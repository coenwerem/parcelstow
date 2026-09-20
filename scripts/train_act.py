#!/usr/bin/env python3
"""Train upright or peg ACT with an episode-level validation partition.

This command does not import Isaac Lab. It corrects padded-action loss
reduction, computes normalization from training episodes only, evaluates fixed
validation chunks, and retains a bounded set of checkpoint candidates.

Run:
  python3 scripts/train_act.py --task upright --model_seed 0
  python3 scripts/train_act.py --task peg --model_seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "manipulation"))

from act_training import (  # noqa: E402
    Normalization,
    build_action_batch,
    compute_normalization,
    demonstration_collection_summary,
    fixed_validation_starts,
    masked_l1_loss,
    sample_episode_starts,
    sha256_file,
    split_episode_indices,
)
from state_act import StateACT, kl_divergence  # noqa: E402

TASK_DEFAULTS = {
    "upright": "outputs/upright/demos/expert_episodes.pt",
    "peg": "outputs/peg/demos/expert_episodes.pt",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=tuple(TASK_DEFAULTS), required=True)
    parser.add_argument("--embodiment", choices=("g1_l6", "panda_allegro_right", "ur5_inspire_right"), default="g1_l6")
    parser.add_argument("--demos", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-review", type=Path, default=None,
                        help="Explicit user approval of exact model sources; required for new embodiments")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=3200)
    parser.add_argument("--temporal_agg", type=int, choices=(0, 1), default=1)
    parser.add_argument("--model_seed", type=int, default=0)
    parser.add_argument("--split_seed", type=int, default=20260904)
    parser.add_argument("--validation_start_seed", type=int, default=20260905)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--validation_samples_per_episode", type=int, default=4)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--keep_top", type=int, default=5)
    return parser.parse_args(argv)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True) + "\n")


def checkpoint_sort_key(candidate: dict) -> tuple[float, int]:
    return float(candidate["validation_chunk_l1"]), int(candidate["epoch"])


def retained_epochs(candidates: list[dict], keep_top: int, final_epoch: int) -> set[int]:
    """Select validation candidates while always preserving the final epoch."""
    if keep_top <= 0:
        raise ValueError("keep_top must be positive")
    top = sorted(candidates, key=checkpoint_sort_key)[:keep_top]
    epochs = {int(candidate["epoch"]) for candidate in top}
    if any(int(candidate["epoch"]) == final_epoch for candidate in candidates):
        epochs.add(final_epoch)
    return epochs


@torch.inference_mode()
def validate(
    model: StateACT,
    episodes,
    starts,
    normalization: Normalization,
    chunk_size: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    chunk_error_sum = 0.0
    chunk_element_count = 0
    first_error_sum = 0.0
    first_element_count = 0
    kl_sum = 0.0
    sample_count = 0
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        observations, actions, is_pad = build_action_batch(
            episodes, batch_starts, normalization, chunk_size, device
        )
        prediction, _, _ = model(observations)
        valid = (~is_pad).unsqueeze(-1).expand_as(prediction)
        chunk_error_sum += float((prediction - actions).abs().masked_select(valid).sum())
        chunk_element_count += int(valid.sum())
        first_error_sum += float((prediction[:, 0] - actions[:, 0]).abs().sum())
        first_element_count += prediction.shape[0] * prediction.shape[2]
        _, _, (mu, logvar) = model(observations, actions, is_pad)
        kl_sum += float(kl_divergence(mu, logvar)[0]) * prediction.shape[0]
        sample_count += prediction.shape[0]
    return {
        "validation_chunk_l1": chunk_error_sum / chunk_element_count,
        "validation_first_action_l1": first_error_sum / first_element_count,
        "validation_kl": kl_sum / sample_count,
    }


def checkpoint_payload(
    model: StateACT,
    normalization: Normalization,
    args,
    epoch: int,
    metrics: dict,
    provenance: dict,
) -> dict:
    return {
        "model": model.state_dict(),
        "obs_mean": normalization.obs_mean,
        "obs_std": normalization.obs_std,
        "act_mean": normalization.act_mean,
        "act_std": normalization.act_std,
        "args": vars(args),
        "epoch": epoch,
        "metrics": metrics,
        "provenance": provenance,
    }


def prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    (path / "checkpoints").mkdir()


def validate_demonstration_contract(source, embodiment: str) -> None:
    """Keep the frozen release default; admit dexterous data only explicitly."""
    expected = (147, 16)
    if embodiment == "panda_allegro_right":
        from panda_allegro_contract import JOINT_NAMES, PROTOCOL_ID, OBSERVATION_SCHEMA
        expected = (94, 23)
        metadata = source.get("embodiment_contract", {})
        required = {"embodiment": embodiment, "protocol_id": PROTOCOL_ID,
                    "observation_schema": OBSERVATION_SCHEMA, "observation_dim": 94,
                    "control_rate_hz": 50, "control": "absolute_joint_position_radians",
                    "joint_names": list(JOINT_NAMES), "task_validated": True}
        for key, value in required.items():
            if metadata.get(key) != value:
                raise ValueError(f"Panda–Allegro demonstration contract mismatch: {key}")
        if not metadata.get("source_sha256") or not source.get("readiness_report_sha256"):
            raise ValueError("Panda–Allegro demonstrations require source and readiness provenance")
        records = source.get("all_records", [])
        if not records:
            raise ValueError("Panda–Allegro demonstrations require all attempt records")
        admitted = sum(bool(record["task_success"]) for record in records)
        if admitted < max(2, math.ceil(0.75 * len(records))):
            raise ValueError("Panda–Allegro collection admission failed")
        if admitted != len(source.get("episodes", [])):
            raise ValueError("Panda–Allegro admitted records and episodes differ")
    elif embodiment == "ur5_inspire_right":
        from ur_inspire_contract import validate_metadata
        expected = (77, 12)
        validate_metadata(source.get("embodiment_contract"), require_task_validated=True)
        readiness_hash = source.get("readiness_report_sha256")
        if not isinstance(readiness_hash, str) or re.fullmatch(r"[0-9a-f]{64}", readiness_hash) is None:
            raise ValueError("UR5–Inspire demonstrations require a valid readiness SHA-256")
        records = source.get("all_records")
        if not isinstance(records, list) or not records:
            raise ValueError("UR5–Inspire demonstrations require all attempt records")
        if any(not isinstance(record, dict) or type(record.get("task_success")) is not bool for record in records):
            raise ValueError("UR5–Inspire attempt task_success must be an explicit boolean")
        admitted = sum(record["task_success"] for record in records)
        if admitted < max(2, math.ceil(0.75 * len(records))):
            raise ValueError("UR5–Inspire collection admission failed")
        if admitted != len(source.get("episodes", [])):
            raise ValueError("UR5–Inspire admitted records and episodes differ")
    elif embodiment != "g1_l6":
        raise ValueError(f"unknown embodiment: {embodiment}")
    if (source.get("obs_dim"), source.get("act_dim")) != expected:
        raise ValueError(f"expected {expected[0]} observations and {expected[1]} actions, got "
                         f"{source.get('obs_dim')} and {source.get('act_dim')}")


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.epochs <= 0 or args.batch <= 0 or args.checkpoint_every <= 0:
        raise ValueError("epochs, batch, and checkpoint_every must be positive")
    if args.embodiment != "g1_l6" and (not args.demos or not args.out_dir):
        raise ValueError("new embodiments require explicit --demos and --out_dir")
    if args.embodiment == "panda_allegro_right" and args.task != "upright":
        raise ValueError("Panda–Allegro currently supports upright training only")
    if args.embodiment == "ur5_inspire_right" and args.task != "upright":
        raise ValueError("UR5–Inspire currently supports upright training only")
    demos_path = Path(args.demos or TASK_DEFAULTS[args.task])
    output_path = Path(
        args.out_dir
        or f"outputs/act_comparability/{args.task}/seed-{args.model_seed}/training"
    )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    source = torch.load(demos_path, map_location="cpu", weights_only=False)
    validate_demonstration_contract(source, args.embodiment)
    model_review = None
    if args.embodiment != "g1_l6":
        from embodiment_model_review import require_model_review
        model_review = require_model_review(args.model_review, source["embodiment_contract"]["source_sha256"])
    episodes = [(obs.float(), action.float()) for obs, action, _ in source["episodes"]]
    split = split_episode_indices(
        len(episodes), args.validation_fraction, args.split_seed
    )
    normalization = compute_normalization(episodes, split.train)
    validation_starts = fixed_validation_starts(
        episodes,
        split.validation,
        args.validation_samples_per_episode,
        args.validation_start_seed,
    )
    demonstration_checksum = sha256_file(demos_path)
    demonstration_collection = demonstration_collection_summary(source, len(episodes))
    prepare_output(output_path)
    provenance = {
        "task": args.task,
        "embodiment": args.embodiment,
        "model_review": model_review,
        "embodiment_contract": source.get("embodiment_contract"),
        "readiness_report_sha256": source.get("readiness_report_sha256"),
        "git_sha": git_sha(),
        "demonstrations": str(demos_path),
        "demonstrations_sha256": demonstration_checksum,
        "episode_count": len(episodes),
        "train_episode_indices": split.train,
        "validation_episode_indices": split.validation,
        "validation_starts": validation_starts,
        "demonstration_collection": demonstration_collection,
    }
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "command": [sys.executable, *sys.argv],
    }
    write_json(
        output_path / "run_config.json",
        {"arguments": vars(args), "provenance": provenance, "environment": environment},
    )

    torch.manual_seed(args.model_seed)
    np.random.seed(args.model_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.model_seed)
    rng = np.random.default_rng(args.model_seed)
    model = StateACT(
        source["obs_dim"],
        source["act_dim"],
        args.chunk_size,
        hidden_dim=args.hidden_dim,
        dim_feedforward=args.dim_feedforward,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    candidates: list[dict] = []
    metrics_path = output_path / "metrics.jsonl"
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        permutation = rng.permutation(split.train)
        starts = sample_episode_starts(episodes, permutation, rng)
        epoch_loss = 0.0
        epoch_l1 = 0.0
        epoch_kl = 0.0
        samples = 0
        for offset in range(0, len(starts), args.batch):
            batch_starts = starts[offset : offset + args.batch]
            observations, actions, is_pad = build_action_batch(
                episodes, batch_starts, normalization, args.chunk_size, device
            )
            prediction, _, (mu, logvar) = model(observations, actions, is_pad)
            reconstruction = masked_l1_loss(prediction, actions, is_pad)
            kl = kl_divergence(mu, logvar)[0]
            loss = reconstruction + args.kl_weight * kl
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_count = len(batch_starts)
            epoch_loss += float(loss.detach()) * batch_count
            epoch_l1 += float(reconstruction.detach()) * batch_count
            epoch_kl += float(kl.detach()) * batch_count
            samples += batch_count

        if epoch % args.checkpoint_every != 0 and epoch != args.epochs:
            continue
        metrics = {
            "epoch": epoch,
            "train_loss": epoch_loss / samples,
            "train_reconstruction_l1": epoch_l1 / samples,
            "train_kl": epoch_kl / samples,
            "elapsed_seconds": time.time() - start_time,
        }
        metrics.update(
            validate(
                model,
                episodes,
                validation_starts,
                normalization,
                args.chunk_size,
                args.batch,
                device,
            )
        )
        candidates.append(metrics)
        keep = retained_epochs(candidates, args.keep_top, args.epochs)
        checkpoint_path = output_path / "checkpoints" / f"epoch-{epoch:04d}.pt"
        if epoch in keep:
            torch.save(
                checkpoint_payload(
                    model, normalization, args, epoch, metrics, provenance
                ),
                checkpoint_path,
            )
        for existing in (output_path / "checkpoints").glob("epoch-*.pt"):
            existing_epoch = int(existing.stem.split("-")[1])
            if existing_epoch not in keep:
                existing.unlink()
        append_jsonl(metrics_path, metrics)
        write_json(
            output_path / "candidates.json",
            {
                "selection_metric": "validation_chunk_l1",
                "keep_top": args.keep_top,
                "retained_epochs": sorted(keep),
                "candidates": candidates,
            },
        )
        print(
            f"[epoch {epoch}] train={metrics['train_loss']:.6f} "
            f"validation={metrics['validation_chunk_l1']:.6f} "
            f"retained={sorted(keep)} elapsed={metrics['elapsed_seconds']:.0f}s",
            flush=True,
        )

    final = candidates[-1]
    if not all(math.isfinite(float(value)) for key, value in final.items() if key != "epoch"):
        raise FloatingPointError("final metrics contain a nonfinite value")
    print(f"Training outputs: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
