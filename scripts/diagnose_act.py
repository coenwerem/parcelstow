#!/usr/bin/env python3
"""Analyze ACT demonstrations, checkpoints, and records without Isaac Lab.

The report measures demonstration coverage, action-chunk phase crossings,
zero-latent action error by phase, fixed-phase rate sensitivity, and exact
expert-ACT initial-pose pairing.

Run:
  python3 scripts/diagnose_act.py --task upright
  python3 scripts/diagnose_act.py --task peg
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "manipulation"))

from act_diagnostics import (  # noqa: E402
    FIXED_ACQUISITION_PHASES,
    RATE_FEATURE_INDEX,
    chunk_crosses_phase_boundary,
    phase_indices,
)
from act_training import Normalization, build_action_batch, sha256_file  # noqa: E402
from state_act import StateACT  # noqa: E402

TASK_DEFAULTS = {
    "upright": {
        "demos": "outputs/upright/demos/expert_episodes.pt",
        "checkpoint": "outputs/upright/act/act_upright.pt",
        "records": "data/records/upright",
        "phase_names": (
            "PARK",
            "APPROACH",
            "PREGRASP_DWELL",
            "CLOSE",
            "GRASP_DWELL",
            "LIFT",
            "REORIENT",
            "TRANSFER",
            "LOWER",
            "PLACE_DWELL",
            "RELEASE",
            "RETREAT",
            "SETTLE",
        ),
    },
    "peg": {
        "demos": "outputs/peg/demos/expert_episodes.pt",
        "checkpoint": "outputs/peg/act/act_peg.pt",
        "records": "data/records/peg",
        "phase_names": (
            "PARK",
            "APPROACH",
            "PREGRASP_DWELL",
            "CLOSE",
            "GRASP_DWELL",
            "LIFT",
            "REORIENT",
            "TRANSFER",
            "INSERT",
            "INSERT_DWELL",
            "RELEASE",
            "RETREAT",
            "SETTLE",
        ),
    },
}
RATE_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", choices=tuple(TASK_DEFAULTS), required=True)
    parser.add_argument("--demos", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--records", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--samples_per_phase", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args(argv)


def load_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def record_summary(record_dir: Path) -> dict:
    by_actor = {
        actor: load_jsonl_gz(record_dir / f"{actor}_episodes.jsonl.gz")
        for actor in ("expert", "act")
    }
    curves = {}
    for actor, rows in by_actor.items():
        curves[actor] = []
        for rate in sorted({float(row["task_rate"]) for row in rows}):
            selected = [row for row in rows if float(row["task_rate"]) == rate]
            curves[actor].append(
                {
                    "rate": rate,
                    "episodes": len(selected),
                    "task_success": sum(bool(row["task_success"]) for row in selected),
                    "acquired": sum(bool(row["acquired"]) for row in selected),
                    "failure_reasons": dict(
                        sorted(Counter(row["failure_reason"] for row in selected).items())
                    ),
                }
            )
    expert_by_key = {
        (float(row["task_rate"]), int(row["episode"])): row
        for row in by_actor["expert"]
    }
    exact_pose_matches = 0
    for row in by_actor["act"]:
        expert = expert_by_key[(float(row["task_rate"]), int(row["episode"]))]
        exact_pose_matches += row["object_initial_pose"] == expert["object_initial_pose"]
    return {
        "source_paths": {
            actor: str(record_dir / f"{actor}_episodes.jsonl.gz")
            for actor in by_actor
        },
        "curves": curves,
        "exact_initial_pose_matches": exact_pose_matches,
        "paired_episode_count": len(by_actor["act"]),
        "unique_indexed_conditions": {
            actor: len(
                {
                    (float(row["task_rate"]), row.get("initial_condition_id"))
                    for row in rows
                    if "initial_condition_id" in row
                }
            )
            for actor, rows in by_actor.items()
        },
    }


def demonstration_summary(source: dict) -> dict:
    admitted = source["records"]
    rejected = [row for row in source["all_records"] if not row["task_success"]]
    admitted_rates = np.asarray([row["task_rate"] for row in admitted], dtype=float)
    rejected_rates = np.asarray([row["task_rate"] for row in rejected], dtype=float)
    positions = np.asarray([row["object_initial_pose"]["pos"] for row in admitted])
    return {
        "admitted": len(admitted),
        "attempted": len(source["all_records"]),
        "rate_spec": source["rate_spec"],
        "admitted_rate": {
            "minimum": float(admitted_rates.min()),
            "mean": float(admitted_rates.mean()),
            "maximum": float(admitted_rates.max()),
        },
        "rejected_rate": None
        if not rejected
        else {
            "minimum": float(rejected_rates.min()),
            "mean": float(rejected_rates.mean()),
            "maximum": float(rejected_rates.max()),
        },
        "rejected_failure_reasons": dict(
            sorted(Counter(row["failure_reason"] for row in rejected).items())
        ),
        "admitted_object_position_bounds": {
            "minimum": positions.min(axis=0).tolist(),
            "maximum": positions.max(axis=0).tolist(),
        },
        "has_initial_condition_ids": all(
            "initial_condition_id" in row for row in source["all_records"]
        ),
        "has_initial_robot_state": all(
            "robot_initial_position" in row for row in source["all_records"]
        ),
    }


def stratified_references(episodes, samples_per_phase: int, seed: int):
    references = {phase: [] for phase in range(13)}
    episode_phases = []
    crossing_counts = {phase: [0, 0] for phase in range(13)}
    for episode_index, (observations, _) in enumerate(episodes):
        phases = phase_indices(observations)
        episode_phases.append(phases)
        for start, phase in enumerate(phases.tolist()):
            references[phase].append((episode_index, start))
            crossing_counts[phase][1] += 1
            crossing_counts[phase][0] += chunk_crosses_phase_boundary(
                phases, start, 100
            )
    rng = np.random.default_rng(seed)
    sampled = {}
    for phase, available in references.items():
        count = min(samples_per_phase, len(available))
        selected = rng.choice(len(available), size=count, replace=False)
        sampled[phase] = [available[int(index)] for index in selected]
    return sampled, episode_phases, crossing_counts


@torch.inference_mode()
def action_metrics(
    model,
    episodes,
    references,
    episode_phases,
    normalization,
    chunk_size,
    batch_size,
    device,
) -> dict:
    model.eval()
    result = {}
    for phase, starts in references.items():
        first_error = 0.0
        first_count = 0
        chunk_error = 0.0
        chunk_count = 0
        crossing_first_error = {False: [0.0, 0], True: [0.0, 0]}
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            observations, actions, is_pad = build_action_batch(
                episodes, batch_starts, normalization, chunk_size, device
            )
            prediction, _, _ = model(observations)
            first_per_sample = (prediction[:, 0] - actions[:, 0]).abs().mean(dim=1)
            first_error += float(first_per_sample.sum())
            first_count += prediction.shape[0]
            valid = (~is_pad).unsqueeze(-1).expand_as(prediction)
            chunk_error += float((prediction - actions).abs().masked_select(valid).sum())
            chunk_count += int(valid.sum())
            for value, (episode_index, start) in zip(first_per_sample, batch_starts):
                crosses = chunk_crosses_phase_boundary(
                    episode_phases[episode_index], start, chunk_size
                )
                crossing_first_error[crosses][0] += float(value)
                crossing_first_error[crosses][1] += 1
        result[phase] = {
            "samples": first_count,
            "first_action_l1": first_error / first_count,
            "chunk_l1": chunk_error / chunk_count,
            "first_action_l1_by_boundary_crossing": {
                str(crosses).lower(): None
                if count == 0
                else error / count
                for crosses, (error, count) in crossing_first_error.items()
            },
        }
    return result


@torch.inference_mode()
def fixed_phase_rate_sensitivity(
    model,
    episodes,
    references,
    normalization,
    device,
    batch_size,
) -> dict:
    selected = []
    for phase in FIXED_ACQUISITION_PHASES:
        selected.extend(references[phase])
    obs_mean = normalization.obs_mean.to(device)
    obs_std = normalization.obs_std.to(device)
    act_mean = normalization.act_mean.to(device)
    act_std = normalization.act_std.to(device)
    predictions = {rate: [] for rate in RATE_GRID}
    for offset in range(0, len(selected), batch_size):
        starts = selected[offset : offset + batch_size]
        observations = torch.stack(
            [episodes[episode_index][0][start] for episode_index, start in starts]
        ).to(device)
        for rate in RATE_GRID:
            modified = observations.clone()
            modified[:, RATE_FEATURE_INDEX] = rate
            normalized = (modified - obs_mean) / obs_std
            prediction, _, _ = model(normalized)
            predictions[rate].append((prediction[:, 0] * act_std + act_mean).cpu())
    predictions = {
        rate: torch.cat(chunks) for rate, chunks in predictions.items()
    }
    baseline = predictions[1.0]
    return {
        f"{rate:g}": float(torch.mean((prediction - baseline) ** 2).sqrt())
        for rate, prediction in predictions.items()
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.samples_per_phase <= 0 or args.batch <= 0:
        raise ValueError("samples_per_phase and batch must be positive")
    defaults = TASK_DEFAULTS[args.task]
    demos_path = Path(args.demos or defaults["demos"])
    checkpoint_path = Path(args.checkpoint or defaults["checkpoint"])
    record_dir = Path(args.records or defaults["records"])
    output_path = Path(
        args.output
        or f"outputs/act_comparability/{args.task}/offline-diagnostic.json"
    )
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    source = torch.load(demos_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    episodes = [(obs.float(), action.float()) for obs, action, _ in source["episodes"]]
    normalization = Normalization(
        obs_mean=checkpoint["obs_mean"],
        obs_std=checkpoint["obs_std"],
        act_mean=checkpoint["act_mean"],
        act_std=checkpoint["act_std"],
    )
    checkpoint_args = checkpoint["args"]
    model = StateACT(
        checkpoint["obs_mean"].shape[0],
        checkpoint["act_mean"].shape[0],
        checkpoint_args["chunk_size"],
        hidden_dim=checkpoint_args["hidden_dim"],
        dim_feedforward=checkpoint_args["dim_feedforward"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    references, episode_phases, crossing = stratified_references(
        episodes, args.samples_per_phase, args.seed
    )
    phase_metrics = action_metrics(
        model,
        episodes,
        references,
        episode_phases,
        normalization,
        checkpoint_args["chunk_size"],
        args.batch,
        device,
    )
    phase_names = defaults["phase_names"]
    for phase, metrics in phase_metrics.items():
        metrics["phase"] = phase
        metrics["phase_name"] = phase_names[phase]
        metrics["boundary_crossing_count"] = crossing[phase][0]
        metrics["available_starts"] = crossing[phase][1]
        metrics["boundary_crossing_fraction"] = crossing[phase][0] / crossing[phase][1]

    report = {
        "task": args.task,
        "demonstrations": {
            "path": str(demos_path),
            "sha256": sha256_file(demos_path),
            **demonstration_summary(source),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "arguments": checkpoint_args,
            "rate_mean": float(checkpoint["obs_mean"][RATE_FEATURE_INDEX]),
            "rate_std": float(checkpoint["obs_std"][RATE_FEATURE_INDEX]),
            "rate_z_scores": {
                f"{rate:g}": (
                    rate - float(checkpoint["obs_mean"][RATE_FEATURE_INDEX])
                )
                / float(checkpoint["obs_std"][RATE_FEATURE_INDEX])
                for rate in RATE_GRID
            },
        },
        "records": record_summary(record_dir),
        "phase_metrics": [phase_metrics[phase] for phase in range(13)],
        "fixed_acquisition_rate_sensitivity_rms": fixed_phase_rate_sensitivity(
            model,
            episodes,
            references,
            normalization,
            device,
            args.batch,
        ),
        "diagnostic": {
            "seed": args.seed,
            "samples_per_phase": args.samples_per_phase,
            "device": str(device),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Diagnostic report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
