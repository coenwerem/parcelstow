#!/usr/bin/env python3
"""Select an ACT checkpoint from nominal development-bank evaluations."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from manipulation.act_model_selection import (
    CandidateResult,
    paired_success_counts,
    select_candidate,
    validate_development_records,
)
from manipulation.file_integrity import sha256_file
from task_registry import get_task


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL from {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"no records found in {path}")
    return rows


def elapsed_seconds(start: str, end: str) -> int:
    """Return elapsed whole seconds between retained ISO-8601 timestamps."""
    return int((datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("upright", "peg"))
    parser.add_argument("--training_dir", type=Path, required=True)
    parser.add_argument("--evaluation_dir", type=Path, required=True)
    parser.add_argument("--expert_record", type=Path)
    parser.add_argument("--demonstration_summary", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--eval_seed", type=int, default=42001)
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--jitter", type=float, default=0.01)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = get_task(args.task)
    candidate_payload = load_json(args.training_dir / "candidates.json")
    run_config_path = args.training_dir / "run_config.json"
    run_config = load_json(run_config_path)
    demonstration_summary_path = args.demonstration_summary or Path(
        f"data/records/{args.task}/demos_summary.jsonl"
    )
    demonstration_summary_rows = load_jsonl(demonstration_summary_path)
    if len(demonstration_summary_rows) != 1:
        raise ValueError(
            f"expected one demonstration summary row in {demonstration_summary_path}"
        )
    demonstration_summary = demonstration_summary_rows[0]
    if demonstration_summary["task"] != task.gym_id:
        raise ValueError("demonstration summary task does not match the selected task")
    if int(demonstration_summary["admitted"]) != int(
        run_config["provenance"]["episode_count"]
    ):
        raise ValueError("demonstration summary and training episode counts differ")
    validation_by_epoch = {
        int(candidate["epoch"]): float(candidate["validation_chunk_l1"])
        for candidate in candidate_payload["candidates"]
    }

    evidence = []
    learner_records_by_epoch = {}
    summaries_by_epoch = {}
    evaluation_files_by_epoch = {}
    for epoch in candidate_payload["retained_epochs"]:
        record_path = args.evaluation_dir / f"act_epoch-{epoch}.jsonl"
        summary_path = args.evaluation_dir / f"summary_epoch-{epoch}.jsonl"
        records = load_jsonl(record_path)
        summary_rows = load_jsonl(summary_path)
        if len(summary_rows) != 1:
            raise ValueError(f"expected one summary row in {summary_path}")
        summary = summary_rows[0]
        checkpoint_path = (
            args.training_dir / "checkpoints" / f"epoch-{int(epoch):04d}.pt"
        )
        bank_sha256 = validate_development_records(
            records,
            summary,
            gym_id=task.gym_id,
            policy="act",
            stage_keys=task.stage_keys,
            expected_episodes=args.episodes,
            expected_seed=args.eval_seed,
            expected_num_envs=args.num_envs,
            expected_jitter=args.jitter,
            checkpoint_path=checkpoint_path,
        )
        candidate = CandidateResult(
            epoch=int(epoch),
            validation_chunk_l1=validation_by_epoch[int(epoch)],
            task_success=int(summary["task_success"]["k"]),
            stage_counts=tuple(int(summary[stage]["k"]) for stage in task.stage_keys),
            bank_sha256=bank_sha256,
        )
        evidence.append(candidate)
        learner_records_by_epoch[int(epoch)] = records
        summaries_by_epoch[int(epoch)] = summary
        evaluation_files_by_epoch[int(epoch)] = (record_path, summary_path)

    selected = select_candidate(evidence)
    result: dict[str, Any] = {
        "task": args.task,
        "status": "development evidence; nominal comparison pending author approval",
        "selection_rule": [
            "highest task-success count",
            "highest lexicographic task-specific stage-count vector",
            "lowest validation chunk L1",
            "earliest epoch",
        ],
        "stage_order": list(task.stage_keys),
        "development_bank": {
            "role": "development",
            "rate": 1.0,
            "episodes": args.episodes,
            "base_seed": args.eval_seed,
            "num_envs": args.num_envs,
            "initial_condition_bank_sha256": selected.bank_sha256,
        },
        "training": {
            "run_config": str(run_config_path),
            "run_config_sha256": sha256_file(run_config_path),
            "git_sha": run_config["provenance"]["git_sha"],
            "demonstrations": run_config["provenance"]["demonstrations"],
            "demonstrations_sha256": run_config["provenance"]["demonstrations_sha256"],
            "model_seed": run_config["arguments"]["model_seed"],
            "elapsed_seconds": candidate_payload["candidates"][-1]["elapsed_seconds"],
            "arguments": run_config["arguments"],
            "environment": run_config["environment"],
            "partition": {
                "train_episode_indices": run_config["provenance"][
                    "train_episode_indices"
                ],
                "validation_episode_indices": run_config["provenance"][
                    "validation_episode_indices"
                ],
                "validation_starts": run_config["provenance"]["validation_starts"],
            },
            "metrics": str(args.training_dir / "metrics.jsonl"),
            "metrics_sha256": sha256_file(args.training_dir / "metrics.jsonl"),
            "candidates": str(args.training_dir / "candidates.json"),
            "candidates_sha256": sha256_file(args.training_dir / "candidates.json"),
            "demonstration_summary": str(demonstration_summary_path),
            "demonstration_summary_sha256": sha256_file(demonstration_summary_path),
            "demonstration_summary_payload": demonstration_summary,
        },
        "candidates": [
            {
                "epoch": candidate.epoch,
                "checkpoint": str(
                    args.training_dir / "checkpoints" / f"epoch-{candidate.epoch:04d}.pt"
                ),
                "checkpoint_sha256": summaries_by_epoch[candidate.epoch][
                    "checkpoint_sha256"
                ],
                "validation_chunk_l1": candidate.validation_chunk_l1,
                "task_success": candidate.task_success,
                "stage_counts": dict(zip(task.stage_keys, candidate.stage_counts, strict=True)),
                "failure_reasons": summaries_by_epoch[candidate.epoch]["failure_reasons"],
                "started_at": learner_records_by_epoch[candidate.epoch][0]["config"]["date"],
                "completed_at": summaries_by_epoch[candidate.epoch]["time"],
                "elapsed_seconds": elapsed_seconds(
                    learner_records_by_epoch[candidate.epoch][0]["config"]["date"],
                    summaries_by_epoch[candidate.epoch]["time"],
                ),
                "episode_records": str(evaluation_files_by_epoch[candidate.epoch][0]),
                "episode_records_sha256": sha256_file(
                    evaluation_files_by_epoch[candidate.epoch][0]
                ),
                "summary": str(evaluation_files_by_epoch[candidate.epoch][1]),
                "summary_sha256": sha256_file(evaluation_files_by_epoch[candidate.epoch][1]),
            }
            for candidate in sorted(evidence, key=lambda item: item.epoch)
        ],
        "selected_epoch": selected.epoch,
        "selected_checkpoint_sha256": summaries_by_epoch[selected.epoch][
            "checkpoint_sha256"
        ],
    }

    if args.expert_record:
        expert_records = load_jsonl(args.expert_record)
        expert_summary_path = args.evaluation_dir / "summary_expert.jsonl"
        expert_summary_rows = load_jsonl(expert_summary_path)
        if len(expert_summary_rows) != 1:
            raise ValueError(f"expected one summary row in {expert_summary_path}")
        expert_bank = validate_development_records(
            expert_records,
            expert_summary_rows[0],
            gym_id=task.gym_id,
            policy="expert",
            stage_keys=task.stage_keys,
            expected_episodes=args.episodes,
            expected_seed=args.eval_seed,
            expected_num_envs=args.num_envs,
            expected_jitter=args.jitter,
            checkpoint_path=None,
        )
        if expert_bank != selected.bank_sha256:
            raise ValueError("expert and selected checkpoint used different banks")
        result["paired_nominal_result"] = paired_success_counts(
            expert_records, learner_records_by_epoch[selected.epoch]
        )
        result["expert"] = {
            "task_success": int(expert_summary_rows[0]["task_success"]["k"]),
            "stage_counts": {
                stage: int(expert_summary_rows[0][stage]["k"])
                for stage in task.stage_keys
            },
            "failure_reasons": expert_summary_rows[0]["failure_reasons"],
            "started_at": expert_records[0]["config"]["date"],
            "completed_at": expert_summary_rows[0]["time"],
            "elapsed_seconds": elapsed_seconds(
                expert_records[0]["config"]["date"], expert_summary_rows[0]["time"]
            ),
            "episode_records": str(args.expert_record),
            "episode_records_sha256": sha256_file(args.expert_record),
            "summary": str(expert_summary_path),
            "summary_sha256": sha256_file(expert_summary_path),
            "checkpoint": None,
            "checkpoint_sha256": None,
        }

    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        print(f"Selection evidence: {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
