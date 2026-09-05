#!/usr/bin/env python3
"""Assess paired nominal expert-learner records at an explicit margin."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from manipulation.file_integrity import sha256_file
from manipulation.nominal_comparability import assess_nominal_pair, validate_margin


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load nonempty JSONL, with gzip selected by the filename suffix."""
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL from {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"no records found in {path}")
    return rows


def margin_argument(value: str) -> float:
    """Parse and validate an explicit CLI noninferiority margin."""
    try:
        return validate_margin(float(value))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-record", type=Path, required=True)
    parser.add_argument("--learner-record", type=Path, required=True)
    parser.add_argument("--margin", type=margin_argument, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = assess_nominal_pair(
        load_jsonl(args.expert_record),
        load_jsonl(args.learner_record),
        margin=args.margin,
    )
    result["input_files"] = {
        "expert_record": str(args.expert_record),
        "expert_record_sha256": sha256_file(args.expert_record),
        "learner_record": str(args.learner_record),
        "learner_record_sha256": sha256_file(args.learner_record),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Nominal comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
