#!/usr/bin/env python3
"""Describe paired nominal expert-learner outcomes without a deficit threshold."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Any

from manipulation.file_integrity import sha256_file
from manipulation.nominal_comparability import describe_nominal_pair


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert-record", type=Path, required=True)
    parser.add_argument("--learner-record", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite report: {args.output}")
    result = describe_nominal_pair(
        load_jsonl(args.expert_record),
        load_jsonl(args.learner_record),
    )
    result["input_files"] = {
        "expert_record": str(args.expert_record),
        "expert_record_sha256": sha256_file(args.expert_record),
        "learner_record": str(args.learner_record),
        "learner_record_sha256": sha256_file(args.learner_record),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Nominal comparison: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
