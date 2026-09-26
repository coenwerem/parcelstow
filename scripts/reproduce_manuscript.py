#!/usr/bin/env python3
"""Validate and reproduce canonical manuscript counts without Isaac Lab or a GPU.

Reads the immutable September 21 inventory and its packaged source series.
Never selects records, merges overlaps, runs a policy, or modifies inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, center - half), min(1, center + half)


def load_bundle(bundle):
    mapping = json.loads((bundle / "FILES.json").read_text())["files"]
    records = {}
    for source, entry in mapping.items():
        raw = (bundle / entry["path"]).read_bytes()
        require(sha256(raw) == entry["sha256"], f"Packaged checksum mismatch: {source}")
        original = gzip.decompress(raw) if entry["encoding"] == "gzip" else raw
        require(sha256(original) == entry["source_sha256"], f"Source checksum mismatch: {source}")
        if entry["path"].startswith("records/"):
            text = gzip.decompress(raw).decode("utf-8")
            records[source] = [json.loads(line) for line in text.splitlines() if line.strip()]
    return records


def validate_conditions(bundle, records):
    conditions = read_csv(bundle / "MANUSCRIPT_EVIDENCE.csv")
    require(len({c["canonical_condition_id"] for c in conditions}) == len(conditions), "Duplicate conditions")
    selected = {}
    results = []
    used = set()
    for c in conditions:
        ident = c["canonical_condition_id"]
        source = c["source_record_path"]
        # The canonical inventory owns selection. Filter only by its rate and
        # recorded line span; never pick a subset based on outcome.
        first, last = int(c["source_line_first"]), int(c["source_line_last"])
        rows = [(i, r) for i, r in enumerate(records[source], 1)
                if first <= i <= last and float(r["task_rate"]) == float(c["task_rate"])]
        for i, _ in rows:
            require((source, i) not in used, f"Overlapping source episode: {source}:{i}")
            used.add((source, i))
        episodes = [r for _, r in rows]
        require(len(episodes) == int(c["episode_count"]), f"Episode count: {ident}")
        key = "initial_condition_id" if c["task"] != "parcel" else "episode"
        require(len({r[key] for r in episodes}) == len(episodes), f"Duplicate episode keys: {ident}")
        require(all(type(r.get("task_success")) is bool for r in episodes), f"Invalid success flag: {ident}")
        successes = sum(r["task_success"] for r in episodes)
        require(successes == int(c["success_count"]), f"Success count: {ident}")
        stages = json.loads(c["stage_counts"])
        for stage, count in stages.items():
            require(sum(bool(r.get(stage, False)) for r in episodes) == count, f"Stage count: {ident}:{stage}")
        failures = dict(Counter(r["failure_reason"] for r in episodes))
        require(failures == json.loads(c["terminal_failure_counts"]), f"Terminal counts: {ident}")
        # Peg events are independently recorded; these are the monitor's direct
        # implications, not an adjacent-stage ordering rule.
        if c["task"] == "peg":
            implications = {"lifted_clear": ("acquired",), "reoriented_upright": ("acquired",),
                            "aligned": ("acquired",), "released": ("inserted",),
                            "settled": ("released",), "task_success": ("inserted", "released", "settled")}
            for r in episodes:
                for premise, consequences in implications.items():
                    require(not r.get(premise) or all(r.get(x) for x in consequences),
                            f"Monitor predicate implication: {ident}:{r[key]}:{premise}")
        lo, hi = wilson(successes, len(episodes))
        selected[ident] = episodes
        results.append({"condition": ident, "task": c["task"], "actor": c["actor"],
                        "speedup_factor": c["task_rate"], "successes": successes, "episodes": len(episodes),
                        "success_rate": successes / len(episodes), "wilson_95_low": lo, "wilson_95_high": hi,
                        "stage_counts": json.dumps(stages, sort_keys=True),
                        "terminal_failure_counts": json.dumps(failures, sort_keys=True),
                        "comparison_status": c["comparison_status"],
                        "speed_degradation_claim_allowed": c["speed_degradation_claim_allowed"]})
    return conditions, selected, results


def paired_results(bundle, selected, bootstrap):
    out = []
    for pair in read_csv(bundle / "PAIRING_STATUS.csv"):
        ident = pair["learner_condition_id"]
        row = {"learner_condition": ident, "expert_condition": pair["expert_condition_id"],
               "status": pair["pairing_status"], "reason": pair["pairing_unavailable_reason"]}
        if pair["pairing_status"] == "VERIFIED_PAIRED":
            key = "episode" if pair["task"] == "parcel" else "initial_condition_id"
            expert = {r[key]: r for r in selected[pair["expert_condition_id"]]}
            learner = {r[key]: r for r in selected[ident]}
            require(expert.keys() == learner.keys(), f"Pairing keys: {ident}")
            differences = []
            counts = Counter()
            for i in sorted(expert):
                e, l = expert[i], learner[i]
                fields = ["seed", "task_rate", "jitter", "corrupt", "action_noise"]
                fields += ["parcel_initial_pose"] if pair["task"] == "parcel" else [
                    "object_initial_pose", "initial_condition_bank_sha256", "bank_role"]
                require(all(e.get(k) == l.get(k) for k in fields), f"Pairing metadata: {ident}:{i}")
                ignored = {"git_sha", "date", "geometry_file", "task"}
                ec = {k: v for k, v in e["config"].items() if k not in ignored}
                lc = {k: v for k, v in l["config"].items() if k not in ignored}
                require(ec == lc, f"Physical configuration: {ident}:{i}")
                a, b = e["task_success"], l["task_success"]
                counts["both_success" if a and b else "expert_only_success" if a else
                       "learner_only_success" if b else "both_failure"] += 1
                differences.append(int(a) - int(b))
            require(len(differences) == int(pair["paired_episode_count"]), f"Paired count: {ident}")
            for name in ["both_success", "expert_only_success", "learner_only_success", "both_failure"]:
                require(counts[name] == int(pair[name]), f"Discordant count: {ident}:{name}")
                row[name] = counts[name]
            row["expert_minus_learner_difference"] = sum(differences) / len(differences)
            if bootstrap:
                import numpy as np
                values = np.asarray(differences, dtype=float)
                draws = np.random.default_rng(0).choice(values, size=(20000, len(values)), replace=True).mean(axis=1)
                lo, hi = np.quantile(draws, [.025, .975], method="linear")
                row.update(paired_95_low=float(lo), paired_95_high=float(hi))
        # Unverified conditions remain standalone. In particular, never reuse
        # the matching subset of a condition that failed full pairing.
        out.append(row)
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "data/manuscript_20260921")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; refuses overwrite")
    parser.add_argument("--bootstrap", action="store_true", help="20,000 paired resamples; requires NumPy")
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        parser.error("output directory already exists; choose a new directory")
    records = load_bundle(args.bundle)
    conditions, selected, results = validate_conditions(args.bundle, records)
    pairs = paired_results(args.bundle, selected, args.bootstrap)
    args.output_dir.mkdir(parents=True)
    with (args.output_dir / "success_counts.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0])); writer.writeheader(); writer.writerows(results)
    (args.output_dir / "paired_results.json").write_text(json.dumps(pairs, indent=2) + "\n")
    report = {"status": "PASS", "conditions": len(conditions), "source_series": len(records),
              "episodes": sum(len(r) for r in selected.values()), "bootstrap_resamples": 20000 if args.bootstrap else 0,
              "pairing_status_counts": dict(Counter(p["status"] for p in pairs)),
              "source_catalog_sha256": sha256((args.bundle / "FILES.json").read_bytes()),
              "scope": "Frozen-record reproduction, not a new policy evaluation or implementation audit"}
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
