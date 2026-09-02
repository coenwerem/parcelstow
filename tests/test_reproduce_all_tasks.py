import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce import _wilson, released_success_counts  # noqa: E402


def test_wilson_intervals_cover_observed_fraction():
    for successes in (0, 39, 75, 100):
        lower, upper = _wilson(successes, 100)
        assert 0.0 <= lower <= successes / 100 <= upper <= 1.0


def test_released_parcel_counts():
    counts = released_success_counts("parcel")
    assert counts["expert"][1.0] == [100, 100]
    assert counts["act"][1.0] == [100, 100]
    assert counts["expert"][2.0] == [84, 100]
    assert counts["act"][2.0] == [53, 100]


def test_released_upright_counts():
    counts = released_success_counts("upright")
    assert counts["expert"][1.0] == [92, 100]
    assert counts["act"][1.0] == [39, 100]
    assert counts["expert"][1.75] == [90, 100]
    assert counts["act"][1.75] == [74, 100]


def test_released_peg_counts_and_acquisition_failures():
    counts = released_success_counts("peg")
    assert counts["expert"][1.0] == [93, 100]
    assert counts["act"][1.0] == [75, 100]

    path = ROOT / "data" / "records" / "peg" / "act_episodes.jsonl.gz"
    acquisition = {}
    with gzip.open(path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["task_rate"] >= 1.5:
                acquisition.setdefault(row["task_rate"], [0, 0])
                acquisition[row["task_rate"]][0] += int(row["acquired"])
                acquisition[row["task_rate"]][1] += 1
    assert acquisition
    assert all(value == [0, 100] for value in acquisition.values())


def test_released_records_use_exact_matched_object_initial_poses():
    for task, pose_key in (("parcel", "parcel_initial_pose"),
                           ("upright", "object_initial_pose"),
                           ("peg", "object_initial_pose")):
        record_dir = ROOT / "data" / "records"
        if task != "parcel":
            record_dir /= task
        actors = {}
        for actor in ("expert", "act"):
            with gzip.open(record_dir / f"{actor}_episodes.jsonl.gz", "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            indexed_rows = {
                (float(row["task_rate"]), int(row.get("initial_condition_id", row["episode"]))): row
                for row in rows
            }
            assert len(indexed_rows) == len(rows)
            if task == "peg":
                assert all("initial_condition_id" in row for row in rows)
            actors[actor] = indexed_rows
        assert actors["expert"].keys() == actors["act"].keys()
        for key in actors["expert"]:
            assert actors["expert"][key][pose_key] == actors["act"][key][pose_key]
