import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reproduce import released_success_counts  # noqa: E402


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
    assert counts["expert"][1.0] == [94, 100]
    assert counts["act"][1.0] == [76, 100]

    import gzip
    import json
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
