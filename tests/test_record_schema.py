"""Schema checks on the released v1 records and on the task identifier
written into newly generated records. The released files under
data/records/ are frozen; these tests pin the fields the analysis code
reads and prove that the extension leaves them untouched."""

import gzip
import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SUMMARY = os.path.join(REPO, "data", "records", "eval_summary.jsonl")
EPISODE_FILES = [
    os.path.join(REPO, "data", "records", f"{name}_episodes.jsonl.gz")
    for name in ("expert", "act", "dp", "dagger")
]
RUNTIME_PY = os.path.join(REPO, "scripts", "manipulation", "stow_runtime.py")

STAGE_KEYS = ["task_success", "acquired", "lifted_clear", "reoriented",
              "preinsert_reached", "inserted", "released", "settled"]
EPISODE_KEYS = ["policy", "task_rate", "task_success", "episode", "env",
                "failure_reason", "config"]


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def first_record_gz(path):
    with gzip.open(path, "rt") as fh:
        return json.loads(fh.readline())


def test_released_summary_schema():
    rows = read_jsonl(SUMMARY)
    assert len(rows) == 28  # 4 policies x 7 speedup factors
    assert {r["policy"] for r in rows} == {"expert", "act", "dp", "dagger"}
    assert {r["rate"] for r in rows} == {0.5, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0}
    for r in rows:
        assert r["episodes"] == 100
        assert "seed" in r and "cycle_time_s" in r
        for key in STAGE_KEYS:
            block = r[key]
            assert set(block) >= {"frac", "k", "n", "wilson"}
            assert len(block["wilson"]) == 2
        # The released rows predate the top-level task identifier.
        assert "task" not in r


def test_released_episode_record_schema():
    for path in EPISODE_FILES:
        rec = first_record_gz(path)
        for key in EPISODE_KEYS:
            assert key in rec, (os.path.basename(path), key)
        assert "task" in rec["config"]
        assert "task" not in rec


def test_new_records_carry_task_identifier():
    with open(RUNTIME_PY) as fh:
        src = fh.read()
    assert 'TASK = "ParcelStow-L6-Distill-Play-v0"' in src
    # run_episodes stamps every new record with a task id defaulting to the
    # ParcelStow gym id, so a second task passes its own without changing
    # the released schema.
    assert '"task": task_id,' in src
    assert "task_id = TASK if task_id is None else task_id" in src
    assert 'out["task"] = records[0]["task"]' in src  # summarize propagates it
