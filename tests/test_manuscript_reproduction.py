"""Failure-path checks for the frozen-evidence reproducer and artifact copier."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


repro = module("reproduce_manuscript", ROOT / "scripts/reproduce_manuscript.py")


@pytest.fixture(scope="module")
def frozen():
    bundle = ROOT / "data/manuscript_20260921"
    records = repro.load_bundle(bundle)
    conditions, selected, results = repro.validate_conditions(bundle, records)
    return bundle, records, conditions, selected, results


def test_frozen_counts_and_pairing(frozen):
    bundle, _, conditions, selected, results = frozen
    assert len(conditions) == len(results) == 136
    pairs = repro.paired_results(bundle, selected, False)
    assert sum(p["status"] == "VERIFIED_PAIRED" for p in pairs) == 104
    assert sum(p["status"] != "VERIFIED_PAIRED" for p in pairs) == 9
    parcel = next(p for p in pairs if p["learner_condition"] == "S180_r2")
    assert parcel["expert_only_success"] == 42
    assert parcel["learner_only_success"] == 11
    assert parcel["expert_minus_learner_difference"] == .31
    for pair in pairs:
        if pair["status"] != "VERIFIED_PAIRED":
            assert "expert_minus_learner_difference" not in pair


def test_success_mismatch_stops(frozen):
    bundle, original, *_ = frozen
    records = copy.deepcopy(original)
    source = "data/records/act_episodes.jsonl.gz"
    records[source][0]["task_success"] = not records[source][0]["task_success"]
    with pytest.raises(ValueError, match="Success count"):
        repro.validate_conditions(bundle, records)


def test_pairing_mismatch_stops(frozen):
    bundle, _, _, selected, _ = frozen
    selected = copy.deepcopy(selected)
    selected["S180_r2"][0]["seed"] += 1
    with pytest.raises(ValueError, match="Pairing metadata"):
        repro.paired_results(bundle, selected, False)


def test_source_checksum_mismatch_stops(tmp_path):
    (tmp_path / "record.txt").write_text("changed")
    (tmp_path / "FILES.json").write_text(json.dumps({"files": {"source": {
        "path": "record.txt", "sha256": "0" * 64, "source_sha256": "0" * 64, "encoding": "identity"}}}))
    with pytest.raises(ValueError, match="checksum mismatch"):
        repro.load_bundle(tmp_path)


def test_existing_output_refused(tmp_path):
    with pytest.raises(SystemExit):
        repro.main(["--output-dir", str(tmp_path)])


def test_peg_independent_events_remain_valid(frozen):
    _, records, *_ = frozen
    peg = [r for source, rows in records.items() if '/peg/' in source for r in rows]
    cases = [r for r in peg if r.get("inserted") and not r.get("aligned")]
    assert len(cases) >= 3  # The frozen DP regression cases remain admitted.
    assert all(not r['task_success'] for r in cases)


def test_local_artifact_copy_and_no_overwrite(tmp_path, monkeypatch):
    # Downloader imports the task registry but never imports Isaac Lab.
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    downloader = module("download_artifacts", ROOT / "scripts/download_artifacts.py")
    monkeypatch.setattr(downloader, "REPO", str(tmp_path / "code"))
    hub = tmp_path / "hf"; hub.mkdir()
    (hub / "model.pt").write_bytes(b"verified weights")
    artifact = {"path": "outputs/model.pt", "hf_path": "model.pt", "bytes": 16,
                "sha256": hashlib.sha256(b"verified weights").hexdigest()}
    downloader.fetch("model", artifact, {}, str(hub))
    assert (tmp_path / "code/outputs/model.pt").read_bytes() == b"verified weights"
    with pytest.raises(FileExistsError):
        downloader.fetch("model", artifact, {}, str(hub))
    artifact["path"] = "outputs/bad.pt"
    (hub / "model.pt").write_bytes(b"version https://git-lfs.github.com/spec/v1")
    with pytest.raises(ValueError, match="failed verification"):
        downloader.fetch("model", artifact, {}, str(hub))
    assert not (tmp_path / "code/outputs/bad.pt").exists()
