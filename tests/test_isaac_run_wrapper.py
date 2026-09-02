import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts" / "isaac_run.sh"


def _environment():
    environment = os.environ.copy()
    environment["ISAAC_CORES"] = str(min(os.sched_getaffinity(0)))
    environment["ISAAC_NICE"] = "0"
    return environment


def test_isaac_wrapper_preserves_child_failure_status(tmp_path):
    log = tmp_path / "failure.log"
    result = subprocess.run(
        [str(WRAPPER), str(log), "/bin/sh", "-c", "exit 7"],
        cwd=ROOT,
        env=_environment(),
        check=False,
    )
    assert result.returncode == 7
    assert log.read_text().endswith("exit 7\n")


def test_isaac_wrapper_preserves_child_success_status(tmp_path):
    log = tmp_path / "success.log"
    result = subprocess.run(
        [str(WRAPPER), str(log), "/bin/sh", "-c", "printf wrapper-ok"],
        cwd=ROOT,
        env=_environment(),
        check=False,
    )
    assert result.returncode == 0
    assert log.read_text() == "wrapper-okexit 0\n"
