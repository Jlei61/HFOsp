import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANAGER = ROOT / "scripts/run_topic4_rev10_sa_managed_command.sh"


def test_managed_command_records_external_termination_as_failure(tmp_path):
    status = tmp_path / "status"
    log = tmp_path / "log"
    process = subprocess.Popen([
        str(MANAGER), str(status), str(log), "test", "deadbeef",
        "/bin/sleep", "30",
    ])
    for _ in range(100):
        if status.exists() and status.read_text().startswith("RUNNING"):
            break
        time.sleep(0.01)
    process.terminate()
    assert process.wait(timeout=5) != 0
    assert status.read_text().startswith("FAILED exit_code=143")
