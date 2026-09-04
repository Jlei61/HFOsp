import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "rev22_completion_audit", ROOT / "scripts/audit_topic4_rev22_completion.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_controller_requires_complete_exact_inventory(tmp_path):
    path = tmp_path / "fit/status/controller.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "status": "COMPLETE", "job_count": 4, "state_counts": {"complete": 4},
    }))
    assert MODULE._controller(tmp_path, "fit", 4)["job_count"] == 4
    with pytest.raises(RuntimeError, match="incomplete"):
        MODULE._controller(tmp_path, "fit", 5)


def test_figure_metadata_verifies_every_output_hash(tmp_path):
    output = tmp_path / "figure.png"
    output.write_bytes(b"figure")
    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({
        "output_sha256": {"figure.png": MODULE._sha256(output)},
    }))
    assert MODULE._verify_figure_metadata(metadata)["outputs"] == 1
    output.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        MODULE._verify_figure_metadata(metadata)
