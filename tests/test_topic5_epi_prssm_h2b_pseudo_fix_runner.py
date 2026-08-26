from __future__ import annotations

import json

import pytest

from scripts.topic5_epi_prssm import run_h2b_pseudo_fix_rerun as runner


def _write_verdict(root, verdict: str) -> None:
    path = root / "seizure_link_preictal/CALIPER_VERIFICATION.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        "verdict": verdict,
        "share_with_caliper_applied": 0.49,
    }))


def test_detached_environment_prefers_conda_runtime_library() -> None:
    value = runner.environment()
    assert value["LD_LIBRARY_PATH"].split(":", 1)[0] == str(
        runner.PYTHON.parent.parent / "lib"
    )


def test_partial_caliper_is_recorded_and_does_not_block_sensitivity(
    tmp_path, monkeypatch,
) -> None:
    _write_verdict(tmp_path, "CALIPER_PARTIAL")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "step",
        lambda *args, **kwargs: {
            "label": "verify_caliper", "returncode": 1, "log": "test.log",
        },
    )

    row = runner.verify_caliper_step()

    assert row["returncode"] == 1
    assert row["scientific_status"] == "CALIPER_PARTIAL"
    assert row["share_with_caliper_applied"] == 0.49


def test_missing_caliper_is_still_an_instrument_failure(
    tmp_path, monkeypatch,
) -> None:
    _write_verdict(tmp_path, "CALIPER_NOT_APPLIED")
    monkeypatch.setattr(runner, "OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "step",
        lambda *args, **kwargs: {
            "label": "verify_caliper", "returncode": 1, "log": "test.log",
        },
    )

    with pytest.raises(RuntimeError, match="not usable"):
        runner.verify_caliper_step()
