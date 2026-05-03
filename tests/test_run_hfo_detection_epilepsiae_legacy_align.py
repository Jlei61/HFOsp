"""TDD tests for `scripts/run_hfo_detection.py` Epilepsiae legacy-align defaults.

Plan reference: docs/archive/epilepsiae_lagpat/detector_drift_root_cause_2026-05-03.md
Path A — verify the 5 detector-layer deltas (Δ5–Δ10) all flow through to the
HFODetectionConfig and load_epilepsiae_block call sites.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_hfo_detection.py"


@pytest.fixture(scope="module")
def script_source() -> str:
    """Read the script as text so we can grep for parameter wiring without
    executing module-level GPU init code (the script imports cusignal at top)."""
    return SCRIPT_PATH.read_text(encoding="utf-8")


def test_epilepsiae_path_uses_5_freq_notch_default(script_source: str) -> None:
    """Δ5: notch_freqs default must include 250 Hz at the Epilepsiae call site."""
    # The list literal `[50.0, 100.0, 150.0, 200.0, 250.0]` must appear in the
    # epilepsiae block. We accept either int or float forms.
    assert (
        "[50.0, 100.0, 150.0, 200.0, 250.0]" in script_source
        or "[50, 100, 150, 200, 250]" in script_source
    ), "Epilepsiae call site must default notch_freqs to 5 freqs (incl. 250 Hz)"


def test_epilepsiae_path_uses_fir_legacy_notch_default(script_source: str) -> None:
    """Δ7: notch_filter_kind default must be 'fir_legacy' for Epilepsiae."""
    # Permissive search: the literal 'fir_legacy' must be in the Epilepsiae
    # block (the only place we set notch_filter_kind in this script).
    assert "fir_legacy" in script_source, (
        "Epilepsiae call site must default notch_filter_kind to 'fir_legacy'"
    )


def test_epilepsiae_path_uses_legacy_align_default(script_source: str) -> None:
    """Δ6 + Δ8: legacy_align default must be True for Epilepsiae detector."""
    # Look for `legacy_align = bool(params.get("legacy_align", True))` in script
    assert 'params.get("legacy_align", True)' in script_source, (
        "Epilepsiae call site must default legacy_align=True"
    )


def test_epilepsiae_path_uses_200s_chunk_default(script_source: str) -> None:
    """Δ10: chunk_sec default must be 200.0 for Epilepsiae."""
    assert 'params.get("chunk_sec", 200.0)' in script_source, (
        "Epilepsiae call site must default chunk_sec=200.0"
    )


def test_epilepsiae_path_uses_zero_chunk_overlap_when_legacy_align(
    script_source: str,
) -> None:
    """Δ9: chunk_overlap_sec must be 0 when legacy_align is True."""
    assert "0.0 if legacy_align else" in script_source, (
        "Epilepsiae call site must force chunk_overlap_sec=0 when legacy_align=True"
    )


def test_load_epilepsiae_block_call_passes_notch_kwargs(script_source: str) -> None:
    """Both notch_freqs and notch_filter_kind must be forwarded into
    load_epilepsiae_block at the Epilepsiae detect path."""
    # This guards against a future refactor that drops the keyword passthrough.
    assert "notch_freqs=notch_freqs" in script_source
    assert "notch_filter_kind=notch_filter_kind" in script_source


def test_yuquan_path_unchanged_no_fir_legacy_notch(script_source: str) -> None:
    """Yuquan must NOT use 'fir_legacy' notch (legacy yuquan p16_cuda doesn't
    apply notch via firwin; SEEGPreprocessor IIR is the canonical Yuquan
    path). Guard against accidental cross-dataset application of the
    Epilepsiae-only Δ7 fix."""
    # Find the run_yuquan_subject function block.
    assert "def run_yuquan_subject" in script_source
    yq_block = script_source.split("def run_yuquan_subject")[1].split("def run_epilepsiae_subject")[0]
    assert "notch_filter_kind" not in yq_block, (
        "Yuquan path must not set notch_filter_kind (Epilepsiae-only)"
    )
