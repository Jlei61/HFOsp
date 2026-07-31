"""Tests for explicit amendment propagation into final Phase-C artifacts."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.adjudicate_topic4_zm_phasec_v2 as A  # noqa: E402


def _fields():
    return {
        "analysis_amendment_path": "results/amendment.json",
        "analysis_amendment_file_sha256": "a" * 64,
        "analysis_amendment_sha256": "b" * 64,
        "analysis_producer_file_sha256": {"analysis.py": "c" * 64},
        "adjudication_v2_wrapper_file_sha256": {"adjudicate.py": "d" * 64},
    }


def test_require_binding_fails_closed_on_missing_amendment():
    with pytest.raises(ValueError, match="analysis_amendment_sha256"):
        A._require_binding(
            {
                "analysis_amendment_path": "results/amendment.json",
                "analysis_amendment_file_sha256": "a" * 64,
            },
            _fields(),
            label="fixture",
        )


def test_build_final_inputs_propagates_binding(monkeypatch):
    fields = _fields()
    monkeypatch.setattr(A, "_analysis_fields", lambda: fields)
    monkeypatch.setattr(
        A.V1,
        "_read",
        lambda _path: fields,
    )
    monkeypatch.setattr(
        A.V1,
        "build_final_inputs",
        lambda **_kwargs: {
            "c0": {"layer": "c0"},
            "c1_primary": {"layer": "c1_primary"},
            "c1_shell": {"layer": "c1_shell"},
            "coverage": {"layer": "coverage"},
        },
    )
    out = A.build_final_inputs(
        c1_native_path=Path("native.json"),
        c1_gate_path=Path("gate.json"),
    )
    assert all(
        row["analysis_amendment_sha256"] == "b" * 64
        for row in out.values()
    )
