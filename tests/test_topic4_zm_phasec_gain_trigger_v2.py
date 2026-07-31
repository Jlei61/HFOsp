"""Tests for v2 conditional-gain trigger provenance propagation."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.lock_topic4_zm_phasec1_gain_triggers_v2 as L2  # noqa: E402


def test_trigger_binding_rejects_base_without_analysis_amendment(monkeypatch):
    monkeypatch.setattr(
        L2.A2,
        "_read_amendment",
        lambda: {"amendment_sha256": "a" * 64},
    )
    monkeypatch.setattr(
        L2.A2,
        "_analysis_producers",
        lambda: {"scripts/analyze_v2.py": "b" * 64},
    )
    monkeypatch.setattr(L2.C1, "_sha256", lambda _path: "c" * 64)
    monkeypatch.setattr(
        L2.C1, "_relative", lambda _path: "results/amendment.json"
    )
    with pytest.raises(ValueError, match="does not bind"):
        L2._bind_v2_provenance(
            {"manifest_sha256": "d" * 64},
            {},
        )


def test_trigger_binding_rehashes_v2_provenance(monkeypatch):
    amendment = {"amendment_sha256": "a" * 64}
    producers = {"scripts/analyze_v2.py": "b" * 64}
    monkeypatch.setattr(L2.A2, "_read_amendment", lambda: amendment)
    monkeypatch.setattr(L2.A2, "_analysis_producers", lambda: producers)
    monkeypatch.setattr(L2.C1, "_sha256", lambda _path: "c" * 64)
    monkeypatch.setattr(
        L2.C1, "_relative", lambda _path: "results/amendment.json"
    )
    base = {
        "analysis_amendment_path": "results/amendment.json",
        "analysis_amendment_file_sha256": "c" * 64,
        "analysis_amendment_sha256": "a" * 64,
        "analysis_producer_file_sha256": producers,
    }
    out = L2._bind_v2_provenance(
        {"schema": "trigger", "manifest_sha256": "old"},
        base,
    )
    assert out["analysis_amendment_sha256"] == "a" * 64
    assert out["manifest_sha256"] != "old"
    body = {k: v for k, v in out.items() if k != "manifest_sha256"}
    assert out["manifest_sha256"] == L2.C1._object_sha(body)
