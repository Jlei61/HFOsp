#!/usr/bin/env python3
"""Versioned Phase-C1 analyzer with corrected whole-sheet runaway scope.

The production manifest locks ``analyze_topic4_zm_phasec1.py`` and
``topic4_zm_phasec_phenotype.py`` byte-for-byte.  This wrapper leaves both
files untouched, patches their analysis-time globals in a fresh process, and
records a separate analysis amendment/provenance layer.  Raw SNN parts and
resource receipts remain governed by the original immutable manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import scripts.analyze_topic4_zm_phasec1 as V1  # noqa: E402
from src import topic4_zm_phasec_phenotype_v2 as P2  # noqa: E402


ANALYSIS_VERSION = "zm_phasec1_analysis_v2_runaway_scope_2026-07-31"
AMENDMENT = V1.OUT / "phasec_analysis_amendment.json"
_INSTALLED = False

# Re-export the pure interfaces used by the dedicated v2 regression tests.
C1_OBSERVABLES_SCHEMA = V1.C1_OBSERVABLES_SCHEMA
_sha256 = V1._sha256
ROOT = V1.ROOT


def _npz_scalar(value):
    return V1._npz_scalar(value)


def _load_phenotype_arrays(part):
    """Load core morphology and the independent all-sheet runaway trace."""
    path_value = part.get("observables_path")
    if not isinstance(path_value, str) or not path_value:
        return {"status": "blocked", "reason": "missing_observables_path"}
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.is_file():
        return {"status": "blocked", "reason": "missing_observables_npz"}
    if part.get("observables_sha256") != _sha256(path):
        return {"status": "blocked", "reason": "observables_sha256_mismatch"}
    required = (
        "phasec1_observables_schema",
        "bin_ms",
        "E_rate_grid",
        "I_rate_grid",
        "source_rate_hz",
        "rest_mask",
        "active_area_fraction",
        "kymograph",
        "axis_positions",
        "readout_kernel_width_mm",
        "carrier_gate_r_all_hz",
        "carrier_gate_bin_ms",
    )
    try:
        with np.load(path, allow_pickle=False) as data:
            missing = [key for key in required if key not in data.files]
            if missing:
                return {
                    "status": "blocked",
                    "reason": (
                        "missing_phenotype_npz_fields:" + ",".join(missing)
                    ),
                }
            if str(_npz_scalar(data["phasec1_observables_schema"])) != (
                C1_OBSERVABLES_SCHEMA
            ):
                return {
                    "status": "blocked",
                    "reason": "phasec1_observables_schema_mismatch",
                }
            arrays = {
                "bin_ms": float(_npz_scalar(data["bin_ms"])),
                "E_rate_grid": np.asarray(data["E_rate_grid"], float),
                "I_rate_grid": np.asarray(data["I_rate_grid"], float),
                "source_rate_hz": np.asarray(
                    data["source_rate_hz"], float
                ),
                "rest_mask": np.asarray(data["rest_mask"], bool),
                "active_area_fraction": np.asarray(
                    data["active_area_fraction"], float
                ),
                "kymograph": np.asarray(data["kymograph"], float),
                "axis_positions": np.asarray(
                    data["axis_positions"], float
                ),
                "readout_kernel_width_mm": float(
                    _npz_scalar(data["readout_kernel_width_mm"])
                ),
                "all_sheet_rate_hz": np.asarray(
                    data["carrier_gate_r_all_hz"], float
                ),
                "all_sheet_bin_ms": float(
                    _npz_scalar(data["carrier_gate_bin_ms"])
                ),
            }
    except (OSError, TypeError, ValueError) as exc:
        return {
            "status": "blocked",
            "reason": f"invalid_observables_npz:{exc}",
        }
    if (
        arrays["all_sheet_rate_hz"].size < 4
        or not np.all(np.isfinite(arrays["all_sheet_rate_hz"]))
        or not np.isfinite(arrays["all_sheet_bin_ms"])
        or arrays["all_sheet_bin_ms"] <= 0
    ):
        return {
            "status": "blocked",
            "reason": "invalid_all_sheet_runaway_trace",
        }
    return {"status": "ok", "path": path, **arrays}


def _analysis_producers():
    files = (
        Path(__file__).resolve(),
        CODE_ROOT / "src/topic4_zm_phasec_phenotype_v2.py",
    )
    return {
        str(path.relative_to(CODE_ROOT)): _sha256(path)
        for path in files
    }


def _canonical_sha(payload):
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _read_amendment():
    if not AMENDMENT.is_file():
        raise ValueError(
            "phasec_analysis_amendment.json must be locked before v2 analysis"
        )
    payload = json.loads(AMENDMENT.read_text())
    body = {k: v for k, v in payload.items() if k != "amendment_sha256"}
    if (
        payload.get("schema") != "zm_phasec_analysis_amendment_v1"
        or payload.get("amendment_sha256") != _canonical_sha(body)
        or payload.get("corrected_analysis_producer_file_sha256")
        != _analysis_producers()
        or payload.get("threshold_changed") is not False
        or payload.get("raw_snn_parts_reused") is not True
    ):
        raise ValueError("Phase-C analysis amendment is invalid or stale")
    return payload


def _bind_analysis_provenance(payload):
    amendment = _read_amendment()
    out = dict(payload)
    out.update({
        "analysis_version": ANALYSIS_VERSION,
        "analysis_amendment_path": V1._relative(AMENDMENT),
        "analysis_amendment_file_sha256": _sha256(AMENDMENT),
        "analysis_amendment_sha256": amendment["amendment_sha256"],
        "analysis_producer_file_sha256": _analysis_producers(),
    })
    return out


def _install_v2():
    """Patch only the current offline analysis process."""
    global _INSTALLED
    if _INSTALLED:
        return
    V1.P = P2
    V1._load_phenotype_arrays = _load_phenotype_arrays

    original_build = V1.build_base_atlas
    original_apply = V1.apply_conditional_gain
    original_dt2 = V1.analyze_dt2_confirmation
    original_combine = V1.combine_resolution_summaries

    def build(*args, **kwargs):
        return _bind_analysis_provenance(original_build(*args, **kwargs))

    def apply(*args, **kwargs):
        return _bind_analysis_provenance(original_apply(*args, **kwargs))

    def dt2(*args, **kwargs):
        return _bind_analysis_provenance(original_dt2(*args, **kwargs))

    def combine(*args, **kwargs):
        return _bind_analysis_provenance(original_combine(*args, **kwargs))

    V1.build_base_atlas = build
    V1.apply_conditional_gain = apply
    V1.analyze_dt2_confirmation = dt2
    V1.combine_resolution_summaries = combine
    _INSTALLED = True


def main(argv=None):
    _install_v2()
    return V1.main(argv)


if __name__ == "__main__":
    main()
