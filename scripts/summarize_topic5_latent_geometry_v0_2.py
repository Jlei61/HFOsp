#!/usr/bin/env python3
"""Aggregate Topic 5.2 C1 geometry at the patient level."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import ANALYSIS_REVISION, OUT, SYSTEM  # noqa: E402


REAL_ARMS = ("L0", "L1", "L2m", "L3")
PRIMARY_ENDPOINTS = (
    "progress_r2_P_minus_O",
    "field_r2_PF_minus_P",
    "field_r2_PF_minus_PF_null",
    "early_emergence_real_minus_C_suffix",
)


def bootstrap_median_ci(values: np.ndarray, seed: int, draws: int = 10000) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    medians = np.median(values[indices], axis=1)
    return [float(x) for x in np.quantile(medians, [0.025, 0.975])]


def one_sided_summary(values: np.ndarray, seed: int) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        p = 1.0
    else:
        p = float(wilcoxon(nonzero, alternative="greater", method="auto").pvalue)
    return {
        "n_patients": int(len(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "ci95_median": bootstrap_median_ci(values, seed),
        "positive": int(np.count_nonzero(values > 0)),
        "negative": int(np.count_nonzero(values < 0)),
        "ties": int(np.count_nonzero(values == 0)),
        "p_one_sided": p,
    }


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, key in enumerate(ordered):
        value = min(1.0, (total - rank) * float(pvalues[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def build_patient_effects(
    cells: pd.DataFrame, emergence: pd.DataFrame, *, canonical_only: bool
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = cells[cells["canonical_ab"]].copy() if canonical_only else cells.copy()
    selected_emergence = (
        emergence[emergence["canonical_ab"]].copy() if canonical_only else emergence.copy()
    )
    metrics = [
        "r2_O", "r2_P", "r2_PF", "r2_PF_null", "output_r2_P", "output_r2_PF",
        "residual_r2_P", "residual_r2_PF", "residual_delta_PF_minus_P",
    ]
    fit_arm = selected.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm"], as_index=False
    )[metrics].median(numeric_only=True)
    patient_arm = fit_arm.groupby(["patient", "public_arm"], as_index=False)[metrics].median()

    efit = selected_emergence.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "phase_bin"], as_index=False
    )[["r2_h", "r2_o", "r2_oh", "incremental_r2_oh_minus_o"]].median()
    epatient = efit.groupby(["patient", "public_arm", "phase_bin"], as_index=False)[
        ["r2_h", "r2_o", "r2_oh", "incremental_r2_oh_minus_o"]
    ].median()

    effect_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []
    for patient in sorted(patient_arm["patient"].unique()):
        arm = patient_arm[patient_arm["patient"].eq(patient)].set_index("public_arm")
        if not set(REAL_ARMS).issubset(arm.index) or "C-suffix" not in arm.index:
            raise RuntimeError(f"patient arm matrix incomplete: {patient}")
        real = arm.loc[list(REAL_ARMS)]
        patient_curve = epatient[epatient["patient"].eq(patient)]
        early_scores: dict[str, float] = {}
        for public_arm in (*REAL_ARMS, "C-suffix"):
            part = patient_curve[
                patient_curve["public_arm"].eq(public_arm)
                & patient_curve["phase_bin"].isin([0, 1])
            ]
            early_scores[public_arm] = float(part["incremental_r2_oh_minus_o"].mean())
        real_early = float(np.median([early_scores[arm_name] for arm_name in REAL_ARMS]))
        c_early = early_scores["C-suffix"]
        effect_rows.append({
            "tier": "canonical_ab_shared" if canonical_only else "generic_all_identifiable",
            "patient": patient,
            "progress_r2_P_minus_O": float(np.median(real["r2_P"] - real["r2_O"])),
            "field_r2_PF_minus_P": float(np.median(real["r2_PF"] - real["r2_P"])),
            "field_r2_PF_minus_PF_null": float(
                np.median(real["r2_PF"] - real["r2_PF_null"])
            ),
            "early_emergence_real_minus_C_suffix": real_early - c_early,
            "output_r2_PF_minus_P": float(
                np.median(real["output_r2_PF"] - real["output_r2_P"])
            ),
            "residual_field_r2_PF_minus_P": float(
                np.median(real["residual_r2_PF"] - real["residual_r2_P"])
            ),
            "real_early_incremental_r2": real_early,
            "C_suffix_early_incremental_r2": c_early,
        })
        for b in range(5):
            phase = patient_curve[patient_curve["phase_bin"].eq(b)].set_index("public_arm")
            for endpoint in ("r2_h", "incremental_r2_oh_minus_o"):
                real_value = float(np.median(phase.loc[list(REAL_ARMS), endpoint]))
                c_value = float(phase.loc["C-suffix", endpoint])
                curve_rows.append({
                    "tier": "canonical_ab_shared" if canonical_only else "generic_all_identifiable",
                    "patient": patient,
                    "phase_bin": b,
                    "endpoint": endpoint,
                    "real_order": real_value,
                    "C_suffix": c_value,
                    "real_minus_C_suffix": real_value - c_value,
                })
    return pd.DataFrame(effect_rows), pd.DataFrame(curve_rows)


def main() -> None:
    audit = json.loads((SYSTEM / "PASS1_AUDIT.json").read_text())
    if audit.get("status") != "PASS" or audit.get("analysis_revision") != ANALYSIS_REVISION:
        raise RuntimeError("Pass 1 audit/revision must be PASS before C1 aggregation")
    cells = pd.read_csv(SYSTEM / "PASS1_CELL_GEOMETRY.csv")
    emergence = pd.read_csv(SYSTEM / "PASS1_FUTURE_FIELD_EMERGENCE.csv")
    if len(cells) != 630 or len(emergence) != 3150:
        raise RuntimeError("Pass 1 aggregate denominator drift")
    effect_parts, curve_parts = [], []
    for canonical_only in (False, True):
        effects, curves = build_patient_effects(
            cells, emergence, canonical_only=canonical_only
        )
        effect_parts.append(effects)
        curve_parts.append(curves)
    effects = pd.concat(effect_parts, ignore_index=True)
    curves = pd.concat(curve_parts, ignore_index=True)

    tiers: dict[str, object] = {}
    for tier, group in effects.groupby("tier", sort=False):
        endpoint_summaries = {
            endpoint: one_sided_summary(
                group[endpoint].to_numpy(float),
                seed=int.from_bytes(__import__("hashlib").sha256(
                    f"{tier}/{endpoint}".encode()
                ).digest()[:8], "little"),
            )
            for endpoint in PRIMARY_ENDPOINTS
        }
        adjusted = holm_adjust({
            endpoint: float(summary["p_one_sided"])
            for endpoint, summary in endpoint_summaries.items()
        })
        for endpoint, adjusted_p in adjusted.items():
            endpoint_summaries[endpoint]["p_holm"] = adjusted_p
            endpoint_summaries[endpoint]["status"] = (
                "SUPPORTED"
                if endpoint_summaries[endpoint]["median"] > 0 and adjusted_p < 0.05
                else "UNSUPPORTED"
            )
        tiers[tier] = {
            "n_patients": int(group["patient"].nunique()),
            "endpoints": endpoint_summaries,
            "C1_status": (
                "SUPPORTED"
                if all(summary["status"] == "SUPPORTED" for summary in endpoint_summaries.values())
                else "UNSUPPORTED"
            ),
            "claim_boundary": (
                "canonical A/B future-field geometry"
                if tier == "canonical_ab_shared"
                else "generic within-fit future-field geometry"
            ),
        }
    payload = {
        "contract": "topic5_latent_geometry_C1_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "status": "COMPLETE",
        "tiers": tiers,
        "aggregation_order": ["state", "event", "seed", "fit", "arm", "patient"],
        "real_order_arms": list(REAL_ARMS),
        "early_phase_bins": [0, 1],
        "negative_r2_policy": "RETAINED_NOT_CLIPPED",
        "target_values_read": False,
    }
    atomic_write_csv(SYSTEM / "C1_PATIENT_EFFECTS.csv", effects)
    atomic_write_csv(SYSTEM / "C1_EMERGENCE_CURVES.csv", curves)
    atomic_write_json(SYSTEM / "LATENT_GEOMETRY_SUMMARY.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
