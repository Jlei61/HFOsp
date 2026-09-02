#!/usr/bin/env python3
"""Aggregate Topic 5.2 C2 teacher-forced dynamics at patient level."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import ANALYSIS_REVISION  # noqa: E402
from scripts.run_topic5_latent_transport_v0_2 import TRANSPORT, TRANSPORT_REVISION  # noqa: E402
from scripts.summarize_topic5_latent_geometry_v0_2 import holm_adjust, one_sided_summary  # noqa: E402


REAL_ARMS = ("L0", "L1", "L2m", "L3")
PRIMARY_ENDPOINTS = (
    "progress_transport_cosine",
    "field_transport_cosine",
    "transverse_contraction",
    "event_to_PF_manifold_convergence",
)
SENSITIVITY_ENDPOINTS = (
    "progress_gain_minus_normal",
    "field_gain_minus_normal",
    "event_to_curve_convergence",
)
CONTROL_SUFFIX = "_real_minus_C_suffix"
# Spec 5.7: transverse contraction counts only when the absolute gain is below one
# *and* below controls.  The other primary endpoints are adjudicated on positivity,
# but their control comparison is reported alongside so a positive cosine is never
# read as an order-specific result.
CONTROL_CONDITIONED_ENDPOINTS = ("transverse_contraction",)


def patient_tables(cells: pd.DataFrame, canonical_only: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = cells[cells["canonical_ab"]].copy() if canonical_only else cells.copy()
    metrics = list(PRIMARY_ENDPOINTS + SENSITIVITY_ENDPOINTS)
    # Each phase contributes equally within a seed; then seed, fit, and arm are collapsed.
    seed = selected.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "seed"], as_index=False
    )[metrics].median()
    fit = seed.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm"], as_index=False
    )[metrics].median()
    arm = fit.groupby(["patient", "public_arm"], as_index=False)[metrics].median()

    effects: list[dict[str, object]] = []
    for patient, group in arm.groupby("patient"):
        indexed = group.set_index("public_arm")
        if not set((*REAL_ARMS, "C-suffix")).issubset(indexed.index):
            raise RuntimeError(f"incomplete patient-arm matrix: {patient}")
        real = indexed.loc[list(REAL_ARMS), metrics].median(axis=0)
        control = indexed.loc["C-suffix", metrics]
        row: dict[str, object] = {
            "tier": "canonical_ab_shared" if canonical_only else "generic_all_identifiable",
            "patient": patient,
        }
        for metric in metrics:
            row[metric] = float(real[metric])
            row[f"{metric}_real_minus_C_suffix"] = float(real[metric] - control[metric])
        effects.append(row)

    # Preserve phase as a descriptive time course using the same hierarchy.
    phase_seed = selected.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "seed", "phase_target"],
        as_index=False,
    )[metrics].median()
    phase_fit = phase_seed.groupby(
        ["patient", "fit_id", "geometry_view", "public_arm", "phase_target"],
        as_index=False,
    )[metrics].median()
    phase_arm = phase_fit.groupby(
        ["patient", "public_arm", "phase_target"], as_index=False
    )[metrics].median()
    curves: list[dict[str, object]] = []
    for (patient, phase), group in phase_arm.groupby(["patient", "phase_target"]):
        indexed = group.set_index("public_arm")
        for metric in metrics:
            real_value = float(indexed.loc[list(REAL_ARMS), metric].median())
            control_value = float(indexed.loc["C-suffix", metric])
            curves.append({
                "tier": "canonical_ab_shared" if canonical_only else "generic_all_identifiable",
                "patient": patient,
                "phase_target": float(phase),
                "endpoint": metric,
                "real_order": real_value,
                "C_suffix": control_value,
                "real_minus_C_suffix": real_value - control_value,
            })
    return pd.DataFrame(effects), pd.DataFrame(curves)


def main() -> None:
    audit = json.loads((TRANSPORT / "TRANSPORT_AUDIT.json").read_text())
    if audit.get("status") != "PASS" or audit.get("transport_revision") != TRANSPORT_REVISION:
        raise RuntimeError("corrected transport audit must pass before C2 aggregation")
    cells = pd.read_csv(TRANSPORT / "TRANSPORT_CELL_PHASE_SUMMARY.csv")
    if len(cells) != 1890:
        raise RuntimeError("transport aggregate denominator drift")
    effects, curves = [], []
    for canonical_only in (False, True):
        effect, curve = patient_tables(cells, canonical_only)
        effects.append(effect)
        curves.append(curve)
    effects_frame = pd.concat(effects, ignore_index=True)
    curves_frame = pd.concat(curves, ignore_index=True)

    tiers: dict[str, object] = {}
    for tier, group in effects_frame.groupby("tier", sort=False):
        endpoints = {}
        for endpoint in (*PRIMARY_ENDPOINTS, *SENSITIVITY_ENDPOINTS):
            seed = int.from_bytes(hashlib.sha256(f"C2/{tier}/{endpoint}".encode()).digest()[:8], "little")
            endpoints[endpoint] = one_sided_summary(group[endpoint].to_numpy(float), seed)
        control = {}
        for endpoint in PRIMARY_ENDPOINTS:
            seed = int.from_bytes(
                hashlib.sha256(f"C2control/{tier}/{endpoint}".encode()).digest()[:8], "little"
            )
            control[endpoint] = one_sided_summary(
                group[f"{endpoint}{CONTROL_SUFFIX}"].to_numpy(float), seed
            )
        control_adjusted = holm_adjust({
            endpoint: float(control[endpoint]["p_one_sided"]) for endpoint in PRIMARY_ENDPOINTS
        })
        adjusted = holm_adjust({
            endpoint: float(endpoints[endpoint]["p_one_sided"])
            for endpoint in PRIMARY_ENDPOINTS
        })
        for endpoint in PRIMARY_ENDPOINTS:
            endpoints[endpoint]["p_holm_primary_family"] = adjusted[endpoint]
            control[endpoint]["p_holm_control_family"] = control_adjusted[endpoint]
            control[endpoint]["status_vs_order_shuffled_control"] = (
                "SUPPORTED" if control[endpoint]["median"] > 0 and control_adjusted[endpoint] < 0.05
                else "UNSUPPORTED"
            )
            endpoints[endpoint]["vs_order_shuffled_control"] = control[endpoint]
            endpoints[endpoint]["status_vs_zero"] = (
                "SUPPORTED" if endpoints[endpoint]["median"] > 0 and adjusted[endpoint] < 0.05
                else "UNSUPPORTED"
            )
            if endpoint in CONTROL_CONDITIONED_ENDPOINTS:
                endpoints[endpoint]["status"] = (
                    "SUPPORTED"
                    if endpoints[endpoint]["status_vs_zero"] == "SUPPORTED"
                    and control[endpoint]["status_vs_order_shuffled_control"] == "SUPPORTED"
                    else "UNSUPPORTED"
                )
                endpoints[endpoint]["status_rule"] = (
                    "SPEC_5_7_ABSOLUTE_GAIN_BELOW_ONE_AND_BELOW_CONTROLS"
                )
            else:
                endpoints[endpoint]["status"] = endpoints[endpoint]["status_vs_zero"]
                endpoints[endpoint]["status_rule"] = "SPEC_5_7_POSITIVITY_ONLY"
        for endpoint in SENSITIVITY_ENDPOINTS:
            endpoints[endpoint]["status"] = "SENSITIVITY_ONLY"
        teacher_forced = (
            "SUPPORTED"
            if all(endpoints[endpoint]["status"] == "SUPPORTED" for endpoint in PRIMARY_ENDPOINTS)
            else "UNSUPPORTED"
        )
        tiers[tier] = {
            "n_patients": int(group["patient"].nunique()),
            "endpoints": endpoints,
            "teacher_forced_C2_status": teacher_forced,
            "C2_status": f"PARTIAL_{teacher_forced}_CLOSED_LOOP_PENDING",
            "interpretation": (
                "A propagation channel requires tangent and field transport, transverse contraction, "
                "and positive event-to-conditional-manifold convergence."
            ),
            "order_specificity": (
                "NOT_ORDER_SPECIFIC" if all(
                    endpoints[endpoint]["vs_order_shuffled_control"][
                        "status_vs_order_shuffled_control"
                    ] == "UNSUPPORTED"
                    for endpoint in PRIMARY_ENDPOINTS
                ) else "PARTIALLY_ORDER_SPECIFIC"
            ),
        }
    payload = {
        "contract": "topic5_latent_dynamical_transport_C2_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "transport_revision": TRANSPORT_REVISION,
        "status": "COMPLETE_TEACHER_FORCED_CLOSED_LOOP_PENDING",
        "tiers": tiers,
        "primary_endpoints": list(PRIMARY_ENDPOINTS),
        "control_reference": (
            "Every primary endpoint is reported both against zero and against the same patient's "
            "order-shuffled arm; the transverse-contraction verdict requires both legs."
        ),
        "aggregation_order": ["state", "phase", "seed", "fit", "arm", "patient"],
        "shared_gamma_convergence_role": "SENSITIVITY_ONLY",
        "negative_metric_policy": "RETAINED_NOT_CLIPPED",
        "target_values_read": False,
    }
    atomic_write_csv(TRANSPORT / "C2_PATIENT_EFFECTS.csv", effects_frame)
    atomic_write_csv(TRANSPORT / "C2_PHASE_CURVES.csv", curves_frame)
    atomic_write_json(TRANSPORT / "DYNAMICAL_TRANSPORT_SUMMARY.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
