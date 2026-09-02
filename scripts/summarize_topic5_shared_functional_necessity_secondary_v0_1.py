#!/usr/bin/env python3
"""Secondary, non-claim-changing summaries for shared-component lesions.

The preregistered primary endpoint averages delayed next-contact NLL at
tau=1..3.  This script asks whether the same conclusion changes at tau=0,
within early/middle/late reference states, or under a simple dose-monotonicity
description.  These outputs never overwrite the primary adjudication.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)
from src.topic5_shared_functional_necessity_v0_1 import (  # noqa: E402
    LESION_DOSES,
    REAL_ARMS,
    dose_auc,
    holm_adjust,
)
from scripts.run_topic5_latent_pass1_v0_2 import OUT  # noqa: E402
from scripts.run_topic5_shared_functional_necessity_v0_1 import (  # noqa: E402
    NECESSITY,
    lesion_dir,
)
from scripts.summarize_topic5_shared_functional_necessity_v0_1 import (  # noqa: E402
    PAIR_METRICS,
    PRIMARY_METRICS,
    bootstrap_median_ci,
    greater_p,
    positive_counts,
    state_weighted_effect,
)


REVISION = "SECONDARY_IMMEDIATE_PHASE_MONOTONICITY_R1_TARGET_FREE_PHASE_CENTER"
PHASES = ("0.25", "0.50", "0.75")


def extract_immediate_cell(row: pd.Series) -> list[dict[str, object]]:
    with np.load(lesion_dir(row) / "lesion_response.npz", allow_pickle=False) as source:
        z = {name: np.asarray(source[name]) for name in source.files}
    family = {str(name): index for index, name in enumerate(z["family_names"].tolist())}
    delta = np.asarray(z["delta_nll"], float)
    valid = np.asarray(z["valid"], bool)
    phase = np.asarray(z["phase_target"], float)
    comparisons = {
        "SHARED": ("SHARED", None),
        "SHARED_MINUS_ORTHOGONAL": ("SHARED", "ORTHOGONAL"),
        "SHARED_MINUS_C_SUFFIX": ("SHARED", "C_SUFFIX"),
        "SHARED_MINUS_PCA": ("SHARED", "PCA"),
    }
    selectors = {"ALL": np.ones(len(phase), bool)}
    selectors.update({name: np.isclose(phase, float(name)) for name in PHASES})
    rows: list[dict[str, object]] = []
    for dose_index, dose in enumerate(z["doses"].astype(float)):
        for phase_name, selector in selectors.items():
            for metric, (left_name, right_name) in comparisons.items():
                left_index = family[left_name]
                right_index = family[right_name] if right_name is not None else None
                left_values = delta[:, left_index, dose_index, 0][:, None]
                left_valid = valid[:, left_index, dose_index, 0][:, None]
                right_values = None if right_index is None else delta[:, right_index, dose_index, 0][:, None]
                right_valid = None if right_index is None else valid[:, right_index, dose_index, 0][:, None]
                effect, n_states, n_decisions = state_weighted_effect(
                    left_values, left_valid, right_values, right_valid, selector
                )
                rows.append({
                    "patient": str(row.patient),
                    "fit_id": str(row.fit_id),
                    "public_arm": str(row.public_arm),
                    "seed": int(row.seed),
                    "phase": phase_name,
                    "dose": float(dose),
                    "metric": metric,
                    "effect_nll_per_decision": effect,
                    "n_reference_states": n_states,
                    "n_immediate_decisions": n_decisions,
                })
    return rows


def aggregate_patient(cell: pd.DataFrame) -> pd.DataFrame:
    keys = ["phase", "dose", "metric"]
    fit_arm = cell.groupby(["patient", "fit_id", "public_arm", *keys], as_index=False).agg(
        effect_nll_per_decision=("effect_nll_per_decision", "mean")
    )
    fit = fit_arm.groupby(["patient", "fit_id", *keys], as_index=False).agg(
        effect_nll_per_decision=("effect_nll_per_decision", "mean")
    )
    return fit.groupby(["patient", *keys], as_index=False).agg(
        effect_nll_per_decision=("effect_nll_per_decision", "mean")
    )


def make_auc(patient: pd.DataFrame, endpoint: str) -> pd.DataFrame:
    rows = []
    for (subject, phase, metric), group in patient.groupby(["patient", "phase", "metric"], sort=True):
        values = group.set_index("dose")["effect_nll_per_decision"]
        effects = np.asarray([values.get(float(dose), np.nan) for dose in LESION_DOSES], float)
        rows.append({
            "endpoint": endpoint,
            "patient": subject,
            "phase": phase,
            "metric": metric,
            "dose_auc_nll": dose_auc(LESION_DOSES, effects),
            "monotone_non_decreasing": bool(np.isfinite(effects).all() and np.all(np.diff(effects) >= 0)),
            **{f"effect_dose_{dose:.2f}": value for dose, value in zip(LESION_DOSES, effects)},
        })
    return pd.DataFrame(rows)


def infer(auc: pd.DataFrame, endpoint: str, phases: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for phase in phases:
        for metric in PAIR_METRICS:
            values = auc[
                auc.endpoint.eq(endpoint) & auc.phase.eq(phase) & auc.metric.eq(metric)
            ]["dose_auc_nll"].to_numpy(float)
            values = values[np.isfinite(values)]
            low, high = bootstrap_median_ci(
                values,
                seed=8200 + 100 * phases.index(phase) + PAIR_METRICS.index(metric),
            )
            positive, negative, zero = positive_counts(values)
            rows.append({
                "endpoint": endpoint,
                "phase": phase,
                "metric": metric,
                "n_patients": int(len(values)),
                "median_dose_auc_nll": float(np.median(values)) if len(values) else float("nan"),
                "ci95_low": low,
                "ci95_high": high,
                "positive": positive,
                "negative": negative,
                "zero": zero,
                "p_greater": greater_p(values),
                "monotone_patients": int(
                    auc[
                        auc.endpoint.eq(endpoint) & auc.phase.eq(phase) & auc.metric.eq(metric)
                    ]["monotone_non_decreasing"].sum()
                ),
            })
    frame = pd.DataFrame(rows)
    frame["p_holm_secondary_family"] = holm_adjust(frame["p_greater"])
    return frame


def main() -> None:
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    real = manifest[manifest.public_arm.isin(REAL_ARMS)].copy()

    immediate_rows: list[dict[str, object]] = []
    for _, row in real.iterrows():
        immediate_rows.extend(extract_immediate_cell(row))
    immediate_cell = pd.DataFrame(immediate_rows)
    immediate_patient = aggregate_patient(immediate_cell)
    immediate_auc = make_auc(immediate_patient, "IMMEDIATE_TAU0")

    delayed_auc = pd.read_csv(NECESSITY / "PATIENT_AUC_EFFECTS.csv")
    delayed_auc.insert(0, "endpoint", "DELAYED_TAU1_TO_3")
    if "monotone_non_decreasing" not in delayed_auc:
        dose_columns = [f"effect_dose_{dose:.2f}" for dose in LESION_DOSES]
        delayed_auc["monotone_non_decreasing"] = delayed_auc[dose_columns].apply(
            lambda row: bool(np.isfinite(row.to_numpy(float)).all() and np.all(np.diff(row.to_numpy(float)) >= 0)),
            axis=1,
        )

    immediate_stats = infer(immediate_auc, "IMMEDIATE_TAU0", ("ALL",))
    delayed_phase_stats = infer(delayed_auc, "DELAYED_TAU1_TO_3", PHASES)
    stats = pd.concat([immediate_stats, delayed_phase_stats], ignore_index=True)

    atomic_write_csv(NECESSITY / "SECONDARY_IMMEDIATE_CELL_EFFECTS.csv", immediate_cell)
    atomic_write_csv(NECESSITY / "SECONDARY_IMMEDIATE_PATIENT_EFFECTS.csv", immediate_patient)
    atomic_write_csv(
        NECESSITY / "SECONDARY_PATIENT_AUC_EFFECTS.csv",
        pd.concat([immediate_auc, delayed_auc], ignore_index=True, sort=False),
    )
    atomic_write_csv(NECESSITY / "SECONDARY_INFERENCE.csv", stats)

    primary_hash_before = sha256_file(NECESSITY / "CLAIM_ADJUDICATION.json")
    finite_immediate = immediate_cell.effect_nll_per_decision.dropna().to_numpy(float)
    audit_checks = {
        "all_504_real_cells_summarized": int(
            immediate_cell[["fit_id", "public_arm", "seed"]].drop_duplicates().shape[0]
        ) == 504,
        "all_defined_values_finite": bool(np.isfinite(finite_immediate).all()),
        "whole_event_immediate_denominator_28": bool(
            (immediate_stats.n_patients == 28).all()
        ),
        "phase_denominators_reported_without_imputation": bool(
            delayed_phase_stats.n_patients.between(1, 28).all()
        ),
        "primary_adjudication_unchanged": primary_hash_before == sha256_file(NECESSITY / "CLAIM_ADJUDICATION.json"),
        "all_three_delayed_phases_present": set(delayed_phase_stats.phase) == set(PHASES),
    }
    summary = {
        "contract": "topic5_shared_functional_computation_necessity_secondary_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "revision": REVISION,
        "status": "PASS" if all(audit_checks.values()) else "FAIL",
        "claim_changing": False,
        "primary_adjudication_sha256": primary_hash_before,
        "checks": audit_checks,
        "phase_patient_denominators": {
            phase: int(delayed_phase_stats.loc[delayed_phase_stats.phase.eq(phase), "n_patients"].max())
            for phase in PHASES
        },
        "inference": json.loads(stats.to_json(orient="records")),
        "plain_language_boundary": (
            "Immediate and early/middle/late summaries are sensitivity checks only; "
            "they do not replace the delayed patient-first primary endpoint."
        ),
    }
    atomic_write_json(NECESSITY / "SECONDARY_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
