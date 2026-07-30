#!/usr/bin/env python3
"""Gate 0: can seizure-specific early-ictal residuals be measured reliably?

The accepted 1--150 Hz target stores one [0,10] s field per seizure, so it can
test patient-mean stability but cannot by itself estimate within-seizure
reliability of a seizure-specific residual.  A separately labelled 1--45 Hz
time-resolved cache supplies a 0--5 versus 5--10 s proxy audit; it does not
silently replace the accepted 1--150 Hz endpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
FIT1 = (
    ROOT
    / "results/topic5_state_conditioned_predictor/fit12_clinical_bb150/"
    "fit1/fig6_fit1_clinical_onset_scaffold_event.csv"
)
EXACT = ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
PROXY = ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_v2_windows"
OUT = (
    ROOT
    / "results/topic5_minimal_sequence_kernel_closeout/"
    "when_gate0_early_ictal_reliability_v0_2"
)
MIN_SEIZURES = 4
MIN_CONTACTS = 6


def _strict_inventory() -> dict[str, list[int]]:
    frame = pd.read_csv(FIT1)
    strict = frame.loc[
        (frame.group_id == "strict_broadband")
        & (frame.time_reference == "clinical_onset")
    ]
    inventory = {
        str(subject): sorted(group.seizure_idx.astype(int).unique().tolist())
        for subject, group in strict.groupby("subject")
    }
    if len(inventory) != 16 or sum(map(len, inventory.values())) != 106:
        raise RuntimeError("strict clinical-onset 16-patient/106-seizure contract drifted")
    return inventory


def _rho(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, float)
    b = np.asarray(right, float)
    keep = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(keep) < MIN_CONTACTS:
        return np.nan
    if np.nanstd(a[keep]) == 0 or np.nanstd(b[keep]) == 0:
        return np.nan
    return float(spearmanr(a[keep], b[keep]).statistic)


def _bootstrap_median(values: np.ndarray, seed: int) -> list[float]:
    data = np.asarray(values, float)
    data = data[np.isfinite(data)]
    if not len(data):
        return [np.nan, np.nan]
    rng = np.random.default_rng(seed)
    draws = np.median(
        data[rng.integers(0, len(data), size=(20_000, len(data)))], axis=1
    )
    return np.quantile(draws, [0.025, 0.975]).tolist()


def _balanced_split_reliability(
    fields: np.ndarray, *, seed: int, draws: int = 1000
) -> np.ndarray:
    n = len(fields)
    if n < MIN_SEIZURES:
        return np.asarray([], float)
    rng = np.random.default_rng(seed)
    size = n // 2
    values = []
    for _ in range(draws):
        left = np.sort(rng.choice(n, size=size, replace=False))
        right = np.setdiff1d(np.arange(n), left)
        values.append(_rho(np.nanmean(fields[left], axis=0), np.nanmean(fields[right], axis=0)))
    return np.asarray(values, float)


def _window_field(trace: np.ndarray, rel_time: np.ndarray, start: float, stop: float):
    mask = (rel_time >= start) & (rel_time < stop)
    if np.count_nonzero(mask) < 2:
        return np.full(trace.shape[0], np.nan)
    return np.nanmean(trace[:, mask], axis=1)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = _strict_inventory()
    exact_rows = []
    proxy_rows = []
    seizure_rows = []
    for patient_index, (subject, seizure_indices) in enumerate(sorted(inventory.items())):
        exact_path = EXACT / f"{subject}.npz"
        proxy_path = PROXY / f"{subject}.npz"
        if not exact_path.exists():
            raise FileNotFoundError(exact_path)
        with np.load(exact_path, allow_pickle=False) as data:
            exact_fields = np.stack(
                [np.asarray(data[f"bb150_auc__{index}"], float) for index in seizure_indices]
            )
            n_contacts = exact_fields.shape[1]
        loso = []
        if len(exact_fields) >= 2:
            for index in range(len(exact_fields)):
                other = np.delete(exact_fields, index, axis=0)
                loso.append(_rho(exact_fields[index], np.nanmean(other, axis=0)))
        split = _balanced_split_reliability(
            exact_fields, seed=2026073000 + patient_index
        )
        exact_rows.append(
            {
                "subject": subject,
                "n_seizures": len(seizure_indices),
                "n_contacts": n_contacts,
                "eligible_split_half": len(seizure_indices) >= MIN_SEIZURES,
                "exact_bb150_loso_patient_mean_rho_median": float(np.nanmedian(loso))
                if len(loso)
                else np.nan,
                "exact_bb150_split_half_rho_median": float(np.nanmedian(split))
                if len(split)
                else np.nan,
                "exact_bb150_split_half_rho_q025": float(np.nanquantile(split, 0.025))
                if len(split)
                else np.nan,
                "exact_bb150_split_half_rho_q975": float(np.nanquantile(split, 0.975))
                if len(split)
                else np.nan,
            }
        )

        if not proxy_path.exists():
            continue
        half_a = []
        half_b = []
        kept_indices = []
        with np.load(proxy_path, allow_pickle=False) as data:
            for seizure_index in seizure_indices:
                trace_key = f"bb_zt__{seizure_index}"
                time_key = f"bb_relt__{seizure_index}"
                if trace_key not in data.files or time_key not in data.files:
                    continue
                trace = np.asarray(data[trace_key], float)
                rel_time = np.asarray(data[time_key], float)
                half_a.append(_window_field(trace, rel_time, 0.0, 5.0))
                half_b.append(_window_field(trace, rel_time, 5.0, 10.0))
                kept_indices.append(seizure_index)
        if len(half_a) < MIN_SEIZURES:
            proxy_rows.append(
                {
                    "subject": subject,
                    "n_seizures": len(half_a),
                    "eligible_proxy_residual": False,
                }
            )
            continue
        half_a = np.stack(half_a)
        half_b = np.stack(half_b)
        residual_a = []
        residual_b = []
        matched = []
        for index in range(len(half_a)):
            other = np.arange(len(half_a)) != index
            current_a = half_a[index] - np.nanmean(half_a[other], axis=0)
            current_b = half_b[index] - np.nanmean(half_b[other], axis=0)
            residual_a.append(current_a)
            residual_b.append(current_b)
            matched.append(_rho(current_a, current_b))
            seizure_rows.append(
                {
                    "subject": subject,
                    "seizure_idx": kept_indices[index],
                    "proxy_residual_half_rho": matched[-1],
                }
            )
        residual_a = np.stack(residual_a)
        residual_b = np.stack(residual_b)
        mismatched = [
            _rho(residual_a[left], residual_b[right])
            for left in range(len(residual_a))
            for right in range(len(residual_b))
            if left != right
        ]
        proxy_rows.append(
            {
                "subject": subject,
                "n_seizures": len(half_a),
                "n_contacts": half_a.shape[1],
                "eligible_proxy_residual": True,
                "proxy_patient_mean_half_rho": _rho(
                    np.nanmean(half_a, axis=0), np.nanmean(half_b, axis=0)
                ),
                "proxy_residual_matched_rho_median": float(np.nanmedian(matched)),
                "proxy_residual_mismatched_rho_median": float(np.nanmedian(mismatched)),
                "proxy_residual_matched_minus_mismatched": float(
                    np.nanmedian(matched) - np.nanmedian(mismatched)
                ),
            }
        )

    exact = pd.DataFrame(exact_rows)
    proxy = pd.DataFrame(proxy_rows)
    seizures = pd.DataFrame(seizure_rows)
    exact.to_csv(OUT / "exact_bb150_patient_mean_reliability.csv", index=False)
    proxy.to_csv(OUT / "proxy_bb45_residual_reliability.csv", index=False)
    seizures.to_csv(OUT / "proxy_bb45_seizure_residual_reliability.csv", index=False)

    exact_eligible = exact.loc[exact.eligible_split_half]
    proxy_eligible = proxy.loc[proxy.eligible_proxy_residual == True]  # noqa: E712
    proxy_values = proxy_eligible.proxy_residual_matched_minus_mismatched.to_numpy()
    proxy_ci = _bootstrap_median(proxy_values, 20260730)
    payload = {
        "status": "COMPLETE",
        "contract": "topic5_minimal_sequence_kernel_closeout_v0_2",
        "strict_target": {
            "band_hz": [1, 150],
            "window_sec": [0, 10],
            "n_patients": len(inventory),
            "n_seizures": sum(map(len, inventory.values())),
            "patients_with_at_least_4_seizures": int(len(exact_eligible)),
            "patient_mean_split_half_rho_median": float(
                exact_eligible.exact_bb150_split_half_rho_median.median()
            ),
            "seizure_specific_residual_reliability": "UNIDENTIFIABLE_FROM_ONE_AGGREGATED_FIELD_PER_SEIZURE",
        },
        "time_resolved_proxy": {
            "band_hz": [1, 45],
            "halves_sec": [[0, 5], [5, 10]],
            "n_patients": int(len(proxy_eligible)),
            "patient_mean_half_rho_median": float(
                proxy_eligible.proxy_patient_mean_half_rho.median()
            ),
            "residual_matched_minus_mismatched_median": float(
                np.nanmedian(proxy_values)
            ),
            "residual_matched_minus_mismatched_bootstrap_ci95": proxy_ci,
            "proxy_reliability_lower_bound_above_zero": bool(proxy_ci[0] > 0),
        },
        "gate0_verdict": "BLOCKED_EXACT_BB150_SEIZURE_RESIDUAL_RELIABILITY_UNIDENTIFIABLE",
        "interpretation": (
            "Patient-mean target stability and a BB45 within-seizure proxy are "
            "reported separately. The proxy cannot unlock seizure-specific "
            "BB150 dynamic prediction."
        ),
        "target_values_read": True,
    }
    (OUT / "GATE0_SUMMARY.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
