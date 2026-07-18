#!/usr/bin/env python3
"""Rebuild the seven-band F2 statistic with frozen gradient fields.

This is a closed re-analysis of the original narrow-axis F2 cohort.  It keeps
the original seven primary bands and the exact accepted [0, 10] s window/event
rows, but replaces the legacy axis field with the frozen gradient-field
contract:

* use ``shared_a/shared_b`` when the complete shared pair exists;
* otherwise use ``own_a/own_b``;
* never choose between shared and own using the ictal result.

For every seizure and band, the observed maxAB score is compared with 1,000
draws from the original F2 spatial-null hierarchy (within shaft first, then
distance-bin and subject-wide fallback where needed).  Smoothing, mirror
selection and the A/B maximum are recomputed for every draw.  Seizures are
folded within subject first.  The seven-band FWER p values reuse the original
F2 Westfall--Young max-over-bands cohort statistic and the figure reuses the
original F2 painter.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_v2_phase1_figures import (  # noqa: E402
    plot_null_per_band_figure,
)
from scripts.run_topic5_old_cache_hybrid_field_comparison import (  # noqa: E402
    select_shared_else_own_scorers,
)
from scripts.run_topic5_tspectral_field_concordance import FIELD_ROOT, _seed  # noqa: E402
from scripts.run_topic5_v2_gates import _cohort_perm_ps  # noqa: E402
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.topic5_t0_features import window_activation  # noqa: E402
from src.topic5_template_axis_field import scorers_from_interictal_record  # noqa: E402
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    exact_name_align_matrix,
    fold_seizure_null_draws,
    jsonable,
    score_observed_bundle,
    score_permutation_matrix,
)


CONFIG = ROOT / "config/topic5_v2_phase1.yaml"
CACHE_ROOT = ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache"
REFERENCE_WINDOWS = (
    ROOT
    / "results/topic5_ictal_recruitment/v2_band_scan/narrow/"
      "phase1_alignment_raw_window_long.csv"
)
OUT = ROOT / "results/topic5_ictal_recruitment/gradient_multiband_significance"
PAPER = ROOT / "results/paper-ready-figure/fig3_gradient_multiband_significance"
PAPER_FIGURES = PAPER / "figures"

CONTRACT = "topic5_gradient_shared_else_own_multiband_onset_0_10_spatial_fwer_v1"
BASE_SEED = 20260718
MIN_PERM = 1000
MIN_CONTACTS = 6
MIN_GROUP_FOR_SHAFT = 4
EXPECTED_REFERENCE_N = 20
EXPECTED_ANALYSIS_N = 19
STEM = "gradient_multiband_significance_onset_0_10s"

_SPATIAL_TIER = {
    "within_shaft_strong": 3,
    "distance_bin_fallback": 2,
    "subject_wide_weak": 1,
}

BAND_LABELS = {
    "delta_HYP_slow": "δ\n1–4",
    "theta_preictal_PAC": "θ\n4–8",
    "alpha_sharp_leq13": "α\n8–13",
    "beta_LVFA_low": "β\n13–30",
    "gamma_LVFA": "γ\n30–80",
    "hg_low_ripple": "R\n80–150",
    "ripple_high": "FR\n150–250",
}


def load_primary_band_contract(path: Path = CONFIG) -> list[dict[str, object]]:
    """Read the exact seven-band family used by the original F2 producer."""
    cfg = yaml.safe_load(path.read_text())
    if cfg["bands"].get("primary_interval") != "half_open":
        raise ValueError("original primary-band interval is not half_open")
    rows = [
        {"band": str(name), "low_hz": float(lo), "high_hz": float(hi),
         "interval": "half_open", "label": BAND_LABELS.get(str(name))}
        for name, lo, hi in cfg["bands"]["primary"]
    ]
    if len(rows) != 7 or any(row["label"] is None for row in rows):
        raise ValueError("the original F2 seven-band family has drifted")
    return rows


def select_reference_window_rows(
    frame: pd.DataFrame,
    bands: Sequence[str],
    *,
    start_sec: float = 0.0,
    end_sec: float = 10.0,
) -> pd.DataFrame:
    """Select the original producer's accepted exact [0,10] s event rows."""
    used = frame["used_fixed_mask"]
    if used.dtype != bool:
        used = used.astype(str).str.lower().eq("true")
    keep = (
        frame["feature"].astype(str).eq("raw")
        & used
        & np.isclose(pd.to_numeric(frame["win_start_rel"], errors="coerce"), start_sec)
        & np.isclose(pd.to_numeric(frame["win_end_rel"], errors="coerce"), end_sec)
        & frame["band"].astype(str).isin(list(bands))
    )
    out = frame.loc[keep, ["subject", "seizure", "band", "ictal_fraction"]].copy()
    out["subject"] = out["subject"].astype(str)
    out["seizure"] = pd.to_numeric(out["seizure"], errors="raise").astype(int)
    duplicated = out.duplicated(["subject", "seizure", "band"], keep=False)
    if duplicated.any():
        raise ValueError("duplicate exact-window reference rows")
    return out.sort_values(["subject", "seizure", "band"]).reset_index(drop=True)


def band_window_activation(
    cache: Mapping[str, np.ndarray], band: str, seizure_idx: int,
) -> np.ndarray:
    """Return the cached robust-z contact activation over onset [0,10] s."""
    z_key = f"{band}__zt__{seizure_idx}"
    relt_key = f"{band}__relt__{seizure_idx}"
    if z_key not in cache or relt_key not in cache:
        raise KeyError(f"missing_activation:{z_key}+{relt_key}")
    return window_activation(
        np.asarray(cache[z_key], float),
        np.asarray(cache[relt_key], float),
        0.0,
        10.0,
    )


def make_original_spatial_permutations(
    contact_names: Sequence[str],
    coords: Sequence[Sequence[float]],
    matched_mask: Sequence[bool],
    n_perm: int,
    seed: int,
    *,
    min_group: int = MIN_GROUP_FOR_SHAFT,
) -> tuple[np.ndarray, str]:
    """Vectorize the original F2 within-shaft/distance/subject fallback null."""
    names = [str(value) for value in contact_names]
    xyz = np.asarray(coords, float)
    matched = np.asarray(matched_mask, bool)
    if xyz.shape != (len(names), 3) or matched.shape != (len(names),):
        raise ValueError("spatial permutation inputs are not contact aligned")
    if min_group < 2:
        raise ValueError("min_group must be >=2")

    finite_idx = [int(i) for i in np.where(matched)[0]]
    by_shaft: dict[str, list[int]] = {}
    for idx in finite_idx:
        by_shaft.setdefault(parse_shaft(names[idx])[0], []).append(idx)

    groups: list[list[int]] = []
    leftovers: list[int] = []
    for members in by_shaft.values():
        if len(members) >= min_group:
            groups.append(members)
        else:
            leftovers.extend(members)

    used_distance = False
    remaining = list(leftovers)
    while len(remaining) >= min_group:
        used_distance = True
        anchor = remaining[0]
        others = sorted(
            remaining[1:],
            key=lambda idx: (
                float(np.linalg.norm(xyz[idx] - xyz[anchor])),
                remaining.index(idx),
            ),
        )
        group = [anchor] + others[:min_group - 1]
        groups.append(group)
        chosen = set(group)
        remaining = [idx for idx in remaining if idx not in chosen]

    used_subject_wide = bool(remaining)
    if len(remaining) >= 2:
        groups.append(remaining)
    strength = (
        "subject_wide_weak" if used_subject_wide
        else "distance_bin_fallback" if used_distance
        else "within_shaft_strong"
    )

    rng = np.random.default_rng(int(seed))
    base = np.arange(len(names), dtype=int)
    permutations = np.tile(base, (int(n_perm), 1))
    for draw in range(int(n_perm)):
        for group in groups:
            idx = np.asarray(group, int)
            if len(idx) > 1:
                permutations[draw, idx] = rng.permutation(idx)
    return permutations, strength


def score_cached_activation_spatial(
    field_record: Mapping[str, object],
    scorers: Mapping[str, Mapping[str, object]],
    score_key: str,
    source_names: Sequence[str],
    activation: Sequence[float],
    *,
    subject: str,
    seizure_idx: int,
    band: str,
    n_perm: int,
    seed: int,
) -> dict[str, object]:
    """Score one event-band against the original spatial-null hierarchy."""
    aligned = exact_name_align_matrix(
        field_record, source_names, np.asarray(activation, float)[:, None]
    )["values"][:, 0]
    finite = np.isfinite(aligned)
    if int(finite.sum()) < MIN_CONTACTS:
        raise ValueError(f"fewer_than_{MIN_CONTACTS}_finite_contacts:{finite.sum()}")
    observed = score_observed_bundle(scorers, aligned)
    data = observed.get(score_key)
    if data is None or not np.isfinite(float(data)):
        raise ValueError(f"nonfinite_observed:{score_key}")
    field = field_record["interictal_field"]
    target_names = [str(value) for value in field["contact_order"]]
    perm_seed = _seed(
        f"gradient-multiband-spatial:{subject}:{seizure_idx}:{band}", seed
    )
    permutations, strength = make_original_spatial_permutations(
        target_names,
        field["coords"],
        finite,
        n_perm,
        perm_seed,
    )
    null = score_permutation_matrix(
        scorers, aligned[None, :], permutations, chunk_draws=100
    )[score_key][:, 0]
    if len(null) != n_perm or not np.isfinite(null).all():
        raise ValueError("nonfinite_or_incomplete_spatial_null")
    prefix = score_key.removesuffix("_maxab")
    return {
        "observed": float(data),
        "null": np.asarray(null, float),
        "null_median": float(np.median(null)),
        "null_p95": float(np.percentile(null, 95)),
        "a_abs": observed.get(f"{prefix}_a_abs"),
        "b_abs": observed.get(f"{prefix}_b_abs"),
        "best_template": observed.get(f"{prefix}_best_template"),
        "n_finite_contacts": int(finite.sum()),
        "permutation_seed": int(perm_seed),
        "spatial_null_strength": strength,
    }


def _resolve_subject_id(bare_subject: str) -> str:
    candidates = [
        name for name in (f"epilepsiae_{bare_subject}", f"yuquan_{bare_subject}")
        if (CACHE_ROOT / f"{name}.json").exists()
    ]
    if len(candidates) != 1:
        raise ValueError(f"cannot uniquely resolve dataset for subject:{bare_subject}")
    return candidates[0]


def _validate_field_record(record: Mapping[str, object], subject: str) -> None:
    if record.get("axis_definition") != "template_propagation_axis_v2":
        raise ValueError(f"{subject}:unexpected_axis_definition")
    if record.get("axis_direction_convention") != "positive_early_to_late":
        raise ValueError(f"{subject}:unexpected_axis_direction_convention")
    field = record.get("interictal_field") or {}
    if field.get("status") != "ok":
        raise ValueError(f"{subject}:interictal_field_{field.get('status')}")


def _subject_fold(
    subject: str,
    dataset: str,
    bare_subject: str,
    band: str,
    plane: str,
    score_key: str,
    events: Sequence[Mapping[str, object]],
    n_perm: int,
) -> tuple[dict[str, object], np.ndarray]:
    observed = float(np.median([float(event["observed"]) for event in events]))
    folded = fold_seizure_null_draws([
        np.asarray(event["null"], float)[:, None] for event in events
    ])[:, 0]
    if folded.shape != (n_perm,) or not np.isfinite(folded).all():
        raise ValueError(f"{subject}:{band}:invalid_subject_null")
    null_median = float(np.median(folded))
    strengths = [str(event["spatial_null_strength"]) for event in events]
    spatial_strength = min(strengths, key=lambda value: _SPATIAL_TIER[value])
    row = {
        "dataset": dataset,
        "subject": subject,
        "reference_subject": bare_subject,
        "band": band,
        "time_reference": (
            "clinical_onset" if dataset == "epilepsiae" else "eeg_onset_only"
        ),
        "window_start_sec": 0.0,
        "window_end_sec": 10.0,
        "field_plane": plane,
        "score_key": score_key,
        "observed_subject_median": observed,
        "spatial_null_subject_median": null_median,
        "delta": observed - null_median,
        "subject_empirical_one_sided_p": float(
            (1 + np.sum(folded >= observed - 1e-15)) / (len(folded) + 1)
        ),
        "n_seizures": len(events),
        "seizure_idxs": ";".join(str(event["seizure_idx"]) for event in events),
        "min_finite_contacts": min(int(event["n_finite_contacts"]) for event in events),
        "spatial_null_strength": spatial_strength,
        "n_spatial_null_draws": int(n_perm),
    }
    return row, folded


def _score_cohort(
    reference: pd.DataFrame,
    bands: Sequence[str],
    *,
    n_perm: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    event_rows: list[dict[str, object]] = []
    subject_rows: list[dict[str, object]] = []
    perm_rows: list[dict[str, object]] = []
    drops: list[dict[str, object]] = []
    routing: list[dict[str, object]] = []

    for bare_subject in sorted(reference["subject"].unique()):
        subject = _resolve_subject_id(bare_subject)
        dataset = subject.split("_", 1)[0]
        meta_path = CACHE_ROOT / f"{subject}.json"
        cache_path = CACHE_ROOT / f"{subject}.npz"
        field_path = FIELD_ROOT / f"{subject}.json"
        if not field_path.exists():
            drops.append({"subject": subject, "drop_reason": "missing_frozen_field"})
            continue
        meta = json.loads(meta_path.read_text())
        record = json.loads(field_path.read_text())
        try:
            _validate_field_record(record, subject)
            all_scorers = scorers_from_interictal_record(record)
            scorers, plane, score_key = select_shared_else_own_scorers(all_scorers)
        except Exception as exc:
            drops.append({
                "subject": subject,
                "drop_reason": f"field_unavailable:{type(exc).__name__}:{exc}",
            })
            continue

        if meta.get("analysis_channels_basis") != "primary_bands_validity":
            raise ValueError(f"{subject}:stale_analysis_channels_basis")
        analysis_channels = set(str(v) for v in meta.get("analysis_channels", []))
        target_order = [
            str(v) for v in record["interictal_field"]["contact_order"]
        ]
        routing.append({
            "dataset": dataset,
            "subject": subject,
            "field_plane": plane,
            "score_key": score_key,
            "n_frozen_contacts": len(target_order),
            "n_fixed_analysis_contacts": len(set(target_order) & analysis_channels),
            "fingerprint_sha256": record["interictal_field"].get("fingerprint_sha256"),
        })

        events_by_band: dict[str, list[dict[str, object]]] = defaultdict(list)
        with np.load(cache_path, allow_pickle=True) as cache:
            source_names = [str(v) for v in meta.get("channels", cache["channels"].tolist())]
            source_fixed = np.asarray([name in analysis_channels for name in source_names], bool)
            subject_ref = reference[reference["subject"] == bare_subject]
            for band in bands:
                band_ref = subject_ref[subject_ref["band"] == band]
                for ref_row in band_ref.itertuples(index=False):
                    seizure_idx = int(ref_row.seizure)
                    try:
                        activation = band_window_activation(cache, band, seizure_idx)
                        activation = np.asarray(activation, float).copy()
                        if len(activation) != len(source_names):
                            raise ValueError("cache_channel_dimension_mismatch")
                        activation[~source_fixed] = np.nan
                        scored = score_cached_activation_spatial(
                            record, scorers, score_key, source_names, activation,
                            subject=subject, seizure_idx=seizure_idx, band=band,
                            n_perm=n_perm, seed=seed,
                        )
                    except Exception as exc:
                        drops.append({
                            "subject": subject,
                            "seizure_idx": seizure_idx,
                            "band": band,
                            "drop_reason": f"score_failed:{type(exc).__name__}:{exc}",
                        })
                        continue
                    event = {
                        "dataset": dataset,
                        "subject": subject,
                        "reference_subject": bare_subject,
                        "seizure_idx": seizure_idx,
                        "band": band,
                        "ictal_fraction_from_original_producer": float(ref_row.ictal_fraction),
                        "time_reference": (
                            "clinical_onset" if dataset == "epilepsiae" else "eeg_onset_only"
                        ),
                        "window_start_sec": 0.0,
                        "window_end_sec": 10.0,
                        "field_plane": plane,
                        "score_key": score_key,
                        **scored,
                    }
                    events_by_band[band].append(event)
                    event_rows.append({
                        key: value for key, value in event.items() if key != "null"
                    })

        for band in bands:
            events = events_by_band.get(band, [])
            if not events:
                drops.append({
                    "subject": subject, "band": band,
                    "drop_reason": "no_resolvable_reference_events",
                })
                continue
            subject_row, folded = _subject_fold(
                subject, dataset, bare_subject, band, plane, score_key,
                events, n_perm,
            )
            subject_rows.append(subject_row)
            base = {
                "subject": subject,
                "feature": "raw_gradient",
                "null_type": "spatial",
                "band": band,
            }
            perm_rows.append({
                **base, "perm_id": -1,
                "perm_subject_median": subject_row["observed_subject_median"],
            })
            perm_rows.extend({
                **base, "perm_id": draw,
                "perm_subject_median": float(value),
            } for draw, value in enumerate(folded))

    return (
        pd.DataFrame(event_rows),
        pd.DataFrame(subject_rows),
        pd.DataFrame(perm_rows),
        pd.DataFrame(drops),
        routing,
    )


def build_cohort_table(
    subjects: pd.DataFrame,
    perm_rows: pd.DataFrame,
    band_contract: Sequence[Mapping[str, object]],
) -> pd.DataFrame:
    """Compute the original F2 cohort statistic and seven-band maxT FWER."""
    bands = [str(row["band"]) for row in band_contract]
    per_band_p, cohort_delta, fwer_p = _cohort_perm_ps(
        perm_rows,
        "raw_gradient",
        "spatial",
        bands,
        max_family=bands,
    )
    rows = []
    for contract in band_contract:
        band = str(contract["band"])
        frame = subjects[subjects["band"] == band].sort_values("subject")
        values = frame["delta"].to_numpy(float)
        rows.append({
            **contract,
            "n_subjects": int(len(frame)),
            "n_epilepsiae_subjects": int((frame["dataset"] == "epilepsiae").sum()),
            "n_yuquan_subjects": int((frame["dataset"] == "yuquan").sum()),
            "n_shared_subjects": int((frame["field_plane"] == "shared").sum()),
            "n_own_fallback_subjects": int((frame["field_plane"] == "own_fallback").sum()),
            "n_seizures": int(frame["n_seizures"].sum()),
            "n_within_shaft_strong_subjects": int(
                (frame["spatial_null_strength"] == "within_shaft_strong").sum()
            ),
            "n_distance_bin_fallback_subjects": int(
                (frame["spatial_null_strength"] == "distance_bin_fallback").sum()
            ),
            "n_subject_wide_weak_subjects": int(
                (frame["spatial_null_strength"] == "subject_wide_weak").sum()
            ),
            "cohort_perm_p_spatial": float(per_band_p[band]),
            "cohort_perm_delta_spatial": float(cohort_delta[band]),
            "max_over_bands_p": float(fwer_p[band]),
            "passes_fwer_0p05": bool(float(fwer_p[band]) < 0.05),
            "median_subject_delta": float(np.median(values)),
            "iqr_subject_delta_low": float(np.percentile(values, 25)),
            "iqr_subject_delta_high": float(np.percentile(values, 75)),
            "n_subject_delta_positive": int(np.sum(values > 0)),
        })
    return pd.DataFrame(rows)


def plot_gradient_multiband_figure(
    subjects: pd.DataFrame,
    cohort: pd.DataFrame,
    band_contract: Sequence[Mapping[str, object]],
    output_path: Path,
    *,
    seed: int,
) -> Path:
    """Call the original F2 painter with a paper-safe annotation layout."""
    bands = [str(row["band"]) for row in band_contract]
    labels = {str(row["band"]): str(row["label"]) for row in band_contract}
    subject_deltas = {
        band: subjects.loc[subjects["band"] == band, "delta"].to_numpy(float)
        for band in bands
    }
    cohort_by_band = cohort.set_index("band")
    passed_n = int(cohort["passes_fwer_0p05"].sum())
    return plot_null_per_band_figure(
        bands,
        labels,
        subject_deltas,
        cohort_by_band["cohort_perm_delta_spatial"].to_dict(),
        cohort_by_band["max_over_bands_p"].to_dict(),
        cohort_by_band["n_subjects"].to_dict(),
        f"Gradient-field multiband concordance · onset 0–10 s "
        f"(n=19) · {passed_n}/7 pass FWER",
        output_path,
        ylabel="Gradient-field alignment − spatial-null median\n(subject-level Δ)",
        save_pdf=True,
        seed=seed,
        figsize=(11.8, 6.8),
        show_exact_annotations=False,
        significance_legend="band passes 7-band FWER",
        nonsignificance_legend="n.s. band",
        cohort_legend="cohort Δ (tested)",
        subject_legend="per-subject Δ",
        title_mode="figure",
        layout_rect=(0.01, 0.01, 0.82, 0.95),
        xtick_fontsize=12.5,
        ytick_fontsize=11.5,
        ylabel_fontsize=13,
        title_fontsize=14,
        legend_fontsize=10.5,
    )


def _write_readme(cohort: pd.DataFrame) -> Path:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    passed = int(cohort["passes_fwer_0p05"].sum())
    detail = "; ".join(
        f"{row.band}: pFWER={row.max_over_bands_p:.4g}, "
        f"Δ={row.cohort_perm_delta_spatial:.3f}"
        for row in cohort.itertuples()
    )
    path = PAPER_FIGURES / "README.md"
    path.write_text(f"""# Topic 5 gradient 多频带显著性图

### {STEM}.png / {STEM}.pdf

严格复用旧 F2 的 7 个 primary 频带、单轴 violin、逐患者点、黑色 cohort Δ 横杠和 7-band max-over-bands FWER 标注。激活量来自旧 producer 已接纳的 onset 后 `[0,10] s` 事件窗口；冻结 gradient field 按结果无关的可用性固定路由：有完整 `shared_a/shared_b` 时取 shared maxAB，否则取 own maxAB，不在 shared 与 own 之间择优。Null 复用旧 producer 的三层空间置换：优先杆内洗牌，杆内触点不足时依次退到 distance-bin 与 subject-wide；每个发作 1000 次，每次都重新做平滑、mirror 与 A/B max，随后按 seizure→subject→cohort 折叠。

**关注点**：n=19（Epilepsiae 17 + Yuquan 2；shared 8、own fallback 11），{passed}/7 个频带通过 FWER。每频带的 spatial-null 强度构成为 within-shaft strong 2、distance-bin fallback 4、subject-wide weak 13，因此这是沿用旧 F2 合同的弱空间-null cohort 证据，不能写成 19 人都通过纯杆内 null。{detail}。Epilepsiae 的 0 点为 clinical onset；Yuquan 没有 clinical-onset 标注，只保留真实 EEG onset，因此图题写 `onset 0–10 s`，不把 19 人统称为 clinical onset。
""")
    return path


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.n_perm < MIN_PERM:
        raise ValueError(f"n_perm must be >= {MIN_PERM}")
    band_contract = load_primary_band_contract()
    bands = [str(row["band"]) for row in band_contract]
    reference_all = pd.read_csv(REFERENCE_WINDOWS)
    reference = select_reference_window_rows(reference_all, bands)
    if reference["subject"].nunique() != EXPECTED_REFERENCE_N:
        raise RuntimeError("original F2 exact-window reference is no longer n=20")

    events, subjects, perm_rows, drops, routing = _score_cohort(
        reference, bands, n_perm=args.n_perm, seed=args.seed,
    )
    if subjects.empty:
        raise RuntimeError("no gradient multiband subject results")
    cohort = build_cohort_table(subjects, perm_rows, band_contract)
    if not (cohort["n_subjects"] == EXPECTED_ANALYSIS_N).all():
        counts = dict(zip(cohort["band"], cohort["n_subjects"]))
        raise RuntimeError(f"gradient multiband denominator drifted from n=19:{counts}")

    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    paths = {
        "event": OUT / f"{STEM}_event.csv",
        "subject": OUT / f"{STEM}_subject.csv",
        "cohort": OUT / f"{STEM}_cohort.csv",
        "drops": OUT / f"{STEM}_drop_inventory.csv",
        "routing": OUT / f"{STEM}_field_routing.csv",
        "null_draws": OUT / f"{STEM}_subject_spatial_null_draws.parquet",
    }
    events.sort_values(["band", "subject", "seizure_idx"]).to_csv(paths["event"], index=False)
    subjects.sort_values(["band", "subject"]).to_csv(paths["subject"], index=False)
    cohort.to_csv(paths["cohort"], index=False)
    drops.to_csv(paths["drops"], index=False)
    pd.DataFrame(routing).sort_values("subject").to_csv(paths["routing"], index=False)
    perm_rows.to_parquet(paths["null_draws"], index=False)

    passed = int(cohort["passes_fwer_0p05"].sum())
    png = PAPER_FIGURES / f"{STEM}.png"
    plot_gradient_multiband_figure(subjects, cohort, band_contract, png, seed=args.seed)
    _write_readme(cohort)

    counts = {
        "reference_subjects": int(reference["subject"].nunique()),
        "analysis_subjects": int(subjects["subject"].nunique()),
        "epilepsiae_subjects": int(subjects.drop_duplicates("subject")["dataset"].eq("epilepsiae").sum()),
        "yuquan_subjects": int(subjects.drop_duplicates("subject")["dataset"].eq("yuquan").sum()),
        "shared_subjects": int(subjects.drop_duplicates("subject")["field_plane"].eq("shared").sum()),
        "own_fallback_subjects": int(subjects.drop_duplicates("subject")["field_plane"].eq("own_fallback").sum()),
        "bands_passing_fwer": passed,
    }
    summary = {
        "contract": CONTRACT,
        "paper_status": "exploratory_gradient_axis_reanalysis_of_original_f2",
        "source_figure_producer": "scripts/plot_topic5_v2_phase1_figures.py::fig2_null_perband",
        "source_event_eligibility": str(REFERENCE_WINDOWS.relative_to(ROOT)),
        "axis_definition": "template_propagation_axis_v2",
        "axis_direction_convention": "positive_early_to_late",
        "field_routing": "shared_a/shared_b if complete else own_a/own_b",
        "routing_is_outcome_independent": True,
        "activation": {
            "cache": str(CACHE_ROOT.relative_to(ROOT)),
            "window_sec": [0.0, 10.0],
            "epilepsiae_time_zero": "clinical_onset",
            "yuquan_time_zero": "eeg_onset_only",
            "primary_bands": band_contract,
            "analysis_mask": "original primary_bands_validity intersect frozen field contact_order",
        },
        "null": {
            "mode": "original_spatial_hierarchy",
            "tiers": [
                "within_shaft_strong",
                "distance_bin_fallback",
                "subject_wide_weak",
            ],
            "min_group_for_shaft": MIN_GROUP_FOR_SHAFT,
            "n_draws_per_seizure_band": int(args.n_perm),
            "mirror_reselected_each_draw": True,
            "ab_max_reselected_each_draw": True,
            "folding": "seizure median within subject for every draw; cohort median over subjects",
        },
        "fwer": {
            "family": bands,
            "method": "original F2 null-centered Westfall-Young max-over-bands cohort statistic",
            "alpha": 0.05,
        },
        "counts": counts,
        "cohort_statistics": cohort.to_dict("records"),
        "drops": drops.to_dict("records"),
        "outputs": {
            key: str(path.relative_to(ROOT)) for key, path in paths.items()
        } | {
            "figure_png": str(png.relative_to(ROOT)),
            "figure_pdf": str(png.with_suffix(".pdf").relative_to(ROOT)),
            "figure_readme": str((PAPER_FIGURES / "README.md").relative_to(ROOT)),
        },
    }
    summary_path = OUT / f"{STEM}_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n")
    (PAPER / paths["cohort"].name).write_text(paths["cohort"].read_text())
    (PAPER / paths["subject"].name).write_text(paths["subject"].read_text())
    (PAPER / summary_path.name).write_text(summary_path.read_text())
    print(cohort.to_string(index=False), flush=True)
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {png}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=MIN_PERM)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
