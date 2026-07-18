#!/usr/bin/env python3
"""Run the original narrow F2 statistic with only the field scorer replaced.

The original F2 window/event eligibility, window -> seizure -> subject fold,
spatial-null hierarchy, 1,000-draw subject-coherent permutation structure and
seven-band Westfall--Young FWER family are retained.  The intended scientific
change is the interictal readout: use the frozen gradient shared A/B field when
available, otherwise the frozen gradient own A/B field.

E916 cannot enter because its frozen gradient field is unavailable.  The old
endpoint-plane coordinates are retained for the distance-bin null fallback so
the spatial randomisation is not silently redefined by the new axis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_v2_phase1_figures import plot_null_per_band_figure  # noqa: E402
from scripts.run_topic5_gradient_multiband_significance import (  # noqa: E402
    BAND_LABELS,
    CACHE_ROOT,
    FIELD_ROOT,
    MIN_GROUP_FOR_SHAFT,
    _resolve_subject_id,
    _validate_field_record,
    build_cohort_table,
    load_primary_band_contract,
)
from scripts.run_topic5_ictal_field_dynamics import load_context  # noqa: E402
from scripts.run_topic5_old_cache_hybrid_field_comparison import (  # noqa: E402
    select_shared_else_own_scorers,
)
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.topic5_t0_features import window_activation  # noqa: E402
from src.topic5_template_axis_field import (  # noqa: E402
    make_field_scorer,
    score_scorer_bundle_batch,
    scorers_from_interictal_record,
)
from src.topic5_tspectral_field_concordance import jsonable  # noqa: E402


REFERENCE_WINDOWS = (
    ROOT
    / "results/topic5_ictal_recruitment/v2_band_scan/narrow/"
      "phase1_alignment_raw_window_long.csv"
)
OUT = ROOT / "results/topic5_ictal_recruitment/gradient_multiband_original_f2"
PAPER = ROOT / "results/paper-ready-figure/fig3_gradient_multiband_significance"
PAPER_FIGURES = PAPER / "figures"

CONTRACT = "topic5_gradient_shared_else_own_original_f2_statistics_v1"
STEM = "gradient_multiband_significance_original_f2_windows"
BASE_SEED = 20260701
MIN_PERM = 1000
MIN_CONTACTS = 4
EXPECTED_REFERENCE_N = 20
EXPECTED_ANALYSIS_N = 19
EXPECTED_WINDOWS = ((-5.0, 5.0), (0.0, 10.0), (5.0, 15.0),
                    (10.0, 20.0), (15.0, 25.0))

_SPATIAL_TIER = {
    "within_shaft_strong": 3,
    "distance_bin_fallback": 2,
    "subject_wide_weak": 1,
}


def select_gradient_scorers(
    field_record: Mapping[str, object],
    *,
    smoothing_policy: str = "frozen_per_model",
) -> tuple[dict, str, str, dict]:
    """Select shared-else-own scorers under an explicit smoothing policy.

    ``frozen_per_model`` is the original frozen-gradient implementation: each
    own plane retains its independently estimated bandwidth.  ``subject_fixed``
    restores the old endpoint-F2 smoothing rule: choose one subject-level
    bandwidth from the selected A plane and rebuild both A and B scorers with
    that same value.  Shared A/B already occupy one plane, so this operation is
    numerically identical there apart from deterministic reconstruction.
    """
    # Validate the serialized contract and fingerprint before rebuilding any
    # weights.  The returned dictionaries are used directly by the per-model
    # branch and serve as the availability truth for both branches.
    frozen = scorers_from_interictal_record(field_record)
    scorers, plane, score_key = select_shared_else_own_scorers(frozen)
    selected_keys = ("shared_a", "shared_b") if plane == "shared" else ("own_a", "own_b")
    sigma_original = {key: float(frozen[key]["sigma"]) for key in selected_keys}
    if smoothing_policy == "frozen_per_model":
        return scorers, plane, score_key, {
            "smoothing_policy": smoothing_policy,
            "sigma_reference": "frozen_per_model",
            "sigma_common": None,
            "sigma_original": sigma_original,
        }
    if smoothing_policy != "subject_fixed":
        raise ValueError(f"unsupported smoothing policy:{smoothing_policy}")

    return rebuild_subject_fixed_scorers(field_record["interictal_field"], frozen)


def rebuild_subject_fixed_scorers(
    field: Mapping[str, object],
    frozen_scorers: Mapping[str, Mapping[str, object]],
) -> tuple[dict, str, str, dict]:
    """Pure fixed-sigma rebuild after the frozen record has been validated."""
    _, plane, score_key = select_shared_else_own_scorers(frozen_scorers)
    selected_keys = ("shared_a", "shared_b") if plane == "shared" else ("own_a", "own_b")
    sigma_original = {
        key: float(frozen_scorers[key]["sigma"]) for key in selected_keys
    }

    prefix = "shared" if plane == "shared" else "own"
    plane_a = "shared" if plane == "shared" else "own_a"
    plane_b = "shared" if plane == "shared" else "own_b"
    sigma = float(field["planes"][plane_a]["sigma"])
    fixed = {
        f"{prefix}_a": make_field_scorer(
            field["earliness_a"], field["planes"][plane_a]["points"],
            field["support_a"], sigma,
        ),
        f"{prefix}_b": make_field_scorer(
            field["earliness_b"], field["planes"][plane_b]["points"],
            field["support_b"], sigma,
        ),
    }
    return fixed, plane, score_key, {
        "smoothing_policy": "subject_fixed",
        "sigma_reference": f"{plane_a}_median_nearest_neighbor_spacing",
        "sigma_common": sigma,
        "sigma_original": sigma_original,
    }


def select_original_f2_rows(frame: pd.DataFrame, bands: Sequence[str]) -> pd.DataFrame:
    """Select every primary-band window accepted by the original F2 producer."""
    used = frame["used_fixed_mask"]
    if used.dtype != bool:
        used = used.astype(str).str.lower().eq("true")
    keep = (
        frame["feature"].astype(str).eq("raw")
        & used
        & frame["band"].astype(str).isin(list(bands))
    )
    columns = [
        "subject", "seizure", "band", "win_start_rel", "win_end_rel",
        "ictal_fraction",
    ]
    out = frame.loc[keep, columns].copy()
    out["subject"] = out["subject"].astype(str)
    out["seizure"] = pd.to_numeric(out["seizure"], errors="raise").astype(int)
    for column in ("win_start_rel", "win_end_rel", "ictal_fraction"):
        out[column] = pd.to_numeric(out[column], errors="raise")
    duplicated = out.duplicated(
        ["subject", "seizure", "band", "win_start_rel", "win_end_rel"],
        keep=False,
    )
    if duplicated.any():
        raise ValueError("duplicate original-F2 window rows")
    windows = tuple(
        map(tuple, out[["win_start_rel", "win_end_rel"]]
            .drop_duplicates().sort_values(["win_start_rel", "win_end_rel"])
            .to_numpy(float))
    )
    if windows != EXPECTED_WINDOWS:
        raise ValueError(f"original F2 window grid drifted:{windows}")
    return out.sort_values(
        ["subject", "band", "seizure", "win_start_rel", "win_end_rel"]
    ).reset_index(drop=True)


def build_original_null_groups(
    target_names: Sequence[str],
    finite_mask: Sequence[bool],
    original_name_order: Sequence[str],
    endpoint_pos: Mapping[str, Sequence[float]],
    *,
    min_group: int = MIN_GROUP_FOR_SHAFT,
) -> tuple[list[np.ndarray], str]:
    """Build original-F2 shaft/distance groups on the old endpoint plane.

    Indices in the returned groups address ``target_names``.  Group insertion
    order follows the old endpoint scorer's mapped-contact order, which keeps
    the randomisation architecture independent of the frozen field JSON order.
    """
    names = [str(value) for value in target_names]
    finite = np.asarray(finite_mask, bool)
    if finite.shape != (len(names),):
        raise ValueError("finite mask is not target-contact aligned")
    target_index = {name: idx for idx, name in enumerate(names)}
    ordered = [
        str(name) for name in original_name_order
        if str(name) in target_index and finite[target_index[str(name)]]
    ]
    missing = [names[i] for i in np.where(finite)[0] if names[i] not in set(ordered)]
    if missing:
        raise ValueError(f"finite gradient contacts absent from old endpoint plane:{missing}")

    by_shaft: dict[str, list[str]] = {}
    for name in ordered:
        by_shaft.setdefault(parse_shaft(name)[0], []).append(name)

    name_groups: list[list[str]] = []
    leftovers: list[str] = []
    for members in by_shaft.values():
        if len(members) >= int(min_group):
            name_groups.append(members)
        else:
            leftovers.extend(members)

    used_distance = False
    remaining = list(leftovers)
    order_index = {name: idx for idx, name in enumerate(remaining)}
    while len(remaining) >= int(min_group):
        used_distance = True
        anchor = remaining[0]
        anchor_xy = np.asarray(endpoint_pos[anchor], float)
        others = sorted(
            remaining[1:],
            key=lambda name: (
                float(np.linalg.norm(np.asarray(endpoint_pos[name], float) - anchor_xy)),
                order_index[name],
            ),
        )
        group = [anchor] + others[:int(min_group) - 1]
        name_groups.append(group)
        selected = set(group)
        remaining = [name for name in remaining if name not in selected]

    used_subject_wide = bool(remaining)
    if len(remaining) >= 2:
        name_groups.append(remaining)
    strength = (
        "subject_wide_weak" if used_subject_wide
        else "distance_bin_fallback" if used_distance
        else "within_shaft_strong"
    )
    groups = [np.asarray([target_index[name] for name in group], int)
              for group in name_groups]
    return groups, strength


def permute_by_groups(values: Sequence[float], groups: Sequence[np.ndarray],
                      rng: np.random.Generator) -> np.ndarray:
    """Apply one original-F2 spatial draw to a target-aligned activation."""
    source = np.asarray(values, float)
    out = source.copy()
    for group in groups:
        idx = np.asarray(group, int)
        if len(idx) >= 2:
            out[idx] = rng.permutation(source[idx])
    return out


def fold_windows_to_subject(scores: np.ndarray, seizure_ids: Sequence[int]) -> np.ndarray | float:
    """Apply the original window -> seizure -> subject median fold."""
    values = np.asarray(scores, float)
    ids = np.asarray(seizure_ids, int)
    if values.ndim not in (1, 2):
        raise ValueError("scores must be window or draw-by-window")
    if values.shape[-1] != len(ids):
        raise ValueError("score windows and seizure ids are not aligned")
    ordered_ids = list(dict.fromkeys(ids.tolist()))
    if values.ndim == 1:
        seizure = [float(np.nanmedian(values[ids == idx])) for idx in ordered_ids]
        return float(np.nanmedian(np.asarray(seizure, float)))
    seizure = np.column_stack([
        np.nanmedian(values[:, ids == idx], axis=1) for idx in ordered_ids
    ])
    return np.nanmedian(seizure, axis=1)


def _score_subject(
    bare_subject: str,
    reference: pd.DataFrame,
    bands: Sequence[str],
    *,
    n_perm: int,
    seed: int,
    smoothing_policy: str = "frozen_per_model",
) -> tuple[list[dict], list[dict], list[dict], dict, dict]:
    subject = _resolve_subject_id(bare_subject)
    dataset = subject.split("_", 1)[0]
    field_path = FIELD_ROOT / f"{subject}.json"
    if not field_path.exists():
        raise FileNotFoundError("missing_frozen_field")
    field_record = json.loads(field_path.read_text())
    _validate_field_record(field_record, subject)
    scorers, plane, score_key, smoothing = select_gradient_scorers(
        field_record, smoothing_policy=smoothing_policy,
    )

    # This is intentionally retained from the old endpoint implementation and
    # is used only for fixed-mask membership/order and spatial-null distances.
    endpoint_ctx = load_context(subject, "narrow")
    endpoint_order = [str(value) for value in endpoint_ctx["mapped"]]
    endpoint_pos = endpoint_ctx["pos"]

    meta_path = CACHE_ROOT / f"{subject}.json"
    cache_path = CACHE_ROOT / f"{subject}.npz"
    meta = json.loads(meta_path.read_text())
    if meta.get("analysis_channels_basis") != "primary_bands_validity":
        raise ValueError(f"{subject}:stale_analysis_channels_basis")
    analysis_channels = set(str(value) for value in meta.get("analysis_channels", []))
    target_names = [
        str(value) for value in field_record["interictal_field"]["contact_order"]
    ]
    old_fixed = analysis_channels & set(endpoint_order)

    events_by_band: dict[str, list[dict]] = defaultdict(list)
    event_rows: list[dict] = []
    drops: list[dict] = []
    with np.load(cache_path, allow_pickle=True) as cache:
        source_names = [str(value) for value in meta.get("channels", cache["channels"].tolist())]
        source_index = {name: idx for idx, name in enumerate(source_names)}
        subject_reference = reference[reference["subject"] == bare_subject]
        for row in subject_reference.itertuples(index=False):
            band = str(row.band)
            seizure_idx = int(row.seizure)
            try:
                activation_source = window_activation(
                    np.asarray(cache[f"{band}__zt__{seizure_idx}"], float),
                    np.asarray(cache[f"{band}__relt__{seizure_idx}"], float),
                    float(row.win_start_rel),
                    float(row.win_end_rel),
                )
                aligned = np.full(len(target_names), np.nan, float)
                for idx, name in enumerate(target_names):
                    if name in old_fixed and name in source_index:
                        aligned[idx] = activation_source[source_index[name]]
                finite = np.isfinite(aligned)
                if int(finite.sum()) < MIN_CONTACTS:
                    raise ValueError(f"fewer_than_{MIN_CONTACTS}_finite_contacts:{finite.sum()}")
                groups, strength = build_original_null_groups(
                    target_names, finite, endpoint_order, endpoint_pos,
                )
            except Exception as exc:
                drops.append({
                    "subject": subject,
                    "seizure_idx": seizure_idx,
                    "band": band,
                    "win_start_rel": float(row.win_start_rel),
                    "win_end_rel": float(row.win_end_rel),
                    "drop_reason": f"window_failed:{type(exc).__name__}:{exc}",
                })
                continue
            event = {
                "dataset": dataset,
                "subject": subject,
                "reference_subject": bare_subject,
                "seizure_idx": seizure_idx,
                "band": band,
                "win_start_rel": float(row.win_start_rel),
                "win_end_rel": float(row.win_end_rel),
                "ictal_fraction": float(row.ictal_fraction),
                "field_plane": plane,
                "score_key": score_key,
                "n_finite_contacts": int(finite.sum()),
                "spatial_null_strength": strength,
                "activation": aligned,
                "groups": groups,
            }
            events_by_band[band].append(event)

    for band in bands:
        if not events_by_band.get(band):
            raise ValueError(f"{subject}:{band}:no_valid_original_f2_windows")

    # One observed batch per subject.
    flat_events = [event for band in sorted(bands) for event in events_by_band[band]]
    observed_matrix = np.vstack([event["activation"] for event in flat_events])
    observed_all = np.asarray(score_scorer_bundle_batch(scorers, observed_matrix)[score_key], float)
    for event, observed in zip(flat_events, observed_all):
        if not np.isfinite(observed):
            raise ValueError(f"{subject}:nonfinite_observed_gradient_score")
        event["observed"] = float(observed)

    # Match the original subject-coherent draw structure: every draw gets one
    # subject RNG and visits bands/windows in a deterministic order.
    null_by_band = {
        band: np.full((n_perm, len(events_by_band[band])), np.nan, float)
        for band in bands
    }
    sub_hash = int(hashlib.sha1(bare_subject.encode()).hexdigest()[:8], 16)
    spatial_sequence = np.random.SeedSequence([int(seed), sub_hash]).spawn(2)[0]
    children = spatial_sequence.spawn(int(n_perm))
    sorted_bands = sorted(bands)
    for draw, child in enumerate(children):
        rng = np.random.default_rng(child)
        permuted = []
        layout: list[tuple[str, int]] = []
        for band in sorted_bands:
            for event_idx, event in enumerate(events_by_band[band]):
                permuted.append(permute_by_groups(event["activation"], event["groups"], rng))
                layout.append((band, event_idx))
        scored = np.asarray(
            score_scorer_bundle_batch(scorers, np.vstack(permuted))[score_key], float
        )
        if not np.isfinite(scored).all():
            raise ValueError(f"{subject}:nonfinite_spatial_null_draw:{draw}")
        for value, (band, event_idx) in zip(scored, layout):
            null_by_band[band][draw, event_idx] = float(value)

    subject_rows: list[dict] = []
    perm_rows: list[dict] = []
    for band in bands:
        events = events_by_band[band]
        seizure_ids = [int(event["seizure_idx"]) for event in events]
        observed = float(fold_windows_to_subject(
            np.asarray([event["observed"] for event in events], float), seizure_ids
        ))
        folded_null = np.asarray(
            fold_windows_to_subject(null_by_band[band], seizure_ids), float
        )
        null_median = float(np.median(folded_null))
        strengths = [str(event["spatial_null_strength"]) for event in events]
        strength = min(strengths, key=lambda value: _SPATIAL_TIER[value])
        subject_rows.append({
            "dataset": dataset,
            "subject": subject,
            "reference_subject": bare_subject,
            "band": band,
            "field_plane": plane,
            "score_key": score_key,
            "observed_subject_median": observed,
            "spatial_null_subject_median": null_median,
            "delta": observed - null_median,
            "subject_empirical_one_sided_p": float(
                (1 + np.sum(folded_null >= observed - 1e-15)) / (len(folded_null) + 1)
            ),
            "n_seizures": len(set(seizure_ids)),
            "n_windows": len(events),
            "spatial_null_strength": strength,
            "n_spatial_null_draws": int(n_perm),
        })
        base = {
            "subject": subject,
            "feature": "raw_gradient_original_f2_windows",
            "null_type": "spatial",
            "band": band,
        }
        perm_rows.append({
            **base, "perm_id": -1, "perm_subject_median": observed,
        })
        perm_rows.extend({
            **base, "perm_id": draw, "perm_subject_median": float(value),
        } for draw, value in enumerate(folded_null))

    for event in flat_events:
        event_rows.append({
            key: value for key, value in event.items()
            if key not in {"activation", "groups"}
        })
    routing = {
        "dataset": dataset,
        "subject": subject,
        "field_plane": plane,
        "score_key": score_key,
        "n_frozen_contacts": len(target_names),
        "n_original_fixed_frozen_contacts": len(old_fixed & set(target_names)),
        "distance_null_coordinates": "original_endpoint_plane_2d",
        "fingerprint_sha256": field_record["interictal_field"].get("fingerprint_sha256"),
        **smoothing,
    }
    return event_rows, subject_rows, perm_rows, routing, {"drops": drops}


def score_cohort(reference: pd.DataFrame, bands: Sequence[str], *, n_perm: int, seed: int,
                 smoothing_policy: str = "frozen_per_model"):
    events, subjects, perm_rows, routing, drops = [], [], [], [], []
    for bare_subject in sorted(reference["subject"].unique()):
        try:
            ev, sub, prm, route, extra = _score_subject(
                bare_subject, reference, bands, n_perm=n_perm, seed=seed,
                smoothing_policy=smoothing_policy,
            )
        except Exception as exc:
            subject = None
            try:
                subject = _resolve_subject_id(bare_subject)
            except Exception:
                subject = bare_subject
            drops.append({
                "subject": subject,
                "drop_reason": f"subject_failed:{type(exc).__name__}:{exc}",
            })
            continue
        events.extend(ev)
        subjects.extend(sub)
        perm_rows.extend(prm)
        routing.append(route)
        drops.extend(extra["drops"])
    return (
        pd.DataFrame(events), pd.DataFrame(subjects), pd.DataFrame(perm_rows),
        pd.DataFrame(routing), pd.DataFrame(drops),
    )


def _plot(subjects: pd.DataFrame, cohort: pd.DataFrame,
          band_contract: Sequence[Mapping[str, object]], output: Path, *, seed: int) -> Path:
    bands = [str(row["band"]) for row in band_contract]
    labels = {str(row["band"]): str(row["label"]) for row in band_contract}
    values = {
        band: subjects.loc[subjects["band"] == band, "delta"].to_numpy(float)
        for band in bands
    }
    by_band = cohort.set_index("band")
    passed = int(cohort["passes_fwer_0p05"].sum())
    return plot_null_per_band_figure(
        bands, labels, values,
        by_band["cohort_perm_delta_spatial"].to_dict(),
        by_band["max_over_bands_p"].to_dict(),
        by_band["n_subjects"].to_dict(),
        f"F2 · gradient field on original windows (n=19) · {passed}/7 pass FWER",
        output,
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
    path = PAPER_FIGURES / "README.md"
    existing = path.read_text() if path.exists() else "# Topic 5 gradient 多频带显著性图\n"
    marker = f"### {STEM}.png / {STEM}.pdf"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    passed = int(cohort["passes_fwer_0p05"].sum())
    detail = "; ".join(
        f"{row.band}: pFWER={row.max_over_bands_p:.4g}, "
        f"Δ={row.cohort_perm_delta_spatial:.3f}"
        for row in cohort.itertuples()
    )
    addition = f"""### {STEM}.png / {STEM}.pdf

这是旧 narrow F2 的受控 gradient-field 替换版。原 producer 接纳的五类重叠窗 `[-5,5]`、`[0,10]`、`[5,15]`、`[10,20]`、`[15,25] s`、`ictal_fraction≥0.5`、window→seizure→subject 中位数、旧 endpoint 二维坐标上的空间置换分组、1000 次置换与七频带 max-over-bands FWER 均保持不变；只把 endpoint field scorer 换成 frozen gradient field，有完整 shared A/B 时使用 shared maxAB，否则使用 own maxAB。

**关注点**：gradient field 可用分母为 n=19，E916 因 `axis_not_available` 排除；这属于新轴的可用性边界。{passed}/7 个频带通过 FWER。{detail}。
"""
    path.write_text(existing.rstrip() + "\n\n" + addition)
    return path


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.n_perm < MIN_PERM:
        raise ValueError(f"n_perm must be >= {MIN_PERM}")
    band_contract = load_primary_band_contract()
    bands = [str(row["band"]) for row in band_contract]
    reference = select_original_f2_rows(pd.read_csv(REFERENCE_WINDOWS), bands)
    if reference["subject"].nunique() != EXPECTED_REFERENCE_N:
        raise RuntimeError("original F2 reference denominator drifted from n=20")

    events, subjects, perm_rows, routing, drops = score_cohort(
        reference, bands, n_perm=args.n_perm, seed=args.seed,
    )
    if subjects.empty:
        raise RuntimeError("no gradient original-F2 subject results")
    # build_cohort_table expects the original F2 feature label in the long null table.
    cohort = build_cohort_table(
        subjects,
        perm_rows.assign(feature="raw_gradient"),
        band_contract,
    )
    if not (cohort["n_subjects"] == EXPECTED_ANALYSIS_N).all():
        counts = dict(zip(cohort["band"], cohort["n_subjects"]))
        raise RuntimeError(f"gradient original-F2 denominator drifted from n=19:{counts}")

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
    events.sort_values(["band", "subject", "seizure_idx", "win_start_rel"]).to_csv(
        paths["event"], index=False
    )
    subjects.sort_values(["band", "subject"]).to_csv(paths["subject"], index=False)
    cohort.to_csv(paths["cohort"], index=False)
    drops.to_csv(paths["drops"], index=False)
    routing.sort_values("subject").to_csv(paths["routing"], index=False)
    perm_rows.to_parquet(paths["null_draws"], index=False)

    png = PAPER_FIGURES / f"{STEM}.png"
    _plot(subjects, cohort, band_contract, png, seed=args.seed)
    readme = _write_readme(cohort)

    unique_subjects = subjects.drop_duplicates("subject")
    counts = {
        "reference_subjects": int(reference["subject"].nunique()),
        "analysis_subjects": int(unique_subjects.shape[0]),
        "epilepsiae_subjects": int(unique_subjects["dataset"].eq("epilepsiae").sum()),
        "yuquan_subjects": int(unique_subjects["dataset"].eq("yuquan").sum()),
        "shared_subjects": int(unique_subjects["field_plane"].eq("shared").sum()),
        "own_fallback_subjects": int(unique_subjects["field_plane"].eq("own_fallback").sum()),
        "bands_passing_fwer": int(cohort["passes_fwer_0p05"].sum()),
        "window_rows": int(len(events)),
    }
    summary = {
        "contract": CONTRACT,
        "analysis_role": "controlled replacement of endpoint field by gradient field in original narrow F2",
        "intended_changed_input_only": "frozen gradient shared-else-own axis and field scorer",
        "unavoidable_denominator_change": "E916 gradient axis_not_available; n=20 to n=19",
        "retained_original_f2_contract": {
            "source_rows": str(REFERENCE_WINDOWS.relative_to(ROOT)),
            "windows_sec": [list(window) for window in EXPECTED_WINDOWS],
            "ictal_fraction_min": 0.5,
            "folding": "window median within seizure; seizure median within subject; cohort median over subjects",
            "fixed_mask": "primary_bands_validity intersect old endpoint mapped contacts and frozen field contacts",
            "minimum_finite_contacts": MIN_CONTACTS,
            "spatial_null": "within shaft then endpoint-plane distance-bin then subject-wide fallback",
            "spatial_null_distance_coordinates": "original endpoint plane 2D",
            "n_permutations": int(args.n_perm),
            "seed": int(args.seed),
            "fwer": "null-centered Westfall-Young max-T across seven primary bands",
        },
        "gradient_field_contract": {
            "axis_definition": "template_propagation_axis_v2",
            "axis_direction_convention": "positive_early_to_late",
            "routing": "shared_a/shared_b if complete else own_a/own_b",
            "routing_is_outcome_independent": True,
        },
        "bands": band_contract,
        "counts": counts,
        "cohort_statistics": cohort.to_dict("records"),
        "drops": drops.to_dict("records"),
        "outputs": {
            key: str(path.relative_to(ROOT)) for key, path in paths.items()
        } | {
            "figure_png": str(png.relative_to(ROOT)),
            "figure_pdf": str(png.with_suffix(".pdf").relative_to(ROOT)),
            "figure_readme": str(readme.relative_to(ROOT)),
        },
    }
    summary_path = OUT / f"{STEM}_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n")
    for source in (paths["cohort"], paths["subject"], summary_path):
        (PAPER / source.name).write_text(source.read_text())
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
