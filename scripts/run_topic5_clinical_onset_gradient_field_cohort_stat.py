#!/usr/bin/env python3
"""Paper-facing onset 0--10 s gradient-field cohort statistic.

The three fixed groups are outcome independent:

* all phenotype-classified seizures, with strict-broadband events read out at
  broadband 1--150 Hz and gamma-nonbroadband events at gamma 30--80 Hz;
* accepted strict-broadband seizures, read out with broadband 1--150 Hz;
* accepted gamma-nonbroadband seizures, read out with gamma 30--80 Hz.

Epilepsiae's historical T0 cache is aligned to clinical onset.  Yuquan has no
clinical-onset annotation, so its historical cache remains honestly aligned to
EEG onset.  Frozen gradient-field routing is shared A/B when a complete shared
field exists and own A/B otherwise.  Every all-contact shuffle is applied to
the activation vector before smoothing, mirror selection and A/B max selection;
seizures are folded within subject before cohort inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3_clinical_onset_gradient_field_cohort import (  # noqa: E402
    plot_clinical_onset_gradient_field_cohort,
)
from scripts.run_topic5_old_cache_hybrid_field_comparison import (  # noqa: E402
    select_shared_else_own_scorers,
)
from scripts.run_topic5_tspectral_field_concordance import (  # noqa: E402
    EPI_CACHE,
    FIELD_ROOT,
    MIN_CONTACTS,
    YUQ_CACHE,
    _seed,
)
from scripts.run_topic5_unstratified_channel_scaffold_diagnostic import (  # noqa: E402
    OLD_CACHE as BB150_CACHE,
)
from src.topic5_template_axis_field import scorers_from_interictal_record  # noqa: E402
from src.topic5_t0_features import window_activation  # noqa: E402
from src.topic5_tspectral_field_concordance import (  # noqa: E402
    exact_name_align_matrix,
    fold_seizure_null_draws,
    jsonable,
    make_contact_permutations,
    paired_sign_flip_p,
    phenotype_selector_sets,
    score_observed_bundle,
    score_permutation_matrix,
)


ARTIFACT_ROOT = Path(os.environ.get("HFOSP_ARTIFACT_ROOT", ROOT)).resolve()
GAMMA30_CACHE = ARTIFACT_ROOT / "results/topic5_ictal_recruitment/v2_band_scan/cache"
OUT = Path(os.environ.get(
    "HFOSP_FIG3D_ANALYSIS_DIR",
    ROOT / "results/topic5_ictal_recruitment/tspectral_field_concordance",
)).resolve()
PAPER = Path(os.environ.get(
    "HFOSP_FIG3D_PAPER_DIR",
    ROOT / "results/paper-ready-figure/fig3-sup-tspectral-field-concordance",
)).resolve()
PAPER_FIGURES = PAPER / "figures"

CONTRACT = "topic5_onset_0_10_gradient_shared_else_own_channel_null_v2_gamma30"
BASE_SEED = 20260717
MIN_PERM = 1000
STEM = "clinical_onset_gradient_field_cohort_stat"

GROUPS: dict[str, dict[str, object]] = {
    "all_phenotype_matched": {
        "label": "All phenotype-matched seizures",
        "selector": "accepted_strict_broadband_or_gamma_nonbroadband",
        "band": "phenotype_matched",
        "cache": "by_phenotype",
        "key_prefix": None,
        "inference_role": "primary_phenotype_matched_cohort",
    },
    "strict_broadband": {
        "label": "Strict broadband · BB 1–150",
        "selector": "accepted_tspectral_strict_broadband",
        "band": "broadband_1_150",
        "cache": "bb150",
        "key_prefix": "bb150_auc",
        "inference_role": "phenotype_stratified_decomposition",
    },
    "gamma_nonbroadband": {
        "label": "Gamma non-BB · 30–80 Hz",
        "selector": "accepted_tspectral_gamma_nonbroadband",
        "band": "gamma_30_80",
        "cache": "gamma30_v2_band_scan",
        "key_prefix": "gamma_LVFA",
        "inference_role": "phenotype_stratified_decomposition",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_phenotype_selector_map() -> dict[str, dict[str, set[int]]]:
    """Load mutually exclusive accepted phenotype selectors from canonical caches."""
    out: dict[str, dict[str, set[int]]] = {}
    for root in (EPI_CACHE, YUQ_CACHE):
        for path in sorted(root.glob("*.json")):
            if path.name == "cache_alignment_summary.json":
                continue
            meta = json.loads(path.read_text())
            if "seizure_idxs" not in meta:
                continue
            subject = str(meta.get("subject", path.stem))
            if subject in out:
                raise ValueError(f"duplicate phenotype selector subject:{subject}")
            selectors = phenotype_selector_sets(meta)
            out[subject] = {
                "strict_broadband": set(selectors["broadband_1_150"]),
                "gamma_nonbroadband": set(selectors["gamma_nonbroadband"]),
            }
    return out


def event_ids_for_group(
    old_eligible: Sequence[int],
    selector: Mapping[str, set[int]],
    group_id: str,
) -> list[int]:
    """Apply only the fixed group selector and the historical cache intersection."""
    eligible = {int(v) for v in old_eligible}
    if group_id == "all_phenotype_matched":
        strict = set(selector.get("strict_broadband", set()))
        gamma = set(selector.get("gamma_nonbroadband", set()))
        if strict & gamma:
            raise ValueError("strict_broadband_and_gamma_nonbroadband_overlap")
        return sorted(eligible & (strict | gamma))
    if group_id == "strict_broadband":
        return sorted(eligible & set(selector.get("strict_broadband", set())))
    if group_id == "gamma_nonbroadband":
        return sorted(eligible & set(selector.get("gamma_nonbroadband", set())))
    raise ValueError(f"unknown group:{group_id}")


def readout_group_for_event(
    group_id: str,
    seizure_idx: int,
    selector: Mapping[str, set[int]],
) -> str:
    """Resolve the fixed phenotype-matched readout for one event."""
    if group_id != "all_phenotype_matched":
        return group_id
    in_strict = seizure_idx in selector.get("strict_broadband", set())
    in_gamma = seizure_idx in selector.get("gamma_nonbroadband", set())
    if in_strict == in_gamma:
        raise ValueError(
            f"pooled_event_must_have_exactly_one_phenotype:{seizure_idx}"
        )
    return "strict_broadband" if in_strict else "gamma_nonbroadband"


def gamma30_onset_activation(
    cache: Mapping[str, np.ndarray], seizure_idx: int
) -> np.ndarray:
    """Slice the cached 30--80 Hz robust-z trace over clinical onset [0,10] s."""
    z_key = f"gamma_LVFA__zt__{seizure_idx}"
    relt_key = f"gamma_LVFA__relt__{seizure_idx}"
    if z_key not in cache or relt_key not in cache:
        raise KeyError(f"missing_activation:{z_key}+{relt_key}")
    return window_activation(
        np.asarray(cache[z_key], float),
        np.asarray(cache[relt_key], float),
        0.0,
        10.0,
    )


def score_cached_activation(
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
    """Score observed and channel-null values from one cached contact activation."""
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
    target_names = [
        str(v) for v in field_record["interictal_field"]["contact_order"]
    ]
    perm_seed = _seed(
        f"clinical-gradient:{subject}:{seizure_idx}:{band}", seed
    )
    permutations = make_contact_permutations(
        target_names, finite, n_perm, perm_seed, mode="all_contact"
    )
    null = score_permutation_matrix(
        scorers, aligned[None, :], permutations, chunk_draws=100
    )[score_key][:, 0]
    if len(null) != n_perm or not np.isfinite(null).all():
        raise ValueError("nonfinite_or_incomplete_channel_null")
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
    }


def _fold_subject(
    subject: str,
    dataset: str,
    group_id: str,
    plane: str,
    score_key: str,
    events: list[dict[str, object]],
    n_perm: int,
    field_fingerprint_sha256: str,
    field_record_path: str,
) -> dict[str, object] | None:
    if not events:
        return None
    data = float(np.median([float(event["observed"]) for event in events]))
    folded = fold_seizure_null_draws([
        np.asarray(event["null"], float)[:, None] for event in events
    ])[:, 0]
    null = float(np.median(folded))
    contract = GROUPS[group_id]
    return {
        "dataset": dataset,
        "subject": subject,
        "group_id": group_id,
        "group_label": contract["label"],
        "inference_role": contract["inference_role"],
        "time_reference": (
            "clinical_onset" if dataset == "epilepsiae" else "eeg_onset_only"
        ),
        "window_start_sec": 0.0,
        "window_end_sec": 10.0,
        "band": contract["band"],
        "field_plane": plane,
        "score_key": score_key,
        "field_fingerprint_sha256": field_fingerprint_sha256,
        "field_record_path": field_record_path,
        "data": data,
        "channel_null_median": null,
        "channel_null_p95": float(np.percentile(folded, 95)),
        "margin": data - null,
        "subject_empirical_one_sided_p": float(
            (1 + np.sum(folded >= data - 1e-15)) / (len(folded) + 1)
        ),
        "n_seizures": len(events),
        "seizure_idxs": ";".join(str(event["seizure_idx"]) for event in events),
        "n_channel_shuffle_draws": int(n_perm),
    }


def _cohort(subjects: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for group_id, contract in GROUPS.items():
        frame = subjects[subjects.group_id == group_id].sort_values("subject")
        if frame.empty:
            continue
        data = frame.data.to_numpy(float)
        null = frame.channel_null_median.to_numpy(float)
        margin = data - null
        rows.append({
            "group_id": group_id,
            "group_label": contract["label"],
            "inference_role": contract["inference_role"],
            "n_subjects": int(len(frame)),
            "n_seizures": int(frame.n_seizures.sum()),
            "n_epilepsiae_subjects": int((frame.dataset == "epilepsiae").sum()),
            "n_yuquan_subjects": int((frame.dataset == "yuquan").sum()),
            "n_shared_subjects": int((frame.field_plane == "shared").sum()),
            "n_own_fallback_subjects": int((frame.field_plane == "own_fallback").sum()),
            "data_median": float(np.median(data)),
            "data_iqr_low": float(np.percentile(data, 25)),
            "data_iqr_high": float(np.percentile(data, 75)),
            "null_median": float(np.median(null)),
            "null_iqr_low": float(np.percentile(null, 25)),
            "null_iqr_high": float(np.percentile(null, 75)),
            "margin_median": float(np.median(margin)),
            "margin_iqr_low": float(np.percentile(margin, 25)),
            "margin_iqr_high": float(np.percentile(margin, 75)),
            "n_data_gt_null": int(np.sum(margin > 0)),
            "wilcoxon_one_sided_data_gt_null_p": float(
                wilcoxon(data, null, alternative="greater").pvalue
            ),
            "two_sided_subject_sign_flip_p": float(
                paired_sign_flip_p(
                    margin,
                    n_perm=100000,
                    seed=_seed(f"clinical-gradient-cohort:{group_id}", seed),
                )
            ),
        })
    return pd.DataFrame(rows)


def _run_scoring(n_perm: int, seed: int):
    selectors = load_phenotype_selector_map()
    subject_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    drops: list[dict[str, object]] = []
    old_subjects = {path.stem for path in BB150_CACHE.glob("*.json")}

    for meta_path in sorted(BB150_CACHE.glob("*.json")):
        subject = meta_path.stem
        dataset = subject.split("_", 1)[0]
        bb_npz_path = BB150_CACHE / f"{subject}.npz"
        gamma_meta_path = GAMMA30_CACHE / f"{subject}.json"
        gamma_npz_path = GAMMA30_CACHE / f"{subject}.npz"
        field_path = FIELD_ROOT / f"{subject}.json"
        if not all(path.exists() for path in (
            bb_npz_path, gamma_meta_path, gamma_npz_path, field_path
        )):
            drops.append({"dataset": dataset, "subject": subject,
                          "drop_reason": "missing_required_cache_or_field"})
            continue
        field_record = json.loads(field_path.read_text())
        field_fingerprint = str(
            (field_record.get("interictal_field") or {}).get(
                "fingerprint_sha256", ""
            )
        )
        try:
            all_scorers = scorers_from_interictal_record(field_record)
            scorers, plane, score_key = select_shared_else_own_scorers(all_scorers)
        except Exception as exc:
            drops.append({"dataset": dataset, "subject": subject,
                          "drop_reason": f"field_unavailable:{type(exc).__name__}:{exc}"})
            continue

        bb_meta = json.loads(meta_path.read_text())
        gamma_meta = json.loads(gamma_meta_path.read_text())
        eligible = [int(v) for v in bb_meta.get("eligible_idxs", [])]
        subject_selector = selectors.get(subject, {
            "strict_broadband": set(), "gamma_nonbroadband": set()
        })
        group_events = {
            group_id: event_ids_for_group(eligible, subject_selector, group_id)
            for group_id in GROUPS
        }
        if not group_events["all_phenotype_matched"]:
            drops.append({
                "dataset": dataset,
                "subject": subject,
                "seizure_idx": None,
                "group_id": "all_phenotype_matched",
                "drop_reason": "no_strict_broadband_or_gamma_nonbroadband_event",
            })
        score_cache: dict[tuple[int, str], dict[str, object]] = {}
        events_by_group: dict[str, list[dict[str, object]]] = {
            group_id: [] for group_id in GROUPS
        }
        with np.load(bb_npz_path, allow_pickle=True) as bb_cache, np.load(
            gamma_npz_path, allow_pickle=True
        ) as gamma_cache:
            bb_names = [str(v) for v in bb_meta.get("channels", bb_cache["channels"].tolist())]
            gamma_names = [
                str(v) for v in gamma_meta.get(
                    "channels", gamma_cache["channels"].tolist()
                )
            ]
            for group_id, seizure_idxs in group_events.items():
                for seizure_idx in seizure_idxs:
                    readout_group = readout_group_for_event(
                        group_id, seizure_idx, subject_selector
                    )
                    readout = GROUPS[readout_group]
                    is_bb = readout["cache"] == "bb150"
                    cache = bb_cache if is_bb else gamma_cache
                    names = bb_names if is_bb else gamma_names
                    cache_key = (
                        f"{readout['key_prefix']}__{seizure_idx}"
                        if is_bb else f"gamma_LVFA__zt__{seizure_idx}"
                    )
                    score_cache_key = (seizure_idx, str(readout["band"]))
                    if cache_key not in cache.files:
                        drops.append({
                            "dataset": dataset, "subject": subject,
                            "seizure_idx": seizure_idx, "group_id": group_id,
                            "drop_reason": f"missing_activation:{cache_key}",
                        })
                        continue
                    if score_cache_key not in score_cache:
                        try:
                            activation = (
                                np.asarray(cache[cache_key], float)
                                if is_bb else gamma30_onset_activation(cache, seizure_idx)
                            )
                            score_cache[score_cache_key] = score_cached_activation(
                                field_record, scorers, score_key, names,
                                activation, subject=subject,
                                seizure_idx=seizure_idx, band=str(readout["band"]),
                                n_perm=n_perm, seed=seed,
                            )
                        except Exception as exc:
                            drops.append({
                                "dataset": dataset, "subject": subject,
                                "seizure_idx": seizure_idx, "group_id": group_id,
                                "drop_reason": f"score_failed:{type(exc).__name__}:{exc}",
                            })
                            continue
                    score = score_cache[score_cache_key]
                    event = {
                        "dataset": dataset, "subject": subject,
                        "seizure_idx": seizure_idx, "group_id": group_id,
                        "group_label": GROUPS[group_id]["label"],
                        "phenotype": readout_group,
                        "time_reference": (
                            "clinical_onset" if dataset == "epilepsiae"
                            else "eeg_onset_only"
                        ),
                        "window_start_sec": 0.0, "window_end_sec": 10.0,
                        "band": readout["band"], "field_plane": plane,
                        "score_key": score_key,
                        "field_fingerprint_sha256": field_fingerprint,
                        "field_record_path": _display_path(field_path),
                        "observed": score["observed"],
                        "null": score["null"],
                        "null_median": score["null_median"],
                        "null_p95": score["null_p95"],
                        "a_abs": score["a_abs"], "b_abs": score["b_abs"],
                        "best_template": score["best_template"],
                        "n_finite_contacts": score["n_finite_contacts"],
                        "permutation_seed": score["permutation_seed"],
                        "mirror_reselected_each_draw": True,
                        "ab_max_reselected_each_draw": True,
                    }
                    events_by_group[group_id].append(event)
                    event_rows.append({
                        key: value for key, value in event.items() if key != "null"
                    })

        for group_id, events in events_by_group.items():
            row = _fold_subject(
                subject,
                dataset,
                group_id,
                plane,
                score_key,
                events,
                n_perm,
                field_fingerprint,
                _display_path(field_path),
            )
            if row is not None:
                subject_rows.append(row)

    for subject, selector in sorted(selectors.items()):
        if subject in old_subjects:
            continue
        dataset = subject.split("_", 1)[0]
        for phenotype in ("strict_broadband", "gamma_nonbroadband"):
            for seizure_idx in sorted(selector[phenotype]):
                for group_id in (phenotype, "all_phenotype_matched"):
                    drops.append({
                        "dataset": dataset, "subject": subject,
                        "seizure_idx": seizure_idx, "group_id": group_id,
                        "drop_reason": "missing_historical_onset_activation_cache",
                    })

    return (
        pd.DataFrame(subject_rows),
        pd.DataFrame(event_rows),
        pd.DataFrame(drops),
    )


def _write_readme(cohort: pd.DataFrame) -> None:
    path = PAPER_FIGURES / "README.md"
    existing = path.read_text() if path.exists() else "# Fig3 supplement figures\n"
    marker = f"### {STEM}.png"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    stat = "; ".join(
        f"{row.group_label}: n={row.n_subjects}/{row.n_seizures}, "
        f"one-sided p={row.wilcoxon_one_sided_data_gt_null_p:.4g}"
        for row in cohort.itertuples()
    )
    addition = f"""### {STEM}.png / {STEM}.pdf

正式paper-facing cohort panel：使用原始onset后`[0,10] s`的缓存能量，冻结gradient field按纯间期几何固定路由（有完整shared A/B则shared，否则own fallback），并以1000次all-contact channel shuffle构造患者配对null。最左组先按事件的互斥表型匹配readout（strict-broadband用1–150 Hz；gamma-nonbroadband用30–80 Hz）再在subject内合并折叠；右两组分别展示strict-broadband和gamma-nonbroadband贡献。每个null draw均重新做平滑、mirror与A/B max选择，先在患者内折叠seizure再做cohort统计。图形严格复用Fig3既有Data–Null violin、box、subject点、配对线和显著性括号；主图panel只以Pooled/Broadband/Gamma作为横轴，Observed与channel-shuffle null移入图例，精确p值保留在图注和统计表。

**关注点**：{stat}。Epilepsiae的cache零点为clinical onset；Yuquan没有clinical onset，保留真实EEG-onset零点而不伪造。1–150 Hz来自既有line-noise-bin-masked旧cache；30–80 Hz来自v2 band-scan的`gamma_LVFA` robust-z轨迹，并在同一onset坐标切取`[0,10] s`。因此本图的正式统计合同必须按此写明，不能与旧`T_spectral + own + within-shaft`表混报。
"""
    path.write_text(existing + addition)


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.n_perm < MIN_PERM:
        raise ValueError(f"n_perm must be >= {MIN_PERM}")
    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    source_npz = sorted(BB150_CACHE.glob("*.npz")) + sorted(GAMMA30_CACHE.glob("*.npz"))
    hashes_before = {_display_path(path): _sha256(path) for path in source_npz}

    subjects, events, drops = _run_scoring(args.n_perm, args.seed)
    if subjects.empty:
        raise RuntimeError("no subject-level results")
    subjects = subjects.sort_values(["group_id", "subject"]).reset_index(drop=True)
    events = events.sort_values(["group_id", "subject", "seizure_idx"]).reset_index(drop=True)
    cohort = _cohort(subjects, args.seed)

    paths = {
        "subject": OUT / f"{STEM}_subject.csv",
        "event": OUT / f"{STEM}_event.csv",
        "cohort": OUT / f"{STEM}_cohort.csv",
        "drops": OUT / f"{STEM}_drop_inventory.csv",
    }
    subjects.to_csv(paths["subject"], index=False)
    events.to_csv(paths["event"], index=False)
    cohort.to_csv(paths["cohort"], index=False)
    drops.to_csv(paths["drops"], index=False)

    hashes_after = {_display_path(path): _sha256(path) for path in source_npz}
    if hashes_before != hashes_after:
        raise RuntimeError("historical onset cache NPZ changed")

    png = PAPER_FIGURES / f"{STEM}.png"
    pdf = png.with_suffix(".pdf")
    plot_clinical_onset_gradient_field_cohort(subjects, cohort, png, pdf)
    _write_readme(cohort)

    selectors = load_phenotype_selector_map()
    summary = {
        "contract": CONTRACT,
        "paper_status": "formal_primary_cohort_panel",
        "supersedes_for_paper": (
            "T_spectral 0-10 s + own field + within-shaft null cohort table"
        ),
        "time_reference": {
            "epilepsiae": "clinical_onset",
            "yuquan": "eeg_onset_only; clinical onset unavailable and not fabricated",
        },
        "window_sec": [0.0, 10.0],
        "field_routing": "shared_a/shared_b if complete else own_a/own_b",
        "field_root": _display_path(FIELD_ROOT),
        "field_fingerprints": {
            str(row.subject): str(row.field_fingerprint_sha256)
            for row in subjects.drop_duplicates("subject").itertuples()
        },
        "routing_is_outcome_independent": True,
        "readout_groups": GROUPS,
        "broadband_cache_contract": (
            "historical bb150_auc; baseline-robust-z 1-150 Hz over onset [0,10] s; "
            "additional line-noise FFT-bin mask"
        ),
        "gamma30_cache_contract": (
            "v2 band-scan gamma_LVFA robust-z trace; line-noise-bin-masked half-open "
            "30-80 Hz; mean over clinical/EEG onset [0,10] s"
        ),
        "null": {
            "mode": "all_contact_channel_shuffle",
            "n_draws_per_seizure": int(args.n_perm),
            "mirror_reselected_each_draw": True,
            "ab_max_reselected_each_draw": True,
            "folding": "seizure median within subject for every draw",
        },
        "canonical_selector_counts": {
            "strict_broadband_events": int(sum(
                len(value["strict_broadband"]) for value in selectors.values()
            )),
            "gamma_nonbroadband_events": int(sum(
                len(value["gamma_nonbroadband"]) for value in selectors.values()
            )),
        },
        "cohort_statistics": cohort.to_dict("records"),
        "source_cache_npz_unchanged": True,
        "outputs": {
            key: _display_path(path) for key, path in paths.items()
        } | {
            "figure_png": _display_path(png),
            "figure_pdf": _display_path(pdf),
        },
    }
    summary_path = OUT / f"{STEM}_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n")
    for path in (paths["subject"], paths["cohort"], paths["drops"], summary_path):
        (PAPER / path.name).write_text(path.read_text())
    print(cohort.to_string(index=False), flush=True)
    print(f"[done] {png}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=MIN_PERM)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
