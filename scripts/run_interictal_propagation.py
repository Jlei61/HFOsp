#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_propagation import (  # noqa: E402
    SEIZURE_PROXIMITY_CONFIGS,
    _build_seizure_proximity_windows,
    _compute_relative_lag_matrix,
    _intersect_seconds,
    _valid_event_indices,
    assign_events_to_templates,
    build_cluster_templates,
    compute_continuous_template_dynamics,
    compute_novel_template_gate,
    compute_rate_state_coupling,
    compute_seizure_proximity_coupling,
    compute_template_recruitment_shift,
    validate_absolute_lag_clustering,
    compute_temporal_cluster_dynamics,
    compute_time_split_reproducibility,
    compute_within_cluster_centered_tau,
    load_subject_propagation_events,
    run_subject_interictal_propagation_pr1,
    summarize_pr5_novel_template_gate,
    summarize_pr5_template_recruitment_shift,
    summarize_propagation_cohort,
)
from src.event_periodicity import load_seizure_times  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("interictal_propagation")

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RESULTS_DIR = Path("results/interictal_propagation")

YUQUAN_SUBJECTS = [
    "zhangkexuan", "pengzihang", "chengshuai", "huangwanling",
    "liyouran", "songzishuo", "zhangbichen", "zhaochenxi",
    "zhaojinrui", "zhourongxuan", "zhangjiaqi",
    "chenziyang", "hanyuxuan", "huanghanwen", "litengsheng",
    "xuxinyi", "zhangjinhan", "sunyuanxin",
]

EPILEPSIAE_SUBJECTS = [
    "1096", "1084", "958", "922", "590", "1150", "442", "1073",
    "253", "1146", "916", "620", "583", "548", "384", "139",
    "1125", "1077", "818", "635",
]

SOZ_FILE_YQ = Path("results/yuquan_soz_core_channels.json")
SOZ_FILE_EPI = Path("results/epilepsiae_soz_core_channels.json")


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
        except Exception:
            pass
        return super().default(obj)


def _epilepsiae_subject_dir(root: Path, subject: str) -> Path:
    """Resolve Epilepsiae per-subject lagPat directory.

    Legacy layout: ``<root>/<subject>/all_recs/`` (the canonical
    `/mnt/epilepsia_data/interilca_inter_results/all_data_lns/` tree).

    Backfill layout: ``<root>/<subject>/`` (the new pipeline writes
    `<stem>_lagPat.npz` directly under the subject dir, with no intermediate
    `all_recs`). Used for Stage D sensitivity audits — see
    `docs/archive/epilepsiae_lagpat/epilepsiae_lagpat_backfill_plan_2026-04-29.md`.

    Probes legacy first (so existing pipelines keep their exact path); falls
    back to flat layout when `all_recs` doesn't exist.
    """
    legacy = root / subject / "all_recs"
    if legacy.exists():
        return legacy
    return root / subject


def _subject_dir(dataset: str, root: Path, subject: str) -> Path:
    if dataset == "yuquan":
        return root / subject
    return _epilepsiae_subject_dir(root, subject)


def _load_soz(path: Path) -> Dict[str, List[str]]:
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
    logger.info("Saved %s", path)


def _run_pr25(
    datasets_list: List,
    per_subject_dir: Path,
) -> None:
    """PR-2.5: augment existing per-subject JSONs with cross-time reproducibility."""
    import numpy as np

    all_results: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, _soz_map in datasets_list:
        for subject in subjects:
            subject_dir = _subject_dir(dataset, root, subject)
            json_path = per_subject_dir / f"{dataset}_{subject}.json"
            key = f"{dataset}/{subject}"

            if not json_path.exists():
                logger.warning("Skip %s: no existing JSON at %s", key, json_path)
                continue

            with open(json_path) as f:
                existing = json.load(f)

            ac = existing.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning("Skip %s: adaptive_cluster missing or errored", key)
                all_results[key] = existing
                continue

            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s: raw data dir missing", key)
                all_results[key] = existing
                continue

            logger.info("PR-2.5 reproducibility: %s", key)
            try:
                loaded = load_subject_propagation_events(subject_dir)
                valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
                labels = np.array(ac["labels"], dtype=int)
                chosen_k = int(ac["chosen_k"])

                if valid_events.size != len(labels):
                    logger.error(
                        "Skip %s: valid_events size %d != labels size %d",
                        key, valid_events.size, len(labels),
                    )
                    all_results[key] = existing
                    continue

                repro = compute_time_split_reproducibility(
                    ranks=loaded["ranks"],
                    bools=loaded["bools"],
                    event_abs_times=loaded["event_abs_times"],
                    block_ids=loaded["block_ids"],
                    chosen_k=chosen_k,
                    adaptive_labels=labels,
                    valid_event_indices=valid_events,
                )
                existing["time_split_reproducibility"] = repro
                logger.info(
                    "  %s: grade=%s  split-half corr=%.3f agree=%.3f",
                    key,
                    repro["reproducibility_grade"],
                    repro["splits"].get("first_half_second_half", {}).get("mean_match_corr", float("nan")),
                    repro["splits"].get("first_half_second_half", {}).get("assignment_agreement", float("nan")),
                )
            except Exception as exc:
                logger.exception("Failed PR-2.5 for %s", key)
                existing["time_split_reproducibility"] = {"error": str(exc)}

            _save(existing, json_path)
            all_results[key] = existing

    cohort = summarize_propagation_cohort(all_results)
    _save(all_results, RESULTS_DIR / "pr1_subject_summary.json")
    _save(cohort, RESULTS_DIR / "pr1_cohort_summary.json")
    logger.info("PR-2.5 done. Cohort reproducibility: %s", cohort.get("reproducibility_analysis", {}))


def _augment_cluster_bias(
    datasets_list: List,
    per_subject_dir: Path,
) -> None:
    """Augment existing per-subject JSONs with within-cluster identity-bias."""
    import numpy as np

    all_results: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, _soz_map in datasets_list:
        for subject in subjects:
            subject_dir = _subject_dir(dataset, root, subject)
            json_path = per_subject_dir / f"{dataset}_{subject}.json"
            key = f"{dataset}/{subject}"

            if not json_path.exists():
                logger.warning("Skip %s: no existing JSON", key)
                continue

            with open(json_path) as f:
                existing = json.load(f)

            ac = existing.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning("Skip %s: adaptive_cluster missing", key)
                all_results[key] = existing
                continue

            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s: raw data missing", key)
                all_results[key] = existing
                continue

            logger.info("Augmenting within-cluster bias: %s", key)
            try:
                loaded = load_subject_propagation_events(subject_dir)
                valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
                labels = np.array(ac["labels"], dtype=int)

                if valid_events.size != len(labels):
                    logger.error("Skip %s: size mismatch %d vs %d", key, valid_events.size, len(labels))
                    all_results[key] = existing
                    continue

                bias = compute_within_cluster_centered_tau(
                    loaded["ranks"], loaded["bools"], labels, valid_events,
                )
                existing["within_cluster_centered"] = bias
                logger.info(
                    "  %s: mean_raw=%.3f  mean_centered=%.3f  bias=%.1f%%",
                    key, bias["mean_raw_tau"], bias["mean_centered_tau"],
                    bias["mean_bias_fraction"] * 100 if np.isfinite(bias["mean_bias_fraction"]) else float("nan"),
                )
            except Exception:
                logger.exception("Failed for %s", key)
                existing["within_cluster_centered"] = {"error": "computation_failed"}

            _save(existing, json_path)
            all_results[key] = existing

    cohort = summarize_propagation_cohort(all_results)
    _save(all_results, RESULTS_DIR / "pr1_subject_summary.json")
    _save(cohort, RESULTS_DIR / "pr1_cohort_summary.json")
    logger.info("Cluster-bias augmentation done.")


def _run_pr4a(
    datasets_list: List,
    per_subject_dir: Path,
    *,
    n_sample: int,
    n_seeds: int,
    bin_hours: float,
) -> None:
    """PR-4A: fixed-template occupancy timeline + day/night summaries."""
    import numpy as np

    all_results: Dict[str, Dict[str, Any]] = {}
    temporal_results: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, soz_map in datasets_list:
        for subject in subjects:
            subject_dir = _subject_dir(dataset, root, subject)
            key = f"{dataset}/{subject}"
            json_path = per_subject_dir / f"{dataset}_{subject}.json"

            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s: raw data dir missing", key)
                continue

            if json_path.exists():
                with open(json_path) as f:
                    existing = json.load(f)
            else:
                logger.info("PR-4A bootstrap base result: %s", key)
                existing = run_subject_interictal_propagation_pr1(
                    subject_dir=subject_dir,
                    dataset=dataset,
                    subject=subject,
                    soz_channels=soz_map.get(subject, []),
                    n_sample=n_sample,
                    n_seeds=n_seeds,
                )

            if "error" in existing:
                all_results[key] = existing
                _save(existing, json_path)
                continue

            ac = existing.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning("Skip %s: adaptive_cluster missing or errored", key)
                all_results[key] = existing
                _save(existing, json_path)
                continue

            loaded = load_subject_propagation_events(subject_dir)
            valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
            labels = np.asarray(ac["labels"], dtype=int)
            chosen_k = int(ac["chosen_k"])

            if valid_events.size != labels.size:
                existing["temporal_dynamics"] = {
                    "error": (
                        f"valid_event_mismatch: valid_events={valid_events.size} "
                        f"labels={labels.size}"
                    )
                }
                all_results[key] = existing
                _save(existing, json_path)
                continue

            if not existing.get("time_split_reproducibility"):
                repro = compute_time_split_reproducibility(
                    ranks=loaded["ranks"],
                    bools=loaded["bools"],
                    event_abs_times=loaded["event_abs_times"],
                    block_ids=loaded["block_ids"],
                    chosen_k=chosen_k,
                    adaptive_labels=labels,
                    valid_event_indices=valid_events,
                )
                existing["time_split_reproducibility"] = repro

            v_ranks = loaded["ranks"][:, valid_events]
            v_bools = loaded["bools"][:, valid_events]
            v_times = loaded["event_abs_times"][valid_events]
            templates = build_cluster_templates(v_ranks, v_bools, labels, chosen_k)
            projected = assign_events_to_templates(v_ranks, v_bools, templates)
            assignable = projected >= 0
            agreement = float(np.mean(projected[assignable] == labels[assignable])) if np.any(assignable) else np.nan

            temporal = compute_temporal_cluster_dynamics(
                event_abs_times=v_times[assignable],
                cluster_labels=projected[assignable],
                n_clusters=chosen_k,
                dataset=dataset,
                coverage_ranges=loaded.get("block_time_ranges"),
                bin_hours=bin_hours,
            )
            temporal.update(
                {
                    "subject": subject,
                    "chosen_k": chosen_k,
                    "stable_k": int(ac.get("stable_k", chosen_k) or chosen_k),
                    "reproducibility_grade": existing.get("time_split_reproducibility", {}).get(
                        "reproducibility_grade"
                    ),
                    "template_projection": {
                        "n_valid_events": int(valid_events.size),
                        "n_assignable": int(np.sum(assignable)),
                        "agreement_with_adaptive_labels": agreement,
                    },
                }
            )
            existing["temporal_dynamics"] = temporal
            temporal_results[key] = temporal
            all_results[key] = existing

            _save(existing, json_path)
            logger.info(
                "PR-4A temporal dynamics: %s  k=%d  grade=%s  assign=%.3f",
                key,
                chosen_k,
                temporal.get("reproducibility_grade"),
                temporal["template_projection"]["agreement_with_adaptive_labels"],
            )

    cohort = summarize_propagation_cohort(all_results)
    _save(all_results, RESULTS_DIR / "pr1_subject_summary.json")
    _save(cohort, RESULTS_DIR / "pr1_cohort_summary.json")
    _save(temporal_results, RESULTS_DIR / "pr4a_temporal_dynamics.json")
    logger.info(
        "PR-4A done. Cohort temporal summary: %s",
        cohort.get("temporal_dynamics_analysis", {}),
    )


def _run_pr4a_followup(
    datasets_list: List,
    per_subject_dir: Path,
    *,
    n_sample: int,
    n_seeds: int,
    smoothing_hours: float,
    bin_hours: float,
) -> None:
    """PR-4D: gap-aware per-template absolute rate decomposition over time."""
    import numpy as np

    all_results: Dict[str, Dict[str, Any]] = {}
    followup_results: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, soz_map in datasets_list:
        for subject in subjects:
            subject_dir = _subject_dir(dataset, root, subject)
            key = f"{dataset}/{subject}"
            json_path = per_subject_dir / f"{dataset}_{subject}.json"

            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s: raw data dir missing", key)
                continue

            if json_path.exists():
                with open(json_path) as f:
                    existing = json.load(f)
            else:
                logger.info("PR-4D bootstrap base result: %s", key)
                existing = run_subject_interictal_propagation_pr1(
                    subject_dir=subject_dir,
                    dataset=dataset,
                    subject=subject,
                    soz_channels=soz_map.get(subject, []),
                    n_sample=n_sample,
                    n_seeds=n_seeds,
                )

            if "error" in existing:
                all_results[key] = existing
                _save(existing, json_path)
                continue

            ac = existing.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning("Skip %s: adaptive_cluster missing or errored", key)
                all_results[key] = existing
                _save(existing, json_path)
                continue

            loaded = load_subject_propagation_events(subject_dir)
            valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
            labels = np.asarray(ac["labels"], dtype=int)
            chosen_k = int(ac["chosen_k"])
            if valid_events.size != labels.size:
                existing["temporal_dynamics_followup"] = {
                    "error": (
                        f"valid_event_mismatch: valid_events={valid_events.size} "
                        f"labels={labels.size}"
                    )
                }
                all_results[key] = existing
                _save(existing, json_path)
                continue

            v_ranks = loaded["ranks"][:, valid_events]
            v_bools = loaded["bools"][:, valid_events]
            v_times = loaded["event_abs_times"][valid_events]
            templates = build_cluster_templates(v_ranks, v_bools, labels, chosen_k)
            projected = assign_events_to_templates(v_ranks, v_bools, templates)
            assignable = projected >= 0
            if not np.any(assignable):
                existing["temporal_dynamics_followup"] = {"error": "no_assignable_events"}
                all_results[key] = existing
                _save(existing, json_path)
                continue

            times_assignable = v_times[assignable]
            projected_assignable = projected[assignable]

            result = compute_continuous_template_dynamics(
                event_abs_times=times_assignable,
                cluster_labels=projected_assignable,
                n_clusters=chosen_k,
                dataset=dataset,
                coverage_ranges=loaded.get("block_time_ranges"),
                smoothing_hours=smoothing_hours,
                bin_hours=bin_hours,
            )
            follow = {
                "subject": subject,
                "dataset": dataset,
                "chosen_k": chosen_k,
                "stable_k": int(ac.get("stable_k", chosen_k) or chosen_k),
                **result,
            }
            existing["temporal_dynamics_followup"] = follow
            followup_results[key] = follow
            all_results[key] = existing
            _save(existing, json_path)
            logger.info(
                "PR-4D: %s  k=%d  n=%d  dom_frac=%.3f",
                key,
                chosen_k,
                int(result.get("n_events_used", 0)),
                result.get("summary", {}).get("dominant_rate_fraction", float("nan")),
            )

    cohort = summarize_propagation_cohort(all_results)
    _save(all_results, RESULTS_DIR / "pr1_subject_summary.json")
    _save(cohort, RESULTS_DIR / "pr1_cohort_summary.json")
    _save(followup_results, RESULTS_DIR / "pr4a_followup_template_mix_dynamics.json")
    logger.info(
        "PR-4D done. Cohort summary: %s",
        cohort.get("temporal_dynamics_followup_analysis", {}),
    )


def _run_pr4b_step0(
    datasets_list: List,
    per_subject_dir: Path,
    *,
    n_sample: int,
    n_seeds: int,
) -> None:
    """PR-4B Step 0: relative-lag validation under fixed rank clusters."""
    import numpy as np

    all_results: Dict[str, Dict[str, Any]] = {}
    lag_validation_results: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, soz_map in datasets_list:
        for subject in subjects:
            subject_dir = _subject_dir(dataset, root, subject)
            key = f"{dataset}/{subject}"
            json_path = per_subject_dir / f"{dataset}_{subject}.json"

            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s: raw data dir missing", key)
                continue

            if json_path.exists():
                with open(json_path) as f:
                    existing = json.load(f)
            else:
                logger.info("PR-4B Step 0 bootstrap base result: %s", key)
                existing = run_subject_interictal_propagation_pr1(
                    subject_dir=subject_dir,
                    dataset=dataset,
                    subject=subject,
                    soz_channels=soz_map.get(subject, []),
                    n_sample=n_sample,
                    n_seeds=n_seeds,
                )

            if "error" in existing:
                all_results[key] = existing
                _save(existing, json_path)
                continue

            ac = existing.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning("Skip %s: adaptive_cluster missing or errored", key)
                all_results[key] = existing
                _save(existing, json_path)
                continue

            loaded = load_subject_propagation_events(subject_dir)
            valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
            labels = np.asarray(ac["labels"], dtype=int)
            chosen_k = int(ac["chosen_k"])

            if valid_events.size != labels.size:
                existing["absolute_lag_validation"] = {
                    "error": (
                        f"valid_event_mismatch: valid_events={valid_events.size} "
                        f"labels={labels.size}"
                    )
                }
                all_results[key] = existing
                _save(existing, json_path)
                continue

            validation = validate_absolute_lag_clustering(
                ranks=loaded["ranks"],
                lag_raw=loaded["lag_raw"],
                bools=loaded["bools"],
                cluster_labels=labels,
                n_clusters=chosen_k,
                valid_event_indices=valid_events,
                n_sample=n_sample,
                seed=0,
                min_shared_channels=3,
                min_participating=5,
            )
            validation.update(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "chosen_k": chosen_k,
                    "stable_k": int(ac.get("stable_k", chosen_k) or chosen_k),
                }
            )
            existing["absolute_lag_validation"] = validation
            lag_validation_results[key] = validation
            all_results[key] = existing
            _save(existing, json_path)
            logger.info(
                "PR-4B Step 0: %s  k=%d  eligible=%.3f  median_r=%.3f  pass=%s",
                key,
                chosen_k,
                validation.get("eligible_fraction", float("nan")),
                validation.get("eligible_median_r", float("nan")),
                validation.get("validation_pass"),
            )

    cohort = summarize_propagation_cohort(all_results)
    _save(all_results, RESULTS_DIR / "pr1_subject_summary.json")
    _save(cohort, RESULTS_DIR / "pr1_cohort_summary.json")
    _save(lag_validation_results, RESULTS_DIR / "pr4b_lag_validation.json")
    logger.info(
        "PR-4B Step 0 done. Cohort lag summary: %s",
        cohort.get("absolute_lag_validation_analysis", {}),
    )


def _run_pr4b_step1(
    datasets_list: List,
    per_subject_dir: Path,
    *,
    n_sample: int,
    n_seeds: int,
    rate_bin_hours: float,
    min_events_per_rate_bin: int,
) -> None:
    """PR-4B Step 1: high-rate vs low-rate within-cluster tau."""
    import numpy as np

    all_results: Dict[str, Dict[str, Any]] = {}
    coupling_results: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, soz_map in datasets_list:
        for subject in subjects:
            subject_dir = _subject_dir(dataset, root, subject)
            key = f"{dataset}/{subject}"
            json_path = per_subject_dir / f"{dataset}_{subject}.json"

            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s: raw data dir missing", key)
                continue

            if json_path.exists():
                with open(json_path) as f:
                    existing = json.load(f)
            else:
                logger.info("PR-4B Step 1 bootstrap base result: %s", key)
                existing = run_subject_interictal_propagation_pr1(
                    subject_dir=subject_dir,
                    dataset=dataset,
                    subject=subject,
                    soz_channels=soz_map.get(subject, []),
                    n_sample=n_sample,
                    n_seeds=n_seeds,
                )

            if "error" in existing:
                all_results[key] = existing
                _save(existing, json_path)
                continue

            ac = existing.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning("Skip %s: adaptive_cluster missing or errored", key)
                all_results[key] = existing
                _save(existing, json_path)
                continue

            loaded = load_subject_propagation_events(subject_dir)
            valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
            labels = np.asarray(ac["labels"], dtype=int)
            chosen_k = int(ac["chosen_k"])

            if valid_events.size != labels.size:
                existing["rate_state_coupling"] = {
                    "error": (
                        f"valid_event_mismatch: valid_events={valid_events.size} "
                        f"labels={labels.size}"
                    )
                }
                all_results[key] = existing
                _save(existing, json_path)
                continue

            coupling = compute_rate_state_coupling(
                event_abs_times=loaded["event_abs_times"],
                ranks=loaded["ranks"],
                lag_raw=loaded["lag_raw"],
                bools=loaded["bools"],
                cluster_labels=labels,
                n_clusters=chosen_k,
                valid_event_indices=valid_events,
                rate_bin_hours=rate_bin_hours,
                min_events_per_bin=min_events_per_rate_bin,
                n_sample=n_sample,
                n_seeds=n_seeds,
                min_shared_channels=3,
                min_center_participation=5,
            )
            coupling.update(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "chosen_k": chosen_k,
                    "stable_k": int(ac.get("stable_k", chosen_k) or chosen_k),
                }
            )
            existing["rate_state_coupling"] = coupling
            coupling_results[key] = coupling
            all_results[key] = existing
            _save(existing, json_path)
            logger.info(
                "PR-4B Step 1: %s  k=%d  high=%d  low=%d  raw_delta=%.4f  centered_delta=%.4f",
                key,
                chosen_k,
                coupling.get("state_event_counts", {}).get("high", 0),
                coupling.get("state_event_counts", {}).get("low", 0),
                coupling.get("subject_raw_delta") or float("nan"),
                coupling.get("subject_centered_delta") or float("nan"),
            )

    cohort = summarize_propagation_cohort(all_results)
    _save(all_results, RESULTS_DIR / "pr1_subject_summary.json")
    _save(cohort, RESULTS_DIR / "pr1_cohort_summary.json")
    _save(coupling_results, RESULTS_DIR / "pr4b_step1_rate_coupling.json")
    logger.info(
        "PR-4B Step 1 done. Cohort rate-coupling summary: %s",
        cohort.get("rate_state_coupling_analysis", {}),
    )


def _run_pr4b_step23(
    datasets_list: List,
    per_subject_dir: Path,
    *,
    n_sample: int,
    n_seeds: int,
    rate_bin_hours: float,
    min_events_per_rate_bin: int,
) -> None:
    """PR-4B Step 2-3: L3 lag span/Pearson plus L1 occupancy-rate."""
    import numpy as np

    all_results: Dict[str, Dict[str, Any]] = {}
    coupling_results: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, soz_map in datasets_list:
        for subject in subjects:
            subject_dir = _subject_dir(dataset, root, subject)
            key = f"{dataset}/{subject}"
            json_path = per_subject_dir / f"{dataset}_{subject}.json"

            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s: raw data dir missing", key)
                continue

            if json_path.exists():
                with open(json_path) as f:
                    existing = json.load(f)
            else:
                logger.info("PR-4B Step 2-3 bootstrap base result: %s", key)
                existing = run_subject_interictal_propagation_pr1(
                    subject_dir=subject_dir,
                    dataset=dataset,
                    subject=subject,
                    soz_channels=soz_map.get(subject, []),
                    n_sample=n_sample,
                    n_seeds=n_seeds,
                )

            if "error" in existing:
                all_results[key] = existing
                _save(existing, json_path)
                continue

            ac = existing.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning("Skip %s: adaptive_cluster missing or errored", key)
                all_results[key] = existing
                _save(existing, json_path)
                continue

            loaded = load_subject_propagation_events(subject_dir)
            valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
            labels = np.asarray(ac["labels"], dtype=int)
            chosen_k = int(ac["chosen_k"])

            if valid_events.size != labels.size:
                existing["rate_state_coupling"] = {
                    "error": (
                        f"valid_event_mismatch: valid_events={valid_events.size} "
                        f"labels={labels.size}"
                    )
                }
                all_results[key] = existing
                _save(existing, json_path)
                continue

            if not existing.get("absolute_lag_validation"):
                validation = validate_absolute_lag_clustering(
                    ranks=loaded["ranks"],
                    lag_raw=loaded["lag_raw"],
                    bools=loaded["bools"],
                    cluster_labels=labels,
                    n_clusters=chosen_k,
                    valid_event_indices=valid_events,
                    n_sample=n_sample,
                    seed=0,
                    min_shared_channels=3,
                    min_participating=5,
                )
                validation.update(
                    {
                        "dataset": dataset,
                        "subject": subject,
                        "chosen_k": chosen_k,
                        "stable_k": int(ac.get("stable_k", chosen_k) or chosen_k),
                    }
                )
                existing["absolute_lag_validation"] = validation

            coupling = compute_rate_state_coupling(
                event_abs_times=loaded["event_abs_times"],
                ranks=loaded["ranks"],
                lag_raw=loaded["lag_raw"],
                bools=loaded["bools"],
                cluster_labels=labels,
                n_clusters=chosen_k,
                valid_event_indices=valid_events,
                rate_bin_hours=rate_bin_hours,
                min_events_per_bin=min_events_per_rate_bin,
                n_sample=n_sample,
                n_seeds=n_seeds,
                min_shared_channels=3,
                min_center_participation=5,
                min_participating_l3=5,
            )
            coupling.update(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "chosen_k": chosen_k,
                    "stable_k": int(ac.get("stable_k", chosen_k) or chosen_k),
                    "l3_validation_pass": bool(
                        existing.get("absolute_lag_validation", {}).get("validation_pass", False)
                    ),
                }
            )
            existing["rate_state_coupling"] = coupling
            coupling_results[key] = coupling
            all_results[key] = existing
            _save(existing, json_path)
            logger.info(
                "PR-4B Step 2-3: %s  k=%d  lag_delta=%.4f  pearson_delta=%.4f  dom_rho=%.4f",
                key,
                chosen_k,
                coupling.get("subject_lag_span_delta") or float("nan"),
                coupling.get("subject_pearson_r_delta") or float("nan"),
                coupling.get("l1", {}).get("dominant_cluster", {}).get(
                    "occupancy_rate_spearman_rho",
                    float("nan"),
                ),
            )

    cohort = summarize_propagation_cohort(all_results)
    _save(all_results, RESULTS_DIR / "pr1_subject_summary.json")
    _save(cohort, RESULTS_DIR / "pr1_cohort_summary.json")
    _save(coupling_results, RESULTS_DIR / "pr4b_coupling_summary.json")
    logger.info(
        "PR-4B Step 2-3 done. Cohort rate-coupling summary: %s",
        cohort.get("rate_state_coupling_analysis", {}),
    )


def _run_pr4c(
    datasets_list: List,
    per_subject_dir: Path,
    *,
    n_sample: int,
    n_seeds: int,
    config_name: str = "main",
) -> None:
    """PR-4C: seizure proximity L1/L2/L3 analysis.

    ``config_name`` selects the named window configuration from
    :data:`SEIZURE_PROXIMITY_CONFIGS`. ``main`` writes the per-subject field
    ``seizure_proximity_coupling`` and cohort artifact
    ``pr4c_seizure_proximity.json``; ``auxiliary`` writes
    ``seizure_proximity_coupling_auxiliary`` and
    ``pr4c_seizure_proximity_auxiliary.json``.
    """
    import numpy as np
    from src.interictal_propagation import SEIZURE_PROXIMITY_CONFIGS

    if config_name not in SEIZURE_PROXIMITY_CONFIGS:
        raise ValueError(
            f"Unknown PR-4C config_name={config_name}; "
            f"expected one of {sorted(SEIZURE_PROXIMITY_CONFIGS)}"
        )
    cfg = SEIZURE_PROXIMITY_CONFIGS[config_name]
    output_key = (
        "seizure_proximity_coupling"
        if config_name == "main"
        else f"seizure_proximity_coupling_{config_name}"
    )
    artifact_stem = (
        "pr4c_seizure_proximity"
        if config_name == "main"
        else f"pr4c_seizure_proximity_{config_name}"
    )

    all_results: Dict[str, Dict[str, Any]] = {}
    seizure_results: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, soz_map in datasets_list:
        for subject in subjects:
            subject_dir = _subject_dir(dataset, root, subject)
            key = f"{dataset}/{subject}"
            json_path = per_subject_dir / f"{dataset}_{subject}.json"

            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s: raw data dir missing", key)
                continue

            if json_path.exists():
                with open(json_path) as f:
                    existing = json.load(f)
            else:
                logger.info("PR-4C bootstrap base result: %s", key)
                existing = run_subject_interictal_propagation_pr1(
                    subject_dir=subject_dir,
                    dataset=dataset,
                    subject=subject,
                    soz_channels=soz_map.get(subject, []),
                    n_sample=n_sample,
                    n_seeds=n_seeds,
                )

            if "error" in existing:
                all_results[key] = existing
                _save(existing, json_path)
                continue

            ac = existing.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning("Skip %s: adaptive_cluster missing or errored", key)
                all_results[key] = existing
                _save(existing, json_path)
                continue

            loaded = load_subject_propagation_events(subject_dir)
            valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
            labels = np.asarray(ac["labels"], dtype=int)
            chosen_k = int(ac["chosen_k"])
            if valid_events.size != labels.size:
                existing[output_key] = {
                    "error": (
                        f"valid_event_mismatch: valid_events={valid_events.size} "
                        f"labels={labels.size}"
                    )
                }
                all_results[key] = existing
                _save(existing, json_path)
                continue

            seizure_times = load_seizure_times(subject, dataset)
            coupling = compute_seizure_proximity_coupling(
                event_abs_times=loaded["event_abs_times"],
                ranks=loaded["ranks"],
                lag_raw=loaded["lag_raw"],
                bools=loaded["bools"],
                cluster_labels=labels,
                n_clusters=chosen_k,
                seizure_times=seizure_times,
                valid_event_indices=valid_events,
                coverage_ranges=loaded.get("block_time_ranges"),
                n_sample=n_sample,
                n_seeds=n_seeds,
                min_shared_channels=3,
                min_center_participation=5,
                min_participating_l3=5,
                **cfg,
            )
            coupling.update(
                {
                    "dataset": dataset,
                    "subject": subject,
                    "chosen_k": chosen_k,
                    "stable_k": int(ac.get("stable_k", chosen_k) or chosen_k),
                    "n_seizures_loaded": int(len(seizure_times)),
                    "config_name": config_name,
                }
            )
            existing[output_key] = coupling
            seizure_results[key] = coupling
            all_results[key] = existing
            _save(existing, json_path)
            logger.info(
                "PR-4C[%s]: %s  seizures=%d  usable=%d  warning=%s",
                config_name,
                key,
                len(seizure_times),
                coupling.get("n_seizures_usable", 0),
                coupling.get("warning"),
            )

    subject_summary_path = RESULTS_DIR / "pr1_subject_summary.json"
    merged: Dict[str, Dict[str, Any]] = {}
    if subject_summary_path.exists():
        try:
            with open(subject_summary_path) as f:
                existing_summary = json.load(f)
            if isinstance(existing_summary, dict):
                merged.update(existing_summary)
        except Exception as exc:
            logger.warning("Failed to merge existing subject summary: %s", exc)
    merged.update(all_results)

    cohort = summarize_propagation_cohort(merged)
    _save(merged, subject_summary_path)
    _save(cohort, RESULTS_DIR / "pr1_cohort_summary.json")
    _save(seizure_results, RESULTS_DIR / f"{artifact_stem}.json")
    logger.info(
        "PR-4C[%s] done. Subjects processed this run=%d, total in summary=%d",
        config_name, len(all_results), len(merged),
    )
    cohort_key = (
        "seizure_proximity_analysis"
        if config_name == "main"
        else f"seizure_proximity_analysis_{config_name}"
    )
    logger.info(
        "PR-4C[%s] cohort seizure summary: %s",
        config_name,
        cohort.get(cohort_key, {}),
    )


def _run_pr5_gate(
    datasets_list: List,
    per_subject_dir: Path,
    *,
    min_state_events_for_gate: int = 30,
) -> None:
    """PR-5-A: novel-template falsification gate (main + auxiliary configs).

    For each subject we reuse the PR-4C P0 fix machinery
    (``_build_seizure_proximity_windows`` + the same ``v_ranks/v_rel/v_bools``
    construction as :func:`compute_seizure_proximity_coupling`), build the
    fixed global template library ``T_global`` exactly as PR-4D does, and
    run :func:`compute_novel_template_gate` on the gate-eligible event pool
    (``min_participating_l3=5``). Cohort-level PASS/FAIL is delegated to
    :func:`summarize_pr5_novel_template_gate` which encodes the §3.5
    thresholds; both window configs must PASS for ``overall_pass=True``.
    """
    import numpy as np

    pr5a_dir = per_subject_dir / "pr5a"
    pr5a_dir.mkdir(parents=True, exist_ok=True)

    config_records: Dict[str, List[Dict[str, Any]]] = {
        "main": [],
        "auxiliary": [],
    }
    per_subject_archive: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, _soz_map in datasets_list:
        for subject in subjects:
            key = f"{dataset}/{subject}"
            json_path = per_subject_dir / f"{dataset}_{subject}.json"
            if not json_path.exists():
                logger.warning("PR-5-A skip %s: per-subject PR-1 JSON missing", key)
                continue

            with open(json_path) as f:
                base = json.load(f)
            ac = base.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning(
                    "PR-5-A skip %s: adaptive_cluster missing/errored", key
                )
                continue

            subject_dir = _subject_dir(dataset, root, subject)
            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("PR-5-A skip %s: raw lagPat dir missing", key)
                continue

            loaded = load_subject_propagation_events(subject_dir)
            valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
            labels = np.asarray(ac["labels"], dtype=int)
            chosen_k = int(ac["chosen_k"])
            if valid_events.size != labels.size:
                logger.warning(
                    "PR-5-A skip %s: label/valid mismatch (labels=%d valid=%d)",
                    key,
                    labels.size,
                    valid_events.size,
                )
                continue

            v_times = loaded["event_abs_times"][valid_events]
            v_ranks = loaded["ranks"][:, valid_events]
            v_bools = loaded["bools"][:, valid_events]
            v_rel = _compute_relative_lag_matrix(
                loaded["lag_raw"][:, valid_events], v_bools
            )
            templates = build_cluster_templates(v_ranks, v_bools, labels, chosen_k)
            seizure_times = load_seizure_times(subject, dataset)
            if not seizure_times:
                logger.warning("PR-5-A skip %s: no seizure times", key)
                continue

            subject_payload: Dict[str, Any] = {
                "dataset": dataset,
                "subject": subject,
                "chosen_k": chosen_k,
                "stable_k": int(ac.get("stable_k", chosen_k) or chosen_k),
                "n_seizures_loaded": int(len(seizure_times)),
                "n_valid_events": int(valid_events.size),
                "configs": {},
            }

            for config_name, cfg in SEIZURE_PROXIMITY_CONFIGS.items():
                proximity = _build_seizure_proximity_windows(
                    v_times,
                    seizure_times,
                    baseline_hours=cfg["baseline_hours"],
                    pre_ictal_hours=cfg["pre_ictal_hours"],
                    post_ictal_hours=cfg["post_ictal_hours"],
                )
                gate_result = compute_novel_template_gate(
                    v_ranks=v_ranks,
                    v_rel=v_rel,
                    v_bools=v_bools,
                    templates=templates,
                    proximity_windows=proximity["usable_windows"],
                    min_participating_l3=5,
                    min_shared_channels=3,
                )
                gate_result["n_seizures_usable"] = int(
                    len(proximity["usable_windows"])
                )
                gate_result["window_hours"] = proximity["state_ranges_hours"]
                gate_result["state_event_counts_window_total"] = (
                    proximity["state_event_counts"]
                )
                # Drop the per-event distributions from cohort archive to
                # keep the consolidated JSON small; per-subject JSON keeps
                # them for downstream exploratory diagnostics.
                cohort_record = {
                    "subject_id": subject,
                    "dataset": dataset,
                    "config_name": config_name,
                    "n_events_by_state": gate_result["n_events_by_state"],
                    "n_l3_excluded_by_state": gate_result["n_l3_excluded_by_state"],
                    "n_seizures_usable": gate_result["n_seizures_usable"],
                    "median_by_state": gate_result["median_by_state"],
                    "delta_pre_minus_baseline": gate_result["delta_pre_minus_baseline"],
                    "delta_post_minus_baseline": gate_result["delta_post_minus_baseline"],
                    "n_clusters": gate_result["n_clusters"],
                }
                if (
                    gate_result["n_events_by_state"]["baseline"] == 0
                    and gate_result["n_events_by_state"]["pre"] == 0
                    and gate_result["n_events_by_state"]["post"] == 0
                ):
                    cohort_record["warning"] = "no_gate_eligible_events"
                config_records[config_name].append(cohort_record)
                subject_payload["configs"][config_name] = gate_result

            per_subject_archive[key] = subject_payload
            _save(subject_payload, pr5a_dir / f"{dataset}_{subject}.json")
            logger.info(
                "PR-5-A %s: usable seizures main=%d aux=%d  events main(b/p/p)=%s aux(b/p/p)=%s",
                key,
                subject_payload["configs"]["main"]["n_seizures_usable"],
                subject_payload["configs"]["auxiliary"]["n_seizures_usable"],
                subject_payload["configs"]["main"]["n_events_by_state"],
                subject_payload["configs"]["auxiliary"]["n_events_by_state"],
            )

    cohort_summary = summarize_pr5_novel_template_gate(
        config_records,
        min_state_events_for_gate=int(min_state_events_for_gate),
    )

    artifact_root = RESULTS_DIR / "pr5a_novel_template_gate.json"
    _save(
        {
            "per_subject": config_records,
            "cohort": cohort_summary,
        },
        artifact_root,
    )

    cohort_path = RESULTS_DIR / "pr1_cohort_summary.json"
    cohort_doc: Dict[str, Any] = {}
    if cohort_path.exists():
        try:
            with open(cohort_path) as f:
                loaded_doc = json.load(f)
            if isinstance(loaded_doc, dict):
                cohort_doc = loaded_doc
        except Exception as exc:
            logger.warning("PR-5-A: failed to merge cohort summary: %s", exc)
    cohort_doc["novel_template_gate"] = cohort_summary
    _save(cohort_doc, cohort_path)

    logger.info(
        "PR-5-A done. overall_pass=%s  main_pass=%s aux_pass=%s  n_subjects(main/aux)=%d/%d",
        cohort_summary.get("overall_pass"),
        cohort_summary.get("main", {}).get("gate_pass"),
        cohort_summary.get("auxiliary", {}).get("gate_pass"),
        cohort_summary.get("main", {}).get("n_subjects_eligible", 0),
        cohort_summary.get("auxiliary", {}).get("n_subjects_eligible", 0),
    )


def _run_pr5_recruitment(
    datasets_list: List,
    per_subject_dir: Path,
    *,
    min_state_events_for_gate: int = 30,
    n_boot: int = 2000,
    bootstrap_seed: int = 42,
) -> None:
    """PR-5-B: template recruitment shift (main + auxiliary configs).

    Hard prerequisites (§2.3 fail-fast):

    1. ``pr5a_novel_template_gate.json`` must exist (PR-5-A run).
    2. ``cohort.overall_pass`` must be ``True`` (gate PASSED in both configs).

    Either condition unmet → ``SystemExit(2)`` and **no** PR-5-B artifact is
    written, so a stale recruitment JSON cannot survive a gate FAIL by
    accident.
    """
    import numpy as np

    gate_json_path = RESULTS_DIR / "pr5a_novel_template_gate.json"
    if not gate_json_path.exists():
        logger.error(
            "PR-5-B abort: gate JSON missing at %s. Run --pr5-gate first.",
            gate_json_path,
        )
        raise SystemExit(2)
    try:
        with open(gate_json_path) as f:
            gate_doc = json.load(f)
    except Exception as exc:
        logger.error("PR-5-B abort: failed to load gate JSON: %s", exc)
        raise SystemExit(2)
    cohort_gate = gate_doc.get("cohort", {}) if isinstance(gate_doc, dict) else {}
    if not bool(cohort_gate.get("overall_pass", False)):
        logger.error(
            "PR-5-B abort: gate overall_pass=%s. PR-5-B not allowed; refusing"
            " to write recruitment artifact.",
            cohort_gate.get("overall_pass"),
        )
        raise SystemExit(2)
    retained_subject_keys_by_config = _pr5a_retained_subject_keys_by_config(gate_doc)

    pr5b_dir = per_subject_dir / "pr5b"
    pr5b_dir.mkdir(parents=True, exist_ok=True)

    config_records: Dict[str, List[Dict[str, Any]]] = {
        "main": [],
        "auxiliary": [],
    }
    per_subject_archive: Dict[str, Dict[str, Any]] = {}

    for dataset, root, subjects, _soz_map in datasets_list:
        for subject in subjects:
            key = f"{dataset}/{subject}"
            json_path = per_subject_dir / f"{dataset}_{subject}.json"
            if not json_path.exists():
                logger.warning("PR-5-B skip %s: PR-1 JSON missing", key)
                continue
            with open(json_path) as f:
                base = json.load(f)
            ac = base.get("adaptive_cluster", {})
            if "error" in ac or "labels" not in ac:
                logger.warning("PR-5-B skip %s: adaptive_cluster missing/errored", key)
                continue

            subject_dir = _subject_dir(dataset, root, subject)
            if not subject_dir.exists() or not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("PR-5-B skip %s: raw lagPat dir missing", key)
                continue

            loaded = load_subject_propagation_events(subject_dir)
            valid_events = _valid_event_indices(loaded["bools"], min_participating=3)
            labels = np.asarray(ac["labels"], dtype=int)
            chosen_k = int(ac["chosen_k"])
            if valid_events.size != labels.size:
                logger.warning(
                    "PR-5-B skip %s: label/valid mismatch (labels=%d valid=%d)",
                    key, labels.size, valid_events.size,
                )
                continue

            v_times = loaded["event_abs_times"][valid_events]
            v_bools = loaded["bools"][:, valid_events]

            n_part = np.sum(v_bools > 0, axis=0).astype(int)

            counts_full = np.bincount(labels, minlength=chosen_k).astype(int)
            if int(counts_full.sum()) == 0:
                logger.warning("PR-5-B skip %s: zero events post valid filter", key)
                continue
            dominant_global_id = int(np.argmax(counts_full))

            seizure_times = load_seizure_times(subject, dataset)
            if not seizure_times:
                logger.warning("PR-5-B skip %s: no seizure times", key)
                continue

            block_ranges = loaded.get("block_time_ranges")
            coverage_ranges = (
                [(float(lo), float(hi)) for lo, hi in block_ranges]
                if block_ranges is not None else None
            )

            subject_payload: Dict[str, Any] = {
                "dataset": dataset,
                "subject": subject,
                "chosen_k": chosen_k,
                "stable_k": int(ac.get("stable_k", chosen_k) or chosen_k),
                "dom_global_id": dominant_global_id,
                "n_seizures_loaded": int(len(seizure_times)),
                "n_valid_events": int(valid_events.size),
                "configs": {},
            }

            for config_name, cfg in SEIZURE_PROXIMITY_CONFIGS.items():
                proximity = _build_seizure_proximity_windows(
                    v_times,
                    seizure_times,
                    baseline_hours=cfg["baseline_hours"],
                    pre_ictal_hours=cfg["pre_ictal_hours"],
                    post_ictal_hours=cfg["post_ictal_hours"],
                )

                state_ranges = {
                    "baseline": cfg["baseline_hours"],
                    "pre": cfg["pre_ictal_hours"],
                    "post": cfg["post_ictal_hours"],
                }
                windows_with_hours: List[Dict[str, Any]] = []
                for w in proximity["usable_windows"]:
                    sz_t = float(w["seizure_time"])
                    covered = {}
                    for state_name, (lo, hi) in state_ranges.items():
                        t0 = sz_t + float(lo) * 3600.0
                        t1 = sz_t + float(hi) * 3600.0
                        if coverage_ranges is None:
                            covered[state_name] = float(hi - lo)
                        else:
                            covered[state_name] = (
                                _intersect_seconds(t0, t1, coverage_ranges) / 3600.0
                            )
                    enriched = dict(w)
                    enriched["state_covered_hours"] = covered
                    windows_with_hours.append(enriched)

                shift = compute_template_recruitment_shift(
                    cluster_labels=labels,
                    n_part_per_event=n_part,
                    n_clusters=chosen_k,
                    dominant_global_id=dominant_global_id,
                    proximity_windows=windows_with_hours,
                    min_participating_l3=5,
                )
                shift["n_seizures_usable"] = int(len(windows_with_hours))
                shift["window_hours"] = proximity["state_ranges_hours"]
                shift["state_event_counts_window_total"] = (
                    proximity["state_event_counts"]
                )
                shift["retained_for_pr5b"] = bool(
                    key in retained_subject_keys_by_config.get(config_name, set())
                )

                cohort_record = {
                    "subject_id": subject,
                    "dataset": dataset,
                    "config_name": config_name,
                    "dom_global_id": shift["dom_global_id"],
                    "dom_window_ids_per_window": shift["dom_window_ids_per_window"],
                    "dom_agreement": shift["dom_agreement"],
                    "n_windows_used": shift["n_windows_used"],
                    "n_windows_total": shift["n_windows_total"],
                    "n_seizures_usable": shift["n_seizures_usable"],
                    "min_participating_l3": shift["min_participating_l3"],
                    "n_clusters": shift["n_clusters"],
                    "weighted_per_state": shift["weighted_per_state"],
                    "deltas": shift["deltas"],
                    "share_state_n_eligible_windows": shift["share_state_n_eligible_windows"],
                    "share_pair_eligible": shift["share_pair_eligible"],
                }
                if shift["retained_for_pr5b"]:
                    config_records[config_name].append(cohort_record)
                subject_payload["configs"][config_name] = shift

            per_subject_archive[key] = subject_payload
            _save(subject_payload, pr5b_dir / f"{dataset}_{subject}.json")
            logger.info(
                "PR-5-B %s: dom_global_id=%d  windows main/aux=%d/%d  agreement main/aux=%.2f/%.2f",
                key, dominant_global_id,
                subject_payload["configs"]["main"]["n_windows_used"],
                subject_payload["configs"]["auxiliary"]["n_windows_used"],
                subject_payload["configs"]["main"]["dom_agreement"]
                if np.isfinite(subject_payload["configs"]["main"]["dom_agreement"]) else float("nan"),
                subject_payload["configs"]["auxiliary"]["dom_agreement"]
                if np.isfinite(subject_payload["configs"]["auxiliary"]["dom_agreement"]) else float("nan"),
            )

    cohort_summary = summarize_pr5_template_recruitment_shift(
        config_records,
        n_boot=int(n_boot),
        bootstrap_seed=int(bootstrap_seed),
    )

    artifact_root = RESULTS_DIR / "pr5b_recruitment_shift.json"
    _save(
        {"per_subject": config_records, "cohort": cohort_summary},
        artifact_root,
    )

    cohort_path = RESULTS_DIR / "pr1_cohort_summary.json"
    cohort_doc: Dict[str, Any] = {}
    if cohort_path.exists():
        try:
            with open(cohort_path) as f:
                loaded_doc = json.load(f)
            if isinstance(loaded_doc, dict):
                cohort_doc = loaded_doc
        except Exception as exc:
            logger.warning("PR-5-B: failed to merge cohort summary: %s", exc)
    cohort_doc["template_recruitment_shift"] = cohort_summary
    _save(cohort_doc, cohort_path)

    sens = cohort_summary.get("sensitivity", {})
    logger.info(
        "PR-5-B done. n_subjects(main/aux)=%d/%d  sensitivity strong=%s medium=%s descriptive=%s",
        cohort_summary.get("main", {}).get("n_subjects", 0),
        cohort_summary.get("auxiliary", {}).get("n_subjects", 0),
        sens.get("overall_strong"),
        sens.get("overall_medium"),
        sens.get("overall_descriptive"),
    )


def _pr5a_retained_subject_keys_by_config(
    gate_doc: Dict[str, Any],
) -> Dict[str, Set[str]]:
    """Return config-specific retained subject keys from PR-5-A artifact.

    PR-5-B must inherit the exact PR-5-A retained subset per config. The gate
    artifact stores every processed subject in ``per_subject[config]`` and the
    config-specific exclusions in ``cohort[config].ineligible_subjects``.
    """
    per_subject = gate_doc.get("per_subject", {}) if isinstance(gate_doc, dict) else {}
    cohort = gate_doc.get("cohort", {}) if isinstance(gate_doc, dict) else {}
    retained: Dict[str, Set[str]] = {}

    for config_name in SEIZURE_PROXIMITY_CONFIGS:
        all_keys: Set[str] = set()
        for rec in per_subject.get(config_name, []):
            if not isinstance(rec, dict):
                continue
            dataset = rec.get("dataset")
            subject = rec.get("subject_id") or rec.get("subject")
            if dataset and subject:
                all_keys.add(f"{dataset}/{subject}")

        ineligible_keys: Set[str] = set()
        config_cohort = cohort.get(config_name, {})
        if isinstance(config_cohort, dict):
            for rec in config_cohort.get("ineligible_subjects", []):
                if not isinstance(rec, dict):
                    continue
                dataset = rec.get("dataset")
                subject = rec.get("subject_id") or rec.get("subject")
                if dataset and subject:
                    ineligible_keys.add(f"{dataset}/{subject}")

        retained[config_name] = all_keys - ineligible_keys
    return retained


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Factored out so tests (and other tooling) can introspect available
    flags without executing ``main``.
    """
    parser = argparse.ArgumentParser(
        description="Interictal group-event internal propagation PR-1 analysis"
    )
    parser.add_argument("--dataset", choices=["yuquan", "epilepsiae", "both"], default="both")
    parser.add_argument("--subjects", nargs="+", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run chengshuai + 548 only")
    parser.add_argument("--pr25", action="store_true", help="PR-2.5: cross-time template reproducibility only")
    parser.add_argument("--augment-cluster-bias", action="store_true",
                        help="Augment existing JSONs with within-cluster identity-bias")
    parser.add_argument("--pr4a", action="store_true", help="PR-4A: temporal occupancy dynamics only")
    parser.add_argument(
        "--pr4a-followup",
        action="store_true",
        help="PR-4D: gap-aware per-template absolute rate decomposition",
    )
    parser.add_argument("--pr4b-step0", action="store_true",
                        help="PR-4B Step 0: absolute lag validation only")
    parser.add_argument("--pr4b-step1", action="store_true",
                        help="PR-4B Step 1: rate-state within-cluster tau only")
    parser.add_argument("--pr4b-step23", action="store_true",
                        help="PR-4B Step 2-3: lag span/Pearson plus occupancy-rate")
    parser.add_argument("--pr4c", action="store_true",
                        help="PR-4C: seizure proximity L1/L2/L3 analysis (main config: 4/1/1 h)")
    parser.add_argument("--pr4c-auxiliary", action="store_true",
                        help="PR-4C: seizure proximity analysis with auxiliary config (2/0.5/1 h) for sensitivity check")
    parser.add_argument("--pr5-gate", action="store_true",
                        help=(
                            "PR-5-A novel-template falsification gate (main + auxiliary configs)."
                            " Hard prerequisite for PR-5-B; per-subject results in"
                            " results/interictal_propagation/per_subject/pr5a/."
                        ))
    parser.add_argument("--pr5-recruitment", action="store_true",
                        help=(
                            "PR-5-B template recruitment shift (main + auxiliary configs)."
                            " Requires --pr5-gate to have produced overall_pass=True"
                            " in pr5a_novel_template_gate.json; aborts with non-zero"
                            " exit code otherwise."
                        ))
    parser.add_argument("--pr5-recruitment-n-boot", type=int, default=2000,
                        help="PR-5-B subject-level cluster bootstrap iterations (default 2000).")
    parser.add_argument("--pr5-recruitment-bootstrap-seed", type=int, default=42,
                        help="PR-5-B bootstrap seed (default 42).")
    parser.add_argument(
        "--pr5-min-state-events",
        type=int,
        default=30,
        help=(
            "PR-5-A per-subject ineligibility threshold (minimum events per"
            " baseline/pre/post state, pooled across usable seizure windows)."
            " Default = 30 per archive §2.3 fail-fast contract."
        ),
    )
    parser.add_argument("--bin-hours", type=float, default=1.0, help="PR-4A occupancy bin width in hours")
    parser.add_argument("--rate-bin-hours", type=float, default=2.0,
                        help="PR-4B local rate bin width in hours")
    parser.add_argument("--min-events-per-rate-bin", type=int, default=20,
                        help="PR-4B minimum events required for an eligible rate bin")
    parser.add_argument(
        "--followup-smoothing-hours",
        type=float,
        default=2.0,
        help="PR-4D Gaussian KDE bandwidth in hours for rate envelope",
    )
    parser.add_argument(
        "--followup-bin-hours",
        type=float,
        default=1.0,
        help="PR-4D histogram bin width in hours",
    )
    parser.add_argument("--n-sample", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument(
        "--epilepsiae-root", type=Path, default=None,
        help=(
            "Override Epilepsiae lagPat root. Default = canonical legacy tree "
            "/mnt/epilepsia_data/interilca_inter_results/all_data_lns/. Pass "
            "results/epilepsiae_lagpat_backfill to consume the new-pipeline "
            "lagPat in Stage D sensitivity audits."
        ),
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help=(
            "Override the per-subject + cohort summary output directory "
            "(default = results/interictal_propagation). Required for Stage D "
            "sensitivity smoke so legacy outputs are not overwritten."
        ),
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    global RESULTS_DIR
    if args.output_root is not None:
        RESULTS_DIR = args.output_root
        logger.info("RESULTS_DIR overridden -> %s", RESULTS_DIR)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    per_subject_dir = RESULTS_DIR / "per_subject"
    soz_yq = _load_soz(SOZ_FILE_YQ)
    soz_epi = _load_soz(SOZ_FILE_EPI)

    if args.smoke:
        yq_subjects = ["chengshuai"]
        epi_subjects = ["548"]
    elif args.subjects:
        yq_subjects = [s for s in args.subjects if s in YUQUAN_SUBJECTS]
        epi_subjects = [s for s in args.subjects if s in EPILEPSIAE_SUBJECTS]
    else:
        yq_subjects = YUQUAN_SUBJECTS
        epi_subjects = EPILEPSIAE_SUBJECTS

    epi_root = args.epilepsiae_root if args.epilepsiae_root is not None else EPILEPSIAE_ROOT
    if args.epilepsiae_root is not None:
        logger.info("Epilepsiae root overridden -> %s", epi_root)

    datasets_list = []
    if args.dataset in ("yuquan", "both"):
        datasets_list.append(("yuquan", YUQUAN_ROOT, yq_subjects, soz_yq))
    if args.dataset in ("epilepsiae", "both"):
        datasets_list.append(("epilepsiae", epi_root, epi_subjects, soz_epi))

    if args.pr25:
        _run_pr25(datasets_list, per_subject_dir)
        return

    if args.augment_cluster_bias:
        _augment_cluster_bias(datasets_list, per_subject_dir)
        return

    if args.pr4a:
        _run_pr4a(
            datasets_list,
            per_subject_dir,
            n_sample=args.n_sample,
            n_seeds=args.n_seeds,
            bin_hours=args.bin_hours,
        )
        return

    if args.pr4a_followup:
        _run_pr4a_followup(
            datasets_list,
            per_subject_dir,
            n_sample=args.n_sample,
            n_seeds=args.n_seeds,
            smoothing_hours=args.followup_smoothing_hours,
            bin_hours=args.followup_bin_hours,
        )
        return

    if args.pr4b_step0:
        _run_pr4b_step0(
            datasets_list,
            per_subject_dir,
            n_sample=args.n_sample,
            n_seeds=args.n_seeds,
        )
        return

    if args.pr4b_step1:
        _run_pr4b_step1(
            datasets_list,
            per_subject_dir,
            n_sample=args.n_sample,
            n_seeds=args.n_seeds,
            rate_bin_hours=args.rate_bin_hours,
            min_events_per_rate_bin=args.min_events_per_rate_bin,
        )
        return

    if args.pr4b_step23:
        _run_pr4b_step23(
            datasets_list,
            per_subject_dir,
            n_sample=args.n_sample,
            n_seeds=args.n_seeds,
            rate_bin_hours=args.rate_bin_hours,
            min_events_per_rate_bin=args.min_events_per_rate_bin,
        )
        return

    if args.pr4c:
        _run_pr4c(
            datasets_list,
            per_subject_dir,
            n_sample=args.n_sample,
            n_seeds=args.n_seeds,
            config_name="main",
        )
        return

    if args.pr4c_auxiliary:
        _run_pr4c(
            datasets_list,
            per_subject_dir,
            n_sample=args.n_sample,
            n_seeds=args.n_seeds,
            config_name="auxiliary",
        )
        return

    if args.pr5_gate:
        _run_pr5_gate(
            datasets_list,
            per_subject_dir,
            min_state_events_for_gate=args.pr5_min_state_events,
        )
        return

    if args.pr5_recruitment:
        _run_pr5_recruitment(
            datasets_list,
            per_subject_dir,
            min_state_events_for_gate=args.pr5_min_state_events,
            n_boot=args.pr5_recruitment_n_boot,
            bootstrap_seed=args.pr5_recruitment_bootstrap_seed,
        )
        return

    subject_results: Dict[str, Dict[str, Any]] = {}
    for dataset, root, subjects, soz_map in datasets_list:
        for subject in subjects:
            subject_dir = _subject_dir(dataset, root, subject)
            if not subject_dir.exists():
                logger.warning("Skip %s/%s: subject dir missing", dataset, subject)
                continue
            if not list(subject_dir.glob("*_lagPat.npz")):
                logger.warning("Skip %s/%s: no lagPat", dataset, subject)
                continue

            key = f"{dataset}/{subject}"
            logger.info("Running %s", key)
            try:
                result = run_subject_interictal_propagation_pr1(
                    subject_dir=subject_dir,
                    dataset=dataset,
                    subject=subject,
                    soz_channels=soz_map.get(subject, []),
                    n_sample=args.n_sample,
                    n_seeds=args.n_seeds,
                )
            except Exception as exc:
                logger.exception("Failed %s", key)
                result = {"dataset": dataset, "subject": subject, "error": str(exc)}

            subject_results[key] = result
            _save(result, per_subject_dir / f"{dataset}_{subject}.json")

    cohort_summary = summarize_propagation_cohort(subject_results)
    _save(subject_results, RESULTS_DIR / "pr1_subject_summary.json")
    _save(cohort_summary, RESULTS_DIR / "pr1_cohort_summary.json")


if __name__ == "__main__":
    main()
