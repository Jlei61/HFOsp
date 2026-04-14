#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.interictal_propagation import (  # noqa: E402
    _valid_event_indices,
    assign_events_to_templates,
    build_cluster_templates,
    compute_rate_state_coupling,
    validate_absolute_lag_clustering,
    compute_temporal_cluster_dynamics,
    compute_time_split_reproducibility,
    compute_within_cluster_centered_tau,
    load_subject_propagation_events,
    run_subject_interictal_propagation_pr1,
    summarize_propagation_cohort,
)


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
            subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"
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
            subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"
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
            subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"
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
            subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"
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
            subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"
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
            subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"
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


def main() -> None:
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
    parser.add_argument("--pr4b-step0", action="store_true",
                        help="PR-4B Step 0: absolute lag validation only")
    parser.add_argument("--pr4b-step1", action="store_true",
                        help="PR-4B Step 1: rate-state within-cluster tau only")
    parser.add_argument("--pr4b-step23", action="store_true",
                        help="PR-4B Step 2-3: lag span/Pearson plus occupancy-rate")
    parser.add_argument("--bin-hours", type=float, default=1.0, help="PR-4A occupancy bin width in hours")
    parser.add_argument("--rate-bin-hours", type=float, default=2.0,
                        help="PR-4B local rate bin width in hours")
    parser.add_argument("--min-events-per-rate-bin", type=int, default=20,
                        help="PR-4B minimum events required for an eligible rate bin")
    parser.add_argument("--n-sample", type=int, default=200)
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

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

    datasets_list = []
    if args.dataset in ("yuquan", "both"):
        datasets_list.append(("yuquan", YUQUAN_ROOT, yq_subjects, soz_yq))
    if args.dataset in ("epilepsiae", "both"):
        datasets_list.append(("epilepsiae", EPILEPSIAE_ROOT, epi_subjects, soz_epi))

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

    subject_results: Dict[str, Dict[str, Any]] = {}
    for dataset, root, subjects, soz_map in datasets_list:
        for subject in subjects:
            subject_dir = root / subject if dataset == "yuquan" else root / subject / "all_recs"
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
