#!/usr/bin/env python3
"""Topic 4 Step 1 batch driver: principal curve + GOF + angle on 35 stable_k=2 cohort.

For each `eligible_for_main = true` subject from `step0_audit.csv`:
    1. Resolve subject_dir (yuquan: /mnt/yuquan_data/yuquan_24h_edf/<sub>;
       epilepsiae: /mnt/epilepsia_data/.../<sub>/all_recs).
    2. Load events via `load_subject_propagation_events`.
    3. Read PR-2 JSON for `adaptive_cluster.labels` (k=2 labels, indexed by
       `_valid_event_indices(bools, min_participating=3)`).
    4. Run `run_step1_subject` with PR-2 labels mapped onto Topic 4 eligibility
       (n_participating >= 6).
    5. Save per-subject JSON and append to cohort CSV.

Outputs:
    results/topic4_attractor/per_subject/<sid>.json
    results/topic4_attractor/step1_cohort_summary.csv
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.interictal_propagation import (  # noqa: E402
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.topic4_attractor_diagnostics import (  # noqa: E402
    TOPIC4_MIN_PARTICIPATING,
    run_step1_subject,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("topic4_step1")

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
PR2_PER_SUBJECT_DIR = REPO_ROOT / "results" / "interictal_propagation" / "per_subject"
OUT_DIR = REPO_ROOT / "results" / "topic4_attractor"
PER_SUBJECT_DIR = OUT_DIR / "per_subject"
COHORT_CSV = OUT_DIR / "step1_cohort_summary.csv"
AUDIT_CSV = OUT_DIR / "step0_audit.csv"

GOF_THRESHOLD = 0.6
N_COMPONENTS = 3
PR2_VALID_MIN_PART = 3  # Matches PR-2's `_valid_event_indices` default


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        return super().default(obj)


def _epilepsiae_subject_dir(subject: str) -> Path:
    legacy = EPILEPSIAE_ROOT / subject / "all_recs"
    if legacy.exists():
        return legacy
    return EPILEPSIAE_ROOT / subject


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    return _epilepsiae_subject_dir(subject)


def _load_main_cohort(audit_csv: Path) -> List[Tuple[str, str, str]]:
    """Return list of (sid, dataset, subject) for eligible_for_main = true rows."""
    out: List[Tuple[str, str, str]] = []
    with open(audit_csv) as f:
        for row in csv.DictReader(f):
            if row.get("eligible_for_main", "").lower() == "true":
                out.append((row["sid"], row["dataset"], row["subject"]))
    return out


def _read_pr2_labels(json_path: Path) -> Tuple[Optional[np.ndarray], Optional[List[List[float]]], Optional[List[str]], int, int]:
    """Return (labels, template_ranks, channel_names_pr2, chosen_k, n_valid_pr2)."""
    if not json_path.exists():
        return None, None, None, -1, -1
    with open(json_path) as f:
        d = json.load(f)
    ac = d.get("adaptive_cluster", {})
    labels = ac.get("labels")
    chosen_k = int(ac.get("chosen_k", -1))
    n_valid = int(ac.get("n_valid_events", -1))
    template_ranks = [c.get("template_rank") for c in ac.get("clusters", []) or []]
    channel_names_pr2 = d.get("channel_names")
    if labels is None:
        return None, template_ranks, channel_names_pr2, chosen_k, n_valid
    return np.asarray(labels, dtype=int), template_ranks, channel_names_pr2, chosen_k, n_valid


def _run_one(sid: str, dataset: str, subject: str) -> Dict[str, Any]:
    sub_dir = _subject_dir(dataset, subject)
    pr2_json = PR2_PER_SUBJECT_DIR / f"{sid}.json"

    base: Dict[str, Any] = {
        "sid": sid,
        "dataset": dataset,
        "subject": subject,
        "subject_dir": str(sub_dir),
        "pr2_json": str(pr2_json),
        "min_participating": TOPIC4_MIN_PARTICIPATING,
        "gof_threshold": GOF_THRESHOLD,
        "n_components": N_COMPONENTS,
    }

    if not sub_dir.exists():
        base["error"] = "subject_dir_missing"
        return base

    loaded = load_subject_propagation_events(sub_dir)
    ranks = np.asarray(loaded["ranks"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=bool)
    chans_loader = list(loaded.get("channel_names", []))

    if ranks.size == 0 or bools.size == 0:
        base["error"] = "loader_empty"
        return base

    labels_pr2, template_ranks, chans_pr2, chosen_k, n_valid_pr2 = _read_pr2_labels(pr2_json)

    if chans_pr2 is not None and chans_loader != list(chans_pr2):
        base["warning"] = "channel_names_mismatch_loader_vs_pr2"
        base["chans_loader"] = chans_loader
        base["chans_pr2"] = list(chans_pr2)

    valid_events_pr2 = _valid_event_indices(bools, min_participating=PR2_VALID_MIN_PART)
    if labels_pr2 is None or labels_pr2.shape[0] != valid_events_pr2.shape[0]:
        base["warning_labels"] = (
            f"pr2_labels_unavailable_or_length_mismatch "
            f"(labels={None if labels_pr2 is None else labels_pr2.shape[0]}, "
            f"valid_events_pr2={valid_events_pr2.shape[0]})"
        )
        labels_pr2 = None

    step1 = run_step1_subject(
        ranks, bools,
        pr2_valid_events=valid_events_pr2,
        pr2_labels=labels_pr2,
        pr2_n_clusters=2,
        min_participating=TOPIC4_MIN_PARTICIPATING,
        n_components=N_COMPONENTS,
        gof_threshold=GOF_THRESHOLD,
    )
    base.update(step1)
    base["chosen_k_pr2"] = int(chosen_k)
    base["n_valid_pr2"] = int(n_valid_pr2)
    return base


def _cohort_row(per_subject: Dict[str, Any]) -> Dict[str, Any]:
    pca = per_subject.get("pca", {}) or {}
    pc_ratio = pca.get("explained_variance_ratio") or []
    gof = per_subject.get("gof", {}) or {}
    angle = per_subject.get("angle_to_kmeans_axis", {}) or {}
    curve = per_subject.get("principal_curve", {}) or {}
    n_in_cluster = per_subject.get("n_in_cluster", []) or []
    return {
        "sid": per_subject.get("sid", ""),
        "dataset": per_subject.get("dataset", ""),
        "subject": per_subject.get("subject", ""),
        "n_events_eligible": per_subject.get("n_events_eligible", 0),
        "n_chan_union": per_subject.get("n_chan_union", 0),
        "pc1_ratio": pc_ratio[0] if len(pc_ratio) >= 1 else "",
        "pc2_ratio": pc_ratio[1] if len(pc_ratio) >= 2 else "",
        "pc3_ratio": pc_ratio[2] if len(pc_ratio) >= 3 else "",
        "cumulative_top3": pca.get("cumulative_top_k", ""),
        "var_explained_curve": gof.get("var_explained_curve", ""),
        "residual_median": gof.get("residual_median", ""),
        "residual_p95": gof.get("residual_p95", ""),
        "angle_deg": angle.get("angle_deg", ""),
        "axis_explained_in_pc": angle.get("axis_explained_in_pc", ""),
        "tangent_source": angle.get("tangent_source", ""),
        "centroid_source": per_subject.get("centroid_source", ""),
        "n_cluster_0": n_in_cluster[0] if len(n_in_cluster) >= 1 else "",
        "n_cluster_1": n_in_cluster[1] if len(n_in_cluster) >= 2 else "",
        "curve_n_iter": curve.get("n_iter", ""),
        "curve_converged": curve.get("converged", ""),
        "gof_pass": per_subject.get("gof_pass", ""),
        "error": per_subject.get("error", ""),
        "warning": per_subject.get("warning", ""),
        "warning_labels": per_subject.get("warning_labels", ""),
    }


def _format_value(val: Any) -> Any:
    if isinstance(val, float):
        if not np.isfinite(val):
            return ""
        return f"{val:.6f}"
    return val


def _write_cohort_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    fields = [
        "sid", "dataset", "subject", "n_events_eligible", "n_chan_union",
        "pc1_ratio", "pc2_ratio", "pc3_ratio", "cumulative_top3",
        "var_explained_curve", "residual_median", "residual_p95",
        "angle_deg", "axis_explained_in_pc", "tangent_source",
        "centroid_source", "n_cluster_0", "n_cluster_1",
        "curve_n_iter", "curve_converged",
        "gof_pass", "error", "warning", "warning_labels",
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: _format_value(r.get(k, "")) for k in fields})
    logger.info("Wrote %s (%d rows)", out_csv, len(rows))


def main() -> int:
    PER_SUBJECT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = _load_main_cohort(AUDIT_CSV)
    logger.info("Main cohort: %d subjects", len(cohort))

    cohort_rows: List[Dict[str, Any]] = []
    for sid, dataset, subject in cohort:
        try:
            result = _run_one(sid, dataset, subject)
        except Exception as exc:  # pragma: no cover
            logger.exception("Failure on %s: %s", sid, exc)
            result = {
                "sid": sid, "dataset": dataset, "subject": subject,
                "error": f"exception:{type(exc).__name__}:{exc}",
            }

        per_path = PER_SUBJECT_DIR / f"{sid}.json"
        with open(per_path, "w") as f:
            json.dump(result, f, cls=NumpyEncoder, indent=2)

        ge = result.get("gof", {}).get("var_explained_curve")
        ang = result.get("angle_to_kmeans_axis", {}).get("angle_deg")
        logger.info(
            "%-25s  n_eligible=%-7s  var_curve=%s  angle=%s%s",
            sid,
            result.get("n_events_eligible", "-"),
            f"{ge:.3f}" if isinstance(ge, (int, float)) and np.isfinite(ge) else "n/a",
            f"{ang:.1f}deg" if isinstance(ang, (int, float)) and np.isfinite(ang) else "n/a",
            "  PASS" if result.get("gof_pass") is True else "",
        )

        cohort_rows.append(_cohort_row(result))

    _write_cohort_csv(cohort_rows, COHORT_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
