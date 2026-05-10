"""Topic 1 × Topic 5 Bridge — Q1 + Q1b + Q3 implementation.

See docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md.
Q2 is deferred and NOT implemented here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# --- Locked constants (see spec §4) -----------------------------------------

ALPHA_WITHIN: float = 0.0167          # α/3 within-subject Bonferroni for 3 features
EFFECT_MIN: float = 0.10              # |ε²| or |r| or Cramér V threshold
P_NULL_BINOMIAL: float = 0.049        # cohort binomial null upper bound
WINDOWS_MIN: List[Tuple[float, float]] = [(-15.0, -1.0), (-30.0, -1.0), (-60.0, -1.0)]
PRIMARY_WINDOW: Tuple[float, float] = (-30.0, -1.0)

COHORT_GAMMA: List[str] = [
    "1073", "1096", "1146", "253", "548",
    "590", "635", "916", "922", "958",
]
SENTINEL_442: str = "442"            # Q1b binary-outlier
SENSITIVITY_BROAD_1084: str = "1084" # broad-band sensitivity


def load_topic5_subtype_labels(
    subject: str,
    band: str,
    results_root: Path,
) -> Dict[str, Any]:
    """Load per-seizure subtype label from topic5 PR-1 z-ER cluster JSON.

    Parameters
    ----------
    subject : str
        Numeric epilepsiae id without prefix, e.g. "442".
    band : str
        Either "gamma_ER" or "broad_ER".
    results_root : Path
        Project results root, typically Path("results") relative to repo root.

    Returns dict with keys:
      - seizure_id_to_subtype : Dict[str, int]   subtype labels (-1 = outlier)
      - n_subtypes : int                          per_band[band]["n_subtypes"]
      - status     : str                          "ok" / "insufficient_n" / ...
    """
    json_path = (
        results_root
        / "data_driven_soz"
        / "layer_a_ictal_er_rank"
        / "seizure_clusters"
        / "per_subject"
        / f"epilepsiae_{subject}__zer_binned.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(f"topic5 PR-1 JSON missing: {json_path}")
    with json_path.open() as fh:
        d = json.load(fh)
    band_d = d["per_band"][band]
    seizure_ids = list(band_d["seizure_ids_kept"])
    labels = list(band_d["subtype_label"])
    if len(seizure_ids) != len(labels):
        raise ValueError(
            f"length mismatch in {json_path}: "
            f"{len(seizure_ids)} ids vs {len(labels)} labels"
        )
    return {
        "seizure_id_to_subtype": dict(zip(seizure_ids, [int(x) for x in labels])),
        "n_subtypes": int(band_d["n_subtypes"]),
        "status": str(band_d["status"]),
    }


def load_seizure_onsets(
    subject: str,
    results_root: Path,
) -> Dict[str, float]:
    """Load per-seizure clinical onset epoch from epilepsiae inventory.

    Returns dict[seizure_id_str → epoch_seconds]. Prefers `clin_onset_epoch`;
    falls back to `eeg_onset_epoch` when clin is NaN. Seizures missing both
    are skipped (not returned).
    """
    inventory = pd.read_csv(
        results_root / "epilepsiae_seizure_inventory.csv",
        dtype={"subject": str, "seizure_id": str},
    )
    sub = inventory[inventory["subject"] == subject]
    if sub.empty:
        raise ValueError(f"subject {subject} not in epilepsiae_seizure_inventory.csv")
    out: Dict[str, float] = {}
    for _, row in sub.iterrows():
        sid = str(row["seizure_id"])
        clin = row.get("clin_onset_epoch")
        eeg = row.get("eeg_onset_epoch")
        if pd.notna(clin):
            out[sid] = float(clin)
        elif pd.notna(eeg):
            out[sid] = float(eeg)
        # else: skip (no usable onset)
    return out


def load_topic1_events_with_templates(
    subject: str,
    results_root: Path,
    artifact_root: Path,
    min_participating: int = 3,
) -> Dict[str, Any]:
    """Load topic1 per-event timestamps aligned with adaptive_cluster labels.

    Pipeline:
      1. load_subject_propagation_events on lagPat NPZ → event_abs_times, bools
      2. _valid_event_indices(bools, min_participating) → idx of valid events
      3. event_abs_times[idx] aligns with adaptive_cluster.labels (length match required)
      4. T0 = template_id with larger fraction across valid events

    Returns dict with:
      - event_abs_times    : (n_valid,) float64 epoch seconds
      - template_labels    : (n_valid,) int   ∈ {0, 1, ...} (raw cluster_id)
      - block_time_ranges  : List[(start_epoch, end_epoch)]
      - n_valid_events     : int
      - t0_template_id     : int  (the cluster_id assigned T0 by larger-fraction rule)
      - t1_template_id     : int  (the other one)
      - cluster_fractions  : Dict[int → float]
    """
    from src.interictal_propagation import (
        load_subject_propagation_events,
        _valid_event_indices,
    )

    # Resolve subject_dir (epilepsiae layout)
    legacy = artifact_root / subject / "all_recs"
    subject_dir = legacy if legacy.exists() else (artifact_root / subject)

    loaded = load_subject_propagation_events(subject_dir)
    bools = loaded["bools"]
    event_abs_times_all = np.asarray(loaded["event_abs_times"], dtype=float)
    valid_idx = _valid_event_indices(bools, min_participating=min_participating)

    # Load adaptive_cluster from topic1 per_subject JSON
    pj = results_root / "interictal_propagation" / "per_subject" / f"epilepsiae_{subject}.json"
    if not pj.exists():
        raise FileNotFoundError(f"topic1 per_subject JSON missing: {pj}")
    with pj.open() as fh:
        topic1 = json.load(fh)
    ac = topic1["adaptive_cluster"]
    if int(ac.get("stable_k", 0)) != 2:
        raise ValueError(f"subject {subject} stable_k != 2: {ac.get('stable_k')}")
    labels = np.asarray(ac["labels"], dtype=int)

    if labels.size != valid_idx.size:
        raise ValueError(
            f"labels size {labels.size} != valid_idx size {valid_idx.size} for {subject}"
        )

    valid_event_abs_times = event_abs_times_all[valid_idx]

    # T0/T1 freeze: T0 = larger-fraction cluster (ties → smaller cluster_id)
    cluster_fractions: Dict[int, float] = {}
    for c in ac["clusters"]:
        cluster_fractions[int(c["cluster_id"])] = float(c["fraction"])
    if len(cluster_fractions) != 2:
        raise ValueError(f"expected 2 clusters, got {len(cluster_fractions)}")
    sorted_clusters = sorted(
        cluster_fractions.items(),
        key=lambda kv: (-kv[1], kv[0]),  # larger fraction first; smaller id ties first
    )
    t0_id = sorted_clusters[0][0]
    t1_id = sorted_clusters[1][0]

    return {
        "event_abs_times": valid_event_abs_times,
        "template_labels": labels,
        "block_time_ranges": list(loaded["block_time_ranges"]),
        "n_valid_events": int(labels.size),
        "t0_template_id": int(t0_id),
        "t1_template_id": int(t1_id),
        "cluster_fractions": {int(k): float(v) for k, v in cluster_fractions.items()},
    }


def freeze_bridge_setup(
    cohort: Sequence[str],
    results_root: Path,
    artifact_root: Path,
    out_path: Path,
) -> Dict[str, Any]:
    """Freeze T0/T1 convention + audit-rerun marker per spec §4 / §6 caveat #3.

    Idempotent: running twice with same input produces byte-identical JSON.
    """
    # Find audit-rerun completion marker in log
    log_dir = results_root / "run_logs"
    marker = None
    for log_file in sorted(log_dir.glob("cohort_zer_audit_*.log")):
        text = log_file.read_text()
        for line in text.splitlines():
            if "[cohort] cohort_summary.csv" in line:
                marker = line.strip()
                break
        if marker:
            break
    if not marker:
        raise RuntimeError(
            "audit-rerun completion marker not found in any run_logs/cohort_zer_audit_*.log; "
            "did the audit-rerun finish?"
        )

    subjects: Dict[str, Any] = {}
    for sid in sorted(cohort):  # sorted = idempotent ordering
        ev = load_topic1_events_with_templates(
            subject=sid,
            results_root=results_root,
            artifact_root=artifact_root,
        )
        subjects[f"epilepsiae_{sid}"] = {
            "topic1_n_valid_events": ev["n_valid_events"],
            "topic1_template_fractions": {
                str(k): round(v, 12) for k, v in sorted(ev["cluster_fractions"].items())
            },
            "t0_template_id": ev["t0_template_id"],
            "t1_template_id": ev["t1_template_id"],
        }

    payload = {
        "schema_version": 1,
        "audit_rerun_marker_log_line": marker,
        "alpha_within": ALPHA_WITHIN,
        "effect_min": EFFECT_MIN,
        "p_null_binomial": P_NULL_BINOMIAL,
        "windows_min": [list(w) for w in WINDOWS_MIN],
        "subjects": subjects,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload
