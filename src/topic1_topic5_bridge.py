"""Topic 1 × Topic 5 Bridge — Q1 + Q1b + Q3 implementation.

See docs/superpowers/specs/2026-05-10-topic1-topic5-bridge-design.md.
Q2 is deferred and NOT implemented here.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

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


# --- Pre-ictal fingerprint (Task 6) ------------------------------------------

@dataclass
class FingerprintRow:
    """One pre-ictal window fingerprint for one seizure."""
    seizure_id: str
    n_events: int
    frac_T0: float          # NaN if n_events == 0
    switch_rate: float      # NaN if n_events <= 1
    last_template: Optional[int]  # None if n_events == 0
    dropped_reason: Optional[str]


def compute_pre_ictal_fingerprint(
    event_times: np.ndarray,
    event_template_ids: np.ndarray,
    seizure_clinical_onset: float,
    window_min_min: float,
    window_min_max: float,
    t0_template_id: int,
    seizure_id: str = "",
) -> Dict[str, Any]:
    """Compute fingerprint for one (seizure × window) cell.

    Window convention: t ∈ [onset + window_min_min * 60, onset + window_min_max * 60)
    (left-inclusive, right-exclusive).

    n_events == 0  → dropped_reason="no_events_in_window", all values NaN/None.
    n_events == 1  → switch_rate=NaN (no transition defined).
    """
    win_lo = seizure_clinical_onset + window_min_min * 60.0
    win_hi = seizure_clinical_onset + window_min_max * 60.0
    mask = (event_times >= win_lo) & (event_times < win_hi)
    in_times = event_times[mask]
    in_labels = event_template_ids[mask]
    n = int(in_times.size)

    if n == 0:
        return {
            "seizure_id": seizure_id,
            "n_events": 0,
            "frac_T0": float("nan"),
            "switch_rate": float("nan"),
            "last_template": None,
            "dropped_reason": "no_events_in_window",
        }

    order = np.argsort(in_times, kind="stable")
    sorted_labels = in_labels[order]
    frac_t0 = float((sorted_labels == t0_template_id).mean())
    if n >= 2:
        switch_rate = float((sorted_labels[1:] != sorted_labels[:-1]).mean())
    else:
        switch_rate = float("nan")
    last_template = int(sorted_labels[-1])

    return {
        "seizure_id": seizure_id,
        "n_events": n,
        "frac_T0": frac_t0,
        "switch_rate": switch_rate,
        "last_template": last_template,
        "dropped_reason": None,
    }


def build_subject_fingerprint_table(
    subject: str,
    band: str,
    results_root: Path,
    artifact_root: Path,
    windows_min: Sequence[Tuple[float, float]],
) -> pd.DataFrame:
    """For one subject and band, compute pre-ictal fingerprint for every
    seizure × window pair; return long-form DataFrame.
    """
    subtypes = load_topic5_subtype_labels(subject, band, results_root)
    onsets = load_seizure_onsets(subject, results_root)
    ev = load_topic1_events_with_templates(
        subject=subject,
        results_root=results_root,
        artifact_root=artifact_root,
    )
    rows: List[Dict[str, Any]] = []
    for sid, subtype in subtypes["seizure_id_to_subtype"].items():
        if sid not in onsets:
            # No usable onset → skip; record drop in audit later
            for win_lo, win_hi in windows_min:
                rows.append({
                    "subject": subject,
                    "band": band,
                    "window_min_min": win_lo,
                    "window_min_max": win_hi,
                    "seizure_id": sid,
                    "subtype_label": int(subtype),
                    "n_events": 0,
                    "frac_T0": float("nan"),
                    "switch_rate": float("nan"),
                    "last_template": None,
                    "dropped_reason": "no_onset",
                })
            continue
        for win_lo, win_hi in windows_min:
            fp = compute_pre_ictal_fingerprint(
                event_times=ev["event_abs_times"],
                event_template_ids=ev["template_labels"],
                seizure_clinical_onset=onsets[sid],
                window_min_min=win_lo,
                window_min_max=win_hi,
                t0_template_id=ev["t0_template_id"],
                seizure_id=sid,
            )
            rows.append({
                "subject": subject,
                "band": band,
                "window_min_min": win_lo,
                "window_min_max": win_hi,
                "seizure_id": sid,
                "subtype_label": int(subtype),
                **{k: fp[k] for k in ("n_events", "frac_T0", "switch_rate", "last_template", "dropped_reason")},
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-feature statistical helpers (Task 8)
# ---------------------------------------------------------------------------

def _mann_whitney_with_effect(
    a: np.ndarray, b: np.ndarray
) -> Tuple[float, float]:
    """Mann-Whitney U two-sided + rank-biserial r effect.

    Returns (p_two_sided, signed_rank_biserial_r).
    Drops NaN values before computation. Returns (1.0, 0.0) for empty / degenerate.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return 1.0, 0.0
    res = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
    n1, n2 = a.size, b.size
    # rank-biserial = 1 - 2U / (n1 * n2); sign indicates direction
    r = 1.0 - 2.0 * float(res.statistic) / (n1 * n2)
    return float(res.pvalue), float(r)


def _kruskal_wallis_with_effect(
    groups: Sequence[np.ndarray],
) -> Tuple[float, float]:
    """Kruskal-Wallis + ε² (epsilon-squared) effect.

    ε² = (H - k + 1) / (n - k); negative values floored at 0.
    Returns (1.0, 0.0) for empty / degenerate.
    """
    cleaned = [np.asarray(g, dtype=float)[np.isfinite(np.asarray(g, dtype=float))] for g in groups]
    cleaned = [g for g in cleaned if g.size > 0]
    if len(cleaned) < 2 or sum(g.size for g in cleaned) < 3:
        return 1.0, 0.0
    res = sp_stats.kruskal(*cleaned)
    h = float(res.statistic)
    k = len(cleaned)
    n = int(sum(g.size for g in cleaned))
    if n - k <= 0:
        return float(res.pvalue), 0.0
    eps2 = max(0.0, (h - k + 1) / (n - k))
    return float(res.pvalue), float(eps2)


def _fisher_or_chi2_with_cramer_v(
    contingency: np.ndarray,
) -> Tuple[float, float]:
    """For 2x2 contingency use Fisher exact two-sided; for >2 rows or columns
    use χ² with Yates correction off; effect = Cramér V.

    Returns (p, V). Returns (1.0, 0.0) on degenerate input.
    """
    table = np.asarray(contingency, dtype=int)
    n = int(table.sum())
    if n == 0 or table.ndim != 2 or min(table.shape) < 2:
        return 1.0, 0.0
    if table.shape == (2, 2):
        res = sp_stats.fisher_exact(table, alternative="two-sided")
        p = float(res.pvalue)
        chi2_stat, _, _, _ = sp_stats.chi2_contingency(table, correction=False)
    else:
        chi2_stat, p, _, _ = sp_stats.chi2_contingency(table, correction=False)
    r, c = table.shape
    denom = n * (min(r, c) - 1)
    v = math.sqrt(chi2_stat / denom) if denom > 0 else 0.0
    return float(p), float(v)


# ---------------------------------------------------------------------------
# Per-subject Q1 test — same-feature dual gate (Task 9)
# ---------------------------------------------------------------------------

def q1_per_subject_test(
    fp_df: pd.DataFrame,
    alpha_within: float = ALPHA_WITHIN,
    effect_min: float = EFFECT_MIN,
    eligibility_floor: int = 4,
) -> Dict[str, Any]:
    """Per-subject Q1 test under same-feature dual gate.

    Locked rule (spec §4.1):
        Per-subject Q1-positive iff
        ∃ feature f ∈ {frac_T0, switch_rate, last_template}:
            p_f < alpha_within  AND  |effect_f| > effect_min

    Cross-feature pickup is forbidden — feature A providing p
    while feature B provides effect must NOT produce a positive.

    Drops seizures with `dropped_reason` set or n_events == 0 from
    every feature's test.
    """
    # Drop dropped/zero-event rows for testing
    eff = fp_df.copy()
    if "dropped_reason" in eff.columns:
        eff = eff[eff["dropped_reason"].isna() | (eff["dropped_reason"] == "")]
    if "n_events" in eff.columns:
        eff = eff[eff["n_events"] > 0]
    n_eligible = int(len(eff))
    passes_floor = n_eligible >= eligibility_floor

    out: Dict[str, Any] = {
        "n_eligible_seizures": n_eligible,
        "passes_eligibility_floor": passes_floor,
        "per_feature": {},
        "subject_positive": False,
        "feature_winner": None,
        "feature_winner_p": None,
        "feature_winner_effect": None,
        "eligibility": "ok",
    }

    if not passes_floor:
        out["eligibility"] = "below_floor"
        return out

    subtypes = sorted(eff["subtype_label"].unique().tolist())
    if len(subtypes) < 2:
        out["eligibility"] = "single_subtype"
        return out

    # frac_T0 → MW (k=2) or KW (k≥3)
    groups_frac = [eff.loc[eff["subtype_label"] == s, "frac_T0"].to_numpy() for s in subtypes]
    if len(subtypes) == 2:
        p_frac, eff_frac = _mann_whitney_with_effect(groups_frac[0], groups_frac[1])
    else:
        p_frac, eff_frac = _kruskal_wallis_with_effect(groups_frac)
    pass_frac = (p_frac < alpha_within) and (abs(eff_frac) > effect_min)
    out["per_feature"]["frac_T0"] = {
        "p": p_frac, "effect": eff_frac, "passed_dual_gate": bool(pass_frac),
    }

    # switch_rate → same logic; additional isfinite filter for n_events=1 NaN cases
    groups_sw_raw = [eff.loc[eff["subtype_label"] == s, "switch_rate"].to_numpy() for s in subtypes]
    groups_sw = [g[np.isfinite(g)] for g in groups_sw_raw]
    if all(g.size > 0 for g in groups_sw):
        if len(subtypes) == 2:
            p_sw, eff_sw = _mann_whitney_with_effect(groups_sw[0], groups_sw[1])
        else:
            p_sw, eff_sw = _kruskal_wallis_with_effect(groups_sw)
    else:
        p_sw, eff_sw = 1.0, 0.0
    pass_sw = (p_sw < alpha_within) and (abs(eff_sw) > effect_min)
    out["per_feature"]["switch_rate"] = {
        "p": p_sw, "effect": eff_sw, "passed_dual_gate": bool(pass_sw),
    }

    # last_template → contingency (subtype × template value)
    sub_arr = eff["subtype_label"].to_numpy()
    lt_arr = eff["last_template"].to_numpy()
    valid_lt = ~pd.isna(lt_arr)
    sub_v = sub_arr[valid_lt]
    lt_v = lt_arr[valid_lt].astype(int)
    if sub_v.size > 0:
        templates_obs = sorted(np.unique(lt_v).tolist())
        if len(templates_obs) < 2 or len(np.unique(sub_v)) < 2:
            p_lt, eff_lt = 1.0, 0.0
        else:
            cont = np.zeros((len(subtypes), len(templates_obs)), dtype=int)
            for i, s in enumerate(subtypes):
                for j, t in enumerate(templates_obs):
                    cont[i, j] = int(((sub_v == s) & (lt_v == t)).sum())
            p_lt, eff_lt = _fisher_or_chi2_with_cramer_v(cont)
    else:
        p_lt, eff_lt = 1.0, 0.0
    pass_lt = (p_lt < alpha_within) and (abs(eff_lt) > effect_min)
    out["per_feature"]["last_template"] = {
        "p": p_lt, "effect": eff_lt, "passed_dual_gate": bool(pass_lt),
    }

    # 3-feature OR: subject positive iff ∃ feature that passes its own dual gate
    passing = [
        (name, info["p"], abs(info["effect"]))
        for name, info in out["per_feature"].items()
        if info["passed_dual_gate"]
    ]
    if passing:
        # winner = highest |effect| among passers
        passing.sort(key=lambda x: -x[2])
        out["subject_positive"] = True
        out["feature_winner"] = passing[0][0]
        out["feature_winner_p"] = passing[0][1]
        out["feature_winner_effect"] = passing[0][2]
    return out
