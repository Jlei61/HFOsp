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
    dropped_subjects: Dict[str, str] = {}
    for sid in sorted(cohort):  # sorted = idempotent ordering
        try:
            ev = load_topic1_events_with_templates(
                subject=sid,
                results_root=results_root,
                artifact_root=artifact_root,
            )
        except Exception as exc:
            reason = repr(exc)
            print(f"[setup] WARNING: skipping epilepsiae_{sid} — {reason}", flush=True)
            dropped_subjects[f"epilepsiae_{sid}"] = reason
            continue
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
        "dropped_subjects": dropped_subjects,
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


# ---------------------------------------------------------------------------
# Cohort aggregation + 3-state verdict (Task 11)
# ---------------------------------------------------------------------------

def q1_cohort_per_window(
    per_subject_results: Dict[str, Any],
    p_null: float = P_NULL_BINOMIAL,
    pass_alpha: float = 0.05,
) -> Dict[str, Any]:
    """Cohort PER-WINDOW judgement for one window.

    `per_subject_results` is dict[subject_key → q1_test_dict] for one window.
    Subjects with passes_eligibility_floor=False are excluded from the denominator.
    """
    eligible = {k: v for k, v in per_subject_results.items() if v.get("passes_eligibility_floor", False)}
    denom = len(eligible)
    n_positive = sum(1 for v in eligible.values() if v.get("subject_positive", False))
    if denom == 0:
        binomial_p = 1.0
    else:
        # one-sided binomial: P(X ≥ n_positive | p_null)
        binomial_p = float(sp_stats.binomtest(n_positive, denom, p_null, alternative="greater").pvalue)
    return {
        "denom": int(denom),
        "n_positive": int(n_positive),
        "binomial_p": binomial_p,
        "per_window_pass": bool(binomial_p < pass_alpha and n_positive > 0),
    }


def cohort_overall_judgement(per_window: Dict[str, Dict[str, Any]]) -> str:
    """3-state cohort verdict per spec §4.4.

    PASS         = ≥ 2/3 windows per_window_pass
    NULL-locked  = 0/3 windows pass AND all windows count ≤ 1/denom
    otherwise    = INDETERMINATE
    """
    n_windows = len(per_window)
    n_passed = sum(1 for v in per_window.values() if v.get("per_window_pass", False))
    if n_passed >= max(2, math.ceil(2 * n_windows / 3)):
        return "COHORT-EXPLORATORY-PASS"
    counts = [v.get("n_positive", 0) for v in per_window.values()]
    if n_passed == 0 and all(c <= 1 for c in counts):
        return "NULL-locked"
    return "INDETERMINATE"


def aggregate_cohort_summary(
    per_subject_dir: Path,
    band: str,
    windows_min: Sequence[Tuple[float, float]],
    cohort: Sequence[str],
    out_path: Path,
) -> Dict[str, Any]:
    """Read per-subject JSONs, aggregate per-window cohort, write cohort_summary.json."""
    per_window: Dict[str, Dict[str, Any]] = {}
    per_window_subject_results: Dict[str, Dict[str, Any]] = {}
    for win_lo, win_hi in windows_min:
        wkey = f"[{win_lo},{win_hi}]"
        ps_results: Dict[str, Any] = {}
        for sid in cohort:
            f = per_subject_dir / f"epilepsiae_{sid}__bridge.json"
            if not f.exists():
                continue
            with f.open() as fh:
                d = json.load(fh)
            band_block = d.get("bands", {}).get(band, {}).get("windows", {})
            if not band_block and "windows" in d:
                # single-band shortcut
                band_block = d["windows"]
            if wkey in band_block:
                w = band_block[wkey]
                ps_results[f"epilepsiae_{sid}"] = {
                    "passes_eligibility_floor": w.get("passes_eligibility_floor", False),
                    "subject_positive": w.get("subject_positive", False),
                    "feature_winner": w.get("feature_winner"),
                    "feature_winner_p": w.get("feature_winner_p"),
                    "feature_winner_effect": w.get("feature_winner_effect"),
                    "n_eligible_seizures": w.get("n_eligible_seizures"),
                }
        per_window_subject_results[wkey] = ps_results
        per_window[wkey] = q1_cohort_per_window(ps_results)
    verdict = cohort_overall_judgement(per_window)
    payload = {
        "schema_version": 1,
        "band": band,
        "cohort": list(cohort),
        "windows": per_window,
        "per_window_subject_results": per_window_subject_results,
        "cohort_judgement": verdict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return payload


# ---------------------------------------------------------------------------
# Q1b 442 binary-outlier sentinel (Task 12)
# ---------------------------------------------------------------------------

def q1b_sentinel_442(
    results_root: Path,
    artifact_root: Path,
    windows_min: Sequence[Tuple[float, float]],
    out_path: Path,
    band: str = "gamma_ER",
) -> Dict[str, Any]:
    """442 sz=9 (subtype_label=-1) vs other 16 seizures fingerprint comparison.

    Q1b is descriptive case-study, not cohort claim (spec §2.2).
    """
    payload: Dict[str, Any] = {
        "subject": "epilepsiae_442",
        "band": band,
        "windows": {},
    }
    for win_lo, win_hi in windows_min:
        df = build_subject_fingerprint_table(
            subject="442",
            band=band,
            results_root=results_root,
            artifact_root=artifact_root,
            windows_min=[(win_lo, win_hi)],
        )
        # Drop dropped/zero-event rows
        eff = df[(df["dropped_reason"].isna()) & (df["n_events"] > 0)]
        outlier = eff[eff["subtype_label"] == -1]
        main = eff[eff["subtype_label"] == 0]
        wkey = f"[{win_lo},{win_hi}]"
        block: Dict[str, Any] = {
            "n_outlier": int(len(outlier)),
            "n_main": int(len(main)),
        }
        if len(outlier) >= 1 and len(main) >= 2:
            p_f, eff_f = _mann_whitney_with_effect(
                outlier["frac_T0"].to_numpy(), main["frac_T0"].to_numpy()
            )
            block["frac_T0"] = {"p": p_f, "effect": eff_f}
            sw_o = outlier["switch_rate"].to_numpy()
            sw_m = main["switch_rate"].to_numpy()
            sw_o = sw_o[np.isfinite(sw_o)]
            sw_m = sw_m[np.isfinite(sw_m)]
            if sw_o.size and sw_m.size:
                p_s, eff_s = _mann_whitney_with_effect(sw_o, sw_m)
                block["switch_rate"] = {"p": p_s, "effect": eff_s}
            # last_template Fisher 2x2
            lt_o = outlier["last_template"].dropna().astype(int).tolist()
            lt_m = main["last_template"].dropna().astype(int).tolist()
            templates = sorted(set(lt_o + lt_m))
            if len(templates) == 2:
                cont = np.array([
                    [lt_o.count(templates[0]), lt_o.count(templates[1])],
                    [lt_m.count(templates[0]), lt_m.count(templates[1])],
                ], dtype=int)
                p_l, eff_l = _fisher_or_chi2_with_cramer_v(cont)
                block["last_template"] = {"p": p_l, "effect": eff_l, "contingency": cont.tolist()}
        else:
            block["frac_T0"] = {"p": 1.0, "effect": 0.0, "skipped_reason": "insufficient_n"}
        payload["windows"][wkey] = block
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return payload


# ---------------------------------------------------------------------------
# Q3 stratifier — descriptive 4-cell table (swap × silhouette)
# ---------------------------------------------------------------------------

# Stratifier maps locked from spec §3.4
SWAP_CLASS_MAP_GAMMA: Dict[str, str] = {
    "1073": "strict",
    "1146": "strict",
    "635": "strict",
    "958": "strict",
    "548": "candidate",
    "1096": "none",
    "253": "none",
    "590": "none",
    "916": "none",
    "922": "none",
}

# Approximate silhouette_median from cluster_geometry; >0.5 = high
SILHOUETTE_MAP_GAMMA: Dict[str, float] = {
    "1073": 0.549, "1146": 0.551, "635": 0.402, "958": 0.413,
    "548": 0.627, "1096": 0.252, "253": 0.182, "590": 0.584,
    "916": 0.647, "922": 0.275,
}


def q3_stratifier_table(
    cohort_summary_path: Path,
    band: str,
    primary_window: Tuple[float, float] = PRIMARY_WINDOW,
    swap_class_map: Dict[str, str] = SWAP_CLASS_MAP_GAMMA,
    silhouette_map: Dict[str, float] = SILHOUETTE_MAP_GAMMA,
    silhouette_high_min: float = 0.5,
) -> pd.DataFrame:
    """4-cell descriptive stratifier (swap × silhouette)."""
    with cohort_summary_path.open() as fh:
        d = json.load(fh)
    wkey = f"[{primary_window[0]},{primary_window[1]}]"
    subj_results = d["per_window_subject_results"][wkey]
    rows = []
    for subj_key, info in subj_results.items():
        sid = subj_key.replace("epilepsiae_", "")
        if sid not in swap_class_map:
            continue
        swap = swap_class_map[sid]
        sil = silhouette_map.get(sid, float("nan"))
        rows.append({
            "subject": sid,
            "swap_class": "real" if swap in ("strict", "candidate") else "none",
            "silhouette_class": "high" if sil > silhouette_high_min else "low",
            "subject_positive": info.get("subject_positive", False),
            "effect_winner": info.get("feature_winner_effect"),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figure helpers (Task 14)
# ---------------------------------------------------------------------------

def _morandi_palette() -> List[str]:
    """Project-wide Morandi palette; falls back to defaults if src.plot_style missing."""
    try:
        from src.plot_style import MORANDI_PALETTE
        return list(MORANDI_PALETTE)
    except Exception:
        return ["#8AA6A3", "#C4A484", "#A9908A", "#7C8B96", "#D4B996"]


def figure_q1_cohort_count_x_window(
    cohort_summary_path: Path, out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    with cohort_summary_path.open() as fh:
        d = json.load(fh)
    windows = list(d["windows"].keys())
    counts = [d["windows"][w]["n_positive"] for w in windows]
    denoms = [d["windows"][w]["denom"] for w in windows]
    pvals = [d["windows"][w]["binomial_p"] for w in windows]
    fig, ax = plt.subplots(figsize=(6, 3.5), dpi=150)
    pal = _morandi_palette()
    ax.bar(range(len(windows)), counts, color=pal[0])
    for i, (n, dn, p) in enumerate(zip(counts, denoms, pvals)):
        ax.text(i, n + 0.1, f"{n}/{dn}\np={p:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels(windows)
    ax.set_ylabel("n subject-positive")
    ax.set_title(f"Q1 cohort count per window — verdict: {d['cohort_judgement']}")
    ax.set_ylim(0, max(denoms) + 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def figure_q1_effect_distribution(
    cohort_summary_path: Path, out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    with cohort_summary_path.open() as fh:
        d = json.load(fh)
    pal = _morandi_palette()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), dpi=150, sharey=True)
    for ax, (wkey, ps) in zip(axes, d["per_window_subject_results"].items()):
        effects = [v.get("feature_winner_effect") or 0.0 for v in ps.values()]
        names = [k.replace("epilepsiae_", "") for k in ps.keys()]
        ax.bar(range(len(names)), effects, color=pal[1])
        ax.axhline(EFFECT_MIN, ls="--", color="grey", label=f"effect_min={EFFECT_MIN}")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, fontsize=8)
        ax.set_title(wkey)
    axes[0].set_ylabel("|effect_winner|")
    fig.suptitle("Q1 per-subject feature-winner effect by window")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def figure_q1_per_subject_strip(
    per_subject_dir: Path,
    cohort: Sequence[str],
    band: str,
    primary_window: Tuple[float, float],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    pal = _morandi_palette()
    fig, ax = plt.subplots(figsize=(11, 5), dpi=150)
    wkey = f"[{primary_window[0]},{primary_window[1]}]"
    plot_subjects = list(cohort) + ["442", "1084"]
    for i, sid in enumerate(plot_subjects):
        f = per_subject_dir / f"epilepsiae_{sid}__bridge.json"
        if not f.exists():
            continue
        with f.open() as fh:
            d = json.load(fh)
        bw = d.get("bands", {}).get(band, {}).get("windows", {}).get(wkey)
        if bw is None and "windows" in d:
            bw = d["windows"].get(wkey)
        if not bw:
            continue
        for row in bw["fingerprint"]:
            color = pal[2] if (row.get("subtype_label") in (-1, 0)) else pal[3]
            y = i + (0.1 if row.get("subtype_label") == -1 else 0.0)
            x = row.get("frac_T0")
            if x is not None and not (isinstance(x, float) and math.isnan(x)):
                ax.plot(x, y, "o", color=color, alpha=0.7)
    ax.set_yticks(range(len(plot_subjects)))
    ax.set_yticklabels(plot_subjects)
    ax.set_xlabel("frac_T0 (pre-ictal)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(f"Q1 per-subject pre-ictal frac_T0 strip — window {wkey}")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def figure_q1b_442_sentinel(
    sentinel_path: Path, out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    with sentinel_path.open() as fh:
        d = json.load(fh)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), dpi=150)
    for ax, (wkey, w) in zip(axes, d["windows"].items()):
        ax.bar([0, 1], [w["frac_T0"].get("effect", 0.0), w.get("switch_rate", {}).get("effect", 0.0)],
               color=_morandi_palette()[:2])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["frac_T0", "switch_rate"])
        ax.set_title(f"{wkey}\n p_frac={w['frac_T0'].get('p', 1.0):.3f}")
        ax.axhline(0, color="grey", lw=0.5)
    fig.suptitle("442 sz=9 (outlier) vs main 16 sz — Q1b")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def figure_q3_stratifier(
    cohort_summary_path: Path, out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    df = q3_stratifier_table(cohort_summary_path, band="gamma_ER")
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    pal = _morandi_palette()
    color_map = {"real": pal[0], "none": pal[1]}
    for swap, sub in df.groupby("swap_class"):
        ax.scatter(sub["silhouette_class"].map({"low": 0, "high": 1}),
                   sub["effect_winner"].fillna(0.0),
                   color=color_map[swap], label=f"swap={swap}", s=60, alpha=0.75)
        for _, row in sub.iterrows():
            ax.annotate(row["subject"], (
                {"low": 0, "high": 1}[row["silhouette_class"]],
                row["effect_winner"] or 0.0,
            ), fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["silhouette ≤ 0.5", "silhouette > 0.5"])
    ax.set_ylabel("|effect_winner| at primary window")
    ax.axhline(EFFECT_MIN, ls="--", color="grey")
    ax.legend()
    ax.set_title("Q3 stratifier: swap × silhouette (descriptive)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-subject runner — full cohort batch (Task 10)
# ---------------------------------------------------------------------------

def run_per_subject(
    cohort: Sequence[str],
    bands: Sequence[str],
    windows_min: Sequence[Tuple[float, float]],
    results_root: Path,
    artifact_root: Path,
    out_dir: Path,
) -> None:
    """For each subject in cohort × each band, build fingerprint table for all
    windows + run per-subject Q1 test, write one JSON per subject."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in cohort:
        out_path = out_dir / f"epilepsiae_{sid}__bridge.json"
        try:
            per_subject_payload: Dict[str, Any] = {
                "subject": f"epilepsiae_{sid}",
                "bands": {},
            }
            for band in bands:
                band_payload: Dict[str, Any] = {"windows": {}}
                df_all = build_subject_fingerprint_table(
                    subject=sid,
                    band=band,
                    results_root=results_root,
                    artifact_root=artifact_root,
                    windows_min=windows_min,
                )
                for win_lo, win_hi in windows_min:
                    key = f"[{win_lo},{win_hi}]"
                    df_w = df_all[
                        (df_all["window_min_min"] == win_lo) & (df_all["window_min_max"] == win_hi)
                    ].copy()
                    # Handle last_template: may contain int or NaN; serialize None for missing
                    df_w = df_w.copy()
                    df_w["last_template_obj"] = df_w["last_template"].apply(
                        lambda x: None if pd.isna(x) else int(x)
                    )
                    df_out = df_w.drop(columns=["last_template"]).rename(
                        columns={"last_template_obj": "last_template"}
                    )
                    fingerprint_records = df_out.to_dict(orient="records")
                    test = q1_per_subject_test(df_w)
                    band_payload["windows"][key] = {
                        "fingerprint": fingerprint_records,
                        "q1_test": test,
                        "subject_positive": test["subject_positive"],
                        "feature_winner": test["feature_winner"],
                        "feature_winner_p": test["feature_winner_p"],
                        "feature_winner_effect": test["feature_winner_effect"],
                        "n_eligible_seizures": test["n_eligible_seizures"],
                        "passes_eligibility_floor": test["passes_eligibility_floor"],
                    }
                per_subject_payload["bands"][band] = band_payload
            # For backward-compat single-band convenience
            if len(bands) == 1:
                per_subject_payload["windows"] = per_subject_payload["bands"][bands[0]]["windows"]
        except Exception as exc:
            per_subject_payload = {
                "subject": f"epilepsiae_{sid}",
                "error": str(exc),
                "status": "failed",
            }
            print(f"  [WARN] epilepsiae_{sid} failed: {exc}")
        with out_path.open("w") as fh:
            json.dump(per_subject_payload, fh, indent=2, default=str)


# ============================================================================
# Q1' (PIVOT 2026-05-10) — Channel-rank correspondence with swap-subset gating
# ============================================================================


def load_atlas_seizure_channel_onsets(
    subject: str,
    band: str,
    results_root: Path,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Load per-seizure channel onset times (t_onset_sec) from atlas v2_3 timing JSON.

    Returns dict[seizure_id_str → dict[ch_name → t_onset_sec | None]].
    None values mean that channel did not reach onset criterion in that seizure.
    """
    p = (
        results_root
        / "data_driven_soz"
        / "layer_a_ictal_er_rank"
        / "per_subject"
        / f"epilepsiae_{subject}.json"
    )
    if not p.exists():
        raise FileNotFoundError(f"atlas v2_3 JSON missing: {p}")
    with p.open() as fh:
        d = json.load(fh)
    if d.get("schema_version") != "pr_t3_1_layer_a_v2_3_timing":
        raise ValueError(f"unexpected schema_version: {d.get('schema_version')} in {p}")
    sr = d["per_er"][band]["seizure_records"]
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for rec in sr:
        sid = str(rec["seizure_id"])
        ch_onsets = rec.get("channel_onsets") or {}
        out[sid] = {
            ch: (None if v is None or v.get("t_onset_sec") is None else float(v["t_onset_sec"]))
            for ch, v in ch_onsets.items()
        }
    return out


def load_swap_channel_subset(
    subject: str,
    results_root: Path,
) -> Dict[str, Any]:
    """Derive swap-channel endpoint set from rank_displacement §8 swap_sweep.

    Endpoint = top-decision_k ∪ bottom-decision_k channels (by rank_a_dense_full),
    restricted to joint_valid. Strict tier per §8.7 is the only contract for
    channel-level downstream consumption; candidate is reported but flagged.
    """
    p = (
        results_root
        / "interictal_propagation"
        / "rank_displacement"
        / "per_subject"
        / f"epilepsiae_{subject}.json"
    )
    if not p.exists():
        raise FileNotFoundError(f"rank_displacement JSON missing: {p}")
    with p.open() as fh:
        d = json.load(fh)
    pairs = d.get("pairs", [])
    if not pairs:
        return {"swap_class": "none", "decision_k": None, "endpoint_channels": [], "channel_names": [], "exit_reason": "no_pair"}
    pair = pairs[0]
    chs = list(pair.get("channel_names", []))
    joint_valid = list(pair.get("joint_valid", []))
    rank_a = list(pair.get("rank_a_dense_full", []))
    sweep = pair.get("swap_sweep", {}) or {}
    swap_class = str(sweep.get("swap_class", "none"))
    decision_k = sweep.get("decision_k")
    if decision_k is None or not chs or not rank_a:
        return {
            "swap_class": swap_class,
            "decision_k": decision_k,
            "endpoint_channels": [],
            "channel_names": chs,
            "exit_reason": "missing_swap_sweep",
        }
    decision_k = int(decision_k)
    valid_idx = [i for i, jv in enumerate(joint_valid) if jv]
    valid_chs_with_rank = [(i, rank_a[i]) for i in valid_idx if rank_a[i] is not None]
    valid_chs_with_rank.sort(key=lambda kv: kv[1])
    if len(valid_chs_with_rank) < 2 * decision_k:
        # Endpoint set may overlap (e.g., 6 ch with k=3 → top 3 + bottom 3 = whole set)
        bottom = valid_chs_with_rank[:decision_k]
        top = valid_chs_with_rank[-decision_k:]
    else:
        bottom = valid_chs_with_rank[:decision_k]
        top = valid_chs_with_rank[-decision_k:]
    endpoint_idx = sorted(set([i for i, _ in bottom] + [i for i, _ in top]))
    endpoint_chs = [chs[i] for i in endpoint_idx]
    return {
        "swap_class": swap_class,
        "decision_k": decision_k,
        "endpoint_channels": endpoint_chs,
        "channel_names": chs,
        "joint_valid_count": len(valid_idx),
        "p_fw": float(sweep.get("p_fw", float("nan"))),
        "swap_score": float(sweep.get("decision_swap_score", float("nan"))),
    }


def compute_seizure_template_alignment(
    seizure_onsets: Dict[str, Optional[float]],
    t0_rank: Dict[str, int],
    t1_rank: Dict[str, int],
    swap_subset: Sequence[str],
    channel_names_topic1: Sequence[str],
    channel_names_atlas: Sequence[str],
    tau_min: float = 0.10,
    min_channels: int = 3,
) -> Dict[str, Any]:
    """Per-seizure ρ_a, ρ_b, assignment ∈ {T0, T1, tie, insufficient_n}.

    Compares seizure channel-onset rank against T0/T1 template ranks within
    the swap-channel subset, restricted to the channel intersection of
    topic1 lagPat ∩ atlas ∩ swap_subset and channels with valid (non-None)
    seizure onset.
    """
    intersect = (
        set(swap_subset)
        & set(channel_names_topic1)
        & set(channel_names_atlas)
        & set(t0_rank.keys())
        & set(t1_rank.keys())
    )
    valid_chs = [
        ch for ch in intersect
        if ch in seizure_onsets and seizure_onsets[ch] is not None
    ]
    n = len(valid_chs)
    if n < min_channels:
        return {
            "assignment": "insufficient_n",
            "rho_a": float("nan"),
            "rho_b": float("nan"),
            "n_swap_channels_used": n,
            "channels_used": valid_chs,
        }
    onset_vec = np.array([seizure_onsets[ch] for ch in valid_chs], dtype=float)
    t0_vec = np.array([t0_rank[ch] for ch in valid_chs], dtype=float)
    t1_vec = np.array([t1_rank[ch] for ch in valid_chs], dtype=float)
    seizure_rank_vec = np.argsort(np.argsort(onset_vec))  # rank 0..n-1
    rho_a = sp_stats.spearmanr(seizure_rank_vec, t0_vec).statistic
    rho_b = sp_stats.spearmanr(seizure_rank_vec, t1_vec).statistic
    if not np.isfinite(rho_a):
        rho_a = 0.0
    if not np.isfinite(rho_b):
        rho_b = 0.0
    diff = float(rho_a - rho_b)
    if diff > tau_min:
        assignment = "T0"
    elif -diff > tau_min:
        assignment = "T1"
    else:
        assignment = "tie"
    return {
        "assignment": assignment,
        "rho_a": float(rho_a),
        "rho_b": float(rho_b),
        "n_swap_channels_used": n,
        "channels_used": valid_chs,
    }


def load_template_ranks_with_t0t1(
    subject: str,
    results_root: Path,
    artifact_root: Path,
) -> Dict[str, Any]:
    """Load adaptive_cluster.template_rank vectors for the 2 clusters,
    map to T0/T1 by fraction-larger rule (same convention as phase-1 freeze).

    Returns: channel_names, t0_template_id, t1_template_id, t0_rank (dict ch → rank), t1_rank.
    """
    p = results_root / "interictal_propagation" / "per_subject" / f"epilepsiae_{subject}.json"
    if not p.exists():
        raise FileNotFoundError(f"topic1 per_subject JSON missing: {p}")
    with p.open() as fh:
        d = json.load(fh)
    chs = list(d.get("channel_names", []))
    ac = d["adaptive_cluster"]
    if int(ac.get("stable_k", 0)) != 2:
        raise ValueError(f"subject {subject} stable_k != 2")
    cluster_fracs: Dict[int, float] = {int(c["cluster_id"]): float(c["fraction"]) for c in ac["clusters"]}
    cluster_ranks: Dict[int, List[int]] = {int(c["cluster_id"]): [int(r) for r in c["template_rank"]] for c in ac["clusters"]}
    sorted_clusters = sorted(cluster_fracs.items(), key=lambda kv: (-kv[1], kv[0]))
    t0_id = sorted_clusters[0][0]
    t1_id = sorted_clusters[1][0]
    if len(cluster_ranks[t0_id]) != len(chs) or len(cluster_ranks[t1_id]) != len(chs):
        raise ValueError(f"template_rank length != channel_names length for {subject}")
    return {
        "channel_names": chs,
        "t0_template_id": t0_id,
        "t1_template_id": t1_id,
        "t0_rank": dict(zip(chs, cluster_ranks[t0_id])),
        "t1_rank": dict(zip(chs, cluster_ranks[t1_id])),
    }


# ---------------------------------------------------------------------------
# Q1' per-subject contingency test (Task 3)
# ---------------------------------------------------------------------------

def q1prime_per_subject_test(
    seizure_alignments: Sequence[Dict[str, Any]],
    subtype_labels: Dict[str, int],
    p_max: float = 0.05,
    cramer_v_min: float = 0.30,
) -> Dict[str, Any]:
    """Per-subject Q1' test: contingency (assignment ∈ {T0,T1} × subtype) →
    Fisher exact (2x2) or χ² (>2x2) + Cramér V + AMI on full assignment list.
    """
    try:
        from sklearn.metrics import adjusted_mutual_info_score
    except Exception as e:
        raise RuntimeError("sklearn required for AMI") from e

    # Filter
    eligible = []
    n_tie = 0
    n_insuf = 0
    for s in seizure_alignments:
        a = s["assignment"]
        sid = s["seizure_id"]
        if sid not in subtype_labels:
            continue
        if a == "tie":
            n_tie += 1
            continue
        if a == "insufficient_n":
            n_insuf += 1
            continue
        eligible.append((sid, a, int(subtype_labels[sid])))

    n_eligible = len(eligible)
    out: Dict[str, Any] = {
        "n_eligible": n_eligible,
        "n_dropped_tie": n_tie,
        "n_dropped_insufficient": n_insuf,
        "p": 1.0,
        "cramer_v": 0.0,
        "ami": 0.0,
        "contingency": [],
        "q1prime_positive": False,
    }
    if n_eligible < 4:
        out["eligibility"] = "below_floor"
        return out

    assignments = [a for _, a, _ in eligible]
    subtypes = [s for _, _, s in eligible]
    a_levels = sorted(set(assignments))
    s_levels = sorted(set(subtypes))
    if len(a_levels) < 2 or len(s_levels) < 2:
        out["eligibility"] = "single_axis"
        return out

    cont = np.zeros((len(a_levels), len(s_levels)), dtype=int)
    for i, a in enumerate(a_levels):
        for j, sv in enumerate(s_levels):
            cont[i, j] = sum(1 for k in range(n_eligible) if assignments[k] == a and subtypes[k] == sv)
    out["contingency"] = cont.tolist()
    out["a_levels"] = a_levels
    out["s_levels"] = s_levels

    p, v = _fisher_or_chi2_with_cramer_v(cont)
    out["p"] = float(p)
    out["cramer_v"] = float(v)

    # AMI uses full per-seizure pairing (label encoded as int)
    a_int = [a_levels.index(a) for a in assignments]
    s_int = [s_levels.index(sv) for sv in subtypes]
    out["ami"] = float(adjusted_mutual_info_score(s_int, a_int))

    out["q1prime_positive"] = bool(p < p_max and v > cramer_v_min)
    out["eligibility"] = "ok"
    return out


# ---------------------------------------------------------------------------
# Q1' per-subject runner (Task 4)
# ---------------------------------------------------------------------------

def run_q1prime_per_subject(
    cohort: Sequence[str],
    band: str,
    results_root: Path,
    artifact_root: Path,
    out_dir: Path,
    tau_min: float = 0.10,
) -> None:
    """For each subject in cohort, run the full Q1' pipeline + write JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for sid in cohort:
        try:
            swap = load_swap_channel_subset(sid, results_root)
            tmpl = load_template_ranks_with_t0t1(sid, results_root, artifact_root)
            atlas = load_atlas_seizure_channel_onsets(sid, band, results_root)
            subtypes = load_topic5_subtype_labels(sid, band, results_root)
        except Exception as exc:
            payload = {
                "subject": f"epilepsiae_{sid}",
                "status": "failed",
                "error": repr(exc),
            }
            with (out_dir / f"epilepsiae_{sid}__q1prime.json").open("w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"[WARN] q1prime epilepsiae_{sid} failed: {exc}")
            continue

        per_seizure: List[Dict[str, Any]] = []
        for seizure_id, sz_onsets in atlas.items():
            align = compute_seizure_template_alignment(
                seizure_onsets=sz_onsets,
                t0_rank=tmpl["t0_rank"],
                t1_rank=tmpl["t1_rank"],
                swap_subset=swap["endpoint_channels"],
                channel_names_topic1=tmpl["channel_names"],
                channel_names_atlas=list(sz_onsets.keys()),
                tau_min=tau_min,
            )
            per_seizure.append({
                "seizure_id": seizure_id,
                **align,
                "subtype_label": subtypes["seizure_id_to_subtype"].get(seizure_id),
            })
        st_map = {sid_: int(lab) for sid_, lab in subtypes["seizure_id_to_subtype"].items() if lab is not None}
        test = q1prime_per_subject_test(per_seizure, st_map)
        payload = {
            "subject": f"epilepsiae_{sid}",
            "band": band,
            "swap_class": swap.get("swap_class"),
            "decision_k": swap.get("decision_k"),
            "swap_endpoint_channels": swap.get("endpoint_channels"),
            "n_swap_endpoint_channels": len(swap.get("endpoint_channels", [])),
            "t0_template_id": tmpl["t0_template_id"],
            "t1_template_id": tmpl["t1_template_id"],
            "topic5_n_subtypes": subtypes["n_subtypes"],
            "n_seizures_atlas": len(atlas),
            "per_seizure": per_seizure,
            "test": test,
        }
        out_path = out_dir / f"epilepsiae_{sid}__q1prime.json"
        with out_path.open("w") as fh:
            json.dump(payload, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Q1' cohort case-series summary + 3-state verdict (Task 5)
# ---------------------------------------------------------------------------

def q1prime_cohort_summary(
    per_subject_results: Dict[str, Any],
    strict_only: bool = True,
) -> Dict[str, Any]:
    """Q1' case-series cohort verdict per spec §10.4.

    CASE-SERIES-PASS = ≥3/4 strict subjects q1prime_positive AND median Cramér V > 0.30
    NULL-locked      = 0/4 strict positive AND median V ≤ 0.10 AND median AMI ≤ 0.05
    INDETERMINATE    = otherwise
    """
    strict_subjects = {
        k: v for k, v in per_subject_results.items()
        if v.get("swap_class") == "strict" and "test" in v
    }
    n_strict = len(strict_subjects)
    pos = [k for k, v in strict_subjects.items() if v["test"].get("q1prime_positive", False)]
    cv_list = [float(v["test"].get("cramer_v", 0.0)) for v in strict_subjects.values()]
    ami_list = [float(v["test"].get("ami", 0.0)) for v in strict_subjects.values()]
    median_cv = float(np.median(cv_list)) if cv_list else 0.0
    median_ami = float(np.median(ami_list)) if ami_list else 0.0
    n_pos = len(pos)

    if n_strict >= 1 and n_pos >= max(3, math.ceil(0.75 * n_strict)) and median_cv > 0.30:
        verdict = "CASE-SERIES-PASS"
    elif n_pos == 0 and median_cv <= 0.10 and median_ami <= 0.05:
        verdict = "NULL-locked"
    else:
        verdict = "INDETERMINATE"

    return {
        "cohort_judgement": verdict,
        "n_strict_total": n_strict,
        "n_strict_positive": n_pos,
        "strict_positive_subjects": pos,
        "median_cramer_v_strict": median_cv,
        "median_ami_strict": median_ami,
        "candidate_sentinel": {
            k: v.get("test", {})
            for k, v in per_subject_results.items()
            if v.get("swap_class") == "candidate"
        },
        "inadmissible_sentinels": {
            k: v.get("test", {})
            for k, v in per_subject_results.items()
            if v.get("swap_class") == "none"
        },
    }


def aggregate_q1prime_cohort(
    per_subject_dir: Path,
    cohort: Sequence[str],
    out_path: Path,
) -> Dict[str, Any]:
    """Read per-subject q1prime JSONs, aggregate to cohort summary, write JSON."""
    per_subject: Dict[str, Any] = {}
    for sid in cohort:
        f = per_subject_dir / f"epilepsiae_{sid}__q1prime.json"
        if not f.exists():
            continue
        with f.open() as fh:
            d = json.load(fh)
        if d.get("status") == "failed":
            continue
        per_subject[d.get("subject", f"epilepsiae_{sid}")] = d
    summary = q1prime_cohort_summary(per_subject)
    payload = {
        "schema_version": 1,
        "cohort": list(cohort),
        "per_subject": per_subject,
        **summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return payload


# ---------------------------------------------------------------------------
# Q1' figures (Task 6)
# ---------------------------------------------------------------------------

def figure_q1prime_per_subject_scatter(
    per_subject_dir: Path,
    cohort: Sequence[str],
    out_path: Path,
) -> None:
    """For each subject: scatter (ρ_a, ρ_b) per seizure colored by topic5 subtype."""
    import matplotlib.pyplot as plt
    pal = _morandi_palette()
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), dpi=150)
    axes_flat = axes.flatten()
    for ax, sid in zip(axes_flat, cohort):
        f = per_subject_dir / f"epilepsiae_{sid}__q1prime.json"
        if not f.exists():
            ax.set_title(f"{sid} — missing")
            continue
        with f.open() as fh:
            d = json.load(fh)
        if d.get("status") == "failed":
            ax.set_title(f"{sid} — failed: {d.get('error', '')[:30]}")
            continue
        per_sz = d.get("per_seizure", [])
        if not per_sz:
            continue
        rho_a = [s.get("rho_a") for s in per_sz if s.get("rho_a") is not None]
        rho_b = [s.get("rho_b") for s in per_sz if s.get("rho_b") is not None]
        subtypes = [
            s.get("subtype_label") if s.get("subtype_label") is not None else -2
            for s in per_sz if s.get("rho_a") is not None
        ]
        unique = sorted(set(subtypes), key=lambda x: (x is None, x))
        for k, st in enumerate(unique):
            xs = [a for a, t in zip(rho_a, subtypes) if t == st]
            ys = [b for b, t in zip(rho_b, subtypes) if t == st]
            ax.scatter(xs, ys, color=pal[k % len(pal)], label=f"st={st}", s=40, alpha=0.75)
        ax.axline((-1, -1), (1, 1), color="grey", lw=0.5, ls="--")
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("ρ_a (vs T0)")
        ax.set_ylabel("ρ_b (vs T1)")
        ax.set_title(f"{sid} (swap={d.get('swap_class')}, V={d.get('test', {}).get('cramer_v', 0):.2f})")
        ax.legend(fontsize=7)
    fig.suptitle("Q1' per-seizure (ρ_a, ρ_b) — cohort 4 strict + 548 candidate + 442 descriptive")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def figure_q1prime_cohort_effect(
    cohort_summary_path: Path, out_path: Path,
) -> None:
    """Cohort bar of per-subject Cramér V + AMI."""
    import matplotlib.pyplot as plt
    with cohort_summary_path.open() as fh:
        d = json.load(fh)
    pal = _morandi_palette()
    subjects = list(d["per_subject"].keys())
    cv = [d["per_subject"][s].get("test", {}).get("cramer_v", 0.0) for s in subjects]
    ami = [d["per_subject"][s].get("test", {}).get("ami", 0.0) for s in subjects]
    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)
    x = np.arange(len(subjects))
    ax.bar(x - 0.2, cv, width=0.4, color=pal[0], label="Cramér V")
    ax.bar(x + 0.2, ami, width=0.4, color=pal[1], label="AMI")
    ax.axhline(0.30, ls="--", color="grey", lw=0.7, label="V min")
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("epilepsiae_", "") for s in subjects], rotation=45)
    ax.set_ylabel("effect")
    ax.set_title(f"Q1' cohort effect — verdict: {d.get('cohort_judgement')}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def figure_q1prime_assignment_x_subtype(
    per_subject_dir: Path,
    cohort: Sequence[str],
    out_path: Path,
) -> None:
    """For each strict + candidate subject: stacked bar of assignment counts by subtype."""
    import matplotlib.pyplot as plt
    pal = _morandi_palette()
    fig, axes = plt.subplots(1, len(cohort), figsize=(3 * len(cohort), 4), dpi=150, sharey=True)
    if len(cohort) == 1:
        axes = [axes]
    for ax, sid in zip(axes, cohort):
        f = per_subject_dir / f"epilepsiae_{sid}__q1prime.json"
        if not f.exists():
            continue
        with f.open() as fh:
            d = json.load(fh)
        if d.get("status") == "failed":
            continue
        cont = d.get("test", {}).get("contingency")
        a_levels = d.get("test", {}).get("a_levels", [])
        s_levels = d.get("test", {}).get("s_levels", [])
        if not cont:
            ax.set_title(f"{sid} no contingency")
            continue
        cont = np.array(cont)
        bottom = np.zeros(len(s_levels))
        for i, a in enumerate(a_levels):
            ax.bar(range(len(s_levels)), cont[i], bottom=bottom, color=pal[i % len(pal)], label=a)
            bottom += cont[i]
        ax.set_xticks(range(len(s_levels)))
        ax.set_xticklabels([f"st={s}" for s in s_levels])
        ax.set_title(f"{sid} (swap={d.get('swap_class')}, p={d.get('test', {}).get('p', 1):.3f})")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("n seizures")
    fig.suptitle("Q1' contingency: assignment × ictal subtype")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
