#!/usr/bin/env python3
"""Topic5 V3d -- scaffold A/B cohort analysis (downstream of the per-subject producer).

Aggregates the per-subject outputs of ``scripts/run_topic5_scaffold_ab_switching.py``
(``cohort_index.{json,csv}`` + ``per_subject/<ds_sid>_scaffold_ab_{summary.json,
per_seizure.csv,contrast.npz}``) into one cohort-level analysis answering three
questions on top of the producer's own per-subject H1 verdict:

  (P1, "坑一" avoid-the-middle) Does peri-onset energy avoid the midpoint of the A<->B
      contrast axis across subjects/geometries, rather than sitting near C_AB=0
      (undecided between the two propagation templates)? -> ``frac_near_zero`` per
      subject + a (rho_AB, frac_near_zero, tier) scatter.
  (typing) Does a subject's seizures reproducibly type into A-dominant vs B-dominant
      beyond chance, rather than each seizure's within-seizure lateral composition
      being indistinguishable from a random relabeling of the subject's own pooled
      windows? -> ``typing_purity_test`` (the primary inferential test in this script).
  (H1) Cohort-level near-onset locking count: k subject_locked out of m H1_eligible
      (straight passthrough + one-sided exact binomial). This script does NOT
      recompute H1 -- that is entirely the producer's job (spec
      docs/superpowers/specs/2026-07-09-topic5-v3d-scaffold-ab-lateral-switching-design.md
      sec4/sec6.3); it only counts and tests the already-computed verdicts.

This is a READ-ONLY downstream analysis: it does not touch the V3d numeric core
(``src/topic5_scaffold_ab_contrast.py``) and does not re-derive C_AB / axis_present /
locking -- all three are read verbatim from the producer's per-subject outputs.

Schema note: the exact per-subject file schema (CSV columns, npz key names) is owned
by ``scripts/run_topic5_scaffold_ab_switching.py``. At the time this script was
written, a full-cohort batch run of that producer was actively writing to
``results/topic5_ictal_recruitment/scaffold_ab_switching/per_subject/``, so that
directory was deliberately NOT read or executed against here (race condition) --
the schema below was instead read from the producer's committed source (its
``COHORT_COLS``, ``_PER_SEIZURE_COLS`` and ``npz_dict`` literals), not from a live
file. Every loader is still defensive on top of that: it discovers the real npz key
names from ``.files`` at runtime (falling back through a small candidate list) and
records which key it used, and it fails soft per subject (a missing/corrupt file is
recorded in ``skipped``, never a crash of the whole aggregate).

Output: ``results/topic5_ictal_recruitment/scaffold_ab_switching/cohort_analysis.json``

CLI: ``python scripts/analyze_topic5_scaffold_ab_cohort.py [--n-perm 1000] [--seed 0]``
(reads the whole cohort; index-driven, no per-subject flags).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = _ROOT / "results/topic5_ictal_recruitment/scaffold_ab_switching"
SUB_DIR = OUT_DIR / "per_subject"
COHORT_INDEX_JSON = OUT_DIR / "cohort_index.json"
COHORT_INDEX_CSV = OUT_DIR / "cohort_index.csv"
COHORT_ANALYSIS_JSON = OUT_DIR / "cohort_analysis.json"

NEAR_ZERO_THRESHOLD = 0.2   # |C_AB| below this -> "near the A/B midpoint" (坑一)
TYPE_SIDE_THRESHOLD = 0.2   # seizure net_side >=/<= this -> A_type/B_type, else mid

# Candidate npz key names per logical array. Exact keys as committed in
# scripts/run_topic5_scaffold_ab_switching.py (`npz_dict`) are listed first;
# the rest are defensive fallbacks in case the running batch used a schema this
# script's author could not see (per the module docstring).
_CAB_KEYS = ["cab", "C_AB", "c_ab", "contrast"]
_PRESENT_KEYS = ["present", "axis_present"]
_CENTERS_KEYS = ["grid_centers", "centers", "window_center_sec"]
_SEIZURE_IDX_KEYS = ["seizure_idx", "seizure_indices", "seizure_ids"]


# ---------------------------------------------------------------------------
# Pure statistical functions -- unit-tested on synthetic data only in
# tests/test_scaffold_ab_cohort_analysis.py. Nothing below this block touches
# a file; everything below is plain arrays in, dict/float out.
# ---------------------------------------------------------------------------

def frac_near_zero(C_AB, present, threshold: float = NEAR_ZERO_THRESHOLD) -> float:
    """Fraction of axis-present windows with |C_AB| < threshold ("坑一": does energy
    avoid the A/B midpoint, or does it often sit undecided between the two templates?).

    Works on any-shape arrays (1D per-seizure or 2D per-subject [n_seizures, n_time])
    as long as C_AB and present broadcast together -- boolean-mask indexing flattens
    either way.

    Denominator is present & isfinite(C_AB) (mirrors the present-plus-finite
    convention used throughout src/topic5_scaffold_ab_contrast.py, e.g. its internal
    ``_range_mask``): present=True should always coincide with a finite C_AB by
    construction, but a NaN under present=True is excluded defensively rather than
    silently miscounted either way. Returns NaN if no window qualifies.
    """
    C_AB = np.asarray(C_AB, float)
    present = np.asarray(present, bool)
    mask = present & np.isfinite(C_AB)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(C_AB[mask]) < threshold))


def _purity(vals) -> float:
    """max(frac_A, frac_B) for one seizure's axis-present C_AB values.

    frac_A/frac_B = fraction of vals with C_AB >= 0.2 / <= -0.2 (TYPE_SIDE_THRESHOLD).
    """
    vals = np.asarray(vals, float)
    if vals.size == 0:
        return float("nan")
    frac_A = float(np.mean(vals >= TYPE_SIDE_THRESHOLD))
    frac_B = float(np.mean(vals <= -TYPE_SIDE_THRESHOLD))
    return max(frac_A, frac_B)


def typing_purity_test(per_seizure_cab_present, n_perm: int = 1000, seed: int = 0) -> dict:
    """Does a subject's seizures reproducibly type A-dominant vs B-dominant beyond chance?

    Parameters
    ----------
    per_seizure_cab_present : sequence of 1D array-like
        One entry per seizure: that seizure's C_AB values restricted to its own
        axis-present windows. Entries with zero usable (finite) values are dropped
        before both the observed statistic and the null (their purity is undefined,
        not 0 -- a seizure with no scaffold signal contributes no typing evidence).
    n_perm : int, default 1000
    seed : int, default 0

    Statistic
    ---------
    obs_mean_purity = mean over seizures of purity_s = max(frac_A, frac_B), where
    frac_A/frac_B are the fraction of that seizure's values with C_AB >= 0.2 /
    C_AB <= -0.2.

    Null
    ----
    Pool ALL values across the subject's seizures, take a random permutation of the
    pool, and re-cut it into chunks sized exactly like the original seizures (a label
    shuffle: every observed value is reused exactly once, so the pooled distribution
    -- including how far it sits from the A/B midpoint -- is preserved bit-for-bit;
    only which seizure each value is attributed to is randomized). Recompute mean
    purity; repeat n_perm times.
    p = (1 + #{null_mean_purity >= obs_mean_purity}) / (n_perm + 1).

    A significant result means seizures are genuinely typed (a seizure's own windows
    cluster on one side more than a random relabeling of the same pooled values would
    predict), not an artifact of the window-level geometry (which the null holds
    fixed by construction).

    Returns
    -------
    dict : {obs_mean_purity, null_p, n_seizures_used}
        NaN-flavored fields if no seizure has any usable (finite) axis-present value.
    """
    arrays = [np.asarray(a, float) for a in per_seizure_cab_present]
    arrays = [a[np.isfinite(a)] for a in arrays]
    arrays = [a for a in arrays if a.size > 0]
    n_seizures_used = len(arrays)
    if n_seizures_used == 0:
        return {"obs_mean_purity": float("nan"), "null_p": float("nan"), "n_seizures_used": 0}

    obs_mean_purity = float(np.mean([_purity(a) for a in arrays]))

    pooled = np.concatenate(arrays)
    sizes = [a.size for a in arrays]
    rng = np.random.default_rng(seed)
    null_means = np.empty(n_perm, float)
    for i in range(n_perm):
        shuffled = rng.permutation(pooled)
        idx = 0
        purities = []
        for sz in sizes:
            purities.append(_purity(shuffled[idx:idx + sz]))
            idx += sz
        null_means[i] = np.mean(purities)

    null_p = float((1 + np.sum(null_means >= obs_mean_purity)) / (n_perm + 1))
    return {"obs_mean_purity": obs_mean_purity, "null_p": null_p, "n_seizures_used": n_seizures_used}


def bimodality_coefficient(net_sides) -> float:
    """Sarle's bimodality coefficient BC = (skew^2 + 1) / (kurt_excess + 3(n-1)^2/((n-2)(n-3))).

    BC > 5/9 ~= 0.555 (the value for a uniform distribution) is the conventional
    "looks more bimodal/multimodal than unimodal" flag; a normal distribution gives
    BC = 1/3. Uses scipy.stats.skew/kurtosis (bias=True -- the population/Fisher-
    Pearson moment coefficients this formula is defined on) opportunistically when
    scipy is importable, else falls back to computing the same moments by hand, so
    this function has no hard scipy dependency. Returns NaN for n<4 or a degenerate
    (zero-variance) input, where the formula is undefined.
    """
    x = np.asarray(net_sides, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 4:
        return float("nan")
    mean = x.mean()
    m2 = float(np.mean((x - mean) ** 2))
    if m2 < 1e-12:
        return float("nan")
    try:
        from scipy import stats as _scipy_stats
        skew = float(_scipy_stats.skew(x, bias=True))
        kurt_excess = float(_scipy_stats.kurtosis(x, fisher=True, bias=True))
    except ImportError:
        skew = float(np.mean((x - mean) ** 3) / m2 ** 1.5)
        kurt_excess = float(np.mean((x - mean) ** 4) / m2 ** 2 - 3.0)
    denom = kurt_excess + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    if not np.isfinite(denom) or abs(denom) < 1e-12:
        return float("nan")
    return float((skew ** 2 + 1.0) / denom)


# ---------------------------------------------------------------------------
# Defensive loaders. See module docstring: the live files could not be
# inspected while a producer batch was writing them, so every reader below
# discovers columns/keys at runtime and fails soft (returns an error string,
# never raises past its own function) rather than assuming the schema holds.
# ---------------------------------------------------------------------------

def _to_float(v):
    if v is None or v == "":
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _to_int(v):
    if v is None or v == "":
        return None
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def _to_bool(v):
    if isinstance(v, bool):
        return v
    if v is None or v == "":
        return None
    if isinstance(v, str):
        vl = v.strip().lower()
        if vl in ("true", "1", "yes"):
            return True
        if vl in ("false", "0", "no"):
            return False
        return None
    try:
        return bool(v)
    except Exception:
        return None


def _json_safe(v):
    """Recursively replace non-finite floats with None so json.dumps produces valid
    JSON (mirrors the `_j` helper in run_topic5_scaffold_ab_switching.py; not
    imported from there since it is that script's private helper and this script
    must not modify/depend on that committed script's internals)."""
    if v is None or isinstance(v, (bool, str)):
        return v
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f if np.isfinite(f) else None


def load_cohort_index():
    """Returns (records, error). records is a list of per-subject dicts (JSON-native
    types if cohort_index.json is used; raw CSV strings if falling back to .csv)."""
    if COHORT_INDEX_JSON.exists():
        try:
            payload = json.loads(COHORT_INDEX_JSON.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, f"cohort_index.json unreadable: {type(exc).__name__}: {exc}"
        if isinstance(payload, list):
            return payload, None
        if isinstance(payload, dict):
            for key in ("subjects", "records", "cohort", "data"):
                v = payload.get(key)
                if isinstance(v, list):
                    return v, None
        return None, ("cohort_index.json has no recognizable subject-list key "
                       f"(top-level: {list(payload) if isinstance(payload, dict) else type(payload)})")
    if COHORT_INDEX_CSV.exists():
        try:
            with COHORT_INDEX_CSV.open(newline="") as fh:
                return list(csv.DictReader(fh)), None
        except Exception as exc:
            return None, f"cohort_index.csv unreadable: {type(exc).__name__}: {exc}"
    return None, f"neither {COHORT_INDEX_JSON.name} nor {COHORT_INDEX_CSV.name} found in {OUT_DIR}"


def _first_key(available, candidates):
    for c in candidates:
        if c in available:
            return c
    return None


def load_subject_contrast_npz(ds_sid: str):
    """Returns (dict{cab,present,centers,seizure_idx,_keys_used}, error)."""
    fp = SUB_DIR / f"{ds_sid}_scaffold_ab_contrast.npz"
    if not fp.exists():
        return None, f"missing_file:{fp.name}"
    try:
        with np.load(fp, allow_pickle=False) as npz:
            available = list(npz.files)
            cab_key = _first_key(available, _CAB_KEYS)
            present_key = _first_key(available, _PRESENT_KEYS)
            centers_key = _first_key(available, _CENTERS_KEYS)
            seizure_idx_key = _first_key(available, _SEIZURE_IDX_KEYS)
            if cab_key is None or present_key is None:
                return None, f"npz missing a cab/present-like key (found keys: {available})"
            cab = np.asarray(npz[cab_key], float)
            present = np.asarray(npz[present_key], bool)
            if cab.shape != present.shape:
                return None, f"npz cab shape {cab.shape} != present shape {present.shape}"
            return {
                "cab": cab,
                "present": present,
                "centers": np.asarray(npz[centers_key], float) if centers_key else None,
                "seizure_idx": np.asarray(npz[seizure_idx_key], int) if seizure_idx_key else None,
                "_keys_used": {"cab": cab_key, "present": present_key,
                               "centers": centers_key, "seizure_idx": seizure_idx_key,
                               "npz_files_seen": available},
            }, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def load_subject_per_seizure(ds_sid: str):
    """Returns (list[dict rows, raw CSV strings], error)."""
    fp = SUB_DIR / f"{ds_sid}_scaffold_ab_per_seizure.csv"
    if not fp.exists():
        return None, f"missing_file:{fp.name}"
    try:
        with fp.open(newline="") as fh:
            return list(csv.DictReader(fh)), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Per-subject analysis
# ---------------------------------------------------------------------------

def analyze_subject(rec: dict, n_perm: int = 1000, seed: int = 0):
    """One cohort_index record -> (result dict, None) on success, or (None, reason)
    if this subject must be skipped (fail-soft: never raises)."""
    ds_sid = rec.get("subject")
    if not ds_sid:
        return None, "cohort_index_record_missing_subject_field"
    if rec.get("status") != "ok":
        return None, f"cohort_status_{rec.get('status')}: {rec.get('drop_reason') or ''}"

    npz_data, npz_err = load_subject_contrast_npz(ds_sid)
    if npz_data is None:
        return None, f"npz_load_failed: {npz_err}"

    seizure_rows, csv_err = load_subject_per_seizure(ds_sid)
    if seizure_rows is None:
        return None, f"per_seizure_csv_load_failed: {csv_err}"

    cab = npz_data["cab"]
    present = npz_data["present"]
    n_sz = cab.shape[0]

    seizure_idx_npz = npz_data["seizure_idx"]
    if seizure_idx_npz is None or len(seizure_idx_npz) != n_sz:
        seizure_idx_npz = np.arange(n_sz)
        join_method = "positional_fallback"
    else:
        join_method = "seizure_idx"

    event_class_by_idx = {}
    for row in seizure_rows:
        si = _to_int(row.get("seizure_idx"))
        if si is not None:
            event_class_by_idx[si] = row.get("event_class")

    net_sides, per_seizure_cab_present, seizure_details = [], [], []
    n_a = n_b = n_mid = n_bimodal = 0
    for i in range(n_sz):
        mask = np.asarray(present[i], bool) & np.isfinite(cab[i])
        vals = cab[i][mask]
        per_seizure_cab_present.append(vals)
        net_side = float(np.mean(vals)) if vals.size > 0 else float("nan")
        net_sides.append(net_side)

        if np.isfinite(net_side) and net_side >= TYPE_SIDE_THRESHOLD:
            stype = "A_type"; n_a += 1
        elif np.isfinite(net_side) and net_side <= -TYPE_SIDE_THRESHOLD:
            stype = "B_type"; n_b += 1
        else:
            stype = "mid"; n_mid += 1

        si = int(seizure_idx_npz[i])
        ec = event_class_by_idx.get(si)
        if ec == "switch":
            n_bimodal += 1
        seizure_details.append({
            "seizure_idx": si, "net_side": net_side, "seizure_type": stype,
            "event_class": ec, "n_present_windows": int(vals.size),
        })

    frac_seizures_bimodal = (n_bimodal / n_sz) if n_sz > 0 else float("nan")
    two_type_distinguishable = bool(n_a >= 2 and n_b >= 2)

    typing = typing_purity_test(per_seizure_cab_present, n_perm=n_perm, seed=seed)
    bc = bimodality_coefficient(net_sides)
    fnz = frac_near_zero(cab, present)

    result = {
        "subject": ds_sid,
        # passthrough (cohort_index)
        "template_pair_tier": rec.get("template_pair_tier"),
        "rho_AB": _to_float(rec.get("rho_AB")),
        "n_seizures": _to_int(rec.get("n_seizures")),
        "n_valid_seizures": _to_int(rec.get("n_valid_seizures")),
        "H1_eligible": _to_bool(rec.get("H1_eligible")),
        "testable": _to_bool(rec.get("testable")),
        "low_dof": _to_bool(rec.get("low_dof")),
        "subject_locked": _to_bool(rec.get("subject_locked")),
        "H1_p": _to_float(rec.get("H1_p")),
        "L_obs": _to_float(rec.get("L_obs")),
        "n_joint": _to_int(rec.get("n_joint")),
        # this script's own metrics
        "n_seizures_npz": n_sz,
        "frac_near_zero": fnz,
        "seizure_type_counts": {"A_type": n_a, "B_type": n_b, "mid": n_mid, "n_bimodal": n_bimodal},
        "two_type_distinguishable": two_type_distinguishable,
        "frac_seizures_bimodal": frac_seizures_bimodal,
        "typing_purity_test": typing,
        "bimodality_coefficient": bc,
        # provenance
        "seizure_join_method": join_method,
        "npz_keys_used": npz_data["_keys_used"],
        "seizure_details": seizure_details,
    }
    return result, None


# ---------------------------------------------------------------------------
# Cohort aggregation
# ---------------------------------------------------------------------------

def aggregate_h1(cohort_records: list) -> dict:
    """k/m one-sided exact binomial over H1_eligible subjects.

    Straight passthrough from cohort_index -- H1_eligible/subject_locked are already
    fully computed upstream by the producer and require no per-subject file access,
    so this uses ALL cohort_index records directly (not gated on whether this
    script's own npz/CSV loading succeeded for the deeper typing/frac_near_zero
    metrics elsewhere in this file).
    """
    h1_eligible = [r for r in cohort_records if _to_bool(r.get("H1_eligible")) is True]
    m = len(h1_eligible)
    k = sum(1 for r in h1_eligible if _to_bool(r.get("subject_locked")) is True)
    out = {"k": k, "m": m}
    if m == 0:
        out.update(p=None, ci=None,
                    note="no H1-eligible subjects in cohort_index; binomial not computed (m=0).")
        return out
    try:
        from scipy.stats import binomtest
    except ImportError:
        out.update(p=None, ci=None, note="scipy unavailable; binomial p/CI not computed.")
        return out
    res = binomtest(k, m, 0.05, alternative="greater")
    ci = res.proportion_ci()
    out.update(
        p=float(res.pvalue), ci=[float(ci.low), float(ci.high)],
        note=("one-sided exact binomial subject-count test vs p0=0.05 (spec sec4/sec6.3); "
              "does not pool seizures across subjects; denominator = H1_eligible subjects "
              "in cohort_index directly (independent of this script's own per-subject file "
              "loading elsewhere in this analysis)."),
    )
    return out


def aggregate_analyzed(analyzed: list) -> dict:
    """typing + avoid-the-middle cohort summaries, over subjects this script could
    itself load (a subset of cohort_index -- see aggregate_h1 for why H1 does not
    use this same subset)."""
    sig_typing = [r for r in analyzed
                  if np.isfinite(r["typing_purity_test"]["null_p"])
                  and r["typing_purity_test"]["null_p"] < 0.05]
    two_type = [r for r in analyzed if r["two_type_distinguishable"]]

    points = [{"subject": r["subject"], "rho_AB": r["rho_AB"],
               "frac_near_zero": r["frac_near_zero"], "template_pair_tier": r["template_pair_tier"]}
              for r in analyzed]
    nonneg = [p["subject"] for p in points if p["rho_AB"] is not None and p["rho_AB"] >= 0]
    note = (f"{len(nonneg)}/{len(points)} subjects have rho_AB >= 0: {nonneg}" if nonneg
            else f"0/{len(points)} subjects have rho_AB >= 0 (every analyzed template pair sits "
                 "on the anti-correlated side of 0).")

    return {
        "typing": {
            "n_subjects_analyzed": len(analyzed),
            "alpha": 0.05,
            "n_typing_significant": len(sig_typing),
            "typing_significant_subjects": [r["subject"] for r in sig_typing],
            "n_two_type_distinguishable": len(two_type),
            "two_type_distinguishable_subjects": [r["subject"] for r in two_type],
        },
        "avoid_middle_scatter": {"points": points, "note": note},
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run(n_perm: int = 1000, seed: int = 0) -> dict:
    cohort_records, err = load_cohort_index()
    if cohort_records is None:
        print(f"FATAL: cannot load cohort index: {err}", file=sys.stderr)
        sys.exit(1)

    analyzed, skipped = [], []
    for rec in cohort_records:
        result, skip_reason = analyze_subject(rec, n_perm=n_perm, seed=seed)
        if result is None:
            skipped.append({"subject": rec.get("subject"), "reason": skip_reason})
        else:
            analyzed.append(result)

    out = {
        "generated_by": "scripts/analyze_topic5_scaffold_ab_cohort.py",
        "spec": "docs/superpowers/specs/2026-07-09-topic5-v3d-scaffold-ab-lateral-switching-design.md",
        "n_subjects_in_cohort_index": len(cohort_records),
        "n_subjects_analyzed": len(analyzed),
        "n_subjects_skipped": len(skipped),
        "skipped": skipped,
        "per_subject": analyzed,
        "cohort": {
            "H1": aggregate_h1(cohort_records),
            **aggregate_analyzed(analyzed),
        },
        "caveats": [
            "typing_purity_test 的 null 只在被试内部打乱各次发作对 C_AB 值的归属，不跨被试混合；显著只说明"
            "该被试的发作确实分侧型（同一次发作的窗更聚在一侧，超过随机重新分配预期），不是队列层面的假设"
            "检验——队列层面只在 cohort.typing 里做计数汇总，不是再做一次跨被试检验。",
            "cohort.H1 的 k/m 直接取 cohort_index 的 H1_eligible/subject_locked passthrough 字段，本脚本"
            "不重算 H1 本身（H1 的完整定义 = producer 的 subject_locking_null，见 spec §6.3）。",
            "frac_near_zero / avoid_middle_scatter 是坑一（能量是否落在 A/B 中点附近）的描述性汇总，不是"
            "显著性检验。",
        ],
    }
    COHORT_ANALYSIS_JSON.parent.mkdir(parents=True, exist_ok=True)
    COHORT_ANALYSIS_JSON.write_text(
        json.dumps(_json_safe(out), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {COHORT_ANALYSIS_JSON} "
          f"({len(analyzed)} analyzed, {len(skipped)} skipped of {len(cohort_records)} in cohort_index)")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perm", type=int, default=1000,
                     help="permutations for the per-subject typing_purity_test null (default 1000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(n_perm=args.n_perm, seed=args.seed)


if __name__ == "__main__":
    main()
