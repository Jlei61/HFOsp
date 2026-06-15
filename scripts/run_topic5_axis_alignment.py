"""Topic 5 A-line — ictal-activation vs interictal-axis ALIGNMENT (axis/gradient collinearity).

Per (subject, eligible seizure): build the ictal activation field by swapping the interictal
axis record's per-channel rank for that channel's activation (broadband_auc_0_10s = the PRIMARY
metric; HFA / ramp / EI-like = sensitivity / exploratory), joined BY NAME, smooth both on the
SAME normalized plane, and take | corr_pair_mirror_invariant |. This is SIGN-FREE: it tests
whether the interictal axis and the ictal activation SPATIAL GRADIENT are COLLINEAR (reversal
allowed). It does NOT test directional propagation or same-direction 'replay'; forward-vs-reverse
sign is a secondary descriptive question, not this statistic.

Per subject: median-over-seizures real |corr| vs B paired-draw null distributions. Four nulls,
each controlling a DIFFERENT confound (they are SEPARATE controls, NOT nested — except 'joint',
which is nested inside both within-shaft and anchor-matched):
  - channel        : full shuffle              -> beat = more than a coarse anatomical anchor
  - within_shaft   : shuffle within a shaft    -> beat = finer than same-shaft geometry
  - anchor_matched : shuffle within activity-bins -> beat = not just 'source is more active'
                     (does NOT also hold shaft fixed)
  - joint          : shuffle within (shaft x activity-bin) -> conservative 'both at once'
Each null reports `effective_shuffle_n` (channels actually moved); a near-degenerate null on a
small/few-shaft subject is flagged, not silently treated as a real control. Per-subject PASS =
real > 95th pct of that null. Cohort = binomial(#pass vs 5%) AND Wilcoxon(real - null_median)
AND leave-one-subject-out worst-case p. Evidence tier: Epilepsiae = primary cohort; yuquan =
descriptive (1 subject). This is an EXPLORATORY family (4 metrics x 4 nulls) -> FDR applied in
the aggregate step; broadband is the pre-registered primary metric, HFA/ramp/EI are sensitivity.

Reuses: src.propagation_contact_plane_readout (make_plane_grid / R_smooth_rank /
corr_pair_mirror_invariant) + src.topic5_axis_alignment (join + null shuffles, TDD'd).
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered")

from src.propagation_contact_plane_readout import (make_plane_grid, R_smooth_rank,
                                                   corr_pair_mirror_invariant, S_THRESH, OVERLAP_MIN)
from src.topic5_axis_alignment import (matched_channels, make_field_record,
                                       channel_shuffle, within_shaft_shuffle, anchor_matched_shuffle,
                                       within_shaft_anchor_shuffle, effective_shuffle_n, along_axis_sign)

ACTIVATION_KEY = {"broadband": "bb_auc", "hfa": "hfa_auc", "ramp": "ramp", "ei": "ei_like"}

CACHE_DIR = Path("results/topic5_ictal_recruitment/t0_feature_cache")
AXIS_DIR = Path("results/spatial_modulation/propagation_geometry/observation_readout/real_subjects")
OUT = Path("results/topic5_ictal_recruitment/axis_alignment")
RNG_SEED = 20260614


def _abs_corr(Fi, Fj):
    r = corr_pair_mirror_invariant(Fi["T"], Fi["S"], Fj["T"], Fj["S"], S_THRESH, OVERLAP_MIN)["corr"]
    return abs(r) if r is not None and np.isfinite(r) else np.nan


def _p95_med(draws):
    """(95th pct, median) of the per-subject null distribution = median-over-seizures per draw."""
    if not draws:
        return None, None
    dist = np.nanmedian(np.asarray(draws, float), axis=0)   # [B]
    return float(np.nanpercentile(dist, 95)), float(np.nanmedian(dist))


def _subject(ds_sid, *, B, rng, activation="bb_auc", sz_subset=None, negative_control=False):
    axis_f = AXIS_DIR / f"{ds_sid}_t_a.json"          # A-primary uses the primary template only
    npz_f = CACHE_DIR / f"{ds_sid}.npz"
    if not axis_f.exists() or not npz_f.exists():
        return None
    axis = json.load(open(axis_f))
    if not axis.get("channels"):
        return None
    data = np.load(npz_f, allow_pickle=True)
    meta = json.load(open(CACHE_DIR / f"{ds_sid}.json"))
    cache_names = [str(x) for x in data["channels"]]
    cidx = {n: i for i, n in enumerate(cache_names)}

    matched = matched_channels(axis, {n: 0.0 for n in cache_names})   # axis chans present ictally
    if len(matched) < 6:
        return {"subject_id": ds_sid, "status": f"insufficient_matched_{len(matched)}"}
    names_m = [c["name"] for c in matched]
    m_in_cache = np.array([cidx[n] for n in names_m])
    has_anchor = any(k.startswith("bact__") for k in data.files)

    X, Y = make_plane_grid()
    inter = [float(c["typical_rank"]) for c in matched]   # interictal along-axis rank (1D sign side-channel)
    inter_rec = make_field_record(matched, inter)
    F_inter = R_smooth_rank(inter_rec, X, Y, None, S_THRESH)
    sigma = F_inter["sigma_xy"]

    def fld(vals):
        return R_smooth_rank(make_field_record(matched, vals), X, Y, sigma, S_THRESH)

    eff = {"channel": effective_shuffle_n(names_m, None, "channel"),
           "within_shaft": effective_shuffle_n(names_m, None, "within_shaft")}
    eff_anchor = eff_joint = None
    real, ch_draws, sh_draws, an_draws, jt_draws = [], [], [], [], []   # *_draws[sz] = [B corrs]
    sign_corrs = []   # 1D along-axis signed spearman per seizure (sign-FREE primary stat unchanged)
    for idx in meta["eligible_idxs"]:
        if sz_subset is not None and idx not in sz_subset:   # split-half robustness filter
            continue
        key = f"{activation}__{idx}"
        if key not in data.files:
            continue
        ict_vals = data[key][m_in_cache].astype(float)
        if np.isfinite(ict_vals).sum() < 6:
            continue
        scored = channel_shuffle(ict_vals, rng) if negative_control else ict_vals  # bad-data gate
        r = _abs_corr(F_inter, fld(scored))
        if not np.isfinite(r):
            continue
        real.append(r)
        s = along_axis_sign(inter, ict_vals)   # 1D side-channel: source(<0)/sink(>0) bias, mirror-free
        if s["sign"] != 0:
            sign_corrs.append(s["signed_corr"])
        ch_draws.append([_abs_corr(F_inter, fld(channel_shuffle(ict_vals, rng))) for _ in range(B)])
        sh_draws.append([_abs_corr(F_inter, fld(within_shaft_shuffle(ict_vals, names_m, rng))) for _ in range(B)])
        if has_anchor and f"bact__{idx}" in data.files:
            anchor = data[f"bact__{idx}"][m_in_cache].astype(float)
            an_draws.append([_abs_corr(F_inter, fld(anchor_matched_shuffle(ict_vals, anchor, rng))) for _ in range(B)])
            jt_draws.append([_abs_corr(F_inter, fld(within_shaft_anchor_shuffle(ict_vals, names_m, anchor, rng))) for _ in range(B)])
            if eff_anchor is None:
                eff_anchor = effective_shuffle_n(names_m, anchor, "anchor")
                eff_joint = effective_shuffle_n(names_m, anchor, "joint")
    if not real:
        return {"subject_id": ds_sid, "status": "no_resolvable_seizure"}
    real_med = float(np.median(real))
    sign_fields = {}
    if sign_corrs:   # >=1 resolvable along-axis sign -> add the 1D source/sink side-channel summary
        sc = np.asarray(sign_corrs, float)
        n_pos, n_neg = int((sc > 0).sum()), int((sc < 0).sum())
        same = max(n_pos, n_neg) / len(sc)                   # majority-sign fraction in [0.5, 1]
        ent = 0.0 if same in (0.0, 1.0) else float(-same * np.log2(same) - (1 - same) * np.log2(1 - same))
        sign_fields = {"sign_signed_median": float(np.median(sc)), "sign_same_frac": float(same),
                       "sign_entropy": ent, "n_sign": len(sc)}
    ch_p95, ch_med = _p95_med(ch_draws)
    sh_p95, sh_med = _p95_med(sh_draws)
    an_p95, an_med = _p95_med(an_draws)
    jt_p95, jt_med = _p95_med(jt_draws)
    return {
        "subject_id": ds_sid, "dataset": ds_sid.split("_", 1)[0], "status": "ok",
        "activation": activation, "n_seizures": len(real), "n_matched_channels": len(matched),
        "real_median_abs_corr": real_med,
        "channel_null_median": ch_med, "channel_null_p95": ch_p95,
        "within_shaft_null_median": sh_med, "within_shaft_null_p95": sh_p95,
        "anchor_matched_null_median": an_med, "anchor_matched_null_p95": an_p95,
        "joint_null_median": jt_med, "joint_null_p95": jt_p95,
        "pass_channel_null": (bool(real_med > ch_p95) if ch_p95 is not None else None),
        "pass_within_shaft_null": (bool(real_med > sh_p95) if sh_p95 is not None else None),
        "pass_anchor_matched_null": (bool(real_med > an_p95) if an_p95 is not None else None),
        "pass_joint_null": (bool(real_med > jt_p95) if jt_p95 is not None else None),
        "effective_shuffle_n": {"channel": eff["channel"], "within_shaft": eff["within_shaft"],
                                "anchor_matched": eff_anchor, "joint": eff_joint},
        **sign_fields,
    }


def main():
    global CACHE_DIR, AXIS_DIR   # repo global-swap convention (cf. OUT etc. module consts)
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--activation", choices=list(ACTIVATION_KEY), default="broadband")
    ap.add_argument("--B", type=int, default=1000, help="null draws per seizure (smoke: 200)")
    ap.add_argument("--cache-dir", default=str(CACHE_DIR), help="T0 feature cache (.npz/.json) dir")
    ap.add_argument("--axis-dir", default=str(AXIS_DIR), help="interictal axis record (_t_a.json) dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    CACHE_DIR = Path(args.cache_dir)
    AXIS_DIR = Path(args.axis_dir)
    rng = np.random.default_rng(RNG_SEED)
    act_key = ACTIVATION_KEY[args.activation]
    out_path = Path(args.out) if args.out else OUT / f"axis_alignment_{args.activation}_B{args.B}.json"

    cached = sorted(p.stem for p in CACHE_DIR.glob("*.npz"))
    if args.subjects:
        cached = [s for s in cached if s in set(args.subjects)]
    print(f"[axis-alignment] activation={args.activation} B={args.B} | {len(cached)} cached subjects",
          flush=True)
    rows = []
    for ds_sid in cached:
        r = _subject(ds_sid, B=args.B, rng=rng, activation=act_key)
        if r is None:
            continue
        rows.append(r)
        if r.get("status") == "ok":
            an = (f"an_p95={r['anchor_matched_null_p95']:.3f}({'PASS' if r['pass_anchor_matched_null'] else 'no'})"
                  if r["anchor_matched_null_p95"] is not None else "an=NA")
            print(f"  {ds_sid}: real={r['real_median_abs_corr']:.3f} | "
                  f"ch_p95={r['channel_null_p95']:.3f}({'PASS' if r['pass_channel_null'] else 'no'}) "
                  f"sh_p95={r['within_shaft_null_p95']:.3f}({'PASS' if r['pass_within_shaft_null'] else 'no'}) "
                  f"{an} | n_sz={r['n_seizures']}", flush=True)
        else:
            print(f"  {ds_sid}: {r['status']}", flush=True)

    ok = [r for r in rows if r.get("status") == "ok"]
    epi = [r for r in ok if r["dataset"] == "epilepsiae"]
    summ = {"activation": args.activation, "B": args.B,
            "evidence_tier": "Epilepsiae=primary cohort; yuquan=descriptive (1 subject)",
            "n_subjects_ok": len(ok), "epilepsiae_primary": _cohort_stats(epi),
            "yuquan_descriptive": [r for r in ok if r["dataset"] == "yuquan"],
            "per_subject": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\nwrote {out_path}")
    cs = summ["epilepsiae_primary"]
    print(f"=== Epilepsiae primary cohort (n={cs['n']}, activation={args.activation}) ===")
    for k, lbl in [("channel", "channel (coarse anchor)"),
                   ("within_shaft", "within-shaft (fine axis)"),
                   ("anchor_matched", "anchor-matched (activity)"),
                   ("joint", "joint shaft+activity")]:
        if cs.get(f"n_pass_{k}") is not None:
            print(f"  beat {lbl:26s}: {cs[f'n_pass_{k}']}/{cs['n']} "
                  f"(binom p={cs.get(f'binom_p_{k}')}, Wilcox p={cs.get(f'wilcoxon_p_{k}')}, "
                  f"LOSO-worst Wilcox p={cs.get(f'loso_wilcoxon_max_p_{k}')})")


def _cohort_stats(rows):
    from scipy.stats import binomtest, wilcoxon
    n = len(rows)
    if n == 0:
        return {"n": 0}

    def _wilcox(diffs):
        try:
            return wilcoxon(diffs, alternative="greater").pvalue if any(d != 0 for d in diffs) else 1.0
        except ValueError:
            return 1.0

    out = {"n": n}
    for key, passk, medk in [("channel", "pass_channel_null", "channel_null_median"),
                             ("within_shaft", "pass_within_shaft_null", "within_shaft_null_median"),
                             ("anchor_matched", "pass_anchor_matched_null", "anchor_matched_null_median"),
                             ("joint", "pass_joint_null", "joint_null_median")]:
        sub = [r for r in rows if r.get(passk) is not None and r.get(medk) is not None]
        if not sub:
            out[f"n_pass_{key}"] = out[f"binom_p_{key}"] = out[f"wilcoxon_p_{key}"] = None
            out[f"loso_binom_max_p_{key}"] = out[f"loso_wilcoxon_max_p_{key}"] = None
            continue
        npass = sum(bool(r[passk]) for r in sub)
        diff = [r["real_median_abs_corr"] - r[medk] for r in sub]
        out[f"n_pass_{key}"] = npass
        out[f"n_eval_{key}"] = len(sub)
        out[f"binom_p_{key}"] = round(float(binomtest(npass, len(sub), 0.05, alternative="greater").pvalue), 5)
        wp = _wilcox(diff)
        out[f"wilcoxon_p_{key}"] = round(float(wp), 5) if np.isfinite(wp) else None
        # leave-one-subject-out: worst-case (max) p — is the signal robust to dropping any one subject?
        lb, lw = [], []
        for j in range(len(sub)):
            s2 = sub[:j] + sub[j + 1:]
            if not s2:
                continue
            nj = sum(bool(r[passk]) for r in s2)
            dj = [r["real_median_abs_corr"] - r[medk] for r in s2]
            lb.append(binomtest(nj, len(s2), 0.05, alternative="greater").pvalue)
            lw.append(_wilcox(dj))
        out[f"loso_binom_max_p_{key}"] = round(float(max(lb)), 5) if lb else None
        out[f"loso_wilcoxon_max_p_{key}"] = round(float(max(lw)), 5) if lw else None
    return out


if __name__ == "__main__":
    main()
