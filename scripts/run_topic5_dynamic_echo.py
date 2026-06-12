"""Topic 5 Stage-2b dynamic-pattern-echo runner (I/O + orchestration).

Spec: docs/superpowers/specs/2026-06-11-topic5-stage2b-dynamic-pattern-echo-design.md
Plan: docs/superpowers/plans/2026-06-11-topic5-stage2b-dynamic-pattern-echo.md

Reads the EXISTING Stage-2 cache (raw feature traces) — NO EDF reload for sentinel.
Computes the early-ictal dynamic echo families (§3): activation-intensity echo(t),
growth-slope echo(t), slope latency, ramp strength, region aggregate; with the
max-over-time null (§2.2). Staged gate: sentinel (MANUAL) -> per-subject -> cohort.

Reuse (don't reinvent): z-construction + cache I/O + masked-template loader live in the
Stage-2 runner (scripts/run_topic5_ictal_recruitment.py), imported as `stage2`. The
pure math lives in src.topic5_dynamic_echo (`dyn`). The plan named some of these helpers
`recruit.X` but they actually live in `stage2` — corrected here (same functions).
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
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

from src import topic5_dynamic_echo as dyn
from src import topic5_ictal_recruitment as recruit          # bipolar_alias_label, assert_channel_identity
from src.propagation_skeleton_geometry import parse_shaft
import scripts.run_topic5_ictal_recruitment as stage2          # cache I/O, _z_from_traces, masked template

OUT = Path("results/topic5_dynamic_echo")
HOP = 0.1
T0, T1 = 0.0, 10.0                  # primary early-ictal window (clinical-onset anchored)
WIDE = (-5.0, 15.0)                 # onset-uncertainty sensitivity (clinical-onset anchored)
MEAN_WIN = (0.0, 5.0)              # pre-registered confirmatory window (broadband echo_mean)
MIN_CH = 8
Z_MIN, DELTA_MIN = 2.0, 1.0
B_SENTINEL = 500
B = 2000
RNG_SEED = 20260611
DETREND = "rolling_median"
N_ANCHOR_BINS = 4
FUSED = ("line_length", "broadband", "hfa", "spectral_edge")
GATE_FEATURES = ("broadband", "line_length")   # max-null gate at sentinel (two canonical amplitude features)
NULL_MODES = ("channel", "within_shaft", "anchor_matched")

# Cached sentinels (built in Stage 2): epi 1146:2/5 + yuquan litengsheng:0.
SENTINELS = (("epilepsiae_1146", (2, 5)), ("yuquan_litengsheng", (0,)))


# ---------------------------------------------------------------------------
# montage + template alignment
# ---------------------------------------------------------------------------
def _ictal_montage_semantics(dataset):
    """Canonical montage-SEMANTICS of the ictal signal AFTER per-dataset aliasing — NOT
    the raw detection reference. yuquan ictal is bipolar then alias-left, which IS the
    template's 'bipolar_aliased_left' semantics; epi ictal is car. (P1 fix: passing the
    raw ICTAL_REFERENCE 'bipolar' would falsely raise on yuquan.)"""
    return "bipolar_aliased_left" if dataset == "yuquan" else "car"


def _aligned_template(ds_sid, ictal_channels, dataset, cluster):
    """Masked narrow template (Main-A) cluster -> per-channel rank over ictal_channels order
    (NaN where no template), via per-dataset alias. cluster in {0,1}."""
    tmpl = stage2._load_masked_template(ds_sid)
    if tmpl is None:
        return None, None
    recruit.assert_channel_identity(template_montage=tmpl["template_montage"],
                                    ictal_montage=_ictal_montage_semantics(dataset))
    t_ch = tmpl["channels"]
    t_rank = np.asarray(tmpl["templates"][cluster], float)
    name2rank = {c: r for c, r in zip(t_ch, t_rank) if np.isfinite(r)}
    trank = np.array([name2rank.get(recruit.bipolar_alias_label(c), np.nan)
                      for c in ictal_channels], float)
    return trank, tmpl


# ---------------------------------------------------------------------------
# common time grid + interpolation (spec §2: per-feature t_center differ by win/2)
# ---------------------------------------------------------------------------
def _common_grid(t0, t1):
    return np.arange(t0, t1 + HOP / 2, HOP)


def _value_on_grid(z_feat, win, pre_sec, t_grid):
    """Interpolate a feature's robust-z (n_ch, n_frames) onto the shared t-grid. Native
    frame center time = frame*hop - pre + win/2 (inverse of stage2._sec_to_frame). Channels
    with < 2 finite samples -> NaN row."""
    n_ch, n_fr = z_feat.shape
    t_center = np.arange(n_fr) * HOP - float(pre_sec) + float(win) / 2.0
    out = np.full((n_ch, t_grid.shape[0]), np.nan)
    for c in range(n_ch):
        fin = np.isfinite(z_feat[c])
        if int(fin.sum()) >= 2:
            out[c] = np.interp(t_grid, t_center[fin], z_feat[c][fin])
    return out


def _shaft_blocks(channels):
    """Within-shaft block ids (single-typed so np.unique never trips str-vs-None)."""
    return np.array([(parse_shaft(recruit.bipolar_alias_label(c))[0] or "__none__")
                     for c in channels], dtype=object)


def _anchor_blocks(activation_grid):
    """Anchor-matched bins = quartiles of per-channel MEAN activation (the magnitude
    confound: template-early may simply be more active overall). Invalid -> -1."""
    m = np.nanmean(activation_grid, axis=1)
    fin = np.isfinite(m)
    bins = np.full(len(m), -1, dtype=int)
    if int(fin.sum()) >= N_ANCHOR_BINS:
        qs = np.quantile(m[fin], np.linspace(0, 1, N_ANCHOR_BINS + 1)[1:-1])
        bins[np.where(fin)[0]] = np.digitize(m[fin], qs)
    return bins.astype(object)


# ---------------------------------------------------------------------------
# per-seizure dynamic echo
# ---------------------------------------------------------------------------
def compute_seizure_dynamic(raw, meta, idx, template_rank, *, window=(T0, T1), mean_win=MEAN_WIN):
    """One seizure -> per-feature dynamic echo (observed). Returns None if z fails.
    template_rank is over the ictal channel order (NaN = not in template)."""
    eeg_rel = meta["eeg_rel_by_idx"].get(str(idx))
    zres = stage2._z_from_traces(raw, meta["pre_sec"], eeg_rel, detrend=DETREND)
    if zres is None:
        return None
    z, _, _, avail = zres
    pre = meta["pre_sec"]
    t_grid = _common_grid(*window)
    grids, out = {}, {"t_axis": t_grid.tolist(), "features": {}}
    for k in [f for f in FUSED if f in avail] + (["er"] if "er" in z else []):
        grids[k] = _value_on_grid(z[k], stage2.FEATURE_WIN[k], pre, t_grid)
    for k, g in grids.items():
        act, dz = dyn.activation_and_slope(g, hop=HOP)
        e_act = dyn.echo_curve(template_rank, act, t_grid, kind="intensity",
                               min_ch=MIN_CH, mean_window=mean_win)
        e_slope = dyn.echo_curve(template_rank, dz, t_grid, kind="intensity",
                                 min_ch=MIN_CH, mean_window=mean_win)
        lat = dyn.slope_latencies(g, t_axis=t_grid, z_min=Z_MIN, delta_min=DELTA_MIN, hop=HOP)
        ramp = dyn.ramp_strength(g, t_axis=t_grid)
        out["features"][k] = {
            "echo_act": {"peak": e_act["echo_peak"], "t_peak": e_act["t_peak"],
                         "mean": e_act["echo_mean"], "curve": _nan_list(e_act["curve"])},
            "echo_slope": {"peak": e_slope["echo_peak"], "t_peak": e_slope["t_peak"],
                           "mean": e_slope["echo_mean"]},
            "latency_align": {
                "t50": dyn.align_score(template_rank, lat["t50_rise"], kind="latency", min_ch=MIN_CH),
                "t_peak": dyn.align_score(template_rank, lat["t_peak"], kind="latency", min_ch=MIN_CH)},
            "ramp_align": {
                "auc_0_2": dyn.align_score(template_rank, ramp["AUC"][(0, 2)], kind="intensity", min_ch=MIN_CH),
                "auc_2_5": dyn.align_score(template_rank, ramp["AUC"][(2, 5)], kind="intensity", min_ch=MIN_CH)},
        }
    out["_grids"] = grids          # kept in-memory for the null pass; stripped before JSON
    out["available"] = list(grids.keys())
    return out


def _nan_list(a):
    return [None if not np.isfinite(x) else round(float(x), 4) for x in np.asarray(a, float)]


# ---------------------------------------------------------------------------
# sentinel
# ---------------------------------------------------------------------------
def _max_null_pvals(template_rank, activation_grid, channels, *, b, rng):
    """3-mode max-over-time null for one feature's activation echo_peak (§2.2).
    Returns {mode: {p, n_null}} for channel / within_shaft / anchor_matched."""
    obs = dyn.echo_curve(template_rank, activation_grid, _grid_t(activation_grid),
                         kind="intensity", min_ch=MIN_CH, mean_window=MEAN_WIN)["echo_peak"]
    blocks_by_mode = {"channel": None,
                      "within_shaft": _shaft_blocks(channels),
                      "anchor_matched": _anchor_blocks(activation_grid)}
    res = {"echo_peak": _f(obs)}
    for mode in NULL_MODES:
        null = dyn.echo_curve_null(template_rank, activation_grid, _grid_t(activation_grid),
                                   kind="intensity", min_ch=MIN_CH, null_mode=mode,
                                   blocks=blocks_by_mode[mode], B=b, rng=rng)
        res[mode] = {"p": _f(dyn.echo_peak_pvalue(obs, null)), "n_null": int(null.size)}
    return res


def _grid_t(activation_grid):
    return np.arange(activation_grid.shape[1]) * HOP + T0


def _f(x):
    x = float(x)
    return None if not np.isfinite(x) else round(x, 4)


def cmd_sentinel(args):
    OUT.mkdir(parents=True, exist_ok=True)
    sent_dir = OUT / "sentinel"
    sent_dir.mkdir(parents=True, exist_ok=True)
    b = args.B if args.B else B_SENTINEL
    rng = np.random.default_rng(RNG_SEED)
    print(f"[sentinel] B={b} primary window [{T0},{T1}]s clinical-anchored; "
          f"gate features={GATE_FEATURES}", flush=True)
    for ds_sid, idxs in SENTINELS:
        loaded = stage2._load_cache(ds_sid)
        if loaded is None:
            print(f"  {ds_sid}: NO CACHE — skip", flush=True)
            continue
        raw_by_idx, meta = loaded
        dataset = meta["dataset"]
        channels = meta["channels"]
        trank0, tmpl = _aligned_template(ds_sid, channels, dataset, cluster=0)
        trank1, _ = _aligned_template(ds_sid, channels, dataset, cluster=1)
        if trank0 is None:
            print(f"  {ds_sid}: NO TEMPLATE — skip", flush=True)
            continue
        n_common = int(np.isfinite(trank0).sum())
        print(f"  {ds_sid} ds={dataset} nchan={len(channels)} template_common={n_common} "
              f"swap={tmpl['swap_class']}", flush=True)
        for idx in idxs:
            if idx not in raw_by_idx:
                print(f"    sz{idx}: not cached — skip", flush=True)
                continue
            dyn_obs = compute_seizure_dynamic(raw_by_idx[idx], meta, idx, trank0,
                                              window=(T0, T1))
            if dyn_obs is None:
                print(f"    sz{idx}: z failed — skip", flush=True)
                continue
            grids = dyn_obs.pop("_grids")
            # primary-window max-null on the two canonical amplitude features (cluster 0)
            gate = {}
            for k in GATE_FEATURES:
                if k in grids:
                    act, _ = dyn.activation_and_slope(grids[k], hop=HOP)
                    gate[k] = _max_null_pvals(trank0, act, channels, b=b, rng=rng)
            # cluster-1 (swap template) confirmatory mean + wide-window sensitivity (observed)
            c1 = compute_seizure_dynamic(raw_by_idx[idx], meta, idx, trank1, window=(T0, T1))
            wide = compute_seizure_dynamic(raw_by_idx[idx], meta, idx, trank0, window=WIDE)
            rec = {
                "ds_sid": ds_sid, "dataset": dataset, "idx": idx,
                "n_channels": len(channels), "template_common": n_common,
                "swap_class": tmpl["swap_class"], "B": b,
                "primary_window": [T0, T1], "mean_window": list(MEAN_WIN),
                "features": dyn_obs["features"], "t_axis": dyn_obs["t_axis"],
                "gate_maxnull": gate,
                "confirmatory_broadband_echo_mean": _broadband_mean(dyn_obs),
                "cluster1_broadband_echo_mean": _broadband_mean(c1) if c1 else None,
                "sensitivity_wide_broadband_echo_mean": _broadband_mean(wide) if wide else None,
                "er_echo_mean": _feat_mean(dyn_obs, "er"),
            }
            (sent_dir / f"{ds_sid}_{idx}.json").write_text(json.dumps(rec, indent=2))
            _print_seizure(rec)
    print("SENTINEL DONE.", flush=True)


def _broadband_mean(obs):
    return _feat_mean(obs, "broadband")


def _feat_mean(obs, feat):
    f = obs["features"].get(feat) if obs else None
    return f["echo_act"]["mean"] if f else None


def _print_seizure(rec):
    print(f"    --- {rec['ds_sid']} sz{rec['idx']} (common={rec['template_common']}) ---", flush=True)
    for k in FUSED:
        f = rec["features"].get(k)
        if not f:
            continue
        ea = f["echo_act"]
        print(f"      {k:13s} echo_act peak={_s(ea['peak'])} @t={_s(ea['t_peak'])}s "
              f"mean[0,5]={_s(ea['mean'])}  lat50={_s(f['latency_align']['t50'])} "
              f"ramp02={_s(f['ramp_align']['auc_0_2'])}", flush=True)
    for k, g in rec["gate_maxnull"].items():
        ch = g.get("channel", {}); sh = g.get("within_shaft", {}); an = g.get("anchor_matched", {})
        print(f"      GATE {k:11s} peak={_s(g.get('echo_peak'))}  p_channel={_s(ch.get('p'))} "
              f"p_shaft={_s(sh.get('p'))} p_anchor={_s(an.get('p'))}", flush=True)
    print(f"      confirmatory broadband echo_mean[0,5]={_s(rec['confirmatory_broadband_echo_mean'])}  "
          f"cluster1={_s(rec['cluster1_broadband_echo_mean'])}  "
          f"wide[-5,15]={_s(rec['sensitivity_wide_broadband_echo_mean'])}  "
          f"er={_s(rec['er_echo_mean'])}", flush=True)


def _s(x):
    return "  nan" if x is None else f"{x:+.3f}"


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Topic 5 Stage-2b dynamic-pattern-echo runner")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sentinel", help="dynamic echo + max-null on cached sentinels (MANUAL GATE)")
    s.add_argument("--B", type=int, default=0, help="null draws (default B_SENTINEL=500)")
    s.set_defaults(func=cmd_sentinel)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
