#!/usr/bin/env python3
"""Fig3-B maxAB scaffold similarity — all-contact channel-shuffle null (per subject).

Adds a within-subject ALL-CONTACT channel-shuffle null to the Fig3-B peri-onset
readout, for the maxAB scaffold-similarity panel ONLY. Not tested here (out of
scope): onset increment, signed A/B fixed-direction polarity, multi-band.

Two load-bearing caveats (see docstring / summary / README):
  * This is the WEAKEST spatial null. Permuting values across every contact rules
    out "any random reassignment of these energy values across all contacts", but
    NOT shaft-level implantation geometry or same-shaft local smoothing. Stronger
    controls (within_shaft_shuffle / spatial_constrained_permute) are NOT run here.
  * Per-window p is POINTWISE and uncorrected. The 66 windows (10 s, 2 s step)
    overlap ~80%; there is no maxT / cluster correction. The top significance
    markers say "which windows pointwise exceed null", not "this interval is
    significant after time-dimension correction".

Observed, per subject and per common window center t:

    O_s(t) = median over that subject's seizures of  max(|r_A(t)|, |r_B(t)|)

Spatial null (per subject; the FULL readout is replayed, never the finished
maxAB): for each seizure and window we take that window's per-channel robust-z
energy vector, permute the values across ALL matched contacts with
``channel_shuffle`` (positions / support / templates / field smoothing / maxAB
selection all unchanged — only the value<->position correspondence is broken),
re-run make_field_record -> support-weighted smoothing -> corr to templates A
and B -> max(|r_A|,|r_B|), then take the median over the SAME seizures. Each
seizure is permuted independently; seizures/windows are never pooled as
independent samples. One-sided per-window p:  (1 + #{r: O^r(t) >= O_s(t)}) / (R+1).

The support-weighted field is linear in the per-channel values with a
value-independent support gate, so a spatial permutation is a matmul; the
vectorised readout below is verified equal to the exact ``score()`` path to
machine precision (``--verify``). Feature = line-noise-notch-filtered 1-150 Hz
summed spectrogram log power, per-channel baseline robust-z (identical to the
observed Fig3-B readout; NO extra FFT-bin line mask).

Tier: single-subject material for Fig3-B. NOT a formal cohort statistic.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.compute_topic5_signed_broadband_similarity import (  # noqa: E402
    _compute_values,
    _load_axis,
    _nan,
    _scorer,
)
from scripts.plot_topic5_signed_broadband_similarity_timecourse import (  # noqa: E402
    _eligible_idxs,
    _on_common_grid,
)
from src.propagation_contact_plane_readout import (  # noqa: E402
    OVERLAP_MIN,
    R_smooth_rank,
    S_THRESH,
    make_plane_grid,
)
from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.topic5_axis_alignment import (  # noqa: E402
    channel_shuffle,
    make_field_record,
    within_shaft_shuffle,
)

try:
    from src.plot_style import FS_LABEL, FS_TICK, savefig_pub, style_panel  # noqa: E402
    _HAVE_PLOT_STYLE = True
except Exception:  # pragma: no cover - styling is best-effort
    _HAVE_PLOT_STYLE = False

OUT_DIR = _ROOT / "results/paper-ready-figure/fig3_peri_onset_field_similarity/spatial_null"
FIG_DIR = OUT_DIR / "figures"

# Locked Fig3-B window contract (docs/figure_style_guide.md 5a).
START_SEC, STOP_SEC = -120.0, 20.0
WINDOW_SEC, STEP_SEC = 10.0, 2.0
BAND = (1.0, 150.0)

COL_OBS = "#A35E48"   # maxAB rust, matches Fig3-B panel a
COL_WS = "#3E6D9C"    # within-shaft null (primary, stronger control)
COL_AC = "#9AA0A6"    # all-contact null (weaker reference)
SIG_ALPHA = 0.05

# Static descriptive text shared by compute + rebuild paths (single source of truth).
_SUMMARY_TEXT = {
    "tier": "single-subject material for Fig3-B; NOT a formal cohort statistic",
    "readout": "maxAB scaffold similarity = max(|r_A|, |r_B|); onset-increment / signed A/B / multiband NOT tested",
    "feature": "1-150 Hz summed spectrogram log power (notch-filtered input at 50/100/150/200 Hz; "
               "NO extra FFT-bin line mask), per-channel baseline robust-z",
    "nulls_def": {
        "all_contact": "channel_shuffle — permute per-channel energy values across EVERY matched contact. "
                       "WEAKEST spatial control: does NOT hold shaft-level implantation geometry / "
                       "same-shaft local smoothing fixed.",
        "within_shaft": "within_shaft_shuffle — permute values only WITHIN each electrode shaft (preserves "
                        "which shaft is hot / shaft geometry; the STRONGER, PRIMARY control). Power depends "
                        "on shaft sizes (see shaft_structure); singleton shafts cannot be shuffled, so a "
                        "subject whose similarity is carried entirely by which shaft is hot will not clear it.",
    },
    "null_construction": "within-subject; full readout replayed per shuffle (values -> make_field_record -> "
                         "support-weighted smoothing -> corr to A/B -> max|r|), median over seizures, "
                         "independent permutation per seizure. NOT a shuffle of the finished maxAB.",
    "corrections": {
        "pointwise_p": "one-sided per window (1 + #{null >= obs}) / (R + 1); UNCORRECTED for the 66 "
                       "overlapping (10 s / 2 s) windows.",
        "maxt_p": "Nichols-Holmes single-step maxT across windows on standardized z (family-wise per-window "
                  "control, one-sided upper).",
        "cluster": "Maris-Oostenveld cluster permutation (cluster-forming = pointwise p<0.05 on standardized "
                   "z; cluster mass = sum z; null = max cluster mass per permutation). This is the "
                   "paper-facing 'significant time interval' verdict.",
    },
    "scope": "exploratory per-subject material; two WITHIN-subject spatial nulls, NOT a formal cohort spatial gate.",
}


# --------------------------------------------------------------------------
# Vectorised readout engine (exact-equivalent to compute script `score()`).
# --------------------------------------------------------------------------
def _pearson_cols(t: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Pearson r of vector `t` (n,) against each column of `M` (n, R)."""
    tc = t - t.mean()
    tnorm = float(np.sqrt((tc * tc).sum()))
    Mc = M - M.mean(axis=0, keepdims=True)
    Mnorm = np.sqrt((Mc * Mc).sum(axis=0))
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = (tc @ Mc) / (tnorm * Mnorm)
    corr = np.asarray(corr, float)
    corr[(tnorm < 1e-12) | (Mnorm < 1e-12)] = np.nan
    return corr


def build_engine(matched: list[dict], ds_sid: str) -> dict:
    """Precompute the fixed support-weighted kernel + template fields + corr
    masks so that ``maxab_batch`` reproduces the exact readout for any per-channel
    value vector. Templates and ictal share `matched`, hence one support field S."""
    X, Y = make_plane_grid()
    gx = X.ravel(); gy = Y.ravel()
    n = X.shape[0]
    pts = np.array([[c["x_norm"], c["y_norm"]] for c in matched], float)
    sup = np.array([c["support"] for c in matched], float)

    # sigma = template A auto bandwidth (median NN spacing), reused everywhere.
    rank_a = np.array([float(c["typical_rank"]) for c in matched], float)
    F_a = R_smooth_rank(make_field_record(matched, rank_a), X, Y, None, S_THRESH)
    sigma = float(F_a["sigma_xy"])
    sig2 = 2.0 * sigma ** 2

    d2 = (gx[:, None] - pts[None, :, 0]) ** 2 + (gy[:, None] - pts[None, :, 1]) ** 2
    W = sup[None, :] * np.exp(-d2 / sig2)          # (n_grid, n_contact)
    S = W.sum(axis=1)                              # (n_grid,)  == smooth_field S

    names = [c["name"] for c in matched]
    fields = {"A": (F_a["T"].ravel(), F_a["S"].ravel())}
    axis_b = _load_axis(ds_sid, "t_b")
    if axis_b is not None:
        b_rank = {c["name"]: float(c["typical_rank"]) for c in axis_b.get("channels", [])
                  if np.isfinite(c.get("typical_rank", np.nan))}
        rb = np.array([b_rank.get(nm, np.nan) for nm in names], float)
        if np.isfinite(rb).sum() >= 4:
            F_b = R_smooth_rank(make_field_record(matched, rb), X, Y, sigma, S_THRESH)
            fields["B"] = (F_b["T"].ravel(), F_b["S"].ravel())

    flip_idx = np.flip(np.arange(n * n).reshape(n, n), axis=0).ravel()
    Sict_m = S[flip_idx]

    # Per-template candidate pixel indices (fixed: ictal support S is value-invariant).
    templates = {}
    used = []
    for key, (Ft, St) in fields.items():
        idx_id = np.where((St >= S_THRESH) & (S >= S_THRESH) & np.isfinite(Ft))[0]
        idx_mir = np.where((St >= S_THRESH) & (Sict_m >= S_THRESH) & np.isfinite(Ft))[0]
        idx_mir_ict = flip_idx[idx_mir]            # ictal pixels feeding the mirror candidate
        templates[key] = {"t_id": Ft[idx_id], "g_id": idx_id, "n_id": idx_id.size,
                          "t_mir": Ft[idx_mir], "g_mir_ict": idx_mir_ict, "n_mir": idx_mir.size}
        used.extend([idx_id, idx_mir_ict])

    # Restrict the smoothed field to the union of pixels any candidate reads.
    needed = np.unique(np.concatenate(used)) if used else np.arange(0)
    for tm in templates.values():
        tm["l_id"] = np.searchsorted(needed, tm["g_id"])
        tm["l_mir"] = np.searchsorted(needed, tm["g_mir_ict"])
    return {"W_need": W[needed], "S_need": S[needed], "templates": templates, "sigma": sigma}


def maxab_batch(engine: dict, vals_matrix: np.ndarray) -> np.ndarray:
    """maxAB = max(|r_A|,|r_B|) for each row of `vals_matrix` (R, n_contact)."""
    W = engine["W_need"]; S = engine["S_need"]
    with np.errstate(invalid="ignore", divide="ignore"):
        T = (W @ vals_matrix.T) / S[:, None]       # (n_need, R)
    Rn = vals_matrix.shape[0]
    per_tmpl = []
    for tm in engine["templates"].values():
        cands = []
        if tm["n_id"] >= OVERLAP_MIN:
            cands.append(np.abs(_pearson_cols(tm["t_id"], T[tm["l_id"], :])))
        if tm["n_mir"] >= OVERLAP_MIN:
            cands.append(np.abs(_pearson_cols(tm["t_mir"], T[tm["l_mir"], :])))
        per_tmpl.append(np.nanmax(np.vstack(cands), axis=0) if cands else np.full(Rn, np.nan))
    return np.nanmax(np.vstack(per_tmpl), axis=0)


# --------------------------------------------------------------------------
# Null models + time-dimension multiple-comparison corrections.
# --------------------------------------------------------------------------
# Two within-subject nulls, weakest -> strongest:
#   all_contact  : channel_shuffle       — permute values across EVERY contact.
#   within_shaft : within_shaft_shuffle  — permute values only WITHIN each shaft
#                  (preserves shaft-level implantation geometry / which shaft is hot).
NULL_MODELS = {
    "all_contact": lambda vals, names, rng: channel_shuffle(vals, rng),
    "within_shaft": lambda vals, names, rng: within_shaft_shuffle(vals, names, rng),
}
NULL_LABELS = {"all_contact": "all-contact shuffle", "within_shaft": "within-shaft shuffle"}
ALPHA_CLUSTER = 0.05


def _shaft_structure(names: list[str]) -> dict:
    sizes: dict[str, int] = {}
    for n in names:
        sizes[parse_shaft(n)[0]] = sizes.get(parse_shaft(n)[0], 0) + 1
    vals = sorted(sizes.values(), reverse=True)
    return {"n_contacts": len(names), "n_shafts": len(sizes),
            "shaft_sizes": vals, "n_singleton_shafts": int(sum(1 for v in vals if v == 1))}


def _standardize(obs: np.ndarray, null: np.ndarray):
    """obs:(W,), null:(R,W). Per-window z by the null mean/std (one-sided upper convention)."""
    mu = null.mean(axis=0)
    sd = null.std(axis=0, ddof=1)
    sd = np.where(sd < 1e-12, np.nan, sd)
    return (obs - mu) / sd, (null - mu) / sd


def _pointwise_p(obs: np.ndarray, null: np.ndarray) -> np.ndarray:
    R = null.shape[0]
    return np.array([(1 + int(np.sum(null[:, t] >= obs[t]))) / (R + 1) for t in range(obs.size)])


def _maxt_p(z_obs: np.ndarray, z_null: np.ndarray) -> np.ndarray:
    """Nichols-Holmes single-step maxT, one-sided upper. Corrected p per window.
    Degenerate windows (z_obs NaN — e.g. a within-shaft null with no within-shaft variation)
    cannot be significant -> p=1; NaN null entries are excluded from the per-permutation max."""
    M = np.max(np.where(np.isfinite(z_null), z_null, -np.inf), axis=1)   # (R,)
    R = M.size
    p = (1 + (M[None, :] >= z_obs[:, None]).sum(axis=1)) / (R + 1)
    return np.where(np.isfinite(z_obs), p, 1.0)


def _runs_above(z: np.ndarray, z_thr: float):
    above = z > z_thr
    out, i, W = [], 0, z.size
    while i < W:
        if above[i]:
            j = i
            while j < W and above[j]:
                j += 1
            out.append((i, j, float(np.nansum(z[i:j]))))
            i = j
        else:
            i += 1
    return out


def _cluster_correction(z_obs: np.ndarray, z_null: np.ndarray, alpha: float = SIG_ALPHA):
    """Maris-Oostenveld cluster permutation, one-sided upper. cluster-forming threshold =
    pooled (1-ALPHA_CLUSTER) null-z quantile; mass = sum of standardized z over the run;
    null = max cluster mass per permutation. Returns (sig_mask, clusters, z_thr)."""
    z_thr = (float(np.nanpercentile(z_null, 100 * (1 - ALPHA_CLUSTER)))
             if np.isfinite(z_null).any() else np.inf)   # inf -> no runs on a degenerate null
    R = z_null.shape[0]
    null_max = np.array([max((m for _, _, m in _runs_above(z_null[r], z_thr)), default=0.0)
                         for r in range(R)])
    sig = np.zeros(z_obs.size, bool)
    clusters = []
    for i0, i1, mass in _runs_above(z_obs, z_thr):
        p = (1 + int(np.sum(null_max >= mass))) / (R + 1)
        clusters.append({"start_idx": int(i0), "end_idx": int(i1), "mass": float(mass), "p": float(p)})
        if p < alpha:
            sig[i0:i1] = True
    return sig, clusters, z_thr


# --------------------------------------------------------------------------
# Per-subject observed + null.
# --------------------------------------------------------------------------
def _keep_window(lo: float) -> bool:
    return (_on_common_grid(lo, start_sec=START_SEC, step_sec=STEP_SEC)
            and (lo >= START_SEC - 1e-9) and (lo + WINDOW_SEC <= STOP_SEC + 1e-9))


def _seizure_args(ds_sid: str, seizure_idx: int) -> SimpleNamespace:
    return SimpleNamespace(
        subject=ds_sid, seizure_idx=int(seizure_idx), start_sec=START_SEC, stop_sec=STOP_SEC,
        band_lo=BAND[0], band_hi=BAND[1], spectral_win_sec=1.0, hop_sec=0.5,
        smooth_sec=WINDOW_SEC, frame_step_sec=STEP_SEC, onset_win_sec=10.0, chunk_ch=16)


def compute_subject(ds_sid: str, n_perm: int, seed: int, verify: bool):
    """Return (per_window rows, summary dict, drops list)."""
    idxs = _eligible_idxs(ds_sid)
    rng = np.random.default_rng(seed)
    obs_by_win: dict[float, list[float]] = {}
    null_by_win = {key: {} for key in NULL_MODELS}   # model -> {lo -> [ (R,) per seizure ]}
    drops = []
    max_fid_err = 0.0
    n_seizures = 0
    shaft = None

    for seizure_idx in idxs:
        try:
            _ds, _i, sw, offset, bl, matched, names, starts, window_vals, _onset = \
                _compute_values(_seizure_args(ds_sid, seizure_idx))
        except Exception as exc:  # fail-closed per seizure
            drops.append({"seizure_idx": int(seizure_idx), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        engine = build_engine(matched, ds_sid)
        n_seizures += 1
        if shaft is None:
            shaft = _shaft_structure(names)
        for lo, vals in zip(starts, window_vals):
            if not _keep_window(float(lo)):
                continue
            vals = np.asarray(vals, float)
            if np.isfinite(vals).all():
                obs = float(maxab_batch(engine, vals[None, :])[0])
                if verify:
                    max_fid_err = max(max_fid_err, abs(obs - _exact_maxab(matched, ds_sid, vals)))
                for key, shuf in NULL_MODELS.items():
                    batch = np.vstack([shuf(vals, names, rng) for _ in range(n_perm)])
                    null_by_win[key].setdefault(float(lo), []).append(maxab_batch(engine, batch))
            else:
                # NaN in the window vector shifts the support gate under permutation; fall back
                # to the exact per-realization score() for this rare window (both null models).
                obs = _exact_maxab(matched, ds_sid, vals)
                for key, shuf in NULL_MODELS.items():
                    null_by_win[key].setdefault(float(lo), []).append(
                        np.array([_exact_maxab(matched, ds_sid, shuf(vals, names, rng))
                                  for _ in range(n_perm)], float))
            obs_by_win.setdefault(float(lo), []).append(obs)

    if n_seizures == 0:
        raise RuntimeError(f"{ds_sid}: no seizure produced windows ({len(drops)} dropped)")

    los = sorted(obs_by_win)
    centers = np.array([lo + WINDOW_SEC / 2.0 for lo in los])
    n_sz = np.array([len(obs_by_win[lo]) for lo in los])
    obs = np.array([float(np.nanmedian(obs_by_win[lo])) for lo in los])
    obs_q25 = np.array([float(np.nanpercentile(obs_by_win[lo], 25)) for lo in los])
    obs_q75 = np.array([float(np.nanpercentile(obs_by_win[lo], 75)) for lo in los])
    # per null model: (R, W) = median over seizures per realization
    null_mats = {}
    for key in NULL_MODELS:
        M = np.empty((n_perm, len(los)))
        for j, lo in enumerate(los):
            M[:, j] = np.nanmedian(np.vstack(null_by_win[key][lo]), axis=0)
        null_mats[key] = M

    meta = {"n_perm": int(n_perm), "seed": int(seed), "n_seizures": int(n_seizures),
            "n_seizure_drops": len(drops), "seizure_drops": drops, "shaft_structure": shaft,
            "fidelity_max_abs_err": float(max_fid_err) if verify else None}
    rows, summary = _finalize(ds_sid, centers, n_sz, obs, obs_q25, obs_q75, null_mats, meta)
    null_npz = {"window_center_sec": centers, "n_seizures": n_sz,
                "obs": obs, "obs_q25": obs_q25, "obs_q75": obs_q75,
                **{f"{key}_null": null_mats[key] for key in NULL_MODELS}}
    return rows, summary, drops, null_npz


def _finalize(ds_sid, centers, n_sz, obs, obs_q25, obs_q75, null_mats, meta):
    """Given per-window observed + per-null (R,W) matrices, apply pointwise / maxT / cluster
    corrections and build (per_window rows, summary). Shared by compute and rebuild paths."""
    W = obs.size
    per_null = {}
    for key, M in null_mats.items():
        z_obs, z_null = _standardize(obs, M)
        cl_sig, clusters, z_thr = _cluster_correction(z_obs, z_null)
        per_null[key] = {"M": M, "pw": _pointwise_p(obs, M), "mt": _maxt_p(z_obs, z_null),
                         "cl_sig": cl_sig, "clusters": clusters, "z_thr": z_thr}

    rows = []
    for j in range(W):
        row = {"window_start_sec": float(centers[j] - WINDOW_SEC / 2.0),
               "window_end_sec": float(centers[j] + WINDOW_SEC / 2.0),
               "window_center_sec": float(centers[j]), "n_seizures": int(n_sz[j]),
               "obs_median_maxAB": float(obs[j]), "obs_q25": float(obs_q25[j]),
               "obs_q75": float(obs_q75[j])}
        for key, d in per_null.items():
            col = d["M"][:, j]
            row[f"{key}_null_median"] = float(np.nanmedian(col))
            row[f"{key}_null_p2.5"] = float(np.nanpercentile(col, 2.5))
            row[f"{key}_null_p97.5"] = float(np.nanpercentile(col, 97.5))
            row[f"{key}_pointwise_p"] = float(d["pw"][j])
            row[f"{key}_maxt_p"] = float(d["mt"][j])
            row[f"{key}_cluster_sig"] = bool(d["cl_sig"][j])
        rows.append(row)

    def _counts(key):
        d = per_null[key]
        sig_cl = [{"start_sec": float(centers[c["start_idx"]]),
                   "end_sec": float(centers[c["end_idx"] - 1]),
                   "n_windows": int(c["end_idx"] - c["start_idx"]), "p": c["p"]}
                  for c in d["clusters"] if c["p"] < SIG_ALPHA]
        return {"n_pointwise_p05": int(np.sum(d["pw"] < SIG_ALPHA)),
                "n_maxt_p05": int(np.sum(d["mt"] < SIG_ALPHA)),
                "n_cluster_sig_windows": int(np.sum(d["cl_sig"])),
                "n_clusters": len(d["clusters"]), "significant_clusters": sig_cl}

    summary = {
        "subject": ds_sid, **_SUMMARY_TEXT,
        "time_range_sec": [START_SEC, STOP_SEC], "window_sec": WINDOW_SEC, "step_sec": STEP_SEC,
        "n_perm": meta["n_perm"], "seed": meta["seed"], "n_seizures": meta["n_seizures"],
        "n_seizure_drops": meta["n_seizure_drops"], "seizure_drops": meta["seizure_drops"],
        "shaft_structure": meta["shaft_structure"], "n_windows": W,
        "obs_median_of_window_medians": float(np.nanmedian(obs)),
        "primary_null": "within_shaft",
        "nulls": {key: _counts(key) for key in per_null},
        "fidelity_max_abs_err": meta.get("fidelity_max_abs_err"),
    }
    return rows, summary


def _exact_maxab(matched, ds_sid, vals) -> float:
    score = _scorer(ds_sid, matched)
    per, _best = score(np.asarray(vals, float))
    return max(_nan(per.get("A", {}).get("abs_corr")), _nan(per.get("B", {}).get("abs_corr")))


# --------------------------------------------------------------------------
# Figure + outputs.
# --------------------------------------------------------------------------
def _shade_runs(ax, x, mask, **kw):
    """axvspan over each contiguous True run of `mask` (window-centered)."""
    i, W = 0, len(mask)
    while i < W:
        if mask[i]:
            j = i
            while j < W and mask[j]:
                j += 1
            ax.axvspan(x[i] - STEP_SEC / 2, x[j - 1] + STEP_SEC / 2, **kw)
            i = j
        else:
            i += 1


def _plot(ds_sid: str, rows: list[dict], summary: dict, out_png: Path, out_pdf: Path) -> None:
    x = np.array([r["window_center_sec"] for r in rows], float)
    obs = np.array([r["obs_median_maxAB"] for r in rows], float)
    oq25 = np.array([r["obs_q25"] for r in rows], float)
    oq75 = np.array([r["obs_q75"] for r in rows], float)
    ws_med = np.array([r["within_shaft_null_median"] for r in rows], float)
    ws_lo = np.array([r["within_shaft_null_p2.5"] for r in rows], float)
    ws_hi = np.array([r["within_shaft_null_p97.5"] for r in rows], float)
    ac_med = np.array([r["all_contact_null_median"] for r in rows], float)
    ws_clu = np.array([r["within_shaft_cluster_sig"] for r in rows], bool)
    ws_maxt = np.array([r["within_shaft_maxt_p"] for r in rows], float) < SIG_ALPHA
    label = ds_sid.replace("epilepsiae_", "E").replace("yuquan_", "Y-")
    ws = summary["nulls"]["within_shaft"]
    ac = summary["nulls"]["all_contact"]

    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    # primary = within-shaft null band (stronger control)
    ax.fill_between(x, ws_lo, ws_hi, color=COL_WS, alpha=0.18, lw=0, zorder=1, label="within-shaft null 95%")
    ax.plot(x, ws_med, color=COL_WS, lw=1.3, ls="--", zorder=2, label="within-shaft null median")
    # weaker reference = all-contact null median only
    ax.plot(x, ac_med, color=COL_AC, lw=1.1, ls=":", zorder=2, label="all-contact null median")
    # observed
    ax.fill_between(x, oq25, oq75, color=COL_OBS, alpha=0.13, lw=0, zorder=3, label="observed IQR")
    ax.plot(x, obs, color=COL_OBS, lw=2.2, zorder=5, label="observed median")
    # cluster-corrected significant spans vs within-shaft (the "significant interval")
    if ws_clu.any():
        _shade_runs(ax, x, ws_clu, color=COL_OBS, alpha=0.11, lw=0, zorder=0)
        ax.axvspan(np.nan, np.nan, color=COL_OBS, alpha=0.11, lw=0, label="within-shaft cluster p<0.05")
    # maxT-significant windows vs within-shaft
    if ws_maxt.any():
        ax.scatter(x[ws_maxt], np.full(ws_maxt.sum(), 0.965), marker="v", s=16,
                   color=COL_WS, zorder=6, label="within-shaft maxT p<0.05")
    ax.axvline(0, color="0.30", ls="--", lw=0.9, zorder=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(float(x.min()), float(x.max()))
    ax.set_xlabel("window center from onset (s)", fontsize=FS_LABEL if _HAVE_PLOT_STYLE else 11)
    ax.set_ylabel("maxAB field similarity |r|", fontsize=FS_LABEL if _HAVE_PLOT_STYLE else 11)
    ax.set_title("maxAB scaffold similarity vs spatial-shuffle nulls",
                 fontsize=(FS_LABEL if _HAVE_PLOT_STYLE else 11), pad=16)
    # subtitle: within-shaft (primary) corrected counts + all-contact pointwise for context
    ax.text(0.5, 1.015,
            f"{label} · {summary['n_seizures']} sz · R={summary['n_perm']} · within-shaft: "
            f"cluster {ws['n_cluster_sig_windows']} / maxT {ws['n_maxt_p05']} / pointwise "
            f"{ws['n_pointwise_p05']} win · all-contact pointwise {ac['n_pointwise_p05']}",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=7.3, color="0.35")
    ax.legend(frameon=False, loc="lower left", fontsize=6.8, handlelength=1.5, ncol=1,
              labelspacing=0.3)
    if _HAVE_PLOT_STYLE:
        ax.tick_params(labelsize=FS_TICK - 2)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16, top=0.85)
    for out in (out_png, out_pdf):
        if _HAVE_PLOT_STYLE:
            savefig_pub(fig, out, dpi=300)
        else:
            fig.savefig(out, dpi=300)
    plt.close(fig)


def _write_readme() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    (FIG_DIR / "README.md").write_text(
        "# Fig3-B maxAB scaffold similarity — 空间置换 null（两档）+ 时间维校正\n\n"
        "### `<subject>_maxab_spatial_null.png / .pdf`\n\n"
        "在 Fig3-B 的 maxAB 面板上叠加**两个被试内空间置换 null + 时间维多重比较校正**。两个 null 都：保持"
        "同一批 seizure / 时间窗 / A|B 模板 / 场平滑 / maxAB 逻辑，只打乱每窗 per-channel 能量值，完整重跑读出、"
        "对 seizure 取中位（每次 seizure 独立置换）。两档强度：\n"
        "- **all-contact**（弱，灰点线=null 中位）：值在**所有触点**间打乱。\n"
        "- **within-shaft**（强，主对比，蓝带）：值只在**每根杆(shaft)内**打乱，保留'哪根杆热'的植入几何。\n\n"
        "图元：粗 rust=观测中位、浅 rust 带=观测 IQR；蓝虚线+蓝带=within-shaft null 中位+95%；灰点线=all-contact "
        "null 中位；浅 rust 竖带=within-shaft **cluster 校正显著区间**；蓝三角=within-shaft **maxT 校正显著窗**；"
        "0 s 虚线=onset。副标题给 within-shaft 的 cluster/maxT/pointwise 显著窗数 + all-contact pointwise。\n\n"
        "**三档显著性（都在 stats CSV，两个 null 各一套）**：pointwise（逐窗，未校正）< maxT（逐窗 FWER）< "
        "cluster（时间维、对持续抬升敏感，= paper-facing '显著区间'）。\n\n"
        "**关注点**：观测中位数是否**离开蓝色 within-shaft null 带**并形成 cluster 显著区间。⚠️within-shaft null "
        "的分辨力取决于每根杆的触点数（见 summary.shaft_structure）；单触点杆无法打乱，若相似度完全由'哪根杆热'"
        "解释，within-shaft null 就贴着观测、几乎无显著窗——这是诚实的强 null 结果。只检验 maxAB scaffold，"
        "不做 onset increment / signed A/B / 多频带。**单被试素材，非 formal cohort spatial gate。**\n",
        encoding="utf-8")


def _write_outputs(ds_sid, rows, summary, null_npz=None) -> dict:
    stats_csv = OUT_DIR / f"{ds_sid}_maxab_spatial_null_stats.csv"
    with stats_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    npz_fp = OUT_DIR / f"{ds_sid}_maxab_spatial_null_matrices.npz"
    if null_npz is not None:
        np.savez_compressed(npz_fp, **null_npz)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out_png = FIG_DIR / f"{ds_sid}_maxab_spatial_null.png"
    out_pdf = FIG_DIR / f"{ds_sid}_maxab_spatial_null.pdf"
    _plot(ds_sid, rows, summary, out_png, out_pdf)
    summary["outputs"] = {"figure_png": str(out_png.relative_to(_ROOT)),
                          "figure_pdf": str(out_pdf.relative_to(_ROOT)),
                          "stats_csv": str(stats_csv.relative_to(_ROOT)),
                          "null_matrices_npz": str(npz_fp.relative_to(_ROOT))}
    (OUT_DIR / f"{ds_sid}_maxab_spatial_null_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    _write_readme()
    return summary


def run_subject(ds_sid: str, n_perm: int, seed: int, verify: bool) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, summary, _drops, null_npz = compute_subject(ds_sid, n_perm, seed, verify)
    return _write_outputs(ds_sid, rows, summary, null_npz)


def rebuild_subject(ds_sid: str) -> dict:
    """Recompute corrections + re-render from the cached null-matrix .npz (obs + both null
    matrices), WITHOUT reloading raw data. Lets correction params / labels change cheaply."""
    npz_fp = OUT_DIR / f"{ds_sid}_maxab_spatial_null_matrices.npz"
    summ_fp = OUT_DIR / f"{ds_sid}_maxab_spatial_null_summary.json"
    if not (npz_fp.exists() and summ_fp.exists()):
        raise FileNotFoundError(f"{ds_sid}: cached .npz/summary missing; run once without --rebuild-from-stats")
    z = np.load(npz_fp)
    old = json.loads(summ_fp.read_text())
    null_mats = {key: z[f"{key}_null"] for key in NULL_MODELS}
    meta = {"n_perm": int(old["n_perm"]), "seed": int(old["seed"]), "n_seizures": int(old["n_seizures"]),
            "n_seizure_drops": int(old.get("n_seizure_drops", 0)), "seizure_drops": old.get("seizure_drops", []),
            "shaft_structure": old.get("shaft_structure"), "fidelity_max_abs_err": old.get("fidelity_max_abs_err")}
    rows, summary = _finalize(ds_sid, z["window_center_sec"], z["n_seizures"], z["obs"],
                              z["obs_q25"], z["obs_q75"], null_mats, meta)
    return _write_outputs(ds_sid, rows, summary, null_npz=None)


PAPER_INDEX = (_ROOT / "results/paper-ready-figure/fig3_peri_onset_field_similarity"
               / "fig3_peri_onset_subject_index.csv")
INDEX_CSV = OUT_DIR / "fig3b_maxab_spatial_null_index.csv"
INDEX_JSON = OUT_DIR / "fig3b_maxab_spatial_null_index.json"
INDEX_COLS = ["subject", "status", "drop_reason", "n_seizures", "n_windows",
              "n_shafts", "n_singleton_shafts",
              "ac_pointwise_sig", "ws_pointwise_sig", "ws_maxt_sig",
              "ws_cluster_sig_windows", "ws_n_sig_clusters",
              "obs_median_of_window_medians", "figure_png", "stats_csv"]


def _ok_subjects() -> list[str]:
    """Subjects that already produced an observed Fig3-B figure (paper index status==ok)."""
    if not PAPER_INDEX.exists():
        raise FileNotFoundError(f"paper index missing: {PAPER_INDEX}")
    with PAPER_INDEX.open() as fh:
        return [r["subject"] for r in csv.DictReader(fh) if r["status"] == "ok"]


def _record_from_summary(summ: dict) -> dict:
    ws = summ["nulls"]["within_shaft"]
    ac = summ["nulls"]["all_contact"]
    sh = summ.get("shaft_structure") or {}
    return {"subject": summ["subject"], "status": "ok", "drop_reason": "",
            "n_seizures": summ["n_seizures"], "n_windows": summ["n_windows"],
            "n_shafts": sh.get("n_shafts", ""), "n_singleton_shafts": sh.get("n_singleton_shafts", ""),
            "ac_pointwise_sig": ac["n_pointwise_p05"],
            "ws_pointwise_sig": ws["n_pointwise_p05"],
            "ws_maxt_sig": ws["n_maxt_p05"],
            "ws_cluster_sig_windows": ws["n_cluster_sig_windows"],
            "ws_n_sig_clusters": len(ws["significant_clusters"]),
            "obs_median_of_window_medians": summ["obs_median_of_window_medians"],
            "figure_png": summ["outputs"]["figure_png"],
            "stats_csv": summ["outputs"]["stats_csv"]}


def _sig_str(rec: dict) -> str:
    return (f"ws-cluster {rec['ws_cluster_sig_windows']}/maxT {rec['ws_maxt_sig']} · "
            f"ac-pw {rec['ac_pointwise_sig']} /{rec['n_windows']}")


def _write_cohort_index(records: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with INDEX_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_COLS, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow({c: rec.get(c, "") for c in INDEX_COLS})
    n_ok = sum(1 for r in records if r["status"] == "ok")
    INDEX_JSON.write_text(json.dumps({
        "figure": "Fig3-B maxAB scaffold similarity — spatial-shuffle nulls (all-contact + within-shaft) "
                  "with maxT / cluster time correction",
        "generated_by": "scripts/run_topic5_fig3b_maxab_spatial_null.py",
        "tier": "single-subject material for Fig3-B; NOT a formal cohort statistic",
        "primary_null": "within_shaft",
        "test": "two within-subject spatial nulls (all_contact channel-shuffle = weak; within_shaft = strong, "
                "primary); full readout replayed, median over seizures; per window pointwise / maxT / cluster "
                "one-sided p; maxAB scaffold only. Index columns: ws_* = within-shaft (primary), ac_* = all-contact.",
        "caveats": [
            "within-shaft (primary) is the stronger control but its power depends on shaft sizes "
            "(n_shafts / n_singleton_shafts); a subject whose similarity is carried by which-shaft-is-hot will "
            "not clear it — that is honest, not a failure. Neither null is a formal cohort spatial gate.",
            "ws_cluster_sig_windows = Maris-Oostenveld cluster-corrected (paper-facing 'significant interval'); "
            "ws_maxt_sig = Nichols-Holmes maxT; ws_pointwise_sig / ac_pointwise_sig = UNCORRECTED pointwise counts.",
            "per-subject material heterogeneity; NOT a cohort claim.",
        ],
        "n_subjects": len(records), "n_ok": n_ok, "n_drop": len(records) - n_ok,
        "subjects": records,
    }, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="epilepsiae_1146")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="batch: explicit subject list")
    ap.add_argument("--all-ok", action="store_true",
                    help="batch: every subject with an observed Fig3-B figure (paper index status==ok)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="batch: reuse a subject's summary JSON if present (resume without recompute)")
    ap.add_argument("--rebuild-from-stats", action="store_true",
                    help="batch: re-render figures + refresh summary text from cached stats CSVs (no recompute)")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verify", action="store_true",
                    help="assert the vectorised readout matches exact score() to 1e-9")
    args = ap.parse_args()

    if args.all_ok:
        subjects = _ok_subjects()
    elif args.subjects:
        subjects = args.subjects
    else:
        subjects = [args.subject]

    if len(subjects) == 1 and not args.all_ok and not args.subjects:
        t0 = time.time()
        summ = run_subject(subjects[0], args.n_perm, args.seed, args.verify)
        print(json.dumps({"subject": summ["subject"], "n_seizures": summ["n_seizures"],
                          "n_windows": summ["n_windows"], "shaft_structure": summ["shaft_structure"],
                          "obs_median_of_window_medians": summ["obs_median_of_window_medians"],
                          "nulls": summ["nulls"], "fidelity_max_abs_err": summ["fidelity_max_abs_err"]},
                         ensure_ascii=False, indent=2))
        print(f"[{subjects[0]}] done in {time.time()-t0:.1f}s -> {summ['outputs']['figure_png']}")
        return

    # fail-closed batch
    print(f"processing {len(subjects)} subject(s)", flush=True)
    records = []
    for i, ds_sid in enumerate(subjects, 1):
        t0 = time.time()
        print(f"[{i}/{len(subjects)}] {ds_sid} ...", flush=True)
        summ_fp = OUT_DIR / f"{ds_sid}_maxab_spatial_null_summary.json"
        if args.rebuild_from_stats:
            try:
                rec = _record_from_summary(rebuild_subject(ds_sid))
                print(f"    rebuilt {_sig_str(rec)}", flush=True)
            except Exception as exc:
                rec = {"subject": ds_sid, "status": "drop", "drop_reason": f"{type(exc).__name__}: {exc}"}
                print(f"    DROP {rec['drop_reason']}", flush=True)
        elif args.skip_existing and summ_fp.exists() and "nulls" in json.loads(summ_fp.read_text()):
            rec = _record_from_summary(json.loads(summ_fp.read_text()))
            print(f"    skip (exists) {_sig_str(rec)}", flush=True)
        else:
            try:
                summ = run_subject(ds_sid, args.n_perm, args.seed, args.verify)
                rec = _record_from_summary(summ)
                print(f"    ok  {_sig_str(rec)} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as exc:
                rec = {"subject": ds_sid, "status": "drop", "drop_reason": f"{type(exc).__name__}: {exc}"}
                print(f"    DROP {rec['drop_reason']}", flush=True)
        records.append(rec)
        _write_cohort_index(records)
    n_ok = sum(1 for r in records if r["status"] == "ok")
    print(f"\nDONE: {n_ok}/{len(records)} ok -> {INDEX_CSV}", flush=True)


if __name__ == "__main__":
    main()
