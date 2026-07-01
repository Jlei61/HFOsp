# src/topic5_v2_band_scan.py
from __future__ import annotations
from pathlib import Path
import numpy as np, yaml
_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CFG = _ROOT / "config/topic5_v2_phase1.yaml"
def load_phase1_config(path=None) -> dict:
    with open(path or _DEFAULT_CFG) as fh:
        return yaml.safe_load(fh)


def line_noise_bin_mask(freqs, harmonics_hz, halfwidth_hz):
    freqs = np.asarray(freqs, float); m = np.zeros(freqs.shape, bool)
    for h in harmonics_hz: m |= np.abs(freqs - float(h)) <= float(halfwidth_hz)
    return m
def band_bin_selection(freqs, lo, hi, line_mask, half_open=False):
    freqs = np.asarray(freqs, float)
    in_band = (freqs >= float(lo)) & ((freqs < float(hi)) if half_open else (freqs <= float(hi)))
    n_band = int(in_band.sum())
    band_mask = in_band & ~np.asarray(line_mask, bool)
    return band_mask, float(band_mask.sum()) / max(n_band, 1), n_band


def masked_band_power_trace(signal, fs, lo, hi, spec_win_sec, spec_hop_sec,
                            harmonics_hz, halfwidth_hz, fs512_hi_safe, half_open=False):
    from src.topic5_ictal_recruitment import _spectrogram_on_hop
    nyq=float(fs)/2.0
    if hi>=nyq: raise ValueError(f"band hi {hi} >= Nyquist {nyq} for fs={fs}")
    f,t,Sxx=_spectrogram_on_hop(signal, fs, spec_win_sec, spec_hop_sec)
    lm=line_noise_bin_mask(f, harmonics_hz, halfwidth_hz)
    bmask,eff,n_band=band_bin_selection(f, lo, hi, lm, half_open=half_open)
    if not bmask.any(): raise ValueError(f"no bins in ({lo},{hi}) after line mask")
    power=Sxx[:,bmask,:].sum(axis=1)
    return {"logpower":np.log(np.maximum(power,1e-30)),"t":t,"eff_frac":eff,
            "fs_edge_flag":bool(float(fs)<=512.0 and float(hi)>float(fs512_hi_safe)),"n_band_bins":n_band}
def robust_z_with_flags(logpower, baseline_idx, hop_sec, min_baseline_valid_sec):
    from src.topic5_ictal_recruitment import baseline_robust_z
    z=baseline_robust_z(logpower, baseline_idx, hop_sec=hop_sec, min_baseline_valid_sec=min_baseline_valid_sec)
    return z, np.all(~np.isfinite(z), axis=1)
def channel_artifact_flags(logpower, z, sat_abs_z, sat_frac, flatline_mad_eps):
    z=np.asarray(z,float); flat=np.all(~np.isfinite(z),axis=1)
    with np.errstate(invalid="ignore"):
        sat=np.nanmean(np.abs(z)>float(sat_abs_z), axis=1) > float(sat_frac)
    sat=np.where(np.isfinite(sat), sat, False)
    bad=flat|sat
    return {"flatline":flat, "saturation":sat, "bad_channel":bad}


def contact_alignment(vals_by_name, rank_a_by_name, rank_b_by_name, oriented_template="a"):
    from scipy.stats import spearmanr, pearsonr
    def _one(rank_by):
        names=[n for n in vals_by_name if n in rank_by
               and np.isfinite(vals_by_name[n]) and np.isfinite(rank_by[n])]
        if len(names)<4: return None
        v=np.array([vals_by_name[n] for n in names]); r=np.array([rank_by[n] for n in names])
        if np.std(v)==0 or np.std(r)==0: return None
        return {"sp":float(spearmanr(v,r).statistic),"pe":float(pearsonr(v,r)[0]),"n":len(names)}
    a,b=_one(rank_a_by_name),_one(rank_b_by_name)
    def g(o,k,d=float("nan")): return o[k] if o else d
    # tie-break: on an exact |sp| tie, prefer the larger raw signed value —
    # a post-hoc cherry-picker reports whichever template looks more
    # confirmatory, not whichever was evaluated first.
    posthoc=max([o for o in (a,b) if o], key=lambda o:(abs(o["sp"]),o["sp"]), default=None)
    return {"signed_pearson_a":g(a,"pe"),"signed_spearman_a":g(a,"sp"),
            "signed_pearson_b":g(b,"pe"),"signed_spearman_b":g(b,"sp"),
            "align_signed_oriented":(g(a,"sp") if oriented_template=="a" else g(b,"sp")),
            "align_signed_posthoc_max":(posthoc["sp"] if posthoc else float("nan")),
            "align_abs_maxab_contact":max([abs(o["sp"]) for o in (a,b) if o], default=float("nan")),
            "n_contacts_a":g(a,"n",0),"n_contacts_b":g(b,"n",0)}
