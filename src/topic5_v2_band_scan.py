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
