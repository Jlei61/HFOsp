"""Topic 5 V2 — multi-band masked band-power cache + subject-fixed analysis mask (Task 6).

Data lynchpin for the Phase 1 frequency-scan backbone: every downstream alignment / null /
gate reads this cache. For each subject we iterate the SAME eligible seizure windows as the
committed ``ictal_field_long_cache`` (via ``iter_subject_seizure_windows`` — identical
eligibility / ``pre_sec`` / ``post_sec`` / baseline-resolution inputs) and, for every config
band, compute:

  masked band-power trace (line-noise harmonics excluded at FFT-bin level, half-open for
  primary bands) -> baseline-robust-z (SAME baseline resolution as the legacy long cache:
  ``resolve_baseline_window`` + ``baseline_robust_z`` with GUARD_SEC / MIN_BASELINE_SEC) ->
  per-channel artifact flags (flatline / saturation -> bad_channel).

Outputs, per band ``B`` and seizure ``idx``:
  - npz: ``{B}__zt__{idx}`` (n_ch, n_bins float32 robust-z), ``{B}__relt__{idx}``
    (per-bin time rel clinical onset = spectrogram t - pre_sec), plus ``channels``.
  - sidecar JSON: per ``(B, idx)`` ``{eff_frac, fs_edge_flag, n_band_bins,
    low_baseline_channels, bad_channels}`` PLUS a subject-level ``analysis_channels``.

``analysis_channels`` (SCIENCE CONTRACT, issue #8): the channel NAMES that are VALID — finite
in z AND not low-baseline — across ALL PRIMARY bands. I.e. the intersection, over every (primary
band x seizure) that was computed, of {channel is finite in z AND not flagged low_baseline}. Only
genuine "can't compute z" validity failures (flatline / degenerate baseline) shrink this fixed
mask; SATURATION is deliberately NOT used here. The saturation flag (|z|>12 for >2% of bins)
fires on genuinely-seizing high-ripple channels — band-power-z legitimately spikes during a
seizure — so excluding it would be circular: it would drop the ictal signal we want to measure,
and z is baseline-self-normalized so magnitude can't separate seizure from amplifier clipping.
``bad_channels`` (flatline|saturation) stays in the sidecar as a downstream sensitivity
diagnostic; it just no longer drives this fixed mask. This mask is what downstream PRIMARY
metrics use (band-wise mask is sensitivity-only). Nyquist-unavailable primary bands (512 Hz
subjects can still do hi<256; 256 Hz subjects drop hi>=128) do not constrain the intersection.

Bands come from ``config/topic5_v2_phase1.yaml`` via ``load_phase1_config``: 7 primary
(HALF-OPEN [lo,hi)) + 4 composites (CLOSED [lo,hi]) + the legacy reproduction band
``legacy_bb_1_45`` (CLOSED; edges = the legacy ``BROAD_BAND`` constant, source of truth). A
band that trips the Nyquist gate for a subject is recorded and skipped, never crashed.

Cache is substrate-independent (the long ictal window is substrate-independent) so it is
written to ``.../v2_band_scan/cache/{ds_sid}.npz`` with NO axis_set in the path; ``--substrate``
only selects the default subject cohort. Reuses ``iter_subject_seizure_windows`` (Task 6a),
``src.topic5_v2_band_scan`` pure math, and ``resolve_baseline_window`` — no reinvented I/O.

Plan: docs/superpowers/plans/2026-07-01-topic5-v2-phase1-band-scan-backbone.md Task 6.
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
warnings.filterwarnings("ignore", message="invalid value encountered")

from scripts.build_topic5_ictal_field_long_cache import (  # noqa: E402
    iter_subject_seizure_windows, HOP, GUARD_SEC, MIN_BASELINE_SEC)
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from scripts.run_topic5_t0_eligibility import BROAD_BAND  # noqa: E402  (legacy bb edges, source of truth)
from src import topic5_ictal_recruitment as recruit  # noqa: E402
from src.topic5_ictal_recruitment import _spectrogram_on_hop  # noqa: E402
from src.ictal_onset_extraction import resolve_baseline_window  # noqa: E402
from src.topic5_v2_band_scan import (  # noqa: E402
    load_phase1_config, line_noise_bin_mask, band_bin_selection,
    robust_z_with_flags, channel_artifact_flags)

OUT_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"


def build_band_specs(cfg, selected=None):
    """Ordered list of (name, lo, hi, half_open, role) + the set of primary band names.

    role in {"primary","composite","repro"}. Primary -> HALF-OPEN [lo,hi); composite/repro ->
    CLOSED [lo,hi] (half_open = name in primary, per the plan). ``selected`` (from ``--bands``)
    filters by name; a requested-but-unknown name is warned and dropped."""
    primary = cfg["bands"]["primary"]
    composites = cfg["bands"]["composites"]
    primary_names = {row[0] for row in primary}
    specs = []
    for name, lo, hi in primary:
        specs.append((name, float(lo), float(hi), True, "primary"))
    for name, lo, hi in composites:
        specs.append((name, float(lo), float(hi), False, "composite"))
    # legacy reproduction band (closed); edges pinned to the legacy BROAD_BAND constant so a
    # v2-recomputed legacy_bb matches the old pipeline's band definition exactly.
    bb_name = cfg["repro_bands"]["bb"]
    specs.append((bb_name, float(BROAD_BAND[0]), float(BROAD_BAND[1]), False, "repro"))
    if selected is not None:
        want = list(dict.fromkeys(selected))  # preserve order, dedupe
        known = {s[0] for s in specs}
        for name in want:
            if name not in known:
                print(f"  [warn] requested --band {name!r} is not a known config band; skipped", flush=True)
        specs = [s for s in specs if s[0] in set(want)]
    return specs, primary_names


def _names_where(flag, channels):
    return [channels[i] for i in np.flatnonzero(np.asarray(flag, bool))]


def build_subject(ds_sid, substrate, specs, primary_names, cfg, out_root):
    """Build the masked multi-band cache for one subject. Returns True if anything was written."""
    ln = cfg["line_noise"]
    art = cfg["artifact"]
    spec_win = float(cfg["power"]["spectrogram_win_sec"])
    spec_hop = float(cfg["power"]["spectrogram_hop_sec"])
    harmonics = list(ln["harmonics_hz"])
    halfwidth = float(ln["halfwidth_hz"])
    fs512_hi_safe = float(cfg["edge"]["fs512_hi_safe_hz"])
    sat_abs_z = float(art["saturation_abs_z"])
    sat_frac = float(art["saturation_frac"])
    flatline_eps = float(art["flatline_mad_eps"])

    dataset, sid = ds_sid.split("_", 1)
    arrays = {}
    bands_qc = {}                 # {band: {str(idx): qc_dict}}
    good_by_band_idx = {}         # {(band, idx): set(good channel names)}
    skipped_bands = {}            # {band: reason} (Nyquist / no-bins; subject-level)
    band_seizure_fail = []        # [(band, idx, reason)] robust-z degeneracies
    drops = []                    # loader-side + channel-consistency drops
    channels = None
    fs = None
    seizure_idxs = []

    for idx, sw, eeg_rel in iter_subject_seizure_windows(ds_sid, substrate, drops=drops):
        ch = [recruit.bipolar_alias_label(c) for c in sw.ch_names]
        if channels is None:
            channels, fs = ch, float(sw.fs)
        elif ch != channels:
            # store only seizures whose montage matches the fixed `channels` array so every
            # stored z row maps to `channels[i]` (name-based analysis mask stays consistent).
            drops.append({"idx": idx, "reason": f"chan_mismatch:{len(ch)}vs{len(channels)}"})
            continue
        seizure_idxs.append(idx)
        # Change A (perf): compute the spectrogram + line-noise bin mask ONCE per seizure — the
        # expensive part is the FFT over ~1000s of hop bins x n_ch — then integrate EVERY band
        # from that single Sxx. The old code called masked_band_power_trace once per band, and it
        # recomputed the full spectrogram internally each time (~12x redundant). _spectrogram_on_hop
        # is deterministic, so the per-band math below is byte-for-byte what masked_band_power_trace
        # did; stored z is unchanged (legacy_bb stays bit-identical to the long cache).
        try:
            f, t, Sxx = _spectrogram_on_hop(sw.signal, sw.fs, spec_win, spec_hop)
        except ValueError as e:                           # win_sec needs more samples than we have
            drops.append({"idx": idx, "reason": f"spectrogram:{e}"})
            continue
        line_mask = line_noise_bin_mask(f, harmonics, halfwidth)
        nyq = float(sw.fs) / 2.0
        relt = (np.asarray(t, float) - float(sw.pre_sec)).astype(np.float32)
        for name, lo, hi, half_open, _role in specs:
            if hi >= nyq:                                 # Nyquist gate (was inside masked_band_power_trace)
                skipped_bands.setdefault(name, f"band hi {hi} >= Nyquist {nyq} for fs={sw.fs}")
                continue
            bmask, eff_frac, n_band = band_bin_selection(f, lo, hi, line_mask, half_open=half_open)
            if not bmask.any():                           # band empty after line-noise mask
                skipped_bands.setdefault(name, f"no bins in ({lo},{hi}) after line mask")
                continue
            power = Sxx[:, bmask, :].sum(axis=1)
            logp = np.log(np.maximum(power, 1e-30))
            fs_edge_flag = bool(float(sw.fs) <= 512.0 and float(hi) > fs512_hi_safe)
            bl = resolve_baseline_window(logp.shape[1], hop_sec=spec_hop, pre_sec=sw.pre_sec,
                                         buffer_sec=GUARD_SEC, eeg_onset_rel_sec=eeg_rel,
                                         min_baseline_valid_sec=MIN_BASELINE_SEC)
            try:
                z, low = robust_z_with_flags(logp, (bl.start_idx, bl.end_idx), spec_hop,
                                             MIN_BASELINE_SEC)
            except ValueError as e:                       # degenerate baseline window
                band_seizure_fail.append((name, idx, str(e)))
                continue
            flags = channel_artifact_flags(logp, z, sat_abs_z, sat_frac, flatline_eps)
            bad = np.asarray(flags["bad_channel"], bool)
            sat = np.asarray(flags["saturation"], bool)   # bad minus low_baseline (disjoint from flatline)
            arrays[f"{name}__zt__{idx}"] = z.astype(np.float32)
            arrays[f"{name}__relt__{idx}"] = relt
            bands_qc.setdefault(name, {})[str(idx)] = {
                "eff_frac": float(eff_frac),
                "fs_edge_flag": fs_edge_flag,
                "n_band_bins": int(n_band),
                "low_baseline_channels": _names_where(low, channels),
                "bad_channels": _names_where(bad, channels),       # flatline|saturation (diagnostic only)
                "saturation_channels": _names_where(sat, channels),  # inspectable; not in the fixed mask
            }
            # Change B (science): analysis-mask good-set = VALIDITY only (finite AND not
            # low-baseline). Excluding saturation here would be circular (see the module docstring):
            # |z|>12 fires on genuinely-seizing high-ripple channels, which is the ictal signal we
            # want to keep. bad_channels above still records flatline|saturation for sensitivity.
            finite = np.any(np.isfinite(z), axis=1)
            good = finite & ~low                          # validity-only (issue #8; drop saturation)
            good_by_band_idx[(name, idx)] = {channels[i] for i in np.flatnonzero(good)}

    if channels is None or not arrays:
        print(f"  [{ds_sid}] nothing cached ({len(drops)} loader drops)", flush=True)
        return False

    # analysis_channels = intersection over (primary band x seizure) of VALID (finite, not
    # low-baseline) names. Degenerate smoke/dev case (no primary band in the build set): fall back
    # to intersecting over the bands that WERE built, recorded via analysis_channels_basis (never a
    # production path — full builds include all 7 primary bands).
    built_band_names = sorted({b for (b, _i) in good_by_band_idx})
    primary_built = [b for b in built_band_names if b in primary_names]
    basis = primary_built if primary_built else built_band_names
    basis_label = "primary_bands_validity" if primary_built else "all_built_bands_validity_fallback"
    analysis_set = None
    for (name, _idx), gset in good_by_band_idx.items():
        if name not in basis:
            continue
        analysis_set = set(gset) if analysis_set is None else (analysis_set & gset)
    analysis_set = analysis_set or set()
    analysis_channels = [c for c in channels if c in analysis_set]   # montage order, deterministic

    meta = {
        "dataset": dataset, "subject": sid, "ds_sid": ds_sid, "substrate": substrate,
        "fs": fs, "channels": channels, "n_channels": len(channels),
        "seizure_idxs": seizure_idxs,
        "spec_win_sec": spec_win, "spec_hop_sec": spec_hop, "hop_sec": HOP,
        "band_specs": {name: {"lo": lo, "hi": hi, "half_open": half_open, "role": role}
                       for name, lo, hi, half_open, role in specs},
        "primary_bands_built": primary_built,
        "skipped_bands": skipped_bands,                 # Nyquist / empty-after-mask (subject-level)
        "band_seizure_fail": [{"band": b, "idx": i, "reason": r} for b, i, r in band_seizure_fail],
        "bands": bands_qc,                              # per (B, idx) QC
        "analysis_channels": analysis_channels,         # SCIENCE CONTRACT (issue #8)
        "analysis_channels_basis": basis_label,
        "n_channels_dropped_by_fixed_mask": len(channels) - len(analysis_channels),
        "baseline": {"guard_sec": GUARD_SEC, "min_baseline_sec": MIN_BASELINE_SEC,
                     "note": "robust-z baseline=[-pre_sec, min(0,eeg_onset_rel)-guard] via "
                             "resolve_baseline_window; same convention as ictal_field_long_cache"},
        "drops": drops,
    }
    cache_dir = out_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    arrays["channels"] = np.array(channels)
    np.savez_compressed(cache_dir / f"{ds_sid}.npz", **arrays)
    json.dump(meta, open(cache_dir / f"{ds_sid}.json", "w"), indent=2, ensure_ascii=False)
    print(f"  [{ds_sid}] wrote {len(seizure_idxs)} sz x {len(bands_qc)} bands | "
          f"analysis_channels={len(analysis_channels)}/{len(channels)} ({basis_label}) | "
          f"skipped={list(skipped_bands)} | {len(drops)} drops", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="explicit subject list (default = SUBJECTS_BY_SUB[substrate])")
    ap.add_argument("--substrate", choices=list(SUBJECTS_BY_SUB), default="broad",
                    help="default subject cohort; the cache itself is substrate-independent")
    ap.add_argument("--bands", nargs="*", default=None,
                    help="subset of config band names to build (default = all 7 primary + 4 "
                         "composites + legacy_bb_1_45)")
    ap.add_argument("--outdir", default=None,
                    help="override output ROOT (default results/.../v2_band_scan); "
                         "writes {outdir}/cache/{ds_sid}.npz")
    ap.add_argument("--restart", action="store_true", help="rebuild even if npz exists")
    args = ap.parse_args()

    cfg = load_phase1_config()
    specs, primary_names = build_band_specs(cfg, selected=args.bands)
    out_root = Path(args.outdir) if args.outdir else OUT_ROOT
    subs = args.subjects or SUBJECTS_BY_SUB[args.substrate]
    print(f"[v2-band-cache] {len(subs)} subjects, {len(specs)} bands "
          f"({sum(r=='primary' for *_, r in specs)} primary) -> {out_root}/cache", flush=True)
    for ds_sid in subs:
        if (out_root / "cache" / f"{ds_sid}.npz").exists() and not args.restart:
            print(f"[cache] {ds_sid} exists, skip", flush=True)
            continue
        print(f"[cache] {ds_sid} ...", flush=True)
        try:
            build_subject(ds_sid, args.substrate, specs, primary_names, cfg, out_root)
        except Exception as e:
            print(f"  SUBJECT ERROR {type(e).__name__}: {e}", flush=True)
    print("V2 BAND CACHE DONE", flush=True)


if __name__ == "__main__":
    main()
