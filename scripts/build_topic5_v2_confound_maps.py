"""Topic 5 V2 Phase 1 -- Task 12a: per-contact CONFOUND covariate maps.

These per-subject maps let downstream (Task 12/13) residualize the producer geometry G_HFO
(``typical_rank``) against confounds so an alignment claim is about the TIMING GEOMETRY of the
interictal propagation, NOT about HFO-rate / baseline-power / anatomical topography. For each
subject we write a channel-name-keyed map per confound:

  hfo_rate            interictal HFO event count per channel = ``events_bool.sum(over events)``
                      from ``load_subject_propagation_events`` on the MASKED propagation pool
                      (the SAME lagPat pool the Task-9 depcheck locates -- broad ->
                      ``_broad_lagpat_dir`` (lagpat_broad_epilepsiae / lagpat_broad_dyn), narrow
                      -> the on-mount ``all_recs`` pool). Participation counts are mask-invariant
                      (the lagPatRank phantom only pollutes non-participating channels' RANKS,
                      never ``eventsBool``), so this rate topography is unaffected by the mask.

  baseline_band_power mean BASELINE log-power per channel for a representative low band
                      (``legacy_bb_1_45`` = the legacy BROAD_BAND (1,45)). The v2 cache stores
                      baseline-normalized z, NOT raw baseline power, so we recompute it from the
                      same seizure windows (``iter_subject_seizure_windows``) with the same
                      ``masked_band_power_trace`` + the SAME ``resolve_baseline_window`` segment
                      the cache uses (GUARD_SEC / MIN_BASELINE_SEC), meaned over the baseline
                      bins then averaged across the subject's eligible seizures.

  broadband_1_250     same construction on the 1-250 Hz broadband (``common_field.broadband_band``).

  shaft_position      anatomical contact order ALONG THE PHYSICAL SHAFT, from the propagation
                      geometry (``shaft`` field + contact numbering). Deliberately NOT
                      ``along_axis_mm``: that is the coordinate along the PROPAGATION axis and is
                      collinear-by-construction with ``typical_rank`` (G_HFO) -- residualizing the
                      timing rank against it would remove the very axis the alignment tests. The
                      shaft-order index is the anatomical confound (deep vs shallow contact),
                      independent of the timing axis.

  soz / resection     per-contact clinical labels, written ONLY if a reliable per-subject label
                      file exists in the repo (``results/{ds}_soz_core_channels.json``). If a
                      subject / label is not reliably available it is OMITTED and logged in
                      ``omitted`` -- never fabricated. No per-contact resection label file exists
                      in the repo, so ``resection`` is always omitted (logged).

Out: ``{outdir}/{substrate}/phase1_confound_maps.json`` -- a dict keyed by ds_sid; each value
carries the per-contact maps above plus provenance + an ``omitted`` log.

Reuses (does NOT reinvent): ``load_subject_propagation_events`` (events), the Task-9 depcheck's
lagPat/geo path convention (``_broad_lagpat_dir`` / ``_narrow_lagpat_dir`` / ``_SUBSTRATE[*].geo``),
``iter_subject_seizure_windows`` + ``masked_band_power_trace`` + ``resolve_baseline_window`` +
``bipolar_alias_label`` (baseline power, byte-identical inputs to the band cache).

Plan: docs/superpowers/plans/2026-07-01-topic5-v2-phase1-band-scan-backbone.md Task 12a (issue #13).
"""
from __future__ import annotations

import argparse
import json
import re
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
    iter_subject_seizure_windows, GUARD_SEC, MIN_BASELINE_SEC)
from scripts.run_topic5_ictal_field_dynamics import SUBJECTS_BY_SUB  # noqa: E402
from scripts.run_topic5_t0_eligibility import BROAD_BAND  # noqa: E402  (legacy_bb_1_45 edges)
# Task-9 depcheck path convention (the MASKED propagation pool + producer geometry planes):
from scripts.run_topic5_v2_order_null_depcheck import (  # noqa: E402
    _SUBSTRATE as _DEPCHECK_SUBSTRATE, _narrow_lagpat_dir)
from src.topic5_event_resolved_alignment import _broad_lagpat_dir  # noqa: E402
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic5_ictal_recruitment import bipolar_alias_label  # noqa: E402
from src.topic5_v2_band_scan import load_phase1_config, masked_band_power_trace  # noqa: E402
from src.ictal_onset_extraction import resolve_baseline_window  # noqa: E402

OUT_ROOT = _ROOT / "results/topic5_ictal_recruitment/v2_band_scan"


# --------------------------------------------------------------------------- hfo_rate
def _lagpat_dir(ds: str, subj: str, substrate: str) -> Path:
    """The lagPat pool the Task-9 depcheck locates events from, resolved to an absolute path."""
    if substrate == "narrow":
        return _narrow_lagpat_dir(ds, subj)                      # on-mount all_recs pool
    return _ROOT / _broad_lagpat_dir(ds, subj)                   # relative results/... -> worktree symlink


def build_hfo_rate(ds: str, subj: str, substrate: str) -> dict:
    """Per-channel interictal event count = eventsBool summed over events."""
    ev = load_subject_propagation_events(_lagpat_dir(ds, subj, substrate))
    bools = np.asarray(ev["bools"], bool)
    if bools.size == 0:
        return {}
    counts = bools.sum(axis=1)                                   # sum over events -> per channel
    return {name: int(counts[i]) for i, name in enumerate(ev["channel_names"])}


# --------------------------------------------------------------------------- baseline power
def build_baseline_power_maps(ds_sid: str, substrate: str, cfg: dict) -> tuple[dict, list]:
    """Mean baseline log-power per channel for (legacy_bb (1,45)) and (broadband (1,250)).

    Same seizure windows / spectrogram / baseline segment as the v2 band cache; per-channel
    baseline mean log-power is averaged across the subject's eligible seizures."""
    ln = cfg["line_noise"]
    harmonics, halfwidth = list(ln["harmonics_hz"]), float(ln["halfwidth_hz"])
    spec_win = float(cfg["power"]["spectrogram_win_sec"])
    spec_hop = float(cfg["power"]["spectrogram_hop_sec"])
    fs512_hi = float(cfg["edge"]["fs512_hi_safe_hz"])
    bb_lo, bb_hi = float(BROAD_BAND[0]), float(BROAD_BAND[1])                 # legacy_bb_1_45
    wide_lo, wide_hi = [float(x) for x in cfg["common_field"]["broadband_band"]]  # 1-250
    bands = [("baseline_band_power", bb_lo, bb_hi), ("broadband_1_250", wide_lo, wide_hi)]

    accum = {name: {} for name, _, _ in bands}          # map -> {channel: [per-seizure baseline means]}
    channels0 = None
    drops: list = []
    for _idx, sw, eeg_rel in iter_subject_seizure_windows(ds_sid, substrate, drops=drops):
        ch = [bipolar_alias_label(c) for c in sw.ch_names]
        if channels0 is None:
            channels0 = ch
        elif ch != channels0:                            # keep only montage-matching seizures (cache rule)
            drops.append({"idx": _idx, "reason": f"chan_mismatch:{len(ch)}vs{len(channels0)}"})
            continue
        nyq = float(sw.fs) / 2.0
        for mapname, lo, hi in bands:
            if hi >= nyq:                                # Nyquist gate: band unavailable this fs
                continue
            try:
                res = masked_band_power_trace(sw.signal, sw.fs, lo, hi, spec_win, spec_hop,
                                              harmonics, halfwidth, fs512_hi, half_open=False)
            except ValueError:
                continue
            logp = np.asarray(res["logpower"], float)    # (n_ch, n_bins)
            bl = resolve_baseline_window(logp.shape[1], hop_sec=spec_hop, pre_sec=sw.pre_sec,
                                         buffer_sec=GUARD_SEC, eeg_onset_rel_sec=eeg_rel,
                                         min_baseline_valid_sec=MIN_BASELINE_SEC)
            seg = logp[:, bl.start_idx:bl.end_idx]
            if seg.shape[1] == 0:
                continue
            per_ch = np.nanmean(seg, axis=1)             # per-channel baseline mean log-power
            for i, name in enumerate(channels0):
                if np.isfinite(per_ch[i]):
                    accum[mapname].setdefault(name, []).append(float(per_ch[i]))

    maps = {mapname: {name: float(np.mean(vals)) for name, vals in accum[mapname].items() if vals}
            for mapname, _, _ in bands}
    return maps, drops


# --------------------------------------------------------------------------- shaft position
def _contact_sort_key(name: str):
    """Natural order within a shaft: trailing contact number (HL1 < HL2 < ... < HL10)."""
    m = re.search(r"(\d+)$", str(name))
    return (int(m.group(1)) if m else 1_000_000, str(name))


def build_shaft_position(ds_sid: str, substrate: str) -> dict | None:
    """0-based contact index along each physical shaft, from the producer geometry ``shaft`` field.

    NOT ``along_axis_mm`` (collinear with the tested timing axis); this is the anatomical
    deep->shallow contact order, an axis-independent topography confound."""
    geo_dir = _DEPCHECK_SUBSTRATE[substrate]["geo"]
    plane = geo_dir / f"{ds_sid}_t_a.json"
    if not plane.exists():
        return None
    g = json.load(open(plane))
    if "channels" not in g:                              # descriptive_only / no_events plane
        return None
    by_shaft: dict = {}
    for c in g["channels"]:
        by_shaft.setdefault(str(c.get("shaft", "")), []).append(str(c["name"]))
    pos: dict = {}
    for _shaft, names in by_shaft.items():
        for rank, name in enumerate(sorted(names, key=_contact_sort_key)):
            pos[name] = float(rank)
    return pos


def _geometry_channel_names(ds_sid: str, substrate: str) -> list:
    plane = _DEPCHECK_SUBSTRATE[substrate]["geo"] / f"{ds_sid}_t_a.json"
    if not plane.exists():
        return []
    g = json.load(open(plane))
    return [str(c["name"]) for c in g.get("channels", [])]


# --------------------------------------------------------------------------- clinical labels
def build_soz_map(ds: str, subj: str, reference_names: list) -> tuple[dict | None, str | None]:
    """Per-contact SOZ membership over ``reference_names``; (None, reason) if not reliably available."""
    fn = _ROOT / f"results/{ds}_soz_core_channels.json"
    if not fn.exists():
        return None, f"no label file {fn.name}"
    d = json.load(open(fn))
    if subj not in d:
        return None, f"subject {subj!r} absent from {fn.name}"
    soz = d[subj]
    if not isinstance(soz, list) or not soz:
        return None, f"empty / non-list SOZ entry for {subj!r} in {fn.name}"
    if not reference_names:
        return None, "no reference channel set (missing geometry/events) to key SOZ against"
    soz_set = set(soz)
    return {name: bool(name in soz_set) for name in reference_names}, None


# --------------------------------------------------------------------------- per subject
def build_subject(ds_sid: str, substrate: str, cfg: dict) -> dict:
    ds, subj = ds_sid.split("_", 1)
    omitted: list = []
    rec: dict = {"ds_sid": ds_sid, "dataset": ds, "subject": subj, "substrate": substrate}

    try:
        rec["hfo_rate"] = build_hfo_rate(ds, subj, substrate)
    except (FileNotFoundError, ValueError, KeyError) as e:
        rec["hfo_rate"] = {}
        omitted.append({"map": "hfo_rate", "reason": f"{type(e).__name__}: {e}"})

    power_maps, power_drops = build_baseline_power_maps(ds_sid, substrate, cfg)
    rec["baseline_band_power"] = power_maps.get("baseline_band_power", {})
    rec["broadband_1_250"] = power_maps.get("broadband_1_250", {})
    for mapname in ("baseline_band_power", "broadband_1_250"):
        if not rec[mapname]:
            omitted.append({"map": mapname, "reason": "no baseline segment / band unavailable (e.g. Nyquist)"})

    shaft = build_shaft_position(ds_sid, substrate)
    if shaft:
        rec["shaft_position"] = shaft
    else:
        omitted.append({"map": "shaft_position", "reason": "no producer geometry plane with channels"})

    ref_names = _geometry_channel_names(ds_sid, substrate) or list(rec["hfo_rate"].keys())
    soz, soz_reason = build_soz_map(ds, subj, ref_names)
    if soz is not None:
        rec["soz"] = soz
    else:
        omitted.append({"map": "soz", "reason": soz_reason})

    # No per-contact resection label file exists in the repo -> always omitted (never fabricated).
    omitted.append({"map": "resection", "reason": "no per-contact resection label file in repo"})

    rec["omitted"] = omitted
    rec["provenance"] = {
        "hfo_rate_source": str(_lagpat_dir(ds, subj, substrate)),
        "baseline_band": {"name": "legacy_bb_1_45", "lo": float(BROAD_BAND[0]), "hi": float(BROAD_BAND[1])},
        "broadband": {"lo": float(cfg["common_field"]["broadband_band"][0]),
                      "hi": float(cfg["common_field"]["broadband_band"][1])},
        "baseline_window": {"guard_sec": GUARD_SEC, "min_baseline_sec": MIN_BASELINE_SEC,
                            "note": "resolve_baseline_window; same segment as v2 band cache"},
        "shaft_position": "anatomical shaft-order index (geometry 'shaft' + contact number); "
                          "NOT along_axis_mm (collinear with tested timing axis)",
        "geometry_plane": str(_DEPCHECK_SUBSTRATE[substrate]["geo"] / f"{ds_sid}_t_a.json"),
        "power_seizure_drops": power_drops,
    }
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--substrate", choices=list(_DEPCHECK_SUBSTRATE), default="broad")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="explicit ds_sid list (default = SUBJECTS_BY_SUB[substrate])")
    ap.add_argument("--outdir", default=None,
                    help="override output ROOT (default results/.../v2_band_scan); "
                         "writes {outdir}/{substrate}/phase1_confound_maps.json")
    args = ap.parse_args()

    cfg = load_phase1_config()
    subjects = args.subjects or SUBJECTS_BY_SUB[args.substrate]
    out_root = Path(args.outdir) if args.outdir else OUT_ROOT
    outdir = out_root / args.substrate
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[confound-maps] {args.substrate}: {len(subjects)} subjects -> {outdir}", flush=True)
    all_maps: dict = {}
    for ds_sid in subjects:
        try:
            rec = build_subject(ds_sid, args.substrate, cfg)
        except Exception as e:                           # one bad subject must not sink the batch
            print(f"  SUBJECT ERROR {ds_sid} {type(e).__name__}: {e}", flush=True)
            continue
        all_maps[ds_sid] = rec
        om = ",".join(o["map"] for o in rec["omitted"])
        print(f"  [{ds_sid}] hfo_rate={len(rec['hfo_rate'])} "
              f"baseline_band_power={len(rec['baseline_band_power'])} "
              f"broadband_1_250={len(rec['broadband_1_250'])} "
              f"shaft_position={len(rec.get('shaft_position', {}))} "
              f"soz={len(rec.get('soz', {})) if 'soz' in rec else 'OMIT'} | omitted[{om}]", flush=True)

    outpath = outdir / "phase1_confound_maps.json"
    json.dump(all_maps, open(outpath, "w"), indent=2, ensure_ascii=False)
    print(f"[done] {args.substrate}: {len(all_maps)}/{len(subjects)} subjects -> {outpath}", flush=True)


if __name__ == "__main__":
    main()
