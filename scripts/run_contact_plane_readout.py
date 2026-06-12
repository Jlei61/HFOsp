#!/usr/bin/env python3
"""真实 subject 2D 传播触点平面读出。
Spec: docs/superpowers/specs/2026-06-11-propagation-contact-plane-readout-design.md
Out:  results/spatial_modulation/propagation_geometry/observation_readout/real_subjects/
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
warnings.filterwarnings("ignore", message="Mean of empty slice")

from src.interictal_propagation import load_subject_propagation_events
from src.lagpat_rank_audit import mask_phantom_ranks
from src.seeg_coord_loader import load_subject_coords
from src.sef_hfo_soz_localization import classify_montage, _first_contact
from src import propagation_skeleton_geometry as G
from src import propagation_contact_plane_readout as R

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
RANKDISP = _ROOT / "results/interictal_propagation_masked/rank_displacement/per_subject"
SOZ_JSON = {ds: _ROOT / f"results/{ds}_soz_core_channels.json"
            for ds in ("yuquan", "epilepsiae")}
OUT = _ROOT / "results/spatial_modulation/propagation_geometry/observation_readout/real_subjects"
EXCLUDE_BAD_DATA = {("yuquan", "pengzihang")}


def _subject_dir(ds, subj):
    return YUQUAN_ROOT / subj if ds == "yuquan" else EPILEPSIAE_ROOT / subj / "all_recs"


def _soz_set(ds, subj):
    try:
        d = json.loads(SOZ_JSON[ds].read_text())
        entry = d.get(subj, d.get(str(subj)))
        if isinstance(entry, dict):
            entry = entry.get("core_channels", [])
        return set(entry or [])
    except Exception:
        return set()


def _load_accepted_templates(ds, subj, names):
    """复用 skeleton runner 的 accepted-template 加载（rank-displacement pair[0]）。"""
    f = RANKDISP / f"{ds}_{subj}.json"
    if not f.exists():
        raise FileNotFoundError(f"rank-displacement JSON missing for {ds}:{subj} ({f})")
    d = json.loads(f.read_text())
    pair = (d.get("pairs") or [{}])[0]
    rd_names = list(pair.get("channel_names") or [])
    ra = np.asarray(pair.get("rank_a_dense_full"), float)
    rb = np.asarray(pair.get("rank_b_dense_full"), float)
    idx_a = {nm: ra[i] for i, nm in enumerate(rd_names)}
    idx_b = {nm: rb[i] for i, nm in enumerate(rd_names)}
    ta = np.array([idx_a.get(nm, np.nan) for nm in names], float)
    tb = np.array([idx_b.get(nm, np.nan) for nm in names], float)
    swap = (((pair.get("swap_sweep") or {}).get("swap_class")) or "none")
    return ta, tb, swap


def build_record_from_events(*, dataset, subject, template_id, names, ranks, bools,
                             lag_raw, coords, mapped, soz_core, montage,
                             lag_time_unit, spacing_mm, template_axis=None):
    """事件数组 -> 标准化 readout record（mount-free，单测入口）。

    1D 采样判定在 compute_axis_frame 之后，用【真实】fr['off_axis'] + participating
    mask（reviewer P1：不得用全零 off-axis 提前判，会系统性误标 1D）。
    """
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    if template_axis is None:
        template_axis = np.array(
            [np.nanmean(r) if np.any(~np.isnan(r)) else np.nan for r in masked])
    eligible = (~np.isnan(template_axis)) & np.asarray(mapped, bool)
    cores = G.build_endpoint_cores(template_axis, eligible, k_primary=3)
    if cores["tier"] == "descriptive_only":
        return {"dataset": dataset, "subject": subject, "template_id": template_id,
                "status": "descriptive_only"}
    fr = G.compute_axis_frame(coords, cores["source_idx"], cores["sink_idx"])
    # 轴外残差 perp_vec：rel - along*u
    src_c = np.array(fr["source_centroid"])
    axis = np.array(fr["sink_centroid"]) - src_c
    u = axis / max(np.linalg.norm(axis), 1e-12)
    rel = np.asarray(coords, float) - src_c
    along = np.asarray(fr["along_axis"], float)
    perp_vec = rel - np.outer(np.where(np.isnan(along), 0.0, along), u)
    perp_vec[np.isnan(coords).any(axis=1)] = np.nan
    participating = bools.any(axis=1) & eligible
    st = R.signed_transverse_axis(perp_vec, participating)
    # 1D 判定（真实 off-axis，不是全零）
    samp = G.classify_sampling_geometry(
        names, participating, np.asarray(fr["off_axis"], float), spacing_mm=spacing_mm)
    one_d = samp.get("geometry") == "1D"
    soz = R.resolve_soz_overlay(list(names), soz_core, montage)
    rec = R.build_readout_record(
        dataset=dataset, subject=subject, template_id=template_id, names=names,
        along_axis_mm=along, axis_length_mm=fr["axis_length"],
        off_axis_mm=np.asarray(fr["off_axis"], float),
        signed_transverse=st["signed_transverse"],
        pc1_variance_explained=st["pc1_variance_explained"],
        masked=masked, lag_raw=lag_raw, bools=bools,
        soz_first_contacts=soz["soz_first_contacts"], lag_time_unit=lag_time_unit,
        one_dimensional_sampling=one_d)
    rec["soz_ambiguous"] = soz["soz_ambiguous"]
    rec["sampling_geometry"] = samp.get("geometry")
    rec["scalars"] = R.compute_cohort_scalars(rec)
    return rec


def process_subject(ds, subj):
    ev = load_subject_propagation_events(_subject_dir(ds, subj))
    if not ev["channel_names"] or np.asarray(ev["ranks"]).size == 0:
        return [{"dataset": ds, "subject": subj, "status": "no_events"}]
    names = list(ev["channel_names"])
    ranks = np.asarray(ev["ranks"], float)
    bools = np.asarray(ev["bools"]) > 0
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    ta, tb, swap = _load_accepted_templates(ds, subj, names)
    labels = G.assign_events_to_templates(masked, ta, tb)
    cr = load_subject_coords(ds, subj, names)
    coords = np.asarray(cr.coords_array_in_requested_order, float)
    mapped = np.asarray(cr.mapped_mask_in_requested_order, bool)
    # montage 类型（single/bipolar）从通道名判定
    montage = "bipolar" if sum("-" in n for n in names) >= len(names) / 2 else "single"
    spacing = 3.5 if ds == "yuquan" else 4.6
    # hard-assert lag_raw 存在（reviewer：不得 silently 用 rank 伪造 time 副图）
    if "lag_raw" not in ev:
        raise KeyError(f"{ds}:{subj} load_subject_propagation_events 缺 lag_raw 键")
    lag_raw = np.where(bools, np.asarray(ev["lag_raw"], float), np.nan)
    out = []
    # 逐 template（A/B 两个 accepted 模板）各出一份 record（spec §8 逐模板处理）。
    # 命名 t_a/t_b：只是 rank-displacement pair 的两支，NOT dominant/minority 的科学事实。
    for tid, lbl in (("t_a", 0), ("t_b", 1)):
        sel = labels == lbl
        if sel.sum() == 0:
            continue
        tmask = masked[:, sel]
        taxis = np.array([np.nanmean(r) if np.any(~np.isnan(r)) else np.nan
                          for r in tmask])
        rec = build_record_from_events(
            dataset=ds, subject=subj, template_id=tid, names=names,
            ranks=ranks[:, sel], bools=bools[:, sel], lag_raw=lag_raw[:, sel],
            coords=coords, mapped=mapped, soz_core=_soz_set(ds, subj),
            montage=montage, lag_time_unit="s", spacing_mm=spacing,
            template_axis=taxis)
        rec["swap_class"] = swap
        out.append(rec)
    return out


def discover_cohort():
    return [tuple(f.stem.split("_", 1)) for f in sorted(RANKDISP.glob("*.json"))]


# Coord-loader failure fingerprints (yuquan chnXyzDict, epilepsiae SQL/MRI). A
# known coord miss is benign (subject lacks coords; ~9 yuquan); anything else is
# a real bug and must NOT be swept up — mirrors run_propagation_skeleton_geometry.
_COORD_MISS_MARKERS = ("coord file not found", "chnxyzdict",
                       "sql not found", "mri not found", "coords missing")
WHITELIST_ERROR_CATEGORIES = {"no_coord_file"}


def _error_category(status):
    s = str(status).lower()
    if any(m in s for m in _COORD_MISS_MARKERS):
        return "no_coord_file"
    return "unexpected"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cohort = ([tuple(s.split(":", 1)) for s in args.subjects] if args.subjects
              else discover_cohort())
    cohort = [(d, s) for d, s in cohort if (d, s) not in EXCLUDE_BAD_DATA]
    n_ok = 0
    n_skip = 0
    for ds, subj in cohort:
        try:
            recs = process_subject(ds, subj)
        except Exception as e:  # noqa: BLE001
            status = f"error: {e}"
            category = _error_category(status)
            # Only KNOWN-benign coord misses are recorded-and-continued; anything
            # else (loader misuse, contract violation) re-raises loudly.
            if category not in WHITELIST_ERROR_CATEGORIES:
                print(f"  [FATAL {ds}:{subj}] {type(e).__name__}: {e}",
                      file=sys.stderr, flush=True)
                raise
            print(f"  [skip {ds}:{subj}] {category}", flush=True)
            (out / f"{ds}_{subj}.json").write_text(json.dumps(
                {"dataset": ds, "subject": subj, "status": status,
                 "error_category": category}, indent=2, default=float))
            n_skip += 1
            continue
        for rec in recs:
            tid = rec.get("template_id")
            name = f"{ds}_{subj}_{tid}.json" if tid else f"{ds}_{subj}.json"
            (out / name).write_text(json.dumps(rec, indent=2, default=float))
            if rec.get("status") not in ("no_events", "descriptive_only"):
                n_ok += 1
    if n_ok == 0:
        raise SystemExit("no usable real readout records — refusing vacuous run")
    print(f"wrote real readout records: {n_ok} usable, {n_skip} skipped (benign), "
          f"{len(cohort)} attempted")


if __name__ == "__main__":
    main()
