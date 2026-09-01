#!/usr/bin/env python3
"""B0.3 -- early ictal field target, anchored on EEG onset.

Rebuilds the peri-onset field from raw signal with the validated Topic 5
primitives rather than reusing ``results/topic5_ictal_recruitment/
ictal_field_long_cache``, because that cache (a) covers only 13 of the 27
dataset subjects (2 Yuquan) and (b) anchors Epilepsiae on the *clinical* onset
while topic5 caveat 9 requires EEG-onset anchoring. The cache is kept as a
parity reference instead (see ``--parity-check``).

Adapters, not core edits: a per-dataset seizure inventory is written with the
anchor column carrying the EEG onset, so ``extract_seizure_window`` runs
unmodified.

Outputs (per subject, on /data):
    early_field/<subject>.npz    per-seizure fields + traces
    early_field/<subject>.json   channel order, reference, normalization, coverage
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1 = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1"
DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b"
DEFAULT_DATA = Path("/data/hfosp_group_event_state_v0_2/agent_b")

# Same band / hop / reference conventions as the accepted Topic 5 ictal work, so
# the target is comparable and the parity check is meaningful.
BROAD_BAND = (1.0, 45.0)
HFA_BAND = (60.0, 100.0)
HOP = 0.1
GUARD_SEC = 60.0
MIN_BASELINE_SEC = 30.0
ICTAL_REFERENCE = {"yuquan": "bipolar", "epilepsiae": "car"}

PRE_SEC = 120.0   # baseline [-120,-60]: 60 s, twice the 30 s minimum
POST_SEC = 15.0   # covers the 10 s sensitivity window with margin
PRIMARY_WINDOW = (0.0, 5.0)
SENSITIVITY_WINDOW = (0.0, 10.0)
RECRUIT_THRESHOLD_Z = 5.0


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def build_adapters(crosswalk: Path, data_root: Path):
    """Write EEG-onset-anchored seizure inventories + a complete Yuquan block table."""
    adir = data_root / "adapter"
    adir.mkdir(parents=True, exist_ok=True)

    matched = [r for r in csv.DictReader(crosswalk.open()) if r["disposition"] == "matched"]
    by_sid = {(r["dataset"], r["subject"], r["seizure_id"]) for r in matched}

    # --- epilepsiae: clin_onset_epoch := eeg_onset_epoch (anchor swap), and
    # C9: verify each row's own block_id actually contains its onset ---
    from src.topic5_h2b_transfer.crosswalk import resolve_block_for_onset
    epi_blocks = {}
    for b in csv.DictReader((MAIN_TREE / "results/epilepsiae_block_inventory.csv").open()):
        epi_blocks.setdefault(b["subject"], []).append(b)

    src = list(csv.DictReader((MAIN_TREE / "results/epilepsiae_seizure_inventory.csv").open()))
    rows, repairs = [], []
    for r in src:
        if ("epilepsiae", f"epilepsiae_{r['subject']}", r["seizure_id"]) not in by_sid:
            continue
        r = dict(r)
        r["clin_onset_epoch"] = r["eeg_onset_epoch"]  # anchor on the EEG onset
        bid, status = resolve_block_for_onset(
            float(r["eeg_onset_epoch"]), epi_blocks.get(r["subject"], []), r["block_id"]
        )
        if status == "claim_repaired":
            repairs.append({"subject": r["subject"], "seizure_id": r["seizure_id"],
                            "claimed_block_id": r["block_id"], "repaired_block_id": bid})
            r["block_id"] = bid
        elif status == "no_block_contains_onset":
            repairs.append({"subject": r["subject"], "seizure_id": r["seizure_id"],
                            "claimed_block_id": r["block_id"], "repaired_block_id": ""})
        rows.append(r)
    (adir / "block_id_repairs.json").write_text(json.dumps(repairs, indent=2))
    if repairs:
        print(f"block_id repaired for {len(repairs)} epilepsiae seizure(s): "
              + ", ".join(f"{x['subject']}/{x['seizure_id']}" for x in repairs), flush=True)
    epi_sz = adir / "epilepsiae_seizure_inventory_eegonset.csv"
    with epi_sz.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(src[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- yuquan: anchor is already eeg_onset_epoch; just restrict to matched ---
    src = list(csv.DictReader((ROOT / "results/dataset_inventory/yuquan_seizure_inventory.csv").open()))
    rows = [r for r in src
            if ("yuquan", f"yuquan_{r['subject']}", r["seizure_id"]) in by_sid]
    yq_sz = adir / "yuquan_seizure_inventory_matched.csv"
    with yq_sz.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(src[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # --- yuquan blocks: rebuild from v0.1 audit so no subject is missing ---
    yq_blocks = adir / "yuquan_block_inventory_full.csv"
    with yq_blocks.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "recording_id", "block_id", "block_stem",
                    "block_start_epoch", "block_end_epoch", "edf_path", "data_path"])
        for r in csv.DictReader((V0_1 / "block_inventory.csv").open()):
            if r["dataset"] != "yuquan":
                continue
            sid = r["subject"].split("_", 1)[1]
            w.writerow([sid, r["record_name"], r["record_name"], r["record_name"],
                        r["block_start_epoch"], r["block_end_epoch"],
                        r["raw_path"], r["raw_path"]])
    return {"epilepsiae_seizures": epi_sz, "yuquan_seizures": yq_sz, "yuquan_blocks": yq_blocks}


def subject_seizures(crosswalk: Path):
    out: dict[str, list[dict]] = {}
    for r in csv.DictReader(crosswalk.open()):
        if r["disposition"] != "matched":
            continue
        out.setdefault(r["subject"], []).append(
            {"seizure_id": r["seizure_id"], "onset_epoch": _f(r["onset_epoch"]),
             "offset_epoch": _f(r["offset_epoch"])}
        )
    for v in out.values():
        v.sort(key=lambda s: s["onset_epoch"])
    return out


def process_subject(subject: str, adapters: dict, data_root: Path) -> dict:
    import src.topic5_ictal_recruitment as recruit
    from src.ictal_onset_extraction import extract_seizure_window, resolve_baseline_window
    from src.topic5_t0_features import window_activation
    from src.topic5_h2b_transfer.early_field import (
        first_crossing_time, laterality_index, save_npz_atomic, spatial_entropy,
    )

    dataset, sid = subject.split("_", 1)
    ref = ICTAL_REFERENCE[dataset]
    if dataset == "epilepsiae":
        sz_csv, blk_csv = adapters["epilepsiae_seizures"], MAIN_TREE / "results/epilepsiae_block_inventory.csv"
    else:
        sz_csv, blk_csv = adapters["yuquan_seizures"], adapters["yuquan_blocks"]

    seizures = subject_seizures(adapters["crosswalk"]).get(subject, [])
    arrays, per_sz, channels, fs_seen = {}, [], None, None
    hemi = None

    for idx, s in enumerate(seizures):
        rec = {"seizure_id": s["seizure_id"], "seizure_index": idx,
               "onset_epoch": s["onset_epoch"], "status": "ok", "reason": ""}
        try:
            sw = extract_seizure_window(
                f"{dataset}/{sid}", idx, pre_sec=PRE_SEC, post_sec=POST_SEC,
                reference=ref, alias_bipolar_to_left=(ref == "bipolar"),
                seizure_inventory_csv=sz_csv, block_inventory_csv=blk_csv,
            )
        except Exception as exc:  # window crosses a block edge, missing file, ...
            rec.update(status="dropped", reason=f"{type(exc).__name__}: {exc}")
            per_sz.append(rec)
            continue

        if channels is None:
            channels, fs_seen = list(sw.ch_names), float(sw.fs)
            try:
                from src.seeg_coord_loader import load_subject_coords
                cr = load_subject_coords(dataset, sid, channels)
                x = cr.coords_array_in_requested_order[:, 0]
                hemi = np.where(np.isfinite(x), np.sign(x).astype(int), 0)
                hemi = np.where(hemi == 0, 0, hemi)
                coord_space = cr.coord_space
                n_mapped = int(cr.mapped_mask_in_requested_order.sum())
            except Exception as exc:
                hemi = np.zeros(len(channels), dtype=int)
                coord_space, n_mapped = f"unavailable: {type(exc).__name__}", 0
            rec["_coord_space"] = coord_space
            rec["_n_mapped"] = n_mapped
        elif list(sw.ch_names) != channels:
            rec.update(status="dropped", reason="channel_order_changed_within_subject")
            per_sz.append(rec)
            continue

        # eeg_rel == 0 by construction: the adapter anchors the window on EEG onset.
        eeg_rel = 0.0
        bb, bt = recruit.band_power_trace(sw.signal, sw.fs, band=BROAD_BAND, win_sec=1.0, hop_sec=HOP)
        hf, ht = recruit.band_power_trace(sw.signal, sw.fs, band=HFA_BAND, win_sec=0.5, hop_sec=HOP)
        blb = resolve_baseline_window(bb.shape[1], hop_sec=HOP, pre_sec=sw.pre_sec,
                                      buffer_sec=GUARD_SEC, eeg_onset_rel_sec=eeg_rel,
                                      min_baseline_valid_sec=MIN_BASELINE_SEC)
        blh = resolve_baseline_window(hf.shape[1], hop_sec=HOP, pre_sec=sw.pre_sec,
                                      buffer_sec=GUARD_SEC, eeg_onset_rel_sec=eeg_rel,
                                      min_baseline_valid_sec=MIN_BASELINE_SEC)
        z_bb = recruit.baseline_robust_z(bb, (blb.start_idx, blb.end_idx), hop_sec=HOP,
                                         min_baseline_valid_sec=MIN_BASELINE_SEC)
        z_hf = recruit.baseline_robust_z(hf, (blh.start_idx, blh.end_idx), hop_sec=HOP,
                                         min_baseline_valid_sec=MIN_BASELINE_SEC)
        bb_relt = (np.asarray(bt, float) - float(sw.pre_sec)).astype(np.float32)
        hfa_relt = (np.asarray(ht, float) - float(sw.pre_sec)).astype(np.float32)

        f5 = window_activation(z_bb, bb_relt, *PRIMARY_WINDOW)
        f10 = window_activation(z_bb, bb_relt, *SENSITIVITY_WINDOW)
        h5 = window_activation(z_hf, hfa_relt, *PRIMARY_WINDOW)
        h10 = window_activation(z_hf, hfa_relt, *SENSITIVITY_WINDOW)
        tcross = first_crossing_time(z_hf, hfa_relt, RECRUIT_THRESHOLD_Z, *SENSITIVITY_WINDOW)

        tag = f"{idx:03d}"
        arrays[f"bb_field_5s__{tag}"] = f5.astype(np.float32)
        arrays[f"bb_field_10s__{tag}"] = f10.astype(np.float32)
        arrays[f"hfa_field_5s__{tag}"] = h5.astype(np.float32)
        arrays[f"hfa_field_10s__{tag}"] = h10.astype(np.float32)
        arrays[f"first_crossing_s__{tag}"] = tcross.astype(np.float32)
        # early trace kept only over [-5, +15] s: enough for path/axis work, small on disk
        keep = (bb_relt >= -5.0) & (bb_relt <= POST_SEC)
        arrays[f"bb_early_trace__{tag}"] = z_bb[:, keep].astype(np.float32)
        arrays[f"bb_early_relt__{tag}"] = bb_relt[keep]

        n_rec = int(np.isfinite(tcross).sum())
        rec.update(
            n_channels=len(channels),
            baseline_bins=int(blb.end_idx - blb.start_idx),
            field_5s_finite=int(np.isfinite(f5).sum()),
            entropy_5s=spatial_entropy(f5),
            entropy_10s=spatial_entropy(f10),
            laterality_5s=laterality_index(f5, hemi),
            n_recruited_10s=n_rec,
            first_recruit_time_s=float(np.nanmin(tcross)) if n_rec else float("nan"),
            max_z_5s=float(np.nanmax(f5)) if np.isfinite(f5).any() else float("nan"),
        )
        per_sz.append(rec)

    out = data_root / "early_field"
    out.mkdir(parents=True, exist_ok=True)
    if arrays:
        save_npz_atomic(out / f"{subject}.npz", arrays)
    meta = {
        "subject": subject, "dataset": dataset, "reference": ref,
        "channels": channels, "n_channels": len(channels) if channels else 0,
        "fs": fs_seen, "hop_sec": HOP,
        "band_broad": list(BROAD_BAND), "band_hfa": list(HFA_BAND),
        "anchor": "eeg_onset_epoch",
        "normalization": (f"baseline robust-z; baseline ends {GUARD_SEC}s before EEG onset, "
                          f"min {MIN_BASELINE_SEC}s valid; pre={PRE_SEC}s post={POST_SEC}s"),
        "primary_window_sec": list(PRIMARY_WINDOW),
        "sensitivity_window_sec": list(SENSITIVITY_WINDOW),
        "recruit_threshold_z": RECRUIT_THRESHOLD_Z,
        "coord_space": next((r.get("_coord_space") for r in per_sz if "_coord_space" in r), None),
        "n_contacts_with_coords": next((r.get("_n_mapped") for r in per_sz if "_n_mapped" in r), 0),
        "seizures": [{k: v for k, v in r.items() if not k.startswith("_")} for r in per_sz],
        "n_ok": sum(1 for r in per_sz if r["status"] == "ok"),
        "n_dropped": sum(1 for r in per_sz if r["status"] == "dropped"),
    }
    tmp = out / f"{subject}.json.tmp"
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    tmp.rename(out / f"{subject}.json")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crosswalk", type=Path, default=DEFAULT_OUT / "support/seizure_crosswalk.csv")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-existing", action="store_true",
                    help="idempotent resume: skip subjects whose json already accounts "
                         "for every matched seizure")
    args = ap.parse_args()

    adapters = build_adapters(args.crosswalk, args.data_root)
    adapters["crosswalk"] = args.crosswalk
    all_sz = subject_seizures(args.crosswalk)
    subs = args.subjects or sorted(all_sz)
    if args.skip_existing:
        keep = []
        for s in subs:
            jp = args.data_root / "early_field" / f"{s}.json"
            if jp.exists():
                try:
                    m = json.loads(jp.read_text())
                    if m.get("n_ok", 0) + m.get("n_dropped", 0) == len(all_sz.get(s, [])):
                        continue
                except Exception:
                    pass
            keep.append(s)
        print(f"resume: {len(subs) - len(keep)} subject(s) already complete", flush=True)
        subs = keep
    print(f"subjects: {len(subs)}  workers: {args.workers}", flush=True)

    results = []
    if args.workers <= 1:
        for s in subs:
            try:
                results.append(process_subject(s, adapters, args.data_root))
                print(f"  {s}: ok={results[-1]['n_ok']} dropped={results[-1]['n_dropped']}", flush=True)
            except Exception:
                traceback.print_exc()
                results.append({"subject": s, "n_ok": 0, "n_dropped": 0, "fatal": True})
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(process_subject, s, adapters, args.data_root): s for s in subs}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    m = fut.result()
                    results.append(m)
                    print(f"  {s}: ok={m['n_ok']} dropped={m['n_dropped']}", flush=True)
                except Exception:
                    print(f"  {s}: FATAL"); traceback.print_exc()
                    results.append({"subject": s, "n_ok": 0, "n_dropped": 0, "fatal": True})

    # The status table describes the whole cohort on disk, not just the subjects
    # this invocation touched -- otherwise re-running a subset silently truncates
    # it (observed: a 2-subject repair rewrote 271 rows down to 60).
    (args.out_root / "support").mkdir(parents=True, exist_ok=True)
    on_disk = []
    for jp in sorted((args.data_root / "early_field").glob("*.json")):
        try:
            on_disk.append(json.loads(jp.read_text()))
        except Exception:
            continue
    rows = []
    for m in on_disk:
        for r in m.get("seizures", []):
            rows.append({"subject": m["subject"], "dataset": m.get("dataset", ""),
                         "reference": m.get("reference", ""), "n_channels": m.get("n_channels", 0),
                         "coord_space": m.get("coord_space", ""), **r})
    if rows:
        keys = sorted({k for r in rows for k in r})
        p = args.out_root / "support/early_field_status.csv"
        tmp = p.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        tmp.rename(p)
        print(f"\nwrote {p}")
    print(f"TOTAL ok={sum(m.get('n_ok',0) for m in results)} "
          f"dropped={sum(m.get('n_dropped',0) for m in results)}")


if __name__ == "__main__":
    main()
