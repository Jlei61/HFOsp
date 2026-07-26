#!/usr/bin/env python
"""Build the prefix-only Figure 6 event-history and signed-target dataset.

This is the Gate-0 producer. It fails closed per subject and writes a complete
attrition row even when no seizure survives. No target seizure is consulted
while estimating or orienting the patient axis.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence
from zoneinfo import ZoneInfo

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/hfosp_fig6_numba")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/hfosp_fig6_mpl")
os.environ.setdefault("_MNE_FAKE_HOME_DIR", "/tmp/hfosp_fig6_mne")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_topic5_t0_eligibility import _inventory_rows
from src.interictal_propagation import load_subject_propagation_events
from src.topic5_state_conditioned_rnn import (
    apply_standardizer,
    axis_split_stability,
    derive_prefix_axis,
    event_feature_matrix,
    fit_prefix_standardizer,
    robust_rebaseline_activation,
    signed_axis_label,
    weighted_inner,
)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _candidate_subjects(parent_table: Path) -> List[str]:
    df = pd.read_csv(parent_table)
    return sorted(
        df.loc[df["group_id"] == "all_phenotype_matched", "subject"]
        .astype(str)
        .unique()
    )


def _raw_subject_dir(ds_sid: str) -> Path:
    dataset, sid = ds_sid.split("_", 1)
    if dataset == "epilepsiae":
        return Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns") / sid / "all_recs"
    if dataset == "yuquan":
        return Path("/mnt/yuquan_data/yuquan_24h_edf") / sid
    raise ValueError(f"unknown dataset: {dataset}")


def _ordered_lagpat_files(subject_dir: Path):
    files = sorted(subject_dir.glob("*_lagPat_withFreqCent.npz"))
    if not files:
        files = sorted(subject_dir.glob("*_lagPat.npz"))
    records = []
    for path in files:
        with np.load(path, allow_pickle=True) as z:
            start = float(np.asarray(z["start_t"]).reshape(-1)[0]) if "start_t" in z else np.nan
        records.append((start, path))
    return [p for _, p in sorted(records, key=lambda x: (x[0] if np.isfinite(x[0]) else np.inf, x[1].name))]


def _packed_path(lagpat: Path) -> Path:
    name = lagpat.name
    if name.endswith("_lagPat_withFreqCent.npz"):
        return lagpat.with_name(name.replace("_lagPat_withFreqCent.npz", "_packedTimes_withFreqCent.npy"))
    return lagpat.with_name(name.replace("_lagPat.npz", "_packedTimes.npy"))


def load_frequency_centroid(subject_dir: Path, union_names: Sequence[str]) -> np.ndarray:
    """Align lagPatFreq to the canonical loader's block and channel ordering."""
    index = {str(name): i for i, name in enumerate(union_names)}
    blocks = []
    for path in _ordered_lagpat_files(subject_dir):
        with np.load(path, allow_pickle=True) as z:
            rank = np.asarray(z["lagPatRank"])
            freq = np.asarray(z["lagPatFreq"], float) if "lagPatFreq" in z else np.full(rank.shape, np.nan)
            names = [str(x) for x in z["chnNames"]]
            n_ch = min(rank.shape[0], freq.shape[0], len(names))
            n_ev = min(rank.shape[1], freq.shape[1])
            packed = _packed_path(path)
            if packed.exists():
                pt = np.asarray(np.load(packed), float)
                if pt.ndim == 2:
                    n_ev = min(n_ev, pt.shape[0])
            block = np.full((len(union_names), n_ev), np.nan)
            for row, name in enumerate(names[:n_ch]):
                if name in index:
                    block[index[name], :] = freq[row, :n_ev]
            blocks.append(block)
    return np.concatenate(blocks, axis=1) if blocks else np.zeros((len(union_names), 0))


def _inventory(ds_sid: str) -> List[dict]:
    dataset, sid = ds_sid.split("_", 1)
    rows, _ = _inventory_rows(dataset, sid)
    return rows


def _float(row: dict, key: str, default=np.nan):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _seizure_intervals(ds_sid: str, inventory: Sequence[dict]):
    dataset = ds_sid.split("_", 1)[0]
    out = []
    for idx, row in enumerate(inventory):
        onset = _float(row, "eeg_onset_epoch")
        offset = _float(row, "eeg_offset_epoch", onset)
        if np.isfinite(onset):
            out.append((idx, onset, offset if np.isfinite(offset) else onset))
    return out


def _crosses_day_boundary(start: float, end: float, timezone: str) -> bool:
    if not np.isfinite(start) or not np.isfinite(end):
        return True
    tz = ZoneInfo(timezone)
    a = datetime.fromtimestamp(start, tz)
    b = datetime.fromtimestamp(end - 1e-3, tz)
    state_a = "day" if 8 <= a.hour < 20 else "night"
    state_b = "day" if 8 <= b.hour < 20 else "night"
    return a.date() != b.date() or state_a != state_b


def eligible_blocks(
    events: Dict[str, object],
    seizure_intervals,
    *,
    post_guard_sec: float,
    timezone: str,
) -> np.ndarray:
    """Fail-closed 1 h parent-block eligibility mask."""
    n_blocks = int(events["n_blocks_used"])
    starts = np.asarray(events["block_start_times"], float)
    good = np.isfinite(starts)
    for block_id in range(n_blocks):
        start = starts[block_id]
        end = start + 3600.0
        if not good[block_id] or _crosses_day_boundary(start, end, timezone):
            good[block_id] = False
            continue
        for _, onset, offset in seizure_intervals:
            guarded_end = offset + post_guard_sec
            if start < guarded_end and end > onset:
                good[block_id] = False
                break
    # A large discontinuity means neither adjacent block may claim to cross a
    # known continuous-recording boundary.
    order = np.argsort(starts)
    for left, right in zip(order[:-1], order[1:]):
        gap = starts[right] - starts[left]
        if np.isfinite(gap) and gap > 5400:
            good[left] = False
            good[right] = False
    return good


def choose_prefix_blocks(
    events: Dict[str, object],
    block_good: np.ndarray,
    *,
    cumulative_hours: int,
) -> np.ndarray:
    starts = np.asarray(events["block_start_times"], float)
    eligible = np.flatnonzero(block_good & np.isfinite(starts))
    if eligible.size < cumulative_hours:
        return np.zeros(0, dtype=int)
    order = eligible[np.argsort(starts[eligible])]
    return order[: int(cumulative_hours)]


def _event_mask_from_blocks(events, blocks):
    return np.isin(np.asarray(events["block_ids"], int), np.asarray(blocks, int))


def _target_for_band(cache, band: str, idx: int, onset_rel: float, cfg):
    zkey = f"{band}__zt__{idx}"
    tkey = f"{band}__relt__{idx}"
    if zkey not in cache or tkey not in cache:
        return None
    return robust_rebaseline_activation(
        np.asarray(cache[zkey], float),
        np.asarray(cache[tkey], float),
        onset_rel=onset_rel,
        baseline_window=tuple(cfg["target"]["baseline_window_seconds"]),
        target_window=tuple(cfg["target"]["primary_window_seconds"]),
        min_baseline_bins=int(cfg["target"]["min_baseline_bins"]),
    )


def build_subject(ds_sid: str, cfg: dict, out_dir: Path, audit_lookup: set):
    dataset, sid = ds_sid.split("_", 1)
    raw_dir = _raw_subject_dir(ds_sid)
    events = load_subject_propagation_events(raw_dir)
    times = np.asarray(events["event_abs_times"], float)
    ranks = np.asarray(events["ranks"], float)
    bools = np.asarray(events["bools"], bool)
    lag_raw = np.asarray(events["lag_raw"], float)
    names = [str(x) for x in events["channel_names"]]
    inventory = _inventory(ds_sid)
    intervals = _seizure_intervals(ds_sid, inventory)
    timezone = "Europe/Berlin" if dataset == "epilepsiae" else "Asia/Shanghai"
    block_good = eligible_blocks(
        events,
        intervals,
        post_guard_sec=float(cfg["cohort"]["seizure_guard_post_minutes"]) * 60,
        timezone=timezone,
    )
    prefix_blocks = choose_prefix_blocks(
        events, block_good, cumulative_hours=int(cfg["cohort"]["calibration_hours"])
    )
    base_row = {
        "dataset": dataset,
        "subject": ds_sid,
        "n_axis_contacts": len(names),
        "n_events_total": int(times.size),
        "n_blocks_total": int(events["n_blocks_used"]),
        "n_blocks_definite_interictal": int(np.sum(block_good)),
        "n_prefix_blocks": int(prefix_blocks.size),
        "prefix_hours_cumulative": int(prefix_blocks.size),
        "prefix_events": 0,
        "prefix_seed_ami": np.nan,
        "prefix_split_axis_correlation": np.nan,
        "n_candidate_seizures": sum((ds_sid, i) in audit_lookup for i in range(len(inventory))),
        "n_eligible_histories": 0,
        "n_primary_targets": 0,
        "gate0_pass": False,
        "reason": "",
    }
    if len(names) < int(cfg["cohort"]["min_axis_contacts"]):
        base_row["reason"] = "too_few_axis_contacts"
        return base_row, []
    if prefix_blocks.size < int(cfg["cohort"]["calibration_hours"]):
        base_row["reason"] = "insufficient_definite_interictal_prefix_blocks"
        return base_row, []
    prefix_mask = _event_mask_from_blocks(events, prefix_blocks) & np.isfinite(times)
    prefix_idx = np.flatnonzero(prefix_mask)
    base_row["prefix_events"] = int(prefix_idx.size)
    if prefix_idx.size < int(cfg["cohort"]["calibration_min_events"]):
        base_row["reason"] = "too_few_prefix_events"
        return base_row, []
    prefix_idx = prefix_idx[np.argsort(times[prefix_idx])]
    try:
        axis = derive_prefix_axis(
            ranks,
            bools,
            prefix_idx,
            seed=int(cfg["model"]["seeds"][0]),
            min_cluster_fraction=float(cfg["cohort"]["calibration_min_cluster_fraction"]),
        )
        split_r = axis_split_stability(
            ranks, bools, prefix_idx, seed=int(cfg["model"]["seeds"][0])
        )
    except Exception as exc:
        base_row["reason"] = f"prefix_axis:{type(exc).__name__}:{exc}"
        return base_row, []
    base_row["prefix_seed_ami"] = float(axis["seed_ami"])
    base_row["prefix_split_axis_correlation"] = float(split_r)
    if axis["seed_ami"] < float(cfg["cohort"]["calibration_min_seed_ami"]):
        base_row["reason"] = "prefix_seed_instability"
        return base_row, []
    if not np.isfinite(split_r) or split_r < float(cfg["cohort"]["calibration_min_split_axis_correlation"]):
        base_row["reason"] = "prefix_split_axis_instability"
        return base_row, []

    freq = load_frequency_centroid(raw_dir, names)
    if freq.shape != ranks.shape:
        freq = np.full_like(ranks, np.nan)
        freq_status = f"shape_mismatch:{freq.shape}!={ranks.shape}"
    else:
        freq_status = "available_as_frequency_centroid_not_energy"
    features, feature_names = event_feature_matrix(
        ranks,
        bools,
        lag_raw,
        times,
        np.asarray(axis["direction_basis"]),
        np.asarray(axis["support_q"]),
        np.asarray(axis["axis_coordinate"]),
        frequency_centroid=freq,
    )
    center, scale = fit_prefix_standardizer(features, prefix_idx)
    features_z = apply_standardizer(features, center, scale).astype(np.float32)

    target_root = ROOT / cfg["outputs"]["target_cache"] / "cache"
    target_npz = target_root / f"{ds_sid}.npz"
    target_json = target_root / f"{ds_sid}.json"
    if not target_npz.exists() or not target_json.exists():
        base_row["reason"] = "missing_target_cache"
        return base_row, []
    with np.load(target_npz, allow_pickle=True) as z:
        cache = {key: z[key] for key in z.files}
    meta = json.loads(target_json.read_text())
    cache_names = [str(x) for x in cache["channels"]]
    cache_idxs = {int(x) for x in meta["seizure_idxs"]}

    prefix_end = float(np.max(np.asarray(events["block_start_times"])[prefix_blocks]) + 3600)
    lookback = float(cfg["history"]["primary_lookback_minutes"]) * 60
    cutoff = float(cfg["history"]["primary_cutoff_minutes"]) * 60
    max_events = int(cfg["history"]["max_events_per_history"])
    history_event_good = block_good[np.asarray(events["block_ids"], int)] & np.isfinite(times)
    arrays = {
        "prefix_features": features_z[prefix_idx],
        "prefix_times": times[prefix_idx].astype(np.float64),
        "feature_names": np.asarray(feature_names),
        "channel_names": np.asarray(names),
        "template_a": np.asarray(axis["template_a"], np.float32),
        "template_b": np.asarray(axis["template_b"], np.float32),
        "support_q": np.asarray(axis["support_q"], np.float32),
        "direction_basis": np.asarray(axis["direction_basis"], np.float32),
        "axis_coordinate": np.asarray(axis["axis_coordinate"], np.float32),
        "standardizer_center": center.astype(np.float32),
        "standardizer_scale": scale.astype(np.float32),
    }
    seizure_rows = []
    band_map = {
        "low_1_8": "target_low_1_8",
        "broad_1_150": "target_broad_1_150",
        "gamma_30_80": "target_gamma_30_80",
        "high_gamma_80_150": "target_high_gamma_80_150",
    }
    for idx, row in enumerate(inventory):
        if (ds_sid, idx) not in audit_lookup or idx not in cache_idxs:
            continue
        eeg_onset = _float(row, "eeg_onset_epoch")
        clin_onset = _float(row, "clin_onset_epoch", eeg_onset)
        if not np.isfinite(eeg_onset) or eeg_onset < prefix_end + lookback + cutoff:
            continue
        hmask = (
            history_event_good
            & (times >= eeg_onset - lookback - cutoff)
            & (times <= eeg_onset - cutoff)
        )
        hidx = np.flatnonzero(hmask)
        hidx = hidx[np.argsort(times[hidx])]
        if hidx.size < int(cfg["cohort"]["min_history_events"]):
            continue
        if hidx.size > max_events:
            take = np.linspace(0, hidx.size - 1, max_events).round().astype(int)
            hidx = hidx[np.unique(take)]
        row_out = {
            "dataset": dataset,
            "subject": ds_sid,
            "seizure_idx": idx,
            "seizure_id": row.get("seizure_id", ""),
            "eeg_onset_epoch": eeg_onset,
            "n_history_events": int(hidx.size),
            "history_start_epoch": float(times[hidx[0]]),
            "history_end_epoch": float(times[hidx[-1]]),
        }
        arrays[f"history_features__{idx}"] = features_z[hidx]
        arrays[f"history_times__{idx}"] = times[hidx].astype(np.float64)
        onset_rel = eeg_onset - clin_onset
        for cache_band, column in band_map.items():
            activation = _target_for_band(cache, cache_band, idx, onset_rel, cfg)
            if activation is None:
                row_out[column] = np.nan
                row_out[f"n_common_{cache_band}"] = 0
                continue
            label = signed_axis_label(
                activation,
                cache_names,
                names,
                np.asarray(axis["direction_basis"]),
                np.asarray(axis["support_q"]),
                min_common_contacts=int(cfg["target"]["min_common_contacts"]),
            )
            row_out[column] = float(label["coefficient"])
            row_out[f"n_common_{cache_band}"] = int(label["n_common"])
            arrays[f"field_{cache_band}__{idx}"] = np.asarray(label["field"], np.float32)
        seizure_rows.append(row_out)

    base_row["n_eligible_histories"] = len(seizure_rows)
    base_row["n_primary_targets"] = int(
        sum(np.isfinite(row["target_low_1_8"]) for row in seizure_rows)
    )
    if base_row["n_primary_targets"] == 0:
        base_row["reason"] = "no_finite_primary_target"
        return base_row, seizure_rows
    base_row["gate0_pass"] = True
    base_row["reason"] = "ok"
    np.savez_compressed(out_dir / "per_subject" / f"{ds_sid}.npz", **arrays)
    axis_record = {
        "contract": cfg["contract"]["name"],
        "subject": ds_sid,
        "axis_source": "prefix_only_masked_rank_k2",
        "prefix_block_ids": prefix_blocks,
        "prefix_end_epoch": prefix_end,
        "prefix_event_count": prefix_idx.size,
        "seed_ami": axis["seed_ami"],
        "split_axis_correlation": split_r,
        "cluster_fractions": axis["cluster_fractions"],
        "channel_names": names,
        "template_a": axis["template_a"],
        "template_b": axis["template_b"],
        "support_a": axis["support_a"],
        "support_b": axis["support_b"],
        "support_q": axis["support_q"],
        "direction_basis": axis["direction_basis"],
        "axis_coordinate": axis["axis_coordinate"],
        "frequency_feature_status": freq_status,
        "hfo_energy_status": cfg["event_features"]["hfo_energy_status"],
    }
    (out_dir / "per_subject" / f"{ds_sid}.json").write_text(
        json.dumps(_jsonable(axis_record), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return base_row, seizure_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "config/topic5_state_conditioned_predictor.yaml")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--calibration-hours", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.calibration_hours is not None:
        cfg["cohort"]["calibration_hours"] = int(args.calibration_hours)
    out_dir = (
        args.out_dir
        if args.out_dir is not None and args.out_dir.is_absolute()
        else ROOT / args.out_dir
        if args.out_dir is not None
        else ROOT / cfg["outputs"]["dataset"]
    )
    (out_dir / "per_subject").mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or _candidate_subjects(ROOT / cfg["cohort"]["parent_event_table"])
    audit = pd.read_csv(ROOT / cfg["cohort"]["eligibility_audit"])
    eligible = audit[
        audit["analysis_eligible"].astype(str).str.lower().isin(("true", "1", "yes"))
    ]
    audit_lookup = {
        (str(row.subject_id), int(row.seizure_idx)) for row in eligible.itertuples()
    }
    attrition, seizures = [], []
    for subject in subjects:
        print(f"[gate0] {subject}", flush=True)
        try:
            row, rows = build_subject(subject, cfg, out_dir, audit_lookup)
        except Exception as exc:
            row = {
                "dataset": subject.split("_", 1)[0],
                "subject": subject,
                "gate0_pass": False,
                "reason": f"unhandled:{type(exc).__name__}:{exc}",
            }
            rows = []
        attrition.append(row)
        seizures.extend(rows)
        print(
            f"  -> {row.get('reason')} | prefix={row.get('prefix_events', 0)} "
            f"targets={row.get('n_primary_targets', 0)}",
            flush=True,
        )
    pd.DataFrame(attrition).to_csv(out_dir / "gate0_attrition.csv", index=False)
    pd.DataFrame(seizures).to_csv(out_dir / "seizure_targets.csv", index=False)
    manifest = {
        "contract": cfg["contract"]["name"],
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": _sha256(args.config),
        "source_spec_sha256": cfg["contract"]["source_spec_sha256"],
        "n_candidate_subjects": len(subjects),
        "calibration_hours_cumulative": int(cfg["cohort"]["calibration_hours"]),
        "n_gate0_subjects": int(sum(bool(r.get("gate0_pass")) for r in attrition)),
        "n_seizures_with_primary_target": int(
            sum(np.isfinite(r.get("target_low_1_8", np.nan)) for r in seizures)
        ),
        "leakage_guards": {
            "axis": "chronological definite-interictal prefix only",
            "orientation": "deterministic first finite channel contrast; no seizure target",
            "target_band": "fixed 1-8 Hz half-open before fitting",
            "target_alignment": "EEG onset [0,10] s; baseline [-120,-90] s",
            "history": "[-65,-5] min",
            "phantom_rank": "eventsBool-masked local rerank",
        },
        "known_input_deviation": {
            "hfo_energy": "not present in accepted lagPat artifact; omitted rather than substituting frequency",
            "frequency_centroid": "included as a separately named event covariate",
        },
    }
    (out_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
