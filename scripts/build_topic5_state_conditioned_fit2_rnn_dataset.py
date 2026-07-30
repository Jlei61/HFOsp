#!/usr/bin/env python
"""Build the clinical-onset BB150 Fit-2 history dataset for the formal RNN.

The prefix axis and all event-feature standardization are estimated from the
same definite-interictal calibration prefix used by Fit 2. The target is the
accepted strict-broadband maxAB score minus its all-contact channel-shuffle
median. It is a coarse, unsigned scaffold-expression target, not a directional
coefficient and not a continuous-time seizure-forecasting label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_topic5_state_conditioned_dataset import (
    _event_mask_from_blocks,
    _float,
    _inventory,
    _raw_subject_dir,
    _seizure_intervals,
    choose_prefix_blocks,
    eligible_blocks,
    load_frequency_centroid,
)
from src.interictal_propagation import load_subject_propagation_events
from src.topic5_state_conditioned_rnn import (
    apply_standardizer,
    axis_split_stability,
    derive_prefix_axis,
    event_feature_matrix,
    fit_prefix_standardizer,
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


def load_targets(cfg: dict) -> pd.DataFrame:
    path = ROOT / cfg["cohort"]["parent_event_table"]
    frame = pd.read_csv(path)
    group = str(cfg["target"]["primary_group"])
    frame = frame[frame.group_id.astype(str) == group].copy()
    keys = ["dataset", "subject", "seizure_idx"]
    if frame.duplicated(keys).any():
        raise ValueError("Fit-2 target table contains duplicate strict-BB events")
    if set(frame.time_reference.astype(str)) != {"clinical_onset"}:
        raise ValueError("Fit-2 RNN target must be clinical-onset aligned")
    if set(frame.band.astype(str)) != {"broadband_1_150"}:
        raise ValueError("Fit-2 RNN target must be strict BB 1-150")
    if not frame.mirror_reselected_each_draw.astype(bool).all():
        raise ValueError("accepted all-contact null did not reselect mirror per draw")
    if not frame.ab_max_reselected_each_draw.astype(bool).all():
        raise ValueError("accepted all-contact null did not reselect A/B per draw")
    frame["target_scaffold_margin_bb150"] = (
        frame.observed.astype(float) - frame.null_median.astype(float)
    )
    frame["target_scaffold_maxab_bb150"] = frame.observed.astype(float)
    frame["target_channel_shuffle_median_bb150"] = frame.null_median.astype(float)
    return frame.sort_values(["subject", "seizure_idx"]).reset_index(drop=True)


def _passed_prefix_subjects(cfg: dict) -> list[str]:
    frame = pd.read_csv(ROOT / cfg["cohort"]["prefix_attrition"])
    passed = frame.prefix_field_pass.astype(str).str.lower().isin(("true", "1", "yes"))
    return sorted(frame.loc[passed, "subject"].astype(str))


def build_subject(subject: str, cfg: dict, targets: pd.DataFrame, out: Path):
    dataset, _sid = subject.split("_", 1)
    subject_targets = targets[targets.subject.astype(str) == subject].copy()
    base = {
        "dataset": dataset,
        "subject": subject,
        "n_candidate_targets": int(len(subject_targets)),
        "n_eligible_histories": 0,
        "gate0_pass": False,
        "reason": "",
    }
    if subject_targets.empty:
        base["reason"] = "no_strict_bb150_target"
        return base, [], []

    events = load_subject_propagation_events(_raw_subject_dir(subject))
    times = np.asarray(events["event_abs_times"], float)
    ranks = np.asarray(events["ranks"], float)
    bools = np.asarray(events["bools"], bool)
    lag_raw = np.asarray(events["lag_raw"], float)
    names = [str(x) for x in events["channel_names"]]
    inventory = _inventory(subject)
    timezone = "Europe/Berlin" if dataset == "epilepsiae" else "Asia/Shanghai"
    block_good = eligible_blocks(
        events,
        _seizure_intervals(subject, inventory),
        post_guard_sec=float(cfg["cohort"]["seizure_guard_post_minutes"]) * 60,
        timezone=timezone,
    )
    prefix_blocks = choose_prefix_blocks(
        events,
        block_good,
        cumulative_hours=int(cfg["cohort"]["calibration_hours"]),
    )
    base.update(
        {
            "n_axis_contacts": len(names),
            "n_events_total": int(times.size),
            "n_blocks_total": int(events["n_blocks_used"]),
            "n_blocks_definite_interictal": int(np.sum(block_good)),
            "n_prefix_blocks": int(prefix_blocks.size),
        }
    )
    if len(names) < int(cfg["cohort"]["min_axis_contacts"]):
        base["reason"] = "too_few_axis_contacts"
        return base, [], []
    if prefix_blocks.size < int(cfg["cohort"]["calibration_hours"]):
        base["reason"] = "insufficient_definite_interictal_prefix_blocks"
        return base, [], []
    prefix_idx = np.flatnonzero(
        _event_mask_from_blocks(events, prefix_blocks) & np.isfinite(times)
    )
    prefix_idx = prefix_idx[np.argsort(times[prefix_idx])]
    base["prefix_events"] = int(prefix_idx.size)
    if prefix_idx.size < int(cfg["cohort"]["calibration_min_events"]):
        base["reason"] = "too_few_prefix_events"
        return base, [], []

    seed = int(cfg["model"]["seeds"][0]) ^ zlib.crc32(subject.encode("utf-8"))
    try:
        axis = derive_prefix_axis(
            ranks,
            bools,
            prefix_idx,
            seed=seed,
            min_cluster_fraction=float(cfg["cohort"]["calibration_min_cluster_fraction"]),
        )
        split_r = axis_split_stability(ranks, bools, prefix_idx, seed=seed)
    except Exception as exc:
        base["reason"] = f"prefix_axis:{type(exc).__name__}:{exc}"
        return base, [], []
    base["prefix_seed_ami"] = float(axis["seed_ami"])
    base["prefix_split_axis_correlation"] = float(split_r)
    if float(axis["seed_ami"]) < float(cfg["cohort"]["calibration_min_seed_ami"]):
        base["reason"] = "prefix_seed_instability"
        return base, [], []
    if (
        not np.isfinite(split_r)
        or float(split_r)
        < float(cfg["cohort"]["calibration_min_split_axis_correlation"])
    ):
        base["reason"] = "prefix_split_axis_instability"
        return base, [], []

    freq = load_frequency_centroid(_raw_subject_dir(subject), names)
    if freq.shape != ranks.shape:
        freq = np.full_like(ranks, np.nan)
        frequency_status = f"shape_mismatch:{freq.shape}!={ranks.shape}"
    else:
        frequency_status = "available_as_frequency_centroid_not_energy"
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
    prefix_end = float(
        np.max(np.asarray(events["block_start_times"], float)[prefix_blocks]) + 3600
    )
    lookback = float(cfg["history"]["primary_lookback_minutes"]) * 60
    cutoff = float(cfg["history"]["primary_cutoff_minutes"]) * 60
    max_events = int(cfg["history"]["max_events_per_history"])
    history_event_good = (
        block_good[np.asarray(events["block_ids"], int)] & np.isfinite(times)
    )
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
    event_audit = []
    for target in subject_targets.itertuples():
        idx = int(target.seizure_idx)
        audit = {
            "dataset": dataset,
            "subject": subject,
            "seizure_idx": idx,
            "eligible_history_target_pair": False,
            "reason": "",
            "n_history_events": 0,
        }
        if idx >= len(inventory):
            audit["reason"] = "seizure_idx_out_of_inventory"
            event_audit.append(audit)
            continue
        clinical_onset = _float(inventory[idx], "clin_onset_epoch")
        if not np.isfinite(clinical_onset):
            audit["reason"] = "missing_clinical_onset"
            event_audit.append(audit)
            continue
        if clinical_onset < prefix_end + lookback + cutoff:
            audit["reason"] = "lookback_not_strictly_after_prefix"
            event_audit.append(audit)
            continue
        hmask = (
            history_event_good
            & (times >= clinical_onset - lookback - cutoff)
            & (times <= clinical_onset - cutoff)
        )
        hidx = np.flatnonzero(hmask)
        hidx = hidx[np.argsort(times[hidx])]
        audit["n_history_events"] = int(hidx.size)
        if hidx.size < int(cfg["cohort"]["min_history_events"]):
            audit["reason"] = "too_few_definite_interictal_history_events"
            event_audit.append(audit)
            continue
        if hidx.size > max_events:
            take = np.linspace(0, hidx.size - 1, max_events).round().astype(int)
            hidx = hidx[np.unique(take)]
        arrays[f"history_features__{idx}"] = features_z[hidx]
        arrays[f"history_times__{idx}"] = times[hidx].astype(np.float64)
        seizure_rows.append(
            {
                "dataset": dataset,
                "subject": subject,
                "seizure_idx": idx,
                "clinical_onset_epoch": clinical_onset,
                "n_history_events": int(hidx.size),
                "history_start_epoch": float(times[hidx[0]]),
                "history_end_epoch": float(times[hidx[-1]]),
                "target_scaffold_margin_bb150": float(
                    target.target_scaffold_margin_bb150
                ),
                "target_scaffold_maxab_bb150": float(
                    target.target_scaffold_maxab_bb150
                ),
                "target_channel_shuffle_median_bb150": float(
                    target.target_channel_shuffle_median_bb150
                ),
            }
        )
        audit["eligible_history_target_pair"] = True
        audit["reason"] = "ok"
        audit["n_history_events"] = int(hidx.size)
        event_audit.append(audit)

    base["n_eligible_histories"] = len(seizure_rows)
    if not seizure_rows:
        base["reason"] = "no_clinical_onset_history_target_pair"
        return base, [], event_audit
    base["gate0_pass"] = True
    base["reason"] = "ok"
    (out / "per_subject").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "per_subject" / f"{subject}.npz", **arrays)
    axis_record = {
        "contract": cfg["contract"]["name"],
        "subject": subject,
        "axis_source": "prefix_only_masked_rank_k2",
        "prefix_block_ids": prefix_blocks,
        "prefix_end_epoch": prefix_end,
        "prefix_event_count": int(prefix_idx.size),
        "seed": seed,
        "seed_ami": float(axis["seed_ami"]),
        "split_axis_correlation": float(split_r),
        "cluster_fractions": axis["cluster_fractions"],
        "channel_names": names,
        "template_a": axis["template_a"],
        "template_b": axis["template_b"],
        "support_a": axis["support_a"],
        "support_b": axis["support_b"],
        "support_q": axis["support_q"],
        "direction_basis": axis["direction_basis"],
        "axis_coordinate": axis["axis_coordinate"],
        "frequency_feature_status": frequency_status,
        "target_information_used_to_build_axis": False,
    }
    (out / "per_subject" / f"{subject}.json").write_text(
        json.dumps(_jsonable(axis_record), indent=2, ensure_ascii=False) + "\n"
    )
    return base, seizure_rows, event_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_state_conditioned_predictor_fit2.yaml",
    )
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out = (
        args.out_dir
        if args.out_dir is not None and args.out_dir.is_absolute()
        else ROOT / args.out_dir
        if args.out_dir is not None
        else ROOT / cfg["outputs"]["dataset"]
    )
    out.mkdir(parents=True, exist_ok=True)
    targets = load_targets(cfg)
    subjects = args.subjects or _passed_prefix_subjects(cfg)
    attrition, seizure_rows, event_audit = [], [], []
    for subject in subjects:
        print(f"[fit2-rnn-dataset] {subject}", flush=True)
        try:
            row, rows, audit_rows = build_subject(subject, cfg, targets, out)
        except Exception as exc:
            row = {
                "dataset": subject.split("_", 1)[0],
                "subject": subject,
                "gate0_pass": False,
                "reason": f"unhandled:{type(exc).__name__}:{exc}",
            }
            rows = []
            audit_rows = []
        attrition.append(row)
        seizure_rows.extend(rows)
        event_audit.extend(audit_rows)
        print(
            f"  -> {row.get('reason')} | targets={row.get('n_candidate_targets', 0)} "
            f"histories={row.get('n_eligible_histories', 0)}",
            flush=True,
        )
    attrition_frame = pd.DataFrame(attrition)
    target_frame = pd.DataFrame(seizure_rows)
    attrition_frame.to_csv(out / "gate0_attrition.csv", index=False)
    target_frame.to_csv(out / "seizure_targets.csv", index=False)
    pd.DataFrame(event_audit).to_csv(out / "event_attrition.csv", index=False)
    manifest = {
        "contract": cfg["contract"]["name"],
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": _sha256(args.config),
        "target_table": cfg["cohort"]["parent_event_table"],
        "target_table_sha256": _sha256(ROOT / cfg["cohort"]["parent_event_table"]),
        "n_candidate_subjects": len(subjects),
        "n_gate0_subjects": int(attrition_frame.gate0_pass.astype(bool).sum()),
        "n_candidate_strict_bb150_events": int(len(targets)),
        "n_history_target_pairs": int(len(target_frame)),
        "target_label": cfg["target"]["primary_label_column"],
        "leakage_guards": {
            "axis": "chronological definite-interictal prefix only",
            "history": "clinical-onset [-65,-5] min",
            "target": "strict BB1-150 clinical-onset maxAB minus all-contact shuffle median",
            "target_information_used_to_build_axis": False,
            "phantom_rank": "eventsBool-masked local rerank",
        },
        "claim_boundary": (
            "seizure-conditioned scaffold-expression strength; "
            "not signed direction and not continuous-time seizure forecasting"
        ),
    }
    (out / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
