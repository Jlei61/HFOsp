#!/usr/bin/env python
"""Build leakage-safe prefix-only TA/TB field records for Figure 6 Fit 2.

Only the interictal calibration prefix is rebuilt. Clinical-onset activation,
phenotype labels, field scoring, mirror/maxAB selection, and spatial nulls are
not read here. The output uses the accepted interictal-field schema so the
existing clinical-onset BB150 scorer can consume it without a second estimator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
)
from src.interictal_propagation import load_subject_propagation_events
from src.topic5_state_conditioned_rnn import axis_split_stability, derive_prefix_axis
from src.topic5_template_axis_field import build_interictal_template_field_record


CANONICAL_FIELD_ROOT = (
    ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
)
CANONICAL_BB150_ROOT = (
    ROOT / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
)
PARENT_EVENT_TABLE = (
    ROOT
    / "results/topic5_ictal_recruitment/tspectral_field_concordance/"
    "clinical_onset_gradient_field_cohort_stat_event.csv"
)
DEFAULT_OUT = (
    ROOT / "results/topic5_state_conditioned_predictor/fit2_prefix_scaffold"
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


def parent_event_contract(path: Path = PARENT_EVENT_TABLE) -> pd.DataFrame:
    """Return the unique accepted parent events with their frozen phenotype."""
    frame = pd.read_csv(path)
    pooled = frame[frame.group_id == "all_phenotype_matched"][
        ["dataset", "subject", "seizure_idx"]
    ].drop_duplicates()
    strict = set(
        map(
            tuple,
            frame[frame.group_id == "strict_broadband"][
                ["subject", "seizure_idx"]
            ].to_numpy(),
        )
    )
    gamma = set(
        map(
            tuple,
            frame[frame.group_id == "gamma_nonbroadband"][
                ["subject", "seizure_idx"]
            ].to_numpy(),
        )
    )
    rows = []
    for row in pooled.itertuples():
        key = (str(row.subject), int(row.seizure_idx))
        phenotype = "strict_broadband" if key in strict else "gamma_nonbroadband"
        if key not in strict and key not in gamma:
            raise ValueError(f"parent event lacks a frozen phenotype: {key}")
        rows.append(
            {
                "dataset": str(row.dataset),
                "subject": str(row.subject),
                "seizure_idx": int(row.seizure_idx),
                "phenotype": phenotype,
            }
        )
    out = pd.DataFrame(rows).sort_values(["subject", "seizure_idx"]).reset_index(drop=True)
    if out.subject.nunique() != 17 or len(out) != 167:
        raise ValueError(
            f"parent contract drift: expected 17/167, got "
            f"{out.subject.nunique()}/{len(out)}"
        )
    return out


def event_primary_onset(row: dict, dataset: str) -> float:
    """Clinical onset is primary; Yuquan has no clinical annotation."""
    if dataset == "epilepsiae":
        onset = _float(row, "clin_onset_epoch")
        return onset
    return _float(row, "eeg_onset_epoch")


def events_after_prefix(
    subject: str,
    parent: pd.DataFrame,
    prefix_end_epoch: float,
    inventory: list[dict],
) -> list[int]:
    """Keep only exact parent events occurring strictly after calibration."""
    dataset = subject.split("_", 1)[0]
    allowed = []
    rows = parent[parent.subject == subject]
    for row in rows.itertuples():
        idx = int(row.seizure_idx)
        if idx >= len(inventory):
            continue
        onset = event_primary_onset(inventory[idx], dataset)
        if np.isfinite(onset) and onset > float(prefix_end_epoch):
            allowed.append(idx)
    return sorted(set(allowed))


def _aligned_geometry(
    accepted: dict, names: list[str]
) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    """Map prefix channels to the accepted, ictal-blind coordinate inventory."""
    accepted_names = [str(x) for x in accepted["names"]]
    coords = np.asarray(accepted["coords"], float)
    shafts = [str(x) for x in accepted["shafts"]]
    lookup = {name: i for i, name in enumerate(accepted_names)}
    keep = np.asarray([name in lookup for name in names], bool)
    mapped_names = [name for name in names if name in lookup]
    mapped_idx = [lookup[name] for name in mapped_names]
    return mapped_names, coords[mapped_idx], [shafts[i] for i in mapped_idx], keep


def _safe_symlink(target: Path, link: Path) -> None:
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f"existing symlink points elsewhere: {link}")
        return
    if link.exists():
        raise RuntimeError(f"refusing to replace existing cache view file: {link}")
    link.symlink_to(target.resolve())


def build_subject(
    subject: str,
    cfg: dict,
    parent: pd.DataFrame,
    out: Path,
) -> dict:
    dataset, sid = subject.split("_", 1)
    row = {
        "dataset": dataset,
        "subject": subject,
        "prefix_field_pass": False,
        "reason": "",
        "n_parent_events": int((parent.subject == subject).sum()),
        "n_events_after_prefix": 0,
    }
    accepted_path = CANONICAL_FIELD_ROOT / f"{subject}.json"
    if not accepted_path.exists():
        row["reason"] = "missing_accepted_coordinate_field"
        return row
    accepted = json.loads(accepted_path.read_text())
    raw_dir = _raw_subject_dir(subject)
    events = load_subject_propagation_events(raw_dir)
    times = np.asarray(events["event_abs_times"], float)
    ranks = np.asarray(events["ranks"], float)
    bools = np.asarray(events["bools"], bool)
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
    row["n_prefix_blocks"] = int(prefix_blocks.size)
    if prefix_blocks.size < int(cfg["cohort"]["calibration_hours"]):
        row["reason"] = "insufficient_definite_interictal_prefix_blocks"
        return row
    prefix_idx = np.flatnonzero(
        _event_mask_from_blocks(events, prefix_blocks) & np.isfinite(times)
    )
    prefix_idx = prefix_idx[np.argsort(times[prefix_idx])]
    row["prefix_events"] = int(prefix_idx.size)
    if prefix_idx.size < int(cfg["cohort"]["calibration_min_events"]):
        row["reason"] = "too_few_prefix_events"
        return row
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
        row["reason"] = f"prefix_axis:{type(exc).__name__}:{exc}"
        return row
    row["prefix_seed_ami"] = float(axis["seed_ami"])
    row["prefix_split_axis_correlation"] = float(split_r)
    if axis["seed_ami"] < float(cfg["cohort"]["calibration_min_seed_ami"]):
        row["reason"] = "prefix_seed_instability"
        return row
    if (
        not np.isfinite(split_r)
        or split_r < float(cfg["cohort"]["calibration_min_split_axis_correlation"])
    ):
        row["reason"] = "prefix_split_axis_instability"
        return row

    mapped_names, coords, shafts, keep = _aligned_geometry(accepted, names)
    if len(mapped_names) < int(cfg["cohort"]["min_axis_contacts"]):
        row["reason"] = "too_few_coordinate_mapped_prefix_contacts"
        return row
    record = build_interictal_template_field_record(
        subject_id=sid,
        dataset=dataset,
        subject=subject,
        stable_k=2,
        names=mapped_names,
        coords=coords,
        rank_ta=np.asarray(axis["template_a"], float)[keep],
        rank_tb=np.asarray(axis["template_b"], float)[keep],
        shafts=shafts,
        support_ta=np.asarray(axis["support_a"], float)[keep],
        support_tb=np.asarray(axis["support_b"], float)[keep],
        support_source="chronological_definite_interictal_prefix_only",
        template_event_counts={
            "prefix_total": int(prefix_idx.size),
            "cluster_a": int(np.sum(np.asarray(axis["labels"]) == 0)),
            "cluster_b": int(np.sum(np.asarray(axis["labels"]) == 1)),
        },
        seed=seed,
    )
    if record.get("interictal_field", {}).get("status") != "ok":
        row["reason"] = (
            "prefix_field:"
            + str(record.get("interictal_field", {}).get("status", "unavailable"))
        )
        return row
    prefix_end = float(
        np.max(np.asarray(events["block_start_times"], float)[prefix_blocks]) + 3600.0
    )
    allowed = events_after_prefix(subject, parent, prefix_end, inventory)
    if not allowed:
        row["reason"] = "no_parent_event_after_prefix"
        return row

    record["prefix_provenance"] = {
        "prefix_block_ids": np.asarray(prefix_blocks, int),
        "prefix_end_epoch": prefix_end,
        "prefix_event_count": int(prefix_idx.size),
        "seed_ami": float(axis["seed_ami"]),
        "split_axis_correlation": float(split_r),
        "target_information_used": False,
        "accepted_coordinate_source": str(accepted_path.relative_to(ROOT)),
    }
    (out / "per_subject").mkdir(parents=True, exist_ok=True)
    (out / "per_subject" / f"{subject}.json").write_text(
        json.dumps(_jsonable(record), ensure_ascii=False, indent=2) + "\n"
    )

    canonical_meta_path = CANONICAL_BB150_ROOT / f"{subject}.json"
    canonical_npz_path = CANONICAL_BB150_ROOT / f"{subject}.npz"
    if not canonical_meta_path.exists() or not canonical_npz_path.exists():
        row["reason"] = "missing_canonical_bb150_cache"
        return row
    meta = json.loads(canonical_meta_path.read_text())
    canonical_eligible = {int(v) for v in meta.get("eligible_idxs", [])}
    allowed = sorted(canonical_eligible & set(allowed))
    if not allowed:
        row["reason"] = "no_canonical_bb150_event_after_prefix"
        return row
    meta["eligible_idxs"] = allowed
    meta["prefix_fit2_filter"] = {
        "subject": subject,
        "prefix_end_epoch": prefix_end,
        "exact_parent_event_only": True,
        "clinical_onset_primary": dataset == "epilepsiae",
    }
    view = out / "bb150_cache_view"
    view.mkdir(parents=True, exist_ok=True)
    (view / f"{subject}.json").write_text(
        json.dumps(_jsonable(meta), ensure_ascii=False, indent=2) + "\n"
    )
    _safe_symlink(canonical_npz_path, view / f"{subject}.npz")

    row.update(
        {
            "prefix_field_pass": True,
            "reason": "ok",
            "prefix_end_epoch": prefix_end,
            "n_field_contacts": int(record["interictal_field"]["n_contacts"]),
            "field_route": (
                "shared"
                if "shared" in record["interictal_field"]["planes"]
                else "own_fallback"
            ),
            "n_events_after_prefix": len(allowed),
            "allowed_seizure_idxs": ";".join(map(str, allowed)),
            "field_fingerprint": record["interictal_field"]["fingerprint_sha256"],
        }
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_state_conditioned_predictor.yaml",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--subjects", nargs="*", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    parent = parent_event_contract()
    subjects = args.subjects or sorted(parent.subject.unique())
    args.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in subjects:
        print(f"[prefix-field] {subject}", flush=True)
        try:
            row = build_subject(subject, cfg, parent, args.outdir)
        except Exception as exc:
            row = {
                "dataset": subject.split("_", 1)[0],
                "subject": subject,
                "prefix_field_pass": False,
                "reason": f"unhandled:{type(exc).__name__}:{exc}",
            }
        rows.append(row)
        print(
            f"  -> {row.get('reason')} | prefix={row.get('prefix_events', 0)} "
            f"events={row.get('n_events_after_prefix', 0)}",
            flush=True,
        )
    attrition = pd.DataFrame(rows)
    attrition.to_csv(args.outdir / "prefix_field_attrition.csv", index=False)
    passed = attrition[
        attrition.prefix_field_pass.astype(str).str.lower().isin(("true", "1", "yes"))
    ]
    event_rows = []
    for row in passed.itertuples():
        for idx in str(row.allowed_seizure_idxs).split(";"):
            if idx:
                event_rows.append(
                    {
                        "dataset": row.dataset,
                        "subject": row.subject,
                        "seizure_idx": int(idx),
                    }
                )
    allowlist = pd.DataFrame(event_rows).sort_values(["subject", "seizure_idx"])
    allowlist.to_csv(args.outdir / "fit2_parent_event_allowlist.csv", index=False)
    manifest = {
        "contract": "topic5_fig6_fit2_prefix_only_scaffold_v2",
        "config": str(args.config.relative_to(ROOT)),
        "config_sha256": _sha256(args.config),
        "parent_event_table": str(PARENT_EVENT_TABLE.relative_to(ROOT)),
        "parent_event_table_sha256": _sha256(PARENT_EVENT_TABLE),
        "accepted_coordinate_root": str(CANONICAL_FIELD_ROOT.relative_to(ROOT)),
        "clinical_onset_primary": True,
        "bb150_primary_control": True,
        "n_candidate_subjects": len(subjects),
        "n_prefix_field_subjects": int(len(passed)),
        "n_parent_events_after_prefix": int(len(allowlist)),
        "target_information_used_to_build_fields": False,
        "outputs": {
            "fields": str((args.outdir / "per_subject").relative_to(ROOT)),
            "bb150_cache_view": str(
                (args.outdir / "bb150_cache_view").relative_to(ROOT)
            ),
            "allowlist": str(
                (args.outdir / "fit2_parent_event_allowlist.csv").relative_to(ROOT)
            ),
        },
    }
    (args.outdir / "prefix_field_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
