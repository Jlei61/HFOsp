#!/usr/bin/env python3
"""Target-free recording-block held-out sensitivity for Topic 5.1 v0.5.

The frozen v0.5 split is chronological at the event level.  This audit keeps
the fitted models unchanged, but restricts evaluation to test events whose raw
recording block contributes no train or validation event to the same fit.  It
therefore asks whether the reported contrast survives removal of the single
train/test boundary-block overlap; it is not presented as a newly refitted
session-held-out model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyse_topic5_multiscale_interictal_v0_5 import interaction, paired_summary  # noqa: E402
from scripts.build_topic5_we_cache import lagpat_dir  # noqa: E402
from src.interictal_propagation import load_subject_propagation_events


DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OLD_ROOT = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
ARMS = (
    "L2M_MACRO_MATCHED_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def strict_unseen_test_events(
    split: np.ndarray,
    event_source_index: np.ndarray,
    raw_block_ids: np.ndarray,
) -> tuple[np.ndarray, dict]:
    """Return indices in the compact ``split>=0`` event sequence.

    A test event is retained only when its raw recording block has no event
    marked train or validation for this fit.  Events excluded from a fit
    (split=-1, e.g. the other train-only mode) do not count as model exposure.
    """
    split = np.asarray(split, dtype=np.int8)
    event_source_index = np.asarray(event_source_index, dtype=np.int64)
    raw_block_ids = np.asarray(raw_block_ids, dtype=np.int64)
    if split.ndim != 1 or event_source_index.shape != split.shape:
        raise ValueError("split and event_source_index must be aligned vectors")
    if len(event_source_index) and (
        event_source_index.min() < 0 or event_source_index.max() >= len(raw_block_ids)
    ):
        raise ValueError("event_source_index lies outside raw block vector")
    event_blocks = raw_block_ids[event_source_index]
    exposed_blocks = set(event_blocks[np.isin(split, (0, 1))].tolist())
    compact = np.flatnonzero(split >= 0)
    compact_split = split[compact]
    compact_blocks = event_blocks[compact]
    strict_mask = np.asarray([
        int(partition) == 2 and int(block) not in exposed_blocks
        for partition, block in zip(compact_split, compact_blocks)
    ], dtype=bool)
    ordinary_test = compact_split == 2
    return np.flatnonzero(strict_mask), {
        "n_compact_events": int(len(compact)),
        "n_test_events": int(ordinary_test.sum()),
        "n_strict_unseen_block_test_events": int(strict_mask.sum()),
        "n_test_events_removed_boundary_block": int(ordinary_test.sum() - strict_mask.sum()),
        "n_exposed_blocks": int(len(exposed_blocks)),
        "n_strict_test_blocks": int(len(set(compact_blocks[strict_mask].tolist()))),
    }


def metrics_path(out: Path, old: Path, fit_id: str, arm: str, seed: int,
                 reused: set[str]) -> Path:
    if fit_id in reused and arm == "L3_LOCAL_PLUS_LEARNED_LR":
        return old / "per_fit" / fit_id / arm / f"seed{seed}" / "metrics.json"
    return out / "formal_units" / fit_id / arm / f"seed{seed}" / "metrics.json"


def summarize_decisions(path: Path, strict_events: set[int], r_local_mm: float) -> dict:
    rows = json.loads((path.parent / "distance_decisions.json").read_text())
    selected = [row for row in rows if int(row["event_index"]) in strict_events]
    distal = [row for row in selected if (
        np.isfinite(row["frontier_distance_mm"])
        and float(row["frontier_distance_mm"]) > r_local_mm
    )]
    keys = [(int(row["event_index"]), int(row["rank_index"])) for row in selected]
    distal_keys = [(int(row["event_index"]), int(row["rank_index"])) for row in distal]
    return {
        "all_contact_nll": float(np.mean([row["contact_nll"] for row in selected]))
        if selected else float("nan"),
        "distal_contact_nll": float(np.mean([row["contact_nll"] for row in distal]))
        if distal else float("nan"),
        "all_n": int(len(selected)),
        "distal_n": int(len(distal)),
        "all_support_sha256": hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
        "distal_support_sha256": hashlib.sha256(json.dumps(distal_keys).encode()).hexdigest(),
    }


def aggregate_patient(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ("all_contact_nll", "distal_contact_nll", "all_n", "distal_n")
    fit = rows.groupby(["subject", "fit_id", "scope", "arm"], sort=False)[list(metrics)].median().reset_index()
    patient = fit.groupby(["subject", "arm"], sort=False)[list(metrics)].mean().reset_index()
    return fit, patient


def contrast_rows(patient: pd.DataFrame) -> pd.DataFrame:
    pivot = patient.pivot(index="subject", columns="arm")
    definitions = {
        "L3_vs_L2m_all": (
            pivot["all_contact_nll"]["L2M_MACRO_MATCHED_RANDOM_LR"]
            - pivot["all_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"]
        ),
        "L3_vs_L2m_distal": (
            pivot["distal_contact_nll"]["L2M_MACRO_MATCHED_RANDOM_LR"]
            - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"]
        ),
        "L3_vs_suffix_all": (
            pivot["all_contact_nll"]["C_L3_ORDER_SHUFFLED"]
            - pivot["all_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"]
        ),
        "L3_vs_suffix_distal": (
            pivot["distal_contact_nll"]["C_L3_ORDER_SHUFFLED"]
            - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"]
        ),
    }
    return pd.concat([
        pd.DataFrame({"subject": series.index, "contrast": label, "gain_nats": series.values})
        for label, series in definitions.items()
    ], ignore_index=True)


def interaction_from(contrast: pd.DataFrame, label: str, J: pd.DataFrame,
                     eligible: set[str] | None = None, seed: int = 20260814) -> dict:
    table = contrast.loc[contrast.contrast == label, ["subject", "gain_nats"]].merge(
        J[["subject", "J_lat_exceedance_burden"]], on="subject", validate="one_to_one"
    )
    if eligible is not None:
        table = table.loc[table.subject.isin(eligible)]
    result = interaction(table.J_lat_exceedance_burden, table.gain_nats, seed=seed)
    result["subjects"] = sorted(table.subject.astype(str).tolist())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    args = parser.parse_args()
    out, old = args.out_root.resolve(), args.old_root.resolve()
    if (out / "TARGET_UNSEAL_AUTHORIZED.json").exists():
        raise RuntimeError("block-heldout sensitivity must finish before target authorization")
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(out / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    reused = set(reuse.loc[reuse.checkpoint_reuse_eligible.astype(bool), "fit_id"].astype(str))
    raw_records: dict[str, dict[str, np.ndarray]] = {}
    rows, audits = [], []
    for fit in census.itertuples():
        subject = str(fit.subject)
        if subject not in raw_records:
            raw = load_subject_propagation_events(lagpat_dir(subject))
            raw_records[subject] = {
                "block_ids": np.asarray(raw["block_ids"], dtype=np.int64),
                "event_abs_times": np.asarray(raw["event_abs_times"], dtype=np.float64),
            }
        events = np.load(out / "cache" / fit.fit_id / "events.npz", allow_pickle=False)
        source_index = np.asarray(events["event_source_index"], dtype=np.int64)
        raw_times = raw_records[subject]["event_abs_times"][source_index]
        cache_times = np.asarray(events["event_abs_time"], dtype=np.float64)
        finite = np.isfinite(raw_times) & np.isfinite(cache_times)
        if not np.allclose(raw_times[finite], cache_times[finite], atol=1e-6, rtol=0):
            raise RuntimeError(f"{fit.fit_id}: raw/cache event time join failed")
        strict, audit = strict_unseen_test_events(
            events["split"], source_index, raw_records[subject]["block_ids"]
        )
        strict_set = set(strict.tolist())
        audits.append({"subject": subject, "fit_id": fit.fit_id, "scope": fit.scope, **audit})
        for arm in ARMS:
            for seed in range(3):
                path = metrics_path(out, old, fit.fit_id, arm, seed, reused)
                metrics = json.loads(path.read_text())
                if metrics.get("target_values_read") is not False:
                    raise RuntimeError(f"target marker not false: {path}")
                values = summarize_decisions(path, strict_set, float(fit.r_local_mm))
                rows.append({
                    "subject": subject, "fit_id": fit.fit_id, "scope": fit.scope,
                    "arm": arm, "seed": seed, **values,
                })
    rows = pd.DataFrame(rows)
    audits = pd.DataFrame(audits)
    if rows.groupby(["fit_id", "seed"]).all_support_sha256.nunique().ne(1).any():
        raise RuntimeError("arms do not share the strict block-heldout all-decision support")
    if rows.groupby(["fit_id", "seed"]).distal_support_sha256.nunique().ne(1).any():
        raise RuntimeError("arms do not share the strict block-heldout distal support")
    fit, patient = aggregate_patient(rows)
    contrasts = contrast_rows(patient)
    J = pd.read_csv(out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv")
    contact_min = census.groupby("subject").n_joint_contacts.min()
    geometry_2d = set(census.groupby("subject").filter(
        lambda frame: bool(np.all(frame.geometry_class == "TWO_DIMENSIONAL"))
    ).subject.astype(str))
    primary_table = contrasts.loc[contrasts.contrast == "L3_vs_L2m_distal"].merge(
        patient.loc[patient.arm == "L3_LOCAL_PLUS_LEARNED_LR", ["subject", "distal_n"]],
        on="subject", validate="one_to_one",
    )
    eligible = set(primary_table.loc[primary_table.distal_n >= 20, "subject"].astype(str))
    primary_with_J = primary_table.loc[primary_table.subject.isin(eligible)].merge(
        J[["subject", "J_lat_exceedance_burden"]], on="subject", validate="one_to_one"
    )
    highest_J = str(primary_with_J.sort_values("J_lat_exceedance_burden").iloc[-1].subject)
    comparison_summary = {
        label: paired_summary(group.gain_nats.to_numpy())
        for label, group in contrasts.groupby("contrast", sort=False)
    }
    summary = {
        "contract": "topic5_v0_5_strict_unseen_recording_block_evaluation_sensitivity",
        "interpretation": "frozen-model evaluation sensitivity; not a block-heldout refit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
        "patients": int(patient.subject.nunique()),
        "fits": int(fit.fit_id.nunique()),
        "comparisons": comparison_summary,
        "primary_interaction": interaction_from(
            contrasts, "L3_vs_L2m_distal", J, eligible, seed=2026081401
        ),
        "fixed_sensitivities": {
            "exclude_6_7_contact_patients": interaction_from(
                contrasts, "L3_vs_L2m_distal", J,
                eligible & set(contact_min[contact_min >= 8].index.astype(str)), seed=2026081402,
            ),
            "exclude_highest_J_patient": interaction_from(
                contrasts, "L3_vs_L2m_distal", J, eligible - {highest_J}, seed=2026081403,
            ),
            "two_dimensional_geometry_only": interaction_from(
                contrasts, "L3_vs_L2m_distal", J, eligible & geometry_2d, seed=2026081404,
            ),
        },
        "highest_J_patient_removed": highest_J,
        "fit_audit": {
            "test_events": int(audits.n_test_events.sum()),
            "strict_test_events": int(audits.n_strict_unseen_block_test_events.sum()),
            "removed_boundary_block_events": int(audits.n_test_events_removed_boundary_block.sum()),
            "fits_with_at_least_one_strict_test_event": int((audits.n_strict_unseen_block_test_events > 0).sum()),
        },
    }
    rows.to_csv(out / "INTERICTAL_BLOCK_HELDOUT_PER_FIT_SEED.csv", index=False)
    audits.to_csv(out / "INTERICTAL_BLOCK_HELDOUT_FIT_AUDIT.csv", index=False)
    fit.to_csv(out / "INTERICTAL_BLOCK_HELDOUT_PER_FIT.csv", index=False)
    patient.to_csv(out / "INTERICTAL_BLOCK_HELDOUT_PER_PATIENT.csv", index=False)
    contrasts.to_csv(out / "INTERICTAL_BLOCK_HELDOUT_PATIENT_CONTRASTS.csv", index=False)
    summary_path = out / "INTERICTAL_BLOCK_HELDOUT_SENSITIVITY.json"
    write_json(summary_path, summary)
    manifest = {
        "status": "PASS", "target_values_read": False,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": sha256_file(Path(__file__)),
        "summary_sha256": sha256_file(summary_path),
        "source_hashes": {
            name: sha256_file(out / name) for name in (
                "FULL_PARENT_FIT_CENSUS.csv", "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv",
                "INTERICTAL_PATIENT_CONTRASTS.csv", "STAGE_E_INTERICTAL_ANALYSIS_COMPLETE.json",
            )
        },
    }
    write_json(out / "INTERICTAL_BLOCK_HELDOUT_PREFREEZE_MANIFEST.json", manifest)


if __name__ == "__main__":
    main()
