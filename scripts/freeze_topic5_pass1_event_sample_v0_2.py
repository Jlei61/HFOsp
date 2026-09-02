#!/usr/bin/env python3
"""Freeze response-blind Pass 1 and Pass 2 event samples for Topic 5.2."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    arrays_sha256,
    atomic_write_csv,
    atomic_write_json,
    canonical_json_sha256,
    response_blind_event_sample,
    sha256_file,
)


PARENT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OUT = ROOT / "results/topic5_latent_propagation_landscape_v0_2"
CAPS = {0: 1024, 1: 512, 2: 512}
SPLIT_NAMES = {0: "axis_train", 1: "axis_validation", 2: "heldout_test"}
REFERENCE_CAP = 64


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    census = pd.read_csv(PARENT / "FULL_PARENT_FIT_CENSUS.csv")
    rows: list[pd.DataFrame] = []
    fit_audits: list[dict[str, object]] = []
    for item in census.itertuples(index=False):
        path = PARENT / "cache" / str(item.fit_id) / "events.npz"
        with np.load(path, allow_pickle=False) as events:
            ranks = np.asarray(events["ranks"])
            split = np.asarray(events["split"])
            source = np.asarray(events["event_source_index"])
            dataset = np.asarray(events["event_dataset_index"])
            abs_time = np.asarray(events["event_abs_time"])
        lengths = np.where(
            np.any(ranks >= 0, axis=1),
            np.max(np.where(ranks >= 0, ranks, -1), axis=1) + 1,
            0,
        )
        sample = response_blind_event_sample(
            patient=str(item.subject),
            split=split,
            event_source_index=source,
            event_dataset_index=dataset,
            phase_defined=lengths >= 2,
            caps=CAPS,
        )
        if sample.empty:
            raise RuntimeError(f"no phase-defined sampled event for {item.fit_id}")
        take = sample["event_array_index"].to_numpy(dtype=int)
        sample.insert(0, "subject", str(item.subject))
        sample.insert(1, "fit_id", str(item.fit_id))
        sample.insert(2, "scope", str(item.scope))
        sample["split_name"] = sample["split"].map(SPLIT_NAMES)
        sample["event_abs_time"] = abs_time[take]
        sample["n_rank_sets"] = lengths[take]
        sample["pass2_reference_event"] = False
        test = sample["split"].eq(2)
        test_order = sample.loc[test].sort_values(
            ["identity_sha256", "event_array_index"], kind="mergesort"
        ).index[:REFERENCE_CAP]
        sample.loc[test_order, "pass2_reference_event"] = True
        sample["target_values_read"] = False
        rows.append(sample)
        fit_audits.append({
            "subject": str(item.subject),
            "fit_id": str(item.fit_id),
            "scope": str(item.scope),
            "events_sha256": sha256_file(path),
            "eligible_phase_events": int(np.count_nonzero((split >= 0) & (lengths >= 2))),
            "selected_events": int(len(sample)),
            "selected_event_steps": int(lengths[take].sum()),
            "reference_events": int(sample["pass2_reference_event"].sum()),
        })
    frame = pd.concat(rows, ignore_index=True)
    audit_frame = pd.DataFrame(fit_audits)

    # Split-plane fits of the same patient must use identical events.
    own_mismatch: list[str] = []
    for patient, group in census[census["scope"].isin(["own_a", "own_b"])].groupby("subject"):
        if set(group["scope"]) != {"own_a", "own_b"}:
            own_mismatch.append(f"{patient}:scope")
            continue
        fit_ids = group["fit_id"].astype(str).tolist()
        identities = [
            set(frame.loc[frame["fit_id"].eq(fit), "identity_sha256"])
            for fit in fit_ids
        ]
        if identities[0] != identities[1]:
            own_mismatch.append(str(patient))
    if own_mismatch:
        raise RuntimeError(f"own_a/own_b event samples differ: {own_mismatch}")

    summary = {
        "contract": "topic5_pass1_response_blind_event_sample_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "selector_inputs": [
            "patient", "split", "event_source_index", "event_dataset_index", "phase_defined"
        ],
        "forbidden_selector_inputs": ["mode", "u_e", "hidden", "effect", "response", "target"],
        "caps": {SPLIT_NAMES[k]: v for k, v in CAPS.items()},
        "pass2_reference_event_cap_per_fit": REFERENCE_CAP,
        "n_patients": int(frame["subject"].nunique()),
        "n_fits": int(frame["fit_id"].nunique()),
        "selected_events": int(len(frame)),
        "selected_event_steps_fit_level": int(audit_frame["selected_event_steps"].sum()),
        "projected_cell_event_steps": int(audit_frame["selected_event_steps"].sum() * 15),
        "reference_events_fit_level": int(audit_frame["reference_events"].sum()),
        "own_pair_event_identity_exact": True,
        "selection_sha256": arrays_sha256({
            "event_array_index": frame["event_array_index"].to_numpy(np.int64),
            "split": frame["split"].to_numpy(np.int8),
            "event_source_index": frame["event_source_index"].to_numpy(np.int64),
            "event_dataset_index": frame["event_dataset_index"].to_numpy(np.int64),
            "pass2_reference_event": frame["pass2_reference_event"].to_numpy(np.uint8),
        }),
        "caps_sha256": canonical_json_sha256(CAPS),
        "target_values_read": False,
    }
    if args.write:
        atomic_write_csv(OUT / "PASS1_EVENT_SAMPLE_MANIFEST.csv", frame)
        atomic_write_csv(OUT / "PASS1_EVENT_SAMPLE_PER_FIT.csv", audit_frame)
        atomic_write_json(OUT / "PASS1_EVENT_SAMPLE_AUDIT.json", summary)
    print(json.dumps({**summary, "written": bool(args.write)}, indent=2))


if __name__ == "__main__":
    main()
