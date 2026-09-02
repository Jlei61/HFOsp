#!/usr/bin/env python3
"""Run the Topic 5.2 E0 complete-state replay audit over all 630 cells."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import (  # noqa: E402
    DecoderState,
    atomic_write_csv,
    atomic_write_json,
    decoder_snapshot,
    load_frozen_cell,
    manual_teacher_forced_trace,
    parameter_state_sha256,
    select_replay_event_indices,
)
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


PARENT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OUT = ROOT / "results/topic5_latent_propagation_landscape_v0_2"


def audit_cell(row: pd.Series, device: torch.device) -> dict[str, object]:
    started = time.perf_counter()
    model, decoder, _, _ = load_frozen_cell(PARENT, row, device)
    model_before = parameter_state_sha256(model)
    decoder_before = parameter_state_sha256(decoder)
    with np.load(PARENT / "cache" / row.fit_id / "events.npz", allow_pickle=False) as handle:
        ranks_all = np.asarray(handle["ranks"])
        split_all = np.asarray(handle["split"])
    keep = split_all >= 0
    ranks = ranks_all[keep]
    split = split_all[keep]
    selected = select_replay_event_indices(ranks, split, per_split=10)
    tensors = build_event_tensors(ranks[selected])
    x = tensors["x"].to(device)
    recruited = tensors["recruited"].to(device)
    valid = tensors["valid"].to(device)
    with torch.no_grad():
        forward_logits, forward_stop = model(x, recruited, valid)
        manual = manual_teacher_forced_trace(model, x, recruited)
    logits_equal = bool(torch.equal(forward_logits, manual["pre_mask_logits"]))
    stop_equal = bool(torch.equal(forward_stop, manual["stop_logits"]))
    all_finite = bool(
        torch.isfinite(manual["hidden"]).all()
        and torch.isfinite(manual["pre_mask_logits"]).all()
        and torch.isfinite(manual["stop_logits"]).all()
    )

    event = 0
    valid_steps = int(valid[event].sum().item())
    step = 0 if valid_steps < 2 else min(valid_steps - 1, valid_steps // 2)
    q = DecoderState(
        manual["hidden"][event:event + 1, step].clone(),
        recruited[event:event + 1, step].clone(),
        step,
    )
    q_clone = q.clone()
    clone_equal = bool(
        torch.equal(q.h, q_clone.h)
        and torch.equal(q.recruited, q_clone.recruited)
        and q.rank_index == q_clone.rank_index
        and q.h.data_ptr() != q_clone.h.data_ptr()
        and q.recruited.data_ptr() != q_clone.recruited.data_ptr()
    )
    base = decoder_snapshot(model, decoder, q, force_continue=True)
    changed_r = q.clone()
    if base["picked"]:
        changed_r.recruited[0, int(base["picked"][0])] = 1.0
    else:
        changed_r.recruited[0, 0] = 1.0 - changed_r.recruited[0, 0]
    recruited_counter = decoder_snapshot(model, decoder, changed_r, force_continue=True)
    recruited_dependency = bool(
        not torch.equal(base["available"], recruited_counter["available"])
        and base["picked"] != recruited_counter["picked"]
    )
    changed_k = q.clone()
    changed_k.rank_index = min(model.n_contacts - 1, q.rank_index + 1)
    rank_counter = decoder_snapshot(model, decoder, changed_k, force_continue=True)
    rank_dependency = bool(
        not torch.equal(base["features"], rank_counter["features"])
        and (
            not torch.equal(base["stop_logit"], rank_counter["stop_logit"])
            or not torch.equal(base["size_logits"], rank_counter["size_logits"])
        )
    )
    model_after = parameter_state_sha256(model)
    decoder_after = parameter_state_sha256(decoder)
    return {
        "patient": row.patient,
        "fit_id": row.fit_id,
        "public_arm": row.public_arm,
        "seed": int(row.seed),
        "checkpoint_source": row.checkpoint_source,
        "n_replay_events": int(len(selected)),
        "n_replay_steps": int(valid.sum().item()),
        "forward_manual_logits_exact": logits_equal,
        "forward_manual_stop_exact": stop_equal,
        "hidden_logits_stop_finite": all_finite,
        "full_q_clone_exact_and_disjoint": clone_equal,
        "same_h_different_r_changes_contact_decision": recruited_dependency,
        "same_h_different_k_changes_decoder_scores": rank_dependency,
        "model_hash_unchanged": model_before == model_after,
        "decoder_hash_unchanged": decoder_before == decoder_after,
        "seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    if args.limit is not None:
        manifest = manifest.iloc[:args.limit]
    device = torch.device(args.device)
    rows = []
    for index, (_, row) in enumerate(manifest.iterrows(), start=1):
        rows.append(audit_cell(row, device))
        if index % 50 == 0:
            print(f"audited {index}/{len(manifest)}", flush=True)
    frame = pd.DataFrame(rows)
    check_columns = [
        "forward_manual_logits_exact",
        "forward_manual_stop_exact",
        "hidden_logits_stop_finite",
        "full_q_clone_exact_and_disjoint",
        "same_h_different_r_changes_contact_decision",
        "same_h_different_k_changes_decoder_scores",
        "model_hash_unchanged",
        "decoder_hash_unchanged",
    ]
    failures = {
        column: int((~frame[column].astype(bool)).sum()) for column in check_columns
    }
    complete = len(manifest) == 630
    status = "PASS" if complete and not any(failures.values()) else "FAIL"
    payload = {
        "contract": "topic5_latent_landscape_e0_replay_audit_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "device": str(device),
        "scheduled_cells": 630,
        "audited_cells": int(len(frame)),
        "complete_matrix": complete,
        "check_failures": failures,
        "total_replay_events": int(frame["n_replay_events"].sum()),
        "total_replay_steps": int(frame["n_replay_steps"].sum()),
        "seconds": float(frame["seconds"].sum()),
        "target_values_read": False,
    }
    suffix = "" if complete else f"_LIMIT{len(frame)}"
    atomic_write_csv(OUT / f"E0_REPLAY_PER_CELL{suffix}.csv", frame)
    atomic_write_json(OUT / f"E0_REPLAY_AUDIT{suffix}.json", payload)
    print(json.dumps(payload, indent=2))
    if status != "PASS" and complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
