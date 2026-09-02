#!/usr/bin/env python3
"""Test whether a frozen true-grid RNN needs local edges around each ECoG patch."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_ecog_physical_neighborhood_v0_1 import (  # noqa: E402
    build_fixed_grid_model,
    enumerate_square_patches,
    matched_dispersed_directed_edge_sets,
    matched_dispersed_edge_sets,
    patch_edge_mask,
)
from src.topic5_wiring_economy_rnn import build_event_tensors  # noqa: E402


DOSES = (1.0, 0.75, 0.5, 0.0)
FIRST_ENTRY_CONTRACT = "no_patch_contact_recruited_before_next_rank_v0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parameter_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(parameter.detach().cpu().numpy()).view(np.uint8))
    return digest.hexdigest()


def event_patch_coverage(
    ranks: np.ndarray, patch_nodes: tuple[int, ...], lesion_mode: str,
) -> tuple[int, int]:
    patch = np.asarray(patch_nodes, dtype=int)
    event_count = int(np.sum(np.any(ranks[:, patch] >= 0, axis=1)))
    entering_decisions = 0
    for full_row in ranks:
        row = full_row[patch]
        if lesion_mode == "inbound_first_entry":
            length = int(full_row[full_row >= 0].max()) + 1
            for step in range(length - 1):
                patch_seen_through_current = bool(
                    np.any((row >= 0) & (row <= step))
                )
                next_has_patch = bool(np.any(row == step + 1))
                entering_decisions += int(
                    (not patch_seen_through_current) and next_has_patch
                )
        else:
            entering_decisions += len(np.unique(row[row > 0]))
    return event_count, int(entering_decisions)


@torch.no_grad()
def baseline_nll(
    model: torch.nn.Module,
    tensors: dict[str, torch.Tensor],
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    output = torch.full(tensors["valid"].shape, torch.nan, dtype=torch.float32)
    for begin in range(0, len(tensors["valid"]), int(batch_size)):
        end = min(len(tensors["valid"]), begin + int(batch_size))
        batch = {name: value[begin:end].to(device) for name, value in tensors.items()}
        logits, _ = model(batch["x"], batch["recruited"], batch["valid"])
        masked = logits.masked_fill(~batch["available"], -1e9)
        log_probability = torch.log_softmax(masked, dim=-1)
        per_step = -(log_probability * batch["target"]).sum(-1)
        per_step /= batch["target"].sum(-1).clamp_min(1.0)
        output[begin:end] = per_step.cpu()
    return output


@torch.no_grad()
def ensemble_nll(
    model: torch.nn.Module,
    node_masks: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Exact teacher-forced contact NLL for many edge lesions in parallel."""
    if int(model.state_dim) != 1 or int(model.recurrent.shape[0]) != 1:
        raise ValueError("v0.1 patch ensemble is frozen to one-state leaky RNNs")
    n_lesions = int(node_masks.shape[0])
    n_events = int(batch["x"].shape[0])
    n_nodes = int(model.n_nodes)
    recurrent = model.recurrent[0].unsqueeze(0) * node_masks
    input_gain = model.input_gain[0, :, 0]
    bias = model.bias[0]
    kappa = torch.sigmoid(model.kappa_logit)
    hidden = torch.zeros(n_lesions, n_events, n_nodes, device=batch["x"].device)
    all_nll: list[torch.Tensor] = []
    for step in range(batch["x"].shape[1]):
        injected = (batch["x"][:, step] @ model.H) * input_gain
        pre = injected.unsqueeze(0) + torch.einsum("lbi,lji->lbj", hidden, recurrent) + bias
        hidden = (1.0 - kappa) * hidden + kappa * torch.tanh(pre)
        for _ in range(int(model.microsteps) - 1):
            pre = torch.einsum("lbi,lji->lbj", hidden, recurrent) + bias
            hidden = (1.0 - kappa) * hidden + kappa * torch.tanh(pre)
        logits = model.contact_bias + model.readout_gain * torch.einsum(
            "lbn,cn->lbc", hidden, model.H
        )
        available = batch["available"][:, step]
        target = batch["target"][:, step]
        masked = logits.masked_fill(~available.unsqueeze(0), -1e9)
        log_probability = torch.log_softmax(masked, dim=-1)
        per_step = -(log_probability * target.unsqueeze(0)).sum(-1)
        per_step /= target.sum(-1).clamp_min(1.0).unsqueeze(0)
        all_nll.append(per_step)
    return torch.stack(all_nll, dim=-1)


def load_true_model(
    subject: str,
    seed_index: int,
    training_root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], Path, list[str]]:
    matches = sorted((training_root / subject).glob(f"TRUE_GRID__*__seed{seed_index}/summary.json"))
    if len(matches) != 1:
        raise FileNotFoundError(f"need one TRUE_GRID summary for {subject} seed {seed_index}; got {len(matches)}")
    summary_path = matches[0]
    summary = json.loads(summary_path.read_text())
    checkpoint_path = Path(summary["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_fixed_grid_model(
        checkpoint["channel_names"], np.asarray(checkpoint["mask"], dtype=np.uint8),
        seed=int(checkpoint["model_seed"]), state_dim=int(checkpoint["state_dim"]),
        microsteps=int(checkpoint["microsteps"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, summary, checkpoint_path, [str(value) for value in checkpoint["channel_names"]]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model, training_summary, checkpoint_path, checkpoint_names = load_true_model(
        args.subject, args.seed_index, args.training_root, device
    )
    before_hash = parameter_hash(model)
    event_path = args.cache_root / args.subject / "events.npz"
    out_dir = args.output_root / args.subject / f"seed{args.seed_index}" / f"patch_{args.patch_side}x{args.patch_side}"
    prior_path = out_dir / "SUMMARY.json"
    if prior_path.exists() and not args.force:
        prior = json.loads(prior_path.read_text())
        if (
            prior.get("checkpoint_sha256") == sha256_file(checkpoint_path)
            and prior.get("events_sha256") == sha256_file(event_path)
            and int(prior.get("n_controls_per_patch", -1)) == int(args.n_controls)
            and int(prior.get("minimum_train_events", -1)) == int(args.minimum_train_events)
            and int(prior.get("minimum_train_entering", -1)) == int(args.minimum_train_entering)
            and int(prior.get("control_draw_multiplier", -1)) == int(args.control_draw_multiplier)
            and prior.get("lesion_mode") == args.lesion_mode
            and (
                args.lesion_mode != "inbound_first_entry"
                or prior.get("first_entry_contract") == FIRST_ENTRY_CONTRACT
            )
            and args.max_patches is None
        ):
            return prior
    with np.load(event_path, allow_pickle=False) as events:
        ranks = np.asarray(events["ranks"], dtype=np.int16)
        split = np.asarray(events["split"], dtype=np.int8)
        channel_names = [str(value) for value in events["channel_names"].tolist()]
    if channel_names != checkpoint_names:
        raise ValueError("event and checkpoint contact order mismatch")

    train_ranks = ranks[split == 0]
    test_ranks = ranks[split == 2]
    tensors = build_event_tensors(test_ranks)
    predict = tensors["valid"] & ~tensors["is_last"]
    base_nll = baseline_nll(model, tensors, device, args.batch_size)
    base_mask = model.node_mask.detach().cpu().numpy().astype(np.float32)
    recurrent_weight = model.recurrent[0].detach().cpu().numpy()
    patches = enumerate_square_patches(channel_names, side=args.patch_side)
    patch_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    eligible: list[tuple[str, tuple[int, ...], int, int]] = []
    for patch_id, nodes in patches:
        n_events, n_enter = event_patch_coverage(train_ranks, nodes, args.lesion_mode)
        if n_events >= args.minimum_train_events and n_enter >= args.minimum_train_entering:
            eligible.append((patch_id, nodes, n_events, n_enter))
    matching_ineligible: list[dict[str, str]] = []

    started = time.time()
    for patch_index, (patch_id, nodes, train_events, train_entering) in enumerate(eligible):
        if args.max_patches is not None and len(patch_rows) >= int(args.max_patches):
            break
        if args.lesion_mode == "inbound_first_entry":
            involved = np.zeros(base_mask.shape[0], dtype=bool)
            involved[list(nodes)] = True
            # Recurrent matrices use [target, source]: keep only outside -> patch boundary edges.
            lesion = base_mask.astype(bool) & involved[:, None] & ~involved[None, :]
            lesion = lesion.astype(np.uint8)
        else:
            lesion = patch_edge_mask(base_mask, nodes).edge_mask
        try:
            if args.lesion_mode == "inbound_first_entry":
                control_masks, control_audits = matched_dispersed_directed_edge_sets(
                    base_mask, recurrent_weight, lesion, forbidden_nodes=nodes,
                    n_controls=args.n_controls,
                    seed=202608170000 + 1000 * args.seed_index + patch_index,
                    candidates_per_control=args.control_draw_multiplier,
                )
            else:
                control_masks, control_audits = matched_dispersed_edge_sets(
                    base_mask, recurrent_weight, lesion,
                    n_controls=args.n_controls,
                    seed=202608160000 + 1000 * args.seed_index + patch_index,
                    candidates_per_control=args.control_draw_multiplier,
                )
        except RuntimeError as error:
            matching_ineligible.append({"patch_id": patch_id, "reason": str(error)})
            continue
        lesion_families = [lesion, *control_masks]
        labels: list[tuple[float, int]] = []
        variant_masks: list[np.ndarray] = []
        for dose in DOSES[1:]:
            for lesion_index, lesion_mask in enumerate(lesion_families):
                variant = base_mask.copy()
                variant[np.asarray(lesion_mask, dtype=bool)] *= float(dose)
                variant_masks.append(variant)
                labels.append((dose, lesion_index))
        variant_tensor = torch.as_tensor(np.stack(variant_masks), dtype=torch.float32, device=device)
        sums_in = np.zeros(len(labels), dtype=np.float64)
        sums_out = np.zeros(len(labels), dtype=np.float64)
        n_in = 0
        n_out = 0
        for begin in range(0, len(test_ranks), args.batch_size):
            end = min(len(test_ranks), begin + args.batch_size)
            batch = {name: value[begin:end].to(device) for name, value in tensors.items()}
            lesion_nll = ensemble_nll(model, variant_tensor, batch).cpu()
            baseline = base_nll[begin:end]
            delta = lesion_nll - baseline.unsqueeze(0)
            valid = predict[begin:end]
            enters = torch.any(batch["target"][:, :, list(nodes)] > 0, dim=-1).cpu() & valid
            if args.lesion_mode == "inbound_first_entry":
                patch_seen_through_current = torch.any(
                    batch["recruited"][:, :, list(nodes)] > 0, dim=-1
                ).cpu()
                enters = enters & ~patch_seen_through_current
                outside = valid & ~enters & ~patch_seen_through_current
            else:
                outside = valid & ~enters
            sums_in += delta[:, enters].sum(-1).numpy()
            sums_out += delta[:, outside].sum(-1).numpy()
            n_in += int(enters.sum())
            n_out += int(outside.sum())

        by_dose: dict[float, list[float]] = {dose: [] for dose in DOSES[1:]}
        control_in_by_dose: dict[float, list[float]] = {dose: [] for dose in DOSES[1:]}
        control_out_by_dose: dict[float, list[float]] = {dose: [] for dose in DOSES[1:]}
        primary_by_dose: dict[float, float] = {}
        primary_in_by_dose: dict[float, float] = {}
        primary_out_by_dose: dict[float, float] = {}
        for row_index, (dose, lesion_index) in enumerate(labels):
            delta_in = float(sums_in[row_index] / max(n_in, 1))
            delta_out = float(sums_out[row_index] / max(n_out, 1))
            selectivity = delta_in - delta_out
            if lesion_index == 0:
                primary_by_dose[dose] = selectivity
                primary_in_by_dose[dose] = delta_in
                primary_out_by_dose[dose] = delta_out
            else:
                by_dose[dose].append(selectivity)
                control_in_by_dose[dose].append(delta_in)
                control_out_by_dose[dose].append(delta_out)
                audit = control_audits[lesion_index - 1]
                control_rows.append({
                    "subject": args.subject, "seed_index": args.seed_index,
                    "patch_id": patch_id, "patch_side": args.patch_side,
                    "dose": dose, "control_index": lesion_index - 1,
                    "delta_nll_entering_patch": delta_in,
                    "delta_nll_outside_patch": delta_out,
                    "selectivity": selectivity,
                    **audit,
                })
        patch_rows.append({
            "subject": args.subject, "seed_index": args.seed_index,
            "patch_id": patch_id, "patch_side": args.patch_side,
            "patch_nodes": " ".join(channel_names[index] for index in nodes),
            "n_train_events_touching_patch": train_events,
            "n_train_decisions_entering_patch": train_entering,
            "n_test_decisions_entering_patch": n_in,
            "n_test_decisions_outside_patch": n_out,
            "n_directed_edges_attenuated": int(lesion.sum()),
            **{
                f"selectivity_dose_{dose:g}": 0.0 if dose == 1.0 else primary_by_dose[dose]
                for dose in DOSES
            },
            **{
                f"matched_control_selectivity_dose_{dose:g}": 0.0 if dose == 1.0 else float(np.median(by_dose[dose]))
                for dose in DOSES
            },
            **{
                f"difference_in_difference_dose_{dose:g}": 0.0 if dose == 1.0 else (
                    primary_by_dose[dose] - float(np.median(by_dose[dose]))
                )
                for dose in DOSES
            },
            **{
                f"delta_nll_entering_patch_dose_{dose:g}": 0.0 if dose == 1.0 else primary_in_by_dose[dose]
                for dose in DOSES
            },
            **{
                f"delta_nll_outside_patch_dose_{dose:g}": 0.0 if dose == 1.0 else primary_out_by_dose[dose]
                for dose in DOSES
            },
            **{
                f"matched_control_delta_nll_entering_patch_dose_{dose:g}": 0.0 if dose == 1.0 else float(np.median(control_in_by_dose[dose]))
                for dose in DOSES
            },
            **{
                f"entry_damage_contrast_dose_{dose:g}": 0.0 if dose == 1.0 else (
                    primary_in_by_dose[dose] - float(np.median(control_in_by_dose[dose]))
                )
                for dose in DOSES
            },
        })

    after_hash = parameter_hash(model)
    if before_hash != after_hash:
        raise RuntimeError("patch evaluation changed trained parameters")
    write_rows(out_dir / "PATCH_RESULTS.csv", patch_rows)
    write_rows(out_dir / "MATCHED_CONTROL_RESULTS.csv", control_rows)
    payload = {
        "schema": "topic5_ecog_patch_necessity_v0.1",
        "subject": args.subject,
        "seed_index": args.seed_index,
        "patch_side": args.patch_side,
        "lesion_mode": args.lesion_mode,
        "first_entry_contract": (
            FIRST_ENTRY_CONTRACT if args.lesion_mode == "inbound_first_entry" else None
        ),
        "n_patches_possible": len(patches),
        "n_patches_train_eligible": len(eligible),
        "n_patches_matching_eligible_evaluated": len(patch_rows),
        "n_patches_matching_ineligible": len(matching_ineligible),
        "matching_ineligible": matching_ineligible,
        "n_controls_per_patch": args.n_controls,
        "control_draw_multiplier": args.control_draw_multiplier,
        "doses": DOSES,
        "minimum_train_events": args.minimum_train_events,
        "minimum_train_entering": args.minimum_train_entering,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "events_path": str(event_path),
        "events_sha256": sha256_file(event_path),
        "parameter_hash_before": before_hash,
        "parameter_hash_after": after_hash,
        "parameter_hash_unchanged": before_hash == after_hash,
        "test_contact_nll": training_summary["test"]["contact_nll"],
        "runtime_sec": time.time() - started,
    }
    (out_dir / "SUMMARY.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, choices=("958", "1084"))
    parser.add_argument("--seed-index", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--patch-side", type=int, default=2, choices=(2, 3))
    parser.add_argument(
        "--lesion-mode", default="symmetric_incident",
        choices=("symmetric_incident", "inbound_first_entry"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-controls", type=int, default=32)
    parser.add_argument("--control-draw-multiplier", type=int, default=64)
    parser.add_argument("--minimum-train-events", type=int, default=200)
    parser.add_argument("--minimum-train-entering", type=int, default=50)
    parser.add_argument("--max-patches", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/cache"
    ))
    parser.add_argument("--training-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/training"
    ))
    parser.add_argument("--output-root", type=Path, default=Path(
        "results/topic5_ecog_physical_neighborhood_rnn_v0_1/patch_necessity"
    ))
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
