#!/usr/bin/env python3
"""Extract target-blind selected-model fields and history interventions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_topic5_interictal_rank_distribution import (  # noqa: E402
    _batch,
    _model,
    load_records,
)
from src.topic5_rank_distribution import (  # noqa: E402
    LinearStateSequenceRNN,
    LowRankLeakySequenceRNN,
    VanillaRateSequenceRNN,
    next_set_stop_loss,
)
from src.topic5_rnn_internal_state import (  # noqa: E402
    prefix_intervention_outputs,
    readout_relevant_local_memory,
    teacher_forced_probability_fields,
)


DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
OLD = (
    ROOT
    / "results/topic5_interictal_rank_distribution/runs/"
    "formal_multiseed_20260725_v1"
)
LOW_RANK = (
    ROOT
    / "results/topic5_low_rank_dynamics/runs/"
    "low_rank_leaky_multiseed_20260725_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_model(control: str, payload: dict) -> torch.nn.Module:
    if control == "linear_state":
        return LinearStateSequenceRNN(**payload["model_kwargs"])
    if control == "vanilla_rnn":
        return VanillaRateSequenceRNN(**payload["model_kwargs"])
    if control.startswith("low_rank_r"):
        return LowRankLeakySequenceRNN(**payload["model_kwargs"])
    raise ValueError(f"unsupported selected control: {control}")


def checkpoint_paths(
    *,
    subject: str,
    seed: int,
    selected: str,
    formal_root: Path,
    shuffle_root: Path,
) -> dict[str, Path]:
    if selected.startswith("low_rank_r"):
        rank = int(selected.rsplit("r", 1)[1])
        selected_true = (
            LOW_RANK / f"seed_{seed}" / f"rank_{rank}" / subject / "checkpoint.pt"
        )
    else:
        selected_true = (
            formal_root
            / selected
            / f"seed_{seed}"
            / subject
            / f"{selected}_checkpoint.pt"
        )
    return {
        "selected_ordered": selected_true,
        "selected_rank_shuffle": (
            shuffle_root
            / selected
            / f"seed_{seed}"
            / subject
            / f"{selected}_rank_shuffle_checkpoint.pt"
        ),
        "unordered_prefix": (
            OLD
            / f"seed_{seed}"
            / subject
            / "unordered_prefix_checkpoint.pt"
        ),
        "full_history_gru": (
            OLD
            / f"seed_{seed}"
            / subject
            / "full_history_gru_checkpoint.pt"
        ),
    }


def load_model(
    label: str,
    path: Path,
    *,
    selected: str,
    feature_dim: int,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("ictal_target_read", True):
        raise RuntimeError(f"{path}: target seal failed")
    if label.startswith("selected_"):
        model = selected_model(selected, payload)
    else:
        model = _model(label, feature_dim, payload["model_kwargs"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload["heldout_local_offset"].to(device), payload


@torch.no_grad()
def intervention_metrics(
    model: torch.nn.Module,
    record,
    offset: torch.Tensor,
    *,
    intervention: str,
    reset_after_rank: int | None,
    device: torch.device,
    batch_size: int,
) -> dict:
    event_losses = []
    bucket_sum = {key: 0.0 for key in ("step0", "step1", "step2", "step3", "step4plus")}
    bucket_n = {key: 0 for key in bucket_sum}
    indices = record.eval_indices
    feature = torch.as_tensor(
        record.contact_features, dtype=torch.float32, device=device
    )
    for start in range(0, len(indices), int(batch_size)):
        chunk = indices[start : start + int(batch_size)]
        batch = _batch(
            record,
            chunk,
            device,
            rank_shuffle=False,
            rng=np.random.default_rng(0),
        )
        output = prefix_intervention_outputs(
            model,
            feature,
            offset,
            batch["group_ids"],
            batch["group_count"],
            intervention=intervention,
            reset_after_rank=reset_after_rank,
        )
        loss = next_set_stop_loss(
            output, batch["group_ids"], batch["group_count"]
        )
        event_losses.append(loss["event_nll"].cpu().numpy())
        step_nll = loss["step_nll"].cpu().numpy()
        step_mask = loss["step_mask"].cpu().numpy()
        for step in range(step_nll.shape[1]):
            key = f"step{step}" if step < 4 else "step4plus"
            bucket_sum[key] += float(step_nll[:, step][step_mask[:, step]].sum())
            bucket_n[key] += int(np.count_nonzero(step_mask[:, step]))
    return {
        "heldout_event_balanced_nll": float(
            np.mean(np.concatenate(event_losses))
        ),
        **{
            f"{key}_prefix_balanced_nll": (
                float(bucket_sum[key] / bucket_n[key])
                if bucket_n[key]
                else np.nan
            )
            for key in bucket_sum
        },
        **{f"{key}_n_prefixes": bucket_n[key] for key in bucket_n},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection-summary", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--shuffle-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    selection_path = (
        args.selection_summary
        if args.selection_summary.is_absolute()
        else ROOT / args.selection_summary
    )
    formal_root = (
        args.formal_root if args.formal_root.is_absolute() else ROOT / args.formal_root
    )
    shuffle_root = (
        args.shuffle_root
        if args.shuffle_root.is_absolute()
        else ROOT / args.shuffle_root
    )
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    selection = json.loads(selection_path.read_text())
    if selection.get("target_values_read", True):
        raise RuntimeError("selection artifact target seal failed")
    selected = selection["target_blind_best_non_gru"]["control"]
    records = load_records(DATASET)
    record = records[args.subject]
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_per_process_memory_fraction(0.20)
    torch.set_num_threads(4)
    paths = checkpoint_paths(
        subject=args.subject,
        seed=args.seed,
        selected=selected,
        formal_root=formal_root,
        shuffle_root=shuffle_root,
    )
    checkpoint_hashes = {}
    fields = {"contact_names": record.contact_names}
    rows = []
    memory_metrics = None
    feature = torch.as_tensor(
        record.contact_features, dtype=torch.float32, device=device
    )
    for label, path in paths.items():
        model, offset, _ = load_model(
            label,
            path,
            selected=selected,
            feature_dim=record.contact_features.shape[1],
            device=device,
        )
        checkpoint_hashes[label] = sha256(path)
        field = teacher_forced_probability_fields(
            model,
            feature,
            offset,
            record.group_ids,
            record.group_count,
            record.eval_indices,
            batch_size=args.batch_size,
        )
        fields[f"{label}_union_participation"] = field["union_participation"]
        fields[f"{label}_summed_next_probability"] = field[
            "summed_next_probability"
        ]
        if label in {"selected_ordered", "full_history_gru"}:
            interventions = [
                ("ordered", None),
                ("reverse_prefix", None),
                ("drop_earliest", None),
                ("reset_after_rank", 0),
                ("reset_after_rank", 1),
                ("reset_after_rank", 2),
            ]
            for intervention, reset_rank in interventions:
                metric = intervention_metrics(
                    model,
                    record,
                    offset,
                    intervention=intervention,
                    reset_after_rank=reset_rank,
                    device=device,
                    batch_size=args.batch_size,
                )
                rows.append(
                    {
                        "subject": args.subject,
                        "dataset": record.dataset,
                        "seed": args.seed,
                        "model": label,
                        "selected_architecture": selected,
                        "intervention": intervention,
                        "reset_after_rank": reset_rank,
                        **metric,
                    }
                )
        if label == "selected_ordered":
            memory_metrics = readout_relevant_local_memory(
                model,
                feature,
                offset,
                record.group_ids,
                record.group_count,
                record.eval_indices,
                max_events=24,
            )
        del model, offset
        if device.type == "cuda":
            torch.cuda.empty_cache()
    np.savez_compressed(output_dir / "teacher_forced_fields.npz", **fields)
    pd.DataFrame(rows).to_csv(
        output_dir / "history_intervention_metrics.csv", index=False
    )
    if memory_metrics is None:
        raise RuntimeError("selected-model readout memory was not computed")
    pd.DataFrame(
        [
            {
                "subject": args.subject,
                "dataset": record.dataset,
                "seed": args.seed,
                "selected_architecture": selected,
                **memory_metrics,
            }
        ]
    ).to_csv(output_dir / "readout_memory_metrics.csv", index=False)
    summary = {
        "status": "COMPLETE",
        "contract": "topic5_ordered_history_architecture_audit_v0_1",
        "subject": args.subject,
        "dataset": record.dataset,
        "seed": args.seed,
        "selected_architecture": selected,
        "n_eval_events": int(len(record.eval_indices)),
        "field_labels": sorted(key for key in fields if key != "contact_names"),
        "checkpoint_hashes": checkpoint_hashes,
        "dataset_npz_sha256": record.input_sha256,
        "target_values_read": False,
        "early_ictal_target_arrays_deserialized": False,
    }
    (output_dir / "DONE.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
