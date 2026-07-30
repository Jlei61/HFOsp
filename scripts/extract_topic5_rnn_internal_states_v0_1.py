#!/usr/bin/env python3
"""Extract target-blind hidden trajectories from one frozen subject/seed cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_rank_distribution import FullHistorySequenceGRU  # noqa: E402
from src.topic5_rnn_internal_state import (  # noqa: E402
    decode_hidden_nll,
    deterministic_event_sample,
    fit_pca,
    pca_summary,
    project_reconstruct,
    split_train80,
    teacher_forced_hidden,
    variance_fidelity,
)


DATASET = ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
RUNS = (
    ROOT
    / "results/topic5_interictal_rank_distribution/runs/"
    "formal_multiseed_20260725_v1"
)
OUT = ROOT / "results/topic5_rnn_internal_state_reduction"
SEED_DIRS = ("seed_20260725", "seed_20260726", "seed_20260727")
CONTROLS = ("full_history_gru", "rank_shuffle_gru")
K_VALUES = (0, 1, 2, 4, 8, 16, 32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_subject(subject: str) -> dict[str, np.ndarray]:
    path = DATASET / "per_subject" / f"{subject}.npz"
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    actual = sha256(path)
    if actual != str(metadata["dataset_npz_sha256"]):
        raise RuntimeError(f"{subject}: dataset fingerprint mismatch")
    with np.load(path, allow_pickle=False) as data:
        record = {
            "path": path,
            "sha256": actual,
            "contact_features": np.asarray(data["contact_features"], np.float32),
            "contact_names": np.asarray(data["contact_names"]).astype(str),
            "group_ids": np.asarray(data["event_group_ids"], np.int16),
            "group_count": np.asarray(data["event_group_count"], np.int16),
            "event_split": np.asarray(data["event_split"], np.uint8),
        }
    return record


def load_model(
    subject: str, seed_dir: str, control: str, feature_dim: int, device: torch.device
) -> tuple[torch.nn.Module, torch.Tensor, Path]:
    path = RUNS / seed_dir / subject / f"{control}_checkpoint.pt"
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload["heldout_subject"] != subject or payload["control"] != control:
        raise RuntimeError(f"{subject}/{seed_dir}/{control}: checkpoint identity drift")
    if bool(payload.get("ictal_target_read", True)):
        raise RuntimeError("checkpoint does not certify target sealing")
    model = FullHistorySequenceGRU(feature_dim, **payload["model_kwargs"])
    model.load_state_dict(payload["model_state"])
    model.to(device).eval()
    return model, payload["heldout_local_offset"].to(device), path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed-dir", required=True, choices=SEED_DIRS)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train-events", type=int, default=2048)
    parser.add_argument("--validation-events", type=int, default=1024)
    parser.add_argument("--heldout-events", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cell_root = OUT / "interictal/cells" / args.seed_dir / args.subject
    output_path = cell_root / "hidden_states.npz"
    status_path = cell_root / "CELL_STATUS.json"
    if output_path.exists() and status_path.exists() and not args.force:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "COMPLETE":
            print(json.dumps({"status": "SKIP_COMPLETE", "cell": str(cell_root)}))
            return
    cell_root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    atomic_json(
        status_path,
        {
            "status": "RUNNING",
            "subject": args.subject,
            "seed_dir": args.seed_dir,
            "target_values_read": False,
        },
    )
    torch.set_num_threads(2)
    device = torch.device(args.device)
    record = load_subject(args.subject)
    all_train = np.flatnonzero(record["event_split"] == 0)
    heldout = np.flatnonzero(record["event_split"] == 1)
    train60, validation20 = split_train80(all_train)
    selected = {
        "train60": deterministic_event_sample(train60, args.train_events),
        "validation20": deterministic_event_sample(
            validation20, args.validation_events
        ),
        "heldout20": deterministic_event_sample(heldout, args.heldout_events),
    }
    arrays: dict[str, np.ndarray] = {
        "contact_names": record["contact_names"],
    }
    for split, indices in selected.items():
        arrays[f"{split}_selected_events"] = indices
    metrics: list[dict] = []
    checkpoints: dict[str, dict[str, str]] = {}
    for control in CONTROLS:
        model, offset, checkpoint = load_model(
            args.subject,
            args.seed_dir,
            control,
            record["contact_features"].shape[1],
            device,
        )
        checkpoints[control] = {
            "path": str(checkpoint.relative_to(ROOT)),
            "sha256": sha256(checkpoint),
        }
        split_states: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for split, indices in selected.items():
            hidden, event_index, step = teacher_forced_hidden(
                model,
                torch.as_tensor(record["contact_features"]),
                offset,
                record["group_ids"],
                record["group_count"],
                indices,
                batch_size=args.batch_size,
            )
            split_states[split] = (hidden, event_index, step)
            arrays[f"{control}_{split}_hidden"] = hidden
            if control == CONTROLS[0]:
                arrays[f"{split}_event_index"] = event_index
                arrays[f"{split}_step"] = step
            else:
                if not np.array_equal(arrays[f"{split}_event_index"], event_index):
                    raise RuntimeError("control event ordering drifted")
                if not np.array_equal(arrays[f"{split}_step"], step):
                    raise RuntimeError("control prefix ordering drifted")

        pca = fit_pca(split_states["train60"][0])
        summary = pca_summary(pca)
        arrays[f"{control}_pca_mean"] = pca.mean
        arrays[f"{control}_pca_components"] = pca.components
        arrays[f"{control}_pca_eigenvalues"] = pca.eigenvalues
        held_hidden, held_event, held_step = split_states["heldout20"]
        original_nll, _ = decode_hidden_nll(
            model,
            torch.as_tensor(record["contact_features"]),
            offset,
            held_hidden,
            record["group_ids"],
            record["group_count"],
            held_event,
            held_step,
        )
        metrics.append(
            {
                "control": control,
                "metric": "pca_inventory",
                "k": -1,
                "value": float(summary["effective_rank"]),
                "k80": int(summary["k80"]),
                "k90": int(summary["k90"]),
                "k95": int(summary["k95"]),
                "original_heldout_event_nll": original_nll,
            }
        )
        for k in K_VALUES:
            if k > held_hidden.shape[1]:
                continue
            reconstructed = project_reconstruct(held_hidden, pca, k)
            nll, _ = decode_hidden_nll(
                model,
                torch.as_tensor(record["contact_features"]),
                offset,
                reconstructed,
                record["group_ids"],
                record["group_count"],
                held_event,
                held_step,
            )
            metrics.append(
                {
                    "control": control,
                    "metric": "pca_reconstruction",
                    "k": int(k),
                    "value": variance_fidelity(
                        held_hidden, reconstructed, pca.mean
                    ),
                    "k80": int(summary["k80"]),
                    "k90": int(summary["k90"]),
                    "k95": int(summary["k95"]),
                    "original_heldout_event_nll": original_nll,
                    "reconstructed_heldout_event_nll": nll,
                    "nll_loss": float(nll - original_nll),
                }
            )
        del model

    np.savez_compressed(output_path, **arrays)
    payload = {
        "contract": "topic5_rnn_internal_state_reduction_v0_1",
        "status": "COMPLETE",
        "subject": args.subject,
        "seed_dir": args.seed_dir,
        "dataset": str(record["path"].relative_to(ROOT)),
        "dataset_sha256": record["sha256"],
        "n_contacts": int(len(record["contact_names"])),
        "split_events": {key: int(len(value)) for key, value in selected.items()},
        "split_prefixes": {
            split: int(len(arrays[f"{split}_event_index"])) for split in selected
        },
        "checkpoints": checkpoints,
        "metrics": metrics,
        "output": str(output_path.relative_to(ROOT)),
        "output_sha256": sha256(output_path),
        "runtime_seconds": float(time.time() - started),
        "peak_rss_gb": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        ),
        "target_values_read": False,
        "early_ictal_arrays_deserialized": False,
    }
    atomic_json(status_path, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
