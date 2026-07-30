#!/usr/bin/env python3
"""Random-subspace sensitivity for one frozen subject/seed hidden-state cell."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.extract_topic5_rnn_internal_states_v0_1 import (  # noqa: E402
    CONTROLS,
    SEED_DIRS,
    load_model,
    load_subject,
)
from src.topic5_rnn_internal_state import (  # noqa: E402
    PCAState,
    decode_hidden_nll,
    project_reconstruct,
    random_orthonormal_bases,
    variance_fidelity,
)


BASE = ROOT / "results/topic5_rnn_internal_state_reduction"
K_VALUES = (1, 2, 4, 8, 16)
N_RANDOM = 8


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--seed-dir", required=True, choices=SEED_DIRS)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    started = time.time()
    out = BASE / "interictal/random_subspace_cells" / args.seed_dir / args.subject
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "CELL_STATUS.json"
    atomic_json(
        status_path,
        {
            "status": "RUNNING",
            "subject": args.subject,
            "seed_dir": args.seed_dir,
            "target_values_read": False,
        },
    )
    device = torch.device(args.device)
    record = load_subject(args.subject)
    state_path = (
        BASE
        / "interictal/cells"
        / args.seed_dir
        / args.subject
        / "hidden_states.npz"
    )
    with np.load(state_path, allow_pickle=False) as data:
        cell = {key: np.asarray(data[key]) for key in data.files}
    event = cell["heldout20_event_index"].astype(np.int64)
    step = cell["heldout20_step"].astype(np.int64)
    rows = []
    seed_number = int(args.seed_dir.split("_")[-1])
    for control_index, control in enumerate(CONTROLS):
        model, offset, _ = load_model(
            args.subject,
            args.seed_dir,
            control,
            record["contact_features"].shape[1],
            device,
        )
        hidden = cell[f"{control}_heldout20_hidden"].astype(np.float32)
        mean = cell[f"{control}_pca_mean"].astype(np.float32)
        pca_components = cell[f"{control}_pca_components"].astype(np.float32)
        eigenvalues = cell[f"{control}_pca_eigenvalues"].astype(np.float64)
        original_nll, _ = decode_hidden_nll(
            model,
            torch.as_tensor(record["contact_features"]),
            offset,
            hidden,
            record["group_ids"],
            record["group_count"],
            event,
            step,
        )
        for k in K_VALUES:
            if k > hidden.shape[1]:
                continue
            pca = PCAState(mean=mean, components=pca_components, eigenvalues=eigenvalues)
            pca_hidden = project_reconstruct(hidden, pca, k)
            pca_nll, _ = decode_hidden_nll(
                model,
                torch.as_tensor(record["contact_features"]),
                offset,
                pca_hidden,
                record["group_ids"],
                record["group_count"],
                event,
                step,
            )
            for random_index, basis in enumerate(
                random_orthonormal_bases(
                    hidden.shape[1],
                    k,
                    [
                        seed_number
                        + control_index * 10_000
                        + k * 101
                        + index
                        for index in range(N_RANDOM)
                    ],
                )
            ):
                random_pca = PCAState(
                    mean=mean,
                    components=basis,
                    eigenvalues=np.ones(hidden.shape[1], dtype=np.float64),
                )
                reconstructed = project_reconstruct(hidden, random_pca, k)
                random_nll, _ = decode_hidden_nll(
                    model,
                    torch.as_tensor(record["contact_features"]),
                    offset,
                    reconstructed,
                    record["group_ids"],
                    record["group_count"],
                    event,
                    step,
                )
                rows.append(
                    {
                        "subject": args.subject,
                        "seed_dir": args.seed_dir,
                        "control": control,
                        "k": k,
                        "random_index": random_index,
                        "random_variance_fidelity": variance_fidelity(
                            hidden, reconstructed, mean
                        ),
                        "pca_variance_fidelity": variance_fidelity(
                            hidden, pca_hidden, mean
                        ),
                        "random_nll_loss": random_nll - original_nll,
                        "pca_nll_loss": pca_nll - original_nll,
                        "pca_advantage_nll": random_nll - pca_nll,
                    }
                )
        del model
    pd.DataFrame(rows).to_csv(out / "random_subspace_metrics.csv", index=False)
    atomic_json(
        status_path,
        {
            "status": "COMPLETE",
            "subject": args.subject,
            "seed_dir": args.seed_dir,
            "n_rows": len(rows),
            "runtime_seconds": float(time.time() - started),
            "target_values_read": False,
            "early_ictal_arrays_deserialized": False,
        },
    )


if __name__ == "__main__":
    main()
