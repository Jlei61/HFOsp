#!/usr/bin/env python3
"""Diagnose whether fitted SPF fields collapsed to a static latent readout.

This is a checkpoint-only development diagnostic.  It uses the already frozen
development-test split and evaluates the future-blind prior mean.  No old
outer-heldout20 event is selected or scored; the frozen NPZ co-locates both
split arrays, so this is an analysis-use guarantee rather than a byte-level
file-read claim.  Non-zero latent use is not evidence for a biological field;
it only rules out the trivial explanation that M4 lost because its latent
state and recurrent trajectory were completely unused.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_topic5_spf_multiround import (  # noqa: E402
    OUTPUT_ROOT,
    _load_context,
    _load_model,
    _run_dirs,
)
from src.topic5_shared_propagation_field import sha256_file  # noqa: E402

TARGET = OUTPUT_ROOT / "round7_field_utilization"
MODELS = ("m4_field", "m4_field_phase")


def _best_history(summary: dict[str, Any], name: str) -> dict[str, Any]:
    model = summary["models"][name]
    best_epoch = int(model["training_adequacy"]["best_epoch"])
    matching = [
        row for row in model["history"] if int(row["epoch"]) == best_epoch
    ]
    if len(matching) != 1:
        raise RuntimeError(
            f"{summary['subject']} {summary['seed']} {name}: "
            f"best epoch {best_epoch} missing from accepted history"
        )
    return matching[0]


@torch.no_grad()
def _trajectory_metrics(
    model: torch.nn.Module,
    groups: torch.Tensor,
    counts: torch.Tensor,
) -> dict[str, float]:
    first = groups == 0
    initial, _ = model.prior_parameters(first)
    state_at = model.state_factory(initial, counts)
    states = []
    logits = []
    active_steps = []
    maximum = int(counts.max().item()) - 1
    for step in range(1, maximum + 1):
        active = step < counts
        state = state_at(step, active)
        states.append(state.detach().cpu())
        logits.append(model.contact_logits(state).detach().cpu())
        active_steps.append(active.detach().cpu())
    state_stack = torch.stack(states, dim=1)
    logit_stack = torch.stack(logits, dim=1)
    active = torch.stack(active_steps, dim=1)
    initial_cpu = initial.detach().cpu()

    total_displacement = []
    mean_step_displacement = []
    temporal_logit_sd = []
    for event in range(len(counts)):
        n_steps = int(active[event].sum().item())
        if n_steps < 1:
            continue
        event_states = state_stack[event, :n_steps]
        total_displacement.append(
            float(torch.linalg.vector_norm(event_states[-1] - initial_cpu[event]))
        )
        path = torch.cat([initial_cpu[event][None], event_states], dim=0)
        mean_step_displacement.append(
            float(
                torch.linalg.vector_norm(
                    path[1:] - path[:-1], dim=1
                ).mean()
            )
        )
        if n_steps > 1:
            temporal_logit_sd.append(
                float(logit_stack[event, :n_steps].std(dim=0).mean())
            )
    return {
        "prior_mean_total_state_displacement": float(
            np.mean(total_displacement)
        ),
        "prior_mean_step_state_displacement": float(
            np.mean(mean_step_displacement)
        ),
        "prior_mean_temporal_logit_sd": float(np.mean(temporal_logit_sd))
        if temporal_logit_sd
        else 0.0,
        "alpha": float(model.alpha.detach().cpu()),
    }


def _worker(run_dir_text: str) -> list[dict[str, Any]]:
    torch.set_num_threads(1)
    context = _load_context(Path(run_dir_text))
    output = []
    for name in MODELS:
        model = _load_model(context, name)
        history = _best_history(context["summary"], name)
        output.append(
            {
                "subject": context["summary"]["subject"],
                "seed": int(context["summary"]["seed"]),
                "model": name,
                "best_epoch": int(
                    context["summary"]["models"][name][
                        "training_adequacy"
                    ]["best_epoch"]
                ),
                "best_epoch_raw_kl_per_event": float(history["raw_kl"]),
                "best_epoch_effective_kl_per_event": float(
                    history["effective_kl"]
                ),
                **_trajectory_metrics(
                    model, context["groups"], context["counts"]
                ),
            }
        )
    return output


def main() -> None:
    rows = []
    with ProcessPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(_worker, str(path)): path for path in _run_dirs()
        }
        for future in as_completed(futures):
            rows.extend(future.result())
    rows.sort(key=lambda row: (row["subject"], row["seed"], row["model"]))
    TARGET.mkdir(parents=True, exist_ok=True)
    with (TARGET / "field_utilization_runs.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for name in MODELS:
        selected = [row for row in rows if row["model"] == name]
        item = {"model": name, "n_runs": len(selected)}
        for metric in (
            "best_epoch_raw_kl_per_event",
            "prior_mean_total_state_displacement",
            "prior_mean_step_state_displacement",
            "prior_mean_temporal_logit_sd",
            "alpha",
        ):
            values = np.asarray([row[metric] for row in selected], dtype=float)
            item[f"{metric}_median"] = float(np.median(values))
            item[f"{metric}_minimum"] = float(np.min(values))
            item[f"{metric}_maximum"] = float(np.max(values))
        summary.append(item)
    with (TARGET / "field_utilization_summary.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    state = {
        "status": "COMPLETE",
        "round": 7,
        "question": (
            "Did the fitted M4 variants lose only because their latent code or "
            "recurrent trajectory collapsed to a static readout?"
        ),
        "n_runs": len(rows),
        "old_heldout20_scored": False,
        "interpretation_limit": (
            "non-zero latent utilization rules out a trivial implementation "
            "collapse but is not evidence for an identifiable biological field"
        ),
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): sha256_file(Path(__file__)),
            "scripts/analyze_topic5_spf_multiround.py": sha256_file(
                ROOT / "scripts/analyze_topic5_spf_multiround.py"
            ),
        },
    }
    (TARGET / "ROUND_STATE.json").write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n"
    )


if __name__ == "__main__":
    main()
