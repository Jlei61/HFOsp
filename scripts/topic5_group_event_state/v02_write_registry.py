#!/usr/bin/env python3
"""Write the shared producer registry entries that Agents B and C read.

One atomic file per producer under ``shared/producers/``; the combined
``checkpoint_registry.json`` is regenerated from them.  A producer whose runs are
incomplete is written with ``status: partial`` and the missing (subject, seed)
cells listed, so a downstream line reports ``not_available`` for those cells
instead of quietly substituting another producer (CC 10).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v02.contract_paths import (  # noqa: E402
    DATASET_ROOT,
    SHARED_ROOT,
)
from src.topic5_group_event_state.v02.registry import (  # noqa: E402
    ProducerEntry,
    ProducerRegistry,
    file_hash,
    payload_hash,
    source_commit,
)
from src.topic5_group_event_state.v02.subject import SubjectTimelineConfig  # noqa: E402

DEFAULT_PRODUCER_ROOT = Path("/data/hfosp_group_event_state_v0_2/agent_a/producers/main")

# P_memoryless is a seed-1 sensitivity arm by design (pre-registered before any
# producer result).  Recording it against a 3-seed expectation would mark it
# ``partial`` and make a downstream line report two seeds as ``not_available``
# that were never meant to exist.
SEEDS_BY_PRODUCER = {"P_memoryless": [1]}
BASELINE_ROOT = Path(
    "/data/hfosp_group_event_state_v0_2/agent_a/future_block/baseline_only"
)

SELECTION_OBJECTIVE = (
    "TRAIN chronological inner-validation: the training objective itself "
    "(local endpoints excluding group_size, plus each horizon's future-block "
    "terms at their frozen weights).  No seizure label and no development-test "
    "number enters checkpoint selection."
)


def _recurrent_entry(producer: str, root: Path, seeds: list[int],
                     subjects: list[str], commit: str) -> ProducerEntry:
    per_subject: dict[str, dict] = {}
    missing: list[str] = []
    hashes: list[str] = []
    encoder = state = None
    weights = None
    for subject in subjects:
        seed_map: dict[str, dict] = {}
        for seed in seeds:
            run = root / "runs" / subject / producer / f"seed{seed}"
            result, ckpt = run / "result.json", run / "checkpoint.pt"
            if not (result.exists() and ckpt.exists()):
                missing.append(f"{subject}/seed{seed}")
                continue
            payload = json.loads(result.read_text())
            h = file_hash(ckpt)
            hashes.append(h)
            if encoder is None:
                # The checkpoint is the authoritative record of what was trained;
                # result.json carries diagnostics, not the frozen configuration.
                import torch

                saved = torch.load(ckpt, map_location="cpu", weights_only=False)
                encoder = saved["config"]["encoder"]
                state = saved["config"]["state"]
            weights = weights or payload.get("future_loss_weights")
            seed_map[str(seed)] = {
                "checkpoint": str(ckpt),
                "checkpoint_sha256": h,
                "anchor_state": str(run / "anchor_state.npz"),
                "event_state": str(run / "event_state.npz"),
                "selected_epoch": payload["selected_epoch"],
                "stop_reason": payload["stop_reason"],
                "n_epochs_run": payload["n_epochs_run"],
                "param_update_magnitude": payload["param_update_magnitude"],
                "train_seconds": payload["train_seconds"],
                "peak_memory_reserved_gb": payload.get("peak_memory_reserved_gb"),
                "config_hash": payload.get("config_hash"),
            }
        if seed_map:
            per_subject[subject] = seed_map
    objective = ["next_event"]
    if producer != "P_local":
        objective += ["future_5m", "future_30m", "future_120m"]
    return ProducerEntry(
        producer_id=producer,
        model_family="group_event_recurrent",
        uses_waveform=bool((encoder or {}).get("use_waveform", True)),
        uses_multiband=bool((encoder or {}).get("use_multiband", True)),
        uses_background=bool((state or {}).get("use_background", False)),
        event_update=bool((state or {}).get("persistent", True)),
        feedback_model="observer_only",
        physical_dt=bool((state or {}).get("use_real_dt", True)),
        training_objective=objective,
        anchor_grid_minutes=SubjectTimelineConfig().grid_seconds / 60.0,
        source_commit=commit,
        config_hash=payload_hash({"encoder": encoder, "state": state,
                                  "future_loss_weights": weights}),
        checkpoint_hash=payload_hash(sorted(hashes)) if hashes else "none",
        dataset_root=str(DATASET_ROOT),
        timeline_config_hash=payload_hash(SubjectTimelineConfig().as_dict()),
        target_builder_hash=payload_hash({
            "builder": "prefix_sums_v1",
            "horizons_seconds": list(SubjectTimelineConfig().horizons_seconds),
        }),
        selection_objective=SELECTION_OBJECTIVE,
        status="complete" if not missing else "partial",
        subjects=per_subject,
        notes={
            "n_subjects": len(per_subject),
            "seeds": seeds,
            "seed_policy": (
                "3 seeds for the core producers; P_memoryless is a pre-registered "
                "seed-1 sensitivity arm and is complete at one seed"
            ),
            "missing_cells": missing,
            "future_loss_weights_example": weights,
            "state_layout": "columns [z_fast | z_slow]; d_fast/d_slow in each result.json",
            "anchor_state_rows": "one row per fixed 5-min anchor of the full grid",
            "event_state_rows": "one row per interictal event in a carry segment",
        },
    )


def _baseline_entry(root: Path, commit: str) -> ProducerEntry:
    subjects = sorted(p.stem for p in (root / "per_subject").glob("*.json"))
    manifest = json.loads((root / "manifest.json").read_text())
    return ProducerEntry(
        producer_id="B_multiscale",
        model_family="multiscale_ewma_glm",
        uses_waveform=False,
        uses_multiband=True,
        uses_background=False,
        event_update=False,
        feedback_model="none",
        physical_dt=True,
        training_objective=["future_5m", "future_30m", "future_120m"],
        anchor_grid_minutes=SubjectTimelineConfig().grid_seconds / 60.0,
        source_commit=commit,
        config_hash=payload_hash(manifest.get("eval_config", {})),
        checkpoint_hash="not_applicable_closed_form_features",
        dataset_root=str(DATASET_ROOT),
        timeline_config_hash=payload_hash(SubjectTimelineConfig().as_dict()),
        target_builder_hash=payload_hash({
            "builder": "prefix_sums_v1",
            "horizons_seconds": list(SubjectTimelineConfig().horizons_seconds),
        }),
        selection_objective=(
            "ridge chosen per endpoint family by chronological CV inside TRAIN; "
            "no development-test number enters the choice"
        ),
        status="complete",
        subjects={s: {"result": str(root / "per_subject" / f"{s}.json")} for s in subjects},
        notes={
            "n_subjects": len(subjects),
            "features": "1/5/30/120 min EWMA of rate, size/STOP, participation "
                        "field, repertoire embedding and band energy, plus clock, "
                        "session geometry and seizure bookkeeping",
            "uses_seizure_times": True,
            "seizure_times_note": "baseline nuisance model only; never an input "
                                  "to a representation producer",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer-root", type=Path, default=DEFAULT_PRODUCER_ROOT)
    parser.add_argument("--baseline-root", type=Path, default=BASELINE_ROOT)
    parser.add_argument("--producers", nargs="+", default=["P_local", "P_slow"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--shared-root", type=Path, default=SHARED_ROOT)
    args = parser.parse_args()

    commit = source_commit(ROOT)
    subjects = sorted(p.name for p in DATASET_ROOT.iterdir() if (p / "index.json").exists())
    registry = ProducerRegistry(args.shared_root)

    written = []
    if (args.baseline_root / "manifest.json").exists():
        registry.write(_baseline_entry(args.baseline_root, commit))
        written.append("B_multiscale")
    for producer in args.producers:
        seeds = SEEDS_BY_PRODUCER.get(producer, args.seeds)
        entry = _recurrent_entry(producer, args.producer_root, seeds, subjects, commit)
        registry.write(entry)
        written.append(f"{producer}({entry.status}, {len(entry.subjects)} subjects)")
    path = registry.refresh_combined_view()
    print(json.dumps({"written": written, "registry": str(path),
                      "producers": registry.list_producers()}, indent=2))


if __name__ == "__main__":
    main()
