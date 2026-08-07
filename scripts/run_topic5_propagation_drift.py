#!/usr/bin/env python3
"""Measure how fast the per-patient propagation ordering drifts, and against what.

Exploratory and descriptive.  It cannot change the frozen V3.0 evidence level and
it is not a cohort claim: the 34-patient cohort has already informed design many
times, so anything found here has to be re-registered as its own contract before
it can be asserted.

Two pre-declared readouts, both computed per patient and then summarised over
patients:

* **cross-recording cost** — at a matched number of intervening events, how much
  lower is the ordering agreement for block pairs that straddle a
  continuity-unit boundary than for pairs inside one unit;
* **wall-clock cost inside one recording** — with intervening event count
  partialled out, does elapsed time still predict lower agreement.

Run `--freeze-rule-only` first: it writes the parameter and decision block and
locks this file's SHA256, so the numbers cannot be produced by a version of the
analysis edited after seeing them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from multiprocessing import Pool
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_propagation_drift import (  # noqa: E402
    block_templates,
    drift_pairs,
    matched_event_distance_contrast,
    partial_spearman,
    rank_residual_fraction,
)

DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"
DEFAULT_OUTPUT = ROOT / "results/topic5_propagation_drift"
RULE_NAME = "DRIFT_RULE_STATE.json"

TRAIN80_SPLIT_CODE = 0

FROZEN = {
    "block_events": 20,
    "min_participation": 0.5,
    "min_support": 0.5,
    "min_shared_contacts": 5,
    "max_pairs_per_patient": 200000,
    "pair_seed": 20260804,
    "d_events_bin_edges": [
        0.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0, 2560.0, 5120.0, 10240.0
    ],
    "min_pairs_per_cell": 20,
    "partial_minimum_pairs": 200,
    "minimum_usable_residual_fraction": 0.05,
}

DECISION = {
    "tier": "EXPLORATORY_DESCRIPTIVE_NOT_A_COHORT_CLAIM",
    "cross_recording_cost": (
        "Per patient take the median of cross-minus-same agreement over matched "
        "event-separation cells; report cohort median, favourable patient count, "
        "two-sided Wilcoxon and sign test. A cell whose within-bin event "
        "imbalance exceeds half the bin width is reported but excluded from the "
        "patient median, because there the bin failed to match the two arms."
    ),
    "wall_clock_cost": (
        "Per patient take the partial rank correlation between agreement and "
        "elapsed seconds with intervening event count partialled out, using "
        "same-source pairs only; a patient whose residual event-count-free "
        "variation falls below the minimum usable residual fraction is recorded "
        "as UNRESOLVED_COLLINEAR rather than as a zero."
    ),
    "forbidden": [
        "activity-dependent shaping",
        "causal plasticity",
        "reopening the V3.0 evidence level or the V3.1 handoff",
        "within-event next-rank mechanism",
    ],
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def jsonable(value):
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def freeze_rule(output: Path) -> dict:
    rule_path = output / RULE_NAME
    if rule_path.exists():
        raise RuntimeError(f"{rule_path} already exists; refusing to re-freeze")
    if (output / "DRIFT_STATE.json").exists():
        raise RuntimeError("results already exist; the rule cannot be newly frozen")
    state = {
        "contract": "topic5_propagation_drift",
        "status": "PRE_RESULT_RULE_FROZEN",
        "supersedes": (
            "A first freeze on 2026-08-04 used dataset event_source_index as the "
            "segment label. That field is a per-event row pointer, so every "
            "patient produced zero blocks and the run returned no scientific "
            "content at all. This freeze corrects the label to the recording name "
            "in the source-mapping artifact. No result was inspected in between "
            "because none existed."
        ),
        "parameters": FROZEN,
        "decision": DECISION,
        "results_exist": False,
        "results_read": False,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "drift_module_sha256": sha256(ROOT / "src/topic5_propagation_drift.py"),
    }
    atomic_json(rule_path, state)
    return state


def analyse_patient(job: tuple[str, str, str]) -> dict:
    subject, dataset_root, mapping_root = job
    dataset = np.load(
        Path(dataset_root) / "per_subject" / f"{subject}.npz", allow_pickle=True
    )
    # `event_source_index` in the dataset is a per-event row pointer, not a unit
    # label; the recording name in the mapping artifact is the real segment id.
    mapping_path = Path(mapping_root) / f"{subject}.npz"
    if not mapping_path.exists():
        return {"subject": subject, "status": "UNRESOLVED_NO_SOURCE_MAPPING"}
    mapping = np.load(mapping_path, allow_pickle=True)
    if "event_source_record_name" not in mapping.files:
        return {"subject": subject, "status": "UNRESOLVED_NO_RECORD_NAME"}
    labels = np.asarray(mapping["event_source_record_name"])
    keep = np.asarray(dataset["event_split"]) == TRAIN80_SPLIT_CODE
    if labels.shape[0] != keep.shape[0]:
        return {"subject": subject, "status": "UNRESOLVED_MAPPING_LENGTH_MISMATCH"}
    blocks = block_templates(
        np.asarray(dataset["event_local_rank"])[keep],
        np.asarray(dataset["event_participation"])[keep],
        labels[keep],
        np.asarray(dataset["event_abs_time"])[keep],
        block_events=FROZEN["block_events"],
        min_participation=FROZEN["min_participation"],
    )
    if len(blocks) < 4:
        return {"subject": subject, "status": "UNRESOLVED_TOO_FEW_BLOCKS", "n_blocks": len(blocks)}

    pairs = drift_pairs(
        blocks,
        max_pairs=FROZEN["max_pairs_per_patient"],
        seed=FROZEN["pair_seed"],
        min_support=FROZEN["min_support"],
        min_shared=FROZEN["min_shared_contacts"],
    )
    cells = matched_event_distance_contrast(
        pairs,
        bin_edges=FROZEN["d_events_bin_edges"],
        min_pairs_per_cell=FROZEN["min_pairs_per_cell"],
    )
    usable = [
        cell
        for cell in cells
        if abs(cell["event_imbalance"])
        <= 0.5 * (cell["d_events_high"] - cell["d_events_low"])
    ]
    same = [row for row in pairs if row["same_source"]]
    residual_fraction = (
        rank_residual_fraction(
            [row["d_seconds"] for row in same], [row["d_events"] for row in same]
        )
        if len(same) >= FROZEN["partial_minimum_pairs"]
        else None
    )
    wall_clock = None
    wall_status = "UNRESOLVED_COLLINEAR"
    if (
        residual_fraction is not None
        and residual_fraction >= FROZEN["minimum_usable_residual_fraction"]
    ):
        wall_clock = partial_spearman(
            [row["similarity"] for row in same],
            [row["d_seconds"] for row in same],
            [row["d_events"] for row in same],
            minimum_n=FROZEN["partial_minimum_pairs"],
        )
        wall_status = "RESOLVED" if wall_clock is not None else "UNRESOLVED_COLLINEAR"

    return {
        "subject": subject,
        "status": "DRIFT_COMPLETE",
        "n_blocks": len(blocks),
        "n_sources": len({row["source_id"] for row in blocks}),
        "n_pairs_scored": len(pairs),
        "n_same_source_pairs": len(same),
        "n_cross_source_pairs": len(pairs) - len(same),
        "n_cells": len(cells),
        "n_usable_cells": len(usable),
        "cross_recording_cost": (
            float(np.median([cell["cross_minus_same"] for cell in usable]))
            if usable
            else None
        ),
        "cross_recording_status": "RESOLVED" if usable else "UNRESOLVED_NO_MATCHED_CELL",
        "wall_clock_partial_rho": wall_clock,
        "wall_clock_status": wall_status,
        "wall_clock_residual_fraction": residual_fraction,
        "median_same_source_similarity": (
            float(np.median([row["similarity"] for row in same])) if same else None
        ),
        "cells": cells,
    }


def cohort_summary(values) -> dict:
    array = np.asarray([v for v in values if v is not None], dtype=float)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"n": 0}
    nonzero = array[array != 0]
    try:
        p = float(wilcoxon(array, alternative="two-sided").pvalue)
    except ValueError:
        p = None
    return {
        "n": int(array.size),
        "median": float(np.median(array)),
        "n_negative": int(np.sum(array < 0)),
        "n_positive": int(np.sum(array > 0)),
        "wilcoxon_two_sided_p": p,
        "sign_test_two_sided_p": (
            float(binomtest(int(np.sum(nonzero > 0)), nonzero.size, 0.5).pvalue)
            if nonzero.size
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--freeze-rule-only", action="store_true")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.freeze_rule_only:
        print(json.dumps(freeze_rule(args.output), indent=2, sort_keys=True))
        return

    rule_path = args.output / RULE_NAME
    if not rule_path.exists():
        raise SystemExit("rule is not frozen; run --freeze-rule-only first")
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if rule.get("runner_sha256") != sha256(Path(__file__).resolve()):
        raise SystemExit("runner changed after the rule was frozen")
    if rule.get("drift_module_sha256") != sha256(ROOT / "src/topic5_propagation_drift.py"):
        raise SystemExit("drift module changed after the rule was frozen")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    innovation = json.loads(
        (
            ROOT / str(config["innovation_output_root"]) / "innovation_validity.json"
        ).read_text(encoding="utf-8")
    )
    statuses = {row["subject"]: row["status"] for row in innovation["patients"]}
    dataset_root = str(ROOT / str(config["dataset_root"]))
    mapping_root = str(ROOT / str(config["source_mapping_root"]))

    jobs = [
        (subject, dataset_root, mapping_root) for subject in sorted(statuses)
    ]
    with Pool(processes=max(1, int(args.workers))) as pool:
        rows = pool.map(analyse_patient, jobs)
    for row in rows:
        row["innovation_status"] = statuses[row["subject"]]
        row["dataset"] = row["subject"].split("_", 1)[0]

    done = [row for row in rows if row["status"] == "DRIFT_COMPLETE"]
    state = {
        "contract": "topic5_propagation_drift",
        "status": "DRIFT_COHORT_COMPLETE",
        "tier": DECISION["tier"],
        "descriptive_only_cannot_change_frozen_evidence_level": True,
        "one_step_is_one_complete_event": True,
        "within_event_next_rank_model_fit": False,
        "n_patients": len(rows),
        "n_complete": len(done),
        "parameters": FROZEN,
        "decision": DECISION,
        "cross_recording_cost": cohort_summary(
            [row.get("cross_recording_cost") for row in done]
        ),
        "cross_recording_unresolved": sum(
            row.get("cross_recording_status") != "RESOLVED" for row in done
        ),
        "wall_clock_partial_rho": cohort_summary(
            [row.get("wall_clock_partial_rho") for row in done]
        ),
        "wall_clock_unresolved_collinear": sum(
            row.get("wall_clock_status") != "RESOLVED" for row in done
        ),
        "rule_sha256": sha256(rule_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "patients": rows,
    }
    atomic_json(args.output / "DRIFT_STATE.json", jsonable(state))
    pd.DataFrame(
        [{key: value for key, value in row.items() if key != "cells"} for row in rows]
    ).to_csv(args.output / "drift_per_patient.csv", index=False)
    pd.DataFrame(
        [
            {"subject": row["subject"], **cell}
            for row in done
            for cell in row["cells"]
        ]
    ).to_csv(args.output / "drift_matched_cells.csv", index=False)
    print(
        json.dumps(
            {
                key: state[key]
                for key in (
                    "status",
                    "n_complete",
                    "cross_recording_cost",
                    "cross_recording_unresolved",
                    "wall_clock_partial_rho",
                    "wall_clock_unresolved_collinear",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
