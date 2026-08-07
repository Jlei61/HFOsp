#!/usr/bin/env python3
"""Day/night control for the frozen elapsed-time drift readout.

Runs after, and never rewrites, `run_topic5_propagation_drift.py` or its
confound sensitivity.  The frozen primary reported that blocks further apart in
seconds agree less about contact ordering at a matched number of intervening
events.  Time of day is confounded with elapsed time, and this repository
already treats day/night as a real stratifier, so the frozen readout is equally
consistent with "day simply looks different from night".

The decisive contrast keeps only pairs whose two blocks sit in the same diurnal
phase and recomputes the same partial correlation with the same three controls.
Block geometry, pair construction and matched-cell binning all reuse the frozen
modules; only the phase stratification is new.

Run `--freeze-rule-only` first.
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
    matched_event_distance_contrast,
)
from src.topic5_propagation_drift_diurnal import (  # noqa: E402
    DAY_START_HOUR,
    NIGHT_START_HOUR,
    TIMEZONE_BY_DATASET,
    as_phase_contrast_pairs,
    assign_block_phase,
    attach_phase,
    phase_exposure,
    timezone_for_dataset,
)
from src.topic5_propagation_drift_sensitivity import (  # noqa: E402
    annotated_pairs,
    partial_spearman_multi,
)

DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"
PRIMARY_OUTPUT = ROOT / "results/topic5_propagation_drift"
DEFAULT_OUTPUT = PRIMARY_OUTPUT / "diurnal_control"
RULE_NAME = "DIURNAL_RULE_STATE.json"
TRAIN80_SPLIT_CODE = 0

DECISION = {
    "tier": "SENSITIVITY_ONLY_CANNOT_UPGRADE_THE_PRIMARY",
    "day_night_rule": (
        f"local hour in [{DAY_START_HOUR}, {NIGHT_START_HOUR}) is day; timezone per "
        f"dataset contract {TIMEZONE_BY_DATASET}"
    ),
    "decisive_readout": (
        "Partial rank correlation between ordering agreement and elapsed seconds, "
        "controlling intervening event count, block span and shared contact count, "
        "computed on same-source pairs whose two blocks share a diurnal phase."
    ),
    "verdict_rule": (
        "If the same-phase-restricted correlation keeps a negative cohort median "
        "with two-sided Wilcoxon p <= 0.05, the frozen elapsed-time readout is not "
        "explained by time of day alone. If it loses the sign or the median moves "
        "to approximately zero, the frozen readout must be re-described as a "
        "day-versus-night difference rather than drift."
    ),
    "exposure_caveat": (
        "A patient whose pairs almost never cross the boundary had no diurnal "
        "confound to control, so its survival is uninformative about the "
        "confound; cross-phase fraction is reported per patient for this reason."
    ),
    "forbidden": [
        "activity-dependent shaping",
        "causal plasticity",
        "reopening the V3.0 evidence level or the V3.1 handoff",
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


def freeze_rule(output: Path, frozen: dict) -> dict:
    rule_path = output / RULE_NAME
    if rule_path.exists():
        raise RuntimeError(f"{rule_path} already exists; refusing to re-freeze")
    if (output / "DIURNAL_STATE.json").exists():
        raise RuntimeError("results already exist; the rule cannot be newly frozen")
    state = {
        "contract": "topic5_propagation_drift_diurnal_control",
        "status": "PRE_RESULT_RULE_FROZEN",
        "parameters_inherited_from": str(PRIMARY_OUTPUT / "DRIFT_RULE_STATE.json"),
        "parameters": frozen,
        "decision": DECISION,
        "results_exist": False,
        "results_read": False,
        "runner_sha256": sha256(Path(__file__).resolve()),
        "diurnal_module_sha256": sha256(
            ROOT / "src/topic5_propagation_drift_diurnal.py"
        ),
    }
    atomic_json(rule_path, state)
    return state


def analyse_patient(job) -> dict:
    subject, dataset_root, mapping_root, frozen = job
    dataset_name = subject.split("_", 1)[0]
    timezone_name = timezone_for_dataset(dataset_name)
    dataset = np.load(
        Path(dataset_root) / "per_subject" / f"{subject}.npz", allow_pickle=True
    )
    mapping_path = Path(mapping_root) / f"{subject}.npz"
    if not mapping_path.exists():
        return {"subject": subject, "status": "UNRESOLVED_NO_SOURCE_MAPPING"}
    mapping = np.load(mapping_path, allow_pickle=True)
    labels = np.asarray(mapping["event_source_record_name"])
    keep = np.asarray(dataset["event_split"]) == TRAIN80_SPLIT_CODE
    blocks = block_templates(
        np.asarray(dataset["event_local_rank"])[keep],
        np.asarray(dataset["event_participation"])[keep],
        labels[keep],
        np.asarray(dataset["event_abs_time"])[keep],
        block_events=frozen["block_events"],
        min_participation=frozen["min_participation"],
    )
    if len(blocks) < 4:
        return {"subject": subject, "status": "UNRESOLVED_TOO_FEW_BLOCKS"}

    phases = assign_block_phase(blocks, timezone_name)
    pairs = attach_phase(
        annotated_pairs(
            blocks,
            max_pairs=frozen["max_pairs_per_patient"],
            seed=frozen["pair_seed"],
            min_support=frozen["min_support"],
            min_shared=frozen["min_shared_contacts"],
        ),
        phases,
    )
    same_source = [row for row in pairs if row["same_source"]]
    same_phase = [row for row in same_source if row["same_phase"]]

    def wall_clock(subset):
        if len(subset) < frozen["partial_minimum_pairs"]:
            return {"rho": None, "status": "UNRESOLVED_TOO_FEW_PAIRS",
                    "residual_fraction": None, "n": len(subset)}
        return partial_spearman_multi(
            [row["similarity"] for row in subset],
            [row["d_seconds"] for row in subset],
            [
                [row["d_events"] for row in subset],
                [row["mean_block_span_seconds"] for row in subset],
                [row["n_shared_contacts"] for row in subset],
            ],
            minimum_n=frozen["partial_minimum_pairs"],
        )

    all_pairs_result = wall_clock(same_source)
    same_phase_result = wall_clock(same_phase)

    phase_cells = matched_event_distance_contrast(
        as_phase_contrast_pairs(same_source),
        bin_edges=frozen["d_events_bin_edges"],
        min_pairs_per_cell=frozen["min_pairs_per_cell"],
    )
    usable = [
        cell
        for cell in phase_cells
        if abs(cell["event_imbalance"])
        <= 0.5 * (cell["d_events_high"] - cell["d_events_low"])
    ]
    exposure = phase_exposure(same_source)
    day_blocks = sum(1 for phase in phases if phase == "day")
    return {
        "subject": subject,
        "dataset": dataset_name,
        "timezone": timezone_name,
        "status": "DIURNAL_COMPLETE",
        "n_blocks": len(blocks),
        "day_block_fraction": day_blocks / float(len(blocks)),
        "n_same_source_pairs": len(same_source),
        "n_same_phase_pairs": len(same_phase),
        "cross_phase_fraction": exposure.get("cross_phase_fraction"),
        "median_d_seconds_same_source": exposure.get("median_d_seconds"),
        "p95_d_seconds_same_source": exposure.get("p95_d_seconds"),
        "max_d_seconds_same_source": exposure.get("max_d_seconds"),
        "wall_clock_all_pairs_rho": all_pairs_result["rho"],
        "wall_clock_all_pairs_status": all_pairs_result["status"],
        "wall_clock_same_phase_rho": same_phase_result["rho"],
        "wall_clock_same_phase_status": same_phase_result["status"],
        "wall_clock_same_phase_residual_fraction": same_phase_result["residual_fraction"],
        "n_usable_phase_cells": len(usable),
        # negative means cross-phase pairs agree less than same-phase pairs
        "cross_phase_minus_same_phase": (
            float(np.median([cell["cross_minus_same"] for cell in usable]))
            if usable
            else None
        ),
    }


def cohort(values) -> dict:
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

    frozen = json.loads(
        (PRIMARY_OUTPUT / "DRIFT_RULE_STATE.json").read_text(encoding="utf-8")
    )["parameters"]
    args.output.mkdir(parents=True, exist_ok=True)
    if args.freeze_rule_only:
        print(json.dumps(freeze_rule(args.output, frozen), indent=2, sort_keys=True))
        return

    rule_path = args.output / RULE_NAME
    if not rule_path.exists():
        raise SystemExit("rule is not frozen; run --freeze-rule-only first")
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if rule.get("runner_sha256") != sha256(Path(__file__).resolve()):
        raise SystemExit("runner changed after the rule was frozen")
    if rule.get("diurnal_module_sha256") != sha256(
        ROOT / "src/topic5_propagation_drift_diurnal.py"
    ):
        raise SystemExit("diurnal module changed after the rule was frozen")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    innovation = json.loads(
        (
            ROOT / str(config["innovation_output_root"]) / "innovation_validity.json"
        ).read_text(encoding="utf-8")
    )
    statuses = {row["subject"]: row["status"] for row in innovation["patients"]}
    jobs = [
        (
            subject,
            str(ROOT / str(config["dataset_root"])),
            str(ROOT / str(config["source_mapping_root"])),
            frozen,
        )
        for subject in sorted(statuses)
    ]
    with Pool(processes=max(1, int(args.workers))) as pool:
        rows = pool.map(analyse_patient, jobs)
    for row in rows:
        row["innovation_status"] = statuses[row["subject"]]

    done = [row for row in rows if row["status"] == "DIURNAL_COMPLETE"]
    exposed = [
        row for row in done if (row.get("cross_phase_fraction") or 0.0) >= 0.05
    ]
    state = {
        "contract": "topic5_propagation_drift_diurnal_control",
        "status": "DIURNAL_COHORT_COMPLETE",
        "tier": DECISION["tier"],
        "decision": DECISION,
        "n_complete": len(done),
        "n_with_real_diurnal_exposure": len(exposed),
        "wall_clock_all_pairs": cohort(
            [row.get("wall_clock_all_pairs_rho") for row in done]
        ),
        "wall_clock_same_phase": cohort(
            [row.get("wall_clock_same_phase_rho") for row in done]
        ),
        "wall_clock_same_phase_exposed_only": cohort(
            [row.get("wall_clock_same_phase_rho") for row in exposed]
        ),
        "cross_phase_minus_same_phase": cohort(
            [row.get("cross_phase_minus_same_phase") for row in done]
        ),
        "rule_sha256": sha256(rule_path),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "patients": rows,
    }
    atomic_json(args.output / "DIURNAL_STATE.json", jsonable(state))
    pd.DataFrame(rows).to_csv(args.output / "diurnal_per_patient.csv", index=False)
    print(
        json.dumps(
            {
                key: state[key]
                for key in (
                    "status",
                    "n_complete",
                    "n_with_real_diurnal_exposure",
                    "wall_clock_all_pairs",
                    "wall_clock_same_phase",
                    "wall_clock_same_phase_exposed_only",
                    "cross_phase_minus_same_phase",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
