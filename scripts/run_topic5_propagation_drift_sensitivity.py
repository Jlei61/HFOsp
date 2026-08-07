#!/usr/bin/env python3
"""Confound sensitivity for the frozen propagation-drift readouts.

Runs after, and never rewrites, `run_topic5_propagation_drift.py`.  Two controls
the frozen primary did not carry are added:

* **block coarseness** — a block always holds 20 events but spans a variable
  amount of time; long-spanning blocks are intrinsically noisier templates;
* **shared-contact attrition** — pairs across a recording boundary may share
  fewer supported contacts.

For the elapsed-time readout the same pair set is scored twice, once with the
frozen single control and once with all three, so the comparison is like for
like.  For the cross-recording readout the shared-contact imbalance between the
two arms is reported, and the contrast is recomputed on well-supported pairs
only.
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
from src.topic5_propagation_drift_sensitivity import (  # noqa: E402
    annotated_pairs,
    partial_spearman_multi,
)

DEFAULT_CONFIG = ROOT / "config/topic5_event_innovation_v3_0.yaml"
PRIMARY_OUTPUT = ROOT / "results/topic5_propagation_drift"
DEFAULT_OUTPUT = PRIMARY_OUTPUT / "confound_sensitivity"
TRAIN80_SPLIT_CODE = 0


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


def analyse_patient(job) -> dict:
    subject, dataset_root, mapping_root, frozen = job
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

    pairs = annotated_pairs(
        blocks,
        max_pairs=frozen["max_pairs_per_patient"],
        seed=frozen["pair_seed"],
        min_support=frozen["min_support"],
        min_shared=frozen["min_shared_contacts"],
    )
    same = [row for row in pairs if row["same_source"]]
    cross = [row for row in pairs if not row["same_source"]]
    if len(same) < frozen["partial_minimum_pairs"]:
        return {"subject": subject, "status": "UNRESOLVED_TOO_FEW_SAME_SOURCE_PAIRS"}

    similarity = [row["similarity"] for row in same]
    seconds = [row["d_seconds"] for row in same]
    events = [row["d_events"] for row in same]
    spans = [row["mean_block_span_seconds"] for row in same]
    shared = [row["n_shared_contacts"] for row in same]

    frozen_equivalent = partial_spearman_multi(
        similarity, seconds, [events], minimum_n=frozen["partial_minimum_pairs"]
    )
    controlled = partial_spearman_multi(
        similarity,
        seconds,
        [events, spans, shared],
        minimum_n=frozen["partial_minimum_pairs"],
    )

    shared_same = float(np.median([row["n_shared_contacts"] for row in same]))
    shared_cross = (
        float(np.median([row["n_shared_contacts"] for row in cross])) if cross else None
    )
    threshold = float(np.median([row["n_shared_contacts"] for row in pairs]))
    well_supported = [row for row in pairs if row["n_shared_contacts"] >= threshold]
    cells = matched_event_distance_contrast(
        well_supported,
        bin_edges=frozen["d_events_bin_edges"],
        min_pairs_per_cell=frozen["min_pairs_per_cell"],
    )
    usable = [
        cell
        for cell in cells
        if abs(cell["event_imbalance"])
        <= 0.5 * (cell["d_events_high"] - cell["d_events_low"])
    ]
    return {
        "subject": subject,
        "status": "SENSITIVITY_COMPLETE",
        "n_same_source_pairs": len(same),
        "n_cross_source_pairs": len(cross),
        "wall_clock_frozen_equivalent_rho": frozen_equivalent["rho"],
        "wall_clock_frozen_equivalent_status": frozen_equivalent["status"],
        "wall_clock_controlled_rho": controlled["rho"],
        "wall_clock_controlled_status": controlled["status"],
        "wall_clock_controlled_residual_fraction": controlled["residual_fraction"],
        "median_shared_contacts_same_source": shared_same,
        "median_shared_contacts_cross_source": shared_cross,
        "shared_contact_imbalance": (
            None if shared_cross is None else shared_cross - shared_same
        ),
        "well_supported_threshold": threshold,
        "n_usable_cells_well_supported": len(usable),
        "cross_recording_cost_well_supported": (
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
    args = parser.parse_args()

    primary_rule = json.loads(
        (PRIMARY_OUTPUT / "DRIFT_RULE_STATE.json").read_text(encoding="utf-8")
    )
    frozen = primary_rule["parameters"]

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
        row["dataset"] = row["subject"].split("_", 1)[0]

    done = [row for row in rows if row["status"] == "SENSITIVITY_COMPLETE"]
    state = {
        "contract": "topic5_propagation_drift_confound_sensitivity",
        "status": "SENSITIVITY_COHORT_COMPLETE",
        "tier": "SENSITIVITY_ONLY_CANNOT_UPGRADE_THE_PRIMARY",
        "parameters_inherited_from": str(PRIMARY_OUTPUT / "DRIFT_RULE_STATE.json"),
        "primary_rule_sha256": sha256(PRIMARY_OUTPUT / "DRIFT_RULE_STATE.json"),
        "n_complete": len(done),
        "wall_clock_frozen_equivalent": cohort(
            [row.get("wall_clock_frozen_equivalent_rho") for row in done]
        ),
        "wall_clock_controlled": cohort(
            [row.get("wall_clock_controlled_rho") for row in done]
        ),
        "shared_contact_imbalance": cohort(
            [row.get("shared_contact_imbalance") for row in done]
        ),
        "cross_recording_cost_well_supported": cohort(
            [row.get("cross_recording_cost_well_supported") for row in done]
        ),
        "known_limitation": (
            "Rank-based partial correlation removes a monotone confound well but "
            "only partly removes a mixture of several controls, so a surviving "
            "association is an upper bound on the confound-free effect, not a "
            "clean estimate of it."
        ),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "patients": rows,
    }
    atomic_json(args.output / "SENSITIVITY_STATE.json", jsonable(state))
    pd.DataFrame(rows).to_csv(args.output / "sensitivity_per_patient.csv", index=False)
    print(
        json.dumps(
            {
                key: state[key]
                for key in (
                    "status",
                    "n_complete",
                    "wall_clock_frozen_equivalent",
                    "wall_clock_controlled",
                    "shared_contact_imbalance",
                    "cross_recording_cost_well_supported",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
