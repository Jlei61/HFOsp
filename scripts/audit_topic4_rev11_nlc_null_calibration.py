#!/usr/bin/env python3
"""Calibrate the rev11-NLC acceptance statistics against matched nulls.

The frozen-substrate confirmation adjudicates ``DIRECTIONAL_REPERTOIRE`` at a
fixed 0.5 threshold and ``PATIENT_GEOMETRY`` at a fixed 0.0 threshold. Neither
threshold is the null expectation of its statistic, and the confirmation runs
all four pathway arms, so the same gates can be evaluated on the Node-only
control. This audit adds, without touching the pre-registered verdict:

* a within-network direction-label permutation null for balanced alignment;
* a contact-correspondence null for the cross-fit patient margin, both with a
  free permutation and with a within-shaft restricted permutation;
* the two D5.2 pooled diagnostics that this rev dropped - the seed-stratified
  permutation p-value of KMeans direction purity and the patient-matched purity
  benchmark;
* the same three acceptance statements evaluated for every arm, so the
  discriminative power of each gate is on the record.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_topic4_rev10_d5_2_spatial_ou_confirmation import (  # noqa: E402
    _patient_matched_benchmark,
    _seed_stratified_permutation,
)
from scripts.audit_topic4_rev11_nlc_frozen_substrate_confirmation import (  # noqa: E402
    ARM_IDS,
)
from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    _canonical_rank_kmeans,
    _load_bundle,
)
from scripts.rescore_topic4_rev10_d6_natural_kmeans import _jsonable  # noqa: E402
from scripts.run_topic4_rev9l_forced_source_worker import _sha256  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_d6_natural_kmeans import (  # noqa: E402
    contact_split_folds,
    crossfit_patient_readout,
    natural_kmeans,
    network_bootstrap,
    normalize_event_ranks,
    patient_profiles,
)
from src.topic4_nlc_null_calibration import (  # noqa: E402
    contact_permutation_draws,
    crossfit_margin,
    direction_label_permutation_draws,
    equal_network_null,
)

DEFAULT_CONFIG = ROOT / "config/topic4_rev11_nlc_frozen_substrate_confirmation.json"
LABEL_DRAWS = 2000
CONTACT_DRAWS = 1000
POOLED_PERMUTATION_REPEATS = 2000
PATIENT_BENCHMARK_DRAWS = 256
BASE_SEED = 20260815


def _arm_calibration(bundle, folds, contract_shafts, config):
    """Per-network observed values and matched null draws for one arm."""
    record_seed = np.asarray([row["seed"] for row in bundle["records"]], int)
    patient_ranks = bundle["patient"]["patient_train_ranks"]
    patient_labels = bundle["patient"]["patient_train_old_labels"]
    profiles = patient_profiles(patient_ranks, patient_labels)
    alignment_observed, margin_observed = {}, {}
    alignment_draws, margin_draws, margin_shaft_draws = {}, {}, {}
    per_network, parity = {}, []
    for seed in config["search"]["confirmation_network_seeds"]:
        key = str(seed)
        index = np.flatnonzero(bundle["clean"] & (record_seed == int(seed)))
        if not len(index):
            continue
        ranks = bundle["ranks"][index]
        crossfit = crossfit_patient_readout(
            ranks, patient_ranks, patient_labels, folds,
        )
        consensus = np.asarray(crossfit["consensus_labels"], int)
        natural = natural_kmeans(ranks, consensus, random_state=int(seed))
        fast = crossfit_margin(normalize_event_ranks(ranks), profiles, folds)
        reference = crossfit["signed_margin"]
        parity.append({
            "network_seed": int(seed),
            "reference_margin": reference,
            "fast_margin": fast,
            "abs_error": (
                None if reference is None or fast is None
                else abs(float(fast) - float(reference))
            ),
        })
        row = {"network_seed": int(seed)}
        if reference is not None:
            margin_observed[key] = float(reference)
            margin_draws[key] = contact_permutation_draws(
                ranks, patient_ranks, patient_labels, folds,
                draws=CONTACT_DRAWS, seed=BASE_SEED + int(seed),
            )
            margin_shaft_draws[key] = contact_permutation_draws(
                ranks, patient_ranks, patient_labels, folds,
                draws=CONTACT_DRAWS, seed=BASE_SEED + 7919 + int(seed),
                shaft_ids=contract_shafts,
            )
            row["observed_crossfit_margin"] = float(reference)
            row["contact_null_median"] = float(np.median(margin_draws[key]))
            row["contact_null_q95"] = float(np.quantile(margin_draws[key], 0.95))
            row["within_shaft_null_median"] = float(
                np.median(margin_shaft_draws[key])
            )
            row["within_shaft_null_q95"] = float(
                np.quantile(margin_shaft_draws[key], 0.95)
            )
        if natural.get("status") == "OK" and (
                natural.get("direction_balanced_alignment") is not None):
            # ``natural_kmeans`` drops events with fewer than three participating
            # contacts, so the direction labels must be taken on the same subset.
            scored = np.asarray(natural["valid_event_mask"], bool)
            alignment_observed[key] = float(natural["direction_balanced_alignment"])
            alignment_draws[key] = direction_label_permutation_draws(
                natural["cluster_labels"], consensus[scored],
                draws=LABEL_DRAWS, seed=BASE_SEED + 104729 + int(seed),
            )
            row["observed_balanced_alignment"] = alignment_observed[key]
            row["label_null_median"] = float(np.median(alignment_draws[key]))
            row["label_null_q95"] = float(np.quantile(alignment_draws[key], 0.95))
            row["n_events"] = int(natural["n_events"])
            row["n_direction_labelled"] = int(
                natural["n_crossfit_direction_labeled"]
            )
        per_network[key] = row
    return {
        "per_network": per_network,
        "fast_crossfit_parity": {
            "rows": parity,
            "max_abs_error": float(max(
                (row["abs_error"] for row in parity
                 if row["abs_error"] is not None), default=0.0,
            )),
        },
        "balanced_alignment_vs_label_permutation_null": equal_network_null(
            alignment_observed, alignment_draws,
        ),
        "crossfit_margin_vs_contact_permutation_null": equal_network_null(
            margin_observed, margin_draws,
        ),
        "crossfit_margin_vs_within_shaft_contact_null": equal_network_null(
            margin_observed, margin_shaft_draws,
        ),
        "equal_network_alignment_bootstrap": network_bootstrap(
            list(alignment_observed.values()),
        ),
        "equal_network_margin_bootstrap": network_bootstrap(
            list(margin_observed.values()),
        ),
    }


def _pooled_diagnostics(bundle):
    """The two D5.2 pooled calibrations this rev dropped."""
    canonical = _canonical_rank_kmeans(bundle)
    seed_ids = np.asarray([
        bundle["records"][int(index)]["seed"]
        for index in canonical["clean_global_index"]
    ], int)
    permutation_p = _seed_stratified_permutation(
        canonical["labels"], canonical["direction"], seed_ids,
        repeats=POOLED_PERMUTATION_REPEATS, seed=BASE_SEED,
    )
    benchmark = _patient_matched_benchmark(
        bundle, canonical["direction"],
        draws=PATIENT_BENCHMARK_DRAWS, seed=BASE_SEED,
    )
    benchmark = {key: value for key, value in benchmark.items() if key != "draws"}
    return {
        "pooled_kmeans_direction_purity": float(canonical["direction_purity"]),
        "pooled_cluster_counts": canonical["cluster_counts"].tolist(),
        "seed_stratified_direction_permutation_p": permutation_p,
        "permutation_repeats": POOLED_PERMUTATION_REPEATS,
        "patient_matched_kmeans_direction_purity": benchmark,
        "reaches_patient_matched_q05": bool(
            float(canonical["direction_purity"]) >= float(benchmark["q05"])
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--arms", nargs="*", default=list(ARM_IDS))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    verdict_path = root / "confirmation_verdict.json"
    verdict = json.loads(verdict_path.read_text())
    contract_path = ROOT / config["inputs"]["contact_contract"]["path"]
    contract = json.loads(contract_path.read_text())
    folds = contact_split_folds(contract)
    acceptance = config["search"]["acceptance"]

    arms = {}
    for candidate_id in args.arms:
        bundle = _load_bundle(
            config_path, root, candidate_id, allow_exploratory_candidate=True,
        )
        shafts = bundle["static"]["shaft_ids"]
        calibration = _arm_calibration(bundle, folds, shafts, config)
        calibration["pooled_diagnostics"] = _pooled_diagnostics(bundle)
        alignment = calibration["equal_network_alignment_bootstrap"]
        margin = calibration["equal_network_margin_bootstrap"]
        calibration["preregistered_gate_replay"] = {
            "n_networks_with_alignment": (
                0 if alignment is None else alignment["n_networks"]
            ),
            "DIRECTIONAL_REPERTOIRE": bool(
                alignment is not None
                and alignment["n_networks"] >= acceptance[
                    "minimum_evaluable_joint_networks"
                ]
                and alignment["network_bootstrap_q05"] > acceptance[
                    "natural_alignment_q05_min"
                ]
            ),
            "PATIENT_GEOMETRY": bool(
                margin is not None
                and margin["network_bootstrap_q05"] > acceptance[
                    "patient_margin_q05_min"
                ]
            ),
        }
        arms[candidate_id] = calibration
        print(json.dumps({
            "arm": candidate_id,
            "alignment_vs_null": calibration[
                "balanced_alignment_vs_label_permutation_null"
            ],
            "margin_vs_contact_null": calibration[
                "crossfit_margin_vs_contact_permutation_null"
            ],
            "gate_replay": calibration["preregistered_gate_replay"],
        }, indent=2, default=float))

    gate_replay = {
        candidate_id: row["preregistered_gate_replay"]
        for candidate_id, row in arms.items()
    }
    passing = [
        candidate_id for candidate_id, row in gate_replay.items()
        if row["DIRECTIONAL_REPERTOIRE"] and row["PATIENT_GEOMETRY"]
    ]
    payload = {
        "status": "REV11NLC_NULL_CALIBRATION_COMPLETE",
        "role": (
            "post-hoc null calibration of the pre-registered confirmation "
            "statistics; it does not change the frozen verdict"
        ),
        "preregistered_verdict": {
            "status": verdict["status"],
            "component_status": verdict["component_status"],
            "path": str(verdict_path.relative_to(ROOT)),
            "sha256": _sha256(verdict_path),
        },
        "why_this_calibration_exists": {
            "DIRECTIONAL_REPERTOIRE": (
                "balanced alignment takes the better of two cluster-to-mode "
                "matchings, so its finite-sample null sits above the 0.5 "
                "acceptance threshold"
            ),
            "PATIENT_GEOMETRY": (
                "events are assigned to the patient prototypes and then scored "
                "against the same prototypes on the disjoint contact fold, and "
                "the two prototypes are anti-correlated, so an axis-aligned "
                "model earns a positive margin without patient-specific "
                "geometry"
            ),
            "arm_replay": (
                "the confirmation ran all four arms, so both gates can be "
                "replayed on the Node-only control"
            ),
        },
        "gates_passed_by_arm": gate_replay,
        "gates_do_not_separate_arms": bool(len(passing) == len(gate_replay)),
        "arms_passing_both_uncalibrated_gates": passing,
        "null_contract": {
            "label_permutation": (
                "within network, permute the cross-fit direction labels of "
                "labelled events; KMeans labels, mode counts and the labelled "
                "subset are held fixed"
            ),
            "contact_permutation": (
                "within network, one contact permutation is applied to every "
                "event, destroying only the model-to-patient contact "
                "correspondence"
            ),
            "within_shaft_contact_permutation": (
                "same, restricted to exchanges inside each shaft, so shaft "
                "membership is preserved"
            ),
            "label_draws": LABEL_DRAWS,
            "contact_draws": CONTACT_DRAWS,
        },
        "arms": arms,
        "inputs": {
            "config": {
                "path": str(config_path.relative_to(ROOT)),
                "sha256": _sha256(config_path),
            },
            "contact_contract": {
                "path": str(contract_path.relative_to(ROOT)),
                "sha256": _sha256(contract_path),
            },
            "analysis_code": {
                "path": str(Path(__file__).resolve().relative_to(ROOT)),
                "sha256": _sha256(Path(__file__).resolve()),
            },
        },
        "analysis_git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
        "claim_boundary": (
            "null calibration only; it re-scores frozen worker artifacts and "
            "adds no simulation, no refit and no new acceptance gate"
        ),
    }
    output = root / "null_calibration.json"
    atomic_write_json(_jsonable(payload), output)
    print(json.dumps({
        "status": payload["status"],
        "gates_do_not_separate_arms": payload["gates_do_not_separate_arms"],
        "arms_passing_both_uncalibrated_gates": passing,
        "output": str(output.relative_to(ROOT)),
    }, indent=2))


if __name__ == "__main__":
    main()
