#!/usr/bin/env python3
"""Compute the two named model--patient similarity tests shown in Fig. 4H.

The displayed matrix is an equal-network mean over the 12 frozen confirmation
networks.  This script tests its two semantic matches separately:

* MTA versus patient TA (raw mode 1 versus mode 1);
* MTB versus patient TB (raw mode 0 versus mode 0).

For each network, one contact permutation is applied to every model event and
is restricted to exchanges within the same shaft.  The model-to-patient
contact correspondence is therefore destroyed while event structure and shaft
membership are retained.  The 12 networks contribute equally to each null
draw.  These are post-hoc calibrations of frozen readouts, not acceptance gates.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    _load_bundle,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_d6_natural_kmeans import (  # noqa: E402
    contact_split_folds,
    crossfit_patient_readout,
    normalize_event_ranks,
    patient_profiles,
)
from src.topic4_nlc_null_calibration import (  # noqa: E402
    contact_permutation_matrix_draws,
    crossfit_matrix,
    equal_network_null,
)


DEFAULT_CONFIG = ROOT / "config/topic4_rev11_nlc_frozen_substrate_confirmation.json"
DEFAULT_NLC_OUTPUT = (
    ROOT
    / "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc"
    / "frozen_substrate_confirmation"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/paper-ready-figure/fig4"
    / "fig4_panelh_pairwise_similarity_statistics.json"
)
CONTACT_DRAWS = 1000
BASE_SEED = 20260815
WITHIN_SHAFT_SEED_OFFSET = 7919

# The frozen semantic audit maps raw mode 1 to TA and raw mode 0 to TB.
SEMANTIC_PAIRS = {
    "MTA_vs_TA": (1, 1),
    "MTB_vs_TB": (0, 0),
}
SEMANTIC_DISPLAY_ORDER = (1, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def compute_statistics(bundle, config_path=DEFAULT_CONFIG, *, draws=CONTACT_DRAWS):
    """Return the audited, separate MTA--TA and MTB--TB null tests."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract_path = ROOT / config["inputs"]["contact_contract"]["path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    folds = contact_split_folds(contract)
    network_seeds = [int(value) for value in config["search"][
        "confirmation_network_seeds"
    ]]
    record_seed = np.asarray([row["seed"] for row in bundle["records"]], int)
    patient_ranks = np.asarray(bundle["patient"]["patient_train_ranks"], float)
    patient_labels = np.asarray(
        bundle["patient"]["patient_train_old_labels"], int,
    )
    profiles = patient_profiles(patient_ranks, patient_labels)
    shaft_ids = np.asarray(bundle["static"]["shaft_ids"])

    observed_by_pair = {key: {} for key in SEMANTIC_PAIRS}
    draws_by_pair = {key: {} for key in SEMANTIC_PAIRS}
    per_network = {}
    parity = []
    observed_matrices = []
    for network_seed in network_seeds:
        key = str(network_seed)
        index = np.flatnonzero(bundle["clean"] & (record_seed == network_seed))
        if not len(index):
            raise RuntimeError(f"no clean events for frozen network {network_seed}")
        ranks = np.asarray(bundle["ranks"][index], float)
        reference = np.asarray(crossfit_patient_readout(
            ranks, patient_ranks, patient_labels, folds,
        )["matrix"], float)
        fast = crossfit_matrix(normalize_event_ranks(ranks), profiles, folds)
        error = float(np.nanmax(np.abs(reference - fast)))
        if not np.allclose(reference, fast, rtol=0.0, atol=1e-12, equal_nan=True):
            raise RuntimeError(
                f"fast/reference matrix mismatch for network {network_seed}: {error}"
            )
        null_matrices = contact_permutation_matrix_draws(
            ranks, patient_ranks, patient_labels, folds,
            draws=int(draws),
            seed=BASE_SEED + WITHIN_SHAFT_SEED_OFFSET + network_seed,
            shaft_ids=shaft_ids,
        )
        network_row = {
            "n_clean_events": int(len(index)),
            "observed_raw_mode_matrix": reference.tolist(),
        }
        for pair_name, (row, column) in SEMANTIC_PAIRS.items():
            values = np.asarray(null_matrices[:, row, column], float)
            if not np.all(np.isfinite(values)):
                raise RuntimeError(
                    f"non-finite {pair_name} null draw for network {network_seed}"
                )
            observed_by_pair[pair_name][key] = float(reference[row, column])
            draws_by_pair[pair_name][key] = values
            network_row[pair_name] = {
                "observed_rho": float(reference[row, column]),
                "null_median": float(np.median(values)),
                "null_q95": float(np.quantile(values, 0.95)),
            }
        per_network[key] = network_row
        parity.append({
            "network_seed": network_seed,
            "max_abs_error": error,
        })
        observed_matrices.append(reference)

    tests = {}
    for pair_name in SEMANTIC_PAIRS:
        summary = equal_network_null(
            observed_by_pair[pair_name], draws_by_pair[pair_name],
        )
        if summary is None:
            raise RuntimeError(f"could not aggregate {pair_name} null")
        summary["stars"] = _stars(float(summary["one_sided_p"]))
        summary["alternative"] = "observed similarity exceeds permuted similarity"
        summary["null"] = "within-shaft model-contact permutation"
        tests[pair_name] = summary

    raw_matrix = np.mean(np.asarray(observed_matrices, float), axis=0)
    display_matrix = raw_matrix[np.ix_(
        SEMANTIC_DISPLAY_ORDER, SEMANTIC_DISPLAY_ORDER,
    )]
    return {
        "schema_version": "fig4_panelh_pairwise_similarity_v1",
        "status": "PAIRWISE_SIMILARITY_NULL_COMPLETE",
        "scientific_role": (
            "post-hoc null calibration of two named frozen similarities; "
            "not a new acceptance gate"
        ),
        "arm": bundle["candidate"]["candidate_id"],
        "n_networks": len(network_seeds),
        "network_seeds": network_seeds,
        "displayed_equal_network_matrix": display_matrix.tolist(),
        "display_order": {
            "rows": ["MTA", "MTB"],
            "columns": ["TA", "TB"],
            "raw_mode_order": list(SEMANTIC_DISPLAY_ORDER),
        },
        "tests": tests,
        "per_network": per_network,
        "fast_reference_parity": {
            "rows": parity,
            "max_abs_error": float(max(row["max_abs_error"] for row in parity)),
        },
        "null_contract": {
            "permutation_unit": "one model-contact permutation per network and draw",
            "restriction": "contacts exchange only within the same shaft",
            "preserved": "event rank structure, event set, shaft membership",
            "destroyed": "model-to-patient contact identity",
            "aggregation": "equal-network mean at each aligned draw index",
            "alternative": "one-sided greater",
            "draws": int(draws),
            "base_seed": BASE_SEED,
            "within_shaft_seed_offset": WITHIN_SHAFT_SEED_OFFSET,
        },
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
            "null_helpers": {
                "path": "src/topic4_nlc_null_calibration.py",
                "sha256": _sha256(ROOT / "src/topic4_nlc_null_calibration.py"),
            },
        },
    }


def compute_and_write(
        bundle=None, *, config_path=DEFAULT_CONFIG,
        output_path=DEFAULT_OUTPUT, draws=CONTACT_DRAWS):
    if bundle is None:
        bundle = _load_bundle(config_path, DEFAULT_NLC_OUTPUT)
    payload = compute_statistics(bundle, config_path, draws=draws)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(payload, output_path)
    return payload


def main() -> None:
    payload = compute_and_write()
    print(json.dumps({
        "status": payload["status"],
        "output": str(DEFAULT_OUTPUT.relative_to(ROOT)),
        "tests": payload["tests"],
    }, indent=2))


if __name__ == "__main__":
    main()
