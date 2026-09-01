"""Close rev10-R2.1 with a network-level shared mode-A capacity audit."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / (
    "results/topic4_sef_hfo/data_driven_core_field_rev10_r"
)


def _sha256(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_summary(path, status):
    payload = json.loads(Path(path).read_text())
    if payload.get("status") != status:
        raise RuntimeError(f"unexpected summary status: {path}")
    return payload


def _candidate_mode_support(summary, candidate_id, seed, mode):
    return int(summary["candidate_details"][candidate_id]["by_seed"][str(seed)][
        "mode_conditioned_joint_support"
    ][mode]["n_joint_in_distribution"])


def fit_library_oracle(summary):
    candidates = sorted(summary["candidate_details"])
    by_seed = {}
    for seed in summary["network_seeds"]:
        positive = [
            candidate for candidate in candidates
            if _candidate_mode_support(summary, candidate, seed, "A") > 0
        ]
        by_seed[str(seed)] = {
            "n_candidates_with_mode_A": len(positive),
            "candidate_ids_with_mode_A": positive,
            "maximum_mode_A_events": max((
                _candidate_mode_support(summary, candidate, seed, "A")
                for candidate in candidates
            ), default=0),
        }
    shared = [
        candidate for candidate in candidates
        if all(
            _candidate_mode_support(summary, candidate, seed, "A") > 0
            for seed in summary["network_seeds"]
        )
    ]
    return {
        "n_candidates": len(candidates),
        "network_seeds": summary["network_seeds"],
        "by_seed": by_seed,
        "shared_mode_A_candidate_ids": shared,
        "n_shared_mode_A_candidates": len(shared),
    }


def phase_candidate_audit(summary, candidate_id):
    return {
        "candidate_id": candidate_id,
        "network_seeds": summary["network_seeds"],
        "networks_with_mode_A": int(next(
            row["networks_with_clean_A"] for row in summary["candidate_rows"]
            if row["candidate_id"] == candidate_id
        )),
        "networks_with_mode_B": int(next(
            row["networks_with_clean_B"] for row in summary["candidate_rows"]
            if row["candidate_id"] == candidate_id
        )),
        "networks_with_both_modes": int(next(
            row["networks_with_both_clean_modes"] for row in summary["candidate_rows"]
            if row["candidate_id"] == candidate_id
        )),
        "mode_A_events_by_seed": {
            str(seed): _candidate_mode_support(summary, candidate_id, seed, "A")
            for seed in summary["network_seeds"]
        },
        "mode_B_events_by_seed": {
            str(seed): _candidate_mode_support(summary, candidate_id, seed, "B")
            for seed in summary["network_seeds"]
        },
    }


def build_verdict(fit, selection, confirmation, confirmation_manifest):
    fit_audit = fit_library_oracle(fit)
    selection_id = selection["diagnostic_best_candidate_id"]
    frozen_id = confirmation_manifest["selection_freeze"][
        "selected_nonzero_candidate_id"
    ]
    selection_audit = phase_candidate_audit(selection, selection_id)
    confirmation_audit = phase_candidate_audit(confirmation, frozen_id)
    baseline = phase_candidate_audit(confirmation, "edge_noop")
    closes_family = bool(
        fit_audit["n_shared_mode_A_candidates"] == 0
        and selection_audit["networks_with_mode_A"] == 0
        and confirmation_audit["networks_with_mode_A"] == 0
    )
    return {
        "status": (
            "REV10R2_STATIC_CONTINUOUS_EDGE_ROUTE_NOT_OBSERVED"
            if closes_family else "REV10R2_STATIC_EDGE_ROUTE_UNRESOLVED"
        ),
        "scientific_scope": (
            "finite 32-direction observation-invariant continuous spatial edge "
            "library at raw logit RMS 0.15 with target-normalized incoming-E budget"
        ),
        "fit_library_oracle": fit_audit,
        "selection": selection_audit,
        "confirmation": confirmation_audit,
        "confirmation_exact_noop": baseline,
        "confirmation_diagnostic_best_candidate_id": confirmation[
            "diagnostic_best_candidate_id"
        ],
        "claim": (
            "The tested static continuous edge redistribution preserved mode B "
            "but did not restore a shared returned mode-A repertoire across fresh "
            "networks; the frozen nonzero candidate did not outperform exact no-op "
            "in confirmation."
        ),
        "optimizer_decision": (
            "Do not compare optimizers: no shared known-good mode-A basin was "
            "observed, and the objective already penalized missing mode A."
        ),
        "beta_decision": (
            "Keep beta closed: the residual is categorical mode-A support, not "
            "a demonstrated radial-width or effective-delay-scale mismatch."
        ),
        "next_mechanism": (
            "Test a frozen low-dimensional dynamic accessibility or inhibitory-state "
            "variable that can switch route occupancy on one static substrate; do "
            "not add contact-conditioned bases, Gaussian cores, or more K."
        ),
        "claim_boundaries": [
            "finite library negative, not a proof over every coefficient in R12",
            "development patient target, not patient-blind generalization",
            "interictal event repertoire only, not ictal lifecycle reproduction",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--out")
    args = parser.parse_args()
    root = Path(args.root)
    paths = {
        "fit": root / "spatial_edge_flow_r2_1/fit_screen_summary_returned_only.json",
        "selection": root / "spatial_edge_flow_r2_1_selection/selection_summary_returned_only.json",
        "confirmation": root / "spatial_edge_flow_r2_1_confirmation/confirmation_summary_returned_only.json",
        "confirmation_manifest": root / "spatial_edge_flow_r2_1_confirmation/candidate_manifest.json",
    }
    fit = _load_summary(paths["fit"], "REV10R_RETURNED_ONLY_FIT_SCREEN_COMPLETE")
    selection = _load_summary(
        paths["selection"], "REV10R_RETURNED_ONLY_SELECTION_COMPLETE",
    )
    confirmation = _load_summary(
        paths["confirmation"], "REV10R_RETURNED_ONLY_CONFIRMATION_COMPLETE",
    )
    confirmation_manifest = json.loads(paths["confirmation_manifest"].read_text())
    if confirmation_manifest.get("status") != (
            "REV10R2_SPATIAL_EDGE_CONFIRMATION_LIBRARY_FROZEN"):
        raise RuntimeError("confirmation manifest is invalid")
    payload = build_verdict(
        fit, selection, confirmation, confirmation_manifest,
    )
    payload["inputs"] = {
        key: {"path": str(path), "sha256": _sha256(path)}
        for key, path in paths.items()
    }
    output = Path(args.out or root / "spatial_edge_flow_r2_1_confirmation/final_verdict.json")
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"], "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
