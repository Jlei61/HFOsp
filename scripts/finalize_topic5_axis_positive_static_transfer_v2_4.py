#!/usr/bin/env python3
"""Build the v2.4 closeout manifest after every frozen analysis is complete."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic5_rnn_axis_positive_static_transfer_v2_4"
FIGURE = (
    ROOT
    / "results/paper-ready-figure/"
    "fig6_rnn_axis_static_transfer_v2_4/figures"
)
SEEDS = (17, 29, 43)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    required = {
        "input_audit": BASE / "input_audit/INPUT_AUDIT_STATUS.json",
        "stage_a0": BASE / "axis_readback_stage_a0/STAGE_A0_STATUS.json",
        "stage_a1": BASE / "formal/AXIS_SELECTION_GATE_STATUS.json",
        "axis_launcher": BASE / "formal/AXIS_SEARCH_LAUNCHER_STATE.json",
        "representation_freeze": (
            BASE / "representations/REPRESENTATION_FREEZE_MANIFEST.json"
        ),
        "target_unlock": BASE / "TARGET_UNLOCK.json",
        "rank_fidelity": (
            BASE / "representations/RANK_DISTRIBUTION_FIDELITY.json"
        ),
        "static_readout": (
            BASE / "static_readout/STATIC_READOUT_GATE_STATUS.json"
        ),
        "static_diagnostics": (
            BASE / "static_readout/STATIC_READOUT_DIAGNOSTICS.json"
        ),
        "target_read_state": (
            BASE / "static_readout/TARGET_READ_STATE.json"
        ),
        "figure_png": FIGURE / "fig6_rnn_axis_static_transfer_v2_4.png",
        "figure_pdf": FIGURE / "fig6_rnn_axis_static_transfer_v2_4.pdf",
        "figure_metadata": (
            FIGURE / "fig6_rnn_axis_static_transfer_v2_4_metadata.json"
        ),
        "figure_readme": FIGURE / "README.md",
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise SystemExit("missing closeout artifacts:\n" + "\n".join(missing))
    audit = read_json(required["input_audit"])
    a0 = read_json(required["stage_a0"])
    a1 = read_json(required["stage_a1"])
    launcher = read_json(required["axis_launcher"])
    representation = read_json(required["representation_freeze"])
    unlock = read_json(required["target_unlock"])
    rank_fidelity = read_json(required["rank_fidelity"])
    static = read_json(required["static_readout"])
    diagnostics = read_json(required["static_diagnostics"])
    target_read = read_json(required["target_read_state"])
    if audit.get("status") != "PASS":
        raise SystemExit("input audit did not pass")
    for name, payload in (
        ("stage_a0", a0),
        ("stage_a1", a1),
        ("rank_fidelity", rank_fidelity),
        ("static_readout", static),
        ("static_diagnostics", diagnostics),
    ):
        if payload.get("status") != "COMPLETE":
            raise SystemExit(f"{name} is incomplete")
    if (
        launcher.get("status") != "COMPLETE"
        or launcher.get("n_tasks_finished") != 27
        or launcher.get("n_tasks_failed") != 0
    ):
        raise SystemExit("axis launcher did not close 27/27 cleanly")
    if representation.get("status") != "FROZEN_INTERICTAL_REPRESENTATIONS":
        raise SystemExit("representations were not frozen")
    if unlock.get("target_values_read"):
        raise SystemExit("unlock chronology drifted")
    if not target_read.get("target_values_read"):
        raise SystemExit("target read was not provenance-recorded")

    subjects = list(map(str, audit["axis_positive_primary_patients"]))
    run_rows: list[dict[str, Any]] = []
    for subject in subjects:
        for seed in SEEDS:
            root = BASE / "formal/axis_search" / subject / f"seed_{seed}"
            config = read_json(root / "resolved_config.json")
            metrics = read_json(root / "metrics.json")
            selection = read_json(root / "AXIS_SELECTION_FROZEN.json")
            candidate_complete = sum(
                (
                    root
                    / f"candidate_{axis_index:02d}"
                    / "axis_two_state_no_source"
                    / "COMPLETE"
                ).exists()
                for axis_index in range(32)
            )
            if (
                config.get("device") != "cpu"
                or candidate_complete != 32
                or metrics.get("status") != "COMPLETE"
                or metrics.get("target_values_read")
                or selection.get("status") != "FROZEN"
                or selection.get("heldout_values_read_for_selection")
            ):
                raise SystemExit(
                    f"{subject}/seed{seed}: formal candidate audit failed"
                )
            run_rows.append(
                {
                    "subject": subject,
                    "seed": seed,
                    "device": config["device"],
                    "n_candidate_complete": candidate_complete,
                    "selected_axis_index": metrics["selected_axis_index"],
                    "heldout_read_after_axis_freeze": metrics[
                        "heldout_values_read_after_axis_freeze"
                    ],
                    "target_values_read": metrics["target_values_read"],
                }
            )

    code_paths = [
        ROOT / "src/topic5_axis_positive_static_transfer_v2_4.py",
        ROOT / "scripts/audit_topic5_axis_positive_static_transfer_v2_4.py",
        ROOT / "scripts/analyze_topic5_axis_readback_stage_a0_v2_4.py",
        ROOT / "scripts/train_topic5_rnn_candidate_axis_v2_4.py",
        ROOT / "scripts/launch_topic5_rnn_candidate_axis_v2_4.py",
        ROOT / "scripts/watch_topic5_rnn_candidate_axis_v2_4.py",
        ROOT / "scripts/analyze_topic5_rnn_candidate_axis_v2_4.py",
        ROOT
        / "scripts/"
        "run_topic5_rnn_candidate_axis_e958_chunk_resume_v2_4.sh",
        ROOT / "scripts/build_topic5_rnn_rank_distributions_v2_4.py",
        ROOT / "scripts/finalize_topic5_rnn_rank_distributions_v2_4.py",
        ROOT / "scripts/launch_topic5_rnn_rank_distributions_v2_4.py",
        ROOT / "scripts/run_topic5_source_free_static_readout_v2_4.py",
        ROOT / "scripts/analyze_topic5_static_readout_diagnostics_v2_4.py",
        ROOT / "scripts/analyze_topic5_rank_distribution_fidelity_v2_4.py",
        ROOT
        / "scripts/paper_figures/"
        "plot_fig6_topic5_rnn_axis_static_transfer_v2_4.py",
        ROOT / "tests/test_topic5_axis_positive_static_transfer_v2_4.py",
    ]
    missing_code = [str(path) for path in code_paths if not path.exists()]
    if missing_code:
        raise SystemExit("missing code paths:\n" + "\n".join(missing_code))
    payload = {
        "contract": "topic5_axis_positive_static_transfer_closeout_v2_4",
        "status": "COMPLETE",
        "scientific_contract_execution": "COMPLETE_TO_FROZEN_STOP_RULES",
        "cohorts": {
            "physical_axis_formal": audit["physical_axis_formal_n"],
            "axis_positive": audit["axis_positive_primary_n"],
            "axis_reversed": audit["axis_reversed_n"],
            "target_ready": audit["target_metadata_eligible_n"],
            "axis_positive_target_overlap": len(
                audit["axis_positive_target_metadata_intersection"]
            ),
        },
        "gates": {
            "axis_positive_construct_validity": a1[
                "gate_a_axis_positive_construct_validity"
            ],
            "source_free_static_readout": static[
                "gate_s_source_free_static_readout"
            ],
            "history_contribution": static[
                "gate_h_history_contribution"
            ],
            "axis_contribution": static["gate_x_axis_contribution"],
            "dynamic_source_conditioned_rollout": static[
                "dynamic_source_conditioned_rollout"
            ],
        },
        "formal_axis_runs": run_rows,
        "resource_recovery_audit": {
            "formal_device": "cpu",
            "quarantine_root": str(
                (
                    BASE / "formal/quarantine"
                ).relative_to(ROOT)
            ),
            "quarantined_gpu_mixed_candidates_excluded": True,
            "reason": (
                "a resource benchmark briefly resumed incomplete E958 "
                "candidates on GPU; every affected candidate directory was "
                "moved out of the formal tree and retrained from a clean CPU "
                "state before aggregation"
            ),
        },
        "target_chronology": {
            "representations_frozen_before_target_read": True,
            "target_values_read": True,
            "dynamic_source_metadata_available": False,
        },
        "rnn_selected_axis_static_sensitivity": (
            "NOT_RUN_TO_PRESERVE_TARGET_BLIND_FREEZE; A1 completed after the "
            "five primary representations had been frozen and the static "
            "target had been read"
        ),
        "code_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in code_paths
        },
        "artifact_sha256": {
            key: sha256(path) for key, path in required.items()
        },
        "tests": {
            "command": (
                "python -m pytest -q "
                "tests/test_topic5_axis_positive_static_transfer_v2_4.py "
                "tests/test_topic5_competitive_propagation_v2_3.py "
                "tests/test_topic5_transition_decomposition_v0_1.py"
            ),
            "expected": "25 passed",
        },
    }
    atomic_json(BASE / "FINAL_REPRODUCIBILITY_MANIFEST.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
