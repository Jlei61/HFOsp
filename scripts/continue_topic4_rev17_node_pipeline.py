#!/usr/bin/env python3
"""Continue rev17 Node-only selection through frozen intervention, fail closed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
PYTHON = Path("/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python")
RESULT_ROOT = ARTIFACT_ROOT / (
    "results/topic4_sef_hfo/data_driven_node_dualmode_rev17"
)
ATLAS_AGGREGATE = RESULT_ROOT / (
    "dual_field_residual_atlas/analysis/dual_field_response_aggregate.json"
)
SELECTION_AGGREGATE = RESULT_ROOT / (
    "dual_field_selection/analysis/fresh_selection_aggregate.json"
)
CONFIRMATION_AUDIT = RESULT_ROOT / (
    "node_confirmation/analysis/confirmation_audit.json"
)
POSTSELECTION_AUDIT = RESULT_ROOT / (
    "node_postselection/analysis/node_postselection_audit.json"
)
FINAL_SCIENCE_AUDIT = RESULT_ROOT / (
    "node_final_science/analysis/node_final_science_audit.json"
)
INTERVENTION_AGGREGATE = RESULT_ROOT / (
    "node_final_science/intervention/analysis/node_intervention_aggregate.json"
)
STATUS_PATH = RESULT_ROOT / "status/rev17_node_continuation_controller.json"
NUMERIC_ENV = {
    "BLIS_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
}


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"rev17 continuation input is missing: {path}")
    return json.loads(path.read_text())


def _head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()


def _worktree_status() -> list[str]:
    return subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=ROOT, text=True,
    ).splitlines()


def _write_status(stage: str, status: str, **details: Any) -> None:
    _atomic_json(STATUS_PATH, {
        "schema_id": "topic4_rev17_node_continuation_controller_v1",
        "stage": stage, "status": status, "git_commit": _head(),
        "worktree_status": _worktree_status(),
        "EE_EtoI_ZM": "off", "updated_at_epoch": time.time(), **details,
    })


def _notify(message: str) -> None:
    subprocess.run(
        ["notify-send", "Topic 4 rev17 Node", message], check=False,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def _run(arguments: Sequence[str | Path]) -> None:
    command = [str(value) for value in arguments]
    environment = os.environ.copy(); environment.update(NUMERIC_ENV)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def _python(script: str, *arguments: str | Path) -> None:
    _run([PYTHON, ROOT / script, *arguments])


def _commit_generated(path: Path, message: str) -> str:
    path = path.resolve()
    relative = path.relative_to(ROOT)
    dirty = _worktree_status()
    allowed = {f"?? {relative}", f" M {relative}", f"M  {relative}",
               f"A  {relative}", f"AM {relative}"}
    unexpected = [row for row in dirty if row not in allowed]
    if unexpected:
        raise RuntimeError(
            f"rev17 continuation worktree has unrelated changes: {unexpected}"
        )
    _run(["git", "add", "--", relative])
    staged = subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--", relative], cwd=ROOT,
        check=False,
    )
    if staged.returncode == 1:
        _run(["git", "commit", "-m", message, "--", relative])
    elif staged.returncode != 0:
        raise RuntimeError("rev17 continuation could not inspect staged config")
    if _worktree_status():
        raise RuntimeError("rev17 continuation worktree is not clean after config commit")
    return _head()


def _wait_for_atlas(interval_seconds: int) -> dict[str, Any]:
    _write_status("atlas", "WAITING_FOR_ATLAS_AGGREGATE")
    while not ATLAS_AGGREGATE.is_file():
        time.sleep(max(60, int(interval_seconds)))
    payload = _load(ATLAS_AGGREGATE)
    if payload.get("status") != "REV17_DUAL_FIELD_RESIDUAL_ATLAS_AGGREGATE_COMPLETE":
        raise RuntimeError("rev17 atlas aggregate is not complete")
    if payload.get("provenance", {}).get("analysis_worktree_clean") is not True:
        raise RuntimeError("rev17 atlas aggregate provenance is not formal")
    return payload


def _selection() -> dict[str, Any]:
    _write_status("selection", "PREPARING")
    config = ROOT / "config/topic4_rev17_dual_field_selection.json"
    _python("scripts/prepare_topic4_rev17_dual_field_selection.py")
    commit = _commit_generated(config, "Freeze rev17 dual-field selection config")
    _python(
        "scripts/freeze_topic4_rev17_dual_field_selection.py",
        "--config", config, "--expected-commit", commit,
    )
    _write_status("selection", "RUNNING_FRESH_NETWORKS", expected_commit=commit)
    _python(
        "scripts/launch_topic4_rev12_node_workers.py",
        "--config", config, "--seed-pool", "selection",
        "--expected-commit", commit, "--unit-prefix", "codex-t4-r17-selection",
        "--maximum-workers", "8", "--estimated-worker-gib", "14",
    )
    _python("scripts/aggregate_topic4_rev17_dual_field_selection.py")
    payload = _load(SELECTION_AGGREGATE)
    if payload.get("status") != "REV17_DUAL_FIELD_FRESH_SELECTION_AGGREGATE_COMPLETE":
        raise RuntimeError("rev17 fresh selection aggregate is invalid")
    if payload.get("selected_candidate_id") is None:
        _write_status("selection", "STOP_NO_ELIGIBLE_SELECTION_CANDIDATE")
        _notify("Stopped: no candidate passed all fresh-network selection clauses")
        return payload
    return payload


def _confirmation() -> dict[str, Any]:
    _write_status("confirmation", "PREPARING")
    config = ROOT / "config/topic4_rev17_node_confirmation.json"
    _python("scripts/prepare_topic4_rev17_node_confirmation.py")
    commit = _commit_generated(config, "Freeze rev17 unseen-network confirmation config")
    _python(
        "scripts/freeze_topic4_rev17_node_confirmation.py",
        "--config", config, "--expected-commit", commit,
    )
    _write_status("confirmation", "RUNNING_UNSEEN_NETWORKS", expected_commit=commit)
    _python(
        "scripts/launch_topic4_rev12_node_workers.py",
        "--config", config, "--seed-pool", "confirmation",
        "--expected-commit", commit, "--unit-prefix", "codex-t4-r17-confirmation",
        "--maximum-workers", "6", "--estimated-worker-gib", "14",
    )
    _python("scripts/audit_topic4_rev17_node_confirmation.py")
    payload = _load(CONFIRMATION_AUDIT)
    if (
        payload.get("status") != "REV17_NODE_CONFIRMATION_SCORED_COMPLETE"
        or payload.get("scientific_confirmation", {}).get("accepted") is not True
    ):
        _write_status("confirmation", "STOP_UNSEEN_CONFIRMATION_REJECTED")
        _notify("Stopped: frozen Node candidate failed unseen-network confirmation")
    return payload


def _postselection() -> dict[str, Any]:
    _write_status("natural_kmeans", "PREPARING")
    config = ROOT / "config/topic4_rev17_node_postselection.json"
    _python("scripts/prepare_topic4_rev17_node_postselection.py")
    _commit_generated(config, "Freeze rev17 natural KMeans audit config")
    _python("scripts/audit_topic4_rev17_node_postselection.py")
    payload = _load(POSTSELECTION_AUDIT)
    if payload.get("status") != "REV17_NODE_POSTSELECTION_ACCEPTED":
        _write_status("natural_kmeans", "STOP_NATURAL_KMEANS_REJECTED")
        _notify("Stopped: frozen Node candidate failed same-network natural KMeans")
    return payload


def _final_science() -> dict[str, Any]:
    _write_status("heldout_topology", "PREPARING")
    config = ROOT / "config/topic4_rev17_node_final_science.json"
    _python("scripts/prepare_topic4_rev17_node_final_science.py")
    _commit_generated(config, "Freeze rev17 held-out and source-topology audit config")
    _python("scripts/audit_topic4_rev17_node_final_science.py")
    payload = _load(FINAL_SCIENCE_AUDIT)
    if payload.get("status") != "REV17_NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION":
        _write_status("heldout_topology", "STOP_HELDOUT_OR_TOPOLOGY_REJECTED")
        _notify("Stopped: frozen Node candidate failed held-out or source topology")
    return payload


def _intervention() -> dict[str, Any]:
    _write_status("intervention", "PREPARING")
    config = ROOT / "config/topic4_rev17_node_intervention.json"
    _python("scripts/prepare_topic4_rev17_node_intervention.py")
    commit = _commit_generated(config, "Freeze rev17 same-checkpoint intervention config")
    _write_status("intervention", "RUNNING_SAME_CHECKPOINT", expected_commit=commit)
    _python(
        "scripts/monitor_topic4_rev17_node_intervention.py",
        "--config", config, "--expected-commit", commit,
        "--worker-cap", "3", "--execute",
    )
    payload = _load(INTERVENTION_AGGREGATE)
    if (
        payload.get("status") != "REV17_NODE_FIELD_FROZEN"
        or payload.get("node_freeze_permitted") is not True
    ):
        _write_status("intervention", "STOP_HOTSPOT_NOT_SELECTIVE")
        _notify("Stopped: same-checkpoint hotspots were not mode selective")
    return payload


def _render() -> None:
    _write_status("figures", "RENDERING")
    _python("scripts/paper_figures/plot_topic4_rev17_node_final_fig4.py")
    _python("scripts/paper_figures/plot_topic4_rev17_node_causal_validation.py")
    _write_status("complete", "REV17_NODE_PIPELINE_COMPLETE_AND_FROZEN")
    _notify("Complete: rev17 Node field frozen and final figures rendered")


def execute(interval_seconds: int = 600) -> str:
    if _worktree_status():
        raise RuntimeError("rev17 continuation must start from a clean worktree")
    _wait_for_atlas(interval_seconds)
    selection = _selection()
    if selection.get("selected_candidate_id") is None:
        return "STOP_NO_ELIGIBLE_SELECTION_CANDIDATE"
    confirmation = _confirmation()
    if confirmation.get("scientific_confirmation", {}).get("accepted") is not True:
        return "STOP_UNSEEN_CONFIRMATION_REJECTED"
    postselection = _postselection()
    if postselection.get("status") != "REV17_NODE_POSTSELECTION_ACCEPTED":
        return "STOP_NATURAL_KMEANS_REJECTED"
    final = _final_science()
    if final.get("status") != "REV17_NODE_FINAL_SCIENCE_ADVANCES_TO_INTERVENTION":
        return "STOP_HELDOUT_OR_TOPOLOGY_REJECTED"
    intervention = _intervention()
    if intervention.get("status") != "REV17_NODE_FIELD_FROZEN":
        return "STOP_HOTSPOT_NOT_SELECTIVE"
    _render()
    return "REV17_NODE_PIPELINE_COMPLETE_AND_FROZEN"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval-seconds", type=int, default=600)
    args = parser.parse_args()
    try:
        status = execute(interval_seconds=args.interval_seconds)
    except Exception as error:
        _write_status("controller", "FAILED", error=repr(error))
        _notify(f"Continuation failed: {error}")
        raise
    print(json.dumps({"status": status, "status_path": str(STATUS_PATH)}))


if __name__ == "__main__":
    main()
