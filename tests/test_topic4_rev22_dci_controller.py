import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_topic4_rev22_dci_controller import (
    WORKER_COMPLETE,
    _artifact_complete,
    _contracts_unchanged,
    _job_state,
    _worker_limit,
    expand_jobs,
)


COMMIT = "a" * 40


def _candidate(candidate_id, block, cube, *, reference=False):
    names = ("g_LEE", "g_LEI", "theta_FT_deg", "AR_FT")
    return {
        "candidate_id": candidate_id,
        "block": block,
        "is_reference": reference,
        "unit_cube": dict(zip(names, cube)),
    }


def _manifest():
    rows = [
        _candidate("dci_p000", "reference", (0.5, 2 / 3, 0.5, 0.5), reference=True),
        _candidate("dci_p001", "full4d", (0.45, 0.55, 0.45, 0.55)),
        _candidate("dci_p002", "geometry_plane", (0.5, 2 / 3, 0.01, 0.99)),
        _candidate("dci_p003", "dose_plane", (0.48, 0.52, 0.5, 0.5)),
        _candidate("dci_p004", "dose_plane", (0.35, 0.65, 0.5, 0.5)),
    ]
    return {"candidate_count": len(rows), "candidates": rows}


def _seeds():
    return {
        "fit": {"units": [
            {"topology_seed": 2511, "dynamics_seed": 2511},
            {"topology_seed": 2512, "dynamics_seed": 2512},
        ]},
        "variance_decomposition_block": {
            "candidates": ["dci_p000", "dci_p001"],
            "units": [
                {"topology_seed": topology, "dynamics_seed": dynamics,
                 "new_run": dynamics != topology}
                for topology in (2511, 2512, 2513, 2514)
                for dynamics in (topology, 3101, 3102)
            ],
            "new_trajectories": 16,
        },
        "qualification": {"units": [
            {"topology_seed": 2601, "dynamics_seed": 2601},
            {"topology_seed": 2602, "dynamics_seed": 2602},
        ]},
        "confirmation": {"units": [
            {"topology_seed": 2621, "dynamics_seed": 2621},
            {"topology_seed": 2622, "dynamics_seed": 2622},
            {"topology_seed": 2623, "dynamics_seed": 2623},
        ]},
    }


def test_fit_expands_every_design_candidate_by_every_fit_unit(tmp_path):
    jobs = expand_jobs("fit", _manifest(), _seeds(), tmp_path, COMMIT)
    assert len(jobs) == 5 * 2
    assert {
        (job["candidate_id"], job["topology_seed"], job["dynamics_seed"])
        for job in jobs
    } == {
        (candidate, seed, seed)
        for candidate in ("dci_p000", "dci_p001", "dci_p002", "dci_p003", "dci_p004")
        for seed in (2511, 2512)
    }


def test_canary_uses_frozen_design_roles_on_first_fit_unit(tmp_path):
    jobs = expand_jobs("canary", _manifest(), _seeds(), tmp_path, COMMIT)
    assert len(jobs) == 4
    assert {job["candidate_id"] for job in jobs} == {
        "dci_p000", "dci_p001", "dci_p002", "dci_p003",
    }
    assert {(job["topology_seed"], job["dynamics_seed"]) for job in jobs} == {(2511, 2511)}


def test_decomposition_launches_only_sixteen_new_split_seed_runs(tmp_path):
    jobs = expand_jobs("decomposition", _manifest(), _seeds(), tmp_path, COMMIT)
    assert len(jobs) == 16
    assert all(job["topology_seed"] != job["dynamics_seed"] for job in jobs)
    assert {job["dynamics_seed"] for job in jobs} == {3101, 3102}
    assert {job["candidate_id"] for job in jobs} == {"dci_p000", "dci_p001"}


def test_equal_and_split_seed_jobs_have_unique_explicit_names(tmp_path):
    fit = expand_jobs("fit", _manifest(), _seeds(), tmp_path, COMMIT)[0]
    split = expand_jobs("decomposition", _manifest(), _seeds(), tmp_path, COMMIT)[0]
    assert fit["stem"].endswith("_topo_2511_dyn_2511")
    assert fit["seed_mode"] == "legacy"
    assert split["stem"].endswith(f"_topo_{split['topology_seed']}_dyn_{split['dynamics_seed']}")
    assert split["seed_mode"] == "split"
    assert fit["unit"] != split["unit"]


def test_qualification_requires_and_obeys_task8_freeze_file(tmp_path):
    with pytest.raises(RuntimeError, match="Task 8 frozen candidate"):
        expand_jobs("qualification", _manifest(), _seeds(), tmp_path, COMMIT)
    fit = tmp_path / "fit.json"
    response = tmp_path / "response.json"
    fit.write_text("fit")
    response.write_text("response")
    bindings = {
        "execution_candidate_manifest_sha256": "candidate-hash",
        "seed_manifest_sha256": "seed-hash",
        "response_design_manifest_sha256": "design-hash",
    }
    frozen = tmp_path / "frozen_candidates.json"
    frozen.write_text(json.dumps({
        "schema_id": "topic4_rev22_dci_frozen_candidates_v1",
        **bindings,
        "input_hashes": {
            "fit_aggregate": {"path": str(fit), "sha256": hashlib.sha256(fit.read_bytes()).hexdigest()},
            "response_fit": {"path": str(response), "sha256": hashlib.sha256(response.read_bytes()).hexdigest()},
        },
        "candidate_ids": ["dci_p001", "dci_p003"],
    }))
    jobs = expand_jobs(
        "qualification", _manifest(), _seeds(), tmp_path, COMMIT,
        frozen_candidates_path=frozen, frozen_bindings=bindings,
    )
    assert len(jobs) == 2 * 2
    assert {job["candidate_id"] for job in jobs} == {"dci_p001", "dci_p003"}
    payload = json.loads(frozen.read_text())
    payload["candidate_ids"] = ["not_in_execution_manifest"]
    frozen.write_text(json.dumps(payload))
    with pytest.raises(RuntimeError, match="outside execution manifest"):
        expand_jobs(
            "confirmation", _manifest(), _seeds(), tmp_path, COMMIT,
            frozen_candidates_path=frozen, frozen_bindings=bindings,
        )


def test_qualification_rejects_stale_task8_binding(tmp_path):
    frozen = tmp_path / "frozen_candidates.json"
    frozen.write_text(json.dumps({
        "schema_id": "topic4_rev22_dci_frozen_candidates_v1",
        "candidate_ids": ["dci_p001"],
        "execution_candidate_manifest_sha256": "old",
    }))
    with pytest.raises(RuntimeError, match="binding mismatch"):
        expand_jobs(
            "qualification", _manifest(), _seeds(), tmp_path, COMMIT,
            frozen_candidates_path=frozen,
            frozen_bindings={"execution_candidate_manifest_sha256": "new"},
        )


def _write_complete_artifact(job, config_sha):
    job["npz"].parent.mkdir(parents=True, exist_ok=True)
    job["npz"].write_bytes(b"frozen arrays")
    digest = hashlib.sha256(job["npz"].read_bytes()).hexdigest()
    job["json"].write_text(json.dumps({
        "status": WORKER_COMPLETE,
        "candidate_id": job["candidate_id"],
        "seed": job["topology_seed"],
        "topology_seed": job["topology_seed"],
        "dynamics_seed": job["dynamics_seed"],
        "seed_mode": job["seed_mode"],
        "arrays": {"path": str(job["npz"]), "sha256": digest},
        "provenance": {
            "expected_git_commit": COMMIT,
            "config_sha256": config_sha,
            "config_sha256_at_expected_commit": config_sha,
            "runtime_modules_match_expected_commit": True,
            "runtime_modules_dirty": False,
        },
    }))


def test_resume_skips_complete_artifact_and_rejects_partial_output(tmp_path):
    config_sha = "b" * 64
    complete, partial = expand_jobs("fit", _manifest(), _seeds(), tmp_path, COMMIT)[:2]
    _write_complete_artifact(complete, config_sha)
    assert _artifact_complete(complete, COMMIT, config_sha)
    assert _job_state(complete, COMMIT, config_sha) == "complete"
    partial["npz"].parent.mkdir(parents=True, exist_ok=True)
    partial["npz"].write_bytes(b"orphan")
    assert _job_state(partial, COMMIT, config_sha) == "invalid_artifact"


def test_worker_limit_reserves_32_gib_and_caps_at_sixteen():
    assert _worker_limit(
        available_gib=200.0, worker_gib=8.0, configured_cap=30, running=0,
    ) == 16
    assert _worker_limit(
        available_gib=80.0, worker_gib=16.0, configured_cap=16, running=1,
    ) == 3
    assert _worker_limit(
        available_gib=31.0, worker_gib=8.0, configured_cap=16, running=0,
    ) == 0


def test_contract_hash_drift_is_fail_closed(tmp_path):
    contract = tmp_path / "manifest.json"
    contract.write_text("frozen")
    frozen_hash = hashlib.sha256(contract.read_bytes()).hexdigest()
    assert _contracts_unchanged({contract: frozen_hash})
    contract.write_text("changed")
    assert not _contracts_unchanged({contract: frozen_hash})
