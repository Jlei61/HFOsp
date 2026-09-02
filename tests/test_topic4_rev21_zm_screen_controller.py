import json
from pathlib import Path

from scripts.run_topic4_rev21_zm_screen_controller import (
    CANARY_CANDIDATES, _complete, _launch, build_jobs,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _inputs():
    config = json.loads((
        ROOT / "config/topic4_rev21_dual_core_zm_transition.json"
    ).read_text())
    manifest = json.loads((
        ARTIFACT_ROOT / "results/topic4_sef_hfo/data_driven_dual_core_zm_transition/"
        "coarse_candidate_manifest.json"
    ).read_text())
    return config, manifest


def test_canary_spans_both_zm_axes_without_using_patient_ictal_data(tmp_path):
    config, manifest = _inputs()
    jobs = build_jobs(config, manifest, tmp_path, "canary", "a" * 40)
    assert len(jobs) == len(CANARY_CANDIDATES) == 6
    assert len({job["topology_seed"] for job in jobs}) == 1
    assert len({job["dynamics_seed"] for job in jobs}) == 1
    assert any("si_0p7" in job["candidate_id"] for job in jobs)
    assert any("sm_2" in job["candidate_id"] for job in jobs)


def test_coarse_grid_uses_four_seed_cells_for_all_17_candidates(tmp_path):
    config, manifest = _inputs()
    jobs = build_jobs(config, manifest, tmp_path, "coarse", "b" * 40)
    assert len(jobs) == 17 * 2 * 2
    by_candidate = {}
    for job in jobs:
        by_candidate.setdefault(job["candidate_id"], set()).add((
            job["topology_seed"], job["dynamics_seed"],
        ))
    assert set(map(len, by_candidate.values())) == {4}
    assert "rev21_zm_off" in by_candidate


def test_timescale_grid_uses_the_same_four_seed_cells(tmp_path):
    config, manifest = _inputs()
    timescale = {"candidates": manifest["candidates"][:9]}
    jobs = build_jobs(config, timescale, tmp_path, "timescale", "c" * 40)
    assert len(jobs) == 9 * 4
    assert len({(job["topology_seed"], job["dynamics_seed"])
                for job in jobs}) == 4


def test_confirmation_uses_fresh_three_by_four_seed_matrix(tmp_path):
    config, manifest = _inputs()
    confirmation = {"candidates": manifest["candidates"][:4]}
    jobs = build_jobs(
        config, confirmation, tmp_path, "confirmation", "d" * 40,
    )
    assert len(jobs) == 4 * 3 * 4
    assert len({job["topology_seed"] for job in jobs}) == 3
    assert len({job["dynamics_seed"] for job in jobs}) == 4


def test_launcher_passes_phase_specific_candidate_manifest(monkeypatch, tmp_path):
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs

    monkeypatch.setattr(
        "scripts.run_topic4_rev21_zm_screen_controller.subprocess.run", fake_run,
    )
    job = {
        "candidate_id": "candidate",
        "topology_seed": 1,
        "dynamics_seed": 2,
        "unit": "unit",
        "json": tmp_path / "out.json",
        "npz": tmp_path / "out.npz",
        "status_path": tmp_path / "status",
        "log": tmp_path / "log",
    }
    manifest = tmp_path / "phase-manifest.json"
    _launch(
        job, config_path=tmp_path / "config.json",
        candidate_manifest=manifest, artifact_root=tmp_path,
        commit="e" * 40, phase="timescale",
    )
    command = captured["command"]
    index = command.index("--candidate-manifest")
    assert command[index + 1] == str(manifest)


def test_complete_rejects_artifact_from_an_old_phase_manifest(tmp_path):
    npz = tmp_path / "out.npz"
    npz.write_bytes(b"artifact")
    import hashlib
    digest = hashlib.sha256(npz.read_bytes()).hexdigest()
    job = {
        "candidate_id": "candidate", "topology_seed": 1,
        "dynamics_seed": 2, "json": tmp_path / "out.json", "npz": npz,
        "expected_commit": "a" * 40,
        "candidate_manifest_sha256": "b" * 64,
    }
    payload = {
        "status": "REV12ND_NODE_WORKER_COMPLETE",
        "candidate_id": "candidate", "topology_seed": 1,
        "dynamics_seed": 2, "model_ictal_rev21": {},
        "arrays": {"sha256": digest},
        "provenance": {
            "expected_git_commit": "a" * 40,
            "candidate_manifest_sha256": "c" * 64,
        },
    }
    job["json"].write_text(json.dumps(payload))
    assert not _complete(job)
    payload["provenance"]["candidate_manifest_sha256"] = "b" * 64
    job["json"].write_text(json.dumps(payload))
    assert _complete(job)
