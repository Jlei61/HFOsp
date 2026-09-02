import json
from pathlib import Path

from scripts.run_topic4_rev21_zm_screen_controller import (
    CANARY_CANDIDATES, build_jobs,
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
