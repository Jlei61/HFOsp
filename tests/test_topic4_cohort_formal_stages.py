"""Run all three aggregation stages on a synthetic tree.

The stages fire unattended hours apart; a crash in stage A stalls the whole
run, so the staged path is exercised here rather than discovered at 02:00.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
LAYOUT_AUDIT = (
    ROOT / "results/topic4_sef_hfo/data_driven_snn_cohort_v1/formal"
    / "cohort_layout_audit.json"
)
CANDIDATES = ("synthetic_aligned", "synthetic_scrambled")

pytestmark = pytest.mark.skipif(
    not LAYOUT_AUDIT.exists(), reason="formal layouts have not been frozen yet",
)


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


aggregate = _module(
    "topic4_formal_stage_aggregate",
    ROOT / "scripts/aggregate_topic4_data_driven_snn_cohort_formal.py",
)


def _write_worker(cohort, candidate: str, seed: int, commit: str,
                  *, scrambled: bool, rng: np.random.Generator) -> None:
    arrays = {}
    for index, subject in enumerate(cohort.subjects):
        with np.load(ROOT / subject["layout_npz"], allow_pickle=False) as loaded:
            width = len(loaded["contact_order"])
            layouts = ["canonical"] + (
                ["real"] if "real_coords_sheet" in loaded else []
            )
        events = 48
        base = np.tile(np.arange(width, dtype=float), (events, 1))
        base[1::2] = base[1::2, ::-1]
        base = base + rng.normal(0.0, 0.35, base.shape)
        if scrambled:
            base = base[:, rng.permutation(width)]
        base[rng.random(base.shape) < 0.2] = np.nan
        for layout in layouts:
            arrays[f"{layout}_{index:02d}_ranks"] = base.astype(np.float32)
    stem = cohort.output_root / "workers" / f"{candidate}_seed_{seed}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    stem.with_suffix(".json").write_text(json.dumps({
        "status": "COMPLETE", "candidate_id": candidate, "seed": seed,
        "output_npz_sha256": aggregate._sha256(stem.with_suffix(".npz")),
        "provenance": {
            "expected_git_commit": commit,
            "runtime_modules_match_expected_commit": True,
            "runtime_modules_dirty": False,
        },
    }))


@pytest.fixture(scope="module")
def staged(tmp_path_factory):
    root = tmp_path_factory.mktemp("formal_stages")
    config = json.loads(CONFIG.read_text())
    config["output_root"] = str(root)
    (root / "cohort_layout_audit.json").write_text(LAYOUT_AUDIT.read_text())
    (root / "candidate_manifest.json").write_text(json.dumps({
        "status": "TOPIC4_DATA_DRIVEN_SNN_COHORT_FORMAL_LIBRARY_FROZEN",
        "candidate_set": {
            "n_candidates": len(CANDIDATES),
            "candidates": [{"candidate_id": name} for name in CANDIDATES],
        },
    }))
    cohort = aggregate.Cohort(config)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    rng = np.random.default_rng(5)
    seeds = [
        int(seed) for key in ("fit_network_seeds", "selection_network_seeds",
                              "confirmation_network_seeds")
        for seed in config["search"][key]
    ]
    for candidate in CANDIDATES:
        for seed in seeds:
            _write_worker(cohort, candidate, seed, commit,
                          scrambled=candidate.endswith("scrambled"), rng=rng)
    return {"config": config, "cohort": cohort, "commit": commit, "root": root}


def test_stage_a_shortlists_without_touching_held_out_blocks(staged):
    payload = aggregate.run_stage_a(staged["cohort"], staged["commit"])
    assert payload["status"] == "STAGE_A_SHORTLIST_READY"
    assert payload["split_used"] == "patient train blocks only"
    assert len(payload["per_subject_ranking"]) == 34
    assert set(payload["stage_b_candidates"]) <= set(CANDIDATES)
    # The aligned candidate keeps contact identity; the scrambled one does not,
    # so selection should favour it for a clear majority of subjects.  Not all
    # of them: one synthetic rank pattern cannot match every patient's own
    # templates, and a scrambled column order can still land close on a small
    # montage.
    aligned_first = sum(
        rows[0]["candidate_id"] == "synthetic_aligned"
        for rows in payload["per_subject_ranking"].values()
    )
    assert aligned_first > len(payload["per_subject_ranking"]) / 2
    (staged["root"] / "stage_a_selection.json").write_text(
        json.dumps(aggregate._json_ready(payload))
    )


def test_stage_b_selects_one_candidate_per_subject(staged):
    payload = aggregate.run_stage_b(staged["cohort"], staged["commit"])
    assert payload["status"] == "STAGE_B_SELECTION_READY"
    assert len(payload["per_subject_selected_candidate"]) == 34
    assert set(payload["stage_c_candidates"]) <= set(CANDIDATES)
    assert sum(payload["selected_candidate_counts"].values()) == 34
    (staged["root"] / "stage_b_selection.json").write_text(
        json.dumps(aggregate._json_ready(payload))
    )


def test_stage_c_confirms_on_held_out_blocks_and_adjudicates(staged):
    payload = aggregate.run_stage_c(staged["cohort"], staged["commit"])
    assert payload["split_used"] == "patient held-out recording blocks"
    assert len(payload["canonical_subjects"]) == 34
    assert len(payload["real_geometry_subjects"]) == 28
    assert payload["denominators"]["primary_canonical_layout"] == 34
    cohort = payload["cohort"]
    assert 0.0 <= cohort["pass_fraction"] <= 1.0
    assert set(cohort["robustness"]["pass_by_confirmation_seed"]) == {
        "1681", "1682", "1683", "1684",
    }
    assert "contact_count_confound" in cohort["robustness"]
    assert payload["status"] in {
        "COHORT_MODEL_SUPPORT_SUPPORTED", "COHORT_MODEL_SUPPORT_INSUFFICIENT",
        "SAME_NETWORK_K2_INSUFFICIENT", "OBSERVATION_LAYOUT_DEPENDENCE_UNRESOLVED",
    }
    representative = payload["representative_subject"]["subject_id"]
    assert representative in {row["subject_id"] for row in payload["canonical_subjects"]}
    for row in payload["canonical_subjects"]:
        assert row["n_seeds"] == 4
        assert row["null_size"] if "null_size" in row else True
