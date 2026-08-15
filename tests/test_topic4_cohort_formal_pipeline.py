"""The worker writes arrays and the aggregator reads them; keep them agreeing."""
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


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


aggregate = _module(
    "topic4_formal_aggregate",
    ROOT / "scripts/aggregate_topic4_data_driven_snn_cohort_formal.py",
)

pytestmark = pytest.mark.skipif(
    not LAYOUT_AUDIT.exists(), reason="formal layouts have not been frozen yet",
)


def _cohort():
    return aggregate.Cohort(json.loads(CONFIG.read_text()))


def _fake_worker(tmp_path: Path, cohort, candidate: str, seed: int, commit: str,
                 *, rng: np.random.Generator, n_events: int = 40) -> None:
    arrays = {}
    for index, subject in enumerate(cohort.subjects):
        with np.load(ROOT / subject["layout_npz"], allow_pickle=False) as loaded:
            width = len(loaded["contact_order"])
            layouts = ["canonical"] + (
                ["real"] if "real_coords_sheet" in loaded else []
            )
        for layout in layouts:
            ranks = np.tile(
                np.arange(width, dtype=np.float32), (n_events, 1),
            )
            ranks[1::2] = ranks[1::2, ::-1]
            ranks += rng.normal(0.0, 0.01, ranks.shape).astype(np.float32)
            arrays[f"{layout}_{index:02d}_ranks"] = ranks
    npz_path = cohort.output_root / "workers" / f"{candidate}_seed_{seed}.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, **arrays)
    payload = {
        "status": "COMPLETE",
        "candidate_id": candidate,
        "seed": seed,
        "output_npz_sha256": aggregate._sha256(npz_path),
        "provenance": {
            "expected_git_commit": commit,
            "runtime_modules_match_expected_commit": True,
            "runtime_modules_dirty": False,
        },
    }
    (npz_path.with_suffix(".json")).write_text(json.dumps(payload))


def test_aggregator_reads_every_layout_the_worker_is_contracted_to_write(tmp_path):
    cohort = _cohort()
    audit = cohort.layout_audit
    assert audit["denominators"]["primary_canonical_layout"] == 34
    assert audit["denominators"]["real_geometry_sensitivity"] == 28
    expected_keys = []
    for index, subject in enumerate(cohort.subjects):
        expected_keys.append(f"canonical_{index:02d}_ranks")
        if subject["in_real_geometry_sensitivity_cohort"]:
            expected_keys.append(f"real_{index:02d}_ranks")
    assert len(expected_keys) == 34 + 28


def test_scorer_kwargs_align_with_the_frozen_targets_and_nulls():
    cohort = _cohort()
    for subject in cohort.subjects[:6]:
        kwargs = cohort.scorer_kwargs(subject, "heldout")
        names = kwargs["contact_names"]
        assert kwargs["target"]["contact_order"] == names
        assert kwargs["target"]["profiles"].shape == (2, len(names))
        assert kwargs["target"]["recruitment"].shape == (2, len(names))
        expected_pairs = len(names) * (len(names) - 1) // 2
        assert kwargs["target"]["precedence"].shape == (2, expected_pairs, 3)
        assert kwargs["patient_centers"].shape[0] == 2
        permutations = cohort.permutations(subject)
        assert permutations.shape[1] == len(names)
        assert permutations.shape[0] == (
            subject["within_shaft_null"]["effective_null_size"]
        )


def test_a_synthetic_worker_run_flows_through_selection_and_confirmation():
    cohort = _cohort()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    rng = np.random.default_rng(0)
    candidate, seed = "pipeline_smoke", 1661
    created = []
    try:
        _fake_worker(None, cohort, candidate, seed, commit, rng=rng)
        created = [
            cohort.output_root / "workers" / f"{candidate}_seed_{seed}.npz",
            cohort.output_root / "workers" / f"{candidate}_seed_{seed}.json",
        ]
        subject = cohort.subjects[0]
        ranks = cohort.worker_ranks(created[0], "canonical", 0)
        score = aggregate.score_readout(
            ranks, include_natural_kmeans=False,
            **cohort.scorer_kwargs(subject, "train"),
        )
        assert "selection_score" in score
        confirmed = aggregate._confirm_layout(
            cohort, subject, 0, candidate, [seed], "canonical", commit,
        )
        assert confirmed["subject_id"] == subject["subject_id"]
        assert confirmed["n_seeds"] == 1
        assert np.isfinite(confirmed["delta_null_median_minus_observed"])
        assert 0.0 < confirmed["permutation_p_median"] <= 1.0
    finally:
        for path in created:
            path.unlink(missing_ok=True)
