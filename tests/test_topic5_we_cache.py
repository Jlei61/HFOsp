"""Contract tests for the WE-SLP-RNN v0.3 cache builder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_topic5_we_cache import (  # noqa: E402
    COHORT,
    FIELD_DIR,
    MIN_PARTICIPATING,
    densify_ranks,
    plane_scopes,
)
from src.interictal_propagation import _valid_event_indices  # noqa: E402


def test_densify_closes_rank_gaps_left_by_dropped_contacts():
    # Rank 1 is gone because the contact that carried it is not in the montage
    # intersection; the event must become a 0,1 event, not a 0,_,2 event.
    ranks = np.array([[0, 2, 2], [-1, 0, 3]], np.int16)
    out = densify_ranks(ranks)
    assert out.tolist() == [[0, 1, 1], [-1, 0, 1]]


def test_densify_leaves_absent_contacts_absent():
    out = densify_ranks(np.array([[-1, -1, 5]], np.int16))
    assert out.tolist() == [[-1, -1, 0]]


def test_cohort_splits_into_eleven_shared_and_ten_own_plane_patients():
    shared, split = [], []
    for subject in COHORT:
        scopes = sorted(plane_scopes(subject))
        (shared if scopes == ["shared"] else split).append(subject)
        assert scopes in (["shared"], ["own_a", "own_b"]), subject
    assert len(shared) == 11, shared
    assert len(split) == 10, split
    assert len(shared) + len(split) == len(COHORT)


def test_every_patient_has_a_solved_plane_for_every_scope_it_contributes():
    for subject in COHORT:
        for scope, plane in plane_scopes(subject).items():
            assert plane["status"] == "ok", (subject, scope)
            points = np.asarray(plane["points"], float)
            assert points.ndim == 2 and points.shape[1] == 2
            assert float(plane["scale_mm"]) > 0.0


def test_plane_contact_order_matches_the_gradient_field_contact_order():
    for subject in COHORT:
        field = json.loads((FIELD_DIR / f"{subject}.json").read_text())["interictal_field"]
        n = len(field["contact_order"])
        for scope, plane in plane_scopes(subject).items():
            assert len(plane["points"]) == n, (subject, scope)


def test_label_join_bridge_is_reconstructible_for_a_patient_whose_subsets_differ():
    # yuquan_chengshuai has 27632 raw events but only 27577 with >= 3
    # participating channels, so labels[event_source_index] would silently
    # mislabel.  The valid-event bridge has to reproduce the label length.
    from src.interictal_propagation import load_subject_propagation_events
    from build_topic5_we_cache import PROP_DIR, lagpat_dir

    subject = "yuquan_chengshuai"
    labels = json.loads((PROP_DIR / f"{subject}.json").read_text())["adaptive_cluster"]["labels"]
    bools = load_subject_propagation_events(lagpat_dir(subject))["bools"]
    valid = _valid_event_indices(bools, min_participating=MIN_PARTICIPATING)
    assert bools.shape[1] > len(labels), "this patient is supposed to have invalid events"
    assert len(valid) == len(labels)


def test_label_join_refuses_to_guess_when_the_bridge_does_not_reconcile(monkeypatch):
    import build_topic5_we_cache as mod

    monkeypatch.setattr(mod, "load_subject_propagation_events",
                        lambda _d: {"bools": np.ones((4, 10), bool)})
    monkeypatch.setattr(mod, "_valid_event_indices", lambda _b, min_participating: np.arange(7))
    with pytest.raises(RuntimeError, match="not certified"):
        mod.event_mode_labels("epilepsiae_1146", np.arange(5))
