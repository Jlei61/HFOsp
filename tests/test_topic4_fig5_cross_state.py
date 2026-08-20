import json
from pathlib import Path

import numpy as np
import pytest

from scripts.freeze_topic4_zm_discovery_boundary import load_audit_config
from src.topic4_fig5_cross_state import (
    NOT_EVALUABLE, evaluate_repertoire, rank_profile_similarity, score_events,
    shaft_groups)

ROOT = Path(__file__).resolve().parents[1]
ZM = ROOT / "results/topic4_sef_hfo/data_driven_zm_ictal_transition"
REFERENCE = (ROOT / "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc"
             / "frozen_substrate_confirmation/workers/joint_04_control_seed_1561.npz")

pytestmark = pytest.mark.skipif(
    not REFERENCE.exists(), reason="frozen reference worker absent in this checkout")


@pytest.fixture(scope="module")
def config():
    return load_audit_config()


@pytest.fixture(scope="module")
def contracts(config):
    from scripts.rescore_topic4_fig5_model_internal_candidates import (
        load_frozen_contracts)
    return load_frozen_contracts(config)


@pytest.fixture(scope="module")
def reference_events(contracts):
    with np.load(REFERENCE, allow_pickle=False) as handle:
        onsets = np.asarray(handle["onsets"], float)
        returned = np.asarray(handle["event_returned"], bool)
        t_on = np.asarray(handle["event_t_on_ms"], float)
        t_off = np.asarray(handle["event_t_off_ms"], float)
    scored = score_events(
        onsets, returned, np.ones_like(returned), groups=contracts["groups"],
        embedding=contracts["embedding"], classifier=contracts["classifier"],
        contact_xy=contracts["contact_xy"], contact_names=contracts["contact_names"],
        event_t_on_ms=t_on, event_t_off_ms=t_off)
    labels, clean = scored["labels"], scored["clean"]
    return {
        "onsets": onsets, "t_on": t_on, "t_off": t_off,
        "mode0": np.flatnonzero(clean & (labels == 0)),
        "mode1": np.flatnonzero(clean & (labels == 1)),
        "ood": np.flatnonzero(np.asarray(scored["ood"], bool) & returned),
    }


def _fixture(reference_events, index):
    index = np.asarray(index, int)
    return (reference_events["onsets"][index], np.ones(len(index), bool),
            np.ones(len(index), bool), reference_events["t_on"][index],
            reference_events["t_off"][index])


def _evaluate(config, contracts, onsets, returned, before, t_on, t_off,
              gate=None, duration_ms=20000.0, t_ictal_ms=None):
    return evaluate_repertoire(
        onsets, returned, before, groups=contracts["groups"],
        embedding=contracts["embedding"], classifier=contracts["classifier"],
        contact_xy=contracts["contact_xy"], contact_names=contracts["contact_names"],
        gate=gate or config["repertoire_gate"], duration_ms=duration_ms,
        event_t_on_ms=t_on, event_t_off_ms=t_off, t_ictal_ms=t_ictal_ms)


def test_reference_run_has_both_modes_available(reference_events):
    assert len(reference_events["mode0"]) >= 5
    assert len(reference_events["mode1"]) >= 5


def test_a_healthy_event_set_is_retained(config, contracts, reference_events):
    index = np.concatenate([reference_events["mode0"][:15],
                            reference_events["mode1"][:15]])
    row = _evaluate(config, contracts, *_fixture(reference_events, index))
    assert row["retained"] is True, row["failing_clauses"]
    assert row["measures"]["n_returned"] == 30
    assert min(row["measures"]["mode_counts"]) >= 3


def test_too_few_events_fails_only_the_count_clause(config, contracts,
                                                    reference_events):
    index = np.concatenate([reference_events["mode0"][:6],
                            reference_events["mode1"][:6]])
    row = _evaluate(config, contracts, *_fixture(reference_events, index))
    assert row["failing_clauses"] == ["n_returned_before_onset_at_least_20"]


def test_a_missing_mode_fails_the_mode_clause(config, contracts, reference_events):
    index = np.concatenate([reference_events["mode0"][:24],
                            reference_events["mode1"][:1]])
    row = _evaluate(config, contracts, *_fixture(reference_events, index))
    assert "both_modes_supported" in row["failing_clauses"]
    assert min(row["measures"]["mode_counts"]) < 3


def test_excessive_ood_fails_the_ood_clause(config, contracts, reference_events):
    if len(reference_events["ood"]) < 20:
        pytest.skip("not enough OOD events in the reference run")
    index = reference_events["ood"][:25]
    row = _evaluate(config, contracts, *_fixture(reference_events, index))
    assert row["measures"]["ood_fraction"] > config["repertoire_gate"]["ood_q95"]
    assert "ood_at_most_reference_q95" in row["failing_clauses"]


def test_a_raised_alignment_floor_fails_the_alignment_clause(
        config, contracts, reference_events):
    index = np.concatenate([reference_events["mode0"][:15],
                            reference_events["mode1"][:15]])
    strict = dict(config["repertoire_gate"], balanced_alignment_q05=1.01)
    row = _evaluate(config, contracts, *_fixture(reference_events, index),
                    gate=strict)
    assert "kmeans_alignment_at_least_reference_q05" in row["failing_clauses"]


def test_censoring_one_shaft_worsens_the_metrics_instead_of_hiding_events(
        config, contracts, reference_events):
    index = np.concatenate([reference_events["mode0"][:15],
                            reference_events["mode1"][:15]])
    onsets, returned, before, t_on, t_off = _fixture(reference_events, index)
    intact = _evaluate(config, contracts, onsets, returned, before, t_on, t_off)
    censored_onsets = onsets.copy()
    censored_onsets[:, contracts["groups"]["SCL"]] = np.nan
    censored = _evaluate(config, contracts, censored_onsets, returned, before,
                         t_on, t_off)
    assert censored["measures"]["n_returned"] == intact["measures"]["n_returned"]
    assert sum(censored["measures"]["mode_counts"]) < sum(
        intact["measures"]["mode_counts"])
    assert censored["retained"] is False
    assert censored["distributions"]["shaft_participation"]["n_joint"] == 0
    assert (censored["distributions"]["recruitment_size"]["median"]
            < intact["distributions"]["recruitment_size"]["median"])


def test_no_ab_label_is_attached_to_a_post_transition_event(
        config, contracts, reference_events):
    index = np.concatenate([reference_events["mode0"][:15],
                            reference_events["mode1"][:15]])
    onsets, returned, before, t_on, t_off = _fixture(reference_events, index)
    before = before.copy()
    before[-5:] = False  # these ended after the transition
    row = _evaluate(config, contracts, onsets, returned, before, t_on, t_off)
    assert row["no_label_assigned_to_runaway"] is True
    for event in row["events"][-5:]:
        assert event["scored"] is False
        assert event["mode"] is None
        assert event["classifier_confidence"] is None
    assert row["measures"]["n_returned"] == 25


def test_displayed_event_is_the_latest_one_not_the_first_in_the_file(
        config, contracts, reference_events):
    index = np.concatenate([reference_events["mode0"][:15],
                            reference_events["mode1"][:15]])
    onsets, returned, before, t_on, t_off = _fixture(reference_events, index)
    row = _evaluate(config, contracts, onsets, returned, before, t_on, t_off)
    order = np.argsort(t_off)[::-1]
    shuffled = _evaluate(config, contracts, onsets[order], returned[order],
                         before[order], t_on[order], t_off[order])
    assert row["displayed_event"]["t_off_ms"] == shuffled["displayed_event"]["t_off_ms"]
    latest = max(event["t_off_ms"] for event in row["events"]
                 if event["scored"] and event["clean"])
    assert row["displayed_event"]["t_off_ms"] == latest
    assert [event["event_index"] for event in row["events"]] == list(range(30))


def test_strict_t_ictal_window_is_reported_beside_the_historical_rule(
        config, contracts, reference_events):
    index = np.concatenate([reference_events["mode0"][:15],
                            reference_events["mode1"][:15]])
    onsets, returned, before, t_on, t_off = _fixture(reference_events, index)
    cut = float(np.median(t_off))
    row = _evaluate(config, contracts, onsets, returned, before, t_on, t_off,
                    t_ictal_ms=cut)
    strict = row["strict_t_ictal_sensitivity"]
    assert strict["n_returned"] + strict["n_events_between_t_ictal_and_t_op"] == 30
    assert strict["n_returned"] < 30


def test_rank_profile_similarity_is_leave_one_out(reference_events):
    onsets = reference_events["onsets"]
    labels = np.zeros(len(onsets), int)
    labels[reference_events["mode1"]] = 1
    clean = np.zeros(len(onsets), bool)
    clean[reference_events["mode0"][:10]] = True
    clean[reference_events["mode1"][:10]] = True
    row = rank_profile_similarity(onsets, labels, clean)
    for name in ("TA_like", "TB_like"):
        assert row[name]["n_events"] == 10
        assert np.isfinite(row[name]["median_leave_one_out_spearman"])
    single = rank_profile_similarity(onsets, labels, np.zeros(len(onsets), bool))
    assert single["TA_like"]["status"] == NOT_EVALUABLE


@pytest.mark.skipif(not (ZM / "repertoire_gate.json").exists(),
                    reason="historical repertoire gate absent")
def test_reproduces_the_historical_canary_repertoire_numbers(config, contracts):
    """Parity: the frozen contract is reused, not re-derived."""
    historical = json.loads((ZM / "repertoire_gate.json").read_text())
    checked = 0
    for key, expected in historical["networks"].items():
        arm, seed = key.rsplit("_s", 1)
        candidate = {"Node": "node_baseline", "Node+EE": "joint_04_ee_only",
                     "Node+EtoI": "joint_04_etoi_only",
                     "Joint": "joint_04_control"}[arm]
        path = ZM / "workers" / f"{candidate}_seed_{seed}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as handle:
            onsets = np.asarray(handle["onsets"], float)
            returned = np.asarray(handle["event_returned"], bool)
            before = np.asarray(handle["event_before_onset"], bool)
            t_on = np.asarray(handle["event_t_on_ms"], float)
            t_off = np.asarray(handle["event_t_off_ms"], float)
        row = _evaluate(config, contracts, onsets, returned, before, t_on, t_off)
        assert row["measures"]["n_returned"] == expected["measures"]["n_returned"]
        assert row["measures"]["mode_counts"] == expected["measures"]["mode_counts"]
        assert row["measures"]["n_clean"] == expected["measures"]["n_clean"]
        assert row["measures"]["ood_fraction"] == pytest.approx(
            expected["measures"]["ood_fraction"], rel=1e-9)
        assert row["retained"] == expected["retained"]
        assert row["failing_clauses"] == expected["failing_clauses"]
        checked += 1
    assert checked >= 12, f"only {checked} historical networks re-scored"
