"""TDD tests for PR-6 template endpoint anatomical anchoring.

Contract: docs/archive/topic1/pr6_template_endpoint_anchoring_plan_2026-04-25.md §9.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.template_anatomical_anchoring import (
    audit_subject_eligibility,
    classify_template_pair_nodes,
    compute_split_half_endpoint_jaccards,
    compute_subject_delta,
    compute_template_anchoring,
    compute_template_anchoring_by_coreness,
    compute_template_coreness,
    compute_template_pair_geometry,
    extract_endpoint_middle,
    extract_endpoint_middle_by_coreness,
    forward_reverse_swap_check,
    soz_breakdown_by_node_class,
)


# ---------------------------------------------------------------------------
# T1. test_extract_endpoint_middle_basic
# ---------------------------------------------------------------------------
def test_extract_endpoint_middle_basic():
    channel_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    template_rank = [7, 6, 5, 4, 3, 2, 1, 0]  # H earliest (rank=0), A latest (rank=7)

    out = extract_endpoint_middle(channel_names, template_rank, n=3)

    assert out["source"] == ["H", "G", "F"]
    assert out["sink"] == ["A", "B", "C"]
    assert set(out["endpoint"]) == {"A", "B", "C", "F", "G", "H"}
    assert set(out["middle"]) == {"D", "E"}
    assert out["exit_reason"] is None


# ---------------------------------------------------------------------------
# T2. test_extract_endpoint_middle_min_n_ch
# ---------------------------------------------------------------------------
def test_extract_endpoint_middle_min_n_ch():
    # n_ch == 6: endpoint covers all, middle empty (allowed but H1 ineligible —
    # compute_subject_delta must skip frac_middle=NaN; audit must mark
    # h1_primary_eligible=False, see test_cohort_exit_audit)
    chs6 = ["A", "B", "C", "D", "E", "F"]
    rank6 = [0, 1, 2, 3, 4, 5]
    out6 = extract_endpoint_middle(chs6, rank6, n=3)
    assert set(out6["endpoint"]) == set(chs6)
    assert out6["middle"] == []
    assert out6["exit_reason"] is None

    # n_ch == 5: helper must signal exit_reason='n_ch<6'
    chs5 = ["A", "B", "C", "D", "E"]
    rank5 = [0, 1, 2, 3, 4]
    out5 = extract_endpoint_middle(chs5, rank5, n=3)
    assert out5["exit_reason"] == "n_ch<6"


def test_extract_endpoint_middle_respects_valid_mask():
    # 8 channels but 2 are non-participating in this cluster — they must NOT
    # be picked as source or sink, and middle must be drawn only from the
    # remaining 6 valid channels.
    channel_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    # ranks: A=0, B=1, ..., H=7; mask out A and H (the would-be source / sink)
    template_rank = [0, 1, 2, 3, 4, 5, 6, 7]
    valid_mask = [False, True, True, True, True, True, True, False]

    out = extract_endpoint_middle(channel_names, template_rank, n=3, valid_mask=valid_mask)
    # source must come from valid channels: B, C, D (smallest valid ranks)
    assert out["source"] == ["B", "C", "D"]
    # sink: G, F, E (largest valid ranks, ordered largest first)
    assert out["sink"] == ["G", "F", "E"]
    # middle is empty since 6 valid channels are exactly endpoint
    assert out["middle"] == []
    assert out["n_valid"] == 6


def test_extract_endpoint_middle_sentinel_negative_one():
    # Split-half encodes invalid channels as rank=-1; default behaviour (no
    # explicit valid_mask) should still skip them.
    channel_names = ["A", "B", "C", "D", "E", "F", "G"]
    template_rank = [-1, 0, 1, 2, 3, 4, 5]  # A is non-participating
    out = extract_endpoint_middle(channel_names, template_rank, n=3)
    # A excluded, source/sink drawn from B..G only
    assert "A" not in out["source"]
    assert "A" not in out["sink"]
    assert out["source"] == ["B", "C", "D"]
    assert out["sink"] == ["G", "F", "E"]


# ---------------------------------------------------------------------------
# T3. test_align_yuquan_bipolar (uses match_bipolar_soz reuse)
# ---------------------------------------------------------------------------
def test_align_yuquan_bipolar():
    channel_names = ["D13-D14", "D14-D15", "E1-E2", "F1-F2", "F2-F3", "F3-F4"]
    # template_rank: D13-D14 earliest, F3-F4 latest
    template_rank = [0, 1, 2, 3, 4, 5]
    soz = ["D13", "D14", "E1", "E2"]

    rec = compute_template_anchoring(channel_names, template_rank, soz, focus_rel_dict=None)

    # source = top-3 smallest rank: D13-D14, D14-D15, E1-E2 (all SOZ via match_bipolar_soz)
    assert rec["source"] == ["D13-D14", "D14-D15", "E1-E2"]
    # sink = top-3 largest rank: F3-F4, F2-F3, F1-F2 (none in SOZ)
    assert rec["sink"] == ["F3-F4", "F2-F3", "F1-F2"]

    assert rec["frac_SOZ_source"] == pytest.approx(3 / 3)
    assert rec["frac_SOZ_sink"] == pytest.approx(0 / 3)
    # endpoint = source ∪ sink = all 6 channels (3 SOZ + 3 non-SOZ)
    assert rec["frac_SOZ_endpoint"] == pytest.approx(3 / 6)
    # middle = empty when n_ch == 6 -> frac_SOZ_middle is NaN
    assert math.isnan(rec["frac_SOZ_middle"])

    # n_ch >= 7 case to exercise frac_SOZ_middle
    channel_names2 = channel_names + ["G1-G2", "G2-G3"]
    template_rank2 = [0, 1, 2, 3, 4, 5, 6, 7]
    rec2 = compute_template_anchoring(channel_names2, template_rank2, soz)
    # source still top 3 (D13-D14, D14-D15, E1-E2 -> 3 SOZ)
    # sink top 3 = G2-G3, G1-G2, F3-F4 -> 0 SOZ
    # middle = F1-F2, F2-F3 -> 0 SOZ
    assert rec2["frac_SOZ_endpoint"] == pytest.approx(3 / 6)
    assert rec2["frac_SOZ_middle"] == pytest.approx(0 / 2)


# ---------------------------------------------------------------------------
# T4. test_align_epilepsiae_focus_rel_3level
# ---------------------------------------------------------------------------
def test_align_epilepsiae_focus_rel_3level():
    focus_rel = {
        "i": ["HRA1"],
        "l": ["BFLA1", "BFLA2"],
        "e": [],
    }
    # 7 channels so middle is non-empty
    channel_names = ["HRA1", "HRB1", "BFLA1", "BFLA2", "BFLA3", "BFLA4", "BFLB1"]
    template_rank = [0, 1, 2, 3, 4, 5, 6]
    soz_channels = focus_rel["i"]  # i-only as SOZ proxy

    rec = compute_template_anchoring(
        channel_names, template_rank, soz_channels, focus_rel_dict=focus_rel
    )

    # source = HRA1, HRB1, BFLA1
    assert rec["source"] == ["HRA1", "HRB1", "BFLA1"]
    # sink top 3 largest = BFLB1, BFLA4, BFLA3
    assert rec["sink"] == ["BFLB1", "BFLA4", "BFLA3"]
    # endpoint = source + sink = 6 channels
    # i in endpoint: HRA1 -> 1/6
    assert rec["frac_i_endpoint"] == pytest.approx(1 / 6)
    # l in endpoint: BFLA1 only (BFLA3/BFLA4/BFLB1/HRB1 are NOT in focus_rel['l']) -> 1/6
    assert rec["frac_l_endpoint"] == pytest.approx(1 / 6)
    # e in endpoint: 0/6
    assert rec["frac_e_endpoint"] == pytest.approx(0 / 6)
    # middle = BFLA2 -> l only (BFLA2 is in focus_rel['l'])
    assert rec["frac_i_middle"] == pytest.approx(0 / 1)
    assert rec["frac_l_middle"] == pytest.approx(1 / 1)
    assert rec["frac_e_middle"] == pytest.approx(0 / 1)


# ---------------------------------------------------------------------------
# T5. test_subject_delta_averages_over_k
# ---------------------------------------------------------------------------
def test_subject_delta_averages_over_k():
    per_template_records = [
        {"frac_SOZ_endpoint": 0.5, "frac_SOZ_middle": 0.1, "frac_SOZ_source": 0.6, "frac_SOZ_sink": 0.4},
        {"frac_SOZ_endpoint": 0.3, "frac_SOZ_middle": 0.2, "frac_SOZ_source": 0.2, "frac_SOZ_sink": 0.4},
    ]
    out = compute_subject_delta(per_template_records)

    # mean over k of (endpoint - middle): (0.5-0.1 + 0.3-0.2)/2 = (0.4 + 0.1)/2 = 0.25
    assert out["delta_endpoint_vs_middle"] == pytest.approx(0.25)
    # mean over k of (source - sink): (0.6-0.4 + 0.2-0.4)/2 = (0.2 + (-0.2))/2 = 0.0
    assert out["delta_source_vs_sink"] == pytest.approx(0.0)
    assert out["n_templates_used"] == 2


def test_subject_delta_handles_nan_middle():
    # When n_ch == 6, frac_middle is NaN; subject delta must skip the NaN
    per_template_records = [
        {"frac_SOZ_endpoint": 0.5, "frac_SOZ_middle": float("nan"), "frac_SOZ_source": 0.6, "frac_SOZ_sink": 0.4},
        {"frac_SOZ_endpoint": 0.3, "frac_SOZ_middle": 0.2, "frac_SOZ_source": 0.2, "frac_SOZ_sink": 0.4},
    ]
    out = compute_subject_delta(per_template_records)
    # only second record has valid middle: delta = 0.3 - 0.2 = 0.1
    assert out["delta_endpoint_vs_middle"] == pytest.approx(0.1)
    assert out["n_templates_endpoint_middle_valid"] == 1


# ---------------------------------------------------------------------------
# T6. test_split_half_centroid_rank_storage
# ---------------------------------------------------------------------------
def test_split_half_centroid_rank_storage():
    """compute_time_split_reproducibility must store cluster_rank_a / cluster_rank_b
    per split for every cluster."""
    from src.interictal_propagation import compute_time_split_reproducibility

    rng = np.random.default_rng(42)
    n_ch = 8
    n_events = 40
    chosen_k = 2

    # Build two distinct rank profiles to make k=2 separable
    ranks = np.zeros((n_ch, n_events), dtype=float)
    bools = np.ones((n_ch, n_events), dtype=bool)
    half = n_events // 2

    # Cluster A: channels 0-3 early, 4-7 late
    profile_a = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    # Cluster B: reversed
    profile_b = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=float)

    for ev in range(half):
        ranks[:, ev] = profile_a + rng.normal(0, 0.3, size=n_ch)
    for ev in range(half, n_events):
        ranks[:, ev] = profile_b + rng.normal(0, 0.3, size=n_ch)

    event_abs_times = np.linspace(0, n_events, n_events).astype(float)
    block_ids = np.zeros(n_events, dtype=int)
    block_ids[half:] = 1

    valid_event_indices = np.arange(n_events)
    # Adaptive labels (the 'truth' labels)
    adaptive_labels = np.concatenate([np.zeros(half, dtype=int), np.ones(half, dtype=int)])

    out = compute_time_split_reproducibility(
        ranks=ranks,
        bools=bools,
        event_abs_times=event_abs_times,
        block_ids=block_ids,
        chosen_k=chosen_k,
        adaptive_labels=adaptive_labels,
        valid_event_indices=valid_event_indices,
    )

    splits = out["splits"]
    assert "first_half_second_half" in splits
    fh = splits["first_half_second_half"]
    # All four PR-6 robustness fields must exist
    for key in (
        "cluster_rank_a",
        "cluster_rank_b",
        "cluster_valid_mask_a",
        "cluster_valid_mask_b",
        "cluster_rank_b_matched_to_a",
        "cluster_valid_mask_b_matched_to_a",
    ):
        assert key in fh, f"missing key {key} in split result"

    # Shape: (k, n_ch)
    arr_a = np.asarray(fh["cluster_rank_a"], dtype=int)
    arr_b = np.asarray(fh["cluster_rank_b"], dtype=int)
    assert arr_a.shape == (chosen_k, n_ch)
    assert arr_b.shape == (chosen_k, n_ch)
    # Valid channels (rank >= 0) must form a permutation of [0..n_valid-1]
    mask_a = np.asarray(fh["cluster_valid_mask_a"], dtype=bool)
    mask_b = np.asarray(fh["cluster_valid_mask_b"], dtype=bool)
    assert mask_a.shape == (chosen_k, n_ch)
    assert mask_b.shape == (chosen_k, n_ch)
    for row, mrow in zip(arr_a, mask_a):
        valid_ranks = sorted(row[mrow].tolist())
        assert valid_ranks == list(range(int(mrow.sum())))
        # Non-valid channels must be -1
        assert all(int(r) == -1 for r, m in zip(row, mrow) if not m)


def test_split_half_cluster_rank_b_matched_to_a():
    """cluster_rank_b_matched_to_a[i] must equal cluster_rank_b[mapping[i]],
    not cluster_rank_b[i]. KMeans labels are arbitrary, so direct same-index
    Jaccard would mix unrelated clusters."""
    from src.interictal_propagation import compute_time_split_reproducibility

    rng = np.random.default_rng(7)
    n_ch = 8
    n_events = 60
    chosen_k = 2
    ranks = np.zeros((n_ch, n_events), dtype=float)
    bools = np.ones((n_ch, n_events), dtype=bool)
    half = n_events // 2
    profile_a = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    profile_b = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
    for ev in range(half):
        ranks[:, ev] = profile_a + rng.normal(0, 0.2, size=n_ch)
    for ev in range(half, n_events):
        ranks[:, ev] = profile_b + rng.normal(0, 0.2, size=n_ch)

    event_abs_times = np.linspace(0, n_events, n_events).astype(float)
    block_ids = np.zeros(n_events, dtype=int)
    block_ids[half:] = 1
    valid_event_indices = np.arange(n_events)
    adaptive_labels = np.concatenate(
        [np.zeros(half, dtype=int), np.ones(half, dtype=int)]
    )

    out = compute_time_split_reproducibility(
        ranks=ranks,
        bools=bools,
        event_abs_times=event_abs_times,
        block_ids=block_ids,
        chosen_k=chosen_k,
        adaptive_labels=adaptive_labels,
        valid_event_indices=valid_event_indices,
    )

    fh = out["splits"]["first_half_second_half"]
    mapping = {int(k): int(v) for k, v in fh["mapping_a_to_b"].items()}
    raw_b = fh["cluster_rank_b"]
    matched_b = fh["cluster_rank_b_matched_to_a"]
    for a_id in range(chosen_k):
        if a_id in mapping:
            b_id = mapping[a_id]
            assert matched_b[a_id] == raw_b[b_id], (
                f"matched_b[{a_id}] != raw_b[mapping[{a_id}]={b_id}]"
            )


# ---------------------------------------------------------------------------
# T7. test_forward_reverse_swap_score
# ---------------------------------------------------------------------------
def test_forward_reverse_swap_score():
    channel_names = ["A", "B", "C", "D", "E", "F", "G", "H"]
    # T0 source = A,B,C  -> T1 sink = B,C,A (full overlap, same set)
    # T0 sink = F,G,H    -> T1 source = G,H,F (full overlap)
    out = forward_reverse_swap_check(
        t0_source=["A", "B", "C"],
        t0_sink=["F", "G", "H"],
        t1_source=["G", "H", "F"],
        t1_sink=["B", "C", "A"],
        channel_names=channel_names,
        n_perm=500,
        seed=0,
    )

    # Both Jaccards = 1.0 -> mean = 1.0
    assert out["swap_score"] == pytest.approx(1.0)
    # null_p must be small (perfect swap is rare under random sampling of 8-channel
    # space with 3-element sets)
    assert out["null_p"] < 0.05
    assert "null_95th" in out
    assert out["null_95th"] < 1.0  # null distribution shouldn't be all 1.0


def test_forward_reverse_swap_score_no_swap():
    """Disjoint endpoints -> swap_score 0 -> null_p high."""
    channel_names = list("ABCDEFGHIJKL")
    out = forward_reverse_swap_check(
        t0_source=["A", "B", "C"],
        t0_sink=["D", "E", "F"],
        t1_source=["G", "H", "I"],  # disjoint from t0_sink
        t1_sink=["J", "K", "L"],  # disjoint from t0_source
        channel_names=channel_names,
        n_perm=500,
        seed=0,
    )
    assert out["swap_score"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# T8. test_cohort_exit_audit
# ---------------------------------------------------------------------------
def test_cohort_exit_audit():
    """audit_subject_eligibility returns one row per candidate with two
    orthogonal flags: endpoint_defined (n_ch >= 6) and h1_primary_eligible
    (n_ch >= 7).  pass == h1_primary_eligible."""
    candidates = [
        # A: stable_k=2, soz=['A','B'], n_ch=10 -> PASS (h1_primary_eligible)
        {
            "subject_id": "A",
            "dataset": "yuquan",
            "stable_k": 2,
            "soz_channels": ["A", "B"],
            "channel_names": ["A", "B"] + [f"C{i}" for i in range(8)],
            "template_ranks": [list(range(10)), list(range(10))[::-1]],
        },
        # B: stable_k=4 -> EXIT 'k!=2'
        {
            "subject_id": "B",
            "dataset": "yuquan",
            "stable_k": 4,
            "soz_channels": ["C0"],
            "channel_names": [f"C{i}" for i in range(10)],
            "template_ranks": [list(range(10))] * 4,
        },
        # C: stable_k=2, soz=[] -> EXIT 'empty_soz'
        {
            "subject_id": "C",
            "dataset": "yuquan",
            "stable_k": 2,
            "soz_channels": [],
            "channel_names": [f"C{i}" for i in range(10)],
            "template_ranks": [list(range(10))] * 2,
        },
        # D: stable_k=2, soz=['Z'] not in channel_names -> EXIT 'no_matched_soz'
        {
            "subject_id": "D",
            "dataset": "yuquan",
            "stable_k": 2,
            "soz_channels": ["Z"],
            "channel_names": [f"C{i}" for i in range(10)],
            "template_ranks": [list(range(10))] * 2,
        },
        # E: stable_k=2, n_ch=5 -> EXIT 'n_ch<6' (endpoint cannot be extracted)
        {
            "subject_id": "E",
            "dataset": "yuquan",
            "stable_k": 2,
            "soz_channels": ["C0"],
            "channel_names": [f"C{i}" for i in range(5)],
            "template_ranks": [list(range(5))] * 2,
        },
        # F: stable_k=2, n_ch=6 -> endpoint_defined=True but H1 ineligible
        # (middle is empty) -> EXIT 'middle_empty', pass=False
        {
            "subject_id": "F",
            "dataset": "yuquan",
            "stable_k": 2,
            "soz_channels": ["C0", "C1"],
            "channel_names": [f"C{i}" for i in range(6)],
            "template_ranks": [list(range(6)), list(range(6))[::-1]],
        },
    ]

    rows = audit_subject_eligibility(candidates)
    assert len(rows) == 6

    by_id = {r["subject_id"]: r for r in rows}

    # A: full PASS
    assert by_id["A"]["pass"] is True
    assert by_id["A"]["endpoint_defined"] is True
    assert by_id["A"]["h1_primary_eligible"] is True
    assert by_id["A"]["exit_reason"] is None

    # B: k!=2
    assert by_id["B"]["pass"] is False
    assert by_id["B"]["endpoint_defined"] is False
    assert by_id["B"]["exit_reason"] == "k!=2"

    # C: empty SOZ
    assert by_id["C"]["pass"] is False
    assert by_id["C"]["endpoint_defined"] is False
    assert by_id["C"]["exit_reason"] == "empty_soz"

    # D: SOZ list non-empty but none match channels
    assert by_id["D"]["pass"] is False
    assert by_id["D"]["endpoint_defined"] is False
    assert by_id["D"]["exit_reason"] == "no_matched_soz"

    # E: n_ch < 6 -> endpoint cannot even be extracted
    assert by_id["E"]["pass"] is False
    assert by_id["E"]["endpoint_defined"] is False
    assert by_id["E"]["exit_reason"] == "n_ch<6"

    # F: n_ch == 6 -> endpoint_defined but H1 ineligible
    assert by_id["F"]["pass"] is False
    assert by_id["F"]["endpoint_defined"] is True
    assert by_id["F"]["h1_primary_eligible"] is False
    assert by_id["F"]["exit_reason"] == "middle_empty"


# ---------------------------------------------------------------------------
# Step 5a — coreness composite sensitivity
# ---------------------------------------------------------------------------
def test_compute_template_coreness_picks_highly_polarized_channel():
    """Channel with stable extreme rank + high participation should outrank
    a channel with mid-range rank, the same participation, and the same IQR."""
    n_ch = 8
    n_events = 50
    rng = np.random.default_rng(0)

    # Cluster 0: channel 0 always rank 0 (perfect early), channel 7 always
    # rank 7 (perfect late), channel 3 always rank ~3.5 (center).  All channels
    # participate every event.
    ranks = np.zeros((n_ch, n_events), dtype=float)
    bools = np.ones((n_ch, n_events), dtype=bool)
    fixed = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
    for ev in range(n_events):
        ranks[:, ev] = fixed + rng.normal(0, 0.15, size=n_ch)
    labels = np.zeros(n_events, dtype=int)

    out = compute_template_coreness(ranks, bools, labels, n_clusters=1)
    rec = out[0]
    assert rec["valid_mask"] == [True] * n_ch
    coreness = np.asarray(rec["coreness"])
    # Channels 0 and 7 (extremes) should have highest coreness; channel 3 / 4
    # (center) lowest among polarity term
    assert coreness[0] > coreness[3]
    assert coreness[7] > coreness[4]


def test_compute_template_coreness_zero_for_non_participating():
    n_ch = 8
    n_events = 30
    ranks = np.zeros((n_ch, n_events), dtype=float)
    for ev in range(n_events):
        ranks[:, ev] = np.arange(n_ch, dtype=float)
    bools = np.ones((n_ch, n_events), dtype=bool)
    bools[2, :] = False  # channel 2 never participates in cluster 0
    labels = np.zeros(n_events, dtype=int)

    rec = compute_template_coreness(ranks, bools, labels, n_clusters=1)[0]
    assert rec["valid_mask"][2] is False
    assert rec["coreness"][2] == 0.0
    # Other channels should still register
    assert rec["valid_mask"][0] is True
    assert rec["coreness"][0] > 0.0


def test_extract_endpoint_middle_by_coreness_matches_main_size():
    """Coreness sensitivity must keep |endpoint| == 2*n so H1 direction is
    comparable to the main top/bottom-3 definition."""
    channel_names = list("ABCDEFGHIJ")  # n_ch=10
    n_ch = 10
    # Coreness: channel 0 highest, then 9, 1, 8, 2, 7, 3, 6, 4, 5 (extremes)
    coreness = [0.95, 0.80, 0.65, 0.40, 0.10, 0.05, 0.45, 0.70, 0.85, 0.90]
    median_rank = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    valid = [True] * 10

    rec = extract_endpoint_middle_by_coreness(
        channel_names,
        {"coreness": coreness, "median_rank": median_rank, "valid_mask": valid},
        n=3,
    )
    # Top-6 by coreness, descending: A (0.95), J (0.90), I (0.85), B (0.80),
    # H (0.70), C (0.65)  → indices [0,9,8,1,7,2]
    # Within these, sort by median_rank: A(0), B(1), C(2), H(7), I(8), J(9)
    # Source = lowest 3 medians: A, B, C
    # Sink = highest 3 medians (largest first): J, I, H
    assert rec["source"] == ["A", "B", "C"]
    assert rec["sink"] == ["J", "I", "H"]
    assert set(rec["endpoint"]) == {"A", "B", "C", "H", "I", "J"}
    # Middle = the OTHER valid channels: D, E, F, G
    assert set(rec["middle"]) == {"D", "E", "F", "G"}
    assert rec["exit_reason"] is None
    assert rec["n_endpoint"] == 6


def test_extract_endpoint_middle_by_coreness_n_valid_too_small():
    channel_names = list("ABCDE")
    coreness = [0.5, 0.4, 0.3, 0.2, 0.1]
    median_rank = [0, 1, 2, 3, 4]
    valid = [True, True, True, False, False]  # only 3 valid; need 6
    rec = extract_endpoint_middle_by_coreness(
        channel_names,
        {"coreness": coreness, "median_rank": median_rank, "valid_mask": valid},
        n=3,
    )
    assert rec["exit_reason"] == "n_valid<2n"


# ---------------------------------------------------------------------------
# Step 5b — split-half endpoint robustness Jaccard
# ---------------------------------------------------------------------------
def test_split_half_endpoint_jaccard_perfect_stability():
    """Identical rank+mask in both splits → all Jaccards = 1.0."""
    channel_names = list("ABCDEFGH")
    rank = [[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]]
    mask = [[True] * 8, [True] * 8]

    out = compute_split_half_endpoint_jaccards(
        channel_names=channel_names,
        cluster_rank_a=rank,
        cluster_valid_mask_a=mask,
        cluster_rank_b_matched_to_a=rank,  # identical
        cluster_valid_mask_b_matched_to_a=mask,
        n=3,
    )
    assert len(out) == 2
    for rec in out:
        assert rec["exit_reason"] is None
        assert rec["jaccard_source"] == pytest.approx(1.0)
        assert rec["jaccard_sink"] == pytest.approx(1.0)
        assert rec["jaccard_endpoint"] == pytest.approx(1.0)


def test_split_half_endpoint_jaccard_full_swap():
    """B half has reversed ranks within the same valid set → source_A vs
    source_B disjoint, jaccard=0."""
    channel_names = list("ABCDEFGH")
    rank_a = [[0, 1, 2, 3, 4, 5, 6, 7]]
    rank_b = [[7, 6, 5, 4, 3, 2, 1, 0]]
    mask_full = [[True] * 8]

    out = compute_split_half_endpoint_jaccards(
        channel_names=channel_names,
        cluster_rank_a=rank_a,
        cluster_valid_mask_a=mask_full,
        cluster_rank_b_matched_to_a=rank_b,
        cluster_valid_mask_b_matched_to_a=mask_full,
        n=3,
    )
    rec = out[0]
    assert rec["exit_reason"] is None
    # Source_A = A,B,C; Source_B = H,G,F → disjoint
    assert rec["jaccard_source"] == pytest.approx(0.0)
    # Sink_A = H,G,F; Sink_B = A,B,C → disjoint
    assert rec["jaccard_sink"] == pytest.approx(0.0)
    # Endpoint sets are SAME 6 channels (A,B,C,F,G,H) → jaccard=1.0
    assert rec["jaccard_endpoint"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Step 4b — node-level template-pair anatomy
# ---------------------------------------------------------------------------
def test_classify_template_pair_nodes_full_swap():
    """T0 source = T1 sink and vice versa: every endpoint channel is a swap_node."""
    channel_names = list("ABCDEFGH")
    out = classify_template_pair_nodes(
        channel_names,
        t0_source=["A", "B", "C"],
        t0_sink=["F", "G", "H"],
        t1_source=["F", "G", "H"],
        t1_sink=["A", "B", "C"],
    )
    assert out["counts"]["swap_node"] == 6
    assert out["counts"]["same_side_node"] == 0
    assert out["counts"]["template_specific_endpoint"] == 0
    assert out["counts"]["shared_endpoint_unassigned"] == 0
    assert out["counts"]["non_endpoint"] == 2  # D, E
    assert set(out["channels_by_class"]["non_endpoint"]) == {"D", "E"}


def test_classify_template_pair_nodes_identical_templates():
    """Identical T0 and T1: all endpoint channels are same_side_node."""
    channel_names = list("ABCDEFGH")
    out = classify_template_pair_nodes(
        channel_names,
        t0_source=["A", "B", "C"],
        t0_sink=["F", "G", "H"],
        t1_source=["A", "B", "C"],
        t1_sink=["F", "G", "H"],
    )
    assert out["counts"]["same_side_node"] == 6
    assert out["counts"]["swap_node"] == 0
    assert out["counts"]["template_specific_endpoint"] == 0


def test_classify_template_pair_nodes_template_specific():
    """Disjoint endpoints: all template_specific_endpoint."""
    channel_names = list("ABCDEFGHIJKLMN")
    out = classify_template_pair_nodes(
        channel_names,
        t0_source=["A", "B", "C"],
        t0_sink=["D", "E", "F"],
        t1_source=["G", "H", "I"],
        t1_sink=["J", "K", "L"],
    )
    assert out["counts"]["template_specific_endpoint"] == 12
    assert out["counts"]["same_side_node"] == 0
    assert out["counts"]["swap_node"] == 0
    assert out["counts"]["non_endpoint"] == 2  # M, N


def test_classify_template_pair_nodes_partial_swap_partial_specific():
    """Realistic mix: 1 swap, 1 same-side, 4 template-specific."""
    channel_names = list("ABCDEFGH")
    # T0 endpoint: A,B,C (source) + F,G,H (sink)
    # T1 endpoint: A,B,F (source) + C,G,H (sink) — wait this would re-use
    # Let me design more carefully:
    # T0: source=[A,B,C], sink=[F,G,H]
    # T1: source=[F,B,D], sink=[C,E,G]
    # A: T0 source only -> template_specific
    # B: T0 source AND T1 source -> same_side
    # C: T0 source AND T1 sink -> swap
    # D: T1 source only -> template_specific
    # E: T1 sink only -> template_specific
    # F: T0 sink AND T1 source -> swap
    # G: T0 sink AND T1 sink -> same_side
    # H: T0 sink only -> template_specific
    out = classify_template_pair_nodes(
        channel_names,
        t0_source=["A", "B", "C"],
        t0_sink=["F", "G", "H"],
        t1_source=["F", "B", "D"],
        t1_sink=["C", "E", "G"],
    )
    assert out["counts"]["swap_node"] == 2  # C, F
    assert out["counts"]["same_side_node"] == 2  # B, G
    assert out["counts"]["template_specific_endpoint"] == 4  # A, D, E, H
    assert out["counts"]["non_endpoint"] == 0
    by_class = out["channels_by_class"]
    assert set(by_class["swap_node"]) == {"C", "F"}
    assert set(by_class["same_side_node"]) == {"B", "G"}


def test_soz_breakdown_by_node_class_yuquan_bipolar():
    """SOZ enrichment per node class, bipolar contact matching."""
    channel_names = ["D13-D14", "D14-D15", "E1-E2", "F1-F2", "F2-F3", "F3-F4", "G1-G2", "G2-G3"]
    # T0: source=D13-D14,D14-D15,E1-E2; sink=F3-F4,F2-F3,F1-F2
    # T1: source=F3-F4,F2-F3,F1-F2; sink=D13-D14,D14-D15,E1-E2 (full swap)
    # → all 6 are swap_node; G1-G2 / G2-G3 are non_endpoint
    soz = ["D13", "D14", "E1", "E2"]
    nc = classify_template_pair_nodes(
        channel_names,
        t0_source=channel_names[:3],
        t0_sink=channel_names[3:6][::-1],  # F3-F4, F2-F3, F1-F2
        t1_source=channel_names[3:6][::-1],
        t1_sink=channel_names[:3],
    )
    assert nc["counts"]["swap_node"] == 6
    breakdown = soz_breakdown_by_node_class(nc, soz)
    # swap_nodes contain D13-D14, D14-D15, E1-E2 (3 SOZ via match) + F1-F2,
    # F2-F3, F3-F4 (0 SOZ) = 3/6 SOZ in swap nodes
    assert breakdown["swap_node"]["n_soz"] == 3
    assert breakdown["swap_node"]["frac_soz"] == pytest.approx(0.5)
    assert breakdown["non_endpoint"]["n_soz"] == 0


def test_template_pair_geometry_full_swap():
    """T0 = [0..7], T1 = [7..0]: bidirectional swap.  Same-side J should be 0,
    swap J should be 1, endpoint shared (same 6 channels), Spearman = -1."""
    channel_names = list("ABCDEFGH")
    t0_rank = [0, 1, 2, 3, 4, 5, 6, 7]
    t1_rank = [7, 6, 5, 4, 3, 2, 1, 0]
    valid = [True] * 8

    out = compute_template_pair_geometry(
        channel_names, t0_rank, t1_rank, valid, valid, n=3
    )
    assert out["exit_reason"] is None
    # T0 source = A,B,C; T1 source = H,G,F → disjoint
    assert out["jaccard_source_same"] == pytest.approx(0.0)
    # T0 sink = H,G,F; T1 sink = A,B,C → disjoint
    assert out["jaccard_sink_same"] == pytest.approx(0.0)
    # T0 source vs T1 sink = A,B,C vs A,B,C → identical
    assert out["jaccard_source_to_sink"] == pytest.approx(1.0)
    assert out["jaccard_sink_to_source"] == pytest.approx(1.0)
    assert out["swap_score"] == pytest.approx(1.0)
    assert out["same_side_score"] == pytest.approx(0.0)
    # Endpoint sets are SAME 6 channels {A,B,C,F,G,H}
    assert out["jaccard_endpoint"] == pytest.approx(1.0)
    # Spearman on full inverse permutation
    assert out["spearman_rank_pair"] == pytest.approx(-1.0)


def test_template_pair_geometry_identical_templates():
    """Identical templates: same-side J = 1, swap J = 0 (source ≠ sink),
    endpoint J = 1, Spearman = +1."""
    channel_names = list("ABCDEFGH")
    rank = [0, 1, 2, 3, 4, 5, 6, 7]
    valid = [True] * 8

    out = compute_template_pair_geometry(
        channel_names, rank, rank, valid, valid, n=3
    )
    assert out["exit_reason"] is None
    assert out["jaccard_source_same"] == pytest.approx(1.0)
    assert out["jaccard_sink_same"] == pytest.approx(1.0)
    assert out["jaccard_endpoint"] == pytest.approx(1.0)
    assert out["jaccard_source_to_sink"] == pytest.approx(0.0)
    assert out["jaccard_sink_to_source"] == pytest.approx(0.0)
    assert out["spearman_rank_pair"] == pytest.approx(1.0)


def test_template_pair_geometry_independent_low_overlap():
    """T0 and T1 use mostly disjoint endpoint channels → all Jaccards low,
    endpoint overlap also low; Spearman near 0 within intersection."""
    channel_names = list("ABCDEFGHIJ")
    # T0 endpoint: A,B,C / H,I,J
    t0_rank = [0, 1, 2, 5, 5, 5, 5, 6, 7, 8]
    t0_valid = [True, True, True, False, False, False, False, True, True, True]
    # T1 endpoint: D,E,F / G (only 7 valid → can't pick 3 sink, fail)
    # Make T1 endpoint use D,E / G,H,I... actually need n_valid >= 6
    t1_rank = [10, 11, 12, 0, 1, 2, 8, 9, 10, 13]
    t1_valid = [False, False, False, True, True, True, True, True, True, False]

    out = compute_template_pair_geometry(
        channel_names, t0_rank, t1_rank, t0_valid, t1_valid, n=3
    )
    # T1 valid=[D,E,F,G,H,I] = 6, so endpoint = D,E,F / I,H,G; sink top-3 = I,H,G
    # T0 valid=[A,B,C,H,I,J] = 6, so endpoint = A,B,C / J,I,H
    # Same-side: source A,B,C vs D,E,F = 0; sink J,I,H vs I,H,G = 2/4 = 0.5
    assert out["exit_reason"] is None
    assert out["jaccard_source_same"] == pytest.approx(0.0)
    # Endpoint overlap: {A,B,C,H,I,J} vs {D,E,F,G,H,I} = {H,I} / {A..J} = 2/10
    assert out["jaccard_endpoint"] == pytest.approx(2.0 / 10.0)


def test_split_half_endpoint_jaccard_no_mapping_exit():
    """When mapping returns None for a cluster, record exit_reason='no_mapping'."""
    channel_names = list("ABCDEFGH")
    rank_a = [[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]]
    mask = [[True] * 8, [True] * 8]
    rank_b_matched = [None, [7, 6, 5, 4, 3, 2, 1, 0]]
    mask_b_matched = [None, [True] * 8]

    out = compute_split_half_endpoint_jaccards(
        channel_names, rank_a, mask, rank_b_matched, mask_b_matched, n=3
    )
    assert out[0]["exit_reason"] == "no_mapping"
    assert out[1]["exit_reason"] is None


def test_compute_template_anchoring_by_coreness_yuquan_bipolar():
    channel_names = ["D13-D14", "D14-D15", "E1-E2", "F1-F2", "F2-F3", "F3-F4", "G1-G2"]
    # 7 channels.  Make first 3 clearly the highest coreness with low median
    # rank (source), last 3 the second-highest coreness with high median rank
    # (sink), middle one (G1-G2) lowest coreness (middle).
    coreness = [0.9, 0.85, 0.8, 0.7, 0.65, 0.6, 0.1]
    median_rank = [0, 1, 2, 5, 6, 7, 4]
    valid = [True] * 7
    soz = ["D13", "D14", "E1", "E2"]

    rec = compute_template_anchoring_by_coreness(
        channel_names,
        {"coreness": coreness, "median_rank": median_rank, "valid_mask": valid},
        soz_channels=soz,
        n=3,
    )
    assert rec["source"] == ["D13-D14", "D14-D15", "E1-E2"]
    assert rec["sink"] == ["F3-F4", "F2-F3", "F1-F2"]
    # Source 3/3 SOZ, sink 0/3, middle 0/1 (G1-G2)
    assert rec["frac_SOZ_source"] == pytest.approx(1.0)
    assert rec["frac_SOZ_sink"] == pytest.approx(0.0)
    assert rec["frac_SOZ_endpoint"] == pytest.approx(0.5)
    assert rec["frac_SOZ_middle"] == pytest.approx(0.0)
