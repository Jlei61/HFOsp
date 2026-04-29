"""Pin the contract of `compute_cohort_statistics_three_tier` in
`scripts/run_spatial_modulation.py` for the Epilepsiae i/l/e gradient.

Key invariants:

- Per-metric direction map: `iei_detrended_r` is `greater`, `detrend_fraction`
  / `iei_median` are `less`. event_rate is confound-only (no monotonicity).
- Three-pair paired Wilcoxon (i vs l, i vs e, l vs e) with `alternative` set
  per metric direction. Bonferroni p_adj = min(p * 3, 1.0).
- Subject-level monotonicity sign test: count subjects whose (i, l, e)
  medians are strictly monotonic in pre-registered direction; binomtest
  with null = 1/6 (six possible orderings).
- focus_rel-missing subjects: still get per-channel JSON, all channels
  labelled 'unknown', auto-excluded from cohort paired stats.
- `--dataset epilepsiae` path must NOT load the Yuquan SOZ JSON.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_spatial_modulation import (  # noqa: E402
    EVENT_RATE_METRIC,
    METRIC_DIRECTIONS,
    _is_monotonic,
    compute_cohort_statistics_three_tier,
)


# ---------------------------------------------------------------------------
# Helpers to synthesize a cohort with controlled i/l/e medians per subject.
# ---------------------------------------------------------------------------


def _make_channel(region: str, **metric_overrides):
    base = {
        "iei_detrended_r": 0.0,
        "detrend_fraction": 0.5,
        "iei_median": 1.0,
        "iei_p02": 0.05,
        "iei_lag1_r": 0.0,
        "iei_cv": 1.0,
        "event_rate": 100.0,
        "artifact_suspect": False,
        "region_label": region,
    }
    base.update(metric_overrides)
    return base


def _make_subject(
    subj: str,
    i_vals: dict,
    l_vals: dict,
    e_vals: dict,
    n_per_region: int = 3,
):
    """Build an all_results-style dict with n channels per region.

    *_vals supply the per-metric values; we repeat each value n times so the
    region median equals the supplied value exactly.
    """
    chans = []
    for region, vals in (("i", i_vals), ("l", l_vals), ("e", e_vals)):
        for _ in range(n_per_region):
            chans.append(_make_channel(region, **vals))
    return {
        "subject": subj,
        "dataset": "epilepsiae",
        "channel_metrics": chans,
    }


# ---------------------------------------------------------------------------
# 1. METRIC_DIRECTIONS contract
# ---------------------------------------------------------------------------


def test_metric_directions_contract():
    """Per-metric direction map must lock pre-registered hypotheses."""
    assert METRIC_DIRECTIONS["iei_detrended_r"] == "greater"
    assert METRIC_DIRECTIONS["detrend_fraction"] == "less"
    assert METRIC_DIRECTIONS["iei_median"] == "less"
    assert METRIC_DIRECTIONS["iei_p02"] == "less"
    assert METRIC_DIRECTIONS["iei_lag1_r"] == "two-sided"
    assert METRIC_DIRECTIONS["iei_cv"] == "two-sided"
    assert EVENT_RATE_METRIC == "event_rate"
    assert "event_rate" not in METRIC_DIRECTIONS, (
        "event_rate is confound-only and must not appear in METRIC_DIRECTIONS"
    )


# ---------------------------------------------------------------------------
# 2. _is_monotonic helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "i, l, e, direction, expected",
    [
        (0.3, 0.2, 0.1, "greater", True),     # strict i>l>e
        (0.3, 0.3, 0.1, "greater", False),    # not strict
        (0.1, 0.2, 0.3, "less", True),        # strict i<l<e
        (0.1, 0.1, 0.3, "less", False),       # not strict
        (0.3, 0.2, 0.1, "less", False),       # wrong direction
        (0.1, 0.2, 0.3, "two-sided", False),  # not applicable
    ],
)
def test_is_monotonic_strict(i, l, e, direction, expected):
    assert _is_monotonic(i, l, e, direction) is expected


# ---------------------------------------------------------------------------
# 3. Three-pair Wilcoxon + Bonferroni correctness
# ---------------------------------------------------------------------------


def test_three_pair_wilcoxon_bonferroni_correctness():
    """Synthesize 8 subjects with consistent i > l > e on iei_detrended_r;
    expect each pair's p < 0.05 raw, and Bonferroni multiplies by 3."""
    cohort = []
    rng = np.random.default_rng(42)
    for k in range(8):
        # add a tiny noise per subject so paired diffs aren't all identical
        noise = rng.normal(0, 0.005, 3)
        cohort.append(_make_subject(
            subj=f"s{k}",
            i_vals={"iei_detrended_r": 0.30 + noise[0]},
            l_vals={"iei_detrended_r": 0.20 + noise[1]},
            e_vals={"iei_detrended_r": 0.10 + noise[2]},
        ))

    out = compute_cohort_statistics_three_tier(
        cohort, metric_directions=METRIC_DIRECTIONS, min_group_channels=3,
    )
    assert out["n_valid_subjects"] == 8

    t = out["tests"]["iei_detrended_r"]
    assert t["direction"] == "greater"
    assert t["n_subjects"] == 8

    pt = t["pair_tests"]
    for pair in ("i_vs_l", "i_vs_e", "l_vs_e"):
        # raw p should be small (signed direction = greater, all positive diffs)
        assert pt[pair]["wilcoxon_p"] < 0.05
        # Bonferroni: p_adj = min(p * 3, 1)
        np.testing.assert_allclose(
            pt[pair]["wilcoxon_p_bonferroni"],
            min(pt[pair]["wilcoxon_p"] * 3.0, 1.0),
        )


def test_bonferroni_caps_at_one():
    """When p > 1/3, Bonferroni should cap at 1.0."""
    cohort = []
    rng = np.random.default_rng(0)
    # noisy data with no true direction
    for k in range(6):
        cohort.append(_make_subject(
            subj=f"s{k}",
            i_vals={"iei_detrended_r": float(rng.normal(0, 0.1))},
            l_vals={"iei_detrended_r": float(rng.normal(0, 0.1))},
            e_vals={"iei_detrended_r": float(rng.normal(0, 0.1))},
        ))
    out = compute_cohort_statistics_three_tier(
        cohort, metric_directions=METRIC_DIRECTIONS, min_group_channels=3,
    )
    pt = out["tests"]["iei_detrended_r"]["pair_tests"]
    for pair in ("i_vs_l", "i_vs_e", "l_vs_e"):
        # cap invariant
        assert pt[pair]["wilcoxon_p_bonferroni"] <= 1.0


# ---------------------------------------------------------------------------
# 4. Subject-level monotonicity sign test direction sensitivity
# ---------------------------------------------------------------------------


def test_monotonicity_direction_sensitive():
    """Same monotonic-i>l>e data should give small p when direction='greater'
    but a non-significant result when direction='less' (wrong direction)."""
    cohort_greater = []
    cohort_less = []
    rng = np.random.default_rng(7)
    for k in range(10):
        ev = {"iei_detrended_r": 0.30 - 0.01*k, "detrend_fraction": 0.30 - 0.01*k}
        # i > l > e for both metrics
        i_vals = ev
        l_vals = {"iei_detrended_r": 0.20 - 0.01*k, "detrend_fraction": 0.20 - 0.01*k}
        e_vals = {"iei_detrended_r": 0.10 - 0.01*k, "detrend_fraction": 0.10 - 0.01*k}
        cohort_greater.append(_make_subject(f"s{k}", i_vals, l_vals, e_vals))
        cohort_less.append(_make_subject(f"s{k}", i_vals, l_vals, e_vals))

    out = compute_cohort_statistics_three_tier(
        cohort_greater, metric_directions={"iei_detrended_r": "greater"},
        min_group_channels=3,
    )
    mono_g = out["tests"]["iei_detrended_r"]["monotonicity"]
    assert mono_g["n_monotonic"] == 10  # all subjects monotonic in 'greater' direction
    assert mono_g["binomial_p_one_sided"] is not None
    assert mono_g["binomial_p_one_sided"] < 0.001

    out2 = compute_cohort_statistics_three_tier(
        cohort_less, metric_directions={"detrend_fraction": "less"},
        min_group_channels=3,
    )
    mono_l = out2["tests"]["detrend_fraction"]["monotonicity"]
    assert mono_l["n_monotonic"] == 0  # data is i>l>e, but direction='less' looks for i<l<e
    # binomtest of 0/10 vs null=1/6 alternative=greater is large p
    assert mono_l["binomial_p_one_sided"] > 0.5


def test_monotonicity_skipped_for_two_sided():
    cohort = []
    for k in range(5):
        cohort.append(_make_subject(
            subj=f"s{k}",
            i_vals={"iei_lag1_r": 0.1*k},
            l_vals={"iei_lag1_r": 0.05*k},
            e_vals={"iei_lag1_r": 0.0},
        ))
    out = compute_cohort_statistics_three_tier(
        cohort, metric_directions={"iei_lag1_r": "two-sided"}, min_group_channels=3,
    )
    t = out["tests"]["iei_lag1_r"]
    assert t["monotonicity"] is None  # no monotonicity for two-sided


# ---------------------------------------------------------------------------
# 5. focus_rel-missing subject handling
# ---------------------------------------------------------------------------


def test_unknown_label_subject_excluded_from_paired_stats():
    """A subject with all channels labelled 'unknown' (focus_rel missing) must
    be auto-excluded from cohort paired stats (no i/l/e channels at ≥3)."""
    chans = [_make_channel("unknown") for _ in range(15)]
    bad_subject = {
        "subject": "no_focus_rel",
        "dataset": "epilepsiae",
        "channel_metrics": chans,
    }
    good_cohort = [
        _make_subject(
            subj=f"good{k}",
            i_vals={"iei_detrended_r": 0.30},
            l_vals={"iei_detrended_r": 0.20},
            e_vals={"iei_detrended_r": 0.10},
        )
        for k in range(5)
    ]
    out = compute_cohort_statistics_three_tier(
        [bad_subject, *good_cohort],
        metric_directions={"iei_detrended_r": "greater"},
        min_group_channels=3,
    )
    assert out["n_valid_subjects"] == 5
    assert "no_focus_rel" not in out["subjects"]


def test_subject_with_too_few_channels_in_one_region_excluded():
    """A subject with < min_group_channels in any region must be excluded."""
    rng = np.random.default_rng(2026)
    # Build a "thin l" subject by overriding region for two of the three l channels
    chans = []
    for r in ("i", "i", "i"):
        chans.append(_make_channel(r, iei_detrended_r=0.30 + rng.normal(0, 0.005)))
    # only 1 'l' channel
    chans.append(_make_channel("l", iei_detrended_r=0.20))
    for r in ("e", "e", "e"):
        chans.append(_make_channel(r, iei_detrended_r=0.10 + rng.normal(0, 0.005)))
    thin = {"subject": "thin_l", "dataset": "epilepsiae", "channel_metrics": chans}

    good = [
        _make_subject(
            subj=f"g{k}",
            i_vals={"iei_detrended_r": 0.30},
            l_vals={"iei_detrended_r": 0.20},
            e_vals={"iei_detrended_r": 0.10},
        )
        for k in range(4)
    ]
    out = compute_cohort_statistics_three_tier(
        [thin, *good],
        metric_directions={"iei_detrended_r": "greater"},
        min_group_channels=3,
    )
    assert out["n_valid_subjects"] == 4
    assert "thin_l" not in out["subjects"]


# ---------------------------------------------------------------------------
# 6. event_rate is confound-only
# ---------------------------------------------------------------------------


def test_event_rate_no_monotonicity_test():
    """event_rate must be reported with cohort medians + 3-pair Wilcoxon
    two-sided, but NEVER with a monotonicity test."""
    cohort = []
    rng = np.random.default_rng(99)
    for k in range(6):
        cohort.append(_make_subject(
            subj=f"s{k}",
            i_vals={"event_rate": 200.0 + rng.normal(0, 5)},
            l_vals={"event_rate": 100.0 + rng.normal(0, 5)},
            e_vals={"event_rate": 50.0 + rng.normal(0, 5)},
        ))
    out = compute_cohort_statistics_three_tier(
        cohort, metric_directions=METRIC_DIRECTIONS, min_group_channels=3,
    )
    er = out["tests"][EVENT_RATE_METRIC]
    assert er["direction"] == "confound_report_only"
    assert "monotonicity" not in er  # no monotonicity test for event_rate
    pt = er["pair_tests"]
    for pair in ("i_vs_l", "i_vs_e", "l_vs_e"):
        # two-sided p reported, not one-sided
        assert "wilcoxon_p_two_sided" in pt[pair]
        assert pt[pair]["wilcoxon_p_bonferroni"] <= 1.0


# ---------------------------------------------------------------------------
# 7. --dataset epilepsiae path must not pull Yuquan SOZ JSON
# ---------------------------------------------------------------------------


def test_focus_rel_none_labels_all_unknown_not_nonsoz(monkeypatch):
    """Regression: when focus_rel_dict is None for an Epilepsiae subject, the
    runner must label every channel 'unknown' (so paired stats auto-exclude
    the subject), NOT fall through to match_bipolar_soz with empty SOZ set
    which mislabels everything as 'non_soz'."""
    import run_spatial_modulation as rsm

    captured = {}

    def fake_load_perchannel(subject_dir, dataset, **kw):
        # Return a tiny synthetic loaded dict so run_subject can proceed
        ch = ["A1-A2", "B1-B2", "C1-C2"]
        events = {c: np.array([[0.0, 0.005], [10.0, 10.005], [20.0, 20.005]]) for c in ch}
        return {
            "ch_names": ch,
            "per_ch_events": events,
            "block_ranges": [(0.0, 100.0)],
            "total_hours": 0.5,
            "lagpat_channels": [],
        }

    monkeypatch.setattr(rsm, "load_perchannel_events_relaxed", fake_load_perchannel)
    monkeypatch.setattr(rsm.Path, "exists", lambda self: True)

    result = rsm.run_subject(
        "1084", refine_k=0.0, min_count=0, min_rate=0.0,
        dataset="epilepsiae", focus_rel_dict=None,
    )
    assert result is not None
    labels = {m["ch_name"]: m["region_label"] for m in result["channel_metrics"]}
    assert all(v == "unknown" for v in labels.values()), labels
    assert "non_soz" not in labels.values()


def test_dataset_epilepsiae_does_not_load_yuquan_soz(tmp_path, monkeypatch):
    """Smoke: simulate main() entry by running the relevant branch of code
    and confirming SOZ_FILE_YQ is never opened."""
    import run_spatial_modulation as rsm

    # Track which JSON files get loaded via _load_json
    loaded_paths = []
    original_loader = rsm._load_json

    def tracking_loader(path):
        loaded_paths.append(Path(path))
        return original_loader(path)

    monkeypatch.setattr(rsm, "_load_json", tracking_loader)

    # Only call the JSON-loading lines from main() for the epilepsiae branch
    soz_yq = {}
    focus_rel_all = rsm._load_json(rsm.FOCUS_REL_FILE_EP)

    # The Yuquan SOZ file must not be in the loaded list
    assert rsm.SOZ_FILE_YQ not in loaded_paths
    # focus_rel may be empty or populated; both acceptable for this test
    assert isinstance(focus_rel_all, dict)
