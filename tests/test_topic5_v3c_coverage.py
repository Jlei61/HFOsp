import math

import numpy as np

from src.topic5_v3_mode_transition import load_v3_config
from src.topic5_v3c_coverage import coverage_metrics, coverage_null_distribution


def test_v3c_config_keys():
    v = load_v3_config()["v3c"]
    assert v["z_cross"] == 2.0 and v["window_sec"] == 30.0 and v["hop_sec"] == 0.1
    assert v["assay_qc"]["t0_frac_max"] == 0.50 and v["assay_qc"]["finite_frac_min"] == 0.40
    assert v["interpretation"]["auc_hb_min"] == 0.60 and v["interpretation"]["auc_ha_band"] == [0.45, 0.55]
    assert v["latency"]["min_surplus"] == 3 and v["latency"]["min_covered_soz"] == 3
    assert v["spatial"]["min_subjects_for_primary"] == 3
    assert v["cohorts"]["primary"] == "broad" and v["nulls"]["n_perm"] == 1000


def test_coverage_metrics_basic():
    m = coverage_metrics(["a", "b", "c", "d"], ["b", "c", "e"])
    assert m["n_axis"] == 4 and m["n_soz"] == 3
    assert m["covered"] == ["b", "c"] and m["surplus"] == ["a", "d"] and m["missed"] == ["e"]
    assert m["coverage"] == 2 / 3
    assert m["surplus_fraction"] == 2 / 4
    assert m["jaccard"] == 2 / 5          # |A∩S|=2, |A∪S|={a,b,c,d,e}=5


def test_coverage_metrics_empty_soz():
    m = coverage_metrics(["a", "b"], [])
    assert math.isnan(m["coverage"]) and m["n_missed"] == 0 and m["surplus"] == ["a", "b"]


def _shaft(name):  # "H1".."H4" -> "H"; "G12" -> "G"
    return "".join(c for c in name if not c.isdigit())


def test_coverage_null_preserves_axis_count_and_size():
    all_clean = ["H1", "H2", "H3", "H4", "G1", "G2", "G3", "G4"]
    axis = ["H1", "H2", "G1"]                      # per-shaft: H=2, G=1
    soz = ["H1", "H2"]
    shaft = {n: _shaft(n) for n in all_clean}
    null = coverage_null_distribution(axis, all_clean, soz, shaft, n_perm=200, rng=0)
    assert null.shape == (200,)
    # coverage = |A_null ∩ {H1,H2}| / 2 ; A_null always has H=2 of {H1..H4}, G=1 of {G1..G4}
    assert set(np.unique(null)).issubset({0.0, 0.5, 1.0})
    assert null.max() == 1.0 and null.min() == 0.0     # both extremes reachable within shaft H


def test_coverage_null_soz_shuffle_regression():
    # If S is shuffled to non-axis positions, observed coverage should sit inside the null (not high)
    all_clean = ["H1", "H2", "H3", "H4"]
    axis = ["H1", "H2"]; shaft = {n: "H" for n in all_clean}
    null_when_soz_offaxis = coverage_null_distribution(axis, all_clean, ["H3", "H4"], shaft, n_perm=200, rng=1)
    # observed coverage(axis={H1,H2}, soz={H3,H4}) = 0; null spans 0..1 -> observed not above null
    assert null_when_soz_offaxis.mean() > 0.0
