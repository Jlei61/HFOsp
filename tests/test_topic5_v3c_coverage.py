import math

from src.topic5_v3_mode_transition import load_v3_config
from src.topic5_v3c_coverage import coverage_metrics


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
