import numpy as np
import pytest
from scripts.run_topic5_directional_replay import process_subject, PRIMARY_COHORT, _report_tier


def test_primary_cohort_is_six_ecog():
    assert PRIMARY_COHORT == ["epilepsiae_442", "epilepsiae_548", "epilepsiae_583",
                              "epilepsiae_1084", "epilepsiae_384", "epilepsiae_958"]


def test_process_subject_442_smoke():
    """Integration: needs real cache + frame + template files present."""
    rec = process_subject("epilepsiae_442", "broadband", n_perm=50, n_boot=50, seed=20260627)
    assert rec["status"] == "ok"
    for k in ("n_sz", "sizes", "R_dir", "R_axial", "p_bimodal", "stability",
              "two_class_eligible", "theta_A", "theta_B", "delta_ictal_deg", "delta_ab_deg",
              "axis_tier", "best_pair_resid_sum_deg", "best_pair_resid_each_deg",
              "best_pair_pairing", "p_align", "report_tier", "geometry_clean",
              "electrode_kind", "coord_aspect", "activation"):
        assert k in rec
    assert rec["axis_tier"] in ("interpretable", "weak_axis", "diagnostic_only")
    assert rec["report_tier"] in ("two_class_mapped", "two_class_unmapped",
                                  "single_axis", "diagnostic_only")


def test_report_tier_geometry_unclean_forces_diagnostic():
    assert _report_tier("interpretable", True, 0.001, geometry_clean=False) == "diagnostic_only"
    assert _report_tier("interpretable", True, 0.001, geometry_clean=True) == "two_class_mapped"
    assert _report_tier("interpretable", False, 0.001, geometry_clean=True) == "single_axis"
