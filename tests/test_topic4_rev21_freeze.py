import json
from pathlib import Path

from scripts.freeze_topic4_rev21_dual_core_zm_transition import build_candidates


ROOT = Path(__file__).resolve().parents[1]


def _rev20_candidate():
    return {
        "candidate_id": "dc_both_scale_1p25",
        "node_field": {"field_sha256":
                       "901f6e839a59ac8d3b2ffc28d13359677b0bf5e2fa7fbd180b38c1d99dc91128"},
        "node_mapping": {"node_gain": 1.0, "signed_depth_shrinkage": 1.0},
        "mechanisms": {"g_EE": 0.625, "g_EtoI": 1.25,
                       "ellipse_angle_deg": 45.0,
                       "ellipse_aspect_ratio": 2.0, "Z_M": "off"},
    }


def test_coarse_library_changes_only_slow_variables():
    config = json.loads((ROOT / "config/topic4_rev21_dual_core_zm_transition.json").read_text())
    rows = build_candidates(config, _rev20_candidate())
    assert len(rows) == 17
    assert rows[0]["slow_variables"] == {"mode": "off"}
    frozen = [(row["node_field"], row["node_mapping"],
               {k: row["mechanisms"][k] for k in
                ("g_EE", "g_EtoI", "ellipse_angle_deg", "ellipse_aspect_ratio")})
              for row in rows]
    assert all(value == frozen[0] for value in frozen)
    active = rows[1:]
    assert {row["slow_variables"]["I_th_EI_scale"] for row in active} == {
        0.7, 0.8, 0.9, 1.0}
    assert {row["slow_variables"]["integrated_M_scale"] for row in active} == {
        0.5, 1.0, 1.5, 2.0}
    products = {
        round(row["slow_variables"]["eta_m"]
              * row["slow_variables"]["tau_adp"], 12)
        for row in active if row["slow_variables"]["integrated_M_scale"] == 1.0
    }
    assert products == {round(0.007451594355587098 * 500.0, 12)}
