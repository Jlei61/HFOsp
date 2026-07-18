import numpy as np

from scripts.plot_topic5_interictal_template_ab_fields import (
    INTERICTAL_FIELD_FIGURE_CONTRACT,
    _canonical_transverse_sign,
    _display_name,
    _load_yuquan_crosswalk,
    _transverse_display_signs,
    build_interictal_ab_panel_payloads,
    draw_interictal_rank_field_panel,
    load_interictal_field_records,
    plot_interictal_ab_atlas,
    plot_interictal_ab_subject,
)
from scripts.plot_topic5_field_vs_ictal_swap import (
    _field_panel,
    draw_topic5_field_panel,
)


def test_public_interictal_field_figure_reuse_contract_is_locked():
    assert INTERICTAL_FIELD_FIGURE_CONTRACT == "topic5_interictal_ab_field_figure_v1"
    assert _field_panel is draw_topic5_field_panel
    for function in (
        build_interictal_ab_panel_payloads,
        draw_interictal_rank_field_panel,
        load_interictal_field_records,
        plot_interictal_ab_subject,
        plot_interictal_ab_atlas,
    ):
        assert callable(function)


def test_canonical_transverse_sign_makes_dominant_component_positive():
    assert _canonical_transverse_sign([0.1, -0.8, 0.2]) == -1
    assert _canonical_transverse_sign([0.7, -0.2, 0.1]) == 1


def test_shared_plane_uses_one_common_transverse_sign():
    planes = {"shared": {"w": [0.1, -0.9, 0.2]}}
    sign_a, sign_b, rmse = _transverse_display_signs(
        "shared", planes, [-2, 0, 2], [-2, 0, 2],
    )
    assert (sign_a, sign_b) == (-1, -1)
    assert rmse == 0.0


def test_own_plane_b_sign_minimizes_same_contact_transverse_rmse():
    planes = {
        "own_a": {"w": [0.9, 0.1, 0.0]},
        "own_b": {"w": [0.0, 0.8, 0.2]},
    }
    y_a = np.array([-3.0, -1.0, 2.0, 4.0])
    y_b = -y_a + np.array([0.1, -0.1, 0.1, -0.1])
    sign_a, sign_b, rmse = _transverse_display_signs("own", planes, y_a, y_b)
    assert sign_a == 1
    assert sign_b == -1
    assert rmse < 0.11


def test_private_yuquan_crosswalk_is_read_without_hard_coding_mapping(tmp_path):
    crosswalk = tmp_path / "crosswalk.md"
    crosswalk.write_text(
        "| Public ID | artifact_folder | evidence |\n"
        "|---|---|---|\n"
        "| Y3 | example_folder | private |\n"
    )
    labels = _load_yuquan_crosswalk(crosswalk)
    assert _display_name("yuquan_example_folder", labels) == "Y3"
    assert _display_name("epilepsiae_1096", labels) == "E1096"
