from scripts.paper_figures.plot_fig5_zm_qualification_boundary import build_payload


def test_boundary_payload_keeps_full_edge_and_comparator_roles_separate():
    records = [
        {
            "candidate_id": "full", "primary_zm_only": True,
            "edge_dose_comparator": False, "parameters": {},
            "model_ictal_qualification": {
                "joint_duty": 0.75, "contact_centroid_shift_hz": 30.0,
                "contact_centroid_ratio": 2.0,
            },
        },
        {
            "candidate_id": "edge", "primary_zm_only": False,
            "edge_dose_comparator": True, "parameters": {},
            "model_ictal_qualification": {
                "joint_duty": 0.95, "contact_centroid_shift_hz": 20.0,
                "contact_centroid_ratio": 2.0,
            },
        },
    ]
    points = build_payload(records)
    assert points[0]["full_learned_edges"] is True
    assert points[0]["edge_expression_comparator"] is False
    assert points[1]["full_learned_edges"] is False
    assert points[1]["edge_expression_comparator"] is True
