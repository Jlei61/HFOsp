from scripts import run_topic5_event_innovation_v3_0_human_test as human


def test_patient_inference_is_two_sided_and_patient_level():
    result = human.patient_inference([1.0, 2.0, -1.0])
    assert result["n"] == 3
    assert result["n_positive"] == 2
    assert "wilcoxon_two_sided_p" in result
    assert "sign_test_two_sided_p" in result
