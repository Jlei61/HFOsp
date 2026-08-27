from src.topic5_continuous_marked_state_h2b.audit import split_contract_audit


def test_split_contract_accepts_disjoint_patient_events():
    value = split_contract_audit({
        "p1": {"TRAIN": ["s1", "s2"], "SELECT": ["s3"], "TEST": ["s4"]},
        "p2": {"LOSO": ["q1", "q2"]},
    })
    assert value["pass"] is True
    assert value["overlapping_seizure_ids"] == {}


def test_split_contract_rejects_same_seizure_in_two_splits():
    value = split_contract_audit({
        "p1": {"TRAIN": ["s1", "s2"], "TEST": ["s2", "s3"]},
    })
    assert value["pass"] is False
    assert value["overlapping_seizure_ids"] == {"p1": ["s2"]}
