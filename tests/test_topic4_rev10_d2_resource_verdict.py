from scripts.audit_topic4_rev10_d2_resource_canary import adjudicate


def _row(candidate, mode, k=0.0, a=0, b=3, both=0, score=10.0):
    return {"candidate_id":candidate,"resource_mode":mode,"resource_k_q_per_ms":k,
            "networks_with_clean_A":a,"networks_with_clean_B":b,"networks_with_both_clean_modes":both,
            "selection_score_equal_network":score,"n_runaway_networks":0}


def _summary(passing=True):
    rows=[_row("edge_noop","off",a=0,both=0)]
    for k in (0.01,0.03,0.1):
        ok=passing and k==0.03
        rows.append(_row(f"local_{k}","local",k,a=2 if ok else 0,both=2 if ok else 0,score=8 if ok else 11))
        rows.append(_row(f"global_{k}","global",k,a=0,both=0,score=12))
    return {"candidate_rows":rows}


def test_local_resource_must_exceed_global_and_off():
    verdict=adjudicate(_summary()); assert verdict["selected_local_candidate_id"]=="local_0.03"


def test_absent_local_specific_support_closes_canary():
    verdict=adjudicate(_summary(False)); assert verdict["status"].endswith("NOT_OBSERVED")
