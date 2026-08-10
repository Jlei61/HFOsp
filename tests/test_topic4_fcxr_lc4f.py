from src.topic4_fcxr_lc4f import adjudicate_screen, derive_candidate


def test_candidate_is_unique_clean_depth():
    lock = {"verdict": "ENTRY_OFFSET_REPAIR_IDENTIFIABLE",
            "candidate": {"theta_h_lc2": 1.73}}
    rows = [{"x_mean_min": x} for x in (0.3776, 0.3837, 0.3902)]
    c = derive_candidate(lock, *rows, {"x_can_terminate_at_observed_D": True}, y_gate=76.6)
    assert c["K_y"] == 3.0 and c["use_m"] is False


def test_positive_screen_requires_full_cycle_prefix_and_guard():
    events = [{"t_on": t, "returned": True} for t in (1000, 3000, 6000)]
    g = adjudicate_screen(regimes=["INTERICTAL"] * 8 + ["ICTAL"] * 3 + ["INTERICTAL"] * 3,
                          win_ms=1000, events=events, numerical_safe=True,
                          refractory_fraction=0, pre_rate_hz=4, post_rate_hz=1,
                          m_current_max=0)
    assert g["verdict"] == "X_DEPTH_OFFSET_CANDIDATE"


def test_persistent_high_is_not_candidate():
    g = adjudicate_screen(regimes=["INTERICTAL"] * 8 + ["ICTAL"] * 8,
                          win_ms=1000, events=[], numerical_safe=True,
                          refractory_fraction=0, pre_rate_hz=4, post_rate_hz=float("nan"),
                          m_current_max=0)
    assert g["verdict"] == "X_DEPTH_OFFSET_NEGATIVE"

