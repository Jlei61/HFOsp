"""M3A-A2 analyzer verdict: R-stay / R-excursion / R-runaway, boundary derived from B+lgr."""
import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "aza", os.path.join(os.path.dirname(__file__), "..", "scripts", "analyze_a2_pilot.py"))
aza = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(aza)


def _ro(rho_peak, tail, max_R, returned_after, B=1.35, lgr=1.16):
    # recovered if q_core_end*q_global_end > lgr/B = 0.859; product 0.92 recovers, 0.45 does not.
    prod = 0.92 if returned_after else 0.45
    return {"a2": {"seizure_boundary": B, "rho_static": lgr, "rho_peak": rho_peak,
                   "q_core_end": prod, "q_global_end": 1.0},
            "activity": {"tail_to_baseline_ratio": tail, "active_E_fraction_peak": 0.2, "peak_E_rate_hz": 10.0},
            "events": [{"R_class": max_R}]}


def test_stay_when_below_boundary():
    assert aza._run_verdict(_ro(1.10, 1.0, "R3", True)) == "R-stay"


def test_excursion_when_crosses_R4a_and_returns():
    assert aza._run_verdict(_ro(1.50, 1.2, "R4a", True)) == "R-excursion"


def test_runaway_when_crosses_but_not_returns():
    assert aza._run_verdict(_ro(1.70, 3.0, "R4b", False)) == "R-runaway"


def test_runaway_when_crosses_returns_but_phenotype_R2():
    # crossed + returned but only R2 (no seizure-like phenotype) -> not an excursion
    assert aza._run_verdict(_ro(1.50, 1.0, "R2", True)) == "R-runaway"


def test_no_boundary_when_B_none():
    r = _ro(1.50, 1.2, "R4a", True); r["a2"]["seizure_boundary"] = None
    assert aza._run_verdict(r) == "NO_BOUNDARY"     # frozen Task-0b runs: phenotype only, no verdict
