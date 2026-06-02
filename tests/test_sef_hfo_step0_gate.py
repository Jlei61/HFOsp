"""Step 0 go/no-go gate. SMOKE tier: structural checks on scaffold runs.
FORMAL tier (unlocks Step 1): requires data-locked provenance + anchored units."""
import json
from pathlib import Path
A = Path("results/topic4_sef_hfo/linear_stability/step0a_stability.json")
B = Path("results/topic4_sef_hfo/finite_pulse/step0b_pulse.json")

def test_smoke_screen_is_fraction_and_n_converged():
    d = json.loads(A.read_text())
    assert 0.0 <= d["low_heterogeneity_screen"]["fraction_closer"] <= 1.0
    assert d["erlang_n_convergence"]["converged"], "Erlang delay n not converged -> dispersion unreliable"

def test_smoke_step0b_sensitivity_stable():
    d = json.loads(B.read_text())
    off = d["runs"]["recovery_off"]["fraction_with_window"]
    assert abs(off - d["sensitivity"]["half_dt"]) <= 0.2, "dt-sensitive -> numerical, not physical"
    assert abs(off - d["sensitivity"]["smaller_L"]) <= 0.2, "L-sensitive -> boundary wrap artifact"

def test_formal_gate_requires_data_locked_provenance_and_window():
    """FORMAL tier: skip on scaffold runs; enforce on data-locked runs."""
    d = json.loads(A.read_text()); prov = d["provenance"]
    if prov.get("source") != "data_locked":
        import pytest; pytest.skip("scaffold run: NOT a formal Step-1-unlock pass")
    assert prov["locked_before_sweep"] is True and prov["hash"]
    b = json.loads(B.read_text())
    assert max(b["runs"]["recovery_off"]["fraction_with_window"],
               b["runs"]["recovery_on"]["fraction_with_window"]) > 0, \
        "no self-limited window with positive margin -> Step 1 stays LOCKED (a finding)"
