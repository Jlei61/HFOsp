# tests/test_sef_hfo_field.py
from src.sef_hfo_field import SEFParams

def test_params_scaffold_invariants():
    p = SEFParams()
    assert p.ell_perp < p.ell_par            # propagation axis exists
    assert p.sigma_I > p.ell_par             # wide inhibition
    assert p.tau_AMPA < p.tau_GABA           # fast excitation, slow inhibition
    assert p.b_a == 0.0                      # recovery OFF by default (switchable)
    assert p.erlang_n >= 1
