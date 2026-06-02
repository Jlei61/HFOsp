# src/sef_hfo_field.py
"""SEF-HFO Step-0 rate field. ALL numeric defaults are TEST-ONLY SCAFFOLDS
(Step 0a/0b screen them); formal results require data-anchored units (see gate).
See docs/archive/topic4/sef_hfo_topic4_v2_plan_2026-06-01.md 2026-06-02 amendment."""
from dataclasses import dataclass

@dataclass(frozen=True)
class SEFParams:
    n: int = 64
    L: float = 64.0
    tau_E: float = 1.0
    tau_I: float = 2.0
    tau_a: float = 20.0           # recovery timescale (anchor to event duration at formal run)
    tau_AMPA: float = 0.5         # E-presynaptic synaptic filter (fast)
    tau_GABA: float = 2.0         # I-presynaptic synaptic filter (slow)
    delay_d: float = 1.0          # conduction delay (Erlang chain mean)
    erlang_n: int = 2             # Erlang stages approximating the delay (n->inf = pure delay)
    J_EE: float = 1.0
    J_EI: float = 1.0
    J_IE: float = 1.0
    J_II: float = 0.5
    ell_par: float = 6.0
    ell_perp: float = 2.0
    axis_angle: float = 0.0
    sigma_I: float = 10.0
    sigma_IE: float = 4.0
    sigma_II: float = 4.0
    gamma_global: float = 0.2
    b_a: float = 0.0              # recovery strength; 0 = OFF
    beta: float = 4.0
    phi_bar: float = 0.0
    sigma_phi: float = 1.0
