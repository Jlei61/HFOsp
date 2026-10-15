"""Small CPU integration check using the copied SNN engine, not a paper-scale run."""
import json
import numpy as np
from modeling.src.snn_engine.params import Params
from modeling.src.snn_engine.model import build_network, simulate
from modeling.src.snn_engine.mz_slow_vars import MZSlowVars, MZSlowVarsConfig


def run_smoke():
    p = Params(L=1.0, density=240.0, T=100.0, dt=0.1, seed=20261015,
               C_EE=20, C_IE=20, C_EI=5, C_II=5)
    first = build_network(p, verbose=False)
    plain = simulate(p, first, verbose=False)
    second = build_network(p, verbose=False)
    disabled = MZSlowVars(second["NE"] + second["NI"], p.V_th, NE=second["NE"],
                         cfg=MZSlowVarsConfig(use_z=False, use_m=False))
    control = simulate(p, second, slow=disabled, verbose=False)
    for name in ("times", "rate_E", "rate_I", "spk_t", "spk_i"):
        np.testing.assert_array_equal(plain[name], control[name])
        assert np.isfinite(plain[name]).all()
    return {"status": "PASS", "neurons": second["NE"] + second["NI"], "duration_ms": p.T,
            "disabled_zm_matches_baseline": True, "scope": "CPU integration only; not a paper fit or state-transition result"}


if __name__ == "__main__":
    print(json.dumps(run_smoke(), indent=2))
