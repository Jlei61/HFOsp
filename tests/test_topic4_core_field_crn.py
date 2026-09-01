"""Locks the premise Stage 1's paired design rests on.

If anyone introduces a spike-dependent RNG call, changing the threshold field
would desynchronise the noise between arms and the paired probe would stop
meaning anything. Equal call counts would still permit a divergent stream, so the
driving normals themselves are recorded and compared.
"""
import sys
import numpy as np
import pytest

sys.path.insert(0, "src/snn_engine")
sys.path.insert(0, ".")


class RecordingGenerator(np.random.Generator):
    """Generator attributes are read-only, so recording needs a subclass."""

    def __init__(self, bit_generator):
        super().__init__(bit_generator)
        self.normals = []
        self.poisson_calls = 0

    def standard_normal(self, *args, **kwargs):
        v = super().standard_normal(*args, **kwargs)
        self.normals.append(float(v) if np.ndim(v) == 0 else float(np.asarray(v).sum()))
        return v

    def poisson(self, *args, **kwargs):
        self.poisson_calls += 1
        return super().poisson(*args, **kwargs)


@pytest.mark.integration
@pytest.mark.slow
def test_changing_the_threshold_field_leaves_the_noise_trajectory_identical(tmp_path):
    from params import Params
    from kick_probe import simulate_kick
    from src.topic4_core_field_runner import get_network

    p = Params(g=3.6, L=6.0, density=40.0, T=200.0, dt=0.1, nu_ext_ratio=1.0, seed=5)
    net, NE, NI, _ = get_network(p, -22.8, 2.0, str(tmp_path))
    N = NE + NI

    def run(vth):
        gen = RecordingGenerator(np.random.PCG64(5))
        net["rng"] = gen
        simulate_kick(p, net, KICK_BOOST=0.0, kick_center=[3.0, 3.0], r_kick=1.0,
                      t_kick=1e9, V_th_per_neuron=vth)
        return gen

    flat = run(np.full(N, 18.0))
    lowered_vth = np.full(N, 18.0)
    lowered_vth[: NE // 4] = 16.0
    lowered = run(lowered_vth)

    assert flat.poisson_calls == lowered.poisson_calls
    assert len(flat.normals) == len(lowered.normals)
    assert np.array_equal(np.asarray(flat.normals), np.asarray(lowered.normals)), (
        "the OU driving noise diverged between threshold fields: common random "
        "numbers no longer hold and the Stage 1 paired probe is invalid"
    )


@pytest.mark.integration
@pytest.mark.slow
def test_two_arms_at_one_seed_start_from_identical_state(tmp_path):
    """No state is inherited between arms: rebuilding the rng at the same seed
    reproduces the run bit for bit."""
    from params import Params
    from kick_probe import simulate_kick
    from src.topic4_core_field_runner import get_network

    p = Params(g=3.6, L=6.0, density=40.0, T=200.0, dt=0.1, nu_ext_ratio=1.0, seed=5)
    net, NE, NI, _ = get_network(p, -22.8, 2.0, str(tmp_path))
    vth = np.full(NE + NI, 18.0)
    outs = []
    for _ in range(2):
        net["rng"] = np.random.default_rng(5)
        outs.append(simulate_kick(p, net, KICK_BOOST=0.0, kick_center=[3.0, 3.0],
                                  r_kick=1.0, t_kick=1e9,
                                  V_th_per_neuron=vth)["E_spk_bool"])
    assert np.array_equal(outs[0], outs[1])
