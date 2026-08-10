import numpy as np
import pytest
from src.topic4_core_field_runner import (
    canonical_checksum, cache_key, connectivity_config,
)


class _P:
    L = 20.0; density = 100.0; f_E = 0.8; seed = 1; g = 3.6
    C_EE = 800; C_IE = 800; C_EI = 200; C_II = 200
    l_EE = 0.380; l_IE = 0.250; l_EI = 0.250; l_II = 0.250
    rho_EE = 0.6; rho_IE = 0.0; rho_EI = 0.0; rho_II = 0.0
    tau0 = 0.1; v_axon = 0.3; delay_dt = 0.1


def test_cache_key_is_stable_for_an_unchanged_config():
    assert cache_key(connectivity_config(_P(), -22.8, 2.0)) == \
           cache_key(connectivity_config(_P(), -22.8, 2.0))


def test_cache_key_can_be_frozen_to_an_explicit_producer_commit():
    frozen = connectivity_config(
        _P(), -22.8, 2.0, git_commit="frozen-connectivity-producer")
    assert frozen["git_commit"] == "frozen-connectivity-producer"
    assert cache_key(frozen) == cache_key(connectivity_config(
        _P(), -22.8, 2.0, git_commit="frozen-connectivity-producer"))
    assert cache_key(frozen) != cache_key(connectivity_config(
        _P(), -22.8, 2.0, git_commit="different-producer"))


@pytest.mark.parametrize("field,value", [
    ("L", 21.0), ("density", 120.0), ("f_E", 0.75), ("seed", 2),
    ("C_EE", 700), ("C_IE", 700), ("C_EI", 150), ("C_II", 150),
    ("l_EE", 0.40), ("l_IE", 0.26), ("l_EI", 0.26), ("l_II", 0.26),
    ("rho_EE", 0.5), ("rho_IE", 0.1), ("rho_EI", 0.1), ("rho_II", 0.1),
    ("tau0", 0.2), ("v_axon", 0.4), ("delay_dt", 0.2),
])
def test_perturbing_any_connectivity_field_changes_the_key(field, value):
    base = cache_key(connectivity_config(_P(), -22.8, 2.0))
    p = _P(); setattr(p, field, value)
    assert cache_key(connectivity_config(p, -22.8, 2.0)) != base, field


@pytest.mark.parametrize("theta,ar", [(-20.0, 2.0), (-22.8, 1.5)])
def test_theta_and_aspect_ratio_are_in_the_key(theta, ar):
    assert cache_key(connectivity_config(_P(), theta, ar)) != \
           cache_key(connectivity_config(_P(), -22.8, 2.0))


def test_canonical_checksum_ignores_the_checksum_field_itself():
    """P0-7: the config stores its own checksum, so verification must recompute
    from the config MINUS that field, not compare a string with itself."""
    cfg = {"a": 1, "b": [1, 2]}
    c = canonical_checksum(cfg)
    assert canonical_checksum({**cfg, "checksum": c}) == c


def test_canonical_checksum_detects_a_changed_field():
    assert canonical_checksum({"a": 1, "b": [1, 3]}) != canonical_checksum({"a": 1, "b": [1, 2]})


import sys


@pytest.mark.integration
@pytest.mark.slow
def test_cache_hit_reproduces_the_built_network_bitwise(tmp_path):
    sys.path.insert(0, "src/snn_engine"); sys.path.insert(0, ".")
    from params import Params
    from src.topic4_core_field_runner import get_network
    p = Params(g=3.6, L=6.0, density=40.0, T=100.0, dt=0.1, nu_ext_ratio=1.0, seed=11)
    a, NE_a, NI_a, hit_a = get_network(p, -22.8, 2.0, str(tmp_path))
    b, NE_b, NI_b, hit_b = get_network(p, -22.8, 2.0, str(tmp_path))
    assert hit_a is False and hit_b is True
    assert (NE_a, NI_a) == (NE_b, NI_b)
    assert np.array_equal(a["pos"], b["pos"]) and np.array_equal(a["labels"], b["labels"])
    assert a["max_delay_steps"] == b["max_delay_steps"]
    for key in ("ampa_by_delay", "gaba_by_delay"):
        assert len(a[key]) == len(b[key])
        for Wa, Wb in zip(a[key], b[key]):
            assert np.array_equal(Wa.toarray(), Wb.toarray())
    assert not [f for f in tmp_path.iterdir() if f.suffix == ".tmp"]


@pytest.mark.integration
@pytest.mark.slow
def test_explicit_commit_cache_is_reused(tmp_path):
    sys.path.insert(0, "src/snn_engine"); sys.path.insert(0, ".")
    from params import Params
    from src.topic4_core_field_runner import get_network
    p = Params(g=3.6, L=6.0, density=40.0, T=100.0, dt=0.1,
               nu_ext_ratio=1.0, seed=12)
    _, _, _, first_hit = get_network(
        p, -22.8, 2.0, str(tmp_path), git_commit="frozen-source")
    _, _, _, second_hit = get_network(
        p, -22.8, 2.0, str(tmp_path), git_commit="frozen-source")
    assert first_hit is False and second_hit is True
    payload = __import__("pickle").load(next(tmp_path.glob("*.pkl")).open("rb"))
    assert payload["config"]["git_commit"] == "frozen-source"
