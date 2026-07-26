"""Task 4 (spec rev3.1 §3.2): matched future-noise continuations.

The load-bearing claim is EMPIRICAL, not architectural: two different scientific arms resumed from
the same snapshot with the same bank must receive a bit-identical external drive, so that any
difference between them is dynamics and not luck. We verify that against the engine's recorded
per-step drive (`zm_ext_nu`, `zm_ext_sum`), which is why the recorder exists.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import src.topic4_zm_noise_bank as NB  # noqa: E402
import src.topic4_zm_checkpoint as CK  # noqa: E402
import src.topic4_zm_fork_state as FS  # noqa: E402
from tests.test_topic4_zm_exact_resume import _continue, reference  # noqa: E402,F401

DT = 0.1


def test_bank_is_deterministic_and_separates_replicates():
    a = NB.build_noise_bank("CFG", 1, 1000, "noise_resample_1")
    b = NB.build_noise_bank("CFG", 1, 1000, "noise_resample_1")
    c = NB.build_noise_bank("CFG", 1, 1000, "noise_resample_2")
    d = NB.build_noise_bank("CFG", 3, 1000, "noise_resample_1")
    assert a["bank_sha"] == b["bank_sha"] and a["rng_state"] == b["rng_state"]
    assert a["bank_sha"] != c["bank_sha"] and a["rng_state"] != c["rng_state"]
    assert a["bank_sha"] != d["bank_sha"], "the bank must depend on the seed"
    assert NB.build_noise_bank("CFG", 1, 1000, "noise_replay")["rng_state"] is None


def test_deleting_the_mean_is_rejected():
    with pytest.raises(ValueError, match="mean_input_only"):
        NB.build_noise_bank("CFG", 1, 0, "noise_off")
    with pytest.raises(ValueError, match="unknown replicate"):
        NB.build_noise_bank("CFG", 1, 0, "whatever")


@pytest.mark.parametrize("replicate", ["noise_replay", "noise_resample_1"])
def test_all_arms_receive_a_bit_identical_external_drive(reference, replicate):  # noqa: F811
    """Arm-order / arm-identity invariance: the drive cannot depend on which arm consumes it."""
    tf = reference["forks"][1]
    bank = NB.build_noise_bank("CFG", 1, tf, replicate)
    n = 400
    drives = {}
    for arm in ("dynamic_replay", "freeze_all", "freeze_zm"):
        res, _, _ = _continue(reference, reference["snaps"][tf], n,
                              slow_wrap=FS.FreezePolicy.for_arm(arm),
                              ckpt_kw=dict(rng_state=bank["rng_state"]))
        drives[arm] = (res["zm_ext_nu"].copy(), res["zm_ext_sum"].copy())
    ref = drives["dynamic_replay"]
    for arm, (nu, ex) in drives.items():
        assert np.array_equal(nu, ref[0]), f"{arm}: external rate diverged"
        assert np.array_equal(ex, ref[1]), f"{arm}: external Poisson draw diverged"


def test_replay_reproduces_the_anchors_own_future_stream(reference):  # noqa: F811
    tf = reference["forks"][0]
    n = 400
    res, _, _ = _continue(reference, reference["snaps"][tf], n)
    assert np.array_equal(res["zm_ext_nu"], reference["full"]["zm_ext_nu"][tf:tf + n])
    assert np.array_equal(res["zm_ext_sum"], reference["full"]["zm_ext_sum"][tf:tf + n])


def test_resamples_are_the_same_ou_process_as_replay(reference):  # noqa: F811
    """A resampled stream must be a different realization of the SAME external process. Comparing
    two 300 ms sample stds of a tau=150 ms OU would be noise-on-noise, so we test the properties
    that are actually estimable at this window: the stationary mean, the OU correlation, and the
    delivered Poisson mean tracking the rate."""
    from params import Params
    from kick_probe import compute_nu_theta

    tf = reference["forks"][1]
    n = 3000
    p = reference["p"]
    NE_NI = reference["NE"] + reference["NI"]
    nu_const = p.nu_ext_ratio * compute_nu_theta(p)[0]
    sigma_xi = (p.sigma_n * 1e-3) * np.sqrt(p.tau_n / 2.0)      # OU stationary sd (engine formula)
    out = {}
    for rep in NB.PAIRED_REPLICATES:
        bank = NB.build_noise_bank("CFG", 1, tf, rep)
        res, _, _ = _continue(reference, reference["snaps"][tf], n,
                              ckpt_kw=dict(rng_state=bank["rng_state"]))
        out[rep] = NB.external_drive_stats(res["zm_ext_nu"], res["zm_ext_sum"], DT, NE_NI)
    for rep, s in out.items():
        assert abs(s["nu_mean"] - nu_const) < 5.0 * sigma_xi, f"{rep}: rate mean drifted off the OU"
        assert s["nu_lag1"] > 0.99, f"{rep}: lost the OU correlation time"
        assert abs(s["ext_mean_per_neuron"] - s["expected_ext_mean_per_neuron"]) \
            < 0.02 * s["expected_ext_mean_per_neuron"], f"{rep}: delivered drive != its rate"
    assert out["noise_resample_1"]["ext_mean_per_neuron"] \
        != out["noise_replay"]["ext_mean_per_neuron"], "resample was not an independent realization"


def test_mean_input_only_keeps_the_mean_and_removes_the_fluctuations(reference):  # noqa: F811
    tf = reference["forks"][1]
    n = 400
    bank = NB.build_noise_bank("CFG", 1, tf, "mean_input_only")
    assert bank["ext_mean_only"] and not bank["is_paired"]
    res, _, _ = _continue(reference, reference["snaps"][tf], n,
                          ckpt_kw=dict(ext_mean_only=True))
    ref, _, _ = _continue(reference, reference["snaps"][tf], n)
    nu = res["zm_ext_nu"]
    assert float(np.ptp(nu)) < 1e-12, "the OU fluctuation was not removed"
    # the retained level is the external SIGNAL mean, not zero and not the anchor's momentary nu
    assert nu[0] > 0.0
    assert abs(float(nu[0]) - float(np.mean(ref["zm_ext_nu"]))) < 0.35 * abs(float(nu[0]))
    # ...and the delivered drive is deterministic (no Poisson variance around it)
    assert float(np.ptp(res["zm_ext_sum"])) < 1e-9


def test_mean_input_only_is_flagged_diagnostic_not_paired():
    assert set(NB.PAIRED_REPLICATES) == {"noise_replay", "noise_resample_1", "noise_resample_2"}
    assert NB.DIAGNOSTIC_REPLICATES == ("mean_input_only",)
