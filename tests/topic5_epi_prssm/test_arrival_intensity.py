"""Contract tests for the arrival-time channel.

The clauses here are science contract, not implementation detail:
  * an interval spanning a metadata gap carries no evidence in either direction;
  * a recorded interval with no discharge in it *is* evidence and must enter the
    compensator -- that silence is most of what identifies the rate;
  * the state enters only through a multiplicative term, so "the state moves the
    rate" stays a single interpretable coefficient;
  * a correctly specified intensity must pass time rescaling.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.topic5_epi_prssm.arrival import RenewalIntensity  # noqa: E402


def _model(state_dim: int = 3, n_patients: int = 2) -> RenewalIntensity:
    torch.manual_seed(0)
    return RenewalIntensity(n_patients, state_dim)


def test_unrecorded_interval_contributes_nothing_in_either_direction():
    model = _model()
    elapsed = torch.tensor([2.0, 900.0, 5.0])
    state = torch.zeros(3, 3)
    patient = torch.zeros(3, dtype=torch.long)
    all_recorded = model(elapsed, state, patient, torch.tensor([True, True, True]))
    gap_in_middle = model(elapsed, state, patient, torch.tensor([True, False, True]))
    assert gap_in_middle["log_intensity"][1] == 0.0
    assert gap_in_middle["compensator"][1] == 0.0
    # dropping the interval must change the likelihood, or the gap rule is a no-op
    assert not torch.isclose(all_recorded["nll"], gap_in_middle["nll"])
    assert gap_in_middle["n_recorded"] == 2


def test_recorded_silence_enters_the_compensator():
    """A long recorded interval must cost more survival than a short one."""
    model = _model()
    state = torch.zeros(1, 3)
    patient = torch.zeros(1, dtype=torch.long)
    short = model(torch.tensor([1.0]), state, patient, torch.tensor([True]))
    long = model(torch.tensor([3600.0]), state, patient, torch.tensor([True]))
    assert float(long["compensator"]) > float(short["compensator"])


def test_state_enters_only_multiplicatively_and_starts_neutral():
    model = _model()
    elapsed = torch.tensor([3.0, 3.0])
    patient = torch.zeros(2, dtype=torch.long)
    recorded = torch.tensor([True, True])
    flat = model(elapsed, torch.zeros(2, 3), patient, recorded)
    varied = model(elapsed, torch.randn(2, 3), patient, recorded)
    # zero-initialised state weight: the model begins as a pure renewal process
    assert torch.allclose(flat["nll"], varied["nll"])
    with torch.no_grad():
        model.state_weight.weight.fill_(0.5)
    moved = model(elapsed, torch.ones(2, 3), patient, recorded)
    assert not torch.allclose(flat["nll"], moved["nll"])


def test_cumulative_baseline_is_monotone():
    model = _model()
    with torch.no_grad():
        model.basis_weight.normal_(0.0, 0.5)
    assert bool((torch.diff(model.cumulative_baseline()) >= 0).all())


def test_time_rescaling_of_a_correct_constant_rate_model_is_unit_exponential():
    """The goodness-of-fit check a mark-only model could never be run against."""
    torch.manual_seed(1)
    rate = 0.2
    elapsed = torch.distributions.Exponential(rate).sample((20000,))
    model = _model(state_dim=1, n_patients=1)
    with torch.no_grad():
        model.basis_weight.zero_()               # flat hazard == exponential intervals
        model.baseline.fill_(math.log(rate))
    recorded = torch.ones(len(elapsed), dtype=torch.bool)
    rescaled = model.rescaled_times(elapsed, torch.zeros(len(elapsed), 1),
                                    torch.zeros(len(elapsed), dtype=torch.long), recorded)
    assert float(rescaled.mean()) == pytest.approx(1.0, abs=0.05)
    assert float(rescaled.std()) == pytest.approx(1.0, abs=0.10)


def test_initialise_from_puts_the_baseline_near_the_empirical_rate():
    model = _model(state_dim=1, n_patients=1)
    elapsed = torch.full((500,), 4.0)
    recorded = torch.ones(500, dtype=torch.bool)
    model.initialise_from(elapsed, recorded)
    assert float(model.baseline) == pytest.approx(math.log(0.25), abs=1e-6)


def test_rate_state_pathway_is_not_born_dead():
    """Both sides zero-initialised makes every gradient in the pathway exactly zero.

    A smoke run caught this as ``t2_event_driven`` reproducing ``renewal_only`` to
    every decimal: the state was identically zero, so the readout had nothing to
    grade against and the drive had no readout to push through.
    """
    from src.topic5_epi_prssm.rate_state import RateState

    torch.manual_seed(0)
    n = 64
    elapsed = torch.rand(n) * 10 + 1
    time_of_day = torch.rand(n) * 6.28
    since_open = torch.log1p(torch.rand(n) * 100)
    load = torch.rand(n) * 5
    segment_start = torch.zeros(n, dtype=torch.bool)
    segment_start[0] = True

    for arm in ("t0_exogenous_clock", "t1_observer", "t2_physical"):
        state_module = RateState(4, arm=arm)
        state = state_module(elapsed, time_of_day, since_open, load, segment_start)
        assert float(state.abs().sum()) > 0, f"{arm}: state is identically zero at init"

        intensity = RenewalIntensity(1, 4)
        out = intensity(elapsed, state, torch.zeros(n, dtype=torch.long),
                        torch.ones(n, dtype=torch.bool))
        out["nll"].backward()
        grad = intensity.state_weight.weight.grad
        assert grad is not None and float(grad.abs().sum()) > 0, (
            f"{arm}: the readout weight has exactly zero gradient, so the rate state "
            "can never be learned and any negative result would be an artefact")


def test_the_clock_arm_carries_no_discharge_information():
    """t0 must be blind to the events; otherwise it is not a clock-only control."""
    from src.topic5_epi_prssm.rate_state import RateState

    torch.manual_seed(0)
    n = 32
    elapsed = torch.rand(n) * 5 + 1
    tod = torch.rand(n) * 6.28
    since = torch.log1p(torch.rand(n) * 50)
    seg = torch.zeros(n, dtype=torch.bool)
    seg[0] = True
    clock = RateState(4, arm="t0_exogenous_clock")
    quiet = clock(elapsed, tod, since, torch.ones(n), seg)
    busy = clock(elapsed, tod, since, torch.full((n,), 20.0), seg)
    assert torch.allclose(quiet, busy), "the clock arm reacted to the discharges"

    observer = RateState(4, arm="t1_observer")
    with torch.no_grad():
        observer.observer_gain.weight.fill_(0.2)
    assert not torch.allclose(observer(elapsed, tod, since, torch.ones(n), seg),
                              observer(elapsed, tod, since,
                                       torch.rand(n) * 20, seg))


def test_a_bounded_variable_needs_both_of_its_boundaries_counted():
    """A bound written on one tail of a two-sided variable reads the wrong failure.

    The resource falls from 1: pinned low means exhausted, pinned high means never
    consumed.  A metric that counts only the low end reports 0.0 -- "not pinned" --
    for a run whose resource never left its ceiling.
    """
    import numpy as np

    exhausted = np.full(1000, 5e-4)
    never_consumed = np.full(1000, 1.0)

    floor_only = lambda v: float((v <= 1.01e-3).mean())
    both = lambda v: float((v <= 1.01e-3).mean() + (v >= 1.0 - 1.01e-3).mean())

    assert floor_only(exhausted) == pytest.approx(1.0)
    assert floor_only(never_consumed) == pytest.approx(0.0)   # the blind spot
    assert both(exhausted) == pytest.approx(1.0)
    assert both(never_consumed) == pytest.approx(1.0)         # now visible


def test_the_fit_actually_moves_its_parameters():
    """One optimiser step per epoch leaves everything at initialisation.

    The first full-cohort arrival run took a single step per epoch, so 25 steps
    fitted the whole model: the time constants came back bit-identical to their
    log-spaced initialisation and every richer arm scored worse than the plainest
    one.  That reads as a negative result and is an optimisation failure.
    """
    from src.topic5_epi_prssm.rate_state import RateState

    torch.manual_seed(0)
    n = 400
    elapsed = torch.distributions.Exponential(0.3).sample((n,)).clamp(min=0.1)
    time_of_day = torch.rand(n) * 6.28
    since_open = torch.log1p(torch.arange(float(n)))
    load = torch.rand(n) * 5
    segment_start = torch.zeros(n, dtype=torch.bool)
    segment_start[0] = True
    patient = torch.zeros(n, dtype=torch.long)
    recorded = ~segment_start

    state_module = RateState(4, arm="t1_observer")
    intensity = RenewalIntensity(1, 4)
    intensity.initialise_from(elapsed, recorded)
    before = state_module.time_constants().clone()

    parameters = list(state_module.parameters()) + list(intensity.parameters())
    optimiser = torch.optim.Adam(parameters, lr=0.01)
    for _ in range(60):
        optimiser.zero_grad()
        state = state_module(elapsed, time_of_day, since_open, load, segment_start)
        loss = intensity(elapsed, state, patient, recorded)["nll"] / recorded.sum()
        loss.backward()
        optimiser.step()

    after = state_module.time_constants()
    assert not torch.allclose(before, after), (
        "the time constants never left their initialisation: the fit is not training")


# --------------------------------------------------------------------------
# contract tests added after the 2026-08-19 review
# --------------------------------------------------------------------------

def test_the_state_for_an_interval_cannot_depend_on_when_that_interval_ends():
    """The survival term must not peek at the outcome it is scoring.

    The first version advanced the state through interval e and then recorded it, so
    the compensator for that interval was computed from the state at t_e^- -- which
    is a function of the arrival time being modelled.  Changing only ``elapsed[e]``
    must leave row e untouched.
    """
    from src.topic5_epi_prssm.rate_state import RateState

    torch.manual_seed(0)
    n = 12
    base = torch.full((n,), 30.0)
    tod = torch.rand(n) * 6.28
    since = torch.log1p(torch.arange(float(n)) * 10)
    load = torch.rand(n) * 5
    seg = torch.zeros(n, dtype=torch.bool)
    seg[0] = True

    for arm in ("t0_exogenous_clock", "t1_observer", "t2_physical"):
        module = RateState(4, arm=arm)
        a = module(base, tod, since, load, seg)
        longer = base.clone()
        longer[5] = 9000.0                      # interval 5 ends much later
        b = module(longer, tod, since, load, seg)
        assert torch.allclose(a[5], b[5]), (
            f"{arm}: row 5 moved when interval 5's own duration changed -- the "
            "compensator is reading its own outcome")
        assert not torch.allclose(a[6], b[6]), (
            f"{arm}: row 6 did not move, so the interval duration never propagates")


def test_observer_and_physical_arms_differ_on_a_perfectly_predicted_event():
    """A discharge that was fully expected informs nothing but still pushes.

    This is the whole identifiability of H3 on this channel: if the two arms behaved
    the same on a predictable stream, their difference could not separate "the
    discharges tell us about the state" from "the discharges move the state".
    """
    from src.topic5_epi_prssm.rate_state import RateState

    torch.manual_seed(0)
    n = 40
    elapsed = torch.full((n,), 20.0)
    tod = torch.zeros(n)
    since = torch.zeros(n)
    load = torch.full((n,), 3.0)                # perfectly predictable
    seg = torch.zeros(n, dtype=torch.bool)
    seg[0] = True

    observer = RateState(4, arm="t1_observer")
    physical = RateState(4, arm="t2_physical")
    physical.load_state_dict(observer.state_dict(), strict=False)
    with torch.no_grad():
        physical.observer_gain.weight.copy_(observer.observer_gain.weight)
        physical.exogenous.weight.copy_(observer.exogenous.weight)
        physical.exogenous.bias.copy_(observer.exogenous.bias)
        physical.physical_gain.weight.fill_(0.1)

    zo = observer(elapsed, tod, since, load, seg)
    zp = physical(elapsed, tod, since, load, seg)
    # the observer's innovations vanish on a constant stream; the physical push does not
    assert float(zo[-1].abs().sum()) < float(zp[-1].abs().sum())


def test_the_rate_drift_truth_actually_drifts_the_rate():
    """A negative control that generates the same data as its comparison tests nothing.

    ``event_rate_only_drift`` used to differ from ``no_state`` only by a list of onset
    markers the fitting arms never see, so the two truths produced bit-identical fits
    across every seed and every arm, and the control's declared purpose -- "the rate
    moves, the spatial state does not" -- was never exercised.
    """
    import numpy as np

    from src.topic5_epi_prssm.synthetic_truths import generate

    plain = generate("no_state", seed=0, n_patients=2, n_events=1200)
    drift = generate("event_rate_only_drift", seed=0, n_patients=2, n_events=1200)

    a = np.asarray(plain.patients[0].delta_t, dtype=float)
    b = np.asarray(drift.patients[0].delta_t, dtype=float)
    assert not np.allclose(a, b), "the rate-drift truth produced the same timeline"

    # the drift must be slow and substantial, not just jitter
    def block_rate(x):
        blocks = np.array_split(x, 8)
        return np.array([1.0 / max(np.mean(v), 1e-9) for v in blocks])
    spread = block_rate(b).max() / block_rate(b).min()
    assert spread > 1.5, f"rate varies only {spread:.2f}x across the record"

    # and the spatial state must stay frozen, or the truth is no longer rate-only
    assert np.allclose(np.asarray(plain.patients[0].participation, dtype=float).mean(),
                       np.asarray(drift.patients[0].participation, dtype=float).mean(),
                       atol=0.05)
