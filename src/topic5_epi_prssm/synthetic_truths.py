"""Just-in-time synthetic truths -- unit tests for each scientific question.

These are not a pre-flight mega-grid.  Each truth exists to calibrate one
discrimination the next Goal depends on, and a truth that turns out to be
unidentifiable limits the interpretation of that model, not the whole project.

A synthetic patient is generated with irregular inter-event intervals, a session
structure with real gaps and a contact count drawn per patient, so the recovery
test exercises the same code paths as the human cohort.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .graph_templates import PatientGraph
from .model import PatientTensors

TRUTHS = (
    # generator / H1
    "no_state", "leaky_state", "graph_recurrent_state", "observer_overpowering",
    # readout / H2a
    "state_conditioned_suffix", "no_state_false_adapter",
    # seizure link / H2b
    "latent_preictal_drift", "event_rate_only_drift",
    # exposure / H3
    "t1_autonomous_resource", "r2_impulse", "r3_integrated_exposure",
    "hidden_common_cause", "event_count_only", "switching_state",
    "observer_resource_substitution", "resource_direct_excitability",
)

#: Truths whose resource acts the way the spec allows: it modulates the gain with
#: which the latent graph state reaches the readout, rather than writing into the
#: contact logits directly.  ``resource_direct_excitability`` deliberately uses the
#: forbidden direct path and exists to mark the family boundary.
RESOURCE_TRUTHS = ("t1_autonomous_resource", "r2_impulse", "r3_integrated_exposure",
                   "event_count_only", "observer_resource_substitution",
                   "hidden_common_cause")

#: what each truth is for, and which comparison it is allowed to license
TRUTH_PURPOSE: dict[str, dict[str, Any]] = {
    "no_state": {"goal": "goal1", "expect": "static wins or ties; a state model must not invent one"},
    "leaky_state": {"goal": "goal1", "expect": "G0 recovers it; graph recurrence adds nothing"},
    "graph_recurrent_state": {"goal": "goal1", "expect": "G1/G2 beat G0, and beat it open-loop"},
    "observer_overpowering": {"goal": "goal1", "expect": "filtered ties; open-loop separates"},
    "state_conditioned_suffix": {"goal": "goal2", "expect": "state adapters beat no_state; swap destroys the gain"},
    "no_state_false_adapter": {"goal": "goal2", "expect": "adapter capacity alone must not create a state gain"},
    "latent_preictal_drift": {"goal": "goal3", "expect": "state moves before onset beyond matched pseudo-onsets"},
    "event_rate_only_drift": {"goal": "goal3", "expect": "rate moves, state does not: the nuisance control must catch it"},
    "t1_autonomous_resource": {"goal": "goal4", "expect": "R1 beats R0; tau_r recoverable within an interval"},
    "r2_impulse": {"goal": "goal4", "expect": "R2 beats matched R1"},
    "r3_integrated_exposure": {"goal": "goal4", "expect": "R3 beats R2 and R1 at the generating timescale"},
    "hidden_common_cause": {"goal": "goal4", "expect": "raw-load gain appears but the innovation challenge kills it"},
    "event_count_only": {"goal": "goal4", "expect": "event-count kernel beats the clock kernel"},
    "switching_state": {"goal": "goal4", "expect": "a smooth resource cannot imitate switching"},
    "observer_resource_substitution": {"goal": "goal4", "expect": "flexible observer-resource correction imitates a resource"},
    "resource_direct_excitability": {"goal": "goal4", "expect": "a resource acting straight on contact excitability is OUTSIDE the model family by contract; the arms should fail to recover it, which marks the boundary rather than refuting a resource"},
}


@dataclass
class SyntheticCohort:
    patients: list[PatientTensors]
    truth: str
    metadata: dict[str, Any]


def generate(truth: str, *, seed: int, n_patients: int = 6, n_events: int = 2500,
             device: str = "cpu") -> SyntheticCohort:
    if truth not in TRUTHS:
        raise ValueError(f"unknown truth {truth!r}")
    rng = np.random.default_rng(seed)
    patients, meta = [], {"per_patient": []}
    for p in range(n_patients):
        n_contacts = int(rng.integers(6, 13))
        patient, info = _one_patient(truth, rng, n_contacts, n_events, device)
        patients.append(patient)
        meta["per_patient"].append(info)
    meta.update({"truth": truth, "seed": seed, "n_patients": n_patients,
                 "n_events": n_events, "purpose": TRUTH_PURPOSE[truth]})
    return SyntheticCohort(patients, truth, meta)


def _one_patient(truth: str, rng, n_contacts: int, n_events: int, device: str):
    coords = np.cumsum(rng.normal(0, 3.5, size=(n_contacts, 3)), axis=0).astype(np.float32)
    base_participation = rng.normal(0.3, 0.8, size=n_contacts)
    base_order = np.sort(rng.normal(0, 1.0, size=n_contacts))[::-1].copy()

    delta_t, session_open, event_time = _timeline(
        rng, n_events, rate_drift=(truth == "event_rate_only_drift"))
    tau_state = 1800.0
    tau_resource = 1200.0
    tau_exposure = 900.0

    z = np.zeros((n_events, n_contacts))
    resource = np.ones(n_events)
    exposure = 0.0
    load_gain = 0.0
    mode = np.zeros(n_events, dtype=int)

    direction = rng.normal(size=n_contacts)
    direction /= np.linalg.norm(direction) + 1e-9
    second = rng.normal(size=n_contacts)
    second -= second.dot(direction) * direction
    second /= np.linalg.norm(second) + 1e-9

    coupling = _coupling_matrix(coords, rng)
    state = np.zeros(n_contacts)
    resource_value, mode_value, activity_state = 1.0, 0, 0.0
    ewma = np.zeros(n_contacts)
    participation = np.zeros((n_events, n_contacts), dtype=bool)
    group_ids = np.full((n_events, n_contacts), -1, dtype=np.int16)
    group_count = np.zeros(n_events, dtype=np.int16)
    ranks = np.full((n_events, n_contacts), np.nan, dtype=np.float32)
    loads = np.zeros(n_events, dtype=np.float32)
    onsets = []

    for e in range(n_events):
        dt = float(delta_t[e])
        decay = np.exp(-dt / tau_state)
        if truth in ("graph_recurrent_state", "observer_overpowering", "state_conditioned_suffix",
                     "latent_preictal_drift") or truth in RESOURCE_TRUTHS:
            drive = coupling @ state
            state = decay * (state + 0.15 * dt / tau_state * drive) + \
                np.sqrt(max(1 - decay ** 2, 1e-6)) * rng.normal(0, 0.6, size=n_contacts)
        elif truth == "leaky_state":
            state = ewma.copy()
        elif truth == "switching_state":
            if rng.random() < 1 - np.exp(-dt / 3600.0):
                mode_value = 1 - mode_value
            state = (direction if mode_value == 0 else second) * 1.2
        else:
            state = np.zeros(n_contacts)

        if truth == "latent_preictal_drift" and (e % 900) > 850:
            state = state + direction * 1.5 * ((e % 900) - 850) / 50.0

        # --- resource / exposure --------------------------------------------
        if truth in RESOURCE_TRUTHS or truth == "resource_direct_excitability":
            # autonomous recovery towards an equilibrium set by a slowly varying
            # latent activity, so R1 has something identifiable to recover
            activity_state = activity_state * np.exp(-dt / tau_resource) + \
                np.sqrt(max(1 - np.exp(-2 * dt / tau_resource), 1e-6)) * rng.normal(0, 1.0)
            equilibrium = float(np.clip(0.75 + 0.2 * np.tanh(activity_state), 0.35, 1.0))
            resource_value = equilibrium + (resource_value - equilibrium) * np.exp(-dt / tau_resource)
        if truth in ("r3_integrated_exposure", "event_count_only"):
            # integrated exposure shifts the *set point* the resource relaxes to,
            # which is the R3 equation in closed form.  Decrementing the resource
            # every event instead would be an integrator and would simply drift to
            # the floor, carrying no information.
            if truth == "r3_integrated_exposure":
                exposure = exposure * np.exp(-dt / tau_exposure)
                shift = 3e-3 * exposure
            else:
                exposure = exposure * np.exp(-1.0 / 20.0)
                shift = 5e-2 * exposure
            target = float(np.clip(equilibrium - shift, 0.15, 1.0))
            resource_value = target + (resource_value - target) * np.exp(-dt / tau_resource)
            resource_value = float(np.clip(resource_value, 0.15, 1.0))
        if truth == "hidden_common_cause":
            hidden = np.sin(2 * np.pi * event_time[e] / 7200.0)
            resource_value = float(np.clip(0.6 + 0.35 * hidden, 0.05, 1.0))
            load_gain = 0.9 * hidden

        z[e] = state
        resource[e] = resource_value
        mode[e] = mode_value

        # --- emit one event --------------------------------------------------
        state_gain = 0.0 if truth in ("no_state", "no_state_false_adapter",
                                      "event_rate_only_drift") else 0.9
        if truth in RESOURCE_TRUTHS:
            # spec-compliant interface: the resource sets how strongly the latent
            # graph state reaches the readout, it never writes a contact logit
            state_gain = state_gain * (0.25 + 1.75 * resource_value)
        participation_logit = base_participation + state_gain * state + load_gain
        if truth == "resource_direct_excitability":
            participation_logit = participation_logit + 1.4 * (resource_value - 1.0)
        if truth == "no_state_false_adapter":
            participation_logit = participation_logit + 0.8 * np.tanh(coords[:, 0] / 10.0)

        probability = 1.0 / (1.0 + np.exp(-participation_logit))
        taking = rng.random(n_contacts) < probability
        if taking.sum() < 2:
            taking[np.argsort(-probability)[:2]] = True
        members = np.flatnonzero(taking)
        order_score = base_order[members] + state_gain * state[members]
        if truth in ("state_conditioned_suffix", "graph_recurrent_state"):
            order_score = order_score + 1.2 * state[members]
        gumbel = rng.gumbel(size=len(members))
        order = members[np.argsort(-(order_score + gumbel))]
        participation[e, members] = True
        group_ids[e, order] = np.arange(len(order), dtype=np.int16)
        group_count[e] = len(order)
        ranks[e, order] = np.linspace(0.0, 1.0, len(order)) if len(order) > 1 else 0.0
        loads[e] = len(members) / n_contacts
        ewma = 0.9 * ewma + 0.1 * (taking.astype(float) - 0.5)

        if truth in ("r2_impulse", "resource_direct_excitability"):
            resource_value = float(np.clip(resource_value * np.exp(-4e-3 * loads[e]), 0.15, 1.0))
        if truth in ("r3_integrated_exposure", "event_count_only"):
            exposure = exposure + loads[e]
        if truth == "event_rate_only_drift" and (e % 900) > 850:
            onsets.append(float(event_time[e]))

    graph = _graph_from_events(participation, group_ids, coords)
    features = np.concatenate([
        np.linspace(0, 1, n_contacts)[:, None],
        np.full((n_contacts, 1), 1.0 / n_contacts),
        (coords - coords.mean(0)) / (coords.std(0) + 1e-6),
        np.ones((n_contacts, 1)),
    ], axis=1).astype(np.float32)

    marks = np.stack([participation.astype(np.float32),
                      np.nan_to_num(ranks), (group_ids == 0).astype(np.float32)], axis=-1)
    split = np.full(n_events, 2, dtype=np.int64)
    split[: int(0.6 * n_events)] = 0
    split[int(0.6 * n_events): int(0.8 * n_events)] = 1
    to = lambda a, d=torch.float32: torch.as_tensor(np.asarray(a), dtype=d, device=device)
    patient = PatientTensors(
        subject=f"synthetic_{rng.integers(0, 1 << 30)}", dataset="synthetic",
        participation=to(participation, torch.bool), group_ids=to(group_ids.astype(np.int64), torch.long),
        n_groups=to(group_count.astype(np.int64), torch.long), marks=to(marks),
        delta_t=to(delta_t), log_delta_t=to(np.log1p(delta_t)),
        session_open=to(session_open.astype(np.float32)), load=to(loads),
        split=to(split, torch.long), event_time=event_time,
        adjacency=to(graph), node_features=to(features),
        baseline_order=to(_static_order_score(participation, group_ids, split == 0)),
        baseline_participation=to(_static_participation_logit(participation, split == 0)),
        baseline_stop=to(np.float32(0.0)), n_contacts=n_contacts, n_events=n_events,
        meta={"truth_state": z, "truth_resource": resource, "truth_mode": mode,
              "onsets": onsets, "tau_state": tau_state, "tau_resource": tau_resource,
              "tau_exposure": tau_exposure},
    )
    return patient, {"n_contacts": n_contacts, "n_events": n_events,
                     "resource_range": [float(resource.min()), float(resource.max())],
                     "state_sd": float(z.std())}


def _timeline(rng, n_events, *, rate_drift: bool = False):
    """Inter-event intervals, optionally with a slow multiplicative rate drift.

    ``rate_drift`` is what makes the ``event_rate_only_drift`` truth mean anything.
    Without it that truth differed from ``no_state`` only by a list of onset markers
    the arms never see, so it produced bit-identical fits and its declared purpose --
    "the rate moves, the spatial state does not" -- was never tested at all.
    """
    delta = np.exp(rng.normal(0.6, 1.1, size=n_events))
    if rate_drift:
        # a smooth multi-scale drift in log-rate; the spatial state stays frozen, so
        # an arrival model should follow this and a mark-only model should not
        index = np.arange(n_events)
        log_rate = (0.9 * np.sin(2 * np.pi * index / 900.0)
                    + 0.5 * np.sin(2 * np.pi * index / 4100.0 + 1.1))
        delta = delta * np.exp(-log_rate)
    session_open = np.zeros(n_events, dtype=bool)
    session_open[0] = True
    breaks = rng.choice(np.arange(1, n_events), size=max(n_events // 600, 1), replace=False)
    session_open[breaks] = True
    delta[session_open] = 300.0
    return delta.astype(np.float32), session_open, np.cumsum(delta).astype(np.float64)


def _coupling_matrix(coords, rng):
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    kernel = np.exp(-0.5 * (distance / (np.median(distance) + 1e-6)) ** 2)
    np.fill_diagonal(kernel, 0.0)
    kernel = kernel / (kernel.sum(1, keepdims=True) + 1e-9)
    return kernel - np.eye(len(coords))


def _graph_from_events(participation, group_ids, coords):
    n = participation.shape[1]
    counts = np.zeros((n, n))
    for e in range(min(len(participation), 2000)):
        index = np.flatnonzero(participation[e])
        if len(index) < 2:
            continue
        g = group_ids[e, index]
        counts[np.ix_(index, index)] += g[:, None] < g[None, :]
    forward = counts / (counts.sum(1, keepdims=True) + 1e-9)
    reverse = counts.T / (counts.T.sum(1, keepdims=True) + 1e-9)
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    geometry = np.exp(-0.5 * (distance / (np.median(distance) + 1e-6)) ** 2)
    np.fill_diagonal(geometry, 0.0)
    geometry = geometry / (geometry.sum(1, keepdims=True) + 1e-9)
    return np.stack([forward, reverse, geometry], axis=0).astype(np.float32)


def _static_participation_logit(participation, train_mask):
    rate = (participation[train_mask].sum(0) + 1.0) / (train_mask.sum() + 2.0)
    return (np.log(rate) - np.log1p(-rate)).astype(np.float32)


def _static_order_score(participation, group_ids, train_mask):
    n = participation.shape[1]
    score = np.zeros(n)
    for i in range(n):
        taken = participation[train_mask, i]
        if taken.sum() == 0:
            continue
        score[i] = -np.mean(group_ids[train_mask, i][taken])
    score = score - score.mean()
    return (score / (score.std() + 1e-6)).astype(np.float32)
