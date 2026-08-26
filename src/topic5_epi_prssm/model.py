"""The assembled Epi-PRSSM, its per-patient tensors and the cohort batch.

The three state objects stay separate down to the field names: ``state`` is the
slow generative graph state ``H``, ``resource`` is ``r``, ``observer_state`` is
``c``, and the fast event state exists only inside ``EventSteps``.

The slow-state scan is batched across patients (padded contact lanes, masked) so
that one Python step advances the whole cohort; the event readout is evaluated
per patient on unpadded tensors so a 6-contact patient never pays for a
52-contact one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from .contracts import FROZEN
from .event_decoder import EventDecoder, build_event_steps
from .event_marks import PatientEvents
from .graph_cells import GeneratorCell
from .graph_templates import PatientGraph
from .observer import FlexibleResourceCorrection, PersistentObserver
from .patient_baseline import PatientBaseline
from .resource_dynamics import ResourceState
from .state_adapter import StateAdapter

#: Elapsed time used where a session opens.  A session boundary is not an
#: inter-event interval, so the generator is advanced by the frozen join
#: threshold rather than by an unknown multi-hour gap; the boundary flag reaches
#: the observer separately so the two are never conflated.
SESSION_OPEN_DELTA_T = FROZEN["session_join_seconds"]


@dataclass
class PatientTensors:
    """One patient's frozen inputs on device."""

    subject: str
    dataset: str
    participation: torch.Tensor
    group_ids: torch.Tensor
    n_groups: torch.Tensor
    marks: torch.Tensor
    delta_t: torch.Tensor
    log_delta_t: torch.Tensor
    session_open: torch.Tensor
    load: torch.Tensor
    split: torch.Tensor
    event_time: np.ndarray
    adjacency: torch.Tensor
    node_features: torch.Tensor
    baseline_order: torch.Tensor
    baseline_participation: torch.Tensor
    baseline_stop: torch.Tensor
    n_contacts: int
    n_events: int
    #: (E, F) causal observable time features -- multi-scale rate, interval,
    #: coverage, time of day.  Only the nuisance-baseline arm reads them.
    nuisance: torch.Tensor | None = None
    #: Graph used by the within-event decoder.  Defaults to ``adjacency``, which is
    #: also what the slow generator propagates along, so the two paths are normally
    #: the same object.  Setting it separately is what lets a graph null shuffle one
    #: path while holding the other fixed -- without that, "the slow message needs
    #: this patient's wiring" cannot be separated from "the decoder needs this
    #: patient's spatial prior", because shuffling hits both at once.
    decoder_adjacency: torch.Tensor | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def split_bounds(self, split_value: int) -> tuple[int, int]:
        index = torch.nonzero(self.split == split_value, as_tuple=False).flatten()
        if len(index) == 0:
            return 0, 0
        return int(index[0]), int(index[-1]) + 1


def build_patient_tensors(events: PatientEvents, graph: PatientGraph,
                          baseline: PatientBaseline, *, device: str = "cpu",
                          dtype: torch.dtype = torch.float32) -> PatientTensors:
    delta = np.array(events.delta_t, dtype=np.float64)
    delta[~np.isfinite(delta)] = SESSION_OPEN_DELTA_T
    delta = np.maximum(delta, 0.0)
    to = lambda a, d=dtype: torch.as_tensor(np.asarray(a), dtype=d, device=device)
    return PatientTensors(
        subject=events.subject, dataset=events.dataset,
        participation=to(events.participation, torch.bool),
        group_ids=to(events.group_ids.astype(np.int64), torch.long),
        n_groups=to(events.group_count.astype(np.int64), torch.long),
        marks=to(events.node_marks()),
        delta_t=to(delta), log_delta_t=to(np.log1p(delta)),
        session_open=to(events.session_opening.astype(np.float32)),
        load=to(events.load), split=to(events.split.astype(np.int64), torch.long),
        event_time=events.event_time,
        adjacency=to(graph.stack()), node_features=to(events.contact_features),
        baseline_order=to(baseline.order_score),
        baseline_participation=to(baseline.participation_logit),
        baseline_stop=to(np.float32(baseline.stop_logit)),
        n_contacts=events.n_contacts, n_events=events.n_events,
        meta={"length_scale_mm": graph.length_scale_mm,
              "graph_train_events": graph.n_train_events,
              "baseline_train_events": baseline.n_train_events},
    )


@dataclass
class CohortBatch:
    """Patients aligned by local step index with padded contact lanes."""

    patients: tuple[PatientTensors, ...]
    starts: np.ndarray
    lengths: np.ndarray
    n_pad: int
    node_mask: torch.Tensor        # (P, n_pad) float
    adjacency: torch.Tensor        # (P, R, n_pad, n_pad)
    device: str = "cpu"

    @property
    def n_patients(self) -> int:
        return len(self.patients)

    @property
    def max_length(self) -> int:
        return int(self.lengths.max()) if len(self.lengths) else 0

    def gather(self, t0: int, t1: int) -> dict[str, torch.Tensor]:
        """Padded per-step inputs for local steps ``[t0, t1)``."""
        span = t1 - t0
        P, N = self.n_patients, self.n_pad
        dev = self.device
        delta = torch.zeros(P, span, device=dev)
        log_delta = torch.zeros(P, span, device=dev)
        session = torch.zeros(P, span, device=dev)
        load = torch.zeros(P, span, device=dev)
        marks = torch.zeros(P, span, N, 3, device=dev)
        active = torch.zeros(P, span, device=dev)
        n_nuisance = next((p.nuisance.shape[-1] for p in self.patients
                           if p.nuisance is not None), 0)
        nuisance = torch.zeros(P, span, n_nuisance, device=dev) if n_nuisance else None
        for p, patient in enumerate(self.patients):
            lo = self.starts[p] + t0
            hi = self.starts[p] + min(t1, self.lengths[p])
            take = max(hi - lo, 0)
            if take <= 0:
                continue
            delta[p, :take] = patient.delta_t[lo:hi]
            log_delta[p, :take] = patient.log_delta_t[lo:hi]
            session[p, :take] = patient.session_open[lo:hi]
            load[p, :take] = patient.load[lo:hi]
            marks[p, :take, : patient.n_contacts] = patient.marks[lo:hi]
            active[p, :take] = 1.0
            if nuisance is not None and patient.nuisance is not None:
                nuisance[p, :take] = patient.nuisance[lo:hi]
        step = {"delta_t": delta, "log_delta_t": log_delta, "session_open": session,
                "load": load, "marks": marks, "active": active}
        if nuisance is not None:
            step["nuisance"] = nuisance
        return step


def build_cohort_batch(patients: Sequence[PatientTensors], starts: Sequence[int],
                       lengths: Sequence[int], *, device: str = "cpu") -> CohortBatch:
    n_pad = max(p.n_contacts for p in patients)
    P = len(patients)
    node_mask = torch.zeros(P, n_pad, device=device)
    n_rel = patients[0].adjacency.shape[0]
    adjacency = torch.zeros(P, n_rel, n_pad, n_pad, device=device)
    for p, patient in enumerate(patients):
        n = patient.n_contacts
        node_mask[p, :n] = 1.0
        adjacency[p, :, :n, :n] = patient.adjacency
    return CohortBatch(tuple(patients), np.asarray(starts, dtype=int),
                       np.asarray(lengths, dtype=int), n_pad, node_mask, adjacency, device)


@dataclass
class SlowState:
    """``z = (H, r)`` plus the observer memory ``c`` and exposure ``x`` beside it."""

    state: torch.Tensor          # (P, N, D)
    resource: torch.Tensor       # (P,)
    observer_state: torch.Tensor # (P, O)
    exposure: torch.Tensor       # (P,)

    def detach(self) -> "SlowState":
        return SlowState(self.state.detach(), self.resource.detach(),
                         self.observer_state.detach(), self.exposure.detach())

    def select(self, index: int) -> "SlowState":
        return SlowState(self.state[index: index + 1], self.resource[index: index + 1],
                         self.observer_state[index: index + 1], self.exposure[index: index + 1])

    def blend(self, other: "SlowState", active: torch.Tensor) -> "SlowState":
        """Keep ``self`` where a patient has run out of events in this window."""
        a1 = active.view(-1, 1, 1)
        a0 = active.view(-1)
        return SlowState(
            other.state * a1 + self.state * (1 - a1),
            other.resource * a0 + self.resource * (1 - a0),
            other.observer_state * a0.view(-1, 1) + self.observer_state * (1 - a0.view(-1, 1)),
            other.exposure * a0 + self.exposure * (1 - a0),
        )


class EpiPRSSM(nn.Module):
    """Shared-parameter generator + observer + adapter + decoder."""

    def __init__(self, *, generator_level: str = "G2", resource_arm: str = "R0",
                 adapter: str = "node_film", state_dim: int | None = None,
                 observer_dim: int | None = None, feature_dim: int = 6,
                 flexible_resource_correction: bool = False,
                 tau_r_seconds: float | None = None, freeze_tau_r: bool = False,
                 tau_x_seconds: float | None = None, exposure_kind: str = "clock",
                 time_mode: str = "clock", unconstrained_gru: bool = False,
                 freeze_state: bool = False, node_resolved_frozen_state: bool = False,
                 state_from_nuisance: bool = False, nuisance_dim: int = 10):
        super().__init__()
        self.state_dim = int(state_dim or FROZEN["state_dim_H"])
        self.observer_dim = int(observer_dim or FROZEN["observer_dim"])
        self.generator_level = generator_level
        self.resource_arm = resource_arm
        self.adapter_mode = adapter
        self.unconstrained_gru = bool(unconstrained_gru)
        self.tau_x_seconds = float(tau_x_seconds) if tau_x_seconds is not None else None
        self.exposure_kind = exposure_kind
        # "event_index" replaces every real interval with one unit step; it is the
        # control that says whether a state uses elapsed time or only event order
        if time_mode not in ("clock", "event_index"):
            raise ValueError(f"unknown time mode {time_mode!r}")
        self.time_mode = time_mode
        # capacity control: the adapter keeps all its parameters but the graph
        # state never moves, so any gain must come from adapter capacity alone
        self.freeze_state = bool(freeze_state)
        # A frozen state fixed at zero gives every contact the same adapter shift,
        # so it cannot express node-resolved structure and under-matches a moving
        # state's capacity.  This variant freezes a *node-resolved* state instead:
        # per-contact, learned from the static covariates, and never updated.
        self.node_resolved_frozen_state = bool(node_resolved_frozen_state)
        if node_resolved_frozen_state:
            self.state_init = nn.Linear(feature_dim, self.state_dim)
        # Observable-timing baseline: the "state" is a deterministic function of the
        # causal multi-scale rate / interval / coverage / time-of-day features and of
        # nothing else.  It never sees which contacts participated in past events, so
        # a latent-state arm that only matches this has shown timing, not repertoire.
        self.state_from_nuisance = bool(state_from_nuisance)
        if state_from_nuisance:
            self.nuisance_head = nn.Linear(nuisance_dim, self.state_dim)

        use_resource = resource_arm != "R0"
        self.generator = GeneratorCell(generator_level, self.state_dim, use_resource=use_resource)
        self.resource = ResourceState(resource_arm, self.state_dim,
                                      tau_r_seconds=tau_r_seconds, freeze_tau=freeze_tau_r)
        self.observer = PersistentObserver(self.state_dim, self.observer_dim)
        self.adapter = StateAdapter(adapter, self.state_dim)
        self.decoder = EventDecoder(self.state_dim, feature_dim)
        self.flexible = FlexibleResourceCorrection(self.observer_dim) if flexible_resource_correction else None
        if unconstrained_gru:
            self.free_cell = nn.GRUCell(self.observer_dim, self.state_dim)
            self.free_write = nn.Linear(self.state_dim, self.state_dim)
            # the unconstrained baseline gets a learnable decay in log space too,
            # so it is not handicapped by a hard-coded constant
            self.free_log_tau = nn.Parameter(torch.full((self.state_dim,), math.log(300.0)))

    # -- state ---------------------------------------------------------------
    def initial_state(self, batch: CohortBatch) -> SlowState:
        P, N, dev = batch.n_patients, batch.n_pad, batch.device
        state = torch.zeros(P, N, self.state_dim, device=dev)
        if self.node_resolved_frozen_state:
            for p, patient in enumerate(batch.patients):
                state[p, : patient.n_contacts] = torch.tanh(
                    self.state_init(patient.node_features))
            state = state * batch.node_mask.unsqueeze(-1)
        return SlowState(state, torch.ones(P, device=dev),
                         torch.zeros(P, self.observer_dim, device=dev),
                         torch.zeros(P, device=dev))

    # -- physical transition -------------------------------------------------
    def decay_exposure(self, exposure: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if self.tau_x_seconds is None:
            return exposure
        if self.exposure_kind == "event_count":
            return exposure * float(np.exp(-1.0 / max(self.tau_x_seconds, 1e-6)))
        return exposure * torch.exp(-torch.clamp(dt / self.tau_x_seconds, max=40.0))

    def propagate(self, z: SlowState, batch: CohortBatch, step: dict[str, torch.Tensor],
                  t: int) -> SlowState:
        """Autonomous transition; reads elapsed time only, never the event marks."""
        dt = step["delta_t"][:, t]
        if self.time_mode == "event_index":
            dt = torch.ones_like(dt)
        exposure = self.decay_exposure(z.exposure, dt)
        resource = self.resource.propagate(z.resource, z.state, dt, batch.node_mask,
                                           exposure=exposure)
        if self.state_from_nuisance:
            features = step.get("nuisance")
            if features is None:
                raise ValueError("the nuisance baseline needs per-event nuisance features")
            projected = torch.tanh(self.nuisance_head(features[:, t]))
            state = projected.unsqueeze(-2).expand(-1, batch.n_pad, -1) \
                * batch.node_mask.unsqueeze(-1)
        elif self.freeze_state:
            state = z.state
        elif self.unconstrained_gru:
            tau = torch.exp(torch.clamp(self.free_log_tau, math.log(0.5), math.log(1e6)))
            decay = torch.exp(-torch.clamp(dt.view(-1, 1, 1) / tau.view(1, 1, -1), max=40.0))
            state = z.state * decay
        else:
            state = self.generator.propagate(z.state, dt, batch.adjacency, resource,
                                             batch.node_mask)
        return SlowState(state, resource, z.observer_state, exposure)

    def absorb(self, z: SlowState, step: dict[str, torch.Tensor], t: int,
               *, load: torch.Tensor | None = None) -> SlowState:
        """Event-load impulse path (R2) and exposure accumulation (R3).

        ``load`` may be overridden with an expected load during observer-off
        rollout; the true future load is never read there.
        """
        value = step["load"][:, t] if load is None else load
        resource = self.resource.absorb_event(z.resource, value)
        exposure = z.exposure if self.tau_x_seconds is None else z.exposure + value
        return SlowState(z.state, resource, z.observer_state, exposure)

    def observe(self, z: SlowState, batch: CohortBatch, step: dict[str, torch.Tensor],
                t: int) -> tuple[SlowState, torch.Tensor, torch.Tensor]:
        marks = step["marks"][:, t]
        observer_state = self.observer.update(
            z.observer_state, marks, step["load"][:, t], step["log_delta_t"][:, t],
            step["session_open"][:, t], batch.node_mask)
        if self.state_from_nuisance or self.freeze_state:
            # no observer correction: this arm's state is fully determined by the
            # observable timing features (or is frozen)
            state, energy = z.state, torch.zeros((), device=z.state.device)
        elif self.unconstrained_gru:
            v = self.observer.encode(marks, step["load"][:, t], step["log_delta_t"][:, t],
                                     step["session_open"][:, t], batch.node_mask)
            pooled = self.free_cell(v, z.state.mean(-2))
            delta = torch.tanh(self.free_write(pooled)).unsqueeze(-2) * batch.node_mask.unsqueeze(-1)
            state = torch.clamp(z.state + delta, -8.0, 8.0)
            energy = (delta ** 2).mean()
        else:
            state, energy = self.observer.correct_graph_state(
                z.state, observer_state, marks, batch.node_mask)
        resource, penalty = z.resource, torch.zeros((), device=state.device)
        if self.flexible is not None:
            resource, penalty = self.flexible(resource, observer_state)
        return SlowState(state, resource, observer_state, z.exposure), energy, penalty

    # -- readout -------------------------------------------------------------
    def score_events(self, patient: PatientTensors, index: torch.Tensor,
                     state: torch.Tensor, resource: torch.Tensor,
                     return_steps: bool = False) -> dict[str, torch.Tensor]:
        """``state`` (T, N_p, D) unpadded, ``resource`` (T,)."""
        steps = build_event_steps(patient.participation[index], patient.group_ids[index],
                                  patient.n_groups[index])
        adapter_out = self.adapter(state, resource)
        return self.decoder(
            steps,
            adjacency=(patient.decoder_adjacency if patient.decoder_adjacency is not None
                       else patient.adjacency),
            node_features=patient.node_features,
            baseline_order=patient.baseline_order,
            baseline_participation=patient.baseline_participation,
            baseline_stop=patient.baseline_stop,
            state=state, resource=resource, adapter=adapter_out, return_steps=return_steps)

    def describe(self) -> dict[str, Any]:
        return {
            "generator_level": self.generator_level, "resource_arm": self.resource_arm,
            "adapter": self.adapter_mode, "state_dim": self.state_dim,
            "observer_dim": self.observer_dim,
            "flexible_resource_correction": self.flexible is not None,
            "unconstrained_gru": self.unconstrained_gru,
            "tau_x_seconds": self.tau_x_seconds, "exposure_kind": self.exposure_kind,
            "time_mode": self.time_mode, "freeze_state": self.freeze_state,
            "node_resolved_frozen_state": self.node_resolved_frozen_state,
            "state_from_nuisance": self.state_from_nuisance,
            "uses_graph_messages": bool(getattr(self.generator, "uses_messages", False))
                                   and not self.unconstrained_gru,
            "n_parameters": int(sum(p.numel() for p in self.parameters())),
        }
