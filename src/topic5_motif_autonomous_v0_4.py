"""Train the motif family on what it is judged by: generating the future itself.

v0.1 built the whole generative evaluation — closed-loop stochastic rollout, a frozen
decoder, common random numbers, endpoint / extent / field summaries and an energy
score — and then trained every model on *teacher-forced next contact*.  The objective
and the yardstick were different tasks, and its own closeout recorded the symptom: the
generated events had roughly the right extent but their endpoints sat a median of
10.2 mm away from the real ones.

This module supplies the bridge.  After an observed prefix of the first two or three
rank sets, the model is rolled forward on its *own* prediction and scored on the
likelihood of the true rank sets.

Four things here are not free choices — a first version got each of them wrong and the
errors were large enough to change the science:

* **The likelihood is the sampler's law, not a lookalike.**  Summing per-contact
  softmax log-probabilities agrees with the exact fixed-cardinality subset law only for
  singletons; measured against it, cardinality two was off by 1.0 nats and three by
  2.8.  The exact law already existed in this repository and is reused here.
* **The true future never enters the support.**  Advancing the recruited mask with the
  observed next rank set makes the likelihood a teacher-forced quantity wearing an
  autonomous name; it moved the second-horizon score by 0.025 when only the first
  horizon's truth changed.  The prefix is a hard exclusion because it is observed; the
  model's own predictions enter only through a soft penalty.
* **The feedback is a legal rank set.**  ``probability * expected_size`` puts more than
  one unit of activity on a single contact (measured 2.9) and corresponds to nothing
  the sampler can emit.  The exact inclusion marginals under the same subset law do,
  and they satisfy ``0 <= m_i <= 1`` with ``sum_i m_i = E[k]``.
* **The prefix is encoded exactly as the sampler encodes it.**  Stepping and updating
  the displacement in the same iteration, rather than in the sampler's order, left
  training and evaluation looking at different direction-gate histories.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from src.topic5_dynamical_motif_rnn_v0_1 import MotifRNN, rollout_displacement_update
from src.topic5_dynamical_motif_rollout_v0_1 import _direction_weight
from src.topic5_shared_propagation_field import conditional_k_subset_log_prob
from src.topic5_strict_history_motif_v0_2 import expected_inclusion

# Large enough to zero a contact's probability, small enough to keep gradients finite.
MASK_PENALTY = 30.0
PRIMARY_HORIZONS = (1, 2, 3)
SENSITIVITY_HORIZONS = (4, 5)


def soft_available_logits(logits: Tensor, recruited: Tensor) -> Tensor:
    """Penalise contacts in proportion to how strongly they are already recruited.

    ``recruited`` runs from 0 to 1.  At the one-hot limit this subtracts the full
    penalty from exactly the taken contacts, which after the softmax is the sampler's
    hard "no repeats" rule; in between it stays differentiable.
    """
    return logits - MASK_PENALTY * recruited.clamp(0.0, 1.0)


def subset_log_likelihood(logits: Tensor, target: Tensor, candidate_mask: Tensor,
                          cardinality: Tensor) -> Tensor:
    """log p(the true rank set | its size) under the exact law the sampler uses.

    Delegates to the repository's ``conditional_k_subset_log_prob`` rather than
    re-deriving it: the normaliser is an elementary symmetric polynomial, and the
    per-contact softmax sum that resembles it is a different distribution as soon as
    more than one contact fires.
    """
    return conditional_k_subset_log_prob(
        logits, target.bool(), candidate_mask.bool(), cardinality=cardinality)


def cardinality_log_likelihood(size_logits: Tensor, cardinality: Tensor) -> Tensor:
    """log p(the true rank-set size)."""
    width = size_logits.shape[-1]
    index = cardinality.clamp(min=0, max=width - 1)
    return torch.log_softmax(size_logits, dim=-1).gather(-1, index[:, None])[:, 0]


def soft_rank_set(logits: Tensor, available: Tensor, size_logits: Tensor,
                  kmax: int) -> Tensor:
    """A legal soft rank set: exact inclusion marginals mixed over the size belief.

    Each entry is a probability that the contact is in the next set, so it lies in
    ``[0, 1]`` and the total mass is the expected cardinality — which is what the
    sampler emits in expectation, and what ``probability * expected_size`` is not.
    """
    width = max(1, min(int(kmax), int(size_logits.shape[-1])))
    cardinality_probability = torch.softmax(size_logits[..., :width], dim=-1)
    return expected_inclusion(logits, available, cardinality_probability, width)


def autonomous_trace(model: MotifRNN, size_head, prefix: Tensor, contacts_xy: Tensor,
                     horizons: int, gate_rule: str = "M2-2RANK",
                     kmax: int = 4) -> dict[str, Tensor]:
    """Roll the model forward on its own prediction after an observed prefix.

    The prefix loop mirrors ``stochastic_rollout`` step for step — gate from the current
    displacement, advance the state with the pending input, then load the next observed
    set and update the displacement — so training and evaluation share one direction-gate
    history.  ``prefix`` is ``(B, K, C)``.
    """
    batch, prefix_len, n_contacts = prefix.shape
    device = prefix.device
    state = torch.zeros(batch, model.n_nodes, device=device, dtype=prefix.dtype)
    terms = model.recurrent_terms()
    unit, _ = model.axis_unit()

    # every observed prefix contact is excluded from every future set, exactly as the
    # sampler does; this is knowledge about the past, not a look at the future
    observed_recruited = (prefix.sum(1) > 0).to(prefix.dtype)
    counts = prefix.sum(-1, keepdim=True).clamp_min(1.0)
    centroid_start = (prefix[:, 0] @ contacts_xy) / counts[:, 0]
    displacement = torch.zeros(batch, 2, device=device, dtype=prefix.dtype)
    pending = prefix[:, 0].clone()

    def advance(current: Tensor, inputs: Tensor, displacement_now: Tensor) -> Tensor:
        # the sampler's own weight helper, not a second expression that happens to agree
        # today: two definitions of the same quantity are how training and evaluation
        # drift apart without any test failing
        gate = model.direction_gate(displacement_now, unit)
        weight = _direction_weight(model, displacement_now, gate, unit)
        return model.step(current, inputs, gate, terms, weight)

    for position in range(1, prefix_len):
        state = advance(state, pending, displacement)
        pending = prefix[:, position]
        centroid_now = (prefix[:, position] @ contacts_xy) / counts[:, position]
        displacement = rollout_displacement_update(
            displacement, centroid_start, centroid_now, position, gate_rule)

    predicted_recruited = torch.zeros_like(observed_recruited)
    contact_logits, size_logits, feedback, support, features_out = [], [], [], [], []
    denominator = max(1, n_contacts - 1)
    for position in range(horizons):
        step_index = prefix_len - 1 + position
        state = advance(state, pending, displacement)
        raw = model.readout(state)
        # the support this step is scored against, recorded BEFORE the step's own
        # prediction is folded in, so the loss can use it without touching the future
        support.append(predicted_recruited.clone())
        penalised = soft_available_logits(raw, predicted_recruited)
        contact_logits.append(penalised)

        features = model.state_features(
            state,
            torch.full((batch,), step_index / denominator, device=device,
                       dtype=prefix.dtype),
            (observed_recruited + predicted_recruited).clamp(max=1.0).mean(dim=-1))
        features_out.append(features)
        sizes = size_head(features)
        size_logits.append(sizes)

        available = observed_recruited < 0.5
        soft = soft_rank_set(penalised, available, sizes, kmax)
        feedback.append(soft)

        pending = soft
        predicted_recruited = (predicted_recruited + soft).clamp(max=1.0)
        centroid_now = (soft @ contacts_xy) / soft.sum(-1, keepdim=True).clamp_min(1e-6)
        displacement = rollout_displacement_update(
            displacement, centroid_start, centroid_now, step_index + 1, gate_rule)

    return {
        "contact_logits": torch.stack(contact_logits, 1),
        "size_logits": torch.stack(size_logits, 1),
        "soft_rank_sets": torch.stack(feedback, 1),
        "predicted_recruited_before": torch.stack(support, 1),
        "state_features": torch.stack(features_out, 1),
        "observed_recruited": observed_recruited,
    }


def autonomous_loss(trace: dict[str, Tensor], targets: Tensor, cardinality: Tensor,
                    valid: Tensor, horizons: tuple[int, ...] = PRIMARY_HORIZONS
                    ) -> tuple[Tensor, dict[str, float]]:
    """Likelihood of the true future rank sets under the autonomous rollout.

    The support at each horizon comes from the trace, never from the observed future: a
    contact is excluded because it was in the *prefix*, or discounted because the model
    itself predicted it.  Folding the true previous rank set into the mask is what made
    an earlier version a teacher-forced score under an autonomous name.
    """
    contact = trace["contact_logits"]
    size = trace["size_logits"]
    observed = trace["observed_recruited"]
    steps = min(contact.shape[1], targets.shape[1], cardinality.shape[1], valid.shape[1])
    candidate = observed < 0.5

    per_step: list[Tensor] = []
    for position in range(steps):
        keep = valid[:, position] & (targets[:, position].sum(-1) > 0)
        if not bool(keep.any()):
            per_step.append(contact.sum() * 0.0)
            continue
        # rows are selected before the likelihood, not after: the exact subset law
        # checks that the target's size equals the cardinality it is given, so a padded
        # step past the end of a short event raises rather than being masked away
        sizes = cardinality[keep, position]
        spatial = subset_log_likelihood(
            contact[keep, position], targets[keep, position], candidate[keep], sizes)
        counts = cardinality_log_likelihood(size[keep, position], sizes - 1)
        per_step.append(-(spatial + counts).mean())

    weight = 1.0 / len(horizons)
    total = sum(weight * per_step[h - 1] for h in horizons if h - 1 < steps)
    detail = {f"h{position + 1}_nll": float(value.detach())
              for position, value in enumerate(per_step)}
    return total, detail


def kinematic_endpoint_extrapolation(prefix_centroids: np.ndarray,
                                     gain: float = 1.0) -> np.ndarray:
    """``r2 + a (r2 - r1)`` — the control the directed-transport motif has to beat.

    If a straight-line continuation of the first two steps predicts the endpoint as well
    as the recurrent model does, "directed transport" is describing inertia rather than
    a recurrent computation.
    """
    centroids = np.asarray(prefix_centroids, dtype=float)
    if centroids.ndim != 3 or centroids.shape[1] < 2:
        raise ValueError("need at least two prefix centroids per event")
    first, second = centroids[:, -2], centroids[:, -1]
    return second + gain * (second - first)


def autonomous_calibration_trace(model: MotifRNN, size_head, prefix: Tensor,
                                 targets: Tensor, valid: Tensor, contacts_xy: Tensor,
                                 gate_rule: str = "M2-2RANK",
                                 kmax: int = 4) -> dict[str, Tensor]:
    """A calibration trace built from autonomous states, in the sampler's schema.

    ``calibrate_temperatures`` and ``fit_size_head`` ask "given these states and these
    targets, which temperature fits best" — a question that does not care where the
    states came from, so they are reused unchanged.  What must change is the states: a
    model that will be *rolled out* has to be calibrated on the states rolling out
    produces, not on teacher-forced ones, or the temperature corrects a distribution the
    decoder never sees.

    The support recorded in ``available`` excludes the observed prefix and whatever the
    model itself predicted — never the observed future.
    """
    trace = autonomous_trace(model, size_head, prefix, contacts_xy,
                             horizons=targets.shape[1], gate_rule=gate_rule, kmax=kmax)
    observed = trace["observed_recruited"]
    steps = trace["contact_logits"].shape[1]
    available = (observed[:, None, :] < 0.5) & (
        trace["predicted_recruited_before"] < 0.5)

    # the STOP head reads the same features the state produced; they are recomputed here
    # rather than carried so that a change in ``state_features`` cannot silently diverge
    stop_logits = torch.stack(
        [model.stop_logit(trace["state_features"][:, position])
         for position in range(steps)], 1)

    present = targets.sum(-1) > 0
    is_last = present & ~torch.cat(
        [present[:, 1:], torch.zeros_like(present[:, :1])], dim=1)
    # detached at the source, not at each consumer: these states are *data* for the
    # decoder fits, and every one of them calls backward repeatedly.  Detaching only
    # where the error first appeared would leave the next consumer to hit it again.
    return {
        "features": trace["state_features"].detach(),
        "contact_logits": trace["contact_logits"].detach(),
        "stop_logits": stop_logits.detach(),
        "target": targets,
        "available": available,
        "predict": valid & present & ~is_last,
        "is_last": is_last,
        "valid": valid & present,
    }


def refit_stop_head_on_autonomous_states(model: MotifRNN, trace: dict[str, Tensor],
                                         max_epochs: int = 300, lr: float = 0.05
                                         ) -> dict[str, float]:
    """Fit only the termination head, with the spatial operator frozen.

    STOP never enters the spatial objective or the spatial model selection, so it is
    fitted afterwards.  Freezing everything else is what makes that separation real
    rather than nominal: the returned hash lets a caller assert the operator did not
    move, and a test does exactly that.
    """
    spatial = [parameter for name, parameter in model.named_parameters()
               if not name.startswith("stop_head")]
    for parameter in spatial:
        parameter.requires_grad_(False)
    head = [parameter for name, parameter in model.named_parameters()
            if name.startswith("stop_head")]
    if not head:
        raise RuntimeError("the model exposes no stop_head to refit")

    valid = trace["valid"]
    features = trace["features"][valid]
    target = trace["is_last"][valid].float()
    optimiser = torch.optim.Adam(head, lr=lr)
    before = float("nan")
    for epoch in range(max_epochs):
        optimiser.zero_grad()
        logits = model.stop_logit(features)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        if epoch == 0:
            before = float(loss.detach())
        loss.backward()
        optimiser.step()
    for parameter in spatial:
        parameter.requires_grad_(True)
    with torch.no_grad():
        after = float(torch.nn.functional.binary_cross_entropy_with_logits(
            model.stop_logit(features), target))
    return {"stop_bce_before": before, "stop_bce_after": after,
            "n_decisions": int(valid.sum())}


def spatial_parameter_hash(model: MotifRNN) -> str:
    """A digest of every parameter except the termination head.

    Used to assert that refitting STOP left the spatial operator untouched; a hash is
    checkable in a test and in the run log, unlike a promise in a docstring.
    """
    import hashlib

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if name.startswith("stop_head"):
            continue
        digest.update(name.encode())
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_autonomous_event_tensors(ranks: np.ndarray, contacts_xy_mm: np.ndarray,
                                   prefix_len: int, horizons: int,
                                   gate_rule: str = "M2-2RANK") -> dict:
    """Split v0.1's padded rank sets into an observed prefix and the future to predict.

    Built from ``build_motif_event_tensors`` rather than from the rank matrix directly,
    so the prefix this round conditions on and the sequences v0.1 scores are the same
    arrays.  Events shorter than ``prefix_len + 1`` have no future left to predict and
    are dropped; the count is returned rather than absorbed, because a silently
    shrinking denominator is how a coverage limit turns into an apparent result.
    """
    from src.topic5_dynamical_motif_rnn_v0_1 import build_motif_event_tensors

    tensors = build_motif_event_tensors(ranks, contacts_xy_mm, gate_rule=gate_rule)
    x, length = tensors["x"], tensors["length"]
    if int(prefix_len) < 1:
        raise ValueError("the prefix must contain at least the first rank set")
    if x.shape[1] < prefix_len + 1:
        raise ValueError(
            f"no event reaches {prefix_len + 1} rank sets; the longest has {x.shape[1]}")

    keep = (length >= prefix_len + 1).nonzero(as_tuple=True)[0]
    prefix = x[keep, :prefix_len]

    n_events, _, n_contacts = x.shape
    targets = torch.zeros(len(keep), horizons, n_contacts, dtype=x.dtype)
    available_steps = min(horizons, x.shape[1] - prefix_len)
    targets[:, :available_steps] = x[keep, prefix_len:prefix_len + available_steps]
    valid = (length[keep, None] > prefix_len + torch.arange(horizons)[None, :])
    return {
        "prefix": prefix,
        "targets": targets,
        "cardinality": targets.sum(-1).long(),
        "valid": valid,
        "event_index": keep.numpy(),
        "n_events_total": int(n_events),
        "n_events_kept": int(len(keep)),
        "n_events_too_short": int(n_events - len(keep)),
        "horizon_coverage": [int(v) for v in valid.sum(0)],
    }


def apply_warm_start(operator, parent_state: dict, added: tuple[str, ...],
                     theta_init: float) -> None:
    """Inherit the parent, then restore what this layer is supposed to introduce.

    ``MotifRNN`` builds all four motif parameters whatever its ``model_id``, so a
    parent that never used ``theta`` still carries ``theta = 0`` in its state, and
    ``load_warm_start`` copies it over the angle this start was constructed with.  The
    order matters: inheriting after the angle is set silently collapses several
    optimisation starts into one, with no error and a plausible-looking spread coming
    from whatever else differed between them.

    The other new parameters go to zero, which is what makes the child reproduce its
    parent exactly; the angle does not, because at zero anisotropy it is inert.
    """
    operator.load_warm_start(parent_state)
    with torch.no_grad():
        for name in added:
            parameter = getattr(operator, name, None)
            if not isinstance(parameter, torch.nn.Parameter):
                continue
            parameter.fill_(float(theta_init) if name == "theta" else 0.0)


def _principal_angle(points: np.ndarray) -> float:
    """Direction of largest spread, in ``[0, pi)``.

    An axis has no head or tail, so the angle is taken modulo pi: the corridor along
    theta and the corridor along theta + pi are the same corridor, and treating them as
    two would report a difference that is only a sign convention.
    """
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 2:
        raise ValueError("a principal axis needs at least two distinct points")
    centred = points - points.mean(axis=0)
    if not np.any(np.abs(centred) > 1e-12):
        raise ValueError("the points coincide; no axis is defined")
    _, _, right = np.linalg.svd(centred, full_matrices=False)
    return float(np.arctan2(right[0, 1], right[0, 0]) % np.pi)


def geometry_axis_angle(contacts_xy_mm: np.ndarray) -> float:
    """The long axis of the implanted contact cloud.

    The control the learned corridor has to beat.  A previous round found the trained
    axis sitting a median of 7.7 degrees from the implantation long axis, which makes
    "this patient has an axial corridor" and "this patient's electrodes lie along a
    line" the same statement unless the two are separated here.
    """
    return _principal_angle(np.asarray(contacts_xy_mm, dtype=float))


def shaft_axis_angle(contacts_xy_mm: np.ndarray, shafts) -> float:
    """The axis along which the electrode shafts themselves are arranged.

    Raises when a patient has fewer than two shafts, rather than silently returning the
    contact-cloud axis: for a single-shaft implant this control does not exist, and a
    fallback would report a comparison that was never run.
    """
    coordinates = np.asarray(contacts_xy_mm, dtype=float)
    labels = np.asarray(shafts)
    unique = sorted(set(labels.tolist()))
    if len(unique) < 2:
        raise ValueError(f"the shaft axis needs at least two shafts, found {len(unique)}")
    centroids = np.stack([coordinates[labels == name].mean(axis=0) for name in unique])
    return _principal_angle(centroids)


def rotated_axis_angles(base: float, offsets=(np.pi / 3.0, 2.0 * np.pi / 3.0)
                        ) -> tuple[float, ...]:
    """Frozen rotations of a reference axis.

    These are the "same implantation shape, wrong direction" arms: they carry exactly
    the geometry the reference axis has, so beating them is what separates a direction
    the propagation picked from the direction the electrodes happen to lie along.
    """
    return tuple(float((base + offset) % np.pi) for offset in offsets)
