"""Score the existing motif family on *when*, not only on *who is next*.

v0.1 compared four propagation motifs on the next-contact distribution and found
essentially nothing: isotropic local diffusion was already the best operator.  Its
own closeout flagged the reason it could not settle the question — every model in
that round used the within-event *rank index* as its only notion of time, while the
data also carries a within-event time proxy that no model ever saw.

That proxy carries distance information the rank order does not: after controlling
for the rank gap, the spatial distance between two contacts still predicts their
time gap in 27 of 28 patients (median partial Spearman +0.125, sign test p=2.2e-7).

So this module adds one thing to the frozen v0.1 family: a two-parameter read-out
of the *same* motif-propagated field that predicts how long the next step took.  The
head is deliberately tiny so that any difference between motifs comes from the
operator rather than from the read-out.

The proxy is the within-event spectral-centroid position.  It is NOT clinical
recruitment time and NOT axonal conduction delay, so nothing here may be converted
into a conduction velocity — that lock is inherited from v0.1 and is not reopened.

What the target does and does not add
-------------------------------------
The rank IS the argsort of this proxy: sorting a patient's contacts by rank and by
proxy gives the same order in 99.7-100% of events, and no patient has a single
negative increment.  So the time target contributes *gap magnitudes* and nothing
about the ordering, which is what makes it a clean question rather than a circular
one — the model is never asked to predict an order it can already see.  It also
fixes the reference: the step index is exactly the ordering information, so the
rank-only baseline is the right thing to score against.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_dynamical_motif_rnn_v0_1 import MotifConfig, MotifRNN, build_motif_event_tensors

# The time head owns two scalars: an offset and a slope on the propagated field.
# Anything richer would let the head compensate for a wrong operator.  The Gaussian
# variance is profiled out analytically rather than learned: fitting it jointly at a
# useful learning rate drove the variance to its floor and then blew the squared term
# up, which would have corrupted the very scores this round rests on.
TIME_HEAD_PARAMETERS = ("time_offset", "time_slope")


def build_time_targets(ranks: np.ndarray, event_lag_raw: np.ndarray) -> dict[str, Tensor]:
    """Per-step time increment of the contact recruited next, teacher-forced.

    ``delta`` at step ``t`` is the lag of the rank-``t+1`` set minus the lag of the
    rank-``t`` set, i.e. how long the event took to make that one step.  Steps whose
    successor does not exist, or whose lag is not finite, are masked out rather than
    imputed: an imputed zero would be indistinguishable from a genuinely fast step.
    """
    ranks = np.asarray(ranks)
    lag = np.asarray(event_lag_raw, dtype=np.float64)
    if lag.shape != ranks.shape:
        raise ValueError("event_lag_raw must align with the rank matrix")
    n_events, _ = ranks.shape
    lengths = np.array([int(row[row >= 0].max()) + 1 if np.any(row >= 0) else 0
                        for row in ranks])
    steps = int(lengths.max())

    delta = np.zeros((n_events, steps), np.float32)
    valid = np.zeros((n_events, steps), bool)
    for event, row in enumerate(ranks):
        length = lengths[event]
        if length < 2:
            continue
        means = np.full(length, np.nan)
        for rank in range(length):
            members = row == rank
            if members.any():
                values = lag[event, members]
                if np.isfinite(values).any():
                    means[rank] = np.nanmean(values[np.isfinite(values)])
        for step in range(length - 1):
            gap = means[step + 1] - means[step]
            if np.isfinite(gap):
                delta[event, step] = gap
                valid[event, step] = True
    return {"time_delta": torch.from_numpy(delta), "time_valid": torch.from_numpy(valid)}


def build_event_tensors_with_time(ranks: np.ndarray, contacts_xy_mm: np.ndarray,
                                  event_lag_raw: np.ndarray,
                                  gate_rule: str = "M2-2RANK") -> dict[str, Tensor]:
    """v0.1's tensors plus the time target, so the two stay index-aligned."""
    tensors = build_motif_event_tensors(ranks, contacts_xy_mm, gate_rule=gate_rule)
    tensors.update(build_time_targets(ranks, event_lag_raw))
    return tensors


class TimeHead(nn.Module):
    """How long the next step took, read off the motif-propagated field.

    ``field`` is the state the operator produced at the contact that was actually
    recruited next.  A motif that transports correctly should make that value order
    the steps by how far the event had to travel, which is exactly the relation the
    rank index cannot express.
    """

    def __init__(self) -> None:
        super().__init__()
        self.time_offset = nn.Parameter(torch.zeros(()))
        self.time_slope = nn.Parameter(torch.zeros(()))

    def predict(self, field: Tensor) -> Tensor:
        return self.time_offset - self.time_slope * field

    def nll(self, field: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        """Mean squared error against the prepared target.

        ``target`` must already be on the scale the caller wants to score, because
        transforming it here as well silently squashes it: the caller standardises
        ``log1p`` of the increment, and applying ``log1p`` a second time clamps every
        negative standardised value to zero and collapses most of the target.

        This is the Gaussian negative log likelihood with the variance profiled out,
        up to a constant and a monotone transform, so it ranks arms identically while
        being impossible to diverge.
        """
        if mask.sum() == 0:
            return field.sum() * 0.0
        return ((target[mask] - self.predict(field)[mask]) ** 2).mean()


class RankOnlyTimeBaseline(nn.Module):
    """Time from the step index alone.

    The runner now uses the closed-form ``time_baseline_scores`` instead, which also
    supplies the two richer references the motifs actually have to beat.  This version
    is kept because the regression tests below are written against it: it is where the
    double-transform bug was caught, and a closed-form solver would have hidden it.

    The 27/28 result is a *partial* relation — distance predicts the time gap after
    the rank gap is accounted for.  Scoring the motifs against zero would therefore
    credit them for the part the step index already explains, so the comparison is
    made against a baseline that has the step index and nothing else.
    """

    def __init__(self, max_steps: int) -> None:
        super().__init__()
        self.per_step = nn.Parameter(torch.zeros(max_steps))

    def nll(self, target: Tensor, mask: Tensor) -> Tensor:
        """Same score as the motif head, on the same prepared target.

        As above, the target is used as supplied; the caller owns the transform so the
        two scores stay directly subtractable.
        """
        if mask.sum() == 0:
            return self.per_step.sum() * 0.0
        steps = torch.arange(target.shape[1], device=target.device)
        predicted = self.per_step[steps.clamp(max=self.per_step.shape[0] - 1)]
        predicted = predicted.unsqueeze(0).expand_as(target)[mask]
        return ((target[mask] - predicted) ** 2).mean()


def recruited_field(states: Tensor, target: Tensor, observation: Tensor) -> Tensor:
    """The propagated field evaluated at the contact recruited next.

    ``states`` is (B, steps, n_nodes); ``target`` is the (B, steps, n_contacts)
    one-hot of what fired next.  The observation operator maps nodes to contacts, so
    the field is compared with the data on the contacts, never on the nodes.
    """
    contact_field = states @ observation.T                     # (B, steps, n_contacts)
    weight = target.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return (contact_field * target).sum(dim=-1) / weight.squeeze(-1)


def direction_persistence(ranks: np.ndarray, contacts_xy_mm: np.ndarray) -> float:
    """Mean cosine between consecutive centroid steps — does the event keep going?

    v0.1 found real events carry more of this than isotropic diffusion generates, so
    it is reported per arm as a secondary target rather than folded into a score.
    """
    ranks = np.asarray(ranks)
    xy = np.asarray(contacts_xy_mm, dtype=float)
    cosines: list[float] = []
    for row in ranks:
        present = row[row >= 0]
        if present.size < 3:
            continue
        length = int(present.max()) + 1
        centroids = []
        for rank in range(length):
            members = row == rank
            if members.any():
                centroids.append(xy[members].mean(axis=0))
        if len(centroids) < 3:
            continue
        steps = np.diff(np.asarray(centroids), axis=0)
        norms = np.linalg.norm(steps, axis=1)
        usable = norms > 1e-9
        if usable.sum() < 2:
            continue
        unit = steps[usable] / norms[usable, None]
        cosines.extend((unit[:-1] * unit[1:]).sum(axis=1).tolist())
    return float(np.mean(cosines)) if cosines else float("nan")


def distance_time_relation(ranks: np.ndarray, coords_mm: np.ndarray,
                           event_lag_raw: np.ndarray, max_events: int = 1500,
                           seed: int = 0) -> dict:
    """The clue this round is built around, as a reusable statistic.

    Within each event, every pair of participating contacts contributes a rank gap,
    a spatial distance and a time gap.  The distance-time association is measured
    *within* rank-gap strata, so the part the step index already explains is removed
    before anything is attributed to geometry.
    """
    from scipy import stats

    ranks = np.asarray(ranks)
    lag = np.asarray(event_lag_raw, dtype=float)
    coords = np.asarray(coords_mm, dtype=float)
    distance = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    rows = np.flatnonzero((ranks >= 0).sum(axis=1) >= 3)
    if rows.size > max_events:
        rows = np.random.default_rng(seed).choice(rows, max_events, replace=False)

    gaps, times, spans = [], [], []
    for event in rows:
        present = np.flatnonzero(ranks[event] >= 0)
        if present.size < 3:
            continue
        left, right = np.triu_indices(present.size, 1)
        a, b = present[left], present[right]
        gaps.append(np.abs(ranks[event][a] - ranks[event][b]))
        times.append(np.abs(lag[event][a] - lag[event][b]))
        spans.append(distance[a, b])
    if not gaps:
        return {"n_pairs": 0, "partial_spearman": float("nan")}
    gaps = np.concatenate(gaps)
    times = np.concatenate(times)
    spans = np.concatenate(spans)
    keep = np.isfinite(times) & np.isfinite(spans) & (times > 0)
    gaps, times, spans = gaps[keep], times[keep], spans[keep]
    if gaps.size < 200:
        return {"n_pairs": int(gaps.size), "partial_spearman": float("nan")}

    ranked_time = np.full(times.shape, np.nan)
    ranked_span = np.full(spans.shape, np.nan)
    for gap in np.unique(gaps):
        inside = gaps == gap
        if inside.sum() < 10:
            continue
        ranked_time[inside] = stats.rankdata(times[inside]) / inside.sum()
        ranked_span[inside] = stats.rankdata(spans[inside]) / inside.sum()
    usable = np.isfinite(ranked_time) & np.isfinite(ranked_span)
    if usable.sum() < 200:
        return {"n_pairs": int(usable.sum()), "partial_spearman": float("nan")}
    rho, pvalue = stats.spearmanr(ranked_span[usable], ranked_time[usable])
    return {"n_pairs": int(usable.sum()), "partial_spearman": float(rho),
            "p_value": float(pvalue)}


def rollout_states(model, x: Tensor, recruited: Tensor, displacement: Tensor
                   ) -> tuple[Tensor, Tensor]:
    """v0.1's teacher-forced pass, but keeping the propagated field at every step.

    ``MotifRNN.forward`` returns only the logits and the stop logits, and the time
    read-out needs the field itself.  The loop below calls the same public operator
    methods in the same order rather than re-deriving anything, so the states are the
    ones that model actually used.
    """
    batch, steps, _ = x.shape
    h = torch.zeros(batch, model.n_nodes, device=x.device, dtype=x.dtype)
    terms = model.recurrent_terms()
    unit, _ = model.axis_unit()
    gate = model.direction_gate(displacement, unit)
    weight = (model.direction_weight(displacement, unit)
              if model.config.direction_mode != "GLOBAL_AXIS" else None)
    states, logits = [], []
    for step in range(steps):
        h = model.step(h, x[:, step], gate[:, step], terms,
                       None if weight is None else weight[:, step])
        states.append(h)
        logits.append(model.readout(h))
    return torch.stack(states, 1), torch.stack(logits, 1)


class FreeLowRankDrive(MotifRNN):
    """The motifs' own cell with the structured operator replaced by a free one.

    An earlier version wrote a separate linear recurrence.  That confounded the
    question: the motifs run a leaky *tanh* cell with an input gain, a node bias and a
    contact read-out, so a free linear cell winning or losing mixed "the operator is
    unconstrained" with "the dynamics are linear".  Subclassing keeps the leak, the
    saturation, both biases, the gains and the observation operator identical and
    changes exactly one thing — the recurrent drive.

    This is a predictive upper bound, not a capacity-matched control: matching the
    parameter count to the digit would put the two differences back together.
    """

    def __init__(self, config: "MotifConfig", rank: int, seed: int = 0) -> None:
        super().__init__(config)
        if int(rank) < 1:
            raise ValueError("rank must be positive")
        torch.manual_seed(int(seed))
        scale = 1.0 / max(1.0, float(self.n_nodes) ** 0.5)
        self.free_left = nn.Parameter(scale * torch.randn(self.n_nodes, int(rank)))
        self.free_right = nn.Parameter(scale * torch.randn(self.n_nodes, int(rank)))

    def recurrent_drive(self, h: Tensor, s: Tensor, terms: dict,
                        weight: Tensor | None = None) -> Tensor:
        return (h @ self.free_left) @ self.free_right.transpose(0, 1)


TIME_BASELINES = ("STEP_ONLY", "STEP_DISTANCE", "STATIC_TARGET")


def time_baseline_scores(target: np.ndarray, mask: np.ndarray, distance: np.ndarray,
                         next_contact: np.ndarray, train: np.ndarray,
                         test: np.ndarray, ridge: float = 0.0) -> dict[str, float]:
    """Least-squares time baselines, each knowing strictly more than the last.

    The motif arms read their field *at the contact that actually fired next*, so they
    are told the destination.  A baseline that knows only the step index is therefore
    not a fair reference: the gain it concedes can come from the destination's distance
    rather than from any dynamics.  These three close that gap:

    ``STEP_ONLY``      how far into the event we are;
    ``STEP_DISTANCE``  and how far the event had to travel to reach the true next set;
    ``STATIC_TARGET``  and whether that particular contact is habitually early or late.

    Solved in closed form on the training split and scored once on the test split, so
    they add no optimisation choices of their own.
    """
    n_events, n_steps = target.shape
    n_contacts = next_contact.shape[2]

    def design(rows: np.ndarray, level: str) -> tuple[np.ndarray, np.ndarray]:
        blocks, values = [], []
        for step in range(n_steps):
            keep = mask[:, step] & rows
            if not keep.any():
                continue
            count = int(keep.sum())
            columns = [np.eye(n_steps)[step][None, :].repeat(count, axis=0)]
            if level in ("STEP_DISTANCE", "STATIC_TARGET"):
                columns.append(distance[keep, step][:, None])
            if level == "STATIC_TARGET":
                columns.append(next_contact[keep, step])
            blocks.append(np.hstack(columns))
            values.append(target[keep, step])
        if not blocks:
            return np.zeros((0, 1)), np.zeros(0)
        return np.vstack(blocks), np.concatenate(values)

    scores: dict[str, float] = {}
    for level in TIME_BASELINES:
        matrix, response = design(train, level)
        held, truth = design(test, level)
        if matrix.shape[0] < matrix.shape[1] + 5 or held.shape[0] == 0:
            scores[level] = float("nan")
            continue
        if ridge > 0:
            # STATIC_TARGET adds one column per contact, which an unregularised solve
            # can overfit on a small patient; the penalty is chosen on validation, never
            # on the split the score is read from
            gram = matrix.T @ matrix + ridge * np.eye(matrix.shape[1])
            coefficients = np.linalg.solve(gram, matrix.T @ response)
        else:
            coefficients, *_ = np.linalg.lstsq(matrix, response, rcond=None)
        scores[level] = float(((truth - held @ coefficients) ** 2).mean())
    return scores


def adjacent_distance_time_relation(tensors: dict) -> dict:
    """The clue restated on exactly the quantity the model predicts.

    ``distance_time_relation`` uses every within-event contact pair, while the model
    predicts the gap between *consecutive* rank sets.  Comparing a per-patient model
    gain against the all-pairs number would relate two different estimands, so this
    computes the association on the adjacent steps, controlling for the step index the
    baseline already has.
    """
    from scipy import stats

    centroid = tensors["centroid"].numpy()
    delta = tensors["time_delta"].numpy()
    mask = tensors["time_valid"].numpy()
    distance = np.zeros_like(delta)
    distance[:, :-1] = np.linalg.norm(centroid[:, 1:] - centroid[:, :-1], axis=-1)

    ranked_time = np.full(delta.shape, np.nan)
    ranked_span = np.full(delta.shape, np.nan)
    for step in range(delta.shape[1]):
        keep = mask[:, step]
        if keep.sum() < 10:
            continue
        ranked_time[keep, step] = stats.rankdata(delta[keep, step])
        ranked_span[keep, step] = stats.rankdata(distance[keep, step])
    usable = np.isfinite(ranked_time) & np.isfinite(ranked_span)
    if usable.sum() < 200:
        return {"n_steps": int(usable.sum()), "adjacent_partial_spearman": float("nan")}
    rho, pvalue = stats.spearmanr(ranked_span[usable], ranked_time[usable])
    return {"n_steps": int(usable.sum()), "adjacent_partial_spearman": float(rho),
            "p_value": float(pvalue)}


def free_rank_for_budget(n_nodes: int, budget: int) -> int:
    """Largest free rank whose parameter count stays inside the motif budget."""
    for rank in range(int(n_nodes), 0, -1):
        if 2 * int(n_nodes) * rank + 3 <= int(budget):
            return int(rank)
    return 1
