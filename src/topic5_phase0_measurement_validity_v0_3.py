"""Topic 5.3 Phase 0 — teachers whose order content is known before anything is fit.

v0.2 ended with two "not established" results.  The direction one was traced to the
instrument: handed the true axis on synthetic data, the pipeline still could not beat
its own control.  The order one was never checked the same way, because v0.2 had no
permutation-invariant model on the free basis to check it against.

This module builds four generators on the *real* patient geometry and the real event
counts, each with a different, known amount of order information:

``T1_ORDER_BLIND``      next contact is a function of the recruited SET only.  Order
                        information is identically zero — the false-positive control.
``T2_SINGLE_DIRECTED``  events genuinely travel along a fixed axis, but the rule reads
                        only the *front* of the recruited set, and the front is a
                        function of the set.  Order information is again identically
                        zero.  This is the diagnostic teacher: it produces data that
                        looks like directed propagation while carrying no extra order
                        information, which is the reading that would explain the real
                        cohort's null without any instrument failure.
``T3_TWO_MODE``         the first two contacts fix a direction that then persists, and
                        events start near the middle of the cloud, so the same set is
                        reachable travelling either way and implies opposite futures.
                        Order information is large — the power teacher.
``T4_HIDDEN_RELAY``     same two-mode rule, but generated on a larger contact set of
                        which the student sees only part, so the direction has to be
                        inferred through an unobserved relay.

The point of T2 is that "no order information" and "the instrument is blind" are
different failures and the gate has to tell them apart.  A design in which every
teacher is detectable cannot do that.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from src.topic5_strict_history_data_v0_2 import (
    PatientInput,
    count_rank_sets,
    derive_recording_blocks,
)

TEACHER_KINDS = ("T1_ORDER_BLIND", "T2_SINGLE_DIRECTED", "T3_TWO_MODE", "T4_HIDDEN_RELAY")

# Teachers whose next-contact rule provably reads the recruited set and nothing else.
# For these the conditional mutual information between the order and the next contact
# is zero by construction, and the estimator in ``order_information`` must return it.
SET_ONLY_KINDS = ("T1_ORDER_BLIND", "T2_SINGLE_DIRECTED")

HIDDEN_FRACTION = 0.30  # share of generating contacts the T4 student never sees


def stable_seed(*parts) -> int:
    """A seed that is the same in every process.

    ``hash()`` on strings is randomised per interpreter, so seeding from it makes the
    teachers silently irreproducible: the same nominal spec draws different parameters
    in different workers and in different runs.
    """
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 32)


@dataclass(frozen=True)
class TeacherSpec:
    kind: str
    patient: str
    seed: int
    coupling_scale: float = 1.0
    direction_scale: float = 3.0
    temperature: float = 1.0


class TeacherField:
    """The generative rule, exposed as a pure function of the ordered prefix.

    Keeping ``logits`` a function of the order (rather than folding the state into the
    sampling loop) is what makes the ground-truth order information computable: the
    estimator can ask the same teacher what it would have done under a different
    order of the same set.
    """

    def __init__(self, spec: TeacherSpec, coords_xy: np.ndarray) -> None:
        if spec.kind not in TEACHER_KINDS:
            raise ValueError(f"unknown teacher {spec.kind}")
        self.spec = spec
        self.coords = np.asarray(coords_xy, dtype=float)
        self.n_contacts = self.coords.shape[0]
        rng = np.random.default_rng(stable_seed(spec.patient, spec.kind, spec.seed))

        centred = self.coords - self.coords.mean(axis=0)
        spread = np.linalg.svd(centred, full_matrices=False)[2]
        self.axis = spread[0]
        projection = centred @ self.axis
        span = max(projection.ptp(), 1e-9)
        self.projection = projection / span

        self.bias = rng.normal(0.0, 1.0, self.n_contacts)
        self.coupling = (spec.coupling_scale
                         * rng.normal(0.0, 1.0, (self.n_contacts, self.n_contacts))
                         / np.sqrt(self.n_contacts))
        distance = np.linalg.norm(self.coords[:, None, :] - self.coords[None, :, :], axis=-1)
        self.locality = np.exp(-distance / max(np.median(distance), 1e-9))

    # -- the rule ----------------------------------------------------------
    def logits(self, order: list[int] | np.ndarray) -> np.ndarray:
        """Next-contact logits given the prefix *in order*.

        For the set-only teachers every term below is a function of ``set(order)``;
        that is the property the ground-truth estimator and the unit tests check.
        """
        order = np.asarray(order, dtype=int)
        recruited = np.zeros(self.n_contacts, dtype=bool)
        recruited[order] = True
        value = self.bias + self.coupling[order].sum(axis=0)

        kind = self.spec.kind
        if kind == "T1_ORDER_BLIND":
            pass
        elif kind == "T2_SINGLE_DIRECTED":
            # the front is max projection over the recruited SET, so this whole term
            # is set-valued even though the events it generates do travel along the axis
            front = self.projection[recruited].max()
            value = value + self.spec.direction_scale * (
                self.locality[order[np.argmax(self.projection[order])]]
                * np.exp(-np.abs(self.projection - front - 0.06) / 0.06))
        elif kind in ("T3_TWO_MODE", "T4_HIDDEN_RELAY"):
            # direction comes from the first two contacts and then persists; given only
            # the set, both directions remain possible and imply opposite futures
            if order.size >= 2:
                step = self.projection[order[1]] - self.projection[order[0]]
                direction = 1.0 if step >= 0 else -1.0
            else:
                direction = 0.0
            current = self.projection[order[-1]]
            ahead = direction * (self.projection - current)
            value = value + self.spec.direction_scale * (
                self.locality[order[-1]] * np.exp(-np.abs(ahead - 0.06) / 0.06))
        value = value / max(self.spec.temperature, 1e-6)
        value[recruited] = -np.inf
        return value

    def probabilities(self, order: list[int] | np.ndarray) -> np.ndarray:
        value = self.logits(order)
        if not np.isfinite(value).any():
            return np.zeros(self.n_contacts)
        shifted = np.exp(value - value[np.isfinite(value)].max())
        shifted[~np.isfinite(value)] = 0.0
        total = shifted.sum()
        return shifted / total if total > 0 else shifted

    def log_path_probability(self, order: np.ndarray, start_weight: np.ndarray) -> float:
        """log P(this exact order | the teacher), used to weight candidate orders."""
        total = float(np.log(max(start_weight[order[0]], 1e-300)))
        for step in range(1, order.size):
            probability = self.probabilities(order[:step])[order[step]]
            total += float(np.log(max(probability, 1e-300)))
        return total


def start_weights(field: TeacherField) -> np.ndarray:
    """Where events begin.

    The two-mode teachers start near the middle of the cloud on purpose: an event that
    starts at one end reveals its direction from the set alone, which would hand the
    order information to a bag model and defeat the test.
    """
    if field.spec.kind in ("T3_TWO_MODE", "T4_HIDDEN_RELAY"):
        weight = np.exp(-((field.projection - np.median(field.projection)) ** 2) / 0.02)
    elif field.spec.kind == "T2_SINGLE_DIRECTED":
        weight = np.exp(-4.0 * (field.projection - field.projection.min()))
    else:
        weight = np.ones(field.n_contacts)
    return weight / weight.sum()


def generate_ranks(field: TeacherField, lengths: np.ndarray, seed: int) -> np.ndarray:
    """One rank row per event, singleton steps (26/28 real SEEG patients are singleton)."""
    rng = np.random.default_rng(seed)
    weight = start_weights(field)
    ranks = np.full((lengths.size, field.n_contacts), -1, dtype=np.int16)
    for event, length in enumerate(lengths):
        order = [int(rng.choice(field.n_contacts, p=weight))]
        for _ in range(int(length) - 1):
            probability = field.probabilities(order)
            if probability.sum() <= 0:
                break
            order.append(int(rng.choice(field.n_contacts, p=probability)))
        ranks[event, np.asarray(order)] = np.arange(len(order), dtype=np.int16)
    return ranks


def order_information(field: TeacherField, ranks: np.ndarray, *, prefix_len: int = 3,
                      n_orders: int = 48, max_events: int = 400,
                      seed: int = 0) -> dict:
    """Ground truth: how much does the order tell you about the next contact, given the set?

    This is I(order ; next contact | set) under the teacher, estimated as

        E_events[ KL( p(.|S, true order) || sum_pi w_pi p(.|S, pi) ) ],  w_pi ∝ P(pi | S),

    with the candidate orders drawn uniformly over permutations of the prefix and
    re-weighted by the probability the teacher would have produced them.

    The re-weighting is the whole estimator.  Weighting the permutations equally would
    charge a monotone teacher for orders it can never generate and report a large
    number for a rule that reads only the set — exactly the wrong answer for T2.
    """
    rng = np.random.default_rng(seed)
    weight = start_weights(field)
    divergences: list[float] = []
    rows = np.flatnonzero((ranks >= 0).sum(axis=1) > prefix_len)
    if rows.size > max_events:
        rows = rng.choice(rows, size=max_events, replace=False)
    for row in rows:
        present = np.flatnonzero(ranks[row] >= 0)
        order = present[np.argsort(ranks[row][present])][:prefix_len]
        candidates = [order] + [rng.permutation(order) for _ in range(n_orders)]
        log_weight = np.array(
            [field.log_path_probability(np.asarray(c), weight) for c in candidates])
        log_weight -= log_weight.max()
        importance = np.exp(log_weight)
        if importance.sum() <= 0:
            continue
        importance /= importance.sum()
        stack = np.stack([field.probabilities(c) for c in candidates])
        marginal = importance @ stack
        truth = stack[0]
        support = (truth > 0) & (marginal > 0)
        if not support.any():
            continue
        divergences.append(float((truth[support]
                                  * np.log(truth[support] / marginal[support])).sum()))
    if not divergences:
        return {"order_information_n_events_scored": 0,
                "order_information_nats": float("nan")}
    return {
        "order_information_n_events_scored": len(divergences),
        "order_information_nats": float(np.mean(divergences)),
        "order_information_median": float(np.median(divergences)),
        "order_information_p90": float(np.percentile(divergences, 90)),
    }


def synthesise_patient(patient: PatientInput, kind: str, seed: int) -> tuple[PatientInput, dict]:
    """Replace only *who fires when*, keeping this patient's geometry and event count."""
    spec = TeacherSpec(kind=kind, patient=patient.patient, seed=seed)
    coords = np.asarray(patient.contacts_xy_mm, dtype=float)
    generating = coords
    hidden: np.ndarray = np.zeros(0, dtype=int)
    if kind == "T4_HIDDEN_RELAY":
        # the relay lives among the real contacts; the student simply never sees it
        rng = np.random.default_rng(stable_seed(patient.patient, "hidden", seed))
        n_hidden = max(1, int(round(HIDDEN_FRACTION * coords.shape[0])))
        hidden = np.sort(rng.choice(coords.shape[0], size=n_hidden, replace=False))

    field = TeacherField(spec, generating)
    real_lengths = count_rank_sets(patient.ranks)
    real_lengths = real_lengths[real_lengths > 0]
    if real_lengths.size == 0:
        raise ValueError(f"{patient.patient} has no usable events to match")
    rng = np.random.default_rng(seed + 17)
    lengths = rng.choice(real_lengths, size=patient.n_events, replace=True)
    lengths = np.clip(lengths, 2, max(2, generating.shape[0] - 1))

    ranks_full = generate_ranks(field, lengths, seed=seed + 101)
    truth = order_information(field, ranks_full, seed=seed + 7)

    visible = np.setdiff1d(np.arange(generating.shape[0]), hidden)
    ranks = np.full((patient.n_events, visible.size), -1, dtype=np.int16)
    for event in range(patient.n_events):
        row = ranks_full[event, visible]
        present = np.flatnonzero(row >= 0)
        if present.size:
            ranks[event, present[np.argsort(row[present])]] = np.arange(present.size)

    synthetic = PatientInput(
        dataset=patient.dataset,
        patient=f"{patient.patient}__{kind}",
        contact_names=[patient.contact_names[i] for i in visible],
        shafts=[patient.shafts[i] for i in visible],
        coords_3d_mm=np.asarray(patient.coords_3d_mm, dtype=float)[visible],
        contacts_xy_mm=coords[visible],
        ranks=ranks,
        split=np.asarray(patient.split).copy(),
        event_abs_time=np.asarray(patient.event_abs_time, dtype=float).copy(),
        event_lag_raw=np.asarray(patient.event_lag_raw, dtype=float)[:, visible],
        recording_block=derive_recording_blocks(np.asarray(patient.event_abs_time, dtype=float)),
        provenance={
            "phase0_teacher": kind,
            "phase0_seed": int(seed),
            "source_patient": patient.patient,
            "n_generating_contacts": int(generating.shape[0]),
            "n_hidden_contacts": int(hidden.size),
            "ground_truth_order_information": truth,
        },
    )
    return synthetic, {
        "patient": patient.patient, "teacher": kind, "seed": int(seed),
        "n_events": int(patient.n_events),
        "n_generating_contacts": int(generating.shape[0]),
        "n_visible_contacts": int(visible.size),
        "median_event_length": float(np.median(lengths)),
        **truth,
    }
