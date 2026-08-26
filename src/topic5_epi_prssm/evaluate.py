"""Held-out evaluation: filtered scores, open-loop horizons, state reset, delta-t shuffle.

Every routine here is patient-first: it returns one number per patient per
endpoint, and cohort statistics are formed later from those patient values.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .contracts import FROZEN
from .model import CohortBatch, EpiPRSSM, SlowState
from .rollout import ENDPOINTS, carry_state, cohort_scan, score_scan, score_window, \
    shuffled_delta_t_batch


@dataclass
class Evaluation:
    filtered: dict[str, dict[str, float]]
    open_loop: dict[str, dict[int, float]]
    open_loop_order: dict[str, dict[int, float]]
    state_reset: dict[str, dict[int, float]]
    delta_t_shuffle: dict[str, dict[str, float]]
    per_event: dict[str, dict[str, np.ndarray]]
    n_open_loop_anchors: dict[str, int]


def _means(scores: dict[str, dict[str, np.ndarray]]) -> dict[str, dict[str, float]]:
    return {s: {k: float(np.mean(v)) for k, v in d.items() if k in ENDPOINTS}
            for s, d in scores.items()}


@torch.no_grad()
def evaluate(model: EpiPRSSM, warm_batch: CohortBatch, eval_batch: CohortBatch,
             *, expected_load: torch.Tensor, horizons=None, reset_horizons=None,
             chunk: int = 256, reset_stride: int = 4, seed: int = 0,
             with_reset: bool = True, with_shuffle: bool = True,
             with_open_loop: bool = True) -> Evaluation:
    horizons = tuple(horizons or FROZEN["open_loop_horizons"])
    reset_horizons = tuple(reset_horizons or FROZEN["state_reset_horizons"])
    model.eval()

    z_warm = carry_state(model, warm_batch, model.initial_state(warm_batch), chunk=chunk)
    per_event, _ = score_window(model, eval_batch, z_warm, chunk=chunk)
    filtered = _means(per_event)

    open_loop, open_loop_order, anchors = ({}, {}, {})
    if with_open_loop:
        open_loop, open_loop_order, anchors = _open_loop(
            model, eval_batch, z_warm, horizons, expected_load, chunk=chunk)

    reset: dict[str, dict[int, float]] = {}
    if with_reset:
        reset = _state_reset(model, eval_batch, z_warm, reset_horizons, stride=reset_stride)

    shuffle: dict[str, dict[str, float]] = {}
    if with_shuffle:
        rng = np.random.default_rng(seed)
        shuffled_eval = shuffled_delta_t_batch(eval_batch, rng)
        shuffled_warm = shuffled_delta_t_batch(warm_batch, rng)
        z_shuffled = carry_state(model, shuffled_warm, model.initial_state(shuffled_warm), chunk=chunk)
        shuffled_scores, _ = score_window(model, shuffled_eval, z_shuffled, chunk=chunk)
        shuffle = _means(shuffled_scores)

    return Evaluation(filtered, open_loop, open_loop_order, reset, shuffle, per_event, anchors)


@torch.no_grad()
def _open_loop(model: EpiPRSSM, batch: CohortBatch, z0: SlowState, horizons,
               expected_load: torch.Tensor, *, chunk: int = 256):
    """Segmented observer-off rollout.

    The stream is cut into consecutive blocks of ``max(horizons)`` events.  At the
    top of each block the observer is closed and the generator integrates on real
    elapsed time alone; the block is then replayed with the observer open so the
    next anchor starts from a properly filtered state.
    """
    horizon_max = max(horizons)
    totals = {h: {p.subject: [] for p in batch.patients} for h in horizons}
    totals_order = {h: {p.subject: [] for p in batch.patients} for h in horizons}
    anchors = {p.subject: 0 for p in batch.patients}
    z = z0
    position = 0
    total = batch.max_length
    while position + horizon_max <= total:
        end = position + horizon_max
        rolled = cohort_scan(model, batch, position, end, z, correction_on=False,
                             expected_load=expected_load)
        scores = score_scan(model, batch, rolled, position)
        for subject, values in scores.items():
            anchors[subject] += 1
            for h in horizons:
                take = min(h, len(values["event_nll"]))
                if take > 0:
                    totals[h][subject].append(float(values["event_nll"][:take].mean()))
                    totals_order[h][subject].append(float(values["order_nll"][:take].mean()))
        z = cohort_scan(model, batch, position, end, z, correction_on=True, collect=False).final
        position = end
    out = {s: {h: float(np.mean(totals[h][s])) if totals[h][s] else float("nan")
               for h in horizons} for s in anchors}
    out_order = {s: {h: float(np.mean(totals_order[h][s])) if totals_order[h][s] else float("nan")
                     for h in horizons} for s in anchors}
    return out, out_order, anchors


@torch.no_grad()
def _state_reset(model: EpiPRSSM, batch: CohortBatch, z0: SlowState, horizons,
                 *, stride: int = 4) -> dict[str, dict[int, float]]:
    """How much worse is a wiped state than an intact one, k events after the wipe?

    A state that is only a smoothed echo of recent events recovers within a few
    events; a state with real memory does not.
    """
    horizon_max = max(horizons)
    deltas = {h: {p.subject: [] for p in batch.patients} for h in horizons}
    z = z0
    position = 0
    total = batch.max_length
    block = 0
    while position + horizon_max <= total:
        end = position + horizon_max
        if block % stride == 0:
            wiped = SlowState(torch.zeros_like(z.state), torch.ones_like(z.resource),
                              torch.zeros_like(z.observer_state), torch.zeros_like(z.exposure))
            intact_run = cohort_scan(model, batch, position, end, z, correction_on=True)
            reset_run = cohort_scan(model, batch, position, end, wiped, correction_on=True)
            intact = score_scan(model, batch, intact_run, position)
            reset = score_scan(model, batch, reset_run, position)
            for subject in intact:
                for h in horizons:
                    k = min(h, len(intact[subject]["event_nll"])) - 1
                    if k >= 0:
                        deltas[h][subject].append(
                            float(reset[subject]["event_nll"][k] - intact[subject]["event_nll"][k]))
            z = intact_run.final
        else:
            z = cohort_scan(model, batch, position, end, z, correction_on=True, collect=False).final
        position = end
        block += 1
    return {s: {h: float(np.mean(deltas[h][s])) if deltas[h][s] else float("nan")
                for h in horizons} for s in deltas[horizons[0]]}


# --------------------------------------------------------------------------
# H2a: state-conditioned readout
# --------------------------------------------------------------------------

@torch.no_grad()
def collect_states(model: EpiPRSSM, warm_batch: CohortBatch, eval_batch: CohortBatch,
                   *, chunk: int = 256) -> dict[str, dict[str, torch.Tensor]]:
    """Filtered pre-event states for every evaluation event, per patient."""
    z = carry_state(model, warm_batch, model.initial_state(warm_batch), chunk=chunk)
    store: dict[str, dict[str, list[torch.Tensor]]] = {}
    position, total = 0, eval_batch.max_length
    while position < total:
        end = min(position + chunk, total)
        result = cohort_scan(model, eval_batch, position, end, z, correction_on=True)
        for p, patient in enumerate(eval_batch.patients):
            take = int(min(end - position, max(int(eval_batch.lengths[p]) - position, 0)))
            if take <= 0:
                continue
            slot = store.setdefault(patient.subject, {"state": [], "resource": [], "index": []})
            slot["state"].append(result.state_minus[:take, p, : patient.n_contacts, :].clone())
            slot["resource"].append(result.resource_minus[:take, p].clone())
            slot["index"].append(torch.arange(eval_batch.starts[p] + position,
                                              eval_batch.starts[p] + position + take))
        z = result.final
        position = end
    return {s: {k: torch.cat(v) for k, v in d.items()} for s, d in store.items()}


@torch.no_grad()
def state_swap_effects(model: EpiPRSSM, eval_batch: CohortBatch,
                       states: dict[str, dict[str, torch.Tensor]], *, seed: int,
                       n_bins: int = 8, chunk: int = 512) -> dict[str, dict[str, float]]:
    """Correct state versus a patient-internal swapped state.

    ``random`` permutes the event-to-state assignment inside the patient.
    ``matched`` permutes only inside bins of equal state magnitude, so the swap
    keeps the latent gauge and destroys only the temporal alignment.
    """
    rng = np.random.default_rng(seed)
    lookup = {p.subject: p for p in eval_batch.patients}
    out: dict[str, dict[str, float]] = {}
    for subject, payload in states.items():
        patient = lookup[subject]
        state, resource, index = payload["state"], payload["resource"], payload["index"]
        n = len(index)
        if n < 32:
            continue
        norm = state.reshape(n, -1).norm(dim=1).cpu().numpy()
        random_perm = rng.permutation(n)
        matched_perm = np.arange(n)
        ranks = np.argsort(np.argsort(norm))
        bins = np.minimum((ranks * n_bins) // max(n, 1), n_bins - 1)
        for b in range(n_bins):
            members = np.flatnonzero(bins == b)
            if len(members) > 1:
                matched_perm[members] = members[rng.permutation(len(members))]
        scores = {"correct": _score_chunks(model, patient, index, state, resource, chunk),
                  "swap_random": _score_chunks(model, patient, index, state[random_perm],
                                               resource[random_perm], chunk),
                  "swap_matched": _score_chunks(model, patient, index, state[matched_perm],
                                                resource[matched_perm], chunk)}
        row: dict[str, float] = {}
        for name, values in scores.items():
            for endpoint, array in values.items():
                row[f"{name}__{endpoint}"] = float(np.mean(array))
        row["n_events"] = float(n)
        row["participation_residualised_order"] = _residualise(
            scores["correct"]["order_nll"], scores["correct"]["n_participants"])
        row["participation_residualised_order_swap_matched"] = _residualise(
            scores["swap_matched"]["order_nll"], scores["swap_matched"]["n_participants"])
        out[subject] = row
    return out


def _score_chunks(model: EpiPRSSM, patient, index: torch.Tensor, state: torch.Tensor,
                  resource: torch.Tensor, chunk: int) -> dict[str, np.ndarray]:
    pieces: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(index), chunk):
        stop = min(start + chunk, len(index))
        scores = model.score_events(patient, index[start:stop], state[start:stop],
                                    resource[start:stop])
        for key in ENDPOINTS:
            pieces.setdefault(key, []).append(scores[key].cpu().numpy())
        pieces.setdefault("n_participants", []).append(
            patient.participation[index[start:stop]].sum(-1).cpu().numpy().astype(float))
    return {k: np.concatenate(v) for k, v in pieces.items()}


def _residualise(values: np.ndarray, covariate: np.ndarray) -> float:
    """Mean of ``values`` after removing a linear dependence on ``covariate``."""
    design = np.stack([np.ones_like(covariate), covariate], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return float(np.mean(values - design @ coefficients) + coefficients[0])


@torch.no_grad()
def ambiguous_prefix_effects(model: EpiPRSSM, eval_batch: CohortBatch,
                             states: dict[str, dict[str, torch.Tensor]],
                             families: dict[str, dict[int, set]], *, seed: int,
                             chunk: int = 512) -> dict[str, dict[str, float]]:
    """Suffix branch log-probability at an ambiguous prefix, correct versus swapped state.

    Only patients whose train inventory supports a branching prefix family enter
    here; the rest are ``not_eligible_for_targeted_analysis``, which is not a
    negative result.
    """
    rng = np.random.default_rng(seed + 7919)
    lookup = {p.subject: p for p in eval_batch.patients}
    out: dict[str, dict[str, float]] = {}
    for subject, payload in states.items():
        depths = families.get(subject, {})
        if not depths:
            continue
        patient = lookup[subject]
        state, resource, index = payload["state"], payload["resource"], payload["index"]
        n = len(index)
        perm = rng.permutation(n)
        row: dict[str, float] = {}
        for depth, prefixes in depths.items():
            member = _prefix_membership(patient, index, depth, prefixes)
            if member.sum() < 30:
                continue
            selected = torch.as_tensor(np.flatnonzero(member))
            correct = _step_logprob(model, patient, index[selected], state[selected],
                                    resource[selected], depth, chunk)
            swapped = _step_logprob(model, patient, index[selected], state[perm][selected],
                                    resource[perm][selected], depth, chunk)
            row[f"depth{depth}_n_events"] = float(len(selected))
            row[f"depth{depth}_suffix_nll_correct"] = float(-np.mean(correct))
            row[f"depth{depth}_suffix_nll_swapped"] = float(-np.mean(swapped))
            row[f"depth{depth}_suffix_state_gain"] = float(np.mean(correct) - np.mean(swapped))
        if row:
            out[subject] = row
    return out


def _prefix_membership(patient, index: torch.Tensor, depth: int, prefixes: set) -> np.ndarray:
    participation = patient.participation[index].cpu().numpy()
    group_ids = patient.group_ids[index].cpu().numpy()
    n_groups = patient.n_groups[index].cpu().numpy()
    member = np.zeros(len(index), dtype=bool)
    for row in range(len(index)):
        if n_groups[row] <= depth:
            continue
        mask = participation[row]
        gid = group_ids[row]
        prefix = tuple(sorted(int(c) for c in np.flatnonzero(mask & (gid < depth) & (gid >= 0))))
        if prefix in prefixes:
            member[row] = True
    return member


def _step_logprob(model: EpiPRSSM, patient, index: torch.Tensor, state: torch.Tensor,
                  resource: torch.Tensor, depth: int, chunk: int) -> np.ndarray:
    pieces = []
    for start in range(0, len(index), chunk):
        stop = min(start + chunk, len(index))
        scores = model.score_events(patient, index[start:stop], state[start:stop],
                                    resource[start:stop], return_steps=True)
        step = scores["order_step_logprob"]
        if step.shape[1] <= depth:
            continue
        valid = scores["select_step"][:, depth]
        pieces.append(step[:, depth][valid].cpu().numpy())
    return np.concatenate(pieces) if pieces else np.zeros(0)


# --------------------------------------------------------------------------
# H2b: probe summaries of a slow state at an arbitrary time
# --------------------------------------------------------------------------

#: Pre-registered in INTERICTAL_MODEL_FREEZE.json before any seizure label is read.
PROBE_ENDPOINTS = ("state_norm", "resource", "expected_load", "first_selection_entropy")


@torch.no_grad()
def probe_summary(model: EpiPRSSM, patient, state: torch.Tensor, resource: torch.Tensor
                  ) -> dict[str, np.ndarray]:
    """Summarise a slow state without an event: no observation is consumed here.

    ``state`` (B, N, D), ``resource`` (B,).
    """
    n_nodes, dim = state.shape[-2], state.shape[-1]
    adapter = model.adapter(state, resource)
    if not adapter.get("state_visible", True):
        state = torch.zeros_like(state)
        resource = torch.ones_like(resource)
    global_terms = adapter["global"]
    base = patient.baseline_order.view(1, -1) + model.decoder.static_node(
        patient.node_features).view(1, -1)
    if adapter.get("node_scale") is not None:
        base = base * adapter["node_scale"]
    if adapter.get("node_shift") is not None:
        base = base + adapter["node_shift"]
    base = base + global_terms[:, 0].view(-1, 1)
    probabilities = torch.softmax(base, dim=-1)
    entropy = -(probabilities * torch.log(probabilities.clamp(min=1e-12))).sum(-1)

    part_in = torch.cat([state, patient.node_features.unsqueeze(0).expand(
        state.shape[0], n_nodes, patient.node_features.shape[-1])], dim=-1)
    part_logit = (model.decoder.participation_head(part_in).squeeze(-1)
                  + patient.baseline_participation.view(1, -1)
                  + global_terms[:, 2].view(-1, 1))
    expected_load = torch.sigmoid(part_logit).mean(-1)

    return {
        "state_norm": (state.reshape(state.shape[0], -1).norm(dim=1)
                       / float(np.sqrt(n_nodes * dim))).cpu().numpy(),
        "resource": resource.cpu().numpy(),
        "expected_load": expected_load.cpu().numpy(),
        "first_selection_entropy": entropy.cpu().numpy(),
    }
