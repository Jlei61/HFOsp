"""Topic 5.2D v0.2 — two-level unordered baselines and strict low-dimensional
ordered-history operators.

Everything ordered enters the prediction through a single ``r``-dimensional
state.  The module deliberately contains no free ``C x C`` transition table, no
second network that reads the ordered prefix and no structure-specific contact
bias, because those are exactly the bypasses the design forbids (spec §3.4).

Model families
--------------
``DIRECT_HORIZON_UPPER_BOUND``
    horizon-specific low-dimensional readouts ``R_h = Q C^out_h`` applied to the
    same prefix state.  It answers "how much future can be decoded at all", and
    can never support a shared-dynamics claim.
``AUTONOMOUS_SHARED_OPERATOR``
    one ``F`` and one contact readout for every horizon: ``z_{t+h}=F^h z_t``.
``ORDERLESS_BAG``
    same frozen basis and the same output restriction, but the state is built
    from the cumulative contact set only, so it cannot read rank order.

Exact subset law
----------------
``p(S | n, l)`` is the conditional-Bernoulli law over the available contacts:
``p(S|n) = prod_{i in S} w_i / e_n(w_available)`` with ``w = exp(l)``.  The
elementary symmetric polynomials are evaluated by a log-space DP, and the same
law drives the stochastic decoder, so likelihood and rollout cannot drift apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

import numpy as np
import torch
from torch import nn

NEG = -1.0e30

BASELINE_LEVELS = ("U_MINIMAL", "U_FULL_SET")
MODEL_FAMILIES = ("DIRECT_HORIZON_UPPER_BOUND", "AUTONOMOUS_SHARED_OPERATOR", "ORDERLESS_BAG")
F_FORMS = ("FULL", "DIAGONAL_ONLY", "BANDWIDTH_1", "STABLE_NORMAL", "LOW_DIMENSIONAL_TANH")
STRUCTURES = (
    "H0_UNORDERED_ONLY",
    "H1_GEOMETRY_LAYOUT",
    "H1_SHAFT_GRADIENT",
    "H1_PATIENT_ALIGNED",
    "H1_ALIGNED_ORDERLESS_BAG",
    "H1_ANGLE_ROTATED_AXIS",
    "H1_IDENTITY_PERMUTED",
    "H1_LOCALITY_REWIRED",
    "H1_FREE_LOW_RANK",
)

# Frozen, arm-independent objective weights (spec §5.4).  They are constants of
# the design rather than tuned quantities, so no arm can be favoured by them.
CHECKPOINT_HORIZONS: tuple[int, ...] = (1, 2, 3)
CHECKPOINT_HORIZON_WEIGHT = 1.0 / 3.0
CHECKPOINT_SUFFIX_WEIGHT = 1.0
TRAIN_HORIZON_WEIGHT = 1.0 / 5.0
TRAIN_SUFFIX_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# elementary symmetric polynomials
# ---------------------------------------------------------------------------
def log_esp_suffix(logits: torch.Tensor, available: torch.Tensor, kmax: int) -> torch.Tensor:
    """``out[:, j, k] = log e_k(w_j .. w_{C-1})``; ``out[:, C] = [0, -inf, ...]``.

    ``out[:, 0, n]`` is therefore ``log e_n`` over every available contact, which
    is the normaliser of the exact subset law, and the whole array is what the
    O(C*kmax) marginal sweep below consumes.
    """
    batch, n_contacts = logits.shape
    masked = torch.where(available, logits, torch.full_like(logits, NEG))
    tail = logits.new_full((batch, kmax + 1), NEG)
    tail[:, 0] = 0.0
    levels = [tail]
    current = tail
    for position in range(n_contacts - 1, -1, -1):
        shifted = torch.cat([current.new_full((batch, 1), NEG), current[:, :-1]], dim=1)
        current = torch.logaddexp(current, shifted + masked[:, position: position + 1])
        levels.append(current)
    return torch.stack(levels[::-1], dim=1)


def log_subset_probability(
    logits: torch.Tensor,
    available: torch.Tensor,
    target: torch.Tensor,
    cardinality: torch.Tensor,
    kmax: int,
) -> torch.Tensor:
    """``log p(S | n, l)`` under the conditional-Bernoulli law.

    Singleton rank sets (26 of the 28 SEEG patients never produce a tie) collapse
    the law to a plain softmax over the available contacts.  The closed form is
    mathematically identical to the general dynamic program and is taken only to
    avoid O(C) tensor-dispatch overhead on every update.
    """
    if kmax == 1:
        masked = torch.where(available, logits, torch.full_like(logits, NEG))
        numerator = (torch.where(target, logits, torch.zeros_like(logits))).sum(dim=1)
        return numerator - torch.logsumexp(masked, dim=1)
    backward = log_esp_suffix(logits, available, kmax)
    log_norm = backward[:, 0, :].gather(1, cardinality.clamp(min=0, max=kmax).unsqueeze(1)).squeeze(1)
    numerator = (torch.where(target, logits, torch.zeros_like(logits))).sum(dim=1)
    return numerator - log_norm


def _conditional_inclusion(
    logits: torch.Tensor,
    available: torch.Tensor,
    backward: torch.Tensor,
) -> torch.Tensor:
    """``q[:, j, k] = P(contact j in S | budget k over contacts j..C-1)``."""
    masked = torch.where(available, logits, torch.full_like(logits, NEG))
    shifted = torch.cat(
        [backward.new_full((backward.shape[0], backward.shape[1], 1), NEG), backward[:, :, :-1]],
        dim=2,
    )
    log_q = masked.unsqueeze(2) + shifted[:, 1:] - backward[:, :-1]
    return log_q.clamp(max=0.0).exp()


def expected_inclusion(
    logits: torch.Tensor,
    available: torch.Tensor,
    cardinality_probability: torch.Tensor,
    kmax: int,
) -> torch.Tensor:
    """Exact marginal ``P(contact in S)`` under a cardinality mixture.

    Uses the classical O(C*kmax) budget sweep for the conditional-Bernoulli law
    rather than an O(C*kmax^2) leave-one-out convolution, so the dense ECoG
    rank sets (cardinality up to 29) stay affordable.
    """
    if kmax == 1:
        masked = torch.where(available, logits, torch.full_like(logits, NEG))
        return torch.softmax(masked, dim=1) * available.to(logits.dtype)
    batch, n_contacts = logits.shape
    backward = log_esp_suffix(logits, available, kmax)
    conditional = _conditional_inclusion(logits, available, backward)
    width = min(int(cardinality_probability.shape[1]), kmax)
    budget = torch.cat(
        [
            logits.new_zeros(batch, 1),
            cardinality_probability[:, :width],
            logits.new_zeros(batch, kmax - width),
        ],
        dim=1,
    )
    marginal = []
    for position in range(n_contacts):
        q = conditional[:, position]
        take = budget * q
        marginal.append(take.sum(dim=1))
        stay = budget - take
        budget = stay + torch.cat([take[:, 1:], take.new_zeros(batch, 1)], dim=1)
    return torch.stack(marginal, dim=1).clamp(0.0, 1.0)


def inclusion_probability(
    logits: torch.Tensor,
    available: torch.Tensor,
    cardinality: torch.Tensor,
    kmax: int,
) -> torch.Tensor:
    """Exact marginal ``P(contact in S | n)`` for a fixed cardinality per row."""
    one_hot = torch.zeros(logits.shape[0], kmax, device=logits.device, dtype=logits.dtype)
    one_hot.scatter_(1, (cardinality.clamp(min=1, max=kmax) - 1).unsqueeze(1), 1.0)
    return expected_inclusion(logits, available, one_hot, kmax)


def sample_subset(
    logits: torch.Tensor,
    available: torch.Tensor,
    cardinality: torch.Tensor,
    kmax: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw one exact conditional-Bernoulli subset per row (rollout decoder).

    The sweep is the sampling counterpart of :func:`expected_inclusion`, so the
    stochastic decoder and the likelihood cannot drift apart.
    """
    if kmax == 1:
        masked = torch.where(available, logits, torch.full_like(logits, NEG))
        draw = torch.multinomial(torch.softmax(masked, dim=1), num_samples=1, generator=generator)
        chosen = torch.zeros_like(available)
        chosen.scatter_(1, draw, cardinality.clamp(min=0, max=1).bool().unsqueeze(1))
        return chosen
    batch, n_contacts = logits.shape
    backward = log_esp_suffix(logits, available, kmax)
    conditional = _conditional_inclusion(logits, available, backward)
    remaining = cardinality.clamp(min=0, max=kmax).clone()
    chosen = torch.zeros_like(available)
    rows = torch.arange(batch, device=logits.device)
    for position in range(n_contacts):
        probability = conditional[rows, position, remaining]
        draw = torch.rand(batch, generator=generator, device=logits.device) < probability
        draw = draw & available[:, position] & (remaining > 0)
        chosen[:, position] = draw
        remaining = remaining - draw.long()
    return chosen


# ---------------------------------------------------------------------------
# tensors handed to every arm
# ---------------------------------------------------------------------------
@dataclass
class UnitTensors:
    prefix_sets: torch.Tensor
    start_set: torch.Tensor
    cumulative_set: torch.Tensor
    target_sets: torch.Tensor
    target_valid: torch.Tensor
    target_available: torch.Tensor
    target_cardinality: torch.Tensor
    suffix5_field: torch.Tensor
    full_suffix_field: torch.Tensor
    suffix_eval_mask: torch.Tensor
    late_field_centroid: torch.Tensor
    latency_proxy: torch.Tensor
    latency_valid: torch.Tensor
    prefix_len: int
    n_contacts: int
    max_cardinality: int
    n_horizons: int

    def index(self, rows: torch.Tensor) -> "UnitTensors":
        return UnitTensors(
            prefix_sets=self.prefix_sets[rows],
            start_set=self.start_set[rows],
            cumulative_set=self.cumulative_set[rows],
            target_sets=self.target_sets[rows],
            target_valid=self.target_valid[rows],
            target_available=self.target_available[rows],
            target_cardinality=self.target_cardinality[rows],
            suffix5_field=self.suffix5_field[rows],
            full_suffix_field=self.full_suffix_field[rows],
            suffix_eval_mask=self.suffix_eval_mask[rows],
            late_field_centroid=self.late_field_centroid[rows],
            latency_proxy=self.latency_proxy[rows],
            latency_valid=self.latency_valid[rows],
            prefix_len=self.prefix_len,
            n_contacts=self.n_contacts,
            max_cardinality=self.max_cardinality,
            n_horizons=self.n_horizons,
        )

    @property
    def n_samples(self) -> int:
        return int(self.start_set.shape[0])


def tensors_from_samples(samples, rows: np.ndarray, device: str = "cpu") -> UnitTensors:
    def to_tensor(array, dtype):
        return torch.as_tensor(np.asarray(array)[rows], dtype=dtype, device=device)

    return UnitTensors(
        prefix_sets=to_tensor(samples.prefix_sets, torch.float32),
        start_set=to_tensor(samples.start_set, torch.float32),
        cumulative_set=to_tensor(samples.cumulative_set, torch.float32),
        target_sets=to_tensor(samples.target_sets, torch.bool),
        target_valid=to_tensor(samples.target_valid, torch.bool),
        target_available=to_tensor(samples.target_available, torch.bool),
        target_cardinality=to_tensor(samples.target_cardinality, torch.long),
        suffix5_field=to_tensor(samples.suffix5_field, torch.float32),
        full_suffix_field=to_tensor(samples.full_suffix_field, torch.float32),
        suffix_eval_mask=to_tensor(samples.suffix_eval_mask, torch.bool),
        late_field_centroid=to_tensor(samples.late_field_centroid, torch.float32),
        latency_proxy=to_tensor(samples.latency_proxy, torch.float32),
        latency_valid=to_tensor(samples.latency_valid, torch.bool),
        prefix_len=samples.prefix_len,
        n_contacts=samples.n_contacts,
        max_cardinality=samples.max_cardinality,
        n_horizons=samples.target_valid.shape[1],
    )


def perturb_prefix_order(batch: UnitTensors, mode: str = "swap_middle") -> UnitTensors:
    """Reorder the ordered prefix while holding everything else fixed.

    The start rank set, the cumulative contact set, the prefix length and the
    contact cardinality are unchanged by construction, so both unordered
    baselines must return bit-identical logits and any change in the merged
    prediction is attributable to the ordered path (spec §6.3).
    """
    prefix_len = batch.prefix_len
    if prefix_len < 3:
        raise ValueError("prefix-order perturbation needs at least three rank sets")
    order = list(range(prefix_len))
    if mode == "swap_middle":
        order[1], order[2] = order[2], order[1]
    elif mode == "reverse_middle":
        order = [0] + order[1:][::-1]
    else:
        raise ValueError(f"unknown perturbation mode {mode!r}")
    permuted = batch.index(torch.arange(batch.n_samples))
    permuted.prefix_sets = batch.prefix_sets[:, torch.as_tensor(order)]
    return permuted


def covariant_rotation(model: "OrderedMotif", rotation: torch.Tensor) -> "OrderedMotif":
    """Rotate the low-dimensional coordinates without changing the subspace.

    ``Q -> QO``, ``C_in -> O^T C_in O``, ``F -> O^T F O``, ``C_out -> O^T C_out O``
    leaves every logit invariant for the full ``F``, which is why the full ``F``
    is primary: a banded or diagonal ``F`` would make the SVD column order a
    silent modelling choice.
    """
    import copy

    rotated = copy.deepcopy(model)
    with torch.no_grad():
        if not model.config.free_basis:
            rotated.basis.copy_(model.basis @ rotation)
        else:
            rotated.input_free.copy_(model.input_free @ rotation)
        rotated.c_in.copy_(rotation.T @ model.c_in @ rotation)
        if hasattr(model, "f_raw"):
            rotated.f_raw.copy_(rotation.T @ model.f_raw @ rotation)
        if model.config.family == "AUTONOMOUS_SHARED_OPERATOR":
            rotated.c_out.copy_(rotation.T @ model.c_out @ rotation)
            rotated.card_w.copy_(rotation.T @ model.card_w)
            if model.config.free_basis:
                rotated.output_free.copy_(model.output_free @ rotation)
        else:
            for horizon in range(model.config.n_horizons):
                rotated.c_out[horizon].copy_(rotation.T @ model.c_out[horizon] @ rotation)
                rotated.card_w[horizon].copy_(rotation.T @ model.card_w[horizon])
                if model.config.free_basis:
                    rotated.output_free[horizon].copy_(model.output_free[horizon] @ rotation)
            rotated.c_suffix.copy_(rotation.T @ model.c_suffix @ rotation)
            if model.config.free_basis:
                rotated.suffix_free.copy_(model.suffix_free @ rotation)
    return rotated


# ---------------------------------------------------------------------------
# unordered baselines
# ---------------------------------------------------------------------------
def unordered_features(batch: UnitTensors, level: str) -> torch.Tensor:
    """Permutation-invariant features (spec §3.2).

    Neither level may read the last rank set, the ordering of the prefix, a
    prefix centroid displacement, mode labels or anything about the future.
    """
    progress = torch.full(
        (batch.n_samples, 1), float(batch.prefix_len), device=batch.start_set.device
    )
    recruited = batch.cumulative_set.sum(dim=1, keepdim=True) / float(batch.n_contacts)
    if level == "U_MINIMAL":
        return torch.cat([batch.start_set, progress, recruited], dim=1)
    if level == "U_FULL_SET":
        return torch.cat([batch.start_set, batch.cumulative_set, progress, recruited], dim=1)
    raise ValueError(f"unknown baseline level {level!r}")


class UnorderedBaseline(nn.Module):
    """``l^base_{e,t,h} = b_h + U_h V_h^T a_{e,t}`` plus matching cardinality and
    suffix heads."""

    def __init__(self, level: str, n_contacts: int, n_features: int, n_horizons: int,
                 max_cardinality: int, rank: int) -> None:
        super().__init__()
        self.level = level
        self.n_contacts = n_contacts
        self.n_horizons = n_horizons
        self.max_cardinality = max_cardinality
        self.rank = rank
        scale = 1.0 / math.sqrt(max(1, n_features))
        self.contact_bias = nn.Parameter(torch.zeros(n_horizons, n_contacts))
        self.contact_u = nn.Parameter(torch.randn(n_horizons, n_contacts, rank) * scale)
        self.contact_v = nn.Parameter(torch.randn(n_horizons, n_features, rank) * scale)
        self.card_bias = nn.Parameter(torch.zeros(n_horizons, max_cardinality))
        self.card_v = nn.Parameter(torch.randn(n_horizons, n_features, max_cardinality) * scale)
        self.suffix_bias = nn.Parameter(torch.zeros(n_contacts))
        self.suffix_u = nn.Parameter(torch.randn(n_contacts, rank) * scale)
        self.suffix_v = nn.Parameter(torch.randn(n_features, rank) * scale)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = torch.einsum("bf,hfr->bhr", features, self.contact_v)
        contact = torch.einsum("bhr,hcr->bhc", projected, self.contact_u) + self.contact_bias
        cardinality = torch.einsum("bf,hfn->bhn", features, self.card_v) + self.card_bias
        suffix = features @ self.suffix_v @ self.suffix_u.T + self.suffix_bias
        return {"contact": contact, "cardinality": cardinality, "suffix": suffix}


# ---------------------------------------------------------------------------
# ordered low-dimensional models
# ---------------------------------------------------------------------------
@dataclass
class MotifConfig:
    structure: str
    family: str
    rank: int
    n_contacts: int
    n_horizons: int
    max_cardinality: int
    f_form: str = "FULL"
    free_basis: bool = False


class OrderedMotif(nn.Module):
    """Strict low-dimensional ordered-history residual.

    ``B_m = Q_m C^in_m`` and ``R_m = Q_m C^out_m`` share one frozen basis, so no
    arm can quietly buy a free contact readout (spec §4.4).
    """

    def __init__(self, config: MotifConfig, basis: np.ndarray | None) -> None:
        super().__init__()
        self.config = config
        rank = config.rank
        if config.free_basis:
            if basis is not None:
                raise ValueError("free model must not receive a frozen basis")
            scale = 1.0 / math.sqrt(config.n_contacts)
            self.input_free = nn.Parameter(torch.randn(config.n_contacts, rank) * scale)
            self.register_parameter("output_free", None)
        else:
            if basis is None:
                raise ValueError("structured model requires a frozen basis")
            tensor = torch.as_tensor(np.asarray(basis, dtype=np.float32))
            if tensor.shape != (config.n_contacts, rank):
                raise ValueError(f"basis shape {tuple(tensor.shape)} != {(config.n_contacts, rank)}")
            self.register_buffer("basis", tensor)
        scale = 1.0 / math.sqrt(rank)
        self.c_in = nn.Parameter(torch.randn(rank, rank) * scale)
        if config.family != "ORDERLESS_BAG":
            self.f_raw = nn.Parameter(torch.randn(rank, rank) * scale)
        if config.family == "AUTONOMOUS_SHARED_OPERATOR":
            self.c_out = nn.Parameter(torch.randn(rank, rank) * scale)
            self.card_w = nn.Parameter(torch.randn(rank) * scale)
            self.card_u = nn.Parameter(torch.zeros(config.max_cardinality))
        else:
            self.c_out = nn.Parameter(torch.randn(config.n_horizons, rank, rank) * scale)
            self.card_w = nn.Parameter(torch.randn(config.n_horizons, rank) * scale)
            self.card_u = nn.Parameter(torch.zeros(config.n_horizons, config.max_cardinality))
            self.c_suffix = nn.Parameter(torch.randn(rank, rank) * scale)
        if config.free_basis:
            out_shape = (config.n_horizons, config.n_contacts, rank) \
                if config.family != "AUTONOMOUS_SHARED_OPERATOR" else (config.n_contacts, rank)
            self.output_free = nn.Parameter(torch.randn(*out_shape) / math.sqrt(config.n_contacts))
            if config.family != "AUTONOMOUS_SHARED_OPERATOR":
                self.suffix_free = nn.Parameter(
                    torch.randn(config.n_contacts, rank) / math.sqrt(config.n_contacts)
                )

    # -- operator -----------------------------------------------------------
    def transition(self) -> torch.Tensor:
        raw = self.f_raw
        form = self.config.f_form
        if form == "FULL":
            return raw
        if form == "DIAGONAL_ONLY":
            return torch.diag(torch.diagonal(raw))
        if form == "BANDWIDTH_1":
            size = raw.shape[0]
            index = torch.arange(size, device=raw.device)
            band = (index.unsqueeze(0) - index.unsqueeze(1)).abs() <= 1
            return raw * band
        if form == "STABLE_NORMAL":
            skew = 0.5 * (raw - raw.T)
            scale = torch.tanh(torch.diagonal(raw).mean())
            return skew + scale * torch.eye(raw.shape[0], device=raw.device)
        if form == "LOW_DIMENSIONAL_TANH":
            return raw
        raise ValueError(f"unknown F form {self.config.f_form!r}")

    def _encode(self, contact_field: torch.Tensor) -> torch.Tensor:
        if self.config.free_basis:
            return contact_field @ self.input_free @ self.c_in
        return contact_field @ self.basis @ self.c_in

    def _decode(self, state: torch.Tensor, horizon: int | None) -> torch.Tensor:
        if self.config.family == "AUTONOMOUS_SHARED_OPERATOR":
            coordinates = state @ self.c_out.T
            if self.config.free_basis:
                return coordinates @ self.output_free.T
            return coordinates @ self.basis.T
        coordinates = state @ self.c_out[horizon].T
        if self.config.free_basis:
            return coordinates @ self.output_free[horizon].T
        return coordinates @ self.basis.T

    def prefix_state(self, batch: UnitTensors) -> torch.Tensor:
        if self.config.family == "ORDERLESS_BAG":
            return self._encode(batch.cumulative_set)
        transition = self.transition()
        nonlinear = self.config.f_form == "LOW_DIMENSIONAL_TANH"
        state = self._encode(batch.prefix_sets[:, 0])
        for step in range(1, batch.prefix_len):
            state = state @ transition.T
            if nonlinear:
                state = torch.tanh(state)
            state = state + self._encode(batch.prefix_sets[:, step])
        return state

    def forward(self, batch: UnitTensors, ordered_path: bool = True) -> dict[str, torch.Tensor]:
        state = self.prefix_state(batch)
        if not ordered_path:
            state = torch.zeros_like(state)
        n_horizons = self.config.n_horizons
        contact, cardinality = [], []
        if self.config.family == "AUTONOMOUS_SHARED_OPERATOR":
            transition = self.transition()
            nonlinear = self.config.f_form == "LOW_DIMENSIONAL_TANH"
            rolled = state
            for _ in range(n_horizons):
                rolled = rolled @ transition.T
                if nonlinear:
                    rolled = torch.tanh(rolled)
                contact.append(self._decode(rolled, None))
                cardinality.append(
                    (rolled @ self.card_w).unsqueeze(1) * self.card_u.unsqueeze(0)
                )
            suffix = None
        else:
            for horizon in range(n_horizons):
                contact.append(self._decode(state, horizon))
                cardinality.append(
                    (state @ self.card_w[horizon]).unsqueeze(1) * self.card_u[horizon].unsqueeze(0)
                )
            coordinates = state @ self.c_suffix.T
            suffix = coordinates @ (self.suffix_free.T if self.config.free_basis else self.basis.T)
        return {
            "contact": torch.stack(contact, dim=1),
            "cardinality": torch.stack(cardinality, dim=1),
            "suffix": suffix,
            "state": state,
        }

    def ordered_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# losses and metrics
# ---------------------------------------------------------------------------
def horizon_losses(
    contact_logits: torch.Tensor,
    cardinality_logits: torch.Tensor,
    batch: UnitTensors,
) -> dict[str, torch.Tensor]:
    """Per-horizon cardinality NLL, exact-subset NLL and their sum.

    Returns sums and counts so callers can aggregate over batches without
    re-weighting long events.  The horizons are folded into the batch dimension
    so the subset law is evaluated once rather than five times; ineligible
    horizons still produce a finite number, which the validity mask then zeroes,
    and their denominators stay separate exactly as before.
    """
    n_samples, n_horizons, n_contacts = contact_logits.shape
    kmax = batch.max_cardinality
    order = batch.target_cardinality.clamp(min=1, max=kmax)
    log_subset = log_subset_probability(
        contact_logits.reshape(-1, n_contacts),
        batch.target_available.reshape(-1, n_contacts),
        batch.target_sets.reshape(-1, n_contacts),
        order.reshape(-1),
        kmax,
    ).reshape(n_samples, n_horizons)
    log_card = torch.log_softmax(cardinality_logits, dim=2).gather(
        2, (order - 1).unsqueeze(2)).squeeze(2)
    valid = batch.target_valid.to(contact_logits.dtype)
    return {
        "cardinality": -(log_card * valid).sum(dim=0),
        "subset": -(log_subset * valid).sum(dim=0),
        "count": valid.sum(dim=0),
    }


def autonomous_suffix_field(
    contact_logits: torch.Tensor,
    cardinality_logits: torch.Tensor,
    batch: UnitTensors,
) -> torch.Tensor:
    """``1 - prod_h (1 - p_h)`` over the contacts still available after the prefix.

    The horizon set is fixed at 1..5 for every event, and the mask is the
    prefix-only no-repeat mask, so the field never reads the true remaining
    event length (spec §4.3).
    """
    n_samples, n_horizons, n_contacts = contact_logits.shape
    kmax = batch.max_cardinality
    mask = batch.suffix_eval_mask.unsqueeze(1).expand(n_samples, n_horizons, n_contacts)
    expected = expected_inclusion(
        contact_logits.reshape(-1, n_contacts),
        mask.reshape(-1, n_contacts),
        torch.softmax(cardinality_logits, dim=2).reshape(-1, kmax),
        kmax,
    ).reshape(n_samples, n_horizons, n_contacts)
    return 1.0 - (1.0 - expected.clamp(0.0, 1.0 - 1e-6)).prod(dim=1)


def balanced_field_scores(
    probability: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Event-balanced BCE and Brier plus the raw unbalanced values.

    Every event contributes with weight one and, inside an event, positives and
    negatives contribute equally, so long events cannot dominate.
    """
    probability = probability.clamp(1e-6, 1.0 - 1e-6)
    positive = (truth > 0.5) & mask
    negative = (truth <= 0.5) & mask
    n_positive = positive.sum(dim=1)
    n_negative = negative.sum(dim=1)
    usable = (n_positive > 0) & (n_negative > 0)
    bce = -(truth * probability.log() + (1 - truth) * (1 - probability).log())
    brier = (probability - truth) ** 2
    out: dict[str, torch.Tensor] = {}
    for name, value in (("bce", bce), ("brier", brier)):
        positive_mean = (value * positive).sum(dim=1) / n_positive.clamp(min=1)
        negative_mean = (value * negative).sum(dim=1) / n_negative.clamp(min=1)
        balanced = 0.5 * (positive_mean + negative_mean)
        out[f"balanced_{name}_sum"] = (balanced * usable).sum()
        raw = (value * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        out[f"raw_{name}_sum"] = (raw * mask.any(dim=1)).sum()
    out["balanced_count"] = usable.sum().to(probability.dtype)
    out["raw_count"] = mask.any(dim=1).sum().to(probability.dtype)
    return out


def combine_logits(base: dict[str, torch.Tensor], residual: dict[str, torch.Tensor] | None
                   ) -> dict[str, torch.Tensor]:
    """Frozen baseline plus the ordered residual.

    An autonomous arm has no suffix readout of its own, so the merged suffix
    logit is dropped entirely rather than silently inheriting the baseline head —
    that makes it structurally impossible to report a direct-style full-suffix
    number for the autonomous family.
    """
    if residual is None:
        return {"contact": base["contact"], "cardinality": base["cardinality"], "suffix": base["suffix"]}
    suffix = None if residual.get("suffix") is None else base["suffix"] + residual["suffix"]
    return {
        "contact": base["contact"] + residual["contact"],
        "cardinality": base["cardinality"] + residual["cardinality"],
        "suffix": suffix,
    }


@dataclass
class EvaluationResult:
    per_horizon: dict[str, list[float]] = field(default_factory=dict)
    scalars: dict[str, float] = field(default_factory=dict)


def primary_field_kind(family: str | None) -> str:
    """Which spatial field carries the arm's primary claim.

    The autonomous family is scored on the accumulated 1–5 step field it can
    generate from one operator; the direct family and the unordered baselines
    are scored on their independent full-suffix readout, which is an upper bound
    and is never mixed with the autonomous number.
    """
    return "suffix5" if family == "AUTONOMOUS_SHARED_OPERATOR" else "full_suffix"


def _field_predictions(
    merged: dict[str, torch.Tensor], batch: UnitTensors
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    fields = {
        "suffix5": (
            autonomous_suffix_field(merged["contact"], merged["cardinality"], batch),
            batch.suffix5_field,
        )
    }
    if merged.get("suffix") is not None:
        fields["full_suffix"] = (torch.sigmoid(merged["suffix"]), batch.full_suffix_field)
    return fields


def evaluate(
    model: OrderedMotif | None,
    baseline_logits: dict[str, torch.Tensor],
    batch: UnitTensors,
    contact_xy: torch.Tensor,
    ordered_path: bool = True,
    chunk: int = 4096,
) -> EvaluationResult:
    """Held-out scoring; never touches gradients.

    Both spatial fields are always produced so that H0, the direct family and
    the autonomous family can be compared on the field that belongs to each
    claim without re-running anything.
    """
    n_horizons = batch.n_horizons
    card_sum = np.zeros(n_horizons)
    subset_sum = np.zeros(n_horizons)
    counts = np.zeros(n_horizons)
    field_totals: dict[str, dict[str, float]] = {}
    endpoint: dict[str, list[float]] = {}
    device = batch.start_set.device
    with torch.no_grad():
        for start in range(0, batch.n_samples, chunk):
            rows = torch.arange(start, min(start + chunk, batch.n_samples), device=device)
            piece = batch.index(rows)
            base = {key: value[rows] for key, value in baseline_logits.items()}
            residual = model(piece, ordered_path=ordered_path) if model is not None else None
            merged = combine_logits(base, residual)
            losses = horizon_losses(merged["contact"], merged["cardinality"], piece)
            card_sum += losses["cardinality"].cpu().numpy()
            subset_sum += losses["subset"].cpu().numpy()
            counts += losses["count"].cpu().numpy()
            for kind, (probability, truth) in _field_predictions(merged, piece).items():
                bucket = field_totals.setdefault(kind, {})
                for key, value in balanced_field_scores(probability, truth, piece.suffix_eval_mask).items():
                    bucket[key] = bucket.get(key, 0.0) + float(value)
                weights = probability * piece.suffix_eval_mask
                total = weights.sum(dim=1, keepdim=True)
                usable = total.squeeze(1) > 1e-6
                predicted = (weights / total.clamp(min=1e-6)) @ contact_xy
                distance = torch.linalg.norm(predicted - piece.late_field_centroid, dim=1)
                store = endpoint.setdefault(kind, [0.0, 0.0])
                store[0] += float((distance * usable).sum())
                store[1] += float(usable.sum())
    result = EvaluationResult()
    with np.errstate(invalid="ignore", divide="ignore"):
        result.per_horizon = {
            "cardinality_nll": list(np.where(counts > 0, card_sum / np.maximum(counts, 1), np.nan)),
            "subset_nll": list(np.where(counts > 0, subset_sum / np.maximum(counts, 1), np.nan)),
            "total_nll": list(
                np.where(counts > 0, (card_sum + subset_sum) / np.maximum(counts, 1), np.nan)
            ),
            "denominator": list(counts),
        }
    scalars: dict[str, float] = {}
    for kind, bucket in field_totals.items():
        balanced = max(bucket.get("balanced_count", 0.0), 1.0)
        raw = max(bucket.get("raw_count", 0.0), 1.0)
        scalars[f"{kind}_balanced_bce"] = bucket.get("balanced_bce_sum", 0.0) / balanced
        scalars[f"{kind}_balanced_brier"] = bucket.get("balanced_brier_sum", 0.0) / balanced
        scalars[f"{kind}_raw_bce"] = bucket.get("raw_bce_sum", 0.0) / raw
        scalars[f"{kind}_raw_brier"] = bucket.get("raw_brier_sum", 0.0) / raw
        scalars[f"{kind}_balanced_denominator"] = bucket.get("balanced_count", 0.0)
        scalars[f"{kind}_endpoint_distance_mm"] = endpoint[kind][0] / max(endpoint[kind][1], 1.0)
        scalars[f"{kind}_endpoint_denominator"] = endpoint[kind][1]
    scalars["primary_field_kind"] = primary_field_kind(
        model.config.family if model is not None else None
    )
    result.scalars = scalars
    return result


def checkpoint_objective(result: EvaluationResult, family: str | None = None) -> float:
    """``L_space`` from spec §5.4 — h=4/5 and STOP never enter it."""
    total = 0.0
    for horizon in CHECKPOINT_HORIZONS:
        value = result.per_horizon["total_nll"][horizon - 1]
        if not math.isnan(value):
            total += CHECKPOINT_HORIZON_WEIGHT * value
    kind = primary_field_kind(family)
    total += CHECKPOINT_SUFFIX_WEIGHT * result.scalars[f"{kind}_balanced_bce"]
    return float(total)


def training_loss(
    merged: dict[str, torch.Tensor],
    batch: UnitTensors,
    field_kind: str,
) -> torch.Tensor:
    losses = horizon_losses(merged["contact"], merged["cardinality"], batch)
    total = merged["contact"].new_zeros(())
    for horizon in range(batch.n_horizons):
        count = losses["count"][horizon]
        if float(count) > 0:
            total = total + TRAIN_HORIZON_WEIGHT * (
                losses["cardinality"][horizon] + losses["subset"][horizon]
            ) / count
    probability, truth = _field_predictions(merged, batch)[field_kind]
    scores = balanced_field_scores(probability, truth, batch.suffix_eval_mask)
    if float(scores["balanced_count"]) > 0:
        total = total + TRAIN_SUFFIX_WEIGHT * scores["balanced_bce_sum"] / scores["balanced_count"]
    return total


# ---------------------------------------------------------------------------
# shared training loop
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    lr: float = 0.05
    max_epochs: int = 400
    patience: int = 30
    batch_size: int = 4096
    eval_every: int = 2
    gradient_clip: float = 5.0
    seed: int = 0
    max_seconds: float = 1800.0
    # Patience counts epochs, so an epoch that contains a single gradient step
    # turns "patience=30" into "30 updates" and leaves the small patients badly
    # under-trained.  The effective batch is shrunk until every epoch performs at
    # least this many updates, which keeps the optimiser budget comparable across
    # montages of very different size.
    min_updates_per_epoch: int = 8

    def effective_batch_size(self, n_train: int) -> int:
        if n_train <= 0:
            return self.batch_size
        target = max(1, math.ceil(n_train / self.min_updates_per_epoch))
        return int(max(32, min(self.batch_size, target)))


def fit(
    module: nn.Module,
    forward: "callable",
    train_batch: UnitTensors,
    valid_batch: UnitTensors,
    valid_objective: "callable",
    config: TrainConfig,
) -> dict:
    """Adam with split-1 checkpoint selection and early stopping.

    ``forward`` maps a :class:`UnitTensors` slice to merged logits; ``valid_objective``
    maps the module to the frozen ``L_space``.  Both are supplied by the caller so
    baselines and ordered residuals share exactly one optimiser contract.
    """
    import copy
    import time

    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    optimiser = torch.optim.Adam(module.parameters(), lr=config.lr)
    best_score = float("inf")
    best_state = copy.deepcopy(module.state_dict())
    best_epoch = -1
    history = []
    started = time.time()
    n_train = train_batch.n_samples
    stalled = 0
    nonfinite_batches = 0
    updates = 0
    device = train_batch.start_set.device
    batch_size = config.effective_batch_size(n_train)
    for epoch in range(config.max_epochs):
        module.train()
        order = torch.randperm(n_train, generator=generator).to(device)
        for start in range(0, n_train, batch_size):
            rows = order[start:start + batch_size]
            piece = train_batch.index(rows)
            loss = forward(piece, rows)
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                continue
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), config.gradient_clip)
            optimiser.step()
            updates += 1
        if epoch % config.eval_every == 0 or epoch == config.max_epochs - 1:
            module.eval()
            score = valid_objective(module)
            history.append({"epoch": epoch, "valid_objective": score})
            if score < best_score - 1e-9:
                best_score, best_epoch = score, epoch
                best_state = copy.deepcopy(module.state_dict())
                stalled = 0
            else:
                stalled += config.eval_every
            if stalled >= config.patience:
                break
        if time.time() - started > config.max_seconds:
            break
    module.load_state_dict(best_state)
    module.eval()
    return {
        "best_valid_objective": best_score,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "gradient_updates": updates,
        "effective_batch_size": batch_size,
        "wall_seconds": time.time() - started,
        "nonfinite_batches": nonfinite_batches,
        "history": history,
        "early_stopped": stalled >= config.patience,
    }
