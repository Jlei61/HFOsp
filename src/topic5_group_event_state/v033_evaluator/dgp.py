"""Synthetic data-generating processes D0-D5 on a real scaffold (plan Task 4).

Only the *targets* are synthetic: future counts (NB around the registry
``log mu_H``) and the contact subset of every real event (conditional Bernoulli
given the real size K).  A hidden state is the leaky integral (tau = 30 min) of
a synthetic per-event mark channel, standardised on TRAIN anchors:

    D0  H only                          (marks visible, no effect)
    D1  count-only state                (z_N drives counts; grammar state-free)
    D2  grammar-only state              (z_G drives subsets; counts H only)
    D3  one shared state                (z_N = z_G)
    D4  two independent states          (z_N, z_G from independent mark sets)
    D5  shared state, marks invisible   (background-only; expected estimator failure)

Effect knobs ``beta_count`` / ``beta_grammar`` are inputs; the assay reports
effects on the oracle deviance-gain scale, never on beta itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import canonical as C
from .scaffold import Scaffold

DGP_KINDS = ("D0", "D1", "D2", "D3", "D4", "D5")
STATE_VIEWS = {"D0": (False, False), "D1": (True, False), "D2": (False, True),
               "D3": (True, True), "D4": (True, True), "D5": (True, True)}
HIDDEN_TAU_SECONDS = 1800.0
MARK_WIDTH = 2
BASE_RATE_CLIP = (0.01, 0.99)
PRIMARY_HORIZON = 1800.0
COUNT_PROFILE_EDGES_SECONDS = (0.0, 300.0, 900.0, 1800.0)
COUNT_PROFILE_WIDTHS_SECONDS = np.diff(COUNT_PROFILE_EDGES_SECONDS)


@dataclass
class SyntheticData:
    kind: str
    subject: str
    beta_count: float
    beta_grammar: float
    generator_seed: int
    noise_seed: int
    hidden_tau: float
    counts: dict[int, np.ndarray]
    log_mu_true: dict[int, np.ndarray]
    count_profile: np.ndarray
    log_mu_profile_h: np.ndarray
    log_mu_profile_true: np.ndarray
    log_r_profile_h: np.ndarray
    z_count: np.ndarray
    z_grammar_anchor: np.ndarray
    z_grammar_event: np.ndarray
    marks: np.ndarray | None
    innovations: dict[str, np.ndarray | None]
    participation: np.ndarray
    base_logits: np.ndarray
    loadings: np.ndarray
    has_state: dict[str, bool]

    def as_meta(self) -> dict[str, Any]:
        return {"kind": self.kind, "subject": self.subject, "beta_count": self.beta_count,
                "beta_grammar": self.beta_grammar, "generator_seed": self.generator_seed,
                "noise_seed": self.noise_seed, "hidden_tau_seconds": self.hidden_tau,
                "mark_width": MARK_WIDTH, "marks_visible": self.marks is not None,
                "count_profile_edges_seconds": list(COUNT_PROFILE_EDGES_SECONDS),
                "count_profile_baseline": "1800s_H_mean_distributed_by_bin_duration_for_synthetic_assay_only",
                "has_state": dict(self.has_state)}


# --------------------------------------------------------------------------- hidden state
def hidden_leaky_state(marks: np.ndarray, event_times: np.ndarray, event_carry: np.ndarray,
                       t_anchor: np.ndarray, anchor_carry: np.ndarray, last_event_pos: np.ndarray,
                       *, tau: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact leaky integral of ``marks`` (reset at every carry-unit change) -> (anchor state, pre-event state)."""

    m = np.asarray(marks, dtype=np.float64)
    t = np.asarray(event_times, dtype=np.float64)
    carry = np.asarray(event_carry, dtype=np.int64)
    n, k = m.shape
    post = np.zeros((n, k), dtype=np.float64)
    pre = np.zeros((n, k), dtype=np.float64)
    state = np.zeros(k, dtype=np.float64)
    for i in range(n):
        if i == 0 or carry[i] != carry[i - 1]:
            state = np.zeros(k, dtype=np.float64)
        else:
            state = state * np.exp(-(t[i] - t[i - 1]) / tau)
        pre[i] = state
        state = state + m[i]
        post[i] = state
    last = np.asarray(last_event_pos, dtype=np.int64)
    ta = np.asarray(t_anchor, dtype=np.float64)
    anchor = np.zeros((ta.size, k), dtype=np.float64)
    has = last >= 0
    if has.any():
        if (carry[last[has]] != np.asarray(anchor_carry, dtype=np.int64)[has]).any():
            raise ValueError("last_event_pos points outside the anchor's carry unit")
        dt = ta[has] - t[last[has]]
        anchor[has] = post[last[has]] * np.exp(-dt / tau)[:, None]
    return anchor, pre


def standardise_on_train(values: np.ndarray, train_rows: np.ndarray) -> tuple[np.ndarray, float, float]:
    v = np.asarray(values, dtype=np.float64)
    ref = v[train_rows]
    mean = float(ref.mean())
    scale = float(ref.std())
    scale = scale if scale > 1e-9 else 1.0
    return (v - mean) / scale, mean, scale


# --------------------------------------------------------------------------- conditional Bernoulli
def _log_sigmoid_pair(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(logits, dtype=np.float64)
    return -np.logaddexp(0.0, -x), -np.logaddexp(0.0, x)   # log p, log (1-p)


def conditional_bernoulli_logpmf(logits: np.ndarray, subset: np.ndarray) -> np.ndarray:
    """``log P(S | |S| = K)``: the negated canonical grammar score (single formula lives in ``canonical``)."""

    return -C.conditional_subset_nll(logits, subset)


def sample_conditional_bernoulli_batch(rng: np.random.Generator, logits: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Exact samples of ``K`` contacts each, ``K`` fixed per row, from the conditional Bernoulli model."""

    lg = np.asarray(logits, dtype=np.float64)
    k_all = np.asarray(sizes, dtype=np.int64)
    e, c = lg.shape
    if k_all.shape != (e,) or (k_all < 0).any() or (k_all > c).any():
        raise ValueError("sizes must be (E,) with 0 <= K <= C")
    out = np.zeros((e, c), dtype=bool)
    if e == 0:
        return out
    logp, log1mp = _log_sigmoid_pair(lg)
    kmax = int(k_all.max())
    # suffix table T[:, j, k] = log P(sum_{i >= j} X_i = k), j = 0..C
    table = np.full((e, c + 1, kmax + 1), -np.inf)
    table[:, c, 0] = 0.0
    for j in range(c - 1, -1, -1):
        stay = table[:, j + 1, :] + log1mp[:, j:j + 1]
        take = np.full_like(stay, -np.inf)
        take[:, 1:] = table[:, j + 1, :-1] + logp[:, j:j + 1]
        table[:, j, :] = np.logaddexp(stay, take)
    remaining = k_all.copy()
    rows = np.arange(e)
    for j in range(c):
        need = remaining > 0
        p_take = np.zeros(e)
        if need.any():
            r = rows[need]
            num = logp[r, j] + table[r, j + 1, remaining[r] - 1]
            den = table[r, j, remaining[r]]
            p_take[need] = np.exp(num - den)
        take = rng.uniform(size=e) < p_take
        out[:, j] = take
        remaining = remaining - take.astype(np.int64)
    if (remaining != 0).any():
        raise RuntimeError("conditional Bernoulli sampler did not hit the requested sizes")
    return out


def sample_conditional_bernoulli(rng: np.random.Generator, logits: np.ndarray, k: int) -> np.ndarray:
    return sample_conditional_bernoulli_batch(rng, np.asarray(logits, dtype=np.float64)[None, :],
                                              np.asarray([int(k)]))[0]


# --------------------------------------------------------------------------- generator
def _base_logits(scaffold: Scaffold) -> np.ndarray:
    rows = scaffold.event_rows("base_fit")
    freq = scaffold.participation[rows].mean(axis=0) if rows.size else np.full(scaffold.n_contacts, 0.5)
    freq = np.clip(freq, *BASE_RATE_CLIP)
    return np.log(freq / (1.0 - freq))


def generate(scaffold: Scaffold, kind: str, *, beta_count: float, beta_grammar: float,
             generator_seed: int, noise_seed: int, hidden_tau: float = HIDDEN_TAU_SECONDS) -> SyntheticData:
    if kind not in DGP_KINDS:
        raise ValueError(f"kind must be one of {DGP_KINDS}, got {kind!r}")
    count_on, grammar_on = STATE_VIEWS[kind]
    gen = np.random.default_rng(int(generator_seed))
    noise = np.random.default_rng(int(noise_seed))
    n, c, a = scaffold.n_events, scaffold.n_contacts, scaffold.n_anchors
    primary = PRIMARY_HORIZON if PRIMARY_HORIZON in [float(h) for h in scaffold.horizons] else float(scaffold.horizons[0])
    train = scaffold.anchor_rows("base_fit", primary)

    def hidden(marks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        proj = gen.normal(size=marks.shape[1])
        anchor, pre = hidden_leaky_state(marks, scaffold.event_times, scaffold.event_carry, scaffold.t_anchor,
                                         scaffold.anchor_carry, scaffold.last_event_pos, tau=hidden_tau)
        z_anchor, mean, scale = standardise_on_train(anchor @ proj, train)
        z_pre = ((pre @ proj) - mean) / scale
        return z_anchor, z_pre

    marks_a = gen.normal(size=(n, MARK_WIDTH))
    z_a, z_a_pre = hidden(marks_a)
    marks_b = z_b = z_b_pre = None
    if kind == "D4":
        marks_b = gen.normal(size=(n, MARK_WIDTH))
        z_b, z_b_pre = hidden(marks_b)
    loadings = gen.normal(size=c)
    base_logits = _base_logits(scaffold)

    z_count = z_a if count_on else np.zeros(a)
    if grammar_on:
        z_grammar_anchor, z_grammar_event = (z_b, z_b_pre) if kind == "D4" else (z_a, z_a_pre)
    else:
        z_grammar_anchor, z_grammar_event = np.zeros(a), np.zeros(n)

    counts: dict[int, np.ndarray] = {}
    log_mu_true: dict[int, np.ndarray] = {}
    for h_i, h in enumerate(scaffold.horizons):
        key = int(h)
        lm = np.asarray(scaffold.log_mu_h[key], dtype=np.float64)
        lm_true = lm + (float(beta_count) * z_count if count_on else 0.0)
        log_mu_true[key] = lm_true
        y = np.zeros(a, dtype=np.int64)
        rows = np.flatnonzero(scaffold.eligible[:, h_i] & np.isfinite(lm_true))
        log_r = scaffold.log_r_h.get(key)
        r = float(np.exp(log_r)) if log_r is not None else 5.0
        if rows.size:
            mu = np.exp(lm_true[rows])
            y[rows] = noise.negative_binomial(r, r / (r + mu))
        counts[key] = y

    # Primary S_N target: three disjoint physical-time bins whose sum is N_0-30min.
    # The synthetic assay has only a cumulative H registry, so its state-free
    # bin means use the registry's 30-minute mean under a constant within-window
    # baseline rate. This is an explicit synthetic construction; human training
    # consumes the materialised per-bin H offsets supplied by Workstream C.
    profile_horizon = int(COUNT_PROFILE_EDGES_SECONDS[-1])
    if profile_horizon not in scaffold.log_mu_h:
        raise ValueError("count-profile assay requires the 1800-second H registry horizon")
    total_log_mu_h = np.asarray(scaffold.log_mu_h[profile_horizon], dtype=np.float64)
    fractions = COUNT_PROFILE_WIDTHS_SECONDS / float(profile_horizon)
    log_mu_profile_h = total_log_mu_h[:, None] + np.log(fractions)[None, :]
    profile_loading = np.array([0.75, 1.0, 1.25], dtype=np.float64)
    log_mu_profile_true = log_mu_profile_h + (
        float(beta_count) * z_count[:, None] * profile_loading[None, :] if count_on else 0.0
    )
    base_log_r = scaffold.log_r_h.get(profile_horizon)
    base_log_r = float(base_log_r) if base_log_r is not None else float(np.log(5.0))
    log_r_profile_h = np.full(3, base_log_r, dtype=np.float64)
    count_profile = np.zeros((a, 3), dtype=np.int64)
    h_i = scaffold.horizon_index(float(profile_horizon))
    rows = np.flatnonzero(scaffold.eligible[:, h_i] & np.isfinite(log_mu_profile_true).all(axis=1))
    for j in range(3):
        r = float(np.exp(log_r_profile_h[j]))
        mu = np.exp(log_mu_profile_true[rows, j])
        count_profile[rows, j] = noise.negative_binomial(r, r / (r + mu))

    logits = base_logits[None, :] + (float(beta_grammar) * loadings[None, :] * z_grammar_event[:, None] if grammar_on else 0.0)
    participation = sample_conditional_bernoulli_batch(noise, np.broadcast_to(logits, (n, c)).copy(), scaffold.event_size())

    if kind == "D5":
        marks = None
    elif kind == "D4":
        marks = np.concatenate([marks_a, marks_b], axis=1)
    else:
        marks = marks_a
    innovations = {
        "count": marks_a if count_on else None,
        "grammar": (marks_b if kind == "D4" else marks_a) if grammar_on else None,
    }
    return SyntheticData(
        kind=kind, subject=scaffold.subject, beta_count=float(beta_count), beta_grammar=float(beta_grammar),
        generator_seed=int(generator_seed), noise_seed=int(noise_seed), hidden_tau=float(hidden_tau),
        counts=counts, log_mu_true=log_mu_true,
        count_profile=count_profile, log_mu_profile_h=log_mu_profile_h,
        log_mu_profile_true=log_mu_profile_true, log_r_profile_h=log_r_profile_h,
        z_count=z_count, z_grammar_anchor=z_grammar_anchor,
        z_grammar_event=z_grammar_event, marks=marks, innovations=innovations, participation=participation,
        base_logits=base_logits, loadings=loadings, has_state={"count": count_on, "grammar": grammar_on},
    )
