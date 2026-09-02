"""Oracle Level 0-2 estimators for the count and grammar views (plan Task 5).

    Level 0  true standardised state at anchors           -> fit the output head only
    Level 1  true event innovation + fixed leaky bank      -> fit the linear readout only
             (tau = 300 / 1800 / 7200 s, standardised on TRAIN anchors)
    Level 2  visible mark channel (+ real tokens)          -> train encoder + bank + readout

Heads and readouts are fitted on TRAIN (base_fit) rows, Level 2 early-stops on
inner_val, and every gain is scored on the development rows (dev_val ∪ dev_test)
through the canonical evaluator (O1, O3).  The H arm always receives the same
TRAIN-only recalibration (intercept + dispersion) as the state arm so a gain is
never an intercept artefact (O2).  The primary detection statistic is the
shared-H-dispersion gain (dynamic contribution only); the per-arm-dispersion gain
is reported alongside for the count view.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma
import torch
from torch import nn

from src.topic5_group_event_state.v032_eval.blocks import block_bootstrap_mean_ci, block_ids_for_times
from src.topic5_group_event_state.v032_model.readout import moment_log_dispersion
from src.topic5_group_event_state.v032_model.state import anchor_states as _anchor_states_torch
from src.topic5_group_event_state.v032_model.state import leaky_bank_trajectory

from . import canonical as C
from .dgp import MARK_WIDTH, SyntheticData, hidden_leaky_state
from .scaffold import PHASE_NAMES, Scaffold

BANK_TAUS = (300.0, 1800.0, 7200.0)
LEVELS = (0, 1, 2)
VIEWS = ("count_profile", "count", "grammar")
PRIMARY_VIEWS = ("count_profile", "grammar")
TRAIN_PHASE = "base_fit"
SELECT_PHASE = "inner_val"
DEV_PHASES = ("dev_val", "dev_test")
LOG_R_BOUNDS = (math.log(0.05), math.log(1e5))
BOOTSTRAP_RESAMPLES = 1000
LEVEL2_HIDDEN = 16
LEVEL2_WRITE = MARK_WIDTH
FAILURE_BY_LEVEL = {0: "head", 1: "scan_alignment", 2: "encoder_optimizer"}
DETECTION_FLOOR_NATS = 1e-9   # a CI lower bound above this counts as detection (guards against float dust at exactly 0)


# --------------------------------------------------------------------------- count head
def fit_count_head(y: np.ndarray, log_mu_h: np.ndarray, features: np.ndarray | None,
                   *, ridge: float = 1e-4, fixed_log_r: float | None = None) -> dict[str, Any]:
    """TRAIN-only NB head; optionally keep H's dispersion frozen for a nested comparison."""

    y = np.asarray(y, dtype=np.float64)
    off = np.asarray(log_mu_h, dtype=np.float64)
    x = np.zeros((y.size, 0)) if features is None else np.asarray(features, dtype=np.float64)
    if x.shape[0] != y.size:
        raise ValueError("features must have one row per target")
    d = x.shape[1]
    n = float(y.size)
    lo, hi = LOG_R_BOUNDS

    def objective(params):
        c, beta = params[0], params[1:1 + d]
        log_r = float(np.clip(params[1 + d], lo, hi)) if fixed_log_r is None else float(fixed_log_r)
        eta = off + c + x @ beta
        nll = C.nb_nll(y, eta, log_r)
        mu = np.exp(eta)
        r = math.exp(log_r)
        d_eta = -(y - mu) * r / (r + mu)
        d_log_r = -r * (digamma(y + r) - digamma(r) + log_r + 1.0 - np.log(r + mu) - (y + r) / (r + mu))
        pieces = [[d_eta.sum()], x.T @ d_eta + n * ridge * beta]
        if fixed_log_r is None:
            pieces.append([d_log_r.sum()])
        grad = np.concatenate(pieces) / n
        return float(nll.sum() / n + 0.5 * ridge * float(beta @ beta)), grad

    mu0 = np.exp(off)
    c0 = math.log(max(y.mean(), 1e-8) / max(mu0.mean(), 1e-8))
    init = np.concatenate([[c0], np.zeros(d)])
    bounds = [(None, None)] * (1 + d)
    if fixed_log_r is None:
        init = np.concatenate([init, [moment_log_dispersion(y, mu0 * math.exp(c0))]])
        bounds.append((lo, hi))
    elif not np.isfinite(float(fixed_log_r)):
        raise ValueError("fixed_log_r must be finite")
    res = minimize(objective, init, jac=True, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8})
    fitted_log_r = float(np.clip(res.x[1 + d], lo, hi)) if fixed_log_r is None else float(fixed_log_r)
    return {"c": float(res.x[0]), "beta": np.asarray(res.x[1:1 + d], dtype=np.float64),
            "log_r": fitted_log_r, "dispersion_frozen": fixed_log_r is not None,
            "converged": bool(res.success), "n_rows": int(y.size), "ridge": ridge}


def predict_count_head(head: dict[str, Any], log_mu_h: np.ndarray, features: np.ndarray | None) -> np.ndarray:
    off = np.asarray(log_mu_h, dtype=np.float64)
    if features is None or head["beta"].size == 0:
        return off + head["c"]
    return off + head["c"] + np.asarray(features, dtype=np.float64) @ head["beta"]


# --------------------------------------------------------------------------- grammar head
def conditional_bernoulli_logpmf_torch(logits: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    """Torch twin of :func:`canonical.conditional_subset_nll` (negated): log P(S | |S| = K).

    Infeasible DP cells use a finite sentinel instead of ``-inf``: the backward of
    ``logaddexp(-inf, -inf)`` is NaN and would poison every gradient upstream.
    """

    neg = -1e300
    lg = logits.to(torch.float64)
    x = subset.to(torch.bool)
    e, c = lg.shape
    logp = -torch.nn.functional.softplus(-lg)
    log1mp = -torch.nn.functional.softplus(lg)
    joint = torch.where(x, logp, log1mp).sum(dim=1)
    k = x.sum(dim=1)
    kmax = int(k.max()) if e else 0
    dp = torch.full((e, kmax + 1), neg, dtype=torch.float64)
    dp[:, 0] = 0.0
    for j in range(c):
        stay = dp + log1mp[:, j:j + 1]
        if kmax:
            take = torch.cat([torch.full((e, 1), neg, dtype=torch.float64), dp[:, :-1] + logp[:, j:j + 1]], dim=1)
            dp = torch.logaddexp(stay, take)
        else:
            dp = stay
    log_z = dp[torch.arange(e), k]
    return joint - log_z


def _base_logits(participation: np.ndarray) -> np.ndarray:
    freq = np.clip(participation.mean(axis=0), 0.01, 0.99)
    return np.log(freq / (1.0 - freq))


def fit_grammar_head(participation: np.ndarray, states: np.ndarray | None, *, n_steps: int = 300,
                     lr: float = 0.05, ridge: float = 1e-4, seed: int = 0,
                     weights: np.ndarray | None = None) -> dict[str, Any]:
    """Conditional-Bernoulli MLE of ``logits = a + W s`` on TRAIN rows (``states=None`` -> a only).

    ``weights`` (optional, per row) let the fit match the per-anchor block-average
    score exactly when rows are (anchor, event) pairs: weight 1 / N_future(anchor).
    """

    torch.manual_seed(int(seed))
    part = torch.from_numpy(np.asarray(participation, dtype=bool))
    e, c = part.shape
    s = torch.zeros((e, 0), dtype=torch.float64) if states is None else torch.from_numpy(np.asarray(states, dtype=np.float64))
    d = s.shape[1]
    wt = torch.ones(e, dtype=torch.float64) if weights is None else torch.from_numpy(np.asarray(weights, dtype=np.float64))
    wt = wt / wt.sum()
    a = torch.tensor(_base_logits(np.asarray(participation, dtype=bool)), dtype=torch.float64, requires_grad=True)
    w = torch.zeros((c, d), dtype=torch.float64, requires_grad=True)
    params = [a, w] if d else [a]
    opt = torch.optim.Adam(params, lr=lr)
    curve = []
    for _ in range(int(n_steps)):
        opt.zero_grad()
        logits = a[None, :] + (s @ w.T if d else 0.0)
        loss = -(wt * conditional_bernoulli_logpmf_torch(logits, part)).sum() + ridge * (w ** 2).sum()
        loss.backward()
        opt.step()
        curve.append(float(loss.detach()))
    return {"a": a.detach().numpy(), "W": w.detach().numpy(), "loss_curve": curve, "n_events": int(e),
            "ridge": ridge, "n_steps": int(n_steps)}


def grammar_logits(head: dict[str, Any], states: np.ndarray | None) -> np.ndarray:
    a = np.asarray(head["a"], dtype=np.float64)
    if states is None or head["W"].shape[1] == 0:
        return np.broadcast_to(a, (1, a.size)).copy() if states is None else np.tile(a, (np.asarray(states).shape[0], 1))
    return a[None, :] + np.asarray(states, dtype=np.float64) @ np.asarray(head["W"], dtype=np.float64).T


# --------------------------------------------------------------------------- rows, blocks, helpers
def _dev_rows(scaffold: Scaffold, horizon: float) -> np.ndarray:
    return np.sort(np.concatenate([scaffold.anchor_rows(p, horizon) for p in DEV_PHASES]))


def _anchor_segment(scaffold: Scaffold) -> np.ndarray:
    starts = scaffold.segment_bounds[:, 0]
    stops = scaffold.segment_bounds[:, 1]
    pos = np.searchsorted(starts, scaffold.t_anchor, side="right") - 1
    ok = (pos >= 0) & (scaffold.t_anchor < stops[np.clip(pos, 0, starts.size - 1)])
    return np.where(ok, pos, -1)


def _blocks(scaffold: Scaffold, rows: np.ndarray, horizon: float) -> np.ndarray:
    seg = _anchor_segment(scaffold)[rows]
    if (seg < 0).any():
        raise ValueError("a scored anchor lies outside every target segment")
    seg_start = {int(i): float(scaffold.segment_bounds[i, 0]) for i in np.unique(seg)}
    return block_ids_for_times(scaffold.t_anchor[rows], seg, seg_start, max(float(horizon), 1800.0))


def _standardise_columns(values: np.ndarray, train_rows: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    ref = v[train_rows]
    mean = ref.mean(axis=0)
    scale = ref.std(axis=0)
    scale = np.where(scale > 1e-9, scale, 1.0)
    return (v - mean) / scale


def _held_event_states(scaffold: Scaffold, anchor_state: np.ndarray) -> np.ndarray:
    """State of the last anchor at/before each event in its carry unit (NaN rows when none)."""

    idx = _held_anchor_index(scaffold)
    out = np.full((scaffold.n_events, anchor_state.shape[1]), np.nan)
    ok = idx >= 0
    out[ok] = anchor_state[idx[ok]]
    return out


def _monotone_codes(ids: np.ndarray) -> np.ndarray:
    ids = np.asarray(ids)
    change = np.r_[True, ids[1:] != ids[:-1]]
    return np.cumsum(change) - 1


def _real_tokens(scaffold: Scaffold) -> np.ndarray:
    part = scaffold.participation.astype(np.float64)
    return np.column_stack([part, np.log1p(part.sum(axis=1))])


# --------------------------------------------------------------------------- states per level
def _fixed_bank_states(innovations: np.ndarray, scaffold: Scaffold, train_rows: np.ndarray) -> np.ndarray:
    blocks = []
    for tau in BANK_TAUS:
        anchor, _pre = hidden_leaky_state(innovations, scaffold.event_times, scaffold.event_carry,
                                          scaffold.t_anchor, scaffold.anchor_carry, scaffold.last_event_pos, tau=tau)
        blocks.append(anchor)
    return _standardise_columns(np.concatenate(blocks, axis=1), train_rows)


def oracle_states(scaffold: Scaffold, data: SyntheticData, *, view: str, level: int,
                  horizon: float) -> dict[str, Any]:
    """Anchor states (A, d) offered to the head at Level 0 / 1; Level 2 trains its own."""

    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}")
    train_rows = scaffold.anchor_rows(TRAIN_PHASE, horizon)
    if level == 0:
        z = data.z_count if view in ("count", "count_profile") else data.z_grammar_anchor
        return {"anchor_state": np.asarray(z, dtype=np.float64)[:, None], "inputs": "true_state_at_anchors",
                "state_dim": 1}
    if level == 1:
        innovation_view = "count" if view == "count_profile" else view
        innov = data.innovations.get(innovation_view)
        if innov is None:
            innov = data.marks
        if innov is None:
            raise ValueError("Level 1 needs a true innovation (or a visible mark channel) for this view")
        s = _fixed_bank_states(np.asarray(innov, dtype=np.float64), scaffold, train_rows)
        return {"anchor_state": s, "inputs": "true_innovation_fixed_leaky_bank_300_1800_7200",
                "state_dim": int(s.shape[1])}
    raise ValueError("oracle_states covers levels 0 and 1 only; Level 2 trains an encoder")


# --------------------------------------------------------------------------- Level 2 trainer
class _EncoderBankReadout(nn.Module):
    def __init__(self, in_dim: int, view: str, n_contacts: int, base_logits: np.ndarray | None,
                 log_r_init: float | np.ndarray) -> None:
        super().__init__()
        self.view = view
        self.encoder = nn.Sequential(nn.Linear(in_dim, LEVEL2_HIDDEN), nn.GELU(), nn.Linear(LEVEL2_HIDDEN, LEVEL2_WRITE))
        self.register_buffer("taus", torch.tensor(list(BANK_TAUS), dtype=torch.float32))
        self.register_buffer("taus_full", torch.tensor(list(BANK_TAUS), dtype=torch.float32).repeat_interleave(LEVEL2_WRITE))
        d = len(BANK_TAUS) * LEVEL2_WRITE
        if view in ("count", "count_profile"):
            n_outputs = 3 if view == "count_profile" else 1
            self.c = nn.Parameter(torch.zeros(n_outputs))
            self.beta = nn.Parameter(torch.zeros(n_outputs, d))
            lr0 = np.asarray(log_r_init, dtype=np.float32).reshape(-1)
            if lr0.size == 1:
                lr0 = np.repeat(lr0, n_outputs)
            if lr0.size != n_outputs:
                raise ValueError("log_r_init does not match count outputs")
            self.log_r = nn.Parameter(torch.from_numpy(lr0.copy()))
        else:
            self.a = nn.Parameter(torch.tensor(np.asarray(base_logits, dtype=np.float32)))
            self.w = nn.Parameter(torch.zeros(n_contacts, d))

    def anchor_states(self, x: torch.Tensor, times: torch.Tensor, codes: torch.Tensor, t_anchor: torch.Tensor,
                      last: torch.Tensor, train_rows: torch.Tensor) -> torch.Tensor:
        u = torch.tanh(self.encoder(x))
        _pre, post = leaky_bank_trajectory(u, times, codes, self.taus, chunk_seconds=3600.0)
        s = _anchor_states_torch(post, times, t_anchor, last, self.taus_full).to(torch.float64)
        ref = s[train_rows]
        scale = ref.std(dim=0, unbiased=False)
        scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        return (s - ref.mean(dim=0)) / scale


def _train_level2(scaffold: Scaffold, data: SyntheticData, *, view: str, horizon: float, seed: int,
                  n_steps: int, log_r_h: float | np.ndarray) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    tokens = _real_tokens(scaffold)
    inputs = "real_tokens_only_mark_channel_hidden"
    if data.marks is not None:
        tokens = np.column_stack([data.marks, tokens])
        inputs = "visible_marks_plus_real_tokens"
    train_events = scaffold.event_rows(TRAIN_PHASE)
    x = torch.from_numpy(_standardise_columns(tokens, train_events)).to(torch.float32)
    times = torch.from_numpy(scaffold.event_times)
    codes = torch.from_numpy(_monotone_codes(scaffold.event_carry))
    t_anchor = torch.from_numpy(scaffold.t_anchor)
    last = torch.from_numpy(scaffold.last_event_pos)
    train_rows_np = scaffold.anchor_rows(TRAIN_PHASE, horizon)
    select_rows_np = scaffold.anchor_rows(SELECT_PHASE, horizon)
    train_rows = torch.from_numpy(train_rows_np)
    h_i = scaffold.horizon_index(horizon)
    base = _base_logits(data.participation[train_events]) if view == "grammar" else None
    model = _EncoderBankReadout(x.shape[1], view, scaffold.n_contacts, base, log_r_h)
    if view in ("count", "count_profile"):
        model.log_r.requires_grad_(False)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    if view == "count":
        y = torch.from_numpy(data.counts[int(horizon)].astype(np.float64))
        off = torch.from_numpy(np.asarray(scaffold.log_mu_h[int(horizon)], dtype=np.float64))
        part = None
    elif view == "count_profile":
        y = torch.from_numpy(data.count_profile.astype(np.float64))
        off = torch.from_numpy(np.asarray(data.log_mu_profile_h, dtype=np.float64))
        part = None
    else:
        part = torch.from_numpy(data.participation)

    pairs: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    if view == "grammar":
        for name, rows_np in (("train", train_rows_np), ("select", select_rows_np)):
            owner, ev, wt = _window_pairs(scaffold, rows_np, horizon)
            pairs[name] = (torch.from_numpy(rows_np[owner]), torch.from_numpy(ev),
                           torch.from_numpy(wt / max(wt.sum(), 1e-12)))

    def loss_on(rows: torch.Tensor, states: torch.Tensor, which: str) -> torch.Tensor:
        if view == "count":
            eta = off[rows] + model.c[0].to(torch.float64) + states[rows] @ model.beta[0].to(torch.float64)
            return C.nb_nll_torch(y[rows], eta, model.log_r[0].to(torch.float64)).mean()
        if view == "count_profile":
            eta = off[rows] + model.c.to(torch.float64) + states[rows] @ model.beta.to(torch.float64).T
            nll = C.nb_nll_torch(y[rows], eta, model.log_r.to(torch.float64))
            return nll.sum(dim=1).mean()
        anchor_rows, ev, wt = pairs[which]
        st = states[anchor_rows]                        # frozen anchor state for every event of its block
        logits = model.a.to(torch.float64)[None, :] + st @ model.w.to(torch.float64).T
        return -(wt * conditional_bernoulli_logpmf_torch(logits, part[ev])).sum()

    best, best_state, stale, curve = math.inf, None, 0, []
    for step in range(1, int(n_steps) + 1):
        model.train()
        opt.zero_grad()
        states = model.anchor_states(x, times, codes, t_anchor, last, train_rows)
        loss = loss_on(train_rows, states, "train")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 10 == 0 or step == int(n_steps):
            model.eval()
            with torch.no_grad():
                states = model.anchor_states(x, times, codes, t_anchor, last, train_rows)
                val = float(loss_on(torch.from_numpy(select_rows_np), states, "select")) if select_rows_np.size else float(loss)
            curve.append({"step": step, "train": float(loss), "select": val})
            if val < best - 1e-7:
                best, stale = val, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if step >= 50 and stale >= 5:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        states = model.anchor_states(x, times, codes, t_anchor, last, train_rows).numpy()
    out = {"anchor_state": states, "inputs": inputs, "state_dim": int(states.shape[1]), "curve": curve,
           "selected_step": int(min(curve, key=lambda r: r["select"])["step"]) if curve else None}
    if view in ("count", "count_profile"):
        c = model.c.detach().numpy().astype(np.float64)
        beta = model.beta.detach().numpy().astype(np.float64)
        log_r = model.log_r.detach().numpy().astype(np.float64)
        out["head"] = {
            "c": float(c[0]) if view == "count" else c,
            "beta": beta[0] if view == "count" else beta,
            "log_r": float(log_r[0]) if view == "count" else log_r,
            "dispersion_frozen": True,
        }
    else:
        out["head"] = {"a": model.a.detach().numpy().astype(np.float64), "W": model.w.detach().numpy().astype(np.float64)}
    return out


def _held_anchor_index(scaffold: Scaffold) -> np.ndarray:
    """Index of the last anchor at or before each event inside the same carry unit (-1 if none)."""

    out = np.full(scaffold.n_events, -1, dtype=np.int64)
    for unit in np.unique(scaffold.event_carry):
        a_idx = np.flatnonzero(scaffold.anchor_carry == unit)
        e_idx = np.flatnonzero(scaffold.event_carry == unit)
        if a_idx.size == 0 or e_idx.size == 0:
            continue
        order = np.argsort(scaffold.t_anchor[a_idx])
        a_idx = a_idx[order]
        pos = np.searchsorted(scaffold.t_anchor[a_idx], scaffold.event_times[e_idx], side="right") - 1
        ok = pos >= 0
        out[e_idx[ok]] = a_idx[pos[ok]]
    return out


def _window_pairs(scaffold: Scaffold, rows: np.ndarray, horizon: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(anchor row position, event index, weight = 1/N_future) for every event inside the rows' target windows.

    Training on these pairs with the anchor's *frozen* state is the exact
    counterpart of the block-average score (spec §8.2: the state is frozen at the
    anchor for the whole future block).
    """

    h_i = scaffold.horizon_index(horizon)
    lo, hi = scaffold.window_lo[rows, h_i], scaffold.window_hi[rows, h_i]
    n_future = hi - lo
    owner = np.repeat(np.arange(rows.size), n_future)
    events = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if rows.size else np.zeros(0, np.int64)
    weight = 1.0 / n_future[owner] if events.size else np.zeros(0)
    return owner, events.astype(np.int64), weight


# --------------------------------------------------------------------------- scoring
def _score_count(scaffold: Scaffold, data: SyntheticData, *, horizon: float, rows: np.ndarray,
                 pred_h: np.ndarray, pred_hs: np.ndarray, log_r_h: float, label: str,
                 pred_hs_free_dispersion: np.ndarray | None = None, log_r_hs_free: float | None = None,
                 seed: int) -> dict[str, Any]:
    y = data.counts[int(horizon)][rows]
    common = dict(subject=scaffold.subject, seed=seed, checkpoint_hash=f"oracle:{label}", split="dev_val+dev_test",
                  anchor_time=scaffold.t_anchor[rows], target=y, prediction_H=pred_h, prediction_H_plus_state=pred_hs,
                  mask=None, weight=None, eligibility="target_window_valid", evidence_label="DIAGNOSTIC_SYNTHETIC")
    shared = C.build_per_anchor_table(dispersion=log_r_h, dispersion_rule="shared", **common)
    sensitivity = None
    if pred_hs_free_dispersion is not None and log_r_hs_free is not None:
        per_arm_common = {**common, "prediction_H_plus_state": pred_hs_free_dispersion}
        per_arm = C.build_per_anchor_table(dispersion={"H": log_r_h, "H_plus_state": log_r_hs_free},
                                           dispersion_rule="per_arm", **per_arm_common)
        sensitivity = C.paired_gain(per_arm)["gain"]
    return {"table": shared, "gain_per_arm_dispersion": sensitivity}


def fit_count_profile_heads(y: np.ndarray, log_mu_h: np.ndarray, features: np.ndarray | None,
                            *, fixed_log_r: np.ndarray | None = None) -> dict[str, Any]:
    """Fit the three disjoint future-count bins; the primary state arm freezes every H dispersion."""

    yy = np.asarray(y, dtype=np.float64)
    off = np.asarray(log_mu_h, dtype=np.float64)
    if yy.ndim != 2 or yy.shape[1] != 3 or off.shape != yy.shape:
        raise ValueError("count profile target and H offset must both be (anchors, 3)")
    frozen = None if fixed_log_r is None else np.asarray(fixed_log_r, dtype=np.float64).reshape(-1)
    if frozen is not None and frozen.size != 3:
        raise ValueError("count profile requires exactly three frozen dispersions")
    heads = [fit_count_head(yy[:, j], off[:, j], features,
                            fixed_log_r=None if frozen is None else float(frozen[j]))
             for j in range(3)]
    return {
        "c": np.asarray([h["c"] for h in heads]),
        "beta": np.stack([h["beta"] for h in heads], axis=0),
        "log_r": np.asarray([h["log_r"] for h in heads]),
        "dispersion_frozen": bool(frozen is not None),
        "converged": bool(all(h["converged"] for h in heads)),
        "n_rows": int(yy.shape[0]),
    }


def predict_count_profile(head: dict[str, Any], log_mu_h: np.ndarray,
                          features: np.ndarray | None) -> np.ndarray:
    off = np.asarray(log_mu_h, dtype=np.float64)
    c = np.asarray(head["c"], dtype=np.float64)
    beta = np.asarray(head["beta"], dtype=np.float64)
    if features is None or beta.shape[1] == 0:
        return off + c[None, :]
    return off + c[None, :] + np.asarray(features, dtype=np.float64) @ beta.T


def _score_count_profile(scaffold: Scaffold, data: SyntheticData, *, rows: np.ndarray,
                         pred_h: np.ndarray, pred_hs: np.ndarray, log_r_h: np.ndarray,
                         label: str, seed: int, pred_hs_free: np.ndarray | None = None,
                         log_r_hs_free: np.ndarray | None = None) -> dict[str, Any]:
    """Joint proper score for [0-5, 5-15, 15-30] min counts (sum of three NB NLLs)."""

    y = np.asarray(data.count_profile[rows], dtype=np.int64)
    lr_h = np.asarray(log_r_h, dtype=np.float64).reshape(1, 3)
    nll_h = C.nb_nll(y, pred_h, lr_h).sum(axis=1)
    nll_hs = C.nb_nll(y, pred_hs, lr_h).sum(axis=1)
    extra: dict[str, np.ndarray] = {}
    if pred_hs_free is not None and log_r_hs_free is not None:
        extra["H_plus_state_free_dispersion"] = C.nb_nll(
            y, pred_hs_free, np.asarray(log_r_hs_free, dtype=np.float64).reshape(1, 3)).sum(axis=1)
    table = C.build_per_anchor_table_from_scores(
        subject=scaffold.subject, seed=seed, checkpoint_hash=f"oracle:{label}", split="dev_val+dev_test",
        anchor_time=scaffold.t_anchor[rows], target=y.sum(axis=1),
        per_anchor_nll={"H": nll_h, "H_plus_state": nll_hs},
        score_family="nb_disjoint_count_profile", mask=None, weight=None,
        eligibility="1800s_target_window_valid", evidence_label="DIAGNOSTIC_SYNTHETIC",
        extra_nll=extra, prediction_H=pred_h, prediction_H_plus_state=pred_hs,
        dispersion=np.broadcast_to(np.asarray(log_r_h, dtype=np.float64), y.shape).copy(),
        dispersion_rule="shared_H_per_bin")
    sensitivity = None
    if extra:
        sensitivity = C.paired_gain(table, control="H", treated="H_plus_state_free_dispersion")["gain"]
    return {"table": table, "gain_per_arm_dispersion": sensitivity}


def _grammar_block_scores(scaffold: Scaffold, data: SyntheticData, *, horizon: float, rows: np.ndarray,
                          logits_h: np.ndarray, logits_hs: np.ndarray) -> dict[str, np.ndarray]:
    """Block-average and first-future-event conditional subset NLL per anchor (NaN when N_future = 0)."""

    h_i = scaffold.horizon_index(horizon)
    lo, hi = scaffold.window_lo[rows, h_i], scaffold.window_hi[rows, h_i]
    n_future = hi - lo
    ev = np.concatenate([np.arange(a, b) for a, b in zip(lo, hi)]) if rows.size else np.zeros(0, np.int64)
    owner = np.repeat(np.arange(rows.size), n_future)
    out: dict[str, np.ndarray] = {"n_future": n_future}
    for name, logits in (("H", logits_h), ("H_plus_state", logits_hs)):
        nll = C.conditional_subset_nll(logits[owner], data.participation[ev]) if ev.size else np.zeros(0)
        block = np.full(rows.size, np.nan)
        first = np.full(rows.size, np.nan)
        if ev.size:
            sums = np.bincount(owner, weights=nll, minlength=rows.size)
            block[n_future > 0] = sums[n_future > 0] / n_future[n_future > 0]
            first_pos = np.r_[0, np.cumsum(n_future)[:-1]]
            first[n_future > 0] = nll[first_pos[n_future > 0]]
        out[f"block_{name}"] = block
        out[f"first_{name}"] = first
    return out


def _bootstrap(table: dict[str, Any], blocks: np.ndarray, seed: int) -> dict[str, Any]:
    c = table["per_anchor_NLL_H"]
    t = table["per_anchor_NLL_H_plus_state"]
    used = table["mask"] & np.isfinite(c) & np.isfinite(t)
    gain_rows = (c - t)[used]
    boot = block_bootstrap_mean_ci(gain_rows, blocks[used], n_boot=BOOTSTRAP_RESAMPLES, seed=seed)
    block_means: list[float] = []
    if used.any():
        _u, inv = np.unique(blocks[used], return_inverse=True)
        block_means = (np.bincount(inv, weights=gain_rows) / np.bincount(inv)).tolist()
    return {"ci_lower": boot["lower"], "ci_upper": boot["upper"], "n_blocks": boot["n_blocks"],
            "n_rows_used": int(used.sum()), "block_gain_means": block_means}


def run_level(scaffold: Scaffold, data: SyntheticData, *, view: str, level: int, horizon: float, seed: int = 0,
              n_steps: int = 300) -> dict[str, Any]:
    if view not in VIEWS or level not in LEVELS:
        raise ValueError("unknown view or level")
    h_key = int(horizon)
    train_rows = scaffold.anchor_rows(TRAIN_PHASE, horizon)
    dev_rows = _dev_rows(scaffold, horizon)
    train_events = scaffold.event_rows(TRAIN_PHASE)
    blocks = _blocks(scaffold, dev_rows, horizon)
    log_mu_h = np.asarray(scaffold.log_mu_h[h_key], dtype=np.float64)
    result: dict[str, Any] = {"view": view, "level": level, "horizon_seconds": float(horizon), "seed": int(seed),
                              "dgp": data.as_meta(), "n_train_rows": int(train_rows.size), "n_dev_rows": int(dev_rows.size)}
    if view == "count_profile":
        y = np.asarray(data.count_profile, dtype=np.int64)
        profile_h = np.asarray(data.log_mu_profile_h, dtype=np.float64)
        head_h = fit_count_profile_heads(y[train_rows], profile_h[train_rows], None)
        pred_h = predict_count_profile(head_h, profile_h[dev_rows], None)
        if level in (0, 1):
            st = oracle_states(scaffold, data, view=view, level=level, horizon=horizon)
            s = st["anchor_state"]
            head = fit_count_profile_heads(y[train_rows], profile_h[train_rows], s[train_rows],
                                           fixed_log_r=head_h["log_r"])
            pred_hs = predict_count_profile(head, profile_h[dev_rows], s[dev_rows])
            result.update({"inputs": st["inputs"], "state_dim": st["state_dim"]})
        else:
            trained = _train_level2(scaffold, data, view=view, horizon=horizon, seed=seed, n_steps=n_steps,
                                    log_r_h=head_h["log_r"])
            head = trained["head"]
            s = trained["anchor_state"]
            pred_hs = predict_count_profile(head, profile_h[dev_rows], s[dev_rows])
            result.update({"inputs": trained["inputs"], "state_dim": trained["state_dim"],
                           "training": {"curve": trained["curve"], "selected_step": trained["selected_step"]}})
        head_free = fit_count_profile_heads(y[train_rows], profile_h[train_rows], s[train_rows])
        pred_hs_free = predict_count_profile(head_free, profile_h[dev_rows], s[dev_rows])
        scored = _score_count_profile(
            scaffold, data, rows=dev_rows, pred_h=pred_h, pred_hs=pred_hs, log_r_h=head_h["log_r"],
            label=f"L{level}", seed=seed, pred_hs_free=pred_hs_free, log_r_hs_free=head_free["log_r"])
        table = scored["table"]
        result["gain_per_arm_dispersion"] = scored["gain_per_arm_dispersion"]
        result["head"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in head.items()}
        result["head_H"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in head_h.items()}
        result["head_free_dispersion_sensitivity"] = {
            k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in head_free.items()
        }
        result["scoring"] = "sum_nb_nll_three_disjoint_bins_shared_H_dispersions"
        result["profile_edges_seconds"] = list(data.as_meta()["count_profile_edges_seconds"])
        result["truth_summary"] = {"has_state": data.has_state["count"],
                                   "z_dev_std": float(np.std(data.z_count[dev_rows]))}
    elif view == "count":
        y = data.counts[h_key]
        head_h = fit_count_head(y[train_rows], log_mu_h[train_rows], None)
        pred_h = predict_count_head(head_h, log_mu_h[dev_rows], None)
        if level in (0, 1):
            st = oracle_states(scaffold, data, view=view, level=level, horizon=horizon)
            s = st["anchor_state"]
            head = fit_count_head(y[train_rows], log_mu_h[train_rows], s[train_rows], fixed_log_r=head_h["log_r"])
            pred_hs = predict_count_head(head, log_mu_h[dev_rows], s[dev_rows])
            result.update({"inputs": st["inputs"], "state_dim": st["state_dim"]})
        else:
            trained = _train_level2(scaffold, data, view=view, horizon=horizon, seed=seed, n_steps=n_steps,
                                    log_r_h=head_h["log_r"])
            head = trained["head"]
            s = trained["anchor_state"]
            pred_hs = log_mu_h[dev_rows] + head["c"] + s[dev_rows] @ head["beta"]
            result.update({"inputs": trained["inputs"], "state_dim": trained["state_dim"],
                           "training": {"curve": trained["curve"], "selected_step": trained["selected_step"]}})
        head_free = fit_count_head(y[train_rows], log_mu_h[train_rows], s[train_rows])
        pred_hs_free = predict_count_head(head_free, log_mu_h[dev_rows], s[dev_rows])
        scored = _score_count(scaffold, data, horizon=horizon, rows=dev_rows, pred_h=pred_h, pred_hs=pred_hs,
                              log_r_h=head_h["log_r"], label=f"L{level}", seed=seed,
                              pred_hs_free_dispersion=pred_hs_free, log_r_hs_free=head_free["log_r"])
        table = scored["table"]
        result["gain_per_arm_dispersion"] = scored["gain_per_arm_dispersion"]
        result["head"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in head.items()}
        result["head_H"] = {**head_h, "beta": head_h["beta"].tolist()}
        result["head_free_dispersion_sensitivity"] = {
            **head_free, "beta": head_free["beta"].tolist()
        }
        result["scoring"] = "nb_count_nll_shared_H_dispersion"
        result["truth_summary"] = {"has_state": data.has_state["count"],
                                   "z_dev_std": float(np.std(data.z_count[dev_rows]))}
    else:
        part = data.participation
        owner, pair_events, pair_w = _window_pairs(scaffold, train_rows, horizon)   # same rows and budget for both arms
        head_h = fit_grammar_head(part[pair_events], None, n_steps=n_steps, seed=seed, weights=pair_w)
        logits_h = grammar_logits(head_h, np.zeros((dev_rows.size, 0)))
        if level in (0, 1):
            st = oracle_states(scaffold, data, view=view, level=level, horizon=horizon)
            s = st["anchor_state"]
            head = fit_grammar_head(part[pair_events], s[train_rows][owner], n_steps=n_steps, seed=seed, weights=pair_w)
            result.update({"inputs": st["inputs"], "state_dim": st["state_dim"]})
        else:
            trained = _train_level2(scaffold, data, view=view, horizon=horizon, seed=seed, n_steps=n_steps, log_r_h=0.0)
            head, s = trained["head"], trained["anchor_state"]
            result.update({"inputs": trained["inputs"], "state_dim": trained["state_dim"],
                           "training": {"curve": trained["curve"], "selected_step": trained["selected_step"]}})
        logits_hs = grammar_logits(head, s[dev_rows])
        scores = _grammar_block_scores(scaffold, data, horizon=horizon, rows=dev_rows, logits_h=logits_h, logits_hs=logits_hs)
        table = C.build_per_anchor_table_from_scores(
            subject=scaffold.subject, seed=seed, checkpoint_hash=f"oracle:L{level}", split="dev_val+dev_test",
            anchor_time=scaffold.t_anchor[dev_rows], target=scores["n_future"],
            per_anchor_nll={"H": scores["block_H"], "H_plus_state": scores["block_H_plus_state"]},
            score_family="conditional_subset_nll", mask=scores["n_future"] > 0, weight=None,
            eligibility="n_future_positive", evidence_label="DIAGNOSTIC_SYNTHETIC",
            extra_nll={"first_H": scores["first_H"], "first_H_plus_state": scores["first_H_plus_state"]})
        first = C.paired_gain(table, control="first_H", treated="first_H_plus_state")
        result["gain_first_future_event"] = first["gain"]
        result["head"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in head.items() if k != "loss_curve"}
        result["scoring"] = "block_average_conditional_subset_nll"
        result["truth_summary"] = {"has_state": data.has_state["grammar"],
                                   "z_dev_std": float(np.std(data.z_grammar_anchor[dev_rows]))}
    gain = C.paired_gain(table)
    boot = _bootstrap(table, blocks, seed)
    result.update({"gain": gain["gain"], "mean_nll_H": gain["mean_nll_control"],
                   "mean_nll_H_plus_state": gain["mean_nll_treated"], **boot,
                   "detected": bool(boot["ci_lower"] is not None and boot["ci_lower"] > DETECTION_FLOOR_NATS),
                   "table": table, "table_meta": table["meta"]})
    return result


def summarise_level(result: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in result.items() if k not in ("table",)}


def run_cascade(scaffold: Scaffold, data: SyntheticData, *, view: str, horizon: float, seed: int = 0,
                n_steps: int = 300, levels: tuple[int, ...] = LEVELS) -> dict[str, Any]:
    results = [run_level(scaffold, data, view=view, level=level, horizon=horizon, seed=seed, n_steps=n_steps)
               for level in levels]
    truth_key = "count" if view == "count_profile" else view
    truth = bool(data.has_state[truth_key])
    out: dict[str, Any] = {"view": view, "horizon_seconds": float(horizon), "dgp": data.as_meta(),
                           "truth_has_state": truth, "levels": [summarise_level(r) for r in results],
                           "oracle_gain_level0": next((r["gain"] for r in results if r["level"] == 0), None)}
    if truth:
        failure = "none"
        for r in results:
            if not r["detected"]:
                failure = FAILURE_BY_LEVEL[r["level"]]
                break
        out["failure_location"] = failure
    else:
        out["failure_location"] = "not_applicable_no_state"
        out["false_positive_by_level"] = {str(r["level"]): bool(r["detected"]) for r in results}
    return out
