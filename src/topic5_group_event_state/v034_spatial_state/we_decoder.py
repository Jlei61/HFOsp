"""Frozen wiring-economy tissue RNN as the S_P contact-sequence decoder.

The decoder is the v0.5 leaky tissue RNN (``LBSSModel``: local recurrence on a
patient-specific tissue plane plus learned nonlocal shortcuts), retrained by
``scripts/train_topic5_lbss_unit_v0_2.py`` on caches whose ``split`` follows
*recorded time* (everything at or after the 70 % boundary unused).  Inside one
event the decoder starts from a zero tissue state; here the cross-event state
enters exactly there, as the *initial tissue state* ``h0`` of the event:

    h0 = b + A(s)        b: state-free adapter (fitted on TRAIN, no state)
                         A: low-rank map from the standardised cross-event state

so the nested ladder is  zero (frozen decoder) -> b (train-mean adapter)
-> b + A(s) (learned state), with the constant / shifted / random arms
evaluated on the same per-event scores.

Per-event scores follow the v0.5 objective: next-rank multi-label BCE over the
still-available contacts plus STOP BCE, and the cardinality-conditioned
contact NLL that v0.5 reports as ``contact_nll``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from src.topic5_lbss_rnn_v0_2 import LBSSConfig, LBSSModel, build_pool_contract
from src.topic5_wiring_economy_rnn import NEG_INF, build_event_tensors
from src.topic5_group_event_state.v033_training_lab.sg_o2 import GrammarPairs

DECODER_DEFAULTS = {"density": 0.10, "added_fraction": 0.10, "r_local_multiplier": 2.0,
                    "state_dim": 1, "stop_weight": 1.0}
TIME_TOLERANCE_SECONDS = 1e-3


# ---------------------------------------------------------------- loading
@dataclass
class FrozenDecoderBundle:
    model: LBSSModel
    ranks: np.ndarray               # (N_cache, C) int16 dense ranks, -1 = not participating
    event_abs_time: np.ndarray      # (N_cache,) float64
    contact_names: tuple[str, ...]
    unit_dir: Path
    cache_dir: Path
    metrics: dict[str, Any]
    split: np.ndarray               # cache split labels (0/1/2/-1) under the recorded-time rule


def load_frozen_decoder(unit_dir: Path, cache_dir: Path, *, device: torch.device,
                        cfg: Mapping[str, Any] = DECODER_DEFAULTS) -> FrozenDecoderBundle:
    unit_dir, cache_dir = Path(unit_dir), Path(cache_dir)
    metrics = json.loads((unit_dir / "metrics.json").read_text())
    if not (unit_dir / "DONE.json").exists() or not metrics.get("best_checkpoint_eligible", False):
        raise PermissionError(f"decoder unit is not a finished, mask-frozen checkpoint: {unit_dir}")
    if metrics.get("target_values_read") is not False:
        raise PermissionError("decoder unit read target values")
    plane = np.load(cache_dir / "plane.npz", allow_pickle=False)
    provenance = json.loads((cache_dir / "provenance.json").read_text())
    raw = np.load(cache_dir / "events_raw.npz", allow_pickle=False)
    events = np.load(cache_dir / "events.npz", allow_pickle=False)
    pools = build_pool_contract(plane["D_mm"], float(cfg["density"]), float(cfg["added_fraction"]),
                                float(cfg["r_local_multiplier"]))
    n_contacts = int(provenance["n_contacts"] if "n_contacts" in provenance else provenance["n_joint_contacts"])
    model = LBSSModel(LBSSConfig(
        arm=str(metrics["arm"]), n_contacts=n_contacts, n_nodes=int(provenance["n_nodes"]),
        observation_operator=plane["H"], node_distance_mm=plane["D_mm"],
        local_mask=pools.local_mask, extra_local_pool=pools.extra_local_pool,
        nonlocal_pool=pools.nonlocal_pool, k_added=pools.k_added,
        seed=int(metrics["seed"]), state_dim=int(cfg["state_dim"]),
    ))
    state = torch.load(unit_dir / "weights.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.freeze_mask()
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    ranks = np.asarray(raw["ranks"], dtype=np.int16)
    times = np.asarray(raw["event_abs_time"], dtype=np.float64)
    if not np.array_equal(times, np.asarray(events["event_abs_time"], dtype=np.float64)):
        raise ValueError("events.npz and events_raw.npz are not aligned")
    return FrozenDecoderBundle(
        model=model, ranks=ranks, event_abs_time=times,
        contact_names=tuple(str(v) for v in raw["contact_names"]), unit_dir=unit_dir, cache_dir=cache_dir,
        metrics=metrics, split=np.asarray(events["split"], dtype=np.int8),
    )


def align_events(our_event_time: np.ndarray, cache_event_time: np.ndarray,
                 tolerance: float = TIME_TOLERANCE_SECONDS) -> np.ndarray:
    """Index into the decoder cache for each of our events, or -1 when the decoder never saw it."""

    ours = np.asarray(our_event_time, dtype=np.float64)
    cache = np.asarray(cache_event_time, dtype=np.float64)
    order = np.argsort(cache, kind="stable")
    sorted_cache = cache[order]
    pos = np.searchsorted(sorted_cache, ours)
    out = np.full(ours.shape, -1, dtype=np.int64)
    for cand in (pos - 1, pos):
        ok = (cand >= 0) & (cand < sorted_cache.size)
        idx = np.where(ok, cand, 0)
        hit = ok & (np.abs(sorted_cache[idx] - ours) <= tolerance)
        out[hit] = order[idx[hit]]
    return out


# ------------------------------------------------------------ per-event NLL
def forward_with_h0(model: LBSSModel, x: Tensor, recruited: Tensor, valid: Tensor,
                    h0: Tensor | None) -> tuple[Tensor, Tensor]:
    """``WEModel.forward`` with the event's initial tissue state supplied (zeros when ``None``)."""

    b, steps, _ = x.shape
    h = torch.zeros(b, model.n_nodes * model.state_dim, device=x.device) if h0 is None else h0
    logits, stops = [], []
    denom = max(1, model.n_contacts - 1)
    for t in range(steps):
        h = model._step(h, x[:, t])
        logits.append(model._readout(h))
        t_norm = torch.full((b,), t / denom, device=x.device)
        stops.append(model._stop(h, t_norm, recruited[:, t].mean(-1)))
    return torch.stack(logits, 1), torch.stack(stops, 1)


def per_event_scores(contact_logits: Tensor, stop_logits: Tensor, batch: Mapping[str, Tensor],
                     stop_weight: float = 1.0) -> dict[str, Tensor]:
    """Per-event versions of the v0.5 objective and of ``cardinality_conditioned_nll``."""

    available = batch["available"]
    target = batch["target"] * available.float()
    predict = batch["valid"] & ~batch["is_last"]
    masked = contact_logits.masked_fill(~available, NEG_INF)
    per_contact = F.binary_cross_entropy_with_logits(masked, target, reduction="none") * available.float()
    per_step = per_contact.sum(-1) / available.float().sum(-1).clamp_min(1.0)
    next_bce = (per_step * predict.float()).sum(-1) / predict.float().sum(-1).clamp_min(1.0)
    stop_bce = (F.binary_cross_entropy_with_logits(stop_logits, batch["is_last"].float(), reduction="none")
                * batch["valid"].float()).sum(-1) / batch["valid"].float().sum(-1).clamp_min(1.0)
    log_prob = torch.log_softmax(masked, dim=-1)
    chosen = (log_prob * target).sum(-1)
    contact_step = -chosen / target.sum(-1).clamp_min(1.0)
    contact_nll = (contact_step * predict.float()).sum(-1) / predict.float().sum(-1).clamp_min(1.0)
    return {"grammar": next_bce + float(stop_weight) * stop_bce, "next_bce": next_bce, "stop_bce": stop_bce,
            "contact_nll": contact_nll, "n_predict": predict.float().sum(-1)}


# ---------------------------------------------------------------- scorer
class WEStateScorer(nn.Module):
    """Frozen tissue decoder; ``h0 = bias + low-rank(state)`` is the only trainable path."""

    def __init__(self, bundle: FrozenDecoderBundle, *, state_dim: int, rank: int, stop_weight: float = 1.0) -> None:
        super().__init__()
        self.decoder = bundle.model
        self.stop_weight = float(stop_weight)
        width = int(self.decoder.n_nodes * self.decoder.state_dim)
        rank = int(min(rank, state_dim, width))
        self.h0_bias = nn.Parameter(torch.zeros(width))
        self.to_h0 = nn.Sequential(nn.Linear(int(state_dim), rank, bias=False), nn.Linear(rank, width, bias=False))
        nn.init.normal_(self.to_h0[0].weight, std=0.02)
        nn.init.normal_(self.to_h0[1].weight, std=1e-3)

    def train(self, mode: bool = True):
        super().train(mode)
        self.decoder.eval()
        return self

    def initial_state(self, state: Tensor | None, *, use_bias: bool, use_state: bool) -> Tensor | None:
        parts = []
        if use_bias:
            parts.append(self.h0_bias)
        if use_state and state is not None:
            parts.append(self.to_h0(state.to(torch.float32)))
        if not parts:
            return None
        h0 = parts[0] if parts[0].dim() == 2 else parts[0].unsqueeze(0)
        for extra in parts[1:]:
            h0 = h0 + (extra if extra.dim() == 2 else extra.unsqueeze(0))
        if h0.shape[0] == 1 and state is not None:
            h0 = h0.expand(state.shape[0], -1)
        return h0

    def scores(self, batch: Mapping[str, Tensor], state: Tensor | None, *, use_bias: bool = True,
               use_state: bool = True) -> dict[str, Tensor]:
        h0 = self.initial_state(state, use_bias=use_bias, use_state=use_state)
        if h0 is not None and h0.shape[0] == 1 and batch["x"].shape[0] != 1:
            h0 = h0.expand(batch["x"].shape[0], -1)
        logits, stops = forward_with_h0(self.decoder, batch["x"], batch["recruited"], batch["valid"], h0)
        return per_event_scores(logits, stops, batch, self.stop_weight)


# ------------------------------------------------------------- pair utils
def restrict_pairs(pairs: GrammarPairs, keep_event: np.ndarray) -> GrammarPairs:
    """Drop pairs whose event the decoder cannot score; rebuild equal-anchor weights."""

    keep = np.asarray(keep_event, dtype=bool)[pairs.pair_event]
    pa = pairs.pair_anchor[keep]
    pe = pairs.pair_event[keep]
    if pa.size == 0:
        raise ValueError("no scorable future events remain after decoder alignment")
    kept_anchor, new_anchor = np.unique(pa, return_inverse=True)
    counts = np.bincount(new_anchor, minlength=kept_anchor.size).astype(np.float64)
    weight = 1.0 / (kept_anchor.size * counts[new_anchor])
    return GrammarPairs(anchor_rows=pairs.anchor_rows[kept_anchor], pair_anchor=new_anchor.astype(np.int64),
                        pair_event=pe, pair_weight=weight).validate()


def split_pairs_by_time(pairs: GrammarPairs, anchor_time: np.ndarray, fraction_tail: float
                        ) -> tuple[GrammarPairs, GrammarPairs]:
    """Chronological split of a pair set by anchor time: head for fitting, tail for rolling inner-validation."""

    order = np.argsort(anchor_time[pairs.anchor_rows], kind="stable")
    n_tail = max(1, int(round(float(fraction_tail) * order.size)))
    tail = np.zeros(order.size, dtype=bool)
    tail[order[-n_tail:]] = True

    def take(mask_local: np.ndarray) -> GrammarPairs:
        keep = mask_local[pairs.pair_anchor]
        pa = pairs.pair_anchor[keep]
        kept, new = np.unique(pa, return_inverse=True)
        counts = np.bincount(new, minlength=kept.size).astype(np.float64)
        return GrammarPairs(anchor_rows=pairs.anchor_rows[kept], pair_anchor=new.astype(np.int64),
                            pair_event=pairs.pair_event[keep], pair_weight=1.0 / (kept.size * counts[new])).validate()

    return take(~tail), take(tail)


def event_batch(tensors: Mapping[str, Tensor], cache_index: Tensor) -> dict[str, Tensor]:
    return {k: v[cache_index] for k, v in tensors.items()}


def decoder_tensors(bundle: FrozenDecoderBundle, device: torch.device) -> dict[str, Tensor]:
    t = build_event_tensors(bundle.ranks)
    return {k: v.to(device) for k, v in t.items()}
