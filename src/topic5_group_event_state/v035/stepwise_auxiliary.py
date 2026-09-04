"""Per-step lag, multiband and waveform heads on the frozen tissue decoder.

The auxiliary heads are first trained without cross-event state.  They are then
frozen, and separate zero-initialised q/m adapters modulate the tissue hidden
state after every observed tied group.  Thus a flat state contrast cannot be
blamed on an untrained endpoint head, and state never enters only as h0.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v034_spatial_state.we_decoder import FrozenDecoderBundle, decoder_tensors

from .contracts import FORMAT_PREFIX, atomic_json, seed_all
from .full_mark_state import FullMarkData
from .stepwise_decoder import DynamicStepAdapter, StepwiseAdapterConfig, StepwiseConditionedDecoder


OFFSETS = (1, 5, 20)
FEATURE_GROUPS = {
    "continuous_lag": (0, 1), "multiband_energy": (1, 6), "multiband_peak_time": (6, 11),
    "cross_band_lag": (11, 21), "waveform_morphology": (21, 36),
}


@dataclass(frozen=True)
class AuxiliaryConfig:
    head_rank: int = 32
    adapter_rank: int = 8
    head_steps: int = 600
    q_steps: int = 600
    m_steps: int = 900
    validate_every: int = 50
    patience: int = 5
    batch_events: int = 96
    head_lr: float = 1e-3
    adapter_lr: float = 5e-4
    seed: int = 20260903


class AuxiliaryHead(nn.Module):
    def __init__(self, hidden: int, contacts: int, features: int, rank: int) -> None:
        super().__init__()
        r = min(rank, hidden)
        self.net = nn.Sequential(nn.Linear(hidden, r), nn.GELU(), nn.Linear(r, contacts * features))
        self.contacts, self.features = contacts, features

    def forward(self, hidden: Tensor) -> Tensor:
        return self.net(hidden).reshape(*hidden.shape[:2], self.contacts, self.features)


def _wave_summary(x: np.ndarray) -> np.ndarray:
    mean = x.mean(-1); rms = np.sqrt(np.mean(np.square(x), axis=-1)); peak = np.max(np.abs(x), axis=-1)
    peak_t = np.argmax(np.abs(x), axis=-1).astype(np.float32) / max(x.shape[-1] - 1, 1)
    line = np.mean(np.abs(np.diff(x, axis=-1)), axis=-1)
    return np.stack((mean, rms, peak, peak_t, line), axis=-1).reshape(x.shape[0], x.shape[1], -1)


def _wave_summary_rows(array: Any, rows: np.ndarray, *, batch_rows: int = 128) -> np.ndarray:
    """Summarise waveform rows without materialising the full N x C x 3 x T tensor."""
    output: list[np.ndarray] = []
    for lo in range(0, rows.size, batch_rows):
        chunk = np.asarray(array[rows[lo:lo + batch_rows]], dtype=np.float32)
        output.append(_wave_summary(np.nan_to_num(chunk)))
    if not output:
        return np.empty((0, int(array.shape[1]), int(array.shape[2]) * 5), dtype=np.float32)
    return np.concatenate(output, axis=0).astype(np.float32, copy=False)


def build_targets(data: FullMarkData, bundle: FrozenDecoderBundle) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
    seq = data.seq; rows = seq.order[data.source_position]
    source_names = [str(row["lagpat_label"]) for row in seq.index["contacts"]]
    lookup = {name: i for i, name in enumerate(source_names)}
    if any(name not in lookup for name in bundle.contact_names):
        missing = [name for name in bundle.contact_names if name not in lookup]
        raise ValueError(f"decoder contacts missing from full-event stream: {missing}")
    contact = np.asarray([lookup[name] for name in bundle.contact_names], dtype=np.int64)
    part = np.asarray(seq.arrays["participation"][rows], dtype=bool)[:, contact]
    ok = np.asarray(seq.arrays["contact_ok"][rows], dtype=bool)[:, contact]
    delay = np.asarray(seq.arrays["relative_delay"][rows], dtype=np.float32)[:, contact, None]
    band = np.asarray(seq.arrays["band_features"][rows], dtype=np.float32)[:, contact]
    energy, peak = band[..., 2], band[..., 0]
    cross = np.asarray(seq.arrays["cross_band_lag"][rows], dtype=np.float32)[:, contact]
    wave = _wave_summary_rows(seq.arrays["waveform"], rows)[:, contact]
    values = np.concatenate((delay, energy, peak, cross, wave), axis=-1)
    band_ok = np.asarray(seq.index["band_available"], dtype=bool)
    common = part & ok
    valid = np.concatenate((
        common[..., None],
        common[..., None] & band_ok[None, None],
        common[..., None] & band_ok[None, None],
        np.broadcast_to(common[..., None], cross.shape),
        np.broadcast_to(common[..., None], wave.shape),
    ), axis=-1)
    valid &= np.isfinite(values)
    return np.nan_to_num(values).astype(np.float32), valid, dict(FEATURE_GROUPS)


def _scale_targets(values: np.ndarray, valid: np.ndarray, fit_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.where(valid[fit_rows], values[fit_rows], np.nan)
    centre = np.nanmedian(selected, axis=(0, 1))
    centre = np.where(np.isfinite(centre), centre, 0.0)
    mad = np.nanmedian(np.abs(selected - centre[None, None]), axis=(0, 1))
    scale = np.where(np.isfinite(mad) & (mad > 1e-6), 1.4826 * mad, 1.0)
    return ((values - centre[None, None]) / scale[None, None]).astype(np.float32), centre, scale


def _loss(pred: Tensor, target: Tensor, target_valid: Tensor, next_group: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
    mask = target_valid[:, None] & (next_group[..., None] > 0)
    error = (pred - target[:, None]).square()
    groups = {}
    terms = []
    for name, (lo, hi) in FEATURE_GROUPS.items():
        current = mask[..., lo:hi]
        value = (error[..., lo:hi] * current).sum() / current.sum().clamp_min(1)
        groups[name] = value; terms.append(value)
    return torch.stack(terms).mean(), groups


def _batch(data: FullMarkData, tensors: dict[str, Tensor], values: Tensor, valid: Tensor,
           source_rows: np.ndarray, offset_j: int, device: torch.device):
    target = data.next_index[source_rows, offset_j]
    good = (target >= 0) & (data.decoder_index[np.maximum(target, 0)] >= 0)
    source = source_rows[good]; target = target[good]
    cache = torch.as_tensor(data.decoder_index[target], dtype=torch.long, device=device)
    batch = {k: v[cache] for k, v in tensors.items()}
    return source, target, batch, values[torch.as_tensor(target, device=device)], valid[torch.as_tensor(target, device=device)]


def _open_loop_state(state: Tensor, mean: Tensor, taus: Tensor, data: FullMarkData,
                     source: np.ndarray, target: np.ndarray) -> Tensor:
    """Evolve an anchor post-state to its target without future event writes."""
    src = torch.as_tensor(source, dtype=torch.long, device=state.device)
    dt = torch.as_tensor(data.event_time[target] - data.event_time[source],
                         dtype=torch.float32, device=state.device).unsqueeze(-1)
    return mean + (state[src] - mean) * torch.exp(-dt / taus)


def _evaluate(wrapper: StepwiseConditionedDecoder, head: AuxiliaryHead, data: FullMarkData,
              tensors: dict[str, Tensor], values: Tensor, valid: Tensor, state: Tensor,
              mean: Tensor, taus: Tensor,
              rows: np.ndarray, *, use_q: bool, use_m: bool, q_adapter: DynamicStepAdapter,
              m_adapter: DynamicStepAdapter, shifted: Tensor | None = None) -> dict[str, Any]:
    totals = {o: {name: [] for name in FEATURE_GROUPS} for o in OFFSETS}
    wrapper.eval(); head.eval()
    with torch.no_grad():
        for j, offset in enumerate(OFFSETS):
            for lo in range(0, rows.size, 256):
                source, target, batch, y, yvalid = _batch(data, tensors, values, valid, rows[lo:lo+256], j, state.device)
                if source.size == 0: continue
                q = torch.as_tensor(data.q_context[source], dtype=torch.float32, device=state.device)
                source_state = state if shifted is None else shifted
                m = _open_loop_state(source_state, mean, taus, data, source, target)
                hidden = wrapper.hidden_sequence(batch["x"], batch["recruited"], batch["valid"],
                                                  q if use_q else None, use_static=False, use_dynamic=use_q,
                                                  extra_context=m if use_m else None,
                                                  extra_adapter=m_adapter if use_m else None)
                pred = head(hidden); _total, groups = _loss(pred, y, yvalid, batch["target"])
                for name, value in groups.items(): totals[offset][name].append(float(value.cpu()))
    return {f"next_{offset}_events": {name: (float(np.mean(v)) if v else None) for name, v in by.items()}
            for offset, by in totals.items()}


def run_auxiliary_heads(data: FullMarkData, bundle: FrozenDecoderBundle, trajectory_path: Path,
                        config: AuxiliaryConfig, *, device: torch.device, out_dir: Path,
                        overwrite: bool = False) -> dict[str, Any]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    card_path = out_dir / "card.json"
    if card_path.exists() and not overwrite: return json.loads(card_path.read_text(encoding="utf-8"))
    seed_all(config.seed)
    with np.load(trajectory_path, allow_pickle=False) as z:
        state_np = np.asarray(z["state_post"], dtype=np.float32)
        mean_np = np.asarray(z["state_mean"], dtype=np.float32)
        taus_np = np.asarray(z["fixed_taus_seconds"], dtype=np.float32)
    values_np, valid_np, groups = build_targets(data, bundle)
    fit_rows = np.flatnonzero(data.phase == "FIT"); inner_rows = np.flatnonzero(data.phase == "INNER")
    selection_rows = np.flatnonzero(data.phase == "SELECTION")
    support: dict[str, dict[str, int]] = {}
    for phase_name, phase_rows in (("FIT", fit_rows), ("INNER", inner_rows), ("SELECTION", selection_rows)):
        support[phase_name] = {}
        for j, offset in enumerate(OFFSETS):
            target = data.next_index[phase_rows, j]
            good = (target >= 0) & (data.decoder_index[np.maximum(target, 0)] >= 0)
            support[phase_name][f"next_{offset}_events"] = int(good.sum())
    if any(sum(support[phase_name].values()) == 0 for phase_name in ("FIT", "INNER", "SELECTION")):
        card = {
            "format": f"{FORMAT_PREFIX}_stepwise_auxiliary_heads_v1",
            "subject": data.subject,
            "status": "NOT_ESTIMABLE",
            "reason": "no frozen-decoder targets in at least one required chronological phase",
            "support": support,
            "selection": {},
            "feature_groups": groups,
            "state_entry": "FiLM after every frozen tissue-RNN step",
            "development_targets_read": False,
            "sealed_partition_opened": False,
            "seizure_outcomes_read": False,
        }
        atomic_json(card_path, card)
        return card
    values_np, centre, scale = _scale_targets(values_np, valid_np, fit_rows)
    values = torch.as_tensor(values_np, device=device); valid = torch.as_tensor(valid_np, device=device)
    state = torch.as_tensor(state_np, device=device)
    mean = torch.as_tensor(mean_np, device=device).unsqueeze(0)
    taus = torch.as_tensor(taus_np, device=device).unsqueeze(0)
    tensors = decoder_tensors(bundle, device)
    wrapper = StepwiseConditionedDecoder(bundle.model, StepwiseAdapterConfig(context_dim=data.q_context.shape[1], rank=config.adapter_rank)).to(device)
    q_adapter = wrapper.dynamic
    m_adapter = DynamicStepAdapter(StepwiseAdapterConfig(context_dim=state.shape[1], rank=config.adapter_rank),
                                   bundle.model.n_nodes * bundle.model.state_dim, bundle.model.n_contacts).to(device)
    head = AuxiliaryHead(bundle.model.n_nodes * bundle.model.state_dim, bundle.model.n_contacts,
                         values.shape[-1], config.head_rank).to(device)

    rng = np.random.default_rng(config.seed)
    histories = {}
    stages = (
        ("head", list(head.parameters()), config.head_lr, config.head_steps, False, False),
        ("q_adapter", list(q_adapter.parameters()), config.adapter_lr, config.q_steps, True, False),
        ("m_adapter", list(m_adapter.parameters()), config.adapter_lr, config.m_steps, True, True),
    )
    for stage, params, lr, max_steps, use_q, use_m in stages:
        for module in (head, wrapper, m_adapter):
            for p in module.parameters(): p.requires_grad_(False)
        for p in params: p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        best, best_state, stale, history = np.inf, None, 0, []
        for step in range(1, max_steps + 1):
            chosen = rng.choice(fit_rows, size=min(config.batch_events, fit_rows.size), replace=fit_rows.size < config.batch_events)
            j = int(rng.integers(0, len(OFFSETS)))
            source, target, batch, y, yvalid = _batch(data, tensors, values, valid, chosen, j, device)
            if source.size == 0: continue
            q = torch.as_tensor(data.q_context[source], dtype=torch.float32, device=device)
            m = _open_loop_state(state, mean, taus, data, source, target)
            hidden = wrapper.hidden_sequence(batch["x"], batch["recruited"], batch["valid"],
                                              q if use_q else None, use_static=False, use_dynamic=use_q,
                                              extra_context=m if use_m else None,
                                              extra_adapter=m_adapter if use_m else None)
            pred = head(hidden); loss, _ = _loss(pred, y, yvalid, batch["target"])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            if step % config.validate_every == 0:
                score = _evaluate(wrapper, head, data, tensors, values, valid, state, mean, taus, inner_rows,
                                  use_q=use_q, use_m=use_m, q_adapter=q_adapter, m_adapter=m_adapter)
                vals = [v for h in score.values() for v in h.values() if v is not None]
                value = float(np.mean(vals)) if vals else np.inf
                history.append({"step": step, "inner_loss": value, "fit_loss": float(loss.detach().cpu())})
                if value < best - 1e-6:
                    best, stale = value, 0
                    best_state = {"head": {k:v.detach().cpu().clone() for k,v in head.state_dict().items()},
                                  "q": {k:v.detach().cpu().clone() for k,v in q_adapter.state_dict().items()},
                                  "m": {k:v.detach().cpu().clone() for k,v in m_adapter.state_dict().items()}}
                else: stale += 1
                if stale >= config.patience: break
        if best_state is None: raise RuntimeError(f"{stage}: no finite inner checkpoint")
        head.load_state_dict(best_state["head"]); q_adapter.load_state_dict(best_state["q"]); m_adapter.load_state_dict(best_state["m"])
        histories[stage] = {"best_inner_loss": best, "history": history,
                            "selected_step": min(history, key=lambda x:x["inner_loss"])["step"]}

    # Correct-time and block-shift share the fitted head/adapters.
    shifted_np = state_np.copy(); shift_valid = np.zeros(state_np.shape[0], bool)
    for seg in np.unique(data.event_segment[selection_rows]):
        rr = selection_rows[data.event_segment[selection_rows] == seg]
        donor = np.roll(rr, rr.size // 2) if rr.size else rr
        ok = np.abs(data.event_time[donor] - data.event_time[rr]) >= 1800 if rr.size else np.zeros(0, bool)
        shifted_np[rr[ok]] = state_np[donor[ok]]; shift_valid[rr[ok]] = True
    shifted = torch.as_tensor(shifted_np, device=device)
    arms = {
        "base_head": _evaluate(wrapper, head, data, tensors, values, valid, state, mean, taus, selection_rows,
                               use_q=False, use_m=False, q_adapter=q_adapter, m_adapter=m_adapter),
        "q_only": _evaluate(wrapper, head, data, tensors, values, valid, state, mean, taus, selection_rows,
                            use_q=True, use_m=False, q_adapter=q_adapter, m_adapter=m_adapter),
        "mark_state_only": _evaluate(wrapper, head, data, tensors, values, valid, state, mean, taus, selection_rows,
                                      use_q=False, use_m=True, q_adapter=q_adapter, m_adapter=m_adapter),
        "q_plus_mark_state": _evaluate(wrapper, head, data, tensors, values, valid, state, mean, taus, selection_rows,
                                       use_q=True, use_m=True, q_adapter=q_adapter, m_adapter=m_adapter),
        "block_shift_mark_state": _evaluate(wrapper, head, data, tensors, values, valid, state, mean, taus,
                                             selection_rows[shift_valid[selection_rows]], use_q=True, use_m=True,
                                             q_adapter=q_adapter, m_adapter=m_adapter, shifted=shifted),
        # Correct-time state on exactly the block-shift support (review 2026-09-04).
        "q_plus_mark_state_on_shift_support": _evaluate(wrapper, head, data, tensors, values, valid, state, mean, taus,
                                                         selection_rows[shift_valid[selection_rows]], use_q=True, use_m=True,
                                                         q_adapter=q_adapter, m_adapter=m_adapter),
    }
    checkpoint = out_dir / "auxiliary_heads.pt"
    torch.save({"head": head.state_dict(), "q_adapter": q_adapter.state_dict(), "m_adapter": m_adapter.state_dict(),
                "config": config.__dict__, "target_centre": centre, "target_scale": scale}, checkpoint)
    card = {"format": f"{FORMAT_PREFIX}_stepwise_auxiliary_heads_v1", "subject": data.subject,
            "trajectory": str(trajectory_path), "feature_groups": groups, "stages": histories, "selection": arms,
            "checkpoint": str(checkpoint), "state_entry": "FiLM after every frozen tissue-RNN step",
            "head_training": "endpoint heads trained first without q/m, then frozen",
            "development_targets_read": False, "sealed_partition_opened": False, "seizure_outcomes_read": False}
    atomic_json(card_path, card); return card
