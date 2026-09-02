"""Frozen-grammar transfer probe for the count-trained v0.3.2 state.

The patient grammar was calibrated on an earlier prefix under the exact
product-form conditional K-subset likelihood.  It stays frozen.  Only a small
context adapter is fitted on ``base_fit`` (epoch selected on ``inner_val``), so
this probe asks whether the count-trained state transfers to event extent and
contact identity beyond the explicit history context.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.dataset import SubjectSequence
from src.topic5_group_event_state.v03.grammar import build_train_only_grammar
from src.topic5_group_event_state.v03.pilot import (
    DATASET_ROOT,
    _legacy_paths,
    grammar_fit_seq_positions,
    nested_partition,
    train_only_contact_features,
    validate_contact_contract,
)
from src.topic5_group_event_state.v02.subject import load_subject_timeline

from .blocks import block_ids_for_times, paired_gain_summary
from .contract import atomic_json, atomic_npz, now_iso
from .history import HistoryFeatureBuilder, history_inputs_from_timeline
from .partition import REFIT_PHASE
from .shift import apply_donor, predefined_session_shifts
from .state_registry import StateBundle
from .timeline import EvalTimeline

GRAMMAR_ROOT = Path("/data/hfosp_group_event_state_v0_3/pilot")
STATE_INPUT_DIM = 16


@dataclass
class ProbeData:
    group_ids: np.ndarray
    group_count: np.ndarray
    context: np.ndarray
    phase_rows: dict[str, np.ndarray]
    event_times: np.ndarray
    event_segment: np.ndarray


class FrozenGrammarContextProbe(nn.Module):
    def __init__(self, grammar: nn.Module, context_dim: int, *, rank: int = 8) -> None:
        super().__init__()
        self.grammar = grammar
        for p in self.grammar.parameters():
            p.requires_grad_(False)
        self.grammar.set_phase("adapter")
        width = int(min(rank, context_dim, STATE_INPUT_DIM))
        self.context_project = nn.Sequential(
            nn.Linear(context_dim, width, bias=False),
            nn.GELU(),
            nn.Linear(width, STATE_INPUT_DIM, bias=False),
        )
        nn.init.normal_(self.context_project[0].weight, std=0.02)
        nn.init.normal_(self.context_project[2].weight, std=1e-3)

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, group_ids: Tensor, group_count: Tensor, context: Tensor):
        return self.grammar(group_ids, group_count, self.context_project(context))


def _load_prefix_calibrated_grammar(subject: str, device: torch.device):
    """Reconstruct and load the v0.3 grammar; learned legacy weights remain unused."""

    checkpoint = GRAMMAR_ROOT / subject / "grammar" / "grammar_v03.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"missing prefix-calibrated grammar {checkpoint}")
    seq = SubjectSequence(DATASET_ROOT / subject)
    legacy_checkpoint, legacy_npz = _legacy_paths(subject)
    validate_contact_contract(subject, seq, legacy_npz)
    old_timeline = load_subject_timeline(subject)
    positions = grammar_fit_seq_positions(old_timeline, nested_partition(old_timeline))
    grammar = build_train_only_grammar(
        legacy_checkpoint,
        train_only_contact_features(seq, positions, legacy_npz),
        state_dim=STATE_INPUT_DIM,
        device=device,
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("legacy_learned_weights_loaded") is not False:
        raise ValueError("grammar checkpoint imported legacy learned weights")
    grammar.load_state_dict(saved["state_dict"], strict=True)
    grammar.set_phase("frozen")
    return grammar, {
        "checkpoint": str(checkpoint),
        "calibration_prefix_events": int(saved["calibration_prefix_events"]),
        "legacy_learned_weights_loaded": False,
        "grammar_fit_stop_epoch": float(saved["grammar_fit_stop_epoch"]),
    }


def _standardize_from_base_fit(x: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    mean = np.nanmean(x[rows], axis=0)
    scale = np.nanstd(x[rows], axis=0)
    scale = np.where(scale > 1e-8, scale, 1.0)
    out = np.nan_to_num((x - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return out, {"mean": mean.tolist(), "scale": scale.tolist(), "fit_phase": "base_fit"}


def event_history_context(tl: EvalTimeline, cfg: Mapping[str, Any]) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    inp = history_inputs_from_timeline(tl)
    hc = cfg["history"]
    builder = HistoryFeatureBuilder(
        inp,
        lookback_seconds=hc["lookback_seconds"],
        ewma_tau_seconds=hc["ewma_tau_seconds"],
        field_tau_seconds=hc["field_tau_seconds"],
    )
    x, names = builder.features(tl.event_times, tl.event_segment, variant="H_strong")
    fit = tl.event_indices("base_fit")
    x, stats = _standardize_from_base_fit(x, fit)
    return x, names, stats


def _state_contexts(
    tl: EvalTimeline,
    state: StateBundle,
    cfg: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    if state.event_pre_state is None:
        raise ValueError("H2a needs event_pre_state or an anchor-held fallback")
    base = tl.event_indices("base_fit")
    s, stats = _standardize_from_base_fit(state.event_pre_state, base)
    out: dict[str, np.ndarray] = {"S_correct": s}
    mean = np.nanmean(s[base], axis=0)
    out["S_mean"] = np.broadcast_to(mean, s.shape).copy()
    specs = predefined_session_shifts(
        tl.event_times, tl.event_session,
        n_shifts=int(cfg["shift"]["n_shifts"]),
        denominator=int(cfg["shift"]["fraction_denominator"]),
        min_distance_seconds=float(cfg["timeline"]["primary_horizon_seconds"])
        + float(cfg["shift"]["min_gap_over_horizon_seconds"]),
    )
    meta = []
    for spec in specs:
        out[f"S_shifted:{spec['shift_id']}"] = apply_donor(s, spec["donor_index"])
        meta.append({k: v for k, v in spec.items() if k != "donor_index"})
    return out, meta, stats


def _loss_terms(probe: FrozenGrammarContextProbe, ids: Tensor, count: Tensor, context: Tensor):
    terms, outputs = probe(ids, count, context)
    active = terms.active_step.float().sum().clamp_min(1.0)
    select = terms.select_step.float().sum().clamp_min(1.0)
    continuation = -outputs["continue_step_log_prob"].sum() / active
    positive_size = -outputs["positive_size_step_log_prob"].sum() / select
    subset = -terms.subset_step_log_prob.sum() / select
    return continuation + positive_size + subset


@torch.no_grad()
def _score(
    probe: FrozenGrammarContextProbe,
    data: ProbeData,
    rows: np.ndarray,
    *,
    batch: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    probe.eval()
    n = rows.size
    continue_event = np.full(n, np.nan)
    size_event = np.full(n, np.nan)
    subset_event = np.full(n, np.nan)
    later_continue = np.full(n, np.nan)
    valid_local = np.flatnonzero(np.isfinite(data.context[rows]).all(axis=1))
    for lo in range(0, valid_local.size, batch):
        local = valid_local[lo : lo + batch]
        pos = rows[local]
        ids = torch.from_numpy(data.group_ids[pos]).to(device).long()
        count = torch.from_numpy(data.group_count[pos]).to(device).long()
        context = torch.from_numpy(data.context[pos]).to(device).float()
        terms, outputs = probe(ids, count, context)
        active = terms.active_step.float()
        select = terms.select_step.float()
        c = -outputs["continue_step_log_prob"]
        k = -outputs["positive_size_step_log_prob"]
        q = -terms.subset_step_log_prob
        continue_event[local] = (c.sum(1) / active.sum(1).clamp_min(1)).cpu().numpy()
        has = select.sum(1) > 0
        size = torch.full((pos.size,), torch.nan, device=device)
        subset = torch.full_like(size, torch.nan)
        size[has] = k.sum(1)[has] / select.sum(1)[has]
        subset[has] = q.sum(1)[has] / select.sum(1)[has]
        size_event[local] = size.cpu().numpy()
        subset_event[local] = subset.cpu().numpy()
        later = active.clone()
        later[:, 0] = 0
        has_later = later.sum(1) > 0
        lc = torch.full_like(size, torch.nan)
        lc[has_later] = (c * later).sum(1)[has_later] / later.sum(1)[has_later]
        later_continue[local] = lc.cpu().numpy()
    return {
        "continue": continue_event,
        "positive_size": size_event,
        "subset_identity": subset_event,
        "later_continuation": later_continue,
    }


def fit_probe(
    grammar: nn.Module,
    data: ProbeData,
    *,
    cfg: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[FrozenGrammarContextProbe, dict[str, Any]]:
    gc = cfg["grammar"]
    # The exact subset dynamic program is dominated by Python launch overhead at
    # the legacy batch of 256.  These patient grammars fit comfortably at 1024
    # (<2 GB in the pilot), which changes no samples or optimisation steps.
    batch = max(int(gc["batch"]), 1024)

    def fresh() -> FrozenGrammarContextProbe:
        torch.manual_seed(int(seed))
        return FrozenGrammarContextProbe(copy.deepcopy(grammar), data.context.shape[1]).to(device)

    model = fresh()
    optimizer = torch.optim.AdamW(model.trainable_parameters, lr=float(gc["adapter_lr"]),
                                  weight_decay=float(gc["weight_decay"]))
    rng = np.random.default_rng(int(seed))
    fit = data.phase_rows["base_fit"]
    fit = fit[np.isfinite(data.context[fit]).all(axis=1)]
    inner = data.phase_rows["inner_val"]
    inner = inner[np.isfinite(data.context[inner]).all(axis=1)]
    if fit.size < int(gc["min_fit_events"]) or inner.size < int(gc["min_inner_events"]):
        raise ValueError(f"insufficient finite H2a rows fit={fit.size} inner={inner.size}")
    history = []
    best = np.inf
    selected = 0
    stale = 0
    max_epochs = int(gc["adapter_max_epochs"])
    patience = int(gc["adapter_patience"])
    for epoch in range(max_epochs):
        model.train()
        order = rng.permutation(fit)
        for lo in range(0, fit.size, batch):
            take = order[lo : lo + batch]
            ids = torch.from_numpy(data.group_ids[take]).to(device).long()
            count = torch.from_numpy(data.group_count[take]).to(device).long()
            context = torch.from_numpy(data.context[take]).to(device).float()
            loss = _loss_terms(model, ids, count, context)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 5.0)
            optimizer.step()
        scored = _score(model, data, inner, batch=batch, device=device)
        val = float(sum(np.nanmean(scored[k]) for k in ("continue", "positive_size", "subset_identity")))
        history.append({"epoch": epoch, "inner_total": val})
        if np.isfinite(val) and val < best - 1e-5:
            best, selected, stale = val, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break

    # Refit a fresh adapter on 0--70% for the selected number of epochs.  The
    # epoch count, not any test result, is the only quantity carried over.
    model = fresh()
    optimizer = torch.optim.AdamW(model.trainable_parameters, lr=float(gc["adapter_lr"]),
                                  weight_decay=float(gc["weight_decay"]))
    refit = data.phase_rows[REFIT_PHASE]
    refit = refit[np.isfinite(data.context[refit]).all(axis=1)]
    rng = np.random.default_rng(int(seed))
    for _epoch in range(selected + 1):
        order = rng.permutation(refit)
        model.train()
        for lo in range(0, order.size, batch):
            take = order[lo : lo + batch]
            ids = torch.from_numpy(data.group_ids[take]).to(device).long()
            count = torch.from_numpy(data.group_count[take]).to(device).long()
            context = torch.from_numpy(data.context[take]).to(device).float()
            loss = _loss_terms(model, ids, count, context)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters, 5.0)
            optimizer.step()
    model.eval()
    return model, {"selected_epoch": int(selected), "best_inner_total": float(best),
                   "selected_at_budget_edge": bool(selected == max_epochs - 1),
                   "selection_history": history, "selection_phase": "inner_val",
                   "refit_phase": REFIT_PHASE, "test_time_fit": False}


def evaluate_h2a_patient_seed(
    tl: EvalTimeline,
    state: StateBundle,
    cfg: Mapping[str, Any],
    *,
    out_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    grammar, grammar_meta = _load_prefix_calibrated_grammar(tl.subject, device)
    h, h_names, h_stats = event_history_context(tl, cfg)
    s_map, shift_meta, s_stats = _state_contexts(tl, state, cfg)
    zero = np.zeros_like(next(iter(s_map.values())), dtype=np.float32)
    contexts = {"H": np.concatenate([h, zero], axis=1)}
    for name, s in s_map.items():
        contexts[f"H+{name}"] = np.concatenate([h, s], axis=1)
    rows = {p: tl.event_indices(p) for p in ("base_fit", "inner_val", REFIT_PHASE, "dev_val", "dev_test")}
    fitted = {}
    scores = {}
    base_seed = int(state.seed)
    for arm, context in contexts.items():
        data = ProbeData(tl.tied_group_id, tl.group_count, context, rows, tl.event_times, tl.event_segment)
        model, fit_meta = fit_probe(grammar, data, cfg=cfg, seed=base_seed, device=device)
        fitted[arm] = fit_meta
        scores[arm] = {
            phase: _score(model, data, rows[phase], batch=max(int(cfg["grammar"]["batch"]), 1024), device=device)
            for phase in ("dev_val", "dev_test")
        }
    summaries: dict[str, Any] = {}
    seg_start = tl.segment_start_map()
    for phase in ("dev_val", "dev_test"):
        idx = rows[phase]
        blocks = block_ids_for_times(tl.event_times[idx], tl.event_segment[idx], seg_start, 1800.0)
        phase_out = {}
        for endpoint in ("continue", "positive_size", "subset_identity", "later_continuation"):
            ref = scores["H"][phase][endpoint]
            phase_out[endpoint] = {}
            for arm in contexts:
                if arm == "H":
                    continue
                phase_out[endpoint][f"{arm}_vs_H"] = paired_gain_summary(
                    ref, scores[arm][phase][endpoint], blocks,
                    n_boot=int(cfg["inference"]["bootstrap_replicates"]),
                    seed=int(cfg["inference"]["bootstrap_seed"]),
                )
            correct = scores["H+S_correct"][phase][endpoint]
            for comparator in ["H+S_mean"] + [f"H+S_shifted:{j}" for j in range(1, 6)]:
                phase_out[endpoint][f"H+S_correct_vs_{comparator}"] = paired_gain_summary(
                    scores[comparator][phase][endpoint], correct, blocks,
                    n_boot=int(cfg["inference"]["bootstrap_replicates"]),
                    seed=int(cfg["inference"]["bootstrap_seed"]),
                )
        summaries[phase] = phase_out
    report = {
        "format": "group_event_state_v0_3_2_h2a_frozen_grammar_probe",
        "generated": now_iso(), "subject": tl.subject, "seed": state.seed,
        "status": "complete", "grammar": grammar_meta,
        "context": {"history_features": len(h_names), "state_dim": state.state_dim,
                    "history_standardization": h_stats, "state_standardization": s_stats},
        "shift": shift_meta, "fit": fitted, "paired": summaries,
        "primary_endpoint": "subset_identity conditional on observed K and prefix",
        "state_training_target": "30 min count residual only",
        "test_time_fit": False, "sealed_partition_opened": False,
        "scope": "development transfer probe; not a seizure or mechanism result",
    }
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"event_time": tl.event_times}
    for arm, phase_scores in scores.items():
        safe = arm.replace("+", "plus").replace(":", "_")
        for phase, endpoint_scores in phase_scores.items():
            for endpoint, values in endpoint_scores.items():
                arrays[f"{safe}_{phase}_{endpoint}"] = values
    atomic_npz(out_dir / f"h2a_arrays_seed_{state.seed}.npz", arrays)
    atomic_json(out_dir / f"h2a_result_seed_{state.seed}.json", report)
    return report
