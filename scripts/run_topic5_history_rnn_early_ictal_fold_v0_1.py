#!/usr/bin/env python3
"""Run one strict patient-LOSO history-state -> early-ictal field fold.

Early-ictal arrays are opened only after target-blind checkpoint provenance
and the independent v0.2 direct-transfer authorization have both passed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from scripts.run_topic5_history_rnn_gate1_sequential_fold_v0_1 import (  # noqa: E402
    MatchedSequentialModel,
    ResidualSequentialModel,
    UnorderedResidualSequentialModel,
    _capacity_matched_unordered_dim,
    _causal_unordered_summary,
)
from src.topic5_history_bridge import (  # noqa: E402
    causal_ewma_contact_fields,
    causal_contact_features,
    centered_field,
    patient_balanced_contact_weights,
    robust_z_field,
    weighted_ridge_fit,
    weighted_ridge_predict,
)
from src.topic5_history_rnn import (  # noqa: E402
    encode_within_event,
    prefix_matched_order_indices,
)
from src.topic5_rank_distribution import LinearStateSequenceRNN  # noqa: E402


ALPHAS = (0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
PRIMARY_SEED = 20260725
CONTACT_FEATURE_CONTRACT = (
    "prefix_participation_support",
    "prefix_participation_support_centered",
    "within_shaft_position",
    "shaft_size_fraction",
    "coord_x_centered_scaled",
    "coord_y_centered_scaled",
    "coord_z_centered_scaled",
    "geometry_present",
)


def _require_g1(g1_root: Path) -> None:
    path = g1_root / "G1_MULTI_SEED_SUMMARY.json"
    if not path.exists():
        raise RuntimeError("G2 locked: multi-seed G1 summary is absent")
    payload = json.loads(path.read_text())
    if bool(payload.get("target_values_read", True)):
        raise RuntimeError("G1 target-seal provenance is invalid")
    if payload.get("status") != "G1_MULTI_SEED_PASS_OPEN_G2":
        raise RuntimeError(f"G2 locked by {payload.get('status')}")


def _authorize_target_access(
    g1_root: Path,
    direct_transfer_contract: Path | None,
) -> str:
    """Authorize target access by the old G1 gate or the revised direct contract."""

    path = g1_root / "G1_MULTI_SEED_SUMMARY.json"
    refit_path = g1_root / "REFIT_SUMMARY.json"
    if path.exists():
        g1 = json.loads(path.read_text())
    elif refit_path.exists():
        g1 = json.loads(refit_path.read_text())
        if (
            g1.get("status") != "COMPLETE"
            or int(g1.get("n_completed_folds", 0)) != 16
            or int(g1.get("n_failed_folds", 1)) != 0
        ):
            raise RuntimeError("target access denied: refit provenance is incomplete")
    else:
        raise RuntimeError("target access denied: checkpoint provenance is absent")
    if bool(g1.get("target_values_read", True)):
        raise RuntimeError("target access denied: G1 target-seal provenance is invalid")
    if g1.get("status") == "G1_MULTI_SEED_PASS_OPEN_G2":
        return "G1_MULTI_SEED_PASS_OPEN_G2"
    if direct_transfer_contract is None:
        raise RuntimeError(f"G2 locked by {g1.get('status')}")
    contract_path = direct_transfer_contract.resolve()
    contract = json.loads(contract_path.read_text())
    required = {
        "status": "DIRECT_TRANSFER_AUTHORIZED_INDEPENDENT_OF_G1",
        "endpoint": "clinical_onset_[0,10]s_1-150Hz_contact_energy",
        "g1_role": "PARALLEL_PROXY_EVIDENCE_NOT_HARD_GATE",
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"direct-transfer contract mismatch for {key}")
    return f"DIRECT_V0_2:{contract_path}"


def _load_models(fold_dir: Path, device: torch.device):
    done = json.loads((fold_dir / "DONE.json").read_text())
    checkpoint = torch.load(
        fold_dir / "checkpoint.pt", map_location="cpu", weights_only=False
    )
    event_payload = torch.load(
        Path(done["event_checkpoint"]), map_location="cpu", weights_only=False
    )
    event_model = LinearStateSequenceRNN(**event_payload["model_kwargs"])
    event_model.load_state_dict(event_payload["model_state"])
    event_model.to(device).eval()
    config = checkpoint["config"]
    event_dim = int(event_payload["model_kwargs"]["hidden_size"])
    contact_dim = int(event_payload["model_kwargs"]["contact_embedding_dim"])
    history_dim = int(config["history_dim"])
    matched = MatchedSequentialModel(event_dim, history_dim, contact_dim).to(device)
    unordered_dim = _capacity_matched_unordered_dim(
        event_dim, contact_dim, history_dim
    )
    unordered = UnorderedResidualSequentialModel(
        event_dim, unordered_dim, contact_dim
    ).to(device)
    chronological = ResidualSequentialModel(
        event_dim,
        history_dim,
        contact_dim,
        initial_half_life_hours=float(config["initial_half_life_hours"]),
    ).to(device)
    matched.load_state_dict(checkpoint["matched_state"])
    unordered.load_state_dict(checkpoint["unordered_residual_state"])
    chronological.load_state_dict(checkpoint["history_state"])
    matched.eval()
    unordered.eval()
    chronological.eval()
    return (
        event_model,
        matched,
        unordered,
        chronological,
        checkpoint["event_embedding_mean"].numpy(),
        checkpoint["event_embedding_scale"].numpy(),
    )


@torch.no_grad()
def _event_embedding(
    event_model,
    contact_features: np.ndarray,
    group_ids: np.ndarray,
    group_count: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    features = torch.as_tensor(contact_features, device=device).unsqueeze(0)
    zero_offset = torch.zeros(
        (len(contact_features), event_model.local_offset_dim),
        dtype=torch.float32,
        device=device,
    )
    states = []
    contact_embedding = None
    for start in range(0, len(group_ids), int(batch_size)):
        stop = min(start + int(batch_size), len(group_ids))
        state, embedding = encode_within_event(
            event_model,
            features.expand(stop - start, -1, -1),
            torch.as_tensor(group_ids[start:stop], dtype=torch.long, device=device),
            torch.as_tensor(group_count[start:stop], dtype=torch.long, device=device),
            local_offset=zero_offset,
        )
        states.append(state.cpu().numpy().astype(np.float32))
        if contact_embedding is None:
            contact_embedding = embedding[0].cpu().numpy().astype(np.float32)
    return np.row_stack(states), np.asarray(contact_embedding)


@torch.no_grad()
def _history_final(
    chronological: ResidualSequentialModel,
    embedding: np.ndarray,
    event_time: np.ndarray,
    *,
    device: torch.device,
    chunk: int = 512,
) -> torch.Tensor:
    state = None
    for start in range(0, len(embedding), int(chunk)):
        stop = min(start + int(chunk), len(embedding))
        delta = np.zeros(stop - start, dtype=np.float32)
        if start > 0:
            delta[0] = float(event_time[start] - event_time[start - 1])
        if stop - start > 1:
            delta[1:] = np.diff(event_time[start:stop]).astype(np.float32)
        reset = torch.zeros((1, stop - start), dtype=torch.bool, device=device)
        if start == 0:
            reset[:, 0] = True
        _, state = chronological.history.forward_masked(
            torch.as_tensor(embedding[start:stop], device=device).unsqueeze(0),
            torch.as_tensor(delta, device=device).unsqueeze(0),
            reset,
            torch.ones((1, stop - start), dtype=torch.bool, device=device),
            initial_state=state,
        )
    return state


@torch.no_grad()
def _history_final_with_diagnostics(
    chronological: ResidualSequentialModel,
    embedding: np.ndarray,
    event_time: np.ndarray,
    *,
    device: torch.device,
    chunk: int = 512,
) -> tuple[torch.Tensor, dict[str, float]]:
    state = None
    state_rows = []
    for start in range(0, len(embedding), int(chunk)):
        stop = min(start + int(chunk), len(embedding))
        delta = np.zeros(stop - start, dtype=np.float32)
        if start > 0:
            delta[0] = float(event_time[start] - event_time[start - 1])
        if stop - start > 1:
            delta[1:] = np.diff(event_time[start:stop]).astype(np.float32)
        reset = torch.zeros((1, stop - start), dtype=torch.bool, device=device)
        if start == 0:
            reset[:, 0] = True
        states, state = chronological.history.forward_masked(
            torch.as_tensor(embedding[start:stop], device=device).unsqueeze(0),
            torch.as_tensor(delta, device=device).unsqueeze(0),
            reset,
            torch.ones((1, stop - start), dtype=torch.bool, device=device),
            initial_state=state,
        )
        state_rows.append(states[0].cpu().numpy())
    trajectory = np.row_stack(state_rows)
    delta = np.diff(trajectory, axis=0)
    diagnostics = {
        "history_state_variance": float(np.mean(np.var(trajectory, axis=0))),
        "history_state_final_l2": float(np.linalg.norm(trajectory[-1])),
        "history_state_step_rms": (
            float(np.sqrt(np.mean(np.square(delta)))) if len(delta) else 0.0
        ),
    }
    return state, diagnostics


@torch.no_grad()
def _history_final_order_control(
    chronological: ResidualSequentialModel,
    embedding: np.ndarray,
    event_time: np.ndarray,
    *,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Reassign every pre-final event identity across the fixed time slots.

    The contract fixes the event set, the within-event ranks, the event count,
    the total span, the IEI slots and the last event, and permutes everything
    before the last event.  The permutation therefore has to cover the whole
    causal prefix: with a two-hour state half-life and prefixes of up to 6125
    events, a recent-window permutation would leave most of the state's
    effective memory in true chronological order and the control would test
    almost nothing.
    """

    if len(embedding) < 3:
        return _history_final(chronological, embedding, event_time, device=device)
    start, indices = prefix_matched_order_indices(
        len(embedding), window=len(embedding),
        rng=np.random.default_rng(int(seed)),
    )
    state = None
    if start > 0:
        state = _history_final(
            chronological,
            embedding[:start],
            event_time[:start],
            device=device,
        )
    recent_embedding = embedding[indices]
    recent_time = event_time[start:]
    delta = np.zeros(len(recent_embedding), dtype=np.float32)
    if state is not None:
        delta[0] = float(recent_time[0] - event_time[start - 1])
    if len(recent_embedding) > 1:
        delta[1:] = np.diff(recent_time).astype(np.float32)
    reset = torch.zeros((1, len(recent_embedding)), dtype=torch.bool, device=device)
    if state is None:
        reset[:, 0] = True
    _, final = chronological.history.forward_masked(
        torch.as_tensor(recent_embedding, device=device).unsqueeze(0),
        torch.as_tensor(delta, device=device).unsqueeze(0),
        reset,
        torch.ones((1, len(recent_embedding)), dtype=torch.bool, device=device),
        initial_state=state,
    )
    return final


def _event_relative_rank(group_ids: np.ndarray, group_count: np.ndarray) -> np.ndarray:
    group = np.asarray(group_ids, dtype=np.float64)
    count = np.asarray(group_count, dtype=np.float64)
    denominator = np.maximum(count - 1.0, 1.0)[:, None]
    rank = group / denominator
    rank[group < 0] = np.nan
    return rank


def _frozen_static_scaffold(
    artifact: Path,
    subject: str,
    contact_names: np.ndarray,
) -> dict[str, np.ndarray]:
    """Load and exactly name-align the paper's target-blind scaffold.

    The frozen paper field is defined only on its own ``contact_order``.  Some
    rank-distribution datasets contain additional contacts that were not part
    of that frozen field.  Those contacts must not be positionally joined or
    silently imputed; ``scaffold_valid`` marks the exact-name intersection and
    the direct-transfer scorer subsequently uses only that fixed denominator.
    """

    path = (
        artifact
        / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
        / f"{subject}.json"
    )
    payload = json.loads(path.read_text())
    if payload.get("status") != "ok" or payload["interictal_field"].get("status") != "ok":
        raise RuntimeError(f"{subject}: frozen interictal scaffold unavailable")
    field = payload["interictal_field"]
    names = np.asarray(field["contact_order"]).astype(str)
    target_names = np.asarray(contact_names).astype(str)
    if len(set(names)) != len(names) or len(set(target_names)) != len(target_names):
        raise RuntimeError(f"{subject}: duplicate contact names prevent exact join")
    target_lookup = {name: index for index, name in enumerate(target_names)}
    missing = [name for name in names if name not in target_lookup]
    if missing:
        raise RuntimeError(
            f"{subject}: frozen scaffold contacts absent from rank dataset: {missing}"
        )
    source_index = {name: index for index, name in enumerate(names)}
    valid = np.asarray([name in source_index for name in target_names], dtype=bool)
    if int(valid.sum()) < 3:
        raise RuntimeError(f"{subject}: fewer than three exact scaffold contacts")

    def align(values: np.ndarray) -> np.ndarray:
        source = centered_field(np.asarray(values, dtype=np.float64))
        if len(source) != len(names):
            raise RuntimeError(f"{subject}: frozen scaffold value length drift")
        aligned = np.full(len(target_names), np.nan, dtype=np.float64)
        for target_index, name in enumerate(target_names):
            if name in source_index:
                aligned[target_index] = source[source_index[name]]
        return aligned

    model = field["field_models"]
    earliness_a = np.asarray(payload["earliness_a"], dtype=np.float64)
    earliness_b = np.asarray(payload["earliness_b"], dtype=np.float64)
    support_a = np.asarray(payload["support_a"], dtype=np.float64)
    support_b = np.asarray(payload["support_b"], dtype=np.float64)
    return {
        "scaffold_valid": valid,
        "scaffold_earliness_a": align(earliness_a),
        "scaffold_earliness_b": align(earliness_b),
        "scaffold_axis_magnitude": align(
            0.5 * (np.abs(earliness_a) + np.abs(earliness_b))
        ),
        "scaffold_support_mean": align(0.5 * (support_a + support_b)),
        "scaffold_support_difference": align(support_a - support_b),
        "scaffold_field_a": align(
            np.asarray(model["own_a"]["template_field"], dtype=np.float64)
        ),
        "scaffold_field_b": align(
            np.asarray(model["own_b"]["template_field"], dtype=np.float64)
        ),
    }


@torch.no_grad()
def _seizure_features(
    subject: str,
    inventory: pd.DataFrame,
    *,
    artifact: Path,
    g0_root: Path,
    target_root: Path,
    event_model,
    matched,
    unordered,
    chronological,
    embedding_mean: np.ndarray,
    embedding_scale: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    dataset_path = (
        artifact
        / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
        / f"{subject}.npz"
    )
    with np.load(dataset_path, allow_pickle=False) as data:
        group_ids = np.asarray(data["event_group_ids"], np.int16)
        group_count = np.asarray(data["event_group_count"], np.int16)
        participation = np.asarray(data["event_participation"], np.uint8)
        base_contact_features = np.asarray(data["contact_features"], np.float32)
        contact_feature_names = tuple(
            str(value) for value in np.asarray(data["contact_feature_names"])
        )
        contact_names = np.asarray(data["contact_names"])
    if contact_feature_names != CONTACT_FEATURE_CONTRACT:
        raise RuntimeError(f"{subject}: contact-feature schema drift")
    with np.load(g0_root / "timeline" / f"{subject}.npz", allow_pickle=False) as data:
        event_time = np.asarray(data["event_time"], np.float64)
        segment_id = np.asarray(data["event_segment_id"], np.int32)
        target_contact_index = np.asarray(data["target_contact_index"], np.int64)
    frozen_scaffold = _frozen_static_scaffold(
        artifact, subject, contact_names
    )
    scaffold_valid = np.asarray(frozen_scaffold.pop("scaffold_valid"), dtype=bool)

    # This is the first target-value access in the direct-transfer pipeline,
    # after explicit v0.2 authorization; G1 is parallel evidence, not a gate.
    with np.load(target_root / f"{subject}.npz", allow_pickle=False) as target_data:
        target_values = {
            int(row.seizure_idx): np.asarray(
                target_data[f"bb150_auc__{int(row.seizure_idx)}"], np.float64
            ).squeeze()
            for row in inventory.itertuples(index=False)
        }

    rows = []
    for seizure in inventory.itertuples(index=False):
        candidate = np.flatnonzero(
            (segment_id == int(seizure.segment_id))
            & (event_time < float(seizure.clinical_onset_epoch) - float(seizure.guard_seconds))
            & (
                event_time
                >= (
                    float(seizure.previous_postictal_end_epoch)
                    if np.isfinite(float(seizure.previous_postictal_end_epoch))
                    else -np.inf
                )
            )
        )
        if len(candidate) != int(seizure.n_causal_events):
            raise RuntimeError(f"{subject} seizure {seizure.seizure_idx}: prefix drift")
        causal_features, support = causal_contact_features(
            base_contact_features, participation[candidate]
        )
        event_state, contact_embedding = _event_embedding(
            event_model,
            causal_features,
            group_ids[candidate],
            group_count[candidate],
            device=device,
        )
        normalized = ((event_state - embedding_mean) / embedding_scale).astype(
            np.float32
        )
        summary = _causal_unordered_summary(
            normalized, event_time[candidate]
        )[-1]
        contact_tensor = torch.as_tensor(
            contact_embedding, device=device
        ).unsqueeze(0)
        summary_tensor = torch.as_tensor(summary, device=device).unsqueeze(0)
        base_state = matched.history(summary_tensor)
        base_field = matched.heads(base_state, contact_tensor)
        unordered_field = unordered(summary_tensor, contact_tensor)
        history_state, state_diagnostics = _history_final_with_diagnostics(
            chronological,
            normalized,
            event_time[candidate],
            device=device,
        )
        shuffle_seed = (
            sum(ord(value) for value in subject) * 1009
            + int(seizure.seizure_idx)
        )
        shuffled_history_state = _history_final_order_control(
            chronological,
            normalized,
            event_time[candidate],
            seed=shuffle_seed,
            device=device,
        )
        history_field = chronological.heads(history_state, contact_tensor)
        shuffled_history_field = chronological.heads(
            shuffled_history_state, contact_tensor
        )
        static_logit = np.log(np.clip(support, 1e-5, 1 - 1e-5)) - np.log(
            np.clip(1 - support, 1e-5, 1 - 1e-5)
        )
        m1_part = (
            static_logit
            + base_field["participation_logits"][0].cpu().numpy()
            + unordered_field["participation_logits"][0].cpu().numpy()
        )
        m1_rank = (
            base_field["relative_rank"][0].cpu().numpy()
            + unordered_field["relative_rank"][0].cpu().numpy()
        )
        history_part = history_field["participation_logits"][0].cpu().numpy()
        history_rank = history_field["relative_rank"][0].cpu().numpy()
        energy = target_values[int(seizure.seizure_idx)]
        if energy.ndim != 1 or len(energy) <= int(target_contact_index.max()):
            raise RuntimeError(f"{subject}: early-ictal target shape drift")
        joined_energy = energy[target_contact_index]
        # The frozen bridge is a contact-centered spatial-field task.  Robust
        # scaling controls seizure-specific amplitude, then exact mean
        # centering removes the arbitrary field shift required by the spec.
        target_z = centered_field(robust_z_field(joined_energy))
        geometry = causal_features[:, 2:8].astype(np.float64)
        geometry = centered_field(geometry.T).T
        fields = {
            "static": centered_field(static_logit),
            "m1_part": centered_field(m1_part),
            "m1_rank": centered_field(m1_rank),
            "history_part": centered_field(history_part),
            "history_rank": centered_field(history_rank),
            "history_shuffle_part": centered_field(
                shuffled_history_field["participation_logits"][0].cpu().numpy()
            ),
            "history_shuffle_rank": centered_field(
                shuffled_history_field["relative_rank"][0].cpu().numpy()
            ),
        }
        relative_rank = _event_relative_rank(
            group_ids[candidate], group_count[candidate]
        )
        # Same permutation the RNN order control uses, so both temporal
        # controls reassign the identical causal prefix over the same slots.
        recent_start, recent_order = prefix_matched_order_indices(
            len(candidate),
            window=len(candidate),
            rng=np.random.default_rng(shuffle_seed),
        )
        full_order = np.arange(len(candidate), dtype=np.int64)
        full_order[recent_start:] = recent_order
        shuffled_participation = participation[candidate][full_order]
        shuffled_relative_rank = relative_rank[full_order]
        cutoff_epoch = float(seizure.clinical_onset_epoch) - float(
            seizure.guard_seconds
        )
        for half_life, label in ((0.5, "0p5"), (2.0, "2"), (6.0, "6")):
            ewma_part, ewma_rank = causal_ewma_contact_fields(
                participation[candidate],
                relative_rank,
                event_time[candidate],
                cutoff_epoch=cutoff_epoch,
                half_life_hours=half_life,
            )
            fields[f"ewma_{label}_part"] = centered_field(ewma_part)
            fields[f"ewma_{label}_rank"] = centered_field(ewma_rank)
            shuffled_ewma_part, shuffled_ewma_rank = causal_ewma_contact_fields(
                shuffled_participation,
                shuffled_relative_rank,
                event_time[candidate],
                cutoff_epoch=cutoff_epoch,
                half_life_hours=half_life,
            )
            fields[f"ewma_{label}_shuffle_part"] = centered_field(
                shuffled_ewma_part
            )
            fields[f"ewma_{label}_shuffle_rank"] = centered_field(
                shuffled_ewma_rank
            )
        for contact_index, contact in enumerate(contact_names):
            if not scaffold_valid[contact_index]:
                continue
            row = {
                "subject": subject,
                "seizure_idx": int(seizure.seizure_idx),
                "seizure_id": str(seizure.seizure_id),
                "history_fingerprint": str(seizure.history_fingerprint),
                "contact_index": int(contact_index),
                "contact": str(contact),
                "n_contacts": int(scaffold_valid.sum()),
                "target_energy": float(joined_energy[contact_index]),
                "target_z": float(target_z[contact_index]),
                **state_diagnostics,
            }
            for index in range(geometry.shape[1]):
                row[f"geometry_{index}"] = float(geometry[contact_index, index])
            for key, value in fields.items():
                row[key] = float(value[contact_index])
            for key, value in frozen_scaffold.items():
                row[key] = float(value[contact_index])
            rows.append(row)
    return pd.DataFrame(rows)


def _features(frame: pd.DataFrame, model: str) -> tuple[np.ndarray, list[str]]:
    geometry = [column for column in frame if column.startswith("geometry_")]
    scaffold = sorted(column for column in frame if column.startswith("scaffold_"))
    columns = geometry + scaffold + ["static"]
    if model != "M0":
        columns += ["m1_part", "m1_rank"]
    if model == "R2":
        columns += ["history_part", "history_rank"]
    elif model == "E0p5":
        columns += ["ewma_0p5_part", "ewma_0p5_rank"]
    elif model == "E2":
        columns += ["ewma_2_part", "ewma_2_rank"]
    elif model == "E6":
        columns += ["ewma_6_part", "ewma_6_rank"]
    elif model == "EM":
        columns += [
            "ewma_0p5_part", "ewma_0p5_rank",
            "ewma_2_part", "ewma_2_rank",
            "ewma_6_part", "ewma_6_rank",
        ]
    elif model not in {"M0", "M1"}:
        raise ValueError(f"unknown direct-transfer model {model}")
    return frame[columns].to_numpy(float), columns


def _r2_shuffled_features(frame: pd.DataFrame) -> np.ndarray:
    shuffled = frame.copy()
    shuffled["history_part"] = shuffled["history_shuffle_part"]
    shuffled["history_rank"] = shuffled["history_shuffle_rank"]
    return _features(shuffled, "R2")[0]


def _r2_zero_state_features(frame: pd.DataFrame) -> np.ndarray:
    zero = frame.copy()
    zero["history_part"] = 0.0
    zero["history_rank"] = 0.0
    return _features(zero, "R2")[0]


def _ewma_shuffled_features(frame: pd.DataFrame, model: str) -> np.ndarray:
    shuffled = frame.copy()
    labels = ("2",) if model == "E2" else ("0p5", "2", "6")
    for label in labels:
        shuffled[f"ewma_{label}_part"] = shuffled[
            f"ewma_{label}_shuffle_part"
        ]
        shuffled[f"ewma_{label}_rank"] = shuffled[
            f"ewma_{label}_shuffle_rank"
        ]
    return _features(shuffled, model)[0]


def _fit_model(train: pd.DataFrame, model_name: str) -> tuple[dict, list[str], float]:
    patient = train.subject.astype(str).to_numpy()
    seizure = train.seizure_id.astype(str).to_numpy()
    count = train.n_contacts.to_numpy(int)
    weight = patient_balanced_contact_weights(patient, seizure, count)
    x, names = _features(train, model_name)
    y = train.target_z.to_numpy(float)
    patients = sorted(train.subject.unique())
    scores = []
    for alpha in ALPHAS:
        fold_mse = []
        for heldout in patients:
            fit = train.subject != heldout
            validation = ~fit
            fit_weight = patient_balanced_contact_weights(
                patient[fit], seizure[fit], count[fit]
            )
            ridge = weighted_ridge_fit(x[fit], y[fit], fit_weight, alpha=alpha)
            prediction = weighted_ridge_predict(ridge, x[validation])
            fold_mse.append(float(np.mean(np.square(prediction - y[validation]))))
        scores.append((float(np.mean(fold_mse)), float(alpha)))
    _, selected_alpha = min(scores)
    return (
        weighted_ridge_fit(x, y, weight, alpha=selected_alpha),
        names,
        selected_alpha,
    )


def _score(prediction: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    prediction = centered_field(np.asarray(prediction, dtype=np.float64))
    target = centered_field(np.asarray(target, dtype=np.float64))
    rho = float(spearmanr(prediction, target).statistic)
    mse = float(np.mean(np.square(prediction - target)))
    denominator = np.linalg.norm(prediction) * np.linalg.norm(target)
    cosine = float(np.dot(prediction, target) / denominator) if denominator > 0 else np.nan
    return rho, mse, cosine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--artifact-root", type=Path, default=ROOT)
    parser.add_argument("--g1-root", type=Path, required=True)
    parser.add_argument("--g0-root", type=Path, required=True)
    parser.add_argument("--direct-transfer-contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--export-feature-table",
        action="store_true",
        help=(
            "Write the complete contact-level readout feature table.  This is "
            "used by data-aligned follow-up readouts and is disabled by default "
            "so the frozen v0.2 artifact schema does not change."
        ),
    )
    args = parser.parse_args()
    g1_root = args.g1_root.resolve()
    target_unlock = _authorize_target_access(
        g1_root, args.direct_transfer_contract
    )
    artifact = args.artifact_root.resolve()
    g0_root = args.g0_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fold_dir = g1_root / f"seed_{PRIMARY_SEED}" / args.heldout_subject
    g1_done = json.loads((fold_dir / "DONE.json").read_text())
    if bool(g1_done.get("target_values_read", True)):
        raise RuntimeError("source HistoryRNN checkpoint violated the target seal")
    (
        event_model,
        matched,
        unordered,
        chronological,
        embedding_mean,
        embedding_scale,
    ) = _load_models(fold_dir, device)
    inventory = pd.read_csv(g0_root / "seizure_causal_history_inventory.csv")
    inventory = inventory.loc[inventory.g2_metadata_eligible].copy()
    target_root = artifact / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
    frames = []
    for subject, subject_inventory in inventory.groupby("subject", sort=True):
        frames.append(
            _seizure_features(
                subject,
                subject_inventory,
                artifact=artifact,
                g0_root=g0_root,
                target_root=target_root,
                event_model=event_model,
                matched=matched,
                unordered=unordered,
                chronological=chronological,
                embedding_mean=embedding_mean,
                embedding_scale=embedding_scale,
                device=device,
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    if args.export_feature_table:
        frame.to_csv(
            output / "all_subject_readout_features.csv.gz",
            index=False,
            compression="gzip",
        )
    train = frame.loc[frame.subject != args.heldout_subject].copy()
    test = frame.loc[frame.subject == args.heldout_subject].copy()
    if test.empty:
        raise RuntimeError("heldout subject has no eligible G2 target")
    prediction_rows = []
    selected = {}
    fitted = {}
    for model_name in ("M0", "M1", "E0p5", "E2", "E6", "EM", "R2"):
        ridge, names, alpha = _fit_model(train, model_name)
        fitted[model_name] = ridge
        x, test_names = _features(test, model_name)
        if names != test_names:
            raise RuntimeError("readout feature order drifted")
        prediction = weighted_ridge_predict(ridge, x)
        selected[model_name] = {
            "alpha": alpha,
            "feature_names": names,
            "coefficient": np.asarray(ridge["coefficient"]).tolist(),
        }
        for row_index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "subject": row.subject,
                    "seizure_idx": int(row.seizure_idx),
                    "seizure_id": str(row.seizure_id),
                    "history_fingerprint": str(row.history_fingerprint),
                    "contact_index": int(row.contact_index),
                    "contact": row.contact,
                    "model": model_name,
                    "prediction": float(prediction[row_index]),
                    "target_z": float(row.target_z),
                    "target_energy": float(row.target_energy),
                }
            )
    # Apply the exact same R2 readout to a strict causal-prefix order control.
    shuffled_prediction = weighted_ridge_predict(
        fitted["R2"], _r2_shuffled_features(test)
    )
    for row_index, (_, row) in enumerate(test.iterrows()):
        prediction_rows.append(
            {
                "subject": row.subject,
                "seizure_idx": int(row.seizure_idx),
                "seizure_id": str(row.seizure_id),
                "history_fingerprint": str(row.history_fingerprint),
                "contact_index": int(row.contact_index),
                "contact": row.contact,
                "model": "R2_ORDER_SHUFFLE",
                "prediction": float(shuffled_prediction[row_index]),
                "target_z": float(row.target_z),
                "target_energy": float(row.target_energy),
            }
        )
    zero_state_prediction = weighted_ridge_predict(
        fitted["R2"], _r2_zero_state_features(test)
    )
    for row_index, (_, row) in enumerate(test.iterrows()):
        prediction_rows.append(
            {
                "subject": row.subject,
                "seizure_idx": int(row.seizure_idx),
                "seizure_id": str(row.seizure_id),
                "history_fingerprint": str(row.history_fingerprint),
                "contact_index": int(row.contact_index),
                "contact": row.contact,
                "model": "R2_ZERO_STATE",
                "prediction": float(zero_state_prediction[row_index]),
                "target_z": float(row.target_z),
                "target_energy": float(row.target_energy),
            }
        )
    for model_name in ("E2", "EM"):
        shuffled_prediction = weighted_ridge_predict(
            fitted[model_name], _ewma_shuffled_features(test, model_name)
        )
        for row_index, (_, row) in enumerate(test.iterrows()):
            prediction_rows.append(
                {
                    "subject": row.subject,
                    "seizure_idx": int(row.seizure_idx),
                    "seizure_id": str(row.seizure_id),
                    "history_fingerprint": str(row.history_fingerprint),
                    "contact_index": int(row.contact_index),
                    "contact": row.contact,
                    "model": f"{model_name}_TIME_SHUFFLE",
                    "prediction": float(shuffled_prediction[row_index]),
                    "target_z": float(row.target_z),
                    "target_energy": float(row.target_energy),
                }
            )
    predictions = pd.DataFrame(prediction_rows)
    seizure_rows = []
    for (seizure_id, model), group in predictions.groupby(
        ["seizure_id", "model"], sort=True
    ):
        rho, mse, cosine = _score(
            group.prediction.to_numpy(), group.target_z.to_numpy()
        )
        seizure_rows.append(
            {
                "subject": args.heldout_subject,
                "seizure_id": seizure_id,
                "model": model,
                "spearman_rho": rho,
                "centered_mse": mse,
                "cosine": cosine,
            }
        )
    seizure_metrics = pd.DataFrame(seizure_rows)

    # G3 within-patient state pairing: keep M1/contact layout fixed and swap
    # only the two chronology-derived contact fields across distinct seizures.
    wrong_rows = []
    seizures = sorted(test.seizure_id.astype(str).unique())
    if len(seizures) >= 2:
        pairing_columns = {
            "R2": ["history_part", "history_rank"],
            "E2": ["ewma_2_part", "ewma_2_rank"],
            "EM": [
                "ewma_0p5_part", "ewma_0p5_rank",
                "ewma_2_part", "ewma_2_rank",
                "ewma_6_part", "ewma_6_rank",
            ],
        }
        for model_name, history_columns in pairing_columns.items():
            ridge = fitted[model_name]
            for shift in range(1, len(seizures)):
                wrong = test.copy()
                mapping = {
                    seizure: seizures[(index + shift) % len(seizures)]
                    for index, seizure in enumerate(seizures)
                }
                lookup = test.set_index(
                    ["seizure_id", "contact_index"]
                )[history_columns]
                for index, row in wrong.iterrows():
                    source = mapping[str(row.seizure_id)]
                    wrong.loc[index, history_columns] = lookup.loc[
                        (source, int(row.contact_index))
                    ].to_numpy()
                x_wrong, _ = _features(wrong, model_name)
                pred_wrong = weighted_ridge_predict(ridge, x_wrong)
                wrong["prediction"] = pred_wrong
                for seizure_id, group in wrong.groupby("seizure_id", sort=True):
                    rho, _, _ = _score(
                        group.prediction.to_numpy(), group.target_z.to_numpy()
                    )
                    wrong_rows.append(
                        {
                            "subject": args.heldout_subject,
                            "seizure_id": seizure_id,
                            "model": model_name,
                            "shift": shift,
                            "wrong_pair_rho": rho,
                        }
                    )
    wrong_pair = pd.DataFrame(
        wrong_rows,
        columns=["subject", "seizure_id", "model", "shift", "wrong_pair_rho"],
    )
    # Evaluation-only seizure-specific residual: heldout targets never enter
    # readout fitting.  Use only patients with >=3 distinct histories so the
    # patient-local mean is not a two-seizure identity.
    residual_rows = []
    distinct = test[["seizure_id", "history_fingerprint"]].drop_duplicates()
    if distinct.history_fingerprint.nunique() >= 3:
        wide = predictions.pivot_table(
            index=["seizure_id", "contact_index"],
            columns="model",
            values="prediction",
        ).reset_index()
        target = test[["seizure_id", "contact_index", "target_z"]]
        wide = wide.merge(target, on=["seizure_id", "contact_index"], validate="one_to_one")
        seizures_for_residual = sorted(wide.seizure_id.astype(str).unique())
        target_matrix = np.row_stack([
            wide.loc[wide.seizure_id.astype(str) == seizure].sort_values("contact_index").target_z
            for seizure in seizures_for_residual
        ])
        from src.topic5_history_bridge import leave_one_seizure_out_residual
        target_residual = leave_one_seizure_out_residual(target_matrix)
        for model_name in ("R2", "E2", "EM"):
            dynamic_matrix = np.row_stack([
                (
                    wide.loc[wide.seizure_id.astype(str) == seizure]
                    .sort_values("contact_index")[model_name].to_numpy()
                    - wide.loc[wide.seizure_id.astype(str) == seizure]
                    .sort_values("contact_index")["M1"].to_numpy()
                )
                for seizure in seizures_for_residual
            ])
            dynamic_residual = leave_one_seizure_out_residual(dynamic_matrix)
            for index, seizure in enumerate(seizures_for_residual):
                rho, mse, cosine = _score(
                    dynamic_residual[index], target_residual[index]
                )
                residual_rows.append({
                    "subject": args.heldout_subject,
                    "seizure_id": seizure,
                    "model": model_name,
                    "residual_rho": rho,
                    "residual_mse": mse,
                    "residual_cosine": cosine,
                })
    residual = pd.DataFrame(
        residual_rows,
        columns=[
            "subject", "seizure_id", "model", "residual_rho",
            "residual_mse", "residual_cosine",
        ],
    )
    predictions.to_csv(output / "heldout_contact_predictions.csv", index=False)
    seizure_metrics.to_csv(output / "heldout_seizure_metrics.csv", index=False)
    wrong_pair.to_csv(output / "heldout_wrong_state_pairing.csv", index=False)
    residual.to_csv(output / "heldout_seizure_specific_residual.csv", index=False)
    model_mean = seizure_metrics.groupby("model").mean(numeric_only=True).to_dict("index")
    result = {
        "status": "COMPLETE",
        "contract": "topic5_history_rnn_direct_early_ictal_transfer_v0_2_loso",
        "heldout_subject": args.heldout_subject,
        "target_values_read": True,
        "target_unlock": target_unlock,
        "history_checkpoint_provenance": {
            "source_fold": str(fold_dir),
            "seed": int(g1_done["seed"]),
            "history_cycles": int(g1_done["config"]["history_cycles"]),
            "learning_rate": float(g1_done["config"]["learning_rate"]),
            "history_dim": int(g1_done["config"]["history_dim"]),
            "initial_half_life_hours": float(
                g1_done["config"]["initial_half_life_hours"]
            ),
            "target_values_read": bool(g1_done["target_values_read"]),
        },
        "n_train_patients": int(train.subject.nunique()),
        "n_test_seizures": int(test.seizure_id.nunique()),
        "n_test_contacts": int(test.contact.nunique()),
        "selected_readouts": selected,
        "contact_feature_contract": list(CONTACT_FEATURE_CONTRACT),
        "heldout_mean_metrics": model_mean,
        "primary_rho_increment_R2_minus_M1": float(
            model_mean["R2"]["spearman_rho"] - model_mean["M1"]["spearman_rho"]
        ),
        "rho_increment_E2_minus_M1": float(
            model_mean["E2"]["spearman_rho"] - model_mean["M1"]["spearman_rho"]
        ),
        "rho_increment_EM_minus_M1": float(
            model_mean["EM"]["spearman_rho"] - model_mean["M1"]["spearman_rho"]
        ),
        "rho_true_R2_minus_order_shuffle": float(
            model_mean["R2"]["spearman_rho"]
            - model_mean["R2_ORDER_SHUFFLE"]["spearman_rho"]
        ),
        "rho_true_R2_minus_zero_state": float(
            model_mean["R2"]["spearman_rho"]
            - model_mean["R2_ZERO_STATE"]["spearman_rho"]
        ),
        "rho_true_E2_minus_time_shuffle": float(
            model_mean["E2"]["spearman_rho"]
            - model_mean["E2_TIME_SHUFFLE"]["spearman_rho"]
        ),
        "rho_true_EM_minus_time_shuffle": float(
            model_mean["EM"]["spearman_rho"]
            - model_mean["EM_TIME_SHUFFLE"]["spearman_rho"]
        ),
        "n_seizure_specific_residuals": int(len(residual)),
        "history_state_diagnostics": {
            key: float(test[key].drop_duplicates().median())
            for key in (
                "history_state_variance",
                "history_state_final_l2",
                "history_state_step_rms",
            )
        },
    }
    (output / "DONE.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
