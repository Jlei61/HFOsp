"""Calibration-prefix contact grammar for the v0.3.3 tuning patients.

This module deliberately keeps the accepted legacy next-set/STOP objective.
It does *not* import the v0.3 product-form exact-subset scorer.  The two tuning
patients have different admissible initialisation paths:

* ``epilepsiae_253`` may use the leave-one-patient-out shared base stored in its
  locked legacy bundle.  Its old patient-local offset is always discarded.
* ``epilepsiae_916`` has no subject bundle.  It reuses only the already locked
  architecture hyperparameters and starts from deterministic random weights.

Every patient-derived statistic, optimisation update and checkpoint choice is
confined to the first 20% of recorded time (fit 0--16%, inner validation
16--20%).  Later development phases are never scored here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn

from src.topic5_group_event_state.v03.partition import recorded_epoch_at_fraction
from src.topic5_interictal_operator import build_contact_features
from src.topic5_rank_distribution import FullHistorySequenceGRU, next_set_stop_loss

from .paths import atomic_write_json, atomic_write_torch, current_commit, file_hash


DATASET_ROOT = Path("/data/hfosp_group_event_state_v0_1/dataset")
HUMAN_INPUT_ROOT = Path(
    "/data/hfosp_group_event_state_v0_3_3/agent_c/human_inputs"
)
LEGACY_ROOT = Path(
    "/home/honglab/leijiaxin/HFOsp/results/topic5_interictal_rank_distribution"
)
TEMPLATE_SUBJECT = "epilepsiae_253"
TUNING_SUBJECTS = ("epilepsiae_253", "epilepsiae_916")
LEGACY_CHECKPOINT = (
    LEGACY_ROOT
    / "runs/formal_multiseed_20260725_v1/seed_20260725"
    / TEMPLATE_SUBJECT
    / "full_history_gru_checkpoint.pt"
)
LEGACY_DATASET = (
    LEGACY_ROOT / "dataset_v0_4/per_subject" / f"{TEMPLATE_SUBJECT}.npz"
)


@dataclass(frozen=True)
class LegacyGrammarCalibrationConfig:
    batch_size: int = 1024
    max_epochs: int = 24
    patience: int = 4
    base_learning_rate: float = 1e-3
    offset_learning_rate: float = 3e-3
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    seed: int = 20260903
    min_fit_events: int = 100
    min_inner_events: int = 20
    # Non-null caps make an explicitly non-scientific smoke run.  Full runs
    # leave them at None and consume the entire calibration prefix.
    max_fit_events: int | None = None
    max_inner_events: int | None = None


@dataclass(frozen=True)
class CalibrationData:
    subject: str
    contact_names: tuple[str, ...]
    contact_features: np.ndarray
    contact_mask: np.ndarray
    group_ids: np.ndarray
    group_count: np.ndarray
    fit_rows: np.ndarray
    inner_rows: np.ndarray
    partition: Mapping[str, Any]
    feature_provenance: Mapping[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_state_hash(state: Mapping[str, Tensor]) -> str:
    """Stable identity over tensor names, dtypes, shapes and raw values."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = torch.as_tensor(state[name]).detach().cpu().contiguous()
        array = tensor.numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(json.dumps(list(array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _group_count(group_ids: np.ndarray) -> np.ndarray:
    groups = np.asarray(group_ids, dtype=np.int64)
    if groups.ndim != 2:
        raise ValueError("group_ids must be [event, contact]")
    return np.maximum(groups.max(axis=1) + 1, 0).astype(np.int64)


def _template_payload(path: Path = LEGACY_CHECKPOINT) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    required = {
        "control", "model_kwargs", "model_state", "heldout_local_offset",
        "heldout_subject", "ictal_target_read",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"architecture template is incomplete: {missing}")
    if payload["control"] != "full_history_gru":
        raise ValueError("architecture template is not the ordered contact GRU")
    if payload["heldout_subject"] != TEMPLATE_SUBJECT:
        raise ValueError("template is not the prelocked E253 held-out bundle")
    if payload["ictal_target_read"] is not False:
        raise ValueError("architecture template read an ictal target")
    return payload


def _subject_mode(subject: str) -> str:
    if subject == TEMPLATE_SUBJECT:
        return "leave_one_patient_out_base_plus_new_prefix_offset"
    if subject == "epilepsiae_916":
        return "architecture_hyperparameters_only_random_base"
    raise ValueError(f"v0.3.3 grammar pilot is locked to {TUNING_SUBJECTS}, got {subject}")


def _deterministic_cap(rows: np.ndarray, cap: int | None) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if cap is None or rows.size <= int(cap):
        return rows
    # The smoke cap remains chronological.  It must never become a random draw
    # from a later phase that could obscure the time boundary.
    return rows[: int(cap)]


def load_calibration_data(
    subject: str,
    *,
    dataset_root: Path = DATASET_ROOT,
    cfg: LegacyGrammarCalibrationConfig = LegacyGrammarCalibrationConfig(),
) -> CalibrationData:
    """Materialise only mark arrays and prefix-derived contact side information."""

    _subject_mode(subject)
    root = Path(dataset_root) / subject
    index = json.loads((root / "index.json").read_text())
    manifest_path = HUMAN_INPUT_ROOT / subject / "manifest_v3.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{subject}: missing locked v0.3.3 human manifest")
    human_manifest = json.loads(manifest_path.read_text())
    artifact_path = Path(str(human_manifest.get("input_path", "")))
    if human_manifest.get("format") != "group_event_state_v0_3_3_human_r0_input_manifest" \
            or human_manifest.get("subject") != subject \
            or human_manifest.get("role") != "tuning" \
            or human_manifest.get("sealed") is not False \
            or human_manifest.get("development_evaluation_used_for_fitting") is not False:
        raise PermissionError(f"{subject}: human input manifest is not an open tuning artifact")
    if not artifact_path.is_file() or file_hash(artifact_path) != human_manifest.get("input_npz_sha256"):
        raise ValueError(f"{subject}: locked human input is missing or changed")
    with np.load(artifact_path, allow_pickle=False) as human:
        event_time = np.asarray(human["event_time"], dtype=np.float64)
        target_bounds = np.asarray(human["target_segment_bounds"], dtype=np.float64)
        metadata = json.loads(str(np.asarray(human["metadata_json"]).item()))
    if metadata.get("subject") != subject or metadata.get("sealed") is not False:
        raise ValueError(f"{subject}: human input metadata identity differs")
    intervals = [
        SimpleNamespace(
            start_epoch=float(left), stop_epoch=float(right),
            duration_seconds=float(right - left),
        )
        for left, right in target_bounds
    ]
    grammar_fit_stop = recorded_epoch_at_fraction(intervals, 0.16)
    calibration_stop = recorded_epoch_at_fraction(intervals, 0.20)
    registered_20 = float(
        human_manifest["report"]["phase_boundaries_epoch"]["20pct"]
    )
    if abs(calibration_stop - registered_20) > 1e-6:
        raise ValueError(f"{subject}: reconstructed 20% boundary differs from locked manifest")
    fit_rows = np.flatnonzero(event_time < grammar_fit_stop)
    inner_rows = np.flatnonzero(
        (event_time >= grammar_fit_stop) & (event_time < calibration_stop)
    )
    fit_rows = _deterministic_cap(fit_rows, cfg.max_fit_events)
    inner_rows = _deterministic_cap(inner_rows, cfg.max_inner_events)
    if fit_rows.size < cfg.min_fit_events or inner_rows.size < cfg.min_inner_events:
        raise ValueError(
            f"{subject}: insufficient calibration prefix "
            f"fit={fit_rows.size}, inner={inner_rows.size}"
        )

    # Match the locked, boundary-audited event stream back to the raw rows.  The
    # two tuning streams have unique event epochs; fail closed if that changes.
    scalars = np.load(root / "scalars.npz")
    order = np.asarray(scalars["interictal_index"], dtype=np.int64)
    full_time = np.asarray(scalars["t_abs"], dtype=np.float64)[order]
    if np.unique(full_time).size != full_time.size or np.unique(event_time).size != event_time.size:
        raise ValueError(f"{subject}: event epochs are not unique; identity matching is ambiguous")
    stream_index = np.searchsorted(full_time, event_time, side="left")
    if np.any(stream_index >= full_time.size) \
            or not np.array_equal(full_time[stream_index], event_time):
        raise ValueError(f"{subject}: locked v0.3.3 events do not map to source stream")
    raw_index = order[stream_index]
    arrays = {
        name: np.load(root / meta["file"], mmap_mode="r")
        for name, meta in index["arrays"].items()
        if name in {"tied_group_id", "participation", "contact_ok"}
    }
    group_ids = np.asarray(arrays["tied_group_id"][raw_index], dtype=np.int64)
    participation = np.asarray(
        arrays["participation"][raw_index[fit_rows]], dtype=np.float32
    )
    contact_ok = np.asarray(
        arrays["contact_ok"][raw_index[fit_rows]], dtype=bool
    )
    support = participation.mean(axis=0)
    contact_mask = contact_ok.any(axis=0)
    names = tuple(str(row["lagpat_label"]) for row in index["contacts"])
    coords_path = Path(dataset_root) / subject / "coords.npy"
    coords = (
        np.asarray(np.load(coords_path), dtype=np.float64)
        if coords_path.exists()
        else np.full((len(names), 3), np.nan, dtype=np.float64)
    )
    features, feature_meta = build_contact_features(names, support, coords)
    if features.shape[0] != group_ids.shape[1]:
        raise ValueError(f"{subject}: contact feature/order mismatch")
    if not np.any(contact_mask):
        raise ValueError(f"{subject}: calibration prefix has no valid contact")

    legacy_order_audit: dict[str, Any] = {"performed": False}
    if subject == TEMPLATE_SUBJECT:
        with np.load(LEGACY_DATASET, allow_pickle=True) as old:
            old_names = tuple(str(v) for v in old["contact_names"].tolist())
        if old_names != names:
            raise ValueError("E253 current/legacy contact order differs")
        legacy_order_audit = {
            "performed": True,
            "legacy_dataset": str(LEGACY_DATASET),
            "legacy_dataset_sha256": _sha256(LEGACY_DATASET),
            "contact_order_equal": True,
            "legacy_event_statistics_loaded": False,
        }

    return CalibrationData(
        subject=subject,
        contact_names=names,
        contact_features=features,
        contact_mask=contact_mask,
        group_ids=group_ids,
        group_count=_group_count(group_ids),
        fit_rows=fit_rows,
        inner_rows=inner_rows,
        partition={
            "phase_names": [
                "CALIBRATION", "STATE_TRAIN", "STATE_SELECTION",
                "DEVELOPMENT_EVALUATION", "SEALED",
            ],
            "boundary_epochs": [
                registered_20,
                float(human_manifest["report"]["phase_boundaries_epoch"]["60pct"]),
                float(human_manifest["report"]["phase_boundaries_epoch"]["70pct"]),
                float(human_manifest["report"]["phase_boundaries_epoch"]["80pct"]),
            ],
            "grammar_fit_stop_epoch": float(grammar_fit_stop),
            "fit_interval": "recorded_time_0_to_16_percent",
            "selection_interval": "recorded_time_16_to_20_percent",
            "later_phases_scored": False,
            "boundary_source": str(manifest_path),
            "boundary_source_sha256": file_hash(manifest_path),
            "event_stream_source": str(artifact_path),
            "event_stream_source_sha256": file_hash(artifact_path),
        },
        feature_provenance={
            **feature_meta,
            "contact_vocabulary": "recording_start_montage_metadata",
            "coordinates": str(coords_path) if coords_path.exists() else "unavailable",
            "participation_support": "grammar_fit_0_to_16_percent_only",
            "contact_mask": "any_valid_waveform_in_grammar_fit_0_to_16_percent",
            "normalization_or_distribution_from_later_phase": False,
            "legacy_order_audit": legacy_order_audit,
        },
    )


class LegacyContactGrammar(nn.Module):
    """Frozen-contract wrapper around the accepted old next-set/STOP scorer."""

    def __init__(
        self,
        base: FullHistorySequenceGRU,
        contact_features: np.ndarray,
        contact_mask: np.ndarray,
        *,
        local_offset_dim: int,
    ) -> None:
        super().__init__()
        self.base = base
        self.register_buffer(
            "contact_features", torch.as_tensor(contact_features, dtype=torch.float32)
        )
        self.register_buffer(
            "contact_mask", torch.as_tensor(contact_mask, dtype=torch.bool)
        )
        self.local_offset = nn.Parameter(
            torch.zeros((contact_features.shape[0], int(local_offset_dim)), dtype=torch.float32)
        )

    def forward(self, group_ids: Tensor, group_count: Tensor) -> Mapping[str, Tensor]:
        batch = int(group_ids.shape[0])
        features = self.contact_features.unsqueeze(0).expand(batch, -1, -1)
        mask = self.contact_mask.unsqueeze(0).expand(batch, -1)
        return self.base(features, mask, group_ids, group_count, self.local_offset)

    def loss(self, group_ids: Tensor, group_count: Tensor) -> Mapping[str, Tensor]:
        return next_set_stop_loss(self(group_ids, group_count), group_ids, group_count)


def build_subject_grammar(
    data: CalibrationData,
    *,
    template_path: Path = LEGACY_CHECKPOINT,
    seed: int = 20260903,
) -> tuple[LegacyContactGrammar, dict[str, Any]]:
    payload = _template_payload(template_path)
    kwargs = {k: int(v) for k, v in payload["model_kwargs"].items()}
    torch.manual_seed(int(seed))
    base = FullHistorySequenceGRU(data.contact_features.shape[1], **kwargs)
    mode = _subject_mode(data.subject)
    if data.subject == TEMPLATE_SUBJECT:
        base.load_state_dict(payload["model_state"], strict=True)
        base_weight_source = "locked_leave_one_patient_out_shared_base"
    else:
        # Do not call load_state_dict in this branch.  In particular neither the
        # E253 base weights nor its patient-local offset may enter E916.
        base_weight_source = "deterministic_random_initialization"
    grammar = LegacyContactGrammar(
        base,
        data.contact_features,
        data.contact_mask,
        local_offset_dim=kwargs["local_offset_dim"],
    )
    provenance = {
        "mode": mode,
        "architecture_template": str(template_path),
        "architecture_template_sha256": _sha256(template_path),
        "architecture_hyperparameters": kwargs,
        "base_weight_source": base_weight_source,
        "template_heldout_subject": payload["heldout_subject"],
        "template_ictal_target_read": payload["ictal_target_read"],
        "template_patient_local_offset_loaded": False,
        "other_patient_weights_loaded": bool(data.subject == TEMPLATE_SUBJECT),
        "other_patient_weights_role": (
            "leave_one_patient_out_pretraining_excluding_E253"
            if data.subject == TEMPLATE_SUBJECT else "none"
        ),
        "initializer_seed": int(seed),
    }
    return grammar, provenance


def _rows_on_device(data: CalibrationData, rows: np.ndarray, device: torch.device):
    ids = torch.as_tensor(data.group_ids[rows], dtype=torch.long, device=device)
    count = torch.as_tensor(data.group_count[rows], dtype=torch.long, device=device)
    return ids, count


@torch.no_grad()
def evaluate_legacy_grammar(
    model: LegacyContactGrammar,
    data: CalibrationData,
    rows: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total = 0.0
    n = 0
    for lo in range(0, rows.size, int(batch_size)):
        take = rows[lo : lo + int(batch_size)]
        ids, count = _rows_on_device(data, take, device)
        event_nll = model.loss(ids, count)["event_nll"]
        total += float(event_nll.sum().cpu())
        n += int(event_nll.numel())
    return {"event_nll": total / max(n, 1), "n_events": n}


def calibrate_legacy_contact_grammar(
    subject: str,
    *,
    out_dir: Path,
    device: torch.device,
    cfg: LegacyGrammarCalibrationConfig = LegacyGrammarCalibrationConfig(),
    overwrite: bool = False,
) -> dict[str, Any]:
    """Fit and inner-select one calibration-only legacy-scoring grammar."""

    out_dir = Path(out_dir)
    checkpoint_path = out_dir / "legacy_contact_grammar_v033.pt"
    report_path = out_dir / "legacy_contact_grammar_v033.json"
    if checkpoint_path.exists() and report_path.exists() and not overwrite:
        return json.loads(report_path.read_text())
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        # Initialising the context before resetting is required by some CUDA /
        # PyTorch combinations; otherwise ``reset_peak_memory_stats(cuda:N)``
        # raises ``Invalid device argument`` despite a valid visible GPU.
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)

    data = load_calibration_data(subject, cfg=cfg)
    model, provenance = build_subject_grammar(data, seed=cfg.seed)
    model.to(device)
    if subject == TEMPLATE_SUBJECT:
        for parameter in model.base.parameters():
            parameter.requires_grad_(False)
        parameter_groups = [
            {"params": [model.local_offset], "lr": cfg.offset_learning_rate}
        ]
        trainable_role = "new_patient_offset_only"
    else:
        parameter_groups = [
            {"params": model.base.parameters(), "lr": cfg.base_learning_rate},
            {"params": [model.local_offset], "lr": cfg.offset_learning_rate},
        ]
        trainable_role = "random_base_and_new_patient_offset"
    optimizer = torch.optim.AdamW(
        parameter_groups, weight_decay=cfg.weight_decay
    )
    rng = np.random.default_rng(cfg.seed)
    best = math.inf
    best_epoch = -1
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(data.fit_rows)
        train_sum = 0.0
        train_n = 0
        for lo in range(0, order.size, cfg.batch_size):
            take = order[lo : lo + cfg.batch_size]
            ids, count = _rows_on_device(data, take, device)
            loss = model.loss(ids, count)["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            trainable = [p for p in model.parameters() if p.requires_grad]
            torch.nn.utils.clip_grad_norm_(trainable, cfg.gradient_clip)
            optimizer.step()
            train_sum += float(loss.detach().cpu()) * int(take.size)
            train_n += int(take.size)
        inner = evaluate_legacy_grammar(
            model, data, data.inner_rows, batch_size=cfg.batch_size, device=device
        )
        row = {
            "epoch": int(epoch),
            "fit_event_nll": train_sum / max(train_n, 1),
            "inner_validation_event_nll": inner["event_nll"],
        }
        history.append(row)
        if np.isfinite(inner["event_nll"]) and inner["event_nll"] < best - 1e-6:
            best = float(inner["event_nll"])
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if best_state is None:
        raise RuntimeError("no finite calibration checkpoint was selected")
    model.load_state_dict(best_state, strict=True)
    model.cpu().eval()
    elapsed = time.monotonic() - started
    peak_gpu = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
    )
    smoke = cfg.max_fit_events is not None or cfg.max_inner_events is not None
    checkpoint = {
        "format": "group_event_contact_grammar_v0_3_3_legacy_scoring",
        "subject": subject,
        "model_state": model.base.state_dict(),
        "local_offset": model.local_offset.detach().cpu(),
        "contact_features": torch.as_tensor(data.contact_features),
        "contact_mask": torch.as_tensor(data.contact_mask),
        "contact_names": list(data.contact_names),
        "architecture_hyperparameters": provenance["architecture_hyperparameters"],
        "hidden_size": int(model.base.hidden_size),
        "contact_embedding_dim": int(model.base.contact_embedding_dim),
        "n_contacts": int(model.contact_features.shape[0]),
        "architecture_provenance": provenance,
        "feature_provenance": data.feature_provenance,
        "partition": data.partition,
        "training": {
            "trainable_role": trainable_role,
            "fit_events": int(data.fit_rows.size),
            "inner_validation_events": int(data.inner_rows.size),
            "selected_epoch": best_epoch,
            "best_inner_validation_event_nll": best,
            "config": asdict(cfg),
            "later_development_rows_read_for_scoring_or_selection": False,
        },
        "scoring_contract": {
            "name": "legacy_next_set_or_STOP",
            "loss_function": "src.topic5_rank_distribution.next_set_stop_loss",
            "tied_group_semantics": "logsumexp_any_member_of_observed_tied_set",
            "exact_subset_likelihood": False,
        },
        "calibrated_grammar_frozen": True,
        "downstream_parameter_contract": {
            "base": "frozen",
            "patient_local_offset": "frozen",
            "contact_features_and_mask": "immutable_buffers",
            "allowed_trainable_addition": "external_low_rank_residual_adapters_only",
            "accepted_adapter_order": "H_mark_then_S_N_residual",
        },
        "scientific_use": not smoke,
        "source_commit": current_commit(),
    }
    # Identity includes base, newly calibrated local offset and immutable
    # contact buffers.  Downstream consumers can therefore prove that only
    # their external residual adapters changed.
    checkpoint["base_tensor_hash"] = tensor_state_hash(model.state_dict())
    atomic_write_torch(checkpoint_path, checkpoint)
    report = {
        "format": "group_event_contact_grammar_v0_3_3_legacy_scoring_report",
        "status": "SMOKE_ONLY" if smoke else "COMPLETE_CALIBRATION_PREFIX_ONLY",
        "subject": subject,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "scientific_use": not smoke,
        "best_inner_validation_event_nll": best,
        "selected_epoch": best_epoch,
        "history": history,
        "architecture_provenance": provenance,
        "feature_provenance": data.feature_provenance,
        "partition": data.partition,
        "scoring_contract": checkpoint["scoring_contract"],
        "base_tensor_hash": checkpoint["base_tensor_hash"],
        "dimensions": {
            "hidden_size": checkpoint["hidden_size"],
            "contact_embedding_dim": checkpoint["contact_embedding_dim"],
            "n_contacts": checkpoint["n_contacts"],
            "contact_feature_dim": int(model.contact_features.shape[1]),
            "local_offset_dim": int(model.base.local_offset_dim),
        },
        "calibrated_grammar_frozen": True,
        "downstream_parameter_contract": checkpoint["downstream_parameter_contract"],
        "data_counts": {
            "fit_events": int(data.fit_rows.size),
            "inner_validation_events": int(data.inner_rows.size),
            "contacts": int(len(data.contact_names)),
        },
        "leakage_audit": {
            "contact_support_and_mask_fit_on": "recorded_time_0_to_16_percent",
            "checkpoint_selected_on": "recorded_time_16_to_20_percent",
            "state_train_or_development_scored": False,
            "old_target_patient_local_offset_loaded": False,
            "E916_other_patient_weights_loaded": (
                provenance["other_patient_weights_loaded"]
                if subject == "epilepsiae_916" else None
            ),
        },
        "resources": {
            "device": str(device),
            "elapsed_seconds": float(elapsed),
            "peak_gpu_allocated_bytes": peak_gpu,
            "peak_gpu_allocated_gib": peak_gpu / (1024.0 ** 3),
            "memory_only_upper_bound_per_24GiB_gpu": (
                max(1, int((20 * 1024 ** 3) // max(peak_gpu, 1))) if peak_gpu else None
            ),
            # The model is tiny, but one job/GPU is the deliberately conservative
            # launch policy: it avoids compute contention and makes the measured
            # peak directly applicable to the full calibration run.
            "recommended_parallel_jobs_per_gpu": 1 if device.type == "cuda" else None,
        },
        "source_commit": current_commit(),
    }
    atomic_write_json(report_path, report)
    return report


def direct_legacy_score(
    model: LegacyContactGrammar, group_ids: Tensor, group_count: Tensor
) -> Mapping[str, Tensor]:
    """Named parity hook used by tests and downstream state adapters."""

    return next_set_stop_loss(model(group_ids, group_count), group_ids, group_count)


def load_calibrated_legacy_grammar(
    checkpoint_path: Path, *, device: torch.device | str = "cpu"
) -> tuple[LegacyContactGrammar, dict[str, Any]]:
    """Reconstruct a calibrated grammar without changing its scoring contract."""

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if payload.get("format") != "group_event_contact_grammar_v0_3_3_legacy_scoring":
        raise ValueError("not a v0.3.3 legacy-scoring grammar checkpoint")
    scoring = dict(payload.get("scoring_contract") or {})
    if scoring.get("name") != "legacy_next_set_or_STOP" \
            or scoring.get("exact_subset_likelihood") is not False:
        raise ValueError("contact grammar scoring contract drifted")
    features = np.asarray(payload["contact_features"], dtype=np.float32)
    mask = np.asarray(payload["contact_mask"], dtype=bool)
    kwargs = {k: int(v) for k, v in payload["architecture_hyperparameters"].items()}
    base = FullHistorySequenceGRU(features.shape[1], **kwargs)
    base.load_state_dict(payload["model_state"], strict=True)
    model = LegacyContactGrammar(
        base, features, mask, local_offset_dim=kwargs["local_offset_dim"]
    )
    with torch.no_grad():
        model.local_offset.copy_(torch.as_tensor(payload["local_offset"]).float())
    actual_hash = tensor_state_hash(model.state_dict())
    if actual_hash != payload.get("base_tensor_hash"):
        raise ValueError("calibrated grammar tensor identity mismatch")
    if payload.get("calibrated_grammar_frozen") is not True:
        raise ValueError("calibrated grammar is not declared frozen")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.to(device).eval()
    return model, payload
