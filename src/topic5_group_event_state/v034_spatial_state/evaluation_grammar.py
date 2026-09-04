"""Calibration-prefix frozen grammars for locked S_P evaluation patients.

Each base checkpoint was trained leave-one-patient-out for the named patient.
Only a fresh local offset is fitted on recorded-time 0--16% and selected on
16--20%.  The state-training, development, seizure and sealed intervals are
not scored or used to construct patient-derived features.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from src.topic5_group_event_state.v03.partition import recorded_epoch_at_fraction
from src.topic5_group_event_state.v033_training_lab.contact_grammar import (
    DATASET_ROOT,
    LEGACY_ROOT,
    CalibrationData,
    LegacyContactGrammar,
    LegacyGrammarCalibrationConfig,
    _deterministic_cap,
    evaluate_legacy_grammar,
    summarize_training_adequacy,
    tensor_state_hash,
)
from src.topic5_group_event_state.v033_training_lab.paths import (
    atomic_write_json,
    atomic_write_torch,
    current_commit,
    file_hash,
)
from src.topic5_interictal_operator import build_contact_features
from src.topic5_rank_distribution import FullHistorySequenceGRU

from .contracts import EVALUATION_SUBJECTS, SEED_CONTRACT, seed_before_model_construction


HUMAN_INPUT_ROOT = Path("/data/hfosp_group_event_state_v0_3_3/agent_c/human_inputs")
LOO_ROOT = (
    LEGACY_ROOT / "runs/formal_multiseed_20260725_v1/seed_20260725"
)
LEGACY_DATASET_ROOT = LEGACY_ROOT / "dataset_v0_4/per_subject"


def _loo_checkpoint(subject: str) -> Path:
    return LOO_ROOT / subject / "full_history_gru_checkpoint.pt"


def load_evaluation_grammar_data(
    subject: str,
    *,
    input_root: Path = HUMAN_INPUT_ROOT,
    dataset_root: Path = DATASET_ROOT,
    cfg: LegacyGrammarCalibrationConfig = LegacyGrammarCalibrationConfig(),
) -> CalibrationData:
    if subject not in EVALUATION_SUBJECTS:
        raise PermissionError(f"grammar calibration is not authorized for {subject}")
    manifest_path = Path(input_root) / subject / "manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_path = Path(str(manifest.get("input_path", "")))
    if manifest.get("format") != "group_event_state_v0_3_3_human_r0_input_manifest" \
            or manifest.get("subject") != subject \
            or manifest.get("role") != "explicit_non_tuning_override" \
            or manifest.get("sealed") is not False \
            or manifest.get("development_evaluation_used_for_fitting") is not False:
        raise PermissionError("evaluation grammar requires an open non-tuning prefix manifest")
    if not artifact_path.is_file() or file_hash(artifact_path) != manifest.get("input_npz_sha256"):
        raise ValueError("evaluation grammar input bytes differ from the manifest")
    with np.load(artifact_path, allow_pickle=False) as human:
        event_time = np.asarray(human["event_time"], dtype=np.float64)
        target_bounds = np.asarray(human["target_segment_bounds"], dtype=np.float64)
        metadata = json.loads(str(np.asarray(human["metadata_json"]).item()))
    if metadata.get("subject") != subject or metadata.get("sealed") is not False:
        raise ValueError("evaluation grammar input metadata differs")
    intervals = [
        SimpleNamespace(
            start_epoch=float(left), stop_epoch=float(right),
            duration_seconds=float(right - left),
        )
        for left, right in target_bounds
    ]
    fit_stop = recorded_epoch_at_fraction(intervals, 0.16)
    calibration_stop = recorded_epoch_at_fraction(intervals, 0.20)
    registered = float(manifest["report"]["phase_boundaries_epoch"]["20pct"])
    if abs(calibration_stop - registered) > 1e-6:
        raise ValueError("reconstructed calibration boundary differs from the manifest")
    # Drop later event rows before mapping any mark array.  Keeping their
    # timestamps merely to choose the prefix would be harmless, but indexing
    # their tied-group marks would violate the calibration-only contract.
    event_time = event_time[event_time < calibration_stop]
    fit_rows = _deterministic_cap(np.flatnonzero(event_time < fit_stop), cfg.max_fit_events)
    inner_rows = _deterministic_cap(
        np.flatnonzero((event_time >= fit_stop) & (event_time < calibration_stop)),
        cfg.max_inner_events,
    )
    if fit_rows.size < cfg.min_fit_events or inner_rows.size < cfg.min_inner_events:
        raise ValueError(f"insufficient calibration events: fit={fit_rows.size}, inner={inner_rows.size}")

    root = Path(dataset_root) / subject
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    scalars = np.load(root / "scalars.npz")
    order = np.asarray(scalars["interictal_index"], dtype=np.int64)
    raw_time = np.asarray(scalars["t_abs"], dtype=np.float64)[order]
    if np.unique(raw_time).size != raw_time.size or np.unique(event_time).size != event_time.size:
        raise ValueError("event epochs are not unique")
    stream_index = np.searchsorted(raw_time, event_time, side="left")
    if np.any(stream_index >= raw_time.size) \
            or not np.array_equal(raw_time[stream_index], event_time):
        raise ValueError("evaluation grammar stream does not map to source data")
    raw_rows = order[stream_index]
    arrays = {
        name: np.load(root / meta["file"], mmap_mode="r")
        for name, meta in index["arrays"].items()
        if name in {"tied_group_id", "participation", "contact_ok"}
    }
    group_ids = np.asarray(arrays["tied_group_id"][raw_rows], dtype=np.int64)
    participation = np.asarray(arrays["participation"][raw_rows[fit_rows]], dtype=np.float32)
    contact_ok = np.asarray(arrays["contact_ok"][raw_rows[fit_rows]], dtype=bool)
    support = participation.mean(axis=0)
    contact_mask = contact_ok.any(axis=0)
    names = tuple(str(row["lagpat_label"]) for row in index["contacts"])
    coords_path = root / "coords.npy"
    coords = (
        np.asarray(np.load(coords_path), dtype=np.float64)
        if coords_path.exists() else np.full((len(names), 3), np.nan)
    )
    features, feature_meta = build_contact_features(names, support, coords)
    if features.shape[0] != group_ids.shape[1] or not np.any(contact_mask):
        raise ValueError("evaluation grammar contact features/mask are invalid")

    legacy_dataset = LEGACY_DATASET_ROOT / f"{subject}.npz"
    if not legacy_dataset.is_file():
        raise FileNotFoundError(f"missing legacy contact-order artifact: {legacy_dataset}")
    with np.load(legacy_dataset, allow_pickle=True) as old:
        old_names = tuple(str(v) for v in old["contact_names"].tolist())
    if old_names != names:
        raise ValueError("current and leave-one-out contact order differ")
    return CalibrationData(
        subject=subject,
        contact_names=names,
        contact_features=features,
        contact_mask=contact_mask,
        group_ids=group_ids,
        group_count=np.maximum(group_ids.max(1) + 1, 0).astype(np.int64),
        fit_rows=fit_rows,
        inner_rows=inner_rows,
        partition={
            "fit_interval": "recorded_time_0_to_16_percent",
            "selection_interval": "recorded_time_16_to_20_percent",
            "grammar_fit_stop_epoch": float(fit_stop),
            "calibration_stop_epoch": float(calibration_stop),
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
            "legacy_dataset": str(legacy_dataset),
            "legacy_dataset_sha256": file_hash(legacy_dataset),
            "contact_order_equal": True,
        },
    )


def _build_loo_grammar(
    data: CalibrationData,
    *,
    seed: int,
) -> tuple[LegacyContactGrammar, dict[str, Any]]:
    path = _loo_checkpoint(data.subject)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("control") != "full_history_gru" \
            or payload.get("heldout_subject") != data.subject \
            or payload.get("ictal_target_read") is not False:
        raise PermissionError("decoder base is not the target-patient-held-out non-ictal bundle")
    kwargs = {key: int(value) for key, value in payload["model_kwargs"].items()}
    seed_before_model_construction(seed)
    base = FullHistorySequenceGRU(data.contact_features.shape[1], **kwargs)
    base.load_state_dict(payload["model_state"], strict=True)
    grammar = LegacyContactGrammar(
        base, data.contact_features, data.contact_mask,
        local_offset_dim=kwargs["local_offset_dim"],
    )
    return grammar, {
        "mode": "leave_one_patient_out_base_plus_new_prefix_offset",
        "architecture_template": str(path),
        "architecture_template_sha256": file_hash(path),
        "architecture_hyperparameters": kwargs,
        "base_weight_source": "locked_leave_one_patient_out_shared_base",
        "template_heldout_subject": payload["heldout_subject"],
        "template_ictal_target_read": payload["ictal_target_read"],
        "template_patient_local_offset_loaded": False,
        "other_patient_weights_role": f"leave_one_patient_out_pretraining_excluding_{data.subject}",
        "initializer_seed": int(seed),
        "seed_contract": SEED_CONTRACT,
    }


def calibrate_evaluation_grammar(
    subject: str,
    *,
    out_dir: Path,
    device: torch.device,
    cfg: LegacyGrammarCalibrationConfig = LegacyGrammarCalibrationConfig(),
    input_root: Path = HUMAN_INPUT_ROOT,
    overwrite: bool = False,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    checkpoint_path = out_dir / "legacy_contact_grammar_v033.pt"
    report_path = out_dir / "legacy_contact_grammar_v033.json"
    if checkpoint_path.exists() and report_path.exists() and not overwrite:
        return json.loads(report_path.read_text(encoding="utf-8"))
    seed_before_model_construction(cfg.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
        torch.cuda.reset_peak_memory_stats(device)
    data = load_evaluation_grammar_data(subject, input_root=input_root, cfg=cfg)
    model, provenance = _build_loo_grammar(data, seed=cfg.seed)
    model.to(device)
    for parameter in model.base.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [{"params": [model.local_offset], "lr": cfg.offset_learning_rate}],
        weight_decay=cfg.weight_decay,
    )
    rng = np.random.default_rng(cfg.seed)
    best = math.inf
    best_epoch = -1
    best_state = None
    stale = 0
    history: list[dict[str, Any]] = []
    started = time.monotonic()
    for epoch in range(cfg.max_epochs):
        model.train()
        order = rng.permutation(data.fit_rows)
        total = 0.0
        n = 0
        for lo in range(0, order.size, cfg.batch_size):
            take = order[lo:lo + cfg.batch_size]
            ids = torch.as_tensor(data.group_ids[take], dtype=torch.long, device=device)
            count = torch.as_tensor(data.group_count[take], dtype=torch.long, device=device)
            loss = model.loss(ids, count)["total"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([model.local_offset], cfg.gradient_clip)
            optimizer.step()
            total += float(loss.detach().cpu()) * int(take.size)
            n += int(take.size)
        inner = evaluate_legacy_grammar(
            model, data, data.inner_rows, batch_size=cfg.batch_size, device=device,
        )
        history.append({
            "epoch": epoch,
            "fit_event_nll": total / max(n, 1),
            "inner_validation_event_nll": inner["event_nll"],
        })
        if np.isfinite(inner["event_nll"]) and inner["event_nll"] < best - 1e-6:
            best = float(inner["event_nll"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    if best_state is None:
        raise RuntimeError("no finite evaluation grammar checkpoint")
    adequacy = summarize_training_adequacy(
        history, selected_epoch=best_epoch, max_epochs=cfg.max_epochs, patience=cfg.patience,
    )
    model.load_state_dict(best_state, strict=True)
    model.cpu().eval()
    smoke = cfg.max_fit_events is not None or cfg.max_inner_events is not None
    scoring = {
        "name": "legacy_next_set_or_STOP",
        "loss_function": "src.topic5_rank_distribution.next_set_stop_loss",
        "tied_group_semantics": "logsumexp_any_member_of_observed_tied_set",
        "exact_subset_likelihood": False,
    }
    frozen = {
        "base": "frozen", "patient_local_offset": "frozen",
        "contact_features_and_mask": "immutable_buffers",
        "allowed_trainable_addition": "external_low_rank_residual_adapters_only",
        "accepted_adapter_order": "H_mark_then_S_N_residual",
    }
    checkpoint: dict[str, Any] = {
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
            "trainable_role": "new_patient_offset_only",
            "fit_events": int(data.fit_rows.size),
            "inner_validation_events": int(data.inner_rows.size),
            "selected_epoch": int(best_epoch),
            "best_inner_validation_event_nll": best,
            "training_adequacy": adequacy,
            "config": asdict(cfg),
            "later_development_rows_read_for_scoring_or_selection": False,
        },
        "scoring_contract": scoring,
        "calibrated_grammar_frozen": True,
        "downstream_parameter_contract": frozen,
        "scientific_use": not smoke,
        "source_commit": current_commit(),
        "seed_contract": SEED_CONTRACT,
    }
    checkpoint["base_tensor_hash"] = tensor_state_hash(model.state_dict())
    atomic_write_torch(checkpoint_path, checkpoint)
    report = {
        "format": "group_event_state_v0_3_4_spatial_state_evaluation_grammar_report_v1",
        "status": "SMOKE_ONLY" if smoke else "COMPLETE_CALIBRATION_PREFIX_ONLY",
        "subject": subject,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "scientific_use": not smoke,
        "best_inner_validation_event_nll": best,
        "selected_epoch": int(best_epoch),
        "history": history,
        "training_adequacy": adequacy,
        "architecture_provenance": provenance,
        "feature_provenance": data.feature_provenance,
        "partition": data.partition,
        "scoring_contract": scoring,
        "base_tensor_hash": checkpoint["base_tensor_hash"],
        "calibrated_grammar_frozen": True,
        "downstream_parameter_contract": frozen,
        "leakage_audit": {
            "base_pretraining_excluded_target_patient": True,
            "contact_support_and_mask_fit_on": "recorded_time_0_to_16_percent",
            "checkpoint_selected_on": "recorded_time_16_to_20_percent",
            "state_train_or_development_scored": False,
            "old_target_patient_local_offset_loaded": False,
        },
        "resources": {
            "device": str(device),
            "elapsed_seconds": float(time.monotonic() - started),
            "peak_gpu_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
        "development_targets_read": False,
        "sealed_partition_opened": False,
        "seizure_outcomes_read": False,
        "seed_contract": SEED_CONTRACT,
        "source_commit": current_commit(),
    }
    atomic_write_json(report_path, report)
    return report
