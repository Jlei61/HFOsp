#!/usr/bin/env python3
"""Build one shared-coordinate outer-fold cache for Topic 5 v0.4."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.topic5_history_bridge import causal_contact_features  # noqa: E402
from src.topic5_history_rnn import encode_within_event  # noqa: E402
from src.topic5_rank_distribution import LinearStateSequenceRNN  # noqa: E402
from src.topic5_static_ab import load_frozen_static_scaffold  # noqa: E402


DEVELOPMENT_SUBJECT = "epilepsiae_1146"
SOURCE_SEED = 20260725


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
    """Encode all events in one outer fold with a single target-blind model."""

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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time_summary(
    embedding: np.ndarray,
    event_time: np.ndarray,
    cutoff: float,
    *,
    tau_hours: float,
) -> np.ndarray:
    age = np.maximum(float(cutoff) - event_time, 0.0)
    weight = np.exp(-age / (float(tau_hours) * 3600.0))
    ewma = np.sum(embedding * weight[:, None], axis=0) / max(float(weight.sum()), 1e-12)
    return np.concatenate(
        [
            ewma,
            embedding.mean(0),
            embedding.max(0),
            embedding[-1],
            np.asarray(
                [np.log1p(len(embedding)), np.log1p(max(event_time[-1] - event_time[0], 0.0))]
            ),
        ]
    ).astype(np.float32)


def _atomic_npz(path: Path, **arrays) -> None:
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def _load_encoder(
    artifact: Path,
    history_root: Path,
    heldout: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, dict]:
    event_checkpoint = (
        artifact
        / "results/topic5_rnn_training_sufficiency_v0_1/formal/converged_teacher_forced"
        / f"seed_{SOURCE_SEED}/{heldout}/checkpoint.pt"
    )
    history_fold = history_root / f"seed_{SOURCE_SEED}" / heldout
    history_checkpoint = history_fold / "checkpoint.pt"
    history_done_path = history_fold / "DONE.json"
    event_payload = torch.load(event_checkpoint, map_location="cpu", weights_only=False)
    history_payload = torch.load(history_checkpoint, map_location="cpu", weights_only=False)
    history_done = json.loads(history_done_path.read_text())
    if event_payload.get("heldout_subject") != heldout:
        raise RuntimeError("event encoder outer-fold mismatch")
    if history_payload.get("heldout_subject") != heldout:
        raise RuntimeError("history checkpoint outer-fold mismatch")
    if bool(event_payload.get("ictal_target_read", True)):
        raise RuntimeError("event encoder is not target blind")
    if bool(history_payload.get("ictal_target_read", True)):
        raise RuntimeError("history initialization is not target blind")
    event_hash = _sha256(event_checkpoint)
    if history_done.get("event_checkpoint_sha256") != event_hash:
        raise RuntimeError("history/event checkpoint provenance mismatch")
    model = LinearStateSequenceRNN(**event_payload["model_kwargs"])
    model.load_state_dict(event_payload["model_state"])
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    provenance = {
        "outer_heldout_subject": heldout,
        "source_seed": SOURCE_SEED,
        "event_checkpoint": str(event_checkpoint),
        "event_checkpoint_sha256": event_hash,
        "history_checkpoint": str(history_checkpoint),
        "history_checkpoint_sha256": _sha256(history_checkpoint),
        "history_target_values_read": bool(history_payload["ictal_target_read"]),
        "event_target_values_read": bool(event_payload["ictal_target_read"]),
    }
    return model, history_payload, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout-subject", required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("/home/honglab/leijiaxin/HFOsp"))
    parser.add_argument(
        "--g0-root",
        type=Path,
        default=ROOT / "results/topic5_history_rnn_early_ictal_field/g0_causal_prefix",
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        default=ROOT / "results/topic5_history_rnn_direct_early_ictal_transfer_v0_2/g1_refit_c30",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "results/topic5_history_conditioned_field_refinement_v0_4",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--embedding-batch-size", type=int, default=8192)
    parser.add_argument("--tau-hours", type=float, default=2.0)
    args = parser.parse_args()
    artifact = args.artifact_root.resolve()
    g0 = args.g0_root.resolve()
    output = args.output_root.resolve() / "cache" / f"outer_{args.heldout_subject}"
    if (output / "DONE.json").exists():
        print((output / "DONE.json").read_text(), end="")
        return
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    event_model, history_payload, provenance = _load_encoder(
        artifact, args.history_root.resolve(), args.heldout_subject, device
    )
    manifest = json.loads((args.output_root.resolve() / "INPUT_MANIFEST.json").read_text())
    primary = list(manifest["cohort"]["primary_subjects"])
    subjects = primary + ([DEVELOPMENT_SUBJECT] if args.heldout_subject == DEVELOPMENT_SUBJECT else [])
    if args.heldout_subject not in subjects:
        raise RuntimeError("heldout subject is outside the frozen v0.4 cohort")
    inventory = pd.read_csv(
        g0 / "seizure_causal_history_inventory.csv",
        dtype={"subject": str, "seizure_id": str},
    )
    inventory = inventory.loc[
        inventory.g2_metadata_eligible.astype(bool) & inventory.subject.isin(subjects)
    ].copy()
    target45_root = artifact / "results/topic5_ictal_recruitment/t0_feature_cache"
    target150_root = artifact / "results/topic5_ictal_recruitment/t0_feature_cache_bb150_1_150"
    dataset_root = artifact / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject"
    mean = np.asarray(history_payload["event_embedding_mean"], np.float32)
    scale = np.asarray(history_payload["event_embedding_scale"], np.float32)
    entries = []
    start_time = time.time()
    for subject_index, subject in enumerate(subjects, start=1):
        dataset_path = dataset_root / f"{subject}.npz"
        with np.load(dataset_path, allow_pickle=False) as data:
            group_ids = np.asarray(data["event_group_ids"], np.int16)
            group_count = np.asarray(data["event_group_count"], np.int16)
            participation = np.asarray(data["event_participation"], np.uint8)
            base_contact_features = np.asarray(data["contact_features"], np.float32)
            contact_names = np.asarray(data["contact_names"]).astype(str)
        with np.load(g0 / "timeline" / f"{subject}.npz", allow_pickle=False) as data:
            event_time = np.asarray(data["event_time"], np.float64)
            segment_id = np.asarray(data["event_segment_id"], np.int32)
            target_contact_index = np.asarray(data["target_contact_index"], np.int64)
        scaffold = load_frozen_static_scaffold(artifact, subject, contact_names)
        valid = np.asarray(scaffold.pop("scaffold_valid"), bool)
        static_a = np.asarray(scaffold["scaffold_field_a"], np.float32)[valid]
        static_b = np.asarray(scaffold["scaffold_field_b"], np.float32)[valid]
        with (
            np.load(target45_root / f"{subject}.npz", allow_pickle=False) as target45,
            np.load(target150_root / f"{subject}.npz", allow_pickle=False) as target150,
        ):
            for seizure in inventory.loc[inventory.subject == subject].itertuples(index=False):
                cutoff = float(seizure.clinical_onset_epoch) - float(seizure.guard_seconds)
                lower = (
                    float(seizure.previous_postictal_end_epoch)
                    if np.isfinite(float(seizure.previous_postictal_end_epoch))
                    else -np.inf
                )
                candidate = np.flatnonzero(
                    (segment_id == int(seizure.segment_id))
                    & (event_time < cutoff)
                    & (event_time >= lower)
                )
                if len(candidate) != int(seizure.n_causal_events):
                    raise RuntimeError(f"{subject}/{seizure.seizure_id}: causal prefix drift")
                if len(candidate) < 1 or np.any(np.diff(event_time[candidate]) < 0):
                    raise RuntimeError(f"{subject}/{seizure.seizure_id}: invalid event timeline")
                causal_features, _ = causal_contact_features(
                    base_contact_features, participation[candidate]
                )
                event_state, contact_embedding = _event_embedding(
                    event_model,
                    causal_features,
                    group_ids[candidate],
                    group_count[candidate],
                    device=device,
                    batch_size=args.embedding_batch_size,
                )
                normalized = ((event_state - mean) / scale).astype(np.float32)
                primary_target = np.asarray(
                    target45[f"bb_auc__{int(seizure.seizure_idx)}"], np.float32
                ).squeeze()[target_contact_index][valid]
                sensitivity_target = np.asarray(
                    target150[f"bb150_auc__{int(seizure.seizure_idx)}"], np.float32
                ).squeeze()[target_contact_index][valid]
                cache_name = f"{subject}__{seizure.seizure_id}.npz"
                cache_path = output / cache_name
                _atomic_npz(
                    cache_path,
                    event_embedding=normalized,
                    event_time=event_time[candidate].astype(np.float64),
                    cutoff_time=np.asarray(cutoff, np.float64),
                    time_summary=_time_summary(
                        normalized,
                        event_time[candidate],
                        cutoff,
                        tau_hours=args.tau_hours,
                    ),
                    contact_embedding=np.asarray(contact_embedding, np.float32)[valid],
                    static_a=static_a,
                    static_b=static_b,
                    target_1_45=primary_target,
                    target_rank_1_45=rankdata(primary_target, method="average").astype(np.float32),
                    target_1_150=sensitivity_target,
                    contact_names=contact_names[valid],
                    source_contact_index=np.flatnonzero(valid).astype(np.int16),
                )
                entries.append(
                    {
                        "subject": subject,
                        "seizure_id": str(seizure.seizure_id),
                        "seizure_idx": int(seizure.seizure_idx),
                        "split": "heldout" if subject == args.heldout_subject else "target_train",
                        "cache_file": cache_name,
                        "sha256": _sha256(cache_path),
                        "n_events": int(len(candidate)),
                        "n_contacts": int(valid.sum()),
                        "history_span_hours": float((event_time[candidate][-1] - event_time[candidate][0]) / 3600.0),
                        "last_event_gap_hours": float((cutoff - event_time[candidate][-1]) / 3600.0),
                        "encoder_checkpoint_sha256": provenance["event_checkpoint_sha256"],
                    }
                )
        print(
            json.dumps(
                {
                    "phase": "outer_fold_cache",
                    "heldout": args.heldout_subject,
                    "subject": subject,
                    "position": subject_index,
                    "total": len(subjects),
                    "elapsed_seconds": round(time.time() - start_time, 2),
                }
            ),
            flush=True,
        )
    encoder_hashes = {entry["encoder_checkpoint_sha256"] for entry in entries}
    if encoder_hashes != {provenance["event_checkpoint_sha256"]}:
        raise RuntimeError("outer fold mixed event-embedding coordinate systems")
    index = {
        "status": "COMPLETE",
        "contract": "topic5_history_conditioned_field_refinement_v0_4_outer_cache",
        "heldout_subject": args.heldout_subject,
        "primary_endpoint": "clinical_onset_[0,10]s_1-45Hz_contact_energy",
        "sensitivity_endpoint": "1-150Hz_no_retrain",
        "outer_fold_shared_encoder": provenance,
        "event_embedding_normalization_source": provenance["history_checkpoint"],
        "n_subjects": len(subjects),
        "n_seizures": len(entries),
        "subjects": subjects,
        "entries": entries,
        "static_ab_boundary": manifest["static_ab_boundary"],
        "elapsed_seconds": time.time() - start_time,
    }
    (output / "INDEX.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "DONE.json").write_text(
        json.dumps(
            {key: value for key, value in index.items() if key != "entries"},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print((output / "DONE.json").read_text(), end="")


if __name__ == "__main__":
    main()
