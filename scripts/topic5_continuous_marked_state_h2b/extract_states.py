#!/usr/bin/env python3
"""CPU-only causal state-cache producer for H2b cross-task transfer."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import pandas as pd

from src.topic5_continuous_marked_state_h2b import contract
from src.topic5_continuous_marked_state_h2b.state_extraction import (
    H2B_STATE_EXTRACTION_REVISION,
    InferenceRawAnchorReader,
    atomic_state_cache,
    build_inference_anchor_inputs,
    build_wrong_time_candidates,
    exact_deterministic_history,
    extract_causal_state_features,
    explicit_observation_summary,
    load_frozen_design,
    load_frozen_explicit_scaler,
    load_frozen_r16_checkpoint,
    materialize_inference_observation_embeddings,
    sha256_file,
    wrong_time_confounders,
)
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.history import history_names


DEFAULT_SOURCE_REPO = Path("/home/honglab/leijiaxin/HFOsp")
CURRENT_OBSERVATION_MAX_AGE_SECONDS = 30.0


def _coverage_segments(path: Path, query_time: np.ndarray
                       ) -> tuple[CoverageTable, np.ndarray, np.ndarray, dict[int, float]]:
    coverage = CoverageTable.load(path)
    segment = np.full(len(query_time), -1, dtype=np.int64)
    continuity_session = np.full(len(query_time), -1, dtype=np.int64)
    for row, (left, right, label) in enumerate(zip(
            coverage.start, coverage.stop, coverage.session)):
        selected = (query_time >= float(left)) & (query_time < float(right))
        if bool(np.any(segment[selected] >= 0)):
            raise ValueError("query time maps to overlapping coverage segments")
        segment[selected] = int(row)
        continuity_session[selected] = int(label)
    if bool(np.any(segment < 0)):
        bad = query_time[segment < 0]
        raise ValueError(
            f"{len(bad)} query anchors are outside admissible recorded coverage: "
            f"{bad.min():.6f}..{bad.max():.6f}"
        )
    starts = {int(row): float(left) for row, left in enumerate(coverage.start)}
    return coverage, segment, continuity_session, starts


def _read_exclusions(path: Path | None) -> list[tuple[float, float]]:
    if path is None:
        raise ValueError("wrong-time donors require an explicit global exclusion CSV")
    frame = pd.read_csv(path)
    required = {"interval_start_epoch", "interval_stop_epoch"}
    if not required.issubset(frame.columns):
        raise ValueError(f"global exclusion CSV requires {sorted(required)}")
    intervals = [
        (float(row.interval_start_epoch), float(row.interval_stop_epoch))
        for row in frame.itertuples(index=False)
    ]
    if not intervals:
        raise ValueError("global exclusion CSV is empty")
    return intervals


def _git_commit(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True,
        text=True, stdout=subprocess.PIPE,
    ).stdout.strip()


def run(args: argparse.Namespace) -> Path:
    if str(args.device) != "cpu":
        raise ValueError("H2b Phase 1 is currently CPU-only while R1.7 owns the GPU")
    source_repo = Path(args.source_repo_root).resolve()
    query_path = Path(args.queries).resolve()
    query = pd.read_csv(query_path)
    if "anchor_time_epoch" not in query.columns:
        raise ValueError("query CSV requires anchor_time_epoch")
    query_time = query.anchor_time_epoch.to_numpy(dtype=np.float64)
    query_id = (
        query.query_id.astype(str).to_numpy()
        if "query_id" in query.columns else np.asarray([
            f"query_{index:06d}" for index in range(len(query))
        ])
    )
    if len(np.unique(query_id)) != len(query_id):
        raise ValueError("query_id must be unique")

    checkpoint = Path(args.checkpoint).resolve()
    model, checkpoint_provenance = load_frozen_r16_checkpoint(
        checkpoint, expected_sha256=args.checkpoint_sha256,
        expected_subject=args.subject, expected_seed=args.seed,
        device="cpu", require_stable_result=True,
    )
    design, design_manifest, design_manifest_path = load_frozen_design(
        source_repo, args.subject
    )
    coverage_path = (
        source_repo
        / "results/epi_prssm/continuous_marked_state/r1/r1_2/coverage"
        / f"{args.subject}.npz"
    )
    coverage, inferred_segment, query_continuity_session, segment_start = (
        _coverage_segments(coverage_path, query_time)
    )
    if "coverage_segment_index" in query.columns:
        declared = query.coverage_segment_index.to_numpy(dtype=np.int64)
        if not np.array_equal(declared, inferred_segment):
            raise ValueError("query-declared coverage segment disagrees with frozen coverage")
        query_segment = declared
    else:
        query_segment = inferred_segment
    if "continuity_session" in query.columns:
        declared_session = query.continuity_session.to_numpy(dtype=np.int64)
        if not np.array_equal(declared_session, query_continuity_session):
            raise ValueError("query continuity session disagrees with frozen coverage")

    baseline_path = (
        source_repo
        / "results/epi_prssm/continuous_marked_state/r1/r1_2/baselines"
        / args.subject / "seed_0/models.pt"
    )
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    scaler = baseline["history_scaler"]
    query_history_unscaled = exact_deterministic_history(
        design=design, subject=args.subject, history_scaler=scaler,
        query_time_epoch=query_time,
        query_continuity_session=query_continuity_session, scaled=False,
    )
    query_history = exact_deterministic_history(
        design=design, subject=args.subject, history_scaler=scaler,
        query_time_epoch=query_time,
        query_continuity_session=query_continuity_session, scaled=True,
    )
    explicit_mean, explicit_scale, scaler_provenance = load_frozen_explicit_scaler(
        source_repo, args.subject
    )
    reader = InferenceRawAnchorReader(
        args.subject, design.event_time, source_repo_root=source_repo
    )
    inference_inputs = build_inference_anchor_inputs(
        reader, coverage, explicit_mean=explicit_mean,
        explicit_scale=explicit_scale,
        allowed_segments=np.unique(query_segment),
    )
    embedding = materialize_inference_observation_embeddings(
        model, inference_inputs, device="cpu",
        batch_size=args.embedding_batch_size,
    )
    features = extract_causal_state_features(
        model,
        observation_time_epoch=inference_inputs.anchor_time_epoch,
        observation_coverage_segment_index=(
            inference_inputs.coverage_segment_index
        ),
        observation_embedding=embedding,
        explicit_observation=inference_inputs.explicit,
        contact_mask=inference_inputs.contact_mask,
        anchor_time_epoch=query_time,
        anchor_coverage_segment_index=query_segment,
        deterministic_history=query_history,
        segment_start=segment_start,
        max_current_observation_age_seconds=CURRENT_OBSERVATION_MAX_AGE_SECONDS,
    )
    if not bool(features.observation_available.all()):
        missing = query_id[~features.observation_available]
        raise ValueError(
            f"{len(missing)} queries have no causal current observation; "
            f"risk-set availability is not equal: {missing[:5].tolist()}"
        )

    # Candidate states are generated on every event-independent observation
    # anchor.  Seizure identity is used only to exclude donor times downstream;
    # it never enters this frozen state scan.
    donor_history_unscaled = exact_deterministic_history(
        design=design, subject=args.subject, history_scaler=scaler,
        query_time_epoch=inference_inputs.anchor_time_epoch,
        query_continuity_session=inference_inputs.continuity_session,
        scaled=False,
    )
    donor_history = exact_deterministic_history(
        design=design, subject=args.subject, history_scaler=scaler,
        query_time_epoch=inference_inputs.anchor_time_epoch,
        query_continuity_session=inference_inputs.continuity_session,
        scaled=True,
    )
    donor_features = extract_causal_state_features(
        model,
        observation_time_epoch=inference_inputs.anchor_time_epoch,
        observation_coverage_segment_index=(
            inference_inputs.coverage_segment_index
        ),
        observation_embedding=embedding,
        explicit_observation=inference_inputs.explicit,
        contact_mask=inference_inputs.contact_mask,
        anchor_time_epoch=inference_inputs.anchor_time_epoch,
        anchor_coverage_segment_index=(
            inference_inputs.coverage_segment_index
        ),
        deterministic_history=donor_history,
        segment_start=segment_start,
        max_current_observation_age_seconds=CURRENT_OBSERVATION_MAX_AGE_SECONDS,
    )
    target_cov = wrong_time_confounders(
        query_history_unscaled, features.current_explicit_summary
    )
    donor_cov = wrong_time_confounders(
        donor_history_unscaled, donor_features.current_explicit_summary
    )
    global_exclusions = _read_exclusions(args.global_exclusions)
    wrong_time = build_wrong_time_candidates(
        target_time_epoch=query_time, target_segment=query_segment,
        target_confounders=target_cov,
        donor_time_epoch=inference_inputs.anchor_time_epoch,
        donor_segment=inference_inputs.coverage_segment_index,
        donor_state=donor_features.persistent_state,
        donor_confounders=donor_cov,
        n_donors=args.wrong_time_donors,
        min_separation_seconds=args.wrong_time_min_separation_minutes * 60.0,
        global_exclusion_intervals=global_exclusions,
        target_exclusion_start=(
            query.exclusion_start_epoch.to_numpy(dtype=np.float64)
            if "exclusion_start_epoch" in query.columns else None
        ),
        target_exclusion_stop=(
            query.exclusion_stop_epoch.to_numpy(dtype=np.float64)
            if "exclusion_stop_epoch" in query.columns else None
        ),
    )

    module_path = Path(__file__).resolve().parents[2] / (
        "src/topic5_continuous_marked_state_h2b/state_extraction.py"
    )
    source_hashes = {
        "state_extraction.py": sha256_file(module_path),
        "extract_states.py": sha256_file(Path(__file__).resolve()),
        "r1_state.py": sha256_file(REPO_ROOT / "src/topic5_continuous_marked_state_r1/state.py"),
        "r1_observer.py": sha256_file(REPO_ROOT / "src/topic5_continuous_marked_state_r1/observer.py"),
        "r1_r1_2.py": sha256_file(REPO_ROOT / "src/topic5_continuous_marked_state_r1/r1_2.py"),
        "r1_r1_3.py": sha256_file(REPO_ROOT / "src/topic5_continuous_marked_state_r1/r1_3.py"),
        "r1_history.py": sha256_file(REPO_ROOT / "src/topic5_continuous_marked_state_r1/history.py"),
        "r1_6_machine_audit": sha256_file(contract.R1_6_MACHINE_AUDIT),
        "query_csv": sha256_file(query_path),
        "global_exclusions": sha256_file(Path(args.global_exclusions).resolve()),
        "design_manifest": sha256_file(design_manifest_path),
        "observation_design": design_manifest["design_sha256"],
        "frozen_explicit_scaler": scaler_provenance[
            "explicit_scaler_result_sha256"
        ],
        "derived_inference_input": inference_inputs.provenance[
            "derived_inference_input_sha256"
        ],
        "coverage": sha256_file(coverage_path),
        "history_baseline": sha256_file(baseline_path),
    }
    source_hashes.update({
        f"raw_cache_{name}": digest for name, digest in
        inference_inputs.provenance["raw_cache_source_hashes"].items()
    })
    output = Path(args.output) if args.output else (
        contract.RESULT_ROOT / "state_cache" / args.subject
        / f"seed_{args.seed}" / "states.npz"
    )
    provenance = {
        **checkpoint_provenance,
        **scaler_provenance,
        **inference_inputs.provenance,
        "h2b_revision": contract.H2B_REVISION,
        "state_extraction_revision": H2B_STATE_EXTRACTION_REVISION,
        "source_repo_root": str(source_repo),
        "artifact_source_repo_commit_at_extraction": _git_commit(source_repo),
        "code_repo_commit_at_extraction": _git_commit(REPO_ROOT),
        "source_hashes": source_hashes,
        "query_input": str(query_path),
        "wrong_time_policy": (
            "same_coverage_segment_soft_confounder_distance_with_probe_adjustment"
        ),
        "wrong_time_donors": int(args.wrong_time_donors),
        "current_observation_max_age_seconds": (
            CURRENT_OBSERVATION_MAX_AGE_SECONDS
        ),
        "old_guarded_observation_cache_used_for_inference": False,
        "state_update_uses_seizure_label": False,
        "coverage_segment_index_is_unique_row": True,
        "continuity_session_used_only_for_history": True,
        "deterministic_history_names": list(history_names(
            int(np.asarray(design.event_group_ids).shape[1])
        )),
        "device": "cpu",
        "omp_num_threads": 1,
    }
    atomic_state_cache(
        output, features=features, query_id=query_id,
        provenance=provenance, wrong_time=wrong_time,
    )
    return Path(output)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--subject", required=True)
    value.add_argument("--seed", required=True, type=int)
    value.add_argument("--checkpoint", required=True, type=Path)
    value.add_argument("--checkpoint-sha256", required=True)
    value.add_argument("--queries", required=True, type=Path)
    value.add_argument("--global-exclusions", required=True, type=Path)
    value.add_argument("--source-repo-root", type=Path, default=DEFAULT_SOURCE_REPO)
    value.add_argument("--output", type=Path)
    value.add_argument("--device", default="cpu", choices=("cpu",))
    value.add_argument("--embedding-batch-size", type=int, default=256)
    value.add_argument("--wrong-time-donors", type=int, default=20)
    value.add_argument("--wrong-time-min-separation-minutes", type=float, default=30.0)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    torch.set_num_threads(1)
    path = run(args)
    print(json.dumps({"status": "COMPLETE", "output": str(path)}, indent=2))
