"""Load a frozen R1.6 confirmation T1 without staging it as R1.5."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import contract
from .coverage import CoverageTable
from .optimizer_audit import R1_6_REVISION
from .optimizer_runtime import load_explicit_target_model
from .r1_2 import (
    _query_states,
    filtered_anchor_states,
    load_full_admissible_event_stream,
)
from .r1_3 import materialize_embedding
from .t2_human import FittedT1Context, _event_coverage_segment


R1_6_MINIMAL_H3_REVISION = "r1_6_frozen_confirmation_minimal_h3_v1"


def load_fitted_r1_6_confirmation_t1(
    subject: str,
    seed: int,
    *,
    device: torch.device | str = "cuda",
    r1_2_root: Path | None = None,
    observation_cache_root: Path | None = None,
    output_root: Path | None = None,
    embedding_batch_size: int = 64,
    require_stable: bool = True,
) -> FittedT1Context:
    """Reconstruct one provenance-preserving frozen R1.6 confirmation fit."""
    r1_2_root = Path(r1_2_root or contract.RESULT_ROOT / "r1_2")
    observation_cache_root = Path(
        observation_cache_root or contract.RESULT_ROOT / "r1_5/cache"
    )
    output_root = Path(
        output_root
        or contract.RESULT_ROOT / "optimizer_identifiability_r1_6"
    )
    tuning_path = output_root / "reports/tuning_summary.json"
    tuning = json.loads(tuning_path.read_text())
    if (
        tuning.get("status") != "COMPLETE"
        or tuning.get("selection_uses_development_validation") is not False
        or tuning.get("formal_test_partition_opened") is not False
        or tuning.get("sealed_opened") is not False
    ):
        raise ValueError("R1.6 tuning summary is not admissible")
    prefix = str(tuning["selected_prefix_config"])
    config = str(tuning["selected_config"])
    result_path = (
        output_root / "confirmation" / prefix / config
        / subject / f"seed_{int(seed)}/result.json"
    )
    result = json.loads(result_path.read_text())
    required = {
        "status": "COMPLETE",
        "revision": R1_6_REVISION,
        "subject": subject,
        "seed": int(seed),
        "selected_prefix_config": prefix,
        "selected_config": config,
        "development_validation_used_for_selection": False,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise ValueError(
                f"R1.6 confirmation field mismatch: {key}="
                f"{result.get(key)!r}, expected {expected!r}"
            )
    if require_stable and result.get("stable_checkpoint") is not True:
        raise ValueError("R1.6 confirmation checkpoint is not stable")

    loaded = load_explicit_target_model(
        subject=subject,
        seed=int(seed),
        device=device,
        r1_2_root=r1_2_root,
        observation_cache_root=observation_cache_root,
        output_root=output_root,
        prefix_config_id=prefix,
    )
    checkpoint_path = Path(result["checkpoint"])
    if contract.sha256_file(checkpoint_path) != result["checkpoint_sha256"]:
        raise ValueError("R1.6 confirmation checkpoint hash mismatch")
    payload = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    if (
        payload.get("revision") != R1_6_REVISION
        or payload.get("subject") != subject
        or payload.get("seed") != int(seed)
        or payload.get("selected_prefix_config") != prefix
        or payload.get("selected_config") != config
    ):
        raise ValueError("R1.6 confirmation checkpoint payload mismatch")
    model = loaded["model"]
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    design = loaded["design"]
    loader = loaded["loader"]
    embedding = materialize_embedding(
        model,
        design,
        loader,
        device=device,
        batch_size=int(embedding_batch_size),
        use_amp=False,
    )
    with torch.no_grad():
        anchor_state = filtered_anchor_states(
            model, design, embedding, device=device
        )
        event_rows = np.arange(len(design.event_time), dtype=np.int64)
        pre_event_state = _query_states(
            model,
            design,
            anchor_state,
            design.event_source_anchor,
            design.event_time,
            design.event_session,
            event_rows,
            state_permutation=None,
            device=device,
        ).float().cpu().numpy()

    coverage_path = r1_2_root / "coverage" / f"{subject}.npz"
    coverage = CoverageTable.load(coverage_path)
    stream = load_full_admissible_event_stream(subject, coverage)
    event_observation = np.zeros(
        (len(design.event_time), embedding.shape[1]), dtype=np.float32
    )
    observed = np.asarray(design.event_source_anchor) >= 0
    event_observation[observed] = embedding[
        np.asarray(design.event_source_anchor[observed], dtype=np.int64)
    ]
    segment = _event_coverage_segment(coverage, design.event_time)
    event_split = np.asarray(design.event_split, dtype=np.int8)
    contract.assert_development_times(
        subject, design.event_time[event_split == 0], "train"
    )
    contract.assert_development_times(
        subject, design.event_time[event_split == 1], "validation"
    )
    strict = result["validation"]["strict_matched_wrong_time"]
    audit = {
        "subject": subject,
        "seed": int(seed),
        "t1_source": "r1_6_frozen_optimizer_confirmation",
        "r1_6_result": str(result_path),
        "r1_6_result_sha256": contract.sha256_file(result_path),
        "r1_6_checkpoint": str(checkpoint_path),
        "r1_6_checkpoint_sha256": result["checkpoint_sha256"],
        "selected_prefix_config": prefix,
        "selected_config": config,
        "selected_total_epoch": int(result["fit_trace"]["selected_total_epoch"]),
        "persistent_minus_memoryless_joint": result["validation"][
            "persistent_minus_memoryless"
        ]["joint_nll_per_event"],
        "correct_minus_wrong_joint": strict[
            "correct_minus_wrong_median"
        ]["joint_nll_per_event"],
        "matched_wrong_time_anchors": int(strict["audit"]["n_matched_anchors"]),
        "seed_stable_t1": bool(result["stable_checkpoint"]),
        "n_events": int(len(design.event_time)),
        "n_recorded_segments": int(len(np.unique(segment))),
        "development_time_contract_verified": True,
        "development_validation_used_for_selection": False,
        "raw_arm_not_run_or_used": True,
        "formal_test_partition_opened": False,
        "sealed_opened": False,
    }
    return FittedT1Context(
        model=model,
        design=design,
        coverage=coverage,
        stream=stream,
        pre_event_state=np.asarray(pre_event_state, dtype=np.float32),
        event_segment=segment,
        audit=audit,
        pre_event_observation=event_observation,
        anchor_embedding=np.asarray(embedding, dtype=np.float32),
    )
