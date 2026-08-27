"""Load frozen R1.7A states and assemble D_mechanism-only N=100 T2."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from . import contract
from .coverage import CoverageTable
from .optimizer_runtime import load_explicit_target_model
from .r1_2 import _query_states, filtered_anchor_states, load_full_admissible_event_stream
from .r1_3 import materialize_embedding
from .r1_7 import R1_7A_REVISION
from .t2_human import FittedT1Context, _event_coverage_segment
from .t2_r2_human import build_r2_arm_designs


R1_7_T2_REVISION = "r1_7a_d_mechanism_t2_r2_n100_v1"


EXPECTED_SUPPORT_LIMITS = (
    "too few TRAIN events for cross-fitting",
    "load innovation is degenerate on TRAIN",
    "composition innovation is degenerate on TRAIN",
    "state-matched placebo has too few TRAIN donors",
    "T2-S1 has no exact one-step pairs",
    "T2-R2.0 H5 has no within-segment pairs",
    "T2-R2.0 H10 has no within-segment pairs",
)


def is_expected_support_limit(error: ValueError) -> bool:
    """Return true only for pre-declared data-support failures.

    Shape, alignment, checkpoint, and other implementation errors must fail the
    job instead of being silently relabelled as a scientifically unestimable
    patient.
    """
    return str(error) in EXPECTED_SUPPORT_LIMITS


def load_fitted_r1_7a_t1(
    subject: str, seed: int, *, device: str = "cuda",
    root: Path | None = None,
    r1_6_root: Path | None = None,
) -> FittedT1Context:
    root = Path(root or contract.RESULT_ROOT / "r1_7a")
    r1_6_root = Path(
        r1_6_root or contract.RESULT_ROOT / "optimizer_identifiability_r1_6"
    )
    summary = json.loads((root / "reports/r1_7a_summary.json").read_text())
    if subject not in summary.get("t2_run_subjects", []):
        raise ValueError(f"{subject}: not eligible for R1.7A T2")
    result_path = root / "fits" / subject / f"seed_{seed}/result.json"
    result = json.loads(result_path.read_text())
    if (result.get("status") != "COMPLETE"
            or result.get("revision") != R1_7A_REVISION
            or result.get("stable_checkpoint") is not True
            or result.get("development_validation_used_for_selection") is not False
            or result.get("sealed_opened") is not False):
        raise ValueError(f"invalid/stable-ineligible R1.7A result: {result_path}")
    frozen = json.loads((r1_6_root / "reports/recommended_optimizer_config.json").read_text())
    upstream = root / "upstream_r1_2"
    loaded = load_explicit_target_model(
        subject=subject, seed=seed, device=device, r1_2_root=upstream,
        observation_cache_root=root / "cache", output_root=root,
        prefix_config_id=frozen["prefix_core"]["config_id"],
    )
    checkpoint = Path(result["checkpoint"])
    if contract.sha256_file(checkpoint) != result["checkpoint_sha256"]:
        raise ValueError("R1.7A checkpoint hash mismatch")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if (payload.get("revision") != R1_7A_REVISION
            or payload.get("subject") != subject or payload.get("seed") != seed):
        raise ValueError("R1.7A checkpoint payload mismatch")
    model = loaded["model"]; model.load_state_dict(payload["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    design, loader = loaded["design"], loaded["loader"]
    embedding = materialize_embedding(
        model, design, loader, device=device, batch_size=64, use_amp=False
    )
    with torch.no_grad():
        anchor_state = filtered_anchor_states(model, design, embedding, device=device)
        rows = np.arange(len(design.event_time), dtype=np.int64)
        pre_event_state = _query_states(
            model, design, anchor_state, design.event_source_anchor,
            design.event_time, design.event_session, rows, device=device,
        ).float().cpu().numpy()
    coverage = CoverageTable.load(upstream / "coverage" / f"{subject}.npz")
    stream = load_full_admissible_event_stream(subject, coverage)
    event_observation = np.zeros((len(design.event_time), embedding.shape[1]), dtype=np.float32)
    observed = design.event_source_anchor >= 0
    event_observation[observed] = embedding[design.event_source_anchor[observed]]
    segment = _event_coverage_segment(coverage, design.event_time)
    d_state = result["d_state"]["support"]
    audit = {
        "subject": subject, "seed": seed,
        "t1_source": "r1_7a_prospective_explicit_persistent_state",
        "r1_7a_result": str(result_path),
        "r1_7a_result_sha256": contract.sha256_file(result_path),
        "r1_7a_checkpoint": str(checkpoint),
        "r1_7a_checkpoint_sha256": result["checkpoint_sha256"],
        "persistent_minus_memoryless_joint": result["d_state"][
            "persistent_minus_memoryless"
        ]["joint_nll_per_event"],
        "correct_minus_wrong_joint": result["d_state"][
            "strict_matched_wrong_time"
        ]["correct_minus_wrong_median"]["joint_nll_per_event"],
        "seed_stable_t1": True,
        "d_mechanism_start": float(d_state["mechanism_start"]),
        "d_mechanism_stop": float(d_state["mechanism_stop"]),
        "development_validation_used_for_selection": False,
        "formal_test_partition_opened": False, "sealed_opened": False,
    }
    return FittedT1Context(
        model=model, design=design, coverage=coverage, stream=stream,
        pre_event_state=np.asarray(pre_event_state, dtype=np.float32),
        event_segment=segment, audit=audit,
        pre_event_observation=event_observation,
        anchor_embedding=np.asarray(embedding, dtype=np.float32),
    )


def build_r1_7a_r2_designs(context: FittedT1Context, *, source: str):
    """Four-arm design: no edge, real, nonoverlap placebo, current event."""
    one_step, horizons, audit = build_r2_arm_designs(
        context, source=source, scale_events=100,
        validation_time_lower=float(context.audit["d_mechanism_start"]),
        include_fitted_intercept_diagnostic=False,
    )
    expected = {
        "no_edge", "real_cumulative", "state_matched_placebo",
        "current_event_only",
    }
    if set(one_step) != expected or any(set(value) != expected for value in horizons.values()):
        raise RuntimeError("R1.7A T2 did not preserve the frozen four-arm contract")
    if audit["free_exposure_intercept_present"] is not False:
        raise RuntimeError("R1.7A T2 exposed a free exposure intercept")
    audit["r1_7_t2_revision"] = R1_7_T2_REVISION
    return one_step, horizons, audit
