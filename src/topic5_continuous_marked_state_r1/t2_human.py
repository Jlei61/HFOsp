"""Human-data assembly for the minimal H3/T2-S1 one-step experiment."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from . import contract
from .bridge_e1 import make_paired_models
from .coverage import CoverageTable
from .r1_2 import (
    FrozenEmbeddingStateModel,
    _bridge_scaler,
    _query_states,
    filtered_anchor_states,
    load_full_anchor_cache,
    load_full_admissible_event_stream,
    load_full_design,
)
from .r1_3 import (
    FullAnchorObservationLoader,
    FullTargetObserverStateModel,
    materialize_embedding,
)
from .t2_s1 import (
    OneStepDesign,
    build_one_step_design,
    fit_load_innovation,
    rolling_event_exposure,
    state_matched_placebo,
)


T2_HUMAN_REVISION = "t2_s1_human_n100_reference_n1000_primary_v1"


@dataclass(frozen=True)
class FittedT1Context:
    model: FullTargetObserverStateModel
    design: object
    coverage: CoverageTable
    stream: object
    pre_event_state: np.ndarray
    event_segment: np.ndarray
    audit: dict


def load_fitted_r1_2_explicit_t1(
    subject: str,
    seed: int,
    *,
    device: torch.device | str = "cuda",
    r1_2_root: Path | None = None,
) -> FittedT1Context:
    """Reconstruct one frozen-observer R1.2 explicit T1 fit.

    This loader is deliberately separate from the formal R1.3 loader.  It is
    used only by the support-selected, development-only very-long H3 screen,
    where the first requirement is a non-degenerate pre-event state.  The
    downstream result records the weaker T1 source explicitly.
    """
    r1_2_root = Path(r1_2_root or contract.RESULT_ROOT / "r1_2")
    design, embedding, cache_manifest = load_full_anchor_cache(
        subject, arm="explicit", output_root=r1_2_root
    )
    baseline_path = r1_2_root / "baselines" / subject / "seed_0/models.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    coverage_path = r1_2_root / "coverage" / f"{subject}.npz"
    coverage = CoverageTable.load(coverage_path)
    stream = load_full_admissible_event_stream(subject, coverage)
    model = FrozenEmbeddingStateModel(
        baseline,
        design.event_history.shape[1],
        stream.n_contacts,
        stream.adjacency,
        observation_dim=embedding.shape[1],
        state_dim=8,
    ).to(device)
    fit_dir = (
        r1_2_root / "t1_full" / subject / f"explicit_d8_seed_{int(seed)}"
    )
    result_path = fit_dir / "result.json"
    checkpoint_path = fit_dir / "model.pt"
    result = json.loads(result_path.read_text())
    if result.get("status") != "COMPLETE" or result.get("sealed_opened") is not False:
        raise ValueError(f"invalid R1.2 explicit fit: {result_path}")
    if contract.sha256_file(checkpoint_path) != result["checkpoint_sha256"]:
        raise ValueError("R1.2 explicit checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
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
    segment = _event_coverage_segment(coverage, design.event_time)
    audit = {
        "subject": subject,
        "seed": int(seed),
        "t1_source": "r1_2_frozen_explicit_observer",
        "r1_2_result": str(result_path),
        "r1_2_result_sha256": contract.sha256_file(result_path),
        "r1_2_checkpoint": str(checkpoint_path),
        "r1_2_checkpoint_sha256": contract.sha256_file(checkpoint_path),
        "selected_epochs": int(result["selected_epochs"]),
        "cache_manifest": cache_manifest,
        "n_events": int(len(design.event_time)),
        "n_recorded_segments": int(len(np.unique(segment))),
        "sealed_opened": False,
        "claim_boundary": (
            "support-selected development H3 instrument; weaker than formal R1.3"
        ),
    }
    return FittedT1Context(
        model=model,
        design=design,
        coverage=coverage,
        stream=stream,
        pre_event_state=pre_event_state.astype(np.float32),
        event_segment=segment,
        audit=audit,
    )


def _event_coverage_segment(coverage: CoverageTable,
                            event_time: np.ndarray) -> np.ndarray:
    event_time = np.asarray(event_time, dtype=np.float64)
    segment = np.searchsorted(coverage.stop, event_time, side="right")
    if np.any(segment >= len(coverage.start)):
        raise ValueError("event occurs after the final recorded segment")
    inside = (
        (event_time >= coverage.start[segment])
        & (event_time < coverage.stop[segment])
    )
    if not bool(inside.all()):
        raise ValueError("event occurs outside recorded coverage")
    return segment.astype(np.int64)


def load_fitted_explicit_t1(subject: str, seed: int, *,
                            device: torch.device | str = "cuda",
                            r1_2_root: Path | None = None,
                            r1_2b_root: Path | None = None,
                            r1_3_root: Path | None = None,
                            observation_cache_root: Path | None = None,
                            embedding_batch_size: int = 64) -> FittedT1Context:
    """Reconstruct and freeze a completed formal explicit R1.3 checkpoint."""
    r1_2_root = Path(r1_2_root or contract.RESULT_ROOT / "r1_2")
    r1_2b_root = Path(r1_2b_root or contract.RESULT_ROOT / "r1_2b")
    r1_3_root = Path(r1_3_root or contract.RESULT_ROOT / "r1_3")
    observation_cache_root = Path(
        observation_cache_root or r1_3_root / "cache"
    )
    cache_manifest_path = observation_cache_root / subject / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    if cache_manifest.get("status") != "COMPLETE":
        raise ValueError(f"invalid R1.3 observation cache: {cache_manifest_path}")
    if cache_manifest.get("sealed_opened") is not False:
        raise ValueError("R1.3 observation cache opened the sealed partition")
    design_path = Path(cache_manifest["design"])
    if contract.sha256_file(design_path) != cache_manifest["design_sha256"]:
        raise ValueError("R1.3 observation-cache design hash mismatch")
    design = load_full_design(design_path)
    baseline_path = r1_2_root / "baselines" / subject / "seed_0/models.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    coverage_path = r1_2_root / "coverage" / f"{subject}.npz"
    coverage = CoverageTable.load(coverage_path)
    stream = load_full_admissible_event_stream(subject, coverage)
    bridge_result_path = r1_2_root / "bridge_e1" / subject / "seed_0/result.json"
    bridge_result = json.loads(bridge_result_path.read_text())
    _, _, sampled, _ = _bridge_scaler(
        subject, baseline_path, bridge_result, stream, coverage
    )
    _, raw_bridge = make_paired_models(
        baseline, sampled, stream.adjacency, seed=0, device=device
    )
    bridge_checkpoint_path = r1_2_root / "bridge_e1" / subject / "seed_0/models.pt"
    bridge_checkpoint = torch.load(
        bridge_checkpoint_path, map_location=device, weights_only=False
    )
    raw_bridge.load_state_dict(bridge_checkpoint["explicit_raw"])
    model = FullTargetObserverStateModel(
        baseline, design.event_history.shape[1], stream.n_contacts,
        stream.adjacency, raw_bridge.observer, use_raw=False, state_dim=8,
    ).to(device)
    fit_dir = r1_3_root / "fits" / subject / f"explicit_seed_{int(seed)}"
    result_path = fit_dir / "result.json"
    checkpoint_path = fit_dir / "model.pt"
    result = json.loads(result_path.read_text())
    if result.get("status") != "COMPLETE" or result.get("sealed_opened") is not False:
        raise ValueError(f"invalid formal R1.3 fit: {result_path}")
    if contract.sha256_file(checkpoint_path) != result["checkpoint_sha256"]:
        raise ValueError("formal R1.3 checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    explicit_path = Path(cache_manifest["explicit"])
    contact_mask_path = Path(cache_manifest["contact_mask"])
    if contract.sha256_file(explicit_path) != cache_manifest["explicit_sha256"]:
        raise ValueError("R1.3 explicit observation cache hash mismatch")
    if contract.sha256_file(contact_mask_path) != cache_manifest["contact_mask_sha256"]:
        raise ValueError("R1.3 contact-mask cache hash mismatch")
    loader = FullAnchorObservationLoader(
        subject, design, stream.event_time,
        sampled.explicit_mean, sampled.explicit_scale,
        cached_explicit=np.load(explicit_path, mmap_mode="r"),
        cached_contact_mask=np.load(contact_mask_path, mmap_mode="r"),
    )
    embedding = materialize_embedding(
        model, design, loader, device=device,
        batch_size=int(embedding_batch_size), use_amp=False,
    )
    with torch.no_grad():
        anchor_state = filtered_anchor_states(
            model, design, embedding, device=device
        )
        event_rows = np.arange(len(design.event_time), dtype=np.int64)
        pre_event_state = _query_states(
            model, design, anchor_state, design.event_source_anchor,
            design.event_time, design.event_session, event_rows,
            state_permutation=None, device=device,
        ).float().cpu().numpy()
    segment = _event_coverage_segment(coverage, design.event_time)
    audit = {
        "subject": subject,
        "seed": int(seed),
        "r1_3_result": str(result_path),
        "r1_3_result_sha256": contract.sha256_file(result_path),
        "r1_3_checkpoint": str(checkpoint_path),
        "r1_3_checkpoint_sha256": contract.sha256_file(checkpoint_path),
        "t1_arm": "explicit",
        "raw_result_does_not_gate_t2": True,
        "n_events": int(len(design.event_time)),
        "n_recorded_segments": int(len(np.unique(segment))),
        "sealed_opened": False,
    }
    return FittedT1Context(
        model=model, design=design, coverage=coverage, stream=stream,
        pre_event_state=pre_event_state.astype(np.float32),
        event_segment=segment, audit=audit,
    )


def build_exposure_arm_designs(context: FittedT1Context, *, scale_events: int
                               ) -> tuple[dict[str, OneStepDesign], dict]:
    """Build capacity-matched real, placebo and current-event T2-S1 arms."""
    design = context.design
    stream_load = np.asarray(context.stream.load, dtype=np.float32)
    if len(stream_load) != len(design.event_time):
        raise ValueError("full stream and full design event denominators diverged")
    train = np.asarray(design.event_split == 0)
    innovation, innovation_audit = fit_load_innovation(
        context.pre_event_state, design.event_history, stream_load, train
    )
    real, eligible = rolling_event_exposure(
        innovation, context.event_segment, int(scale_events)
    )
    placebo, placebo_matched, placebo_audit = state_matched_placebo(
        real, context.pre_event_state, design.event_history, train, eligible,
        exclusion_events=int(scale_events),
    )
    common = eligible & placebo_matched
    arm_exposure = {
        "no_edge": np.zeros_like(real),
        "real_cumulative": real,
        "state_matched_placebo": placebo,
        "current_event_only": innovation,
    }
    arms = {
        label: build_one_step_design(
            design, context.pre_event_state, context.event_segment,
            value, common,
        )
        for label, value in arm_exposure.items()
    }
    reference_index = arms["real_cumulative"].current_index
    for label, value in arms.items():
        if not np.array_equal(value.current_index, reference_index):
            raise RuntimeError(f"T2 arm support changed for {label}")
    audit = {
        "revision": T2_HUMAN_REVISION,
        "scale_events": int(scale_events),
        "exposure": "rolling signed load innovation divided by sqrt(N)",
        "load_expectation_fit": innovation_audit,
        "state_matched_placebo": placebo_audit,
        "eligible_events_before_pairing": int(common.sum()),
        "train_pairs": int((arms["real_cumulative"].split == 0).sum()),
        "validation_pairs": int((arms["real_cumulative"].split == 1).sum()),
        "arms_share_exact_support": True,
        "history_crosses_unrecorded_gap": False,
        "validation_outcome_used_to_fit_innovation": False,
        "sealed_opened": False,
    }
    return arms, audit
