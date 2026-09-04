"""Small, machine-checkable contracts for the full v0.3.5 execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import os
import random
import tempfile

import numpy as np
import torch


FORMAT_PREFIX = "group_event_state_v0_3_5"
DATASET_ROOT = Path("/data/hfosp_group_event_state_v0_1/dataset")
INPUT_ROOT = Path("/data/hfosp_group_event_state_v0_3_3/agent_c/human_inputs")
DECODER_ROOT = Path("/data/hfosp_group_event_state_v0_3_4/we_decoder")
# The causal re-run after the 2026-09-04 review writes to a parallel root so the
# original (segment_fraction-contaminated) artifacts remain as archive evidence.
OUTPUT_ROOT = Path(os.environ.get("HFOSP_GES_V035_OUTPUT_ROOT", "/data/hfosp_group_event_state_v0_3_5"))

V035_SUBJECTS = (
    "epilepsiae_253",
    "epilepsiae_922",
    "epilepsiae_1096",
    "epilepsiae_548",
    "epilepsiae_583",
    "epilepsiae_1146",
    "epilepsiae_384",
    "epilepsiae_1125",
)
V035_COHORT_EXPANSION_SUBJECTS = (
    "epilepsiae_1077",
    "epilepsiae_958",
    "yuquan_chengshuai",
    "yuquan_pengzihang",
    "yuquan_xuxinyi",
    "yuquan_zhangbichen",
    "yuquan_zhangjiaqi",
    "yuquan_zhangkexuan",
)
V035_ALL_DEVELOPMENT_SUBJECTS = V035_SUBJECTS + V035_COHORT_EXPANSION_SUBJECTS
V035_DECODER_FITS = {
    "epilepsiae_253": "epilepsiae_253__own_a",
    "epilepsiae_922": "epilepsiae_922__own_a",
    "epilepsiae_1096": "epilepsiae_1096__own_a",
    "epilepsiae_548": "epilepsiae_548__shared",
    "epilepsiae_583": "epilepsiae_583__shared",
    "epilepsiae_1146": "epilepsiae_1146__shared",
    "epilepsiae_384": "epilepsiae_384__shared",
    "epilepsiae_1125": "epilepsiae_1125__own_a",
    "epilepsiae_1077": "epilepsiae_1077__own_a",
    "epilepsiae_958": "epilepsiae_958__shared",
    "yuquan_chengshuai": "yuquan_chengshuai__shared",
    "yuquan_pengzihang": "yuquan_pengzihang__shared",
    "yuquan_xuxinyi": "yuquan_xuxinyi__own_a",
    "yuquan_zhangbichen": "yuquan_zhangbichen__own_a",
    "yuquan_zhangjiaqi": "yuquan_zhangjiaqi__shared",
    "yuquan_zhangkexuan": "yuquan_zhangkexuan__own_a",
}
CORE_HORIZONS_SECONDS = (300.0, 1800.0, 7200.0)
EXPLORATORY_HORIZONS_SECONDS = (21600.0, 28800.0)
RATE_TAUS_SECONDS = (120.0, 600.0, 1800.0, 7200.0, 28800.0)
LOCKED_SEEDS = (20260903, 20260904, 20260905, 20260906, 20260907)

WORK_PACKAGES = {
    "W0": "timescale, coverage, split and decoder provenance",
    "W1": "static to causal dynamic rate baseline q(t)",
    "W2": "step-wise frozen contact-sequence decoder modulation",
    "W3": "full-event m(t) and multi-horizon H1/H2a",
    "W4": "lag, energy, waveform and same-prefix readouts",
    "W5": "frozen interictal-state seizure transfer H2b",
    "W6": "common-drive versus burden/mark feedback H3",
    "REPORT": "machine summaries, core figures, plain and technical reports",
}
ALLOWED_STATUSES = {"PENDING", "RUNNING", "PARTIAL", "COMPLETE", "NOT_ESTIMABLE"}


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def initialise_scope_manifest(path: Path) -> dict[str, Any]:
    payload = {
        "format": f"{FORMAT_PREFIX}_full_scope_manifest_v1",
        "overall_status": "RUNNING",
        "completion_rule": "every registered package must be COMPLETE or scientifically NOT_ESTIMABLE; MVP/smoke/synthetic/single-subject never completes the goal",
        "subjects": list(V035_SUBJECTS),
        "work_packages": {
            key: {"description": value, "status": "PENDING", "evidence": []}
            for key, value in WORK_PACKAGES.items()
        },
        "development_targets_read": False,
        "sealed_partition_opened": False,
    }
    atomic_json(path, payload)
    return payload


def update_scope_manifest(path: Path, package: str, status: str, evidence: list[str] | None = None) -> dict[str, Any]:
    path = Path(path)
    if package not in WORK_PACKAGES:
        raise KeyError(package)
    if status not in ALLOWED_STATUSES:
        raise ValueError(status)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else initialise_scope_manifest(path)
    payload["work_packages"][package]["status"] = status
    if evidence is not None:
        payload["work_packages"][package]["evidence"] = list(evidence)
    statuses = [row["status"] for row in payload["work_packages"].values()]
    payload["overall_status"] = (
        "COMPLETE" if all(v in {"COMPLETE", "NOT_ESTIMABLE"} for v in statuses)
        else "RUNNING"
    )
    atomic_json(path, payload)
    return payload


@dataclass(frozen=True)
class RateTrainConfig:
    horizons_seconds: tuple[float, ...] = CORE_HORIZONS_SECONDS
    taus_seconds: tuple[float, ...] = RATE_TAUS_SECONDS
    grid_seconds: float = 300.0
    burn_in_seconds: float = 1800.0
    max_steps_static: int = 1200
    max_steps_dynamic: int = 1800
    max_steps_residual: int = 1800
    validate_every: int = 25
    patience_checks: int = 12
    lr_static: float = 3e-2
    lr_dynamic: float = 3e-3
    lr_residual: float = 1e-3
    residual_width: int = 32
    residual_depth: int = 2
    residual_gate_logit: float = -3.0
    weight_decay: float = 1e-4
    gradient_clip: float = 5.0
    seed: int = LOCKED_SEEDS[0]
    # ``observed_support`` permits wall-clock targets to span missing/excluded
    # intervals while counting only genuinely observed seconds as exposure.
    # L0 difficulty baselines use one horizon at a time.  A shared-state
    # producer may use several horizons, but then every horizon must share one
    # split sized from the largest horizon; otherwise a short-horizon FIT block
    # can leak into a long-horizon final holdout.
    window_contract: str = "same_segment_complete"
    merge_artificial_cuts_seconds: float = 0.0
    minimum_exposure_fraction: float = 1.0
    split_contract: str = "legacy_20_60_70_80"
    inner_holdout_horizons: float = 2.0
    selection_holdout_horizons: float = 3.0
    minimum_fit_horizons: float = 4.0

    def validate(self) -> "RateTrainConfig":
        if any(float(v) <= 0 for v in self.horizons_seconds + self.taus_seconds):
            raise ValueError("horizons and taus must be positive")
        if tuple(sorted(self.horizons_seconds)) != tuple(self.horizons_seconds):
            raise ValueError("horizons must be increasing")
        if tuple(sorted(self.taus_seconds)) != tuple(self.taus_seconds):
            raise ValueError("taus must be increasing")
        if min(self.grid_seconds, self.max_steps_static, self.max_steps_dynamic, self.max_steps_residual) <= 0:
            raise ValueError("invalid grid or training budget")
        if self.residual_width <= 0 or self.residual_depth <= 0:
            raise ValueError("residual width/depth must be positive")
        if not np.isfinite(self.residual_gate_logit):
            raise ValueError("residual gate initialisation must be finite")
        if self.window_contract not in {"same_segment_complete", "observed_support"}:
            raise ValueError("unknown future-window contract")
        if self.split_contract not in {
            "legacy_20_60_70_80",
            "horizon_specific_observed_time",
            "shared_multi_horizon_observed_time",
        }:
            raise ValueError("unknown split contract")
        if (
            self.window_contract == "observed_support"
            and len(self.horizons_seconds) != 1
            and self.split_contract != "shared_multi_horizon_observed_time"
        ):
            raise ValueError(
                "multi-horizon observed-support jobs require one shared split"
            )
        if self.split_contract == "horizon_specific_observed_time" and len(self.horizons_seconds) != 1:
            raise ValueError("horizon-specific split requires exactly one horizon")
        if self.split_contract == "shared_multi_horizon_observed_time" and len(self.horizons_seconds) < 2:
            raise ValueError("shared multi-horizon split requires at least two horizons")
        if not 0 < self.minimum_exposure_fraction <= 1:
            raise ValueError("minimum_exposure_fraction must be in (0,1]")
        if min(self.inner_holdout_horizons, self.selection_holdout_horizons,
               self.minimum_fit_horizons) <= 0:
            raise ValueError("long split multipliers must be positive")
        return self

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
