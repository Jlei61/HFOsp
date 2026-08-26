"""Small, fail-closed contract for the development-only R0.1 implementation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REVISION = "continuous_marked_state_r0_1_bridge_v2_full_mark_history"
FIT_REVISION = "bridge_fit_v4_frozen_history_residual"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "results/epi_prssm/continuous_marked_state/r0_1"
UPSTREAM_ROOT = REPO_ROOT / "results/epi_prssm/v0_1"
RAW_STATE_ROOT = REPO_ROOT / "results/epi_prssm/raw_seeg_state/r0_1"
SPLIT_MANIFEST = RAW_STATE_ROOT / "data/split_manifest.json"
COHORT_CACHE = UPSTREAM_ROOT / "cache/cohort_v0_1.pt"

PILOT_SUBJECTS = (
    "epilepsiae_620",
    "epilepsiae_958",
    "epilepsiae_139",
    "yuquan_huanghanwen",
    "yuquan_zhangjiaqi",
    "yuquan_hanyuxuan",
)

ANALYSIS_RATE_HZ = 256
BACKGROUND_SECONDS = 30.0
IED_CORE_HALF_WIDTH_SECONDS = 1.0
OBSERVATION_DIM = 32
STATE_OBSERVATION_DIM = 64
MAX_TRAIN_PAIRS = 6000
MAX_VALIDATION_PAIRS = 2500

RAW_CACHE_BY_DATASET = {
    "epilepsiae": Path("/mnt/yuquan_data/hfosp_cache/raw_seeg_state_r0_1"),
    "yuquan": Path("/mnt/epilepsia_data/hfosp_cache/raw_seeg_state_r0_1"),
}


@dataclass(frozen=True)
class SubjectSplit:
    train_end_epoch: float
    dev_end_epoch: float


def load_split(subject: str) -> SubjectSplit:
    payload = json.loads(SPLIT_MANIFEST.read_text())
    row = payload["subjects"][subject]
    return SubjectSplit(float(row["train_end_epoch"]), float(row["dev_end_epoch"]))


def assert_development_times(subject: str, times: np.ndarray, split: str) -> None:
    """Reject a selected observation/target time outside its named dev interval."""
    values = np.asarray(times, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError(f"{subject}: {split} times are empty or non-finite")
    bound = load_split(subject)
    if split == "train":
        ok = values < bound.train_end_epoch
    elif split == "validation":
        ok = (values >= bound.train_end_epoch) & (values < bound.dev_end_epoch)
    else:
        raise ValueError(f"unsupported development split {split!r}")
    if not bool(ok.all()):
        bad = values[~ok]
        raise ValueError(
            f"SEALED/SPLIT VIOLATION {subject} {split}: "
            f"range={bad.min():.6f}..{bad.max():.6f}"
        )


def raw_cache_dir(subject: str) -> Path:
    dataset = subject.split("_", 1)[0]
    return RAW_CACHE_BY_DATASET[dataset] / subject
