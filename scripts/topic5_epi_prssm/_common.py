"""Shared runner plumbing: threads, cohort loading, job keys, expected load."""
from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)

from src.topic5_epi_prssm.cohort import (  # noqa: E402
    breadth_pilot_subjects, cohort_subjects, eligible_subjects, load_tensors,
)
from src.topic5_epi_prssm.contracts import (  # noqa: E402
    FROZEN, OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision,
    package_hash, sha256_obj,
)
from src.topic5_epi_prssm.model import PatientTensors  # noqa: E402
from src.topic5_epi_prssm.run_registry import JobKey, JobRunner, is_complete  # noqa: E402


def resolve_cohort(name: str) -> tuple[str, ...]:
    if name == "all34":
        return eligible_subjects()
    if name.startswith("breadth"):
        n = int(name.replace("breadth", "") or 8)
        return breadth_pilot_subjects(n)
    if name == "smoke":
        return ("epilepsiae_1084", "yuquan_gaolan")
    raise ValueError(f"unknown cohort {name!r}")


def expected_load_vector(patients) -> torch.Tensor:
    """Train-mean load per patient -- the only load an observer-off rollout may use."""
    values = []
    for patient in patients:
        train = patient.split == 0
        values.append(float(patient.load[train].mean()))
    return torch.as_tensor(values, dtype=torch.float32)


def input_hash(patients) -> str:
    return sha256_obj({p.subject: p.meta.get("source_hashes", {}) for p in patients})


def dataset_of(patients) -> dict[str, str]:
    return {p.subject: p.dataset for p in patients}
