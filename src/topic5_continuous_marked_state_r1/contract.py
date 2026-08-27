"""Fail-closed paths and provenance for continuous marked-state R1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


REVISION = "continuous_marked_state_r1_exact_event_likelihood_v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = REPO_ROOT / "results/epi_prssm/continuous_marked_state/r1"
R0_RESULT_ROOT = REPO_ROOT / "results/epi_prssm/continuous_marked_state/r0_1"
UPSTREAM_ROOT = REPO_ROOT / "results/epi_prssm/v0_1"
RAW_STATE_ROOT = REPO_ROOT / "results/epi_prssm/raw_seeg_state/r0_1"
SPLIT_MANIFEST = RAW_STATE_ROOT / "data/split_manifest.json"

PILOT_SUBJECTS = (
    "epilepsiae_620",
    "epilepsiae_958",
    "epilepsiae_139",
    "yuquan_huanghanwen",
    "yuquan_zhangjiaqi",
    "yuquan_hanyuxuan",
)

BRIDGE_E1_SUBJECTS = (
    "epilepsiae_620",
    "epilepsiae_958",
    "yuquan_huanghanwen",
)

# Development-only subjects selected before fitting from the full-cohort
# contiguous-support audit.  They extend the original six-subject R1 pilot;
# they do not redefine that pilot or create a cohort inference set.
LONG_H3_DISCOVERY_SUBJECTS = (
    "yuquan_chengshuai",      # 20,000-event and physical 6 h support
    "yuquan_pengzihang",      # 5,000-event support over a longer Yuquan segment
    "epilepsiae_922",         # 5,000-event support in dense 2 h blocks
    "yuquan_chenziyang",      # 5,000-event and physical 6 h support
    "yuquan_hanyuxuan",       # 3,000-event / long-clock contrast
)

# R1.5 extension is fixed from the corrected recorded-segment support audit,
# before any R1.5 fit is run.  Only the first three are exact-model unseen;
# the other three were already inspected in older long-record triage and are
# retained as calibration cases, never independent replication subjects.
R1_5_NOVEL_SUBJECTS = (
    "epilepsiae_1096",
    "epilepsiae_384",
    "yuquan_zhangkexuan",
)
R1_5_LONG_CARRYOVER_SUBJECTS = (
    "yuquan_chengshuai",
    "yuquan_chenziyang",
    "yuquan_zhangjiaqi",
)
R1_5_EXTENSION_SUBJECTS = (
    *R1_5_NOVEL_SUBJECTS,
    *R1_5_LONG_CARRYOVER_SUBJECTS,
)

# R1.7A is a prospective development replication set.  These subjects were
# selected only after excluding every subject used for architecture,
# optimiser, epoch-budget, threshold, or long-scale discovery decisions.  The
# selection rule and the resulting frozen list are machine-recorded by
# ``build_r1_7a_inventory.py``; keeping the tuple here makes all CLI entry
# points fail closed before that manifest is read.
R1_7A_SUBJECTS = (
    "epilepsiae_1073",
    "epilepsiae_1077",
    "epilepsiae_1125",
    "epilepsiae_1146",
    "epilepsiae_253",
    "yuquan_liyouran",
    "yuquan_xuxinyi",
    "yuquan_zhangbichen",
    "yuquan_zhaochenxi",
    "yuquan_wangyiyang",
)

EXTENDED_DEVELOPMENT_SUBJECTS = tuple(dict.fromkeys(
    (*PILOT_SUBJECTS, *LONG_H3_DISCOVERY_SUBJECTS, *R1_5_EXTENSION_SUBJECTS,
     *R1_7A_SUBJECTS)
))


def load_split(subject: str) -> tuple[float, float]:
    payload = json.loads(SPLIT_MANIFEST.read_text())
    row = payload["subjects"][subject]
    return float(row["train_end_epoch"]), float(row["dev_end_epoch"])


def assert_development_times(subject: str, values, split: str) -> None:
    import numpy as np

    times = np.asarray(values, dtype=np.float64)
    if times.size == 0 or not np.isfinite(times).all():
        raise ValueError(f"{subject}: {split} times are empty or non-finite")
    train_end, dev_end = load_split(subject)
    if split == "train":
        valid = times < train_end
    elif split == "validation":
        valid = (times >= train_end) & (times < dev_end)
    else:
        raise ValueError(f"unknown development split {split!r}")
    if not bool(valid.all()):
        bad = times[~valid]
        raise ValueError(
            f"SEALED/SPLIT VIOLATION {subject} {split}: "
            f"{bad.min():.6f}..{bad.max():.6f}"
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, target)
