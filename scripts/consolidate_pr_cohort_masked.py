#!/usr/bin/env python3
"""Consolidate PR cohort summary artifact from per-subject JSONs.

After ``scripts/run_pr_parallel.py`` runs N parallel subprocesses, each
writes per-subject JSONs (race-free) and a partial cohort summary (the
last writer wins, earlier ones lost). This script reads ALL per-subject
JSONs and rebuilds the cohort summary by calling the appropriate
``summarize_*`` function.

Usage::

    python scripts/consolidate_pr_cohort_masked.py --pr pr5-gate
    python scripts/consolidate_pr_cohort_masked.py --pr pr5-recruitment
    python scripts/consolidate_pr_cohort_masked.py --pr pr4a
    python scripts/consolidate_pr_cohort_masked.py --pr pr4a-followup

Only operates on ``results/interictal_propagation_masked/``. Mirrors the
schema each ``_run_pr*`` function writes when run sequentially.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "results" / "interictal_propagation_masked"
PER_SUBJECT_DIR = RESULTS_DIR / "per_subject"
PR5A_DIR = PER_SUBJECT_DIR / "pr5a"

YUQUAN_SUBJECTS = [
    "zhangkexuan", "pengzihang", "chengshuai", "huangwanling", "liyouran",
    "songzishuo", "zhangbichen", "zhaochenxi", "zhaojinrui", "zhourongxuan",
    "zhangjiaqi", "chenziyang", "hanyuxuan", "huanghanwen", "litengsheng",
    "xuxinyi", "zhangjinhan", "sunyuanxin", "gaolan", "wangyiyang",
]
EPILEPSIAE_SUBJECTS = [
    "1096", "1084", "958", "922", "590", "1150", "442", "1073",
    "253", "1146", "916", "620", "583", "548", "384", "139",
    "1125", "1077", "818", "635",
]


def _save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  saved -> {path.relative_to(REPO_ROOT)}")


# -------------------- PR-5-A --------------------

def consolidate_pr5_gate() -> None:
    from src.interictal_propagation import summarize_pr5_novel_template_gate

    config_records: Dict[str, List[Dict[str, Any]]] = {"main": [], "auxiliary": []}
    per_subject_archive: Dict[str, Dict[str, Any]] = {}
    skipped: List[str] = []
    if not PR5A_DIR.exists():
        print(f"PR5A dir not found: {PR5A_DIR}")
        return
    files = sorted(PR5A_DIR.glob("*.json"))
    print(f"reading {len(files)} pr5a per_subject JSONs...")
    for fp in files:
        with open(fp) as f:
            payload = json.load(f)
        sid = fp.stem  # "<dataset>_<subject>"
        key = sid.replace("_", "/", 1)
        configs = payload.get("configs") or {}
        if not configs:
            skipped.append(sid)
            continue
        for config_name, gate_result in configs.items():
            if config_name not in config_records:
                config_records[config_name] = []
            rec = dict(gate_result)
            rec["subject_id"] = payload.get("subject")
            rec["dataset"] = payload.get("dataset")
            rec["chosen_k"] = payload.get("chosen_k")
            rec["stable_k"] = payload.get("stable_k")
            rec["n_seizures_loaded"] = payload.get("n_seizures_loaded")
            rec["n_valid_events"] = payload.get("n_valid_events")
            config_records[config_name].append(rec)
        per_subject_archive[key] = payload

    print(f"main records: {len(config_records['main'])}, "
          f"auxiliary records: {len(config_records['auxiliary'])}, "
          f"skipped: {len(skipped)}")

    cohort_summary = summarize_pr5_novel_template_gate(config_records)
    _save(
        {"per_subject": config_records, "cohort": cohort_summary},
        RESULTS_DIR / "pr5a_novel_template_gate.json",
    )

    # Also merge into pr1_cohort_summary.json
    cohort_path = RESULTS_DIR / "pr1_cohort_summary.json"
    cohort_doc: Dict[str, Any] = {}
    if cohort_path.exists():
        with open(cohort_path) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            cohort_doc = loaded
    cohort_doc["novel_template_gate"] = cohort_summary
    _save(cohort_doc, cohort_path)

    print(
        f"overall_pass={cohort_summary.get('overall_pass')}, "
        f"main_pass={cohort_summary.get('main', {}).get('gate_pass')}, "
        f"aux_pass={cohort_summary.get('auxiliary', {}).get('gate_pass')}"
    )


# -------------------- PR-5-B --------------------

def consolidate_pr5_recruitment() -> None:
    from src.interictal_propagation import summarize_pr5_template_recruitment_shift

    pr5b_dir = PER_SUBJECT_DIR / "pr5b"
    if not pr5b_dir.exists():
        print(f"PR5B dir not found: {pr5b_dir}")
        return
    files = sorted(pr5b_dir.glob("*.json"))
    print(f"reading {len(files)} pr5b per_subject JSONs...")
    config_records: Dict[str, List[Dict[str, Any]]] = {"main": [], "auxiliary": []}
    for fp in files:
        with open(fp) as f:
            payload = json.load(f)
        configs = payload.get("configs") or {}
        for config_name, shift in configs.items():
            if config_name not in config_records:
                config_records[config_name] = []
            rec = dict(shift)
            rec["subject_id"] = payload.get("subject")
            rec["dataset"] = payload.get("dataset")
            config_records[config_name].append(rec)

    cohort_summary = summarize_pr5_template_recruitment_shift(config_records)
    _save(
        {"per_subject": config_records, "cohort": cohort_summary},
        RESULTS_DIR / "pr5b_recruitment_shift.json",
    )

    cohort_path = RESULTS_DIR / "pr1_cohort_summary.json"
    cohort_doc: Dict[str, Any] = {}
    if cohort_path.exists():
        with open(cohort_path) as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            cohort_doc = loaded
    cohort_doc["template_recruitment_shift"] = cohort_summary
    _save(cohort_doc, cohort_path)


# -------------------- PR-4A --------------------

def consolidate_pr4a() -> None:
    """Read per_subject `temporal_dynamics` fields and rebuild
    pr4a_temporal_dynamics.json."""
    from src.interictal_propagation import summarize_propagation_cohort
    temporal_results: Dict[str, Dict[str, Any]] = {}
    files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    n = 0
    for fp in files:
        if fp.parent != PER_SUBJECT_DIR:
            continue
        with open(fp) as f:
            d = json.load(f)
        td = d.get("temporal_dynamics")
        if td is None:
            continue
        key = f"{d.get('dataset')}/{d.get('subject')}"
        temporal_results[key] = td
        n += 1
    print(f"PR-4A: collected {n} subjects with temporal_dynamics")
    _save(temporal_results, RESULTS_DIR / "pr4a_temporal_dynamics.json")


# -------------------- PR-4D (followup) --------------------

def consolidate_pr4a_followup() -> None:
    followup_results: Dict[str, Dict[str, Any]] = {}
    files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    n = 0
    for fp in files:
        if fp.parent != PER_SUBJECT_DIR:
            continue
        with open(fp) as f:
            d = json.load(f)
        fr = d.get("template_mix_dynamics")
        if fr is None:
            continue
        key = f"{d.get('dataset')}/{d.get('subject')}"
        followup_results[key] = fr
        n += 1
    print(f"PR-4D: collected {n} subjects with template_mix_dynamics")
    _save(followup_results, RESULTS_DIR / "pr4a_followup_template_mix_dynamics.json")


# -------------------- PR-4C --------------------

def consolidate_pr4c() -> None:
    seizure_results: Dict[str, Dict[str, Any]] = {}
    files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    n = 0
    for fp in files:
        if fp.parent != PER_SUBJECT_DIR:
            continue
        with open(fp) as f:
            d = json.load(f)
        sp = d.get("seizure_proximity_coupling")
        if sp is None:
            continue
        key = f"{d.get('dataset')}/{d.get('subject')}"
        seizure_results[key] = sp
        n += 1
    print(f"PR-4C: collected {n} subjects with seizure_proximity_coupling")
    _save(seizure_results, RESULTS_DIR / "pr4c_seizure_proximity.json")


# -------------------- main --------------------

DISPATCH = {
    "pr4a": consolidate_pr4a,
    "pr4a-followup": consolidate_pr4a_followup,
    "pr4c": consolidate_pr4c,
    "pr5-gate": consolidate_pr5_gate,
    "pr5-recruitment": consolidate_pr5_recruitment,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pr", required=True, choices=sorted(DISPATCH))
    args = ap.parse_args()
    fn = DISPATCH[args.pr]
    print(f"Consolidating --{args.pr} on {RESULTS_DIR.relative_to(REPO_ROOT)}/")
    fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
