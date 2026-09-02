#!/usr/bin/env python3
"""Phase A: freeze the worktree scope and the parent artifact manifest.

Writes WORKTREE_STATUS_BEFORE.txt, WORKTREE_SCOPE.json,
PARENT_ARTIFACT_MANIFEST.json and EXECUTION_AUTHORIZATION.json into the
Topic 5.2D v0.2 result root.  Nothing outside that root is modified.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULT_ROOT = ROOT / "results" / "topic5_capacity_constrained_history_motif_v0_2"
PARENT_ROOT = ROOT / "results" / "topic5_dynamical_motif_rnn_v0_1"
FRAME_CACHE = PARENT_ROOT / "frame_cache" / "GEOMETRY_ONLY_PCA2"
ECOG_ROOT = ROOT / "results" / "topic5_ecog_physical_neighborhood_rnn_v0_1"
FIGURE6_DIR = ROOT / "results" / "paper-ready-figure" / "fig6_interictal_crossstate_response_r5_candidate"

SPEC = ROOT / "docs/superpowers/specs/2026-08-17-topic5-capacity-constrained-structural-identifiability-v0-2-design.md"
PLAN = ROOT / "docs/superpowers/plans/2026-08-17-topic5-capacity-constrained-structural-identifiability-v0-2.md"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True).stdout


def main() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()

    status = git("status", "--porcelain=v1")
    (RESULT_ROOT / "WORKTREE_STATUS_BEFORE.txt").write_text(
        f"# captured {now}\n"
        f"# branch: {git('rev-parse', '--abbrev-ref', 'HEAD').strip()}\n"
        f"# head:   {git('rev-parse', 'HEAD').strip()}\n\n" + status
    )

    tracked_modified = [line[3:] for line in status.splitlines() if line[:2].strip() in {"M", "D", "A", "R"}]
    untracked = [line[3:] for line in status.splitlines() if line.startswith("??")]

    processes = subprocess.run(
        ["ps", "-eo", "pid,ppid,etimes,rss,args"], capture_output=True, text=True
    ).stdout.splitlines()
    stale = [
        line for line in processes
        if "multiprocessing-fork" in line or "topic5" in line
    ]

    scope = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_worktree_scope",
        "captured_utc": now,
        "worktree": str(ROOT),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD").strip(),
        "head_commit": git("rev-parse", "HEAD").strip(),
        "writable_roots": [
            "results/topic5_capacity_constrained_history_motif_v0_2/",
            "results/paper-ready-figure/supp_fig6_strict_history_motif_v0_2/",
            "src/topic5_strict_history_*_v0_2.py",
            "src/topic5_structural_identifiability_v0_2.py",
            "src/topic5_ecog_graph_capacity_v0_2.py",
            "scripts/*topic5_capacity*_v0_2.py",
            "scripts/*topic5_strict_history*_v0_2.py",
            "tests/test_topic5_capacity_constrained_history_motif_v0_2.py",
            "docs/archive/topic5/capacity_constrained_history_motif_v0_2_*.md",
        ],
        "read_only_roots": [
            str(FRAME_CACHE.relative_to(ROOT)),
            str(ECOG_ROOT.relative_to(ROOT)),
            str(FIGURE6_DIR.relative_to(ROOT)),
        ],
        "forbidden_actions": [
            "git commit / git push",
            "modifying or deleting pre-existing uncommitted changes",
            "overwriting results/paper-ready-figure/fig6_interictal_crossstate_response_r5_candidate/",
            "deleting any v0.1 / v0.2 parent result",
        ],
        "preexisting_uncommitted_tracked_files": tracked_modified,
        "preexisting_untracked_paths_count": len(untracked),
        "preexisting_background_processes": stale,
        "spec_sha256": sha256_file(SPEC),
        "plan_sha256": sha256_file(PLAN),
        "host": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "python": sys.version.split()[0],
        },
    }
    (RESULT_ROOT / "WORKTREE_SCOPE.json").write_text(json.dumps(scope, indent=2) + "\n")

    # --- parent artifact manifest -------------------------------------------
    patients = sorted(p.name for p in FRAME_CACHE.iterdir() if p.is_dir())
    per_patient = {}
    for name in patients:
        directory = FRAME_CACHE / name
        provenance = json.loads((directory / "provenance.json").read_text())
        events = np.load(directory / "events.npz", allow_pickle=True)
        plane = np.load(directory / "plane.npz", allow_pickle=False)
        ranks = np.asarray(events["ranks"])
        split = np.asarray(events["split"])
        participation = ranks >= 0
        per_patient[name] = {
            "subject": provenance["subject"],
            "n_contacts": int(ranks.shape[1]),
            "n_events": int(ranks.shape[0]),
            "n_shafts": len(sorted(set(str(v) for v in events["shafts"]))),
            "geometry_class": provenance["geometry_class"],
            "ratio_second_to_first": float(provenance["ratio_second_to_first"]),
            "split_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(split, return_counts=True))},
            "participants_min": int(participation.sum(1).min()),
            "participants_median": float(np.median(participation.sum(1))),
            "participants_max": int(participation.sum(1).max()),
            "tied_rank_events": int(
                (np.where(participation, ranks, -1).max(1) + 1 != participation.sum(1)).sum()
            ),
            "events_sha256": sha256_file(directory / "events.npz"),
            "plane_sha256": sha256_file(directory / "plane.npz"),
            "parent_events_sha256": provenance["events_sha256"],
            "parent_split_sha256": provenance["split_audit"]["split_sha256"],
            "parent_event_split_sha256": provenance["split_audit"]["parent_event_split_sha256"],
            "model_unseen_equals_parent_heldout": bool(
                provenance["split_audit"]["model_unseen_equals_parent_heldout"]
            ),
            "coords_3d_available": bool("coords_3d_mm" in plane.files),
            "sigma_mm": float(provenance["sigma_mm"]),
            "r_local_mm": float(provenance["r_local_mm"]),
        }

    ecog = {}
    for subject in ("958", "1084"):
        directory = ECOG_ROOT / "cache" / subject
        provenance = json.loads((directory / "provenance.json").read_text())
        ecog[subject] = {
            "n_contacts": int(provenance["n_contacts"]),
            "n_events": int(provenance["n_events"]),
            "split_counts": provenance["split_counts"],
            "tied_rank_set_fraction": float(provenance["tied_rank_set_fraction"]),
            "rank_set_count_median": float(provenance["rank_set_count_median"]),
            "participant_count_median": float(provenance["participant_count_median"]),
            "events_sha256": sha256_file(directory / "events.npz"),
            "parent_events_sha256": provenance["events_sha256"],
        }

    figure6 = {}
    if FIGURE6_DIR.exists():
        for path in sorted((FIGURE6_DIR / "figures").rglob("*")):
            if path.is_file():
                figure6[str(path.relative_to(FIGURE6_DIR))] = sha256_file(path)

    manifest = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_parent_manifest",
        "captured_utc": now,
        "seeg_frame": "GEOMETRY_ONLY_PCA2",
        "seeg_frame_root": str(FRAME_CACHE.relative_to(ROOT)),
        "seeg_n_patients": len(patients),
        "seeg_patients": per_patient,
        "ecog_root": str((ECOG_ROOT / "cache").relative_to(ROOT)),
        "ecog_subjects": ecog,
        "protected_figure6_dir": str(FIGURE6_DIR.relative_to(ROOT)),
        "protected_figure6_sha256": figure6,
    }
    (RESULT_ROOT / "PARENT_ARTIFACT_MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")

    authorization = {
        "contract": "topic5_capacity_constrained_history_motif_v0_2_execution_authorization",
        "authorized_utc": now,
        "authorization_source": "user message 2026-08-17 (formal execution authorization, not a review request)",
        "spec": str(SPEC.relative_to(ROOT)),
        "plan": str(PLAN.relative_to(ROOT)),
        "spec_sha256": sha256_file(SPEC),
        "plan_sha256": sha256_file(PLAN),
        "scope": "implement + execute the full frozen matrix, aggregate, supplementary figure, plain + technical reports",
        "frozen_science": "spec + plan scientific content is frozen; only status headers were updated",
        "not_gated_on_results": [
            "aligned structure shows no advantage",
            "free model cannot learn",
            "direct positive but autonomous negative",
            "orderless bag equals ordered",
            "bypass interaction near zero or reversed",
            "cohort median near zero",
            "E958 and E1084 disagree",
            "synthetic recovery weak at low coverage",
        ],
        "only_blocking_conditions": [
            "event/contact/split hash mismatch vs parent",
            "future leakage into basis or training",
            "numerical error (NaN/Inf, dimension mismatch)",
            "checkpoint/hash corruption",
            "worker output unrecoverable from manifest",
        ],
    }
    (RESULT_ROOT / "EXECUTION_AUTHORIZATION.json").write_text(json.dumps(authorization, indent=2) + "\n")

    print(f"wrote Phase A artifacts to {RESULT_ROOT}")
    print(f"  seeg patients: {len(patients)}   ecog subjects: {len(ecog)}")
    print(f"  protected figure6 files: {len(figure6)}")
    print(f"  preexisting uncommitted tracked files: {len(tracked_modified)}")
    print(f"  preexisting background processes: {len(stale)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
