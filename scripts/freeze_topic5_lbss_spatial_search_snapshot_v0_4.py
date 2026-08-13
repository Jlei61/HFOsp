#!/usr/bin/env python3
"""Freeze the target-free LBSS spatial-search runner and training closure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from freeze_topic5_lbss_postprocess_snapshot_v0_2 import dependency_closure, sha256  # noqa: E402


ENTRY_FILES = (
    "scripts/run_topic5_lbss_spatial_search_v0_4.py",
    "scripts/run_topic5_lbss_spatial_decision_watcher_v0_4.py",
    "scripts/prepare_topic5_lbss_selected_primary_root_v0_4.py",
    "scripts/train_topic5_lbss_unit_v0_2.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=Path("results/topic5_lbss_full_tissue_rnn_v0_3"))
    parser.add_argument("--snapshot-name", default="spatial_search_snapshot_v0_4")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    out = args.out_root.resolve()
    forbidden = (
        "TARGET_UNSEAL_AUTHORIZATION.json",
        "TARGET_ACCESS_AUDIT.json",
        "EARLY_ICTAL_SCORING_COMPLETE.json",
    )
    if any((out / name).exists() for name in forbidden):
        raise RuntimeError("spatial-search source cannot be frozen after target access")
    snapshot = out / args.snapshot_name
    if snapshot.exists():
        raise FileExistsError(snapshot)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if status and not args.allow_dirty:
        raise RuntimeError("snapshot requires a clean worktree or --allow-dirty")
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    files = dependency_closure(ENTRY_FILES)
    records: dict[str, str] = {}
    for relative in files:
        source = ROOT / relative
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records[str(relative)] = sha256(destination)
    manifest = snapshot / "SNAPSHOT_MANIFEST.json"
    manifest.write_text(json.dumps({
        "contract": "topic5_lbss_target_free_spatial_search_snapshot_v0_4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_worktree_explicitly_allowed": bool(args.allow_dirty and status),
        "git_status_porcelain": status.splitlines(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "files": records,
        "entry_files": list(ENTRY_FILES),
        "target_values_read": False,
        "immutable": True,
    }, indent=2) + "\n")
    for path in snapshot.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((path for path in snapshot.rglob("*") if path.is_dir()), reverse=True):
        path.chmod(0o555)
    snapshot.chmod(0o555)
    print(json.dumps({
        "snapshot": str(snapshot),
        "n_files": len(files),
        "manifest_sha256": sha256(manifest),
    }))


if __name__ == "__main__":
    main()
