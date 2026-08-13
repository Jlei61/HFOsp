#!/usr/bin/env python3
"""Freeze the reviewed full-tissue LBSS postprocess source and dependencies."""
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
    "scripts/analyse_topic5_lbss_full_tissue_interictal_v0_3.py",
    "scripts/build_topic5_lbss_fields_v0_2.py",
    "scripts/analyse_topic5_lbss_pathways_v0_2.py",
    "scripts/run_topic5_lbss_attenuation_v0_2.py",
    "scripts/audit_topic5_lbss_full_tissue_early_ictal_metadata_v0_3.py",
    "scripts/prepare_topic5_lbss_full_tissue_target_unseal_v0_3.py",
    "scripts/score_topic5_lbss_full_tissue_early_ictal_v0_3.py",
    "scripts/run_topic5_lbss_full_tissue_postprocess_v0_3.py",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_full_tissue_rnn_v0_3"))
    parser.add_argument("--snapshot-name", default="postprocess_snapshot_v3")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    out = args.out_root.resolve()
    if (out / "TARGET_UNSEAL_AUTHORIZATION.json").exists() or (out / "TARGET_ACCESS_AUDIT.json").exists():
        raise RuntimeError("postprocess source cannot be changed after target unseal")
    snapshot = out / args.snapshot_name
    if snapshot.exists():
        raise FileExistsError(snapshot)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
    if status and not args.allow_dirty:
        raise RuntimeError("postprocess snapshot requires a clean worktree or --allow-dirty")
    diff = subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    files = dependency_closure(ENTRY_FILES)
    records = {}
    for relative in files:
        source = ROOT / relative
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records[str(relative)] = sha256(destination)
    payload = {
        "contract": "topic5_lbss_full_tissue_postprocess_snapshot_v0_3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dirty_worktree_explicitly_allowed": bool(args.allow_dirty and status),
        "git_status_porcelain": status.splitlines(),
        "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "files": records,
        "entry_files": list(ENTRY_FILES),
        "self_contained_local_import_closure": True,
        "target_values_read": False,
        "immutable": True,
    }
    manifest = snapshot / "SNAPSHOT_MANIFEST.json"
    manifest.write_text(json.dumps(payload, indent=2) + "\n")
    for path in snapshot.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
    for path in sorted((value for value in snapshot.rglob("*") if value.is_dir()), reverse=True):
        path.chmod(0o555)
    snapshot.chmod(0o555)
    print(json.dumps({
        "snapshot": str(snapshot),
        "n_files": len(files),
        "manifest_sha256": sha256(manifest),
    }))


if __name__ == "__main__":
    main()
