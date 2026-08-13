#!/usr/bin/env python3
"""Freeze the zero-H engagement analyzer and its deferred watcher."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "scripts/analyse_topic5_lbss_latent_engagement_v0_3.py",
    "scripts/run_topic5_lbss_latent_engagement_watcher_v0_3.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--snapshot-name", default="latent_engagement_snapshot_v4")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    out = args.out_root.resolve()
    destination = out / args.snapshot_name
    if destination.exists():
        raise FileExistsError(destination)
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if status and not args.allow_dirty:
        raise RuntimeError("snapshot requires clean worktree or --allow-dirty")
    destination.mkdir(parents=True)
    hashes = {}
    for relative in FILES:
        source = ROOT / relative
        target = destination / Path(relative).name
        shutil.copy2(source, target)
        hashes[target.name] = sha256(target)
    manifest = destination / "SNAPSHOT_MANIFEST.json"
    manifest.write_text(json.dumps({
        "contract": "topic5_lbss_zero_h_latent_engagement_snapshot_v0_4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "dirty_worktree_explicitly_allowed": bool(status and args.allow_dirty),
        "files": hashes,
        "scope": "primary L3 only; follows final primary artifact pointer",
        "target_values_read": False,
        "immutable": True,
    }, indent=2) + "\n")
    for path in destination.iterdir():
        path.chmod(0o444)
    destination.chmod(0o555)
    print(json.dumps({
        "snapshot": str(destination),
        "manifest_sha256": sha256(manifest),
    }))


if __name__ == "__main__":
    main()
