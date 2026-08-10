#!/usr/bin/env python3
"""Create the immutable LBSS v0.2 execution snapshot before formal training."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "scripts/train_topic5_lbss_unit_v0_2.py",
    "scripts/launch_topic5_lbss_v0_2.py",
    "scripts/run_topic5_lbss_detectability_v0_2.py",
    "src/__init__.py",
    "src/topic5_lbss_rnn_v0_2.py",
    "src/topic5_wiring_economy_rnn.py",
    "src/topic5_rnn_motif_v0_4.py",
    "docs/superpowers/specs/2026-08-10-topic5-local-backbone-selective-shortcut-rnn-design.md",
    "docs/superpowers/plans/2026-08-10-topic5-local-backbone-selective-shortcut-rnn.md",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=Path("results/topic5_lbss_rnn_v0_2"))
    parser.add_argument("--replace-before-formal", action="store_true")
    args = parser.parse_args()
    out = args.out_root.resolve()
    if (out / "FORMAL_TRAINING_STATUS.json").exists():
        raise RuntimeError("formal training has started; execution snapshot cannot change")
    snapshot = out / "run_snapshot"
    if snapshot.exists():
        if not args.replace_before_formal:
            raise FileExistsError(snapshot)
        shutil.rmtree(snapshot)
    records = []
    for relative in FILES:
        source = ROOT / relative
        if not source.exists():
            raise FileNotFoundError(source)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        records.append({
            "relative_path": relative,
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        })
    manifest = {
        "contract": "topic5_lbss_immutable_execution_snapshot_v0_2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status_porcelain": subprocess.check_output(
            ["git", "status", "--porcelain"], text=True
        ).strip(),
        "files": records,
        "target_values_read": False,
    }
    manifest_path = snapshot / "SNAPSHOT_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Read-only permissions prevent an accidental edit in the active run tree.
    for file in snapshot.rglob("*"):
        if file.is_file():
            file.chmod(0o444)
    for directory in sorted((item for item in snapshot.rglob("*") if item.is_dir()), reverse=True):
        directory.chmod(0o555)
    snapshot.chmod(0o555)

    contract_path = out / "RUN_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    contract["git_commit"] = manifest["git_commit"]
    contract["immutable_execution_snapshot"] = {
        "path": str(snapshot),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256(manifest_path),
    }
    contract["target_access_count"] = 0
    contract["target_values_read"] = False
    contract_path.write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
