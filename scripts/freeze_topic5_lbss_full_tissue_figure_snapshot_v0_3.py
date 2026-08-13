#!/usr/bin/env python3
"""Freeze a self-contained final-figure/closeout snapshot for LBSS v0.3."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("results/topic5_lbss_full_tissue_rnn_v0_3")
EXTRA_FILES = (
    "scripts/paper_figures/plot_topic5_figure6_lbss_full_tissue_v0_3.py",
    "scripts/plot_topic5_lbss_figure6_v0_2.py",
    "scripts/summarize_topic5_lbss_claims_v0_2.py",
    "scripts/audit_topic5_lbss_full_tissue_closeout_v0_3.py",
    "scripts/run_topic5_lbss_full_tissue_figure_closeout_v0_3.py",
    "scripts/summarize_topic5_lbss_claims_v0_3.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base-snapshot", type=Path, default=None)
    parser.add_argument("--destination-name", default="figure_closeout_snapshot_v3")
    args = parser.parse_args()
    out = args.out_root.resolve()
    base = (
        args.base_snapshot.resolve() if args.base_snapshot is not None
        else out / "postprocess_snapshot_v3"
    )
    destination = out / args.destination_name
    if destination.exists():
        raise FileExistsError(f"figure snapshot already exists: {destination}")
    temporary = destination.with_name(destination.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(base, temporary, copy_function=shutil.copy2)
    temporary.chmod(0o755)
    for path in temporary.rglob("*"):
        path.chmod(0o755 if path.is_dir() else 0o644)
    for relative in EXTRA_FILES:
        source = ROOT / relative
        target = temporary / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.chmod(0o644) if target.exists() else None
        shutil.copy2(source, target)
    files = {
        str(path.relative_to(temporary)): sha256(path)
        for path in sorted(temporary.rglob("*"))
        if path.is_file() and path.name not in {"FIGURE_SNAPSHOT_MANIFEST.json"}
    }
    manifest = {
        "contract": "topic5_lbss_full_tissue_figure_closeout_snapshot_v0_3",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "base_snapshot": str(base),
        "base_snapshot_manifest_sha256": sha256(base / "SNAPSHOT_MANIFEST.json"),
        "files": files,
        "entrypoint": "scripts/run_topic5_lbss_full_tissue_figure_closeout_v0_3.py",
        "immutable": True,
    }
    (temporary / "FIGURE_SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    os.replace(temporary, destination)
    for path in destination.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    destination.chmod(0o555)
    print(json.dumps({
        "snapshot": str(destination), "n_files": len(files),
        "manifest_sha256": sha256(destination / "FIGURE_SNAPSHOT_MANIFEST.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
