#!/usr/bin/env python3
"""Freeze the executable full-tissue LBSS v0.3 source tree before formal runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path("results/topic5_lbss_full_tissue_rnn_v0_3")
FILES = (
    "scripts/launch_topic5_lbss_v0_2.py",
    "scripts/train_topic5_lbss_unit_v0_2.py",
    "scripts/build_topic5_lbss_full_tissue_cache_v0_3.py",
    "scripts/freeze_topic5_lbss_full_tissue_snapshot_v0_3.py",
    "src/topic5_lbss_rnn_v0_2.py",
    "src/topic5_wiring_economy_rnn.py",
    "src/topic5_rnn_motif_v0_4.py",
    "src/topic5_virtual_seeg_operator.py",
    "docs/superpowers/specs/2026-08-12-topic5-lbss-full-tissue-rnn-v0-3-design.md",
    "docs/superpowers/plans/2026-08-12-topic5-lbss-full-tissue-rnn-v0-3.md",
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
    args = parser.parse_args()
    out = args.out_root.resolve()
    smoke = json.loads((out / "SMOKE_TRAINING_COMPLETE.json").read_text())
    if smoke.get("unresolved") != 0 or smoke.get("complete") != smoke.get("scheduled"):
        raise RuntimeError("formal snapshot requires a complete, clean smoke stage")
    snapshot = out / "run_snapshot"
    if snapshot.exists():
        raise RuntimeError(f"immutable snapshot already exists: {snapshot}")
    rows = []
    for relative in FILES:
        source = ROOT / relative
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        rows.append({
            "relative_path": relative,
            "sha256": sha256(destination),
            "bytes": destination.stat().st_size,
        })
    manifest = {
        "contract": "topic5_lbss_full_tissue_immutable_snapshot_v0_3",
        "source_root": str(ROOT),
        "target_values_read": False,
        "input_manifest_sha256": sha256(out / "INPUT_CACHE_MANIFEST.json"),
        "run_contract_sha256": sha256(out / "RUN_CONTRACT.json"),
        "files": rows,
    }
    path = out / "RUN_SNAPSHOT_MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"froze {len(rows)} files under {snapshot}")


if __name__ == "__main__":
    main()
