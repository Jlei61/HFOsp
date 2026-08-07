#!/usr/bin/env python3
"""Verify the development lock, then launch the 28-patient extension exactly once."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RELEASE = (
    ROOT
    / "results/topic5_stable_repertoire_event_history/v2_4/development_acceptance/LOCKED_EXTENSION_RELEASE.json"
)
SPEC = ROOT / "docs/superpowers/specs/2026-08-02-topic5-stable-repertoire-event-history-v2_4.md"
MODULE = ROOT / "src/topic5_stable_repertoire_event_history_v2_4.py"
RUNNER = ROOT / "scripts/run_topic5_stable_repertoire_event_history_v2_4.py"
CONFIGS = {
    20: ROOT / "config/topic5_stable_repertoire_event_history_v2_4.yaml",
    40: ROOT / "config/topic5_stable_repertoire_event_history_v2_4_h40.yaml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, choices=(20, 40), required=True)
    args = parser.parse_args()
    release = json.load(RELEASE.open())
    if release["status"] != "START_LOCKED_28_PATIENT_EXTENSION":
        raise RuntimeError("development gate did not release the extension")
    frozen = release["frozen_hashes"]
    current = {
        "spec_sha256": sha256(SPEC),
        "module_sha256": sha256(MODULE),
        "runner_sha256": sha256(RUNNER),
        f"config_h{args.horizon}_sha256": sha256(CONFIGS[args.horizon]),
    }
    for key, value in current.items():
        if frozen[key] != value:
            raise RuntimeError(f"frozen hash changed: {key}")
    command = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(CONFIGS[args.horizon]),
        "--cohort",
        "extension",
    ]
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()

