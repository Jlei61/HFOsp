#!/usr/bin/env python3
"""Run a v0.5 command inside a mount namespace with ictal targets hidden.

This is an engineering embargo, not an assertion embedded in an output JSON.
The outer process creates a private user/mount namespace.  The inner process
bind-mounts empty directories over every canonical target-value root and over
older target-derived result roots before it launches the requested command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
CANONICAL_ROOT = ROOT.parents[1] if (ROOT.parents[1] / "results").exists() else ROOT
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
SEALED_ENV = "TOPIC5_V0_5_TARGET_SEALED"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protected_roots() -> tuple[Path, ...]:
    canonical_results = CANONICAL_ROOT / "results"
    local_results = ROOT / "results"
    return (
        canonical_results / "topic5_ictal_recruitment/t0_feature_cache_bb150_1_150",
        canonical_results / "topic5_ictal_recruitment/v2_band_scan/cache",
        canonical_results / "topic5_ictal_recruitment/tspectral_field_concordance",
        local_results / "topic5_lbss_full_tissue_rnn_v0_3/early_ictal",
        local_results / "topic5_lbss_rnn_v0_2/early_ictal",
        local_results / "topic5_rnn_full_cohort_field_transfer_v0_1",
        local_results / "paper-ready-figure/fig6_lbss_field_transfer",
        local_results / "paper-ready-figure/fig6_rnn_full_cohort_field_transfer",
    )


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def enter_outer(args: argparse.Namespace, command: list[str]) -> int:
    invocation = [
        "unshare", "-Urnm", "--", sys.executable, str(Path(__file__).resolve()),
        "--inside", "--out-root", str(args.out_root.resolve()), "--", *command,
    ]
    return subprocess.run(invocation, check=False).returncode


def bind_empty_targets() -> list[dict]:
    subprocess.run(["mount", "--make-rprivate", "/"], check=True)
    scratch = Path(tempfile.mkdtemp(prefix="topic5_v05_target_embargo_"))
    empty_directory = scratch / "empty_directory"
    empty_directory.mkdir()
    records = []
    for target in protected_roots():
        target = target.resolve()
        if not target.exists():
            records.append({"path": str(target), "status": "ABSENT"})
            continue
        if not target.is_dir():
            raise RuntimeError(f"protected target root is not a directory: {target}")
        subprocess.run(
            ["mount", "--bind", str(empty_directory), str(target)], check=True,
        )
        visible = list(target.iterdir())
        if visible:
            raise RuntimeError(f"target embargo failed for {target}: {visible[:3]}")
        records.append({"path": str(target), "status": "BIND_HIDDEN_EMPTY"})
    return records


def enter_inner(args: argparse.Namespace, command: list[str]) -> int:
    records = bind_empty_targets()
    manifest = {
        "contract": "topic5_v0_5_physical_target_embargo",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "wrapper": str(Path(__file__).resolve()),
        "wrapper_sha256": sha256(Path(__file__).resolve()),
        "command": command,
        "protected_roots": records,
        "routing_metadata_allowed": str(
            (args.out_root.resolve() / "EARLY_ICTAL_ROUTING_METADATA.csv")
        ),
        "target_values_read": False,
    }
    write_json(args.out_root.resolve() / "TARGET_PHYSICAL_EMBARGO_ACTIVE.json", manifest)
    environment = os.environ.copy()
    environment[SEALED_ENV] = "1"
    return subprocess.run(command, env=environment, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inside", action="store_true")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("a command is required after --")
    if args.inside:
        if os.environ.get(SEALED_ENV):
            raise RuntimeError("nested target embargo invocation is not allowed")
        return enter_inner(args, command)
    return enter_outer(args, command)


if __name__ == "__main__":
    raise SystemExit(main())
