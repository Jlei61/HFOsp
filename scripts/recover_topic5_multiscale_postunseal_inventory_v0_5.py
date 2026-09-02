#!/usr/bin/env python3
"""Resume v0.5 scoring after the recorded condition-inventory incident.

This orchestration is deliberately narrow: it may run only the amended scorer
over already frozen fields/nulls and the already frozen Figure-6 renderer.  It
cannot call a trainer, model loader, field builder, attenuation builder, or
target authorizer.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def archive_stale_failures(out: Path) -> list[str]:
    """Move prior failed-attempt markers aside without deleting evidence."""
    archive = out / "diagnostic_archives" / "postunseal_inventory_first_attempt_2026-08-14"
    archive.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    for name in (
        "POSTTRAINING_PIPELINE_FAILED.json",
        "CLOSEOUT_WATCHER_FAILED.json",
        "POSTUNSEAL_INVENTORY_RECOVERY_FAILED.json",
    ):
        source = out / name
        if not source.exists():
            continue
        destination = archive / name
        if destination.exists():
            destination = archive / f"previous_{name}"
        source.replace(destination)
        moved.append(str(destination))
    note = archive / "ARCHIVE_NOTE.md"
    if moved and not note.exists():
        note.write_text(
            "# Post-unseal inventory first-attempt archive\n\n"
            "The first locked scorer read target values and then stopped before "
            "writing result tables because its inventory validator incorrectly "
            "required an optional matched-local attenuation condition for every "
            "patient. These markers are retained as immutable failure evidence. "
            "The recovery reruns only the amended scorer and frozen figure renderer; "
            "it does not train models or generate fields/nulls.\n"
        )
    return moved


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out_root.resolve()
    authorization_path = out / "TARGET_UNSEAL_AUTHORIZATION.json"
    amendment_path = out / "TARGET_UNSEAL_ENGINEERING_AMENDMENT.json"
    authorization = json.loads(authorization_path.read_text())
    amendment = json.loads(amendment_path.read_text())
    if not (
        authorization.get("authorized") is True
        and amendment.get("status")
        == "POST_UNSEAL_TARGET_INDEPENDENT_INVENTORY_REPAIR"
        and amendment.get("original_authorization_sha256")
        == sha256_file(authorization_path)
        and amendment.get("model_or_field_generation_after_unseal") is False
        and amendment.get("primary_estimand_changed") is False
    ):
        raise RuntimeError("post-unseal inventory recovery contract is invalid")

    driver = Path(__file__).resolve()
    if amendment.get("recovery_driver_sha256") != sha256_file(driver):
        raise RuntimeError("post-unseal recovery driver hash changed")
    archived_failures = archive_stale_failures(out)

    scorer = ROOT / "scripts/score_topic5_multiscale_early_ictal_v0_5.py"
    figure = ROOT / "scripts/paper_figures/plot_topic5_figure6_multiscale_scaffold_v0_5.py"
    if amendment.get("new_scorer_sha256") != sha256_file(scorer):
        raise RuntimeError("amended scorer hash changed")
    snapshot = {
        "contract": "topic5_v0_5_postunseal_inventory_recovery",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "authorization_sha256": sha256_file(authorization_path),
        "amendment_sha256": sha256_file(amendment_path),
        "source_hashes": {
            "driver": sha256_file(driver),
            "scorer": sha256_file(scorer),
            "figure": sha256_file(figure),
        },
        "archived_prior_failure_markers": archived_failures,
        "model_or_field_generation_after_unseal": False,
    }
    write_json(out / "POSTUNSEAL_INVENTORY_RECOVERY_SNAPSHOT.json", snapshot)

    figure_path = (
        ROOT / "results/paper-ready-figure/fig6_multiscale_scaffold_v0_5/figures/"
        "topic5_figure6_multiscale_scaffold_v0_5.png"
    )
    commands = (
        (
            "G_locked_scoring_inventory_repair",
            [sys.executable, str(scorer), "--out-root", str(out)],
        ),
        (
            "H_figure6_inventory_repair",
            [sys.executable, str(figure), "--out-root", str(out)],
        ),
    )
    completed = []
    logs = out / "posttraining_logs"
    logs.mkdir(exist_ok=True)
    for label, command in commands:
        log = logs / f"{label}.log"
        with log.open("w") as stream:
            result = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            write_json(out / "POSTUNSEAL_INVENTORY_RECOVERY_FAILED.json", {
                "status": "FAILED", "step": label,
                "returncode": int(result.returncode), "log": str(log),
                "completed": completed,
                "model_or_field_generation_after_unseal": False,
            })
            raise SystemExit(result.returncode)
        completed.append(label)
    if not figure_path.exists():
        raise FileNotFoundError(figure_path)
    write_json(out / "PIPELINE_COMPLETE.json", {
        "status": "COMPLETE_WITH_RECORDED_TARGET_INDEPENDENT_INVENTORY_REPAIR",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "completed": completed,
        "figure": str(figure_path),
        "figure_sha256": sha256_file(figure_path),
        "target_values_read": True,
        "primary_estimand_changed": False,
        "model_or_field_generation_after_unseal": False,
        "recovery_snapshot_sha256": sha256_file(
            out / "POSTUNSEAL_INVENTORY_RECOVERY_SNAPSHOT.json"
        ),
    })


if __name__ == "__main__":
    main()
