#!/usr/bin/env python3
"""Run the selected-recipe tiny-overfit review when each broad-search card appears.

This supervisor never opens development, seizure, or sealed outputs.  It only
re-runs the TRAIN-slice capacity diagnostic with the recipe selected by the
already-completed STATE_SELECTION search.  The report monitor remains the
read-only consumer of these review artefacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


DEFAULT_DATA_ROOT = Path("/data/hfosp_group_event_state_v0_3_3")
SUBJECTS = ("1096", "1125", "1146", "384", "548", "583", "922")
SCRIPT = Path(__file__).resolve().with_name(
    "audit_group_event_state_v033_selected_recipe_tiny_overfit.py"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def _paths(data_root: Path, subject: str) -> dict[str, Path]:
    request_id = f"human-sn-r0-{subject}-trainability-broad-v1"
    output_root = (
        data_root / "supervisor_reports" / "trainability_incremental" /
        "selected_recipe_tiny_overfit" / f"epilepsiae_{subject}"
    )
    return {
        "card": data_root / "agent_b_expansion" / "search" / request_id /
        "card" / "training_card.json",
        "request": data_root / "shared" / "job_requests" / f"science_{request_id}.json",
        "output_root": output_root,
        "review": output_root / "selected_recipe_tiny_overfit_review.json",
        "log": output_root / "supervisor.log",
    }


def _review_current(paths: dict[str, Path]) -> bool:
    review = _read_json(paths["review"])
    if review is None or not paths["card"].is_file():
        return False
    return (
        review.get("card_sha256") == _sha256(paths["card"])
        and review.get("producer_sha256") == _sha256(SCRIPT)
        and not review.get("development_evaluation_read")
        and not review.get("sealed_partition_opened")
    )


def supervise(data_root: Path, poll_seconds: float, once: bool) -> int:
    report_root = data_root / "supervisor_reports" / "trainability_incremental"
    status_path = report_root / "selected_recipe_audit_supervisor_state.json"
    attempts: dict[str, int] = {}
    errors: dict[str, str] = {}

    while True:
        for subject in SUBJECTS:
            paths = _paths(data_root, subject)
            if not paths["card"].is_file() or _review_current(paths):
                continue
            if not paths["request"].is_file():
                errors[subject] = f"missing request: {paths['request']}"
                continue
            attempts[subject] = attempts.get(subject, 0) + 1
            paths["output_root"].mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable,
                str(SCRIPT),
                "--card", str(paths["card"]),
                "--request", str(paths["request"]),
                "--out-root", str(paths["output_root"]),
                "--device", "cpu",
            ]
            environment = dict(os.environ)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                environment[key] = "1"
            with paths["log"].open("ab") as log:
                result = subprocess.run(
                    command,
                    cwd=str(SCRIPT.parents[1]),
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if result.returncode == 0 and _review_current(paths):
                errors.pop(subject, None)
            else:
                errors[subject] = f"audit exit={result.returncode}; see {paths['log']}"

        cards_present = [subject for subject in SUBJECTS if _paths(data_root, subject)["card"].is_file()]
        reviews_complete = [subject for subject in SUBJECTS if _review_current(_paths(data_root, subject))]
        state = {
            "format": "group_event_state_v0_3_3_selected_recipe_audit_supervisor",
            "updated_epoch": time.time(),
            "pid": os.getpid(),
            "subjects": list(SUBJECTS),
            "cards_present": cards_present,
            "reviews_complete": reviews_complete,
            "pending_cards": [subject for subject in SUBJECTS if subject not in cards_present],
            "pending_reviews": [subject for subject in cards_present if subject not in reviews_complete],
            "attempts": attempts,
            "errors": errors,
            "all_complete": len(reviews_complete) == len(SUBJECTS),
            "development_evaluation_read": False,
            "sealed_partition_opened": False,
        }
        _atomic_json(status_path, state)
        if once or state["all_complete"]:
            return 0 if not errors else 1
        time.sleep(max(5.0, poll_seconds))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(supervise(**vars(parse_args())))
