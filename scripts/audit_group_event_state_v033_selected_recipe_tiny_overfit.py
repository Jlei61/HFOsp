#!/usr/bin/env python3
"""Re-run the tiny-slice capacity check with a card's selected recipe.

The queue runs T0 before hyperparameter search, so the T0 tiny-overfit recipe
can differ from the recipe ultimately selected by the search.  This audit does
not alter the training card.  It writes a separate, provenance-rich result
that states whether the selected recipe passes the same TRAIN-only check.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.topic5_group_event_state.v033_training_lab.diagnostics import tiny_slice_overfit
from src.topic5_group_event_state.v033_training_lab.objective import TRAINABLE_REGISTRY
from src.topic5_group_event_state.v033_training_lab.paths import atomic_write_json
from src.topic5_group_event_state.v033_training_lab.queue import recipe_from_dict
from src.topic5_group_event_state.v033_training_lab.views import view_for_request


FORMAT = "group_event_state_v0_3_3_selected_recipe_tiny_overfit_review"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()

    card = load_json(args.card)
    if card.get("sealed_partition_opened") or card.get("development_evaluation_read"):
        raise ValueError("refusing a card that opened a forbidden partition")
    request = load_json(args.request)
    compact_request = card.get("request") or {}
    if request.get("request_id") != compact_request.get("request_id"):
        raise ValueError("full request does not match the request recorded in the card")
    if request.get("input_hash") != compact_request.get("input_hash") \
            or request.get("split_hash") != compact_request.get("split_hash"):
        raise ValueError("full request input/split hashes do not match the card")
    subject = str(card.get("subject"))
    objective = str((request.get("scientific_target") or {}).get("objective"))
    if objective not in TRAINABLE_REGISTRY:
        raise ValueError(f"unknown objective: {objective}")

    cfg = recipe_from_dict(card["recipe"])
    view, view_meta = view_for_request(request, release_present=True, scaling=cfg.scaling)
    if view.subject != subject:
        raise ValueError(f"subject mismatch: {view.subject} != {subject}")
    if view.input_hash != card.get("input_hash") or view.split_hash != card.get("split_hash"):
        raise ValueError("input/split hash mismatch")

    args.out_root.mkdir(parents=True, exist_ok=True)
    run_root = args.out_root / "run"
    result = tiny_slice_overfit(
        TRAINABLE_REGISTRY[objective](),
        view,
        cfg,
        int(card.get("representative_seed")),
        device=torch.device(args.device),
        steps=int(args.steps),
        out_dir=run_root,
    )

    original = card.get("tiny_overfit") or {}
    other_conditions = {
        key: bool(value)
        for key, value in (card.get("adequacy_conditions") or {}).items()
        if key != "tiny_overfit"
    }
    payload = {
        "format": FORMAT,
        "created_epoch": time.time(),
        "subject": subject,
        "request_id": request.get("request_id"),
        "request_path": str(args.request.resolve()),
        "request_sha256": sha256(args.request),
        "card_path": str(args.card.resolve()),
        "card_sha256": sha256(args.card),
        "input_hash": view.input_hash,
        "split_hash": view.split_hash,
        "view_meta": view_meta,
        "selected_base_recipe_config_hash": cfg.config_hash(),
        "presearch_base_recipe_config_hash": (card.get("t0") or {}).get("config_hash"),
        "presearch_tiny_overfit_config_hash": original.get("config_hash"),
        "presearch_tiny_overfit": original,
        "selected_recipe_tiny_overfit": result,
        "presearch_recipe_mismatch": (card.get("t0") or {}).get("config_hash") != cfg.config_hash(),
        "other_adequacy_conditions": other_conditions,
        "selected_recipe_full_adequacy_conditions_met": bool(result.get("pass")) and all(other_conditions.values()),
        "selection_metric_is_canonical": card.get("selection_metric_is_canonical"),
        "development_evaluation_read": False,
        "sealed_partition_opened": False,
        "source_git_head": git_head(Path(__file__).resolve().parents[1]),
        "producer_path": str(Path(__file__).resolve()),
        "producer_sha256": sha256(Path(__file__).resolve()),
        "pid": os.getpid(),
    }
    atomic_write_json(args.out_root / "selected_recipe_tiny_overfit_review.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
