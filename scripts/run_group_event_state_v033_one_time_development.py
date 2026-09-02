#!/usr/bin/env python3
"""Freeze and execute the one-time v0.3.3 tuning-patient development score."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.topic5_group_event_state.v033_training_lab.development import (
    RELEASE_FORMAT, RELEASE_STATUS, evaluate_released_request,
)
from src.topic5_group_event_state.v033_training_lab.paths import (
    AGENT_B_ROOT, SHARED_ROOT, atomic_write_json, current_commit, file_hash,
)


DEFAULT_RELEASE = SHARED_ROOT / "evaluator_contract" / "ONE_TIME_DEVELOPMENT_EVALUATION_RELEASE.json"


def prepare_release(request_ids: list[str], *, agent_root: Path, path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"development release already exists: {path}")
    requests: dict[str, dict] = {}
    for request_id in request_ids:
        request_path = agent_root / "requests" / request_id / "request.json"
        card_path = agent_root / "search" / request_id / "card_multiseed_v2" / "training_card.json"
        request = json.loads(request_path.read_text())
        card = json.loads(card_path.read_text())
        multi = dict((card.get("diagnostics") or {}).get("multi_seed_diagnostics") or {})
        rows = []
        for row in multi.get("per_seed") or []:
            rows.append({
                "seed": int(row["seed"]),
                "learned_checkpoint_path": row["learned_checkpoint"],
                "learned_checkpoint_sha256": row["learned_checkpoint_sha256"],
                "random_reservoir_checkpoint_path": row["random_checkpoint"],
                "random_reservoir_checkpoint_sha256": row["random_checkpoint_sha256"],
            })
        if len(rows) < 2:
            raise ValueError(f"{request_id}: corrected card lacks multi-seed checkpoint identities")
        requests[request_id] = {
            "subject": request["input_view"]["subject"],
            "request_path": str(request_path), "request_sha256": file_hash(request_path),
            "corrected_card_path": str(card_path), "corrected_card_sha256": file_hash(card_path),
            "evidence_label_at_freeze": card["evidence_label"],
            "checkpoints": rows,
        }
    payload = {
        "format": RELEASE_FORMAT,
        "status": RELEASE_STATUS,
        "issued_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "user_approved_goal": True,
        "development_only": True,
        "sealed": False,
        "selection_feedback_forbidden": True,
        "retraining_after_open_forbidden": True,
        "score_all_named_seeds_before_reduction": True,
        "seed_merge_rule": "median per anchor across seeds, then patient time block",
        "allowed_operations": ["score frozen checkpoints on DEVELOPMENT_EVALUATION once"],
        "forbidden_operations": [
            "hyperparameter selection", "checkpoint selection", "retraining", "replication patients",
            "sealed or formal partition", "human H3",
        ],
        "trainer_evaluation_code_commit": current_commit(),
        "requests": requests,
    }
    atomic_write_json(path, payload)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("prepare-release")
    freeze.add_argument("request_id", nargs="+")
    freeze.add_argument("--agent-root", type=Path, default=AGENT_B_ROOT)
    freeze.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    score = sub.add_parser("evaluate")
    score.add_argument("request_id", nargs="+")
    score.add_argument("--agent-root", type=Path, default=AGENT_B_ROOT)
    score.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    score.add_argument("--out-root", type=Path, default=AGENT_B_ROOT / "development_evaluation")
    score.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "prepare-release":
        path = prepare_release(args.request_id, agent_root=args.agent_root, path=args.release)
        print(json.dumps({"release": str(path), "sha256": file_hash(path)}, ensure_ascii=False))
        return
    for request_id in args.request_id:
        report = evaluate_released_request(
            request_id=request_id,
            request_path=args.agent_root / "requests" / request_id / "request.json",
            card_path=args.agent_root / "search" / request_id / "card_multiseed_v2" / "training_card.json",
            release_path=args.release,
            out_dir=args.out_root / request_id,
            device=torch.device(args.device),
        )
        print(json.dumps({"request_id": request_id, "result": str(args.out_root / request_id / "result.json"),
                          "score": report["score"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
