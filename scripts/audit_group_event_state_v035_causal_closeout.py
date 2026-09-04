#!/usr/bin/env python3
"""Machine audit for the post-review v0.3.5 causal re-run closeout."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path


ROOT = Path("/data/hfosp_group_event_state_v0_3_5_causal")
EXPECTED = {"W1": 36, "W1b": 36, "W2": 21, "W3": 21, "W456": 21}
REPORTS = (
    "dynamic_baseline.json", "h1_h2a.json", "h2b.json", "h3.json",
    "scope_summary.json", "state_training.csv",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    failures: list[str] = []
    done_path = ROOT / "causal_supervisor/queue_done.json"
    if not done_path.exists():
        failures.append("queue_done is missing")
        done = {}
    else:
        done = load(done_path)
    counts = Counter(row.get("kind") for row in done.get("complete", ()))
    if dict(counts) != EXPECTED:
        failures.append(f"completed work-package counts differ: {dict(counts)}")
    if done.get("failed"):
        failures.append(f"queue contains {len(done['failed'])} failures")
    if done.get("finalize_returncode") != 0:
        failures.append(f"finalizer return code is {done.get('finalize_returncode')}")

    rate_cards = sorted((ROOT / "dynamic_rate").glob("**/card.json"))
    noncausal_names = []
    for path in rate_cards:
        card = load(path)
        names = set(card.get("q_names", ()))
        if "segment_elapsed_over_8h" not in names or any("segment_fraction" in n for n in names):
            noncausal_names.append(str(path))
        if card.get("development_targets_read") is not False or card.get("sealed_partition_opened") is not False:
            failures.append(f"unsafe partition flag in {path}")
    if len(rate_cards) != EXPECTED["W1"]:
        failures.append(f"dynamic-rate card count is {len(rate_cards)}, expected 36")
    if noncausal_names:
        failures.append(f"{len(noncausal_names)} rate cards retain the forbidden segment-end feature")

    state_cards = sorted((ROOT / "full_mark_final").glob("**/card.json"))
    by_subject: dict[str, list[int]] = defaultdict(list)
    for path in state_cards:
        card = load(path)
        by_subject[str(card["subject"])].append(int(card["selected_epoch"]))
        trajectory = Path(card.get("state_trajectory", ""))
        checkpoint = Path(card.get("checkpoint", ""))
        if not trajectory.exists() or not checkpoint.exists():
            failures.append(f"state artifact missing for {path}")
        if card.get("development_targets_read") is not False or card.get("sealed_partition_opened") is not False:
            failures.append(f"unsafe partition flag in {path}")
    if len(state_cards) != EXPECTED["W3"]:
        failures.append(f"full-state card count is {len(state_cards)}, expected 21")

    missing_reports = [name for name in REPORTS if not (ROOT / "final_reports" / name).exists()]
    if missing_reports:
        failures.append(f"missing final reports: {missing_reports}")
    scope_path = ROOT / "final_reports/scope_summary.json"
    scope = load(scope_path) if scope_path.exists() else {}
    if scope.get("development_targets_read") is not False or scope.get("sealed_partition_opened") is not False:
        failures.append("unsafe partition flag in final scope summary")

    audit = {
        "format": "group_event_state_v0_3_5_causal_closeout_audit_v1",
        "status": "PASS" if not failures else "FAIL",
        "queue": {
            "n_complete": len(done.get("complete", ())),
            "counts_by_kind": dict(counts),
            "n_failed": len(done.get("failed", ())),
            "finalize_returncode": done.get("finalize_returncode"),
        },
        "causal_feature": {
            "n_rate_cards": len(rate_cards),
            "required": "segment_elapsed_over_8h",
            "forbidden": "segment_fraction or any coverage-segment end",
            "n_violations": len(noncausal_names),
        },
        "state_training": {
            "n_cards": len(state_cards),
            "selected_epochs_by_subject": {k: sorted(v) for k, v in sorted(by_subject.items())},
            "primary_interpretation": "epoch zero is an engineering output, not evidence of a learned event-content state",
        },
        "claim_boundary": {
            "h1": "dynamic burden has heterogeneous development support; fixed-time grammar state is not established",
            "h2b": "risk transfer is low-n development evidence; early ictal field is underpowered",
            "h3": "no stable support; 5000/10000-event scales are not estimable rather than biological negatives",
            "next_version": "the completed model is not the shared cross-horizon S_N/S_G experiment",
        },
        "missing_reports": missing_reports,
        "failures": failures,
        "development_targets_read": False,
        "sealed_partition_opened": False,
    }
    out = ROOT / "final_reports/causal_closeout_audit.json"
    out.write_text(json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
