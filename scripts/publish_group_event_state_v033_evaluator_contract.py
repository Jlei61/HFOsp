#!/usr/bin/env python3
"""Publish the unique v0.3.3 evaluator contract after cross-workstream review.

This is an engineering contract, not a human scientific result.  It records
which pure scoring functions, target bins and dispersion rules both the
training laboratory and independent evaluator must use.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.v032_eval.contract import atomic_json  # noqa: E402
from src.topic5_group_event_state.v033_evaluator import canonical as C  # noqa: E402
from src.topic5_group_event_state.v033_evaluator import oracle as O  # noqa: E402


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _sentinel_summary(power: dict) -> dict:
    if power.get("preset") != "sentinel":
        raise ValueError("canonical publication requires the reviewed sentinel summary")
    if power.get("human_targets_used") is not False or power.get("sealed_partition_opened") is not False:
        raise ValueError("sentinel must be synthetic-only with the sealed partition closed")
    source_commit = str(power.get("source_commit", ""))
    if len(source_commit) != 40 or any(ch not in "0123456789abcdef" for ch in source_commit):
        raise ValueError("sentinel must carry a full source commit")
    views = {}
    for curve in power.get("curves", []):
        cells = curve.get("cells", [])
        d0 = next((c for c in cells if c.get("kind") == "D0"), None)
        d3 = next((c for c in cells if c.get("kind") == "D3"), None)
        if d0 is None or d3 is None:
            raise ValueError(f"sentinel lacks D0/D3 cells for {curve.get('view')}")
        fp = d0.get("false_positive_rate_by_level", {})
        if any(float(value) > 0.0 for value in fp.values()):
            raise ValueError(f"D0 false positive in sentinel view {curve.get('view')}")
        views[str(curve["view"])] = {
            "D0_false_positive_rate_by_level": fp,
            "D3_power_by_level": d3.get("power_by_level", {}),
            "D3_gain_by_level": d3.get("gain_by_level", {}),
            "replicates": {"D0": d0.get("n_replicates"), "D3": d3.get("n_replicates")},
        }
    return {
        "status": "SENTINEL_PASS_D0_NO_FALSE_POSITIVES",
        "scope": "one-replicate diagnostic; not a final power curve",
        "source_commit": source_commit,
        "views": views,
    }


def build_contract(*, power: dict, boundary: dict, discrepancy: dict,
                   training_commit: str, evaluator_test_count: int,
                   joint_test_count: int) -> dict:
    cohort = boundary.get("cohort", {})
    required_boundary = (
        "all_kept_equals_in_target_segments",
        "all_matches_v032_eligibility",
        "all_state_events_exclude_seizure_and_postictal",
    )
    failed = [key for key in required_boundary if cohort.get(key) is not True]
    if failed:
        raise ValueError(f"boundary audit failed: {failed}")
    if boundary.get("sealed_partition_opened") is not False:
        raise ValueError("boundary audit opened the sealed partition")
    if discrepancy.get("canonical_schema_version") != C.SCHEMA_VERSION:
        raise ValueError("E1146 discrepancy audit used another evaluator schema")
    if discrepancy.get("sealed_partition_opened") is not False:
        raise ValueError("E1146 discrepancy audit opened the sealed partition")
    if evaluator_test_count < 1 or joint_test_count < 1:
        raise ValueError("test counts must be positive")
    return {
        "format": "group_event_state_v0_3_3_canonical_evaluator_contract",
        "published_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "ACTIVE_DEVELOPMENT_ONLY",
        "canonical_schema_version": C.SCHEMA_VERSION,
        "tolerance_nats": C.TOLERANCE_NATS,
        "sign_convention": C.SIGN_CONVENTION,
        "count_profile": {
            "target_bins_seconds": [[0, 300], [300, 900], [900, 1800]],
            "interval_convention": "left_closed_right_open_[t+a,t+b)",
            "score": "sum of per-bin NB negative log likelihood",
            "primary_dispersion": "one H-fitted log_r per bin, frozen and shared by H and H_plus_state",
            "free_dispersion": "sensitivity arm only",
        },
        "grammar": {
            "target": "positive-K event subset",
            "score": "conditional independent-Bernoulli subset negative log likelihood given observed K",
            "scope_note": "exact only for this product-form K-subset family, not an arbitrary set distribution",
        },
        "reduction": {
            "primary_gain": "NLL(H) - NLL(H_plus_state)",
            "unmasked_nonfinite": "hard error",
            "paired_rows": "same anchors, targets, masks, weights, predictions and dispersions",
        },
        "phase_contract": {
            "state_selection_phase": O.STATE_SELECTION_PHASE,
            "development_evaluation_phase": O.EVALUATION_PHASE,
            "reported_gain_uses": "dev_test_only",
            "estimability_uses": "dev_test_only",
            "dev_val_plus_dev_test": "descriptive_total_only_never_a_result_denominator",
            "scope_note": "dev_test is development-consumed, not sealed or formal evaluation",
        },
        "implementation": {
            "evaluator_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "training_commit": str(training_commit),
            "training_import": "src.topic5_group_event_state.v033_evaluator.canonical",
            "evaluator_tests_passed": int(evaluator_test_count),
            "joint_evaluator_training_tests_passed": int(joint_test_count),
        },
        "assay": _sentinel_summary(power),
        "boundary_audit": {
            "source_commit": boundary.get("source_commit"),
            "n_subjects": cohort.get("n_subjects"),
            "n_ok": cohort.get("n_ok"),
            "n_events_excluded_seizure_or_postictal": cohort.get("n_events_excluded_seizure_or_postictal"),
            "n_anchors_total": cohort.get("n_anchors_total"),
        },
        "E1146_discrepancy": {
            "source_commit": discrepancy.get("source_commit"),
            "all_published_reproduced": discrepancy.get("audit", {}).get("all_published_reproduced"),
            "first_divergence": discrepancy.get("audit", {}).get("first_divergence"),
            "interpretation": "old branches did not score one identical object; the old sign conflict is retired",
        },
        "evidence_label": "ENGINEERING_AND_SYNTHETIC_ASSAY_ONLY",
        "human_targets_used": False,
        "human_scientific_conclusion": "NONE",
        "sealed_partition_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power", type=Path, default=ROOT / "results/group_event_state/v0_3_3/evaluator_assay/d0_d4_power_curve.json")
    parser.add_argument("--boundary", type=Path, default=ROOT / "results/group_event_state/v0_3_3/evaluator_assay/data_boundary_audit.json")
    parser.add_argument("--discrepancy", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_3/agent_a/e1146_discrepancy_audit.json"))
    parser.add_argument("--training-commit", required=True)
    parser.add_argument("--evaluator-test-count", type=int, required=True)
    parser.add_argument("--joint-test-count", type=int, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "results/group_event_state/v0_3_3/evaluator_assay/canonical_evaluator.json")
    parser.add_argument("--shared", type=Path, default=Path("/data/hfosp_group_event_state_v0_3_3/shared/evaluator_contract/canonical_evaluator.json"))
    args = parser.parse_args()
    payload = build_contract(
        power=_load(args.power), boundary=_load(args.boundary), discrepancy=_load(args.discrepancy),
        training_commit=args.training_commit, evaluator_test_count=args.evaluator_test_count,
        joint_test_count=args.joint_test_count,
    )
    atomic_json(args.out, payload)
    atomic_json(args.shared, payload)
    print(json.dumps({"status": "published", "out": str(args.out), "shared": str(args.shared)}, indent=2))


if __name__ == "__main__":
    main()
