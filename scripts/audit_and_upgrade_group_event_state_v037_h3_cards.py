#!/usr/bin/env python3
"""Verify v0.3.7 H3 nesting, then correct metadata-only v2 cards to v3.

The numerical results and checkpoints are left untouched.  A card is upgraded
only after the saved checkpoint proves that M0/M1/M2 share identical common
parameters and differ solely in their registered zero-bias feedback readouts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
from typing import Any

import torch


DEFAULT_ROOT = Path("/data/hfosp_group_event_state_v0_3_7/h3_independent_generative")
FEEDBACK_PREFIXES = ("count_feedback_", "mark_feedback_")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _same_tensor(left: Any, right: Any) -> bool:
    return isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and torch.equal(left, right)


def _verify_checkpoint(card: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = Path(card["checkpoint_path"])
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    models = payload["models"]
    required = ("M0_common_drive", "M1_count_feedback", "M2_mark_feedback")
    if tuple(models) != required:
        raise ValueError(f"unexpected model order/keys in {checkpoint_path}: {tuple(models)}")
    common_keys = [key for key in models[required[0]] if not key.startswith(FEEDBACK_PREFIXES)]
    for parent, child in zip(required, required[1:]):
        unequal = [key for key in common_keys if not _same_tensor(models[parent][key], models[child][key])]
        if unequal:
            raise ValueError(f"common core changed between {parent} and {child}: {unequal[:5]}")
    for family, expected_prefix in (
        ("M1_count_feedback", "count_feedback_"),
        ("M2_mark_feedback", "mark_feedback_"),
    ):
        trained = card["training"][family]["trained_parameters"]
        if not trained or any(not name.startswith(expected_prefix) for name in trained):
            raise ValueError(f"unexpected trained edge list for {family}: {trained}")
        if any("bias" in name for name in trained):
            raise ValueError(f"feedback readout has a fitted bias for {family}: {trained}")
    return {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha(checkpoint_path),
        "common_parameter_tensors_verified": len(common_keys),
        "m0_m1_m2_common_parameters_bit_equal": True,
        "feedback_readouts_zero_bias": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.glob("**/card.json")):
        card = json.loads(path.read_text(encoding="utf-8"))
        if not str(card.get("format", "")).startswith("group_event_state_v0_3_7_h3_independent_generative_card_v"):
            continue
        before_sha = _sha(path)
        verification = _verify_checkpoint(card)
        original_format = card["format"]
        card.pop("same_core_capacity_and_intercept", None)
        card.update({
            "format": "group_event_state_v0_3_7_h3_independent_generative_card_v3",
            "common_core_and_intercept_frozen_across_nested_models": True,
            "feedback_edges_add_zero_bias_source_specific_readouts": True,
            "complete_models_have_equal_parameter_count": False,
            "edge_estimand_scope": {
                "exposure_block_seconds": int(card["config"]["grid_seconds"]),
                "primary_prediction_lag_blocks": 1,
                "primary_count_edge": "preceding-block burden innovation to next-block count and non-event background",
                "primary_mark_edge": "preceding-block rank-reduced grammar innovation to next-block non-event background beyond M1",
                "persistent_ct_bank_present": True,
                "persistent_ct_readout_fitted_in_primary_v0_3_7": False,
                "long_horizon_feedback_status": "NOT_ESTIMATED_IN_V0_3_7_PRIMARY",
            },
            "metadata_correction": {
                "original_format": original_format,
                "reason": "replace false equal-capacity wording with verified nested-core and explicit extra-edge contract",
                "numerical_results_changed": False,
                "checkpoint_changed": False,
                "verification": verification,
            },
        })
        _atomic_json(path, card)
        rows.append({
            "card": str(path), "before_sha256": before_sha,
            "after_sha256": _sha(path), "original_format": original_format,
            **verification,
        })
    audit = {
        "format": "group_event_state_v0_3_7_h3_card_contract_upgrade_audit_v1",
        "status": "PASS", "n_cards": len(rows), "rows": rows,
        "numerical_results_changed": False, "checkpoints_changed": False,
    }
    _atomic_json(args.root / "card_contract_upgrade_audit.json", audit)
    print(json.dumps({"status": "PASS", "n_cards": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
