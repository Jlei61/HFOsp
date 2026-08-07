#!/usr/bin/env python3
"""Execute frozen v2.6 controls against repaired v2.7 checkpoints.

The control implementations are intentionally reused rather than rewritten.
This adapter changes only the config/output namespace, data preparation,
v2.7 fitter and frozen-state verifier.  It records both implementation and
adapter hashes and refuses checkpoints carrying the v2.6 contract.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_topic5_stateful_event_rnn_v2_7_formal as formal  # noqa: E402
from src.topic5_resource_guard import pin_thread_environment  # noqa: E402


CONTROL_MODULES = {
    "dense": "scripts.evaluate_topic5_stateful_event_rnn_v2_6_dense",
    "state-reset": "scripts.evaluate_topic5_stateful_event_rnn_v2_6_state_reset",
    "memory-curve": "scripts.evaluate_topic5_stateful_event_rnn_v2_6_memory_curve",
    "block-null": "scripts.run_topic5_stateful_event_rnn_v2_6_block_null",
    "reversal-null": "scripts.run_topic5_stateful_event_rnn_v2_6_reversal_null",
    "h40": "scripts.run_topic5_stateful_event_rnn_v2_6_h40",
}


def configure_control(module):
    """Bind one unchanged control implementation to v2.7 infrastructure."""

    replacements = {
        "DEFAULT_CONFIG": formal.DEFAULT_CONFIG,
        "prepare_subject": formal.prepare_subject,
        "fit_profile": formal.fit_profile,
        "jsonable": formal.jsonable,
        "score_dict": formal.score_dict,
        "sha256": formal.sha256,
        "verify_frozen": formal.verify_frozen,
    }
    for name, value in replacements.items():
        if hasattr(module, name):
            setattr(module, name, value)
    return module


def verify_checkpoints(output: Path, subjects) -> None:
    for subject in subjects:
        for seed in (17, 29, 43):
            path = output / "checkpoints" / subject / f"seed_{seed}.pt"
            if not path.exists():
                raise RuntimeError(f"missing v2.7 checkpoint: {path}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if checkpoint.get("contract") != "topic5_stateful_event_sequence_rnn_v2_7":
                raise RuntimeError(f"non-v2.7 checkpoint rejected: {path}")


def _value_after(arguments, flag, default=None):
    try:
        return arguments[arguments.index(flag) + 1]
    except (ValueError, IndexError):
        return default


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--control", choices=tuple(CONTROL_MODULES), required=True)
    known, remaining = parser.parse_known_args()
    pin_thread_environment(1, disable_cuda=True)

    config_path = Path(
        _value_after(remaining, "--config", str(formal.DEFAULT_CONFIG))
    ).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output = ROOT / config["output_root"]
    formal.verify_frozen(config_path, output)
    phase = _value_after(remaining, "--phase")
    if phase == "patients":
        start = remaining.index("--subjects") + 1 if "--subjects" in remaining else len(remaining)
        subjects = []
        for value in remaining[start:]:
            if value.startswith("--"):
                break
            subjects.append(value)
        if not subjects:
            raise RuntimeError("control patients phase requires explicit subjects")
        verify_checkpoints(output, subjects)

    module = configure_control(importlib.import_module(CONTROL_MODULES[known.control]))
    original_argv = sys.argv
    try:
        sys.argv = [str(Path(module.__file__).resolve()), *remaining]
        module.main()
    finally:
        sys.argv = original_argv

    manifest_root = output / "control_adapter"
    manifest_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "contract": config["contract"],
        "control": known.control,
        "status": "CONTROL_ADAPTER_EXECUTION_COMPLETE",
        "phase": phase,
        "implementation_module": CONTROL_MODULES[known.control],
        "implementation_sha256": formal.sha256(Path(module.__file__).resolve()),
        "adapter_sha256": formal.sha256(Path(__file__).resolve()),
        "formal_runner_sha256": formal.sha256(
            ROOT / "scripts/run_topic5_stateful_event_rnn_v2_7_formal.py"
        ),
        "config_sha256": formal.sha256(config_path),
        "checkpoint_contract_required": "topic5_stateful_event_sequence_rnn_v2_7",
    }
    destination = manifest_root / f"{known.control}_{phase}.json"
    destination.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

