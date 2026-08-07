#!/usr/bin/env python3
"""Validation-only epoch-boundary audit for repaired Topic 5 v2.7.

The candidate-selection logic is inherited from the frozen v2.6 audit, while
every retraining call is explicitly rebound to the v2.7 fit implementation.
No test partition is evaluated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_resource_guard import (  # noqa: E402
    configure_torch_threads,
    pin_thread_environment,
)

pin_thread_environment(1)

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from scripts import (  # noqa: E402
    refine_topic5_stateful_event_rnn_v2_6_epoch_boundary as parent_boundary,
)
from scripts.run_topic5_stateful_event_rnn_v2_7 import (  # noqa: E402
    DEFAULT_CONFIG,
    V27_MODULE,
    V27_RUNNER,
    assert_repair_only_config,
    fit_profile as fit_profile_v27,
    jsonable,
    provenance_manifest,
    sha256,
)


def fit_profile(
    subject, profile, datasets, encoder, config, scales, seed
):
    """Adapt the frozen v2.6 boundary-audit call to the v2.7 fitter.

    The parent audit passes ``subject`` for provenance, whereas the repaired
    v2.7 fitter does not need it.  Keeping this seven-argument adapter local to
    the audit preserves the frozen candidate-selection logic without changing
    the v2.7 training grid or implementation.
    """

    del subject
    return fit_profile_v27(profile, datasets, encoder, config, scales, seed)


def refine_subject(subject, config, output, **kwargs):
    """Call the frozen audit logic with repaired v2.7 training."""

    previous = parent_boundary.fit_profile
    parent_boundary.fit_profile = fit_profile
    try:
        return parent_boundary.refine_subject(
            subject, config, output, **kwargs
        )
    finally:
        parent_boundary.fit_profile = previous


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--subjects", nargs="*")
    parser.add_argument("--top-n", type=int, default=3)
    parser.add_argument("--trigger-epoch", type=int, default=35)
    parser.add_argument("--maximum-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=16)
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = assert_repair_only_config(config_path)
    provenance = provenance_manifest(config_path)
    output = ROOT / config["output_root"]
    screen_root = output / "validation_screen/per_subject"
    subjects = (
        sorted(args.subjects)
        if args.subjects
        else sorted(path.stem for path in screen_root.glob("*.json"))
    )
    configure_torch_threads(torch, int(config["torch_num_threads"]))
    rows = []
    for index, subject in enumerate(subjects, 1):
        print(f"[v2.7 epoch-boundary {index}/{len(subjects)}] {subject}", flush=True)
        rows.append(
            refine_subject(
                subject,
                config,
                output,
                top_n=args.top_n,
                trigger_epoch=args.trigger_epoch,
                maximum_epochs=args.maximum_epochs,
                patience=args.patience,
            )
        )
    root = output / "validation_screen"
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(root / "epoch_boundary_summary.csv", index=False)
    summary = {
        "contract": config["contract"],
        "status": "VALIDATION_EPOCH_BOUNDARY_AUDIT_COMPLETE",
        "n_subjects": int(len(rows)),
        "n_triggered": int(
            sum(row["status"] in {"EXTENDED", "ALREADY_EXTENDED"} for row in rows)
        ),
        "n_profile_changed": int(sum(row["profile_changed"] for row in rows)),
        "test_results_read": False,
        "trigger_epoch": int(args.trigger_epoch),
        "maximum_epochs": int(args.maximum_epochs),
        "patience": int(args.patience),
        "top_n": int(args.top_n),
        "config_sha256": sha256(config_path),
        "module_sha256": sha256(V27_MODULE),
        "primary_runner_sha256": sha256(V27_RUNNER),
        "boundary_runner_sha256": sha256(Path(__file__).resolve()),
        "parent_v2_6": provenance["parent_v2_6"],
        "repair_only_grid_match": True,
    }
    destination = root / "EPOCH_BOUNDARY_STATE.json"
    temporary = destination.with_suffix(".json.tmp")
    with temporary.open("w") as stream:
        json.dump(jsonable(summary), stream, indent=2, sort_keys=True)
    temporary.replace(destination)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
