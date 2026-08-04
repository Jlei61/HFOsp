#!/usr/bin/env python3
"""Freeze the cohort field manifest before any early-ictal value is read."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_shared_scaffold_field_readout import (  # noqa: E402
    build_frozen_field_manifest,
    validate_frozen_subject_field_record,
    write_frozen_field_manifest,
)


CODE_PATHS = (
    ROOT / "src/topic5_shared_scaffold_field_readout.py",
    ROOT / "src/topic5_shared_scaffold_rollout.py",
    ROOT / "src/topic5_shared_scaffold_rnn.py",
    ROOT / "scripts/freeze_topic5_shared_scaffold_rollout_subject_v0_2.py",
)


def combined_code_sha256(paths: list[Path] | tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted((item.resolve() for item in paths), key=str):
        if not path.is_file():
            raise FileNotFoundError(path)
        digest.update(str(path.relative_to(ROOT)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def collect_complete_records(
    *,
    output_root: Path,
    subjects: list[str],
    models: list[str],
    freeze_dir: str,
) -> list[dict]:
    """Load the exact expected denominator and reject partial cohort freezes."""

    records = []
    for subject in subjects:
        subject_root = output_root / freeze_dir / "per_subject" / subject
        done_path = subject_root / "DONE.json"
        if not done_path.is_file():
            raise FileNotFoundError(f"missing rollout completion record: {done_path}")
        done = json.loads(done_path.read_text())
        if done.get("status") != "COMPLETE" or done.get("target_values_read") is not False:
            raise ValueError(f"{subject}: field rollout is incomplete or target-unsealed")
        for model in models:
            path = subject_root / f"{model}_field_record.json"
            if not path.is_file():
                raise FileNotFoundError(path)
            record = json.loads(path.read_text())
            validate_frozen_subject_field_record(record)
            if record.get("subject_id") != subject or record.get("model_name") != model:
                raise ValueError(f"field record identity mismatch: {path}")
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "config/topic5_shared_scaffold_propagation_rnn_v0_2.yaml",
    )
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument(
        "--models", nargs="*", default=["structured", "ordinary_gru"]
    )
    parser.add_argument(
        "--source-pool-rule",
        choices=("learned_axis", "normalized_laplacian"),
        default="learned_axis",
    )
    args = parser.parse_args()
    freeze_dir = (
        "field_freeze"
        if args.source_pool_rule == "learned_axis"
        else "field_freeze_diffusion_graph_sensitivity"
    )
    config = yaml.safe_load(args.config.resolve().read_text())
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else ROOT / config["output_root"]
    )
    dataset_manifest_path = (
        Path(config["dataset_artifact_root"]).resolve()
        / config["dataset_root"]
        / "dataset_manifest.json"
    )
    dataset_manifest = json.loads(dataset_manifest_path.read_text())
    subjects = list(args.subjects or dataset_manifest["cohort_subjects"])
    models = list(map(str, args.models))
    records = collect_complete_records(
        output_root=output_root,
        subjects=subjects,
        models=models,
        freeze_dir=freeze_dir,
    )
    rules = sorted({str(record.get("source_pool_rule")) for record in records})
    if len(rules) != 1:
        raise ValueError(f"field records mix source pool rules: {rules}")
    manifest = build_frozen_field_manifest(
        records,
        created_utc=datetime.now(timezone.utc).isoformat(),
        code_sha256=combined_code_sha256(CODE_PATHS),
    )
    manifest.update(
        {
            "expected_subjects": subjects,
            "expected_models": models,
            "source_pool_rule": rules[0],
            "source_pool_rule_tier": (
                "primary" if args.source_pool_rule == "learned_axis" else "sensitivity"
            ),
            "dataset_manifest_path": str(dataset_manifest_path),
            "dataset_manifest_sha256": hashlib.sha256(
                dataset_manifest_path.read_bytes()
            ).hexdigest(),
        }
    )
    # The denominator fields above are part of the immutable manifest.
    manifest.pop("manifest_sha256")
    from src.topic5_shared_scaffold_field_readout import (  # noqa: E402
        frozen_field_manifest_fingerprint,
    )

    manifest["manifest_sha256"] = frozen_field_manifest_fingerprint(manifest)
    destination = output_root / freeze_dir / "FROZEN_FIELD_MANIFEST.json"
    write_frozen_field_manifest(destination, manifest)
    print(
        json.dumps(
            {
                "status": "COMPLETE",
                "path": str(destination),
                "n_subjects": len(subjects),
                "n_records": len(records),
                "manifest_sha256": manifest["manifest_sha256"],
                "target_values_read": False,
            },
            allow_nan=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
