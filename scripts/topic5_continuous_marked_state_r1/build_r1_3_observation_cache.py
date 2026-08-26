#!/usr/bin/env python3
"""Cache deterministic explicit features once; raw waveform remains streamed."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from src.topic5_continuous_marked_state_r1 import contract
from src.topic5_continuous_marked_state_r1.coverage import CoverageTable
from src.topic5_continuous_marked_state_r1.r1_2 import (
    _bridge_scaler, load_full_admissible_event_stream, load_full_design,
)
from src.topic5_continuous_marked_state_r1.r1_3 import R1_3_REVISION
from src.topic5_continuous_marked_state_r1.raw_observation import RawAnchorReader


CACHE_REVISION = "r1_3_normalised_explicit_and_mask_per_anchor_v2"


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject", required=True,
        choices=contract.EXTENDED_DEVELOPMENT_SUBJECTS,
    )
    parser.add_argument(
        "--r1-2-root", type=Path, default=contract.RESULT_ROOT / "r1_2"
    )
    parser.add_argument(
        "--output-root", type=Path, default=contract.RESULT_ROOT / "r1_3"
    )
    args = parser.parse_args()
    upstream_manifest_path = (
        args.r1_2_root / "cache" / args.subject / "manifest.json"
    )
    upstream = json.loads(upstream_manifest_path.read_text())
    if upstream.get("status") != "COMPLETE" or upstream.get("sealed_opened") is not False:
        raise ValueError(f"{args.subject}: invalid R1.2 full-anchor cache")
    design_path = Path(upstream["design"])
    if contract.sha256_file(design_path) != upstream["design_sha256"]:
        raise ValueError(f"{args.subject}: R1.2 design hash mismatch")
    design = load_full_design(design_path)
    contract.assert_development_times(
        args.subject, design.anchor_time[design.anchor_split == 0], "train"
    )
    contract.assert_development_times(
        args.subject, design.anchor_time[design.anchor_split == 1], "validation"
    )
    baseline_path = args.r1_2_root / "baselines" / args.subject / "seed_0/models.pt"
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
    coverage = CoverageTable.load(
        args.r1_2_root / "coverage" / f"{args.subject}.npz"
    )
    stream = load_full_admissible_event_stream(args.subject, coverage)
    bridge_result = json.loads((
        args.r1_2_root / "bridge_e1" / args.subject / "seed_0/result.json"
    ).read_text())
    _, _, sampled, _ = _bridge_scaler(
        args.subject, baseline_path, bridge_result, stream, coverage
    )
    reader = RawAnchorReader(args.subject, stream.event_time)
    explicit = []
    contact_mask = []
    for index, time in enumerate(design.anchor_time):
        value = reader.read(float(time))
        if value is None:
            raise RuntimeError(f"{args.subject}: anchor {index} became unreadable")
        explicit.append(value.explicit)
        contact_mask.append(value.contact_mask)
    explicit = np.stack(explicit).astype(np.float32)
    contact_mask = np.stack(contact_mask).astype(bool)
    explicit = (
        (explicit - sampled.explicit_mean) / sampled.explicit_scale
    ).astype(np.float32)
    if not np.isfinite(explicit).all():
        raise ValueError("R1.3 explicit cache is non-finite")
    output = args.output_root / "cache" / args.subject
    explicit_path = output / "explicit_normalised.npy"
    atomic_npy(explicit_path, explicit)
    contact_mask_path = output / "contact_mask.npy"
    atomic_npy(contact_mask_path, contact_mask)
    manifest = {
        "status": "COMPLETE",
        "contract": contract.REVISION,
        "r1_3_revision": R1_3_REVISION,
        "cache_revision": CACHE_REVISION,
        "subject": args.subject,
        "n_anchors": len(design.anchor_time),
        "n_contacts": stream.n_contacts,
        "n_explicit_features": explicit.shape[-1],
        "design": str(design_path),
        "design_sha256": contract.sha256_file(design_path),
        "upstream_r1_2_cache_manifest": str(upstream_manifest_path),
        "upstream_r1_2_cache_manifest_sha256": contract.sha256_file(
            upstream_manifest_path
        ),
        "explicit": str(explicit_path),
        "explicit_sha256": contract.sha256_file(explicit_path),
        "contact_mask": str(contact_mask_path),
        "contact_mask_sha256": contract.sha256_file(contact_mask_path),
        "explicit_scaler_source": "frozen_bridge_selected_train_anchors",
        "raw_waveform_cached_here": False,
        "raw_waveform_policy": "stream minute chunks from frozen raw Zarr",
        "sealed_opened": False,
    }
    contract.atomic_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
