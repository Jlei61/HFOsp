#!/usr/bin/env python3
"""Two-stage, acyclic write-once lock for Z/M Phase C."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import topic4_zm_phasec_contract as C  # noqa: E402


LEGACY_PRODUCTION_SCHEMA = "zm_phasec_contract_v1.2_2026-07-28"


def invalidate_legacy_manifest(path, *, invalidated_dir=None):
    """Move only the known v1.2 manifest to a recoverable invalidated path."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    schema = payload.get("schema")
    if schema == C.PHASEC_CONTRACT_VERSION:
        return None
    if schema != LEGACY_PRODUCTION_SCHEMA:
        raise C.ImmutableManifestError(
            f"refusing to invalidate unknown manifest schema: {schema}"
        )
    claimed = payload.get("manifest_sha256")
    body = {
        key: value for key, value in payload.items()
        if key != "manifest_sha256"
    }
    if not isinstance(claimed, str) or C._object_sha(body) != claimed:
        raise C.ImmutableManifestError(
            "refusing to invalidate legacy manifest with invalid self hash"
        )
    invalidated = (
        Path(invalidated_dir)
        if invalidated_dir is not None
        else path.parent / "invalidated"
    )
    invalidated.mkdir(parents=True, exist_ok=True)
    destination = invalidated / (
        f"{path.stem}_{LEGACY_PRODUCTION_SCHEMA}_{claimed[:12]}.json"
    )
    try:
        os.link(path, destination)
    except FileExistsError:
        if destination.read_bytes() != path.read_bytes():
            raise C.ImmutableManifestError(
                f"legacy invalidation destination conflicts: {destination}"
            )
    path.unlink()
    for directory in {path.parent, invalidated}:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("input", "final"), required=True,
        help="input is non-production; final requires both coordinate locks",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="write-once manifest path",
    )
    parser.add_argument(
        "--check-only", action="store_true",
        help="build and validate without creating the output",
    )
    parser.add_argument(
        "--invalidate-legacy", action="store_true",
        help=(
            "before final write, recoverably archive only the known v1.2 "
            "manifest; never implied by --resume or --check-only"
        ),
    )
    args = parser.parse_args(argv)
    if args.invalidate_legacy and (
        args.phase != "final" or args.check_only
    ):
        parser.error(
            "--invalidate-legacy is allowed only for a non-check-only final lock"
        )

    if args.phase == "input":
        manifest = C.build_input_manifest(ROOT)
        default_output = ROOT / C.DEFAULT_INPUT_OUTPUT
    else:
        manifest = C.build_final_manifest(ROOT)
        default_output = ROOT / C.DEFAULT_OUTPUT
    output = default_output if args.output is None else args.output
    if args.check_only:
        print(json.dumps({
            "status": "validated",
            "phase": args.phase,
            "schema": manifest["schema"],
            "production_authorized": manifest["production_authorized"],
            "manifest_sha256": manifest["manifest_sha256"],
            "panel_manifest_sha256": manifest["provenance"][
                "panel_manifest_sha256"
            ],
            "identity_measure_ms": manifest["c0"]["protocols"]["identity"][
                "measure_ms"
            ],
            "gain_measure_ms": manifest["c0"]["protocols"]["gain"]["measure_ms"],
            "output": str(output),
        }, sort_keys=True))
        return 0

    invalidated_path = None
    if args.phase == "final":
        if output.exists():
            with output.open(encoding="utf-8") as handle:
                existing_schema = json.load(handle).get("schema")
            if (
                existing_schema == LEGACY_PRODUCTION_SCHEMA
                and not args.invalidate_legacy
            ):
                raise SystemExit(
                    "legacy v1.2 Phase-C manifest is present; rerun explicitly "
                    "with --invalidate-legacy"
                )
        if args.invalidate_legacy:
            invalidated_path = invalidate_legacy_manifest(output)
    status = C.write_manifest_once(output, manifest)
    print(json.dumps({
        "status": status,
        "phase": args.phase,
        "schema": manifest["schema"],
        "production_authorized": manifest["production_authorized"],
        "manifest_sha256": manifest["manifest_sha256"],
        "panel_manifest_sha256": manifest["provenance"][
            "panel_manifest_sha256"
        ],
        "identity_measure_ms": manifest["c0"]["protocols"]["identity"][
            "measure_ms"
        ],
        "gain_measure_ms": manifest["c0"]["protocols"]["gain"]["measure_ms"],
        "output": str(output),
        "invalidated_legacy": (
            None if invalidated_path is None else str(invalidated_path)
        ),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
