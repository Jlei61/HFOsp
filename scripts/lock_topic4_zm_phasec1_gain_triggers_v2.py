#!/usr/bin/env python3
"""Lock C1 conditional-gain triggers from the corrected v2 base atlas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import scripts.analyze_topic4_zm_phasec1 as C1  # noqa: E402
import scripts.analyze_topic4_zm_phasec1_v2 as A2  # noqa: E402
import scripts.lock_topic4_zm_phasec1_gain_triggers as V1  # noqa: E402
import src.topic4_zm_phasec_neighbourhood as N  # noqa: E402


def _bind_v2_provenance(manifest, base):
    amendment = A2._read_amendment()
    expected = {
        "analysis_amendment_path": C1._relative(A2.AMENDMENT),
        "analysis_amendment_file_sha256": C1._sha256(A2.AMENDMENT),
        "analysis_amendment_sha256": amendment["amendment_sha256"],
        "analysis_producer_file_sha256": A2._analysis_producers(),
    }
    if any(base.get(key) != value for key, value in expected.items()):
        raise ValueError(
            "C1 base atlas does not bind the live v2 analysis amendment"
        )
    body = {
        key: value for key, value in manifest.items()
        if key != "manifest_sha256"
    }
    body.update({
        **expected,
        "trigger_analysis_wrapper_file_sha256": {
            str(Path(__file__).resolve().relative_to(CODE_ROOT)):
            C1._sha256(Path(__file__).resolve())
        },
    })
    return {**body, "manifest_sha256": C1._object_sha(body)}


def build_trigger_manifest(base, *, base_atlas_path, denominator_provider=None):
    kwargs = {}
    if denominator_provider is not None:
        kwargs["denominator_provider"] = denominator_provider
    manifest = V1.build_trigger_manifest(
        base,
        base_atlas_path=base_atlas_path,
        **kwargs,
    )
    return _bind_v2_provenance(manifest, base)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-atlas",
        default=str(C1.OUT / "phasec1_base_atlas_dt.json"),
    )
    parser.add_argument("--output", default=str(C1.GAIN_TRIGGER_MANIFEST))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    base_path = Path(args.base_atlas)
    base = C1._load_json(base_path)
    manifest = build_trigger_manifest(base, base_atlas_path=base_path)
    status = (
        "validated_not_written"
        if args.check_only
        else N.write_json_once(Path(args.output), manifest)
    )
    print(json.dumps({
        "status": status,
        "n_triggered_cells": manifest["n_triggered_cells"],
        "manifest_sha256": manifest["manifest_sha256"],
        "analysis_amendment_sha256": manifest[
            "analysis_amendment_sha256"
        ],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
