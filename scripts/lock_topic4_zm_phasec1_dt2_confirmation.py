#!/usr/bin/env python3
"""Lock the conditional, homologous Phase-C1 dt/2 confirmation subset.

The lock is derived once from a completed native C1 maturation window.  It does
not convert native fields to dt/2: every selected cell resolves by identity in
the independently generated dt/2 coordinate manifest, whose full-field NPZ is
already forward-locked by the final Phase-C manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_topic4_zm_phasec_cell as CELL  # noqa: E402
import src.topic4_zm_phasec_contract as PCC  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity"
PHASEC_MANIFEST_PATH = OUT / "phasec_manifest.json"
NATIVE_SUMMARY_PATH = OUT / "phasec1_summary_dt.json"
GAIN_TRIGGER_PATH = OUT / "c1_gain_trigger_manifest.json"
OUTPUT_PATH = OUT / "phasec1_dt2_confirmation_manifest.json"
SCHEMA = "zm_phasec1_dt2_confirmation_manifest_v1_2026-07-28"
GAIN_TRIGGER_SCHEMA = "zm_phasec1_gain_trigger_manifest_v1_2026-07-28"
PHASES = ("rising", "peak")
NOISES = ("noise_replay", "noise_resample_1", "noise_resample_2")
DT2_SEEDS = (1, 3)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _object_sha(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _rel(path):
    return str(Path(path).resolve().relative_to(ROOT))


def _publish_once(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable lock: {path}")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, indent=2, sort_keys=True, ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _coordinate_ref(manifest, resolution):
    row = manifest["c1"]["coordinate_manifests"][resolution]
    return {
        "path": row["path"],
        "file_sha256": row["file_sha256"],
        "manifest_sha256": row["manifest_sha256"],
        "semantic_sha256": row["semantic_sha256"],
    }


def _validate_gain_trigger(trigger_path, manifest, phasec_path):
    """Require the canonical closed native gain-routing decision.

    This applies even when the native non-tonic window is gain-independent:
    the empty/non-empty trigger decision is a write-once upstream branch and
    must be closed before the downstream dt/2 subset can be frozen.
    """
    trigger_path = Path(trigger_path).resolve()
    if not trigger_path.is_file():
        raise RuntimeError(
            "dt2 confirmation requires the closed canonical C1 gain trigger"
        )
    trigger = _read(trigger_path)
    body = {
        key: value for key, value in trigger.items()
        if key != "manifest_sha256"
    }
    if trigger.get("manifest_sha256") != _object_sha(body):
        raise RuntimeError("C1 gain trigger self-hash mismatch")
    native_ref = _coordinate_ref(manifest, "dt")
    native_coordinate_path = ROOT / native_ref["path"]
    if (
        not native_coordinate_path.is_file()
        or _sha(native_coordinate_path) != native_ref["file_sha256"]
    ):
        raise RuntimeError("native C1 coordinate manifest file/hash mismatch")
    native_coordinate = _read(native_coordinate_path)
    CELL._validate_self_hash(
        native_coordinate, label="native C1 coordinate manifest"
    )
    required = {
        "schema": GAIN_TRIGGER_SCHEMA,
        "selection_is_closed": True,
        "resolution": "dt",
        "phasec_manifest_sha256": manifest["manifest_sha256"],
        "phasec_manifest_file_sha256": _sha(phasec_path),
        "coordinate_manifest_sha256": native_ref["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": native_ref[
            "semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": native_ref["file_sha256"],
        "phasec_producer_file_sha256": manifest["provenance"][
            "producer_file_sha256"
        ],
        "coordinate_producer_file_sha256": native_coordinate[
            "producer_file_sha256"
        ],
    }
    for key, wanted in required.items():
        if trigger.get(key) != wanted:
            raise RuntimeError(f"C1 gain trigger parent/provenance mismatch: {key}")
    if trigger.get("producer_file_sha256") != trigger.get(
        "phasec_producer_file_sha256"
    ):
        raise RuntimeError("C1 gain trigger producer provenance mismatch")
    return trigger


def _windows(adjudication):
    """Return all preregistered windows keyed by phenotype/direction/seed."""
    out = {}
    registered = {
        (row["phenotype"], row["direction"])
        for row in adjudication.get("candidates", [])
    }
    for seed_text, seed_row in adjudication.get("seed_results", {}).items():
        seed = int(seed_text)
        for window in seed_row.get("windows", []):
            label = (window["phenotype"], window["direction"])
            if label not in registered:
                continue
            out.setdefault(label, {}).setdefault(seed, []).append(
                tuple(window["cells"])
            )
    return out


def _native_positive_windows(summary):
    rows = []
    for tier, key in (
        ("primary_convex", "primary_adjudication"),
        ("secondary_shell", "secondary_shell_adjudication"),
    ):
        adjudication = summary[key]
        expected_status = (
            "local_maturation_window"
            if tier == "primary_convex"
            else "maturation_candidate_in_secondary_shell"
        )
        if adjudication.get("status") != expected_status:
            continue
        for (phenotype, direction), by_seed in sorted(
            _windows(adjudication).items()
        ):
            if not all(seed in by_seed for seed in DT2_SEEDS):
                continue
            for seed in DT2_SEEDS:
                for cells in sorted(set(by_seed[seed])):
                    rows.append({
                        "tier": tier,
                        "phenotype": phenotype,
                        "direction": direction,
                        "seed": seed,
                        "cells": list(cells),
                    })
    if not rows:
        raise RuntimeError(
            "native C1 has no homologous maturation window supported by both "
            "independent dt2 seeds 1 and 3"
        )
    return rows


def _dt2_coordinate(manifest):
    ref = _coordinate_ref(manifest, "dt2")
    path = ROOT / ref["path"]
    if not path.is_file() or _sha(path) != ref["file_sha256"]:
        raise RuntimeError("dt2 coordinate manifest file/hash mismatch")
    coordinate = _read(path)
    CELL._validate_self_hash(
        coordinate, label="dt2 C1 coordinate manifest"
    )
    if (
        coordinate.get("manifest_sha256") != ref["manifest_sha256"]
        or coordinate.get("semantic_sha256") != ref["semantic_sha256"]
        or coordinate.get("resolution") != "dt2"
    ):
        raise RuntimeError("dt2 coordinate semantic/manifest lock mismatch")
    return coordinate, ref


def _source_family(manifest, seed, phase):
    row = manifest["per_seed"][str(seed)]["resolution_confirmations"]["dt2"]
    return row, row["c0_carrier_states"][phase]


def build_payload(
    *,
    phasec_path=PHASEC_MANIFEST_PATH,
    native_summary_path=NATIVE_SUMMARY_PATH,
    gain_trigger_path=GAIN_TRIGGER_PATH,
):
    phasec_path = Path(phasec_path).resolve()
    native_summary_path = Path(native_summary_path).resolve()
    manifest = _read(phasec_path)
    PCC.validate_manifest(manifest)
    if manifest.get("production_authorized") is not True:
        raise RuntimeError("dt2 confirmation requires final production manifest")
    summary = _read(native_summary_path)
    trigger = _validate_gain_trigger(gain_trigger_path, manifest, phasec_path)
    if (
        summary.get("resolution") != "dt"
        or summary.get("phasec_manifest_sha256")
        != manifest["manifest_sha256"]
        or summary.get("phasec_manifest_file_sha256") != _sha(phasec_path)
        or summary.get("coordinate_manifest_sha256")
        != manifest["c1"]["coordinate_manifests"]["dt"]["manifest_sha256"]
        or summary.get("coordinate_manifest_semantic_sha256")
        != manifest["c1"]["coordinate_manifests"]["dt"]["semantic_sha256"]
        or summary.get("coordinate_manifest_file_sha256")
        != manifest["c1"]["coordinate_manifests"]["dt"]["file_sha256"]
    ):
        raise RuntimeError("native C1 summary/Phase-C parent provenance mismatch")
    if (
        summary.get("gain_trigger_manifest_sha256")
        != trigger["manifest_sha256"]
    ):
        raise RuntimeError(
            "native C1 summary does not match the closed gain trigger"
        )
    windows = _native_positive_windows(summary)
    coordinate, dt2_coordinate_ref = _dt2_coordinate(manifest)

    selected = {}
    for window in windows:
        seed = int(window["seed"])
        coord_seed = coordinate["seeds"].get(str(seed))
        if not isinstance(coord_seed, dict):
            raise RuntimeError(f"dt2 coordinate manifest lacks seed {seed}")
        cell_map = {
            (row["tier"], row["cell_id"]): row
            for row in coord_seed["cells"] if row.get("status") == "valid"
        }
        for cell_id in window["cells"]:
            key = (seed, window["tier"], cell_id)
            cell = cell_map.get((window["tier"], cell_id))
            if cell is None:
                raise RuntimeError(
                    f"native-positive homolog is absent/invalid at dt2: {key}"
                )
            selected.setdefault(key, {
                "seed": seed,
                "tier": window["tier"],
                "cell_id": cell_id,
                "phenotypes": [],
                "directions": [],
                "trajectory_id": cell["trajectory_id"],
                "path_index": int(cell["path_index"]),
                "path_direction": cell["path_direction"],
                "dt2_slow_state_sha256": cell["state_sha256"],
                "coordinate_npz_file_sha256": coord_seed[
                    "npz_file_sha256"
                ],
                "coordinate_npz_semantic_sha256": coord_seed[
                    "npz_semantic_sha256"
                ],
            })
            selected[key]["phenotypes"].append(window["phenotype"])
            selected[key]["directions"].append(window["direction"])

    selected_cells = []
    expected_arms = []
    for key in sorted(selected):
        row = selected[key]
        row["phenotypes"] = sorted(set(row["phenotypes"]))
        row["directions"] = sorted(set(row["directions"]))
        selected_cells.append(row)
        seed, tier, cell_id = key
        for phase in PHASES:
            resolution_row, family = _source_family(manifest, seed, phase)
            banks = {
                bank["replicate"]: bank for bank in family["noise_banks"]
            }
            for noise in NOISES:
                bank = banks[noise]
                expected_arms.append({
                    "schema": CELL.C1_BASE_PART_SCHEMA,
                    "phasec_manifest_sha256": manifest["manifest_sha256"],
                    "phasec_manifest_file_sha256": _sha(phasec_path),
                    "coordinate_manifest_sha256": coordinate[
                        "manifest_sha256"
                    ],
                    "coordinate_manifest_semantic_sha256": coordinate[
                        "semantic_sha256"
                    ],
                    "coordinate_manifest_file_sha256": dt2_coordinate_ref[
                        "file_sha256"
                    ],
                    "coordinate_npz_file_sha256": row[
                        "coordinate_npz_file_sha256"
                    ],
                    "coordinate_npz_semantic_sha256": row[
                        "coordinate_npz_semantic_sha256"
                    ],
                    "seed": seed,
                    "tier": tier,
                    "cell_id": cell_id,
                    "trajectory_id": row["trajectory_id"],
                    "path_index": row["path_index"],
                    "path_direction": row["path_direction"],
                    "phase": phase,
                    "noise": noise,
                    "resolution": "dt2",
                    "path": CELL._c1_base_relative_path(
                        "dt2", seed, tier, cell_id, phase, noise
                    ),
                    "config_sha": resolution_row["config_sha"],
                    "fast_base_state_hash": family["state"]["state_hash"],
                    "state_file_sha256": family["state"]["file_sha256"],
                    "noise_bank_sha": bank["bank_sha"],
                    "slow_state_sha256": row["dt2_slow_state_sha256"],
                    "burn_in_ms": 500.0,
                    "measure_ms": 8000.0,
                })
    if len(expected_arms) != 6 * len(selected_cells):
        raise RuntimeError("dt2 expected-arm coverage is not 2 phases x 3 noises")
    if len({row["path"] for row in expected_arms}) != len(expected_arms):
        raise RuntimeError("duplicate dt2 confirmation output path")

    payload = {
        "schema": SCHEMA,
        "resolution": "dt2",
        "selection_is_closed": True,
        "final_phasec": {
            "path": _rel(phasec_path),
            "file_sha256": _sha(phasec_path),
            "manifest_sha256": manifest["manifest_sha256"],
        },
        "native_summary": {
            "path": _rel(native_summary_path),
            "file_sha256": _sha(native_summary_path),
            "schema": summary.get("schema"),
        },
        "gain_trigger_manifest": {
            "path": _rel(gain_trigger_path),
            "file_sha256": _sha(gain_trigger_path),
            "manifest_sha256": trigger["manifest_sha256"],
            "selection_is_closed": True,
        },
        "coordinate_manifests": {
            "dt": _coordinate_ref(manifest, "dt"),
            "dt2": dt2_coordinate_ref,
        },
        "coordinate_producer_file_sha256": coordinate[
            "producer_file_sha256"
        ],
        "selected_windows": windows,
        "selected_cells": selected_cells,
        "expected_base_arms": expected_arms,
        "claim_boundary": (
            "homologous source-space resolution confirmation only; no entry, "
            "offset, recovery, actuator, observation match, or lifecycle claim"
        ),
    }
    payload["manifest_sha256"] = _object_sha(payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phasec-manifest", default=PHASEC_MANIFEST_PATH)
    parser.add_argument("--native-summary", default=NATIVE_SUMMARY_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--confirm-lock", action="store_true")
    args = parser.parse_args()
    if not args.confirm_lock:
        raise SystemExit("write-once dt2 lock requires --confirm-lock")
    payload = build_payload(
        phasec_path=args.phasec_manifest,
        native_summary_path=args.native_summary,
    )
    _publish_once(args.output, payload)
    print(json.dumps({
        "path": _rel(args.output),
        "manifest_sha256": payload["manifest_sha256"],
        "n_selected_cells": len(payload["selected_cells"]),
        "n_expected_arms": len(payload["expected_base_arms"]),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
