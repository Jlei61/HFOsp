#!/usr/bin/env python3
"""Lock the conditional C1 gain cells after the complete base atlas exists.

This is a selection/manifest step only.  It never launches a simulation.  The
write-once manifest freezes every spike-AI-screen-positive cell, its six base
evidence parts, all 30 expected carrier-gain arms, and the SHA-matched C0
pre-entry denominator parts before any C1 gain result is inspected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import scripts.analyze_topic4_zm_phasec0 as C0  # noqa: E402
import scripts.analyze_topic4_zm_phasec1 as C1  # noqa: E402
import src.topic4_zm_phasec_neighbourhood as N  # noqa: E402


DELTAS_MV = (-0.10, -0.05, 0.0, 0.05, 0.10)


def _arm_label(delta):
    delta = float(delta)
    if delta == 0:
        return "d0_zero"
    return f"d{abs(delta):g}_{'plus' if delta > 0 else 'minus'}"


def _expected_gain_path(resolution, seed, tier, cell_id, phase, noise, delta):
    return (
        C1.OUT / "parts/c1_conditional_gain" / resolution / f"seed{seed}"
        / tier / cell_id / phase / noise / _arm_label(delta) / "gain.json"
    )


def _preentry_denominator_refs(resolution, seed):
    phasec = C1._load_json(C1.PHASEC_MANIFEST)
    phasec_file_sha = C1._sha256(C1.PHASEC_MANIFEST)
    refs = []
    for noise in C1.NOISES:
        locked = C1._resolution_preentry_inputs(
            phasec, resolution=resolution, seed=seed, noise=noise
        )
        for delta in DELTAS_MV:
            if delta == 0:
                path = Path(C0._gain_path(
                    resolution, seed, "pre_entry__natural", noise, 0.0, 0
                ))
            else:
                path = Path(C0._gain_path(
                    resolution,
                    seed,
                    "pre_entry__natural",
                    noise,
                    abs(delta),
                    1 if delta > 0 else -1,
                ))
            if not path.is_file():
                raise RuntimeError(
                    "required C0 pre-entry gain denominator is missing: "
                    f"{path.relative_to(ROOT)}"
                )
            refs.append({
                "schema": "zm_phasec_gain_cell_v1",
                "phasec_manifest_sha256": phasec["manifest_sha256"],
                "phasec_manifest_file_sha256": phasec_file_sha,
                "phasec_producer_file_sha256": phasec["provenance"][
                    "producer_file_sha256"
                ],
                "resolution": resolution,
                "seed": int(seed),
                "state_tag": "pre_entry__natural",
                "noise": noise,
                "replicate": noise,
                "delta_mV": float(delta),
                "threshold_offset_mV": float(delta),
                "signed_delta_abs_mV": abs(float(delta)),
                "sign": int((delta > 0) - (delta < 0)),
                "burn_in_ms": 500.0,
                "measure_ms": 1000.0,
                **locked,
                "path": str(path.relative_to(ROOT)),
                "file_sha256": C1._sha256(path),
            })
    return refs


def _validate_complete_base_inventory(base_atlas):
    by_seed_tier = {}
    duplicate = False
    for cell in base_atlas.get("cells", []):
        key = (int(cell["seed"]), cell["tier"])
        cell_id = cell["cell_id"]
        current = by_seed_tier.setdefault(key, [])
        duplicate = duplicate or cell_id in current
        current.append(cell_id)
        if cell.get("status") == "blocked":
            raise RuntimeError(
                "cannot lock gain triggers with a technically blocked cell: "
                f"{cell['seed']}/{cell_id}"
            )
        if cell.get("status") != "invalid_physical":
            runs = cell.get("run_rows")
            expected_runs = {
                (phase, noise)
                for phase in C1.PHASES for noise in C1.NOISES
            }
            if (
                not isinstance(runs, list)
                or len(runs) != 6
                or {
                    (row.get("phase"), row.get("noise")) for row in runs
                } != expected_runs
                or any(row.get("status") != "complete" for row in runs)
            ):
                raise RuntimeError(
                    "cannot lock gain triggers before every valid cell has "
                    f"terminal 2x3 base evidence: {cell['seed']}/{cell_id}"
                )
    if duplicate:
        raise RuntimeError("duplicate cells in C1 base atlas")
    for seed in C1.SEEDS:
        expected = {
            "primary_convex": set(N.PRIMARY_CELL_NAMES),
            "secondary_shell": set(N.SHELL_CELL_NAMES),
        }
        for tier, wanted in expected.items():
            got = set(by_seed_tier.get((seed, tier), []))
            if got != wanted:
                raise RuntimeError(
                    "cannot lock gain triggers before complete base coverage: "
                    f"seed{seed}/{tier} missing={sorted(wanted-got)} "
                    f"extra={sorted(got-wanted)}"
                )


def _is_locked_trigger_candidate(cell):
    support = cell.get("spike_ai_screen_support")
    if not isinstance(support, dict):
        return False
    phase_counts = support.get("per_phase_pass_count")
    return bool(
        support.get("passes_locked_cell_gate") is True
        and int(support.get("k", -1)) >= 5
        and int(support.get("n", -1)) == 6
        and float(support.get("posterior_median", -1.0)) > 0.80
        and isinstance(phase_counts, dict)
        and all(int(phase_counts.get(phase, -1)) >= 2 for phase in C1.PHASES)
        and cell.get("cell_class") == "spike_AI_screen_candidate"
    )


def _validate_denominator_refs(refs):
    expected = {
        (noise, float(delta))
        for noise in C1.NOISES for delta in DELTAS_MV
    }
    got = {
        (row.get("noise"), float(row.get("delta_mV")))
        for row in refs
    }
    if len(refs) != 15 or got != expected:
        raise RuntimeError("C0 pre-entry denominator coverage is not 3x5")
    for row in refs:
        if (
            not isinstance(row.get("path"), str)
            or not isinstance(row.get("file_sha256"), str)
            or len(row["file_sha256"]) != 64
        ):
            raise RuntimeError("invalid C0 pre-entry denominator provenance")
        required_identity = (
            "schema", "phasec_manifest_sha256",
            "phasec_manifest_file_sha256",
            "phasec_producer_file_sha256", "resolution", "seed",
            "state_tag", "replicate", "threshold_offset_mV",
            "signed_delta_abs_mV", "sign", "burn_in_ms", "measure_ms",
            "config_sha", "fast_base_state_hash", "state_file_sha256",
            "noise_bank_sha",
        )
        if any(key not in row for key in required_identity):
            raise RuntimeError(
                "C0 pre-entry denominator identity lock is incomplete"
            )


def build_trigger_manifest(
    base_atlas,
    *,
    base_atlas_path,
    denominator_provider=_preentry_denominator_refs,
):
    if base_atlas.get("schema") != C1.C1_BASE_ATLAS_SCHEMA:
        raise ValueError("unexpected C1 base-atlas schema")
    if base_atlas.get("matrix", {}).get("complete") is not True:
        raise RuntimeError(
            "cannot lock conditional gain before the complete C1 base atlas"
        )
    _validate_complete_base_inventory(base_atlas)
    resolution = base_atlas["resolution"]
    triggered = []
    denominator_cache = {}
    for cell in sorted(
        (
            row for row in base_atlas["cells"]
            if _is_locked_trigger_candidate(row)
        ),
        key=lambda row: (row["seed"], row["tier"], row["cell_id"]),
    ):
        evidence = [
            {
                "phase": row["phase"],
                "noise": row["noise"],
                "part_path": row["part_path"],
                "part_sha256": row["part_sha256"],
                "locked_arm_identity": row["locked_arm_identity"],
            }
            for row in cell["run_rows"]
        ]
        if (
            len(evidence) != len(C1.PHASES) * len(C1.NOISES)
            or any(
                row["part_sha256"] is None or len(row["part_sha256"]) != 64
                for row in evidence
            )
            or any(
                not isinstance(row.get("locked_arm_identity"), dict)
                for row in evidence
            )
        ):
            raise RuntimeError(
                f"trigger cell lacks complete hashed evidence: {cell['cell_id']}"
            )
        seed = int(cell["seed"])
        if seed not in denominator_cache:
            denominator_cache[seed] = denominator_provider(resolution, seed)
            _validate_denominator_refs(denominator_cache[seed])
        expected_arms = []
        for phase in C1.PHASES:
            for noise in C1.NOISES:
                identity_rows = [
                    row["locked_arm_identity"]
                    for row in cell["run_rows"]
                    if row["phase"] == phase and row["noise"] == noise
                ]
                if len(identity_rows) != 1:
                    raise RuntimeError(
                        "trigger base evidence lacks unique phase/noise identity"
                    )
                base_identity = identity_rows[0]
                for delta in DELTAS_MV:
                    expected_arms.append({
                        "schema": (
                            "zm_phasec1_conditional_gain_part_v1_2026-07-28"
                        ),
                        "phasec_manifest_sha256": base_atlas[
                            "phasec_manifest_sha256"
                        ],
                        "phasec_manifest_file_sha256": base_atlas[
                            "phasec_manifest_file_sha256"
                        ],
                        "coordinate_manifest_sha256": base_atlas[
                            "coordinate_manifest_sha256"
                        ],
                        "coordinate_manifest_semantic_sha256": base_atlas[
                            "coordinate_manifest_semantic_sha256"
                        ],
                        "coordinate_manifest_file_sha256": base_atlas[
                            "coordinate_manifest_file_sha256"
                        ],
                        **base_atlas[
                            "coordinate_npz_provenance_by_seed"
                        ][str(seed)],
                        "trigger_parameter_contract": (
                            "source_core_E_threshold_offset_mV"
                        ),
                        "resolution": resolution,
                        "seed": seed,
                        "tier": cell["tier"],
                        "cell_id": cell["cell_id"],
                        "trajectory_id": cell["trajectory_id"],
                        "path_index": int(cell["path_index"]),
                        "path_direction": cell["path_direction"],
                        "phase": phase,
                        "noise": noise,
                        "delta_mV": float(delta),
                        "threshold_offset_mV": float(delta),
                        "burn_in_ms": 500.0,
                        "measure_ms": 1000.0,
                        "slow_state_sha256": cell[
                            "slow_state_sha256"
                        ],
                        "config_sha": base_identity["config_sha"],
                        "fast_base_state_hash": base_identity[
                            "fast_base_state_hash"
                        ],
                        "state_file_sha256": base_identity[
                            "state_file_sha256"
                        ],
                        "noise_bank_sha": base_identity[
                            "noise_bank_sha"
                        ],
                        "path": str(_expected_gain_path(
                            resolution,
                            seed,
                            cell["tier"],
                            cell["cell_id"],
                            phase,
                            noise,
                            delta,
                        ).relative_to(ROOT)),
                    })
        if len(expected_arms) != 30:
            raise AssertionError("conditional gain arm count drift")
        triggered.append({
            "seed": seed,
            "tier": cell["tier"],
            "cell_id": cell["cell_id"],
            "trajectory_id": cell["trajectory_id"],
            "path_index": int(cell["path_index"]),
            "path_direction": cell["path_direction"],
            "slow_state_sha256": cell["slow_state_sha256"],
            "trigger_rule": {
                "name": "spike_AI_screen_candidate",
                "required_successes": 5,
                "required_per_phase": 2,
                "jeffreys_posterior_median_gt": 0.80,
            },
            "trigger_support": cell["spike_ai_screen_support"],
            "triggering_base_parts": evidence,
            "expected_carrier_gain_arms": expected_arms,
            "reused_c0_preentry_denominators": denominator_cache[seed],
            "gain_status_path": str(C1.gain_status_path(
                resolution, seed, cell["tier"], cell["cell_id"]
            ).relative_to(ROOT)),
        })
    claimed = {
        (int(row["seed"]), row["tier"], row["cell_id"])
        for row in base_atlas["cells"]
        if row.get("gain_trigger_eligible") is True
    }
    selected = {
        (int(row["seed"]), row["tier"], row["cell_id"])
        for row in triggered
    }
    if claimed != selected:
        raise RuntimeError(
            "gain_trigger_eligible flag disagrees with locked spike-AI evidence"
        )

    payload = {
        "schema": C1.C1_GAIN_TRIGGER_SCHEMA,
        "base_atlas_path": str(Path(base_atlas_path).resolve().relative_to(ROOT)),
        "base_atlas_sha256": C1._sha256(base_atlas_path),
        "phasec_manifest_sha256": base_atlas["phasec_manifest_sha256"],
        "phasec_manifest_file_sha256": base_atlas[
            "phasec_manifest_file_sha256"
        ],
        "coordinate_manifest_sha256": base_atlas[
            "coordinate_manifest_sha256"
        ],
        "coordinate_manifest_semantic_sha256": base_atlas[
            "coordinate_manifest_semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": base_atlas[
            "coordinate_manifest_file_sha256"
        ],
        "coordinate_npz_provenance_by_seed": base_atlas[
            "coordinate_npz_provenance_by_seed"
        ],
        "phasec_producer_file_sha256": base_atlas[
            "phasec_producer_file_sha256"
        ],
        "coordinate_producer_file_sha256": base_atlas[
            "coordinate_producer_file_sha256"
        ],
        "producer_file_sha256": base_atlas[
            "phasec_producer_file_sha256"
        ],
        "resolution": resolution,
        "trigger_rule_version": (
            "spike_AI_screen_5of6_2perphase_then_full_gain_v1_2026-07-28"
        ),
        "delta_mV": list(DELTAS_MV),
        "triggered_cells": triggered,
        "n_triggered_cells": len(triggered),
        "selection_is_closed": True,
        "claim_boundary": (
            "conditional frozen-state gain routing only; not maturation, "
            "entry, offset, recovery, actuator efficacy, or lifecycle"
        ),
    }
    manifest = dict(payload)
    manifest["manifest_sha256"] = C1._object_sha(payload)
    return manifest


def lock(
    *,
    base_atlas_path=None,
    output_path=C1.GAIN_TRIGGER_MANIFEST,
    denominator_provider=_preentry_denominator_refs,
):
    base_atlas_path = Path(
        base_atlas_path
        or C1.OUT / "phasec1_base_atlas_dt.json"
    )
    base = C1._load_json(base_atlas_path)
    manifest = build_trigger_manifest(
        base,
        base_atlas_path=base_atlas_path,
        denominator_provider=denominator_provider,
    )
    status = N.write_json_once(output_path, manifest)
    return manifest, status


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
    if args.check_only:
        status = "validated_not_written"
    else:
        status = N.write_json_once(Path(args.output), manifest)
    print(json.dumps({
        "status": status,
        "n_triggered_cells": manifest["n_triggered_cells"],
        "manifest_sha256": manifest["manifest_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
