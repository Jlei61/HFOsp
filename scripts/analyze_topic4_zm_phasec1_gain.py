#!/usr/bin/env python3
"""Adjudicate locked Phase-C1 conditional gain from raw carrier/C0 parts.

The trigger manifest is the complete selection universe.  Missing or
provenance-invalid parts are technical blockers and no status file is written.
Nonlinear, sign-inconsistent, plateau/runaway, or denominator-degenerate
responses are scientific ``tonic_gain_indeterminate`` outcomes, never zero
gain.  This script does not run the SNN.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.analyze_topic4_zm_phasec1 as C1  # noqa: E402
import scripts.run_topic4_zm_phasec_cell as CELL  # noqa: E402
import src.topic4_zm_phasec_metrics as PCM  # noqa: E402
import src.topic4_zm_phasec_neighbourhood as N  # noqa: E402


DELTAS = (-0.10, -0.05, 0.0, 0.05, 0.10)
N_BOOT = 5000


def _load(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _rates(row):
    value = np.asarray(row.get("core_rate_500ms_hz", []), float)
    if value.shape != (2,) or not np.all(np.isfinite(value)):
        raise RuntimeError("gain part lacks two finite 500-ms block rates")
    return value


def gain_curve(rows_by_delta):
    """Return one continuation's exact two-amplitude gain evidence."""
    if set(float(key) for key in rows_by_delta) != set(DELTAS):
        raise RuntimeError("gain curve is not the locked five-arm set")
    rows = {float(key): value for key, value in rows_by_delta.items()}
    if any(row.get("status") == "technical_invalid" for row in rows.values()):
        raise RuntimeError("conditional-gain raw part is technically invalid")
    bank_shas = {row.get("noise_bank_sha") for row in rows.values()}
    if len(bank_shas) != 1 or None in bank_shas:
        raise RuntimeError("gain arms do not share one paired future-noise bank")
    scientific = [
        row.get("scientific_end_reason")
        for row in rows.values() if row.get("status") == "scientific_failure"
    ]
    if scientific:
        return {
            "status": "scientific_indeterminate",
            "reason": "gain_arm_scientific_failure:" + ",".join(
                sorted(set(map(str, scientific)))
            ),
            "linearity_pass": False,
        }
    if any(row.get("status") != "complete" for row in rows.values()):
        raise RuntimeError("conditional-gain raw part is nonterminal")
    if any(row.get("gain_plateau_gate_pass") is not True for row in rows.values()):
        return {
            "status": "scientific_indeterminate",
            "reason": "gain_plateau_or_runaway",
            "linearity_pass": False,
        }
    block = {delta: _rates(row) for delta, row in rows.items()}
    points = []
    slopes = []
    for delta in (0.05, 0.10):
        points.append({
            "delta_mV": delta,
            "rate_vth_minus_hz": float(np.mean(block[-delta])),
            "rate_vth_plus_hz": float(np.mean(block[delta])),
            "rate_baseline_hz": float(np.mean(block[0.0])),
        })
        slopes.append((block[-delta] - block[delta]) / (2.0 * delta))
    result = PCM.paired_local_gain(points)
    if not result.get("linearity_pass"):
        return {
            **result,
            "status": "scientific_indeterminate",
            "reason": "nonlinear_or_sign_inconsistent_gain",
        }
    result.update({
        "status": "ok",
        "gain_hz_per_mV_blocks": np.median(
            np.vstack(slopes), axis=0
        ).tolist(),
    })
    return result


def _validate_carrier_part(
    part,
    expected_arm,
    trigger,
    *,
    trigger_file_sha256,
    coordinate_seed_provenance,
):
    fields = {
        "schema": CELL.C1_GAIN_PART_SCHEMA,
        "trigger_manifest_sha256": trigger["manifest_sha256"],
        "trigger_manifest_file_sha256": trigger_file_sha256,
        "phasec_manifest_sha256": trigger["phasec_manifest_sha256"],
        "phasec_manifest_file_sha256": trigger[
            "phasec_manifest_file_sha256"
        ],
        "coordinate_manifest_sha256": trigger[
            "coordinate_manifest_sha256"
        ],
        "coordinate_manifest_semantic_sha256": trigger[
            "coordinate_manifest_semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": trigger[
            "coordinate_manifest_file_sha256"
        ],
        "coordinate_npz_file_sha256": coordinate_seed_provenance[
            "coordinate_npz_file_sha256"
        ],
        "coordinate_npz_semantic_sha256": coordinate_seed_provenance[
            "coordinate_npz_semantic_sha256"
        ],
    }
    locked_parent = {
        "phasec_manifest_sha256": trigger["phasec_manifest_sha256"],
        "phasec_manifest_file_sha256": trigger[
            "phasec_manifest_file_sha256"
        ],
        "coordinate_manifest_sha256": trigger[
            "coordinate_manifest_sha256"
        ],
        "coordinate_manifest_semantic_sha256": trigger[
            "coordinate_manifest_semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": trigger[
            "coordinate_manifest_file_sha256"
        ],
        **coordinate_seed_provenance,
    }
    trigger_drift = [
        key for key, wanted in locked_parent.items()
        if expected_arm.get(key) != wanted
    ]
    if trigger_drift:
        raise RuntimeError(
            "trigger carrier-arm provenance mismatch:"
            + ",".join(trigger_drift)
        )
    for key in (
        "resolution", "seed", "tier", "cell_id", "trajectory_id",
        "path_index", "path_direction", "phase", "noise", "delta_mV",
        "threshold_offset_mV", "burn_in_ms", "measure_ms",
        "slow_state_sha256", "config_sha", "fast_base_state_hash",
        "state_file_sha256", "noise_bank_sha",
    ):
        fields[key] = expected_arm[key]
    mismatches = [key for key, wanted in fields.items()
                  if part.get(key) != wanted]
    if mismatches:
        raise RuntimeError(
            "conditional-gain raw provenance mismatch:"
            + ",".join(mismatches)
        )
    expected_runtime = {
        "manifest_sha256": trigger["phasec_manifest_sha256"],
        "manifest_file_sha256": trigger[
            "phasec_manifest_file_sha256"
        ],
        "producer_sha256": trigger["phasec_producer_file_sha256"],
        "state_file_sha256": expected_arm["state_file_sha256"],
        "noise_bank_sha": expected_arm["noise_bank_sha"],
        "coordinate_manifest_sha256": trigger[
            "coordinate_manifest_sha256"
        ],
        "coordinate_manifest_semantic_sha256": trigger[
            "coordinate_manifest_semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": trigger[
            "coordinate_manifest_file_sha256"
        ],
        **coordinate_seed_provenance,
        "coordinate_producer_sha256": trigger[
            "coordinate_producer_file_sha256"
        ],
        "trigger_manifest_sha256": trigger["manifest_sha256"],
        "trigger_manifest_file_sha256": trigger_file_sha256,
        "trigger_producer_sha256": trigger["producer_file_sha256"],
    }
    provenance = part.get("runtime_provenance")
    mismatches = [
        key for key, wanted in expected_runtime.items()
        if not isinstance(provenance, dict) or provenance.get(key) != wanted
    ]
    if mismatches:
        raise RuntimeError(
            "conditional-gain runtime provenance mismatch:"
            + ",".join(mismatches)
        )


def _carrier_curves(
    trigger,
    selected,
    *,
    trigger_file_sha256,
    coordinate_seed_provenance,
):
    expected = {
        (row["phase"], row["noise"], float(row["delta_mV"])): row
        for row in selected["expected_carrier_gain_arms"]
    }
    if len(expected) != 30:
        raise RuntimeError("trigger does not contain exactly 30 carrier arms")
    completed = {}
    curves = {}
    for phase in C1.PHASES:
        for noise in C1.NOISES:
            rows = {}
            for delta in DELTAS:
                ref = expected.get((phase, noise, delta))
                if ref is None:
                    raise RuntimeError("trigger carrier arm matrix is incomplete")
                path = ROOT / ref["path"]
                if not path.is_file():
                    raise RuntimeError(f"missing conditional-gain part: {path}")
                part = _load(path)
                _validate_carrier_part(
                    part,
                    ref,
                    trigger,
                    trigger_file_sha256=trigger_file_sha256,
                    coordinate_seed_provenance=(
                        coordinate_seed_provenance
                    ),
                )
                completed[ref["path"]] = C1._sha256(path)
                rows[delta] = part
            curves[(phase, noise)] = gain_curve(rows)
    return curves, completed


def _denominator_curves(selected, trigger):
    refs = {
        (row["noise"], float(row["delta_mV"])): row
        for row in selected["reused_c0_preentry_denominators"]
    }
    if len(refs) != 15:
        raise RuntimeError("trigger does not contain 15 C0 denominators")
    curves, completed = {}, {}
    for noise in C1.NOISES:
        rows = {}
        for delta in DELTAS:
            ref = refs.get((noise, delta))
            if ref is None:
                raise RuntimeError("C0 denominator matrix is incomplete")
            locked_parent = {
                "phasec_manifest_sha256": trigger[
                    "phasec_manifest_sha256"
                ],
                "phasec_manifest_file_sha256": trigger[
                    "phasec_manifest_file_sha256"
                ],
                "phasec_producer_file_sha256": trigger[
                    "phasec_producer_file_sha256"
                ],
                "resolution": trigger["resolution"],
                "seed": int(selected["seed"]),
                "state_tag": "pre_entry__natural",
                "replicate": noise,
                "threshold_offset_mV": float(delta),
            }
            drift = [
                key for key, wanted in locked_parent.items()
                if ref.get(key) != wanted
            ]
            if drift:
                raise RuntimeError(
                    "C0 denominator trigger provenance mismatch:"
                    + ",".join(drift)
                )
            path = ROOT / ref["path"]
            if not path.is_file() or C1._sha256(path) != ref["file_sha256"]:
                raise RuntimeError(f"C0 denominator provenance drift: {path}")
            part = _load(path)
            expected = {
                "schema": ref["schema"],
                "manifest_sha256": ref["phasec_manifest_sha256"],
                "manifest_file_sha256": ref[
                    "phasec_manifest_file_sha256"
                ],
                "resolution": ref["resolution"],
                "seed": int(ref["seed"]),
                "state_tag": ref["state_tag"],
                "replicate": ref["replicate"],
                "delta_mV": float(ref["signed_delta_abs_mV"]),
                "sign": int(ref["sign"]),
                "threshold_offset_mV": float(ref["threshold_offset_mV"]),
                "burn_in_ms": float(ref["burn_in_ms"]),
                "measure_ms": float(ref["measure_ms"]),
                "config_sha": ref["config_sha"],
                "state_hash": ref["fast_base_state_hash"],
                "state_file_sha256": ref["state_file_sha256"],
                "noise_bank_sha": ref["noise_bank_sha"],
            }
            def _matches(got, wanted):
                if isinstance(wanted, float):
                    try:
                        return bool(np.isclose(float(got), wanted))
                    except (TypeError, ValueError):
                        return False
                return got == wanted
            mismatches = [
                key for key, wanted in expected.items()
                if not _matches(part.get(key), wanted)
            ]
            runtime_expected = {
                "manifest_sha256": ref["phasec_manifest_sha256"],
                "manifest_file_sha256": ref[
                    "phasec_manifest_file_sha256"
                ],
                "producer_sha256": ref[
                    "phasec_producer_file_sha256"
                ],
                "state_file_sha256": ref["state_file_sha256"],
                "noise_bank_sha": ref["noise_bank_sha"],
            }
            runtime = part.get("runtime_provenance")
            runtime_mismatch = [
                key for key, wanted in runtime_expected.items()
                if not isinstance(runtime, dict)
                or runtime.get(key) != wanted
            ]
            if (
                part.get("status") not in {"complete", "scientific_failure"}
                or mismatches
                or runtime_mismatch
            ):
                raise RuntimeError(
                    "C0 denominator identity/provenance mismatch:"
                    + ",".join([*mismatches, *runtime_mismatch])
                )
            rows[delta] = part
            completed[ref["path"]] = ref["file_sha256"]
        curves[noise] = gain_curve(rows)
    return curves, completed


def _ratio_rows(carrier, denominator):
    rows = []
    scientific_reasons = []
    for phase in C1.PHASES:
        for noise in C1.NOISES:
            numerator = carrier[(phase, noise)]
            denom = denominator[noise]
            if numerator["status"] != "ok" or denom["status"] != "ok":
                scientific_reasons.append(
                    f"{phase}/{noise}:{numerator.get('reason')}/"
                    f"{denom.get('reason')}"
                )
                continue
            n = np.asarray(numerator["gain_hz_per_mV_blocks"], float)
            d = np.asarray(denom["gain_hz_per_mV_blocks"], float)
            if n.shape != (2,) or d.shape != (2,):
                raise RuntimeError("gain block ratio shape mismatch")
            if not np.all(np.isfinite(n)) or not np.all(np.isfinite(d)):
                raise RuntimeError("gain block ratio contains non-finite values")
            if np.any(d <= 0):
                scientific_reasons.append(
                    f"{phase}/{noise}:invalid_preentry_denominator"
                )
                continue
            rows.append({
                "phase": phase,
                "noise": noise,
                "ratio_blocks": n / d,
                "ratio_point": float(np.median(n / d)),
            })
    return rows, scientific_reasons


def gain_ratio_interval(rows, *, seed, n_boot=N_BOOT):
    if len(rows) != 6:
        raise ValueError("complete C1 gain interval requires six continuations")
    by_phase = {
        phase: [row for row in rows if row["phase"] == phase]
        for phase in C1.PHASES
    }
    if any(len(by_phase[phase]) != 3 for phase in C1.PHASES):
        raise ValueError("gain interval phase strata must each contain three noises")
    point = float(np.median([row["ratio_point"] for row in rows]))
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_boot), float)
    for draw in range(int(n_boot)):
        selected = []
        for phase in C1.PHASES:
            phase_rows = by_phase[phase]
            for index in rng.integers(0, len(phase_rows), size=3):
                row = phase_rows[int(index)]
                blocks = np.asarray(row["ratio_blocks"], float)
                selected.extend(
                    blocks[rng.integers(0, blocks.size, size=blocks.size)]
                )
        draws[draw] = float(np.median(selected))
    return {
        "point": point,
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "n_boot": int(n_boot),
        "structure": (
            "500ms_gain_blocks_then_three_noises_within_each_fast_phase"
        ),
    }


def analyze_triggered_cell(
    trigger,
    selected,
    *,
    trigger_file_sha256,
    coordinate_seed_provenance,
):
    carrier, carrier_files = _carrier_curves(
        trigger,
        selected,
        trigger_file_sha256=trigger_file_sha256,
        coordinate_seed_provenance=coordinate_seed_provenance,
    )
    denominator, denominator_files = _denominator_curves(selected, trigger)
    ratio_rows, scientific = _ratio_rows(carrier, denominator)
    if scientific:
        gain_class = "tonic_gain_indeterminate"
        interval = None
        reason = "scientific_gain_unresolved"
    else:
        interval = gain_ratio_interval(ratio_rows, seed=int(selected["seed"]))
        per_phase_passes = {
            phase: sum(
                row["phase"] == phase and row["ratio_point"] >= 0.50
                for row in ratio_rows
            )
            for phase in C1.PHASES
        }
        n_pass = sum(row["ratio_point"] >= 0.50 for row in ratio_rows)
        if (
            interval["lo"] >= 0.50
            and n_pass >= 5
            and all(per_phase_passes[phase] >= 2 for phase in C1.PHASES)
        ):
            gain_class, reason = "balanced_AI_tonic_cell", "gain_AI_supported"
        elif interval["hi"] < 0.50:
            gain_class, reason = "tonic_non_AI", "gain_AI_rejected"
        else:
            gain_class, reason = (
                "tonic_gain_indeterminate", "gain_ratio_CI_crosses_0p50"
            )
    return {
        "schema": C1.C1_GAIN_STATUS_SCHEMA,
        "trigger_manifest_sha256": trigger["manifest_sha256"],
        "trigger_manifest_file_sha256": trigger_file_sha256,
        "phasec_manifest_sha256": trigger["phasec_manifest_sha256"],
        "phasec_manifest_file_sha256": trigger[
            "phasec_manifest_file_sha256"
        ],
        "coordinate_manifest_sha256": trigger[
            "coordinate_manifest_sha256"
        ],
        "coordinate_manifest_semantic_sha256": trigger[
            "coordinate_manifest_semantic_sha256"
        ],
        "coordinate_manifest_file_sha256": trigger[
            "coordinate_manifest_file_sha256"
        ],
        "resolution": trigger["resolution"],
        "seed": int(selected["seed"]),
        "tier": selected["tier"],
        "cell_id": selected["cell_id"],
        "slow_state_sha256": selected["slow_state_sha256"],
        "coordinate_npz_provenance": coordinate_seed_provenance,
        "phasec_producer_file_sha256": trigger[
            "phasec_producer_file_sha256"
        ],
        "coordinate_producer_file_sha256": trigger[
            "coordinate_producer_file_sha256"
        ],
        "trigger_producer_file_sha256": trigger[
            "producer_file_sha256"
        ],
        "gain_class": gain_class,
        "reason": reason,
        "gain_ratio_interval": interval,
        "scientific_indeterminate_reasons": scientific,
        "carrier_curve_by_phase_noise": {
            f"{phase}|{noise}": carrier[(phase, noise)]
            for phase in C1.PHASES for noise in C1.NOISES
        },
        "preentry_curve_by_noise": denominator,
        "completed_arm_file_sha256": carrier_files,
        "reused_c0_preentry_denominator_sha256": denominator_files,
        "claim_boundary": (
            "conditional frozen-state gain only; not maturation, entry, "
            "offset, recovery, actuator efficacy, or lifecycle"
        ),
    }


def analyze_all(trigger_path=C1.GAIN_TRIGGER_MANIFEST, *, write=True):
    trigger = _load(trigger_path)
    C1._validate_self_hash(trigger, label="C1 gain trigger manifest")
    phasec = _load(C1.PHASEC_MANIFEST)
    phasec_path_sha = C1._sha256(C1.PHASEC_MANIFEST)
    if (
        phasec.get("manifest_sha256")
        != trigger.get("phasec_manifest_sha256")
        or phasec_path_sha
        != trigger.get("phasec_manifest_file_sha256")
        or phasec.get("provenance", {}).get("producer_file_sha256")
        != trigger.get("phasec_producer_file_sha256")
    ):
        raise RuntimeError("trigger/final Phase-C provenance mismatch")
    coordinate_path, coordinate, coordinate_ref = (
        C1._coordinate_path_from_final(phasec, trigger["resolution"])
    )
    C1._validate_self_hash(coordinate, label="C1 coordinate manifest")
    if (
        C1._sha256(coordinate_path)
        != trigger["coordinate_manifest_file_sha256"]
        or coordinate["manifest_sha256"]
        != trigger["coordinate_manifest_sha256"]
        or coordinate["semantic_sha256"]
        != trigger["coordinate_manifest_semantic_sha256"]
        or coordinate.get("producer_file_sha256")
        != trigger["coordinate_producer_file_sha256"]
    ):
        raise RuntimeError("trigger/coordinate manifest provenance mismatch")
    trigger_file_sha256 = C1._sha256(trigger_path)
    outputs = []
    for selected in trigger.get("triggered_cells", []):
        seed_row = coordinate["seeds"][str(int(selected["seed"]))]
        coordinate_seed = {
            "coordinate_npz_file_sha256": seed_row["npz_file_sha256"],
            "coordinate_npz_semantic_sha256": seed_row[
                "npz_semantic_sha256"
            ],
        }
        payload = analyze_triggered_cell(
            trigger,
            selected,
            trigger_file_sha256=trigger_file_sha256,
            coordinate_seed_provenance=coordinate_seed,
        )
        output = C1.gain_status_path(
            trigger["resolution"], int(selected["seed"]),
            selected["tier"], selected["cell_id"],
        )
        status = (
            N.write_json_once(output, payload)
            if write else "validated_not_written"
        )
        outputs.append({
            "path": str(output.relative_to(ROOT)),
            "status": status,
            "gain_class": payload["gain_class"],
        })
    return {
        "trigger_manifest_sha256": trigger["manifest_sha256"],
        "n_triggered": len(trigger.get("triggered_cells", [])),
        "outputs": outputs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trigger-manifest", default=str(C1.GAIN_TRIGGER_MANIFEST)
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    result = analyze_all(args.trigger_manifest, write=not args.check_only)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
