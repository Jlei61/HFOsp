#!/usr/bin/env python3
"""Atomic baseline-only SNN cells for Phase-D conductance calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_topic4_zm_branch_decision as R  # noqa: E402
from src import topic4_zm_anchor_states as AS  # noqa: E402
from src import topic4_zm_checkpoint as CK  # noqa: E402
from src import topic4_zm_fast_carrier_calibration as CAL  # noqa: E402
from src import topic4_zm_fast_carrier_contract as C  # noqa: E402
from src import topic4_zm_fast_carrier_runtime as RT  # noqa: E402
from src import topic4_zm_fast_carrier_state as ST  # noqa: E402
from src import topic4_zm_noise_bank as NB  # noqa: E402


INPUT = ROOT / "results/topic4_sef_hfo/zm_fast_carrier_repair/phaseD_input_manifest_v1_5.json"
AMENDMENT = ROOT / "docs/superpowers/specs/2026-07-31-topic4-zm-fast-carrier-baseline-anchor-amendment.md"
OUT = ROOT / "results/topic4_sef_hfo/zm_fast_carrier_repair/calibration"
PRODUCTION_T_MS = 8500.0
REPLICATES = ("noise_replay", "noise_resample_1")


def _canonical_sha(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _write_npz(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    if path.exists():
        tmp.unlink()
        raise RuntimeError(f"refusing to overwrite existing array artifact: {path}")
    os.replace(tmp, path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to overwrite different receipt: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _core_mask(pos, center, radius):
    delta = np.asarray(pos, float) - np.asarray(center, float)
    return np.sum(delta * delta, axis=1) <= float(radius) ** 2


def _binned_rate(spikes, mask, dt_ms, bin_ms):
    bs = int(round(bin_ms / dt_ms))
    nb = spikes.shape[0] // bs
    count = spikes[: nb * bs, mask].reshape(nb, bs, -1).sum(axis=(1, 2))
    return count / max(1, int(np.sum(mask))) / (bs * dt_ms * 1e-3)


def _binned_lfp(trace, dt_ms, bin_ms):
    bs = int(round(bin_ms / dt_ms))
    nb = trace.shape[0] // bs
    return np.mean(np.abs(trace[: nb * bs]).reshape(nb, bs, trace.shape[1]), axis=1)


def _run(args) -> None:
    manifest = json.loads(INPUT.read_text())
    C.validate_input_manifest(manifest, ROOT)
    if args.replicate not in REPLICATES:
        raise RuntimeError(f"unregistered calibration replicate: {args.replicate}")
    if not args.smoke and float(args.T_ms) != PRODUCTION_T_MS:
        raise RuntimeError(f"production calibration duration must be {PRODUCTION_T_MS} ms")
    if args.mode == "reference" and any(
        value is not None for value in (args.scale_E, args.scale_I, args.scale_M)
    ):
        raise RuntimeError("reference arm cannot receive conductance scales")
    if args.mode == "cell":
        scales = (args.scale_E, args.scale_I, args.scale_M)
        if None in scales or tuple(float(x) for x in scales) not in CAL.scale_lattice():
            raise RuntimeError("cell requires one registered scale triplet")
    else:
        scales = None

    ctx = RT.build_source_locked_context(ROOT, manifest, R)
    state, transformation = ST.load_and_migrate(
        ROOT,
        manifest,
        row_id=("pre_entry", "natural"),
        contract_already_validated=True,
    )
    reference = CAL.build_reference_anchor(
        state,
        n_e=ctx["S"]["NE"],
        v_th_median=float(np.median(ctx["S"]["vth"][: ctx["S"]["NE"]])),
        v_reset=float(ctx["S"]["p"].V_reset),
        eta_m=float(R.ZM.ETA_M),
    )
    conductance = None
    if args.mode == "cell":
        conductance = CAL.candidate_config(reference, scales)
        conductance["gamma"] = 1.0 / 6.0
    slow, diagnostic = RT.make_dynamic_diagnostic_slow(
        ctx, R, conductance_config=conductance
    )
    bank = NB.build_noise_bank(
        manifest["source"]["canonical_config_sha"],
        manifest["source"]["seed"],
        0,
        args.replicate,
    )
    controller = CK.ZMCheckpoint(
        ext_mean_only=bank["ext_mean_only"],
        dump_ext=True,
        rng_state=bank["rng_state"],
    )
    result = R.run_segment(
        ctx,
        slow,
        float(args.T_ms),
        ckpt=controller,
        fresh_rng=True,
        dump_i_spikes=False,
    )
    metrics = R.MC.source_metrics(
        result["E_spk_bool"],
        ctx["core"],
        ctx["S"]["posE"],
        ctx["S"]["L"],
        ctx["dt"],
        lfp_trace=result["lfp_trace"],
        axis_coord=ctx["axis"],
    )
    source_mask = _core_mask(ctx["S"]["posE"], ctx["S"]["src_xy"], R.PP.CORE_R)
    sink_mask = _core_mask(ctx["S"]["posE"], ctx["S"]["snk_xy"], R.PP.CORE_R)
    r_source = _binned_rate(result["E_spk_bool"], source_mask, ctx["dt"], R.MC.BIN_MS)
    r_sink = _binned_rate(result["E_spk_bool"], sink_mask, ctx["dt"], R.MC.BIN_MS)
    lfp_binned = _binned_lfp(result["lfp_trace"], ctx["dt"], R.MC.BIN_MS)
    events = AS.returning_event_stats(metrics["r_core"], R.MC.BIN_MS, metrics["n_bins"])
    diag = diagnostic.diagnostic_summary()
    arrays = {
        "r_core": np.asarray(metrics["r_core"], np.float32),
        "r_source": np.asarray(r_source, np.float32),
        "r_sink": np.asarray(r_sink, np.float32),
        "r_surround": np.asarray(metrics["r_surround"], np.float32),
        "r_all": np.asarray(metrics["r_all"], np.float32),
        "active_fraction": np.asarray(metrics["A_active"], np.float32),
        "spatial_entropy": np.asarray(metrics["H_spatial"], np.float32),
        "kymo_axial": np.asarray(metrics["kymo_axial"], np.float32),
        "lfp_abs_binned": np.asarray(lfp_binned, np.float32),
        "vinf_median_trace": np.asarray(diagnostic.trace_vinf_median, np.float32),
        "tau_eff_median_trace": np.asarray(diagnostic.trace_tau_eff_median, np.float32),
        "exc_charge_mean_trace": np.asarray(diagnostic.trace_exc_charge_mean, np.float32),
        "inh_charge_mean_trace": np.asarray(diagnostic.trace_inh_charge_mean, np.float32),
    }
    stem = (
        f"reference__{args.replicate}"
        if args.mode == "reference"
        else f"sE{args.scale_E:g}_sI{args.scale_I:g}_sM{args.scale_M:g}__{args.replicate}"
    )
    root = OUT / ("smoke_dynamic" if args.smoke else "dynamic_preentry")
    npz_path = root / f"{stem}.npz"
    json_path = root / f"{stem}.json"
    _write_npz(npz_path, arrays)
    body = {
        "schema": "zm_fast_carrier_calibration_cell_v1_2026-07-31",
        "mode": args.mode,
        "data_scope": "dynamic_preentry_t0_to_8500ms_only",
        "input_manifest_sha256": manifest["manifest_sha256"],
        "input_file_sha256": _sha(INPUT),
        "amendment_file_sha256": _sha(AMENDMENT),
        "runtime_git_sha": RT.git_sha(ROOT),
        "source_state_hash": transformation["source_state_hash"],
        "migrated_state_hash": transformation["migrated_state_hash"],
        "source_state_role": "coefficient_anchor_only_not_initial_state",
        "simulation_initial_state": "canonical_t0_z1_m0_SG0",
        "reference_anchor": reference,
        "scales": None if scales is None else list(scales),
        "conductance_config": conductance,
        "replicate": args.replicate,
        "noise_bank_sha256": bank["bank_sha"],
        "T_ms": float(args.T_ms),
        "dt_ms": float(ctx["dt"]),
        "runaway_early_stop_ms": result.get("runaway_early_stop_ms"),
        "n_steps_observed": int(result["E_spk_bool"].shape[0]),
        "total_e_spikes": int(np.sum(result["E_spk_bool"], dtype=np.int64)),
        "median_e_rate_hz": float(np.median(metrics["r_all"])),
        "returning_events": events,
        "peak_active_fraction": float(np.max(metrics["A_active"])),
        "diagnostics": diag,
        "external_drive_sha256": _canonical_sha(
            {
                "nu": hashlib.sha256(np.ascontiguousarray(result["zm_ext_nu"]).tobytes()).hexdigest(),
                "ext": hashlib.sha256(np.ascontiguousarray(result["zm_ext_sum"]).tobytes()).hexdigest(),
            }
        ),
        "array_path": str(npz_path.relative_to(ROOT)),
        "array_file_sha256": _sha(npz_path),
        "candidate_outcomes_accessed": False,
        "production_authorized": False,
    }
    body = _json_safe(body)
    payload = {**body, "manifest_sha256": _canonical_sha(body)}
    _write_json(json_path, payload)
    print(json_path)
    print(payload["manifest_sha256"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("reference", "cell"), required=True)
    parser.add_argument("--replicate", required=True)
    parser.add_argument("--scale-E", type=float)
    parser.add_argument("--scale-I", type=float)
    parser.add_argument("--scale-M", type=float)
    parser.add_argument("--T-ms", type=float, default=PRODUCTION_T_MS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("refusing SNN calibration without --confirm-run")
    _run(args)


if __name__ == "__main__":
    main()
