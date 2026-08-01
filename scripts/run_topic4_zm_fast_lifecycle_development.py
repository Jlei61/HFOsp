#!/usr/bin/env python3
"""Development-first phi screen on real Phase-C Z/M tonic checkpoints.

This is intentionally a thin scientific runner, not a new confirmation system.
Each invocation runs one seed-1 branch-intervention cell and atomically stores
the traces needed for the 24-cell phenotype matrix.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src" / "snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
for name in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(name, "1")

from scripts import run_topic4_zm_branch_decision as R  # noqa: E402
from src import topic4_zm_checkpoint as CK  # noqa: E402
from src import topic4_zm_fast_carrier_runtime as RT  # noqa: E402
from src import topic4_zm_fast_carrier_state as ST  # noqa: E402
from src import topic4_zm_ictal_carrier as IC  # noqa: E402
from src import topic4_zm_noise_bank as NB  # noqa: E402


INPUT = ROOT / "results/topic4_sef_hfo/zm_fast_carrier_repair/phaseD_input_manifest_v1_5.json"
FUTILITY = ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity/phasec_futility_verdict.json"
OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development"
STATES = (
    "bounded_mid__rising",
    "bounded_mid__peak",
    "bounded_late__rising",
    "bounded_late__peak",
)
TAUS_MS = (60.0, 100.0, 160.0)
FRACTIONS = (0.15, 0.30)
REFERENCE_RATE_HZ = 439.22905756378174
FINE_BIN_MS = 2.0
COARSE_BIN_MS = 25.0
PRODUCTION_T_MS = 6000.0
PRODUCTION_BURN_MS = 1000.0


def delta_phi_mV(
    tau_phi_ms: float,
    fraction: float,
    *,
    gap_mV: float = 6.5,
    rate_hz: float = REFERENCE_RATE_HZ,
) -> float:
    """mV jump/spike with the mandatory ms-to-s conversion."""
    tau_s = float(tau_phi_ms) / 1000.0
    if tau_s <= 0 or float(rate_hz) <= 0 or float(gap_mV) <= 0:
        raise ValueError("tau, rate and threshold gap must be positive")
    if not 0 < float(fraction) < 1:
        raise ValueError("fraction must lie strictly between zero and one")
    return float(fraction) * float(gap_mV) / (tau_s * float(rate_hz))


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json_once(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to overwrite different receipt: {path}")
        return
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _write_npz_once(path: Path, arrays: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp.npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        with open(tmp_name, "rb") as stream:
            os.fsync(stream.fileno())
        if path.exists():
            if _sha(path) != _sha(Path(tmp_name)):
                raise RuntimeError(f"refusing to overwrite different array artifact: {path}")
            return
        os.link(tmp_name, path)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _state_parts(state_tag: str) -> tuple[str, str]:
    if state_tag not in STATES:
        raise ValueError(f"unregistered development state: {state_tag}")
    return tuple(state_tag.split("__", 1))


def _source_row(manifest: dict, state_tag: str) -> dict:
    bin_name, fast_phase = _state_parts(state_tag)
    rows = [
        row for row in manifest["source_panel"]
        if row["bin_name"] == bin_name and row["fast_phase"] == fast_phase
    ]
    if len(rows) != 1:
        raise RuntimeError(f"state {state_tag} does not resolve exactly once")
    return rows[0]


def _population_rate(spikes: np.ndarray, dt_ms: float, bin_ms: float) -> np.ndarray:
    x = np.asarray(spikes, bool)
    bs = max(1, int(round(float(bin_ms) / float(dt_ms))))
    nb = x.shape[0] // bs
    if nb == 0:
        return np.zeros(0, dtype=np.float32)
    counts = x[: nb * bs].reshape(nb, bs, x.shape[1]).sum(axis=(1, 2))
    return (counts / x.shape[1] / (bs * dt_ms * 1e-3)).astype(np.float32)


def _core_saturation_fraction(
    spikes: np.ndarray,
    core_mask: np.ndarray,
    dt_ms: float,
    *,
    threshold_fraction: float = 0.8,
    tau_ref_ms: float = 2.0,
) -> float:
    x = np.asarray(spikes, bool)[:, np.asarray(core_mask, bool)]
    duration_s = x.shape[0] * float(dt_ms) * 1e-3
    rates = x.sum(axis=0) / max(duration_s, np.finfo(float).eps)
    active = rates > 0
    if not np.any(active):
        return 0.0
    ceiling_hz = 1000.0 / float(tau_ref_ms)
    return float(np.mean(rates[active] >= threshold_fraction * ceiling_hz))


def _modulation(trace: np.ndarray) -> dict:
    x = np.asarray(trace, float)
    if x.size == 0:
        return {"mean_hz": 0.0, "p05_hz": 0.0, "p95_hz": 0.0, "depth": 0.0, "cv": 0.0}
    mean = float(np.mean(x))
    p05, p95 = (float(v) for v in np.percentile(x, (5, 95)))
    return {
        "mean_hz": mean,
        "p05_hz": p05,
        "p95_hz": p95,
        "depth": float((p95 - p05) / max(mean, 1e-12)),
        "cv": float(np.std(x) / max(mean, 1e-12)),
    }


def _make_slow(ctx: dict, tau_phi_ms: float, fraction: float):
    gap = float(R.PP.CORE_MEAN - ctx["S"]["p"].V_reset)
    delta = delta_phi_mV(tau_phi_ms, fraction, gap_mV=gap)
    cfg = R.ZM._zm_cfg(ctx["S"]["I_th_EI"], **R.ARM_KWARGS)
    cfg = dataclasses.replace(
        cfg,
        use_zm_conductance=False,
        use_phi=True,
        tau_phi=float(tau_phi_ms),
        delta_phi=float(delta),
    )
    base = R.SpatialSlowField(
        ctx["S"]["N"], 18.0, ctx["S"]["posE"], ctx["S"]["posI"],
        ctx["S"]["L"], core_mask_E=ctx["core"], cfg=cfg,
    )
    diagnostic = RT.DiagnosticSlowWrapper(
        base, diagnostic_stride_steps=max(1, int(round(1.0 / ctx["dt"])))
    )
    slow = R.FS.FreezeWrapper(
        diagnostic, R.FS.FreezePolicy.for_arm("freeze_zm")
    )
    return slow, diagnostic, delta


def run_cell(args: argparse.Namespace) -> Path:
    if not args.smoke and float(args.T_ms) != PRODUCTION_T_MS:
        raise RuntimeError(f"production duration must be {PRODUCTION_T_MS:g} ms")
    if not args.smoke and float(args.burn_ms) != PRODUCTION_BURN_MS:
        raise RuntimeError(f"production burn must be {PRODUCTION_BURN_MS:g} ms")
    if float(args.tau_phi_ms) not in TAUS_MS or float(args.fraction) not in FRACTIONS:
        raise RuntimeError("cell lies outside the initial six-point phi panel")

    manifest = json.loads(INPUT.read_text())
    futility = json.loads(FUTILITY.read_text())
    locked_rate = float(futility["seed1_primary_futility"]["core_rate_mean_hz"]["median"])
    if locked_rate != REFERENCE_RATE_HZ:
        raise RuntimeError("Phase-C reference-rate drift")
    row = _source_row(manifest, args.state)
    ctx = RT.build_source_locked_context(ROOT, manifest, R)
    state, transformation = ST.load_and_migrate(
        ROOT,
        manifest,
        row_id=_state_parts(args.state),
        contract_already_validated=True,
    )
    if np.count_nonzero(state["slow.phi_increment"]) != 0:
        raise RuntimeError("branch-intervention phi must start at exact zero")
    bank = NB.build_noise_bank(
        manifest["source"]["canonical_config_sha"],
        int(manifest["source"]["seed"]),
        int(row["t_step"]),
        "noise_replay",
    )
    locked_banks = {b["replicate"]: b for b in row["first_pass_noise_banks"]}
    if locked_banks["noise_replay"]["bank_sha"] != bank["bank_sha"]:
        raise RuntimeError("future-noise bank drift")

    slow, diagnostic, delta = _make_slow(ctx, args.tau_phi_ms, args.fraction)
    z0 = np.array(state["slow.z"], copy=True)
    m0 = np.array(state["slow.m"], copy=True)
    sg0 = float(np.asarray(state["slow.S_G"]))
    controller = CK.ZMCheckpoint(
        initial_state=state,
        rng_state=bank["rng_state"],
        ext_mean_only=bank["ext_mean_only"],
    )

    started = time.time()
    result = R.run_segment(
        ctx,
        slow,
        float(args.T_ms),
        ckpt=controller,
        fresh_rng=True,
        dump_i_spikes=True,
    )
    e_all = np.asarray(result["E_spk_bool"], bool)
    i_all = np.asarray(result["I_spk_bool"], bool)
    burn_steps = min(e_all.shape[0], int(round(float(args.burn_ms) / ctx["dt"])))
    e = e_all[burn_steps:]
    i = i_all[burn_steps:]
    lfp = np.asarray(result["lfp_trace"], float)[burn_steps:]
    fine = IC.fine_rates(e, ctx["core"], ctx["dt"], bin_ms=FINE_BIN_MS)
    i_rate = _population_rate(i, ctx["dt"], FINE_BIN_MS)
    coarse = R.MC.source_metrics(
        e,
        ctx["core"],
        ctx["S"]["posE"],
        ctx["S"]["L"],
        ctx["dt"],
        bin_ms=COARSE_BIN_MS,
        lfp_trace=lfp,
        axis_coord=ctx["axis"],
    )
    phi_e = np.asarray(diagnostic.phi_increment[: ctx["S"]["NE"]], float)
    phi_i = np.asarray(diagnostic.phi_increment[ctx["S"]["NE"] :], float)
    z_drift = float(np.max(np.abs(np.asarray(diagnostic.z) - z0)))
    m_drift = float(np.max(np.abs(np.asarray(diagnostic.m) - m0)))
    if z_drift > 0 or m_drift > 0:
        raise RuntimeError(f"freeze_zm drifted: z={z_drift} m={m_drift}")
    if np.count_nonzero(phi_i) != 0:
        raise RuntimeError("I-cell phi became nonzero")

    arrays = {
        "fine_time_ms": np.asarray(fine["t_ms"], np.float32),
        "fine_core_rate_hz": np.asarray(fine["core"], np.float32),
        "fine_surround_rate_hz": np.asarray(fine["surround"], np.float32),
        "fine_all_e_rate_hz": np.asarray(fine["allE"], np.float32),
        "fine_all_i_rate_hz": np.asarray(i_rate, np.float32),
        "fine_active_fraction": np.asarray(fine["active_frac"], np.float32),
        "coarse_core_rate_hz": np.asarray(coarse["r_core"], np.float32),
        "coarse_surround_rate_hz": np.asarray(coarse["r_surround"], np.float32),
        "coarse_all_e_rate_hz": np.asarray(coarse["r_all"], np.float32),
        "coarse_active_fraction": np.asarray(coarse["A_active"], np.float32),
        "coarse_spatial_entropy": np.asarray(coarse["H_spatial"], np.float32),
        "coarse_kymo_axial": np.asarray(coarse["kymo_axial"], np.float32),
        "lfp_raw_synaptic_proxy": np.asarray(lfp, np.float32),
        "lfp_fs_hz": np.asarray(1000.0 / ctx["dt"]),
        "trace_phi_mean": np.asarray(diagnostic.trace_phi_mean, np.float32),
        "trace_phi_max": np.asarray(diagnostic.trace_phi_max, np.float32),
        "trace_phi_core_mean": np.asarray(diagnostic.trace_phi_core_mean, np.float32),
        "trace_phi_surround_mean": np.asarray(diagnostic.trace_phi_surround_mean, np.float32),
        "trace_S_G": np.asarray(diagnostic.trace_SG, np.float32),
        "trace_m_core_mean": np.asarray(diagnostic.trace_m_core_mean, np.float32),
        "trace_z_core_mean": np.asarray(diagnostic.trace_z_core_mean, np.float32),
    }
    namespace = "smoke" if args.smoke else "discovery"
    stem = f"{args.state}__tau{args.tau_phi_ms:g}__f{args.fraction:g}"
    root = OUT / namespace / "seed1" / stem
    npz_path = root / "traces.npz"
    json_path = root / "summary.json"
    _write_npz_once(npz_path, arrays)
    payload = {
        "schema": "zm_fast_lifecycle_development_cell_v1_2026-08-01",
        "stage": "A_branch_intervention",
        "semantic_scope": "branch_intervention_not_reachability",
        "seed": 1,
        "state": args.state,
        "tau_phi_ms": float(args.tau_phi_ms),
        "fraction": float(args.fraction),
        "delta_phi_mV_per_spike": float(delta),
        "reference_rate_hz": REFERENCE_RATE_HZ,
        "threshold_gap_mV": float(R.PP.CORE_MEAN - ctx["S"]["p"].V_reset),
        "T_ms": float(args.T_ms),
        "burn_ms": float(args.burn_ms),
        "dt_ms": float(ctx["dt"]),
        "input_manifest_sha256": manifest["manifest_sha256"],
        "source_state_hash": transformation["source_state_hash"],
        "migrated_state_hash": transformation["migrated_state_hash"],
        "source_file_sha256": row["file_sha256"],
        "noise_bank_sha256": bank["bank_sha"],
        "runtime_git_sha": RT.git_sha(ROOT),
        "use_zm_conductance": False,
        "freeze_policy": R.FS.FreezePolicy.for_arm("freeze_zm").as_dict(),
        "phi_initial_nonzero": 0,
        "phi_final_mean_mV": float(np.mean(phi_e)),
        "phi_final_max_mV": float(np.max(phi_e)),
        "phi_i_nonzero": int(np.count_nonzero(phi_i)),
        "z_max_abs_drift": z_drift,
        "m_max_abs_drift": m_drift,
        "S_G_initial": sg0,
        "S_G_final": float(diagnostic.S_G),
        "runaway_early_stop_ms": result.get("runaway_early_stop_ms"),
        "observed_ms": float(e_all.shape[0] * ctx["dt"]),
        "postburn_ms": float(e.shape[0] * ctx["dt"]),
        "total_E_spikes_postburn": int(np.sum(e, dtype=np.int64)),
        "total_I_spikes_postburn": int(np.sum(i, dtype=np.int64)),
        "core_modulation": _modulation(fine["core"]),
        "all_E_modulation": _modulation(fine["allE"]),
        "all_I_modulation": _modulation(i_rate),
        "core_rho80_active_fraction": _core_saturation_fraction(
            e, ctx["core"], ctx["dt"], tau_ref_ms=ctx["S"]["p"].tau_ref_E
        ),
        "mean_active_fraction": float(np.mean(fine["active_frac"])) if len(fine["active_frac"]) else 0.0,
        "peak_active_fraction": float(np.max(fine["active_frac"])) if len(fine["active_frac"]) else 0.0,
        "coarse_spatial_entropy_mean": float(np.mean(coarse["H_spatial"])),
        "array_path": str(npz_path.relative_to(ROOT)),
        "array_file_sha256": _sha(npz_path),
        "wall_s": float(time.time() - started),
        "peak_rss_gb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2),
    }
    _write_json_once(json_path, payload)
    print(json_path)
    print(json.dumps({
        "core_mean_hz": payload["core_modulation"]["mean_hz"],
        "core_depth": payload["core_modulation"]["depth"],
        "all_E_mean_hz": payload["all_E_modulation"]["mean_hz"],
        "rho80": payload["core_rho80_active_fraction"],
        "phi_mean": payload["phi_final_mean_mV"],
        "S_G_final": payload["S_G_final"],
        "runaway_ms": payload["runaway_early_stop_ms"],
        "wall_s": payload["wall_s"],
        "rss_gb": payload["peak_rss_gb"],
    }, sort_keys=True))
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cell",))
    parser.add_argument("--state", choices=STATES, required=True)
    parser.add_argument("--tau-phi-ms", type=float, choices=TAUS_MS, required=True)
    parser.add_argument("--fraction", type=float, choices=FRACTIONS, required=True)
    parser.add_argument("--T-ms", type=float, default=PRODUCTION_T_MS)
    parser.add_argument("--burn-ms", type=float, default=PRODUCTION_BURN_MS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("refusing SNN development run without --confirm-run")
    run_cell(args)


if __name__ == "__main__":
    main()
