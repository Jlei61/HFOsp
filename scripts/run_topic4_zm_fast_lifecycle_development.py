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
from src import topic4_lifecycle_worker_receipt as WR  # noqa: E402


INPUT = ROOT / "results/topic4_sef_hfo/zm_fast_carrier_repair/phaseD_input_manifest_v1_5.json"
FUTILITY = ROOT / "results/topic4_sef_hfo/zm_phase_c_tonic_identity/phasec_futility_verdict.json"
OUT = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development"
STATES = (
    "bounded_mid__rising",
    "bounded_mid__peak",
    "bounded_late__rising",
    "bounded_late__peak",
)
FROZEN_MODE_STATES = (
    "pre_entry__natural",
) + STATES
TAUS_MS = (60.0, 100.0, 160.0)
FRACTIONS = (0.15, 0.30)
REFERENCE_RATE_HZ = 439.22905756378174
FINE_BIN_MS = 2.0
COARSE_BIN_MS = 25.0
PRODUCTION_T_MS = 6000.0
PRODUCTION_BURN_MS = 1000.0
RACE_STATE = "bounded_mid__peak"
RACE_PHI_TAU_MS = 60.0
RACE_PHI_FRACTION = 0.15
RACE_I_REFERENCE_RATE_HZ = 11.790075302124023
RACE_TAU_D_MS = (100.0, 300.0, 600.0)
RACE_D_STAR = (0.35, 0.55, 0.75)
RACE_TAU_I_MS = (100.0, 300.0, 600.0)
RACE_F_I = (0.10, 0.25)
RACE_ARMS = ("native", "phi", "i2e", "iadapt", "combined")
CONTROL_CLOCK_VERSION = "relative_to_pre_entry_checkpoint_v2"


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


def i2e_use_from_target(
    tau_ms: float,
    d_star: float,
    *,
    rate_hz: float = RACE_I_REFERENCE_RATE_HZ,
    maximum_use: float = 0.95,
) -> dict:
    """Map a nominal steady resource to a physical multiplicative use rate."""
    tau_s = float(tau_ms) / 1000.0
    if tau_s <= 0 or float(rate_hz) <= 0:
        raise ValueError("tau and I reference rate must be positive")
    if not 0 < float(d_star) < 1:
        raise ValueError("d_star must lie strictly between zero and one")
    nominal = (1.0 / float(d_star) - 1.0) / (tau_s * float(rate_hz))
    applied = min(float(maximum_use), nominal)
    attainable = 1.0 / (1.0 + applied * tau_s * float(rate_hz))
    return {
        "U_nominal": float(nominal),
        "U_applied": float(applied),
        "use_was_capped": bool(applied != nominal),
        "d_star_nominal": float(d_star),
        "d_star_attainable_at_reference_rate": float(attainable),
    }


def delta_i_adaptation_mV(
    tau_ms: float,
    fraction: float,
    *,
    gap_mV: float = 7.0,
    rate_hz: float = RACE_I_REFERENCE_RATE_HZ,
) -> float:
    return delta_phi_mV(tau_ms, fraction, gap_mV=gap_mV, rate_hz=rate_hz)


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
    if state_tag not in FROZEN_MODE_STATES:
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


def _mechanism_kwargs(args: argparse.Namespace, ctx: dict) -> tuple[dict, dict]:
    """Return config edits and an auditable parameter receipt for one race arm."""
    if args.command == "cell":
        return {}, {"arm": "phi", "strength_scale": 1.0}
    arm = str(args.arm)
    if arm not in RACE_ARMS:
        raise ValueError(f"unknown race arm {arm}")
    use_phi = arm != "native"
    edits = {"use_phi": use_phi}
    default_scale = 0.5 if arm == "combined" and args.command != "sprint-cell" else 1.0
    scale = default_scale if args.strength_scale is None else float(args.strength_scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("strength_scale must lie in (0, 1]")
    receipt = {"arm": arm, "strength_scale": scale}
    if arm in {"i2e", "combined"}:
        if args.tau_D_ms is None or args.d_star is None:
            raise ValueError(f"{arm} requires tau_D_ms and d_star")
        dep = i2e_use_from_target(args.tau_D_ms, args.d_star)
        scale = receipt["strength_scale"]
        edits.update(
            use_i2e_depression=True,
            tau_i2e_depression=float(args.tau_D_ms),
            U_i2e_depression=float(scale * dep["U_applied"]),
            d_i2e_min=0.20,
            i2e_tau_cv=float(getattr(args, "i2e_tau_cv", 0.0)),
            i2e_tau_seed=int(getattr(args, "i2e_tau_seed", 0)),
        )
        receipt["i2e_depression"] = {
            **dep,
            "U_applied_after_strength_scale": float(scale * dep["U_applied"]),
            "tau_D_ms": float(args.tau_D_ms),
            "d_min": 0.20,
            "tau_recovery_cv": float(getattr(args, "i2e_tau_cv", 0.0)),
            "tau_recovery_seed": int(getattr(args, "i2e_tau_seed", 0)),
        }
    if arm in {"iadapt", "combined"}:
        if args.tau_aI_ms is None or args.f_aI is None:
            raise ValueError(f"{arm} requires tau_aI_ms and f_aI")
        gap_i = float(ctx["S"]["p"].V_th - ctx["S"]["p"].V_reset)
        delta_i = delta_i_adaptation_mV(
            args.tau_aI_ms, args.f_aI, gap_mV=gap_i
        )
        scale = receipt["strength_scale"]
        edits.update(
            use_i_adaptation=True,
            tau_i_adaptation=float(args.tau_aI_ms),
            delta_i_adaptation=float(scale * delta_i),
        )
        receipt["i_adaptation"] = {
            "tau_aI_ms": float(args.tau_aI_ms),
            "f_aI": float(args.f_aI),
            "delta_mV_per_spike_nominal": float(delta_i),
            "delta_mV_per_spike_after_strength_scale": float(scale * delta_i),
            "I_threshold_gap_mV": gap_i,
            "I_reference_rate_hz": RACE_I_REFERENCE_RATE_HZ,
        }
    return edits, receipt


def _make_slow(ctx: dict, tau_phi_ms: float, fraction: float, *, args=None):
    gap = float(R.PP.CORE_MEAN - ctx["S"]["p"].V_reset)
    delta = delta_phi_mV(tau_phi_ms, fraction, gap_mV=gap)
    cfg = R.ZM._zm_cfg(ctx["S"]["I_th_EI"], **R.ARM_KWARGS)
    edits, receipt = ({}, {"arm": "phi", "strength_scale": 1.0})
    if args is not None:
        edits, receipt = _mechanism_kwargs(args, ctx)
    values = {
        "use_zm_conductance": False,
        "use_phi": True,
        "tau_phi": float(tau_phi_ms),
        "delta_phi": float(delta),
    }
    values.update(edits)
    dynamic = args is not None and args.command in {"dynamic-cell", "sprint-cell"}
    if dynamic:
        if float(args.g_M) < 0.0:
            raise ValueError("g_M must be nonnegative")
        if float(args.g_Z) <= 0.0:
            raise ValueError("g_Z must be positive")
        values["eta_m"] = float(cfg.eta_m) * float(args.g_M)
        # Entry-only contingency: g_Z changes the speed along the native Z
        # coordinate, not its nullcline or its inhibitory sign.
        values["tau_z"] = float(cfg.tau_z) / float(args.g_Z)
        if args.tau_M_ms is not None:
            if float(args.tau_M_ms) <= 0.0:
                raise ValueError("tau_M_ms must be positive")
            values["tau_adp"] = float(args.tau_M_ms)
        receipt["dynamic_slow_flow"] = {
            "g_M": float(args.g_M),
            "eta_m_native": float(cfg.eta_m),
            "eta_m_applied": float(values["eta_m"]),
            "tau_M_ms": float(values.get("tau_adp", cfg.tau_adp)),
            "g_Z": float(args.g_Z),
            "tau_Z_native_ms": float(cfg.tau_z),
            "tau_Z_applied_ms": float(values["tau_z"]),
        }
        if bool(getattr(args, "use_zm_conductance_homotopy", False)):
            values.update({
                "use_zm_conductance_homotopy": True,
                "cond_homotopy_z_native": float(args.homotopy_z_native),
                "cond_homotopy_z_conductance": float(
                    args.homotopy_z_conductance
                ),
                "cond_kappa_E": float(args.cond_kappa_E),
                "cond_kappa_I": float(args.cond_kappa_I),
                "cond_g_M": float(args.cond_g_M),
                "cond_gamma": float(args.cond_gamma),
            })
            receipt["state_dependent_conductance_homotopy"] = {
                key: values[key] for key in (
                    "cond_homotopy_z_native",
                    "cond_homotopy_z_conductance",
                    "cond_kappa_E",
                    "cond_kappa_I",
                    "cond_g_M",
                    "cond_gamma",
                )
            }
        if float(getattr(args, "beta_SG", 0.0)) > 0.0 or float(
            getattr(args, "beta_SG_ramp_per_s", 0.0)
        ) != 0.0:
            # The pool already divides recurrent E; this adds the subtractive
            # component the reduced field named as the missing ingredient for
            # the mean field to have an orbit at all.  0 keeps the parity path.
            values["beta_SG"] = float(args.beta_SG)
            values["beta_SG_ramp_per_s"] = float(args.beta_SG_ramp_per_s)
            receipt["subtractive_pool"] = {
                "beta_SG": float(args.beta_SG),
                "beta_SG_ramp_per_s": float(args.beta_SG_ramp_per_s),
                "open_loop_kinematic_probe": bool(args.beta_SG_ramp_per_s != 0.0),
                "alpha_G": float(cfg.alpha_G),
                "tau_S_ms": float(cfg.tau_S),
                "S_max": float(cfg.S_max),
            }
        if bool(getattr(args, "use_mode_H", False)):
            values.update({
                "use_mode_H": True,
                "tau_mode_H": float(args.tau_mode_H_ms),
                "tau_mode_H_down": float(
                    getattr(args, "tau_mode_H_down_ms", args.tau_mode_H_ms)
                ),
                "rho_mode_H": float(args.rho_mode_H),
                "mode_H_persistent_g_max": float(
                    args.mode_H_persistent_g_max
                ),
                "mode_H_persistent_e_exc": float(
                    args.mode_H_persistent_e_exc_mv
                ),
                "mode_H_common_subtraction": float(
                    args.mode_H_common_subtraction
                ),
                "theta_mode_H_hz": float(args.theta_mode_H_hz),
                "half_mode_H_hz": float(args.half_mode_H_hz),
                "z_mode_base": float(args.z_mode_base),
                "z_mode_susceptible": float(args.z_mode_susceptible),
                "zeta_mode_center": float(args.zeta_mode_center),
                "zeta_mode_slope": float(args.zeta_mode_slope),
                "m_mode_half": float(args.m_mode_half),
                "m_mode_power": float(args.m_mode_power),
            })
            receipt["state_selective_mode_H"] = {
                key: values[key] for key in (
                    "tau_mode_H", "tau_mode_H_down", "rho_mode_H",
                    "mode_H_persistent_g_max", "mode_H_persistent_e_exc",
                    "mode_H_common_subtraction", "theta_mode_H_hz",
                    "half_mode_H_hz", "z_mode_base", "z_mode_susceptible",
                    "zeta_mode_center", "zeta_mode_slope", "m_mode_half",
                    "m_mode_power",
                )
            }
        if bool(getattr(args, "use_mode_M_divisive", False)):
            values.update({
                "use_mode_M_divisive": True,
                "kappa_mode_M": float(args.kappa_mode_M),
                "m_mode_div_ref": float(args.m_mode_div_ref),
                "m_mode_div_power": float(args.m_mode_div_power),
                "m_mode_div_hill_power": float(args.m_mode_div_hill_power),
                "use_mode_M_memory": bool(args.use_mode_M_memory),
                "tau_mode_M_memory_up": float(args.tau_mode_M_memory_up_ms),
                "tau_mode_M_memory_down": float(args.tau_mode_M_memory_down_ms),
            })
            receipt["collective_mode_M_divisive"] = {
                key: values[key] for key in (
                    "kappa_mode_M", "m_mode_div_ref", "m_mode_div_power",
                    "m_mode_div_hill_power",
                    "use_mode_M_memory", "tau_mode_M_memory_up",
                    "tau_mode_M_memory_down",
                )
            }
    cfg = dataclasses.replace(cfg, **values)
    base = R.SpatialSlowField(
        ctx["S"]["N"], 18.0, ctx["S"]["posE"], ctx["S"]["posI"],
        ctx["S"]["L"], core_mask_E=ctx["core"], cfg=cfg,
    )
    diagnostic = RT.DiagnosticSlowWrapper(
        base, diagnostic_stride_steps=max(1, int(round(1.0 / ctx["dt"])))
    )
    freeze_zm = bool(getattr(args, "freeze_zm", False)) if args is not None else False
    policy = "dynamic_replay" if dynamic and not freeze_zm else "freeze_zm"
    slow = R.FS.FreezeWrapper(diagnostic, R.FS.FreezePolicy.for_arm(policy))
    return slow, diagnostic, delta, receipt


def _mechanism_stem(args: argparse.Namespace) -> str:
    stem = str(args.arm)
    if args.arm in {"i2e", "combined"}:
        stem += f"__tauD{args.tau_D_ms:g}__d{args.d_star:g}"
    if args.arm in {"iadapt", "combined"}:
        stem += f"__tauI{args.tau_aI_ms:g}__fI{args.f_aI:g}"
    if args.strength_scale is not None:
        stem += f"__s{args.strength_scale:g}"
    if args.control_uplift_mV > 0:
        stem += (
            f"__ctl{args.control_target}__u{args.control_uplift_mV:g}"
            f"__t{args.control_t0_ms:g}__dur{args.control_duration_ms:g}"
            "__clkrel2"
        )
    if bool(getattr(args, "use_mode_H", False)):
        tau_h_down = float(getattr(args, "tau_mode_H_down_ms", args.tau_mode_H_ms))
        stem += f"__modeH{args.rho_mode_H:g}t{args.tau_mode_H_ms:g}"
        if not np.isclose(tau_h_down, float(args.tau_mode_H_ms)):
            stem += f"d{tau_h_down:g}"
        stem += f"__mc{args.m_mode_half:g}"
        if float(getattr(args, "mode_H_common_subtraction", 0.0)) > 0.0:
            stem += f"cs{args.mode_H_common_subtraction:g}"
        if float(getattr(args, "mode_H_persistent_g_max", 0.0)) > 0.0:
            stem += (
                f"pg{args.mode_H_persistent_g_max:g}"
                f"e{args.mode_H_persistent_e_exc_mv:g}"
            )
    if float(getattr(args, "beta_SG", 0.0)) > 0.0:
        stem += f"__bSG{args.beta_SG:g}"
    if float(getattr(args, "beta_SG_ramp_per_s", 0.0)) != 0.0:
        stem += f"__bSGramp{args.beta_SG_ramp_per_s:g}"
    if bool(getattr(args, "freeze_zm", False)):
        stem += f"__freeze_{args.state}"
    if float(getattr(args, "i2e_tau_cv", 0.0)) > 0.0:
        stem += f"__tauDcv{args.i2e_tau_cv:g}s{args.i2e_tau_seed:d}"
    if not np.isclose(float(getattr(args, "i2e_delay_scale", 1.0)), 1.0):
        stem += f"__i2edelay{args.i2e_delay_scale:g}"
    if float(getattr(args, "i2e_delay_cv", 0.0)) > 0.0:
        stem += f"__i2edelaycv{args.i2e_delay_cv:g}s{args.i2e_delay_seed:d}"
    if bool(getattr(args, "use_dual_gaba", False)):
        stem += (
            f"__dualGf{args.dual_gaba_slow_fraction:g}"
            f"sig{args.dual_gaba_sigma_mm:g}"
            f"c{args.dual_gaba_in_degree:d}"
            f"tr{args.dual_gaba_tau_r_ms:g}"
            f"td{args.dual_gaba_tau_d_ms:g}"
            f"s{args.dual_gaba_seed:d}"
        )
    if bool(getattr(args, "use_inhibitory_subtypes", False)):
        stem += (
            f"__pvSOMq{args.som_source_fraction:g}"
            f"f{args.som_slow_budget_fraction:g}"
            f"sig{args.som_sigma_mm:g}c{args.som_in_degree:d}"
            f"rd{args.som_recruit_delay_scale:g}"
            f"tr{args.som_tau_r_ms:g}td{args.som_tau_d_ms:g}"
            f"s{args.som_seed:d}"
        )
        if bool(getattr(args, "som_shunting", False)):
            stem += f"sh{args.som_shunt_scale:g}eg{args.som_e_gaba_mv:g}"
    if bool(getattr(args, "use_mode_M_divisive", False)):
        stem += (
            f"__mdiv{args.kappa_mode_M:g}"
            f"r{args.m_mode_div_ref:g}p{args.m_mode_div_power:g}"
            f"h{args.m_mode_div_hill_power:g}"
        )
        if bool(getattr(args, "use_mode_M_memory", False)):
            stem += (
                f"mem{args.tau_mode_M_memory_up_ms:g}"
                f"d{args.tau_mode_M_memory_down_ms:g}"
            )
    if bool(getattr(args, "use_zm_conductance_homotopy", False)):
        stem += (
            f"__condhom_z{args.homotopy_z_native:g}"
            f"to{args.homotopy_z_conductance:g}"
            f"g{args.cond_gamma:g}"
        )
    return stem


def control_window_in_engine_time(
    *, source_t_ms: float, relative_t0_ms: float, duration_ms: float
) -> tuple[float, float]:
    """Translate a branch-relative pulse window to the engine's absolute clock.

    ``simulate_kick`` resumes the checkpoint's absolute timestep, whereas the
    lifecycle traces and control manifests use time after the pre-entry fork.
    Keeping this conversion at the runner boundary prevents a relative pulse
    from silently falling before the resumed simulation begins.
    """
    source_t_ms = float(source_t_ms)
    relative_t0_ms = float(relative_t0_ms)
    duration_ms = float(duration_ms)
    if source_t_ms < 0.0 or relative_t0_ms < 0.0 or duration_ms <= 0.0:
        raise ValueError("control clock inputs must be nonnegative with positive duration")
    t0 = source_t_ms + relative_t0_ms
    return t0, t0 + duration_ms


def run_cell(args: argparse.Namespace, *, worker_receipt=None) -> Path:
    launch_git_sha = RT.git_sha(ROOT)
    dynamic = args.command in {"dynamic-cell", "sprint-cell"}
    if dynamic:
        allowed_T = (
            (12000.0, 20000.0, 30000.0, 45000.0, 60000.0)
            if args.command == "sprint-cell" else (30000.0, 60000.0)
        )
        # The pre-entry checkpoint precedes the native onset by 1.35 s.  A
        # 1-s ceiling cannot exercise any activity-gated mechanism and can
        # silently certify a dead sensor.  Keep the smoke short but long enough
        # to cross the locked onset reference.
        if args.smoke and not 0.0 < float(args.T_ms) <= 2500.0:
            raise RuntimeError("dynamic smoke duration must lie in (0,2500] ms")
        if not args.smoke and float(args.T_ms) not in allowed_T:
            raise RuntimeError(f"dynamic prototype duration must be one of {allowed_T}")
        if not args.smoke and float(args.burn_ms) not in (1000.0, 2000.0):
            raise RuntimeError("dynamic prototype equilibration must be 1000 or 2000 ms")
        allowed_gz = (0.8, 1.0, 1.25) if args.command == "sprint-cell" else (1.0, 1.25, 1.5)
        if float(args.g_Z) not in allowed_gz:
            raise RuntimeError("g_Z lies outside the onset-contingency panel")
        if float(args.control_uplift_mV) < 0.0:
            raise RuntimeError("control_uplift_mV must be nonnegative")
        if float(args.control_uplift_mV) > 0.0:
            if args.command != "sprint-cell":
                raise RuntimeError("finite control is only available in sprint-cell")
            if args.control_t0_ms is None or float(args.control_t0_ms) <= 0.0:
                raise RuntimeError("active control requires positive control_t0_ms")
            if not 0.0 < float(args.control_duration_ms) <= 500.0:
                raise RuntimeError("control duration must lie in (0,500] ms")
            if float(args.control_t0_ms) + float(args.control_duration_ms) >= float(args.T_ms):
                raise RuntimeError("control pulse must end before the trajectory ends")
    else:
        if not args.smoke and float(args.T_ms) != PRODUCTION_T_MS:
            raise RuntimeError(f"production duration must be {PRODUCTION_T_MS:g} ms")
        if not args.smoke and float(args.burn_ms) != PRODUCTION_BURN_MS:
            raise RuntimeError(f"production burn must be {PRODUCTION_BURN_MS:g} ms")
    race = args.command == "race-cell"
    if race or dynamic:
        if args.state != RACE_STATE and not (
            dynamic and bool(getattr(args, "freeze_zm", False))
        ):
            raise RuntimeError(f"mechanism race is locked to {RACE_STATE}")
        if (
            float(args.tau_phi_ms) != RACE_PHI_TAU_MS
            or float(args.fraction) != RACE_PHI_FRACTION
        ):
            raise RuntimeError("lifecycle prototype phi coordinate drift")
        if args.command == "sprint-cell":
            if args.arm not in {"i2e", "combined"}:
                raise RuntimeError("sprint-cell requires i2e or combined")
            if float(args.g_M) not in (0.0, 1.0, 3.0, 10.0, 30.0):
                raise RuntimeError("sprint g_M lies outside {0,1,3,10,30}")
            if args.tau_M_ms is not None and float(args.tau_M_ms) not in (500.0, 2000.0):
                raise RuntimeError("sprint tau_M must be 500 or 2000 ms")
            if not 300.0 <= float(args.tau_D_ms) <= 850.0:
                raise RuntimeError("sprint tau_D must lie in [300,850] ms")
            if not 0.55 <= float(args.d_star) <= 0.85:
                raise RuntimeError("sprint d_star must lie in [0.55,0.85]")
            if args.arm == "combined":
                if not 60.0 <= float(args.tau_aI_ms) <= 350.0:
                    raise RuntimeError("sprint tau_aI must lie in [60,350] ms")
                if not 0.0 <= float(args.f_aI) <= 0.12:
                    raise RuntimeError("sprint f_aI must lie in [0,0.12]")
        else:
            if args.tau_D_ms is not None and float(args.tau_D_ms) not in RACE_TAU_D_MS:
                raise RuntimeError("tau_D lies outside the registered race panel")
            if args.d_star is not None and float(args.d_star) not in RACE_D_STAR:
                raise RuntimeError("d_star lies outside the registered race panel")
            if args.tau_aI_ms is not None and float(args.tau_aI_ms) not in RACE_TAU_I_MS:
                raise RuntimeError("tau_aI lies outside the registered race panel")
            if args.f_aI is not None and float(args.f_aI) not in RACE_F_I:
                raise RuntimeError("f_aI lies outside the registered race panel")
    elif float(args.tau_phi_ms) not in TAUS_MS or float(args.fraction) not in FRACTIONS:
        raise RuntimeError("cell lies outside the initial six-point phi panel")

    manifest = json.loads(INPUT.read_text())
    futility = json.loads(FUTILITY.read_text())
    locked_rate = float(futility["seed1_primary_futility"]["core_rate_mean_hz"]["median"])
    if locked_rate != REFERENCE_RATE_HZ:
        raise RuntimeError("Phase-C reference-rate drift")
    source_id = (
        _state_parts(args.state)
        if not dynamic or bool(getattr(args, "freeze_zm", False))
        else ("pre_entry", "natural")
    )
    rows = [
        item for item in manifest["source_panel"]
        if (item["bin_name"], item["fast_phase"]) == source_id
    ]
    if len(rows) != 1:
        raise RuntimeError(f"source {source_id!r} does not resolve exactly once")
    row = rows[0]
    ctx = RT.build_source_locked_context(ROOT, manifest, R)
    state, transformation = ST.load_and_migrate(
        ROOT,
        manifest,
        row_id=source_id,
        contract_already_validated=True,
    )
    delay_receipt = None
    dual_gaba_receipt = None
    subtype_receipt = None
    if bool(args.use_dual_gaba) and bool(args.use_inhibitory_subtypes):
        raise RuntimeError("dual GABA and PV/SOM subtype transforms are mutually exclusive")
    if bool(args.som_shunting) and not bool(args.use_inhibitory_subtypes):
        raise RuntimeError("SOM shunting requires the PV/SOM subtype transform")
    if (
        not np.isfinite(float(args.som_shunt_scale))
        or float(args.som_shunt_scale) < 0.0
        or not np.isfinite(float(args.som_e_gaba_mv))
    ):
        raise RuntimeError("SOM shunt scale/reversal must be finite and scale >=0")
    if (
        not np.isclose(float(args.i2e_delay_scale), 1.0)
        or float(args.i2e_delay_cv) > 0.0
    ):
        ctx["S"]["net"], state, delay_receipt = RT.rescale_i2e_delay_bins(
            ctx["S"]["net"], state,
            n_e=int(ctx["S"]["NE"]), scale=float(args.i2e_delay_scale),
            source_delay_cv=float(args.i2e_delay_cv),
            source_delay_seed=int(args.i2e_delay_seed),
        )
        transformation["migrated_state_hash"] = CK.state_hash(state)
    if bool(args.use_inhibitory_subtypes):
        p = ctx["S"]["p"]
        ctx["S"]["net"], state, subtype_receipt = RT.build_pv_som_inhibitory_subtypes(
            ctx["S"]["net"],
            state,
            n_e=int(ctx["S"]["NE"]),
            som_source_fraction=float(args.som_source_fraction),
            som_slow_budget_fraction=float(args.som_slow_budget_fraction),
            som_sigma_mm=float(args.som_sigma_mm),
            som_in_degree=int(args.som_in_degree),
            som_candidate_count=int(args.som_candidate_count),
            som_recruit_delay_scale=float(args.som_recruit_delay_scale),
            seed=int(args.som_seed),
            dt_ms=float(p.dt),
            delay_dt_ms=float(p.delay_dt),
            tau0_ms=float(p.tau0),
            v_axon_mm_per_ms=float(p.v_axon),
            tau_r_fast_ms=float(p.tau_r_GABA),
            tau_r_som_ms=float(args.som_tau_r_ms),
            tau_d_som_ms=float(args.som_tau_d_ms),
        )
        if bool(args.som_shunting):
            ctx["S"]["net"].update(
                gaba_slow_membrane_mode="shunt",
                gaba_slow_shunt_scale=float(args.som_shunt_scale),
                gaba_slow_e_gaba_mv=float(args.som_e_gaba_mv),
            )
            subtype_receipt.update(
                slow_membrane_mode="shunt",
                som_shunt_scale=float(args.som_shunt_scale),
                som_e_gaba_mv=float(args.som_e_gaba_mv),
            )
        transformation["migrated_state_hash"] = CK.state_hash(state)
    if bool(args.use_dual_gaba):
        p = ctx["S"]["p"]
        ctx["S"]["net"], state, dual_gaba_receipt = RT.build_dual_scale_i2e_gaba(
            ctx["S"]["net"],
            state,
            n_e=int(ctx["S"]["NE"]),
            slow_fraction=float(args.dual_gaba_slow_fraction),
            broad_sigma_mm=float(args.dual_gaba_sigma_mm),
            broad_in_degree=int(args.dual_gaba_in_degree),
            broad_candidate_count=int(args.dual_gaba_candidate_count),
            seed=int(args.dual_gaba_seed),
            dt_ms=float(p.dt),
            delay_dt_ms=float(p.delay_dt),
            tau0_ms=float(p.tau0),
            v_axon_mm_per_ms=float(p.v_axon),
            tau_r_fast_ms=float(p.tau_r_GABA),
            tau_r_slow_ms=float(args.dual_gaba_tau_r_ms),
            tau_d_slow_ms=float(args.dual_gaba_tau_d_ms),
        )
        transformation["migrated_state_hash"] = CK.state_hash(state)
    if worker_receipt is not None:
        worker_receipt.update_context(
            checkpoint_hash=transformation["source_state_hash"],
            source_t_ms=float(row["t_ms"]),
            source_file_sha256=row["file_sha256"],
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

    slow, diagnostic, delta, mechanism = _make_slow(
        ctx, args.tau_phi_ms, args.fraction, args=args
    )
    if delay_receipt is not None:
        mechanism["i2e_delay_rescaling"] = delay_receipt
    if dual_gaba_receipt is not None:
        mechanism["dual_scale_i2e_gaba"] = dual_gaba_receipt
    if subtype_receipt is not None:
        mechanism["pv_som_inhibitory_subtypes"] = subtype_receipt
    z0 = np.array(state["slow.z"], copy=True)
    m0 = np.array(state["slow.m"], copy=True)
    sg0 = float(np.asarray(state["slow.S_G"]))
    controller = CK.ZMCheckpoint(
        initial_state=state,
        rng_state=bank["rng_state"],
        ext_mean_only=bank["ext_mean_only"],
    )

    perturb = None
    control_engine_window = None
    if float(args.control_uplift_mV) > 0.0:
        target = np.zeros(ctx["S"]["N"], dtype=bool)
        if args.control_target == "all_E":
            target[:ctx["S"]["NE"]] = True
        elif args.control_target == "core_E":
            target[:ctx["S"]["NE"]] = np.asarray(ctx["core"], bool)
        else:  # parser choice is the first guard; keep runner fail-closed.
            raise RuntimeError(f"unknown control target {args.control_target!r}")
        source_engine_t_ms = float(row["t_step"]) * float(ctx["dt"])
        if not np.isclose(source_engine_t_ms, float(row["t_ms"]), atol=1e-9, rtol=0.0):
            raise RuntimeError("checkpoint source time disagrees with source timestep")
        # `simulate_kick` resumes from the *state's* absolute step, but the pulse
        # window is built from the manifest's.  If those ever diverge the pulse
        # lands at the wrong absolute time and still produces a plausible trace,
        # so the equality is asserted rather than assumed.
        if int(np.asarray(state["t"])) != int(row["t_step"]):
            raise RuntimeError(
                "resumed engine step differs from the manifest source timestep: "
                f"{int(np.asarray(state['t']))} != {int(row['t_step'])}"
            )
        control_engine_window = control_window_in_engine_time(
            source_t_ms=source_engine_t_ms,
            relative_t0_ms=float(args.control_t0_ms),
            duration_ms=float(args.control_duration_ms),
        )
        perturb = {
            "kind": "inhibitory_pulse",
            "t0": control_engine_window[0],
            "t1": control_engine_window[1],
            "val": float(args.control_uplift_mV),
            "target_mask": target,
        }

    started = time.time()
    result = R.run_segment(
        ctx,
        slow,
        float(args.T_ms),
        ckpt=controller,
        fresh_rng=True,
        dump_i_spikes=True,
        dump_lfp_components=True,
        perturb=perturb,
    )
    e_all = np.asarray(result["E_spk_bool"], bool)
    i_all = np.asarray(result["I_spk_bool"], bool)
    # Keep the dynamic arm's neutral-state equilibration in the saved trace:
    # it is the genuine pre-onset part of the lifecycle, not disposable burn-in.
    burn_steps = 0 if dynamic else min(
        e_all.shape[0], int(round(float(args.burn_ms) / ctx["dt"]))
    )
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
    if (not dynamic or bool(getattr(args, "freeze_zm", False))) and (z_drift > 0 or m_drift > 0):
        raise RuntimeError(f"freeze_zm drifted: z={z_drift} m={m_drift}")
    if np.count_nonzero(phi_i) != 0:
        raise RuntimeError("I-cell phi became nonzero")
    i_adaptation = np.asarray(diagnostic.i_adaptation_increment, float)
    if np.count_nonzero(i_adaptation[: ctx["S"]["NE"]]) != 0:
        raise RuntimeError("E-cell I-adaptation increment became nonzero")
    i2e_resource = np.asarray(diagnostic.i2e_resource, float)
    if np.any(i2e_resource < 0.20 - 1e-12) or np.any(i2e_resource > 1.0 + 1e-12):
        raise RuntimeError("I->E resource left its registered bounds")

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
        "lfp_exc_synaptic_proxy": np.asarray(result["lfp_exc_trace"], np.float32)[burn_steps:],
        "lfp_inh_synaptic_proxy": np.asarray(result["lfp_inh_trace"], np.float32)[burn_steps:],
        "lfp_fs_hz": np.asarray(1000.0 / ctx["dt"]),
        "trace_phi_mean": np.asarray(diagnostic.trace_phi_mean, np.float32),
        "trace_phi_max": np.asarray(diagnostic.trace_phi_max, np.float32),
        "trace_phi_core_mean": np.asarray(diagnostic.trace_phi_core_mean, np.float32),
        "trace_phi_surround_mean": np.asarray(diagnostic.trace_phi_surround_mean, np.float32),
        "trace_S_G": np.asarray(diagnostic.trace_SG, np.float32),
        "trace_m_core_mean": np.asarray(diagnostic.trace_m_core_mean, np.float32),
        "trace_z_core_mean": np.asarray(diagnostic.trace_z_core_mean, np.float32),
        "trace_i2e_resource_mean": np.asarray(
            diagnostic.trace_i2e_resource_mean, np.float32
        ),
        "trace_i2e_resource_min": np.asarray(
            diagnostic.trace_i2e_resource_min, np.float32
        ),
        "trace_i_adaptation_mean": np.asarray(
            diagnostic.trace_i_adaptation_mean, np.float32
        ),
        "trace_i_adaptation_max": np.asarray(
            diagnostic.trace_i_adaptation_max, np.float32
        ),
        # The pool's subtractive strength has to be set against the recurrent
        # current it opposes, so that current has to leave the run measured
        # rather than assumed.  The engine already collects it.
        "trace_Irec_mean": np.asarray(diagnostic.trace_Irec_mean, np.float32),
        "trace_Irec_postdiv_mean": np.asarray(
            diagnostic.trace_Irec_postdiv_mean, np.float32
        ),
        "trace_Isub_mean": np.asarray(diagnostic.trace_Isub_mean, np.float32),
        "trace_beta_SG": np.asarray(diagnostic.trace_beta_SG, np.float32),
    }
    if diagnostic.cfg.use_mode_H:
        arrays.update({
            "trace_mode_H_mean": np.asarray(diagnostic.trace_mode_H_mean, np.float32),
            "trace_mode_H_max": np.asarray(diagnostic.trace_mode_H_max, np.float32),
            "trace_mode_H_rate_max_hz": np.asarray(diagnostic.trace_mode_H_rate_max_hz, np.float32),
            "trace_mode_H_drive_mean": np.asarray(diagnostic.trace_mode_H_drive_mean, np.float32),
            "trace_mode_H_drive_max": np.asarray(diagnostic.trace_mode_H_drive_max, np.float32),
            "trace_mode_H_gain_mean": np.asarray(diagnostic.trace_mode_H_gain_mean, np.float32),
            "trace_mode_H_gain_max": np.asarray(diagnostic.trace_mode_H_gain_max, np.float32),
            "trace_mode_H_gain_core_mean": np.asarray(
                diagnostic.trace_mode_H_gain_core_mean, np.float32
            ),
            "trace_mode_H_persistent_g_mean": np.asarray(
                diagnostic.trace_mode_H_persistent_g_mean, np.float32
            ),
            "trace_mode_H_persistent_g_max": np.asarray(
                diagnostic.trace_mode_H_persistent_g_max, np.float32
            ),
            "trace_mode_H_persistent_g_core_mean": np.asarray(
                diagnostic.trace_mode_H_persistent_g_core_mean, np.float32
            ),
        })
    if diagnostic.cfg.use_mode_M_divisive:
        arrays.update({
            "trace_mode_M_raw_pool": np.asarray(
                diagnostic.trace_mode_M_raw_pool, np.float32
            ),
            "trace_mode_M_pool": np.asarray(diagnostic.trace_mode_M_pool, np.float32),
            "trace_mode_M_memory": np.asarray(
                diagnostic.trace_mode_M_memory, np.float32
            ),
            "trace_mode_M_divisor": np.asarray(
                diagnostic.trace_mode_M_divisor, np.float32
            ),
        })
    if diagnostic.cfg.use_zm_conductance_homotopy:
        arrays.update({
            "trace_cond_lambda_mean": np.asarray(
                diagnostic.trace_cond_lambda_mean, np.float32
            ),
            "trace_cond_lambda_max": np.asarray(
                diagnostic.trace_cond_lambda_max, np.float32
            ),
            "trace_cond_lambda_core_mean": np.asarray(
                diagnostic.trace_cond_lambda_core_mean, np.float32
            ),
            "trace_cond_vinf_mean": np.asarray(
                diagnostic.trace_cond_vinf_mean, np.float32
            ),
            "trace_cond_tau_eff_mean": np.asarray(
                diagnostic.trace_cond_tau_eff_mean, np.float32
            ),
        })
    namespace = (
        "smoke" if args.smoke else
        ("lifecycle_sprint" if args.command == "sprint-cell" else
         ("dynamic" if dynamic else ("race" if race else "discovery")))
    )
    if race or dynamic:
        stem = _mechanism_stem(args)
        if dynamic:
            stem += f"__T{float(args.T_ms) / 1000:g}s"
            if float(args.g_Z) != 1.0:
                stem += f"__gZ{args.g_Z:g}"
        if dynamic and (float(args.g_M) != 1.0 or args.tau_M_ms is not None):
            stem += f"__gM{args.g_M:g}"
            if args.tau_M_ms is not None:
                stem += f"__tauM{args.tau_M_ms:g}"
    else:
        stem = f"{args.state}__tau{args.tau_phi_ms:g}__f{args.fraction:g}"
    root = OUT / namespace / "seed1" / stem
    npz_path = root / "traces.npz"
    json_path = root / "summary.json"
    _write_npz_once(npz_path, arrays)
    payload = {
        "schema": "zm_fast_lifecycle_development_cell_v1_2026-08-01",
        "stage": (
            "joint_dynamic_lifecycle_sprint" if args.command == "sprint-cell" else
            ("dynamic_lifecycle_prototype" if dynamic else
            ("mechanism_race" if race else "A_branch_intervention")
            )
        ),
        "semantic_scope": (
            "seed1_joint_fast_M_control_development" if args.command == "sprint-cell" else
            ("seed1_dynamic_entry_offset_recovery_prototype" if dynamic else
            "branch_intervention_not_reachability"
            )
        ),
        "seed": 1,
        "state": (
            args.state
            if not dynamic or bool(getattr(args, "freeze_zm", False))
            else "pre_entry__natural"
        ),
        "source_t_ms": float(row["t_ms"]),
        "native_onset_reference_t_ms": 8700.0,
        "warm_start_to_native_onset_ms": float(8700.0 - row["t_ms"]),
        "equilibration_ms": float(args.burn_ms) if dynamic else None,
        "tau_phi_ms": float(args.tau_phi_ms),
        "fraction": float(args.fraction),
        "mechanism": mechanism,
        "finite_control": (
            None if perturb is None else {
                "kind": "E_threshold_uplift",
                "target": args.control_target,
                "t0_ms": float(args.control_t0_ms),
                "clock": CONTROL_CLOCK_VERSION,
                "engine_t0_ms": float(control_engine_window[0]),
                "engine_t1_ms": float(control_engine_window[1]),
                "duration_ms": float(args.control_duration_ms),
                "uplift_mV": float(args.control_uplift_mV),
            }
        ),
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
        "runtime_git_sha": launch_git_sha,
        "git_sha_capture_semantics": "captured_before_context_build_and_simulation",
        "use_zm_conductance": False,
        "use_zm_conductance_homotopy": bool(
            diagnostic.cfg.use_zm_conductance_homotopy
        ),
        "freeze_policy": R.FS.FreezePolicy.for_arm(
            "freeze_zm"
            if bool(getattr(args, "freeze_zm", False)) or not dynamic
            else "dynamic_replay"
        ).as_dict(),
        "phi_initial_nonzero": 0,
        "phi_final_mean_mV": float(np.mean(phi_e)),
        "phi_final_max_mV": float(np.max(phi_e)),
        "phi_i_nonzero": int(np.count_nonzero(phi_i)),
        "i_adaptation_E_nonzero": int(
            np.count_nonzero(i_adaptation[: ctx["S"]["NE"]])
        ),
        "i_adaptation_final_mean_mV": float(
            np.mean(i_adaptation[ctx["S"]["NE"] :])
        ),
        "i_adaptation_final_max_mV": float(
            np.max(i_adaptation[ctx["S"]["NE"] :])
        ),
        "i2e_resource_final_mean": float(np.mean(i2e_resource)),
        "i2e_resource_final_min": float(np.min(i2e_resource)),
        "z_max_abs_drift": z_drift,
        "m_max_abs_drift": m_drift,
        "S_G_initial": sg0,
        "S_G_final": float(diagnostic.S_G),
        "mode_H_final_mean": (
            float(diagnostic.mode_H.mean()) if diagnostic.cfg.use_mode_H else None
        ),
        "mode_H_final_max": (
            float(diagnostic.mode_H.max()) if diagnostic.cfg.use_mode_H else None
        ),
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
        "dI_mean": payload["i2e_resource_final_mean"],
        "aI_mean": payload["i_adaptation_final_mean_mV"],
        "S_G_final": payload["S_G_final"],
        "runaway_ms": payload["runaway_early_stop_ms"],
        "wall_s": payload["wall_s"],
        "rss_gb": payload["peak_rss_gb"],
    }, sort_keys=True))
    return json_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cell", "race-cell", "dynamic-cell", "sprint-cell"))
    parser.add_argument("--state", choices=FROZEN_MODE_STATES, default=RACE_STATE)
    parser.add_argument("--tau-phi-ms", type=float, default=RACE_PHI_TAU_MS)
    parser.add_argument("--fraction", type=float, default=RACE_PHI_FRACTION)
    parser.add_argument("--arm", choices=RACE_ARMS, default="phi")
    parser.add_argument("--tau-D-ms", type=float, dest="tau_D_ms")
    parser.add_argument("--d-star", type=float)
    parser.add_argument("--i2e-tau-cv", type=float, default=0.0)
    parser.add_argument("--i2e-tau-seed", type=int, default=0)
    parser.add_argument("--i2e-delay-scale", type=float, default=1.0)
    parser.add_argument("--i2e-delay-cv", type=float, default=0.0)
    parser.add_argument("--i2e-delay-seed", type=int, default=0)
    parser.add_argument("--use-dual-gaba", action="store_true")
    parser.add_argument("--dual-gaba-slow-fraction", type=float, default=0.35)
    parser.add_argument("--dual-gaba-sigma-mm", type=float, default=1.5)
    parser.add_argument("--dual-gaba-in-degree", type=int, default=64)
    parser.add_argument("--dual-gaba-candidate-count", type=int, default=256)
    parser.add_argument("--dual-gaba-seed", type=int, default=0)
    parser.add_argument("--dual-gaba-tau-r-ms", type=float, default=4.0)
    parser.add_argument("--dual-gaba-tau-d-ms", type=float, default=60.0)
    parser.add_argument("--use-inhibitory-subtypes", action="store_true")
    parser.add_argument("--som-source-fraction", type=float, default=0.25)
    parser.add_argument("--som-slow-budget-fraction", type=float, default=0.35)
    parser.add_argument("--som-sigma-mm", type=float, default=1.5)
    parser.add_argument("--som-in-degree", type=int, default=64)
    parser.add_argument("--som-candidate-count", type=int, default=256)
    parser.add_argument("--som-recruit-delay-scale", type=float, default=3.0)
    parser.add_argument("--som-seed", type=int, default=0)
    parser.add_argument("--som-tau-r-ms", type=float, default=4.0)
    parser.add_argument("--som-tau-d-ms", type=float, default=60.0)
    parser.add_argument("--som-shunting", action="store_true")
    parser.add_argument("--som-shunt-scale", type=float, default=1.0)
    parser.add_argument("--som-e-gaba-mv", type=float, default=11.0)
    parser.add_argument("--tau-aI-ms", type=float, dest="tau_aI_ms")
    parser.add_argument("--f-aI", type=float, dest="f_aI")
    parser.add_argument("--strength-scale", type=float)
    parser.add_argument("--control-uplift-mV", type=float, default=0.0)
    parser.add_argument("--control-t0-ms", type=float)
    parser.add_argument("--control-duration-ms", type=float, default=50.0)
    parser.add_argument("--control-target", choices=("all_E", "core_E"), default="all_E")
    parser.add_argument("--g-M", type=float, dest="g_M", default=1.0)
    parser.add_argument("--tau-M-ms", type=float, dest="tau_M_ms")
    parser.add_argument("--g-Z", type=float, dest="g_Z", default=1.0)
    parser.add_argument("--use-mode-H", action="store_true")
    parser.add_argument(
        "--freeze-zm", action="store_true",
        help="freeze native z/m at --state while leaving fast E/I, S_G and mode-H dynamic",
    )
    parser.add_argument("--rho-mode-H", type=float, default=0.0)
    parser.add_argument("--mode-H-persistent-g-max", type=float, default=0.0)
    parser.add_argument("--mode-H-persistent-e-exc-mv", type=float, default=60.0)
    parser.add_argument("--mode-H-common-subtraction", type=float, default=0.0)
    parser.add_argument("--tau-mode-H-ms", type=float, default=250.0)
    parser.add_argument("--tau-mode-H-down-ms", type=float, default=250.0)
    parser.add_argument("--theta-mode-H-hz", type=float, default=40.0)
    parser.add_argument("--half-mode-H-hz", type=float, default=40.0)
    parser.add_argument("--z-mode-base", type=float, default=1.0)
    parser.add_argument("--z-mode-susceptible", type=float, default=0.5)
    parser.add_argument("--zeta-mode-center", type=float, default=0.5)
    parser.add_argument("--zeta-mode-slope", type=float, default=0.1)
    parser.add_argument("--m-mode-half", type=float, default=45.0)
    parser.add_argument("--m-mode-power", type=float, default=4.0)
    parser.add_argument("--use-mode-M-divisive", action="store_true")
    parser.add_argument("--kappa-mode-M", type=float, default=0.0)
    parser.add_argument("--m-mode-div-ref", type=float, default=30.0)
    parser.add_argument("--m-mode-div-power", type=float, default=4.0)
    parser.add_argument("--m-mode-div-hill-power", type=float, default=4.0)
    parser.add_argument("--use-mode-M-memory", action="store_true")
    parser.add_argument("--tau-mode-M-memory-up-ms", type=float, default=3000.0)
    parser.add_argument("--tau-mode-M-memory-down-ms", type=float, default=8000.0)
    parser.add_argument(
        "--beta-SG", type=float, dest="beta_SG", default=0.0,
        help="subtractive shared-pool current on E cells; 0 keeps the parity path",
    )
    parser.add_argument(
        "--beta-SG-ramp-per-s", type=float, dest="beta_SG_ramp_per_s", default=0.0,
        help="KINEMATIC PROBE: walk --beta-SG linearly at this rate per second. "
             "Open loop; nothing in the model generates it, so a run using it is "
             "a trajectory probe and never a carrier result.",
    )
    parser.add_argument("--use-zm-conductance-homotopy", action="store_true")
    parser.add_argument("--homotopy-z-native", type=float, default=0.60)
    parser.add_argument("--homotopy-z-conductance", type=float, default=0.40)
    parser.add_argument("--cond-kappa-E", type=float, default=0.02917206399)
    parser.add_argument("--cond-kappa-I", type=float, default=0.01687781464)
    parser.add_argument("--cond-g-M", type=float, default=1.76213078e-5)
    parser.add_argument("--cond-gamma", type=float, default=1.0 / 6.0)
    parser.add_argument("--T-ms", type=float, default=PRODUCTION_T_MS)
    parser.add_argument("--burn-ms", type=float, default=PRODUCTION_BURN_MS)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("refusing SNN development run without --confirm-run")
    config_text = json.dumps(vars(args), sort_keys=True, separators=(",", ":"))
    config_hash = hashlib.sha256(config_text.encode()).hexdigest()
    receipt_path = (
        OUT / "worker_receipts" /
        f"{args.command}__{config_hash[:16]}.json"
    )
    with WR.WorkerReceipt(
        receipt_path,
        config_hash=config_hash,
        git_sha=RT.git_sha(ROOT),
        command=" ".join(sys.argv),
    ) as receipt:
        artifact = run_cell(args, worker_receipt=receipt)
        result = json.loads(artifact.read_text())
        terminal = (
            "scientific_early_stop"
            if result.get("runaway_early_stop_ms") is not None else "success"
        )
        receipt.finish(terminal, artifact_path=str(artifact.relative_to(ROOT)))


if __name__ == "__main__":
    main()
