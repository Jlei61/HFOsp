"""Runtime helpers shared by Phase-D parity, calibration and carrier forks."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np


def git_sha(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def build_source_locked_context(root: Path, manifest: dict, runner: Any) -> dict:
    """Rebuild static substrate from its old lock without rewriting the lock."""
    source = manifest["source"]
    lock = json.loads((root / source["canonical_config_path"]).read_text())
    seed = int(source["seed"])
    parent = lock["seeds"][str(seed)]
    if parent["config_sha"] != source["canonical_config_sha"]:
        raise RuntimeError("source canonical seed lock drift")
    dt = float(source["dt_ms"])
    static = runner.PP.build_substrate(seed=seed, dt=dt)
    static["seed"] = seed
    static["I_th_EI"] = float(parent["config"]["I_th_EI"])
    montage = static["reg"]["montage_sheet"]
    recorder = runner.LFPRecorder(
        static["p"],
        static["net"]["pos"],
        static["net"]["labels"],
        sites=np.asarray(montage.contacts, float),
    )
    core = runner.ZM._core_mask_E(static)
    along, _ = runner.CG.axis_transverse_coords(
        static["posE"], static["src_xy"], static["axis_unit"]
    )
    return {
        "S": static,
        "rec": recorder,
        "core": core,
        "axis": along,
        "contacts": list(montage.names),
        "cfg_locked": parent["config"],
        "cfg_sha": parent["config_sha"],
        "smoke": False,
        "resolution": "dt",
        "dt": dt,
        "anchor_root": "anchors",
        "runtime_git_sha": git_sha(root),
        "runtime_started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


class DiagnosticSlowWrapper:
    """Observation-only wrapper; every numerical method delegates literally."""

    def __init__(self, inner):
        object.__setattr__(self, "inner", inner)
        object.__setattr__(self, "trace_vinf_median", [])
        object.__setattr__(self, "trace_tau_eff_median", [])
        object.__setattr__(self, "trace_exc_charge_mean", [])
        object.__setattr__(self, "trace_inh_charge_mean", [])
        object.__setattr__(self, "trace_vinf_above_EI", [])
        object.__setattr__(self, "trace_v_above_EI", [])

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if name == "inner" or name.startswith("trace_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        out = self.inner.apply_currents(I_E, I_I, labels, I_E_rec)
        n_e = int(self.inner.nE)
        cfg = self.inner.cfg
        exc = np.asarray(I_E, float)[:n_e].copy()
        if cfg.use_SG and I_E_rec is not None:
            a_s = cfg.alpha_G * self.inner.S_G
            a_h = cfg.alpha_H * self.inner.H if cfg.use_H else 0.0
            frac = (a_s + a_h) / (1.0 + a_s + a_h)
            exc -= np.asarray(I_E_rec, float)[:n_e] * frac
            exc -= cfg.beta_SG * self.inner.S_G
        inh = np.asarray(I_I, float)[:n_e]
        if cfg.use_z:
            inh = self.inner.z[:n_e] * inh
        self.trace_vinf_median.append(float(np.median(out[:n_e])))
        self.trace_tau_eff_median.append(float(self.inner.cfg.cond_tau_m_E))
        self.trace_exc_charge_mean.append(float(np.mean(np.abs(exc))))
        self.trace_inh_charge_mean.append(float(np.mean(np.abs(inh))))
        self.trace_vinf_above_EI.append(float("nan"))
        self.trace_v_above_EI.append(float("nan"))
        return out

    def zm_conductance_step(self, V, I_E, I_I, decay_V):
        out = self.inner.zm_conductance_step(V, I_E, I_I, decay_V)
        e = self.inner.is_E
        cfg = self.inner.zm_conductance_config()
        self.trace_vinf_median.append(float(np.median(out["V_inf"][e])))
        self.trace_tau_eff_median.append(float(np.median(out["tau_eff_ms"][e])))
        self.trace_exc_charge_mean.append(float(np.mean(np.abs(out["I_exc"][e]))))
        self.trace_inh_charge_mean.append(float(np.mean(np.abs(out["I_inh"][e]))))
        self.trace_vinf_above_EI.append(float(np.mean(out["V_inf"][e] > cfg.E_I)))
        self.trace_v_above_EI.append(float(np.mean(np.asarray(V)[e] > cfg.E_I)))
        return out

    def diagnostic_summary(self) -> dict:
        def median(name):
            values = np.asarray(getattr(self, name), float)
            finite = values[np.isfinite(values)]
            return None if not finite.size else float(np.median(finite))

        exc = np.asarray(self.trace_exc_charge_mean, float)
        inh = np.asarray(self.trace_inh_charge_mean, float)
        ratio = float(np.sum(inh) / np.sum(exc)) if np.sum(exc) > 0 else None
        return {
            "median_vinf_mv": median("trace_vinf_median"),
            "median_tau_eff_ms": median("trace_tau_eff_median"),
            "effective_inhibitory_to_excitatory_charge_ratio": ratio,
            "median_fraction_vinf_above_EI": median("trace_vinf_above_EI"),
            "median_fraction_v_above_EI": median("trace_v_above_EI"),
            "n_steps": len(self.trace_vinf_median),
        }


def make_frozen_diagnostic_slow(
    ctx: dict,
    runner: Any,
    *,
    conductance_config: dict | None,
):
    """Build the exact Arm-A or conductance slow layer, frozen at checkpoint."""
    if conductance_config is None:
        cfg = runner.ZM._zm_cfg(ctx["S"]["I_th_EI"], **runner.ARM_KWARGS)
    else:
        cfg = runner.ZM._zm_cfg(ctx["S"]["I_th_EI"], use_SG=False, alpha_G=0.0)
        cfg = dataclasses.replace(
            cfg,
            use_zm_conductance=True,
            cond_kappa_E=float(conductance_config["kappa_E"]),
            cond_kappa_I=float(conductance_config["kappa_I"]),
            cond_g_M=float(conductance_config["g_M"]),
            cond_gamma=float(conductance_config["gamma"]),
            cond_z_spares_global=bool(conductance_config["z_spares_global"]),
            cond_g_L=float(conductance_config["g_L"]),
            cond_E_L=float(conductance_config["E_L"]),
            cond_E_E=float(conductance_config["E_E"]),
            cond_E_I=float(conductance_config["E_I"]),
            cond_E_K=float(conductance_config["E_K"]),
            cond_tau_m_E=float(conductance_config["tau_m_E"]),
        )
    base = runner.SpatialSlowField(
        ctx["S"]["N"],
        18.0,
        ctx["S"]["posE"],
        ctx["S"]["posI"],
        ctx["S"]["L"],
        core_mask_E=ctx["core"],
        cfg=cfg,
    )
    diagnostic = DiagnosticSlowWrapper(base)
    frozen = runner.FS.FreezeWrapper(
        diagnostic, runner.FS.FreezePolicy.for_arm("freeze_all")
    )
    return frozen, diagnostic


__all__ = [
    "DiagnosticSlowWrapper",
    "build_source_locked_context",
    "git_sha",
    "make_frozen_diagnostic_slow",
]
