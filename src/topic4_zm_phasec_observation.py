"""Observation-only helpers for the Z/M Phase-C branch-identity audit.

Nothing in this module advances simulator state.  ``PhaseCCurrentRecorder``
wraps the existing current-based virtual-SEEG recorder and returns its sample
bit-for-bit while retaining a few low-memory *raw synaptic* current summaries.
The checkpoint helpers operate on an already captured exact state.

Terminology is deliberately strict:

* ``raw_synaptic`` means the pre-Z/M currents ``I_E`` and ``I_I`` that the
  existing LFP proxy sees;
* ``effective_snapshot`` means a reconstruction of the canonical Z/M+S_G
  membrane-drive formula at one checkpoint;
* neither quantity is called a transmembrane current.
"""
from __future__ import annotations

import numpy as np


PHASEC_OBSERVATION_VERSION = "zm_phasec_observation_v1_2026-07-28"


class PhaseCEffectiveSlowObserver:
    """Trajectory-transparent observer around ``SpatialSlowField``.

    Place this wrapper *inside* the existing ``FreezeWrapper``.  It calls the
    real ``apply_currents`` exactly once, returns that exact array, and records
    compact means of the terms that actually entered the E-cell membrane
    equation.  ``__setattr__`` delegates slow-state writes so the existing
    checkpoint restore logic still lands on the real slow object.
    """

    _LOCAL = {
        "inner", "core_mask_E", "stride_steps", "_step", "_sample_steps",
        "_traces",
    }

    def __init__(self, inner, core_mask_E, *, stride_steps=5):
        object.__setattr__(self, "inner", inner)
        nE = int(inner.nE)
        core = np.asarray(core_mask_E, bool)
        if core.shape != (nE,) or not np.any(core) or np.all(core):
            raise ValueError("core_mask_E must define non-empty E core and surround")
        if int(stride_steps) < 1:
            raise ValueError("stride_steps must be at least 1")
        object.__setattr__(self, "core_mask_E", core)
        object.__setattr__(self, "stride_steps", int(stride_steps))
        object.__setattr__(self, "_step", 0)
        object.__setattr__(self, "_sample_steps", [])
        names = (
            "raw_excitation", "raw_inhibition", "recurrent_excitation",
            "effective_excitation", "effective_inhibition_z",
            "adaptation_m", "effective_outward_total", "effective_net_drive",
        )
        object.__setattr__(
            self,
            "_traces",
            {f"{name}_{region}_mean_mV": [] for name in names
             for region in ("all", "core", "surround")},
        )

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __setattr__(self, name, value):
        if name in self._LOCAL or name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            setattr(self.inner, name, value)

    def threshold(self, V_th_base):
        return self.inner.threshold(V_th_base)

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        out = self.inner.apply_currents(I_E, I_I, labels, I_E_rec)
        if self._step % self.stride_steps == 0:
            nE = int(self.inner.nE)
            raw_exc = np.asarray(I_E, float)[:nE]
            raw_inh = np.asarray(I_I, float)[:nE]
            rec = (
                np.zeros(nE, float)
                if I_E_rec is None
                else np.asarray(I_E_rec, float)[:nE]
            )
            cfg = self.inner.cfg
            z = np.asarray(self.inner.z, float)[:nE]
            m = np.asarray(self.inner.m, float)[:nE]
            aS = float(cfg.alpha_G) * float(self.inner.S_G)
            aH = (
                float(cfg.alpha_H) * float(self.inner.H)
                if bool(cfg.use_H) else 0.0
            )
            load = aS + aH
            removed = rec * load / (1.0 + load)
            effective_exc = raw_exc - removed
            effective_inh = z * raw_inh
            adaptation = float(cfg.eta_m) * m
            subtractive_sg = float(cfg.beta_SG) * float(self.inner.S_G)
            outward = effective_inh + adaptation + subtractive_sg
            terms = {
                "raw_excitation": raw_exc,
                "raw_inhibition": raw_inh,
                "recurrent_excitation": rec,
                "effective_excitation": effective_exc,
                "effective_inhibition_z": effective_inh,
                "adaptation_m": adaptation,
                "effective_outward_total": outward,
                "effective_net_drive": np.asarray(out, float)[:nE],
            }
            core = self.core_mask_E
            for name, values in terms.items():
                for region, mask in (
                    ("all", np.ones(nE, bool)),
                    ("core", core),
                    ("surround", ~core),
                ):
                    self._traces[f"{name}_{region}_mean_mV"].append(
                        float(np.mean(values[mask]))
                    )
            self._sample_steps.append(self._step)
        self._step += 1
        return out

    def traces(self, *, dt_ms=None):
        payload = {
            "phasec_observation_version": PHASEC_OBSERVATION_VERSION,
            "evidence_label": "effective_membrane_drive",
            "sample_step": np.asarray(self._sample_steps, np.int64),
            "stride_steps": int(self.stride_steps),
            "n_recorded": len(self._sample_steps),
            "claim_boundary": (
                "terms entering the canonical current-based LIF membrane "
                "equation; not biophysical transmembrane conductances"
            ),
        }
        payload.update({
            key: np.asarray(value, np.float32)
            for key, value in self._traces.items()
        })
        if dt_ms is not None:
            dt = float(dt_ms)
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("dt_ms must be finite and positive")
            payload["sample_dt_ms"] = dt * self.stride_steps
            payload["sample_time_ms"] = payload["sample_step"].astype(float) * dt
        return payload


class PhaseCCurrentRecorder:
    """Transparent wrapper around an existing LFP recorder.

    ``sample`` calls the inner recorder exactly once and returns that object
    unchanged.  Every ``stride_steps`` calls it also records raw AMPA/GABA means
    over all E cells, the supplied core, and its surround.  It never receives or
    modifies simulator RNG/state.
    """

    def __init__(self, inner, core_mask_E, *, stride_steps=5):
        if not hasattr(inner, "sample") or not hasattr(inner, "sites"):
            raise TypeError("inner must expose sample(I_E,I_I) and sites")
        if not hasattr(inner, "NE"):
            raise TypeError("inner must expose NE")
        self.inner = inner
        self.sites = inner.sites
        self.NE = int(inner.NE)
        self.core_mask_E = np.asarray(core_mask_E, bool)
        if self.core_mask_E.shape != (self.NE,):
            raise ValueError(f"core_mask_E must have shape ({self.NE},)")
        if not np.any(self.core_mask_E) or np.all(self.core_mask_E):
            raise ValueError("core_mask_E must define non-empty core and surround")
        self.stride_steps = int(stride_steps)
        if self.stride_steps < 1:
            raise ValueError("stride_steps must be at least 1")
        self._step = 0
        self._sample_steps = []
        self._raw_ampa_all = []
        self._raw_gaba_all = []
        self._raw_ampa_core = []
        self._raw_gaba_core = []
        self._raw_ampa_surround = []
        self._raw_gaba_surround = []

    def sample(self, I_E, I_I):
        """Return the inner virtual-SEEG sample without numerical alteration."""
        result = self.inner.sample(I_E, I_I)
        if self._step % self.stride_steps == 0:
            exc = np.asarray(I_E, float)[:self.NE]
            inh = np.asarray(I_I, float)[:self.NE]
            if exc.shape != (self.NE,) or inh.shape != (self.NE,):
                raise ValueError("I_E/I_I are shorter than the recorder E population")
            core = self.core_mask_E
            self._sample_steps.append(self._step)
            self._raw_ampa_all.append(float(np.mean(exc)))
            self._raw_gaba_all.append(float(np.mean(inh)))
            self._raw_ampa_core.append(float(np.mean(exc[core])))
            self._raw_gaba_core.append(float(np.mean(inh[core])))
            self._raw_ampa_surround.append(float(np.mean(exc[~core])))
            self._raw_gaba_surround.append(float(np.mean(inh[~core])))
        self._step += 1
        return result

    def traces(self, *, dt_ms=None):
        """Return compact raw-synaptic traces; no full-neuron arrays are retained."""
        amp_all = np.asarray(self._raw_ampa_all, dtype=np.float32)
        gab_all = np.asarray(self._raw_gaba_all, dtype=np.float32)
        amp_core = np.asarray(self._raw_ampa_core, dtype=np.float32)
        gab_core = np.asarray(self._raw_gaba_core, dtype=np.float32)
        amp_surround = np.asarray(self._raw_ampa_surround, dtype=np.float32)
        gab_surround = np.asarray(self._raw_gaba_surround, dtype=np.float32)

        def ratio(gaba, ampa):
            return np.divide(
                gaba,
                ampa,
                out=np.full(ampa.shape, np.nan, dtype=np.float32),
                where=ampa != 0,
            )

        out = {
            "phasec_observation_version": PHASEC_OBSERVATION_VERSION,
            "evidence_label": "raw_synaptic",
            "sample_step": np.asarray(self._sample_steps, dtype=np.int64),
            "raw_ampa_all_mean_mV": amp_all,
            "raw_gaba_all_mean_mV": gab_all,
            "raw_gaba_to_ampa_all_ratio": ratio(gab_all, amp_all),
            "raw_ampa_core_mean_mV": amp_core,
            "raw_gaba_core_mean_mV": gab_core,
            "raw_gaba_to_ampa_core_ratio": ratio(gab_core, amp_core),
            "raw_ampa_surround_mean_mV": amp_surround,
            "raw_gaba_surround_mean_mV": gab_surround,
            "raw_gaba_to_ampa_surround_ratio": ratio(gab_surround, amp_surround),
            "stride_steps": self.stride_steps,
            "n_recorded": len(self._sample_steps),
            "claim_boundary": (
                "pre-Z/M raw synaptic-current summaries; not effective or "
                "transmembrane currents"
            ),
        }
        if dt_ms is not None:
            dt = float(dt_ms)
            if not np.isfinite(dt) or dt <= 0:
                raise ValueError("dt_ms must be finite and positive")
            out["sample_dt_ms"] = dt * self.stride_steps
            out["sample_time_ms"] = out["sample_step"].astype(float) * dt
        return out


def raw_synaptic_lag(
    traces,
    *,
    dt_ms=None,
    region="core",
    max_lag_ms=50.0,
):
    """Lag of raw GABA behind raw AMPA.

    Positive lag means the GABA trace is best aligned when shifted *later* than
    AMPA.  The function uses normalized correlation over each overlapping
    segment and fails closed on flat or too-short traces.
    """
    if not isinstance(traces, dict) or traces.get("evidence_label") != "raw_synaptic":
        raise ValueError("raw_synaptic traces are required")
    if region not in {"all", "core", "surround"}:
        raise ValueError("region must be all|core|surround")
    exc = np.asarray(traces[f"raw_ampa_{region}_mean_mV"], float)
    inh = np.asarray(traces[f"raw_gaba_{region}_mean_mV"], float)
    if exc.ndim != 1 or inh.shape != exc.shape or exc.size < 5:
        return {
            "status": "insufficient_trace",
            "evidence_label": "raw_synaptic",
            "lag_ms": None,
            "peak_correlation": None,
        }
    if dt_ms is None:
        dt_ms = traces.get("sample_dt_ms")
    dt = float(dt_ms)
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("a positive sampling dt is required")
    max_steps = min(exc.size - 2, int(round(float(max_lag_ms) / dt)))
    if max_steps < 0:
        raise ValueError("max_lag_ms must be nonnegative")
    rows = []
    for lag in range(-max_steps, max_steps + 1):
        if lag > 0:
            x, y = exc[:-lag], inh[lag:]
        elif lag < 0:
            x, y = exc[-lag:], inh[:lag]
        else:
            x, y = exc, inh
        sx, sy = float(np.std(x)), float(np.std(y))
        if x.size < 3 or sx <= 0 or sy <= 0:
            continue
        rows.append((lag, float(np.corrcoef(x, y)[0, 1])))
    if not rows:
        return {
            "status": "degenerate_trace",
            "evidence_label": "raw_synaptic",
            "lag_ms": None,
            "peak_correlation": None,
        }
    lag, corr = max(rows, key=lambda row: row[1])
    return {
        "status": "ok",
        "evidence_label": "raw_synaptic",
        "region": region,
        "lag_steps": int(lag),
        "lag_ms": float(lag * dt),
        "peak_correlation": float(corr),
        "sign_convention": "positive means raw GABA trails raw AMPA",
        "max_lag_ms": float(max_lag_ms),
    }


def _state_vector(state, key, nE):
    if key not in state:
        raise KeyError(f"checkpoint missing {key!r}")
    a = np.asarray(state[key], float)
    if a.ndim != 1 or a.size < int(nE):
        raise ValueError(f"{key} must be a 1D array with at least nE entries")
    return a[:int(nE)]


def reconstruct_effective_snapshot(
    state,
    *,
    nE,
    alpha_G,
    eta_m,
    alpha_H=0.0,
    beta_SG=0.0,
    expected_net=None,
    atol=1e-12,
):
    """Reconstruct the canonical Z/M+S_G E-cell drive at one exact checkpoint.

    For the locked family:

    ``I_exc_eff = I_E - I_E_rec*(aG*S_G+aH*H)/(1+aG*S_G+aH*H)``

    ``I_out_eff = z*I_I + eta_m*m + beta_SG*S_G``

    ``I_net = I_exc_eff - I_out_eff``.

    Unsupported additional current mechanisms must be excluded by the caller's
    configuration gate.  ``expected_net`` may be supplied by a separately
    observed formula path; a mismatch raises rather than silently reporting an
    approximate decomposition.
    """
    nE = int(nE)
    if nE < 1:
        raise ValueError("nE must be positive")
    alpha_G = float(alpha_G)
    alpha_H = float(alpha_H)
    beta_SG = float(beta_SG)
    eta_m = float(eta_m)
    if not np.all(np.isfinite([alpha_G, alpha_H, beta_SG, eta_m])):
        raise ValueError("current coefficients must be finite")
    if min(alpha_G, alpha_H, beta_SG, eta_m) < 0:
        raise ValueError("alpha_G, alpha_H, beta_SG and eta_m must be nonnegative")

    raw_exc = _state_vector(state, "I_E", nE)
    raw_inh = _state_vector(state, "I_I", nE)
    rec_exc = _state_vector(state, "I_E_rec", nE)
    z = _state_vector(state, "slow.z", nE)
    m = _state_vector(state, "slow.m", nE)
    if "slow.S_G" not in state:
        raise KeyError("checkpoint missing 'slow.S_G'")
    S_G = float(np.asarray(state["slow.S_G"]))
    H = float(np.asarray(state.get("slow.H", 0.0)))
    if not np.all(np.isfinite([S_G, H])) or S_G < 0 or H < 0:
        raise ValueError("slow.S_G/slow.H must be finite")
    if np.any((z < 0) | (z > 1)) or np.any(m < 0):
        raise ValueError("canonical snapshot requires z in [0,1] and m>=0")

    divisive_load = alpha_G * S_G + alpha_H * H
    recurrent_removed = rec_exc * divisive_load / (1.0 + divisive_load)
    effective_exc = raw_exc - recurrent_removed
    effective_inh_z = z * raw_inh
    adaptation_m = eta_m * m
    subtractive_sg = np.full(nE, beta_SG * S_G)
    effective_outward = effective_inh_z + adaptation_m + subtractive_sg
    effective_net = effective_exc - effective_outward

    # Independent literal expansion of SpatialSlowField.apply_currents for the
    # canonical Z/M+S_G family.  This assertion guards sign and term omission.
    formula_net = (
        raw_exc
        - effective_inh_z
        - adaptation_m
        - recurrent_removed
        - subtractive_sg
    )
    if not np.allclose(effective_net, formula_net, rtol=0.0, atol=float(atol)):
        raise AssertionError("effective-component identity does not reconstruct formula net")
    if expected_net is not None:
        expected = np.asarray(expected_net, float)
        if expected.shape != (nE,):
            raise ValueError(f"expected_net must have shape ({nE},)")
        if not np.allclose(effective_net, expected, rtol=0.0, atol=float(atol)):
            raise AssertionError("reconstructed effective net != expected formula net")

    return {
        "phasec_observation_version": PHASEC_OBSERVATION_VERSION,
        "evidence_label": "effective_snapshot",
        "raw_ampa_mV": raw_exc,
        "raw_gaba_mV": raw_inh,
        "recurrent_ampa_removed_by_SG_mV": recurrent_removed,
        "effective_excitation_mV": effective_exc,
        "effective_inhibition_z_mV": effective_inh_z,
        "adaptation_m_current_mV": adaptation_m,
        "subtractive_SG_current_mV": subtractive_sg,
        "effective_outward_total_mV": effective_outward,
        "effective_net_drive_mV": effective_net,
        "S_G": S_G,
        "H": H,
        "divisive_load": float(divisive_load),
        "identity_max_abs_error_mV": float(np.max(np.abs(effective_net - formula_net))),
        "claim_boundary": (
            "canonical Z/M+S_G effective drive reconstructed at one exact "
            "checkpoint; not a transmembrane-current time series"
        ),
    }


def _quantile_summary(x, quantiles):
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    if not a.size:
        return {
            "n": 0,
            "mean_mV": None,
            "quantiles_mV": {str(float(q)): None for q in quantiles},
        }
    vals = np.percentile(a, quantiles)
    return {
        "n": int(a.size),
        "mean_mV": float(np.mean(a)),
        "quantiles_mV": {
            str(float(q)): float(v) for q, v in zip(quantiles, vals)
        },
        "fraction_within_0p5mV": float(np.mean((a >= 0.0) & (a <= 0.5))),
        "fraction_within_1mV": float(np.mean((a >= 0.0) & (a <= 1.0))),
        "fraction_above_threshold": float(np.mean(a < 0.0)),
    }


def free_e_threshold_margin_snapshot(
    state,
    vth_per_neuron,
    *,
    nE,
    core_mask_E=None,
    quantiles=(1, 5, 25, 50, 75, 95, 99),
):
    """Distribution of ``V_th - V`` among free E cells in an exact snapshot.

    Checkpoints store the end-of-step, post-reset membrane state.  Therefore
    this is a natural-phase *snapshot margin*, not a pre-spike trajectory or a
    continuous voltage trace.
    """
    nE = int(nE)
    V = _state_vector(state, "V", nE)
    if "ref" not in state:
        raise KeyError("checkpoint missing 'ref'")
    ref = np.asarray(state["ref"])
    if ref.ndim != 1 or ref.size < nE:
        raise ValueError("ref must be a 1D array with at least nE entries")
    ref = ref[:nE]
    vth = np.asarray(vth_per_neuron, float)
    if vth.ndim == 0:
        vth = np.full(nE, float(vth))
    elif vth.ndim == 1 and vth.size >= nE:
        vth = vth[:nE]
    else:
        raise ValueError("vth_per_neuron must be scalar or a vector with at least nE entries")
    free = ref == 0
    margin = vth - V
    out = {
        "phasec_observation_version": PHASEC_OBSERVATION_VERSION,
        "evidence_label": "effective_snapshot",
        "snapshot_quantity": "free_E_Vth_minus_V",
        "all_free_E": _quantile_summary(margin[free], quantiles),
        "free_E_fraction": float(np.mean(free)),
        "nE": nE,
        "claim_boundary": (
            "post-step exact-state voltage margin at selected natural phases; "
            "not a continuous pre-spike or transmembrane-current trace"
        ),
    }
    if core_mask_E is not None:
        core = np.asarray(core_mask_E, bool)
        if core.shape != (nE,):
            raise ValueError(f"core_mask_E must have shape ({nE},)")
        out["core_free_E"] = _quantile_summary(margin[free & core], quantiles)
        out["surround_free_E"] = _quantile_summary(margin[free & ~core], quantiles)
    else:
        out["core_free_E"] = _quantile_summary([], quantiles)
        out["surround_free_E"] = _quantile_summary([], quantiles)
    return out
