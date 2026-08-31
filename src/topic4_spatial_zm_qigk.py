"""Spatial Z/qI--M hybrid for the frozen data-driven Topic 4 SNN.

This module uses one inhibitory-resource variable and one adaptation variable,
rather than stacking four biologically synonymous states.  The inhibitory side
is the qI spatial field (the spatial generalisation of per-neuron Z), while the
recovery side is the original per-neuron M current (the point-neuron analogue of
gK):

    dq(x)/dt = (1-q(x))/tau_q - k_q(x) F_n[K_q * r(x)] q(x)
    dm_i/dt = -m_i/tau_m + k_m[(1-rho_m) s_i + rho_m K_m*r_E(x_i)]
    I_i     = I_i^E - q(x_i) I_i^I - I_M[m_i;x_i],  i in E.

Both ``k_q(x)`` and ``eta_m(x)`` may be modulated by the frozen patient-derived
node field h(x).  Setting both h gains to zero gives homogeneous parameters but
the state remains spatial because it is driven by local activity.  Learned E-E
and E-I edges are not modified here.  ``F_n`` is the explicit Z/qI bridge:
``n=1`` recovers the historical smooth qI saturation, whereas increasing ``n``
approaches the threshold-like Z gate while retaining a continuous spatial field.
Likewise ``rho_m=0`` is the original per-neuron M history and ``rho_m=1`` is a
local gK-like population field sampled back onto the same M variables.
An optional non-negative M-current threshold is the gK activation gate: it
spares low-state fluctuations while allowing strong high-state feedback.  A
positive saturation width changes the current above threshold to a bounded Hill
gate, so ``eta_m(x)`` is a maximum local K-like current rather than an unbounded
linear penalty.  A zero width exactly preserves the historical linear M path.
An optional positive ``m_state_ceiling`` also bounds use-dependent accumulation
as ``dm_build ~ drive*(1-m/m_max)``, the state-level gK limit; zero keeps the
original unbounded M accumulator exactly.
``m_build_gain`` independently scales the M/gK build rate (the analogue of
``k_K``); its default of one exactly preserves the original per-spike M update,
while ``eta_m`` remains the membrane-current coupling.
For the fast-subsystem atlas only, ``q_init_h_gain`` can also seed a deterministic
nonuniform resource state from the same frozen ``h(x)`` field (positive gain =
lower initial q in high-h tissue).  ``q_endpoint_gain`` adds a second,
patient-constrained basis: the maximum of periodic Gaussian fields centred on
the frozen source- and sink-side endpoint contacts (positive gain = lower q at
the two propagation endpoint sets).  The contact centres are inputs, never
fitted to the simulated trajectory.  Zero gains exactly recover the homogeneous
initial condition; no per-neuron random parameter field is introduced.

The implementation intentionally composes the already tested qI field driver in
``src.snn_engine.slow_field``.  It is a development mechanism screen, not a
biophysical identification of chloride or potassium conductances.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sef_hfo_field import convolve_periodic, isotropic_gaussian
from src.snn_engine.slow_field import (
    SpatialSlowField,
    SpatialSlowFieldConfig,
    aq_drive,
)


@dataclass(frozen=True)
class SpatialZMQIGKConfig:
    n_grid: int = 64
    sigma_r_mm: float = 0.5
    tau_rate_ms: float = 20.0
    field_update_ms: float = 1.0
    tau_q_ms: float = 5000.0
    k_q_per_ms: float = 0.02
    q_min: float = 0.05
    q_init: float = 1.0
    q_init_h_gain: float = 0.0
    q_endpoint_gain: float = 0.0
    q_endpoint_sigma_mm: float = 2.0
    freeze_q: bool = False
    sigma_q_mm: float = 1.5
    q_a0: float = 0.0
    q_a50: float = 1.0
    q_hill_n: float = 1.0
    q_eta_e: float = 0.3
    q_eta_i: float = 1.0
    tau_m_ms: float = 62.5
    m_build_gain: float = 1.0
    eta_m: float = 0.05961275484469678
    m_current_threshold: float = 0.0
    m_current_saturation_width: float = 0.0
    m_current_hill_n: float = 1.0
    m_state_ceiling: float = 0.0
    m_spatial_mix: float = 0.0
    sigma_m_mm: float = 0.5
    k_q_h_gain: float = 0.0
    q_floor_h_gain: float = 0.0
    eta_m_h_gain: float = 0.0
    h_smooth_sigma_mm: float = 1.0
    trace_stride_steps: int = 10
    # Duck-typed by kick_probe.  This hybrid never changes recurrent E weights.
    use_SG: bool = False

    def validate(self) -> None:
        if self.n_grid < 2:
            raise ValueError("n_grid must be at least two")
        for name, value in (
            ("sigma_r_mm", self.sigma_r_mm),
            ("tau_rate_ms", self.tau_rate_ms),
            ("field_update_ms", self.field_update_ms),
            ("tau_q_ms", self.tau_q_ms),
            ("sigma_q_mm", self.sigma_q_mm),
            ("q_endpoint_sigma_mm", self.q_endpoint_sigma_mm),
            ("q_a50", self.q_a50),
            ("q_hill_n", self.q_hill_n),
            ("tau_m_ms", self.tau_m_ms),
            ("m_current_hill_n", self.m_current_hill_n),
            ("sigma_m_mm", self.sigma_m_mm),
            ("h_smooth_sigma_mm", self.h_smooth_sigma_mm),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.k_q_per_ms < 0.0:
            raise ValueError("k_q_per_ms must be non-negative")
        if not 0.0 <= self.q_min <= 1.0:
            raise ValueError("q_min must lie in [0, 1]")
        if not self.q_min <= self.q_init <= 1.0:
            raise ValueError("q_init must lie in [q_min, 1]")
        if not 0.0 <= self.q_floor_h_gain <= 1.0 - self.q_min:
            raise ValueError("q_floor_h_gain must keep the q floor in [0, 1]")
        if self.eta_m < 0.0:
            raise ValueError("eta_m must be non-negative")
        if self.m_build_gain < 0.0:
            raise ValueError("m_build_gain must be non-negative")
        if self.m_current_threshold < 0.0:
            raise ValueError("m_current_threshold must be non-negative")
        if self.m_current_saturation_width < 0.0:
            raise ValueError("m_current_saturation_width must be non-negative")
        if self.m_state_ceiling < 0.0:
            raise ValueError("m_state_ceiling must be non-negative")
        if not 0.0 <= self.m_spatial_mix <= 1.0:
            raise ValueError("m_spatial_mix must lie in [0, 1]")
        if self.m_spatial_mix > 0.0 and self.sigma_m_mm >= self.sigma_q_mm:
            raise ValueError("spatial M/gK footprint must be narrower than qI")
        if (abs(self.q_init_h_gain) >= 1.0
                or abs(self.q_endpoint_gain) >= 1.0
                or abs(self.k_q_h_gain) >= 1.0
                or abs(self.eta_m_h_gain) >= 1.0):
            raise ValueError("patient-field gains must have absolute value below one")
        if self.q_eta_i < self.q_eta_e:
            raise ValueError("q_eta_i must be at least q_eta_e")
        if int(self.trace_stride_steps) != self.trace_stride_steps:
            raise ValueError("trace_stride_steps must be an integer")
        if self.trace_stride_steps < 1:
            raise ValueError("trace_stride_steps must be at least one")


def _grid_mean(values, ix, iy, n_grid):
    flat = np.asarray(iy, int) * int(n_grid) + np.asarray(ix, int)
    total = np.bincount(flat, weights=np.asarray(values, float),
                        minlength=n_grid * n_grid)
    count = np.bincount(flat, minlength=n_grid * n_grid)
    fallback = float(np.mean(values)) if len(values) else 0.0
    mean = np.divide(total, count, out=np.full_like(total, fallback),
                     where=count > 0)
    return mean.reshape(n_grid, n_grid)


def _mean_one_bounded_modulation(field, gain):
    """Positive, mean-one modulation with a bounded patient-field contrast.

    ``1 + gain*tanh(zscore)`` prevents a few extreme h values from turning a
    modest mean rate into an order-of-magnitude local rate.  ``abs(gain)<1``
    keeps the multiplier positive before mean normalisation.
    """
    values = np.asarray(field, float)
    if gain == 0.0 or float(np.std(values)) <= 1e-12:
        return np.ones_like(values)
    z = (values - float(np.mean(values))) / float(np.std(values))
    multiplier = 1.0 + float(gain) * np.tanh(z)
    return multiplier / float(np.mean(multiplier))


def periodic_endpoint_field(n_grid, sheet_l_mm, centers_xy, sigma_mm):
    """Endpoint-union field on the periodic sheet from frozen patient contacts.

    The field is deterministic and contains no fitted amplitude or centre.  A
    maximum combines the Gaussian foci so overlap does not create an artificial
    deeper basin between neighbouring endpoint contacts.
    """
    centers = np.asarray(centers_xy, float)
    if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) < 2:
        raise ValueError("endpoint_centers_xy must have shape (n>=2, 2)")
    n = int(n_grid)
    length = float(sheet_l_mm)
    coord = (np.arange(n, dtype=float) + 0.5) * length / n
    xx, yy = np.meshgrid(coord, coord, indexing="xy")
    fields = []
    for center_x, center_y in centers:
        dx = np.abs(xx - float(center_x))
        dy = np.abs(yy - float(center_y))
        dx = np.minimum(dx, length - dx)
        dy = np.minimum(dy, length - dy)
        fields.append(np.exp(
            -(dx * dx + dy * dy) / (2.0 * float(sigma_mm) ** 2)))
    # max, rather than sum, preserves two distinct endpoint basins when their
    # tails overlap along the propagation axis.
    return np.maximum.reduce(fields)


def thresholded_hill_saturation(a, a0, a50, exponent):
    """Z-like activity gate on a continuous qI field.

    Exponent one is exactly the historical qI saturation.  Larger exponents
    interpolate toward the hard threshold used by Z without removing spatial
    continuity from qI.
    """
    x = np.maximum(np.asarray(a, dtype=float) - float(a0), 0.0)
    n = float(exponent)
    xn = x ** n
    return xn / (float(a50) ** n + xn)


class SpatialZMQIGKSlowVars:
    """Slow protocol consumed by ``kick_probe.simulate_kick``.

    qI is stored on a grid and sampled at every E neuron.  M remains a true
    per-neuron spike history.  No random numbers are consumed.
    """

    TRACE_NAMES = (
        "time_ms",
        "q_mean",
        "q_min",
        "q_core_mean",
        "q_surround_mean",
        "m_mean",
        "m_max",
        "adaptation_current_mean",
        "m_spatial_drive_mean",
        "q_drive_mean",
        "q_drive_max",
        "spike_count_E",
        "spike_count_I",
    )

    def __init__(self, N, V_th0, posE, posI, L, h_e, *,
                 core_mask_E=None, endpoint_centers_xy=None, cfg=None):
        self.cfg = cfg or SpatialZMQIGKConfig()
        self.cfg.validate()
        self.N = int(N)
        self.nE = int(np.asarray(posE).shape[0])
        self.NE = self.nE
        self.V_th0 = float(V_th0)
        self.posE = np.asarray(posE, float)
        self.posI = np.asarray(posI, float)
        self.L = float(L)
        self.h_e = np.asarray(h_e, float)
        if self.h_e.shape != (self.nE,):
            raise ValueError(f"h_e must have shape ({self.nE},)")
        if self.N != self.nE + len(self.posI):
            raise ValueError("N must equal nE+nI")
        if endpoint_centers_xy is None:
            self.endpoint_centers_xy = None
        else:
            centers = np.asarray(endpoint_centers_xy, float)
            if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) < 2:
                raise ValueError("endpoint_centers_xy must have shape (n>=2, 2)")
            self.endpoint_centers_xy = centers
        if self.cfg.q_endpoint_gain != 0.0 and self.endpoint_centers_xy is None:
            raise ValueError(
                "q_endpoint_gain requires frozen endpoint_centers_xy")

        qcfg = SpatialSlowFieldConfig(
            n_grid=int(self.cfg.n_grid),
            sigma_r=float(self.cfg.sigma_r_mm),
            tau_a=float(self.cfg.tau_rate_ms),
            use_qI=True,
            tau_q=float(self.cfg.tau_q_ms),
            k_q=0.0,  # updated below with the patient-field multiplier
            q_min=float(self.cfg.q_min),
            q_init=float(self.cfg.q_init),
            sigma_q=float(self.cfg.sigma_q_mm),
            eta_E=float(self.cfg.q_eta_e),
            eta_I=float(self.cfg.q_eta_i),
            a0_q=float(self.cfg.q_a0),
            a50_q=float(self.cfg.q_a50),
            use_gK=False,
            k_K=0.0,
            sigma_K=min(0.5, 0.5 * float(self.cfg.sigma_q_mm)),
        )
        self._qdriver = SpatialSlowField(
            self.N, self.V_th0, self.posE, self.posI, self.L, cfg=qcfg)
        self.q_I = self._qdriver.q_I
        self._ixE, self._iyE = self._qdriver._ixE, self._qdriver._iyE
        n_grid = int(self.cfg.n_grid)
        self._ixI = np.clip(
            (self.posI[:, 0] / self.L * n_grid).astype(int), 0, n_grid - 1)
        self._iyI = np.clip(
            (self.posI[:, 1] / self.L * n_grid).astype(int), 0, n_grid - 1)
        self._flatE = self._iyE * n_grid + self._ixE
        self._flatI = self._iyI * n_grid + self._ixI
        self._Kr = isotropic_gaussian(
            n_grid, self.L, float(self.cfg.sigma_r_mm))
        self._Km = isotropic_gaussian(
            n_grid, self.L, float(self.cfg.sigma_m_mm))
        self._field_count_E = np.zeros((n_grid, n_grid), float)
        self._field_count_I = np.zeros((n_grid, n_grid), float)
        self._field_steps_seen = 0
        self._field_steps_per_update = None
        self._field_cell_count_E = np.bincount(
            self._flatE, minlength=n_grid * n_grid).reshape(n_grid, n_grid)
        self._last_m_drive_E = np.zeros(self.nE, float)

        h_grid = _grid_mean(
            self.h_e, self._ixE, self._iyE, int(self.cfg.n_grid))
        h_kernel = isotropic_gaussian(
            int(self.cfg.n_grid), self.L,
            float(self.cfg.h_smooth_sigma_mm))
        self.h_grid = convolve_periodic(h_grid, h_kernel)
        h_lo, h_hi = float(np.min(self.h_grid)), float(np.max(self.h_grid))
        if h_hi - h_lo <= 1e-12:
            h01 = np.zeros_like(self.h_grid)
        else:
            h01 = (self.h_grid - h_lo) / (h_hi - h_lo)
        if self.endpoint_centers_xy is None:
            self.endpoint_field = np.zeros_like(self.h_grid)
            endpoint_multiplier = np.ones_like(self.h_grid)
        else:
            self.endpoint_field = periodic_endpoint_field(
                int(self.cfg.n_grid), self.L, self.endpoint_centers_xy,
                float(self.cfg.q_endpoint_sigma_mm))
            endpoint_multiplier = _mean_one_bounded_modulation(
                self.endpoint_field, -float(self.cfg.q_endpoint_gain))
        q_init_grid = (
            float(self.cfg.q_init)
            * _mean_one_bounded_modulation(
                self.h_grid, -float(self.cfg.q_init_h_gain))
            * endpoint_multiplier
        )
        q_init_grid *= float(self.cfg.q_init) / float(np.mean(q_init_grid))
        np.clip(q_init_grid, float(self.cfg.q_min), 1.0, out=q_init_grid)
        self.k_q_grid = (
            float(self.cfg.k_q_per_ms)
            * _mean_one_bounded_modulation(self.h_grid, self.cfg.k_q_h_gain)
        )
        self.q_floor_grid = np.clip(
            float(self.cfg.q_min)
            + float(self.cfg.q_floor_h_gain) * (1.0 - h01),
            0.0, 1.0)
        np.maximum(q_init_grid, self.q_floor_grid, out=q_init_grid)
        self.q_init_grid = np.array(q_init_grid, copy=True)
        self.q_I[:] = self.q_init_grid
        eta_multiplier = _mean_one_bounded_modulation(
            self.h_e, self.cfg.eta_m_h_gain)
        self.eta_m_E = float(self.cfg.eta_m) * eta_multiplier

        self.z = np.ones(self.N, dtype=float)
        self.m = np.zeros(self.N, dtype=float)
        self._I_I_last = np.zeros(self.N, dtype=float)
        self._step_index = 0
        self._last_q_drive = np.zeros_like(self.q_I)
        self._trace = {name: [] for name in self.TRACE_NAMES}
        if core_mask_E is None:
            core = self.h_e >= 0.5
        else:
            core = np.asarray(core_mask_E, bool)
            if core.shape != (self.nE,):
                raise ValueError(f"core_mask_E must have shape ({self.nE},)")
        self._core = core

    def threshold(self, V_th_base):
        return V_th_base

    def _m_current_E(self):
        excess = np.maximum(
            self.m[:self.nE] - float(self.cfg.m_current_threshold), 0.0)
        width = float(self.cfg.m_current_saturation_width)
        if width == 0.0:
            activation = excess
        else:
            n = float(self.cfg.m_current_hill_n)
            scaled = excess / width
            activation = np.empty_like(scaled)
            low = scaled <= 1.0
            xn = scaled[low] ** n
            activation[low] = xn / (1.0 + xn)
            inverse_n = (1.0 / scaled[~low]) ** n
            activation[~low] = 1.0 / (1.0 + inverse_n)
        return self.eta_m_E * activation

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        del labels, I_E_rec
        exc = np.asarray(I_E, float)
        inh = np.asarray(I_I, float)
        self._I_I_last = inh
        out = exc - inh
        q_e = self.q_I[self._iyE, self._ixE]
        out[:self.nE] = (
            exc[:self.nE]
            - q_e * inh[:self.nE]
            - self._m_current_E()
        )
        return out

    def step(self, spk, labels, dt):
        spikes = np.asarray(spk, bool)
        if spikes.shape != (self.N,):
            raise ValueError(f"spk must have shape ({self.N},)")
        dt = float(dt)
        if self._field_steps_per_update is None:
            ratio = float(self.cfg.field_update_ms) / dt
            steps = int(round(ratio))
            if steps < 1 or not np.isclose(steps * dt,
                                           float(self.cfg.field_update_ms),
                                           atol=1e-12, rtol=0.0):
                raise ValueError("field_update_ms must lie on the membrane grid")
            self._field_steps_per_update = steps
        fired_e = np.flatnonzero(spikes[:self.nE])
        fired_i = np.flatnonzero(spikes[self.nE:])
        n_grid = int(self.cfg.n_grid)
        if fired_e.size:
            self._field_count_E += np.bincount(
                self._flatE[fired_e], minlength=n_grid * n_grid
            ).reshape(n_grid, n_grid)
        if fired_i.size:
            self._field_count_I += np.bincount(
                self._flatI[fired_i], minlength=n_grid * n_grid
            ).reshape(n_grid, n_grid)
        self._field_steps_seen += 1
        if self._field_steps_seen == self._field_steps_per_update:
            # The accumulated count is converted to the mean per membrane step,
            # matching slow_field's instantaneous-count units.  The 1-ms hold is
            # an explicit slow-field integrator step; membrane/spikes remain at dt.
            scale = 1.0 / float(self._field_steps_seen)
            rE_inst = convolve_periodic(self._field_count_E * scale, self._Kr)
            rI_inst = convolve_periodic(self._field_count_I * scale, self._Kr)
            field_dt = self._field_steps_seen * dt
            alpha = 1.0 - np.exp(-field_dt / float(self.cfg.tau_rate_ms))
            self._qdriver.rE += alpha * (rE_inst - self._qdriver.rE)
            self._qdriver.rI += alpha * (rI_inst - self._qdriver.rI)
            a_q = convolve_periodic(
                aq_drive(
                    self._qdriver.rE,
                    self._qdriver.rI,
                    self.cfg.q_eta_e,
                    self.cfg.q_eta_i,
                ),
                self._qdriver._Kq,
            )
            drive = thresholded_hill_saturation(
                a_q, self.cfg.q_a0, self.cfg.q_a50, self.cfg.q_hill_n)
            if not self.cfg.freeze_q:
                self.q_I += field_dt * (
                    (1.0 - self.q_I) / float(self.cfg.tau_q_ms)
                    - self.k_q_grid * drive * self.q_I
                )
                np.maximum(self.q_I, self.q_floor_grid, out=self.q_I)
                np.minimum(self.q_I, 1.0, out=self.q_I)
            self._last_q_drive = drive
            if float(self.cfg.m_spatial_mix) > 0.0:
                per_cell_spike_probability = np.divide(
                    self._field_count_E * scale,
                    self._field_cell_count_E,
                    out=np.zeros_like(self._field_count_E),
                    where=self._field_cell_count_E > 0,
                )
                m_drive_grid = convolve_periodic(
                    per_cell_spike_probability, self._Km)
                self._last_m_drive_E = m_drive_grid[self._iyE, self._ixE]
            self._field_count_E.fill(0.0)
            self._field_count_I.fill(0.0)
            self._field_steps_seen = 0

        m_e = self.m[:self.nE]
        m_e -= (dt / float(self.cfg.tau_m_ms)) * m_e
        np.maximum(m_e, 0.0, out=m_e)
        mix = float(self.cfg.m_spatial_mix)
        build_gain = float(self.cfg.m_build_gain)
        ceiling = float(self.cfg.m_state_ceiling)
        if ceiling == 0.0:
            if mix == 0.0:
                m_e[spikes[:self.nE]] += build_gain
            else:
                m_e += build_gain * mix * self._last_m_drive_E
                m_e[spikes[:self.nE]] += build_gain * (1.0 - mix)
        else:
            increment = build_gain * mix * self._last_m_drive_E
            if mix < 1.0:
                increment = np.array(increment, copy=True)
                increment[spikes[:self.nE]] += build_gain * (1.0 - mix)
            availability = np.maximum(1.0 - m_e / ceiling, 0.0)
            m_e += increment * availability
            np.minimum(m_e, ceiling, out=m_e)

        q_e = self.q_I[self._iyE, self._ixE]
        self.z[:self.nE] = q_e
        if self._step_index % int(self.cfg.trace_stride_steps) == 0:
            self._record_trace(spikes, dt, q_e)
        self._step_index += 1

    def _record_trace(self, spikes, dt, q_e):
        core = self._core
        m_e = self.m[:self.nE]
        values = {
            "time_ms": (self._step_index + 1) * dt,
            "q_mean": np.mean(q_e),
            "q_min": np.min(q_e),
            "q_core_mean": np.mean(q_e[core]) if np.any(core) else np.nan,
            "q_surround_mean": np.mean(q_e[~core]) if np.any(~core) else np.nan,
            "m_mean": np.mean(m_e),
            "m_max": np.max(m_e),
            "adaptation_current_mean": np.mean(self._m_current_E()),
            "m_spatial_drive_mean": np.mean(self._last_m_drive_E),
            "q_drive_mean": np.mean(self._last_q_drive),
            "q_drive_max": np.max(self._last_q_drive),
            "spike_count_E": np.sum(spikes[:self.nE]),
            "spike_count_I": np.sum(spikes[self.nE:]),
        }
        for name, value in values.items():
            self._trace[name].append(float(value))

    def trace_arrays(self):
        return {name: np.asarray(values, np.float32)
                for name, values in self._trace.items()}

    def summary(self):
        q_e = self.q_I[self._iyE, self._ixE]
        traces = self.trace_arrays()
        return {
            "trace_samples": int(len(traces["time_ms"])),
            "final_q_mean": float(np.mean(q_e)),
            "final_q_min": float(np.min(q_e)),
            "final_q_max": float(np.max(q_e)),
            "initial_q_grid_range": [float(np.min(self.q_init_grid)),
                                       float(np.max(self.q_init_grid))],
            "final_m_mean": float(np.mean(self.m[:self.nE])),
            "maximum_m": float(np.max(self.m[:self.nE])),
            "peak_mean_adaptation_current": float(np.max(
                traces["adaptation_current_mean"], initial=0.0)),
            "k_q_grid_range_per_ms": [float(np.min(self.k_q_grid)),
                                       float(np.max(self.k_q_grid))],
            "q_floor_grid_range": [float(np.min(self.q_floor_grid)),
                                    float(np.max(self.q_floor_grid))],
            "eta_m_range": [float(np.min(self.eta_m_E)),
                              float(np.max(self.eta_m_E))],
        }
