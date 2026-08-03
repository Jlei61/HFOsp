"""Z/M branch-decision Phase-0A: canonical config lock + dynamic-state inventory (spec rev3.1 §1.2/§2).

Two jobs, both FAIL-CLOSED:

1. `build_canonical_config` resolves the locked Z/M+S_G experiment from the REAL builders
   (`scripts/run_m4_phaseplane.build_substrate` constants + `run_zm_snn_native_exit._zm_cfg`), never
   from duplicated literals, and hashes every guarded engine file. Changing any locked field changes
   `config_sha` -> a new experiment family -> snapshots from the old family are rejected on load.

2. `audit_dynamic_state` re-derives, from the ENGINE SOURCE via `ast`, every name mutated inside
   `simulate_kick` and every `self.<attr>` mutated by `SpatialSlowField`, and requires each one to be
   classified exactly once as simulator state / derived-from-config / observer / per-step temporary.
   A new engine variable therefore BLOCKS the line (`blocked_state_inventory`) until it is classified
   -- which is the only way an unclassified current-affecting state cannot silently escape the
   snapshot (spec §2.1 "Any unclassified state that affects membrane current is a P0 stop").

Task 3 adds serialization + freeze policy on top of the `SimulationStateV1` schema defined here.
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(ROOT, "scripts")
_ENGINE = os.path.join(ROOT, "src", "snn_engine")

SCHEMA_VERSION = "zm_fork_state_v1"
INVENTORY_VERSION = "zm_state_inventory_v3_conditional_2026-08-02"

#: every engine file whose bytes can change the trajectory (the 6 blessed + the slow layer)
GUARDED_ENGINE_FILES = (
    "src/snn_engine/kick_probe.py",
    "src/snn_engine/params.py",
    "src/snn_engine/model.py",
    "src/snn_engine/connectivity.py",
    "src/snn_engine/connectivity_rot.py",
    "src/snn_engine/lfp.py",
    "src/snn_engine/slow_field.py",
    "src/snn_engine/mz_slow_vars.py",
)

FREEZE_SEMANTICS = ("dynamic", "freezable_z", "freezable_m", "freezable_sg_family",
                    "always_constant", "not_applicable")


# ================================================================= hashing helpers
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default)


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


def config_sha(cfg):
    """SHA256 over the canonical JSON of the whole config dict (order-independent)."""
    return hashlib.sha256(_canonical_json(cfg).encode()).hexdigest()


# ================================================================= canonical config
def _import_builders():
    """Import the REAL runners (not copies) so the config cannot drift from what runs."""
    for pth in (ROOT, _SCRIPTS, _ENGINE):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    import run_m4_phaseplane as PP  # noqa: E402
    import run_zm_snn_native_exit as ZM  # noqa: E402
    return PP, ZM


def build_canonical_config(seed, I_th_EI, *, arm_kwargs=None, dt=None, resolve_placement=True):
    """The write-once experiment lock (spec §1.2).

    `arm_kwargs` selects the Z/M layer variant; the branch-decision family uses the tested
    S_G anchor (`use_SG=True, alpha_G=16`). `I_th_EI` is the per-seed q75 calibration -- it is a
    LOCKED FIELD, so re-calibrating changes `config_sha` and invalidates old snapshots.
    """
    PP, ZM = _import_builders()
    arm_kwargs = dict(use_SG=True, alpha_G=16.0) if arm_kwargs is None else dict(arm_kwargs)
    slow_cfg = ZM._zm_cfg(float(I_th_EI), **arm_kwargs)
    p = PP.Params(g=PP.G, L=PP.L, density=PP.DENSITY, T=PP.T_SIM, dt=PP.DT if dt is None else dt,
                  nu_ext_ratio=PP.DRIVE, seed=int(seed))
    placement = None
    if resolve_placement:
        m_real, src_names, snk_names = PP.template_source_foci(
            PP.SUBJECT, PP.MONTAGE, PP.PLACEMENT_K_EARLY, root=PP.DATA_ROOT)
        reg = PP.register_to_sheet(m_real, src_names, snk_names, L=PP.L,
                                   target_inter_core_mm=PP.TARGET_INTER_CORE)
        placement = dict(theta_deg=float(reg["theta_deg"]),
                         source_centroid=[float(x) for x in reg["source_centroid"]],
                         sink_centroid=[float(x) for x in reg["sink_centroid"]],
                         center=[float(x) for x in reg["center"]],
                         source_contacts=list(src_names), sink_contacts=list(snk_names))
    return dict(
        schema=SCHEMA_VERSION,
        lockpoint="zA_q75_tz5000__mA0p001_tau500",
        subject=PP.SUBJECT, montage=PP.MONTAGE, placement_k_early=int(PP.PLACEMENT_K_EARLY),
        I_th_EI=float(I_th_EI),
        params={k: (list(v) if isinstance(v, (list, tuple)) else v)
                for k, v in dataclasses.asdict(p).items()},
        slow_field=dataclasses.asdict(slow_cfg),
        arm_kwargs=arm_kwargs,
        substrate=dict(L=float(PP.L), density=float(PP.DENSITY), g=float(PP.G), drive=float(PP.DRIVE),
                       core_mean=float(PP.CORE_MEAN), core_std=float(PP.CORE_STD),
                       core_r=float(PP.CORE_R), base_mean=float(PP.BASE_MEAN),
                       target_inter_core_mm=PP.TARGET_INTER_CORE, AR=2.0,
                       vth_core_seed_offsets=[7, 8], placement=placement),
        protocol=dict(kick_boost=0.0, t_kick=1e9, r_kick=float(PP.R_KICK),
                      kick_center="source_centroid", spontaneous=True,
                      es_dur_ms=100.0, lfp_sample_hz=1.0 / (PP.DT if dt is None else dt) * 1e3),
        engine_sha256={rel: sha256_file(os.path.join(ROOT, rel)) for rel in GUARDED_ENGINE_FILES},
        inventory_version=INVENTORY_VERSION,
    )


# ================================================================= state inventory
def _row(name, category, shape, dtype, time_scale, role, dt_dependent, snapshot,
         freeze_semantics, current_effect, note="", activation_gate=None):
    return dict(name=name, category=category, shape=shape, dtype=dtype, time_scale=time_scale,
                role=role, dt_dependent=bool(dt_dependent), snapshot=bool(snapshot),
                freeze_semantics=freeze_semantics, current_effect=current_effect,
                activation_gate=activation_gate, note=note)


def build_state_inventory():
    """One row per state that the snapshot/restore contract must reason about (spec §2.1)."""
    return [
        # ---- membrane ----
        _row("V", "membrane", "(N,)", "float64", "tau_m 10-20 ms", "simulator", False, True,
             "dynamic", "direct", "LIF membrane potential"),
        _row("ref", "membrane", "(N,)", "int32", "tau_ref 1-2 ms", "simulator", True, True,
             "dynamic", "direct", "refractory countdown in STEPS -> dt dependent"),
        # ---- synaptic ----
        _row("s_E", "synaptic", "(N,)", "float64", "tau_r_AMPA 0.7 ms", "simulator", False, True,
             "dynamic", "direct", "AMPA gate (recurrent + external)"),
        _row("I_E", "synaptic", "(N,)", "float64", "tau_d_AMPA 3.5 ms", "simulator", False, True,
             "dynamic", "direct", "AMPA current"),
        _row("s_I", "synaptic", "(N,)", "float64", "tau_r_GABA 1.0 ms", "simulator", False, True,
             "dynamic", "direct", "GABA gate"),
        _row("I_I", "synaptic", "(N,)", "float64", "tau_d_GABA 18 ms", "simulator", False, True,
             "dynamic", "direct", "GABA current; also the z_inf Heaviside input"),
        _row("s_E_rec", "synaptic", "(N,)", "float64", "tau_r_AMPA", "simulator", False, True,
             "dynamic", "direct", "recurrent-only AMPA gate; allocated only when use_SG"),
        _row("I_E_rec", "synaptic", "(N,)", "float64", "tau_d_AMPA", "simulator", False, True,
             "dynamic", "direct", "recurrent-only AMPA current divided by the S_G pool"),
        # ---- delays ----
        _row("ring_sE", "delays", "(M,N)", "float64", "<= max_delay", "simulator", True, True,
             "dynamic", "direct", "AMPA delay ring; bin index = round(delay/delay_dt)"),
        _row("ring_sI", "delays", "(M,N)", "float64", "<= max_delay", "simulator", True, True,
             "dynamic", "direct", "GABA delay ring"),
        _row("t", "delays", "scalar", "int", "step", "simulator", True, True,
             "always_constant", "indirect", "absolute step index; sets ring slot t%M and t*dt"),
        # ---- external drive ----
        _row("xi", "external", "scalar", "float64", "tau_n 150 ms", "simulator", False, True,
             "dynamic", "direct", "OU state of the external rate"),
        _row("rng_state", "rng", "dict", "PCG64 state", "per draw", "simulator", False, True,
             "dynamic", "direct", "simulator bit-generator state (OU normal + Poisson ext)"),
        # ---- optional hidden fast features (must be OFF in this family; classified anyway) ----
        _row("x_dep", "optional", "(NE,)", "float64", "ee_std_tau_ms", "simulator", False, True,
             "dynamic", "direct", "E->E short-term depression; OFF (ee_std_u=0) in this family"),
        _row("r_ema", "optional", "scalar", "float64", "feedback_tau_ms", "simulator", False, True,
             "dynamic", "direct", "A1c global feedback EMA; OFF (feedback_gain=0)"),
        # ---- early-stop controller (changes truncation -> changes the recorded trajectory) ----
        # the early-stop controller never enters the membrane equation; it decides WHEN the loop
        # truncates, so it changes the recorded trajectory LENGTH and must still be carried across a
        # checkpoint (proved by test_early_stop_controller_state_is_carried_and_matters)
        _row("_es_ema", "control", "scalar", "float64", "20 ms", "simulator", False, True,
             "dynamic", "none", "runaway early-stop EMA; controls truncation only"),
        _row("_es_run", "control", "scalar", "int", "step", "simulator", True, True,
             "dynamic", "none", "consecutive supra-threshold steps; controls truncation only"),
        # ---- slow Z/M (per neuron) ----
        _row("slow.z", "slow_zm", "(N,)", "float64", "tau_z 5000 ms", "simulator", False, True,
             "freezable_z", "direct", "per-E inhibitory efficacy; scales I_I on E cells"),
        _row("slow.m", "slow_zm", "(N,)", "float64", "tau_adp 500 ms", "simulator", False, True,
             "freezable_m", "direct", "per-E adaptation; current eta_m*m subtracted"),
        _row("slow.phi_increment", "fast_adaptation", "(N,)", "float64",
             "tau_phi 60-160 ms", "simulator", False, True,
             "dynamic", "none",
             "gated by use_phi=False in this Phase-C inventory; direct in Phase-D arm D "
             "and remains dynamic in frozen Z/M forks"),
        _row("slow.i2e_resource", "fast_inhibitory_state", "(NI,)", "float64",
             "tau_i2e_depression 100-600 ms", "simulator", False, True,
             "dynamic", "conditional_direct",
             "presynaptic availability scaling I->E edges only; neutral at one when disabled",
             activation_gate="use_i2e_depression"),
        _row("slow.i_adaptation_increment", "fast_inhibitory_state", "(N,)", "float64",
             "tau_i_adaptation 100-600 ms", "simulator", False, True,
             "dynamic", "conditional_direct",
             "I-cell threshold increment; E-cell entries remain exactly zero",
             activation_gate="use_i_adaptation"),
        _row("slow._I_I_last", "slow_zm", "(NE,)", "float64", "step", "simulator", False, True,
             "freezable_z", "none",
             "E-cell I_I stashed by apply_currents for the z_inf Heaviside. INTRA-step scratch: "
             "apply_currents overwrites it before step() reads it, so it carries nothing across a "
             "checkpoint boundary (verified by test_every_current_affecting_state_is_load_bearing). "
             "Snapshotted anyway for completeness."),
        # ---- shared pool S_G family ----
        _row("slow.rE_fast", "shared_pool", "(n_grid,n_grid)", "float64", "tau_s 15 ms", "simulator",
             False, True, "freezable_sg_family", "indirect", "fast rate EMA driving Psi_G"),
        _row("slow.mu_G", "shared_pool", "scalar", "float64", "tau_mu 40 ms", "simulator", False, True,
             "freezable_sg_family", "indirect", "pool activation"),
        _row("slow.S_G", "shared_pool", "scalar", "float64", "tau_S 120 ms", "simulator", False, True,
             "freezable_sg_family", "direct", "pool OUTPUT; divides recurrent E current"),
        # ---- slow fields that are constant in this family but must still round-trip ----
        _row("slow.rE", "slow_field", "(n_grid,n_grid)", "float64", "tau_a 100 ms", "simulator",
             False, True, "dynamic", "none",
             "slow rate EMA; drives q_I/g_K/p/n which are all OFF here -> no current effect"),
        _row("slow.rI", "slow_field", "(n_grid,n_grid)", "float64", "tau_a 100 ms", "simulator",
             False, True, "dynamic", "none", "as slow.rE"),
        _row("slow.q_I", "slow_field", "(n_grid,n_grid)", "float64", "tau_q", "simulator", False,
             True, "always_constant", "direct", "use_qI=False -> pinned at q_init=1"),
        _row("slow.g_K", "slow_field", "(n_grid,n_grid)", "float64", "tau_K", "simulator", False,
             True, "always_constant", "direct", "use_gK=False -> pinned at 0"),
        # the next five are read by apply_currents ONLY behind their own use_* gate, and every one of
        # those gates is False in this family -> no current effect HERE (they are still snapshotted so
        # the state round-trip is complete if a later family turns them on)
        _row("slow.p", "slow_field", "(n_grid,n_grid)", "float64", "tau_p", "simulator", False,
             True, "always_constant", "none", "gated by use_persist=False"),
        _row("slow.n_load", "slow_field", "(n_grid,n_grid)", "float64", "tau_n", "simulator", False,
             True, "always_constant", "none", "gated by use_A=False"),
        _row("slow.a_shunt", "slow_field", "(n_grid,n_grid)", "float64", "tau_n", "simulator", False,
             True, "always_constant", "none", "gated by use_A=False / eta_A=0"),
        _row("slow.h_G", "slow_field", "scalar", "float64", "tau_G", "simulator", False, True,
             "always_constant", "none", "gated by use_hG=False"),
        _row("slow.H", "slow_field", "scalar", "float64", "tau_H", "simulator", False, True,
             "always_constant", "none", "gated by use_H=False"),
        _row("slow.mode_H", "slow_field", "(n_grid,n_grid)", "float64", "tau_mode_H", "simulator", False, True,
             "dynamic", "conditional_direct", "local activity memory; Z-gated and M-closed recurrent-E gain",
             activation_gate="use_mode_H"),
        _row("slow.mode_M_memory", "slow_field", "scalar", "float64", "tau_mode_M_memory", "simulator", False, True,
             "dynamic", "conditional_direct", "slow collective-M load on recurrent-E denominator",
             activation_gate="use_mode_M_memory"),
        _row("slow._t", "slow_field", "scalar", "float64", "ms", "simulator", True, True,
             "always_constant", "none",
             "slow-layer clock; only read by hG_script and persist_onset_ms, both inactive here"),
        # ---- observer ----
        _row("rate_E", "observer", "(nsteps,)", "float64", "step", "observer", True, False,
             "not_applicable", "none", "population spike count per step"),
        _row("rate_I", "observer", "(nsteps,)", "float64", "step", "observer", True, False,
             "not_applicable", "none", ""),
        _row("E_spk_bool", "observer", "(nsteps,NE)", "bool", "step", "observer", True, False,
             "not_applicable", "none", "per-step E raster; source-space readout"),
        _row("lfp_trace", "observer", "(nsteps,n_sites)", "float64", "step", "observer", True, False,
             "not_applicable", "none",
             "current-based virtual SEEG; LFPRecorder is MEMORYLESS (weighted sum of |I_E|+|I_I|), "
             "so it has no filter state to carry across a restore -- continuity is structural"),
        _row("lfp_exc_trace", "observer", "(nsteps,n_sites)", "float64", "step", "observer", True, False,
             "not_applicable", "none", "excitatory-current contribution to lfp_trace"),
        _row("lfp_inh_trace", "observer", "(nsteps,n_sites)", "float64", "step", "observer", True, False,
             "not_applicable", "none", "inhibitory-current contribution to lfp_trace"),
        _row("slow.trace_*", "observer", "(nsteps,)", "float64", "step", "observer", True, False,
             "not_applicable", "none", "slow-layer trace lists; read-only mirrors of the state above"),
    ]


def validate_inventory(rows):
    """Raise on a self-contradictory row (spec §2.1 classification contract)."""
    seen = set()
    for r in rows:
        missing = {"name", "category", "shape", "dtype", "time_scale", "role", "dt_dependent",
                   "snapshot", "freeze_semantics", "current_effect"} - set(r)
        if missing:
            raise ValueError(f"{r.get('name')}: missing inventory fields {sorted(missing)}")
        if r["name"] in seen:
            raise ValueError(f"{r['name']}: appears more than once in the inventory")
        seen.add(r["name"])
        if r["role"] not in ("simulator", "observer"):
            raise ValueError(f"{r['name']}: role must be simulator|observer")
        if r["current_effect"] not in (
            "direct", "indirect", "conditional_direct", "none"
        ):
            raise ValueError(
                f"{r['name']}: current_effect must be "
                "direct|indirect|conditional_direct|none"
            )
        if r["current_effect"] == "conditional_direct" and not r.get("activation_gate"):
            raise ValueError(
                f"{r['name']}: conditional_direct requires activation_gate"
            )
        if r["freeze_semantics"] not in FREEZE_SEMANTICS:
            raise ValueError(f"{r['name']}: unknown freeze_semantics {r['freeze_semantics']!r}")
        if r["role"] == "observer" and r["current_effect"] != "none":
            raise ValueError(
                f"{r['name']}: observer-only state cannot have current_effect={r['current_effect']!r}")
        if r["role"] == "observer" and r["snapshot"]:
            raise ValueError(f"{r['name']}: observer state must not be part of the simulator snapshot")
        if r["role"] == "simulator" and not r["snapshot"]:
            raise ValueError(f"{r['name']}: simulator state must be in the snapshot (exact resume)")
    return rows


# ================================================================= fail-closed source audit
#: mutated inside `simulate_kick` but a pure function of (config, net) -- recomputed on restore
_KICK_DERIVED = {
    "M", "N", "NE", "NI", "labels", "pos", "ampa", "gaba", "dt", "nsteps", "net", "slow", "rng",
    "decay_sE", "decay_IE", "decay_sI", "decay_II", "tau_m", "decay_V", "ref_steps", "ext_incr",
    "ampa_bins", "gaba_bins", "a_indptr", "a_dst", "a_dly", "a_w", "g_indptr", "g_dst", "g_dly",
    "g_w", "nu_theta", "nu_sig_const", "nu_signal_fn", "sigma_n_inv_ms", "sigma_xi", "ou_a", "ou_b",
    "center", "is_E", "dist_c", "rk", "tk", "kick_mask", "outside_mask", "e_gaba", "E_A",
    "ee_std_on", "x_rec_f", "track_rec", "fb_dyn", "fb_static", "fb_on", "alpha_fb", "inv_dt_ms",
    "conductance_on", "cond_cfg", "i2e_dep_on",
    "_es_alpha", "_es_dur", "fb_override_trace", "base_vth", "rate_E_hz", "rate_I_hz", "res",
    # Z/M branch-decision checkpoint hook: controller handle + its hoisted gate flags. All are
    # functions of the `zm_ckpt=` argument; `t_start` comes from the restored snapshot's absolute
    # step. None of them is state the snapshot has to carry.
    "zm_ckpt", "_ck_mean", "_ck_dump", "_ck_snap", "_st0", "t_start",
}
#: recorded for analysis; provably cannot feed back into the membrane update
_KICK_OBSERVER = {
    "rate_E", "rate_I", "spk_t", "spk_i", "ras_keepE", "ras_keepI", "ras_keep", "ras_mask",
    "spk_inside", "spk_outside", "E_spk_bool", "I_spk_bool", "_peak_act", "I_E_peak", "I_I_peak",
    "lfp_trace", "lfp_current_proxy_trace", "lfp_exc_trace", "lfp_inh_trace",
    "I_global_trace", "xdep_mean", "xdep_min",
    "xdep_mask_mean", "t0",
    # written only in the same statement that breaks the loop -> nothing to carry across a
    # checkpoint; it is the truncation MARKER reported as runaway_early_stop_ms.
    "_stop_t",
}
#: fully overwritten before every use within a timestep -> nothing to carry across a checkpoint
_KICK_TEMP = {
    "tm", "nu_now", "nu_vec", "ext", "slot", "I_net", "V_th_eff", "Vtmp", "V_inf", "free", "spk",
    "idx", "st", "cnt", "tot", "spE", "spI", "x_per_edge", "d_per_edge", "w_eff",
    "conductance_state", "ig_t", "I_fb", "_na", "_tgt", "g", "_", "tg",
}
#: true simulator state carried by the snapshot (names as they appear in `simulate_kick`)
_KICK_STATE = {"V", "ref", "s_E", "I_E", "s_I", "I_I", "s_E_rec", "I_E_rec", "ring_sE", "ring_sI",
               "xi", "t", "x_dep", "r_ema", "_es_ema", "_es_run"}

#: SpatialSlowField attributes that are pure functions of cfg/geometry
_SLOW_DERIVED = {"cfg", "N", "nE", "nI", "L", "posE", "posI", "_Kq", "_Kk", "_Kn", "_Kp", "_alpha_a",
                 "_alpha_s", "_ixE", "_iyE", "_ixE_core", "_iyE_core", "_ixE_surr", "_iyE_surr",
                 "_core_mask_E", "is_E", "hG_script"}
_SLOW_STATE = {"z", "m", "phi_increment", "i2e_resource", "i_adaptation_increment",
               "_I_I_last", "rE", "rI", "rE_fast", "mu_G", "S_G", "q_I", "g_K", "p",
               "n_load", "a_shunt", "h_G", "H", "mode_H", "mode_M_memory", "_t"}


# ================================================================= freeze policy (Task 3)
#: the six scientific arms of spec §6.1 -> which slow coordinates are held at their snapshot value
ARM_FREEZE_TABLE = {
    "dynamic_replay": dict(freeze_z=False, freeze_m=False, freeze_sg_family=False),
    "freeze_z":       dict(freeze_z=True,  freeze_m=False, freeze_sg_family=False),
    "freeze_zm":      dict(freeze_z=True,  freeze_m=True,  freeze_sg_family=False),
    "freeze_zsg":     dict(freeze_z=True,  freeze_m=False, freeze_sg_family=True),
    "freeze_all":     dict(freeze_z=True,  freeze_m=True,  freeze_sg_family=True),
    "dynamic_z_only": dict(freeze_z=False, freeze_m=True,  freeze_sg_family=True),
}
#: NOT a primary arm (spec §2.2): freezing the pool OUTPUT while its sensor keeps drifting is mixed
#: semantics, so it carries a different name and can never support a carrier verdict.
DIAGNOSTIC_ARMS = {
    "freeze_sg_output_only": dict(
        freeze_z=True, freeze_m=True,
        freeze_sg_family=False, freeze_sg_output_only=True,
    ),
    # Phase 2B only: let the specified Z recovery and M adaptation evolve
    # together while the minimal fast carrier's shared pool is held fixed.
    # This is not a discovery arm and can never establish a carrier window.
    "dynamic_zm_freeze_sg": dict(
        freeze_z=False, freeze_m=False, freeze_sg_family=True,
    ),
}


@dataclasses.dataclass(frozen=True)
class FreezePolicy:
    """q(t>t_f)=q(t_f) with the coordinate's membrane-current effect still active."""
    freeze_z: bool = False
    freeze_m: bool = False
    freeze_sg_family: bool = False
    freeze_sg_output_only: bool = False
    arm: str = "dynamic_replay"

    @classmethod
    def for_arm(cls, arm):
        if arm in ARM_FREEZE_TABLE:
            return cls(arm=arm, **ARM_FREEZE_TABLE[arm])
        if arm in DIAGNOSTIC_ARMS:
            return cls(arm=arm, **DIAGNOSTIC_ARMS[arm])
        raise ValueError(f"unknown arm {arm!r}; primary={sorted(ARM_FREEZE_TABLE)} "
                         f"diagnostic={sorted(DIAGNOSTIC_ARMS)}")

    @property
    def is_primary(self):
        return self.arm in ARM_FREEZE_TABLE

    def as_dict(self):
        return dataclasses.asdict(self)


class FreezeWrapper:
    """Wraps a `SpatialSlowField` and holds the frozen coordinates at their snapshot values.

    Implementation: let the real `step` run, then write the frozen coordinates back. That is exactly
    `q(t>t_f)=q(t_f)` -- the coordinate is still READ by `apply_currents` every step, so its current
    effect is retained, and the spatial field is preserved element-wise (never collapsed to a mean,
    never reset to a default). The trace tails are rewritten too, so the recorded slow coordinates
    are the ones the membrane actually saw.
    """

    _Z_TRACES = ("trace_z_mean", "trace_z_min", "trace_z_core_mean", "trace_z_surround_mean")
    _M_TRACES = ("trace_m_mean", "trace_m_max", "trace_m_core_mean", "trace_m_surround_mean")

    def __init__(self, inner, policy: FreezePolicy):
        self.inner = inner
        self.policy = policy

    def __getattr__(self, name):  # delegate everything not defined here (cfg, uses_shunt, traces...)
        return getattr(self.inner, name)

    def apply_currents(self, I_E, I_I, labels=None, I_E_rec=None):
        return self.inner.apply_currents(I_E, I_I, labels, I_E_rec)

    def threshold(self, V_th_base):
        return self.inner.threshold(V_th_base)

    def _capture(self):
        s, pol, out = self.inner, self.policy, {}
        if pol.freeze_z:
            out["z"] = s.z.copy()
        if pol.freeze_m:
            out["m"] = s.m.copy()
        if pol.freeze_sg_family:
            out["rE_fast"] = s.rE_fast.copy()
            out["mu_G"] = float(s.mu_G)
            out["S_G"] = float(s.S_G)
        elif pol.freeze_sg_output_only:
            out["S_G"] = float(s.S_G)
        return out

    def _restore(self, saved):
        s, pol = self.inner, self.policy
        if "z" in saved:
            s.z[:] = saved["z"]
            self._retrace(self._Z_TRACES, s.z[:s.nE])
        if "m" in saved:
            s.m[:] = saved["m"]
            self._retrace(self._M_TRACES, s.m[:s.nE])
        if "rE_fast" in saved:
            s.rE_fast[:] = saved["rE_fast"]
            _set_tail(s.trace_rEfast_max, float(s.rE_fast.max()))
        if "mu_G" in saved:
            s.mu_G = saved["mu_G"]
            _set_tail(s.trace_muG, s.mu_G)
        if "S_G" in saved:
            s.S_G = saved["S_G"]
            _set_tail(s.trace_SG, s.S_G)

    def _retrace(self, names, vE):
        s = self.inner
        core = s._core_mask_E
        stat = dict(mean=float(vE.mean()), min=float(vE.min()), max=float(vE.max()))
        for nm in names:
            tr = getattr(s, nm, None)
            if not tr:
                continue
            if nm.endswith("_core_mean") and core is not None:
                _set_tail(tr, float(vE[core].mean()))
            elif nm.endswith("_surround_mean") and core is not None:
                _set_tail(tr, float(vE[~core].mean()))
            elif nm.endswith("_mean"):
                _set_tail(tr, stat["mean"])
            elif nm.endswith("_min"):
                _set_tail(tr, stat["min"])
            elif nm.endswith("_max"):
                _set_tail(tr, stat["max"])

    def step(self, spk, labels, dt):
        saved = self._capture()
        self.inner.step(spk, labels, dt)
        if saved:
            self._restore(saved)


def _set_tail(lst, value):
    if lst:
        lst[-1] = value


def _targets(node):
    if isinstance(node, ast.Assign):
        return list(node.targets)
    if isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        return [node.target]
    if isinstance(node, (ast.For, ast.AsyncFor)):
        return [node.target]
    return []


def _base_name(node):
    while isinstance(node, (ast.Subscript, ast.Attribute, ast.Starred)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def _base_self_attr(node):
    last_attr = None
    while isinstance(node, (ast.Subscript, ast.Attribute, ast.Starred)):
        if isinstance(node, ast.Attribute):
            last_attr = node
        node = node.value
    if isinstance(node, ast.Name) and node.id == "self" and last_attr is not None:
        return last_attr.attr
    return None


def _walk_targets(node, resolve):
    out = set()
    for t in _targets(node):
        parts = list(ast.walk(t)) if isinstance(t, (ast.Tuple, ast.List)) else [t]
        for part in parts:
            got = resolve(part)
            if got:
                out.add(got)
    return out


def engine_mutable_names():
    """{scope: set(names)} re-derived from the engine source, not from a hand-kept list."""
    kick = ast.parse(open(os.path.join(_ENGINE, "kick_probe.py")).read())
    fn = next(n for n in ast.walk(kick)
              if isinstance(n, ast.FunctionDef) and n.name == "simulate_kick")
    kick_names = set()
    for n in ast.walk(fn):
        kick_names |= _walk_targets(n, _base_name)

    slow = ast.parse(open(os.path.join(_ENGINE, "slow_field.py")).read())
    cls = next(n for n in ast.walk(slow)
               if isinstance(n, ast.ClassDef) and n.name == "SpatialSlowField")
    out = {"simulate_kick": kick_names}
    for m in [x for x in cls.body if isinstance(x, ast.FunctionDef)]:
        names = set()
        for n in ast.walk(m):
            names |= _walk_targets(n, _base_self_attr)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) \
                    and n.func.attr in ("append", "extend"):
                got = _base_self_attr(n.func.value)
                if got:
                    names.add(got)
        if names:
            out[f"SpatialSlowField.{m.name}"] = names
    return out


def unclassified_engine_names(extra_names=None):
    """Names the engine mutates that no bucket claims. Empty dict == the audit passes."""
    found = engine_mutable_names()
    for scope, names in (extra_names or {}).items():
        found.setdefault(scope, set())
        found[scope] = set(found[scope]) | set(names)
    kick_known = _KICK_STATE | _KICK_DERIVED | _KICK_OBSERVER | _KICK_TEMP
    slow_known = _SLOW_STATE | _SLOW_DERIVED
    out = {}
    for scope, names in found.items():
        if scope == "simulate_kick":
            rest = set(names) - kick_known
        else:
            rest = {n for n in names if not n.startswith("trace_")} - slow_known
        if rest:
            out[scope] = sorted(rest)
    return out


def audit_dynamic_state(extra_names=None):
    """The Phase-0A gate. status='ok' or 'blocked_state_inventory' (spec §13)."""
    rows = build_state_inventory()
    problems = []
    try:
        validate_inventory(rows)
    except ValueError as e:  # pragma: no cover - exercised via validate_inventory tests
        problems.append(str(e))
    unknown = unclassified_engine_names(extra_names=extra_names)
    if unknown:
        problems.append(f"unclassified engine mutables: {unknown}")
    return dict(status="ok" if not problems else "blocked_state_inventory",
                inventory_version=INVENTORY_VERSION,
                n_rows=len(rows),
                n_simulator_rows=sum(r["role"] == "simulator" for r in rows),
                unclassified=unknown, problems=problems,
                engine_scopes={k: len(v) for k, v in engine_mutable_names().items()})
