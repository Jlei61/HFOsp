"""Simulation glue: network cache, provenance, and one arm run.

Calls the blessed engine and the existing read-out chain; changes neither.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import tempfile

import numpy as np

CONNECTIVITY_FIELDS = (
    "L", "density", "f_E", "seed", "g",
    "C_EE", "C_IE", "C_EI", "C_II",
    "l_EE", "l_IE", "l_EI", "l_II",
    "rho_EE", "rho_IE", "rho_EI", "rho_II",
    "tau0", "v_axon", "delay_dt",
)
# Everything an artifact's numbers actually depend on. The placement helper was
# missing for a long time, so artifacts were stamped "clean" while depending on
# an uncommitted addition to it -- the stamp asserted reproducibility the tree
# could not deliver. If a module can change a number, it belongs here.
TRACKED_MODULES = (
    "src/topic4_core_field.py",
    "src/topic4_core_field_scoring.py",
    "src/topic4_core_field_report.py",
    "src/topic4_core_field_runner.py",
    "src/topic4_core_field_stage2.py",
    "src/topic4_core_field_stage3.py",
    "src/topic4_core_field_profile.py",
    "src/topic4_core_connectivity.py",
    "src/topic4_core_field_cmaes.py",
    "src/sef_hfo_subject_placement.py",
    "src/sef_hfo_snn_adapter.py",
    "src/sef_hfo_events.py",
    "src/sef_hfo_heterogeneity.py",
    "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py",
    "scripts/run_topic4_core_field_stage3_fit.py",
    "scripts/run_topic4_core_field_stage3_joint_fit.py",
    "scripts/run_topic4_core_field_stage3_joint_confirm.py",
    "scripts/run_topic4_core_field_stage3_confirm_fit.py",
    "scripts/run_topic4_core_field_stage3_profile_round1.py",
    "scripts/calibrate_topic4_core_field_stage3_joint_observable.py",
    "scripts/audit_topic4_data_driven_core_mechanism.py",
    "src/snn_engine/kick_probe.py",
    "src/snn_engine/params.py",
    "src/snn_engine/connectivity.py",
    "src/snn_engine/connectivity_rot.py",
    "src/snn_engine/lfp.py",
)


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def canonical_checksum(obj, drop=("checksum",)):
    """SHA256 of the canonical JSON with `drop` fields removed.

    Verification must recompute from content, never compare a stored string with
    itself (third-review P0-7).
    """
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k not in drop}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def provenance():
    """What code actually produced an artifact."""
    dirty = _git("status", "--porcelain", *TRACKED_MODULES, default="?")
    return dict(
        git_commit=_git("rev-parse", "HEAD"),
        tracked_modules_dirty=(bool(dirty.strip()) if dirty != "?" else None),
        module_sha256={m: hashlib.sha256(open(m, "rb").read()).hexdigest()
                       for m in TRACKED_MODULES if os.path.exists(m)},
        numpy_version=np.__version__,
    )


def connectivity_config(p, theta_deg, ar):
    """Every field that can change the connectivity graph. Keying on
    (seed, theta, L, density, AR) alone would silently hit a stale cache."""
    cfg = {f: getattr(p, f) for f in CONNECTIVITY_FIELDS}
    cfg["theta_EE_deg"] = float(theta_deg)
    cfg["AR"] = float(ar)
    cfg["numpy_version"] = np.__version__
    cfg["rng_bit_generator"] = "PCG64"
    cfg["git_commit"] = _git("rev-parse", "HEAD")
    return cfg


def cache_key(config):
    return canonical_checksum(config, drop=())


def get_network(p, theta_deg, ar, cache_dir):
    """Build or load the connectivity graph.

    Field-independent, so ONE build per (seed, theta) serves every arm. Written
    via a temp file plus atomic rename: Stage 1 parallelises over seeds precisely
    so two workers never race here, and the rename makes a partial file
    impossible even if that assumption is ever broken.
    """
    import sys
    eng = os.path.join("src", "snn_engine")
    for path in (eng, os.getcwd()):
        if path not in sys.path:
            sys.path.insert(0, path)
    from connectivity import place_neurons
    from connectivity_rot import build_connectivity_rot

    cfg = connectivity_config(p, theta_deg, ar)
    key = cache_key(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        return payload["net"], payload["NE"], payload["NI"], True

    rng = np.random.default_rng(p.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(theta_deg), AR=ar, verbose=False)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump({"net": net, "NE": NE, "NI": NI, "config": cfg},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return net, NE, NI, False


def atomic_write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)


def _placement(cfg):
    """Frozen shared-plane montage, core centroids and axis. Never refits."""
    from src.sef_hfo_subject_placement import (
        gradient_shared_template_foci, register_to_sheet, template_source_foci)
    m_real, _, _, _ = gradient_shared_template_foci(cfg["subject"], 3)
    _, src_names, snk_names = template_source_foci(cfg["subject"], "narrow", 3)
    reg = register_to_sheet(m_real, src_names, snk_names,
                            L=cfg["engine"]["L"], target_inter_core_mm=None)
    axis = reg["sink_centroid"] - reg["source_centroid"]
    reg["axis_unit_vec"] = axis / np.linalg.norm(axis)
    return reg


def run_arm_on_network(arm, seed, cfg, net, NE, NI, reg, cmrun):
    """One arm on an ALREADY-BUILT network. The caller owns the network so the
    eight arms at a seed share one build (third-review P0-8).

    k_dir and part_min are passed EXPLICITLY: read_event binds them from module
    globals at def time, so mutating cmrun.KDIR does nothing (Stage 0 parity).
    """
    from kick_probe import simulate_kick
    from lfp import LFPRecorder
    from params import Params
    from src.sef_hfo_events import detect_events
    from src.sef_hfo_heterogeneity import sample_core_field
    from src.sef_hfo_snn_adapter import snn_event_envelope
    from src.topic4_core_field import (
        arm_h, axis_coords, build_vth, core_thresholds, manual_mask,
        sample_core_quantiles, signed_depth)

    e = cfg["engine"]
    msheet = reg["montage_sheet"]
    src_xy, snk_xy = reg["source_centroid"], reg["sink_centroid"]
    axis_unit = reg["axis_unit_vec"]
    posE = net["pos"][:NE]
    is_E = np.zeros(len(net["pos"]), bool); is_E[:NE] = True
    mask = manual_mask(posE, src_xy, snk_xy, e["core_r"])

    if arm == "manual_hard":
        cf1 = sample_core_field(net["pos"], is_E, src_xy, e["core_r"],
                                np.random.default_rng(seed + 7), core_mean=e["core_mean"],
                                core_std=e["core_std"], base_mean=e["v_base"])
        cf2 = sample_core_field(net["pos"], is_E, snk_xy, e["core_r"],
                                np.random.default_rng(seed + 8), core_mean=e["core_mean"],
                                core_std=e["core_std"], base_mean=e["v_base"])
        vth = np.minimum(cf1["vth"], cf2["vth"])
        h_sum = float(mask.sum())
    else:
        s, r = axis_coords(posE, reg["center"], axis_unit)
        geom = dict(sep=float(np.linalg.norm(snk_xy - src_xy)),
                    s_support=(float(s.min()) + cfg["field"]["AXIAL_MARGIN"],
                               float(s.max()) - cfg["field"]["AXIAL_MARGIN"]),
                    M=cfg["field"]["M"], sigma_perp=e["core_r"],
                    shift_mm=cfg["field"]["SHIFT_MM"])
        h = arm_h(arm, s, r, geom, float(cfg["N_core_manual"]), manual_mask_E=mask)
        d = signed_depth(core_thresholds(
            sample_core_quantiles(NE, cfg["quantile_seed"]), e["core_mean"], e["core_std"]),
            e["v_base"])
        vth = build_vth(h, d, n_total=NE + NI, n_E=NE, v_base=e["v_base"])
        h_sum = float(h.sum())

    p = Params(g=e["g"], L=e["L"], density=e["density"], T=cfg["duration_ms"],
               dt=e["dt"], nu_ext_ratio=cmrun.DRIVE, seed=seed)
    k_dir = int(e["k_dir"])
    valid = cmrun.valid_mask(msheet, posE, e["L"], p.Rr)
    rec = LFPRecorder(p, net["pos"], net["labels"], sites=msheet.contacts)
    net["rng"] = np.random.default_rng(seed)
    res = simulate_kick(p, net, KICK_BOOST=0.0, kick_center=list(reg["center"]),
                        r_kick=e["core_r"], t_kick=1e9, V_th_per_neuron=vth,
                        lfp_recorder=rec)
    spk = res["E_spk_bool"]

    af, bin_w = cmrun.active_fraction(spk, e["dt"], cmrun.BIN_MS)
    nb0, nb1 = int(cmrun.BASELINE_MS[0] / bin_w), int(cmrun.BASELINE_MS[1] / bin_w)
    floor = float(np.percentile(af[nb0:nb1], 95)) if nb1 > nb0 else float(af.min())
    bar = floor + cmrun.CAL_FRAC * (float(af.max()) - floor)
    events = detect_events(af, bin_w, event_on_frac=bar)
    env_f, fdt, _ = snn_event_envelope(spk, posE, msheet, e["dt"])

    recs = []
    for ev in events:
        rd = cmrun.read_event(env_f, fdt, msheet, valid, (ev["t_on"], ev["t_off"]),
                              axis_unit, k_dir=k_dir, part_min=2 * k_dir + 1)
        recs.append(dict(n_part=int(rd["n_part"]), sign=rd["sign"], ranks=rd["ranks"]))
    return dict(arm=arm, seed=int(seed), events=recs, n_events=len(recs),
                h_sum=h_sum, config_checksum=cfg["checksum"], provenance=provenance())
