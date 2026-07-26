#!/usr/bin/env python
"""Freeze the PRE-EDIT `simulate_kick` default-path outputs before the checkpoint hook lands.

Run this ONCE, on the unmodified engine, so `tests/test_topic4_zm_checkpoint_hook.py` can assert
byte equality against a real pre-edit trajectory rather than against the post-edit code judging
itself (spec rev3.1 §3.1 step 1-3). Records the pre-edit guarded SHA alongside the arrays.

  python scripts/gen_topic4_zm_preedit_fixture.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
for _p in (_ROOT, os.path.join(_ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

FIX = os.path.join(_ROOT, "tests", "fixtures", "topic4_zm_preedit_parity.npz")


def build_cases():
    """(name, runner) pairs. Case A is the historic BASELINE_SHA anchor (slow=None). Case B drives
    the real Z/M+S_G layer with a kick so delay rings, refractory, recurrent current and the slow
    fields are all non-trivial at the fork point."""
    from params import Params
    from connectivity import place_neurons, build_connectivity
    from kick_probe import simulate_kick
    from slow_field import SpatialSlowField, SpatialSlowFieldConfig
    from lfp import LFPRecorder

    def substrate(seed=1, T=200.0):
        p = Params(L=1.0, density=400.0, T=T, dt=0.1, seed=seed, nu_ext_ratio=1.0)
        rng = np.random.default_rng(seed)
        pos, labels, NE, NI = place_neurons(p, rng)
        net = build_connectivity(p, pos, labels, NE, NI, rng, verbose=False)
        return p, net, pos, labels, NE, NI

    def case_A():
        p, net, pos, labels, NE, NI = substrate()
        net["rng"] = np.random.default_rng(p.seed)
        return simulate_kick(p, net, 5.0, kick_center=np.array([p.L / 2, p.L / 2]), r_kick=0.3,
                             t_kick=50.0, verbose=False)

    def case_B():
        p, net, pos, labels, NE, NI = substrate()
        N = NE + NI
        vth = np.full(N, 18.0)
        vth[:5] = 16.0
        cfg = SpatialSlowFieldConfig(use_qI=False, use_gK=False, use_z=True, use_m=True,
                                     tau_z=200.0, I_th_EI=0.6, tau_adp=200.0, eta_m=0.5,
                                     use_SG=True, alpha_G=16.0, r50_psi=0.05, n_grid=16)
        core = np.linalg.norm(pos[:NE] - np.array([p.L / 2, p.L / 2]), axis=1) <= 0.3
        slow = SpatialSlowField(N, 18.0, pos[:NE], pos[NE:], p.L, core_mask_E=core, cfg=cfg)
        rec = LFPRecorder(p, pos, labels)
        net["rng"] = np.random.default_rng(p.seed)
        return simulate_kick(p, net, 5.0, slow=slow, kick_center=np.array([p.L / 2, p.L / 2]),
                             r_kick=0.3, t_kick=50.0, V_th_per_neuron=vth, lfp_recorder=rec,
                             verbose=False, early_stop_runaway=True, es_thresh_hz=400.0)

    return [("A_plain", case_A), ("B_zm_sg_lfp", case_B)]


def main():
    from src.topic4_zm_fork_state import sha256_file
    out = {}
    for name, fn in build_cases():
        res = fn()
        out[f"{name}__E_spk_bool"] = res["E_spk_bool"]
        out[f"{name}__rate_E"] = res["rate_E"]
        out[f"{name}__rate_I"] = res["rate_I"]
        out[f"{name}__spk_inside"] = res["spk_inside"]
        out[f"{name}__spk_outside"] = res["spk_outside"]
        if res.get("lfp_trace") is not None:
            out[f"{name}__lfp_trace"] = res["lfp_trace"]
        print(f"[{name}] steps={res['E_spk_bool'].shape[0]} spikes={int(res['E_spk_bool'].sum())} "
              f"sha1(raster)={hashlib.sha1(res['E_spk_bool'].tobytes()).hexdigest()[:16]}")
    os.makedirs(os.path.dirname(FIX), exist_ok=True)
    np.savez_compressed(FIX, **out)
    meta = dict(pre_edit_kick_probe_sha256=sha256_file(os.path.join(_ROOT, "src/snn_engine/kick_probe.py")),
                pre_edit_slow_field_sha256=sha256_file(os.path.join(_ROOT, "src/snn_engine/slow_field.py")),
                cases=[n for n, _ in build_cases()], arrays=sorted(out))
    with open(FIX.replace(".npz", ".json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[fixture] wrote {FIX}\n[fixture] pre-edit kick_probe SHA256 = "
          f"{meta['pre_edit_kick_probe_sha256']}")


if __name__ == "__main__":
    main()
