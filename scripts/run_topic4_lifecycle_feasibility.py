#!/usr/bin/env python
"""Compute lifecycle design diagnostics for the registered Z/M substrate.

Every constant is read from its own source of truth -- the engine parameter file, the
substrate builder, and the locked result artifacts of the runs that actually executed
-- so the verdict tracks the code and cannot drift from prose.

    python scripts/run_topic4_lifecycle_feasibility.py

The output is deliberately diagnostic: risk flags are not a scientific NO-GO.
Writes ``results/topic4_sef_hfo/zm_lifecycle_feasibility/feasibility_verdict.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from params import Params                                      # noqa: E402
import run_m4_phaseplane as PP                                 # noqa: E402
from src.topic4_lifecycle_feasibility import (                 # noqa: E402
    LIFECYCLE_FEASIBILITY_VERSION,
    brake_authority,
    operating_point_headroom,
    screen_mechanism,
    slow_variable_reversal,
    timescale_separation,
)

RES = os.path.join(ROOT, "results", "topic4_sef_hfo")
LIFECYCLE_JSON = os.path.join(RES, "zm_snn_native_exit", "lifecycle_seed1.json")
FUTILITY_JSON = os.path.join(RES, "zm_phase_c_tonic_identity", "phasec_futility_verdict.json")
REFERENCE_JSON = os.path.join(
    RES, "zm_fast_carrier_repair", "calibration", "dynamic_preentry",
    "reference__noise_replay.json",
)
OUT_DIR = os.path.join(RES, "zm_lifecycle_feasibility")

# z sits near 0.75 at baseline by construction (I_th_EI is calibrated as the q75 of
# the interictal E-cell inhibitory current, so z_inf is 1 for ~75% of timesteps), and
# the bounded arms reach z_core ~ 0.34-0.37 in their high-activity state
# (lifecycle_seed1.json z_core_final).  The latch verdict does NOT depend on where
# exactly the entry point is placed: z_inf_ictal = 0 lies beyond ANY entry point in
# (0, 0.75), so the check fails for every admissible choice.
Z_INTERICTAL = 0.75
Z_ENTRY = 0.30
Z_INF_ICTAL = 0.0     # z_inf = H(I_th_EI - I_I) is 0 for every elevated-inhibition state


def _load(path, label):
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {label}: {path}\n"
            "This screen reads the locked artifacts of the runs that executed; "
            "it does not re-simulate."
        )
    with open(path) as fh:
        return json.load(fh)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--target-ictal-duration-ms", type=float, default=20000.0,
        help="registered design target for a terminating ictal state (NOT a measurement)",
    )
    a = ap.parse_args()

    p = Params()
    life = _load(LIFECYCLE_JSON, "Z/M lifecycle result")
    futility = _load(FUTILITY_JSON, "Phase C futility verdict")
    reference = _load(REFERENCE_JSON, "Phase D dynamic interictal reference")

    cfg = life["rows"][0]["cfg"]
    eta_m, tau_adp = float(life["eta_m"]), float(life["tau_adp"])
    tau_S, alpha_G = float(cfg["tau_S"]), 16.0     # alpha_G of the containment arms (ARMS['sg'])
    v_th_core = float(PP.CORE_MEAN)

    events = reference["returning_events"]
    ied_interval_ms = float(reference["T_ms"]) / int(events["n_events"])

    core_rate = futility["seed1_primary_futility"]["core_rate_mean_hz"]
    modulation = futility["seed1_primary_futility"]["modulation_depth"]
    modulation_gate = float(futility["registered_logic"]["run_positive_modulation_min"])

    checks = [
        brake_authority(
            name="m_adaptation",
            gain_mv_per_unit=eta_m,
            tau_accum_ms=tau_adp,
            tau_ref_ms=p.tau_ref_E,
            v_th_mv=v_th_core,
            v_reset_mv=p.V_reset,
        ),
        slow_variable_reversal(
            name="z_inhibitory_efficacy",
            u_interictal=Z_INTERICTAL,
            u_entry=Z_ENTRY,
            u_inf_ictal=Z_INF_ICTAL,
        ),
        timescale_separation(
            name="S_G_divisive_pool",
            tau_recover_ms=tau_S,
            interictal_event_interval_ms=ied_interval_ms,
            target_ictal_duration_ms=a.target_ictal_duration_ms,
        ),
        operating_point_headroom(
            name="phase_c_tonic_core",
            target_rate_hz=float(core_rate["median"]),
            tau_ref_ms=p.tau_ref_E,
        ),
    ]

    verdict = screen_mechanism("zm_registered_lifecycle_substrate", checks)
    verdict["provenance"] = {
        "engine_params": "src/snn_engine/params.py::Params",
        "core_threshold": "scripts/run_m4_phaseplane.py::CORE_MEAN",
        "slow_var_constants": os.path.relpath(LIFECYCLE_JSON, ROOT),
        "interictal_event_interval": os.path.relpath(REFERENCE_JSON, ROOT),
        "operating_point": os.path.relpath(FUTILITY_JSON, ROOT),
    }
    verdict["substrate_constants"] = {
        "V_th_base_mv": p.V_th,
        "V_th_core_mv": v_th_core,
        "V_reset_mv": p.V_reset,
        "tau_ref_E_ms": p.tau_ref_E,
        "tau_m_E_ms": p.tau_m_E,
        "eta_m_mv_per_unit": eta_m,
        "tau_adp_ms": tau_adp,
        "tau_z_ms": float(life["tau_z"]),
        "alpha_G": alpha_G,
        "tau_S_ms": tau_S,
        "interictal_event_interval_ms": ied_interval_ms,
        "n_returning_events": int(events["n_events"]),
        "target_ictal_duration_ms": a.target_ictal_duration_ms,
    }
    # Descriptive quantities only.  tau_S is a post-drive decay scale, not a bound
    # on the duration of a state that a continuously driven pool can affect.
    verdict["derived"] = {
        "S_G_post_activity_decay_ms": tau_S,
        "S_G_divisive_ceiling_fraction": alpha_G / (1.0 + alpha_G),
        "phase_c_core_rate_hz": core_rate,
        "phase_c_modulation_depth": modulation,
        "registered_non_tonic_modulation_gate": modulation_gate,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, "feasibility_verdict.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2, sort_keys=True)

    print(f"[{LIFECYCLE_FEASIBILITY_VERSION}] {verdict['mechanism']}: {verdict['verdict']}")
    for c in verdict["checks"]:
        print(f"  [{'clear' if c['passed'] else 'RISK'}] {c['name']}: {c['reason']}")
    print(f"\n[write] {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
