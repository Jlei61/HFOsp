"""P0a offline calibration for the M4-3A load->shunt variable (spec 4.0, 6.1).

Feed real M4-1 Arm0 rate traces (4 regimes) through the n->a ODE WITHOUT running the
network, check a(t) against the calibration table, and evaluate the sensor-free HARD
gate (Delta_a_IED, R_A). Produces calibrated params so the network sweep (Task 8) does
not waste budget, and so n50 is not a soft ictal gate.

PROXY (P1-4): the runbook feeds '{label}__rate' (global sheet-mean rate) as u_n, which
is a cheap proxy (= P0a). Locking u_n0/n50 needs the field-derived u_n (Task 4
trace_un_mean) from an Arm0 replay (= P0b). Trace source:
results/topic4_m4_dynamic_p1_sweep*/p1_sweep_traces.npz (Arm0 label = 'p1_arm0').
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from src.sef_hfo_m4_load_shunt import (
    LoadShuntParams, load_shunt_step, event_triggered_a_response, compute_R_A)


def run_a_trace(u_series, dt, p: LoadShuntParams):
    """Integrate the ODE over a 1D drive series; return the a(t) series."""
    u = np.asarray(u_series, float)
    n = p.n_base
    a_out = np.empty(u.size, dtype=float)
    for t in range(u.size):
        n, a = load_shunt_step(n, u[t], dt, p)
        a_out[t] = float(a)
    return a_out


def calibrate_regimes(regime_series, dt, p: LoadShuntParams, event_idx=None, a_block=None):
    """Run each regime's drive through the ODE, summarize a(t), and evaluate the sensor-free
    HARD gate (P1-3). a_block = network-measured a level that blocks an IED kick (P0b); None
    -> block check pending (not a pass). Returns {'table', 'metrics', 'gate'}."""
    event_idx = event_idx or {}
    table, a_by_regime = {}, {}
    for name, u in regime_series.items():
        a = run_a_trace(u, dt, p)
        a_by_regime[name] = a
        row = {"a_max": float(a.max()), "a_mean": float(a.mean()),
               "a_mid": float(a[a.size // 2]), "a_end": float(a[-1])}
        if name in event_idx:
            row["delta_a_ied"] = event_triggered_a_response(
                a, event_idx[name], dt, pre_ms=200, post0_ms=10, post1_ms=200)
        table[name] = row

    metrics, gate = {}, {}
    d = table.get("isolated_ied", {}).get("delta_a_ied")
    if d is not None and "bounded_ictal" in a_by_regime:
        ic = a_by_regime["bounded_ictal"]
        n1s = int(round(1000.0 / dt))
        delta_ictal = float(ic.max() - ic[:min(n1s, ic.size)].mean())   # 1s+ ictal accumulation
        metrics["R_A"] = compute_R_A(delta_ictal, d)
        sigma_bl = float(a_by_regime.get("quiet", ic).std())
        gate["sigma_baseline"] = sigma_bl
        gate["delta_a_ied"] = float(d)
        gate["soft_gate_fail"] = bool(d <= 0.0)                          # IED did not move a -> hidden sensor
        gate["baseline_jitter_pass"] = bool(d >= 2.0 * sigma_bl)
        gate["magnitude_pass"] = bool(d >= 0.005 * p.a_max)
        gate["delta_ied_pass"] = bool(d > 0.0 and gate["baseline_jitter_pass"] and gate["magnitude_pass"])
        gate["R_A"] = metrics["R_A"]
        gate["R_A_pass"] = bool(d > 0.0 and gate["R_A"] >= 5.0)          # d>0 guard: inf can't sneak through
        if "quiet" in a_by_regime:
            gate["a_interictal_mean"] = float(a_by_regime["quiet"].mean())
            gate["interictal_block_pass"] = (bool(gate["a_interictal_mean"] < a_block)
                                             if a_block is not None else None)
        else:
            gate["a_interictal_mean"] = None
            gate["interictal_block_pass"] = None
        gate["sensor_free_pass"] = bool(
            gate["delta_ied_pass"] and gate["R_A_pass"] and (not gate["soft_gate_fail"])
            and gate["interictal_block_pass"] is True)                   # None (pending) / False -> not certified
    return {"table": table, "metrics": metrics, "gate": gate}


def _load_regime(npz_path, key, seg=None):
    arr = np.asarray(np.load(npz_path)[key], float)
    if seg is not None:
        arr = arr[seg[0]:seg[1]]
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--tau-n", type=float, default=20000.0)
    ap.add_argument("--k-n", type=float, default=1.0)
    ap.add_argument("--rho-n", type=float, default=0.1)
    ap.add_argument("--n50", type=float, default=0.4)
    ap.add_argument("--hill-h", type=float, default=2.0)
    ap.add_argument("--a-max", type=float, default=1.0)
    ap.add_argument("--u-n0", type=float, default=0.0)
    ap.add_argument("--a-block", type=float, default=None)   # P0b: network-measured IED-block a level
    # --regime name=npz:key[:start:end]  (repeatable); --event name=idx[,idx]
    ap.add_argument("--regime", action="append", default=[])
    ap.add_argument("--event", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    p = LoadShuntParams(tau_n=a.tau_n, k_n=a.k_n, rho_n=a.rho_n, n_base=0.0, n50=a.n50,
                        hill_h=a.hill_h, a_max=a.a_max, u_n0=a.u_n0)
    p.validate()
    regimes = {}
    for spec in a.regime:
        name, rhs = spec.split("=", 1)
        parts = rhs.split(":")
        npz, key = parts[0], parts[1]
        seg = (int(parts[2]), int(parts[3])) if len(parts) == 4 else None
        regimes[name] = _load_regime(npz, key, seg)
    events = {}
    for spec in a.event:
        name, idx = spec.split("=", 1)
        events[name] = [int(x) for x in idx.split(",")]
    out = calibrate_regimes(regimes, a.dt, p, events, a_block=a.a_block)
    out["params"] = p.__dict__
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "p0_calibration.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
