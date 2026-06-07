"""Quiet-but-excitable operating-point prefilter for Topic-4 SEF-HFO (2026-06-07).

WHY (plain language)
--------------------
We are looking for the resting state of cortex that the interictal HFO model should
sit in: a sheet that is SILENT on its own (the background does not light up by itself)
and is NOT a self-running oscillator (it is below the point where a travelling wave
would appear spontaneously) -- yet, if you poke a big enough patch of it hard enough,
that poke can launch a SINGLE self-extinguishing wave that travels in one direction
and then dies. This is the "excitable medium" picture. The smallest poke that still
launches such a wave is the "critical nucleus" -- below that size the poke just fizzles.

We sweep two knobs of the rate-field model:
  * g     = inhibition strength ratio (higher g = more inhibition = harder to ignite),
  * drive = external Poisson input as a fraction of threshold (sets the resting rate).
For each (g, drive) cell we ask, with NO new physics, only the existing classifier:
  1. is the rest state quiet (a no-poke run stays flat)?
  2. is the homogeneous state below the Turing-Hopf instability (linear stability
     "stable", i.e. it would NOT spontaneously break into a travelling wave)?
  3. is it excitable (does at least one poke radius launch a self-limited wave)?
  4. is it NOT pathological (no poke radius blows up into runaway / global synchrony)?
A cell passing ALL FOUR is a QUIET_EXCITABLE candidate; we record its critical nucleus.

This is a CHEAP prefilter for the expensive spiking-network (SNN) search: only the
quiet-but-excitable cells are worth building a big SNN for. We do NOT change any
verdict threshold and we do NOT sweep amplitude (fixed amp=8).

HOW (mechanics)
---------------
g is varied by reassigning the module inhibitory weights W_EI / W_II (they enter
mean_field, the linear-stability determinant, and the field integrator all via bare
module-global lookups -- so one reassignment reaches all three). w_ee_mult stays 1.0
(we never touch the excitatory gain / pathological branch). Linear stability uses the
module regime string == "stable" (NOT re_max<0: the module maps a failed root search
to re_max=-inf with regime="unresolved", and has a marginal "candidate" band, both of
which a raw re_max<0 test would wrongly admit). All four poke radii are classified
before criterion 4 is decided (an early exit could miss a runaway at a larger radius).

Run: PYTHONPATH="$PWD" python scripts/preflight_sef_hfo_excitable_phase.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.sef_hfo_lif as M
from src.sef_hfo_lif import (
    mean_field,
    lif_gains,
    closed_loop_leading,
    integrate_lif_field,
    classify_response,
    _grid,
    _STIM_T,
)

OUT = Path("results/topic4_sef_hfo/observation_layer/excitable_phase_prefilter.json")

# Grid (SMALL by design -- one amplitude, no amplitude sweep).
G_VALUES = [3.6, 3.8, 4.0, 4.1, 4.2]
DRIVE_VALUES = [0.3, 0.4, 0.5, 0.6, 0.7]
R_KICK_VALUES = [0.15, 0.4, 0.8, 1.2]
AMP = 8.0

# Field / stimulus geometry (task spec: 16mm box, off-center disk at x=-3).
N_GRID = 96
L_BOX = 16.0
STIM_X0 = -3.0
DT = 0.25
T_MAX = 300.0
THETA_EE = np.deg2rad(45.0)

# Criterion 1 quietness threshold (rest must not self-activate).
QUIET_MAX = 0.02

# Default module g (Brunel G_INH) -- used to restore between cells / at exit.
G_DEFAULT = 3.6


def set_g(g: float) -> None:
    """Reassign module inhibitory weights so g reaches mean_field, the linear-stability
    determinant, and the field integrator (all read W_EI / W_II as module globals)."""
    M.W_EI = 1.07 * g * 0.1575   # = 1.07 * g * W_EE_base
    M.W_II = g * 0.2625          # = g * W_IE_base


def _disk_stim(r_kick: float):
    """Off-center disk poke at (STIM_X0, 0), radius r_kick, amplitude AMP, on for _STIM_T."""
    X, Y = _grid(N_GRID, L_BOX)
    mask = ((X - STIM_X0) ** 2 + Y ** 2 <= r_kick ** 2).astype(float)
    zero = 0.0 * mask
    on = AMP * mask
    return lambda t: (on if t < _STIM_T else zero)


def _run_cell(g: float, drive: float) -> dict:
    """One (g, drive) cell: operating point, linear stability, no-kick quietness,
    and a classified response for each kick radius. Returns a fully-populated row."""
    set_g(g)
    row: dict = {"g": g, "drive": drive}

    # --- operating point (may fail to find a clean root) ---
    try:
        op = mean_field(drive)
    except RuntimeError as exc:
        row["mean_field_failed"] = True
        row["error"] = str(exc)
        row["candidate"] = False
        return row
    row["mean_field_failed"] = False
    row["nuE_Hz"] = float(op["nuE"] * 1000.0)
    row["nuI_Hz"] = float(op["nuI"] * 1000.0)
    row["n_clean_roots"] = int(op["n_clean_roots"])

    # --- linear stability (Turing-Hopf gate) ---
    gains = lif_gains(op)
    stab = closed_loop_leading(gains["E"], gains["I"])
    row["re_max"] = float(stab["re_max"])
    row["k_star"] = float(stab["k_star"])
    row["regime"] = str(stab["regime"])
    row["is_hopf"] = bool(stab["is_hopf"])
    row["freq_Hz"] = float(stab["freq_Hz"])
    row["stab_converged"] = bool(stab["converged"])

    # --- criterion 1: background quietness (no-kick reference) ---
    ext0, _front0 = integrate_lif_field(
        op, lambda t: 0.0, dt=DT, t_max=T_MAX, theta_EE=THETA_EE, n=N_GRID, L=L_BOX
    )
    nokick_max_ext = float(ext0.max())
    row["nokick_max_ext"] = nokick_max_ext
    background_quiet = nokick_max_ext < QUIET_MAX

    # --- per-kick classification (classify ALL radii before deciding crit 4) ---
    kicks = []
    for r_kick in R_KICK_VALUES:
        stim = _disk_stim(r_kick)
        ext, front = integrate_lif_field(
            op, stim, dt=DT, t_max=T_MAX, theta_EE=THETA_EE, n=N_GRID, L=L_BOX
        )
        lbl, info = classify_response(ext, front, stim_x0=STIM_X0, stim_r=r_kick, dt=DT)
        kicks.append({
            "r_kick_mm": r_kick,
            "label": lbl,
            "max_ext": float(info["max_ext"]),
            "adv_mm": float(info["adv_mm"]),
            "returned": bool(info["returned"]),
            "dur_ms": float(info["dur_ms"]),
        })
    row["kicks"] = kicks

    labels = [k["label"] for k in kicks]
    prop_radii = [k["r_kick_mm"] for k in kicks if k["label"] == "self_limited_propagation"]
    pathological = any(l in ("runaway", "global_synchronous") for l in labels)

    not_self_oscillating = row["regime"] == "stable"   # encode crit 2 as regime string
    excitable = len(prop_radii) > 0
    critical_nucleus_mm = min(prop_radii) if prop_radii else None

    row["background_quiet"] = bool(background_quiet)
    row["not_self_oscillating"] = bool(not_self_oscillating)
    row["excitable"] = bool(excitable)
    row["pathological"] = bool(pathological)
    row["critical_nucleus_mm"] = critical_nucleus_mm
    row["candidate"] = bool(
        background_quiet and not_self_oscillating and excitable and not pathological
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    rows = []
    try:
        for g in G_VALUES:
            for drive in DRIVE_VALUES:
                rows.append(_run_cell(g, drive))
    finally:
        set_g(G_DEFAULT)   # leave the module as we found it

    candidates = [r for r in rows if r.get("candidate")]
    candidates_sorted = sorted(
        candidates, key=lambda r: (r["critical_nucleus_mm"], r["g"], -r["drive"])
    )
    failed = [{"g": r["g"], "drive": r["drive"], "error": r.get("error")}
              for r in rows if r.get("mean_field_failed")]

    out = {
        "description": (
            "Quiet-but-excitable operating-point prefilter for the SEF-HFO SNN search. "
            "Per (g, drive) cell: operating point, linear stability (Turing-Hopf gate), "
            "no-kick background quietness, and classified response for each kick radius. "
            "QUIET_EXCITABLE candidate = background quiet AND linear-stability regime "
            "'stable' (below Turing-Hopf) AND >=1 kick gives self_limited_propagation "
            "AND no kick gives runaway/global_synchronous."
        ),
        "grid": {
            "g": G_VALUES, "drive": DRIVE_VALUES, "r_kick_mm": R_KICK_VALUES,
            "amp": AMP, "n_grid": N_GRID, "L_box_mm": L_BOX, "stim_x0_mm": STIM_X0,
            "dt_ms": DT, "t_max_ms": T_MAX, "theta_EE_deg": 45.0,
            "stim_duration_ms": float(_STIM_T), "quiet_max_ext": QUIET_MAX,
        },
        "cells": rows,
        "candidates": candidates_sorted,
        "n_candidates": len(candidates_sorted),
        "failed_cells": failed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))

    # --- stdout summary table ---
    print(f"\nSEF-HFO quiet-but-excitable prefilter  (amp={AMP}, {N_GRID}px / {L_BOX}mm)")
    print(f"{'g':>4} {'drive':>6} {'nuE_Hz':>9} {'regime':>11} {'k*':>5} "
          f"{'nuc_mm':>7} {'cand':>5}")
    print("-" * 56)
    for r in rows:
        if r.get("mean_field_failed"):
            print(f"{r['g']:>4} {r['drive']:>6} {'FAILED (mean_field no clean root)':>40}")
            continue
        nuc = r["critical_nucleus_mm"]
        nuc_s = f"{nuc:.2f}" if nuc is not None else "  -"
        print(f"{r['g']:>4} {r['drive']:>6} {r['nuE_Hz']:>9.4f} {r['regime']:>11} "
              f"{r['k_star']:>5.2f} {nuc_s:>7} {('YES' if r['candidate'] else ''):>5}")
    print("-" * 56)
    print(f"{len(candidates_sorted)} QUIET_EXCITABLE candidate cell(s); "
          f"{len(failed)} mean_field failure(s).")
    if candidates_sorted:
        print("Best candidates (smallest critical nucleus first):")
        for r in candidates_sorted[:3]:
            print(f"  g={r['g']}  drive={r['drive']}  nucleus={r['critical_nucleus_mm']:.2f}mm  "
                  f"nuE={r['nuE_Hz']:.4f}Hz  regime={r['regime']}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
