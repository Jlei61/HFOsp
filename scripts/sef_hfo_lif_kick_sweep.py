"""Closure #2 — the rate-field's excitable window across drive (the "0.6 re-run" as a sweep).

For each external-drive ratio, set the canonical LIF rate-field operating point (fsolve
mean_field) and fire the validated off-center finite kick; classify the response
(extinction / local_bump / self_limited_propagation / global_synchronous / runaway).
Finds WHERE the rate-field is excitable -- not hard-coded to 0.6, because at low drive the
rest is near-silent and a kick may just fizzle (advisor's caveat).

Caveat: integrate_lif_field uses the white-noise mean_field operating point (the gain that
closure #1 showed is an UNDERESTIMATE vs the spiking-measured rate). So the rate-field's
excitable window in nominal ratio is shifted relative to the spiking net by that gain offset;
the comparison is mechanism (excitable->self-limited) + window shape, not ratio-for-ratio.

Run: python scripts/sef_hfo_lif_kick_sweep.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sef_hfo_lif import (  # noqa: E402
    mean_field, integrate_lif_field, classify_response,
    _grid, _DEFAULT_N, _DEFAULT_L, _STIM_X0, _STIM_R, _STIM_T,
)

OUT = Path("results/topic4_sef_hfo/lif_snn/data")
A = 8.0


def run(ratio):
    op = mean_field(ratio)
    n, L = _DEFAULT_N, _DEFAULT_L
    X, Y = _grid(n, L)
    mask = ((X - _STIM_X0) ** 2 + Y ** 2 <= _STIM_R ** 2).astype(float)
    stim = lambda t: (A * mask) if t < _STIM_T else (0.0 * mask)
    ext, front = integrate_lif_field(op, stim, dt=0.25, t_max=300.0, b_a=0.0,
                                     theta_EE=0.0, n=n, L=L)
    lbl, info = classify_response(ext, front, stim_x0=_STIM_X0, stim_r=_STIM_R, dt=0.25)
    return dict(ratio=ratio, nuE_Hz=op["nuE"] * 1000.0, label=lbl,
                adv_mm=info["adv_mm"], max_ext=info["max_ext"],
                dur_ms=info["dur_ms"], returned=bool(info["returned"]))


def main():
    ratios = [0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3]
    rows = [run(r) for r in ratios]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kick_sweep.json").write_text(json.dumps({"A": A, "per_ratio": rows}, indent=2, default=float))
    print("ratio nuE(Hz)  label                       adv_mm max_ext dur_ms ret")
    for r in rows:
        print(f"{r['ratio']:.2f} {r['nuE_Hz']:6.3f}  {r['label']:27s} {r['adv_mm']:5.1f} "
              f"{r['max_ext']:.3f} {r['dur_ms']:5.0f} {str(r['returned'])[0]}")
    sl = [r["ratio"] for r in rows if r["label"] == "self_limited_propagation"]
    print(f"\nexcitable window (self_limited_propagation): {sl}")


if __name__ == "__main__":
    main()
