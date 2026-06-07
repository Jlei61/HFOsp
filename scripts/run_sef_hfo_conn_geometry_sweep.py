"""Track G geometry sweep (spec §5.1): ρ and ℓ_∥ at FIXED total E→E mass."""
import json
import numpy as np
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.sef_hfo_lif import mean_field, integrate_lif_field, classify_response, ELL_PAR
from src.sef_hfo_field import _grid
from scripts.sef_hfo_step0d_anisotropy_control import principal_axis

OUT = Path("results/topic4_sef_hfo/connectivity_geometry/geometry_sweep.json")


def offcenter_pulse(A=8.0, T=30.0, R=2.0, x0=-3.0, n=96, L=12.0):
    X, Y = _grid(n, L)
    disk = ((X - x0) ** 2 + Y ** 2) <= R ** 2
    return lambda t: (A * disk if t < T else 0.0)


def run():
    op = mean_field(1.0)
    rho_rows = []
    for rho in [1.0, 0.9, 0.75, 0.5, 0.33]:
        ell_par = 0.9
        ell_perp = rho * ell_par
        out = integrate_lif_field(op, offcenter_pulse(), ell_par=ell_par,
                                  ell_perp=ell_perp, return_peak_field=True, t_max=150.0)
        ext, front, peak = out[0], out[1], out[2]
        angle, ratio = principal_axis(peak - op["nuE"])
        label, info = classify_response(ext, front)
        rho_rows.append(dict(rho=rho, ell_par=ell_par, ell_perp=ell_perp,
                             axis_angle_deg=angle, anisotropy_ratio=ratio,
                             label=label, adv_mm=info["adv_mm"], dur_ms=info["dur_ms"]))
    ell_rows = []
    for ell_par in [0.36, 0.54, 0.72, 0.9, 1.2]:
        ell_perp = 0.5 * ell_par  # hold rho=0.5, vary scale
        out = integrate_lif_field(op, offcenter_pulse(), ell_par=ell_par,
                                  ell_perp=ell_perp, return_field=True, t_max=150.0)
        ext, front = out[0], out[1]
        label, info = classify_response(ext, front)
        ell_rows.append(dict(ell_par=ell_par, ell_perp=ell_perp, label=label,
                             adv_mm=info["adv_mm"], dur_ms=info["dur_ms"], max_ext=info["max_ext"]))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(
        note="Track G geometry sweep at fixed total E→E mass (unit-sum kernel). "
             "θ_EE rotation locked separately in step0d.",
        operating_point=dict(nuE=op["nuE"], nuI=op["nuI"]),
        rho_sweep=rho_rows, ell_sweep=ell_rows), indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    run()
