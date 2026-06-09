"""SNN heterogeneity multi-seed + position sweep (next-step roadmap priority 1).

Answers the two load-bearing questions the 3-seed grid left open:
  (A) is the matched (variance-only) NULL robust across seeds + positions?
  (B) does variance modulate the ignited-state magnitude (mean_only vs unmatched)
      once we have proper CIs, or is it still unresolved?
Plus the ignition mean-gating across positions, and the position dependence of the
ignited-state index.

Design: fixed kick at the axis end; sweep the pathology-core POSITION (along-axis
line + off-axis), all OFF the kick; per (position, seed) run baseline + the 3
conditions PAIRED (same noise/kick/surround), compute Δ vs baseline + ignition +
self-limit. Aggregate per (position, condition): ignition rate, and Δ split into
evoked (non-igniting seeds) vs ignited-index (igniting seeds) with mean±sem.

Reuses the validated grid `_run_one`/`_build`/`_montage`/`_provenance` (DRY).
Kick amplitude sweep is a separate phase (roadmap), not here.
Modes:  --quick (L=1, 2 pos x 2 seeds) ; default --full (L=3, --seeds N, ~7 pos)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "scripts")
import run_sef_hfo_snn_hetero_grid as G            # noqa: E402  (reuse engine-backed helpers)
sys.path.insert(0, os.getcwd())
from src.sef_hfo_heterogeneity import sample_threshold_fields   # noqa: E402

OUT = G.OUT
CONDS = G.CONDS
PATCH_R = G.PATCH_R


def _positions(L):
    """Along-axis line (5) + off-axis (2), all OFF the end-kick. Returns
    (positions dict name->xy, kick xy)."""
    u = np.array([np.cos(np.deg2rad(G.THETA)), np.sin(np.deg2rad(G.THETA))])
    perp = np.array([-u[1], u[0]]); ctr = np.array([L / 2, L / 2])
    along = {f"axis{f:+.1f}": tuple(ctr + f * (L / 2) * u)
             for f in (-0.5, -0.3, -0.1, 0.1, 0.3)}
    off = {"offaxis+": tuple(ctr + 0.35 * (L / 2) * perp),
           "offaxis-": tuple(ctr - 0.35 * (L / 2) * perp)}
    return {**along, **off}, tuple(ctr + 0.75 * (L / 2) * u)


def _ci(v):
    v = np.asarray(v, float)
    n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    return dict(mean=float(v.mean()) if n else float("nan"), sd=sd,
                sem=float(sd / np.sqrt(n)) if n > 1 else 0.0, n=int(n), vals=v.tolist())


def _sweep(L, density, positions, kick, seeds):
    G._engine_guard()
    p, net, nu_theta, NE = G._build(L, density, seeds[0])
    is_E = net["labels"] == 0
    montage = G._montage(L)
    rows = []; t0 = time.time(); k = 0; tot = len(positions) * len(seeds)
    for pname, pc in positions.items():
        on_patch = bool(np.linalg.norm(np.array(kick) - np.array(pc)) <= PATCH_R)
        for s in seeds:
            fields = sample_threshold_fields(net["pos"], is_E, pc, PATCH_R,
                                             np.random.default_rng(s))
            cm = fields["core_mask"]
            b = G._run_one(p, net, nu_theta, NE, kick, fields["baseline"], montage, cm, s)
            rec = dict(position=pname, patch=list(pc), seed=s, kick_on_patch=on_patch,
                       base_prekick_ignited=b["prekick_ignited"], base_rest_rate=b["rest_rate"])
            for cond in CONDS:
                c = G._run_one(p, net, nu_theta, NE, kick, fields[cond], montage, cm, s)
                rec[f"d_core_{cond}"] = c["core_paf"] - b["core_paf"]
                rec[f"ignited_{cond}"] = c["prekick_ignited"]
                rec[f"iglat_{cond}"] = c["ignition_latency"]
                rec[f"returned_{cond}"] = c["returned"]
                rec[f"tail_{cond}"] = c["tail_complete"]      # returned is weak where False
            rows.append(rec); k += 1
            print(f"[{k}/{tot}] {pname} seed{s} " + " ".join(
                f"{cc[:4]}={rec[f'd_core_{cc}']:+.2f}(ig{int(rec[f'ignited_{cc}'])})"
                for cc in CONDS) + f" ({time.time()-t0:.0f}s)", flush=True)

    agg = {}
    for pname in positions:
        prows = [r for r in rows if r["position"] == pname]
        agg[pname] = {"kick_on_patch": prows[0]["kick_on_patch"]}
        for cond in CONDS:
            ig = np.array([r[f"ignited_{cond}"] for r in prows], bool)
            d = np.array([r[f"d_core_{cond}"] for r in prows], float)
            tc = np.array([r.get(f"tail_{cond}", True) for r in prows], bool)
            agg[pname][cond] = dict(
                ignition_rate=float(ig.mean()),
                d_core_evoked=(_ci(d[~ig]) if (~ig).any() else None),   # non-igniting seeds
                d_core_ignited=(_ci(d[ig]) if ig.any() else None),      # igniting seeds (index)
                returned_rate=float(np.mean([r[f"returned_{cond}"] for r in prows])),
                tail_complete_rate=float(tc.mean()))                    # <1.0 → returned is weak
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweep_metrics.json").write_text(json.dumps(
        dict(provenance=G._provenance(p, seeds[0]), kick=list(kick), n_seeds=len(seeds),
             positions={n: list(v) for n, v in positions.items()},
             aggregate=agg, raw=rows), indent=2))
    print(f"wrote sweep_metrics.json ({len(rows)} position-seed cells, "
          f"{len(seeds)} seeds x {len(positions)} pos, {time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seeds", type=int, default=12)
    a = ap.parse_args()
    if a.quick:
        L, dens = 1.0, 4000.0
        pos, kick = _positions(L)
        pos = dict(list(pos.items())[:2])
        _sweep(L, dens, pos, kick, list(range(1, 3)))
    else:
        L, dens = 3.0, 1800.0
        pos, kick = _positions(L)
        _sweep(L, dens, pos, kick, list(range(1, a.seeds + 1)))


if __name__ == "__main__":
    main()
