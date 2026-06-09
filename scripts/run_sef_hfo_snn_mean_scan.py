"""SNN mean-amplitude scan (next-step, 2026-06-09 review). Verifies the joint law
the sweep left open: how the two axes of the local threshold distribution —
mean-down amplitude × spread width — JOINTLY set core self-ignition. Answers the
core question: is mean=16 just "a too-strong rung (ignites the moment you lower)",
or is there a REAL ignition BOUNDARY as the core mean walks 18→16, and does the
wide low-threshold tail help only NEAR that boundary?

Design (advisor-locked):
  axes      : core mean ∈ {18,17.5,17,16.5,16} × std ∈ {wide 1.5, narrow 0.5}.
              (18,wide) = healthy baseline; (18,*) rows are the SANITY ANCHOR that
              must reproduce the sweep's matched-null (~0 ignition) — a built-in
              regression against setup drift.
  geometry  : fixed mid core + end kick (sweep showed ignition is position-
              independent; spend the budget on the mean axis + seeds instead).
  seeds     : CONNECTION seed freed (each network independently rebuilt) — the fix
              for the sweep's "1 fixed skeleton" caveat. 6 connection × 2 field/noise
              = 12 independent realizations per cell. This separates connection vs
              (field+noise) — NOT the three-way 连接/阈值场/噪声 (field+noise tied);
              small-N independent networks = SCREEN-PLUS, not a formal verdict.
  report    : every trial records ignition/latency/returned/tail + core_paf/axis.
              SPLIT (sweep discipline): ignition prob + latency from ALL trials;
              evoked sync + axis ONLY from non-igniting trials, paired vs the same-
              network/seed healthy baseline.

Pre-registered reads (acceptance gate, encode the conclusion not existence):
  - graded/locatable rise inside (16,18) with mean=18 ~0  -> REAL boundary (16 just past it)
  - already saturated at 17.5                              -> 18 at the edge; need finer steps
  - mean=18 ignites >0                                     -> bug/drift; must match sweep 0/84
  - tail effect measured as HORIZONTAL boundary-shift (wide curve at higher mean
    than narrow), NOT a vertical prob-gap (the latter needs ~100/cell).

Reuses the validated grid `_build`/`_run_one`/`_montage`/`_provenance`/`_engine_guard`.
Modes: --quick (L=1, 2 conn × 1 fn, smoke) ; default (L=3, --conn/--fn).
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
from src.sef_hfo_heterogeneity import sample_core_field   # noqa: E402

OUT = G.OUT
PATCH_R = G.PATCH_R
MEANS = [18.0, 17.5, 17.0, 16.5, 16.0]
STDS = {"wide": 1.5, "narrow": 0.5}
BASE_MEAN, BASE_STD = 18.0, 1.5


def _geom(L):
    ctr = np.array([L / 2, L / 2])
    u = np.array([np.cos(np.deg2rad(G.THETA)), np.sin(np.deg2rad(G.THETA))])
    return tuple(ctr), tuple(ctr + 0.6 * (L / 2) * u)        # mid patch, end kick


def _field_seed(fn, m, sname):
    return int(fn * 100003 + int(round(m * 10)) * 7 + (1 if sname == "narrow" else 0))


def _ci(v):
    v = np.asarray(v, float); n = len(v)
    sd = float(v.std(ddof=1)) if n > 1 else 0.0
    return dict(mean=float(v.mean()) if n else float("nan"), sd=sd,
                sem=float(sd / np.sqrt(n)) if n > 1 else 0.0, n=int(n), vals=v.tolist())


def _cross_mean(means_desc, rates):
    """Mean level where ignition rate crosses 0.5 as mean DECREASES (linear interp
    between bracketing points). means_desc descending; rates aligned. Returns nan if
    never crosses (all <0.5 or all >=0.5)."""
    m = np.asarray(means_desc, float); r = np.asarray(rates, float)
    for i in range(len(m) - 1):
        if (r[i] - 0.5) * (r[i + 1] - 0.5) <= 0 and r[i] != r[i + 1]:
            f = (0.5 - r[i]) / (r[i + 1] - r[i])
            return float(m[i] + f * (m[i + 1] - m[i]))
    return float("nan")


def _scan(L, density, conn_seeds, fn_seeds, means=None, out_tag=""):
    means = list(MEANS if means is None else means)
    G._engine_guard()
    patch, kick = _geom(L)
    montage = G._montage(L)
    rows = []; t0 = time.time(); k = 0
    tot = len(conn_seeds) * len(fn_seeds) * (len(means) * len(STDS) - 1)
    p_ref = None
    for cs in conn_seeds:
        p, net, nu_theta, NE = G._build(L, density, cs)      # INDEPENDENT network per conn seed
        p_ref = p
        is_E = net["labels"] == 0
        for fn in fn_seeds:
            bf = sample_core_field(net["pos"], is_E, patch, PATCH_R,
                                   np.random.default_rng(_field_seed(fn, BASE_MEAN, "wide")),
                                   core_mean=BASE_MEAN, core_std=BASE_STD)
            cm = bf["core_mask"]
            b = G._run_one(p, net, nu_theta, NE, kick, bf["vth"], montage, cm, fn)
            cell_ig = {}                                       # (m,sname) -> ignited, for the progress line
            for m in means:
                for sname, sval in STDS.items():
                    is_base = (m == BASE_MEAN and sname == "wide")
                    if is_base:
                        c = b
                    else:
                        cf = sample_core_field(net["pos"], is_E, patch, PATCH_R,
                                               np.random.default_rng(_field_seed(fn, m, sname)),
                                               core_mean=m, core_std=sval)
                        c = G._run_one(p, net, nu_theta, NE, kick, cf["vth"], montage, cm, fn)
                        k += 1
                    d_axis = (None if (c["axis_deg"] is None or b["axis_deg"] is None)
                              else G._undirected_diff(c["axis_deg"], b["axis_deg"]))
                    cell_ig[(m, sname)] = bool(c["prekick_ignited"])
                    rows.append(dict(
                        conn_seed=cs, fn_seed=fn, mean=m, std=sname, is_baseline=is_base,
                        ignited=bool(c["prekick_ignited"]), latency=c["ignition_latency"],
                        returned=bool(c["returned"]), tail_complete=bool(c["tail_complete"]),
                        d_core_paf=float(c["core_paf"] - b["core_paf"]),
                        d_axis_deg=d_axis, base_ignited=bool(b["prekick_ignited"])))
            print(f"[conn{cs} fn{fn}] " + "  ".join(
                f"{m:.1f}:" + "".join("I" if cell_ig[(m, s)] else "." for s in STDS)
                for m in means) + f"  (w/n per mean; {time.time()-t0:.0f}s, {k}/{tot})", flush=True)

    # aggregate per (mean, std): ignition rate, latency (igniting), evoked sync (non-igniting)
    agg = {}
    for sname in STDS:
        agg[sname] = {}
        for m in means:
            sub = [r for r in rows if r["mean"] == m and r["std"] == sname]
            ig = np.array([r["ignited"] for r in sub], bool)
            lat = np.array([r["latency"] for r in sub if r["ignited"]], float)
            dpaf = np.array([r["d_core_paf"] for r in sub if not r["ignited"]], float)
            agg[sname][f"{m:.1f}"] = dict(
                ignition_rate=float(ig.mean()), n=int(len(sub)),
                latency_igniting=(_ci(lat) if lat.size else None),
                d_core_paf_evoked=(_ci(dpaf) if dpaf.size else None))
    means_desc = sorted(means, reverse=True)
    cross = {s: _cross_mean(means_desc, [agg[s][f"{m:.1f}"]["ignition_rate"] for m in means_desc])
             for s in STDS}
    boundary_shift = (cross["wide"] - cross["narrow"]
                      if np.isfinite(cross["wide"]) and np.isfinite(cross["narrow"]) else None)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"mean_scan{out_tag}_metrics.json").write_text(json.dumps(dict(
        provenance=G._provenance(p_ref, conn_seeds[0]),
        means=means, stds=STDS, base=(BASE_MEAN, BASE_STD),
        conn_seeds=list(conn_seeds), fn_seeds=list(fn_seeds), patch=list(patch), kick=list(kick),
        aggregate=agg, ignition_cross_mean=cross, boundary_shift_wide_minus_narrow=boundary_shift,
        raw=rows), indent=2))
    san = [r for r in rows if r["mean"] == 18.0]
    san_ig = float(np.mean([r["ignited"] for r in san]))
    print(f"\nSANITY mean=18 ignition rate = {san_ig:.3f} (must be ~0 to match sweep matched-null)")
    print(f"ignition cross-mean: wide={cross['wide']}, narrow={cross['narrow']}, "
          f"shift(wide-narrow)={boundary_shift}")
    print(f"wrote mean_scan{out_tag}_metrics.json ({len(rows)} rows, {time.time()-t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--conn", type=int, default=6)
    ap.add_argument("--conn-start", type=int, default=1,
                    help="first connection seed (use FRESH seeds for a 2nd pass — reusing "
                         "seeds reproduces cells bit-identically at shared means)")
    ap.add_argument("--fn", type=int, default=2)
    ap.add_argument("--means", type=str, default=None,
                    help="comma-separated core means (default 18,17.5,17,16.5,16)")
    ap.add_argument("--out-tag", type=str, default="",
                    help="suffix for the output file (e.g. _fine) — keep separate from the coarse run")
    a = ap.parse_args()
    means = ([float(x) for x in a.means.split(",")] if a.means else None)
    if a.quick:
        _scan(1.0, 4000.0, conn_seeds=[1, 2], fn_seeds=[1], means=means, out_tag=a.out_tag)
    else:
        cs = list(range(a.conn_start, a.conn_start + a.conn))
        _scan(3.0, 1800.0, conn_seeds=cs, fn_seeds=list(range(1, a.fn + 1)),
              means=means, out_tag=a.out_tag)


if __name__ == "__main__":
    main()
