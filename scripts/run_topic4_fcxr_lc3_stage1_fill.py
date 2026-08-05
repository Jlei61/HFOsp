#!/usr/bin/env python
"""Run stage-1 grid cells the primary pass will not reach for hours.

The primary pass sizes its pool with the registered `choose_map_workers`, whose
96 GiB floor is calibrated for the 110 GiB reconnaissance rows. A stage-1 cell
peaks near 15 GiB, so on a machine with 172 GiB free and 4 of 80 cores busy that
floor pins the run to two workers and the nine cells to fifteen-plus hours.

This takes the tail of the grid at an explicit width, reusing the primary's own
`_run_cell` so the science is identical -- same point, same seed, same frozen
relay, same registered values at the centre. Cells already on disk are skipped by
that function, so the two passes cannot duplicate work beyond one cell in flight,
and a duplicate would write the same deterministic result.

Nothing about the primary pass is touched. It keeps its own pool, its own
sizing, and its own claim on the cells it has already started.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "scripts"), os.path.join(ROOT, "src", "snn_engine")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse  # noqa: E402
import json  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402

import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc3_stage1 as S1  # noqa: E402

PER_CELL_GIB = S1.BASE_RSS_GIB + S1.GIB_PER_SIM_SECOND * (S1.RUN_MS / 1000.0)
HEADROOM_GIB = 40.0     # what stays free after every worker is at its peak


def _grid():
    return [(t, s) for t in S1.TAU_Z_GRID for s in S1.THETA_SCALE]


def _pending(cells):
    out = []
    for tau_z, scale in cells:
        path = os.path.join(S1.OUT, f"cell_{S1._cell_id(tau_z, scale)}.json")
        if os.path.isfile(path) and GEO._load_json(path).get("status") == "COMPLETE":
            continue
        out.append((tau_z, scale))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-run", action="store_true")
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--from-index", type=int, default=0,
                    help="0-based position in the grid to start from; the primary "
                         "pass works forward from 0, so the tail is collision-free")
    args = ap.parse_args()
    if not args.confirm_run:
        raise SystemExit("40k stage-1 fill requires --confirm-run")

    mem = GEO._meminfo()["mem_available_gib"]
    need = args.workers * PER_CELL_GIB + HEADROOM_GIB
    if mem < need:
        raise SystemExit(f"{args.workers} workers need {need:.0f} GiB at peak "
                         f"({PER_CELL_GIB:.1f} GiB each plus {HEADROOM_GIB:.0f} GiB "
                         f"headroom); {mem:.0f} GiB available")

    cells = _pending(_grid()[args.from_index:])
    print(f"[fill] {len(cells)} cells from index {args.from_index}, "
          f"{args.workers} workers, {PER_CELL_GIB:.1f} GiB each, "
          f"{mem:.0f} GiB available", flush=True)
    if not cells:
        return

    rows, pending = [], list(cells)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        while pending or futures:
            while pending and len(futures) < args.workers:
                futures[pool.submit(S1._run_cell, pending.pop(0))] = True
            done, _ = wait(list(futures), return_when=FIRST_COMPLETED)
            for fut in done:
                futures.pop(fut)
                r = fut.result()
                rows.append(r)
                print(f"[fill] {r['cell_id']}: {r['stage']} onset={r['onset_ms']} "
                      f"events_before={r['n_returning_before_onset']} "
                      f"class={r['entry_class']}", flush=True)
    print(json.dumps(dict(n_cells=len(rows),
                          cells=[r["cell_id"] for r in rows]), indent=2))


if __name__ == "__main__":
    main()
