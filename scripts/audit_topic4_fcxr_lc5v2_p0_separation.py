#!/usr/bin/env python3
"""Outcome-blind baseline/early-ictal separation audit for the LC5v2 p0 field."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.topic4_fcxr_lc5 import AtomicStageBundle, json_sanitize, load_sparse_spike_stream  # noqa: E402
from src.topic4_fcxr_lc5_finite_episode import coarsen_sparse_stream  # noqa: E402
from src.topic4_mz_fcxr_pump import pump_activation  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/p0_separation_audit"
SOURCE = ROOT / "results/topic4_sef_hfo/fcxr_lc5_episode_pump/u1_capture/u1_sparse_spikes.npz"
EXACT = ROOT / "results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/exact_load_audit"
A_LOAD = 0.014675
TAU_MS = 8000.0
SAMPLE_MS = 5


def _write_json(path, value):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(json_sanitize(value), indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _npz_atomic(path, **arrays):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def select_policy(rows):
    by_name = {row["name"]: row for row in rows}
    q99 = by_name["q099"]
    if (q99["baseline_active_sample_fraction"] <= 0.01
            and q99["early_median_active_cells_per_sample"] >= 0.75):
        return "q099"
    return None


def run_audit():
    if OUT.is_dir():
        return json.loads((OUT / "summary.json").read_text())
    full = load_sparse_spike_stream(SOURCE)
    stream = coarsen_sparse_stream(full, source_dt_ms=0.05, target_dt_ms=1.0, stop_ms=14000)
    del full
    u = np.zeros(stream.n_cells)
    spike = np.zeros(stream.n_cells)
    previous = np.empty(0, np.int64)
    pos = 0
    baseline, early = [], []
    for step in range(stream.n_steps):
        if previous.size:
            spike[previous] = 0.0
        phi = pump_activation(u, 3)
        if 7000 <= step < 11000 and (step - 7000) % SAMPLE_MS == 0:
            baseline.append(phi.astype(np.float32))
        if 12000 <= step < 14000 and (step - 12000) % SAMPLE_MS == 0:
            early.append(phi.astype(np.float32))
        end = int(np.searchsorted(stream.steps, step, side="right"))
        cells = stream.cells[pos:end]
        if cells.size:
            spike[cells] = 1.0
        previous, pos = cells, end
        np.maximum(u + A_LOAD * spike - phi / TAU_MS, 0.0, out=u)
    baseline = np.stack(baseline)
    early = np.stack(early)
    with np.load(EXACT / "fields.npz", allow_pickle=False) as z:
        old = np.asarray(z["p0_exact"], float)
    fields = {"old_mean": old.astype(np.float32)}
    policies = [("old_mean", old)] + [
        (f"q{int(q * 100):03d}", np.quantile(baseline, q, axis=0))
        for q in (0.90, 0.95, 0.99, 1.0)
    ]
    rows = []
    for name, p0 in policies:
        fields[name] = np.asarray(p0, np.float32)
        b_excess = np.maximum(baseline - p0, 0.0)
        e_excess = np.maximum(early - p0, 0.0)
        b_active, e_active = b_excess > 0.0, e_excess > 0.0
        integral = e_excess.sum(axis=0) * SAMPLE_MS
        rows.append({
            "name": name,
            "baseline_active_sample_fraction": float(b_active.mean()),
            "baseline_any_cell_fraction": float(b_active.any(axis=0).mean()),
            "baseline_population_excess_mean": float(b_excess.mean()),
            "early_active_sample_fraction": float(e_active.mean()),
            "early_any_cell_fraction": float(e_active.any(axis=0).mean()),
            "early_median_active_cells_per_sample": float(np.median(e_active.mean(axis=1))),
            "early_population_excess_mean": float(e_excess.mean()),
            "early_percell_excess_integral_median_ms": float(np.median(integral)),
        })
    selected = select_policy(rows)
    if selected is None:
        raise RuntimeError("P0_SEPARATION_NOT_FOUND")
    exact = json.loads((EXACT / "summary.json").read_text())
    denom = next(row for row in rows if row["name"] == selected)[
        "early_percell_excess_integral_median_ms"
    ]
    force = float(exact["dose_exact"]["recurrent_force_integral_median_ms"])
    gammas = (0.001, 0.003, 0.005)
    imax = {str(gamma): float(gamma * force / denom) for gamma in gammas}
    summary = {
        "status": "P0_SEPARATION_PASS", "selected_policy": selected,
        "selection_rule": {"baseline_active_sample_fraction_max": 0.01,
                           "early_median_active_cells_per_sample_min": 0.75},
        "baseline_window_ms": [7000, 11000], "early_window_ms": [12000, 14000],
        "sample_ms": SAMPLE_MS, "a_load": A_LOAD, "tau_ms": TAU_MS, "h": 3,
        "rows": rows, "recurrent_force_integral_median_ms": force,
        "selected_excess_integral_median_ms": denom, "Imax_by_gamma": imax,
        "source_spike_sha256": stream.sha256,
    }
    with AtomicStageBundle(OUT) as bundle:
        _write_json(bundle.path("summary.json"), summary)
        _npz_atomic(bundle.path("p0_fields.npz"), **fields)
        bundle.commit(required=["summary.json", "p0_fields.npz"])
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm-audit", action="store_true")
    args = parser.parse_args()
    if not args.confirm_audit:
        raise SystemExit("pass --confirm-audit")
    print(json.dumps(json_sanitize(run_audit()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
