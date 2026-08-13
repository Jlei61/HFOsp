#!/usr/bin/env python3
"""Full-window 1-ms versus 0.05-ms calibration audit on a locked spatial/rate subset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_topic4_fcxr_lc5v2_u2 as U2  # noqa: E402
from src.topic4_fcxr_lc5 import SparseSpikeStream, load_sparse_spike_stream  # noqa: E402
from src.topic4_fcxr_lc5_finite_episode import coarsen_sparse_stream  # noqa: E402


OUT = U2.OUT / "lc5v2p1_calibration_parity"
STOP_MS = 14000.0
BASELINE_MS = (7000.0, 11000.0)
EARLY_MS = (12000.0, 14000.0)
SAMPLE_MS = 5.0
N_PER_CATEGORY = 16


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, value):
    U2._write_json(path, value)


def _subset_stream(stream, selected, stop_ms, source_dt_ms=0.05):
    selected = np.asarray(selected, np.int64)
    mapping = np.full(stream.n_cells, -1, np.int64)
    mapping[selected] = np.arange(selected.size)
    stop_step = int(round(float(stop_ms) / float(source_dt_ms)))
    right = int(np.searchsorted(stream.steps, stop_step, side="left"))
    old_cells = stream.cells[:right]
    keep = mapping[old_cells] >= 0
    steps = np.asarray(stream.steps[:right][keep], np.int64)
    cells = np.asarray(mapping[old_cells[keep]], np.int64)
    order = np.lexsort((cells, steps))
    return SparseSpikeStream(
        steps[order], cells[order], stop_step, selected.size
    )


def _sampled_replay(stream, *, dt_ms, tau_ms, a_load):
    n = stream.n_cells
    bounds = np.searchsorted(stream.steps, np.arange(stream.n_steps + 1), side="left")
    u = np.zeros(n, float)
    baseline, early = [], []
    sample_every = int(round(SAMPLE_MS / float(dt_ms)))
    b0, b1 = (int(round(x / float(dt_ms))) for x in BASELINE_MS)
    e0, e1 = (int(round(x / float(dt_ms))) for x in EARLY_MS)
    for step in range(stream.n_steps):
        uh = u ** 3
        phi = uh / (1.0 + uh)
        if b0 <= step < b1 and (step - b0) % sample_every == 0:
            baseline.append(phi.copy())
        if e0 <= step < e1:
            early.append(phi.copy())
        u -= (float(dt_ms) / float(tau_ms)) * phi
        cells = stream.cells[bounds[step]:bounds[step + 1]]
        if cells.size:
            u[cells] += float(a_load)
        np.maximum(u, 0.0, out=u)
    baseline = np.asarray(baseline, float)
    early = np.asarray(early, float)
    p0 = np.quantile(baseline, 0.99, axis=0)
    excess_integral = np.sum(np.maximum(early - p0[None, :], 0.0), axis=0) * float(dt_ms)
    return {
        "p0": p0,
        "early_phi_median": float(np.median(early)),
        "excess_integral_ms": excess_integral,
        "n_baseline_samples": int(baseline.shape[0]),
        "n_early_steps": int(early.shape[0]),
    }


def choose_subset(S, baseline_rate, high_rate):
    pos = np.asarray(S["posE"], float)
    axis = np.asarray(S["axis_unit"], float)
    center = np.asarray(S["center"], float)
    rel = pos - center
    perpendicular = np.abs(rel[:, 0] * axis[1] - rel[:, 1] * axis[0])
    core = np.asarray(U2.OLD_SLOW.build_core_masks(S), bool)
    masks = {
        "core": core,
        "axial_noncore": (~core) & (perpendicular <= 1.5),
        "off_axis": (~core) & (perpendicular >= 4.0),
        "high_rate_tail": np.asarray(high_rate) >= np.quantile(high_rate, 0.99),
    }
    selected, categories = [], []
    score = np.asarray(baseline_rate, float) + np.asarray(high_rate, float)
    used = set()
    for name, mask in masks.items():
        candidates = np.flatnonzero(mask)
        candidates = candidates[np.argsort(score[candidates])[::-1]]
        chosen = [int(i) for i in candidates if int(i) not in used][:N_PER_CATEGORY]
        if len(chosen) < N_PER_CATEGORY:
            raise RuntimeError(f"insufficient cells for parity category {name}")
        selected.extend(chosen)
        categories.extend([name] * len(chosen))
        used.update(chosen)
    return np.asarray(selected, np.int64), np.asarray(categories)


def run_audit():
    summary_path = OUT / "summary.json"
    if summary_path.is_file():
        return json.loads(summary_path.read_text())
    source = load_sparse_spike_stream(U2.SOURCE / "u1_sparse_spikes.npz")
    with np.load(U2.SOURCE / "u1_rate_fields.npz", allow_pickle=False) as z:
        baseline_rate = np.asarray(z["baseline_rate_hz"], float)
        high_rate = np.asarray(z["high_rate_hz"], float)
    S = U2.PP.build_substrate(U2.CONNECTION_SEED)
    selected, categories = choose_subset(S, baseline_rate, high_rate)
    exact_stream = _subset_stream(source, selected, STOP_MS, U2.DT_MS)
    coarse_stream = coarsen_sparse_stream(
        exact_stream, source_dt_ms=U2.DT_MS, target_dt_ms=1.0, stop_ms=STOP_MS
    )
    calibration = json.loads((U2.CAL / "finite_episode_calibration.json").read_text())
    rows = []
    for tau_ms in (3000.0, 8000.0, 15000.0):
        a_load = float(calibration["tau"][f"tau{int(tau_ms)}"]["a_load"])
        exact = _sampled_replay(exact_stream, dt_ms=U2.DT_MS, tau_ms=tau_ms, a_load=a_load)
        coarse = _sampled_replay(coarse_stream, dt_ms=1.0, tau_ms=tau_ms, a_load=a_load)
        p0_abs = np.abs(exact["p0"] - coarse["p0"])
        exact_excess = float(np.median(exact["excess_integral_ms"]))
        coarse_excess = float(np.median(coarse["excess_integral_ms"]))
        rows.append({
            "tau_ms": tau_ms, "a_load": a_load,
            "p0_abs_diff_q50_q95_max": [
                float(np.quantile(p0_abs, 0.5)), float(np.quantile(p0_abs, 0.95)),
                float(np.max(p0_abs)),
            ],
            "early_phi_median_exact": exact["early_phi_median"],
            "early_phi_median_coarse": coarse["early_phi_median"],
            "early_phi_median_abs_diff": abs(exact["early_phi_median"] - coarse["early_phi_median"]),
            "excess_integral_median_ms_exact": exact_excess,
            "excess_integral_median_ms_coarse": coarse_excess,
            "excess_integral_median_relative_diff": abs(exact_excess - coarse_excess) / exact_excess,
            "exact_n_baseline_samples": exact["n_baseline_samples"],
            "coarse_n_baseline_samples": coarse["n_baseline_samples"],
        })
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / "representative_cells.npz", indices=selected, categories=categories,
        baseline_rate_hz=baseline_rate[selected], high_rate_hz=high_rate[selected],
    )
    payload = {
        "status": "COMPLETE", "scientific_role": "calibration instrument parity, no tuning",
        "source_spike_sha256": source.sha256, "source_file_sha256": _sha(U2.SOURCE / "u1_sparse_spikes.npz"),
        "subset_n": int(selected.size), "category_counts": {
            name: int(np.sum(categories == name)) for name in np.unique(categories)
        },
        "baseline_window_ms": list(BASELINE_MS), "early_window_ms": list(EARLY_MS),
        "rows": rows,
    }
    _write_json(summary_path, payload)
    return payload


if __name__ == "__main__":
    print(json.dumps(run_audit(), indent=2, sort_keys=True))
