#!/usr/bin/env python3
"""Axis-A Stage A1-0a — ignition feasibility scan (NOT a science comparison).

Plan: docs/superpowers/plans/2026-06-14-sef-hfo-axisA-lesion-propagation-fingerprint.md
Stage A1 step 0a. The ONLY question here: which (core_mean, core_std) single-focus
operating points IGNITE (produce self-terminated directional clean events) — so the
formal A1 (mean-/rate-matched) knows where the narrow heterogeneity band can be lit.
This step REPORTS feasibility only: ignite yes/no, clean-event count/rate, excluded
counts. It MUST NOT be written up as "heterogeneity causes a fingerprint difference".

Runs the existing read-out runner (no new engine code) over the grid, then aggregates.
Outputs under .../snn_cm_spontaneous/a1_0a_feasibility/:
  feasibility_table.csv / feasibility_table.json   (one row per (mean,std,seed))
  figures/README.md

Usage:
  python scripts/run_sef_hfo_axisA_a1_0a_feasibility.py --workers 3 [--run] [--aggregate-only]
By default it both runs the grid and aggregates. --aggregate-only skips simulation.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"
FEAS = BASE / "a1_0a_feasibility"
RUNNER = ROOT / "scripts/run_sef_hfo_snn_cm_spontaneous_readout.py"

LESION = "oneend_neg"          # single-focus heterogeneity scan
CORE_MEANS = [17.0, 16.5, 16.0]   # 17.0 = baseline op; lower to find where narrow ignites
CORE_STDS = [0.5, 1.0, 1.5]       # narrow / mid / wide
SEEDS = [1]                     # pass 1 = seed 1 only (feasibility); add seed 2 at the boundary
T = 2000.0                      # feasibility window (baseline used 3500; this only needs ignition)
THREADS_PER_RUN = 8             # per-run thread cap so N parallel runs don't oversubscribe 80 cores


def _tag(mean: float, std: float, seed: int) -> str:
    return f"a10a_neg_m{mean}_std{std}_s{seed}"   # matches the pilot tag (resume reuses it)


def _cells():
    for mean in CORE_MEANS:
        for std in CORE_STDS:
            for seed in SEEDS:
                yield mean, std, seed


def _run_cell(mean: float, std: float, seed: int) -> dict:
    tag = _tag(mean, std, seed)
    if (FEAS / f"readout_{tag}.json").exists():   # resume: reuse the pilot / prior cells
        return {"tag": tag, "rc": 0, "skipped": True}
    log = FEAS / "logs" / f"{tag}.log"
    cmd = [sys.executable, str(RUNNER), "--lesion", LESION,
           "--core-mean", str(mean), "--core-std", str(std), "--seed", str(seed),
           "--T", str(T), "--tag", tag, "--out", str(FEAS)]
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = str(THREADS_PER_RUN)
    with open(log, "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    return {"tag": tag, "rc": rc, "skipped": False}


def run_grid(workers: int) -> None:
    (FEAS / "logs").mkdir(parents=True, exist_ok=True)
    cells = list(_cells())
    print(f"A1-0a feasibility: {len(cells)} cells, workers={workers}, T={T}, lesion={LESION}")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_run_cell, *c): c for c in cells}
        for f in as_completed(futs):
            r = f.result()
            print(f"  done {r['tag']} rc={r['rc']}" + (" (skipped/resumed)" if r.get("skipped") else ""))


def aggregate() -> None:
    rows = []
    for mean, std, seed in _cells():
        tag = _tag(mean, std, seed)
        jp = FEAS / f"readout_{tag}.json"
        if not jp.exists():
            rows.append(dict(core_mean=mean, core_std=std, seed=seed, tag=tag,
                             status="MISSING", n_events=None, n_clean=None,
                             clean_rate=None, ignited=None))
            continue
        d = json.loads(jp.read_text())
        nev = int(d.get("n_events", 0) or 0)
        nfw = int(d.get("n_clean_forward", 0) or 0)
        nrv = int(d.get("n_clean_reverse", 0) or 0)
        ntr = int(d.get("n_truncated_directional", 0) or 0)
        nclean = nfw + nrv
        rows.append(dict(
            core_mean=mean, core_std=std, seed=seed, tag=tag, status="ok",
            n_events=nev, n_clean_forward=nfw, n_clean_reverse=nrv,
            n_clean=nclean, n_truncated_directional=ntr,
            clean_rate=(round(nclean / nev, 4) if nev else 0.0),
            ignited=bool(nev > 0), ignited_clean=bool(nclean > 0)))

    # per-(mean,std) ignition reliability across seeds
    band = {}
    for r in rows:
        if r["status"] != "ok":
            continue
        k = (r["core_mean"], r["core_std"])
        band.setdefault(k, {"n_seed": 0, "n_ignited_clean": 0, "clean_counts": []})
        band[k]["n_seed"] += 1
        band[k]["n_ignited_clean"] += int(r["ignited_clean"])
        band[k]["clean_counts"].append(r["n_clean"])
    band_summary = [
        dict(core_mean=m, core_std=s, n_seed=v["n_seed"],
             ignited_clean_fraction=round(v["n_ignited_clean"] / v["n_seed"], 3),
             clean_counts=v["clean_counts"])
        for (m, s), v in sorted(band.items())]

    out = dict(
        stage="A1-0a ignition feasibility (REPORTS feasibility only; NOT a science comparison)",
        lesion=LESION, T=T, core_means=CORE_MEANS, core_stds=CORE_STDS, seeds=SEEDS,
        rows=rows, band_summary=band_summary)
    (FEAS / "feasibility_table.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # CSV
    cols = ["core_mean", "core_std", "seed", "tag", "status", "n_events",
            "n_clean", "clean_rate", "ignited_clean"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    (FEAS / "feasibility_table.csv").write_text("\n".join(lines) + "\n")
    print(f"aggregated -> {FEAS/'feasibility_table.csv'} + .json")
    for b in band_summary:
        print(f"  m={b['core_mean']} std={b['core_std']}: "
              f"ignited_clean {b['ignited_clean_fraction']} counts={b['clean_counts']}")


def write_readme() -> None:
    rd = FEAS / "figures"
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "README.md").write_text(
        "# A1-0a 点火可行性扫描（ignition feasibility）\n\n"
        "本目录是 Axis-A Stage A1 的 **0a 子步**：只回答一个问题——单灶在哪些 "
        "`(core_mean, core_std)` 工作点上**点得着**（产生自终止的方向性 clean 事件）。"
        "**这一步只报可点火性，绝不下「异质性导致 fingerprint 差异」的科学结论**——那是正式 "
        "A1（mean-/rate-matched control）的事。窄档（std=0.5）在 baseline mean=17.0 已知点不着，"
        "所以向下扫 mean 找窄档能点着的边界，给正式 A1 定可比工作点。\n\n"
        "### feasibility_table.csv / feasibility_table.json\n"
        "逐 `(core_mean, core_std, seed)` 一行：`n_events`（自发事件数）、`n_clean`（自终止方向性 "
        "clean 事件数 = forward+reverse）、`clean_rate`、`ignited_clean`（是否点着）。"
        "JSON 另含 `band_summary`：每个 `(mean,std)` 跨 seed 的 `ignited_clean_fraction` + clean 计数。\n"
        "**关注点**：窄档 std=0.5 在 mean=17.0→16.0 哪一档开始 `ignited_clean_fraction` 抬起来——"
        "那条边界决定正式 A1 用哪个 mean 才能让三档（窄/中/宽）在**可比工作点**下比较；"
        "宽档(1.5)应在 17.0 就点着（与 baseline 一致），是 sanity。\n")
    print(f"wrote {rd/'README.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    if not args.aggregate_only:
        run_grid(args.workers)
    aggregate()
    write_readme()


if __name__ == "__main__":
    main()
