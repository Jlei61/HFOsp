"""Summarise Leg A's map, fail closed on anything that does not add up.

Every reported layer keeps its own denominator. A cell where only one of four
networks produced a readable direction is not the same evidence as one where all
four did, and averaging the finite values without saying so would present the
two identically.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_report import PRIMARY_KEY  # noqa: E402
from src.topic4_core_field_runner import (atomic_write_json,  # noqa: E402
                                          canonical_checksum, provenance)
from src.topic4_core_field_scoring import load_patient_templates  # noqa: E402
from src.topic4_core_field_stage3 import high_scoring_region  # noqa: E402

OUT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
DROP = ("checksum", "provenance")


def _audit(cells, cfg, sigma, seeds):
    """Fail closed: the set on disk must be exactly the frozen design."""
    n_cells = len(cfg["grid"]["centers"])
    problems = []
    seen = {}
    for path, rec in cells:
        if rec.get("config_checksum") != cfg["checksum"]:
            problems.append(f"{os.path.basename(path)}: design hash mismatch")
        if "provenance" not in rec:
            problems.append(f"{os.path.basename(path)}: no provenance")
        key = (round(float(rec["cx"]), 6), round(float(rec["cy"]), 6), int(rec["seed"]))
        if key in seen:
            problems.append(f"duplicate cell/seed {key}")
        seen[key] = path
    got_seeds = sorted({int(r["seed"]) for _, r in cells})
    if got_seeds != sorted(seeds):
        problems.append(f"seed set {got_seeds} != frozen {sorted(seeds)}")
    if len(cells) != n_cells * len(seeds):
        problems.append(f"{len(cells)} artefacts != {n_cells} cells x {len(seeds)} seeds")
    errored = [os.path.basename(p) for p, r in cells if "error" in r]
    if errored:
        problems.append(f"{len(errored)} simulations errored: {errored[:3]}")
    return problems


def _aggregate(cells, cfg, tgt, rule, n_cells, seeds):
    from scripts.run_topic4_core_field_stage3_sweep import _score
    per = {i: [] for i in range(n_cells)}
    index = {(round(c[0], 6), round(c[1], 6)): i
             for i, c in enumerate(cfg["grid"]["centers"])}
    for _, rec in cells:
        i = index[(round(float(rec["cx"]), 6), round(float(rec["cy"]), 6))]
        per[i].append(_score(rec["events"], cfg, tgt, rule))

    n = int(cfg["grid"]["n"])
    shape = (n, n)
    S = np.full(shape, np.nan)
    n_valid = np.zeros(shape, int)      # denominator for the MATCH layer only
    n_runs = np.zeros(shape, int)       # denominator for every other layer
    bidir_frac = np.full(shape, np.nan)
    recruited = np.full(shape, np.nan)
    events = np.full(shape, np.nan)
    for i, rows in per.items():
        r, c = divmod(i, n)
        n_runs[r, c] = len(rows)
        vals = [x["S_rank"] for x in rows if x["S_rank"] is not None]
        n_valid[r, c] = len(vals)
        if vals:
            S[r, c] = float(np.mean(vals))
        if rows:
            bidir_frac[r, c] = float(np.mean([x["n_dir"] == 2 for x in rows]))
            recruited[r, c] = float(np.mean([x["recruited_min"] for x in rows]))
            events[r, c] = float(np.mean([x["n_events"] for x in rows]))
    return dict(S_rank=S, n_valid=n_valid, n_runs=n_runs,
                bidirectional_fraction=bidir_frac,
                recruited_min=recruited, n_events=events, per_cell=per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--allow-partial", action="store_true",
                    help="summarise an incomplete sweep, recording what is missing")
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.out, "config", "sweep_config.json")))
    if canonical_checksum(cfg, drop=DROP) != cfg["checksum"]:
        raise SystemExit("sweep config checksum mismatch")
    tgt = load_patient_templates(cfg["subject"], PRIMARY_KEY[0])
    rule = PRIMARY_KEY[1]
    n_cells = len(cfg["grid"]["centers"])
    seeds = list(cfg["sweep_seeds"])

    maps, audits = {}, {}
    for label, sigma in (("primary", cfg["sigmas"]["primary"]),
                         ("sensitivity", cfg["sigmas"]["sensitivity"])):
        paths = sorted(glob.glob(os.path.join(a.out, "cells", f"sigma{sigma:g}", "*.json")))
        cells = [(p, json.load(open(p))) for p in paths]
        if not cells:
            audits[label] = ["no artefacts on disk"]
            continue
        problems = _audit(cells, cfg, sigma, seeds)
        audits[label] = problems
        if problems and not a.allow_partial:
            raise SystemExit(f"[{label}] fail closed:\n  " + "\n  ".join(problems))
        good = [(p, r) for p, r in cells if "error" not in r]
        maps[label] = _aggregate(good, cfg, tgt, rule, n_cells, seeds)

    if "primary" not in maps:
        raise SystemExit("no primary map to summarise")

    # the region, not the argmax: see the winner's-curse discipline in the plan
    m = maps["primary"]
    region = high_scoring_region(m["S_rank"], m["n_valid"],
                                 top_frac=cfg["region"]["top_frac"],
                                 min_valid=cfg["region"]["min_valid"])
    centers = np.asarray(cfg["grid"]["centers"], float).reshape(
        int(cfg["grid"]["n"]), int(cfg["grid"]["n"]), 2)

    def _layer(x):
        return [[None if not np.isfinite(v) else float(v) for v in row] for row in x]

    summary = dict(
        stage="stage3_legA_sweep_summary",
        config_checksum=cfg["checksum"], provenance=provenance(),
        artefact_audit=audits,
        complete=all(not v for v in audits.values()),
        maps={k: dict(S_rank=_layer(v["S_rank"]),
                      n_valid=v["n_valid"].tolist(),
                      n_runs=v["n_runs"].tolist(),
                      bidirectional_fraction=_layer(v["bidirectional_fraction"]),
                      recruited_min=_layer(v["recruited_min"]),
                      n_events=_layer(v["n_events"]),
                      undefined_fraction=float(np.mean(~np.isfinite(v["S_rank"]))),
                      n_valid_histogram={str(k): int(c) for k, c in
                                         zip(*np.unique(v["n_valid"], return_counts=True))})
              for k, v in maps.items()},
        high_scoring_region=dict(
            mask=region.tolist(), n_cells=int(region.sum()),
            centers=[[float(x) for x in centers[r, c]]
                     for r, c in zip(*np.where(region))],
            rule=(f"top {cfg['region']['top_frac']:.0%} of cells with at least "
                  f"{cfg['region']['min_valid']} valid seeds, as a region -- "
                  f"never the single best cell"),
            confirm_seeds=list(cfg["confirm_seeds"]),
            confirmed=None),
        reading_notes=[
            "S_rank is undefined wherever a cell produced only one direction; "
            "those cells are counted in undefined_fraction, not silently dropped",
            "n_valid is the denominator for the match layer ONLY; direction and "
            "recruitment are defined for every completed run, so a cell where no "
            "network was bidirectional is a measured zero, not an unmeasured cell",
            "the two probe sizes are separate maps and are never combined per cell",
        ])
    atomic_write_json(summary, os.path.join(a.out, "sweep_summary.json"))

    for label, v in maps.items():
        S, nv = v["S_rank"], v["n_valid"]
        print(f"[{label}] undefined {np.mean(~np.isfinite(S)):.0%} of cells | "
              f"n_valid hist {dict(zip(*np.unique(nv, return_counts=True)))}")
        if np.isfinite(S).any():
            print(f"          S_rank range {np.nanmin(S):+.3f} .. {np.nanmax(S):+.3f}")
    print(f"[region] {int(region.sum())} cells: "
          f"{[[round(x,1) for x in centers[r, c]] for r, c in zip(*np.where(region))]}")
    for label, probs in audits.items():
        if probs:
            print(f"[audit:{label}] {len(probs)} problem(s): {probs[:3]}")


if __name__ == "__main__":
    main()
