#!/usr/bin/env python3
"""Selection-aware contact nulls for the rev5 target-informed bridge."""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_fig5_target_informed_bridge import (  # noqa: E402
    jsonable, lse, score_energy_burden, score_energy_field)


def _permutation(rng, shafts, mode):
    base = np.arange(len(shafts))
    if mode == "all_contact":
        return rng.permutation(base)
    if mode == "within_shaft":
        out = base.copy()
        for shaft in sorted(set(shafts.tolist())):
            idx = np.flatnonzero(shafts == shaft)
            out[idx] = rng.permutation(idx)
        return out
    if mode == "within_shaft_circular":
        out = base.copy()
        for shaft in sorted(set(shafts.tolist())):
            idx = np.flatnonzero(shafts == shaft)
            out[idx] = np.roll(idx, int(rng.integers(0, len(idx))))
        return out
    raise ValueError(mode)


def _permute_target(target, permutation):
    out = deepcopy(target)
    for endpoint in ("pre", "early", "increment"):
        for key in ("median", "q025", "q975", "bootstrap_iqr"):
            out[endpoint][key] = np.asarray(out[endpoint][key], float)[permutation]
    return out


def _candidate_score(row, target, shafts):
    model_pre = np.asarray(row["model_pre_robust_z"], float)
    model_early = np.asarray(row["model_early_robust_z"], float)
    field = score_energy_field(model_pre, model_early, target, shafts)
    energy = score_energy_burden(model_early, target)
    return float(np.mean([energy["D_energy"], field["J_field"]])
                 + lse([energy["D_energy"], field["J_field"]]))


def selection_null(records, target, shafts, *, mode, draws, seed):
    rng = np.random.default_rng(int(seed))
    observed = min(_candidate_score(row, target, shafts) for row in records)
    null = np.empty(int(draws), float)
    for draw in range(int(draws)):
        permutation = _permutation(rng, shafts, mode)
        surrogate = _permute_target(target, permutation)
        null[draw] = min(_candidate_score(row, surrogate, shafts) for row in records)
    return {
        "mode": mode,
        "draws": int(draws),
        "seed": int(seed),
        "observed_minimum_J_early": float(observed),
        "null_q05": float(np.quantile(null, 0.05)),
        "null_median": float(np.median(null)),
        "null_q95": float(np.quantile(null, 0.95)),
        "lower_tail_p": float((1 + np.sum(null <= observed + 1e-15)) / (len(null) + 1)),
        "null_minimum_J_early": null,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_data_driven_zm_target_informed_bridge_v1.json")
    parser.add_argument("--draws", type=int, default=4096)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    out = ROOT / config["output_root"]
    rescore = json.loads((out / "existing_candidate_rescore.json").read_text())
    target_payload = json.loads((out / "clinical_target.json").read_text())
    with np.load(out / "clinical_target_vectors.npz", allow_pickle=False) as target_npz:
        shafts = target_npz["shaft_ids"].astype(str)
    records = [row for row in rescore["records"]
               if row.get("status") == "BRIDGE_EVALUABLE"
               and row.get("primary_zm_only")]
    if not records:
        raise RuntimeError("no bridge-evaluable primary candidate for selection null")
    target = target_payload["summaries"]["sensitivity_10_150"]
    modes = ("all_contact", "within_shaft", "within_shaft_circular")
    results = [selection_null(
        records, target, shafts, mode=mode, draws=args.draws,
        seed=20260821 + index * 10000)
        for index, mode in enumerate(modes)]
    payload = {
        "status": "SELECTION_AWARE_NULL_COMPLETE",
        "candidate_ids": [row["candidate_id"] for row in records],
        "n_candidates": len(records),
        "matched_band_hz": [10.0, 150.0],
        "nulls": results,
        "boundary": ("development target and Stage-1 candidate set; the null repeats "
                     "the minimum-over-candidates selection operation"),
    }
    (out / "selection_aware_null.json").write_text(
        json.dumps(jsonable(payload), indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "n_candidates": len(records)}))


if __name__ == "__main__":
    main()
