"""Can a network wired under a spatial prior reproduce the patient's propagation?

This is a sufficiency question, not an attribution one: give the model the first
contact of a held-out event it never saw and let it generate the rest, then ask
how close the generated order is to the real one.

Two things inflate that number if left alone.

The given first contact is free.  On a short event it is a large share of the
correlation, so the agreement is recomputed with the seeded contacts dropped from
both sides.

A deterministic generator scored against one noisy event can beat two noisy
events scored against each other.  If every event is a common order plus
independent noise, the correlation between two events is the reliability rho and
the best any deterministic predictor can reach on a single event is sqrt(rho).
That is the ceiling the model is measured against, not rho itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_wiring_economy_rnn import (  # noqa: E402
    WEConfig,
    WEModel,
    build_event_tensors,
    rollout,
)

OUT_ROOT = ROOT / "results/topic5_wiring_economy_slp_rnn_v0_3_own_scale"
ARMS = ("SPATIAL_SET", "RANDOM_SET", "DENSE_TISSUE", "STATIC_CONTACT",
        "SPATIAL_SET_shuffled")
MAX_EVENTS = 400


def event_pair_reliability(ranks: np.ndarray, seed: int = 0, n_pairs: int = 2000) -> float:
    """Median order correlation between two held-out events of the same patient."""
    values = ranks.astype(float).copy()
    values[values < 0] = np.nan
    if len(values) < 20:
        return float("nan")
    rng = np.random.default_rng(seed)
    out = []
    for i, j in rng.choice(len(values), size=(n_pairs, 2)):
        if i == j:
            continue
        a, b = values[i], values[j]
        m = np.isfinite(a) & np.isfinite(b)
        if m.sum() < 3 or len(np.unique(a[m])) < 2 or len(np.unique(b[m])) < 2:
            continue
        r = spearmanr(a[m], b[m]).statistic
        if np.isfinite(r):
            out.append(r)
    return float(np.median(out)) if out else float("nan")


def agreement(observed: np.ndarray, generated: List[List[int]], drop: set[int]) -> float:
    """Spearman between the real and generated order, seeded contacts removed."""
    order = {c: i for i, step in enumerate(generated) for c in step}
    shared = [c for c in np.flatnonzero(observed >= 0)
              if c in order and c not in drop]
    if len(shared) < 3:
        return float("nan")
    x = [float(observed[c]) for c in shared]
    y = [float(order[c]) for c in shared]
    if len(np.unique(x)) < 2 or len(np.unique(y)) < 2:
        return float("nan")
    r = spearmanr(x, y).statistic
    return float(r) if np.isfinite(r) else float("nan")


def replay_fit(out_root: Path, fit_id: str, arm: str, seed: int) -> Dict[str, Any] | None:
    unit = out_root / "per_subject" / fit_id / f"{arm}_rnn" / f"seed{seed}"
    if not (unit / "weights.pt").exists():
        return None
    provenance = json.loads((out_root / "cache" / fit_id / "provenance.json").read_text())
    plane = np.load(out_root / "cache" / fit_id / "plane.npz")
    events = np.load(out_root / "cache" / fit_id / "events.npz")
    ranks, split, mode = events["ranks"], events["split"], events["mode"]
    keep = split >= 0
    tensors = build_event_tensors(ranks[keep])
    kept, part, kept_mode = ranks[keep], split[keep], mode[keep]

    base = arm.replace("_shuffled", "")
    model = WEModel(WEConfig(
        arm=base, cell="rnn", n_contacts=int(provenance["n_contacts"]),
        n_nodes=int(provenance["n_nodes"]), seed=seed,
        observation_operator=None if base == "STATIC_CONTACT" else plane["H"],
        node_distance_mm=None if base == "STATIC_CONTACT" else plane["D_mm"]))
    model.load_state_dict(torch.load(unit / "weights.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()

    test = np.flatnonzero(part == 2)[:MAX_EVENTS]
    if test.size < 10:
        return None
    steps = int(tensors["valid"].shape[1])
    starts = [np.flatnonzero(kept[i] == 0) for i in test]
    generated = rollout(model, starts, provenance["n_contacts"], steps, torch.device("cpu"))

    with_start, without_start, lengths = [], [], []
    for i, start, gen in zip(test, starts, generated):
        with_start.append(agreement(kept[i], gen, drop=set()))
        without_start.append(agreement(kept[i], gen, drop=set(int(c) for c in start)))
        flat = [c for s in gen for c in s]
        lengths.append(len(flat) / max(1, int((kept[i] >= 0).sum())))
    per_mode = {}
    for m in (0, 1):
        sel = [k for k, i in enumerate(test) if kept_mode[i] == m]
        if len(sel) >= 10:
            per_mode[str(m)] = float(np.nanmedian([without_start[k] for k in sel]))
    return {
        "fit_id": fit_id, "arm": arm, "seed": seed,
        "n_events": int(test.size),
        "with_start": float(np.nanmedian(with_start)),
        "without_start": float(np.nanmedian(without_start)),
        "length_ratio": float(np.median(lengths)),
        "by_mode": per_mode,
        "scope": provenance["scope"],
        "reliability": event_pair_reliability(kept[part == 2]),
    }


def per_patient(rows: List[Dict[str, Any]], key: str, f2s: Dict[str, str]) -> Dict[str, float]:
    by_fit: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is not None and np.isfinite(v):
            by_fit[r["fit_id"]].append(float(v))
    by_subject: Dict[str, List[float]] = defaultdict(list)
    for fit, vals in by_fit.items():
        by_subject[f2s[fit]].append(float(np.mean(vals)))
    return {s: float(np.mean(v)) for s, v in by_subject.items()}


def paired(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, Any]:
    keys = sorted(set(a) & set(b))
    x = np.array([a[k] for k in keys])
    y = np.array([b[k] for k in keys])
    return {"n": len(keys), "median_delta": float(np.median(x - y)),
            "n_higher": int((x > y).sum()),
            "p": float(wilcoxon(x, y).pvalue) if np.any(x != y) else float("nan")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    f2s = {r["fit_id"]: r["subject"] for r in manifest["fits"]}

    rows: List[Dict[str, Any]] = []
    for fit_id in sorted(f2s):
        for arm in ARMS:
            for seed in args.seeds:
                row = replay_fit(out_root, fit_id, arm, seed)
                if row:
                    rows.append(row)
        print(f"  replayed {fit_id}", flush=True)

    reliability = per_patient([r for r in rows if r["arm"] == "SPATIAL_SET"],
                              "reliability", f2s)
    ceiling = {k: float(np.sqrt(max(v, 0.0))) for k, v in reliability.items()}
    result: Dict[str, Any] = {
        "contract": "topic5_we_sufficiency_v0_3",
        "question": "given only the first contact of a held-out event, does the network "
                    "regenerate the rest of the patient's propagation order",
        "n_patients": len(reliability),
        "event_pair_reliability": reliability,
        "noise_ceiling_sqrt_reliability": ceiling,
        "arms": {},
    }
    for arm in ARMS:
        sub = [r for r in rows if r["arm"] == arm]
        result["arms"][arm] = {
            "with_start": per_patient(sub, "with_start", f2s),
            "without_start": per_patient(sub, "without_start", f2s),
            "length_ratio": per_patient(sub, "length_ratio", f2s),
        }

    spatial = result["arms"]["SPATIAL_SET"]["without_start"]
    result["contrasts"] = {
        "spatial_vs_no_recurrence": paired(
            spatial, result["arms"]["STATIC_CONTACT"]["without_start"]),
        "spatial_vs_order_destroyed": paired(
            spatial, result["arms"]["SPATIAL_SET_shuffled"]["without_start"]),
        "spatial_vs_noise_ceiling": paired(spatial, ceiling),
        "spatial_vs_uniform_sparse": paired(
            spatial, result["arms"]["RANDOM_SET"]["without_start"]),
        "spatial_vs_all_to_all": paired(
            spatial, result["arms"]["DENSE_TISSUE"]["without_start"]),
    }
    result["per_unit"] = rows

    path = out_root / "analysis" / "sufficiency_rnn.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result, indent=2))

    print()
    print("held-out same-start generation, seeded contacts removed (n = "
          f"{len(spatial)} patients)")
    print(f"{'arm':26s} {'median':>7} {'IQR':>16}")
    for arm in ARMS:
        v = np.array(list(result["arms"][arm]["without_start"].values()))
        print(f"{arm:26s} {np.median(v):7.3f} "
              f"[{np.percentile(v, 25):6.3f},{np.percentile(v, 75):6.3f}]")
    c = np.array([ceiling[k] for k in sorted(ceiling)])
    print(f"{'noise ceiling':26s} {np.median(c):7.3f} "
          f"[{np.percentile(c, 25):6.3f},{np.percentile(c, 75):6.3f}]")
    print()
    for name, block in result["contrasts"].items():
        print(f"  {name:30s} {block['median_delta']:+.3f}  higher in "
              f"{block['n_higher']}/{block['n']}  p={block['p']:.2e}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
