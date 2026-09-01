#!/usr/bin/env python3
"""B0.4 -- what can the early-ictal-field target support, before any state exists?

Two numbers per patient, both computed with TRAIN-only information:

``patient_average_baseline``
    the H2b spec's baseline #3: predict a held-out seizure's early field by the
    mean of the TRAIN seizures' fields. This is what a frozen state has to beat.

``target_ceiling_heldout``
    how well one held-out seizure resembles another of the same patient, i.e.
    how reproducible the target is at the single-seizure level.

    This is NOT an upper bound on what a predictor can score, and the measured
    numbers say so directly: the patient-average baseline beats it in 5 of 12
    patients (916: 0.940 vs 0.917; 253: 0.367 vs -0.249), because averaging
    several TRAIN fields cancels seizure-specific noise that no single seizure
    can. Read it as "how much of a single seizure's field is idiosyncratic":
    where it is low, a patient contributes little power to distinguish a state
    from a static prototype -- "assay not estimable" rather than a biological
    negative (H2b spec §8).

Writing this down *before* reading any producer is the point: it fixes what a
null result is allowed to mean.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_h2b_transfer.normalization import (  # noqa: E402
    fit_field_normalization,
    fit_route_templates,
)
from src.topic5_h2b_transfer.risk_grid import (  # noqa: E402
    DEFAULT_POSTICTAL_EXCLUSION_SECONDS,
    group_seizure_episodes,
)

DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b"
DEFAULT_DATA = Path("/data/hfosp_group_event_state_v0_2/agent_b")
# H2b spec §1: the first 5 s is the primary target, 10 s is the sensitivity.
PRIMARY_KEY = "hfa_field_5s"
SENSITIVITY_KEY = "hfa_field_10s"


def _rho(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4:
        return float("nan")
    r = spearmanr(a[ok], b[ok]).statistic
    return float(r) if np.isfinite(r) else float("nan")


def _pairwise_median(fields):
    rs = []
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            r = _rho(fields[i], fields[j])
            if np.isfinite(r):
                rs.append(r)
    return float(np.median(rs)) if rs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--crosswalk", type=Path, default=DEFAULT_OUT / "support/seizure_crosswalk.csv")
    ap.add_argument("--max-routes", type=int, default=3)
    args = ap.parse_args()

    matched: dict[str, list[dict]] = {}
    for r in csv.DictReader(args.crosswalk.open()):
        if r["disposition"] != "matched":
            continue
        matched.setdefault(r["subject"], []).append(
            {"seizure_id": r["seizure_id"], "onset_epoch": float(r["onset_epoch"]),
             "offset_epoch": float(r["offset_epoch"])}
        )
    for v in matched.values():
        v.sort(key=lambda s: s["onset_epoch"])

    rows = []
    D = args.data_root / "early_field"
    for jp in sorted(D.glob("*.json")):
        subject = jp.stem
        meta = json.loads(jp.read_text())
        z = np.load(D / f"{subject}.npz")
        ok = {s["seizure_id"]: i for i, s in enumerate(meta["seizures"]) if s["status"] == "ok"}
        sz = matched.get(subject, [])
        if not sz or not ok:
            continue

        episodes = group_seizure_episodes(sz, gap_seconds=DEFAULT_POSTICTAL_EXCLUSION_SECONDS)
        n_train_ep = max(1, math.ceil(len(episodes) / 2))
        train_ids, held_ids = [], []
        for ei, ep in enumerate(episodes):
            bucket = train_ids if ei < n_train_ep else held_ids
            bucket.append(ep[0]["seizure_id"])  # only episode leads are targets

        def fields_for(ids, key):
            out, kept = [], []
            for sid in ids:
                i = ok.get(sid)
                if i is None:
                    continue
                out.append(z[f"{key}__{i:03d}"])
                kept.append(sid)
            return (np.vstack(out) if out else np.empty((0, 0))), kept

        def support_for(key):
            Ftr, ktr = fields_for(train_ids, key)
            Fho, kho = fields_for(held_ids, key)
            base = []
            if len(ktr) >= 1 and len(kho) >= 1:
                proto = np.nanmean(Ftr, axis=0)  # TRAIN-only, frozen
                base = [b for b in (_rho(proto, f) for f in Fho) if np.isfinite(b)]
            return {
                "ceiling": _pairwise_median(list(Fho)) if len(kho) >= 2 else float("nan"),
                "train_repro": _pairwise_median(list(Ftr)) if len(ktr) >= 2 else float("nan"),
                "baseline": float(np.median(base)) if base else float("nan"),
                "kept_train": ktr, "kept_held": kho,
            }

        prim = support_for(PRIMARY_KEY)
        sens = support_for(SENSITIVITY_KEY)
        kept_train, kept_held = prim["kept_train"], prim["kept_held"]
        ceiling, train_repro = prim["ceiling"], prim["train_repro"]
        baseline_scores = [prim["baseline"]] if np.isfinite(prim["baseline"]) else []
        fields_for_routes = lambda ids: fields_for(ids, PRIMARY_KEY)

        routes = None
        if len(kept_train) >= 2:
            all_ids = kept_train + kept_held
            F_all, _ = fields_for_routes(all_ids)
            try:
                routes = fit_route_templates(
                    F_all, train_index=list(range(len(kept_train))),
                    max_routes=min(args.max_routes, len(kept_train)),
                )
            except ValueError:
                routes = None
        norm = None
        if len(kept_train) >= 1:
            F_all, _ = fields_for_routes(kept_train + kept_held)
            norm = fit_field_normalization(F_all, train_index=list(range(len(kept_train))))

        rows.append({
            "subject": subject,
            "dataset": meta["dataset"],
            "n_channels": meta["n_channels"],
            "n_seizures_ok": len(ok),
            "n_episodes": len(episodes),
            "n_train_leads": len(kept_train),
            "n_heldout_leads": len(kept_held),
            "train_field_reproducibility": round(train_repro, 4) if np.isfinite(train_repro) else "",
            "target_ceiling_heldout": round(ceiling, 4) if np.isfinite(ceiling) else "",
            "patient_average_baseline": (round(float(np.median(baseline_scores)), 4)
                                         if baseline_scores else ""),
            "target_ceiling_heldout_10s": (round(sens["ceiling"], 4)
                                           if np.isfinite(sens["ceiling"]) else ""),
            "patient_average_baseline_10s": (round(sens["baseline"], 4)
                                             if np.isfinite(sens["baseline"]) else ""),
            "n_routes": routes.n_routes if routes else "",
            "route_support": "|".join(str(s) for s in routes.support) if routes else "",
            "route_under_supported": (sum(routes.under_supported) if routes else ""),
            "normalization_n_train": norm.n_train if norm else "",
        })

    p = args.out_root / "support/field_target_support.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    tmp.rename(p)
    print(f"wrote {p}\n")

    def col(name):
        return np.array([float(r[name]) for r in rows if r[name] != ""], float)

    ceil_v, base_v = col("target_ceiling_heldout"), col("patient_average_baseline")
    print(f"{'subject':>20} {'ok':>4} {'HO':>4} {'ceiling':>8} {'pat-avg':>8}")
    for r in sorted(rows, key=lambda r: -(float(r["target_ceiling_heldout"]) if r["target_ceiling_heldout"] != "" else -9)):
        print(f"{r['subject']:>20} {r['n_seizures_ok']:4d} {r['n_heldout_leads']:4d} "
              f"{str(r['target_ceiling_heldout']):>8} {str(r['patient_average_baseline']):>8}")
    if ceil_v.size:
        print(f"\ntarget ceiling (held-out seizures resembling each other): "
              f"median {np.median(ceil_v):+.3f}   below +0.3: {(ceil_v < 0.3).sum()}/{ceil_v.size}")
    if base_v.size:
        print(f"patient-average-field baseline                        : median {np.median(base_v):+.3f}")


if __name__ == "__main__":
    main()
