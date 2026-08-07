"""Re-run Leg A's high-scoring region on seeds that had no say in choosing it.

The region was picked as the top slice of ninety-eight cells scored on four
networks each. Whatever optimism that selection introduced is inherited by the
value read off the map, so the only way to know its size is to measure the same
cells again on networks that took no part in the choice. The two numbers are
reported side by side and their difference IS the winner's curse estimate; it is
never subtracted away or reported alone.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_report import PRIMARY_KEY  # noqa: E402
from src.topic4_core_field_runner import (atomic_write_json,  # noqa: E402
                                          canonical_checksum, provenance)
from src.topic4_core_field_scoring import load_patient_templates  # noqa: E402

OUT = "results/topic4_sef_hfo/data_driven_core_field_stage3"
STAGE2 = "results/topic4_sef_hfo/data_driven_core_field"
DROP = ("checksum", "provenance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()

    cfg = json.load(open(os.path.join(a.out, "config", "sweep_config.json")))
    if canonical_checksum(cfg, drop=DROP) != cfg["checksum"]:
        raise SystemExit("sweep config checksum mismatch")
    summ = json.load(open(os.path.join(a.out, "sweep_summary.json")))
    region = summ["high_scoring_region"]
    if not region["centers"]:
        raise SystemExit("no high-scoring region to confirm")

    confirm_seeds = list(cfg["confirm_seeds"])
    overlap = set(confirm_seeds) & set(cfg["sweep_seeds"])
    if overlap:
        raise SystemExit(f"confirmation seeds must be independent; overlap {overlap}")

    sys.path.insert(0, os.path.join("src", "snn_engine"))
    from scripts.run_topic4_core_field_stage3_sweep import _evaluate, _score
    sigma = float(cfg["sigmas"]["primary"])
    cache = os.path.join(STAGE2, "network_cache")
    jobs = [(c[0], c[1], sigma, sd, cfg, cache)
            for c in region["centers"] for sd in confirm_seeds]
    print(f"[confirm] {len(region['centers'])} cells x {len(confirm_seeds)} "
          f"independent seeds = {len(jobs)} sims", flush=True)

    with Pool(a.workers, maxtasksperchild=1) as pool:
        res = pool.map(_evaluate, jobs)

    tgt = load_patient_templates(cfg["subject"], PRIMARY_KEY[0])
    rule = PRIMARY_KEY[1]
    S_map = np.asarray(summ["maps"]["primary"]["S_rank"], dtype=object)
    n = int(cfg["grid"]["n"])
    index = {(round(c[0], 6), round(c[1], 6)): i
             for i, c in enumerate(cfg["grid"]["centers"])}

    rows = []
    for c in region["centers"]:
        i = index[(round(c[0], 6), round(c[1], 6))]
        r, col = divmod(i, n)
        vals = [_score(x["events"], cfg, tgt, rule)["S_rank"]
                for x in res if "error" not in x
                and abs(x["cx"] - c[0]) < 1e-6 and abs(x["cy"] - c[1]) < 1e-6]
        vals = [v for v in vals if v is not None]
        built = S_map[r][col]
        rows.append(dict(center=[float(c[0]), float(c[1])],
                         map_value=None if built is None else float(built),
                         confirmed_mean=float(np.mean(vals)) if vals else None,
                         confirmed_n_valid=len(vals),
                         n_confirm_seeds=len(confirm_seeds)))

    paired = [(x["map_value"], x["confirmed_mean"]) for x in rows
              if x["map_value"] is not None and x["confirmed_mean"] is not None]
    optimism = (float(np.mean([b - a_ for a_, b in paired])) if paired else None)

    out = dict(
        stage="stage3_legA_confirmation",
        cells=rows,
        winners_curse=dict(
            mean_confirmed_minus_map=optimism, n_paired=len(paired),
            reading=("negative means the map was optimistic about this region by "
                     "that much; report both numbers, never the map value alone")),
        confirm_seeds=confirm_seeds, sweep_seeds=list(cfg["sweep_seeds"]),
        n_errors=sum(1 for x in res if "error" in x),
        config_checksum=cfg["checksum"], provenance=provenance())
    atomic_write_json(out, os.path.join(a.out, "region_confirmation.json"))

    print(f"\n{'centre':>16} {'map':>8} {'confirmed':>10} {'valid':>6}")
    for x in rows:
        mv = "  n/a" if x["map_value"] is None else f"{x['map_value']:+.3f}"
        cv = "  n/a" if x["confirmed_mean"] is None else f"{x['confirmed_mean']:+.3f}"
        print(f"  ({x['center'][0]:5.1f},{x['center'][1]:5.1f}) {mv:>8} {cv:>10} "
              f"{x['confirmed_n_valid']}/{x['n_confirm_seeds']}")
    if optimism is not None:
        print(f"\nwinner's curse: confirmed - map = {optimism:+.3f} "
              f"over {len(paired)} cells")


if __name__ == "__main__":
    main()
