#!/usr/bin/env python3
"""Refuse to interpret an artefact that is older than the thing that produced it.

Four separate times in one night this leg computed a "result" from a stale input:
a stage that skipped its work, a caliper whose evidence was never persisted, a
launcher that adopted an overwrite request as already done, and finally a chain that
omitted the aggregation step entirely.  None of them raised anything -- each produced
a plausible number from the previous day's file.

The common boundary is the same: nothing checked that a consumer's input was newer
than its producer's output.  This check does exactly that, and nothing else.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from _common import OUTPUT_ROOT, atomic_write_json  # noqa: E402

H2B = OUTPUT_ROOT / "seizure_link_preictal"


def newest(pattern: str) -> tuple[str, float] | None:
    files = glob.glob(pattern)
    if not files:
        return None
    newest_path = max(files, key=lambda f: Path(f).stat().st_mtime)
    return newest_path, Path(newest_path).stat().st_mtime


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance-seconds", type=float, default=120.0)
    parser.add_argument("--layers", nargs="+",
                        default=["linear_graph_recurrent", "leaky_state",
                                 "resource_anchored_on_best_family"])
    parser.add_argument("--leads", nargs="+",
                        default=["lead30m", "lead60m", "lead15m", "lead5m"])
    args = parser.parse_args()

    producer = newest(str(H2B / "per_subject/*.json"))
    if producer is None:
        raise SystemExit("no producer output at all")

    # EVERY required artefact, not the newest one.  Taking the newest let a single
    # freshly-written layer vouch for three stale ones on the first run of this check.
    checks, stale = {}, []
    for layer in args.layers:
        for lead in args.leads:
            for kind in ("preictal_effects", "preictal_denominators"):
                label = f"{kind}__{layer}__{lead}"
                path = H2B / f"{kind}__{layer}__{lead}.csv"
                if not path.exists():
                    checks[label] = {"status": "MISSING"}
                    stale.append(label)
                    continue
                lag = producer[1] - path.stat().st_mtime
                fresh = lag <= args.tolerance_seconds
                checks[label] = {"status": "FRESH" if fresh else "STALE",
                                 "seconds_older_than_producer": round(lag, 1)}
                if not fresh:
                    stale.append(label)

    summary = {
        "contract": "topic5_epi_prssm_downstream_freshness",
        "producer_newest": producer[0],
        "n_checked": len(checks),
        "checks": checks,
        "verdict": "FRESH" if not stale else "STALE",
        "stale": stale,
    }
    atomic_write_json(H2B / "DOWNSTREAM_FRESHNESS.json", summary)
    print(json.dumps(summary, indent=1))
    if stale:
        raise SystemExit(f"stale downstream artefacts: {stale}")


if __name__ == "__main__":
    main()
