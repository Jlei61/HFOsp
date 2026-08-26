#!/usr/bin/env python3
"""Aggregate the just-in-time synthetic recovery tests into one table.

A truth that is unidentifiable at this sample size limits the interpretation of
the comparison it was built for, and nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.topic5_epi_prssm.contracts import (  # noqa: E402
    OUTPUT_ROOT, atomic_write_csv, atomic_write_json, code_revision, package_hash,
)
from src.topic5_epi_prssm.synthetic_truths import TRUTH_PURPOSE  # noqa: E402

OUT = OUTPUT_ROOT / "synthetic"


def main() -> None:
    #: Truths whose generator was rewritten during this run: their resource acted
    #: straight on contact excitability, a path the spec forbids the model to use,
    #: so it was changed to modulate the gain with which the latent state reaches
    #: the readout.  Only these truths are subject to supersession; a truth whose
    #: generator never changed keeps every seed it ran.
    REWRITTEN = {"t1_autonomous_resource", "r2_impulse", "r3_integrated_exposure",
                 "event_count_only", "observer_resource_substitution", "hidden_common_cause"}
    records = []
    for path in sorted(OUT.glob("*.json")):
        record = json.loads(path.read_text())
        if "truth" not in record:
            continue
        record["__path"] = str(path)
        record["__mtime"] = path.stat().st_mtime
        records.append(record)
    newest_hash = {}
    for record in records:
        key = record["truth"]
        if key not in REWRITTEN:
            continue
        if key not in newest_hash or record["__mtime"] > newest_hash[key][0]:
            newest_hash[key] = (record["__mtime"], record["package_hash"])
    def _superseded(record) -> bool:
        key = record["truth"]
        return key in newest_hash and record["package_hash"] != newest_hash[key][1]
    superseded = [{"truth": r["truth"], "seed": r["seed"], "package_hash": r["package_hash"],
                   "path": r["__path"],
                   "reason": "generator rewritten so the resource acts through the state gain "
                             "instead of straight on contact excitability"}
                  for r in records if _superseded(r)]
    records = [r for r in records if not _superseded(r)]

    rows, per_truth = [], {}
    for record in records:
        verdict = record["verdict"]
        row = {"truth": record["truth"], "seed": record["seed"],
               "goal": record["purpose"]["goal"], "expected": record["purpose"]["expect"],
               "status": verdict.get("status"), "winner": verdict.get("winner"),
               "ranking": " < ".join(verdict.get("ranking", [])),
               "spread": verdict.get("spread")}
        for arm, value in (verdict.get("validation_by_arm") or {}).items():
            row[f"val_{arm}"] = value
        for arm, value in (verdict.get("open_loop_h20_by_arm") or {}).items():
            row[f"ol20_{arm}"] = value
        rows.append(row)
        per_truth.setdefault(record["truth"], []).append(verdict)
    frame = pd.DataFrame(rows)
    atomic_write_csv(OUT / "synthetic_recovery.csv", frame)

    summary = {}
    for truth, verdicts in sorted(per_truth.items()):
        winners = [v.get("winner") for v in verdicts if v.get("winner")]
        counts = {w: winners.count(w) for w in set(winners)}
        identifiable = sum(1 for v in verdicts if v.get("status") == "IDENTIFIABLE")
        summary[truth] = {
            "goal": TRUTH_PURPOSE[truth]["goal"], "expected": TRUTH_PURPOSE[truth]["expect"],
            "n_seeds": len(verdicts), "n_identifiable": identifiable,
            "winner_counts": counts,
            "modal_winner": max(counts, key=counts.get) if counts else None,
            "median_spread": float(np.median([v.get("spread", np.nan) for v in verdicts])),
        }
    atomic_write_json(OUT / "SYNTHETIC_RECOVERY_SUMMARY.json", {
        "contract": "topic5_epi_prssm_v0_1_synthetic_recovery_summary",
        "n_runs": len(rows), "by_truth": summary,
        "n_superseded_runs": len(superseded), "superseded": superseded,
        "note": "each truth is compared only with the models adjacent to it; an unidentifiable "
                "truth limits that comparison alone",
        "code_revision": code_revision(), "package_hash": package_hash(),
    })
    for truth, block in summary.items():
        print(f"{truth:32s} seeds={block['n_seeds']} ident={block['n_identifiable']} "
              f"modal={block['modal_winner']}  expect: {block['expected'][:60]}")


if __name__ == "__main__":
    main()
