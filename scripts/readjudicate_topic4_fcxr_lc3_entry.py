#!/usr/bin/env python
"""Re-adjudicate each entry ledger's reproduction verdict from the stored events.

The runner's own guard filtered both event lists by "ends before the 20 s cut",
which keeps a tail event the short run truncated while dropping the whole copy
the long run recorded, so a bit-identical trajectory can report one event too
many. Both event lists are on disk, so the corrected verdict is a re-reading, not
a re-simulation.

The original verdict is preserved alongside the corrected one. A verdict that
changes is recorded as changed rather than overwritten.
"""
from __future__ import annotations

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.topic4_fcxr_lc3_reproduction import events_reproduce  # noqa: E402

BASE = os.path.join(ROOT, "results", "topic4_sef_hfo",
                    "fcxr_lc3_dx_spatial_instability")
LEDGERS = os.path.join(BASE, "entry_ledger")
RECON = os.path.join(BASE, "dynamic_reconnaissance")
OUT = os.path.join(LEDGERS, "reproduction_readjudicated.json")


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(LEDGERS, "entry_noise*.json"))):
        if ".DONE." in path or ".RUNNING" in path:
            continue
        led = json.load(open(path))
        if led.get("status") != "COMPLETE":
            continue
        seed = int(led["noise_seed"])
        rec_path = os.path.join(RECON, f"recon_noise{seed}.json")
        if not os.path.isfile(rec_path):
            continue
        rec = json.load(open(rec_path))
        out = events_reproduce(led["events"], rec["events"], cut_ms=float(led["T_ms"]))
        rows.append(dict(
            noise_seed=seed,
            original_verdict=led.get("reproduces_recorded_trajectory"),
            original_detail=led.get("reproduction_detail"),
            corrected_verdict=out["reproduces"], corrected_detail=out["detail"],
            changed=bool(led.get("reproduces_recorded_trajectory") != out["reproduces"]),
            n_compared=out["n_compared"], margin_ms=out["margin_ms"],
            comparable_until_ms=out["comparable_until_ms"],
            onset_ms=led["onset_ms"],
            n_returning_before_onset=led["event_ledger"]["n_returning_before_onset"],
        ))

    payload = dict(
        schema="fcxr-lc3-reproduction-readjudication-1.0",
        rule=("compare only events both runs could see whole: the span ends one "
              "longest-event margin short of the cut"),
        n_seeds=len(rows), n_reproducing=sum(1 for r in rows if r["corrected_verdict"]),
        n_changed=sum(1 for r in rows if r["changed"]),
        note=("no re-simulation; both event lists were already on disk. Original "
              "verdicts are preserved, not overwritten."),
        rows=rows)
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    print(json.dumps(payload, indent=2))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
