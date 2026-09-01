#!/usr/bin/env python3
"""Sensitivity of the H2b denominator to the postictal exclusion length.

The spec fixes 60 min as primary and asks for 30/120 min as sensitivity. The
question this answers is narrow but load-bearing: is the held-out denominator
sitting on a cliff edge of that arbitrary choice?
"""
from __future__ import annotations
import csv, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.topic5_h2b_transfer.risk_grid import (  # noqa: E402
    group_seizure_episodes, lead_anchor_status, merge_spans)

V01 = Path("/home/honglab/leijiaxin/HFOsp/results/epi_prssm/group_event_state/v0_1")
DS = Path("/data/hfosp_group_event_state_v0_1/dataset")
OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b"
LEADS = ((300., "5min"), (1800., "30min"), (7200., "2h"), (21600., "6h"))


def main():
    cov = {s.name: {Path(x).stem for x in json.loads((s / "index.json").read_text())["source_shards"]}
           for s in sorted(p for p in DS.iterdir() if p.is_dir())}
    spans = {s: [] for s in cov}
    for r in csv.DictReader((V01 / "block_inventory.csv").open()):
        if r["subject"] in cov and r["record_name"] in cov[r["subject"]]:
            spans[r["subject"]].append((float(r["block_start_epoch"]), float(r["block_end_epoch"])))
    spans = {k: merge_spans(v) for k, v in spans.items()}

    sz = {}
    for r in csv.DictReader((OUT / "support/seizure_crosswalk.csv").open()):
        if r["disposition"] == "matched":
            sz.setdefault(r["subject"], []).append(
                {"seizure_id": r["seizure_id"], "onset_epoch": float(r["onset_epoch"]),
                 "offset_epoch": float(r["offset_epoch"])})
    for v in sz.values():
        v.sort(key=lambda s: s["onset_epoch"])

    out = {"primary_postictal_exclusion_sec": 3600.0, "arms": {}}
    for excl, label in ((1800., "30min"), (3600., "60min_primary"), (7200., "120min")):
        tot_ep = tot_ho = 0
        per_lead = {n: 0 for _, n in LEADS}
        for subj, v in sz.items():
            eps = group_seizure_episodes(v, gap_seconds=excl)
            ntr = max(1, math.ceil(len(eps) / 2))
            tot_ep += len(eps); tot_ho += len(eps) - ntr
            for ei, ep in enumerate(eps):
                if ei < ntr:
                    continue
                for lead, name in LEADS:
                    if lead_anchor_status(ep[0]["onset_epoch"] - lead, spans[subj], v,
                                          postictal_exclusion_seconds=excl) == "ok":
                        per_lead[name] += 1
        out["arms"][label] = {"n_episodes": tot_ep, "n_heldout_episodes": tot_ho,
                              "heldout_with_anchor": per_lead}
    p = OUT / "support/postictal_sensitivity.json"
    tmp = p.with_suffix(".json.tmp"); tmp.write_text(json.dumps(out, indent=2)); tmp.rename(p)
    print(f"wrote {p}")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
