#!/usr/bin/env python3
"""B4 -- aggregate H2b to cohort level, patient-first.

Discipline the contract fixes (§10 of the common contract):

* seeds are repeat fits, not samples -- collapse them within a patient first;
* the denominator is held-out **episodes**, never grid rows or seizure rows;
* a cell that could not be estimated stays visible as such and never becomes 0;
* the sign test is over patients, and leave-one-patient-out is reported so a
  single patient cannot carry a cohort statement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b"
LEADS = ("5min", "30min", "2h", "6h")


def _binom_two_sided(k, n, p=0.5):
    if n == 0:
        return float("nan")
    from math import comb
    probs = [comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    obs = probs[k]
    return float(sum(x for x in probs if x <= obs + 1e-12))


def collect_b2(machine: Path):
    """(producer, lead, subject) -> dict with per-seed increments."""
    out: dict = {}
    for f in sorted(machine.glob("b2_field__*.json")):
        d = json.loads(f.read_text())
        prov = d.get("provenance", {})
        pid = prov.get("producer")
        if pid is None or d.get("tag") != "registry_producer":
            continue
        subj, seed = d["subject"], str(prov.get("seed"))
        for lead, rec in d.get("leads", {}).items():
            key = (pid, lead, subj)
            e = out.setdefault(key, {"seeds": {}, "n_scored": rec.get("n_scored", 0),
                                     "status": rec.get("status")})
            if rec.get("status") == "ok":
                e["seeds"][seed] = {"inc": rec["median_increment"],
                                    "base": rec["median_baseline"],
                                    "state": rec["median_state"]}
                e["n_scored"] = rec.get("n_scored", 0)
    return out


def collect_b1(machine: Path):
    out: dict = {}
    for f in sorted(machine.glob("b1__*.json")):
        d = json.loads(f.read_text())
        prov = d.get("provenance", {})
        pid = prov.get("producer")
        if pid is None or d.get("tag") != "registry_producer":
            continue
        subj, seed = d["subject"], str(prov.get("seed"))
        key = (pid, subj)
        e = out.setdefault(key, {"seeds": {}, "status": d.get("status"),
                                 "eval_events": d.get("eval_events", 0)})
        if d.get("status") == "ok":
            e["seeds"][seed] = d["nested_increment_log_score"]["mean_gain"]
            e["eval_events"] = d.get("eval_events", 0)
        e.setdefault("statuses", []).append(d.get("status"))
    return out


def summarise(per_subject_values, label):
    """Median over patients + sign test + leave-one-patient-out range."""
    v = np.array([x for x in per_subject_values if np.isfinite(x)], float)
    if v.size == 0:
        return {"n_patients": 0, "status": "not_estimable"}
    n_pos = int((v > 0).sum())
    loo = [float(np.median(np.delete(v, i))) for i in range(v.size)] if v.size > 1 else []
    return {
        "n_patients": int(v.size),
        "median": round(float(np.median(v)), 5),
        "n_positive": n_pos,
        "sign_test_p": round(_binom_two_sided(n_pos, int(v.size)), 4),
        "loo_median_min": round(min(loo), 5) if loo else None,
        "loo_median_max": round(max(loo), 5) if loo else None,
        "label": label,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    machine = args.out_root / "machine"

    b2 = collect_b2(machine)
    b1 = collect_b1(machine)
    report = {"b2_early_field": {}, "b1_survival": {}}

    producers = sorted({k[0] for k in b2} | {k[0] for k in b1})
    for pid in producers:
        report["b2_early_field"][pid] = {}
        for lead in LEADS:
            subj_vals, seed_spread, n_ep, n_cells, n_unest = [], [], 0, 0, 0
            for (p, l, s), e in b2.items():
                if p != pid or l != lead:
                    continue
                n_cells += 1
                if not e["seeds"]:
                    n_unest += 1
                    continue
                incs = [x["inc"] for x in e["seeds"].values()]
                subj_vals.append(float(np.median(incs)))
                if len(incs) > 1:
                    seed_spread.append(float(np.max(incs) - np.min(incs)))
                n_ep += e["n_scored"]
            s = summarise(subj_vals, "median per-patient increment in field score")
            s.update(n_heldout_episodes_scored=int(n_ep),
                     n_cells=n_cells, n_cells_not_estimable=n_unest,
                     median_seed_spread=(round(float(np.median(seed_spread)), 5)
                                         if seed_spread else None))
            report["b2_early_field"][pid][lead] = s

        subj_vals, n_ev, n_cells, n_unest = [], 0, 0, 0
        for (p, s_), e in b1.items():
            if p != pid:
                continue
            n_cells += 1
            if not e["seeds"]:
                n_unest += 1
                continue
            subj_vals.append(float(np.median(list(e["seeds"].values()))))
            n_ev += e["eval_events"]
        s = summarise(subj_vals, "median per-patient increment in survival log score")
        s.update(n_eval_events=int(n_ev), n_cells=n_cells, n_cells_not_estimable=n_unest)
        report["b1_survival"][pid] = s

    p = args.out_root / "machine/cohort_summary.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.rename(p)
    print(f"wrote {p}\n")

    print("=== B2  early ictal field (increment over the patient-average baseline) ===")
    print(f"{'producer':<15}{'lead':>7}{'pts':>5}{'median':>10}{'pos':>6}{'sign p':>9}"
          f"{'LOO range':>18}{'ep':>5}{'n/a':>5}")
    for pid in producers:
        for lead in LEADS:
            s = report["b2_early_field"][pid][lead]
            if s.get("n_patients", 0) == 0:
                print(f"{pid:<15}{lead:>7}{0:>5}   not_estimable"); continue
            loo = f"[{s['loo_median_min']:+.4f},{s['loo_median_max']:+.4f}]"
            print(f"{pid:<15}{lead:>7}{s['n_patients']:>5}{s['median']:>+10.4f}"
                  f"{s['n_positive']:>6}{s['sign_test_p']:>9.3f}{loo:>18}"
                  f"{s['n_heldout_episodes_scored']:>5}{s['n_cells_not_estimable']:>5}")

    print("\n=== B1  seizure survival (increment in log score) ===")
    print(f"{'producer':<15}{'pts':>5}{'median':>10}{'pos':>6}{'sign p':>9}{'events':>8}{'n/a':>5}")
    for pid in producers:
        s = report["b1_survival"][pid]
        if s.get("n_patients", 0) == 0:
            print(f"{pid:<15}{0:>5}   not_estimable"); continue
        print(f"{pid:<15}{s['n_patients']:>5}{s['median']:>+10.4f}{s['n_positive']:>6}"
              f"{s['sign_test_p']:>9.3f}{s['n_eval_events']:>8}{s['n_cells_not_estimable']:>5}")


if __name__ == "__main__":
    main()
