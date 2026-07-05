"""M3A-A2 pilot analyzer — per-run R-stay / R-excursion / R-runaway verdict + bout R-class gate.
Mirrors analyze_a1c_pilot.py / analyze_m3a_a1b.py. Reads readout_*.json; writes status_a2_pilot.json.

Verdict (spec §4.2 + §4.3.2):
  R-stay      : rho never crossed the (Task-0b-locked) boundary B.
  R-excursion : crossed B AND seizure-like phenotype (R4a or R3, ignited) AND returned (absolute tail
                <= 1.5 AND tanks refilled above the seizure-entry product lgr/B).
  R-runaway   : crossed B but did not return.
Recovery threshold derives from B + lgr (NOT a hardcoded 0.74).
"""
import os, sys, json, glob
from collections import Counter
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAIL_GATE = 1.5          # A1c absolute tail
IGNITE_PEAK = 0.05       # A1c ignition (active fraction)
SEIZURE_RCLASSES = ("R4a", "R3")   # R4a primary; R3 -> R4a-like allowed
_RORDER = ["R0", "R1", "R2", "R3", "R4a", "R4b"]


def _max_rclass(events):
    mx = "R0"
    for ev in events:
        rc = ev.get("R_class", "R0")
        if rc in _RORDER and _RORDER.index(rc) > _RORDER.index(mx):
            mx = rc
    return mx


def _run_verdict(r):
    a2 = r.get("a2", {}); act = r.get("activity", {})
    B = a2.get("seizure_boundary")
    if B is None:
        return "NO_BOUNDARY"                      # frozen Task-0b runs: characterize phenotype, not verdict
    crossed = a2.get("rho_peak", 0.0) >= B
    if not crossed:
        return "R-stay"
    ignited = (act.get("active_E_fraction_peak", 0.0) >= IGNITE_PEAK) or (act.get("peak_E_rate_hz", 0.0) >= 3.0)
    returned = act.get("tail_to_baseline_ratio", 1e9) <= TAIL_GATE
    lgr = a2.get("rho_static", 1.0)
    seizure_product = lgr / B                     # rho crosses back below B when q-product > lgr/B
    q_recovered = (a2.get("q_core_end", 0.0) * a2.get("q_global_end", 1.0)) > seizure_product
    max_R = _max_rclass(r.get("events", []))
    seizure_pheno = (max_R in SEIZURE_RCLASSES) and ignited
    if seizure_pheno and returned and q_recovered:
        return "R-excursion"
    return "R-runaway"


def main(base):
    rows = []
    for f in sorted(glob.glob(os.path.join(base, "readout_*.json"))):
        r = json.load(open(f))
        a2 = r.get("a2", {})
        rows.append(dict(tag=os.path.basename(f)[8:-5], verdict=_run_verdict(r),
                         mode=a2.get("mode"), k_use=a2.get("k_use"), q_target=a2.get("q_target"),
                         rho_static=a2.get("rho_static"), seizure_boundary=a2.get("seizure_boundary"),
                         rho_peak=a2.get("rho_peak"), q_core_min=a2.get("q_core_min"),
                         a_core_mean=a2.get("a_core_mean"), tail=r.get("activity", {}).get("tail_to_baseline_ratio"),
                         frac_seizure_band=a2.get("frac_time_seizure_band"),
                         I_I_over_I_E_core=a2.get("I_I_over_I_E_core"),
                         max_R=_max_rclass(r.get("events", [])), n_events=r.get("n_events")))
    status = dict(base=os.path.relpath(base, ROOT),
                  tier="MECHANISM-SCREEN (NOT seizure-mechanism validation)",
                  tail_gate=TAIL_GATE, n_runs=len(rows), per_run=rows,
                  verdict_counts=dict(Counter(x["verdict"] for x in rows)),
                  caveat="PLACEHOLDER params; rho boundary = Task-0b frozen-q calibrated B; recovery is "
                         "emergent (fast self-limit + slow refill), NOT slow-var-caused termination.")
    json.dump(status, open(os.path.join(base, "status_a2_pilot.json"), "w"), indent=1)
    print(json.dumps(status["verdict_counts"], indent=1))
    return status


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(ROOT, "results/topic4_sef_hfo/m3a_slowvars/a2_abbott_lg"))
