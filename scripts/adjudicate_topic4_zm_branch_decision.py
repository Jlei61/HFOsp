#!/usr/bin/env python
"""Read every completed phase manifest and write ONE fail-closed branch verdict (plan Task 13).

  python scripts/adjudicate_topic4_zm_branch_decision.py --verify-gates

`--verify-gates` re-runs the Phase-0 gate tests through pytest and records the result, so the
verdict cannot claim exact-resume parity that nobody re-checked. This script runs no simulation and
derives no new metric: if a phase is absent, the verdict degrades rather than guessing.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.topic4_zm_branch_verdict as BV  # noqa: E402
import src.topic4_zm_minimal_carrier as MC  # noqa: E402
import src.topic4_zm_neighbourhood as NBH  # noqa: E402

OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision")
PHASE0 = os.path.join(OUT, "phase0")
GATE_TESTS = ["tests/test_topic4_zm_fork_state.py", "tests/test_topic4_zm_checkpoint_hook.py",
              "tests/test_topic4_zm_exact_resume.py", "tests/test_topic4_zm_noise_bank.py",
              "tests/test_topic4_zm_minimal_carrier.py", "tests/test_topic4_zm_branch_verdict.py",
              "tests/test_topic4_zm_neighbourhood.py",
              "tests/test_topic4_zm_empirical_carrier.py", "tests/test_snn_shunting.py",
              "tests/test_topic4_zm_ictal_carrier.py",
              "tests/test_topic4_zm_branch_runner_diagnostics.py",
              "tests/test_topic4_zm_carrier_morphology.py",
              "tests/test_topic4_zm_source_rhythm.py",
              "tests/test_topic4_zm_effective_rank.py",
              "tests/test_topic4_zm_modal_operator.py",
              "tests/test_topic4_zm_boundaries.py",
              "tests/test_snn_gates.py", "tests/test_zm_slow_field_parity.py",
              "tests/test_a1c_feedback.py"]

#: the FULL spec contract (spec rev3.1 §5.2/§5.3/§6.1) -- what a complete matrix would cover
SPEC_FULL = dict(seeds=[1, 3, 4], bins=["bounded_early", "bounded_mid", "bounded_late"],
                 phases=["trough", "rising", "peak"],
                 arms=["freeze_all", "freeze_zm", "freeze_zsg", "freeze_z", "dynamic_replay",
                       "dynamic_z_only"])
#: the LADDER this run actually committed to, declared before results were inspected: the four
#: carrier-question arms at the mid slow bin in two natural fast phases, on three primary seeds.
#: Everything the ladder drops is listed in the coverage report; nothing is silently truncated.
PLANNED = dict(seeds=[1, 3, 4], bins=["bounded_mid"], phases=["trough", "peak"],
               arms=["freeze_all", "freeze_zm", "freeze_zsg", "freeze_z"])


def verify_gates():
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    t0 = time.time()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", *GATE_TESTS],
                       cwd=_ROOT, capture_output=True, text=True, env=env)
    tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ""
    ev = dict(returncode=p.returncode, summary=tail, tests=GATE_TESTS,
              wall_s=round(time.time() - t0, 1), timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
              passed=bool(p.returncode == 0))
    os.makedirs(PHASE0, exist_ok=True)
    with open(os.path.join(PHASE0, "gate_evidence.json"), "w") as f:
        json.dump(ev, f, indent=2)
    print(f"[gates] {'PASS' if ev['passed'] else 'FAIL'}: {tail}")
    return ev


def load_json(path, default=None):
    return json.load(open(path)) if os.path.exists(path) else default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-gates", action="store_true")
    a = ap.parse_args()

    gates = verify_gates() if a.verify_gates else load_json(
        os.path.join(PHASE0, "gate_evidence.json"), dict(passed=False, summary="not verified"))
    inv = load_json(os.path.join(PHASE0, "state_inventory.json"), {})
    guard = load_json(os.path.join(PHASE0, "engine_guard_change.json"), {})
    ref_lock = load_json(os.path.join(OUT, "phase0c", "carrier_target_lock.json"))
    ref_blocked = load_json(os.path.join(OUT, "phase0c", "blocked_reference_artifacts.json"))

    anchors, eligible, ied = {}, [], {}
    for p in sorted(glob.glob(os.path.join(OUT, "anchors", "seed*", "anchor.json"))):
        m = json.load(open(p))
        s = int(m["seed"])
        anchors[s] = dict(eligible=m["selection"]["eligibility"]["eligible"],
                          reasons=m["selection"]["eligibility"]["reasons"],
                          bounded_ms=m["selection"]["eligibility"]["bounded_ms"],
                          escalation_ms=m["selection"]["eligibility"]["escalation_ms"],
                          n_states=len(m.get("captured_states", [])),
                          returning_events=m["selection"]["eligibility"]["returning_events"],
                          config_sha=m["config_sha"])
        if anchors[s]["eligible"]:
            eligible.append(s)
            ied[s] = m["locks"]["ied_lifetime_ms"]

    rows = []
    for p in sorted(glob.glob(os.path.join(OUT, "forks", "seed*", "fork_matrix.json"))):
        rows.extend(json.load(open(p))["rows"])

    cells = BV.classify_matrix(rows, ied, MC.classify_replicas) if rows else {}
    per_arm = BV.carrier_window(cells) if cells else {}
    coverage = BV.coverage_report(cells, PLANNED) if cells else None
    if coverage:
        coverage["declared_ladder"] = PLANNED
        coverage["spec_full_matrix"] = BV.coverage_report(cells, SPEC_FULL)

    nb_rows, nb_by_seed = [], {}
    for p in sorted(glob.glob(os.path.join(OUT, "neighbourhood", "seed*", "neighbourhood.json"))):
        m = json.load(open(p))
        nb_rows.extend(m["rows"])
        nb_by_seed[int(m["seed"])] = m
    neighbourhood = None
    if nb_rows:
        local_pos, local_neg, fam_result = [], [], {}
        all_complete = True
        all_agree = True
        for s, m in nb_by_seed.items():
            # The runner must certify the full paired-noise/minimal-subsystem
            # neighbourhood contract.  Raw "survived" counts are insufficient:
            # they ignore stationarity, IED lifetime, compatible phases and
            # representation agreement.
            audit = m.get("audit") or {}
            complete = bool(audit.get("complete", False))
            agree = bool(audit.get("representations_agree", False))
            all_complete &= complete
            all_agree &= agree
            fam_result[s] = audit.get("family_results", {})
            if complete and bool(audit.get("local_carrier_window", False)):
                local_pos.append(s)
            if complete and bool(audit.get("formal_local_negative", False)):
                local_neg.append(s)
        evidence_complete = bool(
            all_complete and set(BV.PRIMARY_SEEDS).issubset(nb_by_seed) and
            set(BV.PRIMARY_SEEDS).issubset(local_pos + local_neg))
        neighbourhood = dict(NBH.branch_verdict(
            False, local_pos, eligible, representations_agree=all_agree,
            local_negative_seeds=local_neg, evidence_complete=evidence_complete),
                             family_results=fam_result, n_rows=len(nb_rows),
                             seeds_audited=sorted(nb_by_seed),
                             complete=bool(evidence_complete),
                             representations_agree=bool(all_agree),
                             local_positive_seeds=sorted(local_pos),
                             local_negative_seeds=sorted(local_neg))

    smallest = MC.smallest_positive_subsystem({
        arm: ("stable_carrier" if info["status"] == "carrier_window"
              else "transient_carrier_like")
        for arm, info in per_arm.items()
        if not info.get("is_control_arm", False)
    }) if per_arm else None

    long_rows, dt2_rows = [], []
    for p in sorted(glob.glob(os.path.join(
            OUT, "confirmations", "long", "seed*", "fork_matrix.json"))):
        long_rows.extend(json.load(open(p)).get("rows", []))
    for p in sorted(glob.glob(os.path.join(
            OUT, "confirmations", "dt2", "seed*", "fork_matrix.json"))):
        dt2_rows.extend(json.load(open(p)).get("rows", []))
    window_arms = [
        arm for arm, info in per_arm.items()
        if info.get("status") == "carrier_window" and not info.get("is_control_arm", False)
    ]
    confirmation_by_arm = {
        arm: BV.confirmation_gate(long_rows, dt2_rows, arm=arm)
        for arm in window_arms
    }
    passed_confirmations = [
        v for v in confirmation_by_arm.values() if v.get("status") == "passed"
    ]
    if passed_confirmations:
        confirmation = passed_confirmations[0]
    elif confirmation_by_arm:
        # Missing evidence must dominate an explicit failure in another arm:
        # one unconfirmed candidate cannot be silently discarded to promote or
        # reject a different one.
        pending = [v for v in confirmation_by_arm.values() if v.get("status") == "pending"]
        confirmation = pending[0] if pending else next(iter(confirmation_by_arm.values()))
    else:
        confirmation = None

    result = BV.adjudicate(
        state_inventory_ok=bool(inv.get("audit", {}).get("status") == "ok"),
        exact_resume_ok=bool(gates.get("passed")),
        eligible_seeds=eligible, cells=cells, per_arm=per_arm,
        neighbourhood=neighbourhood,
        reference_lock=ref_lock, smallest_subsystem=smallest, coverage=coverage,
        confirmation=confirmation)
    result = BV.apply_observation_status(result, ref_lock)

    out = dict(
        version=BV.VERDICT_VERSION, timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        git_sha=subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT, capture_output=True,
                               text=True).stdout.strip(),
        verdict=result["verdict"], layers=result["layers"], reason=result["reason"],
        observation_layer_blocked=result["observation_layer_blocked"],
        actuator_authorized=result["actuator_authorized"],
        reference_artifacts=("locked" if ref_lock else
                             ("blocked" if ref_blocked else "not_attempted")),
        gates=gates, state_inventory=inv.get("audit"), engine_guard_change=guard,
        anchors=anchors, eligible_seeds=eligible,
        per_arm=per_arm, smallest_positive_subsystem=smallest,
        coverage=coverage, cells=BV.summarize_cells(cells) if cells else [],
        neighbourhood=neighbourhood, confirmation=confirmation,
        confirmation_by_arm=confirmation_by_arm,
        n_long_confirmation_rows=len(long_rows), n_dt2_confirmation_rows=len(dt2_rows),
        conditional_phases_not_run=[
            "Task 9A slow-coordinate functional rank", "Task 9B modal/operator audit",
            "Task 10 Z-entry probability boundary", "Task 11 existing-coordinate offset boundary",
            "Task 12 matched offline exit-driver selection"]
        if result["verdict"] != "carrier_at_visited_states" else [],
        n_fork_rows=len(rows), n_neighbourhood_rows=len(nb_rows))
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "branch_verdict.json.tmp")
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, os.path.join(OUT, "branch_verdict.json"))
    print(f"\n=== VERDICT: {out['verdict']} ===\n{out['reason']}")
    print(f"layers: {json.dumps(out['layers'], indent=2)}")
    print(f"eligible seeds: {eligible}; fork rows: {len(rows)}; nbh rows: {len(nb_rows)}")
    if coverage:
        print(f"coverage: {coverage['n_cells_planned_run']}/"
              f"{coverage['n_cells_planned']} planned cells "
              f"({coverage['n_cells_extra']} extra; {coverage['n_not_run']} planned not run)")
    for arm, v in sorted(per_arm.items()):
        print(f"  {arm:16s} {v['status']:28s} positives={v['positive_cells']} "
              f"seeds={v['seeds']} phases={v['phases']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
