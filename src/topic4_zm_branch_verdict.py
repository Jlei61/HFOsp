"""Fail-closed branch adjudicator (spec rev3.1 §13, plan Task 13).

A PURE function of already-written manifests. It never runs a simulation, never re-derives a metric
and never relaxes a threshold: if a required piece of evidence is missing, ineligible, or only exists
in the smoke namespace, the verdict degrades to a `blocked_*` / `insufficient_*` / `no_evidence`
label. Unknown never becomes Branch F, and a source-only carrier never authorizes an actuator.

Top-level verdict is exactly one of `VERDICTS`.
"""
from __future__ import annotations

import numpy as np

VERDICT_VERSION = "zm_branch_verdict_v1.4_2026-07-28_offset_routing"

VERDICTS = (
    "blocked_state_inventory",
    "blocked_exact_resume",
    "blocked_reference_artifacts",
    "insufficient_bounded_anchors",
    "representation_sensitive_no_branch",
    "carrier_at_visited_states",
    "branch_T_slow_trajectory_repair",
    "branch_F_fast_carrier_repair",
    "branch_M_calibration",
    "existing_M_lifecycle_candidate",
    "phase3_driver_selection_required",
    "observation_layer_blocked",
    "no_evidence",
)

ADJACENT_BINS = (("bounded_early", "bounded_mid"), ("bounded_mid", "bounded_late"))
PRIMARY_SEEDS = (1, 3, 4)
POSITIVE = ("stable_carrier", "metastable_carrier")
CENTRAL_CONFIRMATION_STATE = ("bounded_mid", "peak")
LONG_CONFIRMATION_MS = 20_000.0
DT2_CONFIRMATION_MS = 8_000.0
OFFSET_REACHED = {
    "M_sufficient_and_reached",
    "M_SG_joint_offset_reached",
    "M_Z_recovery_offset_reached",
}
OFFSET_PHASE3 = {
    "M_boundary_far_unreached",
    "M_is_carrier_component",
    "M_shapes_but_no_offset_surface",
}


def _group(rows, keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def classify_matrix(rows, ied_lifetime_by_seed, classifier):
    """(seed, bin, phase, arm) -> class, from the paired-noise replicates of that cell."""
    cells = {}
    for key, rs in _group(rows, ("seed", "bin_name", "fast_phase", "arm")).items():
        paired = [r for r in rs if r["replicate"] != "mean_input_only"]
        ied = ied_lifetime_by_seed.get(key[0], float("nan"))
        cells[key] = dict(classifier(paired, ied), n_replicates=len(paired), arm=key[3],
                          seed=key[0], bin_name=key[1], fast_phase=key[2],
                          is_control_arm=bool(rs[0].get("is_control_arm")))
    return cells


def carrier_window(cells, min_seeds=2, min_phases=2):
    """The §6.3 carrier-window contract, applied per arm.

    Requires a positive class with: compatible support in two ADJACENT slow bins, convergence across
    at least two natural fast phases, and confirmation in at least two of three eligible primary
    seeds. A single positive seed is an `isolated_carrier_candidate`, never a window.
    """
    per_arm = {}
    arms = sorted({k[3] for k in cells})
    for arm in arms:
        pos = [k for k, v in cells.items() if k[3] == arm and v["klass"] in POSITIVE]
        seeds = sorted({k[0] for k in pos})
        phases = sorted({k[2] for k in pos})
        bins = {k[1] for k in pos}
        is_control = any(v.get("is_control_arm", False)
                         for k, v in cells.items() if k[3] == arm)

        # "Compatible support" cannot be assembled from disjoint lucky cells
        # (for example seed 1 early/trough plus seed 3 mid/peak).  A formal
        # window requires the SAME adjacent-bin pair and the SAME two natural
        # phases to be positive in at least min_seeds primary seeds.
        witnesses = []
        for b0, b1 in ADJACENT_BINS:
            for i, p0 in enumerate(phases):
                for p1 in phases[i + 1:]:
                    supporting = []
                    for seed in seeds:
                        required = ((seed, b0, p0, arm), (seed, b0, p1, arm),
                                    (seed, b1, p0, arm), (seed, b1, p1, arm))
                        if all(k in cells and cells[k]["klass"] in POSITIVE for k in required):
                            supporting.append(seed)
                    if len(supporting) >= min_seeds:
                        witnesses.append(dict(bins=[b0, b1], phases=[p0, p1],
                                              seeds=supporting))
        adjacent = bool(witnesses)
        ok = bool(witnesses) and not is_control
        status = "carrier_window" if ok else (
            "control_window" if witnesses and is_control else
            "isolated_carrier_candidate" if pos else "no_carrier")
        per_arm[arm] = dict(status=status, positive_cells=len(pos), seeds=seeds, phases=phases,
                            bins=sorted(bins), adjacent_bins=bool(adjacent),
                            is_control_arm=bool(is_control), compatible_witnesses=witnesses,
                            unmet=[] if ok else [
                                r for r, c in (("control arm cannot establish a carrier",
                                               not is_control),
                                              ("needs>=2 seeds", len(seeds) >= min_seeds),
                                              ("needs>=2 fast phases", len(phases) >= min_phases),
                                              ("needs compatible support in the same adjacent bins "
                                               "and phases", bool(witnesses)))
                                if not c])
    return per_arm


def coverage_report(cells, planned):
    """What was actually run vs what the contract asks for -- never left implicit."""
    got = {(k[0], k[1], k[2], k[3]) for k in cells}
    planned_cells = {
        (s, b, p, a)
        for s in planned.get("seeds", [])
        for b in planned.get("bins", [])
        for p in planned.get("phases", [])
        for a in planned.get("arms", [])
    }
    seeds = sorted({k[0] for k in got})
    bins = sorted({k[1] for k in got})
    phases = sorted({k[2] for k in got})
    arms = sorted({k[3] for k in got})
    missing = []
    for s in planned.get("seeds", []):
        for b in planned.get("bins", []):
            for p in planned.get("phases", []):
                for a in planned.get("arms", []):
                    if (s, b, p, a) not in got:
                        missing.append(dict(seed=s, bin_name=b, fast_phase=p, arm=a))
    planned_run = got & planned_cells
    extras = got - planned_cells
    return dict(seeds_run=seeds, bins_run=bins, phases_run=phases, arms_run=arms,
                n_cells_run=len(got), n_cells_planned_run=len(planned_run),
                n_cells_extra=len(extras),
                extra_cells=[dict(seed=s, bin_name=b, fast_phase=p, arm=a)
                             for s, b, p, a in sorted(extras)[:200]],
                n_cells_planned=len(planned_cells),
                not_run=missing[:200], n_not_run=len(missing))


def confirmation_gate(long_rows, dt2_rows, *, arm, min_seeds=2):
    """Fail-closed §6.2 confirmation of a discovery carrier window.

    The central homologous state is locked as ``bounded_mid__peak``.  The
    discovery posterior already carries the three paired-future-noise
    replicates; the confirmation asks a separate question and therefore uses
    the native ``noise_replay`` continuation in each supporting seed:

    * the original-dt state must remain stationary for at least 20 s;
    * a newly selected state from an independently rerun dt/2 anchor must
      remain stationary for at least the primary 8 s horizon.

    Rows with the wrong resolution, horizon, tier, state, arm, or replicate are
    not evidence.  Missing evidence is ``pending``; two fully observed seeds
    with fewer than ``min_seeds`` passes are an explicit ``failed`` result.
    """
    b, p = CENTRAL_CONFIRMATION_STATE

    def eligible(rows, *, resolution, tier, min_ms):
        out = {}
        for r in rows or []:
            if (
                r.get("bin_name") == b
                and r.get("fast_phase") == p
                and r.get("arm") == arm
                and r.get("replicate") == "noise_replay"
                and r.get("resolution") == resolution
                and r.get("evidence_tier") == tier
                and float(r.get("T_cont_ms", 0.0)) >= float(min_ms)
            ):
                out[int(r["seed"])] = r
        return out

    long_by_seed = eligible(
        long_rows, resolution="dt", tier="long_confirmation",
        min_ms=LONG_CONFIRMATION_MS,
    )
    dt2_by_seed = eligible(
        dt2_rows, resolution="dt2", tier="dt2_confirmation",
        min_ms=DT2_CONFIRMATION_MS,
    )
    complete = sorted(set(long_by_seed) & set(dt2_by_seed))

    def passed(r):
        return bool(r.get("survived")) and bool(r.get("stationarity_ok"))

    seeds_passed = [
        s for s in complete
        if passed(long_by_seed[s]) and passed(dt2_by_seed[s])
    ]
    failed = [s for s in complete if s not in seeds_passed]
    if len(seeds_passed) >= int(min_seeds):
        status = "passed"
    elif len(complete) >= int(min_seeds):
        status = "failed"
    else:
        status = "pending"
    return dict(
        status=status,
        arm=arm,
        central_state=f"{b}__{p}",
        min_seeds=int(min_seeds),
        long_horizon_ms=LONG_CONFIRMATION_MS,
        dt2_horizon_ms=DT2_CONFIRMATION_MS,
        long_horizon_seeds=sorted(long_by_seed),
        dt2_seeds=sorted(dt2_by_seed),
        complete_seeds=complete,
        seeds_passed=seeds_passed,
        failed_seeds=failed,
    )


def adjudicate(*, state_inventory_ok, exact_resume_ok, eligible_seeds, cells, per_arm,
               neighbourhood=None, reference_lock=None, smallest_subsystem=None,
               offset_result=None, coverage=None, confirmation=None):
    """The single fail-closed decision (spec §13). Returns the verdict plus every layer's status."""
    layers = dict(
        state_gate="ok" if state_inventory_ok else "blocked",
        exact_resume="ok" if exact_resume_ok else "blocked",
        anchors=dict(eligible_seeds=sorted(set(eligible_seeds)),
                     n_eligible=len(set(eligible_seeds))),
        source_space_carrier="not_established",
        observation_space_carrier="not_established",
        carrier_confirmation="not_established",
        entry="not_established",
        offset="not_established",
        recovery_lifecycle="not_established",
    )
    obs_blocked = not (reference_lock or {}).get("sufficient_reference_sample", False)
    layers["observation_space_carrier"] = ("blocked_reference_artifacts" if obs_blocked
                                           else "not_established")

    if not state_inventory_ok:
        return dict(verdict="blocked_state_inventory", layers=layers, reason="unclassified state")
    if not exact_resume_ok:
        return dict(verdict="blocked_exact_resume", layers=layers,
                    reason="split/resume parity not proven")

    windows = {a: v for a, v in per_arm.items()
               if v["status"] == "carrier_window" and not v.get("is_control_arm", False)}
    isolated = {a: v for a, v in per_arm.items() if v["status"] == "isolated_carrier_candidate"}
    n_elig = len(set(eligible_seeds))

    if windows:
        confirm_status = (confirmation or {}).get("status", "pending")
        layers["carrier_confirmation"] = confirm_status
        if confirm_status != "passed":
            layers["source_space_carrier"] = "provisional_carrier_window"
            return dict(
                verdict="no_evidence",
                layers=layers,
                carrier_arms=sorted(windows),
                smallest_subsystem=smallest_subsystem,
                isolated=sorted(isolated),
                coverage=coverage,
                confirmation=confirmation,
                reason=(
                    "an 8 s discovery carrier window exists, but the mandatory "
                    f"20 s plus native-dt/2 confirmation is {confirm_status}"
                ),
            )
        layers["source_space_carrier"] = "carrier_window"
        layers["carrier_confirmation"] = "passed"
        verdict = "carrier_at_visited_states"
        if offset_result:
            verdict = offset_result.get("verdict", verdict)
        return dict(verdict=verdict, layers=layers, carrier_arms=sorted(windows),
                    smallest_subsystem=smallest_subsystem, isolated=sorted(isolated),
                    coverage=coverage,
                    reason="a positive carrier window survived the replication contract")

    layers["source_space_carrier"] = ("isolated_carrier_candidate" if isolated
                                      else "no_carrier_in_completed_cells")

    if coverage and int(coverage.get("n_not_run", 0)) > 0:
        layers["source_space_carrier"] = "partial_no_carrier_evidence"
        return dict(
            verdict="no_evidence", layers=layers, coverage=coverage,
            isolated=sorted(isolated),
            reason=(
                "the declared discovery fork ladder is incomplete "
                f"({coverage.get('n_cells_planned_run', 0)}/"
                f"{coverage.get('n_cells_planned', 0)} planned cells; "
                f"{coverage.get('n_cells_extra', 0)} extra cells); "
                "partial negatives cannot be promoted to a visited-state verdict"
            ),
        )

    spec_cov = (coverage or {}).get("spec_full_matrix") or {}
    if int(spec_cov.get("n_not_run", 0)) > 0 and not isolated:
        layers["source_space_carrier"] = "no_carrier_in_completed_discovery_ladder"

    if n_elig < 3:
        return dict(verdict="insufficient_bounded_anchors", layers=layers, coverage=coverage,
                    isolated=sorted(isolated),
                    reason=f"a formal negative needs three eligible bounded anchors; have {n_elig}")

    if neighbourhood is None:
        return dict(verdict="no_evidence", layers=layers, coverage=coverage,
                    isolated=sorted(isolated),
                    reason="visited states show no carrier, but the local slow-state neighbourhood "
                           "audit that separates Branch T from Branch F was not run")

    nb = neighbourhood
    if nb["verdict"] == "representation_sensitive_no_branch":
        return dict(verdict="representation_sensitive_no_branch", layers=layers,
                    coverage=coverage, reason=nb["reason"])
    if nb["verdict"] in ("branch_T_slow_trajectory_repair", "branch_F_fast_carrier_repair"):
        return dict(verdict=nb["verdict"], layers=layers, coverage=coverage, reason=nb["reason"])
    return dict(verdict="no_evidence", layers=layers, coverage=coverage, reason=nb["reason"])


def apply_observation_status(result, reference_lock):
    """A source-only result keeps `observation_layer_blocked` visible at the top level and can never
    authorize an actuator (spec §4.5 / §11).

    Task 12 is an offline driver-selection step.  Even with a complete
    observation reference, ``phase3_driver_selection_required`` means that
    selection has not happened yet; if one driver later passes, the locked
    plan still requires a new actuator spec.  Therefore this branch-decision
    workflow never directly authorizes implementation.
    """
    blocked = not (reference_lock or {}).get("sufficient_reference_sample", False)
    result = dict(result)
    result["observation_layer_blocked"] = bool(blocked)
    result["actuator_authorized"] = False
    result["actuator_authorization_reason"] = (
        "requires a passing offline driver selection and a separately approved "
        "actuator specification"
    )
    return result


def apply_offset_status(result, offset_summary):
    """Route a completed Phase-2B result without inventing lifecycle recovery.

    A reached carrier-exit boundary remains an offset-layer result because
    returning interictal events have not been demonstrated.  A narrowly
    unreached boundary routes to the pre-registered M-calibration branch.
    Phase 3 is authorized only for the explicit no-usable-offset outcomes.
    Ambiguous/static-only outcomes remain at the carrier verdict.
    """

    out = dict(result)
    out["layers"] = dict(result.get("layers") or {})
    if (
        out.get("verdict") != "carrier_at_visited_states"
        or not offset_summary
        or int(offset_summary.get("n_complete_seeds", 0)) < 2
    ):
        return out
    offset_verdict = offset_summary.get("verdict")
    if not offset_verdict or offset_verdict == "no_evidence":
        out["layers"]["offset"] = "existing_coordinate_offset_unresolved"
        return out

    out["layers"]["offset"] = offset_verdict
    if offset_verdict == "M_boundary_near_but_unreached":
        out["verdict"] = "branch_M_calibration"
        out["reason"] = (
            "a registered existing-coordinate offset boundary is narrowly "
            "outside the observed range; only a later M-calibration spec is allowed"
        )
    elif offset_verdict in OFFSET_PHASE3:
        out["verdict"] = "phase3_driver_selection_required"
        out["reason"] = (
            "a confirmed carrier exists, but every valid existing slow-coordinate "
            "family lacks a usable offset; matched offline driver selection is required"
        )
    elif offset_verdict in OFFSET_REACHED:
        out["reason"] = (
            "a confirmed carrier and an existing-coordinate offset boundary are "
            "present, but returning-event recovery and a lifecycle are not established"
        )
    else:
        out["reason"] = (
            "the existing-coordinate offset audit is incomplete or dynamically "
            "unreached; no calibration, Phase-3, or lifecycle branch is authorized"
        )
    return out


def summarize_cells(cells):
    """Flat, sortable view for the archive table."""
    out = []
    for k, v in sorted(cells.items()):
        post = v.get("posterior") or {}
        out.append(dict(seed=k[0], bin_name=k[1], fast_phase=k[2], arm=k[3], klass=v["klass"],
                        k=post.get("k"), n=post.get("n"),
                        p_median=None if not post else round(post["median"], 3),
                        p_lo=None if not post else round(post["lo"], 3),
                        p_hi=None if not post else round(post["hi"], 3),
                        is_control_arm=v["is_control_arm"]))
    return out


def median_lifetime(rows, **filt):
    sel = [r["lifetime_ms"] for r in rows
           if all(r.get(k) == v for k, v in filt.items())]
    return float(np.median(sel)) if sel else float("nan")
