#!/usr/bin/env python3
"""Finalize FCXR-LC2-GX1 frozen entry/offset diagnostics."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "topic4_sef_hfo", "fcxr_lc2_core",
                   "gx1_entry_offset_diagnostics")
ARCHIVE = os.path.join(ROOT, "docs", "archive", "topic4", "sef_hfo",
                       "fcxr_lc2_gx1_entry_offset_diagnostics_2026-08-02.md")
POST_REVIEW_NEXT_PROGRAM = "LC3_DX_STATE_PLANE_AND_SPATIAL_INSTABILITY_AUDIT"
MECHANISM_MAP_LABELS = [
    "FINITE_H_HIGH_STATE_POSITIVE",
    "D_SELECTIVE_ONSET_CANDIDATE",
    "SAME_D_BISTABILITY_NOT_FOUND",
    "X_OFFSET_PATH_REACHABLE",
    "X_FIXED_D_DYNAMIC_RANGE_INSUFFICIENT",
    "COUPLED_D_X_OFFSET_UNTESTED",
    "DYNAMIC_LIFECYCLE_UNTESTED",
    "SPATIAL_INSTABILITY_UNTESTED",
]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def _load_path(path):
    with open(path) as f:
        return json.load(f)


def _locked_artifact(lock, key):
    return _load_path(lock["artifacts"][key]["path"])


def _write_json(name, payload):
    path = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def _write_text(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(text.rstrip() + "\n")
    os.replace(tmp, path)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _executable_contract_sha256():
    """Same two-file contract hash the runner locks; recorded so post-run code fixes stay visible."""
    h = hashlib.sha256()
    for rel in ("scripts/run_topic4_fcxr_lc2_gx1.py", "scripts/run_topic4_fcxr_lc2_forks.py"):
        with open(os.path.join(ROOT, rel), "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def choose_next_hypothesis(strip_verdict, x_verdict):
    natural = strip_verdict == "NATURAL_SELECTIVITY_WINDOW_CANDIDATE"
    reachable = x_verdict in (
        "X_PATH_REACHABLE_RANGE_INSUFFICIENT",
        "X_OFFSET_ALREADY_REACHABLE_IN_CURRENT_PATH",
    )
    bypass = x_verdict == "H_ACTUATOR_BYPASSES_X_AT_MAXIMAL_SHUTDOWN"
    if natural and reachable:
        return "KEEP_H_EQUATION_CALIBRATE_X_RANGE"
    if natural and bypass:
        return "SHARED_PATH_X_H_COUPLING_ONLY"
    if strip_verdict == "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP" and reachable:
        return "LOCAL_D_DEPENDENT_H_GAIN_ONLY_X_RANGE_SEPARATE"
    if strip_verdict == "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP" and bypass:
        return "CAUSAL_2X2_D_GATE_BY_SHARED_X_H_PATH"
    return "MEASUREMENT_REPAIR_NO_STRUCTURAL_CLAIM"


def summarize_entry_geometry(strip):
    """Describe the strongest locked-strip component without upgrading it to a basin."""
    points = list(strip.get("point_rows", []))
    selective = []
    for point in points:
        arms = {a["arm"]: a for a in point.get("arms", []) if "arm" in a}
        if set(arms) != {"healthy_low", "susceptible_low", "susceptible_high"}:
            continue
        healthy_low = arms["healthy_low"].get("workpoint_label") == "INTERICTAL_WORKPOINT"
        susceptible_low_high = arms["susceptible_low"].get("workpoint_label") in (
            "FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
        susceptible_high_high = arms["susceptible_high"].get("workpoint_label") in (
            "FINITE_HIGH_FIXED", "FINITE_HIGH_ORBIT")
        if healthy_low and susceptible_low_high and susceptible_high_high:
            selective.append(point["point_id"])
    return dict(
        component_label=("D_SELECTIVE_ONE_WAY_IGNITION_WITHOUT_DUAL_BASIN"
                         if selective else "NO_D_SELECTIVE_IGNITION_COMPONENT"),
        selective_one_way_points=selective,
        natural_dual_basin_window=bool(strip.get("n_window_points", 0)),
        explicit_d_gate_status=(
            "AUTHORIZED_AS_FALSIFIABLE_HYPOTHESIS_NOT_PROVEN_SUFFICIENT"
            if strip.get("verdict") == "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP"
            else "NOT_AUTHORIZED"),
    )


def archived_relay_loads(forks_map, point_id):
    """Frozen-relay arms already run at THIS anchor in the archived LC2 E4 fork map.

    These are the loads the spec calls the observed physiological range.  They are read from the archive
    rather than restated as constants, so the range verdict cannot drift away from its evidence.
    """
    loads = []
    for row in forks_map.get("rows", []):
        if row.get("candidate_run_id") != point_id:
            continue
        if float(row.get("x_availability", 1.0)) >= 1.0:
            continue
        label = row.get("required_low_workpoint_label")
        loads.append(dict(x_availability=float(row["x_availability"]), arm=row.get("arm"),
                          required_low_workpoint_label=label,
                          returned_to_interictal=label == "INTERICTAL_WORKPOINT",
                          tail_rate_hz=float(row["state_tail_1s"]["rate_mean_hz"])))
    return sorted(loads, key=lambda d: -d["x_availability"])


def summarize_x_authority(xmap, archived_loads):
    rows = sorted((r for r in xmap.get("rows", []) if "x_availability" in r),
                  key=lambda r: float(r["x_availability"]), reverse=True)
    returning = sorted(float(v) for v in xmap.get("returning_availabilities", []))
    nonreturning = sorted(
        float(r["x_availability"]) for r in rows
        if r.get("required_low_workpoint_label") != "INTERICTAL_WORKPOINT")
    smallest_nonreturning_above = None
    if returning:
        ret_hi = max(returning)
        above = [v for v in nonreturning if v > ret_hi]
        smallest_nonreturning_above = min(above) if above else None
    archived_returning = [a for a in archived_loads if a["returned_to_interictal"]]
    if not archived_loads:
        archived_status = "NO_ARCHIVED_LOAD_AT_THIS_ANCHOR"
    elif archived_returning:
        archived_status = "SUFFICIENT_FOR_THIS_H_BRANCH"
    else:
        archived_status = "INSUFFICIENT_FOR_THIS_H_BRANCH"
    return dict(
        current_x_path_reachable=bool(returning),
        h_actuator_bypasses_x=False if returning else None,
        largest_tested_returning_availability=max(returning) if returning else None,
        smallest_tested_nonreturning_availability_above_return=(
            smallest_nonreturning_above),
        experimental_return_bracket=(
            [max(returning), smallest_nonreturning_above]
            if returning and smallest_nonreturning_above is not None else None),
        archived_relay_loads_at_this_anchor=archived_loads,
        archived_relay_load_source="LC2 E4 frozen_fork_map.json, same candidate_run_id",
        archived_range_status=archived_status,
        physiological_validity_of_returning_probe="NOT_ESTABLISHED",
    )


def summarize_strip_resolution(strip):
    """How many independent conditions the locked strip actually resolves.

    The actuator is `rho * sigmoid((h-theta)/k)`.  Where that gate is pinned at 1 for the whole run the
    actuator degenerates to a constant additive `rho`, and the trajectory carries no information about
    `tau_H` or `theta_H`.  Reporting 12 parameter points without this is an overstatement of coverage.
    """
    out = {}
    for arm in ("healthy_low", "susceptible_low", "susceptible_high"):
        cells = [a for p in strip.get("point_rows", []) for a in p.get("arms", [])
                 if a.get("arm") == arm]
        rates = {round(float(c["state_tail_1s"]["rate_mean_hz"]), 9) for c in cells}
        gates = [min(float(g) / float(c["rho"]) for g in c["gH_trace"])
                 for c in cells if float(c.get("rho", 0.0)) > 0.0 and c.get("gH_trace")]
        gate_min = min(gates) if gates else None
        out[arm] = dict(
            n_points=len(cells), n_distinct_tail_rates=len(rates),
            min_gate_over_run=gate_min,
            gate_pinned_open_whole_run=bool(gate_min is not None and gate_min > 1.0 - 1e-6),
        )
    degenerate = [k for k, v in out.items() if v["gate_pinned_open_whole_run"]]
    return dict(per_arm=out, gate_pinned_arms=degenerate,
                arms_that_resolve_tau_and_theta=[k for k in out if k not in degenerate])


def summarize_x_initial_condition(xmap):
    """The offset arms start from an analytic H head start, not from the converged high branch."""
    ref = next((r for r in xmap.get("rows", [])
                if "x_availability" in r and float(r["x_availability"]) == 1.0 and r.get("h_trace")),
               None)
    if ref is None:
        return dict(status="NO_FULL_AVAILABILITY_REFERENCE")
    theta, tau = float(ref["theta"]), float(ref["tau_ms"])
    h_init = float(ref["h_init_scale"]) * theta
    h_converged = max(float(v) for v in ref["h_trace"])
    extra_s = tau * math.log(h_converged / h_init) / 1000.0 if h_converged > h_init else 0.0
    return dict(
        h_initial=h_init, h_converged_at_full_availability=h_converged,
        head_start_ratio=h_converged / h_init,
        extra_above_theta_decay_s_if_started_converged=extra_s,
        run_duration_s=float(ref["T_ms"]) / 1000.0,
        required_low_window_s=float(ref["post_offset_required_ms"]) / 1000.0,
        offset_tested_against="analytic 2*theta head start, not the converged high branch",
        margin_ok=bool(float(ref["T_ms"]) / 1000.0 - extra_s
                       > float(ref["post_offset_required_ms"]) / 1000.0),
    )


def build_candidate_verdict(strip, xmap, archived_loads=()):
    rows = [a for p in strip["point_rows"] for a in p["arms"]] + list(xmap["rows"])
    safe = sum(not bool(r.get("numerical_failure", True)) for r in rows)
    preregistered_next = choose_next_hypothesis(strip["verdict"], xmap["verdict"])
    entry = summarize_entry_geometry(strip)
    entry["lifecycle_interpretation"] = "D_SELECTIVE_MONOSTABLE_ONSET_CANDIDATE"
    entry["same_D_bistability_required_for_lifecycle"] = False
    entry["explicit_d_gate_status"] = "DEFERRED_PENDING_LC3_CURRENT_EQUATION_AUDIT"
    xauth = summarize_x_authority(xmap, list(archived_loads))
    xauth["fixed_D_dynamic_range_status"] = "INSUFFICIENT"
    xauth["coupled_D_X_offset_status"] = "UNTESTED"
    return dict(
        status="COMPLETE",
        scientific_tier="frozen_component_diagnostic",
        canonical_verdict="GX1_MECHANISM_MAP_ACCEPTED",
        mechanism_map_labels=list(MECHANISM_MAP_LABELS),
        selectivity_strip_verdict=strip["verdict"],
        x_authority_verdict=xmap["verdict"],
        preregistered_next_hypothesis=preregistered_next,
        authorized_next_program=POST_REVIEW_NEXT_PROGRAM,
        strategy_adjudication=(
            "The GX1 same-D dual-basin gate was a local frozen-geometry diagnostic, not a necessary "
            "condition for a seizure lifecycle. Test the current H equation in the coupled D-X plane "
            "and audit the spatial instability before adding an explicit D gate."),
        n_strip_rows=int(strip["n_rows"]),
        n_strip_points=int(strip["n_points"]),
        n_strip_pass=int(strip["n_pass"]),
        n_strip_window_points=int(strip["n_window_points"]),
        n_x_rows=int(xmap["n_rows"]),
        numerical_safe_rows=int(safe),
        numerical_total_rows=len(rows),
        entry_geometry=entry,
        strip_resolution=summarize_strip_resolution(strip),
        x_authority=xauth,
        x_initial_condition=summarize_x_initial_condition(xmap),
        dynamic_lifecycle_tested=False,
        spatial_instability_tested=False,
        morphology_tested=False,
        forbidden_claims=[
            "spontaneous seizure lifecycle",
            "bistability or hysteresis from the frozen strip alone",
            "physiological validity of x=0 or x=0.1",
            "patient-like ictal morphology",
            "12 independent susceptible-high conditions: the H gate is pinned open in that arm, so it "
            "resolves rho only and is blind to tau_H and theta_H",
            "removal of a converged high branch: the offset arms start from an analytic 2*theta head "
            "start",
            "same-D bistability is necessary for a seizure lifecycle",
            "the current dynamic X range is insufficient in the coupled D-X system: only fixed-D was tested",
            "the D-selective transition is axial or local: spatial instability was not tested",
        ],
        finalized_at=_now(),
    )


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "savefig.dpi": 220,
    })


def plot_strip(strip, path):
    _style()
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.7), constrained_layout=True)
    families = ("H1", "H6")
    rhos = (0.025, 0.05, 0.075)
    thetas = (1.0, 1.25)
    labels = {"INTERICTAL_WORKPOINT": 0, "ELEVATED_EVENT_TRAIN": 1,
              "FINITE_HIGH_FIXED": 2, "FINITE_HIGH_ORBIT": 2}
    cmap = ListedColormap(["#4c78a8", "#f2cf5b", "#d1495b", "#777777"])
    for iy, family in enumerate(families):
        pts = {p["point_id"]: p for p in strip["point_rows"] if p["family"] == family}
        for ix, arm in enumerate(("healthy_low", "susceptible_low", "susceptible_high")):
            z = np.full((len(thetas), len(rhos)), 3.0)
            rates = np.full_like(z, np.nan)
            for ti, ts in enumerate(thetas):
                for ri, rho in enumerate(rhos):
                    p = next((q for q in pts.values()
                              if q["theta_scale"] == ts and q["rho_fraction"] == rho), None)
                    if p is None:
                        continue
                    a = next(q for q in p["arms"] if q["arm"] == arm)
                    z[ti, ri] = labels.get(a.get("workpoint_label"), 3)
                    rates[ti, ri] = float(a["state_tail_1s"]["rate_mean_hz"])
            ax = axes[iy, ix]
            ax.imshow(z, vmin=-0.5, vmax=3.5, cmap=cmap, aspect="auto")
            for ti in range(len(thetas)):
                for ri in range(len(rhos)):
                    if np.isfinite(rates[ti, ri]):
                        ax.text(ri, ti, f"{rates[ti, ri]:.1f}", ha="center", va="center",
                                color="white" if z[ti, ri] in (0, 2, 3) else "#222222",
                                fontsize=8, fontweight="bold")
            ax.set_xticks(range(len(rhos)), [f"{v:.3f}" for v in rhos])
            ax.set_yticks(range(len(thetas)), [f"{v:.2f}" for v in thetas])
            ax.set_xlabel(r"$\rho_H/g_{sat}$")
            if ix == 0:
                ax.set_ylabel(f"{family}\n" + r"$\theta$ scale")
            else:
                ax.set_ylabel(r"$\theta$ scale")
            if iy == 0:
                ax.set_title(arm.replace("_", " "))
    fig.legend(handles=[
        Patch(facecolor="#4c78a8", label="interictal workpoint"),
        Patch(facecolor="#f2cf5b", label="elevated event train"),
        Patch(facecolor="#d1495b", label="finite high state"),
        Patch(facecolor="#777777", label="unresolved"),
    ], loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=4,
       frameon=False, fontsize=8)
    headline = ("no parameter point keeps both low starts interictal while holding a high start"
                if strip["verdict"] == "NO_NATURAL_SELECTIVITY_WINDOW_IN_LOCKED_STRIP"
                else strip["verdict"])
    fig.suptitle(f"GX1 entry strip: {headline}\n"
                 "cell text = final-1 s rate (Hz); the susceptible-high column repeats per "
                 r"$\rho$ because its H gate is pinned open",
                 fontsize=11, fontweight="bold")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _rolling(x, n):
    x = np.asarray(x, float)
    if n <= 1:
        return x
    if x.size < n:
        return np.full_like(x, np.nan)
    y = np.convolve(x, np.ones(n) / n, mode="valid")
    return np.r_[np.full(n - 1, np.nan), y]


def plot_x(xmap, path, roll_hi):
    _style()
    fig, axes = plt.subplots(2, 1, figsize=(10.2, 6.4), sharex=True,
                             constrained_layout=True)
    palette = ["#4c78a8", "#59a14f", "#f28e2b", "#d1495b"]
    rows = sorted(xmap["rows"], key=lambda r: -float(r["x_availability"]))
    theta = float(rows[0]["theta"])
    for row, color in zip(rows, palette):
        x = float(row["x_availability"])
        dt = float(row["trace_dt_ms"])
        t = np.arange(len(row["rate_trace"])) * dt / 1000.0
        axes[0].plot(t, _rolling(row["rate_trace"], max(1, int(round(300.0 / dt)))),
                     lw=1.4, color=color, label=f"relay availability = {x:g}")
        axes[1].plot(t, row["h_trace"], lw=1.4, color=color)
    # The decision rule is "rolling rate above the accepted interictal band", not an arbitrary 20 Hz.
    axes[0].axhline(roll_hi, color="#555555", ls="--", lw=0.9,
                    label=f"interictal band upper edge = {roll_hi:.2f} Hz")
    axes[1].axhline(theta, color="#555555", ls="--", lw=0.9,
                    label=r"H gate midpoint $\theta$ = " + f"{theta:.2f}")
    axes[0].set_ylabel("300-ms rolling rate (Hz)")
    axes[0].legend(frameon=False, ncol=2, fontsize=8)
    axes[1].set_ylabel("mean H (recurrent-conductance units)")
    axes[1].set_xlabel("time (s)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Shutting the presynaptic relay does remove the high state,\n"
                 "but only far below the relay loads this model actually reaches",
                 fontsize=11, fontweight="bold")
    fig.text(0.5, -0.012, f"verdict: {xmap['verdict']}", ha="center", fontsize=7.5,
             color="#555555")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_logic(strip, xmap, path):
    _style()
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    ax.axis("off")
    natural = strip["verdict"] == "NATURAL_SELECTIVITY_WINDOW_CANDIDATE"
    reachable = xmap["verdict"] in ("X_PATH_REACHABLE_RANGE_INSUFFICIENT",
                                    "X_OFFSET_ALREADY_REACHABLE_IN_CURRENT_PATH")
    cells = [
        (0.05, 0.55, "Natural selectivity\n+ X path reachable",
         "Keep H; calibrate X range", natural and reachable),
        (0.52, 0.55, "Natural selectivity\n+ maximal-X bypass",
         "Shared X/H path only", natural and not reachable),
        (0.05, 0.08, "No natural window\n+ X path reachable",
         "Test local D-gate; calibrate X separately", (not natural) and reachable),
        (0.52, 0.08, "No natural window\n+ maximal-X bypass",
         "Causal 2x2: D gate x shared path", (not natural) and (not reachable)),
    ]
    for x, y, title, action, active in cells:
        ax.add_patch(plt.Rectangle((x, y), 0.43, 0.35,
                                   fc="#f7c98b" if active else "#edf0f2",
                                   ec="#333333" if active else "#aaaaaa", lw=1.6 if active else 0.8))
        ax.text(x + 0.215, y + 0.23, title, ha="center", va="center",
                fontsize=9, fontweight="bold")
        ax.text(x + 0.215, y + 0.09, action, ha="center", va="center", fontsize=8)
    ax.text(0.5, 0.98, "GX1 pre-registered routing (audit trail, superseded)", ha="center", va="top",
            fontsize=12, fontweight="bold")
    observed = ("Observed: no dual-basin window | X path reachable"
                if (not natural) and reachable else
                f"Observed: {strip['verdict']} | {xmap['verdict']}")
    ax.text(0.5, 0.035, observed, ha="center", va="bottom", fontsize=8.5)
    ax.text(0.5, -0.045,
            "Post-review: same-D dual basin is not required; next = D-X state plane + spatial audit",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1f5a7a")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _resource_summary():
    """Every engineering number in the archive is read back from the run's own logs."""
    peaks, swaps = [], []
    log = os.path.join(OUT, "resource_log.jsonl")
    if os.path.isfile(log):
        with open(log) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                peaks.append(float(rec["peak_rss_gib"]))
                swaps.append(float(rec["mem"]["swap_used_mib"]))
    watchdog, marks = {}, list(swaps)
    for stage in ("S1", "X1"):
        wpath = os.path.join(OUT, f"{stage}_WATCHDOG.json")
        if os.path.isfile(wpath):
            watchdog[stage] = float(_load_path(wpath).get("elapsed_hours", float("nan")))
        dpath = os.path.join(OUT, f"{stage}_DONE.json")
        if os.path.isfile(dpath):
            done = _load_path(dpath)
            marks += [float(done[k]["swap_used_mib"]) for k in ("resource_before", "resource_after")
                      if k in done]
    return dict(peak_rss_gib=max(peaks) if peaks else None,
                swap_delta_mib=(max(marks) - min(marks)) if marks else None,
                watchdog_hours=watchdog, n_resource_rows=len(peaks))


def _rate_range(points, arm):
    vals = sorted(float(a["state_tail_1s"]["rate_mean_hz"])
                  for p in points for a in p["arms"] if a["arm"] == arm)
    if not vals:
        return "n/a"
    return f"{vals[0]:.1f}" if len(vals) == 1 or vals[-1] - vals[0] < 0.05 else \
        f"{vals[0]:.1f}--{vals[-1]:.1f}"


def _status(verdict):
    entry = verdict["entry_geometry"]
    xauth = verdict["x_authority"]
    return f"""# FCXR-LC2-GX1 status

Status: **ACCEPTED — GX1 frozen mechanism map**

- Canonical verdict: `{verdict['canonical_verdict']}`
- S1: `{verdict['selectivity_strip_verdict']}`
- X1: `{verdict['x_authority_verdict']}`
- Pre-registered local routing result: `{verdict['preregistered_next_hypothesis']}`
- Post-review authorized next program: `{verdict['authorized_next_program']}`
- Numerical safety: {verdict['numerical_safe_rows']}/{verdict['numerical_total_rows']} rows
- Entry component: `{entry['component_label']}`
- Natural low/high dual-basin window: **no**
- Same-D bistability required for lifecycle: **no**
- X path reachable: **{str(xauth['current_x_path_reachable']).lower()}**; tested return bracket:
  `{xauth['experimental_return_bracket']}`; archived loads at this anchor:
  `{xauth['archived_range_status']}`
- Coupled D-X offset: **untested**
- Spatial instability / eigenmode: **untested**
- Strip resolution: the `{', '.join(verdict['strip_resolution']['gate_pinned_arms']) or 'none'}` arm(s)
  run with the H gate pinned open, so they resolve `rho` only
- Offset arms started from `{verdict['x_initial_condition'].get('offset_tested_against')}`
- Dynamic lifecycle: **not tested**
- M/K/A/ELR: **not used**

GX1 identifies a D-selective monostable-onset candidate and a structurally reachable X offset path.
It does not establish a spontaneous interictal-ictal-interictal lifecycle, a coupled D-X return path,
an axial onset mode, or patient-like ictal morphology.
"""


def _archive(verdict, strip, xmap):
    entry = verdict["entry_geometry"]
    xauth = verdict["x_authority"]
    res = verdict["strip_resolution"]
    xic = verdict["x_initial_condition"]
    xrows = {float(row["x_availability"]): row for row in xmap["rows"]}
    sel_ids = entry["selective_one_way_points"]
    sel = [p for p in strip["point_rows"] if p["point_id"] in sel_ids]
    sel_desc = (", ".join(sorted({f"{p['family']} theta_scale={p['theta_scale']:g}" for p in sel}))
                + f" 的 {len(sel)} 个 rho 点" if sel else "无")
    hi_arm = res["per_arm"]["susceptible_high"]
    rs = _resource_summary()
    loads = xauth["archived_relay_loads_at_this_anchor"]
    load_desc = ("、".join(f"{a['x_availability']:g}（尾段 {a['tail_rate_hz']:.1f} Hz，"
                          f"{'回到间期' if a['returned_to_interictal'] else '仍为高态'}）"
                          for a in loads) or "本 anchor 无归档 relay 负载臂")
    wd = "; ".join(f"{k} watchdog {v:.3f} h" for k, v in sorted(rs["watchdog_hours"].items()))
    return f"""# FCXR-LC2-GX1 frozen entry/offset diagnostics — 2026-08-02

## 一句话结论

GX1 在不改方程、不接动态慢变量的条件下，分别检验现有 H 方程是否自带易感性选择窗，以及 X
理论最大关断是否有权把 H 高态拉回间期。正式结果是：

- S1：`{strip['verdict']}`（{strip['n_pass']}/{strip['n_points']} 点通过，
  {strip['n_window_points']} 个点属于相邻窗）；
- X1：`{xmap['verdict']}`；
- 原预注册局部路由：`{verdict['preregistered_next_hypothesis']}`；
- 经核心科学目标复审后，下一获准程序：`{verdict['authorized_next_program']}`。

这不是一个笼统的双阴性。S1 在 {sel_desc}上都看到了同一分解：
健康 `D=0/H_low` 保持 {_rate_range(sel, 'healthy_low')} Hz 的间期工作点，而易感 `D=0.15/H_low` 已经升到
{_rate_range(sel, 'susceptible_low')} Hz；易感高初值也维持 {_rate_range(sel, 'susceptible_high')} Hz。也就是说，现有方程已经出现
**D 选择性的单向点火**，但同一个易感 D 下低初值和高初值都落到高态，因此不是低/高双盆地。
这不等于 onset 机制失败：合法的 seizure onset 可以来自单稳态分支交换、Hopf、SNIC、
noise-assisted transition 或 slow-wave bursting，并不要求同一点迟滞。当前 H1 因而被验收为
`D_SELECTIVE_ONSET_CANDIDATE`，尚待 D-X 状态平面和空间稳定性检验。

X1 则给出清楚的权限括号：availability=1.0/0.5 仍为高态（尾段
{xrows[1.0]['state_tail_1s']['rate_mean_hz']:.1f}/{xrows[0.5]['state_tail_1s']['rate_mean_hz']:.1f} Hz），
0.1/0.0 在末段连续 {xrows[0.0]['post_offset_required_ms'] / 1000.0:.0f} s 回到间期（尾段
{xrows[0.1]['state_tail_1s']['rate_mean_hz']:.3f}/{xrows[0.0]['state_tail_1s']['rate_mean_hz']:.3f} Hz）。
所以 H 没有结构性绕过 X。同一 anchor 上归档的 LC2 E4 frozen-relay 负载臂为
{load_desc}，状态 `{xauth['archived_range_status']}`——即当前 relay 实际达到的负载区间不足以终止这条 H 分支，
能终止的 {xauth['largest_tested_returning_availability']:g} 远在其之下，且不具有生理标定资格。
但该结论只在固定病理 `D=0.15` 下成立；完整系统可能通过 `X↑ -> rate↓ -> Z恢复 -> D↓`
共同跨过 offset surface，因此 `COUPLED_D_X_OFFSET_UNTESTED`，目前不能直接判定必须增强 X。

## 本轮结论覆盖不到的两处

1. **易感高初值臂不检验 H 的时间常数和阈值。** 该臂从 `H(0)=2*theta` 出发，门
   `sigmoid((h-theta)/k)` 在整段运行里被钉在 1（最小值
   {hi_arm['min_gate_over_run']:.9f}），于是 `rho*S_H(h)` 退化成常数附加电导 `rho`，`tau_H` 与
   `theta_H` 从方程里掉出去。{hi_arm['n_points']} 个参数点在这一臂只产生
   {hi_arm['n_distinct_tail_rates']} 条不同轨迹（按 rho 分组逐位相同）。因此"12 点都验证了高态可维持"
   是对覆盖度的高估：可维持性只在 rho 这一个轴上被验证过。真正区分 H 家族的是两条低初值臂。
2. **offset 臂没有从收敛的高分支出发。** 四条 X 臂都从 `H(0)=2*theta={xic['h_initial']:.3f}` 开始，
   而 availability=1 跑到的自洽高分支是 `H={xic['h_converged_at_full_availability']:.3f}`（{xic['head_start_ratio']:.2f} 倍）。
   若从收敛值出发，H 衰减到 theta 以下多需要 {xic['extra_above_theta_decay_s_if_started_converged']:.2f} s，
   仍显著短于 {xic['run_duration_s']:.2f} s 的记录长度与 {xic['required_low_window_s']:.0f} s 的判定窗
   （余量充足={str(xic['margin_ok']).lower()}），所以本轮结论在这个界内成立；但严格意义上被检验的是
   "高 H 起步能否被压住"，不是"已收敛发作能否被掐断"。

## 测了什么

S1 固定 connection seed 1 / noise 401，在 H1/H6 两个既有家族上扫描低于旧下界的三个 H 增益
和两个阈值尺度。每点同时要求健康低初值、易感低初值保持间期，且易感高初值保持有限高态。相邻
两点同时通过才算自然参数窗。

X1 从同一个解析高 H 初值出发，把 recurrent relay availability 冻结为 1、0.5、0.1、0，检验
现有 X 路径的理论最大终止权限。x=0 只是一条结构性因果探针，不是生理参数。

## 科学边界

本轮只允许说明 frozen entry/offset component control。没有接 dynamic Z/X，没有跑无 kick
lifecycle，没有测试 M 形态、K 招募、A/ELR，也没有比较真实 E1146 ictal morphology。因此不能称为
迟滞、双稳态、极限环或可恢复发作闭环。

## 核心科学目标复审与下一授权边界

GX1 原预注册决策表落在“no natural window + X path reachable”，局部路由原本指向显式
D-dependent H gain。然而，同一 D 的双盆地不是完整 lifecycle 的必要条件，而且 H1 已出现
健康低态到易感高态的单稳态型 candidate。直接加入显式 D gate 会把“Z 控制 onset”写进方程，
存在先射箭再画靶的风险。

因此正式授权改为：**保留当前 H1 方程，先完成 D-X 状态平面、early spatial mode audit 和
由状态平面约束的 dynamic no-kick pilot。** 只有 D-X 平面不存在可闭合路径、transition 在多 seed
下不稳，或 leading mode 是全局共同模态时，才允许改用 local E/I-balance H sensor。显式 D gate
只保留为后续 mechanistic control。GX2 `D gate × shared X/H path` 2×2 继续不得执行。

## 工程与资源

- strip trajectories: {strip['n_rows']}; X trajectories: {xmap['n_rows']};
- numerical safe: {verdict['numerical_safe_rows']}/{verdict['numerical_total_rows']};
  numerically failed strip points: {strip.get('n_numerical_failure_points', 'n/a')};
- the six blessed engine files were checked by the execution lock.  The module that implements the H gate
  and the frozen relay (`src/snn_engine/mz_slow_vars.py`) is **not** in that blessed set; it was last
  modified in `fe9674a2` (2026-08-02 01:30 +0800), before the GX1 lock, and `cmd_lock` now pins it under
  `mechanism_module_hashes` for future runs;
- long stages used setsid/nohup, exact PID watchdogs, stage locks and sentinels;
- {wd};
- peak single-cell RSS {rs['peak_rss_gib']:.3f} GiB over {rs['n_resource_rows']} recorded trajectories;
  swap delta {rs['swap_delta_mib']:.0f} MiB;
- the spec's `+256 MiB stop new submission` rule is **not** implemented: all futures are submitted up
  front, so only the `+512 MiB` hard stop is active, and it tears down the whole stage rather than the
  newest worker;
- final commit and test counts are recorded in `run_manifest.json` after sign-off.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-finalize", action="store_true")
    args = ap.parse_args()
    if not args.confirm_finalize:
        raise SystemExit("--confirm-finalize is required")
    strip = _load("selectivity_strip.json")
    xmap = _load("x_authority_map.json")
    if strip.get("status") != "COMPLETE" or xmap.get("status") != "COMPLETE":
        raise SystemExit("S1 and X1 must both be complete")
    lock = _load("execution_lock.json")
    archived_loads = archived_relay_loads(_locked_artifact(lock, "frozen_map"),
                                          xmap["rows"][0]["point_id"])
    roll_hi = float(_locked_artifact(lock, "baseline_contract")["roll_hi_hz"])
    verdict = build_candidate_verdict(strip, xmap, archived_loads)
    figures = os.path.join(OUT, "figures")
    os.makedirs(figures, exist_ok=True)
    plot_strip(strip, os.path.join(figures, "selectivity_strip.png"))
    plot_x(xmap, os.path.join(figures, "x_authority.png"), roll_hi)
    plot_logic(strip, xmap, os.path.join(figures, "failure_logic.png"))
    readme = """### selectivity_strip.png

两行分别是 H1/H6，三列分别检验健康低初值、易感低初值和易感高初值；格内数字是末 1 秒平均率。
颜色区分间期、升高事件串、有限高态和未解析结果。
**关注点**：没有一个点满足“两个低态不点燃、易感高态能维持”；H1 高阈值行显示的是健康低态
保留、易感低态也点燃的单向 D 选择性，不要误读成双稳态。第三列每个 rho 只有一个数值、
跨家族与阈值完全重复，是因为高初值把 H 门钉在全开——该列只分辨 rho，不分辨 H 的时间常数与阈值。

### x_authority.png

同一个易感高 H 初值下，比较四档 frozen relay availability 的 300 ms 平滑率与 H 轨迹。
上图虚线是判据真正使用的间期带上沿（不是 20 Hz），下图虚线是 H 门的中点 theta。
x=0 是理论最大权限因果探针，不代表生理可实现值。
**关注点**：availability=0.1 已让末段连续至少 2 秒回到间期，说明通路可达；0.5 仍维持高态，
且同 anchor 归档的 0.872/0.786 负载臂也维持高态，故证据指向动态范围不足而不是 H 绕过 X。
四条臂都从 H=2*theta 起步，比 availability=1 收敛到的高分支低约 3 倍，所以这里检验的是
“高 H 起步能否被压住”，不是“已收敛发作能否被掐断”。

### failure_logic.png

把 S1 的自然选择窗结论与 X1 的路径权限结论组成原预注册决策表，浅橙格表示数据落点；这张表现在
只保留作审计轨迹，不再决定下一实验。
**关注点**：同一 D 下双盆地并非 lifecycle 的必要条件，因此局部 D gate 与 X 单独扩程均暂缓；
正式下一步是保持当前方程，先做 D-X 状态平面和空间失稳审计。共享路径与完整 2×2 仍未获授权。
"""
    _write_text(os.path.join(figures, "README.md"), readme)
    _write_json("candidate_verdict.json", verdict)
    _write_text(os.path.join(OUT, "STATUS.md"), _status(verdict))
    _write_text(ARCHIVE, _archive(verdict, strip, xmap))
    # Sign-off fields are added by hand after the tests and the visual QA; re-finalizing must not silently
    # drop them.
    prior = _load("run_manifest.json") if os.path.isfile(os.path.join(OUT, "run_manifest.json")) else {}
    manifest = dict(status="FINALIZED", head=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        candidate_verdict=verdict, artifacts={}, finalized_at=_now())
    for key in ("result_archive_commit", "verification"):
        if key in prior:
            manifest[key] = prior[key]
    res = _resource_summary()
    manifest.setdefault("verification", {}).update(
        numerical_safe_rows=f"{verdict['numerical_safe_rows']}_OF_{verdict['numerical_total_rows']}",
        max_single_cell_rss_gib=res["peak_rss_gib"], swap_delta_mib=res["swap_delta_mib"],
        executable_contract_sha256_at_lock=lock["source_sha256"],
        executable_contract_sha256_now=_executable_contract_sha256())
    manifest["verification"].pop("executable_contract_sha256", None)
    for rel in ("execution_lock.json", "selectivity_strip_manifest.json",
                "selectivity_strip.json", "x_authority_manifest.json",
                "x_authority_map.json", "candidate_verdict.json", "STATUS.md",
                "figures/selectivity_strip.png", "figures/x_authority.png",
                "figures/failure_logic.png", "figures/README.md"):
        path = os.path.join(OUT, rel)
        manifest["artifacts"][rel] = dict(path=path, sha256=_sha256(path))
    _write_json("run_manifest.json", manifest)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
