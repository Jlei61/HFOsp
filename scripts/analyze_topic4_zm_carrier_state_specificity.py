#!/usr/bin/env python3
"""Does the passing persistent-conductance dose carry across the frozen Z/M
operating points the slow variables traverse, and is it off before entry?

`pre_entry__natural` is the earliest registered checkpoint, roughly 1.35 s
ahead of the native onset.  It is a pre-ictal reference, not an interictal
baseline, so whether it can test the Z gate at all depends on how far z has
already fallen there; the adjudication reads that gate rather than assuming it.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.analyze_topic4_zm_mode_h_pilot import _row  # noqa: E402
from scripts.analyze_topic4_zm_conductance_homotopy import credible_carrier  # noqa: E402
from src.snn_engine.slow_field import zero_baseline_sigmoid  # noqa: E402


IN = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/smoke/seed1"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
REFERENCE_STATE = "pre_entry__natural"
# Above this the Z gate is more open than shut, so the reference point offers
# no "mechanism off" condition and selectivity cannot be read from this panel.
GATE_OPEN_CEILING = 0.5
# Ordered along the slow-variable trajectory the lifecycle has to traverse.
TRAVERSED_STATES = (
    "bounded_mid__rising",
    "bounded_mid__peak",
    "bounded_late__rising",
    "bounded_late__peak",
)


def _close(a, b):
    return a is not None and np.isclose(float(a), float(b))


def arm_key(summary):
    """(frozen operating point, persistent dose) for arms in the locked panel."""
    mech = summary.get("mechanism", {})
    subtype = mech.get("pv_som_inhibitory_subtypes")
    mode = mech.get("state_selective_mode_H") or {}
    state = summary.get("state")
    if not subtype or state not in (REFERENCE_STATE,) + TRAVERSED_STATES:
        return None
    if not (
        _close(summary.get("T_ms"), 2500.0)
        and _close(mode.get("rho_mode_H"), 0.0)
        and _close(mode.get("mode_H_persistent_e_exc"), 60.0)
        and _close(mode.get("tau_mode_H_down"), 250.0)
        and _close(mode.get("mode_H_common_subtraction"), 0.0)
        and _close(subtype.get("tau_d_som_ms"), 60.0)
        and _close(subtype.get("som_source_fraction_realized"), 0.25)
        and _close(subtype.get("som_slow_integrated_budget_fraction"), 0.35)
        and _close(subtype.get("som_recruit_delay_scale"), 3.0)
        and int(subtype.get("seed", 1)) == 1
        and subtype.get("slow_membrane_mode") != "shunt"
    ):
        return None
    return state, float(mode.get("mode_H_persistent_g_max", 0.0))


def z_gate_open_at_freeze(summary, array):
    """How far the Z susceptibility gate is open at the frozen core z.

    Uses the engine's own gate so the reported number is the one the mechanism
    actually multiplies by, not a re-derivation that can drift from it.
    """
    mode = summary["mechanism"]["state_selective_mode_H"]
    z0 = float(np.asarray(array["trace_z_core_mean"], float)[0])
    base = float(mode.get("z_mode_base", 1.0))
    susceptible = float(mode.get("z_mode_susceptible", 0.5))
    zeta = float(np.clip((base - z0) / (base - susceptible), 0.0, 1.0))
    return float(zero_baseline_sigmoid(
        np.asarray([zeta]),
        float(mode.get("zeta_mode_center", 0.5)),
        float(mode.get("zeta_mode_slope", 0.1)),
    )[0])


def traces_identical(a, b):
    """Bit equality on the shared traces; the Z gate either couples or it does not."""
    shared = sorted(set(a).intersection(b))
    return bool(shared) and all(
        np.array_equal(np.asarray(a[key]), np.asarray(b[key])) for key in shared
    )


def adjudicate(rows):
    """A carrier confined to one point cannot carry a lifecycle, and a mechanism
    that still couples where its gate is shut cannot produce entry or exit.

    The selectivity question is only answerable when the reference point sits
    on the closed side of the Z gate; otherwise the panel has no off state and
    the verdict says so instead of reading engagement as a gate failure.
    """
    # A fixed-dose sweep over operating points: the dose is the one carried by
    # the reference pair, so a dose band at one point must not redefine it.
    doses = sorted(
        {dose for state, dose in rows if state == REFERENCE_STATE and dose > 0.0}
    )
    if len(doses) != 1:
        raise RuntimeError(f"state sweep needs exactly one dosed reference arm: {doses}")
    dose = doses[0]
    if (REFERENCE_STATE, 0.0) not in rows:
        raise RuntimeError("state sweep is missing the no-mechanism reference control")
    reference = rows[(REFERENCE_STATE, dose)]
    gate_open = reference["z_gate_open_at_freeze"]
    testable = bool(gate_open <= GATE_OPEN_CEILING)
    inert = bool(
        reference.get("identical_to_no_mechanism", False)
        and not reference["credible_carrier"]
    )
    carrier_states = sorted(
        state for state in TRAVERSED_STATES
        if rows.get((state, dose), {}).get("credible_carrier")
    )
    if not carrier_states:
        headline = "NO_CARRIER_AT_ANY_FROZEN_OPERATING_POINT"
        coordinate = "the dose that passed on one point does not generalise; re-open the dose band"
    elif not testable:
        headline = "CARRIER_ON_THE_LATE_ARC_SELECTIVITY_UNTESTED"
        coordinate = (
            "no operating point in this panel sits on the closed side of the Z gate, "
            "so register a checkpoint whose z is above the gate before claiming entry"
        )
    elif not inert:
        headline = "MECHANISM_NOT_STATE_SELECTIVE"
        coordinate = "the mechanism still couples where its gate is shut, so entry and exit cannot exist"
    elif len(carrier_states) == 1:
        headline = "CARRIER_CONFINED_TO_ONE_OPERATING_POINT"
        coordinate = "a single-point carrier vanishes as soon as the slow variables move"
    else:
        headline = "STATE_SELECTIVE_CARRIER_ACROSS_TRAVERSED_POINTS"
        coordinate = "none; release Z/M and ask whether M terminates this carrier"
    return {
        "verdict": headline,
        "dose": dose,
        "carrier_states": carrier_states,
        "selectivity_testable": testable,
        "reference_point_inert": inert,
        "reference_z_gate_open_at_freeze": gate_open,
        "reference_persistent_g_core_mean_peak": reference[
            "persistent_g_core_mean_peak"
        ],
        "next_coordinate": coordinate,
        "claim_boundary": "seed-1 frozen-Z/M 2.5-s operating-point sweep",
    }


def main():
    found = {}
    for root in sorted(IN.glob("*pvSOM*")):
        sp, tp = root / "summary.json", root / "traces.npz"
        if not sp.is_file() or not tp.is_file():
            continue
        summary = json.loads(sp.read_text())
        key = arm_key(summary)
        if key is not None:
            if key in found:
                raise RuntimeError(f"duplicate state-sweep arm: {key}")
            found[key] = (root, summary)

    rows, arrays = {}, {}
    for key, (root, summary) in sorted(found.items()):
        row, array = _row(f"{key[0]}__g{key[1]:g}", root, summary)
        row["core_mean_hz"] = float(summary["core_modulation"]["mean_hz"])
        row["core_rho80_active_fraction"] = float(summary["core_rho80_active_fraction"])
        row["credible_carrier"] = credible_carrier(row)
        row["persistent_g_core_mean_peak"] = float(
            np.max(array.get("trace_mode_H_persistent_g_core_mean", np.zeros(1)))
        )
        row["z_gate_open_at_freeze"] = z_gate_open_at_freeze(summary, array)
        rows[key], arrays[key] = row, array
    for key in [key for key in rows if key[0] == REFERENCE_STATE and key[1] > 0.0]:
        rows[key]["identical_to_no_mechanism"] = traces_identical(
            arrays[key], arrays.get((REFERENCE_STATE, 0.0), {})
        )

    verdict = adjudicate(rows)
    # One artifact, one question: keep the swept dose and its interictal control.
    rows = {
        key: row for key, row in rows.items()
        if key[1] == verdict["dose"] or key == (REFERENCE_STATE, 0.0)
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "carrier_state_specificity_summary.json").write_text(
        json.dumps(
            {
                "schema": "topic4_zm_carrier_state_specificity_v1_2026-08-04",
                "verdict": verdict,
                "rows": {f"{s}__g{d:g}": row for (s, d), row in rows.items()},
            },
            indent=2, sort_keys=True, allow_nan=False,
        ) + "\n"
    )

    trajectory = TRAVERSED_STATES + (REFERENCE_STATE,)
    order = sorted(rows, key=lambda key: (trajectory.index(key[0]), -key[1]))
    fig, axes = plt.subplots(
        len(order), 2, figsize=(11, 2.6 * len(order)), constrained_layout=True
    )
    for ir, key in enumerate(order):
        row, a = rows[key], arrays[key]
        axes[ir, 0].plot(a["fine_time_ms"] / 1000.0, a["fine_core_rate_hz"],
                         color="#d95f45", lw=.7)
        axes[ir, 0].set(xlabel="time (s)",
                        ylabel=f"{key[0]}\ng={key[1]:g}\ncore Hz")
        axes[ir, 0].set_title(
            f"gap {row['post_onset_deep_gap_fraction']}; "
            f"PC1 {row['spatial_pc1']:.3f}; "
            f"carrier {row['credible_carrier']}"
        )
        kymo = a["coarse_kymo_axial"]
        axes[ir, 1].imshow(kymo, origin="lower", aspect="auto", cmap="magma",
                           extent=[0, .025 * kymo.shape[1], 0, kymo.shape[0]])
        axes[ir, 1].set(
            xlabel="time (s)", ylabel="axis bin",
            title=f"slow exc. g core peak {row['persistent_g_core_mean_peak']:.4f}",
        )
    fig.suptitle(verdict["verdict"], fontsize=14)
    fig_dir = OUT / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "carrier_state_specificity.png", dpi=170)
    plt.close(fig)

    readme = fig_dir / "README.md"
    prior = readme.read_text() if readme.exists() else ""
    marker = "### carrier_state_specificity.png"
    if marker not in prior:
        readme.write_text(
            prior.rstrip() + "\n\n" + marker + "\n\n"
            "固定慢兴奋强度，沿慢变量会经过的几个冻结工作点各跑一条轨迹，"
            "最后两行是发作前参考点的加机制/不加机制配对。"
            "左列给核心放电与该点是否过 carrier 判据，右列给轴向时空图"
            "以及该点实际被调动起来的慢兴奋强度。\n\n"
            "**关注点**：carrier 是只在一个工作点出现（慢变量一动就消失），"
            "还是覆盖整段轨迹；以及参考点那一对是否逐位相同。"
            "注意先看该点的 Z 门开度——若门已经开着，这套面板就没有"
            "\"机制关闭\"的对照状态，特异性问题在这里是测不了的。\n"
        )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
