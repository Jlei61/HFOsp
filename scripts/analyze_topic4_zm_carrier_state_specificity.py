#!/usr/bin/env python3
"""Does the passing persistent-conductance dose carry across the frozen Z/M
operating points the slow variables traverse, and is it inert interictally?"""
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


IN = ROOT / "results/topic4_sef_hfo/zm_fast_lifecycle_development/smoke/seed1"
OUT = ROOT / "results/topic4_sef_hfo/zm_mode_lifecycle"
INTERICTAL_STATE = "pre_entry__natural"
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
    if not subtype or state not in (INTERICTAL_STATE,) + TRAVERSED_STATES:
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


def traces_identical(a, b):
    """Bit equality on the shared traces; the Z gate either couples or it does not."""
    shared = sorted(set(a).intersection(b))
    return bool(shared) and all(
        np.array_equal(np.asarray(a[key]), np.asarray(b[key])) for key in shared
    )


def adjudicate(rows):
    """A carrier confined to one point cannot carry a lifecycle, and a mechanism
    that also fires at the interictal point cannot produce entry or exit."""
    doses = sorted({dose for _, dose in rows if dose > 0.0})
    if not doses:
        raise RuntimeError("state sweep has no dosed arm")
    dose = doses[0]
    for key in ((INTERICTAL_STATE, dose), (INTERICTAL_STATE, 0.0)):
        if key not in rows:
            raise RuntimeError(f"state sweep is missing the interictal pair: {key}")
    interictal = rows[(INTERICTAL_STATE, dose)]
    inert = bool(
        interictal.get("identical_to_no_mechanism", False)
        and not interictal["credible_carrier"]
    )
    carrier_states = sorted(
        state for state in TRAVERSED_STATES
        if rows.get((state, dose), {}).get("credible_carrier")
    )
    if not carrier_states:
        headline = "NO_CARRIER_AT_ANY_FROZEN_OPERATING_POINT"
        coordinate = "the dose that passed on one point does not generalise; re-open the dose band"
    elif not inert:
        headline = "MECHANISM_NOT_STATE_SELECTIVE"
        coordinate = "the Z gate does not close interictally, so entry and exit cannot exist"
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
        "interictal_point_inert": inert,
        "interictal_persistent_g_core_mean_peak": interictal[
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
        rows[key], arrays[key] = row, array
    doses = sorted({dose for _, dose in rows if dose > 0.0})
    if doses and (INTERICTAL_STATE, doses[0]) in arrays:
        rows[(INTERICTAL_STATE, doses[0])]["identical_to_no_mechanism"] = (
            traces_identical(
                arrays[(INTERICTAL_STATE, doses[0])],
                arrays.get((INTERICTAL_STATE, 0.0), {}),
            )
        )

    verdict = adjudicate(rows)
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

    order = [key for key in sorted(rows) if key[1] > 0.0]
    order += [key for key in sorted(rows) if key[1] == 0.0]
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
            "最后两行是间期侧工作点的加机制/不加机制配对。"
            "左列给核心放电与该点是否过 carrier 判据，右列给轴向时空图"
            "以及该点实际被调动起来的慢兴奋强度。\n\n"
            "**关注点**：carrier 是只在一个工作点出现（慢变量一动就消失），"
            "还是覆盖整段轨迹；以及间期侧那一对是否逐位相同——"
            "相同才说明这个机制在间期确实是关着的。\n"
        )
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
