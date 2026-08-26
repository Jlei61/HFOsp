#!/usr/bin/env python3
"""asset_id: epi_prssm_seizure_link   paper_slot: TBD

Core conclusion
---------------
The matched-pseudo-onset design cannot answer the pre-ictal question in this cohort:
a seizure onset has no non-seizure counterpart that resembles it on rate, interval and
coverage at the same time, so any apparent state difference is inseparable from the
way the discharges themselves change before a seizure.

Evidence chain, one panel per question
--------------------------------------
A  How many seizures are we actually talking about, and what does each number mean?
B  Does the effect depend on how well the pre-ictal window was observed?
C  Can a seizure be balanced against its matched non-seizure times at all?
D  Does the effect differ between a patient's own seizure subtypes?

C is the load-bearing panel: it is what turns "the signal did not survive balancing"
from a judgement into a measurement.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.topic5_epi_prssm.contracts import OUTPUT_ROOT, code_revision, package_hash
from src.topic5_epi_prssm.figure_style import (
    COLOR, DOUBLE_COLUMN_MM, LW_MAIN, LW_REFERENCE, figure, panel_letter, save_asset,
    zero_line,
)

FS_TICK, FS_AXIS, FS_TITLE, FS_LEGEND = 7.0, 7.6, 8.4, 7.0
ASSET = "epi_prssm_seizure_link"
H2B = OUTPUT_ROOT / "seizure_link_preictal"
DEN = OUTPUT_ROOT / "h2b_denominators"
SENS = OUTPUT_ROOT / "h2b_sensitivity"

PRIMARY_LAYER, PRIMARY_LEAD = "linear_graph_recurrent", "lead30m"
BALANCE_THRESHOLDS = (0.25, 0.5, 0.75, 1.0, 1.5)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def panel_a(ax, flow: pd.DataFrame) -> dict:
    """The denominator chain, drawn so no number can stand in for another."""
    row = flow[flow.lead == PRIMARY_LEAD]
    if row.empty:
        ax.text(0.5, 0.5, "denominators unavailable", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS_TICK, color="#8A8A8A")
        ax.set_xticks([]); ax.set_yticks([])
        return {}
    row = row.iloc[0]
    steps = [("patients in the cohort", row.step_1_cohort_patients, COLOR["scaffold"]),
             ("with an analysable seizure", row.step_2_patients_with_any_analysable_seizure,
              COLOR["scaffold"])]
    seiz = [("eligible seizures", row.step_3_seizures_eligible_all, COLOR["G1"]),
            ("pre-ictal window observed",
             row.step_4_seizures_meeting_observation_premise, COLOR["G2"])]
    y = np.arange(len(seiz))
    ax.barh(y, [s[1] for s in seiz], color=[s[2] for s in seiz], height=0.5)
    for i, (label, value, _) in enumerate(seiz):
        ax.text(value + row.step_3_seizures_eligible_all * 0.02, i, f"{int(value)}",
                va="center", fontsize=FS_TICK, color="#333333")
    ax.set_yticks(y)
    ax.set_yticklabels([label for label, _, _ in seiz], fontsize=FS_TICK)
    ax.invert_yaxis()
    ax.set_xlim(0, row.step_3_seizures_eligible_all * 1.22)
    ax.set_ylim(len(seiz) - 0.4, -0.9)
    ax.set_xlabel("seizures", fontsize=FS_AXIS)
    ax.text(0.02, 0.955,
            f"{int(steps[0][1])} patients, {int(steps[1][1])} with a usable seizure",
            transform=ax.transAxes, ha="left", va="top", fontsize=FS_TICK,
            color="#6A6A6A")
    ax.set_title("What each denominator counts", fontsize=FS_TITLE, loc="left", pad=4)
    return {"flow": {k: int(v) for k, v in row.items() if str(k).startswith("step_")}}


def panel_b(ax, grid: pd.DataFrame) -> dict:
    """Does the effect track how well the window was observed?"""
    block = grid[(grid.layer == PRIMARY_LAYER) & (grid.tier == "primary")
                 & (grid.reading == "open_loop_at_onset")]
    if block.empty:
        return {}
    leads = ["lead5m", "lead15m", "lead30m", "lead60m"]
    minutes = [5, 15, 30, 60]
    drawn = {}
    for population, label, tint in (("all_eligible", "all eligible", COLOR["G1"]),
                                    ("high_observability", "window observed", COLOR["G2"])):
        rows = block[block.population == population].set_index("lead")
        xs, ys, los, his = [], [], [], []
        for lead, m in zip(leads, minutes):
            if lead not in rows.index:
                continue
            r = rows.loc[lead]
            xs.append(m); ys.append(r.median_delta)
            los.append(r.ci_low); his.append(r.ci_high)
        if not xs:
            continue
        ax.plot(xs, ys, "-o", ms=3.2, lw=LW_MAIN, color=tint, label=label)
        ax.fill_between(xs, los, his, color=tint, alpha=0.13, lw=0)
        drawn[population] = {"minutes": xs, "median": ys}
    zero_line(ax)
    ax.set_xscale("log"); ax.set_xticks(minutes)
    ax.set_xticklabels([str(m) for m in minutes])
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_xlabel("minutes before onset the observer closes", fontsize=FS_AXIS)
    ax.set_ylabel("readout shift vs matched times (SD)", fontsize=FS_AXIS, labelpad=2)
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.32 * (high - low), high)
    ax.legend(loc="lower center", ncol=2, handlelength=1.0, borderpad=0.2,
              fontsize=FS_LEGEND)
    ax.set_title("Observability does not explain it", fontsize=FS_TITLE, loc="left", pad=4)
    return drawn


def panel_c(ax, effects: pd.DataFrame) -> dict:
    """Whether a seizure can be balanced against its matched times at all."""
    dims = [c for c in effects.columns
            if c.startswith("nuisanceonly__") and c.endswith("_z")]
    if not dims:
        return {}
    Z = effects[dims].apply(pd.to_numeric, errors="coerce")
    counts, patients = [], []
    for thr in BALANCE_THRESHOLDS:
        ok = (Z.abs() < thr).all(axis=1) & Z.notna().all(axis=1)
        counts.append(int(ok.sum()))
        patients.append(int(effects[ok].subject.nunique()))
    x = np.arange(len(BALANCE_THRESHOLDS))
    ax.bar(x, counts, color=COLOR["onset"], width=0.6)
    total = len(effects)
    for i, (n, p) in enumerate(zip(counts, patients)):
        ax.text(i, n + total * 0.018, f"{n}" + (f"\n{p} pts" if n else ""),
                ha="center", va="bottom", fontsize=FS_TICK, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t:g}" for t in BALANCE_THRESHOLDS], fontsize=FS_TICK)
    ax.set_xlabel("how closely every nuisance must match (SD)", fontsize=FS_AXIS)
    ax.set_ylabel(f"seizures balanced on all {len(dims)}", fontsize=FS_AXIS,
                  labelpad=2)
    ax.set_ylim(0, max(max(counts), 1) * 1.45)
    ax.set_title("No seizure has a matched counterpart", fontsize=FS_TITLE,
                 loc="left", pad=4)
    return {"thresholds": list(BALANCE_THRESHOLDS), "n_balanced": counts,
            "n_patients": patients, "n_dimensions": len(dims), "n_seizures": total,
            "dimensions": [d.replace("nuisanceonly__", "").replace("_z", "")
                           for d in dims]}


def panel_d(ax, card: dict) -> dict:
    """Does the effect differ between a patient's own subtypes?"""
    block = (card or {}).get("subtype_interaction__broad_ER") or {}
    if block.get("status") != "OK":
        ax.text(0.5, 0.5, "too few patients have two subtypes\nlarge enough to compare",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS_TICK, color="#8A8A8A", style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title("Subtype difference", fontsize=FS_TITLE, loc="left", pad=4)
        return {"status": block.get("status")}
    rows = pd.DataFrame(block["per_patient"]).sort_values("spread").reset_index(drop=True)
    x = np.arange(len(rows))
    ax.bar(x - 0.19, rows.spread, width=0.36, color=COLOR["G2"], label="observed")
    ax.bar(x + 0.19, rows.null_median, width=0.36, color="#C9C9C9",
           label="relabelled at random")
    # a spread of order 100 SD is the same variance degeneracy that appears whenever a
    # matched set is nearly constant; a linear axis would let one such patient erase
    # the rest, and hiding it would be worse
    ax.set_yscale("symlog", linthresh=1.0)
    extreme = rows[rows.spread > 20.0]
    if len(extreme):
        ax.annotate("degenerate\nmatched set",
                    xy=(extreme.index[0] - 0.19, extreme.spread.iloc[0]),
                    xytext=(max(len(rows) - 2.9, 0.2), extreme.spread.iloc[0] * 0.055),
                    fontsize=FS_TICK, color="#8A4A4A", ha="center", va="top",
                    arrowprops=dict(arrowstyle="->", lw=0.7, color="#8A4A4A"))
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{i+1}" for i in range(len(rows))], fontsize=FS_TICK)
    ax.set_xlabel("patients with two comparable subtypes", fontsize=FS_AXIS)
    ax.set_ylabel("spread between subtypes (SD)", fontsize=FS_AXIS, labelpad=2)
    ax.legend(loc="upper left", handlelength=0.9, borderpad=0.2,
              fontsize=FS_LEGEND)
    ax.set_title("Subtype difference is within its own null", fontsize=FS_TITLE,
                 loc="left", pad=4)
    return {"n_patients": int(len(rows)),
            "n_above_own_null": block.get("n_patients_above_their_own_null"),
            "sign_test_p": block.get("sign_test_p")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    flow = _read_csv(DEN / f"denominator_flow__{PRIMARY_LAYER}.csv")
    grid = _read_csv(SENS / "h2b_sensitivity_grid.csv")
    effects = _read_csv(H2B / f"preictal_effects__{PRIMARY_LAYER}__{PRIMARY_LEAD}.csv")
    card = json.loads((SENS / "H2B_SENSITIVITY_CARD.json").read_text()) \
        if (SENS / "H2B_SENSITIVITY_CARD.json").exists() else {}

    fig, axes = figure(DOUBLE_COLUMN_MM, 116.0, nrows=2, ncols=2)
    fig.subplots_adjust(left=0.115, right=0.982, top=0.912, bottom=0.100,
                        wspace=0.32, hspace=0.48)
    drawn = {"A_denominators": panel_a(axes[0, 0], flow),
             "B_observability": panel_b(axes[0, 1], grid),
             "C_balance": panel_c(axes[1, 0], effects),
             "D_subtype": panel_d(axes[1, 1], card)}
    for ax, letter, dx in zip(axes.flat, "ABCD", (-0.115, -0.135, -0.135, -0.115)):
        panel_letter(ax, letter, dx=dx, dy=1.055)

    root = OUTPUT_ROOT / "figures/revisions" / args.run_id
    out = save_asset(fig, ASSET, root=root, metadata={
        "asset_id": ASSET, "paper_slot": "TBD", "run_id": args.run_id,
        "status": "EXPLORATORY_DEVELOPMENT",
        "core_conclusion": "in this cohort a seizure onset has no non-seizure "
                           "counterpart that matches it on rate, interval and coverage "
                           "at once, so the matched-pseudo-onset design cannot separate "
                           "a state shift from the way the discharges themselves change",
        "panels": drawn,
        "denominator_rule": {
            "population_layer": "every eligible seizure at the lead",
            "sensitivity_layer": "the subset whose pre-ictal window was observed",
            "do_not_write": ["203 seizures were analysed",
                             "the cohort had 203 seizures"],
        },
        "claim_boundary": [
            "development partition only; the untouched test partition was never opened",
            "panel C is a property of the matching design in this cohort, not a "
            "statement that no pre-ictal change exists",
        ],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }, readme_entries=[{
        "filename": f"{ASSET}.png",
        "body": "A 说清每个分母各自数的是什么——可分析的发作总数与"
                "「发作前确实观测到放电」的子集不是一回事；B 看效应会不会随观测质量变化；"
                "C 问一个更基本的问题：一次发作到底能不能找到在所有混杂维度上都像它的"
                "非发作时刻；D 看同一位患者自己的发作亚型之间有没有差别。",
        "focus": "C 是承重面板：全部六个维度同时配平到 0.5 个标准差以内的发作是 0 次，"
                 "所以 B 里那条曲线无法与「发作前放电本来就变慢变稀」分开。"
                 "这是这个队列里匹配设计的性质，不是「没有发作前变化」。",
    }])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
