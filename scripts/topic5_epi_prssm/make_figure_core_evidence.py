#!/usr/bin/env python3
"""asset_id: epi_prssm_core_evidence   paper_slot: TBD

Core conclusion the figure must defend
--------------------------------------
Interictal discharge history carries information that changes which contacts a
future discharge recruits, and that information is not a restatement of how many
parameters the model has.

Evidence chain, one panel per question
--------------------------------------
A  Does the information survive when the observer is switched off, and for how long?
B  Does the message need this patient's own wiring, and on which of the two graph
   paths -- the slow state's propagation, or the within-event readout?
C  Is the effect carried by the state itself?  Substituting a magnitude-matched state
   from another moment of the same patient changes no parameter at all.
D  Does the state change the continuation of an event, given the same opening?

A and C are different questions: A asks whether the state predicts the future, C asks
whether the state is what does it.  B and D are likewise distinct: B is about the
spatial support of the message, D about a within-event branch point.
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

from src.topic5_epi_prssm.contracts import OUTPUT_ROOT, code_revision, package_hash
from src.topic5_epi_prssm.figure_style import (
    COLOR, DOUBLE_COLUMN_MM, LW_MAIN, LW_REFERENCE,
    figure, panel_letter, save_asset, zero_line,
)

#: 7 pt is the floor for every rendered glyph at the final 180 mm width.  The
#: project defaults sit below it (tick text is 6.8 pt), so this figure sets its own.
FS_TICK, FS_AXIS, FS_TITLE, FS_LEGEND = 7.0, 7.6, 8.4, 7.0

ASSET = "epi_prssm_core_evidence"
LADDER = OUTPUT_ROOT / "generator_ladder"
EVENTS = OUTPUT_ROOT / "event_distribution"
GRAPH = OUTPUT_ROOT / "graph_null"

HORIZONS = [5, 10, 20, 40]
#: arms shown in A; the frozen reference is the zero line, so it is not a series
OPEN_LOOP_ARMS = [("ct_ewma_g0", "leaky state"), ("g1_graph_clds", "graph state"),
                  ("g2_graph_gru_ode", "gated graph state")]
ARM_TINT = {"ct_ewma_g0": COLOR["G0"], "g1_graph_clds": COLOR["G1"],
            "g2_graph_gru_ode": COLOR["G2"]}


def _read(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def panel_a(ax, card: dict) -> dict:
    """Open-loop horizon: the observer is off, only elapsed time is supplied."""
    contrasts = card.get("open_loop_contrasts") or {}
    drawn = {}
    for arm, label in OPEN_LOOP_ARMS:
        xs, ys, los, his = [], [], [], []
        for h in HORIZONS:
            entry = contrasts.get(f"{arm}::vs_frozen_state_node::H{h}")
            if not entry or entry.get("median_delta") is None:
                continue
            xs.append(h); ys.append(entry["median_delta"])
            ci = entry.get("ci") or [np.nan, np.nan]
            los.append(ci[0]); his.append(ci[1])
        if not xs:
            continue
        colour = ARM_TINT[arm]
        ax.plot(xs, ys, "-o", ms=3.2, lw=LW_MAIN, color=colour, label=label)
        ax.fill_between(xs, los, his, color=colour, alpha=0.13, lw=0)
        drawn[arm] = {"horizons": xs, "median": ys}
    zero_line(ax)
    ax.set_xscale("log"); ax.set_xticks(HORIZONS)
    ax.set_xticklabels([str(h) for h in HORIZONS])
    ax.xaxis.set_minor_locator(__import__("matplotlib").ticker.NullLocator())
    ax.set_xlabel("events predicted after the observer is off", fontsize=FS_AXIS)
    ax.set_ylabel("Δ NLL vs frozen state (nats/event)", fontsize=FS_AXIS,
                  labelpad=2)
    ax.set_title("Predicts ahead with the observer off", fontsize=FS_TITLE,
                 loc="left", pad=4)
    low, high = ax.get_ylim()
    ax.set_ylim(low - 0.42 * (high - low), high)
    ax.legend(loc="lower center", ncol=3, handlelength=1.0, borderpad=0.2,
              columnspacing=0.9, fontsize=FS_LEGEND)
    return drawn


def panel_b(ax, card: dict) -> dict:
    """Which graph path needed this patient's own wiring.

    Grouped by path rather than by null, because the question is about the path; the
    two nulls are two ways of asking it and belong side by side inside each group.
    """
    contrasts = (card or {}).get("contrasts") or {}
    provenance = (card or {}).get("run_provenance") or {}
    paths = [("@generator", "slow state's graph"), ("@decoder", "within-event readout")]
    nulls = [("degree_preserving_rewire", "rewired", COLOR["G1"]),
             ("forward_only_shuffled", "relabelled", COLOR["G2"])]
    drawn, values = {}, {}
    for suffix, _ in paths:
        for null, _, _ in nulls:
            entry = contrasts.get(f"event_nll::real-vs-{null}{suffix}")
            if entry and entry.get("median_delta") is not None:
                values[(suffix, null)] = -entry["median_delta"]
                drawn[f"{null}{suffix}"] = entry
    if not values:
        ax.text(0.5, 0.5, "regenerating on the current model package",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=FS_TICK, color="#8A8A8A", style="italic")
        ax.set_xticks([]); ax.set_yticks([])
        for side in ax.spines.values():
            side.set_visible(False)
    else:
        y = np.arange(len(paths))
        height = 0.34
        for j, (null, null_label, tint) in enumerate(nulls):
            vals = [values.get((suffix, null), np.nan) for suffix, _ in paths]
            ax.barh(y + (j - 0.5) * height, vals, height=height, color=tint,
                    label=null_label)
        # group names go inside the axes: as y tick labels they reach far enough left
        # to sit on top of the neighbouring panel
        ax.set_yticks([])
        ax.invert_yaxis()
        ax.set_xlabel("cost of shuffling that path (nats/event)", fontsize=FS_AXIS)
        top = max(v for v in values.values() if np.isfinite(v))
        ax.set_xlim(0, top * 1.55)
        for row, (_, label) in zip(y, paths):
            ax.text(top * 0.035, row - height * 0.95, label, va="bottom", ha="left",
                    fontsize=FS_TICK, color="#333333")
        ax.legend(loc="center right", handlelength=0.9, borderpad=0.25,
                  fontsize=FS_LEGEND, title="graph null", title_fontsize=FS_LEGEND)
    ax.set_title("Which graph path needs the wiring", fontsize=FS_TITLE,
                 loc="left", pad=4)
    return {"contrasts": drawn, "provenance": provenance}


def panel_c(ax, card: dict) -> dict:
    """Substituting the state changes no parameter, so a cost is the state itself."""
    swaps = (card or {}).get("state_swap") or {}
    families = [("node_film", "per-contact"), ("edge_gate", "+ pair coupling"),
                ("initial_state", "opening only")]
    sources = ["g0", "g2", "g3"]
    drawn, x, series = {}, np.arange(len(families)), {}
    for source in sources:
        vals = []
        for family, _ in families:
            entry = swaps.get(f"order_nll::{family}_{source}::swap_matched")
            # 0.0 is falsy, so a truthiness test turns the architecturally exact zero
            # of the opening-only control into a missing value in both the drawn
            # series and the metadata.  Test for absence explicitly.
            value = entry.get("median_delta") if entry else None
            vals.append(-value if value is not None else np.nan)
        series[source] = vals
    frozen = [swaps.get(f"order_nll::{f}_frozen::swap_matched") for f, _ in families]
    width = 0.24
    for i, source in enumerate(sources):
        ax.bar(x + (i - 1) * width, series[source], width=width,
               color=[COLOR["G0"], COLOR["G2"], COLOR["G3"]][i],
               label={"g0": "leaky", "g2": "gated graph", "g3": "graph + resource"}[source])
        drawn[source] = series[source]
    ax.axhline(0.0, color="#333333", lw=LW_REFERENCE)
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in families], fontsize=FS_TICK)
    ax.set_ylabel("cost of a swapped state (nats/event)", fontsize=FS_AXIS)
    ax.set_title("The state itself carries it", fontsize=FS_TITLE, loc="left", pad=4)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.34)
    # An exactly-zero bar reads as missing data unless it is said out loud. This
    # adapter adds the same number to every candidate, and that cancels when the
    # candidates are compared, so its zero is architectural rather than measured.
    ax.annotate("exactly 0: adds the same\nnumber to every candidate",
                xy=(x[-1], 0.0), xytext=(x[-1] - 0.05, top * 0.62), ha="center",
                fontsize=FS_TICK, color="#6A6A6A",
                arrowprops=dict(arrowstyle="->", lw=0.7, color="#8A8A8A"))
    ax.legend(loc="upper left", handlelength=0.9, borderpad=0.2, fontsize=FS_LEGEND)
    drawn["frozen_controls_are_zero_by_construction"] = [
        e["median_delta"] if e else None for e in frozen]
    return drawn


def panel_d(ax, card: dict) -> dict:
    """Same opening, different continuation."""
    block = (card or {}).get("ambiguous_prefix") or {}
    moving = block.get("by_arm") or {}
    control = block.get("negative_control_arms") or {}
    depths = [1, 2, 3]
    drawn = {}
    for arm, depths_map in sorted(moving.items()):
        if arm.startswith("initial_state"):
            continue
        ys = []
        for d in depths:
            entry = depths_map.get(str(d)) or depths_map.get(d)
            ys.append(entry["median_delta"] if entry else np.nan)
        colour = COLOR["G1"] if arm.startswith("node_film") else COLOR["G2"]
        ax.plot(depths, ys, "-o", ms=2.8, lw=LW_MAIN * 0.8, color=colour, alpha=0.75)
        drawn[arm] = ys
    ax.axhline(0.0, color="#8A8A8A", lw=LW_MAIN, zorder=1)
    ax.plot([], [], "-", lw=LW_MAIN, color="#8A8A8A", label="frozen state (exactly 0)")
    ax.plot([], [], "-o", ms=2.8, color=COLOR["G1"], label="per-contact gain")
    ax.plot([], [], "-o", ms=2.8, color=COLOR["G2"], label="gain + pair coupling")
    ax.set_xticks(depths)
    ax.set_xlabel("contacts shared by the opening", fontsize=FS_AXIS)
    ax.set_ylabel("gain from the true state (nats/event)", fontsize=FS_AXIS,
                  labelpad=2)
    top = ax.get_ylim()[1]
    ax.set_ylim(-0.001, top * 1.42)
    ax.set_title("Same opening, different continuation", fontsize=FS_TITLE,
                 loc="left", pad=4)
    ax.legend(loc="upper right", handlelength=1.0, borderpad=0.2, labelspacing=0.2,
              fontsize=FS_LEGEND)
    drawn["negative_control_arms"] = sorted(control)
    return drawn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cohort", default="all34")
    args = parser.parse_args()

    ladder = _read(LADDER / "GENERATOR_EVIDENCE_CARD.json")
    events = _read(EVENTS / "H2A_EVIDENCE_CARD.json")
    graph = _read(GRAPH / "GRAPH_NULL_EVIDENCE_CARD.json")

    fig, axes = figure(DOUBLE_COLUMN_MM, 116.0, nrows=2, ncols=2)
    fig.subplots_adjust(left=0.115, right=0.982, top=0.912, bottom=0.100,
                        wspace=0.30, hspace=0.46)
    drawn = {
        "A_open_loop": panel_a(axes[0, 0], ladder),
        "B_graph_path": panel_b(axes[0, 1], graph),
        "C_state_swap": panel_c(axes[1, 0], events),
        "D_prefix_branch": panel_d(axes[1, 1], events),
    }
    # the right column carries long y labels, so its letters sit above the axes
    # rather than beside them
    for ax, letter, dx in zip(axes.flat, "ABCD", (-0.155, -0.085, -0.155, -0.085)):
        panel_letter(ax, letter, dx=dx, dy=1.055)

    root = OUTPUT_ROOT / "figures/revisions" / args.run_id
    out = save_asset(fig, ASSET, root=root, metadata={
        "asset_id": ASSET, "paper_slot": "TBD", "run_id": args.run_id,
        "status": "EXPLORATORY_DEVELOPMENT",
        "core_conclusion": "interictal discharge history carries information that "
                           "changes which contacts a future discharge recruits, and "
                           "that information is not a restatement of parameter count",
        "panel_questions": {
            "A": "does the information survive with the observer off, and for how long",
            "B": "which graph path needs this patient's own wiring",
            "C": "is the state itself what carries the effect",
            "D": "does the state change the continuation given the same opening",
        },
        "panels": drawn,
        "graph_panel_provenance": (graph or {}).get("run_provenance"),
        "claim_boundary": [
            "development partition only; the untouched formal-test partition was "
            "never opened",
            "a prediction gain is not evidence of a mechanism",
            "panel B speaks about patient-specific weighted spatial relations with "
            "correct contact-identity alignment; these graphs are nearly complete, so "
            "a degree-preserving rewire permutes edge weights rather than rewiring",
        ],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }, readme_entries=[{
        "filename": f"{ASSET}.png",
        "body": "四问合成一张：A 关掉观测后还能往前预测多久；B 两条图通路里哪一条需要"
                "这位患者自己的接线；C 把状态换成同一位患者另一个时刻——参数一个没变，"
                "所以代价只能来自状态本身；D 同样的开头之后，走法会不会因当时状态而不同。",
        "focus": "B 两组条形按通路分开：打乱事件内读出那条的代价约为打乱慢状态传播那条的"
                 "四倍，两种打乱方式给出同一结论；较小的传播分量是换种子漂移的 4–6 倍，"
                 "所以不是噪声。C 里的冻结对照恒为 0 是构造使然，不是结论。"
                 "全部条形只用同一个模型包的运行，按臂×种子去重。",
    }])
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
