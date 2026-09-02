#!/usr/bin/env python3
"""Final Figure-6 render with the 2026-08-14 review legibility repairs.

Both the frozen r2 finalizer and the panel producer it drives are hash-pinned:
r2 verifies its own SHA256 against its pre-unseal manifest, and the producer
``plot_topic5_figure6_multiscale_scaffold_v0_5.py`` is pinned by
``POSTTRAINING_PIPELINE_SNAPSHOT.json``.  Both therefore stay byte-identical.
This wrapper composes every repair on top of them at run time:

* panel A names its two flanking rank strips and states how many of the learned
  shortcuts are actually drawn.  The producer draws only the three strongest of
  them and leaves both strips unlabelled, so three red arcs read as the whole
  learned nonlocal set and the strips read as stray colourbars.
* panel B states how many *distinct* generated sequences the 30 plotted columns
  contain.  Free rollout is deterministic given the frozen model and the true
  first rank, and these held-out events share few distinct first ranks, so the
  generated block is not 30 independent reproductions.
* panel D gives the two RNN maps one shared timing colourbar instead of two
  identical ones, names every field's x axis, and reserves a gutter for the
  energy colourbar tick labels, which otherwise overprint the panel-E y-axis
  label on the final canvas.
* panel E marks its two non-primary paired views ``n.s.``.  Both show a large
  RNN-versus-shuffle separation while their patient-level one-sided tests are
  P=0.202 and P=0.153, and the unannotated panel reads as a positive result.
  The frozen Panel-E decision only forbids awarding a *significance star* to a
  non-primary view; marking one not significant moves in the safe direction.
  Its primary scatter also drops to four of its six frozen ticks and a shorter
  axis name, because at this width the six labels overprint each other and the
  long name runs off the canvas.
* panel G names its contrast as distal-only.  ``Selected benefit`` alone reads
  as an overall benefit, but the plotted contrasts are the distal-transition
  ones, whose signs differ from the all-transition contrasts.
* panel H renames its ``Local`` curve to ``Local in L3``.  In panel G ``Local``
  is the local-only model arm; in panel H it is the matched local-backbone edge
  subset attenuated inside L3.  Two adjacent panels used one word for two
  different objects.

No estimand, denominator, cohort, null, patient count or significance star
changes; axis limits and plotted values are untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import finalize_topic5_figure6_multiscale_scaffold_v0_5_r2 as r2  # noqa: E402
from scripts.paper_figures import (  # noqa: E402
    plot_topic5_figure6_multiscale_scaffold_v0_5 as base,
)

NS_COLOR = "#4a5157"
DISCLOSURE = (
    "**读图必须知道的三条披露**：(1) A 的红色 shortcut 只画了该 fit 最强的 3 条，"
    "legend 标出实际总条数，全部边参与每一次计算；A 左右两条竖条已标为 Input rank / "
    "Generated rank，是该事件的输入与生成 rank，不是 colorbar。(2) B 的 Generated 列是"
    "「冻结模型 + 真实第一 rank」的确定性函数，30 列里互不相同的序列条数已标在该列下方，"
    "远少于 30，不能读成 30 次独立复现。(3) E 左/中两格视觉分离明显，但患者级单侧检验分别为 "
    "P=0.202 与 P=0.153，图上已标 n.s.；该 family 唯一可获显著性标记的是最右侧的 joint "
    "primary interaction，其联合判据 P=0.684。\n\n"
)


def mark_not_significant(ax, p_value: float) -> None:
    """Draw the same bracket ``paired_axis`` uses for a star, labelled n.s."""
    low, high = ax.get_ylim()
    span = max(high - low, 1e-9)
    ax.set_ylim(low, high + .16 * span)
    y = high + .01 * span
    ax.plot([0, 0, 1, 1], [y - .02 * span, y, y, y - .02 * span],
            color=NS_COLOR, lw=.8, clip_on=False)
    ax.text(.5, y + .015 * span, f"n.s. (P={p_value:.2f})", ha="center",
            va="bottom", fontsize=8.2, color=NS_COLOR)


def distinct_generated_columns(ax) -> int:
    """Count distinct plotted sequences in one drawn ``Generated`` heat map."""
    image = ax.get_images()[0]
    matrix = np.ma.filled(np.asarray(image.get_array(), float), np.nan)
    matrix = np.where(np.isfinite(matrix), matrix, -1.0)
    columns = np.ascontiguousarray(matrix.T)
    return len({column.tobytes() for column in columns})


def annotated_full_tissue_graph(frozen):
    def wrapped(ax, out, fit_id, canonical_root):
        frozen(ax, out, fit_id, canonical_root)
        with np.load(out / "per_fit" / fit_id / base.L3 / "seed0" / "graph.npz",
                     allow_pickle=False) as graph:
            n_added = int(np.asarray(graph["added_mask"]).astype(bool).sum())
        # Label below each strip: the |h| inset colourbar occupies the upper
        # right, so a label above the output strip would overprint it.
        for x_position, text in ((-.057, "Input\nrank"), (.9825, "Generated\nrank")):
            ax.text(x_position, .23, text, transform=ax.transAxes, ha="center",
                    va="top", fontsize=7, linespacing=.95, color=base.DARK)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_texts()[1].set_text(f"Selected shortcut (3 of {n_added} drawn)")
        wrapped.stats = {"added_edges": n_added, "shortcuts_drawn": 3}
    wrapped.stats = {}
    return wrapped


def annotated_event_reproduction(frozen):
    def wrapped(fig, spec, out, old, canonical):
        before = len(fig.axes)
        frozen(fig, spec, out, old, canonical)
        # Creation order inside the frozen producer is (0,0) (0,1) (1,0) (1,1),
        # so index 1 and index 3 are the two ``Generated`` heat maps.
        created = fig.axes[before:]
        distinct = {
            "TA": distinct_generated_columns(created[1]),
            "TB": distinct_generated_columns(created[3]),
        }
        created[3].set_xlabel(
            f"30 held-out events per row; {distinct['TA']} (TA) / {distinct['TB']} (TB) "
            "distinct generated sequences",
            fontsize=7.4, labelpad=3,
        )
        wrapped.stats = {"plotted_events_per_row": 30, "distinct_generated": distinct}
    wrapped.stats = {}
    return wrapped


def relaid_cross_state_fields(fig, spec, out: Path, canonical: Path) -> dict:
    """Panel D with a shared timing colourbar, named x axes and a label gutter."""
    sub = spec.subgridspec(1, 6, width_ratios=(1, 1, .055, 1, .055, .34), wspace=.24)
    axes = [fig.add_subplot(sub[0, index]) for index in (0, 1, 3)]
    bars = [fig.add_subplot(sub[0, index]) for index in (2, 4)]
    field = json.loads((
        canonical / "results/interictal_propagation_masked/template_gradient_fields"
        / "per_subject" / f"{base.SUBJECT}.json"
    ).read_text())["interictal_field"]
    order = list(map(str, field["contact_order"]))
    with np.load(out / "model_fields/intact/per_patient" / base.SUBJECT
                 / f"{base.L3}.npz", allow_pickle=False) as model:
        names = model["contacts"].astype(str).tolist()
        take = np.asarray([names.index(name) for name in order])
        rank_a = 1.0 - np.asarray(model["A_canonical_full"], float)[take]
        rank_b = 1.0 - np.asarray(model["B_canonical_full"], float)[take]
        support_a = np.asarray(model["A_participation"], float)[take]
        support_b = np.asarray(model["B_participation"], float)[take]
    with np.load(out / "early_ictal/per_patient_targets" / f"{base.SUBJECT}.npz",
                 allow_pickle=False) as target:
        names = target["contacts"].astype(str).tolist()
        lookup = dict(zip(names, np.asarray(target["median_broadband_energy"], float)))
        energy = np.asarray([lookup[name] for name in order])
        n_seizures = int(target["n_seizures"])
    points, xlim, ylim = base.field_geometry(field)
    timing_image = base.draw_field(
        axes[0], points, rank_a, support_a, xlim, ylim, cmap=base.TIMING_CMAP,
        vmin=0, vmax=1, title="RNN TA", title_color=base.RED, show_y=True,
    )
    base.draw_field(
        axes[1], points, rank_b, support_b, xlim, ylim, cmap=base.TIMING_CMAP,
        vmin=0, vmax=1, title="RNN TB", title_color=base.BLUE, show_y=False,
    )
    energy_image = base.draw_field(
        axes[2], points, energy, np.ones_like(energy), xlim, ylim,
        cmap=base.ENERGY_CMAP, vmin=float(np.nanmin(energy)),
        vmax=float(np.nanmax(energy)), title="Early-ictal broadband",
        title_color=base.DARK, show_y=False,
    )
    for axis in axes:
        axis.set_xlabel("Propagation axis (mm)", fontsize=9)
    bar = fig.colorbar(timing_image, cax=bars[0], orientation="vertical")
    bar.set_ticks([0, 1], labels=["Early", "Late"])
    bar.ax.set_title("RNN\nrank", fontsize=8, pad=2, linespacing=.95)
    bar.ax.tick_params(labelsize=8, pad=1)
    bar = fig.colorbar(energy_image, cax=bars[1], orientation="vertical")
    bar.ax.set_title("Energy\nz", fontsize=8, pad=2, linespacing=.95)
    bar.ax.tick_params(labelsize=8, pad=1)
    return {"subject": base.SUBJECT, "n_seizures": n_seizures,
            "timing_colourbar": "SHARED_BY_RNN_TA_AND_TB",
            "field_x_axis": "Propagation axis (mm)"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=r2.DEFAULT_OUT)
    args, _unknown = parser.parse_known_args()
    out = args.out_root.resolve()

    original_main = base.main
    frozen_graph = base.draw_full_tissue_graph
    frozen_events = base.draw_event_reproduction
    frozen_fields = base.draw_cross_state_fields
    graph_wrapper = annotated_full_tissue_graph(frozen_graph)
    events_wrapper = annotated_event_reproduction(frozen_events)

    def patched_main() -> None:
        frozen_early = base.draw_early_cohort
        frozen_mechanism = base.draw_mechanism_row

        def early_with_ns(fig, spec, unused_out):
            before = len(fig.axes)
            stats = frozen_early(fig, spec, unused_out)
            created = fig.axes[before:]
            mark_not_significant(created[0], float(stats["oracle_p_vs_null"]))
            mark_not_significant(created[1], float(stats["mixture_p_vs_null"]))
            scatter = created[2]
            kept = np.asarray([0.0, .05, .25, .60])
            scatter.set_xticks(np.sqrt(kept), ["0", ".05", ".25", ".60"])
            scatter.set_xlabel("Nonlocality J\n(sqrt scale)")
            return {
                **stats,
                "nonprimary_views_marked_not_significant": True,
                "primary_scatter_ticks": "SUBSET_OF_FROZEN_TICKS_SAME_LIMITS",
            }

        def mechanism_with_labels(fig, spec, unused_out):
            before = len(fig.axes)
            stats = frozen_mechanism(fig, spec, unused_out)
            created = fig.axes[before:]
            panel_g, panel_h = created[1], created[2]
            panel_g.set_ylabel("Selected benefit,\ndistal transitions (nats)")
            panel_g.set_xticks(range(3), ("vs Local", "vs Nearby", "vs Matched"))
            legend = panel_h.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    if text.get_text().startswith("Local ("):
                        text.set_text(text.get_text().replace("Local (", "Local in L3 (", 1))
            return {
                **stats,
                "panel_g_contrast_scope": "DISTAL_TRANSITIONS_ONLY",
                "panel_h_local_curve": "L3_MATCHED_LOCAL_BACKBONE_EDGES_NOT_THE_L0_ARM",
            }

        base.draw_full_tissue_graph = graph_wrapper
        base.draw_event_reproduction = events_wrapper
        base.draw_cross_state_fields = relaid_cross_state_fields
        base.draw_early_cohort = early_with_ns
        base.draw_mechanism_row = mechanism_with_labels
        original_main()

    base.main = patched_main
    saved_argv = sys.argv
    try:
        sys.argv = [str(Path(r2.__file__).resolve()), "--out-root", str(out)]
        r2.main()
    finally:
        sys.argv = saved_argv
        base.main = original_main
        base.draw_full_tissue_graph = frozen_graph
        base.draw_event_reproduction = frozen_events
        base.draw_cross_state_fields = frozen_fields

    figure = r2.DEFAULT_FIGURE.resolve()
    metadata = json.loads((figure / "FIGURE6_METADATA.json").read_text())
    for key, required in (
        ("panel_c", "v0.5_true_suffix_vs_split_matched_reassigned_suffix"),
        ("panel_e", "oracle_plus_train_prevalence_mixture_plus_primary_J_interaction"),
    ):
        if metadata.get(key, {}).get("contract") != required:
            raise RuntimeError(f"r3 render lost the frozen {key} contract")
    if not metadata["panel_e"].get("nonprimary_views_marked_not_significant"):
        raise RuntimeError("panel E non-primary views were not marked n.s.")
    if metadata["panels_f_i"].get("panel_g_contrast_scope") != "DISTAL_TRANSITIONS_ONLY":
        raise RuntimeError("panel G distal scope label missing")
    if metadata["panel_d"].get("timing_colourbar") != "SHARED_BY_RNN_TA_AND_TB":
        raise RuntimeError("panel D shared timing colourbar missing")
    if not graph_wrapper.stats or not events_wrapper.stats:
        raise RuntimeError("panel A/B annotations did not run")
    metadata["panel_a"] = graph_wrapper.stats
    metadata["panel_b"] = events_wrapper.stats
    metadata["finalizer_r3"] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": r2.sha256_file(Path(__file__).resolve()),
        "scope": "VISUAL_LEGIBILITY_ONLY_NO_ESTIMAND_OR_STAR_CHANGE",
        "frozen_producers_left_byte_identical": [
            "scripts/finalize_topic5_figure6_multiscale_scaffold_v0_5_r2.py",
            "scripts/paper_figures/plot_topic5_figure6_multiscale_scaffold_v0_5.py",
        ],
        "repairs": [
            "panel_A_input_and_generated_rank_strip_labels_and_drawn_shortcut_count",
            "panel_B_distinct_generated_sequence_disclosure",
            "panel_D_shared_timing_colourbar_x_axis_labels_and_energy_bar_gutter",
            "panel_E_non_primary_views_marked_not_significant",
            "panel_E_primary_scatter_tick_thinning_and_shorter_axis_name",
            "panel_G_distal_scope_label",
            "panel_H_local_curve_disambiguated_from_panel_G_local_arm",
        ],
    }
    r2.write_json(figure / "FIGURE6_METADATA.json", metadata)

    readme = figure / "README.md"
    text = readme.read_text()
    if DISCLOSURE not in text:
        text = text.replace("**关注点**：", DISCLOSURE + "**关注点**：", 1)
        readme.write_text(text)

    stem = figure / "topic5_figure6_multiscale_scaffold_v0_5"
    assets = {path.name: r2.sha256_file(path)
              for path in [stem.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg")]}
    r2.write_json(figure / "FIGURE6_COMPLETE.json", {
        "status": "COMPLETE_FINALIZED_R3", "assets_sha256": assets,
        "panel_c_decision_sha256": metadata["finalizer_r2"]["panel_c_decision_sha256"],
        "panel_e_decision_sha256": metadata["finalizer_r2"]["panel_e_decision_sha256"],
        "panel_i_decision_sha256": metadata["finalizer_r2"]["panel_i_decision_sha256"],
    })
    r2.write_json(out / "FIGURE6_FINAL_RENDER_COMPLETE.json", {
        "status": "PASS_R3", "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": True, "visual_changes_were_prefrozen": True,
        "postreview_visual_repairs": metadata["finalizer_r3"]["repairs"],
        "assets_sha256": assets,
    })
    print(json.dumps({"figure": str(stem.with_suffix(".png")),
                      "panel_a": metadata["panel_a"], "panel_b": metadata["panel_b"],
                      "assets": assets}, indent=2))


if __name__ == "__main__":
    main()
