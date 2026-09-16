#!/usr/bin/env python3
"""Render the compact Fig.4C Node x E-to-E x E-to-I ablation panel."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
    load_scoring_contract,
)
from src.topic4_nlc_pathway_mechanism import (  # noqa: E402
    ARM_IDS,
    bootstrap_mean,
    formal_mode_assignments,
    network_mode_endpoints,
)
from src.topic4_shaft_aware import contract_groups  # noqa: E402


HISTORICAL_CONFIG = ROOT / "config/topic4_rev11_nlc_frozen_substrate_confirmation.json"
DEFAULT_OUTPUT = (
    ROOT
    / "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc"
    / "frozen_substrate_confirmation/figures"
)
MAIN_FIGURES = ROOT / "results/paper-ready-figure/fig4/figures"
ARM_LABELS = ("Node", "+EE", "+E-to-I", "+EE+EI")
ARM_COLORS = ("#777777", "#D7892F", "#2A9D8F", "#202020")
TA_COLOR = "#C43C39"
TB_COLOR = "#277DA1"
DISPLAY_ENDPOINTS = {
    "Mode 1 share (%)": "TB_like_fraction",
    "Mode 2 share (%)": "TB_like_fraction",
    "KMeans match (%)": "natural_alignment",
    "OOD (%)": "ood_fraction_returned",
}


def _classifier(manifest):
    output = dict(manifest["direction_classifier"])
    for key in (
        "coef", "class_centers", "class_precisions", "ood_distance_thresholds",
    ):
        output[key] = np.asarray(output[key], float)
    return output


def _historical_rows(config_path, root):
    config = json.loads(config_path.read_text())
    manifest = json.loads((root / "candidate_manifest.json").read_text())
    verdict = json.loads((root / "confirmation_verdict.json").read_text())
    contract = json.loads(
        (ROOT / config["inputs"]["contact_contract"]["path"]).read_text()
    )
    groups = contract_groups(contract)
    _, embedding, _, _ = load_scoring_contract(
        config["inputs"]["shaft_aware_target_npz"]["path"],
        config["inputs"]["shaft_aware_floors"]["path"],
        "FULL_TIMING", fixed_events_per_mode=6,
    )
    classifier = _classifier(manifest)
    natural = {
        row["candidate_id"]: row["network_objective_inputs"]
        for row in verdict["candidate_rows"]
    }
    seeds = list(map(int, config["search"]["confirmation_network_seeds"]))
    duration_ms = float(config["search"]["simulation"]["duration_ms"])
    rows = {arm_id: {} for arm_id in ARM_IDS}
    for arm_id in ARM_IDS:
        for seed in seeds:
            path = root / "workers" / f"{arm_id}_seed_{seed}.npz"
            with np.load(path, allow_pickle=False) as loaded:
                onsets = np.asarray(loaded["onsets"], float)
                returned = np.asarray(loaded["event_returned"], bool)
            assigned = formal_mode_assignments(
                onsets, returned, groups=groups, embedding=embedding,
                classifier=classifier, minimum_recruited_contacts=3,
            )
            assigned["returned"] = returned
            endpoints = network_mode_endpoints(assigned, duration_ms)
            rows[arm_id][str(seed)] = {
                **endpoints,
                "natural_alignment": natural[arm_id][str(seed)]["natural"],
            }
    return rows, seeds, {
        "endpoint_status": "POST_HOC_ON_FROZEN_EXACT_ABLATION",
        "source_verdict": str((root / "confirmation_verdict.json").relative_to(ROOT)),
        "new_confirmation_required_for_main_claim": True,
    }, None


def _mechanism_rows(root):
    verdict_path = root / "mechanism_verdict.json"
    verdict = json.loads(verdict_path.read_text())
    if not verdict.get("figure_eligible"):
        raise RuntimeError("pathway mechanism result is not figure eligible")
    return verdict["per_network"], verdict["network_seeds"], {
        "endpoint_status": "PRE_REGISTERED_INDEPENDENT_CONFIRMATION",
        "source_verdict": str(verdict_path.relative_to(ROOT)),
        "new_confirmation_required_for_main_claim": False,
    }, verdict["paired_differences_vs_node"]


def _metric_arrays(rows, seeds):
    mode_1 = np.asarray([
        [rows[arm][str(seed)]["TA_like_count"] for arm in ARM_IDS]
        for seed in seeds
    ], float)
    mode_2 = np.asarray([
        [rows[arm][str(seed)]["TB_like_count"] for arm in ARM_IDS]
        for seed in seeds
    ], float)
    total = mode_1 + mode_2
    mode_1_share = np.divide(
        100.0 * mode_1, total, out=np.full_like(total, np.nan), where=total > 0,
    )
    mode_2_share = np.divide(
        100.0 * mode_2, total, out=np.full_like(total, np.nan), where=total > 0,
    )
    return {
        "Mode 1 share (%)": mode_1_share,
        "Mode 2 share (%)": mode_2_share,
        "KMeans match (%)": 100.0 * np.asarray([
            [rows[arm][str(seed)]["natural_alignment"] for arm in ARM_IDS]
            for seed in seeds
        ], float),
        "OOD (%)": 100.0 * np.asarray([
            [rows[arm][str(seed)]["ood_fraction_returned"] for arm in ARM_IDS]
            for seed in seeds
        ], float),
    }


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7.0,
        "axes.titlesize": 8.0,
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.4,
        "ytick.labelsize": 6.4,
        "axes.linewidth": 0.65,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def _significant_node_arms(title, paired_differences):
    """Return non-Node arm indices whose paired 90% CI excludes zero."""
    if paired_differences is None:
        return []
    endpoint = DISPLAY_ENDPOINTS[title]
    significant = []
    for arm_index, arm_id in enumerate(ARM_IDS[1:], start=1):
        result = paired_differences[arm_id][endpoint]
        if result.get("status") != "OK":
            continue
        if float(result["q05"]) > 0.0 or float(result["q95"]) < 0.0:
            significant.append(arm_index)
    return significant


def _draw_significance_bracket(axis, arm_index, level):
    y = 103.0 + 8.0 * level
    height = 2.0
    axis.plot(
        [0.0, 0.0, float(arm_index), float(arm_index)],
        [y, y + height, y + height, y],
        color="#333333", lw=0.7, clip_on=False, zorder=5,
    )
    axis.text(
        0.5 * arm_index, y + height + 0.8, "*",
        ha="center", va="bottom", fontsize=8.0, color="#222222", zorder=6,
    )


def _draw_axis(axis, values, title, *, draws, seed, significant_arms=()):
    x = np.arange(len(ARM_IDS), dtype=float)
    for row in values:
        finite = np.isfinite(row)
        axis.plot(
            x[finite], row[finite], color="#B8B8B8", lw=0.45,
            alpha=0.48, zorder=1,
        )
        axis.scatter(
            x[finite], row[finite], s=7, facecolor="white",
            edgecolor="#8E8E8E", linewidth=0.35, alpha=0.78, zorder=2,
        )
    for index, color in enumerate(ARM_COLORS):
        summary = bootstrap_mean(
            values[:, index], draws=draws, seed=seed + index,
        )
        axis.errorbar(
            index, summary["mean"],
            yerr=[[summary["mean"] - summary["q05"]],
                  [summary["q95"] - summary["mean"]]],
            fmt="o", ms=4.0, color=color, ecolor=color,
            elinewidth=1.0, capsize=2.0, capthick=0.8, zorder=4,
        )
    axis.set_title(title, loc="left", weight="bold", pad=3.0)
    axis.set_xticks(x, ARM_LABELS, rotation=28, ha="right", rotation_mode="anchor")
    axis.tick_params(length=2.2, width=0.55, pad=1.5)
    axis.spines["left"].set_color("#777777")
    axis.spines["bottom"].set_color("#777777")
    axis.margins(x=0.10)
    for level, arm_index in enumerate(significant_arms):
        _draw_significance_bracket(axis, arm_index, level)


def _save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".png"), dpi=600, facecolor="white",
        bbox_inches="tight", pad_inches=0.025,
    )
    fig.savefig(
        stem.with_suffix(".pdf"), facecolor="white",
        bbox_inches="tight", pad_inches=0.025,
    )


def _install_main(stem, metadata):
    MAIN_FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        shutil.copy2(stem.with_suffix(suffix), MAIN_FIGURES / f"fig4-panelc{suffix}")
    (MAIN_FIGURES / "fig4-panelc-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    registry_path = MAIN_FIGURES.parent / "figure4_panel_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        registry["status"] = (
            "CANDIDATE_WITH_CONFIRMED_NLC_PATHWAY_PANEL"
            if not metadata["source"]["new_confirmation_required_for_main_claim"]
            else "CANDIDATE_WITH_POSTHOC_NLC_PATHWAY_PANEL"
        )
        registry.setdefault("panel_contract", {})["c"] = (
            "paired Node / +EE / +E-to-I / +EE+EI pathway ablation; "
            "Mode 1/2 shares, natural KMeans match and OOD"
        )
        registry.setdefault("panel_metadata", {})["c"] = (
            "results/paper-ready-figure/fig4/figures/fig4-panelc-metadata.json"
        )
        registry["replacement_contract"] = {
            "target": "independent frozen-endpoint pathway confirmation",
            "layout_and_filenames_stable": True,
            "current_result_is_final": not metadata["source"][
                "new_confirmation_required_for_main_claim"
            ],
        }
        registry_path.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False) + "\n"
        )
    required = [MAIN_FIGURES / f"fig4-panel{panel}.png" for panel in "abcdefg"]
    if all(path.exists() for path in required):
        from scripts.paper_figures.build_main_figures_1_2 import (  # noqa: E402
            _compose_complete_layout,
        )
        _compose_complete_layout(
            figures_dir=MAIN_FIGURES,
            stem="fig4-complete-layout",
            canvas_size=(8000, 5900),
            placements={
                "a": (MAIN_FIGURES / "fig4-panela.png", (160, 160, 2000, 1750)),
                "b": (MAIN_FIGURES / "fig4-panelb.png", (2140, 160, 3900, 1750)),
                "c": (MAIN_FIGURES / "fig4-panelc.png", (4040, 160, 7840, 1750)),
                "d": (MAIN_FIGURES / "fig4-paneld.png", (160, 2050, 4430, 3570)),
                "e": (MAIN_FIGURES / "fig4-panele.png", (4590, 2050, 7840, 3570)),
                "f": (MAIN_FIGURES / "fig4-panelf.png", (160, 3890, 4430, 5710)),
                "g": (MAIN_FIGURES / "fig4-panelg.png", (4590, 3890, 7840, 5710)),
            },
            labels={
                "A": (35, 25), "B": (2015, 25), "C": (3915, 25),
                "D": (35, 1915), "E": (4465, 1915),
                "F": (35, 3755), "G": (4465, 3755),
            },
            anchors={key: "top" for key in "abcdefg"},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HISTORICAL_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--install-main", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    if (root / "mechanism_verdict.json").exists():
        rows, seeds, source, paired_differences = _mechanism_rows(root)
        stem_name = "fig4c_nlc_pathway_ablation_confirmation"
    else:
        rows, seeds, source, paired_differences = _historical_rows(config_path, root)
        stem_name = "fig4c_nlc_pathway_ablation_posthoc"
    metrics = _metric_arrays(rows, seeds)
    annotated_comparisons = {
        title: [ARM_IDS[index] for index in _significant_node_arms(
            title, paired_differences,
        )]
        for title in metrics
    }
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(7.15, 2.62))
    for index, (title, values) in enumerate(metrics.items()):
        _draw_axis(
            axes[index], values, title, draws=4096,
            seed=20260820 + 20 * index,
            significant_arms=_significant_node_arms(title, paired_differences),
        )
    axes[0].title.set_color(TA_COLOR)
    axes[1].title.set_color(TB_COLOR)
    for axis in axes:
        axis.set_ylim(0.0, 120.0)
    fig.text(
        0.5, 0.018,
        "* paired vs Node: 90% network-bootstrap CI excludes 0; "
        "n=12, 4,096 draws; no multiplicity correction",
        ha="center", va="bottom", fontsize=5.2, color="#555555",
    )
    fig.subplots_adjust(left=0.055, right=0.995, bottom=0.29, top=0.91, wspace=0.42)
    stem = Path(args.output_dir).resolve() / stem_name
    _save(fig, stem)
    plt.close(fig)
    metadata = {
        "status": "FIG4C_NLC_PATHWAY_ABLATION_RENDERED",
        "source": source,
        "config": str(config_path.relative_to(ROOT)),
        "network_seeds": list(map(int, seeds)),
        "n_paired_networks": len(seeds),
        "arm_order": list(ARM_IDS),
        "arm_labels": list(ARM_LABELS),
        "event_labels": {
            "Mode 1": "frozen patient classifier label 0; no pathological meaning",
            "Mode 2": "frozen patient classifier label 1; no pathological meaning",
        },
        "display_metrics": {
            "Mode 1 share (%)": "Mode 1 / (Mode 1 + Mode 2) among formal events",
            "Mode 2 share (%)": "Mode 2 / (Mode 1 + Mode 2) among formal events",
            "KMeans match (%)": "balanced alignment of de novo KMeans K=2 clusters with frozen Mode 1/2 labels",
            "OOD (%)": "fraction of returned events outside frozen patient support",
        },
        "natural_clusters_are_not_relabelled_as_patient_modes": True,
        "statistics": {
            "summary": "equal-network mean and 90% network bootstrap interval; raw paired networks shown",
            "star_rule": "paired arm-minus-Node 90% network-bootstrap CI excludes zero",
            "bootstrap_draws": 4096,
            "multiplicity_correction": "none",
            "annotated_arm_vs_node": annotated_comparisons,
        },
        "canvas_has_no_claim_text_or_internal_panel_letter": True,
        "outputs": [
            str(stem.with_suffix(".png").relative_to(ROOT)),
            str(stem.with_suffix(".pdf").relative_to(ROOT)),
        ],
    }
    stem.with_name(stem.name + "_metadata").with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    if args.install_main:
        _install_main(stem, metadata)
    print(json.dumps({
        "status": metadata["status"],
        "endpoint_status": source["endpoint_status"],
        "output": str(stem.with_suffix(".png")),
        "installed_main": bool(args.install_main),
    }, indent=2))


if __name__ == "__main__":
    main()
