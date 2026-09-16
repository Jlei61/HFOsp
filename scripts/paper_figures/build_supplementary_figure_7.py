#!/usr/bin/env python3
"""Build paper-ready Supplementary Figure 7 from the frozen NLC ablation."""
from __future__ import annotations

import argparse
import hashlib
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
DEFAULT_SOURCE = (
    ROOT
    / "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc"
    / "pathway_mechanism_confirmation"
)
DEFAULT_OUTPUT = (
    ROOT
    / "results/paper-ready-figure/supp_fig7_nlc_pathway_confirmation"
)
KMEANS_SOURCE_ROOT = (
    ROOT
    / "results/paper-ready-figure/archive/2026-09-03_pre_fig4_a_e_reorder"
    / "fig4"
)
KMEANS_SOURCE_FIGURES = KMEANS_SOURCE_ROOT / "figures"
KMEANS_SOURCE_REGISTRY = KMEANS_SOURCE_ROOT / "figure4_panel_registry.json"

from scripts.paper_figures.plot_fig4c_nlc_pathway_ablation import (  # noqa: E402
    ARM_IDS,
    TA_COLOR,
    TB_COLOR,
    _draw_axis,
    _mechanism_rows,
    _metric_arrays,
    _significant_node_arms,
)
from src.supplementary_figure_style import (  # noqa: E402
    apply_supplementary_rcparams,
    normalize_axis_text,
)


PANEL_KEYS = ("a", "b", "c", "d")
PANEL_LETTERS = ("A", "B", "C", "D")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _style() -> None:
    apply_supplementary_rcparams()


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.035,
    )
    fig.savefig(
        stem.with_suffix(".pdf"),
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.035,
    )


def _format_axis(axis: plt.Axes, title: str, values: np.ndarray, index: int, paired) -> None:
    _draw_axis(
        axis,
        values,
        title,
        draws=4096,
        seed=20260820 + 20 * index,
        significant_arms=_significant_node_arms(title, paired),
    )
    axis.set_ylim(0.0, 120.0)
    axis.set_yticks(np.arange(0.0, 121.0, 20.0))
    if index == 0:
        axis.title.set_color(TA_COLOR)
    elif index == 1:
        axis.title.set_color(TB_COLOR)


def _render_panels(metrics, paired, figures_dir: Path) -> list[Path]:
    outputs: list[Path] = []
    for index, ((title, values), panel) in enumerate(zip(metrics.items(), PANEL_KEYS)):
        fig, axis = plt.subplots(figsize=(3.20, 2.25))
        _format_axis(axis, title, values, index, paired)
        # These atomic panels are reduced once more by the fixed raster
        # compositor; compensate so their final visible type matches panel E.
        normalize_axis_text(axis, scale=1.45)
        axis.title.set_fontsize(11.6)
        fig.subplots_adjust(left=0.16, right=0.98, bottom=0.25, top=0.86)
        stem = figures_dir / f"supp_fig7-panel{panel}"
        _save(fig, stem)
        plt.close(fig)
        outputs.extend([stem.with_suffix(".png"), stem.with_suffix(".pdf")])
    return outputs


def _install_kmeans_panel(figures_dir: Path, output_dir: Path) -> tuple[list[Path], dict]:
    registry = json.loads(KMEANS_SOURCE_REGISTRY.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "paper_figure4_combined_a_g_v11":
        raise ValueError("unexpected KMeans source registry schema")
    outputs: list[Path] = []
    for suffix in (".png", ".pdf"):
        source = KMEANS_SOURCE_FIGURES / f"fig4-panele{suffix}"
        target = figures_dir / f"supp_fig7-panele{suffix}"
        shutil.copy2(source, target)
        outputs.append(target)
    (output_dir / "supp_fig7_panele_pairwise_similarity_statistics.json").unlink(
        missing_ok=True
    )
    details = registry["panel_details"]["e"]
    return outputs, {
        "source_registry": str(KMEANS_SOURCE_REGISTRY.relative_to(ROOT)),
        "source_registry_sha256": _sha256(KMEANS_SOURCE_REGISTRY),
        "mode_counts": details["event_counts_MTA_MTB"],
        "event_subset": registry["kmeans_event_subset"],
        "former_main_panel": "Figure 4E in the pre-reorder A-G package",
        "display": "masked-rank heatmap, per-contact rank distribution, one shared colorbar",
    }


def _render_complete(figures_dir: Path) -> list[Path]:
    from scripts.paper_figures.build_main_figures_1_2 import (
        _compose_complete_layout,
    )

    paths = _compose_complete_layout(
        figures_dir=figures_dir,
        stem="supp_fig7-complete-layout",
        canvas_size=(9000, 4900),
        placements={
            "a": (figures_dir / "supp_fig7-panela.png", (120, 180, 2860, 2250)),
            "b": (figures_dir / "supp_fig7-panelb.png", (3070, 180, 5810, 2250)),
            "c": (figures_dir / "supp_fig7-panelc.png", (6020, 180, 8760, 2250)),
            "d": (figures_dir / "supp_fig7-paneld.png", (1450, 2700, 4250, 4750)),
            "e": (figures_dir / "supp_fig7-panele.png", (4000, 2700, 8780, 4750)),
        },
        labels={
            "A": (20, 20), "B": (2970, 20), "C": (5920, 20),
            "D": (1050, 2540), "E": (3900, 2540),
        },
        anchors={"a": "top", "b": "top", "c": "top", "d": "top", "e": "top"},
        label_font_size=132,
    )
    return [ROOT / path for path in paths]


def _write_readme(figures_dir: Path) -> Path:
    readme = figures_dir / "README.md"
    readme.write_text(
        """### supp_fig7-complete-layout.png / .pdf

**Supplementary Fig. 7 | Frozen local-connectivity ablation and model-event KMeans structure.**

**A,** Mode 1 share among classified formal events, defined as Mode 1/(Mode 1 + Mode 2). **B,** Complementary Mode 2 share. **C,** Balanced alignment between de novo KMeans K = 2 clusters and the frozen Mode 1/2 labels. The natural clusters were not relabelled as patient modes. **D,** Fraction of returned events outside the frozen patient-distribution support (out of distribution, OOD). Across A–D, Node, +EE, +E-to-I and +EE+EI were evaluated for 20 s in each of 12 paired network seeds (1581–1592) under frozen topology, delays and separately conserved incoming-weight budgets. Open circles denote individual networks, grey lines connect the same network seed across arms, filled circles show equal-network means and error bars show 90% network-bootstrap confidence intervals (CIs). Stars indicate that the paired arm-minus-Node 90% network-bootstrap CI excluded zero (4,096 resamples; no multiplicity correction): +E-to-I in A and B, +EE+EI in C, and +E-to-I and +EE+EI in D. **E,** Masked recruitment-rank heatmap for 627 formal clean model events, grouped by frozen MTA and MTB KMeans labels, with aligned per-contact rank distributions and one shared first-to-last color scale. Mode 1/2 and MTA/MTB are frozen model labels and have no independent pathological interpretation. These analyses estimate development-case model-internal pathway effects and event structure; they do not establish patient causal connectivity, anatomical-core recovery or patient-blind/real-geometry generalization.

**关注点**：A–D 的统计单位是 network seed（n = 12），不是事件；E 的 627 列是 pooled model events，不是独立患者或网络。图只支持 development-case 模型内部 pathway effect 与事件结构。

### supp_fig7-panela.png / .pdf

Supplementary Fig. 7A 的无角标独立导出，展示 Mode 1 事件占比；图形元素与统计定义以上方完整图注为准。

**关注点**：完整投稿图请使用带 A–E 角标的 `supp_fig7-complete-layout`。

### supp_fig7-panelb.png / .pdf

Supplementary Fig. 7B 的无角标独立导出，展示与 A 互补的 Mode 2 事件占比；图形元素与统计定义以上方完整图注为准。

**关注点**：Mode 1/2 是冻结分类器标签，不是临床病理亚型。

### supp_fig7-panelc.png / .pdf

Supplementary Fig. 7C 的无角标独立导出，展示 de novo KMeans K = 2 与冻结 Mode 1/2 标签的 balanced match。

**关注点**：自然簇未被重命名为患者模式；联合臂 match 的下降不能写成患者模式几何恢复。

### supp_fig7-paneld.png / .pdf

Supplementary Fig. 7D 的无角标独立导出，展示返回事件的 OOD 比例；图形元素与统计定义以上方完整图注为准。

**关注点**：较低 OOD 不代表真实几何或患者外泛化。

### supp_fig7-panele.png / .pdf

原主图 KMeans panel 的无角标独立导出。热图展示 627 个 formal clean model events 的 masked recruitment ranks，并按冻结 MTA/MTB 标签分组；右侧为对齐的逐触点 rank distribution 与唯一共享色条。

**关注点**：MTA `n=437`、MTB `n=190` 是 pooled model events，不是患者数或独立网络数。
""",
        encoding="utf-8",
    )
    return readme


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    verdict_path = source_root / "mechanism_verdict.json"
    config_path = ROOT / "config/topic4_rev11_nlc_pathway_mechanism_confirmation.json"
    if not verdict_path.exists():
        raise FileNotFoundError(verdict_path)

    rows, seeds, source, paired = _mechanism_rows(source_root)
    if len(seeds) != 12:
        raise RuntimeError(f"FigS7 contract requires 12 paired networks, found {len(seeds)}")
    metrics = _metric_arrays(rows, seeds)
    mode_sum = metrics["Mode 1 share (%)"] + metrics["Mode 2 share (%)"]
    if not np.allclose(mode_sum[np.isfinite(mode_sum)], 100.0):
        raise RuntimeError("Mode 1 and Mode 2 shares do not sum to 100%")

    _style()
    outputs = _render_panels(metrics, paired, figures_dir)
    kmeans_outputs, kmeans_details = _install_kmeans_panel(figures_dir, output_dir)
    outputs.extend(kmeans_outputs)
    outputs.extend(_render_complete(figures_dir))
    readme = _write_readme(figures_dir)

    annotated = {
        title: [ARM_IDS[index] for index in _significant_node_arms(title, paired)]
        for title in metrics
    }
    metadata = {
        "schema_version": "supplementary_figure7_nlc_pathway_and_kmeans_v3",
        "status": "SUPPLEMENTARY_FIGURE_7_REVISION_READY",
        "asset_id": "data_driven_snn_nlc_pathway_confirmation",
        "paper_slot": "FigS7-A-E",
        "source": {
            **source,
            "root": str(source_root.relative_to(ROOT)),
            "mechanism_verdict_sha256": _sha256(verdict_path),
            "config": str(config_path.relative_to(ROOT)),
            "config_sha256": _sha256(config_path),
        },
        "network_seeds": list(map(int, seeds)),
        "n_paired_networks": len(seeds),
        "independent_unit": "network seed",
        "arm_order": list(ARM_IDS),
        "panel_contract": {
            "a": "Mode 1 share (%)",
            "b": "Mode 2 share (%)",
            "c": "de novo KMeans K=2 balanced match with frozen Mode 1/2 labels (%)",
            "d": "returned-event OOD fraction (%)",
            "e": "masked-rank KMeans heatmap and per-contact rank distribution",
        },
        "statistics": {
            "summary": "equal-network mean and 90% network-bootstrap interval; raw paired networks shown",
            "star_rule": "paired arm-minus-Node 90% network-bootstrap CI excludes zero",
            "bootstrap_draws": 4096,
            "multiplicity_correction": "none",
            "annotated_arm_vs_node": annotated,
            "panel_e_kmeans": kmeans_details,
        },
        "scientific_boundary": (
            "development-only, model-internal static-pathway effect pattern; not patient causal "
            "connectivity, anatomical-core recovery, or real-geometry/patient-blind generalization"
        ),
        "panel_files_have_internal_letters": False,
        "complete_layout_has_panel_letters": True,
        "visual_status": "CODEX_RENDER_QA_PASSED_PENDING_AUTHOR_ACCEPTANCE",
        "outputs": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in outputs
        },
        "readme": str(readme.relative_to(ROOT)),
    }
    metadata_path = output_dir / "supp_fig7_nlc_pathway_confirmation_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": metadata["status"],
        "complete_layout": str((figures_dir / "supp_fig7-complete-layout.png")),
        "metadata": str(metadata_path),
    }, indent=2))


if __name__ == "__main__":
    main()
