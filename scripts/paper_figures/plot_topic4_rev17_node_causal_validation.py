#!/usr/bin/env python3
"""Render rev17 source-topology and same-checkpoint intervention evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures import plot_topic4_rev17_node_final_fig4 as final_fig4  # noqa: E402


ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
DEFAULT_CONFIG = final_fig4.DEFAULT_CONFIG
DEFAULT_POSTSELECTION = final_fig4.DEFAULT_AUDIT
DEFAULT_FINAL_SCIENCE = final_fig4.DEFAULT_FINAL_AUDIT
DEFAULT_INTERVENTION = final_fig4.DEFAULT_INTERVENTION_AGGREGATE
DEFAULT_FREEZE = final_fig4.DEFAULT_FREEZE_MANIFEST
DEFAULT_OUTPUT = ARTIFACT_ROOT / (
    "results/paper-ready-figure/fig4_rev17_node_causal_validation/figures"
)
MODE_NAMES = {0: "MTB", 1: "MTA"}
MODE_COLORS = {0: "#2783B8", 1: "#C43B49"}
SEMANTIC_MODE_ORDER = (1, 0)
EFFECT_KEYS = (
    ("own", "own_mode_effect_event_abolished_then_delay"),
    ("other", "opposite_mode_effect_event_abolished_then_delay"),
    ("matched", "matched_control_effect_event_abolished_then_delay"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"rev17 causal-validation input is missing: {path}")
    return json.loads(path.read_text())


def _resolve_record(record: Mapping[str, Any], artifact_root: Path) -> Path:
    raw = Path(str(record.get("path", "")))
    candidates = [raw] if raw.is_absolute() else [artifact_root / raw, ROOT / raw]
    for path in candidates:
        if path.is_file() and _sha256(path) == str(record.get("sha256")):
            return path.resolve()
    raise RuntimeError(f"rev17 causal-validation input changed: {raw}")


def intervention_effect_records(selectivity: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Keep event loss and surviving-event delay as separate endpoints."""
    if selectivity.get("primary_ordered_effect") != (
        "native supported-mode loss first, then nonnegative onset delay"
    ):
        raise RuntimeError("rev17 intervention endpoint contract changed")
    records: list[dict[str, Any]] = []
    for hotspot_mode in SEMANTIC_MODE_ORDER:
        mode = selectivity.get("modes", {}).get(str(hotspot_mode), {})
        for network in mode.get("per_network", []):
            for effect_name, key in EFFECT_KEYS:
                value = network.get(key)
                if not isinstance(value, list) or len(value) != 2:
                    raise RuntimeError("rev17 intervention ordered effect is malformed")
                lost = int(value[0])
                delay = float(value[1])
                if lost not in (0, 1) or delay < 0.0 or not np.isfinite(delay):
                    raise RuntimeError("rev17 intervention ordered effect is invalid")
                if lost and delay != 0.0:
                    raise RuntimeError("abolished events cannot carry an onset delay")
                records.append({
                    "hotspot_mode": hotspot_mode,
                    "hotspot_name": MODE_NAMES[hotspot_mode],
                    "network_seed": int(network["network_seed"]),
                    "effect": effect_name,
                    "event_lost": bool(lost),
                    "delay_if_retained_ms": None if lost else delay,
                    "selective_network": bool(network["selective"]),
                })
    if not records:
        raise RuntimeError("rev17 intervention has no crossed effects")
    return records


def topology_plot_records(final_science: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose observed topology quality beside each matched label null."""
    records = []
    for label, key in (
        ("Exact anchor", "reference_source_topology_permutation"),
        ("Rev17 field", "source_topology_permutation"),
    ):
        row = final_science.get(key, {})
        values = {
            "observed": float(row.get("observed_weakest_mode_quality", np.nan)),
            "q05": float(row.get("null_q05", np.nan)),
            "q50": float(row.get("null_q50", np.nan)),
            "q95": float(row.get("null_q95", np.nan)),
            "p": float(row.get("upper_tail_p", np.nan)),
        }
        if not np.isfinite(list(values.values())).all():
            raise RuntimeError("rev17 source-topology audit is incomplete")
        if not values["q05"] <= values["q50"] <= values["q95"]:
            raise RuntimeError("rev17 source-topology null quantiles are unordered")
        records.append({"label": label, **values})
    return records


def _load_bundle(
    *, config_path: Path, postselection_path: Path, final_science_path: Path,
    intervention_path: Path, freeze_path: Path, artifact_root: Path,
) -> dict[str, Any]:
    config = _load_json(config_path)
    postselection = _load_json(postselection_path)
    final_science = _load_json(final_science_path)
    intervention = _load_json(intervention_path)
    freeze = _load_json(freeze_path)
    candidate_id = final_fig4.validate_final_figure_gate(
        config=config, postselection=postselection, final_science=final_science,
        intervention=intervention, freeze_manifest=freeze,
    )
    frozen_intervention = freeze.get("intervention_aggregate", {})
    if (
        Path(str(frozen_intervention.get("path", ""))).resolve()
        != intervention_path.resolve()
        or frozen_intervention.get("sha256") != _sha256(intervention_path)
    ):
        raise RuntimeError("rev17 causal-validation freeze does not bind the aggregate")

    worker_jsons, worker_arrays = [], []
    for record in intervention.get("inputs", {}).get("workers", []):
        json_path = _resolve_record(record["json"], artifact_root)
        npz_path = _resolve_record(record["npz"], artifact_root)
        payload = _load_json(json_path)
        if (
            payload.get("candidate_id") != candidate_id
            or payload.get("status")
            != "REV17_NODE_CROSSED_INTERVENTION_WORKER_COMPLETE"
            or payload.get("mechanism_freeze")
            != {"EE": "off", "E_to_I": "off", "Z_M": "off"}
            or payload.get("arrays", {}).get("sha256") != _sha256(npz_path)
        ):
            raise RuntimeError("rev17 causal-validation worker identity changed")
        worker_jsons.append(payload)
        with np.load(npz_path, allow_pickle=False) as loaded:
            worker_arrays.append({key: np.asarray(loaded[key]) for key in loaded.files})
    if len(worker_jsons) != len(intervention.get("network_seeds", [])):
        raise RuntimeError("rev17 causal-validation worker count changed")

    support = {
        mode: np.mean([
            np.asarray(arrays[f"mode{mode}_early_probability"], float)
            for arrays in worker_arrays
        ], axis=0)
        for mode in SEMANTIC_MODE_ORDER
    }
    if support[0].shape != support[1].shape or support[0].ndim != 2:
        raise RuntimeError("rev17 mode-specific source maps do not align")
    targets = {
        mode: {
            "hotspot": np.asarray([
                payload["targets"][str(mode)]["dominant"]["xy_mm"]
                for payload in worker_jsons
            ], float),
            "matched": np.asarray([
                payload["targets"][str(mode)]["matched_off_template"]["xy_mm"]
                for payload in worker_jsons
            ], float),
        }
        for mode in SEMANTIC_MODE_ORDER
    }
    return {
        "candidate_id": candidate_id,
        "support": support,
        "targets": targets,
        "topology": topology_plot_records(final_science),
        "effects": intervention_effect_records(intervention["selectivity"]),
        "network_seeds": [int(value) for value in intervention["network_seeds"]],
        "input_paths": {
            "postselection": postselection_path,
            "final_science": final_science_path,
            "intervention": intervention_path,
            "freeze": freeze_path,
            "workers": [Path(row["json"]["path"]) for row in intervention["inputs"]["workers"]],
        },
    }


def _panel_letter(ax: plt.Axes, letter: str, *, x: float = -0.13) -> None:
    ax.text(x, 1.17, letter, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="top", ha="left")


def _draw_support(ax: plt.Axes, bundle: Mapping[str, Any], mode: int,
                  *, vmax: float) -> Any:
    source = np.asarray(bundle["support"][mode], float)
    image = ax.imshow(
        source, origin="lower", extent=(0, 20, 0, 20), cmap="magma",
        vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal",
    )
    targets = bundle["targets"][mode]
    ax.scatter(
        targets["hotspot"][:, 0], targets["hotspot"][:, 1], s=38,
        facecolor="none", edgecolor=MODE_COLORS[mode], linewidth=1.2,
        label="hotspot",
    )
    ax.scatter(
        targets["matched"][:, 0], targets["matched"][:, 1], s=24,
        color="#656D72", marker="x", linewidth=1.0, label="matched",
    )
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="x (mm)")
    ax.set_title(f"{MODE_NAMES[mode]} early support", color=MODE_COLORS[mode],
                 fontsize=8.5, fontweight="bold", pad=3)
    ax.tick_params(labelsize=6.5, length=2)
    return image


def _draw_topology(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    colors = ("#90979B", "#2B8C6B")
    for x, (row, color) in enumerate(zip(rows, colors)):
        ax.vlines(x, row["q05"], row["q95"], color="#B5BABD", linewidth=7,
                  zorder=1)
        ax.hlines(row["q50"], x - 0.13, x + 0.13, color="#525A5E",
                  linewidth=1.0, zorder=2)
        ax.scatter(x, row["observed"], s=42, color=color, edgecolor="white",
                   linewidth=0.7, zorder=3)
    ax.set_xticks(range(len(rows)), [row["label"] for row in rows])
    ax.set_ylabel("Weakest-mode topology quality", fontsize=7.3)
    ax.set_title("Source topology vs label null", fontsize=8.5,
                 fontweight="bold", pad=3)
    ax.tick_params(axis="x", labelsize=6.6, length=0)
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.35, y=0.12)


def _effect_x(mode: int, effect: str) -> float:
    block = 0 if mode == 1 else 4
    return float(block + {"own": 0, "other": 1, "matched": 2}[effect])


def _draw_effects(ax_loss: plt.Axes, ax_delay: plt.Axes,
                  rows: list[dict[str, Any]]) -> None:
    seeds = sorted({row["network_seed"] for row in rows})
    offsets = {seed: offset for seed, offset in zip(seeds, (-0.10, 0.0, 0.10))}
    for mode in SEMANTIC_MODE_ORDER:
        mode_rows = [row for row in rows if row["hotspot_mode"] == mode]
        color = MODE_COLORS[mode]
        for seed in seeds:
            selected = [row for row in mode_rows if row["network_seed"] == seed]
            selected.sort(key=lambda row: ("own", "other", "matched").index(row["effect"]))
            xs = np.asarray([_effect_x(mode, row["effect"]) + offsets[seed]
                             for row in selected])
            losses = np.asarray([float(row["event_lost"]) for row in selected])
            ax_loss.plot(xs, losses, color=color, alpha=0.18, linewidth=0.7)
            ax_loss.scatter(xs, losses, color=color, s=19, alpha=0.82,
                            edgecolor="white", linewidth=0.35, zorder=3)
            retained = [row for row in selected if not row["event_lost"]]
            if retained:
                dx = np.asarray([_effect_x(mode, row["effect"]) + offsets[seed]
                                 for row in retained])
                dy = np.asarray([row["delay_if_retained_ms"] for row in retained], float)
                ax_delay.plot(dx, dy, color=color, alpha=0.18, linewidth=0.7)
                ax_delay.scatter(dx, dy, color=color, s=19, alpha=0.82,
                                 edgecolor="white", linewidth=0.35, zorder=3)

    positions = [0, 1, 2, 4, 5, 6]
    labels = ["own", "other", "matched"] * 2
    for ax in (ax_loss, ax_delay):
        ax.set_xticks(positions, labels, rotation=28, ha="right")
        ax.tick_params(axis="x", labelsize=6.2, length=0)
        ax.tick_params(axis="y", labelsize=6.5, length=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.axvline(3, color="#D6DADC", linewidth=0.6)
        ax.text(1, 1.01, "MTA hotspot", color=MODE_COLORS[1], fontsize=6.8,
                ha="center", va="bottom", transform=ax.get_xaxis_transform())
        ax.text(5, 1.01, "MTB hotspot", color=MODE_COLORS[0], fontsize=6.8,
                ha="center", va="bottom", transform=ax.get_xaxis_transform())
    ax_loss.set_ylim(-0.18, 1.18)
    ax_loss.set_yticks([0, 1], ["retained", "lost"])
    ax_loss.set_title("Native-mode survival", fontsize=8.5,
                      fontweight="bold", pad=18)
    ax_delay.set_ylim(bottom=-2)
    ax_delay.set_ylabel("Onset delay if retained (ms)", fontsize=7.2)
    ax_delay.set_title("Delay among surviving events", fontsize=8.5,
                       fontweight="bold", pad=18)


def render(
    *, config_path: Path = DEFAULT_CONFIG,
    postselection_path: Path = DEFAULT_POSTSELECTION,
    final_science_path: Path = DEFAULT_FINAL_SCIENCE,
    intervention_path: Path = DEFAULT_INTERVENTION,
    freeze_path: Path = DEFAULT_FREEZE,
    output_dir: Path = DEFAULT_OUTPUT,
    artifact_root: Path = ARTIFACT_ROOT,
) -> dict[str, Any]:
    bundle = _load_bundle(
        config_path=config_path.resolve(),
        postselection_path=postselection_path.resolve(),
        final_science_path=final_science_path.resolve(),
        intervention_path=intervention_path.resolve(),
        freeze_path=freeze_path.resolve(), artifact_root=artifact_root.resolve(),
    )
    plt.rcParams.update({
        "font.family": "Arial", "font.size": 7.0, "axes.linewidth": 0.65,
        "xtick.major.width": 0.65, "ytick.major.width": 0.65,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    figure = plt.figure(figsize=(7.15, 5.0), constrained_layout=False)
    grid = figure.add_gridspec(
        2, 5, left=0.07, right=0.985, bottom=0.105, top=0.93,
        width_ratios=(1.0, 1.0, 0.075, 1.12, 1.12),
        height_ratios=(1.0, 0.78), wspace=0.48, hspace=0.58,
    )
    ax_mta = figure.add_subplot(grid[0, 0])
    ax_mtb = figure.add_subplot(grid[0, 1])
    cax = figure.add_subplot(grid[0, 2])
    ax_topology = figure.add_subplot(grid[0, 3:])
    ax_loss = figure.add_subplot(grid[1, :2])
    ax_delay = figure.add_subplot(grid[1, 3:])
    vmax = max(float(np.max(bundle["support"][mode])) for mode in SEMANTIC_MODE_ORDER)
    image = _draw_support(ax_mta, bundle, 1, vmax=vmax)
    _draw_support(ax_mtb, bundle, 0, vmax=vmax)
    ax_mta.set_ylabel("y (mm)")
    ax_mtb.set_yticklabels([])
    colorbar = figure.colorbar(image, cax=cax)
    colorbar.ax.set_title("Early support\nprobability", fontsize=6.2, pad=3)
    colorbar.ax.tick_params(labelsize=6, length=1.5)
    ax_mtb.legend(frameon=False, fontsize=5.8, loc="upper right",
                  handletextpad=0.3, borderpad=0.1)
    _draw_topology(ax_topology, bundle["topology"])
    _draw_effects(ax_loss, ax_delay, bundle["effects"])
    _panel_letter(ax_mta, "A", x=-0.22)
    _panel_letter(ax_topology, "B")
    _panel_letter(ax_loss, "C")
    _panel_letter(ax_delay, "D")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / "fig4_rev17_node_causal_validation"
    figure.savefig(stem.with_suffix(".png"), dpi=450, facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), dpi=450, facecolor="white")
    plt.close(figure)
    files = {
        suffix: {"path": str(stem.with_suffix(f".{suffix}")),
                 "sha256": _sha256(stem.with_suffix(f".{suffix}"))}
        for suffix in ("png", "pdf")
    }
    metadata = {
        "schema_id": "topic4_rev17_node_causal_validation_figure_v1",
        "candidate_id": bundle["candidate_id"],
        "node_freeze_status": "REV17_NODE_FIELD_FROZEN",
        "network_seeds": bundle["network_seeds"],
        "numeric_label_to_mode": {"0": "MTB", "1": "MTA"},
        "topology": bundle["topology"],
        "intervention_effects": bundle["effects"],
        "ordered_endpoint": (
            "native supported-mode loss first; onset delay only among retained events"
        ),
        "EE_EtoI_ZM": "off",
        "files": files,
        "plotting_only": True,
        "SNN_simulation_run": False,
    }
    metadata_path = Path(str(stem) + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    readme = f"""### {stem.name}

Panel A 显示三张未见网络 leave-one-network-out 构建的 MTA/MTB 最早 10% 空间支持；圆圈是预测热点，叉号是按 `h`、实际 `Delta Vtheta`、E 细胞数和基线放电率匹配的离模板对照。Panel B 将冻结候选和 exact dual-field anchor 的最弱模式起始拓扑质量分别放在各自的占比保持标签置换零假设旁。Panel C 逐网络显示同 checkpoint 阈值抬高是否使原模式消失；Panel D 只对仍保留原模式的事件显示 onset delay，未把事件消失错误记作 0 ms。EE、E-to-I 和 Z/M 均关闭。

**关注点**：热点效应要同时强于另一模式和匹配位置；图中每个点是一张独立网络，二元事件消失优先于延迟解释。
"""
    (output_dir / "README.md").write_text(readme)
    summary = {
        "status": "REV17_NODE_CAUSAL_VALIDATION_FIGURE_RENDERED",
        "candidate_id": bundle["candidate_id"], "files": files,
        "metadata": str(metadata_path), "readme": str(output_dir / "README.md"),
        "plotting_only": True, "SNN_simulation_run": False,
    }
    (output_dir / "render_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--postselection", type=Path, default=DEFAULT_POSTSELECTION)
    parser.add_argument("--final-science", type=Path, default=DEFAULT_FINAL_SCIENCE)
    parser.add_argument("--intervention", type=Path, default=DEFAULT_INTERVENTION)
    parser.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(render(
        config_path=args.config, postselection_path=args.postselection,
        final_science_path=args.final_science, intervention_path=args.intervention,
        freeze_path=args.freeze, output_dir=args.output_dir,
        artifact_root=args.artifact_root,
    ), indent=2))


if __name__ == "__main__":
    main()
