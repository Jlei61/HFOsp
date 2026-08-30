#!/usr/bin/env python3
"""Refresh Figure 4G/H with all-event Timing+Space patient templates.

The frozen SNN events, model KMeans labels and contact geometry are unchanged.
Only the E1146 patient-side event labels/profiles are replaced by the all-event
Timing+Space clustering, after an explicit TA/TB semantic alignment to the
new frozen field artifact.  Panel H and its matched within-shaft null are then
recomputed from the same 12 frozen networks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import plot_interictal_propagation as propagation_plot  # noqa: E402
from scripts.paper_figures import plot_fig4_spatial_edge_flow_validation as spatial  # noqa: E402
from src.topic4_d6_natural_kmeans import (  # noqa: E402
    contact_split_folds,
    crossfit_patient_readout,
    normalize_event_ranks,
    patient_profiles,
)
from src.topic4_nlc_null_calibration import (  # noqa: E402
    _assign_to_profiles,
    _matrix,
    _mode_profiles,
    equal_network_null,
)


TA_COLOR = "#C43C39"
TB_COLOR = "#277DA1"
SHAFT_COLORS = {"ICL": "#E67E22", "SCL": "#159EAE"}
SEMANTIC_PAIRS = {"MTA_vs_TA": (1, 1), "MTB_vs_TB": (0, 0)}
SEMANTIC_DISPLAY_ORDER = (1, 0)
DEFAULT_SOURCE_ROOT = ROOT
DEFAULT_TEMPLATE = (
    ROOT
    / "results/interictal_propagation_masked/"
    "template_gradient_fields_all_events_timing_plus_space/per_subject/"
    "epilepsiae_1146.json"
)
DEFAULT_OUTPUT = ROOT / "results/paper-ready-figure/fig4gh_all_event_timing_plus_space"
CONFIG_RELATIVE = Path("config/topic4_rev11_nlc_frozen_substrate_confirmation.json")
NLC_OUTPUT_RELATIVE = Path(
    "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/"
    "frozen_substrate_confirmation"
)
OLD_KMEANS_METADATA_RELATIVE = NLC_OUTPUT_RELATIVE / "figures/fig4b_nlc_kmeans_consistency_metadata.json"
OLD_PAIRWISE_RELATIVE = Path(
    "results/paper-ready-figure/fig4/fig4_panelh_pairwise_similarity_statistics.json"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def _mean_profiles(ranks: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return patient_profiles(np.asarray(ranks, float), np.asarray(labels, int))


def _crossfit_matrix(
    normalized_ranks: np.ndarray,
    profiles: np.ndarray,
    folds: tuple[np.ndarray, np.ndarray],
) -> np.ndarray:
    stack = []
    for assignment_contacts, evaluation_contacts in (
        (folds[0], folds[1]),
        (folds[1], folds[0]),
    ):
        labels = _assign_to_profiles(normalized_ranks, profiles, assignment_contacts)
        model = _mode_profiles(normalized_ranks, labels)
        stack.append(_matrix(model, profiles, evaluation_contacts))
    stack = np.asarray(stack, float)
    count = np.sum(np.isfinite(stack), axis=0)
    return np.divide(
        np.nansum(stack, axis=0), count,
        out=np.full(stack.shape[1:], np.nan), where=count > 0,
    )


def _contact_permutation_matrix_draws(
    ranks: np.ndarray,
    patient_ranks: np.ndarray,
    patient_labels: np.ndarray,
    folds: tuple[np.ndarray, np.ndarray],
    *,
    draws: int,
    seed: int,
    shaft_ids: np.ndarray,
) -> np.ndarray:
    normalized = normalize_event_ranks(ranks)
    profiles = patient_profiles(patient_ranks, patient_labels)
    rng = np.random.default_rng(int(seed))
    blocks = [
        np.flatnonzero(shaft_ids == shaft) for shaft in np.unique(shaft_ids)
    ]
    values = np.full((int(draws), 2, 2), np.nan)
    for draw in range(int(draws)):
        order = np.arange(normalized.shape[1])
        for block in blocks:
            order[block] = rng.permutation(block)
        values[draw] = _crossfit_matrix(normalized[:, order], profiles, folds)
    return values


def _semantic_relabel(
    bundle: dict,
    template_path: Path,
) -> dict:
    template = json.loads(template_path.read_text(encoding="utf-8"))
    discovery = template["template_discovery"]
    sampled = np.asarray(discovery["sampled_event_indices"], int)
    labels = np.asarray(discovery["event_labels"], int)
    if len(sampled) != len(labels) or len(np.unique(sampled)) != len(sampled):
        raise ValueError("invalid all-event label index")
    with np.load(bundle["target_path"], allow_pickle=False) as target:
        training_indices = np.asarray(target["patient_train_event_indices"], int)
    lookup = {int(index): int(label) for index, label in zip(sampled, labels)}
    try:
        new_numeric = np.asarray([lookup[int(index)] for index in training_indices], int)
    except KeyError as exc:
        raise ValueError(f"patient training event is absent from all-event clustering: {exc}")

    patient_ranks = np.asarray(bundle["patient"]["patient_train_ranks"], float)
    cluster_profiles = _mean_profiles(patient_ranks, new_numeric)
    field = template["interictal_field"]
    field_names = [str(value) for value in field["contact_order"]]
    model_names = bundle["static"]["contact_names"].astype(str).tolist()
    if set(field_names) != set(model_names):
        raise ValueError("new field and Fig4 patient target do not share the same contacts")
    field_order = np.asarray([field_names.index(name) for name in model_names], int)
    field_profiles = np.asarray([field["rank_a"], field["rank_b"]], float)[:, field_order]
    similarity = np.full((2, 2), np.nan)
    for cluster in (0, 1):
        for field_index in (0, 1):
            finite = np.isfinite(cluster_profiles[cluster]) & np.isfinite(field_profiles[field_index])
            similarity[cluster, field_index] = float(
                spearmanr(
                    cluster_profiles[cluster, finite],
                    field_profiles[field_index, finite],
                ).statistic
            )
    direct = float(similarity[0, 0] + similarity[1, 1])
    swapped = float(similarity[0, 1] + similarity[1, 0])
    if direct >= swapped:
        cluster_to_semantic = {0: spatial.TA_MODE, 1: spatial.TB_MODE}
        assignment = "cluster0=TA,cluster1=TB"
    else:
        cluster_to_semantic = {0: spatial.TB_MODE, 1: spatial.TA_MODE}
        assignment = "cluster0=TB,cluster1=TA"
    semantic = np.asarray([cluster_to_semantic[int(value)] for value in new_numeric], int)
    old = np.asarray(bundle["patient"]["patient_train_old_labels"], int)
    if len(semantic) != len(old):
        raise ValueError("updated patient labels do not align with the frozen training target")
    old_profiles = spatial._patient_profiles(bundle)[0]
    bundle["patient"]["patient_train_old_labels"] = semantic
    new_profiles = spatial._patient_profiles(bundle)[0]
    return {
        "template": template,
        "old_labels": old,
        "new_labels": semantic,
        "old_profiles": old_profiles,
        "new_profiles": new_profiles,
        "semantic_alignment": {
            "cluster_vs_field_spearman": similarity.tolist(),
            "direct_sum": direct,
            "swapped_sum": swapped,
            "assignment": assignment,
        },
        "label_change": {
            "n_training_events": int(len(old)),
            "n_changed": int(np.sum(old != semantic)),
            "fraction_changed": float(np.mean(old != semantic)),
            "old_counts_raw_mode_0_1": np.bincount(old, minlength=2).tolist(),
            "new_counts_raw_mode_0_1": np.bincount(semantic, minlength=2).tolist(),
        },
    }


def _display_payload(bundle: dict) -> dict:
    canonical = spatial._canonical_rank_kmeans(bundle)
    selected = canonical["clean_global_index"]
    n_contacts = len(bundle["static"]["contact_names"])
    display_ranks = normalize_event_ranks(bundle["ranks"][selected]) * (n_contacts - 1)
    frozen_labels, mapping = spatial._map_kmeans_clusters_to_modes(
        canonical["labels"], canonical["direction_contingency"],
    )
    labels = np.where(frozen_labels == spatial.TA_MODE, 0, 1).astype(int)
    names = np.asarray(bundle["static"]["contact_names"], str)
    patient = spatial._patient_profiles(bundle)[0] * (n_contacts - 1)
    patient_rank_matrix = np.asarray(bundle["patient"]["patient_train_ranks"], float).T
    order = propagation_plot._fixed_channel_order(
        patient_rank_matrix, np.isfinite(patient_rank_matrix),
    )
    model = np.asarray([
        spatial._column_stats(display_ranks[labels == mode])[0]
        for mode in (0, 1)
    ])
    return {
        "labels": labels,
        "names": names,
        "order": order,
        "model_profiles": model,
        "patient_profiles": patient,
        "cluster_mapping": mapping.tolist(),
        "counts": np.bincount(labels, minlength=2).tolist(),
        "n_contacts": n_contacts,
    }


def _compute_pairwise(bundle: dict, config_path: Path, *, draws: int) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    contract_path = config_path.parents[0].parent / config["inputs"]["contact_contract"]["path"]
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    folds = contact_split_folds(contract)
    seeds = [int(value) for value in config["search"]["confirmation_network_seeds"]]
    record_seed = np.asarray([row["seed"] for row in bundle["records"]], int)
    patient_ranks = np.asarray(bundle["patient"]["patient_train_ranks"], float)
    patient_labels = np.asarray(bundle["patient"]["patient_train_old_labels"], int)
    shaft_ids = np.asarray(bundle["static"]["shaft_ids"])
    observed_by_pair = {key: {} for key in SEMANTIC_PAIRS}
    draws_by_pair = {key: {} for key in SEMANTIC_PAIRS}
    observed_matrices = []
    per_network = {}
    for network_seed in seeds:
        index = np.flatnonzero(bundle["clean"] & (record_seed == network_seed))
        if not len(index):
            raise RuntimeError(f"no clean events for network {network_seed}")
        ranks = np.asarray(bundle["ranks"][index], float)
        observed = np.asarray(
            crossfit_patient_readout(ranks, patient_ranks, patient_labels, folds)["matrix"],
            float,
        )
        nulls = _contact_permutation_matrix_draws(
            ranks, patient_ranks, patient_labels, folds, draws=draws,
            seed=20260815 + 7919 + network_seed, shaft_ids=shaft_ids,
        )
        row = {"n_clean_events": int(len(index)), "observed_raw_mode_matrix": observed.tolist()}
        for name, (matrix_row, matrix_column) in SEMANTIC_PAIRS.items():
            values = np.asarray(nulls[:, matrix_row, matrix_column], float)
            observed_by_pair[name][str(network_seed)] = float(observed[matrix_row, matrix_column])
            draws_by_pair[name][str(network_seed)] = values
            row[name] = {
                "observed_rho": float(observed[matrix_row, matrix_column]),
                "null_median": float(np.median(values)),
                "null_q95": float(np.quantile(values, 0.95)),
            }
        per_network[str(network_seed)] = row
        observed_matrices.append(observed)
    tests = {}
    for name in SEMANTIC_PAIRS:
        summary = equal_network_null(observed_by_pair[name], draws_by_pair[name])
        if summary is None:
            raise RuntimeError(f"failed to aggregate {name} null")
        summary["stars"] = _stars(float(summary["one_sided_p"]))
        summary["alternative"] = "observed similarity exceeds permuted similarity"
        summary["null"] = "within-shaft model-contact permutation"
        tests[name] = summary
    raw_matrix = np.mean(np.asarray(observed_matrices, float), axis=0)
    display = raw_matrix[np.ix_(SEMANTIC_DISPLAY_ORDER, SEMANTIC_DISPLAY_ORDER)]
    return {
        "schema_version": "fig4_panelh_pairwise_similarity_all_event_space_v1",
        "status": "PAIRWISE_SIMILARITY_NULL_COMPLETE",
        "n_networks": len(seeds),
        "network_seeds": seeds,
        "displayed_equal_network_matrix": display.tolist(),
        "display_order": {"rows": ["MTA", "MTB"], "columns": ["TA", "TB"]},
        "tests": tests,
        "per_network": per_network,
        "null_contract": {
            "restriction": "contacts exchange only within the same shaft",
            "aggregation": "equal-network mean",
            "draws": int(draws),
            "alternative": "one-sided greater",
        },
    }


def _save(fig: plt.Figure, figures: Path, panel: str) -> dict:
    png = figures / f"fig4-panel{panel}.png"
    pdf = figures / f"fig4-panel{panel}.pdf"
    fig.savefig(png, dpi=600, facecolor="white", bbox_inches="tight", pad_inches=0.045)
    fig.savefig(pdf, facecolor="white", bbox_inches="tight", pad_inches=0.045)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _draw_g(payload: dict, figures: Path) -> dict:
    order = payload["order"]
    names = payload["names"]
    model = payload["model_profiles"]
    patient = payload["patient_profiles"]
    y = np.arange(len(order), dtype=float)
    fig, ax = plt.subplots(figsize=(5.45, 5.1), facecolor="white")
    fig.subplots_adjust(left=0.23, right=0.975, bottom=0.23, top=0.90)
    specifications = (
        (model[0], TA_COLOR, "-", "o", "MTA model"),
        (patient[0], TA_COLOR, "--", None, "TA patient"),
        (model[1], TB_COLOR, "-", "o", "MTB model"),
        (patient[1], TB_COLOR, "--", None, "TB patient"),
    )
    for values, color, linestyle, marker, label in specifications:
        selected = np.asarray(values)[order]
        finite = np.isfinite(selected)
        ax.plot(
            selected[finite], y[finite], linestyle, color=color,
            lw=2.6 if linestyle == "-" else 2.2, marker=marker, ms=5.4, label=label,
        )
    ax.set_yticks(y, names[order], fontsize=13)
    for tick, contact in zip(ax.get_yticklabels(), names[order]):
        shaft = "".join(character for character in contact if not character.isdigit())
        tick.set_color(SHAFT_COLORS.get(shaft, "#333333"))
    ax.invert_yaxis()
    ax.set_xlim(-0.4, payload["n_contacts"] - 0.6)
    ax.set_xticks([0, 4, 8, 12, 14])
    ax.set_xlabel("Mean rank", fontsize=16)
    ax.tick_params(axis="x", labelsize=13)
    ax.grid(axis="x", color="#E3E7E9", lw=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Rank profiles", fontsize=19, fontweight="bold", pad=8)
    fig.legend(
        handles=[
            Line2D([0], [0], color=TA_COLOR, lw=2.6, marker="o", ms=5.0, label="MTA"),
            Line2D([0], [0], color=TA_COLOR, lw=2.2, ls="--", label="TA"),
            Line2D([0], [0], color=TB_COLOR, lw=2.6, marker="o", ms=5.0, label="MTB"),
            Line2D([0], [0], color=TB_COLOR, lw=2.2, ls="--", label="TB"),
        ],
        frameon=False, fontsize=12.5, ncol=4, loc="lower center",
        bbox_to_anchor=(0.60, 0.015), columnspacing=0.9,
        handlelength=1.6, handletextpad=0.35,
    )
    return _save(fig, figures, "g")


def _draw_h(pairwise: dict, figures: Path) -> dict:
    matrix = np.asarray(pairwise["displayed_equal_network_matrix"], float)
    tests_by_cell = {(0, 0): pairwise["tests"]["MTA_vs_TA"], (1, 1): pairwise["tests"]["MTB_vs_TB"]}
    fig = plt.figure(figsize=(4.6, 4.7), facecolor="white")
    grid = fig.add_gridspec(
        1, 2, width_ratios=(1.0, 0.075), left=0.19, right=0.94,
        bottom=0.15, top=0.84, wspace=0.10,
    )
    ax = fig.add_subplot(grid[0, 0])
    cax = fig.add_subplot(grid[0, 1])
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_aspect("equal")
    ax.set_xticks((0, 1), ("TA", "TB"), fontsize=16, fontweight="bold")
    ax.set_yticks((0, 1), ("MTA", "MTB"), fontsize=16, fontweight="bold")
    for tick, color in zip(ax.get_xticklabels(), (TA_COLOR, TB_COLOR)):
        tick.set_color(color)
    for tick, color in zip(ax.get_yticklabels(), (TA_COLOR, TB_COLOR)):
        tick.set_color(color)
    for row in range(2):
        for column in range(2):
            foreground = "white" if abs(matrix[row, column]) >= 0.55 else "#111111"
            value_y = row - 0.10 if (row, column) in tests_by_cell else row
            ax.text(
                column, value_y, f"{matrix[row, column]:+.2f}", ha="center", va="center",
                fontsize=19, fontweight="bold", color=foreground,
            )
            if (row, column) in tests_by_cell:
                ax.text(
                    column, row + 0.22, tests_by_cell[(row, column)]["stars"],
                    ha="center", va="center", fontsize=16.5,
                    fontweight="bold", color=foreground,
                )
    ax.set_title("Cross-fit similarity", fontsize=19, fontweight="bold", pad=12)
    colorbar = fig.colorbar(image, cax=cax, ticks=(-1, -0.5, 0, 0.5, 1))
    colorbar.set_label("ρ", fontsize=17, labelpad=5)
    colorbar.ax.tick_params(labelsize=13)
    return _save(fig, figures, "h")


def build(source_root: Path, template_path: Path, output_root: Path, *, draws: int) -> dict:
    config_path = source_root / CONFIG_RELATIVE
    nlc_output = source_root / NLC_OUTPUT_RELATIVE
    spatial.ROOT = source_root
    bundle = spatial._load_bundle(config_path, nlc_output)
    relabel = _semantic_relabel(bundle, template_path)
    payload = _display_payload(bundle)
    pairwise = _compute_pairwise(bundle, config_path, draws=draws)
    figures = output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    outputs = {"g": _draw_g(payload, figures), "h": _draw_h(pairwise, figures)}
    from scripts.paper_figures import build_main_figures_1_2 as layout

    layout.ROOT = source_root
    canonical = source_root / "results/paper-ready-figure/fig4/figures"
    complete_layout = layout._compose_complete_layout(
        figures_dir=figures,
        stem="fig4-complete-layout",
        canvas_size=(10500, 6500),
        placements={
            "a": (canonical / "fig4-panela.png", (140, 150, 2330, 2050)),
            "b": (canonical / "fig4-panelb.png", (2490, 150, 4790, 2050)),
            "c": (canonical / "fig4-panelc.png", (4460, 150, 10360, 2050)),
            "d": (canonical / "fig4-paneld.png", (140, 2180, 6040, 4240)),
            "e": (canonical / "fig4-panele.png", (6170, 2180, 10360, 4240)),
            "f": (canonical / "fig4-panelf.png", (140, 4380, 5510, 6380)),
            "g": (figures / "fig4-panelg.png", (5680, 4380, 7820, 6380)),
            "h": (figures / "fig4-panelh.png", (7980, 4380, 10360, 6380)),
        },
        labels={
            "A": (25, 20), "B": (2355, 20), "C": (4325, 20),
            "D": (25, 2050), "E": (6035, 2050),
            "F": (25, 4250), "G": (5545, 4250), "H": (7845, 4250),
        },
        anchors={
            "a": "top", "b": "top-left", "c": "top-left",
            "d": "top", "e": "top", "f": "top", "g": "top", "h": "top",
        },
        label_font_size=132,
    )
    pairwise_path = output_root / "fig4_panelh_pairwise_similarity_statistics.json"
    pairwise_path.write_text(json.dumps(pairwise, indent=2) + "\n", encoding="utf-8")

    old_metadata_path = source_root / OLD_KMEANS_METADATA_RELATIVE
    old_metadata = json.loads(old_metadata_path.read_text(encoding="utf-8"))
    old_pairwise_path = source_root / OLD_PAIRWISE_RELATIVE
    old_pairwise = json.loads(old_pairwise_path.read_text(encoding="utf-8"))
    old_matrix = np.asarray(old_metadata["displayed_matrix"], float)
    new_matrix = np.asarray(pairwise["displayed_equal_network_matrix"], float)
    metadata = {
        "schema_version": "figure4gh_all_event_timing_plus_space_v1",
        "simulation_rerun": False,
        "frozen_model_events_and_kmeans": True,
        "updated_component": "E1146 patient template labels and profiles only",
        "template_source": {"path": str(template_path), "sha256": _sha256(template_path)},
        "label_change": relabel["label_change"],
        "semantic_alignment": relabel["semantic_alignment"],
        "patient_profile_max_abs_change_normalized_rank": float(
            np.nanmax(np.abs(relabel["new_profiles"] - relabel["old_profiles"]))
        ),
        "model_cluster_counts_MTA_MTB": payload["counts"],
        "old_matrix": old_matrix.tolist(),
        "new_matrix": new_matrix.tolist(),
        "matrix_delta": (new_matrix - old_matrix).tolist(),
        "old_tests": old_pairwise["tests"],
        "new_tests": pairwise["tests"],
        "outputs": outputs,
        "complete_layout": complete_layout,
        "statistics": str(pairwise_path),
        "scientific_boundary": (
            "development-case patient-template comparison only; the SNN simulation and model "
            "KMeans are frozen, and the refresh does not establish patient-blind generalization."
        ),
    }
    metadata_path = figures / "fig4gh-all-event-space-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    mta = pairwise["tests"]["MTA_vs_TA"]
    mtb = pairwise["tests"]["MTB_vs_TB"]
    (figures / "README.md").write_text(
        f"""# Figure 4G/H — all-event Timing+Space refresh

### fig4-panelg.png / .pdf

模型 MTA/MTB 实线保持冻结；患者 TA/TB 虚线改由 all-event Timing+Space 标签在同一
30,049 个 E1146 training events 上重算。共有 {relabel['label_change']['n_changed']}/
{relabel['label_change']['n_training_events']} 个训练事件改变模板归属。

**关注点**：只更新患者端模板，未重跑 SNN，也未改变模型 KMeans。

### fig4-panelh.png / .pdf

12 张冻结网络等权的 contact-split cross-fit Spearman 矩阵。MTA–TA 为
rho={mta['observed_equal_network_mean']:.3f}、P={mta['one_sided_p']:.6g}（{mta['stars']}）；
MTB–TB 为 rho={mtb['observed_equal_network_mean']:.3f}、P={mtb['one_sided_p']:.6g}
（{mtb['stars']}）。两项均重新使用 1,000 次 matched within-shaft contact permutation。

**关注点**：这是 E1146 development-case 的模板一致性复核，不是 cohort 或 patient-blind 泛化。
""",
        encoding="utf-8",
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    result = build(
        args.source_root.resolve(), args.template.resolve(), args.output_root.resolve(),
        draws=args.draws,
    )
    print(json.dumps({
        "label_change": result["label_change"],
        "old_matrix": result["old_matrix"],
        "new_matrix": result["new_matrix"],
        "new_tests": result["new_tests"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
