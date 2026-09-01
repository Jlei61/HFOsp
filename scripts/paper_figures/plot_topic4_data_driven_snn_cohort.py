#!/usr/bin/env python3
"""Cohort-level figure for the formal data-driven SNN run.

Panels answer four independent questions and nothing else: who is in the
cohort, does each subject beat its own matched contact-identity null, does a
single network hold two reproducible event clusters, and does the answer
survive swapping the readout geometry.  The representative subject's readout
and clustering are separate figures drawn with the accepted painters.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
PASS_COLOR = "#2166ac"
FAIL_COLOR = "#b2182b"
NULL_COLOR = "#9e9e9e"
VERDICT_TEXT = {
    "COHORT_MODEL_SUPPORT_SUPPORTED":
        "held-out contact-order structure recovered above matched nulls",
    "COHORT_MODEL_SUPPORT_INSUFFICIENT":
        "held-out contact-order structure NOT recovered above matched nulls",
    # Deliberately does not say "supervised alignment holds": that is the very
    # claim the readout-geometry gate can withdraw, and the two gates can fail
    # together.
    "SAME_NETWORK_K2_INSUFFICIENT":
        "one network rarely holds two clusters matching distinct patient modes",
    "OBSERVATION_LAYOUT_DEPENDENCE_UNRESOLVED":
        "the effect does not survive the readout geometry",
    "TARGET_BLIND_FIELD_LIBRARY_CAPACITY_FAIL":
        "the target-blind field library had no capacity",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _short(subject_id: str) -> str:
    return subject_id.replace("epilepsiae_", "E").replace("yuquan_", "Y")


def _format_p(value: float) -> str:
    if not np.isfinite(value):
        return "not evaluable"
    return f"{value:.3g}" if value >= 1e-4 else f"{value:.1e}"


def _banner(fig, result: dict) -> None:
    """The accepted verdict goes on the canvas, qualifiers included."""
    cohort = result["cohort"]
    status = result["status"]
    primary = cohort["primary_test"]
    verdict = result["verdict"]
    failed = verdict.get("failed_gates") or ([status] if verdict.get("reasons") else [])
    headline = " ; ".join(VERDICT_TEXT.get(gate, gate) for gate in failed) or (
        VERDICT_TEXT.get(status, status)
    )
    if len(failed) > 1:
        headline = f"{len(failed)} gates not met: " + headline
    sensitivity = cohort.get("sensitivity") or {}
    detail = (
        f"{cohort['n_subjects']} patients | "
        f"{cohort['pass_fraction']:.0%} beat their own shuffle "
        f"(gate {cohort['pass_fraction_min']:.0%}) | "
        f"median advantage {primary['median_delta']:+.4f}, "
        f"p = {_format_p(primary['wilcoxon_p'])} | "
        f"real implant geometry {sensitivity.get('real_geometry_median_delta', float('nan')):+.4f}"
        f" | two clusters in one network in "
        f"{cohort['same_network_k2_fraction']:.0%}"
    )
    # Every broken gate goes on the canvas; naming one hides the others.
    reasons = verdict.get("all_reasons") or verdict.get("reasons") or []
    if reasons:
        detail += "\nnot met: " + " | ".join(reasons)
    color = PASS_COLOR if status == "COHORT_MODEL_SUPPORT_SUPPORTED" else FAIL_COLOR
    fig.text(0.5, 0.995, headline, ha="center", va="top", fontsize=13,
             fontweight="bold", color=color)
    fig.text(0.5, 0.963, detail, ha="center", va="top", fontsize=9.2,
             color="#333333", linespacing=1.45)


def _panel_denominator(ax, result: dict) -> None:
    cohort = result["cohort"]
    denominators = result["denominators"]
    counts = [
        denominators["primary_canonical_layout"],
        int(round(cohort["pass_fraction"] * cohort["n_subjects"])),
        int(round(cohort["same_network_k2_fraction"] * cohort["n_subjects"])),
        denominators["real_geometry_sensitivity"],
    ]
    labels = [
        "in the cohort", "beat their own shuffle",
        "two clusters in\none network", "have real 3-D geometry",
    ]
    colors = [NULL_COLOR, PASS_COLOR, PASS_COLOR, "#7fb0d5"]
    y = np.arange(len(counts))[::-1]
    ax.barh(y, counts, color=colors, height=0.68)
    for position, value in zip(y, counts):
        ax.text(value + 0.6, position, str(value), ha="left", va="center",
                fontsize=11, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, max(counts) + 6)
    ax.set_xlabel("patients")
    ax.set_title("who the numbers are about", loc="left", fontweight="bold",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)


def _panel_null(ax, rows: list[dict]) -> None:
    order = sorted(rows, key=lambda row: row["delta_null_median_minus_observed"])
    x = np.arange(len(order))
    observed = np.asarray([row["observed_weakest_mode_loss"] for row in order])
    null_median = np.asarray([row["null_median"] for row in order])
    passed = np.asarray([bool(row["subject_endpoint_pass"]) for row in order])
    ax.vlines(x, np.minimum(observed, null_median),
              np.maximum(observed, null_median), color="#dddddd", linewidth=1.4)
    ax.scatter(x, null_median, s=26, color=NULL_COLOR, zorder=3,
               label="shuffled contact identity within each shaft")
    ax.scatter(x[passed], observed[passed], s=30, color=PASS_COLOR, zorder=4,
               label="model, better than its own shuffle")
    ax.scatter(x[~passed], observed[~passed], s=30, color=FAIL_COLOR, zorder=4,
               label="model, not better")
    ax.set_xticks(x, [_short(row["subject_id"]) for row in order],
                  rotation=90, fontsize=6.5)
    ax.set_ylabel("mismatch to the patient's weaker mode\n(lower is closer)")
    ax.set_title(
        "does the model match held-out patient data better than the same\n"
        "contacts relabelled inside their own shaft?",
        loc="left", fontweight="bold", fontsize=11,
    )
    ax.legend(frameon=False, fontsize=8.5, loc="lower left", ncol=1)
    ax.spines[["top", "right"]].set_visible(False)


def _panel_same_network(ax, rows: list[dict], n_seeds: int) -> None:
    order = sorted(
        rows,
        key=lambda row: (
            -row["natural_kmeans"]["n_seeds_with_same_network_k2"],
            row["subject_id"],
        ),
    )
    x = np.arange(len(order))
    seeds = np.asarray([
        row["natural_kmeans"]["n_seeds_with_same_network_k2"] for row in order
    ])
    ood = np.asarray([
        float(row["per_seed"][0].get("ood_fraction") or 0.0) for row in order
    ])
    threshold = max(3, n_seeds - 1)
    ax.bar(x, seeds, color=np.where(seeds >= threshold, PASS_COLOR, NULL_COLOR),
           width=0.72)
    ax.axhline(threshold - 0.5, color="#444444", linewidth=0.9, linestyle="--")
    ax.text(len(x) - 0.4, threshold - 0.42, "counts as recovered above this line",
            ha="right", va="bottom", fontsize=8, color="#444444")
    ax.set_ylim(0, n_seeds + 0.9)
    ax.set_yticks(np.arange(n_seeds + 1))
    ax.set_ylabel("networks holding two\nmatching clusters")
    ax.set_xticks(x, [_short(row["subject_id"]) for row in order],
                  rotation=90, fontsize=6.5)
    ax.set_title(
        "does one network contain both propagation modes at once,\n"
        "without being told the patient labels?",
        loc="left", fontweight="bold", fontsize=11,
    )
    ax.spines[["top"]].set_visible(False)
    twin = ax.twinx()
    twin.scatter(x, ood, s=16, color="#e08214", zorder=4)
    twin.set_ylim(0, 1.0)
    twin.set_ylabel("events outside the patient's\nown event cloud",
                    color="#e08214", fontsize=9)
    twin.tick_params(axis="y", colors="#e08214")
    twin.spines[["top"]].set_visible(False)


def _panel_layout(ax, result: dict) -> None:
    sensitivity = result["cohort"].get("sensitivity")
    if not sensitivity:
        ax.text(0.5, 0.5, "no real-geometry arm", ha="center", va="center")
        ax.axis("off")
        return
    real_lookup = {
        row["subject_id"]: row for row in result["real_geometry_subjects"]
    }
    shared = [
        row for row in result["canonical_subjects"]
        if row["subject_id"] in real_lookup
    ]
    canonical = np.asarray([
        row["delta_null_median_minus_observed"] for row in shared
    ])
    real = np.asarray([
        real_lookup[row["subject_id"]]["delta_null_median_minus_observed"]
        for row in shared
    ])
    limit = float(np.max(np.abs(np.concatenate([canonical, real])))) * 1.15
    limit = max(limit, 1e-6)
    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8)
    ax.plot([-limit, limit], [-limit, limit], color="#dddddd", linewidth=1.0,
            linestyle="--", zorder=1)
    agree = np.sign(canonical) == np.sign(real)
    ax.scatter(canonical[agree], real[agree], s=34, color=PASS_COLOR, zorder=3)
    ax.scatter(canonical[~agree], real[~agree], s=34, color=FAIL_COLOR, zorder=3,
               marker="x")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xlabel("advantage over shuffle,\ncontact-order readout")
    ax.set_ylabel("advantage over shuffle,\nreal implant geometry")
    ax.set_title(
        f"does it survive moving the contacts?\n"
        f"{int(agree.sum())} of {len(shared)} agree in sign "
        f"(coin flip would give {len(shared) / 2:.0f})",
        loc="left", fontweight="bold", fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)


def render(result: dict, output: Path) -> dict:
    n_seeds = len(result["confirmation_seeds"])
    rows = result["canonical_subjects"]
    fig = plt.figure(figsize=(16.4, 9.4))
    grid = fig.add_gridspec(
        2, 3, height_ratios=(1.0, 1.0), width_ratios=(0.78, 1.32, 1.02),
        left=0.105, right=0.955, bottom=0.085, top=0.885, hspace=0.42,
        wspace=0.42,
    )
    _panel_denominator(fig.add_subplot(grid[0, 0]), result)
    _panel_null(fig.add_subplot(grid[0, 1:]), rows)
    _panel_same_network(fig.add_subplot(grid[1, :2]), rows, n_seeds)
    _panel_layout(fig.add_subplot(grid[1, 2]), result)
    _banner(fig, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "png": _relative(output.with_suffix(".png")),
        "png_sha256": _sha256(output.with_suffix(".png")),
        "pdf": _relative(output.with_suffix(".pdf")),
        "pdf_sha256": _sha256(output.with_suffix(".pdf")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--result", type=Path, default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output_root = ROOT / config["output_root"]
    result_path = args.result or (output_root / "cohort_result.json")
    if not result_path.exists():
        print(json.dumps({"status": "COHORT_RESULT_ABSENT_FIGURE_SKIPPED"}))
        return
    result = json.loads(result_path.read_text())
    figures = output_root / "figures"
    files = render(result, figures / "topic4_data_driven_snn_cohort_statistics")
    metadata = {
        "schema_version": "topic4_data_driven_snn_cohort_figure_v1",
        "science_status": {
            "status": result["status"],
            "verdict": result["verdict"],
            "result_json": str(result_path.relative_to(ROOT)),
            "result_json_sha256": _sha256(result_path),
        },
        "denominators": result["denominators"],
        "cohort": result["cohort"],
        "representative_subject": result["representative_subject"],
        "files": files,
        "scientific_boundary": config["claim_boundary"],
    }
    (figures / "topic4_data_driven_snn_cohort_statistics_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({"status": result["status"], **files}, indent=2))


if __name__ == "__main__":
    main()
