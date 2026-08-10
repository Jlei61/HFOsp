"""Correct count mismatch and issue the scientific rev9-L L2 diagnosis."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.getcwd())
from src.topic4_component_pair_search import DESCRIPTOR_NAMES, score_candidate  # noqa: E402
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = "config/topic4_rev9l_component_pair_edge.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _provenance(expected_commit):
    paths = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py":
            continue
        try:
            paths.add(str(path.relative_to(ROOT)))
        except ValueError:
            continue
    paths.add(str(Path(__file__).resolve().relative_to(ROOT)))
    paths = sorted(paths)
    expected = subprocess.check_output(
        ["git", "rev-parse", expected_commit], cwd=ROOT, text=True).strip()
    current = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *paths], cwd=ROOT,
        text=True).strip()
    hashes = {path: _sha256(ROOT / path) for path in paths}
    expected_hashes = {
        path: hashlib.sha256(subprocess.check_output(
            ["git", "show", f"{expected}:{path}"], cwd=ROOT)).hexdigest()
        for path in paths
    }
    if dirty or current != expected or any(
            hashes[path] != expected_hashes[path] for path in paths):
        raise RuntimeError("L2 scientific review differs from expected commit")
    return {
        "git_commit": current,
        "expected_git_commit": expected,
        "runtime_modules_dirty": False,
        "runtime_module_sha256": hashes,
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def _read(path, status):
    payload = json.loads(Path(path).read_text())
    if payload["status"] != status or payload.get("patient_heldout_scores_computed") is not False:
        raise RuntimeError(f"invalid L2 input: {path}")
    return payload, {"path": str(path), "sha256": _sha256(path)}


def _rescore(row, floor, objective):
    source_for = {"A": "component_2", "B": "component_1"}
    readable = {mode: row["geometry"][source]["curve_usable_fraction"]
                for mode, source in source_for.items()}
    ood = {mode: row["geometry"][source]["ood_fraction"]
           for mode, source in source_for.items()}
    return score_candidate(
        row["mode_descriptors"], floor["floor"], readable, ood,
        readable_weight=objective["readable_fraction_penalty_weight"],
        tau=objective["weakest_mode_lse_tau"],
        ood_weight=objective["ood_weight"])


def _field_kl(summary, stage):
    values = {}
    for record in summary["worker_inputs"]:
        payload = json.loads(Path(record["json"]["path"]).read_text())
        diagnostics = payload["network"]["edge_diagnostics"]["residual_target_groups"]
        medians = [diagnostics[key]["kl_median"]
                   for key in ("component_1_dominant", "component_2_dominant")
                   if diagnostics[key]["kl_median"] is not None]
        values.setdefault(record["candidate_id"], []).append(max(medians, default=0.0))
    return {candidate: float(np.median(rows)) for candidate, rows in values.items()}


def _prototype_weak(row):
    losses = [0.5 * (1.0 - row["mode_descriptors"]["modes"][mode][
        "curve_prototype_spearman"]) for mode in ("A", "B")]
    return float(max(losses))


def _plot(fit_rows, selection_rows, baseline_id, selected_id, field_kl, floor,
          output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), constrained_layout=True)
    common = [key for key in selection_rows if key in fit_rows]
    x = [fit_rows[key]["score"]["objective"] for key in common]
    y = [selection_rows[key]["corrected_score"]["objective"] for key in common]
    axes[0].scatter(x, y, color="#277da1", s=38)
    lo, hi = min(x + y), max(x + y)
    axes[0].plot([lo, hi], [lo, hi], ls="--", color="0.6", lw=1)
    for key in common:
        axes[0].annotate(key.replace("sobol_", ""),
                         (fit_rows[key]["score"]["objective"],
                          selection_rows[key]["corrected_score"]["objective"]),
                         xytext=(2, 2), textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("fit objective (n=6 floor)")
    axes[0].set_ylabel("selection objective (n=3 floor)")
    axes[0].set_title("A  Out-of-fit stability", loc="left", weight="bold")

    base, best = selection_rows[baseline_id], selection_rows[selected_id]
    positions = np.arange(len(DESCRIPTOR_NAMES))
    width = 0.36
    for offset, mode, color in ((-width / 2, "A", "#d1495b"),
                                (width / 2, "B", "#277da1")):
        base_values = [base["mode_descriptors"]["modes"][mode][key]
                       / floor["floor"]["modes"][mode][key]["median"]
                       for key in DESCRIPTOR_NAMES]
        best_values = [best["mode_descriptors"]["modes"][mode][key]
                       / floor["floor"]["modes"][mode][key]["median"]
                       for key in DESCRIPTOR_NAMES]
        axes[1].bar(positions + offset, base_values, width, color=color, alpha=0.25)
        axes[1].scatter(positions + offset, best_values, color=color, s=28,
                        label=f"mode {mode}")
    axes[1].axhline(1.0, color="0.4", ls="--", lw=1)
    axes[1].set_xticks(positions, ["recruit", "precedence", "profile", "event cloud"],
                       rotation=25, ha="right")
    axes[1].set_ylabel("distance / patient floor median")
    axes[1].set_title("B  Baseline bars, selected dots", loc="left", weight="bold")
    axes[1].legend(frameon=False, fontsize=7)

    all_fit = list(fit_rows.values())
    axes[2].scatter(
        [row["mode_descriptors"]["modes"]["A"]["curve_prototype_spearman"]
         for row in all_fit],
        [row["score"]["mode_scores"]["A"] for row in all_fit],
        color="0.55", s=24)
    axes[2].scatter(
        fit_rows[selected_id]["mode_descriptors"]["modes"]["A"][
            "curve_prototype_spearman"],
        fit_rows[selected_id]["score"]["mode_scores"]["A"],
        marker="*", s=150, color="#d1495b", label="selected")
    axes[2].set_xlabel("mode A prototype Spearman")
    axes[2].set_ylabel("full mode A score")
    axes[2].set_title("C  Prototype is not the objective", loc="left", weight="bold")
    axes[2].legend(frameon=False, fontsize=7)

    axes[3].scatter(
        [field_kl[row["candidate_id"]] for row in all_fit],
        [row["score"]["objective"] for row in all_fit],
        c=[row["score"]["mode_scores"]["A"] for row in all_fit],
        cmap="magma", s=28)
    axes[3].set_xlabel("field-target residual KL (median max C1/C2)")
    axes[3].set_ylabel("fit weakest-mode objective")
    axes[3].set_title("D  Fit-distortion landscape", loc="left", weight="bold")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_l2_scientific_review.{suffix}", dpi=300)
    plt.close(fig)
    (output_dir / "README.md").write_text(
        "### rev9l_l2_scientific_review.png\n"
        "A 图用各自匹配事件数的 patient-training floor 比较 fit 与 selection；B 图把 scalar baseline（浅柱）和 selection 候选（实点）的四项绝对距离除以 patient floor；C 图显示只提高 prototype correlation 不等于完整模式改善；D 图改用 C1/C2 field-target KL，避免 background target 把 distortion 中位数压到零。\n\n"
        "**关注点**：selection 改善是否稳定、mode A 四项是否接近患者抽样地板，以及连接扰动是否换来实质的 weakest-mode 收益。\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--selection-floor", required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    root = Path(config["output_root"])
    fit, fit_input = _read(
        root / "sobol_fit/sobol_fit_summary.json", "REV9L_L2_SOBOL_FIT_COMPLETE")
    selection, selection_input = _read(
        root / "selection_confirmation/selection_confirmation_summary.json",
        "REV9L_L2_SELECTION_CONFIRMATION_COMPLETE")
    floor6, floor6_input = _read(
        config["objective"]["floor_output"],
        "REV9L_L2_PATIENT_TRAINING_FLOOR_COMPLETE")
    floor3, floor3_input = _read(
        args.selection_floor, "REV9L_L2_PATIENT_TRAINING_FLOOR_COMPLETE")
    if (floor6["n_events_per_mode_per_draw"] != 6
            or floor3["n_events_per_mode_per_draw"] != 3):
        raise RuntimeError("fit/selection floors do not match event counts")

    fit_rows = {row["candidate_id"]: row for row in fit["candidates"]}
    selection_rows = {row["candidate_id"]: row for row in selection["candidates"]}
    for row in selection_rows.values():
        for mode in ("A", "B"):
            if row["mode_descriptors"]["modes"][mode]["n_model_events"] != 3:
                raise RuntimeError("selection event count changed")
        row["corrected_score"] = _rescore(row, floor3, config["objective"])
    corrected_rank = sorted(
        selection_rows, key=lambda key: selection_rows[key]["corrected_score"]["objective"])
    selected_id, baseline_id = corrected_rank[0], "sobol_000"
    baseline = selection_rows[baseline_id]
    selected = selection_rows[selected_id]
    common = list(selection_rows)
    rank_result = spearmanr(
        [fit_rows[key]["score"]["objective"] for key in common],
        [selection_rows[key]["corrected_score"]["objective"] for key in common])
    fit_eligible = [row for row in fit["candidates"] if row["score"].get("eligible")]
    prototype_best = min(fit_eligible, key=_prototype_weak)
    field_kl = _field_kl(fit, "sobol_fit")
    mode_a_above_q95 = {
        metric: bool(selected["mode_descriptors"]["modes"]["A"][metric]
                     > floor3["floor"]["modes"]["A"][metric]["q95"])
        for metric in DESCRIPTOR_NAMES
    }
    raw_delta = {
        mode: {
            metric: float(selected["mode_descriptors"]["modes"][mode][metric]
                          - baseline["mode_descriptors"]["modes"][mode][metric])
            for metric in DESCRIPTOR_NAMES
        } for mode in ("A", "B")
    }
    objective_gain = float(
        baseline["corrected_score"]["objective"]
        - selected["corrected_score"]["objective"])
    payload = {
        "status": "REV9L_L2_SCIENTIFIC_REVIEW_COMPLETE",
        "safe_claim": (
            "the six-parameter conserved edge residual yields only a small "
            "selection-network improvement and does not recover patient mode A"),
        "fit_n_per_mode": 6,
        "selection_n_per_mode": 3,
        "selection_floor_mismatch_fixed": True,
        "original_selection_summary_retained_as_uncorrected": True,
        "corrected_selection_rank": corrected_rank,
        "corrected_selected_candidate_id": selected_id,
        "scalar_baseline_id": baseline_id,
        "corrected_objective": {
            "selected": selected["corrected_score"],
            "baseline": baseline["corrected_score"],
            "baseline_minus_selected": objective_gain,
            "relative_improvement": float(
                objective_gain / baseline["corrected_score"]["objective"]),
        },
        "fit_to_selection_rank_spearman": {
            "rho": float(rank_result.statistic),
            "pvalue": float(rank_result.pvalue),
            "n": len(common),
        },
        "selection_raw_descriptor_delta_selected_minus_baseline": raw_delta,
        "mode_a_selected_above_patient_floor_q95": mode_a_above_q95,
        "prototype_only_ablation": {
            "best_candidate_id": prototype_best["candidate_id"],
            "mode_a_prototype_spearman": prototype_best[
                "mode_descriptors"]["modes"]["A"]["curve_prototype_spearman"],
            "full_objective": prototype_best["score"]["objective"],
            "full_objective_rank": fit["ranked_candidate_ids"].index(
                prototype_best["candidate_id"]) + 1,
        },
        "diagnosis": {
            "objective": "OLD_OBJECTIVE_INSUFFICIENT_NEW_WEAKEST_MODE_OBJECTIVE_NEEDED",
            "optimizer": "NOT_ATTRIBUTABLE_NO_FULL_KNOWN_GOOD_SOLUTION",
            "edge_family": "SMALL_EFFECT_DIRECTION_FOUND_MODE_A_RECOVERY_FAIL",
            "beta": "KEEP_CLOSED_ROUTE_SHAPE_NOT_RADIAL_SCALE",
            "patient_interictal_reproduction": "NO_PARTIAL_PROPAGATION_PHENOTYPE_ONLY",
        },
        "field_target_residual_kl": field_kl,
        "patient_heldout_scores_computed": False,
        "inputs": {
            "fit": fit_input, "selection": selection_input,
            "fit_floor_n6": floor6_input, "selection_floor_n3": floor3_input,
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
        },
        "provenance": _provenance(args.expected_commit),
    }
    output_dir = root / "scientific_review"
    output_path = output_dir / "l2_scientific_review.json"
    atomic_write_json(payload, output_path)
    _plot(fit_rows, selection_rows, baseline_id, selected_id, field_kl,
          floor3, output_dir / "figures")
    print(json.dumps({
        "status": payload["status"],
        "corrected_selected_candidate_id": selected_id,
        "relative_improvement": payload["corrected_objective"]["relative_improvement"],
        "mode_a_above_floor_q95": mode_a_above_q95,
        "diagnosis": payload["diagnosis"],
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
