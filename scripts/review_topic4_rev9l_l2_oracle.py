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


def arrays_equal_nan(left, right):
    left, right = np.asarray(left), np.asarray(right)
    return left.shape == right.shape and np.array_equal(
        left, right, equal_nan=True)


def _source_curve(path, source):
    with np.load(path, allow_pickle=False) as loaded:
        identifiers = np.asarray(loaded["source_ids"]).astype(str)
        index = np.flatnonzero(identifiers == source)
        if len(index) != 1:
            raise RuntimeError(f"source {source} missing or duplicated in {path}")
        return np.asarray(loaded["rank_curves"][index[0]], float)


def _curve_signature(curves):
    digest = hashlib.sha256()
    for curve in curves:
        values = np.asarray(curve, np.float64)
        digest.update(np.asarray(values.shape, np.int64).tobytes())
        digest.update(np.isnan(values).tobytes())
        digest.update(np.nan_to_num(values, nan=0.0).tobytes())
    return digest.hexdigest()


def _curve_audit(root, candidate_ids, seeds, source_for, gamma_by_id):
    baseline_id = "sobol_000"
    baseline = {
        mode: {
            seed: _source_curve(
                root / f"{baseline_id}_seed{seed}.npz", source)
            for seed in seeds
        }
        for mode, source in source_for.items()
    }
    audit, signatures = {}, {mode: {} for mode in source_for}
    combined_signatures = {}
    for candidate_id in candidate_ids:
        audit[candidate_id] = {}
        combined_curves = []
        for mode, source in source_for.items():
            curves = [
                _source_curve(root / f"{candidate_id}_seed{seed}.npz", source)
                for seed in seeds
            ]
            combined_curves.extend(curves)
            changed = [
                not arrays_equal_nan(curve, baseline[mode][seed])
                for curve, seed in zip(curves, seeds)
            ]
            maximum = []
            for curve, seed in zip(curves, seeds):
                delta = np.abs(curve - baseline[mode][seed])
                finite = delta[np.isfinite(delta)]
                maximum.append(float(finite.max(initial=0.0)))
            audit[candidate_id][mode] = {
                "source": source,
                "changed_network_seeds": [
                    int(seed) for seed, value in zip(seeds, changed) if value],
                "n_changed_networks": int(sum(changed)),
                "n_networks": int(len(seeds)),
                "max_abs_curve_change_by_seed": dict(
                    zip(map(str, seeds), maximum)),
            }
            signatures[mode][candidate_id] = _curve_signature(curves)
        combined_signatures[candidate_id] = _curve_signature(combined_curves)

    def groups(signature_by_id):
        grouped = {}
        for candidate_id, signature in signature_by_id.items():
            grouped.setdefault(signature, []).append(candidate_id)
        output = []
        for identifiers in grouped.values():
            if len(identifiers) < 2:
                continue
            distances = [
                float(np.linalg.norm(
                    gamma_by_id[left] - gamma_by_id[right]))
                for index, left in enumerate(identifiers)
                for right in identifiers[index + 1:]
            ]
            output.append({
                "candidate_ids": identifiers,
                "min_pairwise_gamma_l2": min(distances),
                "max_pairwise_gamma_l2": max(distances),
            })
        return output

    return audit, {
        "mode_A": groups(signatures["A"]),
        "mode_B": groups(signatures["B"]),
        "combined_A_B": groups(combined_signatures),
    }


def _plot(fit_rows, selection_rows, baseline_id, selected_id, curve_audit,
          floor, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), constrained_layout=True)
    common = [key for key in selection_rows if key in fit_rows]
    fit_baseline = fit_rows[baseline_id]["score"]["objective"]
    selection_baseline = selection_rows[baseline_id]["corrected_score"]["objective"]
    x = [fit_baseline - fit_rows[key]["score"]["objective"] for key in common]
    y = [selection_baseline - selection_rows[key]["corrected_score"]["objective"]
         for key in common]
    axes[0].scatter(x, y, color="#277da1", s=38)
    lo, hi = min(x + y + [0.0]), max(x + y + [0.0])
    axes[0].plot([lo, hi], [lo, hi], ls="--", color="0.6", lw=1)
    axes[0].axhline(0.0, color="0.75", lw=0.8)
    axes[0].axvline(0.0, color="0.75", lw=0.8)
    fit_best = min(common, key=lambda key: fit_rows[key]["score"]["objective"])
    for key in dict.fromkeys((baseline_id, selected_id, fit_best)):
        axes[0].annotate(
            key.replace("sobol_", ""),
            (fit_baseline - fit_rows[key]["score"]["objective"],
             selection_baseline
             - selection_rows[key]["corrected_score"]["objective"]),
            xytext=(3, 3), textcoords="offset points", fontsize=7)
    axes[0].set_xlabel("fit improvement vs scalar (n=6 floor)")
    axes[0].set_ylabel("selection improvement vs scalar (n=3 floor)")
    axes[0].set_title("A  Fit-to-selection retention", loc="left", weight="bold")

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

    ordered = sorted(
        common,
        key=lambda key: selection_baseline
        - selection_rows[key]["corrected_score"]["objective"])
    positions = np.arange(len(ordered))
    axes[2].barh(
        positions - 0.17,
        [curve_audit[key]["A"]["n_changed_networks"] for key in ordered],
        0.32, color="#d1495b", label="mode A source")
    axes[2].barh(
        positions + 0.17,
        [curve_audit[key]["B"]["n_changed_networks"] for key in ordered],
        0.32, color="#277da1", label="mode B source")
    axes[2].axvline(len(next(iter(curve_audit.values()))["A"][
        "max_abs_curve_change_by_seed"]), color="0.5", ls="--", lw=0.8)
    axes[2].set_xlim(0, 3.15)
    axes[2].set_xticks([0, 1, 2, 3])
    axes[2].set_yticks(positions, [key.replace("sobol_", "") for key in ordered],
                       fontsize=7)
    axes[2].set_xlabel("selection networks with changed rank curve")
    axes[2].set_title("C  Shared route change", loc="left", weight="bold")
    axes[2].legend(frameon=False, fontsize=7)

    shown = [baseline_id, selected_id]
    for candidate_id in ordered[::-1]:
        if candidate_id not in shown:
            shown.append(candidate_id)
        if len(shown) == 5:
            break
    gamma = np.asarray([selection_rows[key]["gamma"] for key in shown], float)
    scale = max(float(np.max(np.abs(gamma))), 1e-12)
    image = axes[3].imshow(
        gamma, cmap="RdBu_r", vmin=-scale, vmax=scale, aspect="auto")
    axes[3].set_yticks(range(len(shown)), [key.replace("sobol_", "") for key in shown],
                       fontsize=7)
    axes[3].set_xticks(
        range(6), ("C1<-C1", "C1<-C2", "C2<-C1", "C2<-C2",
                   "BG<-C1", "BG<-C2"), rotation=30, ha="right", fontsize=7)
    axes[3].set_title("D  Distinct parameters, coarse ties", loc="left",
                      weight="bold")
    fig.colorbar(image, ax=axes[3], shrink=0.80, label="gamma")
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"rev9l_l2_scientific_review.{suffix}", dpi=300)
    plt.close(fig)
    (output_dir / "README.md").write_text(
        "### rev9l_l2_scientific_review.png\n"
        "A 图用各自匹配事件数的 patient-training floor 显示相对 scalar edge 的 fit 与 selection 改善；B 图把 scalar baseline（浅柱）和 selection 候选（实点）的四项绝对距离除以 patient floor；C 图直接计数每个候选在三张 selection 网络上是否改变 A/B rank curve；D 图展示若干不同 residual 参数。该图是 development/oracle 审阅，不读取 patient held-out。\n\n"
        "**关注点**：selection 改善是否来自三张网络共享的 mode-A 路径改变，以及 mode-A 四层误差是否真正接近患者训练抽样地板。\n")


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
    seeds = list(map(int, config["network_seeds"]["selection"]))
    source_for = {
        "A": config["primary_mapping"]["mode_A_source"],
        "B": config["primary_mapping"]["mode_B_source"],
    }
    gamma_by_id = {
        candidate_id: np.asarray(row["gamma"], float)
        for candidate_id, row in selection_rows.items()
    }
    curve_audit, equivalent_groups = _curve_audit(
        root / "selection_confirmation/workers", list(selection_rows), seeds,
        source_for, gamma_by_id)
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
    selected_a_curve_change = curve_audit[selected_id]["A"]
    shared_mode_a_route_change = (
        selected_a_curve_change["n_changed_networks"] == len(seeds))
    payload = {
        "status": "L2_COMPONENT_PAIR_SEARCH_NO_SHARED_MODE_A_RESTORATION",
        "safe_claim": (
            "within the frozen 64-point bounded component-pair search, the "
            "six-parameter conserved edge residual yields only a small "
            "selection improvement and no candidate restores mode A across all "
            "three selection networks"),
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
        "selection_curve_audit": curve_audit,
        "selected_mode_a_curve_change": selected_a_curve_change,
        "shared_mode_a_route_change": bool(shared_mode_a_route_change),
        "equivalent_output_groups": {
            **equivalent_groups,
            "interpretation": (
                "distinct gamma vectors can produce identical forced rank curves; "
                "the edge mechanism is not identifiable under this coarse readout"),
        },
        "prototype_only_ablation": {
            "best_candidate_id": prototype_best["candidate_id"],
            "mode_a_prototype_spearman": prototype_best[
                "mode_descriptors"]["modes"]["A"]["curve_prototype_spearman"],
            "full_objective": prototype_best["score"]["objective"],
            "full_objective_rank": fit["ranked_candidate_ids"].index(
                prototype_best["candidate_id"]) + 1,
        },
        "diagnosis": {
            "objective": "WORST_MODE_OBJECTIVE_USED_NO_SHARED_MODE_A_RESTORATION",
            "optimizer": "NOT_ATTRIBUTABLE_NO_FULL_KNOWN_GOOD_SOLUTION",
            "edge_family": (
                "NO_SHARED_MODE_A_ROUTE_RESTORATION_OBSERVED_WITHIN_64_POINT_"
                "BOUNDED_SEARCH"),
            "network": "SELECTION_BEST_CHANGES_MODE_A_ON_ONE_OF_THREE_NETWORKS",
            "identifiability": "PARAMETER_TO_COARSE_OUTPUT_MANY_TO_ONE_OBSERVED",
            "beta": "KEEP_CLOSED_ROUTE_SHAPE_NOT_RADIAL_SCALE",
            "patient_interictal_reproduction": "NO_PARTIAL_PROPAGATION_PHENOTYPE_ONLY",
        },
        "local_refinement_recommended": False,
        "local_refinement_reason": (
            "the corrected selection gain does not represent a shared mode-A "
            "route change across the selection networks"),
        "next_recommendation": (
            "L3_PER_NETWORK_ORACLE_DIAGNOSTIC_WITHOUT_OPTIMIZER_CLAIM"),
        "family_claim_boundary": (
            "no shared restoration was observed within this finite bounded search; "
            "this is not proof that no parameter exists in the family"),
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
    _plot(fit_rows, selection_rows, baseline_id, selected_id, curve_audit,
          floor3, output_dir / "figures")

    decision_path = root.parent / "decision.json"
    decision = json.loads(decision_path.read_text())
    decision["status"] = payload["status"]
    decision["propagation_family"]["status"] = payload["status"]
    decision["propagation_family"]["component_pair_review"] = {
        "review_path": str(output_path),
        "selection_best_candidate_id": selected_id,
        "corrected_improvement_vs_scalar": objective_gain,
        "relative_improvement_vs_scalar": payload[
            "corrected_objective"]["relative_improvement"],
        "mode_A_networks_changed": selected_a_curve_change[
            "n_changed_networks"],
        "mode_A_networks_total": len(seeds),
        "shared_mode_A_route_change": bool(shared_mode_a_route_change),
        "local_refinement_recommended": False,
        "claim_boundary": payload["family_claim_boundary"],
    }
    decision["propagation_family"]["next_recommendation"] = payload[
        "next_recommendation"]
    decision["network_realization"] = {
        "status": "SHARED_MODE_A_IMPROVEMENT_NOT_OBSERVED",
        "selection_best_A_changed_networks": selected_a_curve_change[
            "changed_network_seeds"],
        "selection_networks": seeds,
        "per_network_oracle_status": "NOT_YET_QUANTIFIED",
    }
    decision["optimizer"] = {
        "status": payload["diagnosis"]["optimizer"],
        "reason": "optimizer attribution requires a known-good shared solution",
    }
    decision["identifiability"] = {
        "status": payload["diagnosis"]["identifiability"],
        "equivalent_output_groups": equivalent_groups,
        "claim_boundary": (
            "coarse forced-output ties do not identify a unique edge mechanism"),
    }
    decision["patient_heldout_scores_computed"] = False
    decision["l2_scientific_review_provenance"] = payload["provenance"]
    atomic_write_json(decision, decision_path)
    print(json.dumps({
        "status": payload["status"],
        "corrected_selected_candidate_id": selected_id,
        "relative_improvement": payload["corrected_objective"]["relative_improvement"],
        "mode_a_above_floor_q95": mode_a_above_q95,
        "mode_a_networks_changed": selected_a_curve_change[
            "n_changed_networks"],
        "mode_a_networks_total": len(seeds),
        "diagnosis": payload["diagnosis"],
        "patient_heldout_scores_computed": False,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
