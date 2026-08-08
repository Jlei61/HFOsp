"""Patient-first interictal analysis for the locked v0.4 model matrix."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

from src.topic5_wiring_economy_rnn import WEConfig, WEModel, build_event_tensors

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT / "scripts"))
from train_topic5_we_unit import evaluate  # noqa: E402


MODEL_ORDER = (
    "M0_NO_REC", "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL",
    "M4_SPATIAL_GROWTH", "M5_SPATIAL_LOW", "M6_SPATIAL_MID",
    "M7_SPATIAL_HIGH", "M8_UNIFORM_COST_MID", "C_ORDER_SHUFFLED",
    "C_FULL_RANK_SHUFFLED",
)
MINIMUM_DENSE_BENEFIT = 0.01
COLORS = {
    "M0_NO_REC": "#9d9da1", "M1_DENSE": "#222222", "M2_UNIFORM_SET": "#7f7f7f",
    "M3_FIXED_LOCAL": "#4c78a8", "M4_SPATIAL_GROWTH": "#72b7b2",
    "M5_SPATIAL_LOW": "#f2cf5b", "M6_SPATIAL_MID": "#e45756",
    "M7_SPATIAL_HIGH": "#b279a2", "M8_UNIFORM_COST_MID": "#f58518",
    "C_ORDER_SHUFFLED": "#5c6bc0", "C_FULL_RANK_SHUFFLED": "#9467bd",
}


def logical_model(directory: str) -> tuple[str, str]:
    model, cell = directory.rsplit("__", 1)
    return model, cell


def median(values: list[float]) -> float:
    finite = np.asarray([value for value in values if np.isfinite(value)], float)
    return float(np.median(finite)) if finite.size else float("nan")


def paired_test(values: np.ndarray, tolerance: float = 1e-9) -> dict[str, Any]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    positive = int((values > tolerance).sum())
    negative = int((values < -tolerance).sum())
    tied = int(len(values) - positive - negative)
    nonzero = values[np.abs(values) > tolerance]
    p = float(wilcoxon(nonzero, method="auto").pvalue) if nonzero.size else 1.0
    return {"n": int(len(values)), "median": median(values.tolist()),
            "positive": positive, "negative": negative, "tied": tied, "p_two_sided": p}


def bootstrap_ci(values: np.ndarray, seed: int = 20260809, draws: int = 10000) -> list[float]:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    samples = np.median(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    return np.quantile(samples, [0.025, 0.975]).astype(float).tolist()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def seed_removed_sequence_agreement(observed: np.ndarray, generated: list[list[int]]) -> float:
    """Rank correlation after deleting the supplied first rank from both sides."""
    observed = np.asarray(observed, int)
    generated_order = {contact: rank for rank, rank_set in enumerate(generated[1:])
                       for contact in rank_set}
    shared = [int(contact) for contact in np.flatnonzero(observed > 0)
              if int(contact) in generated_order]
    if len(shared) < 3:
        return float("nan")
    value = spearmanr([observed[contact] - 1 for contact in shared],
                      [generated_order[contact] for contact in shared]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def load_real_kept_events(out_root: Path, fit_id: str) -> tuple[np.ndarray, np.ndarray]:
    events = np.load(out_root / "cache" / fit_id / "events.npz")
    keep = events["split"] >= 0
    return np.asarray(events["ranks"])[keep], np.asarray(events["split"])[keep]


def rollout_rows_for_unit(out_root: Path, metrics_path: Path, metrics: dict[str, Any],
                          model: str, cell: str) -> list[dict[str, Any]]:
    ranks, split = load_real_kept_events(out_root, metrics["fit_id"])
    with gzip.open(metrics_path.parent / "heldout_rollouts.json.gz", "rt", encoding="utf-8") as handle:
        records = json.load(handle)
    rows = []
    for record in records:
        index = int(record["kept_event_index"])
        if split[index] != 2:
            raise RuntimeError(f"rollout is not a real heldout event: {metrics_path}: {index}")
        generated = record["generated_rank_sets"]
        observed_postseed = int((ranks[index] > 0).sum())
        generated_postseed = sum(len(rank_set) for rank_set in generated[1:])
        rows.append({
            "subject": metrics["subject"], "fit_id": metrics["fit_id"],
            "scope": metrics["fit_scope"], "model": model, "cell": cell,
            "seed": int(metrics["seed"]), "event_source_index": int(record["event_source_index"]),
            "event_abs_time": float(record["event_abs_time"]), "mode": int(record["mode"]),
            "seed_removed_rollout_spearman": seed_removed_sequence_agreement(ranks[index], generated),
            "observed_postseed_contacts": observed_postseed,
            "generated_postseed_contacts": generated_postseed,
            "postseed_length_ratio": generated_postseed / max(1, observed_postseed),
        })
    return rows


def rescore_real_targets(out_root: Path, metrics_path: Path, metrics: dict[str, Any],
                         device: torch.device) -> dict[str, float]:
    """Put shuffled-training controls on the same untouched heldout target as every arm."""
    if metrics.get("shuffle_mode", "none") == "none":
        return {key: float(metrics["test"][key])
                for key in ("next_bce", "stop_bce", "contact_nll", "top1")}
    cache = out_root / "cache" / metrics["fit_id"]
    plane = np.load(cache / "plane.npz")
    provenance = json.loads((cache / "provenance.json").read_text())
    ranks, split = load_real_kept_events(out_root, metrics["fit_id"])
    cfg = metrics["config"]
    config = WEConfig(
        arm=metrics["arm"], cell=cell_from_metrics(metrics),
        n_contacts=int(provenance["n_contacts"]), n_nodes=int(provenance["n_nodes"]),
        state_dim=int(cfg["state_dim"]), density=float(cfg["density"]),
        eta=float(cfg["eta"]), d0_mm=float(cfg["d0_mm"]), seed=int(metrics["seed"]),
        observation_operator=None if metrics["arm"] == "STATIC_CONTACT" else plane["H"],
        node_distance_mm=None if metrics["arm"] == "STATIC_CONTACT" else plane["D_mm"],
    )
    model = WEModel(config).to(device)
    state = torch.load(metrics_path.parent / "weights.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    tensors = build_event_tensors(ranks)
    return evaluate(model, tensors, device, event_mask=(split == 2))


def cell_from_metrics(metrics: dict[str, Any]) -> str:
    return str(metrics["cell"])


def load_rows(out_root: Path, device: torch.device) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    event_rows = []
    for path in sorted((out_root / "per_subject").glob("*/*__*/seed*/metrics.json")):
        directory = path.parents[1].name
        if directory.startswith("SMOKE_"):
            continue
        try:
            model_id, cell = logical_model(directory)
        except ValueError:
            continue
        if model_id not in MODEL_ORDER:
            continue
        metrics = json.loads(path.read_text())
        rescored = rescore_real_targets(out_root, path, metrics, device)
        unit_event_rows = rollout_rows_for_unit(out_root, path, metrics, model_id, cell)
        event_rows.extend(unit_event_rows)
        rows.append({
            "subject": metrics["subject"], "fit_id": metrics["fit_id"],
            "scope": metrics["fit_scope"], "model": model_id, "cell": cell,
            "seed": metrics["seed"], "converged": metrics["converged"],
            "next_bce": rescored["next_bce"], "stop_bce": rescored["stop_bce"],
            "contact_nll": rescored["contact_nll"], "top1": rescored["top1"],
            "validation_contact_nll": metrics["validation"]["contact_nll"],
            "rollout_spearman": median([row["seed_removed_rollout_spearman"] for row in unit_event_rows]),
            "length_ratio": median([row["postseed_length_ratio"] for row in unit_event_rows]),
            "generator_degenerate": bool(metrics.get("generator_degenerate") is True),
            "evaluation_target": "real_unshuffled_heldout",
            "edge_count": metrics["edge_count"],
            "mean_edge_len_mm": metrics.get("mean_edge_len_mm", float("nan")),
            "c_wiring": metrics["c_wiring"], "n_epochs": metrics["n_epochs"],
            "seconds": metrics["seconds"],
        })
    return rows, event_rows


def aggregate(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_fit: dict[tuple, list[dict]] = defaultdict(list)
    for row in rows:
        by_fit[(row["subject"], row["fit_id"], row["scope"], row["model"], row["cell"])].append(row)
    fits = []
    metrics = ("next_bce", "stop_bce", "contact_nll", "top1", "validation_contact_nll", "rollout_spearman",
               "length_ratio", "mean_edge_len_mm", "c_wiring", "n_epochs", "seconds")
    for key, values in by_fit.items():
        subject, fit_id, scope, model, cell = key
        fit = {"subject": subject, "fit_id": fit_id, "scope": scope,
               "model": model, "cell": cell, "n_seeds": len(values),
               "all_converged": all(row["converged"] for row in values),
               "any_generator_degenerate": any(row["generator_degenerate"] for row in values)}
        fit.update({metric: median([row[metric] for row in values]) for metric in metrics})
        fits.append(fit)

    by_patient: dict[tuple, list[dict]] = defaultdict(list)
    for fit in fits:
        by_patient[(fit["subject"], fit["model"], fit["cell"])].append(fit)
    patients = []
    for (subject, model, cell), values in by_patient.items():
        row = {"subject": subject, "model": model, "cell": cell,
               "n_fits": len(values), "all_converged": all(v["all_converged"] for v in values),
               "any_generator_degenerate": any(v["any_generator_degenerate"] for v in values),
               "fit_aggregation": "shared" if len(values) == 1 and values[0]["scope"] == "shared" else "mean_own_a_own_b"}
        row.update({metric: median([value[metric] for value in values]) for metric in metrics})
        patients.append(row)
    return fits, patients


def stats_for(patients: list[dict[str, Any]], cell: str) -> dict[str, Any]:
    table = {(row["subject"], row["model"]): row for row in patients if row["cell"] == cell}
    subjects = sorted({subject for subject, model in table if model == "M0_NO_REC"})
    available_models = [model for model in MODEL_ORDER if all((subject, model) in table for subject in subjects)]
    comparisons = {}
    for model in available_models:
        if model == "M0_NO_REC":
            continue
        gain = np.array([table[(subject, "M0_NO_REC")]["contact_nll"]
                         - table[(subject, model)]["contact_nll"] for subject in subjects])
        comparisons[f"{model}_vs_M0"] = {**paired_test(gain), "bootstrap_95ci": bootstrap_ci(gain)}
        stop_gain = np.array([table[(subject, "M0_NO_REC")]["stop_bce"]
                              - table[(subject, model)]["stop_bce"] for subject in subjects])
        rollout_gain = np.array([table[(subject, model)]["rollout_spearman"]
                                 - table[(subject, "M0_NO_REC")]["rollout_spearman"]
                                 for subject in subjects])
        comparisons[f"{model}_vs_M0"]["stop_bce_gain"] = paired_test(stop_gain)
        comparisons[f"{model}_vs_M0"]["seed_removed_rollout_gain"] = paired_test(rollout_gain)
    if "M6_SPATIAL_MID" in available_models and "C_ORDER_SHUFFLED" in available_models:
        gain = np.array([table[(subject, "C_ORDER_SHUFFLED")]["contact_nll"]
                         - table[(subject, "M6_SPATIAL_MID")]["contact_nll"] for subject in subjects])
        comparisons["M6_true_order_vs_shuffle"] = {**paired_test(gain), "bootstrap_95ci": bootstrap_ci(gain, seed=17)}

    factorial = {}
    required = {"M2_UNIFORM_SET", "M4_SPATIAL_GROWTH", "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID"}
    if required.issubset(available_models):
        score = lambda subject, model: -table[(subject, model)]["contact_nll"]
        definitions = {
            "growth_at_zero": lambda s: score(s, "M4_SPATIAL_GROWTH") - score(s, "M2_UNIFORM_SET"),
            "growth_at_mid": lambda s: score(s, "M6_SPATIAL_MID") - score(s, "M8_UNIFORM_COST_MID"),
            "cost_uniform": lambda s: score(s, "M8_UNIFORM_COST_MID") - score(s, "M2_UNIFORM_SET"),
            "cost_spatial": lambda s: score(s, "M6_SPATIAL_MID") - score(s, "M4_SPATIAL_GROWTH"),
            "interaction": lambda s: ((score(s, "M6_SPATIAL_MID") - score(s, "M4_SPATIAL_GROWTH"))
                                      - (score(s, "M8_UNIFORM_COST_MID") - score(s, "M2_UNIFORM_SET"))),
        }
        for name, function in definitions.items():
            values = np.array([function(subject) for subject in subjects])
            factorial[name] = {**paired_test(values), "bootstrap_95ci": bootstrap_ci(values, seed=31)}

    return {"cell": cell, "n_patients": len(subjects), "available_models": available_models,
            "comparisons": comparisons, "factorial": factorial}


def patient_bootstrap(event_rows: list[dict[str, Any]], draws: int = 2000,
                      seed: int = 20260809) -> dict[str, Any]:
    grouped: dict[tuple, list[float]] = defaultdict(list)
    meta: dict[str, set[str]] = defaultdict(set)
    for row in event_rows:
        value = float(row["seed_removed_rollout_spearman"])
        if np.isfinite(value):
            grouped[(row["subject"], row["fit_id"], row["model"], row["cell"],
                     int(row["event_source_index"]))].append(value)
            meta[row["subject"]].add(row["fit_id"])
    event_median = {key: float(np.median(value)) for key, value in grouped.items()}
    rng = np.random.default_rng(seed)
    output = {}
    contrasts = [(model, "M0_NO_REC") for model in MODEL_ORDER if model != "M0_NO_REC"]
    contrasts.append(("M6_SPATIAL_MID", "C_ORDER_SHUFFLED"))
    for subject, fit_ids in sorted(meta.items()):
        output[subject] = {}
        for model, baseline in contrasts:
            fit_differences = []
            for fit_id in sorted(fit_ids):
                a = {key[-1]: value for key, value in event_median.items()
                     if key[:4] == (subject, fit_id, model, "rnn")}
                b = {key[-1]: value for key, value in event_median.items()
                     if key[:4] == (subject, fit_id, baseline, "rnn")}
                common = sorted(set(a) & set(b))
                if common:
                    fit_differences.append(np.asarray([a[index] - b[index] for index in common], float))
            if not fit_differences:
                continue
            observed = float(np.mean([np.median(values) for values in fit_differences]))
            samples = np.empty(draws, float)
            for draw in range(draws):
                samples[draw] = np.mean([
                    np.median(rng.choice(values, len(values), replace=True))
                    for values in fit_differences
                ])
            output[subject][f"{model}_minus_{baseline}"] = {
                "estimate": observed,
                "bootstrap_95ci": np.quantile(samples, [0.025, 0.975]).astype(float).tolist(),
                "strict_positive": bool(np.quantile(samples, 0.025) > 0),
                "n_events_by_fit": [int(len(values)) for values in fit_differences],
            }
    return output


def plot_stage(out_root: Path, patients: list[dict[str, Any]], stats: dict[str, Any]) -> None:
    cell = "rnn"
    table = {(row["subject"], row["model"]): row for row in patients if row["cell"] == cell}
    subjects = sorted({subject for subject, model in table if model == "M0_NO_REC"})
    models = [m for m in ("M0_NO_REC", "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL",
                           "M4_SPATIAL_GROWTH", "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID")
              if all((subject, m) in table for subject in subjects)]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.linewidth": 0.7})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), constrained_layout=True)
    for x, model in enumerate(models):
        values = np.array([table[(subject, model)]["contact_nll"] for subject in subjects])
        axes[0].scatter(np.full(len(values), x) + np.linspace(-0.10, 0.10, len(values)), values,
                        s=12, alpha=0.55, color=COLORS[model], linewidths=0)
        axes[0].plot([x - 0.20, x + 0.20], [np.median(values)] * 2, color="#111111", linewidth=1.3)
    axes[0].set_xticks(range(len(models)), [m.split("_", 1)[0] for m in models])
    axes[0].set_ylabel("Held-out contact NLL")
    axes[0].set_title("Interictal prediction", loc="left", fontweight="bold")

    for x, model in enumerate(models):
        values = np.array([table[(subject, model)]["rollout_spearman"] for subject in subjects])
        axes[1].scatter(np.full(len(values), x) + np.linspace(-0.10, 0.10, len(values)), values,
                        s=12, alpha=0.55, color=COLORS[model], linewidths=0)
        axes[1].plot([x - 0.20, x + 0.20], [np.nanmedian(values)] * 2, color="#111111", linewidth=1.3)
    axes[1].axhline(0, color="#999999", linewidth=0.7)
    axes[1].set_xticks(range(len(models)), [m.split("_", 1)[0] for m in models])
    axes[1].set_ylabel("Rollout rank correlation")
    axes[1].set_title("Free propagation", loc="left", fontweight="bold")

    names = ["growth_at_zero", "growth_at_mid", "cost_uniform", "cost_spatial", "interaction"]
    names = [name for name in names if name in stats["factorial"]]
    values = [stats["factorial"][name]["median"] for name in names]
    ci = np.array([stats["factorial"][name]["bootstrap_95ci"] for name in names]) if names else np.empty((0, 2))
    axes[2].barh(range(len(names)), values, color=["#72b7b2", "#e45756", "#f58518", "#b279a2", "#4c78a8"][:len(names)])
    if names:
        axes[2].errorbar(values, range(len(names)), xerr=np.abs(ci - np.asarray(values)[:, None]).T,
                         fmt="none", ecolor="#222222", linewidth=0.8, capsize=2)
    axes[2].axvline(0, color="#555555", linewidth=0.8)
    axes[2].set_yticks(range(len(names)), [name.replace("_", " ") for name in names])
    axes[2].set_xlabel("Δ predictive score")
    axes[2].set_title("Growth × wiring cost", loc="left", fontweight="bold")
    for label, axis in zip("abc", axes):
        axis.text(-0.17, 1.04, label, transform=axis.transAxes, fontsize=12,
                  fontweight="bold", va="bottom")
    figure_dir = out_root / "figures"
    stem = figure_dir / "stage_d_interictal_model_matrix"
    fig.savefig(stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def convergence_audit(out_root: Path) -> dict[str, Any]:
    """Audit convergence, final slopes, graph resources and producer identity."""
    records = []
    for path in sorted((out_root / "per_subject").glob("*/*__*/seed*/metrics.json")):
        if path.parents[1].name.startswith("SMOKE_"):
            continue
        metrics = json.loads(path.read_text())
        history = json.loads((path.parent / "history.json").read_text())
        tail = np.asarray([row["val"] for row in history[-5:]], float)
        slope = (float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
                 if len(tail) >= 2 else float("nan"))
        expected_edges = (0 if metrics["arm"] == "STATIC_CONTACT" else
                          metrics["n_nodes"] * (metrics["n_nodes"] - 1)
                          if metrics["arm"] == "DENSE_TISSUE" else
                          int(round(metrics["config"]["density"] * metrics["n_nodes"]
                                    * (metrics["n_nodes"] - 1))))
        snapshots = path.parent / "snapshots"
        snapshot_applicable = metrics["arm"] != "STATIC_CONTACT"
        records.append({
            "fit_id": metrics["fit_id"], "model_id": metrics["model_id"],
            "cell": metrics["cell"], "seed": int(metrics["seed"]),
            "converged": bool(metrics["converged"]), "hit_ceiling": bool(metrics["hit_ceiling"]),
            "n_epochs": int(metrics["n_epochs"]), "last5_validation_slope": slope,
            "edge_count": int(metrics["edge_count"]), "expected_edge_count": int(expected_edges),
            "edge_budget_valid": int(metrics["edge_count"]) == int(expected_edges),
            "snapshot_applicable": snapshot_applicable,
            "all_four_snapshots_present": (not snapshot_applicable) or all(
                (snapshots / f"{name}.npz").exists()
                for name in ("INIT", "REWIRE_MID", "MASK_FREEZE", "FINAL")
            ),
            "producer_hashes": metrics["producer_hashes"],
        })
    grouped = {}
    for model_id in sorted({row["model_id"] for row in records}):
        selected = [row for row in records if row["model_id"] == model_id]
        grouped[model_id] = {
            "n": len(selected), "n_converged": sum(row["converged"] for row in selected),
            "n_hit_ceiling": sum(row["hit_ceiling"] for row in selected),
            "median_epochs": float(np.median([row["n_epochs"] for row in selected])),
            "median_last5_validation_slope": float(np.nanmedian([
                row["last5_validation_slope"] for row in selected
            ])),
            "all_edge_budgets_valid": all(row["edge_budget_valid"] for row in selected),
            "all_four_snapshots_present": all(row["all_four_snapshots_present"] for row in selected),
        }
    hashes = {}
    for key in ("trainer", "model", "v0_4_contract", "input_manifest"):
        values = sorted({row["producer_hashes"][key] for row in records})
        hashes[key] = {"n_unique": len(values), "values": values}
    payload = {
        "contract": "topic5_rnn_motif_convergence_audit_v0_4",
        "n_units": len(records), "expected_units": 1426,
        "n_converged": sum(row["converged"] for row in records),
        "n_hit_ceiling": sum(row["hit_ceiling"] for row in records),
        "all_edge_budgets_valid": all(row["edge_budget_valid"] for row in records),
        "all_four_snapshots_present": all(row["all_four_snapshots_present"] for row in records),
        "producer_hashes": hashes, "by_model_cell": grouped,
        "rows": records,
    }
    (out_root / "CONVERGENCE_AUDIT.json").write_text(json.dumps(payload, indent=2))
    return payload


def adequacy_and_retention(patients: list[dict[str, Any]], cell: str,
                           delta_ni: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table = {(row["subject"], row["model"]): row for row in patients if row["cell"] == cell}
    subjects = sorted({subject for subject, model in table if model == "M0_NO_REC"})
    models = [model for model in MODEL_ORDER if model != "M0_NO_REC"
              and all((subject, model) in table for subject in subjects)]
    tiers = {}
    retention_rows = []
    for model in models:
        rollout_gain = np.asarray([
            table[(subject, model)]["rollout_spearman"]
            - table[(subject, "M0_NO_REC")]["rollout_spearman"] for subject in subjects
        ], float)
        dense_gap = np.asarray([
            table[(subject, model)]["contact_nll"]
            - table[(subject, "M1_DENSE")]["contact_nll"] for subject in subjects
        ], float) if all((subject, "M1_DENSE") in table for subject in subjects) else np.asarray([])
        dense_gap_ci = bootstrap_ci(dense_gap, seed=101 + MODEL_ORDER.index(model)) if dense_gap.size else [float("nan")] * 2
        length_ratio = np.asarray([table[(subject, model)]["length_ratio"] for subject in subjects], float)
        noncollapse = bool(np.isfinite(length_ratio).all() and np.nanmedian(length_ratio) > 0
                           and np.isfinite([table[(subject, model)]["rollout_spearman"]
                                            for subject in subjects]).any())
        recurrence_positive = bool(np.nanmedian(rollout_gain) > 0)
        noninferior = bool(dense_gap.size and np.isfinite(delta_ni)
                           and dense_gap_ci[1] < delta_ni)
        tier = ("ADEQUATE_STRONG" if recurrence_positive and noncollapse and noninferior
                else "ADEQUATE_PARTIAL" if recurrence_positive and noncollapse else "INADEQUATE")
        tiers[model] = {
            "tier": tier,
            "recurrence_gain": {**paired_test(rollout_gain),
                                "bootstrap_95ci": bootstrap_ci(rollout_gain, seed=71)},
            "rollout_noncollapse": noncollapse,
            "median_postseed_length_ratio": median(length_ratio.tolist()),
            "dense_gap_model_minus_dense_nll": {
                **paired_test(dense_gap), "bootstrap_95ci": dense_gap_ci,
            } if dense_gap.size else None,
            "delta_NI": delta_ni,
            "noninferior_to_dense": noninferior,
        }
        for subject in subjects:
            denominator = (table[(subject, "M0_NO_REC")]["contact_nll"]
                           - table[(subject, "M1_DENSE")]["contact_nll"])
            value = ((table[(subject, "M0_NO_REC")]["contact_nll"]
                      - table[(subject, model)]["contact_nll"]) / denominator
                     if denominator > MINIMUM_DENSE_BENEFIT else float("nan"))
            retention_rows.append({
                "subject": subject, "cell": cell, "model": model,
                "dense_benefit_nll": denominator,
                "dense_benefit_retention": value,
                "retention_available": bool(denominator > MINIMUM_DENSE_BENEFIT),
            })
    return {"cell": cell, "minimum_dense_benefit": MINIMUM_DENSE_BENEFIT,
            "models": tiers}, retention_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    rows, event_rows = load_rows(out_root, torch.device(args.device))
    audit = convergence_audit(out_root)
    if audit["n_units"] != audit["expected_units"]:
        raise RuntimeError(f"formal convergence audit incomplete: {audit['n_units']}/1426")
    fits, patients = aggregate(rows)
    write_csv(out_root / "interictal_per_event.csv", event_rows)
    write_csv(out_root / "interictal_per_fit_seed.csv", rows)
    write_csv(out_root / "interictal_fit_metrics.csv", fits)
    write_csv(out_root / "interictal_per_patient.csv", patients)
    cells = sorted({row["cell"] for row in patients})
    statistics = {cell: stats_for(patients, cell) for cell in cells}

    # Freeze the non-inferiority margin from eight montage-stratified development fits.
    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    ordered = sorted(manifest["fits"], key=lambda row: (row["n_contacts"], row["fit_id"]))
    positions = sorted(set(round(i * (len(ordered) - 1) / 7) for i in range(8)))
    development = [ordered[index]["fit_id"] for index in positions]
    fit_table = {(row["fit_id"], row["model"], row["cell"]): row for row in fits}
    dense_gain = [fit_table[(fit_id, "M0_NO_REC", "rnn")]["validation_contact_nll"]
                  - fit_table[(fit_id, "M1_DENSE", "rnn")]["validation_contact_nll"]
                  for fit_id in development
                  if (fit_id, "M0_NO_REC", "rnn") in fit_table and (fit_id, "M1_DENSE", "rnn") in fit_table]
    positive = [value for value in dense_gain if value > 0]
    delta_ni = 0.10 * median(positive) if positive else float("nan")
    adequacy = {}
    retention_rows = []
    for cell in cells:
        cell_adequacy, cell_retention = adequacy_and_retention(patients, cell, delta_ni)
        adequacy[cell] = cell_adequacy
        retention_rows.extend(cell_retention)
    write_csv(out_root / "dense_benefit_retention.csv", retention_rows)
    (out_root / "task_adequacy_tiers.json").write_text(json.dumps(adequacy, indent=2))
    bootstraps = patient_bootstrap(event_rows)
    (out_root / "interictal_bootstrap.json").write_text(json.dumps(bootstraps, indent=2))
    (out_root / "factorial_effects_interictal.json").write_text(json.dumps(
        {cell: statistics[cell]["factorial"] for cell in statistics}, indent=2
    ))
    pareto_rows = [
        {key: row[key] for key in ("subject", "model", "cell", "contact_nll",
                                   "rollout_spearman", "mean_edge_len_mm", "c_wiring")}
        for row in patients
    ]
    write_csv(out_root / "accuracy_wiring_pareto.csv", pareto_rows)
    summary = {"n_raw_units": len(rows), "n_fit_rows": len(fits), "n_patient_rows": len(patients),
               "n_event_rows": len(event_rows),
               "development_fits": development, "development_dense_gains": dense_gain,
               "delta_NI": delta_ni, "minimum_dense_benefit": MINIMUM_DENSE_BENEFIT,
               "primary_evaluation_target": "real_unshuffled_heldout_for_all_models",
               "rollout_seed_policy": "supplied rank 1 deleted from both observed and generated score",
               "statistics": statistics, "task_adequacy": adequacy,
               "convergence_audit": {
                   "n_converged": audit["n_converged"],
                   "n_hit_ceiling": audit["n_hit_ceiling"],
                   "all_edge_budgets_valid": audit["all_edge_budgets_valid"],
                   "all_four_snapshots_present": audit["all_four_snapshots_present"],
               },
               "target_values_read": False}
    (out_root / "INTERICTAL_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    (out_root / "stage_d_scientific_drift_audit.json").write_text(json.dumps({
        "status": "ALIGNED",
        "target_values_read": False,
        "scientific_question": (
            "which recurrent connectivity constraints are sufficient for the same patient-specific "
            "heldout interictal propagation task"
        ),
        "primary_corrections_applied": [
            "all models rescored against the untouched heldout sequence",
            "supplied first-rank contacts removed from observed and generated rollout correlation",
            "patient-first own_a/own_b aggregation kept separate from later field aggregation",
        ],
        "not_claimed": ["true connectome recovery", "seizure prediction", "hidden neural manifold"],
    }, indent=2))
    if "rnn" in statistics and len(statistics["rnn"]["available_models"]) >= 7:
        plot_stage(out_root, patients, statistics["rnn"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
