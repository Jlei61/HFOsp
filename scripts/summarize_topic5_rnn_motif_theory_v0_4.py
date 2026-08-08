#!/usr/bin/env python3
"""Target-free enrichment, task relation and architecture audit of effective motifs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from score_topic5_rnn_motif_early_ictal_v0_4 import paired_summary  # noqa: E402


def stable_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "little") % (2**63 - 1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: raise RuntimeError(f"empty motif table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def safe_spearman(x, y) -> dict[str, Any]:
    x, y = np.asarray(x, float), np.asarray(y, float)
    use = np.isfinite(x) & np.isfinite(y)
    if use.sum() < 5 or np.std(x[use]) == 0 or np.std(y[use]) == 0:
        return {"n": int(use.sum()), "rho": None, "p": None}
    result = spearmanr(x[use], y[use])
    return {"n": int(use.sum()), "rho": float(result.statistic), "p": float(result.pvalue)}


def pairwise_seed_stability(paths: list[Path]) -> float:
    """Median rank correlation of the full effective operator across seeds.

    Inactive and active edges are both retained, so this audits the stability of
    the effective topology and its weights rather than cherry-picking shared
    surviving edges.  Diagonal entries are excluded because self-edges are
    forbidden by contract.
    """
    vectors = []
    for path in sorted(paths):
        with np.load(path, allow_pickle=False) as data:
            value = np.asarray(data["edge_effective_influence"], float)
        use = ~np.eye(value.shape[0], dtype=bool)
        vectors.append(value[use])
    correlations = []
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            if np.std(vectors[left]) <= 0 or np.std(vectors[right]) <= 0:
                continue
            correlations.append(float(spearmanr(vectors[left], vectors[right]).statistic))
    return float(np.nanmedian(correlations)) if correlations else float("nan")


def unit_summary(path: Path, draws: int) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        summary = json.loads(str(data["summary_json"].item()))
        mask = np.asarray(data["edge_mask"], bool)
        influence = np.asarray(data["edge_effective_influence"], float)
        distance = np.asarray(data["edge_distance_mm"], float)
        connector = np.asarray(data["connector_nodes"], bool)
    active = np.flatnonzero(mask.ravel())
    values = influence.ravel()[active]
    lengths = distance.ravel()[active]
    if len(active) < 10:
        raise RuntimeError(f"too few active edges for motif null: {path}")
    long = lengths >= np.quantile(lengths, 0.75)
    local = lengths <= np.quantile(lengths, 0.50)
    n_top = max(1, int(np.ceil(0.10 * len(values))))
    order = np.argsort(values, kind="stable")[-n_top:]
    observed_long_top = float(long[order].mean())
    rng = np.random.default_rng(stable_seed(str(path)))
    null = np.empty(draws, float)
    for draw in range(draws):
        permuted = rng.permutation(values)
        null_order = np.argsort(permuted, kind="stable")[-n_top:]
        null[draw] = float(long[null_order].mean())
    mean_all = float(np.mean(values))
    output = {
        **summary,
        "long_top_fraction": observed_long_top,
        "long_top_null_median": float(np.median(null)),
        "long_top_enrichment": observed_long_top - float(np.median(null)),
        "long_top_empirical_p": float((1 + np.sum(null >= observed_long_top - 1e-15)) / (draws + 1)),
        "local_effective_ratio": float(np.mean(values[local]) / max(mean_all, 1e-12)),
        "long_effective_ratio": float(np.mean(values[long]) / max(mean_all, 1e-12)),
        "connector_node_fraction": float(connector.mean()),
        "motif_score": float(
            np.mean(values[local]) / max(mean_all, 1e-12)
            + observed_long_top - float(np.median(null))
        ),
        "influence_permutation_draws": draws,
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    influence_paths = sorted(
        (out_root / "effective_influence").glob("*/*__*/seed*/influence.npz")
    )
    unit_rows = [unit_summary(path, args.draws) for path in influence_paths]
    write_csv(out_root / "effective_motif_fit_seed.csv", unit_rows)
    numeric = [key for key, value in unit_rows[0].items() if isinstance(value, (int, float))
               and key not in {"seed"}]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in unit_rows:
        grouped.setdefault((row["subject"], row["model"], row["cell"]), []).append(row)
    fit_paths: dict[tuple[str, str, str], list[Path]] = {}
    for path, row in zip(influence_paths, unit_rows):
        fit_paths.setdefault((row["fit_id"], row["model"], row["cell"]), []).append(path)
    fit_stability = {
        key: pairwise_seed_stability(paths) for key, paths in fit_paths.items()
    }
    patient_rows = []
    for (subject, model, cell), selected in sorted(grouped.items()):
        item = {"subject": subject, "model": model, "cell": cell, "n_fit_seed_units": len(selected)}
        item.update({key: float(np.nanmedian([row[key] for row in selected])) for key in numeric})
        subject_fits = sorted({row["fit_id"] for row in selected})
        item["effective_operator_seed_stability"] = float(np.nanmedian([
            fit_stability.get((fit_id, model, cell), np.nan) for fit_id in subject_fits
        ]))
        patient_rows.append(item)
    write_csv(out_root / "effective_motif_patient.csv", patient_rows)

    inter = {(row["subject"], row["model"], row["cell"]): row
             for row in read_csv(out_root / "interictal_per_patient.csv")}
    fields = {(row["subject"], row["model"], row["cell"]): row
              for row in read_csv(out_root / "model_field_patient_metrics.csv")}
    models = sorted({(row["model"], row["cell"]) for row in patient_rows})
    associations = {}
    enrichment = {}
    for model, cell in models:
        selected = [row for row in patient_rows if row["model"] == model and row["cell"] == cell]
        enrichment[f"{model}|{cell}"] = {
            "long_top_enrichment": paired_summary(
                [row["long_top_enrichment"] for row in selected], seed=stable_seed(model + cell + "long")
            ),
            "local_effective_ratio_minus_one": paired_summary(
                [row["local_effective_ratio"] - 1.0 for row in selected],
                seed=stable_seed(model + cell + "local"),
            ),
        }
        motif, rollout, fidelity, wiring = [], [], [], []
        for row in selected:
            key = (row["subject"], model, cell)
            if key not in inter or key not in fields: continue
            motif.append(row["motif_score"])
            rollout.append(float(inter[key]["rollout_spearman"]))
            fidelity.append(float(fields[key]["matched_empirical_r"]))
            wiring.append(float(inter[key]["c_wiring"]))
        associations[f"{model}|{cell}"] = {
            "motif_vs_rollout": safe_spearman(motif, rollout),
            "motif_vs_empirical_field_fidelity": safe_spearman(motif, fidelity),
            "motif_vs_wiring_cost": safe_spearman(motif, wiring),
        }

    patient_lookup = {(row["subject"], row["model"], row["cell"]): row for row in patient_rows}
    subjects = sorted({row["subject"] for row in patient_rows})
    proposal_difference = [
        patient_lookup[(subject, "M6_SPATIAL_MID", "rnn")]["motif_score"]
        - patient_lookup[(subject, "C_ORDER_SHUFFLED", "rnn")]["motif_score"]
        for subject in subjects
        if (subject, "M6_SPATIAL_MID", "rnn") in patient_lookup
        and (subject, "C_ORDER_SHUFFLED", "rnn") in patient_lookup
    ]
    architecture = {}
    for model in sorted({row["model"] for row in patient_rows}):
        rnn = [row["motif_score"] for row in patient_rows if row["model"] == model and row["cell"] == "rnn"]
        gru = [row["motif_score"] for row in patient_rows if row["model"] == model and row["cell"] == "gru"]
        architecture[model] = {
            "rnn_median": float(np.nanmedian(rnn)) if rnn else None,
            "gru_median": float(np.nanmedian(gru)) if gru else None,
            "same_direction_relative_to_zero": bool(rnn and gru and np.nanmedian(rnn) * np.nanmedian(gru) >= 0),
        }
    lesion = json.loads((out_root / "MATCHED_LESION_SUMMARY.json").read_text())
    m6_key = "M6_SPATIAL_MID|rnn"
    m6_enrichment = enrichment.get(m6_key, {})
    m6_association = associations.get(m6_key, {})
    stability = paired_summary(
        [row["effective_operator_seed_stability"] for row in patient_rows
         if row["model"] == "M6_SPATIAL_MID" and row["cell"] == "rnn"],
        seed=stable_seed("M6 effective operator seed stability"),
    )
    lesion_stats = lesion.get("statistics", {})

    def positive_significant(value: dict[str, Any] | None, estimate: str = "median") -> bool:
        return bool(value and (value.get(estimate) or 0) > 0
                    and (value.get("wilcoxon_p") or 1) < 0.05)

    local_lesion = lesion_stats.get("M6_SPATIAL_MID|local_backbone_edges")
    long_lesion = lesion_stats.get("M6_SPATIAL_MID|long_range_high_influence_edges")
    connector_lesion = lesion_stats.get("M6_SPATIAL_MID|connector_nodes")
    task_relation = any(
        (value.get("rho") or 0) > 0 and (value.get("p") or 1) < 0.05
        for key, value in m6_association.items() if key != "motif_vs_wiring_cost"
    )
    proposal = paired_summary(proposal_difference, seed=stable_seed("M6 proposal"))
    claim_components = {
        "local_effective_enrichment": positive_significant(
            m6_enrichment.get("local_effective_ratio_minus_one")
        ),
        "long_range_effective_enrichment": positive_significant(
            m6_enrichment.get("long_top_enrichment")
        ),
        "effective_operator_seed_stability": positive_significant(stability),
        "task_relation": task_relation,
        "local_backbone_matched_lesion": positive_significant(
            local_lesion, "median_specificity_contact_nll"
        ),
        "long_range_or_connector_matched_lesion": (
            positive_significant(long_lesion, "median_specificity_contact_nll")
            or positive_significant(connector_lesion, "median_specificity_contact_nll")
        ),
        "not_binary_proposal_only": positive_significant(proposal),
    }
    payload = {
        "contract": "topic5_rnn_motif_theory_summary_v0_4",
        "target_values_read": False,
        "effective_weight_permutation_null": "influence values permuted over the frozen active edge mask",
        "enrichment": enrichment,
        "task_and_wiring_associations": associations,
        "effective_operator_seed_stability": {m6_key: stability},
        "M6_true_order_minus_order_shuffle_motif_score": proposal,
        "architecture_direction": architecture,
        "matched_lesion": lesion,
        "M6_motif_claim_components": claim_components,
        "M6_motif_claim_pass": all(claim_components.values()),
        "claim_rule": (
            "the same local-backbone plus long-range-connector motif must show both enrichments, "
            "cross-seed operator stability, task relation, local and long/connector matched-lesion "
            "specificity, and true-order benefit over the identical order-shuffled proposal"
        ),
    }
    (out_root / "EFFECTIVE_MOTIF_SUMMARY.json").write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
