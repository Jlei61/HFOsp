#!/usr/bin/env python3
"""Target-free matched lesions of preassigned v0.4 recurrent motifs."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from analyse_topic5_rnn_motif_interictal_v0_4 import seed_removed_sequence_agreement  # noqa: E402
from build_topic5_rnn_motif_fields_v0_4 import aggregate_records, empirical_score  # noqa: E402
from train_topic5_we_unit import evaluate  # noqa: E402
from src.topic5_rnn_motif_v0_4 import RolloutSizeHead, rollout_with_size_head  # noqa: E402
from src.topic5_wiring_economy_rnn import WEConfig, WEModel, build_event_tensors  # noqa: E402


THEORY_MODELS = {
    "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL", "M4_SPATIAL_GROWTH",
    "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID", "C_ORDER_SHUFFLED",
}
TARGET_DRAWS = 500
MINIMUM_VALID = 200


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def number(value: Any) -> float:
    return float(value) if value is not None else float("nan")


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    use = np.isfinite(a) & np.isfinite(b)
    if use.sum() < 3 or np.std(a[use]) == 0 or np.std(b[use]) == 0:
        return float("nan")
    value = spearmanr(a[use], b[use]).statistic
    return float(value) if np.isfinite(value) else float("nan")


def instantiate(out_root: Path, metrics_path: Path, device: torch.device):
    metrics = json.loads(metrics_path.read_text())
    cache = out_root / "cache" / metrics["fit_id"]
    plane = dict(np.load(cache / "plane.npz"))
    events = dict(np.load(cache / "events.npz"))
    provenance = json.loads((cache / "provenance.json").read_text())
    cfg = metrics["config"]
    config = WEConfig(
        arm=metrics["arm"], cell=metrics["cell"], n_contacts=int(provenance["n_contacts"]),
        n_nodes=int(provenance["n_nodes"]), state_dim=int(cfg["state_dim"]),
        density=float(cfg["density"]), eta=float(cfg["eta"]), d0_mm=float(cfg["d0_mm"]),
        seed=int(metrics["seed"]), observation_operator=plane["H"],
        node_distance_mm=plane["D_mm"],
    )
    model = WEModel(config).to(device)
    model.load_state_dict(torch.load(metrics_path.parent / "weights.pt", map_location=device, weights_only=True))
    model.eval()
    decoder = RolloutSizeHead(model.n_contacts).to(device)
    decoder.load_state_dict(torch.load(metrics_path.parent / "rollout_size_head.pt",
                                       map_location=device, weights_only=True))
    decoder.eval()
    return model, decoder, metrics, plane, events, provenance


def evenly(indices: np.ndarray, maximum: int) -> np.ndarray:
    if len(indices) <= maximum:
        return indices
    return indices[np.unique(np.linspace(0, len(indices) - 1, maximum).round().astype(int))]


def evaluate_lesion(model: WEModel, decoder: RolloutSizeHead, tensors: dict[str, torch.Tensor],
                    selected: np.ndarray, ranks: np.ndarray, mode: np.ndarray,
                    provenance: dict[str, Any], empirical: dict[str, Any],
                    device: torch.device) -> dict[str, float]:
    mask = np.zeros(len(ranks), bool); mask[selected] = True
    prediction = evaluate(model, tensors, device, event_mask=mask)
    starts = [np.flatnonzero(ranks[index] == 0) for index in selected]
    generated = rollout_with_size_head(model, decoder, starts, device)
    agreement = [seed_removed_sequence_agreement(ranks[index], sequence)
                 for index, sequence in zip(selected, generated)]
    length = [sum(len(rank_set) for rank_set in sequence[1:]) / max(1, int((ranks[index] > 0).sum()))
              for index, sequence in zip(selected, generated)]
    grouped: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for index, sequence in zip(selected, generated):
        template = provenance["mode_to_template"].get(str(int(mode[index])))
        if template in ("a", "b"):
            grouped[template.upper()].append({"generated_rank_sets": sequence})
    order = [str(value) for value in empirical["contact_order"]]
    contacts = [str(value) for value in provenance["contacts"]]
    take = np.asarray([order.index(value) for value in contacts], int)
    fidelity = []
    for template, rank_key in (("A", "rank_a"), ("B", "rank_b")):
        if not grouped[template]:
            continue
        generated_field = aggregate_records(grouped[template], len(contacts))["canonical_full"]
        reference = empirical_score(np.asarray(empirical[rank_key], float)[take])
        fidelity.append(safe_corr(generated_field, reference))
    return {
        "contact_nll": float(prediction["contact_nll"]),
        "stop_bce": float(prediction["stop_bce"]),
        "rollout_spearman": float(np.nanmedian(agreement)),
        "postseed_length_ratio": float(np.nanmedian(length)),
        "interictal_field_fidelity": float(np.nanmean(fidelity)) if fidelity else float("nan"),
    }


def generate_template_fields(
    model: WEModel, decoder: RolloutSizeHead, ranks: np.ndarray, mode: np.ndarray,
    selected: np.ndarray, provenance: dict[str, Any], device: torch.device,
) -> dict[str, list[float]]:
    """Generate target-free A/B fields from every held-out event for lesion readout."""
    starts = [np.flatnonzero(ranks[index] == 0) for index in selected]
    generated = rollout_with_size_head(model, decoder, starts, device)
    grouped: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
    for index, sequence in zip(selected, generated):
        template = provenance["mode_to_template"].get(str(int(mode[index])))
        if template in ("a", "b"):
            grouped[template.upper()].append({"generated_rank_sets": sequence})
    output: dict[str, list[float]] = {}
    for template in ("A", "B"):
        if grouped[template]:
            output[template] = aggregate_records(
                grouped[template], len(provenance["contacts"])
            )["canonical_full"].tolist()
    return output


def edge_descriptor(edges: np.ndarray, strength: np.ndarray, distance: np.ndarray,
                    mask: np.ndarray, nodes_xy: np.ndarray) -> dict[str, float]:
    source, target = edges[:, 0], edges[:, 1]
    endpoint = np.unique(edges)
    # Rows are presynaptic/source nodes and columns are postsynaptic/target
    # nodes.  Keep the two degree directions separate: matching only their sum
    # can silently pair an outgoing hub lesion with an incoming hub control.
    # The frozen v0.4 contract explicitly requires endpoint in/out degree.
    in_degree = mask.sum(0)
    out_degree = mask.sum(1)
    points = nodes_xy[endpoint]
    extent = float(np.linalg.norm(points[:, None] - points[None, :], axis=-1).max()) if len(points) else 0.0
    return {
        "total_weight": float(strength[source, target].sum()),
        "mean_length": float(distance[source, target].mean()),
        "mean_in_degree": float(np.mean(in_degree[endpoint])),
        "mean_out_degree": float(np.mean(out_degree[endpoint])),
        "extent": extent,
    }


def within(value: float, target: float, fraction: float) -> bool:
    return abs(value - target) <= fraction * max(abs(target), 1e-12)


def edge_descriptor_matches(current: dict[str, float], target: dict[str, float]) -> bool:
    """Return whether an edge draw satisfies the frozen directed calipers."""
    return bool(
        within(current["total_weight"], target["total_weight"], 0.10)
        and within(current["mean_length"], target["mean_length"], 0.10)
        and abs(current["mean_in_degree"] - target["mean_in_degree"]) <= 1.0
        and abs(current["mean_out_degree"] - target["mean_out_degree"]) <= 1.0
        and within(current["extent"], target["extent"], 0.10)
    )


def matched_edge_draws(target_mask: np.ndarray, mask: np.ndarray, strength: np.ndarray,
                       distance: np.ndarray, nodes_xy: np.ndarray, draws: int,
                       seed: int) -> list[np.ndarray]:
    target_edges = np.argwhere(target_mask)
    active = np.argwhere(mask & ~target_mask)
    if len(target_edges) < 1 or len(active) < len(target_edges):
        return []
    descriptor = edge_descriptor(target_edges, strength, distance, mask, nodes_xy)
    length_edges = distance[active[:, 0], active[:, 1]]
    bins = np.quantile(distance[mask], np.linspace(0, 1, 6))
    target_bin = np.clip(np.digitize(distance[target_edges[:, 0], target_edges[:, 1]], bins[1:-1]), 0, 4)
    active_bin = np.clip(np.digitize(length_edges, bins[1:-1]), 0, 4)
    counts = np.bincount(target_bin, minlength=5)
    rng = np.random.default_rng(seed)
    valid = []
    attempts = 0
    while len(valid) < draws and attempts < 200000:
        attempts += 1
        pieces = []
        feasible = True
        for group, count in enumerate(counts):
            if count == 0:
                continue
            candidates = np.flatnonzero(active_bin == group)
            if len(candidates) < count:
                feasible = False; break
            pieces.append(active[rng.choice(candidates, int(count), replace=False)])
        if not feasible:
            break
        chosen = np.concatenate(pieces)
        if len(np.unique(chosen[:, 0] * mask.shape[0] + chosen[:, 1])) != len(chosen):
            continue
        current = edge_descriptor(chosen, strength, distance, mask, nodes_xy)
        if edge_descriptor_matches(current, descriptor):
            lesion = np.zeros_like(mask, bool); lesion[chosen[:, 0], chosen[:, 1]] = True
            valid.append(lesion)
    return valid


def node_descriptor(nodes: np.ndarray, mask: np.ndarray, strength: np.ndarray,
                    nodes_xy: np.ndarray) -> dict[str, float]:
    incident = np.zeros_like(mask, bool); incident[nodes, :] = True; incident[:, nodes] = True
    degree = mask.sum(0) + mask.sum(1)
    points = nodes_xy[nodes]
    radius = float(np.linalg.norm(points - points.mean(0), axis=1).max()) if len(points) else 0.0
    return {"incident_weight": float(strength[incident & mask].sum()),
            "mean_degree": float(degree[nodes].mean()), "radius": radius}


def matched_node_draws(target_nodes: np.ndarray, mask: np.ndarray, strength: np.ndarray,
                       nodes_xy: np.ndarray, draws: int, seed: int) -> list[np.ndarray]:
    nodes = np.flatnonzero(target_nodes)
    pool = np.flatnonzero(~target_nodes)
    if len(nodes) < 1 or len(pool) < len(nodes):
        return []
    descriptor = node_descriptor(nodes, mask, strength, nodes_xy)
    rng = np.random.default_rng(seed)
    valid = []
    for _ in range(200000):
        chosen = np.sort(rng.choice(pool, len(nodes), replace=False))
        current = node_descriptor(chosen, mask, strength, nodes_xy)
        if (within(current["incident_weight"], descriptor["incident_weight"], 0.10)
                and abs(current["mean_degree"] - descriptor["mean_degree"]) <= 1.0
                and within(current["radius"], descriptor["radius"], 0.10)):
            valid.append(chosen)
            if len(valid) >= draws:
                break
    return valid


def apply_edge_lesion(model: WEModel, lesion: np.ndarray) -> None:
    model.node_mask[torch.as_tensor(lesion, device=model.node_mask.device)] = 0.0


def apply_node_lesion(model: WEModel, nodes: np.ndarray) -> None:
    model.node_mask[nodes, :] = 0.0; model.node_mask[:, nodes] = 0.0


def choose_units(out_root: Path) -> list[tuple[Path, Path]]:
    candidates: dict[tuple[str, str, str], list[tuple[Path, float]]] = {}
    for metrics_path in sorted((out_root / "per_subject").glob("*/*__rnn/seed*/metrics.json")):
        model = metrics_path.parents[1].name.rsplit("__", 1)[0]
        if model not in THEORY_MODELS:
            continue
        metrics = json.loads(metrics_path.read_text())
        influence = (out_root / "effective_influence" / metrics["fit_id"]
                     / metrics_path.parents[1].name / metrics_path.parent.name / "influence.npz")
        if not influence.exists():
            continue
        key = (metrics["fit_id"], model, "rnn")
        candidates.setdefault(key, []).append((metrics_path, float(metrics["validation"]["contact_nll"])))
    selected = []
    for values in candidates.values():
        median = np.median([value for _, value in values])
        metrics_path, _ = min(values, key=lambda item: (abs(item[1] - median), str(item[0])))
        metrics = json.loads(metrics_path.read_text())
        influence = (out_root / "effective_influence" / metrics["fit_id"]
                     / metrics_path.parents[1].name / metrics_path.parent.name / "influence.npz")
        selected.append((metrics_path, influence))
    return sorted(selected)


def run_unit(out_root: Path, metrics_path: Path, influence_path: Path, device: torch.device,
             target_draws: int, max_events: int) -> dict[str, Any]:
    model, decoder, metrics, plane, events, provenance = instantiate(out_root, metrics_path, device)
    keep = events["split"] >= 0
    ranks = np.asarray(events["ranks"])[keep]
    split = np.asarray(events["split"])[keep]
    mode = np.asarray(events["mode"])[keep]
    tensors = build_event_tensors(ranks)
    selected = evenly(np.flatnonzero(split == 2), max_events)
    all_heldout = np.flatnonzero(split == 2)
    manifest = json.loads((out_root / "INPUT_MANIFEST.json").read_text())
    empirical = json.loads((Path(manifest["input_roots"]["field"]) / f"{metrics['subject']}.json").read_text())["interictal_field"]
    with np.load(influence_path, allow_pickle=False) as influence:
        mask = np.asarray(influence["edge_mask"], bool)
        strength = np.asarray(influence["edge_strength"], float)
        effective = np.asarray(influence["edge_effective_influence"], float)
        local = np.asarray(influence["local_backbone_mask"], bool)
        long_high = np.asarray(influence["long_high_mask"], bool)
        connector_nodes = np.asarray(influence["connector_nodes"], bool)
    highest = np.zeros_like(mask, bool)
    active_effect = effective[mask]
    threshold = np.quantile(active_effect, 0.90)
    highest[mask & (effective >= threshold)] = True
    lesions: dict[str, tuple[str, np.ndarray]] = {
        "highest_effective_edges": ("edge", highest),
        "local_backbone_edges": ("edge", local),
        "long_range_high_influence_edges": ("edge", long_high),
        "connector_nodes": ("node", connector_nodes),
    }
    original_mask = model.node_mask.detach().clone()
    baseline = evaluate_lesion(model, decoder, tensors, selected, ranks, mode, provenance, empirical, device)
    baseline_fields = generate_template_fields(
        model, decoder, ranks, mode, all_heldout, provenance, device
    )
    outputs = {}
    for lesion_name, (kind, target) in lesions.items():
        n_target = int(target.sum())
        minimum = 3 if kind == "edge" else 2
        if n_target < minimum:
            outputs[lesion_name] = {"status": "motif_not_estimable", "n_target": n_target}
            continue
        if kind == "edge":
            matched = matched_edge_draws(target, mask, strength, np.asarray(plane["D_mm"], float),
                                         np.asarray(plane["nodes_xy_mm"], float), target_draws,
                                         int(metrics["seed"]) + 9001)
        else:
            matched = matched_node_draws(target, mask, strength, np.asarray(plane["nodes_xy_mm"], float),
                                         target_draws, int(metrics["seed"]) + 19001)
        model.node_mask.copy_(original_mask)
        if kind == "edge": apply_edge_lesion(model, target)
        else: apply_node_lesion(model, np.flatnonzero(target))
        targeted = evaluate_lesion(model, decoder, tensors, selected, ranks, mode, provenance, empirical, device)
        targeted_fields = generate_template_fields(
            model, decoder, ranks, mode, all_heldout, provenance, device
        )
        random_metrics = []
        if len(matched) >= MINIMUM_VALID:
            for control in matched:
                model.node_mask.copy_(original_mask)
                if kind == "edge": apply_edge_lesion(model, control)
                else: apply_node_lesion(model, control)
                random_metrics.append(evaluate_lesion(
                    model, decoder, tensors, selected, ranks, mode, provenance, empirical, device
                ))
        model.node_mask.copy_(original_mask)
        outputs[lesion_name] = {
            "status": "inference_available" if len(random_metrics) >= MINIMUM_VALID else "matched_inference_unavailable",
            "n_target": n_target, "n_valid_matched_draws": len(matched),
            "baseline": baseline, "targeted": targeted,
            "field_contacts": [str(value) for value in provenance["contacts"]],
            "baseline_fields": baseline_fields,
            "targeted_fields": targeted_fields,
            "matched": {metric: [row[metric] for row in random_metrics] for metric in baseline},
        }
    return {
        "contract": "topic5_rnn_motif_matched_lesion_unit_v0_4", "target_values_read": False,
        "subject": metrics["subject"], "fit_id": metrics["fit_id"], "scope": metrics["fit_scope"],
        "model": metrics["model_id"].rsplit("__", 1)[0], "cell": metrics["cell"],
        "seed": int(metrics["seed"]), "seed_selection": "closest_to_three_seed_validation_median",
        "n_heldout_events_for_matched_metrics": int(len(selected)),
        "n_heldout_events_for_targeted_fields": int(len(all_heldout)),
        "target_draws": target_draws,
        "minimum_valid_matched_draws": MINIMUM_VALID, "lesions": outputs,
        "matching_contract": {
            "draws": "without replacement within each draw; repeated draws allowed",
            "edge_total_weight_caliper": 0.10,
            "edge_mean_length_caliper": 0.10,
            "edge_mean_in_degree_absolute_caliper": 1.0,
            "edge_mean_out_degree_absolute_caliper": 1.0,
            "edge_spatial_extent_caliper": 0.10,
            "node_incident_weight_caliper": 0.10,
            "node_mean_degree_absolute_caliper": 1.0,
            "node_radius_caliper": 0.10,
            "connector_node_operation": (
                "remove all incoming and outgoing recurrent edges incident to the selected "
                "tissue nodes; retain direct input injection and observation-operator readout"
            ),
        },
    }


def aggregate(out_root: Path) -> None:
    records = [json.loads(path.read_text()) for path in sorted((out_root / "matched_lesions").glob("**/LESION_DONE.json"))]
    fit_rows = []
    for record in records:
        for lesion, values in record["lesions"].items():
            if values["status"] == "motif_not_estimable":
                continue
            row = {key: record[key] for key in ("subject", "fit_id", "scope", "model", "cell", "seed")}
            row.update({"lesion": lesion, "status": values["status"],
                        "n_target": values["n_target"], "n_valid_matched_draws": values["n_valid_matched_draws"]})
            for metric, direction in (("contact_nll", 1), ("stop_bce", 1),
                                      ("rollout_spearman", -1), ("postseed_length_ratio", -1),
                                      ("interictal_field_fidelity", -1)):
                target_damage = direction * (number(values["targeted"][metric]) - number(values["baseline"][metric]))
                matched_damage = [direction * (number(value) - number(values["baseline"][metric]))
                                  for value in values["matched"].get(metric, [])]
                row[f"target_damage_{metric}"] = target_damage
                row[f"matched_median_damage_{metric}"] = float(np.nanmedian(matched_damage)) if matched_damage else np.nan
                row[f"specificity_{metric}"] = target_damage - row[f"matched_median_damage_{metric}"]
            fit_rows.append(row)
    write_csv(out_root / "matched_lesion_fit_metrics.csv", fit_rows)
    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for row in fit_rows:
        grouped.setdefault((row["subject"], row["model"], row["cell"], row["lesion"]), []).append(row)
    patient_rows = []
    numeric = [key for key in fit_rows[0] if key.startswith(("target_damage_", "matched_median_", "specificity_"))]
    for key, rows in grouped.items():
        subject, model, cell, lesion = key
        item = {"subject": subject, "model": model, "cell": cell, "lesion": lesion,
                "n_fits": len(rows), "all_inference_available": all(row["status"] == "inference_available" for row in rows)}
        item.update({name: float(np.nanmean([row[name] for row in rows])) for name in numeric})
        patient_rows.append(item)
    write_csv(out_root / "matched_lesion_patient_metrics.csv", patient_rows)
    statistics = {}
    for model, lesion in sorted({(row["model"], row["lesion"]) for row in patient_rows}):
        values = np.asarray([row["specificity_contact_nll"] for row in patient_rows
                             if row["model"] == model and row["lesion"] == lesion
                             and row["all_inference_available"]], float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            continue
        nonzero = values[np.abs(values) > 1e-9]
        statistics[f"{model}|{lesion}"] = {
            "n": len(values), "median_specificity_contact_nll": float(np.median(values)),
            "positive": int((values > 1e-9).sum()),
            "wilcoxon_p": float(wilcoxon(nonzero, method="auto").pvalue) if len(nonzero) else 1.0,
        }
    (out_root / "MATCHED_LESION_SUMMARY.json").write_text(json.dumps({
        "contract": "topic5_rnn_motif_matched_lesion_summary_v0_4", "target_values_read": False,
        "n_selected_fit_model_units": len(records), "statistics": statistics,
        "cell_scope": "leaky_rnn_primary_only; GRU is limited to effective-reach architecture replication",
        "target_draws": TARGET_DRAWS, "minimum_valid_matched_draws": MINIMUM_VALID,
        "connector_node_operation": (
            "incident recurrent-edge removal; direct input and observation readout retained"
        ),
        "interpretation_rule": "motif wording requires enrichment, task relation and positive matched-lesion specificity",
    }, indent=2))
    (out_root / "stage_g_scientific_drift_audit.json").write_text(json.dumps({
        "status": "ALIGNED",
        "target_values_read": False,
        "scientific_question": (
            "which input-output effective organization supports heldout interictal propagation "
            "under matched target-free perturbations"
        ),
        "primary_motif": "local high-influence backbone plus sparse long-range connector organization",
        "connector_perturbation_scope": (
            "incoming/outgoing recurrent edges at connector nodes; not complete node ablation"
        ),
        "primary_evidence_required": ["enrichment", "task association", "matched-lesion specificity"],
        "cell_scope": "leaky RNN primary; GRU effective-reach replication only",
        "not_claimed": ["edge-level connectome recovery", "hidden-unit neuron identity",
                        "causal perturbation of the human brain"],
    }, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target-draws", type=int, default=TARGET_DRAWS)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    if not args.aggregate_only:
        units = choose_units(out_root)[args.shard_index::args.n_shards]
        if args.limit is not None:
            units = units[:args.limit]
        for index, (metrics_path, influence_path) in enumerate(units, 1):
            metrics = json.loads(metrics_path.read_text())
            output = (out_root / "matched_lesions" / metrics["fit_id"]
                      / metrics_path.parents[1].name / "LESION_DONE.json")
            if output.exists():
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            result = run_unit(out_root, metrics_path, influence_path, torch.device(args.device),
                              args.target_draws, args.max_events)
            temporary = output.with_suffix(".tmp")
            temporary.write_text(json.dumps(clean_json(result), indent=2, allow_nan=False)); temporary.replace(output)
            print(json.dumps({"complete": index, "total": len(units), "fit_id": metrics["fit_id"]}), flush=True)
    if args.n_shards == 1 or args.aggregate_only:
        aggregate(out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
