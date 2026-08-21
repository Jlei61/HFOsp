#!/usr/bin/env python3
"""Compare hand dual cores and the final continuous Node field."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_topic4_rev10_sa_shaft_aware_target import (  # noqa: E402
    _calibrated_mode_score,
    _distance_row,
    _flatten_distances,
    _load_old_reference,
    _summarize_floor_rows,
)
from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    _classifier_from_manifest,
)
from scripts.run_topic4_core_field_stage3_profile_round1 import axial_map  # noqa: E402
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic4_core_field_profile import (  # noqa: E402
    fit_profile_modes,
    fixed_count_indices,
    normalized_rank_curve,
    profile_grid,
    split_by_block,
)
from src.topic4_d6_natural_kmeans import natural_kmeans  # noqa: E402
from src.topic4_shaft_aware import (  # noqa: E402
    build_event_features,
    centered_smooth_max,
    contract_groups,
    contract_pairs,
    describe_events,
    transform_patient_embedding,
)
from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402


DEFAULT_CONFIG = ROOT / "config/topic4_dual_core_vs_free_field.json"
ARM_ORDER = ("hand_dual_core", "continuous_free_field")


def _sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            _jsonable(payload), indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _patient_table(sa_config, contract, partition):
    """Rebuild one side of the frozen recording-block split."""
    data = load_subject_propagation_events(sa_config["inputs"]["patient_root"])
    names = np.asarray([str(value) for value in data["channel_names"]])
    canonical = np.asarray([row["contact_name"] for row in contract["contacts"]])
    if set(names) != set(canonical):
        raise RuntimeError("patient and comparison contact sets differ")
    lookup = {name: index for index, name in enumerate(names)}
    reorder = np.asarray([lookup[name] for name in canonical], int)
    blocks = np.asarray(data["block_ids"])
    train, heldout = split_by_block(
        blocks, float(sa_config["patient_split"]["heldout_fraction"]),
        int(sa_config["patient_split"]["seed"]),
    )
    selected = train if partition == "train" else heldout
    opposite = heldout if partition == "train" else train
    if set(blocks[selected]).intersection(set(blocks[opposite])):
        raise RuntimeError("recording-block split leaked")
    axial = {
        row["contact_name"]: row["shared_axis_coordinate_mm"]
        for row in contract["contacts"]
    }
    grid = profile_grid(axial)
    reference = _load_old_reference(
        ROOT / sa_config["inputs"]["old_rank_curve_reference"]
    )
    if not np.allclose(grid, reference["grid"], rtol=0.0, atol=1e-12):
        raise RuntimeError("patient direction grid changed")
    raw_ranks = np.asarray(data["ranks"], float)
    raw_onsets = np.asarray(data["lag_raw"], float)
    raw_mask = np.asarray(data["bools"], bool)
    curves, onsets, ranks, event_blocks, event_indices = [], [], [], [], []
    for event_index in selected:
        participating = np.flatnonzero(raw_mask[:, event_index])
        curve = normalized_rank_curve({
            names[index]: float(raw_ranks[index, event_index])
            for index in participating
        }, axial, grid=grid)
        if curve is None:
            continue
        onset = raw_onsets[:, event_index].copy()
        rank = raw_ranks[:, event_index].copy()
        onset[~raw_mask[:, event_index]] = np.nan
        rank[~raw_mask[:, event_index]] = np.nan
        curves.append(curve)
        onsets.append(onset[reorder])
        ranks.append(rank[reorder])
        event_blocks.append(blocks[event_index])
        event_indices.append(event_index)
    modes = fit_profile_modes(np.asarray(curves, float), reference)
    if modes.get("status") != "ok":
        raise RuntimeError(f"could not assign frozen patient modes for {partition}")
    return {
        "partition": partition,
        "onsets": np.asarray(onsets, float),
        "ranks": np.asarray(ranks, float),
        "labels": np.asarray(modes["labels"], int),
        "blocks": np.asarray(event_blocks),
        "event_indices": np.asarray(event_indices, int),
        "mode_counts": np.asarray(modes["cluster_counts"], int),
        "n_blocks": int(len(np.unique(event_blocks))),
    }


def _embedding(target_path):
    with np.load(target_path, allow_pickle=False) as loaded:
        return {
            "center": np.asarray(loaded["feature_center"], float),
            "scale": np.asarray(loaded["feature_scale"], float),
            "components": np.asarray(loaded["pca_components"], float),
            "directions": np.asarray(loaded["sw_directions"], float),
        }


def _patient_targets(patient, groups, pairs, embedding, seed):
    z = transform_patient_embedding(
        build_event_features(patient["onsets"], groups)["features"], embedding,
    )
    rng = np.random.default_rng(int(seed))
    output = {}
    for mode in (0, 1):
        index = np.flatnonzero(patient["labels"] == mode)
        take = min(4096, len(index))
        selected = np.sort(rng.choice(index, size=take, replace=False))
        output[mode] = {
            "descriptor": describe_events(patient["onsets"][index], groups, pairs),
            "reference_z": z[selected],
            "n_events": int(len(index)),
        }
    return output


def _patient_floors(patient, targets, groups, pairs, embedding, count, seed):
    rng = np.random.default_rng(int(seed))
    output = {}
    for mode in (0, 1):
        index = np.flatnonzero(patient["labels"] == mode)
        by_block = {}
        for event_index in index:
            by_block.setdefault(int(patient["blocks"][event_index]), []).append(
                int(event_index)
            )
        if len(by_block) < int(count):
            raise RuntimeError("held-out patient has too few blocks for floor")
        rows = []
        block_ids = np.asarray(sorted(by_block), int)
        for _ in range(256):
            chosen_blocks = rng.choice(block_ids, size=int(count), replace=False)
            chosen = np.asarray([
                rng.choice(by_block[int(block)]) for block in chosen_blocks
            ], int)
            rows.append(_flatten_distances(_distance_row(
                patient["onsets"][chosen], targets[mode], groups, pairs, embedding,
            )))
        output[str(mode)] = _summarize_floor_rows(rows)
    return output


def _model_arm(config, manifest, contract, embedding, arm):
    groups = contract_groups(contract)
    classifier = _classifier_from_manifest(manifest)
    if arm == "continuous_free_field":
        root = ROOT / "results/topic4_sef_hfo/data_driven_local_connectivity_rev11_nlc/frozen_substrate_confirmation"
        candidate = "node_baseline"
    else:
        root = ROOT / config["output_root"]
        candidate = config["manual_dual_core"]["candidate_id"]
    expected_names = np.asarray([row["contact_name"] for row in contract["contacts"]])
    blocks, inputs, static = {}, [], None
    for seed in config["search"]["confirmation_network_seeds"]:
        stem = root / "workers" / f"{candidate}_seed_{seed}"
        json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        payload = json.loads(json_path.read_text())
        if payload["arrays"]["sha256"] != _sha256(npz_path):
            raise RuntimeError(f"worker array changed: {npz_path}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            names = np.asarray(loaded["contact_names"]).astype(str)
            if not np.array_equal(names, expected_names):
                raise RuntimeError(f"contact order changed: {npz_path}")
            onsets = np.asarray(loaded["onsets"], float)
            ranks = np.asarray(loaded["ranks"], float)
            returned = np.asarray(loaded["event_returned"], bool)
            positions_e = np.asarray(loaded["positions_E"], float)
            h = np.asarray(loaded["h"], float)
            if static is None:
                static = {
                    "positions_E": positions_e,
                    "h": h,
                    "delta_vtheta": np.asarray(loaded["delta_vtheta"], float),
                    "contact_xy_mm": np.asarray(loaded["contact_xy_mm"], float),
                    "contact_names": names,
                    "shaft_ids": np.asarray(loaded["shaft_ids"]).astype(str),
                }
        assigned = assign_direction_modes(
            onsets, groups=groups, embedding=embedding, classifier=classifier,
        )
        labels = np.asarray(assigned["labels"], int)
        ood = np.asarray(assigned["ood"], bool)
        masks = _event_masks(onsets, returned, ood, groups)
        readable = masks["readable"]
        distribution = masks["distribution"]
        formal_kmeans = masks["formal_kmeans"]
        blocks[int(seed)] = {
            "onsets": onsets, "ranks": ranks, "returned": returned,
            "labels": labels, "ood": ood, "readable": readable,
            "distribution": distribution, "formal_kmeans": formal_kmeans,
            "positions_E": positions_e, "h": h,
        }
        inputs.append({
            "seed": int(seed), "json": str(json_path),
            "json_sha256": _sha256(json_path), "npz": str(npz_path),
            "npz_sha256": _sha256(npz_path),
        })
    return {"arm": arm, "candidate": candidate, "blocks": blocks,
            "inputs": inputs, "static": static}


def _event_masks(onsets, returned, ood, groups):
    """Keep single-shaft absence in the distribution, not the KMeans view."""
    onsets = np.asarray(onsets, float)
    returned = np.asarray(returned, bool)
    ood = np.asarray(ood, bool)
    readable = np.sum(np.isfinite(onsets), axis=1) >= 3
    distribution = returned & readable & ~ood
    icl = np.isfinite(onsets[:, np.asarray(groups["ICL"], int)]).any(axis=1)
    scl = np.isfinite(onsets[:, np.asarray(groups["SCL"], int)]).any(axis=1)
    return {
        "readable": readable,
        "distribution": distribution,
        "formal_kmeans": distribution & icl & scl,
    }


def _bernoulli_js(left, right):
    left = np.asarray([left, 1.0 - left], float)
    right = np.asarray([right, 1.0 - right], float)
    middle = 0.5 * (left + right)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = [
            np.where(value > 0, value * np.log2(value / middle), 0.0)
            for value in (left, right)
        ]
    return float(0.5 * (np.sum(terms[0]) + np.sum(terms[1])))


def _score_network(block, targets, floors, groups, pairs, embedding, sa_config,
                   count, patient_b_fraction, seed):
    mask = block["distribution"]
    labels = block["labels"]
    modes, objectives = {}, []
    for mode in (0, 1):
        available = np.flatnonzero(mask & (labels == mode))
        local = fixed_count_indices(len(available), int(count))
        if local is None:
            modes[str(mode)] = {
                "status": "INSUFFICIENT_MODE_SUPPORT",
                "n_available": int(len(available)), "n_required": int(count),
            }
            continue
        selected = available[local]
        raw = _flatten_distances(_distance_row(
            block["onsets"][selected], targets[mode], groups, pairs, embedding,
        ))
        objective = _calibrated_mode_score(raw, floors[str(mode)], sa_config)
        modes[str(mode)] = {
            "status": "OK", "n_available": int(len(available)),
            "n_scored": int(len(selected)), "selected_indices": selected,
            "raw": raw, "objective": objective,
        }
        objectives.append(objective)
    selected_labels = labels[mask]
    mode_b_fraction = (
        float(np.mean(selected_labels == 1)) if len(selected_labels) else None
    )
    formal = block["formal_kmeans"]
    natural = natural_kmeans(
        block["ranks"][formal], labels[formal], random_state=int(seed),
    )
    returned_readable = block["returned"] & block["readable"]
    output = {
        "status": "OK" if len(objectives) == 2 else "INSUFFICIENT_MODE_SUPPORT",
        "mode_counts_distribution": np.bincount(
            labels[mask], minlength=2,
        ),
        "n_returned_readable": int(np.sum(returned_readable)),
        "n_distribution_events": int(np.sum(mask)),
        "ood_fraction_returned_readable": float(np.mean(
            block["ood"][returned_readable]
        )) if np.any(returned_readable) else None,
        "multishaft_fraction_distribution": float(np.mean(
            block["formal_kmeans"][mask]
        )) if np.any(mask) else None,
        "scl_recruitment_fraction_distribution": float(np.isfinite(
            block["onsets"][mask][:, groups["SCL"]]
        ).mean()) if np.any(mask) else None,
        "mode_b_fraction": mode_b_fraction,
        "mode_proportion_js": (
            _bernoulli_js(mode_b_fraction, patient_b_fraction)
            if mode_b_fraction is not None else None
        ),
        "natural_kmeans": {
            key: value for key, value in natural.items()
            if key not in {"valid_event_mask", "cluster_labels"}
        },
        "modes": modes,
    }
    if len(objectives) == 2:
        mode_scores = [row["mode_score"] for row in objectives]
        output.update({
            "mean_mode_score": float(np.mean(mode_scores)),
            "weak_mode_score": centered_smooth_max(mode_scores, 0.25),
            "recruitment": float(np.mean([row["recruitment"] for row in objectives])),
            "precedence": float(np.mean([row["precedence"] for row in objectives])),
            "profile": float(np.mean([row["profile"] for row in objectives])),
            "event_cloud": float(np.mean([row["event_cloud"] for row in objectives])),
        })
    return output


def _paired_bootstrap(rows, endpoint, *, lower_is_better, draws, seed):
    pairs = []
    for network_seed in sorted(rows["hand_dual_core"]):
        left = rows["continuous_free_field"][network_seed].get(endpoint)
        right = rows["hand_dual_core"][network_seed].get(endpoint)
        if left is not None and right is not None:
            pairs.append((network_seed, float(left), float(right)))
    if not pairs:
        return None
    delta = np.asarray([left - right for _, left, right in pairs], float)
    rng = np.random.default_rng(int(seed))
    sample = rng.choice(delta, size=(int(draws), len(delta)), replace=True)
    means = sample.mean(axis=1)
    nonzero = delta[np.abs(delta) > 1e-12]
    p_value = (
        float(wilcoxon(nonzero).pvalue) if len(nonzero) >= 2 else None
    )
    return {
        "endpoint": endpoint,
        "lower_is_better": bool(lower_is_better),
        "n_paired_networks": int(len(delta)),
        "continuous_minus_hand_by_seed": {
            str(network_seed): left - right for network_seed, left, right in pairs
        },
        "mean_continuous_minus_hand": float(np.mean(delta)),
        "median_continuous_minus_hand": float(np.median(delta)),
        "network_bootstrap_q05": float(np.quantile(means, 0.05)),
        "network_bootstrap_q95": float(np.quantile(means, 0.95)),
        "probability_continuous_better": float(np.mean(
            means < 0.0 if lower_is_better else means > 0.0
        )),
        "paired_wilcoxon_p_two_sided": p_value,
    }


def _write_csv(path, scored):
    fields = [
        "count", "arm", "seed", "status", "n_returned_readable",
        "n_distribution_events", "mode_A_count", "mode_B_count",
        "ood_fraction_returned_readable", "multishaft_fraction_distribution",
        "scl_recruitment_fraction_distribution", "mode_b_fraction",
        "mode_proportion_js", "natural_alignment", "weak_mode_score",
        "mean_mode_score", "recruitment", "precedence", "profile", "event_cloud",
    ]
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for count, arms in scored.items():
            for arm, seeds in arms.items():
                for seed, row in seeds.items():
                    counts = row["mode_counts_distribution"]
                    writer.writerow({
                        "count": count, "arm": arm, "seed": seed,
                        "status": row["status"],
                        "n_returned_readable": row["n_returned_readable"],
                        "n_distribution_events": row["n_distribution_events"],
                        "mode_A_count": counts[0], "mode_B_count": counts[1],
                        "ood_fraction_returned_readable": row["ood_fraction_returned_readable"],
                        "multishaft_fraction_distribution": row["multishaft_fraction_distribution"],
                        "scl_recruitment_fraction_distribution": row["scl_recruitment_fraction_distribution"],
                        "mode_b_fraction": row["mode_b_fraction"],
                        "mode_proportion_js": row["mode_proportion_js"],
                        "natural_alignment": row["natural_kmeans"].get(
                            "direction_balanced_alignment"
                        ),
                        **{key: row.get(key) for key in (
                            "weak_mode_score", "mean_mode_score", "recruitment",
                            "precedence", "profile", "event_cloud",
                        )},
                    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    root = ROOT / config["output_root"]
    manifest_path = root / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["config"]["sha256"] != _sha256(config_path):
        raise RuntimeError("comparison manifest/config mismatch")
    contract = json.loads(
        (ROOT / config["inputs"]["contact_contract"]["path"]).read_text()
    )
    groups, pairs = contract_groups(contract), contract_pairs(contract)
    sa_config = json.loads(
        (ROOT / config["inputs"]["shaft_aware_scoring_config"]["path"]).read_text()
    )
    target_path = ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]
    embedding = _embedding(target_path)
    patient = {
        partition: _patient_table(sa_config, contract, partition)
        for partition in ("train", "heldout")
    }
    targets = {
        partition: _patient_targets(
            table, groups, pairs, embedding, 20260821 + index,
        ) for index, (partition, table) in enumerate(patient.items())
    }
    arms = {
        arm: _model_arm(config, manifest, contract, embedding, arm)
        for arm in ARM_ORDER
    }

    counts = [
        int(config["search"]["comparison"][key])
        for key in (
            "primary_events_per_mode_per_network",
            "sensitivity_events_per_mode_per_network",
        )
    ]
    scored, floor_outputs, comparisons = {}, {}, {}
    primary_patient = patient["heldout"]
    patient_b = float(np.mean(primary_patient["labels"] == 1))
    endpoints = {
        "weak_mode_score": True, "mean_mode_score": True,
        "recruitment": True, "precedence": True, "profile": True,
        "event_cloud": True, "mode_proportion_js": True,
        "ood_fraction_returned_readable": True,
        "multishaft_fraction_distribution": False,
        "scl_recruitment_fraction_distribution": False,
    }
    for count in counts:
        key = str(count)
        floors = _patient_floors(
            primary_patient, targets["heldout"], groups, pairs, embedding,
            count, seed=20260831 + count,
        )
        floor_outputs[key] = floors
        scored[key] = {}
        for arm, bundle in arms.items():
            scored[key][arm] = {
                str(seed): _score_network(
                    block, targets["heldout"], floors, groups, pairs, embedding,
                    sa_config, count, patient_b, seed,
                ) for seed, block in bundle["blocks"].items()
            }
        comparisons[key] = {
            endpoint: _paired_bootstrap(
                scored[key], endpoint, lower_is_better=lower,
                draws=int(config["search"]["comparison"][
                    "paired_network_bootstrap_draws"
                ]), seed=int(config["search"]["comparison"]["bootstrap_seed"]) + index,
            ) for index, (endpoint, lower) in enumerate(endpoints.items())
        }
        for arm in ARM_ORDER:
            values = {
                seed: row["natural_kmeans"].get("direction_balanced_alignment")
                for seed, row in scored[key][arm].items()
            }
            for seed, value in values.items():
                scored[key][arm][seed]["natural_alignment"] = value
        comparisons[key]["natural_alignment"] = _paired_bootstrap(
            scored[key], "natural_alignment", lower_is_better=False,
            draws=int(config["search"]["comparison"]["paired_network_bootstrap_draws"]),
            seed=int(config["search"]["comparison"]["bootstrap_seed"]) + 99,
        )

    geometry = {}
    centers = np.asarray(config["manual_dual_core"]["centers_mm"], float)
    historical_radius = float(config["manual_dual_core"]["historical_radius_mm"])
    for seed, block in arms["hand_dual_core"]["blocks"].items():
        positions = np.asarray(block["positions_E"])
        nearest = np.min(np.linalg.norm(
            positions[:, None, :] - centers[None, :, :], axis=2,
        ), axis=1)
        h = np.asarray(block["h"], float)
        geometry[str(seed)] = {
            "exact_budget_sum_h": float(np.sum(h)),
            "effective_cutoff_mm": float(np.max(nearest[h > 0.5])),
            "historical_radius_count": int(np.sum(nearest <= historical_radius)),
        }

    summary = {
        "status": "TOPIC4_DUAL_CORE_VS_FREE_FIELD_COMPLETE",
        "scientific_role": config["scientific_role"],
        "primary_target": "patient heldout recording blocks; development-only",
        "patient": {
            partition: {
                "n_events": int(len(table["onsets"])),
                "n_blocks": table["n_blocks"],
                "mode_counts": table["mode_counts"],
            } for partition, table in patient.items()
        },
        "arms": {
            arm: {"candidate": bundle["candidate"], "worker_inputs": bundle["inputs"]}
            for arm, bundle in arms.items()
        },
        "manual_geometry_by_seed": geometry,
        "patient_floors": floor_outputs,
        "per_network": scored,
        "paired_comparisons": comparisons,
        "claim_boundary": config["claim_boundary"],
        "provenance": {
            "git_commit": os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip(),
            "config": {"path": str(config_path), "sha256": _sha256(config_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256(manifest_path)},
        },
    }
    _atomic_json(root / "comparison_summary.json", summary)
    _write_csv(root / "per_network_metrics.csv", scored)

    arrays = {
        "patient_heldout_ranks": patient["heldout"]["ranks"],
        "patient_heldout_labels": patient["heldout"]["labels"],
    }
    for arm, bundle in arms.items():
        arrays[f"{arm}_positions_E"] = bundle["static"]["positions_E"]
        arrays[f"{arm}_h"] = bundle["static"]["h"]
        pooled_ranks, pooled_labels, pooled_seed, pooled_formal = [], [], [], []
        for seed, block in bundle["blocks"].items():
            pooled_ranks.append(block["ranks"])
            pooled_labels.append(block["labels"])
            pooled_seed.append(np.full(len(block["ranks"]), seed, int))
            pooled_formal.append(block["formal_kmeans"])
        arrays[f"{arm}_ranks"] = np.concatenate(pooled_ranks)
        arrays[f"{arm}_labels"] = np.concatenate(pooled_labels)
        arrays[f"{arm}_seed"] = np.concatenate(pooled_seed)
        arrays[f"{arm}_formal_kmeans"] = np.concatenate(pooled_formal)
    _atomic_npz(root / "comparison_plot_data.npz", **arrays)
    print(json.dumps({
        "status": summary["status"],
        "patient": summary["patient"],
        "primary_comparisons": comparisons[str(counts[0])],
    }, indent=2, default=_jsonable))


if __name__ == "__main__":
    main()
