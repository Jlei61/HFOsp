#!/usr/bin/env python3
"""Decompose the Node patient-fit/direction trade-off without new simulation."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, relative: str) -> Path:
    local = ROOT / relative
    return local if local.exists() else artifact_root / relative


def _rows(summary: dict) -> dict[str, dict]:
    return {row["candidate_id"]: row for row in summary["rows"]}


def _bootstrap(values: list[float], *, draws: int, confidence: float,
               seed: int) -> dict:
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, len(values), size=(draws, len(values)))]
    means = np.mean(sampled, axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_manual_minus_reference": float(np.mean(values)),
        "ci_low": float(np.quantile(means, alpha)),
        "ci_high": float(np.quantile(means, 1.0 - alpha)),
        "manual_lower_networks": int(np.sum(values < 0.0)),
        "manual_higher_networks": int(np.sum(values > 0.0)),
        "n_networks": int(len(values)),
        "values": values.tolist(),
    }


def _get(record: dict, path: tuple[str, ...]) -> float:
    value = record
    for key in path:
        value = value[key]
    return float(value)


LOSS_PATHS = {
    "recruitment": ("soft_objective", "modes", "{mode}", "recruitment"),
    "precedence": ("soft_objective", "modes", "{mode}", "precedence"),
    "profile": ("soft_objective", "modes", "{mode}", "profile"),
    "cloud": ("soft_objective", "modes", "{mode}", "cloud"),
}


def conflict_decomposition(capacity: dict, reference: dict, *, draws: int,
                           confidence: float, seed: int) -> dict:
    manual = {int(row["seed"]): row for row in capacity["per_network"]}
    baseline = {int(row["seed"]): row for row in reference["per_network"]}
    if manual.keys() != baseline.keys():
        raise RuntimeError("capacity and comparison networks do not align")
    output = {"modes": {}}
    for mode_index, mode in enumerate(("0", "1")):
        losses = {}
        for term_index, (term, path) in enumerate(LOSS_PATHS.items()):
            concrete = tuple(mode if key == "{mode}" else key for key in path)
            values = [
                _get(manual[network], concrete) - _get(baseline[network], concrete)
                for network in sorted(manual)
            ]
            losses[term] = _bootstrap(
                values, draws=draws, confidence=confidence,
                seed=seed + 1000 * mode_index + term_index,
            )
        precedence_classes = {}
        for class_index, pair_class in enumerate(("ICL-ICL", "SCL-SCL", "ICL-SCL")):
            path = (
                "soft_objective", "modes", mode, "raw",
                "precedence_classes", pair_class,
            )
            values = [
                _get(manual[network], path) - _get(baseline[network], path)
                for network in sorted(manual)
            ]
            precedence_classes[pair_class] = _bootstrap(
                values, draws=draws, confidence=confidence,
                seed=seed + 1000 * mode_index + 100 + class_index,
            )
        dynamics = {}
        for endpoint_index, (name, path) in enumerate({
            "causal_direction": (
                "soft_causal_direction", "modes", mode, "alignment_score",
            ),
            "causal_monotonicity": (
                "soft_causal_monotonicity", "modes", mode, "alignment_score",
            ),
            "soft_occupancy": (
                "soft_objective", "modes", mode, "soft_occupancy",
            ),
            "effective_events": (
                "soft_objective", "modes", mode, "effective_events",
            ),
        }.items()):
            values = [
                _get(manual[network], path) - _get(baseline[network], path)
                for network in sorted(manual)
            ]
            dynamics[name] = _bootstrap(
                values, draws=draws, confidence=confidence,
                seed=seed + 1000 * mode_index + 200 + endpoint_index,
            )
        output["modes"][mode] = {
            "loss_terms_positive_means_manual_is_worse": losses,
            "raw_precedence_positive_means_manual_is_worse": precedence_classes,
            "dynamics_positive_means_manual_is_higher": dynamics,
        }
    mode_0_losses = output["modes"]["0"][
        "loss_terms_positive_means_manual_is_worse"
    ]
    mode_1_direction = output["modes"]["1"][
        "dynamics_positive_means_manual_is_higher"
    ]["causal_direction"]
    output["observed_tradeoff"] = {
        "mode_0_all_four_losses_worse": bool(
            all(row["ci_low"] > 0.0 for row in mode_0_losses.values())
        ),
        "mode_1_direction_higher": bool(mode_1_direction["ci_low"] > 0.0),
        "interpretation": (
            "Observed candidate trade-off only; not proof that the patient target "
            "and opposite causal propagation are mathematically incompatible."
        ),
    }
    return output


GRADIENT_ENDPOINTS = {
    "soft_objective": (
        "soft_objective", "objective", "lower",
    ),
    "mode_0": ("soft_objective", "modes", "0", "mean", "lower"),
    "mode_1": ("soft_objective", "modes", "1", "mean", "lower"),
    "mode_0_direction": (
        "soft_causal_direction", "modes", "0", "alignment_score", "higher",
    ),
    "mode_1_direction": (
        "soft_causal_direction", "modes", "1", "alignment_score", "higher",
    ),
    "mode_0_monotonicity": (
        "soft_causal_monotonicity", "modes", "0", "alignment_score", "higher",
    ),
    "mode_1_monotonicity": (
        "soft_causal_monotonicity", "modes", "1", "alignment_score", "higher",
    ),
}


def _utility(worker: dict, endpoint: str) -> float:
    contract = GRADIENT_ENDPOINTS[endpoint]
    value = _get(worker, tuple(contract[:-1]))
    return -value if contract[-1] == "lower" else value


def _ridge_loo(design: np.ndarray, values: np.ndarray,
               lambdas: list[float]) -> dict:
    best = None
    identity = np.eye(design.shape[1])
    for ridge in lambdas:
        predictions = []
        for omitted in range(len(values)):
            keep = np.arange(len(values)) != omitted
            x_train = design[keep]
            gradient = np.linalg.solve(
                x_train.T @ x_train + ridge * identity,
                x_train.T @ values[keep],
            )
            predictions.append(float(design[omitted] @ gradient))
        predictions = np.asarray(predictions)
        mse = float(np.mean((predictions - values) ** 2))
        if best is None or mse < best[0]:
            best = (mse, ridge, predictions)
    mse, ridge, predictions = best
    pearson = float(pearsonr(values, predictions).statistic)
    spearman = float(spearmanr(values, predictions).statistic)
    sign_fraction = float(np.mean(np.sign(values) == np.sign(predictions)))
    gradient = np.linalg.solve(
        design.T @ design + ridge * identity, design.T @ values,
    )
    return {
        "ridge": float(ridge),
        "loo_rmse": float(np.sqrt(mse)),
        "slope_sd": float(np.std(values)),
        "loo_pearson": pearson,
        "loo_spearman": spearman,
        "loo_sign_agreement_fraction": sign_fraction,
        "gradient": gradient.tolist(),
    }


def gradient_identifiability(manifest: dict, summary: dict, *, contract: dict) -> dict:
    rows = _rows(summary)
    directions = []
    pairs = []
    for candidate in manifest["candidates"]:
        record = candidate["node_field"].get("residual_coordinates", {})
        if record.get("sign") != 1:
            continue
        positive_id = candidate["candidate_id"]
        negative_id = positive_id[:-1] + "m"
        radius = float(record["radius"])
        directions.append(np.asarray(record["coefficients"], float) / radius)
        pairs.append((positive_id, negative_id, radius))
    design = np.asarray(directions)
    singular_values = np.linalg.svd(design, compute_uv=False)
    endpoints = {}
    lambdas = [float(value) for value in contract["ridge_lambdas"]]
    for endpoint in GRADIENT_ENDPOINTS:
        slopes_by_network = []
        seeds = None
        for positive_id, negative_id, radius in pairs:
            positive = {
                int(row["seed"]): row for row in rows[positive_id]["per_network"]
            }
            negative = {
                int(row["seed"]): row for row in rows[negative_id]["per_network"]
            }
            if positive.keys() != negative.keys():
                raise RuntimeError("Stage-Z antithetic networks do not align")
            if seeds is None:
                seeds = sorted(positive)
            slopes_by_network.append([
                (
                    _utility(positive[network], endpoint)
                    - _utility(negative[network], endpoint)
                ) / (2.0 * radius)
                for network in seeds
            ])
        slopes_by_network = np.asarray(slopes_by_network)
        mean_slopes = np.mean(slopes_by_network, axis=1)
        fit = _ridge_loo(design, mean_slopes, lambdas)
        network_gradients = []
        identity = np.eye(design.shape[1])
        for network_index in range(slopes_by_network.shape[1]):
            gradient = np.linalg.solve(
                design.T @ design + fit["ridge"] * identity,
                design.T @ slopes_by_network[:, network_index],
            )
            network_gradients.append(gradient)
        cosines = []
        for left in range(len(network_gradients)):
            for right in range(left + 1, len(network_gradients)):
                a, b = network_gradients[left], network_gradients[right]
                denominator = np.linalg.norm(a) * np.linalg.norm(b)
                cosines.append(float(np.dot(a, b) / denominator) if denominator else 0.0)
        fit["network_gradient_pairwise_cosines"] = cosines
        fit["network_gradient_cosine_median"] = float(np.median(cosines))
        fit["n_antithetic_directions"] = int(len(pairs))
        endpoints[endpoint] = fit
    required = list(contract["required_endpoints"])
    identifiable = bool(all(
        endpoints[name]["loo_pearson"] >= float(contract["minimum_loo_pearson"])
        and endpoints[name]["loo_sign_agreement_fraction"] >= float(
            contract["minimum_sign_agreement_fraction"]
        )
        and endpoints[name]["network_gradient_cosine_median"] >= float(
            contract["minimum_network_gradient_cosine"]
        )
        for name in required
    ))
    return {
        "status": (
            "CURRENT_THREE_NETWORK_GRADIENT_IDENTIFIABLE" if identifiable else
            "CURRENT_THREE_NETWORK_GRADIENT_NOT_IDENTIFIABLE"
        ),
        "identifiable": identifiable,
        "design": {
            "shape": list(design.shape),
            "rank": int(np.linalg.matrix_rank(design)),
            "condition_number": float(np.linalg.cond(design)),
            "singular_values": singular_values.tolist(),
        },
        "endpoints": endpoints,
        "contract": contract,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args()
    config = json.loads(args.config.resolve().read_text())
    artifact_root = args.artifact_root.resolve()
    loaded, inputs = {}, {}
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        observed = _sha256(path)
        if observed != record["sha256"]:
            raise RuntimeError(f"Stage-AE input changed: {record['path']}")
        loaded[name] = json.loads(path.read_text())
        inputs[name] = {"path": str(path), "sha256": observed}
    if not loaded["stage_ad_audit"]["status"].startswith(
            "NODE_ONLY_DIRECTIONAL_CAPACITY_POSITIVE"):
        raise RuntimeError("Stage-AD does not support a conflict audit")
    stage_aa = _rows(loaded["stage_aa_summary"])
    stage_ad = _rows(loaded["stage_ad_summary"])
    capacity_id = config["capacity_candidate_id"]
    comparisons = {}
    bootstrap = config["bootstrap"]
    for index, comparison_id in enumerate(config["comparison_candidate_ids"]):
        comparisons[comparison_id] = conflict_decomposition(
            stage_ad[capacity_id], stage_aa[comparison_id],
            draws=int(bootstrap["draws"]),
            confidence=float(bootstrap["confidence"]),
            seed=int(bootstrap["seed"]) + 10000 * index,
        )
    gradient = gradient_identifiability(
        loaded["stage_z_manifest"], loaded["stage_z_summary"],
        contract=config["gradient_identifiability"],
    )
    mode0_conflict = comparisons["stage_z_g04_m"]["observed_tradeoff"][
        "mode_0_all_four_losses_worse"
    ]
    mode1_gain = comparisons["stage_z_g04_m"]["observed_tradeoff"][
        "mode_1_direction_higher"
    ]
    status = (
        "OBSERVED_CROSS_MODE_TRADEOFF_NEW_PAIRED_DESIGN_REQUIRED"
        if mode0_conflict and mode1_gain and not gradient["identifiable"] else
        "JOINT_OBJECTIVE_CONFLICT_AUDIT_INCONCLUSIVE"
    )
    payload = {
        "schema_id": "topic4_rev12_nd_joint_objective_conflict_audit_v1",
        "status": status,
        "paired_conflict_decomposition": comparisons,
        "stage_z_gradient_identifiability": gradient,
        "patient_heldout_used": False,
        "manual_field_used_as_target": False,
        "inputs": inputs,
        "claim_boundary": config["claim_boundary"],
    }
    output_root = artifact_root / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "joint_objective_conflict_audit.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "status": status,
        "gradient_status": gradient["status"],
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
