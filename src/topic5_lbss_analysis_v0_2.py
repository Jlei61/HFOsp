"""Target-free loaders and perturbation helpers for LBSS v0.2."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import torch

from src.topic5_lbss_rnn_v0_2 import LBSSConfig, LBSSModel, build_pool_contract
from src.topic5_rnn_motif_v0_4 import RolloutSizeHead


def upsert_figure_readme(readme: Path, heading: str, entry: str) -> None:
    """Replace one figure's README section in place, keeping every other section.

    The stage scripts previously either clobbered the whole file with
    ``write_text`` or appended blindly, so re-running a plot deleted the other
    figures' notes or produced duplicate sections.
    """
    sections: list[str] = []
    if readme.exists():
        current = readme.read_text()
        sections = [block for block in current.split("\n### ") if block.strip()]
        sections = [block if block.startswith("### ") else "### " + block.lstrip("# ")
                    for block in sections]
        sections = [block for block in sections if not block.startswith(f"### {heading}")]
    sections.append(entry.strip() + "\n")
    readme.write_text("\n".join(block.rstrip() + "\n" for block in sections))


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(mask, np.uint8)).view(np.uint8)).hexdigest()


def instantiate_lbss(
    out_root: Path,
    metrics_path: Path,
    device: torch.device,
) -> tuple[LBSSModel, RolloutSizeHead, dict, dict, dict, dict]:
    metrics = json.loads(metrics_path.read_text())
    cache = out_root / "cache" / metrics["fit_id"]
    plane_file = np.load(cache / "plane.npz", allow_pickle=False)
    events_file = np.load(cache / "events.npz", allow_pickle=False)
    plane = {key: plane_file[key] for key in plane_file.files}
    events = {key: events_file[key] for key in events_file.files}
    provenance = json.loads((cache / "provenance.json").read_text())
    cfg = metrics["config"]
    pools = build_pool_contract(
        plane["D_mm"], cfg["density"], cfg["added_fraction"],
        cfg.get("r_local_multiplier", 2.0),
    )
    model = LBSSModel(LBSSConfig(
        arm=metrics["arm"],
        n_contacts=int(provenance["n_contacts"]),
        n_nodes=int(provenance["n_nodes"]),
        observation_operator=plane["H"],
        node_distance_mm=plane["D_mm"],
        local_mask=pools.local_mask,
        extra_local_pool=pools.extra_local_pool,
        nonlocal_pool=pools.nonlocal_pool,
        k_added=pools.k_added,
        seed=int(metrics["seed"]),
        state_dim=int(cfg["state_dim"]),
    )).to(device)
    model.load_state_dict(torch.load(metrics_path.parent / "weights.pt", map_location=device, weights_only=True))
    model.freeze_mask()
    model.eval()
    decoder = RolloutSizeHead(int(provenance["n_contacts"])).to(device)
    decoder.load_state_dict(torch.load(
        metrics_path.parent / "rollout_size_head.pt", map_location=device, weights_only=True
    ))
    decoder.eval()
    return model, decoder, metrics, plane, events, provenance


def attenuate_mask(model: LBSSModel, node_mask: np.ndarray, alpha: float) -> LBSSModel:
    mask = torch.as_tensor(np.asarray(node_mask, bool), device=model.recurrent.device)
    if mask.shape != model.node_mask.shape:
        raise ValueError("attenuation mask must align to the tissue recurrent graph")
    if bool((mask & ~model.node_mask.bool()).any()):
        raise ValueError("attenuation may target only active edges in this model")
    with torch.no_grad():
        model.recurrent[:, mask] *= 1.0 - float(alpha)
    return model


def endpoint_density(
    strength: np.ndarray,
    mask: np.ndarray,
    observation_operator: np.ndarray,
) -> dict[str, np.ndarray]:
    weighted = np.asarray(strength, float) * np.asarray(mask, bool)
    source_node = weighted.sum(axis=0)
    target_node = weighted.sum(axis=1)

    def normalize(value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, float)
        total = value.sum()
        return value / total if total > 0 else np.zeros_like(value)

    h = np.asarray(observation_operator, float)
    return {
        "source_node": normalize(source_node),
        "target_node": normalize(target_node),
        "source_contact": normalize(h @ source_node),
        "target_contact": normalize(h @ target_node),
    }


def edge_set_descriptors(
    mask: np.ndarray,
    strength: np.ndarray,
    nodes_xy_mm: np.ndarray,
    observation_operator: np.ndarray,
) -> dict[str, float | int | list[int]]:
    """Descriptors used to match a local control to the learned LR target."""
    selected = np.asarray(mask, bool)
    weight = np.asarray(strength, float) * selected
    source_weight = weight.sum(axis=0)
    target_weight = weight.sum(axis=1)
    source_nodes = selected.any(axis=0)
    target_nodes = selected.any(axis=1)
    source_degree = selected.sum(axis=0).astype(int)
    target_degree = selected.sum(axis=1).astype(int)
    xy = np.asarray(nodes_xy_mm, float)
    support = np.asarray(observation_operator, float).max(axis=0) > 1e-6

    def extent(node_weight: np.ndarray) -> float:
        total = float(node_weight.sum())
        if total <= 0:
            return 0.0
        centre = (xy * node_weight[:, None]).sum(axis=0) / total
        return float(np.sqrt(((xy - centre) ** 2).sum(axis=1).dot(node_weight) / total))

    return {
        "edge_count": int(selected.sum()),
        "total_abs_weight": float(weight.sum()),
        "unique_source_nodes": int(source_nodes.sum()),
        "unique_target_nodes": int(target_nodes.sum()),
        "source_extent_mm": extent(source_weight),
        "target_extent_mm": extent(target_weight),
        "supported_source_nodes": int((source_nodes & support).sum()),
        "supported_target_nodes": int((target_nodes & support).sum()),
        # Sorted profiles match recurrent fan-out/fan-in heterogeneity without
        # forcing a local control to reuse the nonlocal edge endpoints.
        "source_degree_profile": np.sort(source_degree).tolist(),
        "target_degree_profile": np.sort(target_degree).tolist(),
    }


def _within_relative(value: float, target: float, lower: float, upper: float) -> bool:
    if target <= 1e-12:
        return abs(value - target) <= 1e-12
    ratio = value / target
    return lower <= ratio <= upper


def _count_within(value: int, target: int) -> bool:
    return abs(int(value) - int(target)) <= max(2, int(np.ceil(0.20 * max(1, int(target)))))


def local_control_match_score(candidate: dict[str, float], target: dict[str, float]) -> float:
    """Scale-free mismatch score; it ranks only candidates already inside calipers."""
    ratios = []
    for key in ("total_abs_weight", "source_extent_mm", "target_extent_mm"):
        denominator = max(abs(float(target[key])), 1e-12)
        ratios.append((float(candidate[key]) - float(target[key])) / denominator)
    for key in (
        "unique_source_nodes", "unique_target_nodes",
        "supported_source_nodes", "supported_target_nodes",
    ):
        denominator = max(abs(float(target[key])), 1.0)
        ratios.append((float(candidate[key]) - float(target[key])) / denominator)
    for key in ("source_degree_profile", "target_degree_profile"):
        left = np.asarray(candidate[key], float)
        right = np.asarray(target[key], float)
        ratios.append(float(np.abs(left - right).sum() / max(1.0, right.sum())))
    return float(np.sqrt(np.mean(np.square(ratios))))


def match_local_control_subsets(
    local_mask: np.ndarray,
    target_mask: np.ndarray,
    strength: np.ndarray,
    nodes_xy_mm: np.ndarray,
    observation_operator: np.ndarray,
    seed: int,
    max_candidate_draws: int = 20_000,
    keep_valid: int = 500,
    evaluate_best: int = 16,
) -> dict:
    """Freeze matched K-edge local controls without looking at model endpoints."""
    local = np.asarray(local_mask, bool)
    target = np.asarray(target_mask, bool)
    if target.sum() < 1 or local.sum() < target.sum():
        raise ValueError("local-control matching requires K active target and at least K local edges")
    edge_index = np.argwhere(local)
    k = int(target.sum())
    target_desc = edge_set_descriptors(
        target, strength, nodes_xy_mm, observation_operator
    )
    rng = np.random.default_rng(int(seed))
    accepted: list[tuple[float, np.ndarray, dict[str, float]]] = []
    seen: set[str] = set()
    for _ in range(int(max_candidate_draws)):
        take = rng.choice(len(edge_index), size=k, replace=False)
        mask = np.zeros_like(local)
        mask[tuple(edge_index[take].T)] = True
        digest = mask_sha256(mask)
        if digest in seen:
            continue
        seen.add(digest)
        desc = edge_set_descriptors(mask, strength, nodes_xy_mm, observation_operator)
        valid = (
            _within_relative(desc["total_abs_weight"], target_desc["total_abs_weight"], 0.75, 1.25)
            and _count_within(desc["unique_source_nodes"], target_desc["unique_source_nodes"])
            and _count_within(desc["unique_target_nodes"], target_desc["unique_target_nodes"])
            and _within_relative(desc["source_extent_mm"], target_desc["source_extent_mm"], 0.70, 1.30)
            and _within_relative(desc["target_extent_mm"], target_desc["target_extent_mm"], 0.70, 1.30)
            and _count_within(desc["supported_source_nodes"], target_desc["supported_source_nodes"])
            and _count_within(desc["supported_target_nodes"], target_desc["supported_target_nodes"])
            and np.abs(
                np.asarray(desc["source_degree_profile"], float)
                - np.asarray(target_desc["source_degree_profile"], float)
            ).sum() / max(1.0, sum(target_desc["source_degree_profile"])) <= 0.30
            and np.abs(
                np.asarray(desc["target_degree_profile"], float)
                - np.asarray(target_desc["target_degree_profile"], float)
            ).sum() / max(1.0, sum(target_desc["target_degree_profile"])) <= 0.30
        )
        if valid:
            accepted.append((local_control_match_score(desc, target_desc), mask, desc))
    accepted.sort(key=lambda item: (item[0], mask_sha256(item[1])))
    accepted = accepted[: int(keep_valid)]
    selected = accepted[: min(int(evaluate_best), len(accepted))]
    shape = (0,) + local.shape if not selected else (len(selected),) + local.shape
    masks = np.empty(shape, dtype=np.uint8) if not selected else np.stack(
        [item[1].astype(np.uint8) for item in selected]
    )
    return {
        "target_descriptors": target_desc,
        "n_candidate_draws": int(max_candidate_draws),
        "n_unique_candidates": len(seen),
        "n_valid_matched_draws": len(accepted),
        "inferential_eligible": len(accepted) >= 200,
        "selected_masks": masks,
        "selected_scores": np.asarray([item[0] for item in selected], float),
        "selected_descriptors": [item[2] for item in selected],
        "selected_hashes": [mask_sha256(item[1]) for item in selected],
    }
