#!/usr/bin/env python3
"""Target-free contact influence and weighted motif analysis for v0.4."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_rnn_motif_v0_4 import RolloutSizeHead, state_features  # noqa: E402
from src.topic5_we_graph_analysis import module_of_each_node  # noqa: E402
from src.topic5_wiring_economy_rnn import WEConfig, WEModel, build_event_tensors  # noqa: E402


THEORY_MODELS = {
    "M1_DENSE", "M2_UNIFORM_SET", "M3_FIXED_LOCAL", "M4_SPATIAL_GROWTH",
    "M6_SPATIAL_MID", "M8_UNIFORM_COST_MID", "C_ORDER_SHUFFLED",
}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("empty influence summary")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def parse_shaft(name: str) -> str:
    value = str(name).upper().replace("-", "").replace("_", "")
    return "".join(character for character in value if not character.isdigit())


def instantiate(out_root: Path, metrics_path: Path, device: torch.device
                ) -> tuple[WEModel, RolloutSizeHead, dict, dict, dict]:
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
    decoder = RolloutSizeHead(int(provenance["n_contacts"])).to(device)
    decoder.load_state_dict(torch.load(metrics_path.parent / "rollout_size_head.pt",
                                       map_location=device, weights_only=True))
    decoder.eval()
    return model, decoder, metrics, plane, {"events": events, "provenance": provenance}


def prefix_inventory(tensors: dict[str, torch.Tensor], split: np.ndarray,
                     max_prefixes: int) -> list[tuple[int, int]]:
    candidates = []
    for event in np.flatnonzero(split == 2):
        valid = tensors["valid"][event].numpy()
        is_last = tensors["is_last"][event].numpy()
        for step in np.flatnonzero(valid & ~is_last):
            if step >= 1 and int(tensors["available"][event, step].sum()) >= 3:
                candidates.append((int(event), int(step)))
    if len(candidates) <= max_prefixes:
        return candidates
    take = np.linspace(0, len(candidates) - 1, max_prefixes).round().astype(int)
    return [candidates[index] for index in np.unique(take)]


def hidden_before(model: WEModel, x: torch.Tensor, step: int) -> torch.Tensor:
    h = torch.zeros(1, model.n_nodes * model.state_dim, device=x.device)
    with torch.no_grad():
        for index in range(step):
            h = model._step(h, x[index:index + 1])
    return h


def probabilities(model: WEModel, h: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    logits = model._readout(h).masked_fill(~available[None], -1e9)
    return torch.softmax(logits, dim=-1)[0]


@torch.no_grad()
def open_loop(model: WEModel, decoder: RolloutSizeHead, h_before: torch.Tensor,
              current: torch.Tensor, recruited: torch.Tensor, step: int,
              pulse_contact: int | None, pulse_scale: float, horizon: int = 3) -> torch.Tensor:
    h = h_before.clone()
    x = current.clone()
    if pulse_contact is not None:
        x[pulse_contact] += float(pulse_scale)
    recruited = recruited.clone()
    rows = []
    for lag in range(horizon):
        h = model._step(h, x[None])
        available = ~recruited.bool()
        if not bool(available.any()):
            rows.append(torch.zeros(model.n_contacts, device=x.device))
            continue
        p = probabilities(model, h, available)
        t = step + lag
        t_norm = torch.full((1,), t / max(1, model.n_contacts - 1), device=x.device)
        stop = torch.sigmoid(model._stop(h, t_norm, recruited.float().mean()[None])).item() > 0.5
        if stop:
            rows.append(torch.zeros_like(p))
            rows.extend(torch.zeros_like(p) for _ in range(lag + 1, horizon))
            break
        rows.append(p)
        feature = state_features(model, h, t, recruited.float().mean()[None])
        k = int(decoder(feature).argmax(-1).item()) + 1
        k = min(k, int(available.sum()))
        logits = model._readout(h).masked_fill(~available[None], -1e9)[0]
        chosen = torch.topk(logits, k=k, largest=True, sorted=True).indices
        x = torch.zeros_like(x); x[chosen] = 1.0
        recruited[chosen] = True
    if len(rows) < horizon:
        rows.extend(torch.zeros(model.n_contacts, device=x.device) for _ in range(len(rows), horizon))
    return torch.stack(rows[:horizon])


def teacher_jacobian(model: WEModel, h_before: torch.Tensor, current: torch.Tensor,
                     available: torch.Tensor) -> np.ndarray:
    def function(x: torch.Tensor) -> torch.Tensor:
        h = model._step(h_before, x[None])
        return probabilities(model, h, available)
    x = current.detach().clone().requires_grad_(True)
    jacobian = torch.autograd.functional.jacobian(function, x, vectorize=True)
    return jacobian.detach().cpu().numpy()


def edge_influence_rnn(model: WEModel, h_before: torch.Tensor, current: torch.Tensor,
                       available: torch.Tensor) -> np.ndarray:
    """Exact one-step deletion sensitivity factor for the leaky state-dim-1 cell."""
    if model.cell != "rnn" or model.state_dim != 1:
        strength = model.edge_strength().cpu().numpy()
        activity = np.abs(h_before.detach().cpu().numpy().reshape(model.n_nodes))
        return strength * activity[None, :]
    with torch.no_grad():
        w = model.masked_recurrent()[0]
        u = model._inject(current[None])[:, 0].reshape(1, -1)[0]
        h0 = h_before[0]
        pre = u + h0 @ w.T + model.bias[0]
        kappa = torch.sigmoid(model.kappa_logit)[0]
        h = (1 - kappa) * h0 + kappa * torch.tanh(pre)
        p = probabilities(model, h[None], available)
        jac_prob = torch.diag(p) - p[:, None] * p[None, :]
        node_sensitivity = []
        for node in range(model.n_nodes):
            direction = model.readout_gain * model.H[:, node]
            node_sensitivity.append((jac_prob @ direction).abs().mean())
        node_sensitivity = torch.stack(node_sensitivity)
        local_slope = kappa * (1.0 - torch.tanh(pre).square())
        effect = (w.abs() * h0.abs()[None, :] * local_slope[:, None]
                  * node_sensitivity[:, None])
        return effect.cpu().numpy()


def weighted_participation(weight: np.ndarray, membership: np.ndarray) -> np.ndarray:
    undirected = np.abs(weight) + np.abs(weight.T)
    out = np.zeros(len(weight), float)
    for node in range(len(weight)):
        total = float(undirected[node].sum())
        if total <= 0:
            continue
        shares = [float(undirected[node, membership == group].sum()) / total
                  for group in np.unique(membership[membership >= 0])]
        out[node] = 1.0 - np.sum(np.square(shares))
    return out


def summarize_unit(out_root: Path, metrics_path: Path, device: torch.device,
                   max_prefixes: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    model, decoder, metrics, plane, context = instantiate(out_root, metrics_path, device)
    events = context["events"]
    keep = events["split"] >= 0
    ranks = np.asarray(events["ranks"])[keep]
    split = np.asarray(events["split"])[keep]
    tensors = build_event_tensors(ranks)
    prefixes = prefix_inventory(tensors, split, max_prefixes)
    if not prefixes:
        raise RuntimeError(f"no eligible influence prefixes: {metrics['fit_id']}")
    h_matrix = torch.as_tensor(plane["H"], dtype=torch.float32, device=device)
    train_norms = []
    for event in np.flatnonzero(split == 0):
        for step in np.flatnonzero(tensors["valid"][event].numpy()):
            train_norms.append(float((tensors["x"][event, step].to(device) @ h_matrix).norm()))
    amplitude = float(np.median(train_norms))
    contact_norm = np.linalg.norm(np.asarray(plane["H"], float), axis=1)
    n_contacts = model.n_contacts
    tf_sum = np.zeros((n_contacts, n_contacts), float)
    tf_count = np.zeros((n_contacts, n_contacts), int)
    pulse_sum = np.zeros((3, n_contacts, n_contacts), float)
    pulse_count = np.zeros((3, n_contacts, n_contacts), int)
    edge_sum = np.zeros((model.n_nodes, model.n_nodes), float)
    edge_count = 0
    lag1_agreement = []
    for event, step in prefixes:
        x_grid = tensors["x"][event].to(device)
        current = x_grid[step]
        h0 = hidden_before(model, x_grid, step)
        recruited = tensors["recruited"][event, step].to(device).bool()
        available = tensors["available"][event, step].to(device).bool()
        jacobian = teacher_jacobian(model, h0, current, available)
        valid = np.outer(available.cpu().numpy(), available.cpu().numpy())
        tf_sum[valid] += jacobian[valid]
        tf_count[valid] += 1
        edge_sum += edge_influence_rnn(model, h0, current, available)
        edge_count += 1
        base = open_loop(model, decoder, h0, current, recruited, step, None, 0.0)
        for contact in torch.nonzero(available, as_tuple=False).flatten().cpu().numpy():
            if contact_norm[contact] <= 1e-12:
                continue
            scale = amplitude / contact_norm[contact]
            pulsed = open_loop(model, decoder, h0, current, recruited, step,
                               int(contact), scale)
            delta = (pulsed - base).cpu().numpy()
            for lag in range(3):
                outputs = available.cpu().numpy()
                pulse_sum[lag, outputs, contact] += delta[lag, outputs]
                pulse_count[lag, outputs, contact] += 1
            column = jacobian[:, contact]
            if np.linalg.norm(column) > 0 and np.linalg.norm(delta[0]) > 0:
                lag1_agreement.append(float(np.dot(column, delta[0])
                                            / (np.linalg.norm(column) * np.linalg.norm(delta[0]))))
    teacher = np.divide(tf_sum, tf_count, out=np.zeros_like(tf_sum), where=tf_count > 0)
    pulse = np.divide(pulse_sum, pulse_count, out=np.zeros_like(pulse_sum), where=pulse_count > 0)
    edge_effect = edge_sum / max(edge_count, 1)

    contact_xy = np.asarray(plane["contacts_xy_mm"], float)
    contact_distance = np.linalg.norm(contact_xy[:, None] - contact_xy[None, :], axis=-1)
    shafts = [parse_shaft(name) for name in context["provenance"]["contacts"]]
    same_shaft = np.equal.outer(shafts, shafts)
    off_contact = ~np.eye(n_contacts, dtype=bool)
    reach = []
    for lag in range(3):
        weight = np.abs(pulse[lag]) * off_contact
        reach.append(float((weight * contact_distance).sum() / max(weight.sum(), 1e-12)))

    graph = dict(np.load(metrics_path.parent / "graph.npz"))
    mask = np.asarray(graph["mask"], bool)
    distance = np.asarray(graph["D_mm"], float)
    strength = np.asarray(graph["strength"], float) * mask
    effective = edge_effect * mask
    candidate = ~np.eye(mask.shape[0], dtype=bool)
    q50, q75 = np.quantile(distance[candidate], [0.50, 0.75])
    local_active = mask & (distance <= q50)
    local_threshold = np.quantile(effective[local_active], 0.75) if local_active.any() else np.inf
    high_threshold = np.quantile(effective[mask], 0.90) if mask.any() else np.inf
    local_backbone = local_active & (effective >= local_threshold)
    long_high = mask & (distance >= q75) & (effective >= high_threshold)
    membership, _ = module_of_each_node(mask, seed=int(metrics["seed"]))
    participation = weighted_participation(effective, membership)
    connector_threshold = np.quantile(participation, 0.75)
    incident_long = long_high.any(0) | long_high.any(1)
    connector_nodes = incident_long & (participation >= connector_threshold)
    motif_estimable = bool(local_backbone.sum() >= 3 and long_high.sum() >= 3
                           and connector_nodes.sum() >= 2)
    row = {
        "subject": metrics["subject"], "fit_id": metrics["fit_id"], "scope": metrics["fit_scope"],
        "model": metrics["model_id"].rsplit("__", 1)[0], "cell": metrics["cell"],
        "seed": int(metrics["seed"]), "n_prefixes": len(prefixes),
        "pulse_amplitude": amplitude, "tf_finite_pulse_lag1_cosine": float(np.nanmedian(lag1_agreement)),
        "lag1_reach_mm": reach[0], "lag2_reach_mm": reach[1], "lag3_reach_mm": reach[2],
        "lag1_abs_influence": float(np.mean(np.abs(pulse[0][off_contact]))),
        "lag2_abs_influence": float(np.mean(np.abs(pulse[1][off_contact]))),
        "lag3_abs_influence": float(np.mean(np.abs(pulse[2][off_contact]))),
        "lag1_same_shaft_abs": float(np.mean(np.abs(pulse[0][same_shaft & off_contact]))),
        "lag1_cross_shaft_abs": float(np.mean(np.abs(pulse[0][~same_shaft]))),
        "n_active_edges": int(mask.sum()), "n_local_backbone_edges": int(local_backbone.sum()),
        "n_long_high_edges": int(long_high.sum()), "n_connector_nodes": int(connector_nodes.sum()),
        "motif_estimable": motif_estimable,
        "effective_edge_definition": ("exact_leaky_one_step_deletion_sensitivity"
                                      if metrics["cell"] == "rnn" else "gru_gate_rms_activity_proxy"),
    }
    arrays = {
        "contacts": np.asarray(context["provenance"]["contacts"], dtype="U64"),
        "teacher_forced_lag1": teacher.astype(np.float32),
        "open_loop_pulse_lag123": pulse.astype(np.float32),
        "contact_distance_mm": contact_distance.astype(np.float32),
        "edge_effective_influence": effective.astype(np.float32),
        "edge_strength": strength.astype(np.float32), "edge_distance_mm": distance.astype(np.float32),
        "edge_mask": mask.astype(np.uint8), "local_backbone_mask": local_backbone.astype(np.uint8),
        "long_high_mask": long_high.astype(np.uint8), "connector_nodes": connector_nodes.astype(np.uint8),
        "weighted_participation": participation.astype(np.float32),
    }
    return row, arrays


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-prefixes", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    paths = []
    for path in sorted((out_root / "per_subject").glob("*/*__*/seed*/metrics.json")):
        directory = path.parents[1].name
        if directory.startswith("SMOKE_"):
            continue
        model, _ = directory.rsplit("__", 1)
        if model in THEORY_MODELS:
            paths.append(path)
    all_paths = list(paths)
    if args.limit is not None and args.n_shards == 1 and not args.aggregate_only:
        all_paths = all_paths[:args.limit]
    if not args.aggregate_only:
        paths = all_paths[args.shard_index::args.n_shards]
        if args.limit is not None:
            paths = paths[:args.limit]
        shard_rows = []
        for index, path in enumerate(paths, 1):
            output = out_root / "effective_influence" / path.parents[2].name / path.parents[1].name / path.parent.name / "influence.npz"
            output.parent.mkdir(parents=True, exist_ok=True)
            if output.exists():
                with np.load(output, allow_pickle=False) as data:
                    row = json.loads(str(data["summary_json"].item()))
            else:
                row, arrays = summarize_unit(out_root, path, torch.device(args.device), args.max_prefixes)
                arrays["summary_json"] = np.asarray(json.dumps(row))
                np.savez_compressed(output, **arrays)
            shard_rows.append(row)
            if index % 20 == 0:
                print(json.dumps({"complete": index, "total": len(paths),
                                  "shard": args.shard_index}), flush=True)
        if args.n_shards > 1:
            write_csv(out_root / f"effective_influence_shard_{args.shard_index:02d}.csv", shard_rows)
    if args.n_shards == 1 or args.aggregate_only:
        rows = []
        for path in all_paths:
            output = out_root / "effective_influence" / path.parents[2].name / path.parents[1].name / path.parent.name / "influence.npz"
            if not output.exists():
                raise RuntimeError(f"missing effective influence output: {output}")
            with np.load(output, allow_pickle=False) as data:
                rows.append(json.loads(str(data["summary_json"].item())))
        write_csv(out_root / "effective_influence_fit_seed.csv", rows)
        summary = {
            "contract": "topic5_rnn_motif_effective_influence_v0_4",
            "target_values_read": False, "n_units": len(rows),
            "expected_units": len(all_paths), "max_prefixes_per_unit": args.max_prefixes,
            "primary_observables": ["lag1_2_3_open_loop_contact_reach", "local_backbone_long_range_connector"],
            "teacher_forced_scope": "lag1_local_probability_jacobian_only",
            "open_loop_scope": "lag1_to_lag3_deterministic_decoder_without_future_ranks",
            "rank_step_not_real_time": True,
        }
        (out_root / "EFFECTIVE_INFLUENCE_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
