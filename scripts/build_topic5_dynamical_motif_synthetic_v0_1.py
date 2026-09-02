#!/usr/bin/env python3
"""Synthetic identifiability cells for the Topic 5.2 motif ladder.

Each cell instantiates one motif model with known parameters on a controlled
geometry, generates rank-set events from it, and writes them into the same
frame-cache layout the real data uses.  The normal trainer and aggregator then
run on the cell unchanged, so the resulting map describes *this pipeline's*
power, not an idealised estimator's.

A cell where the truth is not recovered is a power statement about that corner
of the design; it is not an engineering failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_lbss_rnn_v0_2 import build_pool_contract  # noqa: E402
from src.topic5_virtual_seeg_operator import kernel_sigma_mm, resolve_full_tissue_layout  # noqa: E402

NODE_SEED = 20260812

CELLS = [
    # truth, strength, contacts, events, ranks, noise, tie, stop spread
    {"truth": "DM0_ISOTROPIC", "strength": 0.0},
    {"truth": "DM1_FREE_AXIS", "strength": 0.15},
    {"truth": "DM1_FREE_AXIS", "strength": 0.45},
    {"truth": "DM1_FREE_AXIS", "strength": 0.90},
    {"truth": "DM2_LOCAL_DIRECTIONAL", "strength": 0.15},
    {"truth": "DM2_LOCAL_DIRECTIONAL", "strength": 0.50},
    {"truth": "DM2_LOCAL_DIRECTIONAL", "strength": 1.20},
    {"truth": "DM3_AXIS_FEEDFORWARD_TRANSIENT", "strength": 0.05},
    {"truth": "DM3_AXIS_FEEDFORWARD_TRANSIENT", "strength": 0.20},
    {"truth": "DM3_AXIS_FEEDFORWARD_TRANSIENT", "strength": 0.60},
]
SIZES = [
    {"n_contacts": 8, "n_events": 4000, "label": "small"},
    {"n_contacts": 16, "n_events": 12000, "label": "medium"},
    {"n_contacts": 8, "n_events": 800, "label": "few_events"},
]
NOISE_LEVELS = [{"noise": 0.0, "tie_rate": 0.0, "label": "clean"},
                {"noise": 0.6, "tie_rate": 0.05, "label": "noisy"}]


def shaft_like_contacts(n_contacts: int, n_shafts: int = 3, pitch: float = 4.0,
                        spacing: float = 12.0) -> np.ndarray:
    """Contacts on parallel shafts, the layout SEEG actually produces."""
    per_shaft = int(np.ceil(n_contacts / n_shafts))
    points = []
    for shaft in range(n_shafts):
        for index in range(per_shaft):
            points.append([index * pitch, shaft * spacing])
    return np.asarray(points[:n_contacts], dtype=np.float32)


def build_cell_geometry(n_contacts: int) -> dict:
    contacts = shaft_like_contacts(n_contacts)
    sigma = float(np.float32(kernel_sigma_mm(contacts, floor_mm=0.0)))
    layout = resolve_full_tissue_layout(contacts.astype(float), sigma, seed=NODE_SEED)
    nodes = np.asarray(layout.nodes_xy, dtype=float)
    distance = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
    pools = build_pool_contract(distance)
    return {"contacts": contacts, "nodes": nodes, "H": np.asarray(layout.H, float),
            "D": distance, "sigma": sigma, "pools": pools}


def generate_events(
    contacts_xy_mm: np.ndarray,
    n_events: int,
    seed: int,
    *,
    ell_mm: float,
    eta: float,
    beta: float,
    theta: float,
    noise: float,
    tie_rate: float,
    max_ranks: int,
    stop_probability: float,
) -> np.ndarray:
    """Generate rank-set sequences from a transparent contact-level transport rule.

    The earlier version rolled the tissue RNN forward, but with a unit input
    gain the retained bump dominated the read-out and the emitted sequences
    carried no axial drift at all -- P(step forward) was 0.500 even at the
    largest directional strength, so the cell tested recovery on noise.

    Here the next contact is drawn directly from

        score_c = -d_u(r_c, r_now)^2 / 2  +  beta * s * u.(r_c - r_now) / ell

    where ``d_u`` is the elliptical distance with semi-axes ``ell e^{eta}``
    along ``u`` and ``ell e^{-eta}`` across it, and ``s`` is the sign-and-size
    of the movement so far along ``u``.  The induced drift is therefore an
    explicit, checkable property of the cell rather than an emergent one.
    """
    rng = np.random.default_rng(int(seed))
    xy = np.asarray(contacts_xy_mm, dtype=float)
    n_contacts = xy.shape[0]
    u = np.array([np.cos(theta), np.sin(theta)])
    u_perp = np.array([-u[1], u[0]])
    ell_par, ell_orth = ell_mm * np.exp(eta), ell_mm * np.exp(-eta)

    ranks = np.full((n_events, n_contacts), -1, dtype=np.int16)
    start = rng.integers(0, n_contacts, size=n_events)
    ranks[np.arange(n_events), start] = 0
    current = xy[start].copy()
    origin = current.copy()
    alive = np.ones(n_events, dtype=bool)
    recruited = np.zeros((n_events, n_contacts), dtype=bool)
    recruited[np.arange(n_events), start] = True

    for step in range(1, max_ranks):
        alive &= ~recruited.all(axis=1)
        alive &= rng.random(n_events) >= stop_probability
        if not alive.any():
            break
        moved = current - origin
        s = np.tanh((moved @ u) / max(ell_mm, 1e-9))
        offset = xy[None, :, :] - current[:, None, :]
        along, across = offset @ u, offset @ u_perp
        score = -0.5 * ((along / ell_par) ** 2 + (across / ell_orth) ** 2)
        score = score + beta * s[:, None] * along / max(ell_mm, 1e-9)
        if noise > 0:
            score = score + noise * rng.normal(size=score.shape)
        score = np.where(recruited, -np.inf, score)
        probability = np.exp(score - score.max(axis=1, keepdims=True))
        total = probability.sum(axis=1, keepdims=True)
        probability = np.divide(probability, total, out=np.zeros_like(probability),
                                where=total > 0)
        draw = rng.random(n_events)
        pick = (probability.cumsum(axis=1) < draw[:, None]).sum(axis=1)
        pick = np.clip(pick, 0, n_contacts - 1)
        valid = alive & (total[:, 0] > 0)
        rows = np.flatnonzero(valid)
        ranks[rows, pick[rows]] = step
        recruited[rows, pick[rows]] = True
        tie = rows[rng.random(rows.size) < tie_rate]
        if tie.size:
            second = probability[tie].copy()
            second[np.arange(tie.size), pick[tie]] = 0.0
            mass = second.sum(axis=1, keepdims=True)
            usable = tie[mass[:, 0] > 0]
            if usable.size:
                normalised = second[mass[:, 0] > 0] / mass[mass[:, 0] > 0]
                extra = (normalised.cumsum(axis=1)
                         < rng.random(usable.size)[:, None]).sum(axis=1)
                extra = np.clip(extra, 0, n_contacts - 1)
                ranks[usable, extra] = step
                recruited[usable, extra] = True
        current[rows] = xy[pick[rows]]
        alive = valid
    return ranks


def axial_drift(ranks: np.ndarray, contacts_xy_mm: np.ndarray, theta: float) -> dict:
    """Model-free drift of the emitted cell, for the identifiability index."""
    xy = np.asarray(contacts_xy_mm, dtype=float)
    q = xy @ np.array([np.cos(theta), np.sin(theta)])
    steps = []
    for row in ranks:
        present = row[row >= 0]
        if present.size < 2:
            continue
        length = int(present.max()) + 1
        centroids = [q[row == t].mean() for t in range(length) if np.any(row == t)]
        steps.extend(np.diff(centroids).tolist())
    steps = np.asarray(steps)
    if steps.size == 0:
        return {"n_steps": 0}
    return {"n_steps": int(steps.size), "p_forward": float(np.mean(steps > 0)),
            "mean_step_mm": float(steps.mean())}


def write_cell(out_root: Path, cell_id: str, geometry: dict, ranks: np.ndarray,
               truth: dict) -> dict:
    directory = out_root / "frame_cache" / "SYNTHETIC" / cell_id
    directory.mkdir(parents=True, exist_ok=True)
    n_events, n_contacts = ranks.shape
    lengths = np.asarray([int(row[row >= 0].max()) + 1 if np.any(row >= 0) else 0
                          for row in ranks])
    keep = lengths >= 2
    ranks, lengths = ranks[keep], lengths[keep]
    n_events = int(ranks.shape[0])
    split = np.full(n_events, -1, dtype=np.int8)
    train_cut, calibration_cut = int(0.7 * n_events), int(0.85 * n_events)
    split[:train_cut] = 0
    split[train_cut:calibration_cut] = 1
    split[calibration_cut:int(0.95 * n_events)] = 2
    np.savez_compressed(
        directory / "plane.npz",
        contacts_xy_mm=geometry["contacts"].astype(np.float32),
        nodes_xy_mm=geometry["nodes"].astype(np.float32),
        H=geometry["H"].astype(np.float32),
        D_mm=geometry["D"].astype(np.float32),
        sigma_mm=np.asarray([geometry["sigma"]], np.float32),
        local_mask=geometry["pools"].local_mask.astype(np.uint8),
    )
    np.savez_compressed(
        directory / "events.npz",
        ranks=ranks.astype(np.int16), split=split,
        event_group_count=lengths.astype(np.int16),
        event_lag_raw=np.zeros_like(ranks, dtype=np.float32),
        event_abs_time=np.arange(n_events, dtype=np.float64),
        event_source_index=np.arange(n_events, dtype=np.int64),
        prefix_posterior=np.tile(np.asarray([0.5, 0.5], np.float32), (n_events, 1)),
        full_train_mode=np.zeros(n_events, np.int8),
        prefix_mode=np.zeros(n_events, np.int8),
        contact_names=np.asarray([f"C{i}" for i in range(n_contacts)], dtype=str),
        shafts=np.asarray([f"S{i // max(1, n_contacts // 3)}" for i in range(n_contacts)],
                          dtype=str),
    )
    np.savez_compressed(
        directory / "train_only_modes.npz",
        templates=np.zeros((2, n_contacts), np.float32),
        centers=np.zeros((2, n_contacts), np.float32),
        train_counts=np.asarray([train_cut, 0]),
        temperature=np.asarray([1.0], np.float32),
        own_cluster=np.asarray([0], np.int8),
    )
    provenance = {
        "contract": "topic5_dynamical_motif_synthetic_cell_v0_1",
        "frame": "SYNTHETIC", "subject": cell_id, "n_contacts": n_contacts,
        "n_events": n_events, "n_nodes": int(geometry["nodes"].shape[0]),
        "sigma_mm": geometry["sigma"],
        "r_local_mm": float(geometry["pools"].r_local_mm),
        "local_edges": int(geometry["pools"].local_mask.sum()),
        "ground_truth": truth,
        "target_values_read": False,
    }
    (directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False, default=float) + "\n")
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path,
                        default=ROOT / "results/topic5_dynamical_motif_rnn_v0_1")
    parser.add_argument("--max-ranks", type=int, default=8)
    parser.add_argument("--stop-probability", type=float, default=0.28)
    args = parser.parse_args()

    rows = []
    for size in SIZES:
        geometry = build_cell_geometry(size["n_contacts"])
        for noise in NOISE_LEVELS:
            for cell in CELLS:
                cell_id = (f"{cell['truth'].split('_')[0].lower()}"
                           f"_s{cell['strength']:.2f}_{size['label']}_{noise['label']}")
                truth_parameters = {"theta": 0.9, "eta_raw": 0.0, "beta": 0.0, "gamma_raw": 0.0}
                if cell["truth"] == "DM1_FREE_AXIS":
                    truth_parameters["eta_raw"] = cell["strength"]
                elif cell["truth"] == "DM2_LOCAL_DIRECTIONAL":
                    truth_parameters.update({"eta_raw": 0.3, "beta": cell["strength"]})
                elif cell["truth"] == "DM3_AXIS_FEEDFORWARD_TRANSIENT":
                    truth_parameters.update({"eta_raw": 0.3, "beta": 0.5,
                                             "gamma_raw": cell["strength"]})
                ell_mm = float(np.median(np.linalg.norm(
                    geometry["contacts"][:, None, :] - geometry["contacts"][None, :, :], axis=-1)
                    [~np.eye(size["n_contacts"], dtype=bool)]) / 2.0)
                ranks = generate_events(
                    geometry["contacts"], size["n_events"],
                    seed=int(hashlib.sha256(cell_id.encode()).hexdigest()[:8], 16),
                    ell_mm=ell_mm, eta=truth_parameters["eta_raw"],
                    beta=truth_parameters["beta"], theta=truth_parameters["theta"],
                    noise=noise["noise"], tie_rate=noise["tie_rate"],
                    max_ranks=args.max_ranks, stop_probability=args.stop_probability)
                drift = axial_drift(ranks, geometry["contacts"], truth_parameters["theta"])
                provenance = write_cell(args.out_root, cell_id, geometry, ranks, {
                    "model_id": cell["truth"], "strength": cell["strength"],
                    "parameters": truth_parameters, "noise": noise["noise"],
                    "tie_rate": noise["tie_rate"], "size_label": size["label"],
                    "generator": "contact_level_elliptical_transport_v0_1",
                    "generator_ell_mm": ell_mm, "induced_drift": drift,
                })
                rows.append({"cell_id": cell_id, "truth": cell["truth"],
                             "strength": cell["strength"], "size": size["label"],
                             "noise_label": noise["label"], "noise": noise["noise"],
                             "tie_rate": noise["tie_rate"],
                             "n_contacts": size["n_contacts"],
                             "n_events_kept": provenance["n_events"],
                             "n_nodes": provenance["n_nodes"],
                             "generator_ell_mm": ell_mm,
                             "induced_p_forward": drift.get("p_forward"),
                             "induced_mean_step_mm": drift.get("mean_step_mm")})
                print(f"[toy] {cell_id}: {provenance['n_events']} events, "
                      f"P(forward)={drift.get('p_forward', float('nan')):.4f}", flush=True)
    frame = pd.DataFrame(rows)
    (args.out_root / "toy_identifiability").mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_root / "toy_identifiability" / "CELL_MANIFEST.csv", index=False)
    print(f"[toy] {len(frame)} cells written")


if __name__ == "__main__":
    main()
