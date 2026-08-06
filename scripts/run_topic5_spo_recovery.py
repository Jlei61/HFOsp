"""Can these parameters be recovered at all?  Asked before any patient is fitted.

This is the gate v0.1 should have been designed around from the start.  There,
the free graph was asked to recover edge identity and failed, which retired two
hypotheses after the code was already written.  Here the same question is asked
first, of quantities chosen because they might survive it: the sign of the axial
drift, the ordering of the two diffusions, how far activity effectively reaches,
and how strong the recovery process is.

Events are generated from a known operator on a real patient's geometry, with
that patient's real observation kernel and real event lengths, and the fitter
sees nothing but the resulting contact rank sets.

A failure here is not a reason to stop: it decides which layers the patient
analysis is allowed to report, exactly as in v0.1.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_spatial_propagation_operator import (  # noqa: E402
    D_MAX, OperatorConfig, SPOModel, build_grid,
)
from src.topic5_virtual_seeg_operator import build_observation_operator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"
V1_CACHE = (ROOT.parent / "topic5-slp-rnn"
            / "results/topic5_spatial_latent_propagation_rnn_v0_1/cache")

# Eight cells, not a full grid: the question is whether each quantity moves with
# its generator, and a factorial would spend the budget on interactions nobody
# asked about.
# Chosen so the operator, not the per-contact bias, decides what is generated.
# Verified by assert_generator_is_informative below rather than by inspection.
GENERATOR_ETA = 60.0
GENERATOR_BIAS_SD = 0.15

CELLS = [
    {"name": "drift_forward",      "v": +0.35, "D_par": 0.08, "D_perp": 0.02, "beta": 0.2},
    {"name": "drift_backward",     "v": -0.35, "D_par": 0.08, "D_perp": 0.02, "beta": 0.2},
    {"name": "drift_none",         "v": 0.00,  "D_par": 0.05, "D_perp": 0.05, "beta": 0.2},
    {"name": "axial_anisotropy",   "v": +0.15, "D_par": 0.10, "D_perp": 0.01, "beta": 0.2},
    {"name": "transverse_dominant", "v": +0.15, "D_par": 0.01, "D_perp": 0.10, "beta": 0.2},
    {"name": "isotropic",          "v": +0.15, "D_par": 0.05, "D_perp": 0.05, "beta": 0.2},
    {"name": "recovery_absent",    "v": +0.25, "D_par": 0.06, "D_perp": 0.03, "beta": 0.0},
    {"name": "recovery_strong",    "v": +0.25, "D_par": 0.06, "D_perp": 0.03, "beta": 1.2},
]


def _inv_bounded(y: float, hi: float) -> float:
    z = min(max(y / hi, 1e-6), 1 - 1e-6)
    return float(np.log(z / (1 - z)))


def make_generator(config: OperatorConfig, cell: dict, seed: int) -> SPOModel:
    model = SPOModel(config)
    with torch.no_grad():
        op = model.operator
        op.raw_D_parallel.fill_(_inv_bounded(cell["D_par"], D_MAX))
        op.raw_D_perp.fill_(_inv_bounded(cell["D_perp"], D_MAX))
        op.v.fill_(float(np.arctanh(np.clip(cell["v"] / 0.5, -0.95, 0.95))))
        op.raw_gamma_a.fill_(_inv_bounded(0.25, 1.0))
        op.raw_gamma_r.fill_(_inv_bounded(0.35, 1.0))
        op.raw_beta.fill_(float(np.log(np.expm1(max(cell["beta"], 1e-6)))))
        op.raw_xi.fill_(float(np.log(np.expm1(0.4))))
        # A contact's footprint is spread over ~50 cells and read back through
        # the same kernel, so a unit injection reaches the logits at about 1/50
        # of its size. At eta=2 the field moved the logits by 0.04 against a
        # contact bias with sd 0.3 -- the operator was a rounding error on its
        # own generated data, and two opposite drifts produced identical events.
        op.raw_eta.fill_(float(np.log(np.expm1(GENERATOR_ETA))))
        model.contact_bias.normal_(0.0, GENERATOR_BIAS_SD,
                                   generator=torch.Generator().manual_seed(seed))
    model.eval()
    return model


@torch.no_grad()
def generate_events(model: SPOModel, n_events: int, lengths: np.ndarray,
                    seed: int) -> np.ndarray:
    """Rank sets produced by the operator itself, seen only through contacts."""
    rng = np.random.default_rng(seed)
    c = model.config.n_contacts
    out = np.full((n_events, c), -1, np.int16)
    for e in range(n_events):
        target_length = int(lengths[rng.integers(len(lengths))])
        recruited = torch.zeros(1, c)
        x_t = torch.zeros(1, c)
        x_t[0, rng.integers(c)] = 1.0
        state = model.initial_state(1, x_t.device)
        rank = 0
        out[e, x_t[0] > 0] = rank
        recruited = x_t.clone()
        for _ in range(target_length - 1):
            t_norm = torch.full((1, 1), rank / max(target_length - 1, 1))
            state, logits, _ = model.step(state, x_t, recruited, t_norm)
            logits = logits.masked_fill(recruited > 0, -1e9)
            probability = torch.sigmoid(logits)[0].numpy()
            available = np.flatnonzero(recruited[0].numpy() == 0)
            if not len(available):
                break
            p = probability[available]
            if p.sum() <= 0:
                break
            k = 1 + int(rng.random() < 0.25)          # sometimes two join at once
            k = min(k, len(available))
            chosen = rng.choice(available, size=k, replace=False, p=p / p.sum())
            rank += 1
            out[e, chosen] = rank
            x_t = torch.zeros(1, c)
            x_t[0, chosen] = 1.0
            recruited[0, chosen] = 1.0
    return out


def fit(config: OperatorConfig, events: np.ndarray, epochs: int,
        lr: float, seed: int) -> SPOModel:
    from src.topic5_spatial_latent_rnn import build_event_tensors, next_set_stop_loss

    torch.manual_seed(seed)
    model = SPOModel(config)
    tensors = build_event_tensors(events)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    n = tensors.x.shape[0]
    batch = min(64, max(8, n // 8))
    for _ in range(epochs):
        order = torch.randperm(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            logits, stop = model(tensors.x[idx], tensors.recruited[idx],
                                 tensors.valid[idx])
            loss, _, _ = next_set_stop_loss(
                logits, stop, tensors.target[idx], tensors.available[idx],
                tensors.valid[idx], tensors.is_last[idx],
            )
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimiser.step()
    model.eval()
    return model


def assert_generator_is_informative(base: dict, lengths: np.ndarray,
                                    n_events: int, axis_mm: np.ndarray) -> dict:
    """Two opposite drifts must generate visibly different events.

    If they do not, the operator is not driving its own synthetic data and every
    recovery verdict below is a statement about the sampler, not about
    identifiability. This is a guard, not a diagnostic: the run stops here.
    """
    forward = {"name": "guard_fwd", "v": +0.35, "D_par": 0.08, "D_perp": 0.02, "beta": 0.2}
    backward = dict(forward, name="guard_bwd", v=-0.35)
    events = {}
    for cell in (forward, backward):
        generator = make_generator(
            OperatorConfig(variant="ANISOTROPIC_RECOVERY", seed=1, **base), cell, 1
        )
        events[cell["name"]] = generate_events(generator, n_events, lengths, 1)
    a, b = events["guard_fwd"], events["guard_bwd"]
    disagreement = float((a != b).mean())
    # Where does each event start relative to where it ends, along the axis?
    def axial_trend(ranks: np.ndarray, order: np.ndarray) -> float:
        keep = ranks >= 0
        if keep.sum() < 3:
            return np.nan
        return float(stats.spearmanr(ranks[keep], order[keep]).statistic)
    # The contact's real position on the propagation axis, not its index in the
    # array: the array order carries no geometry and would make the trend
    # meaningless.
    order = np.asarray(axis_mm, float)
    trend_a = np.nanmedian([axial_trend(r, order) for r in a])
    trend_b = np.nanmedian([axial_trend(r, order) for r in b])
    report = {
        "cell_disagreement_fraction": disagreement,
        "median_rank_vs_contact_index_spearman_forward": float(trend_a),
        "median_rank_vs_contact_index_spearman_backward": float(trend_b),
        "trend_separation": float(trend_a - trend_b),
    }
    if disagreement < 0.15 or not np.isfinite(report["trend_separation"]):
        raise SystemExit(
            "generator guard failed: opposite drifts produced near-identical "
            f"events (disagreement {disagreement:.3f}). The operator is not "
            "driving its own synthetic data, so no recovery verdict is meaningful. "
            "Raise GENERATOR_ETA or lower GENERATOR_BIAS_SD."
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="epilepsiae_1146")
    parser.add_argument("--n-events", type=int, default=900)
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--lr", type=float, default=3e-3)
    # Swept, not fixed. The drift a rank can produce is v * microsteps in grid
    # cells, and a contact reads a disc of radius 3 sigma; if the first is much
    # smaller than the second the displacement is invisible whatever v is. A
    # verdict from one microstep count could not tell those two apart.
    parser.add_argument("--microsteps", type=int, nargs="+", default=[3, 6])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    args = parser.parse_args()

    (OUT / "synthetic").mkdir(parents=True, exist_ok=True)
    plane = np.load(V1_CACHE / args.subject / "plane_coordinates.npz", allow_pickle=True)
    contacts, sigma = plane["xy_mm"], float(plane["sigma_mm"][0])
    reference = np.load(V1_CACHE / args.subject / "events.npz")["group_ids"]
    lengths = np.array([r[r >= 0].max() + 1 for r in reference if (r >= 0).any()])

    centres, shape, mask = build_grid(contacts, sigma)
    H = build_observation_operator(contacts, centres, sigma)
    base = dict(n_contacts=len(contacts), grid_shape=shape,
                microsteps=args.microsteps, observation_operator=H, grid_mask=mask)
    print(f"{args.subject}: {len(contacts)} contacts, grid {shape}, "
          f"{int(mask.sum())} cells in domain, event lengths "
          f"{lengths.min()}-{lengths.max()}")

    guard = assert_generator_is_informative(
        base, lengths, min(args.n_events, 300), contacts[:, 0]
    )
    print(f"generator guard: {guard['cell_disagreement_fraction']:.3f} of ranks differ "
          f"between opposite drifts; axial trend {guard['median_rank_vs_contact_index_spearman_forward']:+.3f} "
          f"vs {guard['median_rank_vs_contact_index_spearman_backward']:+.3f}")

    rows = []
    for microsteps in args.microsteps:
      base = dict(base, microsteps=microsteps)
      for cell in CELLS:
        for seed in args.seeds:
            generator = make_generator(
                OperatorConfig(variant="ANISOTROPIC_RECOVERY", seed=seed, **base), cell, seed
            )
            events = generate_events(generator, args.n_events, lengths, seed)
            fitted = fit(
                OperatorConfig(variant="ANISOTROPIC_RECOVERY", seed=seed + 100, **base),
                events, args.epochs, args.lr, seed + 100,
            )
            est = fitted.parameter_estimates()
            rows.append({
                "cell": cell["name"], "seed": seed, "microsteps": microsteps,
                "true_v": cell["v"], "fitted_v": est["v"],
                "true_D_parallel": cell["D_par"], "fitted_D_parallel": est["D_parallel"],
                "true_D_perp": cell["D_perp"], "fitted_D_perp": est["D_perp"],
                "true_beta": cell["beta"], "fitted_beta": est["beta"],
                "true_anisotropy": cell["D_par"] / max(cell["D_perp"], 1e-9),
                "fitted_anisotropy": est["anisotropy"],
                "fitted_axial_reach": est["effective_axial_reach_per_rank"],
                "fitted_recovery_strength": est["recovery_strength"],
                "n_events": int(len(events)),
            })
            print(f"  K={microsteps} {cell['name']:20s} seed{seed}  v {cell['v']:+.2f}->"
                  f"{est['v']:+.3f}   D_par {cell['D_par']:.3f}->{est['D_parallel']:.3f}"
                  f"   D_perp {cell['D_perp']:.3f}->{est['D_perp']:.3f}"
                  f"   beta {cell['beta']:.2f}->{est['beta']:.3f}", flush=True)

    with (OUT / "synthetic_parameter_recovery.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # --- the three layers, each with its own pre-set bar --------------------
    def layers_for(subset: list[dict]) -> dict:
        signed_ = [r for r in subset if abs(r["true_v"]) > 1e-9]
        sign_ = float(np.mean([np.sign(r["fitted_v"]) == np.sign(r["true_v"])
                               for r in signed_])) if signed_ else float("nan")
        rho_ = stats.spearmanr(
            [np.log(r["true_anisotropy"]) for r in subset],
            [np.log(max(r["fitted_anisotropy"], 1e-9)) for r in subset],
        ).statistic
        absent_ = [r["fitted_recovery_strength"] for r in subset if r["true_beta"] == 0.0]
        strong_ = [r["fitted_recovery_strength"] for r in subset if r["true_beta"] > 1.0]
        return {"drift_sign_agreement": sign_, "anisotropy_spearman": float(rho_),
                "recovery_absent": float(np.median(absent_)) if absent_ else None,
                "recovery_strong": float(np.median(strong_)) if strong_ else None}

    by_microsteps = {str(k): layers_for([r for r in rows if r["microsteps"] == k])
                     for k in args.microsteps}
    # The verdict takes the most favourable microstep count: a failure has to
    # hold where the operator had its best chance, or it is a budget artefact.
    best_k = max(args.microsteps,
                 key=lambda k: by_microsteps[str(k)]["drift_sign_agreement"])
    rows_best = [r for r in rows if r["microsteps"] == best_k]
    signed = [r for r in rows_best if abs(r["true_v"]) > 1e-9]
    sign_agreement = float(np.mean([
        np.sign(r["fitted_v"]) == np.sign(r["true_v"]) for r in signed
    ]))
    aniso_rho = stats.spearmanr(
        [np.log(r["true_anisotropy"]) for r in rows_best],
        [np.log(max(r["fitted_anisotropy"], 1e-9)) for r in rows_best],
    ).statistic
    absent = [r["fitted_recovery_strength"] for r in rows_best if r["true_beta"] == 0.0]
    strong = [r["fitted_recovery_strength"] for r in rows_best if r["true_beta"] > 1.0]
    recovery_ordered = bool(np.median(strong) > np.median(absent)) if absent and strong else False

    gate = {
        "contract": "topic5_spo_recovery_gate_v0_2",
        "asked_before_any_patient_was_fitted": True,
        "subject_geometry": args.subject,
        "n_cells": len(CELLS), "seeds": args.seeds,
        "generator_guard": guard,
        "microsteps_swept": args.microsteps,
        "verdict_taken_from_microsteps": best_k,
        "by_microsteps": by_microsteps,
        "drift_sign": {
            "agreement": sign_agreement, "floor": 0.80,
            "n_signed_cells": len(signed),
            "status": "RECOVERABLE" if sign_agreement >= 0.80 else "NOT_RECOVERABLE",
        },
        "anisotropy_ordering": {
            "spearman": float(aniso_rho), "floor": 0.60,
            "status": "RECOVERABLE" if aniso_rho >= 0.60 else "NOT_RECOVERABLE",
        },
        "recovery_strength_ordering": {
            "median_when_absent": float(np.median(absent)) if absent else None,
            "median_when_strong": float(np.median(strong)) if strong else None,
            "status": "RECOVERABLE" if recovery_ordered else "NOT_RECOVERABLE",
        },
    }
    gate["reportable_layers"] = {
        "drift_direction": gate["drift_sign"]["status"] == "RECOVERABLE",
        "anisotropy": gate["anisotropy_ordering"]["status"] == "RECOVERABLE",
        "recovery_strength": gate["recovery_strength_ordering"]["status"] == "RECOVERABLE",
    }
    (OUT / "synthetic" / "RECOVERY_GATE.json").write_text(json.dumps(gate, indent=1))

    print("\n=== recovery gate ===")
    print(f"  drift sign        {sign_agreement:.3f} (floor 0.80)  "
          f"{gate['drift_sign']['status']}")
    print(f"  anisotropy order  rho={aniso_rho:+.3f} (floor 0.60)  "
          f"{gate['anisotropy_ordering']['status']}")
    print(f"  recovery order    absent {gate['recovery_strength_ordering']['median_when_absent']} "
          f"vs strong {gate['recovery_strength_ordering']['median_when_strong']}  "
          f"{gate['recovery_strength_ordering']['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
