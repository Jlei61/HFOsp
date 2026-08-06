"""Why a parameter fails to recover: is the data flat in it, or did the fit miss?

A bare NOT_RECOVERABLE cannot tell those apart, and they mean opposite things --
one is a property of the observation, the other is a defect in my optimiser. So
every other parameter is pinned at the value that generated the data and one is
swept. If the loss barely moves, the data cannot constrain it. If it has a clear
minimum that the fit did not find, the fit is the problem.

Run against the same generator the recovery gate uses, so the two are comparable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topic5_spatial_latent_rnn import build_event_tensors, next_set_stop_loss  # noqa: E402
from src.topic5_spatial_propagation_operator import (  # noqa: E402
    D_MAX, OperatorConfig, V_MAX, build_grid,
)
from src.topic5_virtual_seeg_operator import build_observation_operator  # noqa: E402
import scripts.run_topic5_spo_recovery as gate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic5_spatial_propagation_operator_v0_2"

TRUTH = {"name": "profile", "v": +0.35, "D_par": 0.08, "D_perp": 0.02, "beta": 0.2}


@torch.no_grad()
def loss_at(base: dict, cell: dict, tensors) -> float:
    model = gate.make_generator(
        OperatorConfig(variant="ANISOTROPIC_RECOVERY", seed=1, **base), cell, 1
    )
    logits, stop = model(tensors.x, tensors.recruited, tensors.valid)
    value, _, _ = next_set_stop_loss(logits, stop, tensors.target, tensors.available,
                                     tensors.valid, tensors.is_last)
    return float(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="epilepsiae_1146")
    parser.add_argument("--n-events", type=int, default=600)
    parser.add_argument("--microsteps", type=int, nargs="+", default=[3, 6])
    args = parser.parse_args()

    plane = np.load(gate.V1_CACHE / args.subject / "plane_coordinates.npz",
                    allow_pickle=True)
    contacts, sigma = plane["xy_mm"], float(plane["sigma_mm"][0])
    reference = np.load(gate.V1_CACHE / args.subject / "events.npz")["group_ids"]
    lengths = np.array([r[r >= 0].max() + 1 for r in reference if (r >= 0).any()])
    centres, shape, mask = build_grid(contacts, sigma)
    H = build_observation_operator(contacts, centres, sigma)

    report: dict = {
        "contract": "topic5_spo_profile_likelihood_v0_2",
        "question": ("does the loss depend on this parameter at all, and does it "
                     "depend on its sign as well as its size"),
        "subject_geometry": args.subject,
        "generating_truth": TRUTH,
        "profiles": {},
    }

    for microsteps in args.microsteps:
        base = dict(n_contacts=len(contacts), grid_shape=shape, microsteps=microsteps,
                    observation_operator=H, grid_mask=mask)
        events = gate.generate_events(
            gate.make_generator(
                OperatorConfig(variant="ANISOTROPIC_RECOVERY", seed=1, **base), TRUTH, 1
            ), args.n_events, lengths, 1
        )
        tensors = build_event_tensors(events)

        drift_grid = np.linspace(-0.9 * V_MAX, 0.9 * V_MAX, 13)
        drift_loss = np.array([loss_at(base, dict(TRUTH, v=float(v)), tensors)
                               for v in drift_grid])
        # The two things a sign-blind observation looks like: a big change with
        # |v| and almost none between +v and -v.
        magnitude_span = float(drift_loss.max() - drift_loss.min())
        pairs = [(drift_loss[i], drift_loss[-1 - i]) for i in range(len(drift_grid) // 2)]
        sign_span = float(np.mean([abs(a - b) for a, b in pairs]))

        aniso_grid = np.linspace(0.05 * D_MAX, 0.95 * D_MAX, 9)
        aniso_loss = np.array([loss_at(base, dict(TRUTH, D_par=float(d)), tensors)
                               for d in aniso_grid])

        beta_grid = np.array([0.0, 0.1, 0.2, 0.4, 0.8, 1.6])
        beta_loss = np.array([loss_at(base, dict(TRUTH, beta=float(b)), tensors)
                              for b in beta_grid])

        report["profiles"][str(microsteps)] = {
            "drift": {
                "grid": drift_grid.tolist(), "loss": drift_loss.tolist(),
                "span_over_magnitude": magnitude_span,
                "mean_gap_between_mirrored_signs": sign_span,
                "sign_share_of_signal": float(sign_span / max(magnitude_span, 1e-12)),
                "argmin": float(drift_grid[drift_loss.argmin()]),
                "reading": (
                    "the loss moves with how far activity travels but barely with "
                    "which way: sign carries "
                    f"{sign_span / max(magnitude_span, 1e-12):.1%} of the signal that "
                    "magnitude does"
                    if sign_span < 0.25 * magnitude_span else
                    "the loss separates the two directions"
                ),
            },
            "axial_diffusion": {
                "grid": aniso_grid.tolist(), "loss": aniso_loss.tolist(),
                "span": float(aniso_loss.max() - aniso_loss.min()),
                "argmin": float(aniso_grid[aniso_loss.argmin()]),
                "truth": TRUTH["D_par"],
            },
            "recovery_strength": {
                "grid": beta_grid.tolist(), "loss": beta_loss.tolist(),
                "span": float(beta_loss.max() - beta_loss.min()),
                "argmin": float(beta_grid[beta_loss.argmin()]),
                "truth": TRUTH["beta"],
                # Unlike drift, changing beta changes the overall amplitude of
                # the field, which shifts calibration as well as fit. The span
                # is therefore meaningful -- the loss does depend strongly on
                # this parameter -- but the location of the minimum is not an
                # estimate of the truth, and must not be read as one. The
                # ordering test in the gate, which refits and compares across
                # generators, is the evidence for this layer.
                "argmin_is_not_an_estimate": True,
                "caveat": (
                    "beta scales the field as well as fitting it, so the minimum "
                    "of this profile is confounded with calibration; use the "
                    "gate's refit-and-compare ordering instead"
                ),
            },
        }
        d = report["profiles"][str(microsteps)]
        print(f"\n=== {microsteps} internal steps ===")
        print(f"  drift      span {d['drift']['span_over_magnitude']:.4f} over |v|, "
              f"{d['drift']['mean_gap_between_mirrored_signs']:.4f} between mirrored "
              f"signs  ({d['drift']['sign_share_of_signal']:.1%} of it)")
        print(f"             {d['drift']['reading']}")
        print(f"  axial diffusion  span {d['axial_diffusion']['span']:.4f}, "
              f"argmin {d['axial_diffusion']['argmin']:.3f} against truth "
              f"{TRUTH['D_par']}")
        print(f"  recovery         span {d['recovery_strength']['span']:.4f}, "
              f"argmin {d['recovery_strength']['argmin']:.3f} against truth "
              f"{TRUTH['beta']}")

    (OUT / "synthetic").mkdir(parents=True, exist_ok=True)
    (OUT / "synthetic" / "PROFILE_LIKELIHOOD.json").write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT / 'synthetic' / 'PROFILE_LIKELIHOOD.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
