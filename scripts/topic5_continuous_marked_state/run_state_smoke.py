#!/usr/bin/env python3
"""Toy recovery plus a real-feature forward smoke for T1/T2 primitives."""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from src.topic5_continuous_marked_state import contract
from src.topic5_continuous_marked_state.bridge import BridgeArrays
from src.topic5_continuous_marked_state.state import T1T2Core, correction_off_rollout


def _exposure(innovation: np.ndarray, dt: np.ndarray, tau: float) -> np.ndarray:
    u = 0.0
    out = np.zeros_like(innovation, dtype=np.float64)
    for i in range(len(innovation)):
        u *= np.exp(-float(dt[i]) / tau)
        out[i] = u
        u += float(innovation[i])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", default="yuquan_huanghanwen", choices=contract.PILOT_SUBJECTS)
    args = parser.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # Instrument sensitivity: real causal exposure, not a shifted placebo,
    # must recover a synthetic exposure-driven target.
    n = 2000
    dt = rng.exponential(2.0, n)
    eta = rng.normal(size=n)
    real_u = _exposure(eta, dt, 60.0)
    shifted_u = _exposure(np.roll(eta, 137), dt, 60.0)
    target = 0.8 * real_u + rng.normal(scale=0.3, size=n)
    cut = 1400
    def fit_score(feature: np.ndarray) -> float:
        x = np.stack([np.ones(cut), feature[:cut]], axis=1)
        beta = np.linalg.lstsq(x, target[:cut], rcond=None)[0]
        pred = beta[0] + beta[1] * feature[cut:]
        return float(np.mean((target[cut:] - pred) ** 2))
    real_mse = fit_score(real_u)
    placebo_mse = fit_score(shifted_u)

    arrays = BridgeArrays.load(
        contract.RESULT_ROOT / "bridge/features" / f"{args.subject}.npz"
    )
    obs = torch.as_tensor(arrays.spectral[:32, :8], dtype=torch.float32)
    core = T1T2Core(observation_dim=8, state_dim=4, t2=True)
    dt_min = (arrays.next_time[:32] - arrays.current_time[:32]) / 60.0
    innovations = (arrays.stop_fraction[:32] - arrays.stop_fraction[:32].mean()).tolist()
    z0 = torch.zeros(4)
    first = correction_off_rollout(
        core, z0, list(obs), dt_min.tolist(), innovations, 60.0, anchor_index=7
    )
    changed = list(obs.clone())
    for i in range(8, len(changed)):
        changed[i] = changed[i] + 100.0
    second = correction_off_rollout(
        core, z0, changed, dt_min.tolist(), innovations, 60.0, anchor_index=7
    )
    future_invariant = bool(torch.equal(first[8:], second[8:]))
    eigen = torch.linalg.eigvals(core.generator.matrix()).real.detach().numpy()
    output = {
        "contract": contract.REVISION,
        "subject": args.subject,
        "synthetic_real_exposure_mse": real_mse,
        "synthetic_placebo_exposure_mse": placebo_mse,
        "synthetic_real_better": bool(real_mse < placebo_mse),
        "correction_off_future_observation_invariant": future_invariant,
        "max_generator_eigen_real": float(eigen.max()),
        "real_forward_all_finite": bool(torch.isfinite(first).all()),
        "n_real_events_smoked": int(len(first)),
        "sealed_opened": False,
        "claim_boundary": "instrument and forward smoke only; not an H1/H3 result",
    }
    path = contract.RESULT_ROOT / "state_smoke/T1_T2_SMOKE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(output, indent=2, sort_keys=True))
    os.replace(tmp, path)
    print(json.dumps(output, sort_keys=True))


if __name__ == "__main__":
    main()
