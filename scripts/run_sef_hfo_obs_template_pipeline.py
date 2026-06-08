"""(b) Push MODEL events through the REAL propagation-template pipeline (2026-06-08).

Goal = the INTEGRATION milestone (advisor 2026-06-08): model output traverses the
real clustering pipeline with NO special-casing and lands in the real subject schema,
field-for-field comparable to a real patient — reported as plumbing + sanity, NOT as
"the model reproduces the patient mechanism". (With ONE imposed connectivity axis the
template space is ~1-D, so stable_k≈2 forward/reverse is FORCED, not discovered; the
genuine mechanism test is the downstream heterogeneity work.)

Each model event: rate field (the C1 recipe), seeded at the -end (forward) or +end
(reverse) of the connectivity axis, read through ONE fixed virtual montage (3 non-parallel
4 mm shafts, like the four-control), -> per-event rank vector + participation. Within-arm
variability is injected by per-event axis-angle + seed-position jitter (the deterministic
rate field has no intrinsic trial-to-trial noise).

Stack -> feed the REAL src.interictal_propagation.compute_adaptive_cluster_stereotypy
(the exact function run_interictal_propagation calls). Smoke-test on 10 events first
(advisor): >=10 participants/event, non-degenerate within-arm spread, cluster fn returns.

Run smoke:  PYTHONPATH="$PWD" python scripts/run_sef_hfo_obs_template_pipeline.py --smoke
Run full:   PYTHONPATH="$PWD" python scripts/run_sef_hfo_obs_template_pipeline.py --n-per-arm 60
"""
from __future__ import annotations

import os
import sys
import json
import argparse

import numpy as np

sys.path.insert(0, os.getcwd())
from src.sef_hfo_lif import mean_field, DETECT                              # noqa: E402
from src.sef_hfo_rate_adapter import rate_event_envelope                    # noqa: E402
from src.sef_hfo_observation import (build_shaft, merge_montages,           # noqa: E402
                                     extract_lagpat, attach_geometry)
from src.sef_hfo_snn_adapter import event_window_for_run                    # noqa: E402
from src.interictal_propagation import compute_adaptive_cluster_stereotypy  # noqa: E402
from scripts.run_sef_hfo_obs_increment3a import _integrate, RATIO, DT       # noqa: E402

OUT = "results/topic4_sef_hfo/observation_layer/template_pipeline"
L, NGRID, PITCH = 24.0, 96, 4.0
AXIS_DEG = 45.0
SHAFTS = (15.0, 75.0, 135.0)            # 3 non-parallel shafts (four-control montage)


def _montage():
    return merge_montages([build_shaft(np.deg2rad(a), PITCH, 6, (0.0, 0.0), chr(65 + i))
                           for i, a in enumerate(SHAFTS)])


def _one_event(op, montage, theta_deg, kick_xy, noise_sd=0.0, rng=None):
    frames, _off, on_ext, off_ext = _integrate(op, np.deg2rad(theta_deg), 2.0, kick_xy, NGRID, L)
    win = event_window_for_run(on_ext, off_ext, DT)
    if win is None:
        return None
    env = rate_event_envelope(frames, NGRID, L, montage, 0.5 * PITCH)
    if noise_sd > 0.0 and rng is not None:
        # observation noise on the recorded signal (the mean-field model is deterministic;
        # real SEEG is noisy) -> jitters per-contact first-crossings -> within-arm rank spread
        env = env + rng.normal(0.0, noise_sd * (float(env.max()) - float(env.min())), size=env.shape)
    floor = float(env.min()); margin = 0.10 * (float(env.max()) - floor)
    art = attach_geometry(extract_lagpat(env, DT, [win], floor, margin, 0.5, DT), montage)
    return art.ranks[:, 0], art.bools[:, 0]


def generate(n_per_arm, frac=0.5, jit_perp=0.18, noise_sd=0.015, seed=0):
    """Forward arm (kick at -end) + reverse arm (kick at +end). Within-arm variability
    from PERPENDICULAR seed jitter + per-event observation noise on the recorded signal.
    The connectivity axis is FIXED at 45 deg: only theta in {0,45,90} yields a clean event
    on the periodic grid (intermediate angles fail), so axis-angle jitter is NOT usable;
    along-axis jitter pushes the kick into the periodic boundary; the deterministic
    mean-field model has no intrinsic trial-to-trial rank-order spread, so it is injected
    as observation noise (real SEEG is noisy)."""
    rng = np.random.default_rng(seed)
    op = mean_field(RATIO)
    m = _montage()
    half = L / 2.0
    u = np.array([np.cos(np.deg2rad(AXIS_DEG)), np.sin(np.deg2rad(AXIS_DEG))])
    up = np.array([-u[1], u[0]])
    R, B, arms = [], [], []
    for arm, sgn in (("forward", -1.0), ("reverse", +1.0)):
        for _ in range(n_per_arm):
            kick = sgn * frac * half * u + rng.uniform(-jit_perp, jit_perp) * half * up
            res = _one_event(op, m, AXIS_DEG, kick, noise_sd=noise_sd, rng=rng)
            if res is None:
                continue
            R.append(res[0]); B.append(res[1]); arms.append(arm)
    ranks = np.array(R).T          # (n_channel, n_event)
    bools = np.array(B).T
    return ranks, bools, np.array(arms), list(m.names)


def _within_arm_spread(ranks, bools, arms):
    """Median pairwise rank disagreement within each arm (0 = identical = degenerate)."""
    out = {}
    for arm in ("forward", "reverse"):
        cols = np.flatnonzero(arms == arm)
        if len(cols) < 2:
            out[arm] = None; continue
        diffs = []
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                a, b = ranks[:, cols[i]], ranks[:, cols[j]]
                m = np.isfinite(a) & np.isfinite(b)
                if m.sum() >= 3:
                    diffs.append(float(np.mean(np.abs(a[m] - b[m]))))
        out[arm] = (round(float(np.median(diffs)), 3) if diffs else None)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="10-event smoke (5/arm)")
    ap.add_argument("--n-per-arm", type=int, default=60)
    ap.add_argument("--noise", type=float, default=0.015, help="observation-noise sd (frac of env range)")
    a = ap.parse_args()
    n_per_arm = 5 if a.smoke else a.n_per_arm

    print(f"[template-pipeline] generating {2 * n_per_arm} model events "
          f"({n_per_arm}/arm) on a {len(SHAFTS)}-shaft {PITCH}mm montage, noise={a.noise} ...",
          flush=True)
    ranks, bools, arms, names = generate(n_per_arm, noise_sd=a.noise)
    n_ev = ranks.shape[1]
    npart = bools.sum(axis=0)
    print(f"  generated {n_ev} events; participants/event: min={npart.min()} "
          f"median={int(np.median(npart))} max={npart.max()} (need >=10)")
    print(f"  within-arm rank spread (0=degenerate): {_within_arm_spread(ranks, bools, arms)}")

    print("[template-pipeline] running REAL compute_adaptive_cluster_stereotypy ...", flush=True)
    res = compute_adaptive_cluster_stereotypy(ranks, bools, names, use_masked_features=True)
    print(f"  stable_k = {res.get('stable_k')}  (chosen_reason={res.get('chosen_reason')})")
    fr = res.get("candidate_forward_reverse_pairs")
    print(f"  candidate forward/reverse pairs = {fr}")

    if not a.smoke:
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "model_subject_adaptive_cluster.json"), "w") as f:
            json.dump({"dataset": "sef_hfo_model", "subject": "rate_field_45deg",
                       "n_channels": len(names), "n_events_total": int(n_ev),
                       "channel_names": names,
                       "arm_counts": {arm: int((arms == arm).sum()) for arm in ("forward", "reverse")},
                       "adaptive_cluster": res}, f, indent=2, default=lambda o: None)
        print("  wrote", os.path.join(OUT, "model_subject_adaptive_cluster.json"))


if __name__ == "__main__":
    main()
