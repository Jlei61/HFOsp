#!/usr/bin/env python3
"""H3: does past IED exposure still add anything once the state is controlled?

The probe is deliberately linear and runs on **frozen** states, so nothing here
can rescue a state that does not carry the information.  For every scale K it
compares six arms on held-out test time:

``no_edge``                 state only
``real_exposure``           state + the excess of the last K events over what the
                            model itself predicted for them
``intercept_matched``       state + a surrogate column with the same variance and
                            lag-1 autocorrelation as the exposure but no event
                            content.  This is the control that catches the
                            "a saturated event jump becomes a free intercept"
                            failure this project hit on 2026-08-26.
``delayed_exposure``        exposure from a window ending K events earlier
                            (no overlap with the recent window)
``state_matched_placebo``   exposure taken from the event whose state is closest
                            but which is at least 2K events away
``current_event_only``      exposure from the single previous event

Independent-window count (``n_test / K``) is reported next to every result so a
sliding-window count can never be mistaken for a sample size.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_group_event_state.source_audit import write_json_atomic  # noqa: E402

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1 = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1"
SCALES = (100, 1000, 5000, 10000)


def _rolling_mean(x: np.ndarray, k: int) -> np.ndarray:
    """Mean of the k values strictly before each position (NaN before that)."""

    out = np.full(x.size, np.nan)
    if x.size <= k:
        return out
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    out[k:] = (csum[k:-1] - csum[:-k-1]) / k
    return out


def _ar1_surrogate(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same variance and lag-1 autocorrelation, no event content."""

    finite = x[np.isfinite(x)]
    if finite.size < 3:
        return np.zeros_like(x)
    centred = finite - finite.mean()
    denom = float(np.sum(centred[:-1] ** 2))
    phi = float(np.clip(np.sum(centred[:-1] * centred[1:]) / denom, -0.99, 0.99)) if denom > 0 else 0.0
    sigma = float(np.std(finite)) * np.sqrt(max(1.0 - phi**2, 1e-6))
    out = np.zeros(x.size)
    noise = rng.normal(0.0, sigma, x.size)
    for i in range(1, x.size):
        out[i] = phi * out[i - 1] + noise[i]
    return out + float(np.mean(finite))


def _ridge_blocked_r2(X: np.ndarray, y: np.ndarray, n_folds: int = 5, alpha: float = 1.0) -> float:
    """Blocked (contiguous-fold) out-of-sample R^2; blocks respect time order."""

    n = y.size
    if n < n_folds * 10:
        return float("nan")
    edges = np.linspace(0, n, n_folds + 1).astype(int)
    preds = np.full(n, np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        test = np.zeros(n, bool)
        test[a:b] = True
        train = ~test
        Xt, yt = X[train], y[train]
        mu, sd = Xt.mean(0), Xt.std(0)
        sd = np.where(sd > 1e-9, sd, 1.0)
        Xt = (Xt - mu) / sd
        Xe = (X[test] - mu) / sd
        gram = Xt.T @ Xt + alpha * np.eye(Xt.shape[1])
        w = np.linalg.solve(gram, Xt.T @ (yt - yt.mean()))
        preds[test] = Xe @ w + yt.mean()
    ss_res = float(np.sum((y - preds) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def analyse_run(run_dir: Path, scales=SCALES, seed: int = 0) -> dict | None:
    states_path = run_dir / "test_states.npy"
    series_path = run_dir / "test_series.npz"
    result_path = run_dir / "result.json"
    if not (states_path.exists() and series_path.exists() and result_path.exists()):
        return None
    result = json.loads(result_path.read_text())
    z = np.load(states_path).astype(np.float64)
    series = dict(np.load(series_path))
    n = z.shape[0]
    if series["size_true"].size != n:
        return None
    rng = np.random.default_rng(seed)

    # "excess" = the part of each observed event the model did not expect
    size_excess = series["size_true"] - series["size_pred"]
    dt = series["dt_prev"].astype(np.float64)
    with np.errstate(all="ignore"):
        timing_excess = np.log(np.clip(dt, 1e-3, None)) - series["timing_mu"]
    timing_excess[~np.isfinite(dt)] = 0.0

    # target: the next event's size, i.e. one step ahead of every predictor
    y = series["size_true"][1:]
    z_pred = z[:-1]
    out: dict = {
        "subject": result["subject"],
        "arm": result["arm"],
        "seed": result["seed"],
        "n_test_events": int(n),
        "state_dim": int(z.shape[1]),
        "scales": {},
    }
    for k in scales:
        if n <= 3 * k:
            out["scales"][str(k)] = {"status": "insufficient_test_events", "n_test_events": int(n)}
            continue
        exposure = np.stack(
            [_rolling_mean(size_excess, k), _rolling_mean(timing_excess, k)], axis=1
        )[:-1]
        delayed = np.stack(
            [
                _rolling_mean(np.concatenate([np.zeros(k), size_excess[:-k]]), k),
                _rolling_mean(np.concatenate([np.zeros(k), timing_excess[:-k]]), k),
            ],
            axis=1,
        )[:-1]
        surrogate = np.stack(
            [_ar1_surrogate(exposure[:, 0], rng), _ar1_surrogate(exposure[:, 1], rng)], axis=1
        )
        current = np.stack([size_excess[:-1], timing_excess[:-1]], axis=1)

        # State-matched placebo: the exposure of the most state-similar event that
        # is at least 2k events away.  Scanning every candidate for every row is
        # O(n^2) and does not finish on a 47k-event test split, so the match is
        # made against a fixed random pool of candidates.
        m = exposure.shape[0]
        placebo = np.zeros_like(exposure)
        if m:
            zc = (z_pred - z_pred.mean(0)) / (z_pred.std(0) + 1e-9)
            pool_rng = np.random.default_rng(seed + 4242)
            pool = pool_rng.choice(m, size=min(m, 1024), replace=False)
            pool = pool[np.isfinite(exposure[pool]).all(1)]
            if pool.size:
                d2 = (
                    np.einsum("ij,ij->i", zc, zc)[:, None]
                    - 2.0 * zc @ zc[pool].T
                    + np.einsum("ij,ij->i", zc[pool], zc[pool])[None, :]
                )
                too_close = np.abs(np.arange(m)[:, None] - pool[None, :]) < 2 * k
                d2 = np.where(too_close, np.inf, d2)
                best = np.argmin(d2, axis=1)
                usable = np.isfinite(d2[np.arange(m), best])
                placebo[usable] = exposure[pool[best[usable]]]

        valid = np.isfinite(exposure).all(1) & np.isfinite(delayed).all(1) & np.isfinite(y)
        if valid.sum() < 200:
            out["scales"][str(k)] = {"status": "insufficient_valid_rows", "n_valid": int(valid.sum())}
            continue
        base = z_pred[valid]
        yy = y[valid]
        arms = {
            "no_edge": base,
            "real_exposure": np.hstack([base, exposure[valid]]),
            "intercept_matched": np.hstack([base, surrogate[valid]]),
            "delayed_exposure": np.hstack([base, delayed[valid]]),
            "state_matched_placebo": np.hstack([base, placebo[valid]]),
            "current_event_only": np.hstack([base, current[valid]]),
        }
        scores = {name: _ridge_blocked_r2(X, yy) for name, X in arms.items()}
        out["scales"][str(k)] = {
            "status": "ok",
            "n_valid_rows": int(valid.sum()),
            "n_independent_windows": float(valid.sum() / k),
            "r2": scores,
            "gain_over_no_edge": {
                name: (scores[name] - scores["no_edge"]) for name in arms if name != "no_edge"
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=V0_1 / "runs")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--arms", nargs="+", default=["a4_full_multimodal_state"])
    parser.add_argument("--out", type=Path, default=V0_1 / "h3_exposure.json")
    args = parser.parse_args()

    records = []
    for run_dir in sorted((args.runs_root / args.tag).glob("*")):
        if not run_dir.is_dir():
            continue
        if not any(f"__{a}__" in run_dir.name for a in args.arms):
            continue
        rec = analyse_run(run_dir)
        if rec:
            records.append(rec)
            print(f"{rec['subject']} {rec['arm']} seed{rec['seed']}: "
                  + " ".join(f"K={k}:{v.get('status','')}" for k, v in rec["scales"].items()), flush=True)

    # patient-first: median over seeds within patient, then across patients
    summary: dict = {"n_runs": len(records), "scales": {}}
    for k in SCALES:
        per_patient: dict[str, list[dict]] = {}
        for rec in records:
            entry = rec["scales"].get(str(k), {})
            if entry.get("status") == "ok":
                per_patient.setdefault(rec["subject"], []).append(entry)
        if not per_patient:
            summary["scales"][str(k)] = {"n_patients": 0}
            continue
        names = ["real_exposure", "intercept_matched", "delayed_exposure",
                 "state_matched_placebo", "current_event_only"]
        gains = {
            name: np.array([
                float(np.median([e["gain_over_no_edge"][name] for e in v]))
                for v in per_patient.values()
            ])
            for name in names
        }
        summary["scales"][str(k)] = {
            "n_patients": len(per_patient),
            "median_independent_windows": float(np.median([
                np.median([e["n_independent_windows"] for e in v]) for v in per_patient.values()
            ])),
            "median_gain": {n: float(np.median(g)) for n, g in gains.items()},
            "n_patients_positive": {n: int((g > 0).sum()) for n, g in gains.items()},
            "real_minus_intercept_matched": {
                "median": float(np.median(gains["real_exposure"] - gains["intercept_matched"])),
                "n_positive": int((gains["real_exposure"] > gains["intercept_matched"]).sum()),
                "n_patients": int(gains["real_exposure"].size),
            },
        }
    write_json_atomic({"summary": summary, "runs": records}, args.out)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
