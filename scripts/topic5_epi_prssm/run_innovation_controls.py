#!/usr/bin/env python3
"""Goal 4 Task 4.5 -- the H3a innovation and directionality leg.

Expected load is built from the *frozen T1* state plus nuisances with a blocked,
future-blind cross-fit; the innovation is what the observed load did beyond that
expectation.  A T2 state is never used to residualise T2's own load, and the
outcome is a non-load endpoint, so a positive result cannot be a restatement of
"more contacts participated".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from _common import (  # noqa: E402
    FROZEN, JobKey, JobRunner, OUTPUT_ROOT, atomic_write_csv, atomic_write_json,
    code_revision, is_complete, load_tensors, package_hash, resolve_cohort,
    sha256_obj, torch,
)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.topic5_epi_prssm.evaluate import probe_summary  # noqa: E402
from src.topic5_epi_prssm.model import EpiPRSSM, build_cohort_batch  # noqa: E402
from src.topic5_epi_prssm.rollout import cohort_scan  # noqa: E402
from src.topic5_epi_prssm.stats import paired_effect  # noqa: E402

GOAL = "goal4_innovation"
OUT = OUTPUT_ROOT / "exposure_mechanism"
EXPOSURE_TAUS = FROZEN["exposure_tau_primary_seconds"]
#: the pre-registered primary grid stays the reporting default; a wider sweep is
#: exploratory and must be labelled as such in whatever consumes the summary.

OUTCOME_HORIZON = 20
N_FOLDS = 5
EMBARGO = 200
RATE_WINDOW = 1800.0


TAU_FREEZE = OUTPUT_ROOT / "manifests/RESOURCE_TAU_FREEZE.json"


def frozen_t1_arm() -> str:
    """The arm named by the resource-tau freeze -- never a post-hoc best pick."""
    if not TAU_FREEZE.exists():
        raise SystemExit(f"{TAU_FREEZE} missing: freeze tau_r before running the "
                         "innovation controls, otherwise the checkpoint is chosen post hoc")
    tau = float(json.loads(TAU_FREEZE.read_text())["tau_r_seconds"])
    return f"t1_r1_tau{int(tau)}"


def find_t1_checkpoint(cohort: str, arm: str | None = None) -> dict:
    """Return the completed run for the frozen arm, choosing among seeds only.

    Selecting the best-scoring arm across the tau grid would re-open the choice the
    freeze exists to close, and did: an earlier run scored tau=60 s while the freeze
    declared 7200 s.
    """
    wanted = arm or frozen_t1_arm()
    best = None
    for path in sorted((OUT / "runs").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("cohort") != cohort or record.get("arm") != wanted:
            continue
        if record.get("evaluation") is None:
            continue
        score = float(np.mean([v["event_nll"] + v["participation_nll"]
                               for v in record["evaluation"]["filtered"].values()]))
        if best is None or score < best[0]:
            best = (score, record)
    if best is None:
        raise SystemExit(f"no completed run for the frozen arm {wanted!r} (cohort {cohort}); "
                         "the innovation controls must not fall back to another tau")
    return best[1]


@torch.no_grad()
def per_event_features(model: EpiPRSSM, patient, chunk: int = 512) -> pd.DataFrame:
    """Causal pre-event features for the whole stream, from the frozen T1 model."""
    batch = build_cohort_batch([patient], [0], [patient.n_events])
    z = model.initial_state(batch)
    rows: list[dict[str, np.ndarray]] = []
    position = 0
    while position < patient.n_events:
        end = min(position + chunk, patient.n_events)
        result = cohort_scan(model, batch, position, end, z, correction_on=True)
        take = end - position
        state = result.state_minus[:take, 0, : patient.n_contacts, :]
        resource = result.resource_minus[:take, 0]
        summary = probe_summary(model, patient, state, resource)
        scores = model.score_events(patient, torch.arange(position, end), state, resource)
        rows.append({**summary,
                     "order_nll": scores["order_nll"].cpu().numpy(),
                     "event_nll": scores["event_nll"].cpu().numpy()})
        z = result.final
        position = end
    merged = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    event_time = patient.event_time
    delta = patient.delta_t.cpu().numpy()
    lo = np.searchsorted(event_time, event_time - RATE_WINDOW, side="left")
    hi = np.arange(len(event_time))
    merged["log_iei"] = np.log1p(delta)
    merged["local_rate"] = (hi - lo) / (RATE_WINDOW / 3600.0)
    hour = ((event_time + (1.0 if patient.dataset == "epilepsiae" else 8.0) * 3600.0)
            / 3600.0) % 24.0
    merged["tod_sin"] = np.sin(2 * np.pi * hour / 24.0)
    merged["tod_cos"] = np.cos(2 * np.pi * hour / 24.0)
    merged["session"] = patient.meta["session_index"]
    merged["load"] = patient.load.cpu().numpy()
    merged["split"] = patient.split.cpu().numpy()
    merged["event_time"] = event_time
    merged["delta_t"] = delta
    return pd.DataFrame(merged)


def blocked_cross_fit(frame: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Expanding, future-blind folds with an embargo between fit and prediction."""
    X = frame[feature_names].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(X)), X])
    y = frame["load"].to_numpy(dtype=float)
    predicted = np.full(len(y), np.nan)
    n = len(y)
    edges = np.linspace(0, n, N_FOLDS + 1).astype(int)
    for fold in range(1, N_FOLDS + 1):
        fit_stop = max(edges[fold - 1] - EMBARGO, 0)
        predict_lo, predict_hi = edges[fold - 1], edges[fold]
        if fit_stop < 50:
            # never average the block being predicted: y[:predict_hi] contains it.
            # With too little history to fit, the fold carries no prediction at all.
            predicted[predict_lo:predict_hi] = (y[:fit_stop].mean() if fit_stop > 0 else np.nan)
            continue
        A = X[:fit_stop]
        coefficients = np.linalg.lstsq(A.T @ A + 1e-3 * np.eye(A.shape[1]),
                                       A.T @ y[:fit_stop], rcond=None)[0]
        predicted[predict_lo:predict_hi] = X[predict_lo:predict_hi] @ coefficients
    return predicted


def causal_kernel(values: np.ndarray, delta_t: np.ndarray, tau: float,
                  kind: str = "clock") -> np.ndarray:
    out = np.zeros(len(values))
    carry = 0.0
    for i in range(len(values)):
        carry = carry * (np.exp(-delta_t[i] / tau) if kind == "clock" else np.exp(-1.0 / tau))
        out[i] = carry
        carry = carry + values[i]
    return out


def forward_mean(values: np.ndarray, horizon: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    for i in range(len(values)):
        stop = min(i + 1 + horizon, len(values))
        if stop > i + 1:
            out[i] = (cumulative[stop] - cumulative[i + 1]) / (stop - i - 1)
    return out


def residualise(y: np.ndarray, frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    keep = np.isfinite(y)
    design = np.column_stack([np.ones(len(frame)), frame[names].to_numpy(dtype=float)])
    coefficients, *_ = np.linalg.lstsq(design[keep], y[keep], rcond=None)
    residual = np.full(len(y), np.nan)
    residual[keep] = y[keep] - design[keep] @ coefficients
    return residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="all34")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--taus", default="", help="comma-separated exploratory tau sweep "
                        "in seconds; the pre-registered primary grid is the default")
    parser.add_argument("--outcome-horizon", type=int, default=None)
    parser.add_argument("--tag", default="", help="suffix for the output file, so an "
                        "exploratory sweep never overwrites the primary summary")
    args = parser.parse_args()

    global EXPOSURE_TAUS, OUTCOME_HORIZON
    if args.taus:
        EXPOSURE_TAUS = tuple(float(t) for t in args.taus.split(","))
    if args.outcome_horizon:
        OUTCOME_HORIZON = int(args.outcome_horizon)

    record = find_t1_checkpoint(args.cohort)
    checkpoint = OUT / "checkpoints" / f"{record['job_id']}.pt"
    key = JobKey(goal=GOAL, family=record["arm"], arm="innovation", seed=int(record["seed"]),
                 split="development", cohort=args.cohort,
                 config_hash=sha256_obj({"taus": list(EXPOSURE_TAUS),
                                         "horizon": OUTCOME_HORIZON,
                                         "tag": args.tag})[:16],
                 input_hash=sha256_obj({"job": record["job_id"]})[:16],
                 code_revision=package_hash()[:16])
    suffix = f"__{args.tag}" if args.tag else ""
    target = OUT / f"innovation_controls_summary{suffix}.json"
    if target.exists() and is_complete(key) and not args.overwrite:
        print("SKIPPED_EXISTING")
        return

    with JobRunner(key) as job:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        patients = load_tensors(resolve_cohort(args.cohort))
        model = EpiPRSSM(feature_dim=payload.get("feature_dim",
                                                 patients[0].node_features.shape[-1]),
                         **payload["spec"])
        model.load_state_dict(payload["state_dict"])
        model.eval()

        nuisance = ["log_iei", "local_rate", "tod_sin", "tod_cos"]
        state_features = ["state_norm", "expected_load", "first_selection_entropy", "resource"]
        rows = []
        rng = np.random.default_rng(FROZEN["bootstrap_seed"])
        for patient in patients:
            frame = per_event_features(model, patient)
            development = frame[frame["split"] <= 1].reset_index(drop=True)
            if len(development) < 500:
                rows.append({"subject": patient.subject, "status": "too_few_events",
                             "n_events": len(development)})
                continue
            predicted = blocked_cross_fit(development, state_features + nuisance)
            innovation = development["load"].to_numpy() - predicted
            innovation = np.nan_to_num(innovation)
            outcome = forward_mean(development["order_nll"].to_numpy(), OUTCOME_HORIZON)
            outcome = residualise(outcome, development, nuisance)
            delta = development["delta_t"].to_numpy()
            sessions = development["session"].to_numpy()
            for tau in EXPOSURE_TAUS:
                exposure = causal_kernel(innovation, delta, tau, "clock")
                row = {"subject": patient.subject, "dataset": patient.dataset,
                       "status": "ok", "tau_seconds": float(tau),
                       "n_events": int(len(development)),
                       "expected_load_r2": float(1 - np.nanvar(innovation)
                                                 / max(np.nanvar(development["load"]), 1e-12))}
                row["real"] = _spearman(exposure, outcome)
                # --- controls ---------------------------------------------
                bins = np.minimum((np.argsort(np.argsort(development["state_norm"].to_numpy()))
                                   * 10) // max(len(development), 1), 9)
                matched = innovation.copy()
                for b in range(10):
                    members = np.flatnonzero(bins == b)
                    if len(members) > 1:
                        matched[members] = innovation[members][rng.permutation(len(members))]
                row["state_matched_shuffle"] = _spearman(
                    causal_kernel(matched, delta, tau, "clock"), outcome)
                row["time_reversal"] = _spearman(
                    causal_kernel(innovation[::-1], delta[::-1], tau, "clock")[::-1], outcome)
                row["event_count_kernel"] = _spearman(
                    causal_kernel(innovation, delta, 20.0, "event_count"), outcome)
                block = innovation.copy()
                order = rng.permutation(np.unique(sessions))
                offset = 0
                for session in order:
                    members = np.flatnonzero(sessions == session)
                    block[offset:offset + len(members)] = innovation[members]
                    offset += len(members)
                row["session_block_shuffle"] = _spearman(
                    causal_kernel(block, delta, tau, "clock"), outcome)
                row["raw_load_kernel"] = _spearman(
                    causal_kernel(development["load"].to_numpy(), delta, tau, "clock"), outcome)
                rows.append(row)
        frame = pd.DataFrame(rows)
        atomic_write_csv(OUT / f"innovation_controls{suffix}.csv", frame)
        summary = _summarise(frame, record)
        atomic_write_json(target, summary)
        job.outputs = {"summary": str(target),
                       "csv": str(OUT / f"innovation_controls{suffix}.csv")}
        job.metrics = {"n_patients": int(frame[frame.status == "ok"].subject.nunique())
                       if len(frame) else 0}
    print(json.dumps(summary["by_tau"], indent=2)[:1200])


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 50 or np.std(x[keep]) < 1e-12:
        return float("nan")
    return float(stats.spearmanr(x[keep], y[keep]).statistic)


def _summarise(frame: pd.DataFrame, record: dict) -> dict:
    ok = frame[frame.status == "ok"] if "status" in frame else frame
    by_tau = {}
    for tau, group in ok.groupby("tau_seconds"):
        block = {"n_patients": int(group.subject.nunique())}
        real = group.set_index("subject")["real"].to_dict()
        for control in ("state_matched_shuffle", "time_reversal", "event_count_kernel",
                        "session_block_shuffle", "raw_load_kernel"):
            other = group.set_index("subject")[control].to_dict()
            effect = paired_effect(real, other, label=f"tau{tau}::real-vs-{control}",
                                   lower_is_better=False)
            block[f"real_minus_{control}"] = effect.as_dict()
        zeros = {s: 0.0 for s in real}
        block["real_vs_zero"] = paired_effect(real, zeros, label=f"tau{tau}::real-vs-zero",
                                              lower_is_better=False).as_dict()
        by_tau[float(tau)] = block
    return {
        "contract": "topic5_epi_prssm_v0_1_innovation_controls",
        "frozen_t1_job_id": record["job_id"], "frozen_t1_arm": record["arm"],
        "outcome": f"mean masked-order NLL over the next {OUTCOME_HORIZON} events, "
                   "residualised on IEI, local rate and time of day",
        "expected_load_model": "frozen T1 state summary + IEI + local rate + time of day, "
                               f"blocked expanding cross-fit with a {EMBARGO}-event embargo",
        "by_tau": by_tau,
        "claim_boundary": [
            "a raw-load effect that does not survive the innovation challenge may only be "
            "called a history-dependent predictor, never event-driven shaping",
        ],
        "code_revision": code_revision(), "package_hash": package_hash(),
    }


if __name__ == "__main__":
    main()
