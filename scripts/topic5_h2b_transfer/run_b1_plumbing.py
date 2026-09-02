#!/usr/bin/env python3
"""B1 -- plumbing_only: exercise the survival pipeline end to end.

This runs the *whole* B1 loop (risk rows -> frozen state at anchors -> nested
baseline vs baseline+state -> censored scoring on held-out episodes) using a
**v0.1** trajectory as a stand-in state.

    The v0.1 arms are NOT the v0.2 producers. Every output is tagged
    ``plumbing_only`` and MUST NOT be reported as a v0.2 human result, in either
    direction. Its only job is to prove the rows, leads, censoring and schema
    are wired correctly before a real producer exists.

Fitting follows the rolling origin: rows before the first held-out episode fit
the model, rows from there on are evaluated, and the state arm is the baseline
arm plus frozen state features -- same rows, nested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.topic5_h2b_transfer.attach import attach_state_to_anchors  # noqa: E402
from src.topic5_h2b_transfer.risk_grid import (  # noqa: E402
    DEFAULT_POSTICTAL_EXCLUSION_SECONDS,
    HORIZON_EDGES_SECONDS,
    group_seizure_episodes,
)
from src.topic5_h2b_transfer.registry import read_registry, resolve_subject_arms  # noqa: E402
from src.topic5_h2b_transfer.scoring import (  # noqa: E402
    brier_by_bin,
    discrete_time_log_score,
    nested_increment,
)

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1_RUNS = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1/runs/main"
DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b"
DEFAULT_DATA = Path("/data/hfosp_group_event_state_v0_2/agent_b")
N_BINS = len(HORIZON_EDGES_SECONDS)
MAX_STATE_AGE_SEC = 3600.0
STATE_COMPONENTS = 8


def load_risk_rows(path: Path):
    rows = []
    for r in csv.DictReader(path.open()):
        rows.append({
            "anchor_epoch": float(r["anchor_epoch"]),
            "outcome_bin": (int(r["outcome_bin"]) if r["outcome_bin"] != "" else None),
            "censored": r["censored"] == "1",
            "next_seizure_id": r["next_seizure_id"],
            "last_observed_bin": int(r["last_observed_bin"]),
            "time_since_prev_seizure_sec": (float(r["time_since_prev_seizure_sec"])
                                            if r["time_since_prev_seizure_sec"] != "" else None),
            "seconds_into_state_span": float(r["seconds_into_state_span"]),
        })
    rows.sort(key=lambda r: r["anchor_epoch"])
    return rows


def baseline_features(rows):
    """clock / session position / coverage / time since previous seizure."""
    t = np.array([r["anchor_epoch"] for r in rows])
    hour = (t / 3600.0) % 24.0
    since = np.array([r["time_since_prev_seizure_sec"] if r["time_since_prev_seizure_sec"] is not None
                      else np.nan for r in rows])
    has_prev = np.isfinite(since).astype(float)
    log_since = np.where(np.isfinite(since), np.log1p(np.clip(since, 0, None)), 0.0)
    postictal = np.where(np.isfinite(since) & (since < 2 * 3600.0), 1.0, 0.0)
    into = np.array([r["seconds_into_state_span"] for r in rows])
    day = (t - t.min()) / 86400.0
    return np.column_stack([
        np.sin(2 * np.pi * hour / 24.0), np.cos(2 * np.pi * hour / 24.0),
        np.sin(4 * np.pi * hour / 24.0), np.cos(4 * np.pi * hour / 24.0),
        log_since, has_prev, postictal,
        np.log1p(np.clip(into, 0, None)), day,
    ])


def expand_person_period(X, rows, keep):
    """One (row, bin) record per bin the row was genuinely at risk in."""
    feats, ys, owners, bins = [], [], [], []
    for i in np.flatnonzero(keep):
        r = rows[i]
        for k in range(N_BINS):
            if r["last_observed_bin"] < k:
                break
            if r["outcome_bin"] is not None and r["outcome_bin"] < k:
                break
            onehot = np.zeros(N_BINS)
            onehot[k] = 1.0
            feats.append(np.concatenate([onehot, X[i]]))
            ys.append(1.0 if r["outcome_bin"] == k else 0.0)
            owners.append(i)
            bins.append(k)
            if r["outcome_bin"] == k:
                break
    if not feats:
        return None
    return np.vstack(feats), np.array(ys), np.array(owners), np.array(bins)


def hazards_for(model, X, rows, idx):
    H = np.zeros((len(rows), N_BINS))
    for k in range(N_BINS):
        onehot = np.zeros((len(idx), N_BINS))
        onehot[:, k] = 1.0
        H[idx, k] = model.predict_proba(np.hstack([onehot, X[idx]]))[:, 1]
    return H


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--producer", default=None,
                    help="registry producer id; omit for the v0.1 plumbing_only stand-in")
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--arm", default="a1_static_recent_history")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--crosswalk", type=Path, default=DEFAULT_OUT / "support/seizure_crosswalk.csv")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    rows = load_risk_rows(args.data_root / "risk_sets" / f"{args.subject}.csv")
    if not rows:
        raise SystemExit(f"no risk rows for {args.subject}")

    sz = []
    for r in csv.DictReader(args.crosswalk.open()):
        if r["disposition"] == "matched" and r["subject"] == args.subject:
            sz.append({"seizure_id": r["seizure_id"], "onset_epoch": float(r["onset_epoch"]),
                       "offset_epoch": float(r["offset_epoch"])})
    sz.sort(key=lambda s: s["onset_epoch"])
    episodes = group_seizure_episodes(sz, gap_seconds=DEFAULT_POSTICTAL_EXCLUSION_SECONDS)
    n_train_ep = max(1, math.ceil(len(episodes) / 2))
    if n_train_ep >= len(episodes):
        raise SystemExit(f"{args.subject}: no held-out episode under the rolling origin")
    split_epoch = episodes[n_train_ep][0]["onset_epoch"]

    if args.producer:
        reg = read_registry(args.registry) if args.registry else read_registry()
        arms_r = resolve_subject_arms(reg, args.subject, seed=str(args.seed))
        if args.producer not in arms_r:
            raise SystemExit(f"producer {args.producer!r} not in registry {list(arms_r)}")
        a = arms_r[args.producer]
        if a.status != "ok":
            out = {"subject": args.subject, "producer": args.producer,
                   "status": "not_available", "reason": a.reason,
                   "registry_version": reg.version}
            d = args.out_root / "machine"; d.mkdir(parents=True, exist_ok=True)
            (d / f"b1__{args.subject}__{args.producer}__seed{args.seed}.json").write_text(
                json.dumps(out, indent=2))
            print(json.dumps(out, indent=2)); return
        states, t_abs = a.state, a.t_anchor
        tag = "registry_producer"
        warn = ""
        provenance = {"producer": args.producer, "seed": a.seed,
                      "source_commit": a.source_commit, "config_hash": a.config_hash,
                      "checkpoint_hash": a.checkpoint_hash, "registry_version": reg.version}
        max_age = 330.0
    else:
        run = V0_1_RUNS / f"{args.subject}__{args.arm}__seed{args.seed}"
        if not (run / "test_states.npy").exists():
            raise SystemExit(f"no v0.1 trajectory at {run}")
        states = np.load(run / "test_states.npy")
        t_abs = np.load(run / "test_series.npz")["t_abs"]
        tag = "plumbing_only"
        warn = "v0.1 stand-in trajectory; NOT a v0.2 producer and NOT a human result"
        provenance = {"stand_in": f"v0.1 {args.arm} seed{args.seed}"}
        max_age = MAX_STATE_AGE_SEC

    anchors = np.array([r["anchor_epoch"] for r in rows])
    att = attach_state_to_anchors(anchors, t_abs, states, max_age_seconds=max_age)

    Xb = baseline_features(rows)
    is_train = anchors < split_epoch
    is_eval = ~is_train
    usable = att.available & np.isfinite(Xb).all(axis=1)

    # Nested arms must run on identical rows: only anchors where the frozen
    # state actually exists can enter either arm.
    train_mask = is_train & usable
    eval_mask = is_eval & usable
    out = {
        "tag": tag, "warning": warn, "provenance": provenance,
        "subject": args.subject, "arm": args.arm, "seed": args.seed,
        "n_rows_total": len(rows),
        "n_rows_state_available": int(usable.sum()),
        "n_train_rows": int(train_mask.sum()), "n_eval_rows": int(eval_mask.sum()),
        "n_episodes": len(episodes), "n_heldout_episodes": len(episodes) - n_train_ep,
        "split_epoch": split_epoch,
        "state_dim": int(states.shape[1]),
        "max_state_age_sec": max_age,
    }

    out["split_mode"] = "seizure_rolling_origin"
    if train_mask.sum() < 50 and usable.sum() >= 200:
        # The v0.1 trajectory exists only on v0.1's own TEST split, which lies
        # entirely after this subject's seizure-based rolling origin, so the
        # scientific split leaves almost no TRAIN rows with a state. For a
        # PLUMBING run the split is therefore taken inside the state-covered
        # span. This is legitimate only because nothing here is a result; the
        # scientific split is still reported next to it so the mismatch is visible.
        cover = np.flatnonzero(usable)
        cut = anchors[cover][len(cover) // 2]
        train_mask = usable & (anchors < cut)
        eval_mask = usable & (anchors >= cut)
        out["split_mode"] = "within_state_coverage_plumbing_split"
        out["plumbing_split_epoch"] = float(cut)
        out["n_train_rows"] = int(train_mask.sum())
        out["n_eval_rows"] = int(eval_mask.sum())

    if train_mask.sum() < 50 or eval_mask.sum() < 20:
        out["status"] = "insufficient_rows_for_plumbing"
        _write(out, args); return

    # State capacity is selected on TRAIN, like C: eight components of a 96-dim
    # state overfit a few thousand person-period rows, and the arm then fits
    # worse than an intercept. Both the component count and the penalty are
    # chosen inside TRAIN, so no evaluation row informs either.
    age = np.where(att.available, att.age_seconds, 0.0)[:, None]

    def build_state_block(k):
        pca = PCA(n_components=min(k, int(train_mask.sum()), states.shape[1]))
        pca.fit(att.state[train_mask])
        Xs = np.zeros((len(rows), pca.n_components_))
        Xs[usable] = pca.transform(att.state[usable])
        return np.hstack([Xb, Xs, np.log1p(np.clip(age, 0, None))])

    def fit_with_cv(X, rows, train_mask):
        """Pick C by chronological CV inside TRAIN; no eval row informs it."""
        tr_idx = np.flatnonzero(train_mask)
        cut = tr_idx[int(0.7 * len(tr_idx))] if len(tr_idx) > 20 else None
        best, best_c = -np.inf, 1.0
        if cut is not None:
            inner_tr = train_mask.copy(); inner_tr[cut:] = False
            inner_va = train_mask.copy(); inner_va[:cut] = False
            a = expand_person_period(X, rows, inner_tr)
            b = expand_person_period(X, rows, inner_va)
            if a is not None and b is not None and len(np.unique(a[1])) > 1:
                for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
                    m = LogisticRegression(max_iter=3000, C=c).fit(a[0], a[1])
                    pr = np.clip(m.predict_proba(b[0])[:, 1], 1e-6, 1 - 1e-6)
                    ll = float(np.mean(b[1] * np.log(pr) + (1 - b[1]) * np.log1p(-pr)))
                    if ll > best:
                        best, best_c = ll, c
        tr = expand_person_period(X, rows, train_mask)
        return LogisticRegression(max_iter=3000, C=best_c).fit(tr[0], tr[1]), best_c, tr

    def inner_score(X):
        tr_idx = np.flatnonzero(train_mask)
        if len(tr_idx) <= 20:
            return -np.inf
        cut = tr_idx[int(0.7 * len(tr_idx))]
        itr = train_mask.copy(); itr[cut:] = False
        iva = train_mask.copy(); iva[:cut] = False
        a, b = expand_person_period(X, rows, itr), expand_person_period(X, rows, iva)
        if a is None or b is None or len(np.unique(a[1])) < 2:
            return -np.inf
        best = -np.inf
        for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            m = LogisticRegression(max_iter=3000, C=c).fit(a[0], a[1])
            pr = np.clip(m.predict_proba(b[0])[:, 1], 1e-6, 1 - 1e-6)
            best = max(best, float(np.mean(b[1] * np.log(pr) + (1 - b[1]) * np.log1p(-pr))))
        return best

    candidates = {k: build_state_block(k) for k in (1, 2, 4, 8)}
    k_star = max(candidates, key=lambda k: inner_score(
        StandardScaler().fit(candidates[k][train_mask]).transform(candidates[k])))
    Xfull = candidates[k_star]
    scaler = StandardScaler().fit(Xfull[train_mask])
    Xb_s = scaler.transform(Xfull)[:, :Xb.shape[1]]
    Xfull_s = scaler.transform(Xfull)

    res = {}
    chosen_C = {}
    arms_spec = (("intercept_only", np.zeros((len(rows), 0))),
                 ("baseline", Xb_s), ("baseline_plus_state", Xfull_s))
    for name, X in arms_spec:
        tr = expand_person_period(X, rows, train_mask)
        if tr is None:
            out["status"] = "no_person_period_rows"; _write(out, args); return
        if len(np.unique(tr[1])) < 2:
            out["status"] = "train_has_no_event_variation"; _write(out, args); return
        model, c, tr = fit_with_cv(X, rows, train_mask)
        chosen_C[name] = c
        idx = np.flatnonzero(eval_mask)
        H = hazards_for(model, X, rows, idx)
        ll = discrete_time_log_score(
            H[idx],
            [rows[i]["outcome_bin"] for i in idx],
            [rows[i]["last_observed_bin"] for i in idx],
            [rows[i]["censored"] for i in idx],
        )
        br = brier_by_bin(
            H[idx],
            [rows[i]["outcome_bin"] for i in idx],
            [rows[i]["last_observed_bin"] for i in idx],
            [rows[i]["censored"] for i in idx],
        )
        res[name] = {"log_score": ll, "brier_by_bin": br}

    # Engineering invariant: an arm that fits far worse than the intercept-only
    # reference is an unusable fit, not evidence against the state. Reporting its
    # negative increment as "the state hurts" would be a fabricated finding.
    ref = float(np.mean(res["intercept_only"]["log_score"]))
    worse = {k: bool(float(np.mean(v["log_score"])) - ref < -0.05)
             for k, v in res.items() if k != "intercept_only"}
    out["chosen_C"] = chosen_C
    out["chosen_state_components"] = int(k_star)
    out["mean_log_score_intercept_only"] = ref
    out["arm_worse_than_intercept"] = worse

    inc = nested_increment(res["baseline"]["log_score"], res["baseline_plus_state"]["log_score"])
    # A 5-min grid row is not a seizure: many consecutive rows point at the same
    # next seizure. Report both, and never let the row count pose as sample size
    # ("event rows 不冒充样本量", common contract §10).
    _ev_rows = [i for i in np.flatnonzero(eval_mask) if rows[i]["outcome_bin"] is not None]
    out["eval_event_rows"] = int(len(_ev_rows))
    out["eval_distinct_seizures"] = int(len({rows[i]["next_seizure_id"] for i in _ev_rows}))
    out["eval_events"] = out["eval_distinct_seizures"]  # the honest denominator
    # A survival log score with no events in the evaluation set is not a weak
    # result, it is an uninformative one: nothing distinguishes the arms.
    if out["eval_events"] == 0:
        out["status"] = "ok_but_no_events_in_eval"
    elif any(worse.values()):
        out["status"] = "unusable_fit_worse_than_intercept"
    else:
        out["status"] = "ok"
    out["mean_log_score"] = {k: float(np.mean(v["log_score"])) for k, v in res.items()}
    out["brier_by_bin"] = {k: [None if not np.isfinite(x) else round(float(x), 5)
                               for x in v["brier_by_bin"]] for k, v in res.items()}
    out["nested_increment_log_score"] = inc
    _write(out, args)


def _write(out, args):
    d = args.out_root / "machine"
    d.mkdir(parents=True, exist_ok=True)
    label = args.producer or args.arm
    prefix = "b1" if args.producer else "b1_plumbing"
    p = d / f"{prefix}__{args.subject}__{label}__seed{args.seed}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, default=float))
    tmp.rename(p)
    print(json.dumps({k: v for k, v in out.items()
                      if k in ("subject", "status", "split_mode", "n_train_rows",
                               "n_eval_rows", "eval_events", "chosen_C",
                               "chosen_state_components",
                               "mean_log_score_intercept_only", "arm_worse_than_intercept",
                               "mean_log_score", "nested_increment_log_score")},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
