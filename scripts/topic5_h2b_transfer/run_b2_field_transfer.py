#!/usr/bin/env python3
"""B2 -- does the frozen interictal state predict the next seizure's early field?

For each held-out episode lead and each pre-registered lead time
(6 h / 2 h / 30 min / 5 min):

    baseline   patient-average of the TRAIN early fields (uniform weights)
    state arm  the same TRAIN fields, re-weighted by how similar the frozen
               state at ``onset - lead`` is to the state before each TRAIN
               seizure at *its* own ``onset - lead``

Identical TRAIN fields in both arms, so the increment isolates the state.

The softmax temperature is chosen per patient by leave-one-out **within TRAIN**
and then frozen; the held-out seizures never inform it. A lead whose anchor does
not exist is reported ``not_estimable`` and never back-filled.

Until a v0.2 producer exists this runs on a v0.1 stand-in trajectory and every
output is tagged ``plumbing_only``.
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

from src.topic5_h2b_transfer.attach import attach_state_to_anchors  # noqa: E402
from src.topic5_h2b_transfer.field_predict import (  # noqa: E402
    predict_field,
    state_similarity_weights,
)
from src.topic5_h2b_transfer.risk_grid import (  # noqa: E402
    DEFAULT_POSTICTAL_EXCLUSION_SECONDS,
    group_seizure_episodes,
    lead_anchor_status,
    merge_spans,
)
from src.topic5_h2b_transfer.registry import read_registry, resolve_subject_arms  # noqa: E402
from src.topic5_h2b_transfer.scoring import field_score  # noqa: E402

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1_RUNS = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1/runs/main"
V0_1 = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1"
DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b"
DEFAULT_DATA = Path("/data/hfosp_group_event_state_v0_2/agent_b")

LEADS = ((5 * 60.0, "5min"), (30 * 60.0, "30min"), (2 * 3600.0, "2h"), (6 * 3600.0, "6h"))
TEMPERATURE_GRID = (0.05, 0.1, 0.25, 0.5, 1.0, 4.0)
MAX_STATE_AGE_SEC = 3600.0
FIELD_KEY = "hfa_field_5s"          # H2b spec §1 primary
FIELD_KEY_SENS = "hfa_field_10s"    # sensitivity


def state_spans_for(subject):
    covered = set()
    idx_path = Path("/data/hfosp_group_event_state_v0_1/dataset") / subject / "index.json"
    if idx_path.exists():
        covered = {Path(s).stem for s in json.loads(idx_path.read_text()).get("source_shards", [])}
    spans = []
    for r in csv.DictReader((V0_1 / "block_inventory.csv").open()):
        if r["subject"] == subject and r["record_name"] in covered:
            spans.append((float(r["block_start_epoch"]), float(r["block_end_epoch"])))
    return merge_spans(spans)


def choose_temperature(train_fields, train_states, grid=TEMPERATURE_GRID):
    """Leave-one-out within TRAIN only. Returns (tau, loo_score)."""
    n = len(train_fields)
    if n < 3:
        return None, float("nan")
    best, best_tau = -np.inf, None
    for tau in grid:
        scores = []
        for i in range(n):
            others = [j for j in range(n) if j != i]
            w = state_similarity_weights(train_states[i], train_states[others], tau)
            pred = predict_field(train_fields[others], w)
            s = field_score(pred, train_fields[i])
            if np.isfinite(s):
                scores.append(s)
        if scores and np.median(scores) > best:
            best, best_tau = float(np.median(scores)), tau
    return best_tau, best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--producer", default=None,
                    help="registry producer id (P_local / P_slow / B_multiscale). "
                         "Omit to run the v0.1 plumbing_only stand-in.")
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--arm", default="a1_static_recent_history")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--field-key", default=FIELD_KEY)
    ap.add_argument("--crosswalk", type=Path, default=DEFAULT_OUT / "support/seizure_crosswalk.csv")
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    fj = args.data_root / "early_field" / f"{args.subject}.json"
    if not fj.exists():
        raise SystemExit(f"no early field for {args.subject}")
    meta = json.loads(fj.read_text())
    fz = np.load(args.data_root / "early_field" / f"{args.subject}.npz")
    ok_idx = {s["seizure_id"]: i for i, s in enumerate(meta["seizures"]) if s["status"] == "ok"}

    sz = [{"seizure_id": r["seizure_id"], "onset_epoch": float(r["onset_epoch"]),
           "offset_epoch": float(r["offset_epoch"])}
          for r in csv.DictReader(args.crosswalk.open())
          if r["disposition"] == "matched" and r["subject"] == args.subject]
    sz.sort(key=lambda s: s["onset_epoch"])
    episodes = group_seizure_episodes(sz, gap_seconds=DEFAULT_POSTICTAL_EXCLUSION_SECONDS)
    n_train_ep = max(1, math.ceil(len(episodes) / 2))
    train_leads = [ep[0] for ep in episodes[:n_train_ep]]
    held_leads = [ep[0] for ep in episodes[n_train_ep:]]

    if args.producer:
        reg = read_registry(args.registry) if args.registry else read_registry()
        arms = resolve_subject_arms(reg, args.subject, seed=str(args.seed))
        if args.producer not in arms:
            raise SystemExit(f"producer {args.producer!r} not in registry {list(arms)}")
        arm = arms[args.producer]
        if arm.status != "ok":
            # Contract: report, never fall back to a different producer.
            out = {"subject": args.subject, "producer": args.producer,
                   "status": "not_available", "reason": arm.reason,
                   "registry_version": reg.version}
            d = args.out_root / "machine"; d.mkdir(parents=True, exist_ok=True)
            pth = d / f"b2_field__{args.subject}__{args.producer}__seed{args.seed}.json"
            pth.write_text(json.dumps(out, indent=2))
            print(json.dumps(out, indent=2)); return
        states, t_abs = arm.state, arm.t_anchor
        provenance = {"producer": args.producer, "seed": arm.seed,
                      "source_commit": arm.source_commit, "config_hash": arm.config_hash,
                      "checkpoint_hash": arm.checkpoint_hash,
                      "anchor_path": arm.anchor_path,
                      "registry_version": reg.version}
        tag, warn = "registry_producer", ""
        # the state lives on a 5-min grid, so the usable anchor is the grid point
        # immediately preceding onset - lead
        max_age = 330.0
    else:
        run = V0_1_RUNS / f"{args.subject}__{args.arm}__seed{args.seed}"
        if not (run / "test_states.npy").exists():
            raise SystemExit(f"no trajectory at {run}")
        states = np.load(run / "test_states.npy")
        t_abs = np.load(run / "test_series.npz")["t_abs"]
        provenance = {"stand_in": f"v0.1 {args.arm} seed{args.seed}"}
        tag, warn = "plumbing_only", ("v0.1 stand-in trajectory; NOT a v0.2 producer "
                                      "and NOT a human result")
        max_age = MAX_STATE_AGE_SEC
    spans = state_spans_for(args.subject)

    out = {
        "tag": tag,
        "warning": warn,
        "provenance": provenance,
        "subject": args.subject, "arm": args.arm, "seed": args.seed,
        "field_key": args.field_key,
        "n_episodes": len(episodes), "n_train_leads": len(train_leads),
        "n_heldout_leads": len(held_leads),
        "leads": {},
    }

    for lead_sec, lead_name in LEADS:
        def anchor_state(seizure):
            t = seizure["onset_epoch"] - lead_sec
            st = lead_anchor_status(t, spans, sz,
                                    postictal_exclusion_seconds=DEFAULT_POSTICTAL_EXCLUSION_SECONDS)
            if st != "ok":
                return None, st
            a = attach_state_to_anchors(np.array([t]), t_abs, states,
                                        max_age_seconds=max_age)
            if not a.available[0]:
                return None, "no_state_trajectory_at_anchor"
            return a.state[0], "ok"

        tr_fields, tr_states, tr_reasons = [], [], []
        for s in train_leads:
            i = ok_idx.get(s["seizure_id"])
            st, why = anchor_state(s)
            if i is None or st is None:
                tr_reasons.append(why if i is not None else "no_field")
                continue
            tr_fields.append(fz[f"{args.field_key}__{i:03d}"])
            tr_states.append(st)

        rec = {"n_train_usable": len(tr_fields), "per_seizure": [],
               "train_drop_reasons": tr_reasons}
        if len(tr_fields) < 2:
            rec["status"] = "not_estimable_insufficient_train_anchors"
            out["leads"][lead_name] = rec
            continue

        tr_fields_a = np.vstack(tr_fields)
        tr_states_a = np.vstack(tr_states)
        tau, loo = choose_temperature(tr_fields_a, tr_states_a)
        rec["temperature"] = tau
        rec["train_loo_score"] = None if not np.isfinite(loo) else round(float(loo), 4)

        base_s, state_s = [], []
        for s in held_leads:
            i = ok_idx.get(s["seizure_id"])
            st, why = anchor_state(s)
            row = {"seizure_id": s["seizure_id"]}
            if i is None:
                row["status"] = "no_field"
            elif st is None:
                row["status"] = why
            else:
                actual = fz[f"{args.field_key}__{i:03d}"]
                b = field_score(predict_field(tr_fields_a, None), actual)
                if tau is None:
                    row["status"] = "no_temperature_train_too_small"
                    row["baseline"] = None if not np.isfinite(b) else round(float(b), 4)
                else:
                    w = state_similarity_weights(st, tr_states_a, tau)
                    v = field_score(predict_field(tr_fields_a, w), actual)
                    row.update(status="ok",
                               baseline=None if not np.isfinite(b) else round(float(b), 4),
                               state=None if not np.isfinite(v) else round(float(v), 4))
                    if np.isfinite(b) and np.isfinite(v):
                        base_s.append(b); state_s.append(v)
            rec["per_seizure"].append(row)

        rec["n_scored"] = len(base_s)
        if base_s:
            d = np.array(state_s) - np.array(base_s)
            rec.update(status="ok",
                       median_baseline=round(float(np.median(base_s)), 4),
                       median_state=round(float(np.median(state_s)), 4),
                       median_increment=round(float(np.median(d)), 4),
                       n_positive=int((d > 0).sum()))
        else:
            rec["status"] = "not_estimable_no_scored_heldout"
        out["leads"][lead_name] = rec

    d = args.out_root / "machine"
    d.mkdir(parents=True, exist_ok=True)
    label = args.producer or args.arm
    p = d / f"b2_field__{args.subject}__{label}__seed{args.seed}.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=2, default=float))
    tmp.rename(p)
    print(f"{args.subject}  episodes={len(episodes)} train={len(train_leads)} held={len(held_leads)}")
    for name, rec in out["leads"].items():
        print(f"   {name:>6}  {rec.get('status','?'):<42} "
              f"n={rec.get('n_scored','-'):>3}  base={rec.get('median_baseline','-')}  "
              f"state={rec.get('median_state','-')}  inc={rec.get('median_increment','-')}")


if __name__ == "__main__":
    main()
