#!/usr/bin/env python3
"""B0.2/B0.3 support -- fixed 5-min risk sets, censoring, and per-lead coverage.

Answers the question that decides whether H2b is estimable at all, *before* any
producer exists: for each held-out seizure and each pre-registered lead
(6 h / 2 h / 30 min / 5 min), does a frozen-state anchor actually exist?

Coverage sources (kept apart on purpose):
  state       = blocks that entered the consolidated dataset (index.json::source_shards)
  monitoring  = epilepsiae SQL block inventory; yuquan EDF blocks
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_h2b_transfer.risk_grid import (  # noqa: E402
    DEFAULT_POSTICTAL_EXCLUSION_SECONDS,
    HORIZON_EDGES_SECONDS,
    build_risk_rows,
    group_seizure_episodes,
    lead_anchor_status,
    merge_spans,
)

MAIN_TREE = Path("/home/honglab/leijiaxin/HFOsp")
V0_1 = MAIN_TREE / "results/epi_prssm/group_event_state/v0_1"
DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b"
DEFAULT_DATA = Path("/data/hfosp_group_event_state_v0_2/agent_b")

#: B2 reads the frozen state at these leads before the same seizure.
LEADS_SECONDS = (5 * 60.0, 30 * 60.0, 2 * 3600.0, 6 * 3600.0)
LEAD_NAMES = ("5min", "30min", "2h", "6h")

SPLIT_FRACTIONS = (0.7, 0.1, 0.2)  # train / val / development-test, by physical time


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def load_state_spans(dataset_root: Path, block_inventory: Path):
    """subject -> merged spans of blocks that actually entered the dataset."""
    covered: dict[str, set[str]] = {}
    meta: dict[str, dict] = {}
    for sub in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        idx_path = sub / "index.json"
        if not idx_path.exists():
            continue
        idx = json.loads(idx_path.read_text())
        covered[sub.name] = {Path(s).stem for s in idx.get("source_shards", [])}
        meta[sub.name] = {"dataset": idx["dataset"], "n_events": idx.get("n_events")}
    spans: dict[str, list[tuple[float, float]]] = {s: [] for s in covered}
    for row in csv.DictReader(block_inventory.open()):
        s = row["subject"]
        if s in covered and row["record_name"] in covered[s]:
            spans[s].append((_f(row["block_start_epoch"]), _f(row["block_end_epoch"])))
    return {s: merge_spans(v) for s, v in spans.items()}, meta


def load_monitoring_spans(epi_blocks: Path, yuquan_state_spans, v0_1_blocks: Path):
    """Recording coverage: epilepsiae from SQL; yuquan from its EDF blocks."""
    spans: dict[str, list[tuple[float, float]]] = {}
    for row in csv.DictReader(epi_blocks.open()):
        s = f"epilepsiae_{row['subject']}"
        spans.setdefault(s, []).append((_f(row["block_start_epoch"]), _f(row["block_end_epoch"])))
    for row in csv.DictReader(v0_1_blocks.open()):
        if row["dataset"] != "yuquan":
            continue
        spans.setdefault(row["subject"], []).append(
            (_f(row["block_start_epoch"]), _f(row["block_end_epoch"]))
        )
    return {s: merge_spans(v) for s, v in spans.items()}


def physical_time_split(spans, fractions=SPLIT_FRACTIONS):
    """Chronological boundaries on *cumulative recorded physical time* (§7.1)."""
    total = sum(e - s for s, e in spans)
    if total <= 0:
        return (math.nan, math.nan)
    want_train, want_val = total * fractions[0], total * (fractions[0] + fractions[1])
    acc, b_train, b_val = 0.0, None, None
    for s, e in spans:
        dur = e - s
        if b_train is None and acc + dur >= want_train:
            b_train = s + (want_train - acc)
        if b_val is None and acc + dur >= want_val:
            b_val = s + (want_val - acc)
        acc += dur
    return (b_train if b_train is not None else spans[-1][1],
            b_val if b_val is not None else spans[-1][1])


def period_of(t, bounds):
    b_train, b_val = bounds
    if not math.isfinite(b_train):
        return "unknown"
    return "train" if t < b_train else ("val" if t < b_val else "development_test")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", type=Path, default=Path("/data/hfosp_group_event_state_v0_1/dataset"))
    ap.add_argument("--block-inventory", type=Path, default=V0_1 / "block_inventory.csv")
    ap.add_argument("--epilepsiae-blocks", type=Path, default=MAIN_TREE / "results/epilepsiae_block_inventory.csv")
    ap.add_argument("--crosswalk", type=Path, default=DEFAULT_OUT / "support/seizure_crosswalk.csv")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--postictal-exclusion-seconds", type=float, default=DEFAULT_POSTICTAL_EXCLUSION_SECONDS)
    args = ap.parse_args()

    state_spans, meta = load_state_spans(args.dataset_root, args.block_inventory)
    monitoring = load_monitoring_spans(args.epilepsiae_blocks, state_spans, args.block_inventory)

    seizures: dict[str, list[dict]] = {}
    for row in csv.DictReader(args.crosswalk.open()):
        if row["disposition"] != "matched":
            continue
        seizures.setdefault(row["subject"], []).append(
            {
                "seizure_id": row["seizure_id"],
                "onset_epoch": _f(row["onset_epoch"]),
                "offset_epoch": _f(row["offset_epoch"]),
            }
        )
    for v in seizures.values():
        v.sort(key=lambda s: s["onset_epoch"])

    (args.out_root / "support").mkdir(parents=True, exist_ok=True)
    (args.data_root / "risk_sets").mkdir(parents=True, exist_ok=True)

    lead_rows, support_rows = [], []
    event_times: dict[str, "np.ndarray"] = {}
    for subject in sorted(state_spans):
        spans = state_spans[subject]
        mon = monitoring.get(subject, spans)
        sz = seizures.get(subject, [])
        bounds = physical_time_split(spans)

        rows = build_risk_rows(
            subject=subject,
            state_spans=spans,
            monitoring_spans=mon,
            seizures=sz,
            postictal_exclusion_seconds=args.postictal_exclusion_seconds,
        )

        rp = args.data_root / "risk_sets" / f"{subject}.csv"
        tmp = rp.with_suffix(".csv.tmp")
        with tmp.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "subject", "anchor_epoch", "state_span_index", "seconds_into_state_span",
                "time_to_next_seizure_sec", "next_seizure_id", "outcome_bin", "censored",
                "beyond_horizon", "last_observed_bin", "observed_horizon_sec",
                "time_since_prev_seizure_sec", "prev_seizure_id", "state_period",
            ])
            for r in rows:
                w.writerow([
                    r.subject, f"{r.anchor_epoch:.3f}", r.state_span_index,
                    f"{r.seconds_into_state_span:.3f}",
                    "" if r.time_to_next_seizure_sec is None else f"{r.time_to_next_seizure_sec:.3f}",
                    r.next_seizure_id or "",
                    "" if r.outcome_bin is None else r.outcome_bin,
                    int(r.censored), int(r.beyond_horizon), r.last_observed_bin,
                    f"{r.observed_horizon_sec:.3f}",
                    "" if r.time_since_prev_seizure_sec is None else f"{r.time_since_prev_seizure_sec:.3f}",
                    r.prev_seizure_id or "", period_of(r.anchor_epoch, bounds),
                ])
        tmp.rename(rp)

        # D9: a cluster is one predictable episode. Rolling origin is applied to
        # EPISODES, and only the lead seizure of an episode can be a target --
        # every follower sits inside its own episode's postictal exclusion.
        # Interictal event times: how fresh is the state a lead anchor can carry?
        if subject not in event_times:
            import numpy as _np
            zp = args.dataset_root / subject / "scalars.npz"
            if zp.exists():
                _z = _np.load(zp)
                event_times[subject] = _np.sort(_z["t_abs"][~_z["is_ictal"]])
            else:
                event_times[subject] = _np.empty(0)

        def _age(t):
            import numpy as _np
            arr = event_times[subject]
            if arr.size == 0:
                return None
            i = int(_np.searchsorted(arr, t, side="left")) - 1
            return float(t - arr[i]) if i >= 0 else None

        episodes = group_seizure_episodes(sz, gap_seconds=args.postictal_exclusion_seconds)
        n, n_ep = len(sz), len(episodes)
        n_train_ep = max(1, math.ceil(n_ep / 2)) if n_ep else 0
        for ei, ep in enumerate(episodes):
            role = "train" if ei < n_train_ep else "held_out"
            lead_sz = ep[0]
            for lead, name in zip(LEADS_SECONDS, LEAD_NAMES):
                status = lead_anchor_status(
                    anchor_epoch=lead_sz["onset_epoch"] - lead,
                    state_spans=spans,
                    seizures=sz,
                    postictal_exclusion_seconds=args.postictal_exclusion_seconds,
                )
                lead_rows.append({
                    "subject": subject, "seizure_id": lead_sz["seizure_id"],
                    "episode_index": ei, "episode_size": len(ep),
                    "role": role, "lead": name, "lead_seconds": lead,
                    "anchor_epoch": lead_sz["onset_epoch"] - lead, "status": status,
                    "onset_epoch": lead_sz["onset_epoch"],
                    "state_period": period_of(lead_sz["onset_epoch"], bounds),
                    "zero_duration_lead": int(lead_sz["offset_epoch"] <= lead_sz["onset_epoch"]),
                    "seconds_since_last_event": _age(lead_sz["onset_epoch"] - lead),
                })

        n_ho_ep = n_ep - n_train_ep
        by_lead = {}
        for name in LEAD_NAMES:
            by_lead[name] = sum(
                1 for r in lead_rows
                if r["subject"] == subject and r["lead"] == name
                and r["role"] == "held_out" and r["status"] == "ok"
            )
        n_events = sum(1 for r in rows if r.outcome_bin is not None)
        support_rows.append({
            "subject": subject,
            "dataset": meta[subject]["dataset"],
            "state_coverage_hours": round(sum(e - s for s, e in spans) / 3600.0, 2),
            "monitoring_hours": round(sum(e - s for s, e in mon) / 3600.0, 2),
            "n_state_spans": len(spans),
            "n_seizures_matched": n,
            "n_episodes": n_ep,
            "n_train_episodes": n_train_ep,
            "n_heldout_episodes": n_ho_ep,
            "n_zero_duration_seizures": sum(1 for s in sz if s["offset_epoch"] <= s["onset_epoch"]),
            "max_episode_size": max((len(e) for e in episodes), default=0),
            "n_risk_rows": len(rows),
            "n_rows_with_event": n_events,
            "n_rows_censored": sum(1 for r in rows if r.censored),
            "n_rows_beyond_horizon": sum(1 for r in rows if r.beyond_horizon),
            **{f"heldout_anchor_{k}": v for k, v in by_lead.items()},
        })

    lp = args.out_root / "support/lead_coverage.csv"
    tmp = lp.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(lead_rows[0].keys()))
        w.writeheader()
        w.writerows(lead_rows)
    tmp.rename(lp)

    sp = args.out_root / "support/support_inventory.csv"
    tmp = sp.with_suffix(".csv.tmp")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(support_rows[0].keys()))
        w.writeheader()
        w.writerows(support_rows)
    tmp.rename(sp)

    print(f"wrote {lp}\nwrote {sp}\nrisk sets -> {args.data_root/'risk_sets'}")
    tot = {k: sum(r[k] for r in support_rows) for k in
           ("n_seizures_matched", "n_episodes", "n_heldout_episodes", "n_zero_duration_seizures",
            "n_risk_rows", "n_rows_with_event", "n_rows_censored", "n_rows_beyond_horizon")}
    print("\nCOHORT TOTALS:", json.dumps(tot, indent=2))
    print("\nheld-out EPISODES with a usable state anchor, by lead:")
    for name in LEAD_NAMES:
        k = f"heldout_anchor_{name}"
        n_sz = sum(r[k] for r in support_rows)
        n_pat = sum(1 for r in support_rows if r[k] > 0)
        print(f"    {name:6s}  seizures={n_sz:4d}   patients={n_pat:3d}")


if __name__ == "__main__":
    main()
