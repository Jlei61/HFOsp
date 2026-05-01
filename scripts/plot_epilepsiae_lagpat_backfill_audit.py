"""Stage C audit figures (plan §4 task C.1, output dir convention).

Inputs:  results/epilepsiae_lagpat_backfill/audit/per_record_audit.csv
         results/epilepsiae_lagpat_backfill/audit/cohort_audit_summary.json
Outputs: results/epilepsiae_lagpat_backfill/audit/figures/{*.png, README.md}

Three histograms (filtered to *_eligible == True):
  - chn_overlap_hist.png:        chn_overlap_jaccard distribution per record
  - event_count_ratio_hist.png:  log10(count_ratio) per record
  - rank_template_corr_hist.png: rank_template_corr per record

Bucket boundaries (plan §4 Task C.2) overlaid as vertical lines.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

AUDIT_ROOT = Path("results/epilepsiae_lagpat_backfill/audit")
FIG_DIR = AUDIT_ROOT / "figures"


def _load_records() -> List[dict]:
    p = AUDIT_ROOT / "per_record_audit.csv"
    with p.open() as f:
        return list(csv.DictReader(f))


def _eligible_floats(rows: List[dict], val_key: str, eligible_key: str) -> np.ndarray:
    out: List[float] = []
    for r in rows:
        if r.get(eligible_key) != "True":
            continue
        v = r.get(val_key, "")
        if v == "":
            continue
        try:
            f = float(v)
        except ValueError:
            continue
        if math.isnan(f) or math.isinf(f):
            continue
        out.append(f)
    return np.asarray(out, dtype=float)


def plot_chn_overlap(rows: List[dict]) -> None:
    vals = _eligible_floats(rows, "chn_overlap_jaccard", "chn_overlap_jaccard_eligible")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=40, range=(0, 1), color="steelblue", alpha=0.85)
    ax.axvline(0.5, ls="--", color="orange", label="moderate (0.5)")
    ax.axvline(0.7, ls="--", color="green", label="stable (0.7)")
    ax.set_xlabel("chn_overlap_jaccard (per paired record)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Channel set Jaccard overlap — {len(vals)} eligible paired records"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "chn_overlap_hist.png", dpi=140)
    plt.close(fig)


def plot_event_count_ratio(rows: List[dict]) -> None:
    vals = _eligible_floats(rows, "count_ratio", "count_ratio_eligible")
    log_vals = np.log10(vals[vals > 0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(log_vals, bins=50, color="indianred", alpha=0.85)
    # Bucket boundaries on log10 scale
    for x, lab, c in [
        (math.log10(0.5), "moderate low (0.5)", "orange"),
        (math.log10(0.7), "stable low (0.7)", "green"),
        (math.log10(1.4), "stable high (1.4)", "green"),
        (math.log10(2.0), "moderate high (2.0)", "orange"),
    ]:
        ax.axvline(x, ls="--", color=c, alpha=0.7, label=lab)
    ax.set_xlabel("log10(count_ratio = n_events_new / n_events_legacy)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Event count ratio — {len(log_vals)} eligible paired records "
        f"(median={float(np.median(vals)):.3g})"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "event_count_ratio_hist.png", dpi=140)
    plt.close(fig)


def plot_rank_template_corr(rows: List[dict]) -> None:
    vals = _eligible_floats(rows, "rank_template_corr", "rank_template_eligible")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(vals, bins=40, range=(-1, 1), color="seagreen", alpha=0.85)
    ax.axvline(0, ls="--", color="gray", label="zero")
    ax.set_xlabel("rank_template_corr (Pearson r over shared chns)")
    ax.set_ylabel("count")
    ax.set_title(
        f"Rank template correlation — {len(vals)} eligible paired records "
        f"(median={float(np.median(vals)):.3g})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rank_template_corr_hist.png", dpi=140)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = _load_records()
    plot_chn_overlap(rows)
    plot_event_count_ratio(rows)
    plot_rank_template_corr(rows)
    print(f"wrote {FIG_DIR}/")


if __name__ == "__main__":
    main()
