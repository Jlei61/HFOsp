"""Per-subject absolute-timeline plotter for SEF-ITP Phase 3 v2.

Per user 2026-05-24:
  - swap-k / decision_k is a TEMPLATE-PAIR statistic, not per-event SOZ extent.
  - This figure shows PER-EVENT source (top-k earliest channels per event,
    median centroid RMS across events in a 55-min window) and sink (last-k
    latest channels), for k = 2, 3, 4, 5, 6 — gradient color shows how
    spatial spread grows as k increases. If k=2 is tight and k=6 is also
    tight → seed and immediate network are both spatially focal. If k=2 is
    tight but k=6 wide → small seed triggers wider network.

  IMPORTANT CAVEAT — even k=2 tight does NOT prove SOZ extent. It says
  per-event propagation seeds are spatially compact at the channels they
  fire from. SEEG sampling and seed-vs-extent identification are separate.

Output:
  results/topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory/figures/timeline_<dataset>_<sid>.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


PER_SUBJECT_DIR = Path("results/topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory/per_subject")
FIGURES_DIR = Path("results/topic4_sef_itp/phase3_ictal_adjacent/v2_trajectory/figures")

K_VALUES = (2, 3, 4, 5, 6)


def _row_from_window(w: dict) -> dict:
    g = w.get("per_event_endpoint_geometry") or {}
    out = {
        "t_start": w["t_start"],
        "t_end": w["t_end"],
        "rate_per_hour": w.get("rate_per_hour"),
    }
    for k in K_VALUES:
        src = g.get(f"source_top_{k}") or {}
        snk = g.get(f"sink_last_{k}") or {}
        out[f"source_{k}"] = src.get("rms_median")
        out[f"sink_{k}"] = snk.get("rms_median")
        out[f"source_{k}_n"] = src.get("n_events_used", 0)
        out[f"sink_{k}_n"] = snk.get("n_events_used", 0)
    return out


def _flatten_subject_windows(d: dict) -> Tuple[pd.DataFrame, List[float], str]:
    rows = []
    full = d.get("full_sweep_trajectory") or []
    if full:
        for w in full:
            rows.append(_row_from_window(w))
        source = "full_sweep"
    else:
        for sz in d.get("per_seizure_trajectory") or []:
            for w in (sz.get("trajectory") or []):
                rows.append(_row_from_window(w))
        source = "per_seizure_anchored"
    df = pd.DataFrame(rows)
    sz_onsets = sorted({sz["seizure_onset_t"] for sz in d.get("per_seizure_trajectory") or []})
    return df, sz_onsets, source


SWAP_CLASS_PLAIN = {
    "strict": "strong swap evidence",
    "candidate": "moderate swap evidence",
    "none": "no swap evidence",
}


def plot_subject_timeline(subject_label: str, swap_class: str, df: pd.DataFrame,
                          seizure_onsets: List[float], source: str, out_path: Path) -> None:
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    if df.empty:
        return
    t0 = min(df["t_start"].min(), min(seizure_onsets) if seizure_onsets else df["t_start"].min())
    df = df.copy()
    df["t_h"] = (df["t_start"] - t0) / 3600.0
    df = df.sort_values("t_h").drop_duplicates(subset=["t_h"], keep="first")
    seizure_h = [(t - t0) / 3600.0 for t in seizure_onsets]
    ts = df["t_h"].values

    COL_RATE = "0.35"
    COL_SEIZURE = "#d62728"
    cmap_src = plt.get_cmap("Blues")
    cmap_snk = plt.get_cmap("Oranges")
    # Light-to-dark gradient for k=2..6
    src_colors = [cmap_src(0.35 + 0.15 * i) for i in range(len(K_VALUES))]
    snk_colors = [cmap_snk(0.35 + 0.15 * i) for i in range(len(K_VALUES))]

    fig, axes = plt.subplots(3, 1, figsize=(13, 8.2), sharex=True,
                             gridspec_kw={"hspace": 0.18, "height_ratios": [1.0, 1.3, 1.3]})

    # --- Panel 1: HFO event rate ---
    ax = axes[0]
    rates = df["rate_per_hour"].astype(float).values
    m = np.isfinite(rates)
    if m.any():
        ax.bar(ts[m], rates[m], width=0.45, color=COL_RATE,
               alpha=0.85, edgecolor="none")
        ax.set_ylim(0, float(np.nanmax(rates[m])) * 1.08)
    ax.set_ylabel("HFO events / hour", fontsize=11)

    # --- Panel 2: per-event source RMS, 5 k values gradient ---
    ax = axes[1]
    all_src = []
    for i, k in enumerate(K_VALUES):
        v = df[f"source_{k}"].astype(float).values
        m = np.isfinite(v)
        if m.any():
            ax.plot(ts[m], v[m], color=src_colors[i], linewidth=1.1, marker="o",
                    markersize=2.6, alpha=0.92, zorder=2 + i)
            all_src.append(v[m])
    ax.set_ylabel("source RMS (mm)\nper-event top-k earliest", fontsize=11)
    if all_src:
        ymax = float(np.nanmax(np.concatenate(all_src)))
        ax.set_ylim(0, ymax * 1.15)

    # --- Panel 3: per-event sink RMS, 5 k values gradient ---
    ax = axes[2]
    all_snk = []
    for i, k in enumerate(K_VALUES):
        v = df[f"sink_{k}"].astype(float).values
        m = np.isfinite(v)
        if m.any():
            ax.plot(ts[m], v[m], color=snk_colors[i], linewidth=1.1, marker="s",
                    markersize=2.6, alpha=0.92, zorder=2 + i)
            all_snk.append(v[m])
    ax.set_ylabel("sink RMS (mm)\nper-event last-k latest", fontsize=11)
    if all_snk:
        ymax = float(np.nanmax(np.concatenate(all_snk)))
        ax.set_ylim(0, ymax * 1.15)

    for ax in axes:
        for st in seizure_h:
            ax.axvline(st, color=COL_SEIZURE, linestyle="--", alpha=0.55, linewidth=0.7,
                       zorder=0)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=10)
        ax.margins(x=0.005)

    axes[-1].set_xlabel("Hours from recording start", fontsize=11)

    n_sz = len(seizure_h)
    grouping = SWAP_CLASS_PLAIN.get(swap_class, swap_class) if swap_class != "?" else "PR-2 cohort"
    fig.text(0.005, 0.985,
             f"{subject_label}    grouping: {grouping}    {n_sz} seizures",
             fontsize=11, fontweight="bold", ha="left", va="top")
    fig.text(0.005, 0.96,
             "Per-event endpoint centroid spread — does NOT prove SOZ extent; "
             "only says event-by-event propagation seeds (source) and termini "
             "(sink) are spatially compact at the channels they fire from.",
             fontsize=8, style="italic", ha="left", va="top", color="0.35")

    # Single shared legend — show k gradient + seizure marker
    handles = [
        mlines.Line2D([], [], color=COL_RATE, linewidth=5, label="HFO event rate"),
    ]
    for i, k in enumerate(K_VALUES):
        handles.append(mlines.Line2D([], [], color=src_colors[i], marker="o", markersize=4,
                                     linewidth=1.1, label=f"source k={k}"))
    for i, k in enumerate(K_VALUES):
        handles.append(mlines.Line2D([], [], color=snk_colors[i], marker="s", markersize=4,
                                     linewidth=1.1, label=f"sink k={k}"))
    handles.append(mlines.Line2D([], [], color=COL_SEIZURE, linestyle="--", linewidth=1.4,
                                 label="Seizure onset"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.005))

    fig.subplots_adjust(top=0.93, bottom=0.10, left=0.075, right=0.99)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="dataset_subject e.g. epilepsiae_1146")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.subject:
        files = [PER_SUBJECT_DIR / f"{args.subject}.json"]
    elif args.all:
        files = sorted(PER_SUBJECT_DIR.glob("*.json"))
    else:
        parser.error("provide --subject or --all")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for p in files:
        if not p.exists():
            print(f"  missing: {p}")
            continue
        d = json.loads(p.read_text())
        if d.get("exit_reason") != "ok":
            print(f"  skip {p.stem} (exit={d.get('exit_reason')})")
            continue
        df, sz_onsets, source = _flatten_subject_windows(d)
        if df.empty:
            print(f"  skip {p.stem} (no windows)")
            continue
        subj = f"{d['dataset']}_{d['subject_id']}"
        sc = d.get("swap_class_full_data") or "?"
        out_path = FIGURES_DIR / f"timeline_{subj}.png"
        plot_subject_timeline(subj, sc, df, sz_onsets, source, out_path)
        print(f"  -> {out_path.name}  ({len(df)} windows, {len(sz_onsets)} seizures)")


if __name__ == "__main__":
    main()
