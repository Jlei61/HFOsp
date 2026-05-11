"""Per-subject bridge figure for Q1' — interictal templates × per-seizure t_onset.

For each target subject, produces one figure:

  ┌───────────────────────────────┬─────────────────────────┬─────────────────────────┐
  │ Interictal templates          │ Most T0-like seizure    │ Most T1-like seizure    │
  │   T0 / T1 mean ± SD rank      │   t_onset_sec stars     │   t_onset_sec stars     │
  │   ★ swap-endpoint channels    │   on swap channels      │   on swap channels      │
  ├───────────────────────────────┴─────────────────────────┼─────────────────────────┤
  │ Δρ per seizure sorted, colored by subtype                │ (ρ_a, ρ_b) scatter      │
  │                                                          │  colored by subtype      │
  └──────────────────────────────────────────────────────────┴─────────────────────────┘

All three top panels share y-axis = swap-endpoint channels, ordered by joint mean
rank (matches the standard pattern from
`scripts/plot_pr6_swap_cluster_rank_multiples.py`).

Seizure t_onset_sec is read directly from atlas v2_3
`channel_onsets[ch].t_onset_sec` for gamma_ER band — the same value that produces
the ✦ stars in `results/data_driven_soz/.../per_seizure/epilepsiae_<sid>_seizure_<idx>.png`.

Pre-figure validation (run separately): channels, T0/T1 cluster IDs, picked
seizures, lagPat alignment all confirmed for {958, 548, 922}.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(REPO))

from src import topic1_topic5_bridge as br  # noqa: E402
from src.interictal_propagation import (  # noqa: E402
    _valid_event_indices,
    load_subject_propagation_events,
)
from src.topic1_topic5_bridge import _morandi_palette  # noqa: E402

RESULTS = REPO / "results"
ARTIFACT_EPI = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
ARTIFACT_YUQ = Path("/mnt/yuquan_data/yuquan_24h_edf")
FIG_DIR = RESULTS / "topic1_topic5_bridge" / "figures"

TARGETS: List[Tuple[str, str]] = [
    ("epilepsiae", "958"),
    ("epilepsiae", "548"),
    ("epilepsiae", "922"),
]
BAND = "gamma_ER"
DETECTION_WINDOW_SEC = (-120.0, 30.0)

PAL = _morandi_palette()
COL_T0 = "#3B6F8A"      # blue
COL_T1 = "#B85450"      # red
COL_TIE = "#A0A0A0"     # grey
COL_NO_ONSET = "#CCCCCC"


def subject_dir(dataset: str, sid: str) -> Path:
    if dataset == "epilepsiae":
        legacy = ARTIFACT_EPI / sid / "all_recs"
        return legacy if legacy.exists() else (ARTIFACT_EPI / sid)
    return ARTIFACT_YUQ / sid


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def _joint_mean_rank_order(
    ranks: np.ndarray, bools: np.ndarray, valid_events: np.ndarray,
    subset_indices: List[int],
) -> List[int]:
    """Return subset_indices sorted by joint mean rank ascending (NaN at end)."""
    mean_rank: Dict[int, float] = {}
    for ci in subset_indices:
        vals = np.asarray(ranks[ci, valid_events], dtype=float)
        mask = np.asarray(bools[ci, valid_events], dtype=bool)
        vv = vals[mask & np.isfinite(vals)]
        if vv.size > 0:
            mean_rank[ci] = float(np.mean(vv))
        else:
            mean_rank[ci] = float("inf")
    return sorted(subset_indices, key=lambda ci: mean_rank[ci])


def _cluster_mean_sd(
    ranks: np.ndarray, bools: np.ndarray, valid_events: np.ndarray,
    labels: np.ndarray, channel_indices: List[int], cluster_id: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n_ch = len(channel_indices)
    means = np.full(n_ch, np.nan, dtype=float)
    stds = np.full(n_ch, np.nan, dtype=float)
    sel = labels == cluster_id
    eidx = valid_events[sel]
    if eidx.size == 0:
        return means, stds
    for k, ci in enumerate(channel_indices):
        vals = np.asarray(ranks[ci, eidx], dtype=float)
        mask = np.asarray(bools[ci, eidx], dtype=bool)
        vv = vals[mask & np.isfinite(vals)]
        if vv.size > 0:
            means[k] = float(np.mean(vv))
            stds[k] = float(np.std(vv))
    return means, stds


def _load_subject_data(dataset: str, sid: str) -> Dict[str, object]:
    """Load all data needed for one subject's figure."""
    swap = br.load_swap_channel_subset(sid, RESULTS, dataset=dataset)
    tmpl = br.load_template_ranks_with_t0t1(sid, RESULTS, ARTIFACT_EPI, dataset=dataset)
    atlas = br.load_atlas_seizure_channel_onsets(sid, BAND, RESULTS, dataset=dataset)
    subtypes = br.load_topic5_subtype_labels(sid, BAND, RESULTS, dataset=dataset)

    # Q1' per-seizure JSON (cached from the cohort run)
    q1p_path = RESULTS / "topic1_topic5_bridge" / "q1prime_per_subject" / f"{dataset}_{sid}__q1prime.json"
    with q1p_path.open() as fh:
        q1p = json.load(fh)

    # lagPat NPZ for mean ± SD
    loaded = load_subject_propagation_events(subject_dir(dataset, sid))
    ranks = np.asarray(loaded["ranks"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=bool)
    channel_names = list(loaded["channel_names"])

    # Topic 1 propagation per_subject JSON for adaptive_cluster.labels
    prop_path = RESULTS / "interictal_propagation" / "per_subject" / f"{dataset}_{sid}.json"
    with prop_path.open() as fh:
        prop = json.load(fh)
    labels = np.asarray(prop["adaptive_cluster"]["labels"], dtype=int)
    valid_events = _valid_event_indices(bools, min_participating=3)
    if labels.size != valid_events.size:
        raise ValueError(
            f"{dataset}_{sid}: labels size {labels.size} != valid_events {valid_events.size}"
        )

    return {
        "swap": swap,
        "tmpl": tmpl,
        "atlas": atlas,
        "subtypes": subtypes,
        "q1p": q1p,
        "ranks": ranks,
        "bools": bools,
        "channel_names": channel_names,
        "labels": labels,
        "valid_events": valid_events,
    }


def _pick_extreme_seizures(per_seizure: List[Dict[str, object]]) -> Tuple[Dict, Dict]:
    """Pick max-Δρ (T0-like) and min-Δρ (T1-like) among valid-assignment seizures."""
    cand = []
    for s in per_seizure:
        if s.get("assignment") not in ("T0", "T1", "tie"):
            continue
        ra, rb = s.get("rho_a"), s.get("rho_b")
        if ra is None or rb is None or not np.isfinite(ra) or not np.isfinite(rb):
            continue
        delta = float(ra) - float(rb)
        cand.append((delta, s))
    if len(cand) < 2:
        raise ValueError("not enough valid seizures to pick extremes")
    cand.sort(key=lambda kv: kv[0])
    return cand[-1][1], cand[0][1]  # most T0-like, most T1-like


def _format_subtype(s: Optional[int]) -> str:
    if s is None:
        return "no label"
    if int(s) == -1:
        return "outlier"
    return f"subtype={int(s)}"


# ---------------------------------------------------------------------------
# Panel drawing
# ---------------------------------------------------------------------------

def _draw_interictal_panel(ax, t0_mean, t0_sd, t1_mean, t1_sd, ch_labels):
    """T0/T1 mean ± SD rank curves; channels are y-axis (top-down)."""
    n_ch = len(ch_labels)
    y_pos = np.arange(n_ch, dtype=float)

    for mean, sd, color, label in [
        (t0_mean, t0_sd, COL_T0, "T0 (forward)"),
        (t1_mean, t1_sd, COL_T1, "T1 (reverse)"),
    ]:
        valid = np.isfinite(mean)
        if not valid.any():
            continue
        ax.fill_betweenx(
            y_pos[valid], (mean - sd)[valid], (mean + sd)[valid],
            color=color, alpha=0.13, linewidth=0,
        )
        ax.plot(mean[valid], y_pos[valid], "-", color=color, lw=2.0, alpha=0.9, zorder=8)
        for k in range(n_ch):
            if not np.isfinite(mean[k]):
                continue
            ax.scatter(
                [mean[k]], [y_pos[k]], marker="*", s=180, color=color,
                edgecolors="black", linewidths=0.6, zorder=12, clip_on=False,
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ch_labels, fontsize=9, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlabel("cluster mean rank")
    ax.set_title("Interictal templates  (mean ± SD over events;  ★ = swap channel)", fontsize=10)
    legend_handles = [
        Line2D([0], [0], color=COL_T0, lw=2.2, label="T0 (forward)"),
        Line2D([0], [0], color=COL_T1, lw=2.2, label="T1 (reverse)"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right", frameon=True)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")


def _draw_seizure_panel(ax, ch_labels, sz_onsets_for_chs, sz_meta_text, accent_color):
    """t_onset_sec stars per swap channel; x = time relative to clinical onset.

    accent_color: single color used for all valid stars (matches T0 or T1 in title,
    so reader can see at a glance which template this seizure resembles).
    Channels with no onset get an × on the right edge.
    """
    n_ch = len(ch_labels)
    y_pos = np.arange(n_ch, dtype=float)
    n_active = 0

    for k, ch in enumerate(ch_labels):
        # Light row guideline (helps the eye scan across)
        ax.axhline(y_pos[k], color="#EAEAEA", lw=0.5, zorder=1)
        t = sz_onsets_for_chs.get(ch)
        if t is None or not np.isfinite(t):
            ax.scatter(
                [DETECTION_WINDOW_SEC[1] + 3], [y_pos[k]], marker="x",
                s=60, color=COL_NO_ONSET, linewidths=1.2, zorder=6, clip_on=False,
            )
            continue
        n_active += 1
        ax.scatter(
            [t], [y_pos[k]], marker="*", s=240, color=accent_color,
            edgecolors="black", linewidths=0.7, zorder=10,
        )

    ax.axvline(0, color="black", lw=1.0, ls="--", zorder=2, label="clin. onset" if n_active else None)
    ax.set_xlim(DETECTION_WINDOW_SEC[0] - 2, DETECTION_WINDOW_SEC[1] + 8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("t rel. clinical onset (s)")
    sz_meta_text_full = sz_meta_text + f"\nn_active={n_active}/{n_ch} swap ch ( × = no onset)"
    ax.set_title(sz_meta_text_full, fontsize=10)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":")


def _draw_delta_bar(ax, q1p, swap_subset_count, subtype_colors):
    """Sorted Δρ bar across all valid-assignment seizures; colored by subtype."""
    rows = []
    for s in q1p["per_seizure"]:
        ra, rb = s.get("rho_a"), s.get("rho_b")
        if ra is None or rb is None or not np.isfinite(ra) or not np.isfinite(rb):
            continue
        rows.append({
            "delta": float(ra) - float(rb),
            "subtype": s.get("subtype_label"),
            "assignment": s.get("assignment"),
        })
    df = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    if df.empty:
        ax.text(0.5, 0.5, "no valid seizures", transform=ax.transAxes, ha="center")
        return
    colors = [subtype_colors.get(s, COL_TIE) for s in df["subtype"]]
    ax.bar(range(len(df)), df["delta"], color=colors, edgecolor="black", lw=0.3)
    ax.axhline(0, color="grey", lw=0.5)
    ax.axhline(+0.10, color="grey", lw=0.4, ls=":")
    ax.axhline(-0.10, color="grey", lw=0.4, ls=":")
    ax.set_ylabel("Δρ = ρ_a − ρ_b")
    ax.set_xlabel(f"seizures (n={len(df)}, sorted)")
    ax.set_xticks([])
    median_d = df["delta"].median()
    counts = df["assignment"].value_counts().to_dict()
    title = (
        f"Δρ per seizure  |  T0={counts.get('T0', 0)}  T1={counts.get('T1', 0)}  "
        f"tie={counts.get('tie', 0)}  median Δρ={median_d:+.3f}  "
        f"|  swap subset = {swap_subset_count} ch"
    )
    ax.set_title(title, fontsize=10)


def _draw_scatter(ax, q1p, subtype_colors):
    df = pd.DataFrame(q1p["per_seizure"])
    df = df[df["rho_a"].notna() & df["rho_b"].notna()]
    if df.empty:
        ax.text(0.5, 0.5, "no valid seizures", transform=ax.transAxes, ha="center")
        return
    labeled = df[df["subtype_label"].notna()]
    subtypes = sorted({int(s) for s in labeled["subtype_label"]})
    for st in subtypes:
        sub = labeled[labeled["subtype_label"] == st]
        col = subtype_colors.get(st, COL_TIE)
        ax.scatter(sub["rho_a"], sub["rho_b"], color=col, s=40, alpha=0.85,
                   edgecolor="black", lw=0.3,
                   label=f"subtype={st}" if st >= 0 else "outlier")
    no_lab = df[df["subtype_label"].isna()]
    if len(no_lab):
        ax.scatter(no_lab["rho_a"], no_lab["rho_b"], color="#D9D9D9", s=40,
                   alpha=0.6, edgecolor="none", label="no label")
    ax.axline((-1, -1), (1, 1), color="grey", ls=":", lw=0.5)
    ax.axline((-1, 1), (1, -1), color="grey", ls="--", lw=0.5)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("ρ_a (Spearman vs T0)")
    ax.set_ylabel("ρ_b (Spearman vs T1)")
    ax.set_title("(ρ_a, ρ_b) per seizure", fontsize=10)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.25, linestyle=":")


# ---------------------------------------------------------------------------
# Main per-subject driver
# ---------------------------------------------------------------------------

def plot_subject(dataset: str, sid: str, out_path: Path) -> None:
    data = _load_subject_data(dataset, sid)
    swap = data["swap"]
    tmpl = data["tmpl"]
    q1p = data["q1p"]
    channel_names = data["channel_names"]
    ranks = data["ranks"]
    bools = data["bools"]
    labels = data["labels"]
    valid_events = data["valid_events"]

    # Resolve swap channels' indices in lagPat ordering, then sort by joint mean rank
    name_to_idx = {ch: i for i, ch in enumerate(channel_names)}
    swap_chs = [ch for ch in swap["endpoint_channels"] if ch in name_to_idx]
    if len(swap_chs) < 3:
        raise ValueError(
            f"{dataset}_{sid}: swap-endpoint ∩ lagPat = {len(swap_chs)} channels (< 3)"
        )
    swap_indices_unordered = [name_to_idx[ch] for ch in swap_chs]
    swap_indices = _joint_mean_rank_order(ranks, bools, valid_events, swap_indices_unordered)
    swap_labels = [channel_names[i] for i in swap_indices]

    # Cluster mean ± SD on swap channels for T0 / T1 cluster IDs (from bridge_setup freeze)
    t0_id = int(tmpl["t0_template_id"])
    t1_id = int(tmpl["t1_template_id"])
    t0_mean, t0_sd = _cluster_mean_sd(ranks, bools, valid_events, labels, swap_indices, t0_id)
    t1_mean, t1_sd = _cluster_mean_sd(ranks, bools, valid_events, labels, swap_indices, t1_id)

    # Pick extreme seizures
    sz_t0_like, sz_t1_like = _pick_extreme_seizures(q1p["per_seizure"])
    sz_t0_id = str(sz_t0_like["seizure_id"])
    sz_t1_id = str(sz_t1_like["seizure_id"])
    sz_t0_delta = float(sz_t0_like["rho_a"]) - float(sz_t0_like["rho_b"])
    sz_t1_delta = float(sz_t1_like["rho_a"]) - float(sz_t1_like["rho_b"])
    sz_t0_subtype = sz_t0_like.get("subtype_label")
    sz_t1_subtype = sz_t1_like.get("subtype_label")

    # Atlas channel_onsets for the picked seizures (only swap channels)
    atlas = data["atlas"]
    sz_t0_onsets = {ch: atlas.get(sz_t0_id, {}).get(ch) for ch in swap_labels}
    sz_t1_onsets = {ch: atlas.get(sz_t1_id, {}).get(ch) for ch in swap_labels}

    # Subtype color palette (per subject — subtype labels are subject-internal)
    raw_subs = [s.get("subtype_label") for s in q1p["per_seizure"]]
    subtypes_present = sorted({
        int(s) for s in raw_subs if s is not None and not (isinstance(s, float) and np.isnan(s))
    })
    subtype_colors: Dict[int, str] = {}
    for k, st in enumerate(subtypes_present):
        if st == -1:
            subtype_colors[st] = "#8C8C8C"  # outlier = grey
        else:
            subtype_colors[st] = PAL[k % len(PAL)]

    # Build figure
    fig = plt.figure(figsize=(15.5, 9.5), dpi=150, facecolor="white")
    gs = GridSpec(
        nrows=2, ncols=3, figure=fig,
        height_ratios=[3.0, 1.6], width_ratios=[1.0, 1.0, 1.0],
        left=0.06, right=0.97, top=0.88, bottom=0.07,
        hspace=0.42, wspace=0.16,
    )

    ax_inter = fig.add_subplot(gs[0, 0])
    ax_t0sz = fig.add_subplot(gs[0, 1], sharey=ax_inter)
    ax_t1sz = fig.add_subplot(gs[0, 2], sharey=ax_inter)
    ax_bar = fig.add_subplot(gs[1, 0:2])
    ax_scatter = fig.add_subplot(gs[1, 2])

    _draw_interictal_panel(ax_inter, t0_mean, t0_sd, t1_mean, t1_sd, swap_labels)
    _draw_seizure_panel(
        ax_t0sz, swap_labels, sz_t0_onsets,
        f"Most T0-like seizure (max Δρ)\n"
        f"sz={sz_t0_id[-6:]}  Δρ={sz_t0_delta:+.2f}  {_format_subtype(sz_t0_subtype)}",
        accent_color=COL_T0,
    )
    _draw_seizure_panel(
        ax_t1sz, swap_labels, sz_t1_onsets,
        f"Most T1-like seizure (min Δρ)\n"
        f"sz={sz_t1_id[-6:]}  Δρ={sz_t1_delta:+.2f}  {_format_subtype(sz_t1_subtype)}",
        accent_color=COL_T1,
    )
    _draw_delta_bar(ax_bar, q1p, len(swap_labels), subtype_colors)
    _draw_scatter(ax_scatter, q1p, subtype_colors)

    suptitle = (
        f"{dataset}_{sid}  |  swap_class={swap['swap_class']}  "
        f"(decision_k={swap['decision_k']})  |  T0=cluster_id {t0_id}  T1=cluster_id {t1_id}  "
        f"|  band={BAND}  |  topic5 status={q1p.get('topic5_status', '?')}  "
        f"n_subtypes={q1p.get('topic5_n_subtypes', '?')}"
    )
    fig.suptitle(suptitle, fontsize=11)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"figure → {out_path}")


def main() -> None:
    for ds, sid in TARGETS:
        out = FIG_DIR / f"q1prime_bridge_subject_{ds}_{sid}.png"
        try:
            plot_subject(ds, sid, out)
        except Exception as e:
            print(f"[ERROR] {ds}_{sid}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
