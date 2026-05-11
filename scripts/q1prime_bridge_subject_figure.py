"""Per-subject bridge figure for Q1' — interictal templates × per-seizure z-ER heatmap.

Layout (gridspec 2×3):

  TOP:
    [1] Interictal templates  — drawn by `_draw_subject_panel`
        from scripts/plot_pr6_swap_cluster_rank_multiples.py
        (the canonical project standard: T0/T1 mean ± SD rank curves,
        ★ at swap-endpoint channels, ○ at non-swap, bold y-labels for swap).
    [2] Most-T0-like seizure z-ER heatmap zoomed to swap channels × [-30, +20]s
    [3] Most-T1-like seizure z-ER heatmap, same zoom

  BOTTOM:
    [Δρ per seizure sorted bar, colored by ictal subtype]
    [(ρ_a, ρ_b) scatter, colored by ictal subtype]

The right two heatmaps reuse the same z-ER computation pipeline as
scripts/plot_ictal_er_atlas.py::render_per_seizure
(extract_seizure_window → compute_er → baseline_zscore_er),
giving values directly comparable to
results/data_driven_soz/.../atlas_v2_3/figures/per_seizure/epilepsiae_<sid>_seizure_<idx>.png.

Each picked seizure's title shows the atlas seizure_idx so cross-ref with
the per_seizure PNG is direct.
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
import numpy as np
import pandas as pd

REPO = Path("/home/honglab/leijiaxin/HFOsp")
sys.path.insert(0, str(REPO))

from src import topic1_topic5_bridge as br  # noqa: E402
from src.ictal_onset_extraction import (  # noqa: E402
    GAMMA_ER_BANDS,
    baseline_zscore_er,
    compute_er,
    extract_seizure_window,
    resolve_baseline_window,
)
from src.plot_style import (  # noqa: E402
    COL_CLUSTER_T0,
    COL_CLUSTER_T1,
    COL_SWAP_LABEL,
    FS_LABEL,
    FS_TICK,
    FS_TITLE,
    savefig_pub,
    style_panel,
)
from src.topic1_topic5_bridge import _morandi_palette  # noqa: E402
from scripts.plot_pr6_swap_cluster_rank_multiples import (  # noqa: E402
    _draw_subject_panel,
    _swap_nodes_rank_displacement,
)
from scipy import stats as sp_stats  # noqa: E402

RESULTS = REPO / "results"
FIG_DIR = RESULTS / "topic1_topic5_bridge" / "figures"

TARGETS: List[Tuple[str, str]] = [
    ("epilepsiae", "958"),
    ("epilepsiae", "548"),
    ("epilepsiae", "922"),
]
BAND = "gamma_ER"
ZOOM_T = (-30.0, 20.0)  # seconds rel. clinical onset for zoom heatmap

# Manual override for T0/T1 picks per subject — keyed by (dataset, sid).
# Set to None to use automatic strict-swap picker.
# Values are seizure_id strings (matching atlas seizure_records).
MANUAL_PICK_OVERRIDES: Dict[Tuple[str, str], Dict[str, Optional[str]]] = {
    # 548: auto picker chose sz_idx=26 (Δρ_strict=-1.10, subtype=2) for T1;
    # user requested a different example. Override to sz_idx=21
    # (sz_id=54801302102, Δρ_strict=-1.00, subtype=1, n_strict=5/5) — almost
    # equal T1 strength, full strict-swap coverage, different subtype.
    ("epilepsiae", "548"): {"t1": "54801302102"},
}

# ER computation constants (match atlas v2_3 pipeline)
ER_PRE_SEC = 300.0
ER_POST_SEC = 60.0
ER_HOP_SEC = 0.1
ER_WIN_SEC = 1.0

PAL = _morandi_palette()
COL_TIE = "#A0A0A0"


# ---------------------------------------------------------------------------
# Per-seizure z-ER computation (mirrors atlas v2_3 / plot_ictal_er_atlas)
# ---------------------------------------------------------------------------

def _compute_seizure_zer(
    dataset: str, sid: str, seizure_idx: int,
) -> Dict[str, object]:
    """Run extract → compute_er → baseline_zscore on one seizure.

    Tries decreasing post_sec windows to handle seizures near block boundaries
    (same fallback ladder as plot_ictal_er_atlas.py::render_per_seizure).

    Returns:
      zer        : (n_channels, n_frames) z-scored ER
      ch_names   : list of channels matching zer rows
      t_er       : (n_frames,) seconds relative to clinical onset
      eeg_rel    : eeg onset (sec) relative to clinical onset, or None
      post_sec   : actual post_sec used (for diagnostic)
    """
    sw = None
    last_exc: Optional[Exception] = None
    used_post = None
    for post_attempt in (ER_POST_SEC, 30.0, 15.0):
        try:
            sw = extract_seizure_window(
                f"{dataset}/{sid}", seizure_idx,
                pre_sec=ER_PRE_SEC, post_sec=post_attempt,
                results_root=RESULTS, reference="car",
            )
            used_post = post_attempt
            break
        except (ValueError, IndexError) as exc:
            last_exc = exc
            continue
    if sw is None:
        raise RuntimeError(
            f"{dataset}/{sid} seizure {seizure_idx}: window extraction failed "
            f"across post_sec fallback (last: {last_exc})"
        )
    er = compute_er(
        sw.signal, fs=sw.fs,
        fast_band=GAMMA_ER_BANDS["fast"],
        slow_band=GAMMA_ER_BANDS["slow"],
        win_sec=ER_WIN_SEC, hop_sec=ER_HOP_SEC,
    )
    eeg_rel = (
        sw.eeg_onset_epoch - sw.clin_onset_epoch
        if sw.eeg_onset_epoch is not None else None
    )
    bl = resolve_baseline_window(
        n_time_frames=er.shape[1], hop_sec=ER_HOP_SEC,
        pre_sec=ER_PRE_SEC, eeg_onset_rel_sec=eeg_rel,
    )
    zer = baseline_zscore_er(
        er, (bl.start_idx, bl.end_idx), hop_sec=ER_HOP_SEC,
    )
    # ER frame i centered at: (i * hop) + win/2 - pre_sec
    n_frames = zer.shape[1]
    t_er = np.arange(n_frames) * ER_HOP_SEC + ER_WIN_SEC / 2.0 - ER_PRE_SEC
    return {
        "zer": zer,
        "ch_names": list(sw.ch_names),
        "t_er": t_er,
        "eeg_rel": eeg_rel,
        "post_sec_used": used_post,
    }


# ---------------------------------------------------------------------------
# Seizure-side panel: zoom z-ER heatmap on swap channels
# ---------------------------------------------------------------------------

def _draw_seizure_heatmap(
    ax,
    zer: np.ndarray,
    ch_names_zer: List[str],
    t_er: np.ndarray,
    swap_channels_ordered: List[str],
    swap_nodes_strict: set,
    channel_onsets: Dict[str, Optional[float]],
    eeg_rel: Optional[float],
    title_text: str,
    vmax: float = 3.0,
):
    """z-ER heatmap rows = swap_channels_ordered, x = t_er zoomed.

    y-tick styling matches left interictal panel: bold orange (COL_SWAP_LABEL)
    only for channels in `swap_nodes_strict`; others muted grey.
    Axes box forced square via ax.set_box_aspect(1.0).
    """
    name_to_row = {ch: i for i, ch in enumerate(ch_names_zer)}
    rows: List[int] = []
    missing: List[int] = []
    for plot_idx, ch in enumerate(swap_channels_ordered):
        if ch in name_to_row:
            rows.append(name_to_row[ch])
        else:
            missing.append(plot_idx)
            rows.append(-1)

    t_mask = (t_er >= ZOOM_T[0]) & (t_er <= ZOOM_T[1])
    t_slice = t_er[t_mask]
    if t_slice.size < 2:
        ax.text(0.5, 0.5, "zoom window has no ER frames",
                transform=ax.transAxes, ha="center")
        return None

    n_swap = len(swap_channels_ordered)
    mat = np.full((n_swap, t_slice.size), np.nan, dtype=float)
    for k, r in enumerate(rows):
        if r >= 0:
            mat[k] = zer[r, t_mask]

    extent = (float(t_slice[0]), float(t_slice[-1]), n_swap - 0.5, -0.5)
    im = ax.imshow(
        mat, aspect="auto", interpolation="nearest",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=extent, origin="upper",
    )

    y_pos = np.arange(n_swap, dtype=float)
    for k, ch in enumerate(swap_channels_ordered):
        t = channel_onsets.get(ch)
        if t is None or not np.isfinite(t):
            continue
        if not (ZOOM_T[0] <= t <= ZOOM_T[1]):
            continue
        ax.scatter([t], [y_pos[k]], marker="*", s=200, color="white",
                   edgecolors="black", linewidths=1.0, zorder=10)

    for k in missing:
        ax.scatter([ZOOM_T[1] + 1.0], [y_pos[k]], marker="x", s=60,
                   color="#666", clip_on=False, zorder=8)

    ax.axvline(0.0, color="black", lw=1.4, ls="--", zorder=3)
    if eeg_rel is not None and ZOOM_T[0] <= eeg_rel <= ZOOM_T[1]:
        ax.axvline(eeg_rel, color="#444", lw=1.0, ls=":", zorder=3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(swap_channels_ordered, fontsize=FS_TICK)
    # Match left panel: bold orange ONLY for strict swap nodes
    for tick, ch in zip(ax.get_yticklabels(), swap_channels_ordered):
        if ch in swap_nodes_strict:
            tick.set_fontweight("bold")
            tick.set_color(COL_SWAP_LABEL)
        else:
            tick.set_color("#888888")
    ax.set_xlabel("t rel. clin. onset (s)", fontsize=FS_LABEL)
    ax.set_xlim(ZOOM_T[0], ZOOM_T[1] + 1.5)
    ax.set_title(title_text, fontsize=FS_TITLE, pad=10)
    # Square box (plot_style "保证方型" request)
    ax.set_box_aspect(1.0)
    return im


# ---------------------------------------------------------------------------
# Bottom summary panels
# ---------------------------------------------------------------------------

def _draw_delta_bar(ax, per_seizure, swap_subset_count, subtype_colors):
    rows = []
    for s in per_seizure:
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
        return None
    colors = [subtype_colors.get(s, COL_TIE) for s in df["subtype"]]
    bars = ax.bar(range(len(df)), df["delta"], color=colors,
                  edgecolor="black", lw=0.4)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axhline(+0.10, color="grey", lw=0.5, ls=":")
    ax.axhline(-0.10, color="grey", lw=0.5, ls=":")
    ax.set_ylabel("Δρ = ρ(T0) − ρ(T1)", fontsize=FS_LABEL)
    ax.set_xlabel(f"seizures (n={len(df)}, sorted)", fontsize=FS_LABEL)
    ax.set_xticks([])
    median_d = df["delta"].median()
    counts = df["assignment"].value_counts().to_dict()
    title = (
        f"Δρ per seizure   T0={counts.get('T0', 0)}   T1={counts.get('T1', 0)}   "
        f"tie={counts.get('tie', 0)}   median={median_d:+.3f}   "
        f"swap={swap_subset_count} ch"
    )
    ax.set_title(title, fontsize=FS_TITLE, pad=10)
    return bars


def _draw_scatter(ax, per_seizure, subtype_colors):
    df = pd.DataFrame(per_seizure)
    df = df[df["rho_a"].notna() & df["rho_b"].notna()]
    if df.empty:
        ax.text(0.5, 0.5, "no valid seizures", transform=ax.transAxes, ha="center")
        return
    labeled = df[df["subtype_label"].notna()]
    subtypes = sorted({int(s) for s in labeled["subtype_label"]})
    for st in subtypes:
        sub = labeled[labeled["subtype_label"] == st]
        col = subtype_colors.get(st, COL_TIE)
        ax.scatter(sub["rho_a"], sub["rho_b"], color=col, s=60, alpha=0.85,
                   edgecolor="black", lw=0.4)
    no_lab = df[df["subtype_label"].isna()]
    if len(no_lab):
        ax.scatter(no_lab["rho_a"], no_lab["rho_b"], color="#D9D9D9", s=60,
                   alpha=0.6, edgecolor="none")
    ax.axline((-1, -1), (1, 1), color="grey", ls=":", lw=0.6)
    ax.axline((-1, 1), (1, -1), color="grey", ls="--", lw=0.6)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("ρ vs T0", fontsize=FS_LABEL)
    ax.set_ylabel("ρ vs T1", fontsize=FS_LABEL)
    ax.set_title("per-seizure (ρ_T0, ρ_T1)", fontsize=FS_TITLE, pad=10)
    ax.grid(True, alpha=0.25, linestyle=":")


# ---------------------------------------------------------------------------
# Per-subject driver
# ---------------------------------------------------------------------------

def _seizure_idx_from_id(atlas_subj_json: dict, band: str, sz_id: str) -> Optional[int]:
    """Look up atlas seizure_idx for a given seizure_id string."""
    sr = atlas_subj_json["per_er"][band]["seizure_records"]
    for rec in sr:
        if str(rec.get("seizure_id")) == sz_id:
            return int(rec.get("seizure_idx", -1))
    return None


def _compute_strict_swap_alignment(
    atlas: Dict[str, Dict[str, Optional[float]]],
    t0_rank: Dict[str, int],
    t1_rank: Dict[str, int],
    strict_channels: Sequence[str],
    min_channels: int = 3,
) -> Dict[str, Dict[str, object]]:
    """Recompute (ρ_T0, ρ_T1, Δρ, n_used) per seizure using ONLY strict swap nodes.

    The original Q1' computation used the *endpoint* set (top-k ∪ bottom-k of T0_rank),
    which includes channels extreme in T0 alone. The cleaner Q1' axis restricts to
    *strict swap* nodes: (top-k T0 ∩ bottom-k T1) ∪ (bottom-k T0 ∩ top-k T1).

    Returns dict[sz_id → {rho_t0, rho_t1, delta_rho, n_strict_used, assignment}].
    """
    out: Dict[str, Dict[str, object]] = {}
    for sz_id, sz_onsets in atlas.items():
        valid_chs = [
            ch for ch in strict_channels
            if ch in t0_rank and ch in t1_rank
            and ch in sz_onsets and sz_onsets[ch] is not None
            and np.isfinite(sz_onsets[ch])
        ]
        n = len(valid_chs)
        if n < min_channels:
            out[sz_id] = {
                "rho_t0": float("nan"), "rho_t1": float("nan"),
                "delta_rho": float("nan"), "n_strict_used": n,
                "assignment": "insufficient_n",
            }
            continue
        onset_vec = np.array([sz_onsets[ch] for ch in valid_chs], dtype=float)
        seizure_rank = np.argsort(np.argsort(onset_vec)).astype(float)
        t0_vec = np.array([t0_rank[ch] for ch in valid_chs], dtype=float)
        t1_vec = np.array([t1_rank[ch] for ch in valid_chs], dtype=float)
        rho_t0 = sp_stats.spearmanr(seizure_rank, t0_vec).statistic
        rho_t1 = sp_stats.spearmanr(seizure_rank, t1_vec).statistic
        if not np.isfinite(rho_t0):
            rho_t0 = 0.0
        if not np.isfinite(rho_t1):
            rho_t1 = 0.0
        diff = float(rho_t0 - rho_t1)
        if diff > 0.10:
            assg = "T0"
        elif -diff > 0.10:
            assg = "T1"
        else:
            assg = "tie"
        out[sz_id] = {
            "rho_t0": float(rho_t0),
            "rho_t1": float(rho_t1),
            "delta_rho": diff,
            "n_strict_used": n,
            "assignment": assg,
        }
    return out


def _pick_extreme_seizures_strict(
    per_seizure: List[Dict[str, object]],
    strict_alignments: Dict[str, Dict[str, object]],
    n_strict: int,
) -> Tuple[Dict, Dict]:
    """Pick max/min Δρ using STRICT-swap-only alignments.

    Gate: n_strict_used >= ceil(n_strict * 0.75), fallback to ≥ max(3, n_strict//2).
    Within gate: prefer max n_strict_used, then Δρ extremity.
    """
    def _build(min_n: int):
        t0p, t1p = [], []
        for s in per_seizure:
            sid = str(s.get("seizure_id"))
            al = strict_alignments.get(sid)
            if al is None or al["assignment"] not in ("T0", "T1", "tie"):
                continue
            n_used = int(al["n_strict_used"])
            if n_used < min_n:
                continue
            delta = float(al["delta_rho"])
            entry = (n_used, delta, s, al)
            (t0p if delta > 0 else t1p).append(entry)
        return t0p, t1p

    gate_strict = max(3, (n_strict * 3 + 3) // 4)
    t0p, t1p = _build(gate_strict)
    if not t0p or not t1p:
        gate_loose = max(3, n_strict // 2)
        t0p, t1p = _build(gate_loose)
    if not t0p or not t1p:
        raise ValueError(
            f"strict-swap alignment: not enough candidates "
            f"(n_strict={n_strict}, t0_pool={len(t0p)}, t1_pool={len(t1p)})"
        )
    t0p.sort(key=lambda r: (-r[0], -r[1]))   # most strict-swap stars, then max +Δρ
    t1p.sort(key=lambda r: (-r[0], r[1]))    # most strict-swap stars, then most -Δρ
    return t0p[0], t1p[0]


def _pick_extreme_seizures(
    per_seizure: List[Dict[str, object]], swap_size: int,
) -> Tuple[Dict, Dict]:
    """Pick representative T0-like / T1-like seizures.

    Primary criterion: maximum n_swap_channels_used (most CUSUM stars on
    swap-endpoint channels — so the right-side heatmap shows a real
    swap-node pattern, not mostly × markers).
    Secondary: |Δρ| in the appropriate direction.

    Gate: n_swap ≥ ceil(swap_size * 0.75). If too few pass, fall back to ≥ swap_size // 2.
    """
    def _build_cand(min_n: int):
        t0_pool: List[Tuple[int, float, dict]] = []
        t1_pool: List[Tuple[int, float, dict]] = []
        for s in per_seizure:
            if s.get("assignment") not in ("T0", "T1", "tie"):
                continue
            ra, rb = s.get("rho_a"), s.get("rho_b")
            if ra is None or rb is None or not np.isfinite(ra) or not np.isfinite(rb):
                continue
            n_used = int(s.get("n_swap_channels_used", 0))
            if n_used < min_n:
                continue
            delta = float(ra) - float(rb)
            (t0_pool if delta > 0 else t1_pool).append((n_used, delta, s))
        return t0_pool, t1_pool

    gate_strict = max(3, (swap_size * 3 + 3) // 4)  # ceil(swap_size * 0.75)
    t0_pool, t1_pool = _build_cand(gate_strict)
    if not t0_pool or not t1_pool:
        gate_loose = max(3, swap_size // 2)
        t0_pool, t1_pool = _build_cand(gate_loose)
        used_gate = gate_loose
    else:
        used_gate = gate_strict
    if not t0_pool or not t1_pool:
        raise ValueError(
            f"not enough valid seizures across both directions "
            f"(swap_size={swap_size}, gate={used_gate})"
        )
    # Prefer stars-rich first, then Δρ extremity
    t0_pool.sort(key=lambda r: (-r[0], -r[1]))   # max n_swap, max +Δρ
    t1_pool.sort(key=lambda r: (-r[0], r[1]))    # max n_swap, min Δρ (most negative)
    return t0_pool[0][2], t1_pool[0][2]


def _format_subtype(s: Optional[int]) -> str:
    if s is None:
        return "no topic5 label"
    if int(s) == -1:
        return "outlier"
    return f"subtype={int(s)}"


def plot_subject(dataset: str, sid: str, out_path: Path) -> None:
    # --- load metadata ---
    swap = br.load_swap_channel_subset(sid, RESULTS, dataset=dataset)
    tmpl = br.load_template_ranks_with_t0t1(
        sid, RESULTS, Path("/dev/null"), dataset=dataset,
    )
    q1p_path = (
        RESULTS / "topic1_topic5_bridge" / "q1prime_per_subject"
        / f"{dataset}_{sid}__q1prime.json"
    )
    with q1p_path.open() as fh:
        q1p = json.load(fh)
    atlas_subj = json.load(
        (RESULTS / "data_driven_soz" / "layer_a_ictal_er_rank"
         / "per_subject" / f"{dataset}_{sid}.json").open()
    )

    # Swap channels in their joint-rank order — get it from the same path
    # the left panel will use (_fixed_channel_order via _draw_subject_panel),
    # but we need it for the right heatmap rows. Re-derive here.
    from src.interictal_propagation import (
        _valid_event_indices, load_subject_propagation_events,
    )
    if dataset == "epilepsiae":
        subj_dir = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns") / sid / "all_recs"
        if not subj_dir.exists():
            subj_dir = subj_dir.parent
    else:
        subj_dir = Path("/mnt/yuquan_data/yuquan_24h_edf") / sid
    loaded = load_subject_propagation_events(subj_dir)
    ranks = np.asarray(loaded["ranks"], dtype=float)
    bools = np.asarray(loaded["bools"], dtype=bool)
    channel_names = list(loaded["channel_names"])
    valid_events = _valid_event_indices(bools, min_participating=3)

    # joint mean rank order for swap channels
    name_to_idx = {ch: i for i, ch in enumerate(channel_names)}
    swap_chs_present = [ch for ch in swap["endpoint_channels"] if ch in name_to_idx]
    swap_idx_unordered = [name_to_idx[ch] for ch in swap_chs_present]

    def _mean_rank(ci):
        vals = ranks[ci, valid_events]
        mask = bools[ci, valid_events]
        vv = vals[mask & np.isfinite(vals)]
        return float(vv.mean()) if vv.size else float("inf")
    swap_idx_ordered = sorted(swap_idx_unordered, key=_mean_rank)
    swap_labels = [channel_names[i] for i in swap_idx_ordered]

    # --- Strict swap node set (from rank_displacement §8) ---
    swap_nodes_set, _cid_a, _cid_b, _decision_k = _swap_nodes_rank_displacement(
        dataset, sid,
    )
    strict_labels = [ch for ch in swap_labels if ch in swap_nodes_set]
    if len(strict_labels) < 3:
        # Fall back to full endpoint when strict tier is too small (e.g. swap=none subjects)
        strict_labels = list(swap_labels)
        used_strict_fallback = True
    else:
        used_strict_fallback = False
    n_strict = len(strict_labels)

    # Load atlas channel_onsets keyed by seizure_id (string) for alignment
    atlas_by_sz_id = br.load_atlas_seizure_channel_onsets(
        sid, BAND, RESULTS, dataset=dataset,
    )

    # --- Recompute alignment on STRICT swap channels only ---
    strict_alignments = _compute_strict_swap_alignment(
        atlas=atlas_by_sz_id,
        t0_rank=tmpl["t0_rank"],
        t1_rank=tmpl["t1_rank"],
        strict_channels=strict_labels,
        min_channels=3,
    )
    (sz_t0_n, sz_t0_dr, sz_t0, sz_t0_al), (sz_t1_n, sz_t1_dr, sz_t1, sz_t1_al) = \
        _pick_extreme_seizures_strict(q1p["per_seizure"], strict_alignments, n_strict)

    # Optional manual override (per-subject t0/t1 sz_id specified at top of file)
    override = MANUAL_PICK_OVERRIDES.get((dataset, sid), {})
    for side in ("t0", "t1"):
        forced_id = override.get(side)
        if forced_id is None:
            continue
        forced_seiz = next(
            (s for s in q1p["per_seizure"] if str(s.get("seizure_id")) == forced_id),
            None,
        )
        forced_al = strict_alignments.get(forced_id)
        if forced_seiz is None or forced_al is None:
            raise RuntimeError(
                f"{dataset}_{sid}: manual override {side}={forced_id} not found"
            )
        replacement = (
            int(forced_al["n_strict_used"]),
            float(forced_al["delta_rho"]),
            forced_seiz,
            forced_al,
        )
        if side == "t0":
            sz_t0_n, sz_t0_dr, sz_t0, sz_t0_al = replacement
        else:
            sz_t1_n, sz_t1_dr, sz_t1, sz_t1_al = replacement

    sz_t0_id = str(sz_t0["seizure_id"])
    sz_t1_id = str(sz_t1["seizure_id"])
    sz_t0_idx = _seizure_idx_from_id(atlas_subj, BAND, sz_t0_id)
    sz_t1_idx = _seizure_idx_from_id(atlas_subj, BAND, sz_t1_id)
    if sz_t0_idx is None or sz_t1_idx is None:
        raise RuntimeError(
            f"{dataset}_{sid}: could not resolve seizure_idx "
            f"(t0={sz_t0_idx} for sz_id={sz_t0_id}, t1={sz_t1_idx} for sz_id={sz_t1_id})"
        )

    # per-channel onsets for each picked seizure (gamma band)
    sr_dict = {
        int(rec["seizure_idx"]): {
            ch: (entry or {}).get("t_onset_sec")
            for ch, entry in (rec.get("channel_onsets") or {}).items()
        }
        for rec in atlas_subj["per_er"][BAND]["seizure_records"]
    }
    onsets_t0 = sr_dict.get(sz_t0_idx, {})
    onsets_t1 = sr_dict.get(sz_t1_idx, {})

    # --- compute z-ER for both seizures ---
    print(f"  [zer] computing for {dataset}_{sid} sz_idx={sz_t0_idx} (T0-like)...")
    zer_t0 = _compute_seizure_zer(dataset, sid, sz_t0_idx)
    print(f"  [zer] computing for {dataset}_{sid} sz_idx={sz_t1_idx} (T1-like)...")
    zer_t1 = _compute_seizure_zer(dataset, sid, sz_t1_idx)

    # T0/T1 cluster IDs from bridge_setup freeze (larger-fraction rule)
    t0_id = int(tmpl["t0_template_id"])
    t1_id = int(tmpl["t1_template_id"])

    # subtype colors
    raw_subs = [s.get("subtype_label") for s in q1p["per_seizure"]]
    subtypes_present = sorted({
        int(s) for s in raw_subs
        if s is not None and not (isinstance(s, float) and np.isnan(s))
    })
    subtype_colors: Dict[int, str] = {}
    for k, st in enumerate(subtypes_present):
        if st == -1:
            subtype_colors[st] = "#8C8C8C"
        else:
            subtype_colors[st] = PAL[k % len(PAL)]

    # --- build figure (plot_style.py conventions) ---
    # Top row: 3 panels — slim interictal rank distribution + 2 SQUARE heatmaps.
    # Bottom row: independent gridspec — wide Δρ bar + scatter, widths free.
    # Colorbar pinned to T1 panel; legend far-right edge.
    fig = plt.figure(figsize=(20.5, 11.0), facecolor="white")

    # Top GridSpec — heatmaps square via set_box_aspect(1.0); rank panel stays slim
    gs_top = GridSpec(
        nrows=1, ncols=3, figure=fig,
        width_ratios=[0.55, 1.0, 1.0],
        left=0.045, right=0.78, top=0.94, bottom=0.50,
        wspace=0.40,
    )
    ax_inter = fig.add_subplot(gs_top[0, 0])
    ax_t0sz = fig.add_subplot(gs_top[0, 1])
    ax_t1sz = fig.add_subplot(gs_top[0, 2])

    # Bottom GridSpec — independent of top column widths
    gs_bot = GridSpec(
        nrows=1, ncols=2, figure=fig,
        width_ratios=[2.2, 1.0],
        left=0.07, right=0.78, top=0.40, bottom=0.07,
        wspace=0.28,
    )
    ax_bar = fig.add_subplot(gs_bot[0, 0])
    ax_scatter = fig.add_subplot(gs_bot[0, 1])

    # Left: canonical interictal template panel (NOT forced square — keep slim)
    _draw_subject_panel(
        ax_inter, dataset, sid,
        swap_nodes=swap_nodes_set,
        cid_t0=t0_id, cid_t1=t1_id,
        title_text="",
    )
    n_lagpat = len(channel_names)
    ax_inter.set_title(
        f"E:{sid}\ninterictal templates   n_swap={len(swap_nodes_set)}",
        fontsize=FS_TITLE, pad=10,
    )
    ax_inter.set_xlabel("rank", fontsize=FS_LABEL)
    ax_inter.set_xlim(0, n_lagpat - 1)
    ax_inter.set_ylim(n_lagpat - 0.5, -0.5)
    # NO set_box_aspect here — slim rectangle by default

    # Middle/right: z-ER zoom heatmap restricted to STRICT SWAP rows only
    # (per user: "T0/T1 不应该只考虑swap节点之间的时序" — ρ on strict swap only).
    # Channel highlight rule is moot (all rows ARE strict swap), but keep
    # bold-orange styling consistent with the left panel.
    im_t0 = _draw_seizure_heatmap(
        ax_t0sz, zer_t0["zer"], zer_t0["ch_names"], zer_t0["t_er"],
        strict_labels, set(strict_labels), onsets_t0, zer_t0["eeg_rel"],
        title_text=(
            f"most T0-like   sz_idx={sz_t0_idx:02d}   subtype={_format_subtype(sz_t0.get('subtype_label'))}\n"
            f"Δρ_strict={sz_t0_dr:+.2f}   n_strict={sz_t0_n}/{n_strict}"
            + ("   (strict tier empty → fallback endpoint)" if used_strict_fallback else "")
        ),
    )
    im_t1 = _draw_seizure_heatmap(
        ax_t1sz, zer_t1["zer"], zer_t1["ch_names"], zer_t1["t_er"],
        strict_labels, set(strict_labels), onsets_t1, zer_t1["eeg_rel"],
        title_text=(
            f"most T1-like   sz_idx={sz_t1_idx:02d}   subtype={_format_subtype(sz_t1.get('subtype_label'))}\n"
            f"Δρ_strict={sz_t1_dr:+.2f}   n_strict={sz_t1_n}/{n_strict}"
            + ("   (strict tier empty → fallback endpoint)" if used_strict_fallback else "")
        ),
    )

    # Apply style_panel (no-op for already-set spines but unifies tick params)
    for ax in (ax_inter, ax_t0sz, ax_t1sz, ax_bar, ax_scatter):
        style_panel(ax)

    # Bottom (uses original Q1' endpoint-Δρ for cohort view — that's the
    # subject-summary axis, not the strict-swap axis used in the seizure panels).
    _draw_delta_bar(ax_bar, q1p["per_seizure"], len(swap_labels), subtype_colors)
    _draw_scatter(ax_scatter, q1p["per_seizure"], subtype_colors)

    # Single shared legend on right (plot_style §6)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_handles = [
        Line2D([0], [0], color=COL_CLUSTER_T0, lw=2.5, marker="*",
               markersize=11, markeredgecolor="black",
               label="T0 mean ± SD"),
        Line2D([0], [0], color=COL_CLUSTER_T1, lw=2.5, marker="*",
               markersize=11, markeredgecolor="black",
               label="T1 mean ± SD"),
        Line2D([0], [0], marker="*", color="white", markersize=12,
               markeredgecolor="black", markeredgewidth=1.0, lw=0,
               label="CUSUM ✦"),
        Line2D([0], [0], marker="x", color="#666", lw=0, markersize=8,
               label="no onset"),
        Line2D([0], [0], color="black", ls="--", lw=1.4, label="clin. onset"),
        Line2D([0], [0], color="#444", ls=":", lw=1.0, label="EEG onset"),
    ]
    # Subtype color swatches
    for st in sorted(subtype_colors.keys()):
        lbl = "outlier" if st == -1 else f"subtype {st}"
        legend_handles.append(Patch(facecolor=subtype_colors[st],
                                    edgecolor="black", label=lbl))

    # Legend block: top-right corner, anchored to figure (further right per user)
    fig.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(0.86, 0.95),
        ncol=1, frameon=False, fontsize=FS_TICK,
    )

    # Colorbar pinned to T1 heatmap's right edge via inset_axes (auto-aligns
    # whatever the actual axes position is — no manual positioning drift)
    if im_t1 is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        cax = inset_axes(
            ax_t1sz,
            width="4%", height="100%",
            loc="lower left",
            bbox_to_anchor=(1.04, 0.0, 1.0, 1.0),
            bbox_transform=ax_t1sz.transAxes,
            borderpad=0,
        )
        cb = fig.colorbar(im_t1, cax=cax)
        cb.set_label("z-ER (gamma)", fontsize=FS_LABEL)
        cb.ax.tick_params(labelsize=FS_TICK)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save PDF first, then PNG (savefig_pub closes figure) — plot_style §8
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    savefig_pub(fig, out_path, dpi=200)
    print(f"figure → {out_path}  +  {pdf_path}")


def main() -> None:
    for ds, sid in TARGETS:
        out = FIG_DIR / f"q1prime_bridge_subject_{ds}_{sid}.png"
        try:
            plot_subject(ds, sid, out)
        except Exception as e:
            print(f"[ERROR] {ds}_{sid}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
