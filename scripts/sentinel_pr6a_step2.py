"""PR-6-A Step 2 sentinel driver.

Loads ictal-onset windows for the two PR-6-A sentinel subjects, runs the
ER + baseline z-score primitives from :mod:`src.ictal_onset_extraction`
under both gamma_ER and broad_ER band configurations, and writes
multi-channel z-ER trace figures that the user inspects to confirm SOZ
channels show a clear z-ER rise around clinical onset.

Sentinel cohort (locked in archive plan §3 末尾):
    - sentinel_A: epilepsiae/548  — k=2, in PR-5-A retained main, in
      forward/reverse subset, n_seizures=31 (highest among the 9-subset).
    - sentinel_B: epilepsiae/916  — k=2, in PR-5-A retained main,
      ordinary k=2 (NOT in forward/reverse subset), strong grade,
      n_seizures=51.

Output:
    results/interictal_propagation/ictal_alignment/_sentinel_step2/
        <subject>_<seizure_idx>_<gamma_ER|broad_ER>.png
        sentinel_step2_summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ictal_onset_extraction import (
    BROAD_ER_BANDS,
    GAMMA_ER_BANDS,
    baseline_window_indices,
    baseline_zscore_er,
    compute_er,
    extract_seizure_window,
)


SENTINEL_COHORT = [
    {
        "key": "sentinel_A",
        "subject": "epilepsiae/548",
        "rationale": "k=2, PR-5-A retained main, in 9-subset forward/reverse, n_seizures=31 (max in subset)",
    },
    {
        "key": "sentinel_B",
        "subject": "epilepsiae/916",
        "rationale": "k=2, PR-5-A retained main, NOT in forward/reverse subset, ordinary k=2, n_seizures=51",
    },
]

OUT_DIR = _PROJECT_ROOT / "results" / "interictal_propagation" / "ictal_alignment" / "_sentinel_step2"


def _load_focus_rel() -> dict:
    path = _PROJECT_ROOT / "results" / "epilepsiae_electrode_focus_rel.json"
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _focal_channels(subject: str, focus_rel: dict) -> List[str]:
    sid = subject.split("/", 1)[1]
    rec = focus_rel.get(sid, {})
    return list(rec.get("i", []))


def _plot_zER_panel(
    z: np.ndarray,
    t_axis_er: np.ndarray,
    ch_names: List[str],
    focal_set: set[str],
    title: str,
    outfile: Path,
    band_label: str,
    onset_sec: float = 0.0,
) -> dict:
    """Plot multi-channel z-ER. Returns rise-around-onset summary."""

    fig, axes = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )
    ax_main, ax_sum = axes

    valid_mask = ~np.isnan(z).any(axis=1)
    n_ch = z.shape[0]

    for ch_idx in range(n_ch):
        if not valid_mask[ch_idx]:
            continue
        name = ch_names[ch_idx]
        is_focal = name.upper() in {c.upper() for c in focal_set}
        if is_focal:
            ax_main.plot(t_axis_er, z[ch_idx], color="#c0392b", alpha=0.95, lw=1.3,
                         label=f"focal: {name}", zorder=5)
        else:
            ax_main.plot(t_axis_er, z[ch_idx], color="#7f8c8d", alpha=0.18, lw=0.6, zorder=1)

    ax_main.axvline(onset_sec, color="black", lw=1.4, ls="--", label="clin_onset")
    ax_main.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax_main.set_ylabel("z-ER")
    ax_main.set_title(title, fontsize=11)
    ax_main.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)

    win_pre = (-30.0, 0.0)
    win_post = (0.0, 30.0)
    pre_mask = (t_axis_er >= win_pre[0]) & (t_axis_er < win_pre[1])
    post_mask = (t_axis_er >= win_post[0]) & (t_axis_er <= win_post[1])
    focal_idx = [i for i, n in enumerate(ch_names)
                 if n.upper() in {c.upper() for c in focal_set} and valid_mask[i]]
    nonfocal_idx = [i for i in range(n_ch) if i not in focal_idx and valid_mask[i]]

    def _band_max(idx_list: List[int], mask: np.ndarray) -> float:
        if not idx_list:
            return float("nan")
        return float(np.nanmedian(np.nanmax(z[np.ix_(idx_list, mask)], axis=1)))

    summary = {
        "n_channels_total": int(n_ch),
        "n_channels_valid": int(valid_mask.sum()),
        "n_focal_valid": len(focal_idx),
        "n_nonfocal_valid": len(nonfocal_idx),
        "focal_zER_pre30s_max_median": _band_max(focal_idx, pre_mask),
        "focal_zER_post30s_max_median": _band_max(focal_idx, post_mask),
        "nonfocal_zER_pre30s_max_median": _band_max(nonfocal_idx, pre_mask),
        "nonfocal_zER_post30s_max_median": _band_max(nonfocal_idx, post_mask),
    }

    bar_x = np.arange(4)
    bar_y = [
        summary["focal_zER_pre30s_max_median"],
        summary["focal_zER_post30s_max_median"],
        summary["nonfocal_zER_pre30s_max_median"],
        summary["nonfocal_zER_post30s_max_median"],
    ]
    colors = ["#c0392b", "#c0392b", "#7f8c8d", "#7f8c8d"]
    ax_sum.bar(bar_x, bar_y, color=colors, alpha=0.7,
               edgecolor=["#7b241c"] * 2 + ["#34495e"] * 2)
    ax_sum.set_xticks(bar_x)
    ax_sum.set_xticklabels(["focal pre", "focal post", "nonfocal pre", "nonfocal post"],
                           rotation=15, fontsize=9)
    ax_sum.set_ylabel("median(max z-ER)")
    ax_sum.axhline(0, color="black", lw=0.5, alpha=0.4)
    ax_sum.set_xlabel(f"time relative to clin_onset (s)  |  band={band_label}")

    ax_main.set_xlim(t_axis_er[0], t_axis_er[-1])
    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    plt.close(fig)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seizures", type=int, default=2,
                        help="Per-sentinel number of seizures to plot (default 2)")
    parser.add_argument("--seizure-indices", type=int, nargs="*", default=None,
                        help="Override: explicit zero-based seizure indices (applied to both sentinels)")
    parser.add_argument("--max-attempts", type=int, default=8,
                        help="Skip seizures whose pre/post window crosses block boundary; try up to this many")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    focus_rel = _load_focus_rel()

    summary = {
        "step": "PR-6-A Step 2 sentinel ER + baseline z-score visual inspection",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sentinels": [],
    }

    bands = (GAMMA_ER_BANDS, BROAD_ER_BANDS)

    for sentinel in SENTINEL_COHORT:
        subj = sentinel["subject"]
        focal_set = set(_focal_channels(subj, focus_rel))
        sentinel_log = {
            "key": sentinel["key"],
            "subject": subj,
            "rationale": sentinel["rationale"],
            "focal_channels": sorted(focal_set),
            "seizures": [],
        }
        print(f"\n=== {sentinel['key']}  {subj}  focal_channels(i)={len(focal_set)} ===")

        if args.seizure_indices is not None:
            candidate_idx = list(args.seizure_indices)
        else:
            candidate_idx = list(range(args.max_attempts))

        ok_seizures = 0
        for seizure_idx in candidate_idx:
            if ok_seizures >= args.n_seizures and args.seizure_indices is None:
                break
            try:
                window = extract_seizure_window(
                    subj, seizure_idx,
                    pre_sec=300.0, post_sec=30.0,
                    results_root=_PROJECT_ROOT / "results",
                    reference="car",
                )
            except (ValueError, IndexError) as exc:
                print(f"  seizure {seizure_idx}: SKIP ({exc})")
                continue

            print(f"  seizure {seizure_idx}: block={window.block_stem} "
                  f"fs={window.fs} n_ch={window.signal.shape[0]} "
                  f"n_samples={window.signal.shape[1]}")

            seizure_record = {
                "seizure_idx": seizure_idx,
                "seizure_id": window.seizure_id,
                "block_stem": window.block_stem,
                "clin_onset_epoch": window.clin_onset_epoch,
                "n_channels": window.signal.shape[0],
                "fs": window.fs,
                "bands": {},
            }

            for band in bands:
                er = compute_er(
                    window.signal, fs=window.fs,
                    fast_band=band["fast"], slow_band=band["slow"],
                    win_sec=1.0, hop_sec=0.1,
                )
                hop_sec = 0.1
                win_sec = 1.0
                bl_start, bl_end = baseline_window_indices(
                    er.shape[1],
                    hop_sec=hop_sec,
                    pre_sec=window.pre_sec,
                    buffer_sec=60.0,
                    onset_t_sec=0.0,
                )
                z = baseline_zscore_er(er, baseline_idx_window=(bl_start, bl_end), hop_sec=hop_sec)

                t_axis_er = (np.arange(er.shape[1]) * hop_sec + win_sec / 2.0) - window.pre_sec

                title = (
                    f"{subj} | seizure_id={window.seizure_id} idx={seizure_idx} | "
                    f"band={band['key']} (fast={band['fast']} slow={band['slow']}) | "
                    f"baseline=[-{window.pre_sec:.0f}s, -60s] | red=focal(i)"
                )
                sid = subj.split("/", 1)[1]
                outfile = OUT_DIR / f"epilepsiae_{sid}_{seizure_idx:02d}_{band['key']}.png"
                stats = _plot_zER_panel(
                    z, t_axis_er, list(window.ch_names),
                    focal_set, title, outfile, band["key"],
                )
                stats["png"] = str(outfile.relative_to(_PROJECT_ROOT))
                seizure_record["bands"][band["key"]] = stats
                print(f"    {band['key']}: focal post30s={stats['focal_zER_post30s_max_median']:.2f}  "
                      f"vs pre30s={stats['focal_zER_pre30s_max_median']:.2f}  -> {outfile.name}")

            sentinel_log["seizures"].append(seizure_record)
            ok_seizures += 1

        summary["sentinels"].append(sentinel_log)

    summary_path = OUT_DIR / "sentinel_step2_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nSummary -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
