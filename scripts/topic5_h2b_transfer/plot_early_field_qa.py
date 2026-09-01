#!/usr/bin/env python3
"""B0.3 visual QA -- onset and channel alignment, at least 3 seizures per cohort.

One question per panel: *does the fast-activity rise actually start where we
placed the EEG onset, on channels that are in the order we think they are?*
Seizure choice is deterministic (first eligible seizure of the first eligible
subjects, alphabetically) so the montage cannot be cherry-picked.

Emits PNG + PDF + metadata JSON; the figures README is written alongside.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DATA = Path("/data/hfosp_group_event_state_v0_2/agent_b")
DEFAULT_OUT = ROOT / "results/epi_prssm/group_event_state/v0_2/h2b/figures"
STEM = "early_ictal_field_onset_alignment_qa"


def pick(data_root: Path, per_cohort: int):
    """First eligible seizure of the first eligible subjects, alphabetically."""
    chosen = {"epilepsiae": [], "yuquan": []}
    for jp in sorted((data_root / "early_field").glob("*.json")):
        meta = json.loads(jp.read_text())
        ds = meta.get("dataset")
        if ds not in chosen or len(chosen[ds]) >= per_cohort:
            continue
        ok = [(i, s) for i, s in enumerate(meta["seizures"]) if s["status"] == "ok"]
        if not ok:
            continue
        i, s = ok[0]
        chosen[ds].append((meta["subject"], i, s, meta))
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-cohort", type=int, default=3)
    args = ap.parse_args()

    chosen = pick(args.data_root, args.per_cohort)
    cohorts = [c for c in ("epilepsiae", "yuquan") if chosen[c]]
    if not cohorts:
        raise SystemExit("no eligible seizures found yet")
    nrow = max(len(chosen[c]) for c in cohorts)

    # symmetric, shared colour limits from the pooled robust spread (§0.1: a
    # signed quantity gets a diverging map centred on zero)
    pool = []
    panels = {}
    for c in cohorts:
        for subject, i, s, meta in chosen[c]:
            z = np.load(args.data_root / "early_field" / f"{subject}.npz")
            tr = z[f"bb_early_trace__{i:03d}"]
            rt = z[f"bb_early_relt__{i:03d}"]
            tc = z[f"first_crossing_s__{i:03d}"]
            panels[(subject, i)] = (tr, rt, tc, s, meta)
            pool.append(np.abs(tr[np.isfinite(tr)]))
    vmax = float(np.nanpercentile(np.concatenate(pool), 99)) if pool else 5.0
    vmax = max(vmax, 1.0)

    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6,
                         "xtick.major.width": 0.6, "ytick.major.width": 0.6})
    fig = plt.figure(figsize=(3.5 * len(cohorts) + 0.75, 2.15 * nrow + 0.55))
    gs = GridSpec(nrow, len(cohorts) + 1, figure=fig,
                  width_ratios=[1] * len(cohorts) + [0.045],
                  wspace=0.28, hspace=0.55)

    im = None
    for ci, c in enumerate(cohorts):
        for ri in range(nrow):
            ax = fig.add_subplot(gs[ri, ci])
            if ri >= len(chosen[c]):
                ax.axis("off")
                continue
            subject, i, s, meta = chosen[c][ri]
            tr, rt, tc, s, meta = panels[(subject, i)]
            # contacts ordered by recruitment time; never-recruited last
            key = np.where(np.isfinite(tc), tc, np.inf)
            order = np.argsort(key, kind="stable")
            im = ax.pcolormesh(rt, np.arange(tr.shape[0]), tr[order],
                               cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                               shading="nearest", rasterized=True)
            ax.axvline(0.0, color="k", lw=1.1)
            # common limits so the panels are directly comparable; the two
            # cohorts otherwise differ by a hop (-4.95 vs -5.00)
            ax.set_xlim(-5.0, 15.0)
            ax.set_ylim(-0.5, tr.shape[0] - 0.5)
            label = subject.split("_", 1)[1]
            # cohort name rides on the first panel of each column instead of an
            # overlay axes, which would sit on top of real data (§0.2)
            cohort_name = {"epilepsiae": "Epilepsiae", "yuquan": "Yuquan"}[c]
            if ri == 0:
                ax.set_title(f"{cohort_name}\n{label} · seizure {i + 1}",
                             fontsize=8, pad=4, loc="left")
            else:
                ax.set_title(f"{label} · seizure {i + 1}", fontsize=8, pad=3, loc="left")
            if ri == nrow - 1 or ri == len(chosen[c]) - 1:
                ax.set_xlabel("time from EEG onset (s)")
            if ci == 0:
                ax.set_ylabel("contacts\n(ordered by fast-activity arrival)", fontsize=7.5)
            ax.tick_params(labelsize=7)

    cax = fig.add_subplot(gs[:, -1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("band power (baseline robust-z, 1–45 Hz)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png, pdf = args.out_dir / f"{STEM}.png", args.out_dir / f"{STEM}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    meta_out = {
        "figure": STEM,
        "question": "Does the peri-onset rise start at the EEG onset we assigned, "
                    "on channels in the recorded order?",
        "anchor": "eeg_onset_epoch (not clinical onset, not cache relt=0)",
        "band": [1.0, 45.0],
        "colour": "diverging RdBu_r centred on 0; signed robust-z",
        "colour_limit_z": vmax,
        "contact_order": "ascending first fast-activity crossing (z>=5, 60-100 Hz); "
                         "never-recruited contacts last",
        "selection_rule": "first eligible seizure of the first eligible subjects, "
                          "alphabetically -- deterministic, not curated",
        "panels": [
            {"subject": subject, "seizure_index": i,
             "seizure_id": panels[(subject, i)][3]["seizure_id"],
             "n_channels": panels[(subject, i)][4]["n_channels"],
             "reference": panels[(subject, i)][4]["reference"]}
            for c in cohorts for subject, i, _s, _m in chosen[c]
        ],
    }
    (args.out_dir / f"{STEM}_metadata.json").write_text(json.dumps(meta_out, indent=2))
    print(f"wrote {png}\nwrote {pdf}\nwrote {args.out_dir / (STEM + '_metadata.json')}")
    for c in cohorts:
        print(f"  {c}: " + ", ".join(f"{s}#{i+1}" for s, i, _x, _m in chosen[c]))


if __name__ == "__main__":
    main()
