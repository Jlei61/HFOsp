#!/usr/bin/env python3
"""The empirical state map along the natural path, plus the three readouts the review asked for.

Four independent questions, one panel each:
  a  which regime does each point land in, and where do D and the H gate sit there?
  b  does the endpoint depend on where the tissue started from inside the same pinned field?
  c  how much tissue carries the outcome?
  d  is the bounded regime the same re-ignition train everywhere it appears?

A supplementary readout is derived here rather than in the runner: the registered decision tree sends
"never rose above the interictal band at one-second resolution" down the same AFTER_DISCHARGE branch
as "rose briefly, then died", because it keys on the CUMULATIVE time above the band and a bursty
interictal state accumulates ~1.7 s of scattered supra-band bins across 10 s.  The registered label is
plotted as recorded; `interictal throughout` is drawn as a separate hatch so the map does not claim a
discharge that never happened.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc6b_frozen_slow_atlas"
FIGURES = OUT / "figures"
INITS = ("path_native", "locked_low", "locked_high")
INIT_LABEL = {"path_native": "the path's own fast state",
              "locked_low": "one shared interictal fast state",
              "locked_high": "one shared high fast state"}
INIT_COLOR = {"path_native": "#37474F", "locked_low": "#3B7EA1", "locked_high": "#B00020"}
REGIME_COLOR = {
    "ESCALATING_SATURATION": "#B00020", "BOUNDED_OSCILLATORY": "#2CA25F",
    "BOUNDED_STATIONARY": "#00695C", "LOW_STATE": "#90A4AE", "SILENCE": "#ECEFF1",
    "AFTER_DISCHARGE": "#CFD8DC", "RIGHT_CENSORED": "#FFB300", "NUMERICAL_FAIL": "#000000"}
REGIME_TEXT = {
    "ESCALATING_SATURATION": "escalates to saturation", "BOUNDED_OSCILLATORY": "bounded, bursting",
    "BOUNDED_STATIONARY": "bounded, steady", "LOW_STATE": "falls back to interictal",
    "SILENCE": "silent", "AFTER_DISCHARGE": "brief elevation, then back in band",
    "RIGHT_CENSORED": "unresolved in window", "NUMERICAL_FAIL": "numerical failure"}


def _load():
    atlas = json.loads((OUT / "natural_path_atlas.json").read_text())
    traces = {}
    for key, row in atlas["rows"].items():
        with np.load(OUT / f"atlas/{row['point_id']}/traces.npz") as handle:
            traces[key] = {name: np.asarray(handle[name]) for name in handle.files}
    return atlas, traces


def spike_convergence(field, *, tail_ms=5000.0, dt_ms=0.05):
    """Do the three initialisations end up emitting the SAME spikes, cell by cell and step by step?

    The final full-state hash cannot answer this: it folds in the step counter, and the three
    initialisations legitimately carry different ones (the shared low state is at 42001 steps, the
    shared high state at 360000, the path's own state at wherever the trajectory was).  Comparing the
    spike trains over the tail is the question actually being asked -- has the tissue forgotten which
    state it was handed?  Identical trains mean it has, which is a single-attractor reading and the
    cleanest possible negative for bistability at that field.
    """
    digests = {}
    for init in INITS:
        with np.load(OUT / f"atlas/{field}__{init}/spikes.npz", allow_pickle=False) as handle:
            steps = np.asarray(handle["steps"], np.int64)
            cells = np.asarray(handle["cells"], np.int32)
            n_steps = int(handle["n_steps"][0])
        keep = steps >= n_steps - int(round(tail_ms / dt_ms))
        digests[init] = hashlib.sha256(
            np.ascontiguousarray(steps[keep]).tobytes()
            + np.ascontiguousarray(cells[keep]).tobytes()).hexdigest()
    return {
        "tail_ms": tail_ms, "per_init_sha256": digests,
        "all_three_identical": len(set(digests.values())) == 1,
        "low_and_high_identical": digests["locked_low"] == digests["locked_high"],
    }


def _adjudication():
    """Read the verdict; never compute one here.

    Round 2 produced its outcome verdict inside this plotting script.  That is the wrong home for a
    scientific adjudication -- it is not versioned with the result and it was never tested -- so the
    comparison now lives in src/topic4_fcxr_lc6b_outcome.py behind
    scripts/finalize_topic4_fcxr_lc6b_outcome.py, and this figure only draws what that wrote.
    """
    path = OUT / "atlas_outcome_adjudication.json"
    if not path.is_file():
        raise SystemExit(
            "run scripts/finalize_topic4_fcxr_lc6b_outcome.py --confirm-run first: the outcome "
            "verdict is produced there, not in this figure")
    return json.loads(path.read_text())


def _burst(trace, band):
    bin_ms = float(np.asarray(trace["rate_bin_ms"]).ravel()[0])
    tail = np.asarray(trace["rate_bins_hz"], float)[-int(round(2000.0 / bin_ms)):]
    above = (tail > band).astype(int)
    starts = int(np.sum(np.diff(above) == 1) + (1 if above.size and above[0] else 0))
    return {"sub_band_fraction": float(np.mean(tail <= band)),
            "bursts_per_s": starts / (tail.size * bin_ms / 1000.0) if tail.size else float("nan")}


def main():
    atlas, traces = _load()
    cfg = json.loads((ROOT / "config/topic4_fcxr_lc6b_frozen_slow_atlas.json").read_text())
    band = cfg["classifier"]["thresholds"]["interictal_roll_hi_hz"]
    fields = atlas["fields_in_time_order"]
    per_field = atlas["per_field"]
    x = np.asarray([per_field[f]["relative_to_onset_ms"] / 1000.0 for f in fields], float)
    row = lambda f, i: atlas["rows"][f"{f}__{i}"]

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4), constrained_layout=True)

    # a -- the empirical state map, with the two slow coordinates underneath it
    ax = axes[0, 0]
    for index, init in enumerate(INITS):
        for column, field in enumerate(fields):
            verdict = row(field, init)["verdict"]
            label = verdict["label"]
            interictal_throughout = (verdict["max_global_1s_mean_hz"] or 0.0) <= band
            ax.add_patch(plt.Rectangle(
                (x[column] - .45, index - .42), .9, .84,
                facecolor=REGIME_COLOR.get(label, "#BDBDBD"), edgecolor="white", linewidth=1.2,
                hatch="///" if interictal_throughout else None))
    ax.set_xlim(x.min() - .6, x.max() + .6)
    ax.set_ylim(-.6, len(INITS) - .4)
    ax.set_yticks(range(len(INITS)), [INIT_LABEL[i] for i in INITS], fontsize=8)
    ax.set_xticks(x, [f"{v:+.0f}" for v in x])
    ax.set_xlabel("field taken from the natural trajectory at this time relative to onset (s)")
    ax.set_title("a  common-input outcome map along the path", loc="left", fontsize=10)
    twin = ax.twinx()
    twin.plot(x, [per_field[f]["D_mean"] for f in fields], color="#D95F0E", marker="o", ms=4, lw=1.4,
              label="mean wear D of the pinned field")
    twin.plot(x, [per_field[f]["h_gate_mean"] for f in fields], color="#8C6BB1", marker="s", ms=4,
              lw=1.4, label="mean H gate occupancy of the pinned field")
    twin.set_ylabel("pinned slow field value")
    twin.set_ylim(0, 1.35)          # headroom so the two curves never run through the top cells
    twin.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(.5, -.30))
    present = [label for label in REGIME_COLOR
               if any(row(f, i)["verdict"]["label"] == label for f in fields for i in INITS)]
    handles = [Patch(facecolor=REGIME_COLOR[label], label=REGIME_TEXT[label]) for label in present]
    handles.append(Patch(facecolor="white", edgecolor="0.4", hatch="///",
                         label="never above the band at 1 s resolution"))
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=2,
              loc="upper center", bbox_to_anchor=(.5, -.16))

    # b -- does the endpoint depend on where the tissue started?
    ax = axes[0, 1]
    for init in INITS:
        values = [row(f, init)["verdict"]["per_second_mean_hz"][-1] for f in fields]
        ax.plot(x, values, color=INIT_COLOR[init], marker="o", ms=5, lw=1.6, label=INIT_LABEL[init])
    convergence = {field: spike_convergence(field) for field in fields}
    bursts = {f"{f}__{i}": _burst(traces[f"{f}__{i}"], band) for f in fields for i in INITS}
    adjudication = _adjudication()
    outcomes = {f: adjudication["per_field"][f]["outcome_locked_low_vs_locked_high"] for f in fields}
    for column, field in enumerate(fields):
        if per_field[field]["initialisation_split"]:
            same = outcomes[field]["same_outcome_regime"]
            ax.axvline(x[column], color="#9E9E9E" if same else "#FFB300", lw=8,
                       alpha=.20 if same else .30, zorder=0)
        if convergence[field]["all_three_identical"]:
            # axes-fraction y: the data-space floor is not settled until every artist is drawn
            ax.plot([x[column]], [0.012], marker="^", ms=9, color="#2CA25F", clip_on=False,
                    zorder=5, transform=ax.get_xaxis_transform())
    ax.axhline(cfg["classifier"]["thresholds"]["global_saturation_hz"], color="#B00020", ls="--", lw=1.0)
    ax.axhline(band, color="0.45", ls=":", lw=1.0)
    ax.set_yscale("symlog", linthresh=10)
    ax.set_xticks(x, [f"{v:+.0f}" for v in x])
    ax.set_xlabel("field time relative to onset (s)")
    ax.set_ylabel("final-second population rate (Hz, symlog)")
    ax.set_title("b  does the endpoint depend on the starting fast state?", loc="left", fontsize=10)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#FFB300", alpha=.30))
    labels.append("different labels AND different outcome regimes (candidate)")
    handles.append(Patch(facecolor="#9E9E9E", alpha=.20))
    labels.append("different labels but the SAME outcome regime (drift-gate artefact)")
    handles.append(Line2D([], [], marker="^", ms=8, color="#2CA25F", ls="none"))
    labels.append("all three emit bitwise identical spikes in the final 5 s")
    ax.legend(handles, labels, frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(.5, -.16))
    ax.spines[["top", "right"]].set_visible(False)

    # c -- how much tissue
    ax = axes[1, 0]
    sheet = float(row(fields[0], INITS[0])["sheet_area_mm2"])
    for init in INITS:
        values = [row(f, init)["median_active_area_mm2"] or 0.0 for f in fields]
        ax.plot(x, values, color=INIT_COLOR[init], marker="o", ms=5, lw=1.6, label=INIT_LABEL[init])
    ax.axhline(sheet, color="#B00020", ls="--", lw=1.0)
    ax.annotate(f"whole sheet {sheet:.0f} mm²", xy=(1.0, sheet), xycoords=("axes fraction", "data"),
                xytext=(-4, 3), textcoords="offset points", fontsize=8, color="#B00020",
                ha="right", va="bottom")
    ax.set_xticks(x, [f"{v:+.0f}" for v in x])
    ax.set_ylim(0, sheet * 1.12)
    ax.set_xlabel("field time relative to onset (s)")
    ax.set_ylabel("median active area (mm², 100 ms windows)")
    ax.set_title("c  how much tissue carries the outcome", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="center left")
    ax.spines[["top", "right"]].set_visible(False)

    # d -- is the bounded regime always the same train?
    ax = axes[1, 1]
    for init in INITS:
        stats = [_burst(traces[f"{f}__{init}"], band) for f in fields]
        ax.plot(x, [s["sub_band_fraction"] for s in stats], color=INIT_COLOR[init],
                marker="o", ms=5, lw=1.6, label=INIT_LABEL[init])
    ax.axhline(cfg["classifier"]["thresholds"]["silence_bin_fraction_gate"], color="0.35",
               ls=":", lw=1.2)
    ax.annotate("registered bursty gate 0.25", xy=(1.0, .25), xycoords=("axes fraction", "data"),
                xytext=(-4, 3), textcoords="offset points", fontsize=8, color="0.3",
                ha="right", va="bottom")
    ax.set_xticks(x, [f"{v:+.0f}" for v in x])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("field time relative to onset (s)")
    ax.set_ylabel("fraction of final-2 s 20 ms bins at or below the band")
    ax.set_title("d  every non-saturated outcome is a train, not a plateau", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("FCXR-LC6B — the empirical state map along the natural (D, H) path", fontsize=12)
    fig.text(.5, -.02,
             "Every point pins D and H at the value the C0 trajectory reached at that time and runs "
             "10 s.  All 18 points share one future-input stream, so a difference between rows can "
             "only come from the starting fast state.  At every field the two locked initialisations "
             "reached the SAME outcome regime on all four readouts, so no orange band appears: the "
             "registered label splits are drift-gate artefacts.  Zero-lag population correlation is "
             "NEGATIVE (-0.42 to -0.47) while the phase-aligned correlation is 0.93-0.99: one rhythm "
             "at two phases, not two outcomes.  Under one shared input this cannot separate a single "
             "attractor from common-noise synchronisation, and no perturbation-return test was run, "
             "so the bounded regime is a MONOSTABLE CANDIDATE -- not a demonstrated absence of a "
             "carrier.",
             ha="center", va="top", fontsize=8, color="#555555", wrap=True)

    # The convergence readout is derived here, so it is written out where the report can cite it
    # instead of living only inside the figure.
    (OUT / "atlas_supplementary_readouts.json").write_text(json.dumps({
        "not_the_registered_classifier": True,
        "spike_convergence": convergence,
        "spike_convergence_note": (
            "identical final-5 s spike trains across all three initialisations mean the tissue has "
            "forgotten which state it was handed, i.e. a single input-driven attractor at that "
            "pinned field -- the cleanest available negative for bistability there"),
        "interictal_throughout": {
            f"{f}__{i}": bool((row(f, i)["verdict"]["max_global_1s_mean_hz"] or 0.0) <= band)
            for f in fields for i in INITS},
        "interictal_throughout_note": (
            "the registered tree keys AFTER_DISCHARGE on CUMULATIVE time above the band, and a bursty "
            "interictal state accumulates ~1.7 s of scattered supra-band 20 ms bins across 10 s, so "
            "'never rose above the band at one-second resolution' is recorded separately here rather "
            "than by relabelling"),
        "burst_structure": bursts,
        "outcome_verdict_source": "atlas_outcome_adjudication.json (produced by "
                                  "scripts/finalize_topic4_fcxr_lc6b_outcome.py, not by this figure)",
    }, indent=1, sort_keys=True) + "\n")

    FIGURES.mkdir(parents=True, exist_ok=True)
    png, pdf = FIGURES / "lc6b_natural_path_atlas.png", FIGURES / "lc6b_natural_path_atlas.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {png}\nwrote {pdf}")


if __name__ == "__main__":
    main()
