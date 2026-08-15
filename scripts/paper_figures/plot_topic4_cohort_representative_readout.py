#!/usr/bin/env python3
"""Field, both model modes and the same-network readout for one cohort subject.

Column order follows the Topic 4 standard in `docs/figure_style_guide.md`:
substrate, mode A source, mode B source, electrode readout.

The rev10 producer's `_plot_mode_density` is not called: it draws the Figure 2A
implantation of one development subject and keys its colours to that module's
own mode numbering, neither of which holds for an arbitrary cohort subject.
The pure computations underneath it -- field interpolation, onset density, mean
propagation direction and the readout band-pass -- are imported and reused, so
the numbers on this canvas come from the accepted code path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig4_spatial_edge_flow_validation import (  # noqa: E402
    TA_COLOR,
    TB_COLOR,
    TRACE_BAND_HZ,
    _bandpass_contact_activity,
    _field_grid,
    _mode_mean_direction,
    _mode_onset_density,
    normalize_event_ranks,
)
from scripts.paper_figures.plot_topic4_cohort_representative_kmeans import (  # noqa: E402
    _relative,
    _representative_seed,
    _sha256,
)
from src.topic4_cohort_formal_scoring import score_readout  # noqa: E402

DEFAULT_CONFIG = ROOT / "config/topic4_data_driven_snn_cohort_formal_v1.json"
MODE_COLORS = (TA_COLOR, TB_COLOR)
MODE_NAMES = ("model mode A", "model mode B")
SHADE_COLORS = ("#F6D9C8", "#D4E7F2")


def _adapter(coords, positions_e, h_e, onsets, ranks, labels, clean):
    """The minimal shape the reused onset-density helpers actually read."""
    return {
        "static": {"contact_xy_mm": coords, "positions_E": positions_e, "h": h_e},
        "onsets": onsets, "ranks": ranks, "labels": labels, "clean": clean,
    }


def build(config: dict, result: dict) -> dict:
    from scripts.aggregate_topic4_data_driven_snn_cohort_formal import Cohort

    cohort = Cohort(config)
    subject_id = result["representative_subject"]["subject_id"]
    row = next(item for item in result["canonical_subjects"]
               if item["subject_id"] == subject_id)
    index = next(position for position, subject in enumerate(cohort.subjects)
                 if subject["subject_id"] == subject_id)
    seed = _representative_seed(row)
    npz_path = (
        cohort.output_root / "workers" / f"{row['candidate_id']}_seed_{seed}.npz"
    )
    key = f"canonical_{index:02d}"
    with np.load(npz_path, allow_pickle=False) as loaded:
        if f"{key}_contact_envelope" not in loaded:
            raise RuntimeError(
                f"{npz_path.name} holds no contact envelope; re-run "
                f"run_topic4_data_driven_snn_cohort_formal_worker.py for "
                f"candidate {row['candidate_id']} seed {seed} with "
                f"--store-contact-envelope"
            )
        payload = {
            "ranks": np.asarray(loaded[f"{key}_ranks"], float),
            "onsets": np.asarray(loaded[f"{key}_onsets"], float),
            "coords": np.asarray(loaded[f"{key}_contact_xy_mm"], float),
            "names": [str(value) for value in loaded[f"{key}_contact_names"]],
            "envelope": np.asarray(loaded[f"{key}_contact_envelope"], float),
            "envelope_dt_ms": float(loaded["contact_envelope_dt_ms"]),
            "positions_E": np.asarray(loaded["positions_E"], float),
            "h_E": np.asarray(loaded["h_E"], float),
            "event_t_on_ms": np.asarray(loaded["event_t_on_ms"], float),
            "event_t_off_ms": np.asarray(loaded["event_t_off_ms"], float),
        }
    kwargs = cohort.scorer_kwargs(cohort.subjects[index], "heldout")
    score = score_readout(payload["ranks"], **kwargs)
    if score.get("status") != "EVALUABLE":
        raise RuntimeError(f"representative subject is not evaluable: {score['status']}")
    readable = np.isfinite(payload["ranks"]).sum(axis=1) >= int(
        kwargs["minimum_contacts"]
    )
    labels = np.asarray(score["natural_kmeans"]["aligned_labels"], int)
    full_labels = np.full(len(payload["ranks"]), -1, int)
    full_labels[readable] = labels
    with np.load(
        cohort.target_root / f"{subject_id}_target.npz", allow_pickle=False,
    ) as loaded:
        train = np.vstack([
            np.asarray(loaded["train_ta_rank_samples"], float),
            np.asarray(loaded["train_tb_rank_samples"], float),
        ]).T
    from scripts import plot_interictal_propagation as propagation_plot

    return {
        "channel_order": propagation_plot._fixed_channel_order(
            train, np.isfinite(train),
        ),
        **payload,
        "subject_id": subject_id,
        "candidate_id": row["candidate_id"],
        "seed": seed,
        "npz": npz_path,
        "labels": full_labels,
        "clean": readable,
        "score": score,
        "row": row,
    }


def _draw_field(ax, data):
    xx, yy, hh = _field_grid(data["positions_E"], data["h_E"], size=110)
    vmax = max(float(np.quantile(data["h_E"], 0.995)), 1e-9)
    mesh = ax.contourf(xx, yy, np.minimum(hh, vmax), levels=18, cmap="plasma",
                       vmin=0.0, vmax=vmax)
    coords, names = data["coords"], data["names"]
    shafts = [name.rstrip("0123456789") for name in names]
    for shaft in sorted(set(shafts)):
        selected = np.asarray([value == shaft for value in shafts])
        ax.plot(coords[selected, 0], coords[selected, 1], "-o", color="white",
                markersize=3.4, linewidth=1.0, alpha=0.95, zorder=6)
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)",
           ylabel="sheet y (mm)")
    ax.set_aspect("equal")
    ax.set_title("substrate", fontsize=14, weight="bold", pad=8)
    return mesh


def _draw_mode(ax, data, mode, *, show_ylabel):
    adapter = _adapter(
        data["coords"], data["positions_E"], data["h_E"], data["onsets"],
        data["ranks"], data["labels"], data["clean"],
    )
    xx, yy, hh = _field_grid(data["positions_E"], data["h_E"], size=110)
    vmax = max(float(np.quantile(data["h_E"], 0.995)), 1e-9)
    ax.contourf(xx, yy, np.minimum(hh, vmax), levels=18, cmap="plasma",
                vmin=0.0, vmax=vmax, alpha=0.94)
    density, contact_mass = _mode_onset_density(adapter, mode, xx, yy)
    if np.max(density, initial=0.0) > 0.0:
        ax.contour(xx, yy, density, levels=(0.22, 0.45, 0.70),
                   colors=MODE_COLORS[mode], linewidths=(1.1, 1.6, 2.2),
                   alpha=0.98)
    coords = data["coords"]
    ax.scatter(coords[:, 0], coords[:, 1], s=14, facecolor="none",
               edgecolor="white", linewidth=0.7, alpha=0.75, zorder=6)
    present = contact_mass > 0.0
    ax.scatter(coords[present, 0], coords[present, 1],
               s=45 + 220 * contact_mass[present], facecolor="white",
               edgecolor=MODE_COLORS[mode], linewidth=1.3, alpha=0.92, zorder=8)
    direction = _mode_mean_direction(adapter, mode)
    if direction is not None:
        start, stop = direction
        for color, width in (("white", 5.2), (MODE_COLORS[mode], 2.7)):
            ax.annotate("", xy=stop, xytext=start, zorder=10,
                        arrowprops={"arrowstyle": "-|>", "color": color,
                                    "lw": width, "mutation_scale": 15,
                                    "shrinkA": 0, "shrinkB": 0})
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)")
    ax.set_ylabel("sheet y (mm)" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_aspect("equal")
    ax.set_title(MODE_NAMES[mode], fontsize=14, color=MODE_COLORS[mode],
                 weight="bold", pad=8)
    ax.text(0.035, 0.965, f"events  n={int(np.sum(data['labels'] == mode))}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.0,
            color="white", weight="bold",
            bbox={"facecolor": MODE_COLORS[mode], "edgecolor": "none",
                  "alpha": 0.86, "pad": 2.2})


def _event_pair(data):
    """One event per mode from the same network, as far apart in time as possible."""
    starts = data["event_t_on_ms"]
    picks = []
    for mode in (0, 1):
        available = np.flatnonzero(data["labels"] == mode)
        if not len(available):
            return None
        picks.append(available)
    best, best_gap = None, -1.0
    for first in picks[0]:
        for second in picks[1]:
            gap = abs(float(starts[first] - starts[second]))
            if gap > best_gap:
                best, best_gap = (int(first), int(second)), gap
    return best


def _draw_readout(ax, data, pair):
    if pair is None:
        ax.text(0.5, 0.5, "one network did not produce both modes",
                transform=ax.transAxes, ha="center", va="center",
                color="#9B2F2A", weight="bold", fontsize=12)
        ax.axis("off")
        return
    dt = data["envelope_dt_ms"]
    traces = _bandpass_contact_activity(data["envelope"], dt, TRACE_BAND_HZ)
    scale = float(np.percentile(np.abs(traces), 99)) or 1.0
    order = data["channel_order"]
    names = [data["names"][index] for index in order]
    ticks, tick_labels, breaks = [], [], []
    offset, gap = 0.0, 0.0
    for position, event in enumerate(pair):
        start = float(data["event_t_on_ms"][event])
        stop = float(data["event_t_off_ms"][event])
        pad = 0.35 * max(stop - start, 20.0)
        lo = max(0, int((start - pad) / dt))
        hi = min(traces.shape[1], int((stop + pad) / dt))
        span = (hi - lo) * dt
        gap = gap or 0.12 * span
        ax.axvspan(offset, offset + span, color=SHADE_COLORS[position],
                   zorder=0, linewidth=0)
        time = offset + np.arange(hi - lo) * dt
        for row, contact in enumerate(order):
            ax.plot(time, row + traces[contact, lo:hi] / scale * 0.42,
                    color="#333333", linewidth=0.7)
            onset = data["onsets"][event, contact]
            if np.isfinite(onset):
                ax.plot(offset + (onset - lo * dt), row, "o", markersize=3.4,
                        color=MODE_COLORS[position], zorder=5)
        ax.text(offset + 0.5 * span, len(names) - 0.15,
                f"{MODE_NAMES[position]}\nat {start / 1000.0:.1f} s",
                ha="center", va="bottom", fontsize=10.5, weight="bold",
                color=MODE_COLORS[position], linespacing=1.25)
        # Each window carries its own local clock; the two events are seconds
        # apart in the same run and must not read as one continuous trace.
        for local in (0.0, 0.5 * span, span):
            ticks.append(offset + local)
            tick_labels.append(f"{local:.0f}")
        if position + 1 < len(pair):
            breaks.append((offset + span, gap))
        offset += span + gap
    for start, width in breaks:
        for fraction in (0.35, 0.65):
            centre = start + fraction * width
            ax.plot([centre - 0.16 * width, centre + 0.16 * width],
                    [-0.9, len(names) + 0.3], color="#999999", linewidth=1.1,
                    zorder=3, clip_on=False)
    ax.set_yticks(np.arange(len(names)), names, fontsize=7.5)
    ax.set_ylim(-1.0, len(names) + 0.9)
    ax.set_xlim(0, offset - gap)
    ax.set_xticks(ticks, tick_labels, fontsize=9)
    ax.set_xlabel(f"time within each event window (ms), "
                  f"{int(TRACE_BAND_HZ[0])}-{int(TRACE_BAND_HZ[1])} Hz "
                  f"firing-density envelope")
    ax.spines[["top", "right"]].set_visible(False)


def render(data: dict, output: Path) -> dict:
    fig = plt.figure(figsize=(20.4, 5.6), facecolor="white")
    grid = fig.add_gridspec(
        1, 5, width_ratios=(1.0, 1.0, 1.0, 0.055, 1.62), left=0.042,
        right=0.988, bottom=0.155, top=0.845, wspace=0.30,
    )
    mesh = _draw_field(fig.add_subplot(grid[0, 0]), data)
    _draw_mode(fig.add_subplot(grid[0, 1]), data, 0, show_ylabel=False)
    _draw_mode(fig.add_subplot(grid[0, 2]), data, 1, show_ylabel=False)
    bar = fig.colorbar(mesh, cax=fig.add_subplot(grid[0, 3]))
    bar.set_label("node field h", fontsize=10)
    bar.ax.tick_params(labelsize=9)
    _draw_readout(fig.add_subplot(grid[0, 4]), data, _event_pair(data))
    verdict = data["row"]
    fig.text(
        0.5, 0.985,
        f"{data['subject_id']} | one network | beat its own within-shaft "
        f"shuffle: {'yes' if verdict['subject_endpoint_pass'] else 'no'}",
        ha="center", va="top", fontsize=12.5, weight="bold",
        color=TA_COLOR if verdict["subject_endpoint_pass"] else "#9B2F2A",
    )
    fig.text(
        0.5, 0.035,
        "contact positions are the target-blind contact-order layout, not this "
        "patient's anatomy; the readout is a firing-density envelope, not a "
        "clinical SEEG voltage",
        ha="center", va="bottom", fontsize=9.5, color="#444444",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=240)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    return {
        "png": _relative(output.with_suffix(".png")),
        "png_sha256": _sha256(output.with_suffix(".png")),
        "pdf": _relative(output.with_suffix(".pdf")),
        "pdf_sha256": _sha256(output.with_suffix(".pdf")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-commit", default=None)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output_root = ROOT / config["output_root"]
    result_path = output_root / "cohort_result.json"
    if not result_path.exists():
        print(json.dumps({"status": "COHORT_RESULT_ABSENT_FIGURE_SKIPPED"}))
        return
    result = json.loads(result_path.read_text())
    data = build(config, result)
    figures = output_root / "figures"
    files = render(data, figures / "topic4_cohort_representative_readout")
    metadata = {
        "schema_version": "topic4_cohort_representative_readout_v1",
        "science_status": {
            "cohort_status": result["status"], "verdict": result["verdict"],
            "result_json_sha256": _sha256(result_path),
        },
        "subject_id": data["subject_id"],
        "candidate_id": data["candidate_id"],
        "confirmation_seed": data["seed"],
        "worker_npz": _relative(data["npz"]),
        "worker_npz_sha256": _sha256(data["npz"]),
        "readout_band_hz": list(TRACE_BAND_HZ),
        "envelope_dt_ms": data["envelope_dt_ms"],
        "n_events_by_mode": [
            int(np.sum(data["labels"] == mode)) for mode in (0, 1)
        ],
        "files": files,
        "scientific_boundary": config["claim_boundary"],
    }
    (figures / "topic4_cohort_representative_readout_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({"subject": data["subject_id"], **files}, indent=2))


if __name__ == "__main__":
    main()
