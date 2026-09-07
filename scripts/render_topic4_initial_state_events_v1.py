#!/usr/bin/env python3
"""Chronological displays of every detected window (plan §8, F5).

Minimum dynamics seed x three arms: contact-map montage, native strobograms and
an all-events GIF (native sheet + contact readout + timeline + core state).
Every other run: native strobogram pages + its event table. No window is
selected by likeness; nothing is aligned across arms by event number.
"""
from __future__ import annotations

import argparse
import csv
import gc
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import topic4_initial_state_runtime as rt  # noqa: E402
from scripts.analyze_topic4_initial_state_v1 import ARM_COLOR, ARM_LABEL, MODE_COLOR, load_runs  # noqa: E402

PATIENT = Path("/home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/event_envelope_fields/epilepsiae_1146_event_envelope_field_cache.npz")
OFFSETS_MS = np.array([-16, 0, 16, 32, 64])


def patient_positions(names):
    with np.load(PATIENT, allow_pickle=True) as p:
        order = p["contact_order"].astype(str).tolist()
        points = np.asarray(p["points_mm"], float)
    return points[[order.index(n) for n in names]]


def contact_map(ax, xy, t, title, names, labels=False):
    valid = np.isfinite(t)
    rel = t - (np.nanmin(t) if valid.any() else 0.0)
    ax.scatter(*xy[~valid].T, s=42, facecolors="none", edgecolors="0.65", linewidth=0.8)
    if valid.any():
        ax.scatter(*xy[valid].T, c=rel[valid], s=48, cmap="viridis", vmin=0, vmax=100,
                   edgecolors="0.25", linewidth=0.4)
    ax.set(xlim=(-21, 23), ylim=(-13, 24), aspect="equal", title=title)
    ax.set_xticks([]); ax.set_yticks([]); ax.title.set_fontsize(7.5)
    if labels:
        for (x, y), n in zip(xy, names):
            ax.text(x, y + 1, n, fontsize=5, ha="center")


def montage(run, xy, folder):
    z, rec = run["arrays"], run["record"]
    names = z["contact_names"].astype(str).tolist()
    t = np.asarray(z["centroid_ms"], float)
    modes = np.asarray(z["event_mode"], int)
    states = np.asarray(z["event_support_state"], int)
    primary = set(int(i) for i in z["primary_event_indices"])
    windows = np.asarray(z["windows_ms"], float)
    n = len(t)
    cols = 6
    rows = max(1, int(np.ceil(n / cols)))
    height = 2.3 * rows + 1.6
    fig, axes = plt.subplots(rows, cols, figsize=(2.3 * cols, height))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for i in range(n):
        status = "primary" if i in primary else "excluded"
        state = {1: "supported", 0: "uncertain", -1: "OOD"}[int(states[i])] if modes[i] >= 0 else "unreadable"
        contact_map(axes[i], xy, t[i], f"#{i + 1}  {windows[i][0] / 1000:.2f} s | M{modes[i]} {state}\n{status}", names,
                    labels=(i == 0))
    fig.subplots_adjust(top=1 - 0.9 / height, bottom=0.7 / height, hspace=0.45, wspace=0.08)
    fig.suptitle(f"{ARM_LABEL[rec['arm']]} | graph {rec['topology_seed']} | seed {rec['dynamics_seed']}\n"
                 "All detected windows in chronological order; colour = relative centroid 0-100 ms; hollow = absent",
                 fontsize=10)
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 100), cmap="viridis")
    fig.colorbar(sm, cax=fig.add_axes([0.3, 0.25 / height, 0.4, 0.12 / height]), orientation="horizontal",
                 label="Relative centroid (ms); values above 100 clipped in colour only")
    fig.savefig(folder / "all_event_contact_maps.png", dpi=130)
    fig.savefig(folder / "all_event_contact_maps.pdf")
    plt.close(fig)
    return n


def strobograms(run, folder, centers, radius):
    z, rec = run["arrays"], run["record"]
    t = np.asarray(z["centroid_ms"], float)
    movie = z["sheet_activity_counts"]
    dt = float(z["sheet_activity_frame_ms"])
    modes = np.asarray(z["event_mode"], int)
    primary = set(int(i) for i in z["primary_event_indices"])
    vmax = max(1.0, float(movie.max()))
    pages = 0
    for page, start in enumerate(range(0, len(t), 6)):
        count = min(6, len(t) - start)
        fig, axes = plt.subplots(count, len(OFFSETS_MS), figsize=(10, count * 1.85), squeeze=False)
        for row, i in enumerate(range(start, start + count)):
            valid = np.isfinite(t[i])
            t0 = np.nanmin(t[i]) if valid.any() else float(z["windows_ms"][i][0])
            indices = np.clip(np.round((t0 + OFFSETS_MS) / dt - 0.5).astype(int), 0, len(movie) - 1)
            for col, frame in enumerate(indices):
                ax = axes[row, col]
                ax.imshow(movie[frame], origin="lower", cmap="magma", vmin=0, vmax=vmax,
                          interpolation="nearest", extent=(0, 20, 0, 20))
                for c in centers:
                    ax.add_patch(Circle(c, radius, fill=False, edgecolor="cyan", lw=0.6))
                ax.scatter(*z["contact_xy_mm"].T, s=4, facecolors="none", edgecolors="white", lw=0.3)
                label = f"#{i + 1} M{modes[i]} {'primary' if i in primary else 'excl.'}" if col == 0 else f"{OFFSETS_MS[col]:+d} ms"
                ax.set(xticks=[], yticks=[], title=label)
                ax.title.set_fontsize(7.5)
        fig.suptitle(f"{ARM_LABEL[rec['arm']]} | graph {rec['topology_seed']} | seed {rec['dynamics_seed']}\n"
                     "Native sheet (1 mm / 2 ms, fixed scale), all detected windows chronologically; cyan = core disks",
                     fontsize=9.5)
        fig.subplots_adjust(top=1 - 0.75 / (count * 1.85), bottom=0.02, hspace=0.28, wspace=0.03)
        fig.savefig(folder / f"native_chronological_page_{page + 1:02d}.png", dpi=110)
        plt.close(fig)
        pages += 1
    return pages


def movie(run, xy, folder, centers, radius, step_ms=12.0):
    z, rec = run["arrays"], run["record"]
    names = z["contact_names"].astype(str).tolist()
    table = np.asarray(z["centroid_ms"], float)
    modes = np.asarray(z["event_mode"], int)
    states = np.asarray(z["event_support_state"], int)
    primary = set(int(i) for i in z["primary_event_indices"])
    windows = np.asarray(z["windows_ms"], float)
    field = z["sheet_activity_counts"]
    env = np.asarray(z["contact_envelope"], float)
    dt = float(z["sheet_activity_frame_ms"])
    assert dt == float(z["contact_envelope_dt_ms"]) == 2.0
    centers_t = (np.arange(len(field)) + 0.5) * dt
    vmax = max(1.0, float(field.max()))
    emax = max(1e-12, float(env.max()))
    trace_t = np.asarray(z["trace_time_ms"], float) / 1000.0
    kernel = np.ones(20) / 20.0
    core_a = np.convolve(np.asarray(z["trace_coreA_mean_V"], float), kernel, mode="same")
    core_b = np.convolve(np.asarray(z["trace_coreB_mean_V"], float), kernel, mode="same")
    fig, axs = plt.subplots(2, 2, figsize=(11, 6.6), dpi=64)
    fig.subplots_adjust(left=0.06, right=0.975, bottom=0.09, top=0.84, hspace=0.5, wspace=0.3)
    raw, heat, timeline, state_ax = axs.ravel()
    raw.set(title="Native sheet: active E counts / 2 ms", xlabel="x (mm)", ylabel="y (mm)", xlim=(0, 20), ylim=(0, 20))
    im = raw.imshow(field[0], origin="lower", extent=(0, 20, 0, 20), cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
    for c in centers:
        raw.add_patch(Circle(c, radius, fill=False, edgecolor="cyan", lw=1))
    raw.scatter(*z["contact_xy_mm"].T, s=12, facecolors="none", edgecolors="white", lw=0.5)
    heat.set(title="Contact readout in the 250 ms window (x = centroid)", xlabel="Time from first centroid (ms)",
             yticks=range(len(names)), yticklabels=names)
    heat.tick_params(axis="y", labelsize=6)
    him = heat.imshow(env[:, :125], aspect="auto", origin="lower", cmap="magma", norm=PowerNorm(0.5, 0, emax),
                      extent=(-125, 125, -0.5, len(names) - 0.5))
    marks = heat.scatter(np.zeros(len(names)), np.arange(len(names)), s=12, c="cyan", marker="x")
    cursor = heat.axvline(0, c="white", ls="--", lw=0.8)
    rate = field.sum((1, 2))
    timeline.plot(centers_t / 1000, rate, lw=0.5, c="0.2")
    for i, (start, stop) in enumerate(windows):
        timeline.axvspan(start / 1000, stop / 1000, color=MODE_COLOR.get(int(modes[i]), "0.5"),
                         alpha=0.35 if i in primary else 0.12)
    tcursor = timeline.axvline(0, c="red", lw=1)
    timeline.set(xlim=(0, centers_t[-1] / 1000), title="Whole run: active E / 2 ms; shading = detected windows\n(mode colour; faint = excluded)",
                 xlabel="Simulation time (s)")
    state_ax.plot(trace_t, core_a, color=ARM_COLOR["B1"], lw=0.7, label="core A mean V")
    state_ax.plot(trace_t, core_b, color=ARM_COLOR["B2"], lw=0.7, label="core B mean V")
    scursor = state_ax.axvline(0, c="red", lw=1)
    allv = np.concatenate([core_a, core_b])
    state_ax.set(xlim=(0, centers_t[-1] / 1000), ylim=(float(np.percentile(allv, 2)), float(allv.max()) + 1.0),
                 title="Core state (1 ms trace, 20 ms smoothing; display clipped at the 2nd percentile)",
                 xlabel="Simulation time (s)", ylabel="mV")
    state_ax.legend(fontsize=7, frameon=False, loc="upper right")
    title = fig.text(0.5, 0.975, "", ha="center", va="top", fontsize=10.5)
    stamp = raw.text(0.02, 0.98, "", transform=raw.transAxes, va="top", color="white", fontsize=8.5)
    fig.text(0.5, 0.012, "Same graph, same cores, same noise law throughout; only the t=0 membrane voltage differs between arms. "
             "Model density is not patient HFO amplitude.", ha="center", fontsize=7.5)
    for ax in axs.ravel():
        ax.title.set_fontsize(8.5); ax.tick_params(labelsize=7)
    frames, durations, records = [], [], []
    stride = max(1, int(round(step_ms / dt)))
    for i, row in enumerate(table):
        start, stop = windows[i]
        ix = np.flatnonzero((centers_t >= start) & (centers_t < stop))[::stride]
        valid = np.isfinite(row)
        t0 = float(np.nanmin(row)) if valid.any() else float(start)
        him.set_data(env[:, int(start / dt):int(stop / dt)])
        him.set_extent((start - t0, stop - t0, -0.5, len(names) - 0.5))
        heat.set_xlim(start - t0, stop - t0)
        marks.set_offsets(np.column_stack([np.where(valid, row - t0, np.nan), np.arange(len(names))]))
        state = {1: "supported", 0: "uncertain", -1: "OOD"}[int(states[i])] if modes[i] >= 0 else "unreadable"
        gap = None if i == 0 else start - windows[i - 1][1]
        timing = ("first window" if gap is None else (f"overlap {-gap:.0f} ms" if gap < 0 else f"skipped gap {gap:.0f} ms"))
        first_frame = len(frames)
        for j, f in enumerate(ix):
            rel = centers_t[f] - t0
            im.set_data(field[f]); cursor.set_xdata([rel] * 2)
            tcursor.set_xdata([centers_t[f] / 1000] * 2); scursor.set_xdata([centers_t[f] / 1000] * 2)
            title.set_text(f"{ARM_LABEL[rec['arm']]} | graph {rec['topology_seed']} | seed {rec['dynamics_seed']} | window {i + 1}/{len(table)}\n"
                           f"M{modes[i]} / {state} / {'primary' if i in primary else 'excluded from the primary table'} | {timing}")
            stamp.set_text(f"t = {centers_t[f] / 1000:.3f} s\ncentroid lag {rel:+.0f} ms")
            fig.canvas.draw()
            rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            frames.append(Image.fromarray(rgb).quantize(colors=96, method=Image.Quantize.FASTOCTREE))
            durations.append(400 if j == 0 else 60)
        records.append({"detected_event_number": i + 1, "window_ms": [float(start), float(stop)],
                        "primary": i in primary, "mode": int(modes[i]), "support_state": int(states[i]),
                        "first_gif_frame": first_frame, "n_frames": int(len(ix))})
    plt.close(fig)
    path = folder / "all_chronological_events.gif"
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False, disposal=2)
    expected = len(frames)
    del frames
    gc.collect()
    decoded = 0
    if expected:
        with Image.open(path) as gif:
            for k in range(gif.n_frames):
                gif.seek(k); gif.load(); decoded += 1
                assert gif.info["duration"] == durations[k]
    return {"path": str(path), "sha256": rt.sha(path) if expected else None, "frames": expected,
            "frames_decoded": decoded, "all_frames_decoded": decoded == expected,
            "events": records, "n_windows": int(len(table)),
            "all_windows_rendered": all(r["n_frames"] > 0 for r in records) and len(records) == len(table),
            "frame_step_ms": step_ms, "first_frame_duration_ms": 400, "frame_duration_ms": 60,
            "selection": "lowest dynamics seed of the stage, all three arms, every detected window chronologically; no likeness selection",
            "aligned_across_arms_by_event_number": False, "agent_visual_review": False, "human_acceptance": False}


def write_event_table(run, folder):
    z, rec = run["arrays"], run["record"]
    names = z["contact_names"].astype(str).tolist()
    centroid = np.asarray(z["centroid_ms"], float)
    with open(folder / "event_table.csv", "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["detected_index", "window_start_ms", "window_end_ms", "event_time_ms", "primary_eligible",
                         "exclusion_reasons", "mode", "support_state", "distance_mode0", "distance_mode1"] + [f"centroid_{n}_ms" for n in names])
        for event in rec["events"]:
            i = event["detected_index"]
            writer.writerow([i, *event["window_ms"], event["event_time_ms"], event["primary_eligible"],
                             "|".join(event["primary_exclusion_reasons"]), event["mode"], event["support_state"],
                             event["distance_mode0"], event["distance_mode1"]] +
                            [("" if not np.isfinite(v) else f"{v:.2f}") for v in centroid[i]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=Path, default=rt.DESIGN_PATH)
    parser.add_argument("--stage", choices=("screen", "replication"), required=True)
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--source", choices=("formal", "qualification"), default="formal")
    args = parser.parse_args()
    design = rt.load_design(args.design)
    out = rt.output_root(design) / (args.stage if args.source == "formal" else "qualification/render_dry_run")
    runs = load_runs(design, args.stage, source=args.source)
    if not runs:
        raise RuntimeError("no completed runs")
    candidate = rt.candidate_record(design)
    centers = candidate["node_field"]["centers_mm"]
    radius = candidate["geometry"]["distance_cutoff_mm"]
    names = runs[0]["arrays"]["contact_names"].astype(str).tolist()
    xy = patient_positions(names)
    seed = min(r["record"]["dynamics_seed"] for r in runs)
    report = {"stage": args.stage, "minimum_seed": seed, "created_unix": time.time(), "featured": {}, "other_runs": {}}
    for run in runs:
        rec = run["record"]
        stem = run["job"]["stem"]
        if rec["dynamics_seed"] == seed:
            folder = out / "figures" / f"events_seed{seed}" / rec["arm"]
            folder.mkdir(parents=True, exist_ok=True)
            n = montage(run, xy, folder)
            pages = strobograms(run, folder, centers, radius)
            write_event_table(run, folder)
            entry = {"n_windows": n, "strobogram_pages": pages, "folder": str(folder)}
            if not args.skip_gif:
                entry["gif"] = movie(run, xy, folder, centers, radius)
            report["featured"][rec["arm"]] = entry
            print("featured", stem, entry.get("gif", {}).get("frames"), flush=True)
        else:
            folder = out / "native_frames" / stem
            folder.mkdir(parents=True, exist_ok=True)
            pages = strobograms(run, folder, centers, radius)
            write_event_table(run, folder)
            report["other_runs"][stem] = {"strobogram_pages": pages, "n_windows": int(len(run["arrays"]["windows_ms"]))}
        run["arrays"].close()
    rt.write(out / "figures" / "events_render_metadata.json", report)
    print({"stage": args.stage, "seed": seed, "featured": list(report["featured"]), "other_runs": len(report["other_runs"])})


if __name__ == "__main__":
    main()
