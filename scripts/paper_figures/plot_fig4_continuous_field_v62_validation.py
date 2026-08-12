"""Render the two frozen Fig.4 validation figures for rev10-SA V6.2.

This is a plotting-only consumer.  It does not rerun the SNN, alter the field,
or select a candidate.  Patient-template agreement fails closed when either
supervised direction lacks the frozen number of clean events.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# Figure production is small; unrestricted BLAS threads only add instability.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.interpolate import griddata
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_sa_observation_invariant_field_v6_2.json"
DEFAULT_OUTPUT = ROOT / (
    "results/topic4_sef_hfo/data_driven_core_field_rev10_sa/"
    "observation_invariant_field_v6_2_final"
)
DEFAULT_CANDIDATE = "v62_density_t050"
MODE_COLORS = ("#C43C39", "#277DA1")
SHAFT_COLORS = {"ICL": "#E67E22", "SCL": "#159EAE"}
GROUP_COLORS = ("#6A51A3", "#2A9D55")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def normalize_event_ranks(ranks):
    """Normalize finite ranks within each event and preserve missing contacts."""
    ranks = np.asarray(ranks, dtype=float)
    if ranks.ndim != 2:
        raise ValueError("ranks must have shape (events, contacts)")
    output = np.full_like(ranks, np.nan)
    for event_index, row in enumerate(ranks):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        values = row[finite]
        span = float(np.max(values) - np.min(values))
        output[event_index, finite] = (
            (values - np.min(values)) / span if span > 0.0 else 0.0
        )
    return output


def formal_clean_mask(onsets, labels, ood, groups, event_returned=None):
    """Couple direction identity, dual-shaft recruitment, and patient support."""
    onsets = np.asarray(onsets, dtype=float)
    labels = np.asarray(labels, dtype=int)
    ood = np.asarray(ood, dtype=bool)
    if labels.shape != (len(onsets),) or ood.shape != (len(onsets),):
        raise ValueError("event labels, OOD mask, and onsets must align")
    icl = np.isfinite(onsets[:, np.asarray(groups["ICL"], int)]).any(axis=1)
    scl = np.isfinite(onsets[:, np.asarray(groups["SCL"], int)]).any(axis=1)
    returned = (
        np.ones(len(onsets), bool) if event_returned is None
        else np.asarray(event_returned, bool)
    )
    if returned.shape != (len(onsets),):
        raise ValueError("event_returned must align with onsets")
    return returned & icl & scl & ~ood


def matrix_acceptance_status(labels, clean, required_per_mode):
    counts = np.bincount(np.asarray(labels, int)[np.asarray(clean, bool)], minlength=2)
    valid = bool(np.all(counts >= int(required_per_mode)))
    return valid, counts


def _column_stats(values):
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    mean = np.divide(
        np.nansum(values, axis=0), count,
        out=np.full(values.shape[1], np.nan), where=count > 0,
    )
    centered = np.where(finite, values - mean[None, :], 0.0)
    variance = np.divide(
        np.sum(centered ** 2, axis=0), count,
        out=np.full(values.shape[1], np.nan), where=count > 0,
    )
    return mean, np.sqrt(variance)


def _column_quantile(values, quantile):
    values = np.asarray(values, float)
    output = np.full(values.shape[1], np.nan)
    for column in range(values.shape[1]):
        finite = values[:, column]
        finite = finite[np.isfinite(finite)]
        if len(finite):
            output[column] = np.quantile(finite, quantile)
    return output


def _event_slope(row):
    row = np.asarray(row, float)
    finite = np.isfinite(row)
    if np.sum(finite) < 2:
        return 0.0
    return float(np.polyfit(np.flatnonzero(finite), row[finite], 1)[0])


def _classifier_from_manifest(manifest):
    classifier = dict(manifest["direction_classifier"])
    for key in (
        "coef", "class_centers", "class_precisions", "ood_distance_thresholds",
    ):
        classifier[key] = np.asarray(classifier[key], dtype=float)
    return classifier


def _field_grid(positions, values, *, size=84, limit=20.0):
    axis = np.linspace(0.0, float(limit), int(size))
    xx, yy = np.meshgrid(axis, axis)
    zz = griddata(positions, values, (xx, yy), method="linear")
    if np.isnan(zz).any():
        nearest = griddata(positions, values, (xx, yy), method="nearest")
        zz = np.where(np.isfinite(zz), zz, nearest)
    return xx, yy, zz


def _plot_contacts(ax, xy, names, shaft_ids, *, annotate=False, z=None):
    for shaft in ("ICL", "SCL"):
        selected = np.flatnonzero(np.asarray(shaft_ids).astype(str) == shaft)
        if not len(selected):
            continue
        kwargs = {} if z is None else {"zs": np.asarray(z)[selected]}
        if z is None:
            ax.plot(xy[selected, 0], xy[selected, 1], color=SHAFT_COLORS[shaft],
                    lw=1.15, alpha=0.90, zorder=5)
            ax.scatter(xy[selected, 0], xy[selected, 1], s=34,
                       color=SHAFT_COLORS[shaft], edgecolor="white",
                       linewidth=0.65, zorder=6)
        else:
            ax.plot(xy[selected, 0], xy[selected, 1], np.asarray(z)[selected],
                    color=SHAFT_COLORS[shaft], lw=1.15, zorder=6)
            ax.scatter(xy[selected, 0], xy[selected, 1], **kwargs, s=27,
                       color=SHAFT_COLORS[shaft], edgecolor="white",
                       linewidth=0.55, depthshade=False, zorder=7)
        if annotate and z is None:
            for index in selected:
                ax.annotate(str(names[index]), xy[index], xytext=(3, 3),
                            textcoords="offset points", fontsize=6.2,
                            color=SHAFT_COLORS[shaft], zorder=8)


def _load_bundle(config_path, output_root, candidate_id):
    sys.path.insert(0, str(ROOT))
    from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
        load_scoring_contract,
    )
    from src.topic4_shaft_aware import contract_groups  # noqa: E402
    from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402

    config_path = Path(config_path).resolve()
    output_root = Path(output_root).resolve()
    config = _json(config_path)
    manifest_path = output_root / "candidate_manifest.json"
    summary_path = output_root / "v62_mode_boundary_final_search_summary.json"
    manifest, summary = _json(manifest_path), _json(summary_path)
    if manifest["config"]["sha256"] != _sha256(config_path):
        raise RuntimeError("candidate manifest and V6.2 config do not match")
    if candidate_id != summary["display_candidate_id"]:
        raise RuntimeError("Fig.4 validation must use the frozen display candidate")

    contract_path = ROOT / config["inputs"]["contact_contract"]["path"]
    target_path = ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]
    floor_path = ROOT / config["inputs"]["shaft_aware_floors"]["path"]
    if _sha256(contract_path) != config["inputs"]["contact_contract"]["sha256"]:
        raise RuntimeError("contact contract hash mismatch")
    if _sha256(target_path) != config["inputs"]["shaft_aware_target_npz"]["sha256"]:
        raise RuntimeError("patient target hash mismatch")
    contract = _json(contract_path)
    groups = contract_groups(contract)
    names, embedding, _, _ = load_scoring_contract(
        target_path, floor_path, "FULL_TIMING",
        fixed_events_per_mode=int(config["search"]["objective"]["fixed_events_per_mode"]),
    )
    classifier = _classifier_from_manifest(manifest)

    blocks, records, worker_inputs = [], [], []
    static = None
    cursor = 0
    for seed in config["search"]["network_seeds"]:
        stem = output_root / "workers" / f"{candidate_id}_seed_{seed}"
        json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        payload = _json(json_path)
        if payload["arrays"]["sha256"] != _sha256(npz_path):
            raise RuntimeError(f"worker array hash mismatch: {npz_path.name}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            worker_names = np.asarray(loaded["contact_names"]).astype(str)
            if not np.array_equal(worker_names, names.astype(str)):
                raise RuntimeError(f"contact order changed: {npz_path.name}")
            onsets = np.asarray(loaded["onsets"], float)
            ranks = np.asarray(loaded["ranks"], float)
            block = {
                "seed": int(seed), "npz_path": npz_path,
                "onsets": onsets, "ranks": ranks,
                "event_t_on_ms": np.asarray(loaded["event_t_on_ms"], float),
                "event_t_off_ms": np.asarray(loaded["event_t_off_ms"], float),
                "event_returned": np.asarray(loaded["event_returned"], bool),
                "contact_envelope": np.asarray(loaded["contact_envelope"], float),
                "contact_envelope_dt_ms": float(loaded["contact_envelope_dt_ms"]),
            }
            if static is None:
                static = {
                    "contact_names": worker_names,
                    "shaft_ids": np.asarray(loaded["shaft_ids"]).astype(str),
                    "contact_xy_mm": np.asarray(loaded["contact_xy_mm"], float),
                    "positions_E": np.asarray(loaded["positions_E"], float),
                    "h": np.asarray(loaded["h"], float),
                    "delta_vtheta": np.asarray(loaded["delta_vtheta"], float),
                }
        blocks.append(block)
        records.extend({"seed": int(seed), "local_index": local,
                        "global_index": cursor + local}
                       for local in range(len(onsets)))
        cursor += len(onsets)
        worker_inputs.append({
            "seed": int(seed), "json": str(json_path),
            "json_sha256": _sha256(json_path), "npz": str(npz_path),
            "npz_sha256": _sha256(npz_path),
        })
    onsets = np.concatenate([block["onsets"] for block in blocks], axis=0)
    ranks = np.concatenate([block["ranks"] for block in blocks], axis=0)
    event_returned = np.concatenate([
        block["event_returned"] for block in blocks
    ], axis=0)
    assigned = assign_direction_modes(
        onsets, groups=groups, embedding=embedding, classifier=classifier,
    )
    all_event_clean = formal_clean_mask(
        onsets, assigned["labels"], assigned["ood"], groups,
    )
    clean = formal_clean_mask(
        onsets, assigned["labels"], assigned["ood"], groups,
        event_returned=event_returned,
    )
    required = int(config["search"]["objective"]["fixed_events_per_mode"])
    matrix_valid, clean_counts = matrix_acceptance_status(
        assigned["labels"], clean, required,
    )
    expected = summary["candidate_details"][candidate_id][
        "mode_conditioned_joint_support"
    ]
    all_event_clean_counts = np.bincount(
        np.asarray(assigned["labels"], int)[all_event_clean], minlength=2,
    )
    if all_event_clean_counts.tolist() != [
        expected["A"]["n_joint_in_distribution"],
        expected["B"]["n_joint_in_distribution"],
    ]:
        raise RuntimeError("reconstructed all-event pool disagrees with summary")
    with np.load(target_path, allow_pickle=False) as target:
        patient = {key: np.asarray(target[key]) for key in (
            "patient_train_ranks", "patient_train_old_labels",
            "patient_train_block_ids",
        )}
    return {
        "config": config, "config_path": config_path,
        "manifest_path": manifest_path, "summary": summary,
        "summary_path": summary_path, "candidate_id": candidate_id,
        "groups": groups, "blocks": blocks, "records": records,
        "static": static, "onsets": onsets, "ranks": ranks,
        "labels": np.asarray(assigned["labels"], int),
        "ood": np.asarray(assigned["ood"], bool),
        "embedding": np.asarray(assigned["embedding"], float),
        "probability_B": np.asarray(assigned["probability_B"], float),
        "event_returned": event_returned,
        "all_event_clean_counts": all_event_clean_counts,
        "clean": clean, "clean_counts": clean_counts,
        "required_per_mode": required, "matrix_valid": matrix_valid,
        "patient": patient, "worker_inputs": worker_inputs,
        "target_path": target_path, "contract_path": contract_path,
    }


def _representative_pair(bundle):
    labels, clean = bundle["labels"], bundle["clean"]
    a_indices = np.flatnonzero(clean & (labels == 0))
    if not len(a_indices):
        raise RuntimeError("no formal mode-A event is available for Fig.4A")
    a_index = int(a_indices[0])
    a_record = bundle["records"][a_index]
    same_seed_b = [
        index for index in np.flatnonzero(clean & (labels == 1))
        if bundle["records"][int(index)]["seed"] == a_record["seed"]
    ]
    if not same_seed_b:
        raise RuntimeError("the mode-A network has no formal mode-B event")
    block = next(row for row in bundle["blocks"] if row["seed"] == a_record["seed"])
    a_local = a_record["local_index"]
    a_center = 0.5 * (
        block["event_t_on_ms"][a_local] + block["event_t_off_ms"][a_local]
    )
    b_index = min(
        same_seed_b,
        key=lambda index: abs(
            0.5 * (
                block["event_t_on_ms"][bundle["records"][int(index)]["local_index"]]
                + block["event_t_off_ms"][bundle["records"][int(index)]["local_index"]]
            ) - a_center
        ),
    )
    return a_index, int(b_index), block


def _plot_landscape(ax, static):
    pos, h = static["positions_E"], static["h"]
    xx, yy, hh = _field_grid(pos, h)
    vmax = max(float(np.quantile(h, 0.995)), 1e-6)
    surface = ax.plot_surface(
        xx, yy, np.minimum(hh, vmax), cmap="plasma", vmin=0.0, vmax=vmax,
        linewidth=0, antialiased=True, shade=False, alpha=0.97,
        rasterized=True,
    )
    ax.contour(xx, yy, hh, zdir="z", offset=0.0, levels=7, cmap="plasma",
               linewidths=0.55, alpha=0.75)
    contact_h = griddata(pos, h, static["contact_xy_mm"], method="linear")
    contact_h = np.nan_to_num(contact_h, nan=0.0) + 0.025 * vmax
    _plot_contacts(
        ax, static["contact_xy_mm"], static["contact_names"],
        static["shaft_ids"], z=contact_h,
    )
    ax.set(xlim=(0, 20), ylim=(0, 20), zlim=(0, 1.12 * vmax),
           xlabel="sheet x (mm)", ylabel="sheet y (mm)", zlabel="h")
    ax.set_title("continuous field landscape", weight="bold", pad=7)
    ax.view_init(elev=31, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.58))
    ax.tick_params(labelsize=7.2, pad=1)
    colorbar = plt.colorbar(surface, ax=ax, fraction=0.04, pad=0.01, shrink=0.72)
    colorbar.set_label("pathology field h", fontsize=8)


def _style_sheet(ax, title):
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    ax.set_aspect("equal")
    ax.set_xlabel("sheet x (mm)")
    ax.set_title(title, weight="bold", pad=7)
    ax.spines[["top", "right"]].set_visible(False)


def _plot_delta(ax, static):
    delta = static["delta_vtheta"]
    vmax = max(float(np.quantile(np.abs(delta), 0.995)), 1e-6)
    image = ax.scatter(
        static["positions_E"][:, 0], static["positions_E"][:, 1],
        c=delta, s=2.7, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        lw=0, alpha=0.80, rasterized=True,
    )
    _plot_contacts(ax, static["contact_xy_mm"], static["contact_names"],
                   static["shaft_ids"])
    _style_sheet(ax, r"actual $\Delta V_\theta=-h d$ (mV)")
    colorbar = plt.colorbar(image, ax=ax, fraction=0.047, pad=0.025)
    colorbar.set_label(r"$V_{\theta,i}-V_{\theta,0}$ (mV)", fontsize=8)


def _earliest_contact_frequency(bundle, mode):
    selected = bundle["onsets"][bundle["clean"] & (bundle["labels"] == mode)]
    counts = np.zeros(bundle["onsets"].shape[1], float)
    for onset in selected:
        finite = np.isfinite(onset)
        if not np.any(finite):
            continue
        earliest = finite & np.isclose(onset, np.nanmin(onset))
        counts[earliest] += 1.0 / int(np.sum(earliest))
    return counts / max(1, len(selected))


def _plot_contact_event(ax, bundle, global_index, mode):
    static = bundle["static"]
    pos, h = static["positions_E"], static["h"]
    xx, yy, hh = _field_grid(pos, h)
    levels = np.unique(np.quantile(h, (0.75, 0.90, 0.98)))
    ax.contour(xx, yy, hh, levels=levels, colors="#303030",
               linestyles="--", linewidths=0.65, alpha=0.55)
    xy, onset = static["contact_xy_mm"], bundle["onsets"][global_index]
    finite = np.isfinite(onset)
    ax.scatter(xy[~finite, 0], xy[~finite, 1], s=34, facecolor="white",
               edgecolor="#B9B9B9", linewidth=0.8, zorder=4)
    relative = onset[finite] - np.nanmin(onset)
    span = max(float(np.max(relative, initial=0.0)), 1.0)
    image = ax.scatter(
        xy[finite, 0], xy[finite, 1], c=relative / span, s=68,
        cmap="viridis", vmin=0.0, vmax=1.0, edgecolor="white",
        linewidth=0.75, zorder=7,
    )
    _plot_contacts(ax, xy, static["contact_names"], static["shaft_ids"])
    earliest = finite & np.isclose(onset, np.nanmin(onset))
    ax.scatter(xy[earliest, 0], xy[earliest, 1], marker="*", s=125,
               color="#111111", edgecolor="white", linewidth=0.55, zorder=9)
    density = _earliest_contact_frequency(bundle, mode)
    present = density > 0
    ax.scatter(xy[present, 0], xy[present, 1], s=52 + 150 * density[present],
               facecolor="none", edgecolor=MODE_COLORS[mode], linewidth=1.1,
               alpha=0.78, zorder=8)
    _style_sheet(ax, f"model mode {'AB'[mode]}")
    ax.text(0.03, 0.96,
            f"formal joint+ID n={int(bundle['clean_counts'][mode])}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            color=MODE_COLORS[mode], weight="bold")
    return image


def _nice_scale(value):
    if not np.isfinite(value) or value <= 0.0:
        return 1.0
    exponent = np.floor(np.log10(value))
    fraction = value / (10.0 ** exponent)
    level = next(item for item in (1.0, 2.0, 5.0, 10.0) if fraction <= item)
    return float(level * (10.0 ** exponent))


def _plot_readout(ax, bundle, a_index, b_index, block):
    records = bundle["records"]
    pair = [(0, a_index), (1, b_index)]
    times = []
    for _, global_index in pair:
        local = records[global_index]["local_index"]
        times.extend([block["event_t_on_ms"][local], block["event_t_off_ms"][local]])
    width = max(1000.0, max(times) - min(times) + 280.0)
    center = 0.5 * (min(times) + max(times))
    duration = block["contact_envelope"].shape[1] * block["contact_envelope_dt_ms"]
    start = max(0.0, center - 0.5 * width)
    stop = min(duration, start + width)
    start = max(0.0, stop - width)
    dt = block["contact_envelope_dt_ms"]
    sample = (np.arange(block["contact_envelope"].shape[1]) * dt >= start) & (
        np.arange(block["contact_envelope"].shape[1]) * dt <= stop
    )
    trace = block["contact_envelope"][:, sample]
    t = np.arange(block["contact_envelope"].shape[1])[sample] * dt - start
    union = np.zeros(trace.shape[0], bool)
    for _, global_index in pair:
        union |= np.isfinite(bundle["onsets"][global_index])
    selected_contacts = np.flatnonzero(union)
    common_scale = _nice_scale(float(np.quantile(trace[selected_contacts], 0.99)))
    gain = 0.68 / common_scale
    offsets = np.arange(len(selected_contacts), dtype=float) * 1.18
    for mode, global_index in pair:
        local = records[global_index]["local_index"]
        ax.axvspan(block["event_t_on_ms"][local] - start,
                   block["event_t_off_ms"][local] - start,
                   color=MODE_COLORS[mode], alpha=0.14, lw=0)
    for row, contact in enumerate(selected_contacts):
        shaft = bundle["static"]["shaft_ids"][contact]
        ax.plot(t, trace[contact] * gain + offsets[row],
                color=SHAFT_COLORS[shaft], lw=0.82, alpha=0.95)
    for mode, global_index in pair:
        onset = bundle["onsets"][global_index]
        xs, ys = [], []
        for row, contact in enumerate(selected_contacts):
            if not np.isfinite(onset[contact]):
                continue
            x = onset[contact] - start
            sample_index = int(np.clip(round(onset[contact] / dt), 0,
                                       block["contact_envelope"].shape[1] - 1))
            y = offsets[row] + block["contact_envelope"][contact, sample_index] * gain
            xs.append(x)
            ys.append(y)
        order = np.argsort(xs)
        ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], color="#111111",
                lw=0.65, alpha=0.72, zorder=7)
        ax.scatter(xs, ys, s=9, color="#111111", zorder=8)
    names = bundle["static"]["contact_names"][selected_contacts]
    ax.set_xlim(0, width)
    ax.set_ylim(-0.65, offsets[-1] + 1.05)
    ax.set_yticks(offsets, names, fontsize=7.7)
    ax.invert_yaxis()
    ax.set_xlabel("simulation time (ms)")
    ax.set_ylabel("model-current envelope", labelpad=3)
    ax.set_title("direct electrode readout", loc="left", weight="bold", pad=7)
    ax.spines[["top", "right"]].set_visible(False)
    mode_legend = ax.legend(handles=[
        Patch(facecolor=MODE_COLORS[0], alpha=0.20, label="model mode A"),
        Patch(facecolor=MODE_COLORS[1], alpha=0.20, label="model mode B"),
    ], frameon=False, fontsize=7.8, ncol=2, loc="upper center",
       bbox_to_anchor=(0.50, -0.13))
    ax.add_artist(mode_legend)
    ax.legend(handles=[
        Line2D([0], [0], color=SHAFT_COLORS["ICL"], lw=1.8, label="ICL"),
        Line2D([0], [0], color=SHAFT_COLORS["SCL"], lw=1.8, label="SCL"),
    ], title="contact family", frameon=False, fontsize=7.7,
       title_fontsize=7.7, ncol=2, loc="upper right")
    bar_x, bar_y = 0.025 * width, offsets[0] + 0.02
    ax.plot([bar_x, bar_x], [bar_y, bar_y + 0.68], color="#111111", lw=1.4,
            clip_on=False)
    ax.text(bar_x + 0.012 * width, bar_y + 0.34,
            f"{common_scale:g} a.u.\nmodel-current envelope",
            va="center", fontsize=7.0)
    return {
        "seed": int(block["seed"]), "start_ms": float(start),
        "stop_ms": float(stop), "common_amplitude_scale_au": common_scale,
        "active_contact_count": int(len(selected_contacts)),
    }


def _save_figure(fig, stem):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return {
        "png": {"path": str(stem.with_suffix(".png")),
                "sha256": _sha256(stem.with_suffix(".png"))},
        "pdf": {"path": str(stem.with_suffix(".pdf")),
                "sha256": _sha256(stem.with_suffix(".pdf"))},
    }


def _render_direct(bundle, output_dir):
    a_index, b_index, block = _representative_pair(bundle)
    fig = plt.figure(figsize=(22.0, 4.9), facecolor="white")
    grid = fig.add_gridspec(
        1, 5, width_ratios=(1.12, 1.0, 1.0, 1.0, 2.35),
        left=0.036, right=0.992, bottom=0.18, top=0.87, wspace=0.31,
    )
    axes = [fig.add_subplot(grid[0, 0], projection="3d")]
    axes.extend(fig.add_subplot(grid[0, index]) for index in range(1, 5))
    _plot_landscape(axes[0], bundle["static"])
    _plot_delta(axes[1], bundle["static"])
    image_a = _plot_contact_event(axes[2], bundle, a_index, 0)
    _plot_contact_event(axes[3], bundle, b_index, 1)
    colorbar = fig.colorbar(image_a, ax=axes[2:4], fraction=0.025, pad=0.025)
    colorbar.set_ticks((0, 1))
    colorbar.set_ticklabels(("early", "late"))
    readout = _plot_readout(axes[4], bundle, a_index, b_index, block)
    fig.suptitle(
        "Continuous data-driven field: direct model readout  |  "
        "fresh-network coexistence not confirmed",
        fontsize=13.0, weight="bold", y=0.985,
    )
    stem = Path(output_dir) / "fig4a_continuous_field_v62_direct_readout"
    files = _save_figure(fig, stem)
    representative = []
    for mode, index in ((0, a_index), (1, b_index)):
        record = bundle["records"][index]
        local_block = next(row for row in bundle["blocks"]
                           if row["seed"] == record["seed"])
        representative.append({
            "mode": "AB"[mode], "global_index": int(index),
            "seed": int(record["seed"]), "local_index": int(record["local_index"]),
            "t_on_ms": float(local_block["event_t_on_ms"][record["local_index"]]),
            "t_off_ms": float(local_block["event_t_off_ms"][record["local_index"]]),
            "n_recruited_contacts": int(np.isfinite(bundle["onsets"][index]).sum()),
        })
    metadata = _base_metadata(bundle, "Fig4A continuous-field direct readout")
    metadata.update({
        "files": files, "representative_events": representative,
        "direct_readout": readout,
        "readout_boundary": (
            "worker artifacts retain the continuous 30-80 Hz model-current "
            "envelope and contact onsets, not raw band-passed current or clinical "
            "SEEG voltage; spatial event panels therefore show contact-level onset"
        ),
    })
    Path(str(stem) + "_metadata.json").write_text(json.dumps(metadata, indent=2))
    return stem, metadata


def _canonicalize_kmeans(labels, z):
    labels = np.asarray(labels, int)
    means = [float(np.mean(z[labels == cluster, 0])) for cluster in (0, 1)]
    order = np.argsort(means)
    mapping = {int(old): int(new) for new, old in enumerate(order)}
    return np.asarray([mapping[int(value)] for value in labels], int), mapping


def _kmeans_audit(z, direction_labels):
    with threadpool_limits(limits=1):
        primary_raw = KMeans(
            n_clusters=2, n_init=100, random_state=0,
        ).fit_predict(z)
        primary, primary_mapping = _canonicalize_kmeans(primary_raw, z)
        label_sets = [primary]
        for seed in range(1, 12):
            raw = KMeans(
                n_clusters=2, n_init=20, random_state=seed,
            ).fit_predict(z)
            canonical, _ = _canonicalize_kmeans(raw, z)
            label_sets.append(canonical)
    pairwise = [
        adjusted_mutual_info_score(label_sets[left], label_sets[right])
        for left in range(len(label_sets)) for right in range(left + 1, len(label_sets))
    ]
    labels = primary
    contingency = np.zeros((2, 2), int)
    for cluster, mode in zip(labels, direction_labels):
        contingency[int(cluster), int(mode)] += 1
    return labels, {
        "algorithm": "KMeans", "n_clusters": 2, "primary_n_init": 100,
        "stability_n_init": 20, "random_state_primary": 0,
        "stability_random_states": list(range(12)),
        "cluster_counts": np.bincount(labels, minlength=2).tolist(),
        "pairwise_seed_ami_median": float(np.median(pairwise)),
        "pairwise_seed_ami_min": float(np.min(pairwise)),
        "ami_with_supervised_direction": float(
            adjusted_mutual_info_score(labels, direction_labels)
        ),
        "direction_contingency": contingency.tolist(),
        "direction_purity": [
            float(np.max(row) / max(1, np.sum(row))) for row in contingency
        ],
        "raw_to_display_mapping": primary_mapping,
    }


def _profile(matrix, selected):
    values = np.asarray(matrix, float)[np.asarray(selected, bool)]
    return _column_stats(values)


def _patient_profiles(bundle):
    ranks = normalize_event_ranks(bundle["patient"]["patient_train_ranks"])
    labels = bundle["patient"]["patient_train_old_labels"].astype(int)
    blocks = bundle["patient"]["patient_train_block_ids"]
    profiles, lows, highs = [], [], []
    for mode in (0, 1):
        profile, _ = _column_stats(ranks[labels == mode])
        block_profiles = []
        for block in np.unique(blocks[labels == mode]):
            selected = (labels == mode) & (blocks == block)
            if np.any(selected):
                block_profiles.append(_column_stats(ranks[selected])[0])
        block_profiles = np.asarray(block_profiles, float)
        profiles.append(profile)
        lows.append(_column_quantile(block_profiles, 0.05))
        highs.append(_column_quantile(block_profiles, 0.95))
    return np.asarray(profiles), np.asarray(lows), np.asarray(highs)


def _within_cluster_tau(norm_ranks, labels):
    output = []
    for cluster in (0, 1):
        values = norm_ranks[labels == cluster]
        tau = []
        for left in range(len(values)):
            for right in range(left + 1, len(values)):
                shared = np.isfinite(values[left]) & np.isfinite(values[right])
                if np.sum(shared) >= 3:
                    value = kendalltau(values[left, shared], values[right, shared]).statistic
                    if np.isfinite(value):
                        tau.append(float(value))
        output.append(float(np.median(tau)) if tau else None)
    return output


def _descriptive_similarity(model_profiles, patient_profiles):
    matrix = np.full((2, 2), np.nan)
    for row in range(2):
        for column in range(2):
            finite = np.isfinite(model_profiles[row]) & np.isfinite(patient_profiles[column])
            if np.sum(finite) >= 3:
                matrix[row, column] = spearmanr(
                    model_profiles[row, finite], patient_profiles[column, finite]
                ).statistic
    return matrix


def _render_kmeans(bundle, output_dir):
    clean_index = np.flatnonzero(bundle["clean"])
    clean_ranks = normalize_event_ranks(bundle["ranks"][clean_index])
    clean_direction = bundle["labels"][clean_index]
    clean_z = bundle["embedding"][clean_index]
    kmeans_labels, audit = _kmeans_audit(clean_z, clean_direction)
    patient_profiles, patient_low, patient_high = _patient_profiles(bundle)
    model_profiles = np.asarray([
        _column_stats(clean_ranks[kmeans_labels == cluster])[0]
        for cluster in (0, 1)
    ])
    descriptive_matrix = _descriptive_similarity(model_profiles, patient_profiles)
    within_tau = _within_cluster_tau(clean_ranks, kmeans_labels)

    slope = np.asarray([_event_slope(row) for row in clean_ranks])
    order = np.lexsort((slope, kmeans_labels))
    ordered_labels = kmeans_labels[order]
    split = int(np.sum(ordered_labels == 0))
    names = bundle["static"]["contact_names"]

    fig = plt.figure(figsize=(18.8, 5.2), facecolor="white")
    grid = fig.add_gridspec(
        1, 5, width_ratios=(2.9, 0.10, 1.0, 1.45, 1.25),
        left=0.052, right=0.985, bottom=0.18, top=0.84, wspace=0.38,
    )
    ax_heat = fig.add_subplot(grid[0, 0])
    ax_cbar = fig.add_subplot(grid[0, 1])
    ax_dist = fig.add_subplot(grid[0, 2])
    ax_profile = fig.add_subplot(grid[0, 3])
    ax_matrix = fig.add_subplot(grid[0, 4])

    shown = np.ma.masked_invalid(clean_ranks[order].T)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#D7D7D7")
    image = ax_heat.imshow(shown, aspect="auto", origin="upper",
                           interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    if 0 < split < len(order):
        ax_heat.axvline(split - 0.5, color="#B22222", lw=1.4)
    ax_heat.set_yticks(np.arange(len(names)), names, fontsize=8.2)
    ax_heat.set_xlabel("formal clean events")
    ax_heat.set_ylabel("fixed contact identity")
    ax_heat.set_title("clustered event heatmap", weight="bold", pad=24)
    for cluster, left, right in ((0, 0, split), (1, split, len(order))):
        ax_heat.text(0.5 * (left + right - 1), 1.012,
                     f"unmapped group {cluster + 1}  n={right - left}",
                     transform=ax_heat.get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7.7,
                     color=GROUP_COLORS[cluster],
                     bbox={"facecolor": "white", "edgecolor": "none",
                           "alpha": 0.78, "pad": 1.4})
    colorbar = fig.colorbar(image, cax=ax_cbar)
    colorbar.set_ticks((0, 1))
    colorbar.set_ticklabels(("first", "last"))
    colorbar.set_label("within-event rank", fontsize=8)

    values, positions = [], []
    for contact in range(len(names)):
        finite = clean_ranks[:, contact]
        finite = finite[np.isfinite(finite)]
        if len(finite):
            values.append(finite)
            positions.append(contact)
    violin = ax_dist.violinplot(values, positions=positions, vert=False,
                                widths=0.78, showmeans=False, showmedians=True,
                                showextrema=False)
    for body in violin["bodies"]:
        body.set_facecolor("#727272")
        body.set_edgecolor("none")
        body.set_alpha(0.45)
    violin["cmedians"].set_color("#222222")
    ax_dist.set_xlim(-0.04, 1.04)
    ax_dist.set_ylim(ax_heat.get_ylim())
    ax_dist.set_yticks(ax_heat.get_yticks(), [])
    ax_dist.tick_params(axis="y", length=0)
    ax_dist.set_xlabel("within-event rank")
    ax_dist.set_title("rank distribution", weight="bold", pad=7)
    ax_dist.spines[["top", "right"]].set_visible(False)

    y = np.arange(len(names))
    for cluster in (0, 1):
        mean, std = _profile(clean_ranks, kmeans_labels == cluster)
        finite = np.isfinite(mean)
        ax_profile.fill_betweenx(y[finite], (mean - std)[finite],
                                 (mean + std)[finite], color=GROUP_COLORS[cluster],
                                 alpha=0.12, lw=0)
        ax_profile.plot(mean[finite], y[finite], "-o",
                        color=GROUP_COLORS[cluster], lw=1.8, ms=3.6,
                        label=f"group {cluster + 1}")
    for mode in (0, 1):
        finite = np.isfinite(patient_profiles[mode])
        ax_profile.fill_betweenx(y[finite], patient_low[mode, finite],
                                 patient_high[mode, finite],
                                 color=MODE_COLORS[mode], alpha=0.06, lw=0)
        ax_profile.plot(patient_profiles[mode, finite], y[finite], "--",
                        color=MODE_COLORS[mode], lw=1.35,
                        label=f"patient {'AB'[mode]}")
    ax_profile.set_xlim(-0.08, 1.08)
    ax_profile.set_ylim(ax_heat.get_ylim())
    ax_profile.set_yticks(ax_heat.get_yticks(), [])
    ax_profile.tick_params(axis="y", length=0)
    ax_profile.set_xlabel("mean normalized rank")
    ax_profile.set_title("cluster rank profile", weight="bold", pad=7)
    ax_profile.legend(frameon=False, fontsize=7.2, loc="upper right", ncol=2,
                      columnspacing=0.7, handlelength=1.5)
    ax_profile.spines[["top", "right"]].set_visible(False)

    ax_matrix.set_facecolor("#EFEFEF")
    ax_matrix.set_xlim(-0.5, 1.5)
    ax_matrix.set_ylim(1.5, -0.5)
    for value in (-0.5, 0.5, 1.5):
        ax_matrix.axvline(value, color="white", lw=2)
        ax_matrix.axhline(value, color="white", lw=2)
    ax_matrix.set_xticks((0, 1), ("patient A", "patient B"), fontsize=8.5)
    ax_matrix.set_yticks((0, 1), ("model A", "model B"), fontsize=8.5)
    ax_matrix.set_aspect("equal")
    ax_matrix.set_title("model vs patient", weight="bold", pad=7)
    ax_matrix.text(0.5, 0.42, "N/A", transform=ax_matrix.transAxes,
                   ha="center", va="center", fontsize=21, weight="bold",
                   color="#9B2F2A")
    ax_matrix.text(
        0.5, 0.24,
        f"mode A support {int(bundle['clean_counts'][0])} < "
        f"{bundle['required_per_mode']}",
        transform=ax_matrix.transAxes, ha="center", va="center",
        fontsize=8.2, color="#9B2F2A",
    )
    for spine in ax_matrix.spines.values():
        spine.set_color("#C43C39")
        spine.set_linewidth(2.0)

    fig.suptitle(
        "KMeans modes against patient data  |  consistency not evaluable",
        fontsize=13.0, weight="bold", y=0.985,
    )
    fig.text(
        0.983, 0.045,
        f"formal A/B={int(bundle['clean_counts'][0])}/{int(bundle['clean_counts'][1])}"
        f"   clusters={audit['cluster_counts'][0]}/{audit['cluster_counts'][1]}"
        f"   KMeans stability AMI={audit['pairwise_seed_ami_median']:.2f}",
        ha="right", va="bottom", fontsize=8.6, color="0.32",
    )
    stem = Path(output_dir) / "fig4b_continuous_field_v62_kmeans_consistency"
    files = _save_figure(fig, stem)
    audit["within_cluster_pairwise_kendall_tau_median"] = within_tau
    audit["descriptive_cluster_vs_patient_spearman_not_for_acceptance"] = (
        descriptive_matrix.tolist()
    )
    metadata = _base_metadata(bundle, "Fig4B continuous-field KMeans consistency")
    metadata.update({
        "files": files, "kmeans": audit,
        "matrix_status": "NOT_EVALUABLE_INSUFFICIENT_MODE_A_SUPPORT",
        "matrix_valid": False,
        "matrix_rule": (
            "requires the frozen fixed_events_per_mode in both supervised "
            "directions after joint-shaft and patient-support filtering; KMeans "
            "clusters cannot substitute for a missing direction"
        ),
    })
    Path(str(stem) + "_metadata.json").write_text(json.dumps(metadata, indent=2))
    return stem, metadata


def _base_metadata(bundle, figure):
    return {
        "figure": figure, "plotting_only": True,
        "candidate_id": bundle["candidate_id"],
        "candidate_role": (
            "frozen display candidate; returned events only; not a selected success"
        ),
        "source_status": bundle["summary"]["status"],
        "selected_candidate_id": bundle["summary"]["selected_candidate_id"],
        "event_filter": (
            "supervised patient-trained direction AND both shafts recruited AND "
            "inside class-conditional patient support"
        ),
        "n_detected_events": int(len(bundle["onsets"])),
        "n_formal_clean_events": int(np.sum(bundle["clean"])),
        "formal_clean_mode_counts": {
            "A": int(bundle["clean_counts"][0]),
            "B": int(bundle["clean_counts"][1]),
        },
        "historical_all_event_clean_mode_counts": {
            "A": int(bundle["all_event_clean_counts"][0]),
            "B": int(bundle["all_event_clean_counts"][1]),
        },
        "detected_event_count": int(len(bundle["event_returned"])),
        "returned_event_count": int(np.sum(bundle["event_returned"])),
        "nonreturned_event_count_excluded": int(np.sum(~bundle["event_returned"])),
        "required_events_per_mode": int(bundle["required_per_mode"]),
        "inputs": {
            "config": {"path": str(bundle["config_path"]),
                       "sha256": _sha256(bundle["config_path"])},
            "manifest": {"path": str(bundle["manifest_path"]),
                         "sha256": _sha256(bundle["manifest_path"])},
            "summary": {"path": str(bundle["summary_path"]),
                        "sha256": _sha256(bundle["summary_path"])},
            "patient_target": {"path": str(bundle["target_path"]),
                               "sha256": _sha256(bundle["target_path"])},
            "contact_contract": {"path": str(bundle["contract_path"]),
                                 "sha256": _sha256(bundle["contract_path"])},
            "workers": bundle["worker_inputs"],
        },
        "claim_boundary": (
            "development-only fresh-network validation; this figure does not "
            "establish full patient interictal reproduction, patient blind "
            "generalization, or a causal core mechanism"
        ),
    }


def _write_readme(output_dir):
    path = Path(output_dir) / "README.md"
    existing = path.read_text() if path.exists() else ""
    entries = """
### fig4a_continuous_field_v62_direct_readout

这张图按冻结 Fig.4 直接读出合同展示同一个 V6.2 连续场：三维 h landscape、神经元实际承受的 signed Delta Vtheta、唯一 returned formal mode A 与同网络最近 returned formal mode B 的逐触点传播，以及同一网络的连续 30-80 Hz model-current envelope。圆环只汇总 returned、双杆、patient-support 内的 formal clean events；星号表示当前代表事件的最早触点。

**关注点**：A 只有 1 个 joint+in-distribution event，图中出现一例 A 不等于 fresh-network 双模式 repertoire 已经通过；worker 未保存 raw band-passed current，因此这里是直接模型电流包络，不是临床 SEEG 波形。

### fig4b_continuous_field_v62_kmeans_consistency

这张图只使用与 Fig4A 同一批 worker artifact 中 returned、双杆、patient-support 内的 formal clean events，按冻结布局展示 KMeans heatmap、逐触点 rank 分布、cluster profile 与患者 prototype。KMeans 可自然切成两组，但 supervised patient mode A 只有 1 个 clean event，低于冻结的每模式 6 个评分预算，因此 model-vs-patient 矩阵显示 N/A，不能把两个 pooled clusters 改名为患者 A/B。

**关注点**：区分“聚类数值稳定”和“两个患者传播模式均有足够支持”；本轮只能验收前者，不能验收患者模式一致性。

"""
    marker = "### fig4a_continuous_field_v62_direct_readout"
    if marker not in existing:
        path.write_text(existing.rstrip() + "\n\n" + entries.lstrip())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--candidate-id", default=DEFAULT_CANDIDATE)
    args = parser.parse_args()
    bundle = _load_bundle(args.config, args.output_root, args.candidate_id)
    figure_dir = Path(args.output_root) / "figures"
    direct_stem, _ = _render_direct(bundle, figure_dir)
    kmeans_stem, _ = _render_kmeans(bundle, figure_dir)
    _write_readme(figure_dir)
    print(f"wrote {direct_stem}.png / .pdf / _metadata.json")
    print(f"wrote {kmeans_stem}.png / .pdf / _metadata.json")


if __name__ == "__main__":
    main()
