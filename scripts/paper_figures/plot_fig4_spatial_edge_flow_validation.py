"""Render Fig.4 direct-readout and KMeans validation for rev10-R2."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

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
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_r2_spatial_edge_flow.json"
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


def _is_spatial_ou(bundle):
    return bundle.get("candidate", {}).get(
        "spatial_ou", {},
    ).get("mode") == "local"


def normalize_event_ranks(ranks):
    ranks = np.asarray(ranks, float)
    output = np.full_like(ranks, np.nan)
    for index, row in enumerate(ranks):
        finite = np.isfinite(row)
        if not np.any(finite):
            continue
        values = row[finite]
        span = float(np.max(values) - np.min(values))
        output[index, finite] = (
            (values - np.min(values)) / span if span > 0.0 else 0.0
        )
    return output


def formal_clean_mask(onsets, labels, ood, groups, event_returned=None):
    onsets = np.asarray(onsets, float)
    icl = np.isfinite(onsets[:, np.asarray(groups["ICL"], int)]).any(axis=1)
    scl = np.isfinite(onsets[:, np.asarray(groups["SCL"], int)]).any(axis=1)
    returned = (
        np.ones(len(onsets), bool) if event_returned is None
        else np.asarray(event_returned, bool)
    )
    if returned.shape != (len(onsets),):
        raise ValueError("event_returned must align with onsets")
    return returned & icl & scl & ~np.asarray(ood, bool)


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
    output = np.full(values.shape[1], np.nan)
    for column in range(values.shape[1]):
        selected = values[:, column]
        selected = selected[np.isfinite(selected)]
        if len(selected):
            output[column] = np.quantile(selected, quantile)
    return output


def _classifier_from_manifest(manifest):
    classifier = dict(manifest["direction_classifier"])
    for key in (
        "coef", "class_centers", "class_precisions", "ood_distance_thresholds",
    ):
        classifier[key] = np.asarray(classifier[key], float)
    return classifier


def _returned_summary_filename(config):
    role = config.get("scientific_role", "")
    if role in {
        "development_only_dynamic_accessibility_canary",
        "development_only_inhibitory_resource_accessibility_canary",
        "development_only_dynamic_ee_std_accessibility_canary",
        "development_only_translation_invariant_spatial_ou_accessibility_canary",
        "development_only_translation_invariant_spatial_ou_low_amplitude_bracket",
        "development_only_translation_invariant_spatial_ou_kmeans_grid",
    }:
        return "canary_summary_returned_only.json"
    phase = config.get("search", {}).get("phase", "fit")
    return {
        "fit": "fit_screen_summary_returned_only.json",
        "selection": "selection_summary_returned_only.json",
        "confirmation": "confirmation_summary_returned_only.json",
    }[phase]


def _load_bundle(
        config_path, output_root, candidate_id=None,
        *, allow_exploratory_candidate=False):
    sys.path.insert(0, str(ROOT))
    from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
        load_scoring_contract,
    )
    from src.topic4_shaft_aware import contract_groups  # noqa: E402
    from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402

    config_path, output_root = Path(config_path).resolve(), Path(output_root).resolve()
    config = _json(config_path)
    manifest_path = output_root / "candidate_manifest.json"
    phase = config.get("search", {}).get("phase", "fit")
    summary_path = output_root / _returned_summary_filename(config)
    manifest, summary = _json(manifest_path), _json(summary_path)
    if manifest["config"]["sha256"] != _sha256(config_path):
        raise RuntimeError("manifest and rev10-R2 config do not match")
    frozen_id = (
        manifest["selection_freeze"]["selected_nonzero_candidate_id"]
        if phase == "confirmation"
        else summary["diagnostic_best_candidate_id"]
    )
    if (candidate_id is not None and candidate_id != frozen_id
            and not allow_exploratory_candidate):
        raise RuntimeError("figure candidate must equal the pre-network frozen candidate")
    candidate_id = frozen_id if candidate_id is None else candidate_id
    candidates = {
        row["candidate_id"]: row for row in manifest["candidate_set"]["candidates"]
    }
    candidate = candidates[candidate_id]

    contract_path = ROOT / config["inputs"]["contact_contract"]["path"]
    target_path = ROOT / config["inputs"]["shaft_aware_target_npz"]["path"]
    floor_path = ROOT / config["inputs"]["shaft_aware_floors"]["path"]
    contract = _json(contract_path)
    groups = contract_groups(contract)
    names, embedding, _, _ = load_scoring_contract(
        target_path, floor_path, "FULL_TIMING", fixed_events_per_mode=6,
    )
    classifier = _classifier_from_manifest(manifest)
    blocks, records, worker_inputs = [], [], []
    cursor, static = 0, None
    seed_key = {
        "fit": "fit_network_seeds",
        "selection": "selection_network_seeds",
        "confirmation": "confirmation_network_seeds",
    }[phase]
    for seed in config["search"][seed_key]:
        stem = output_root / "workers" / f"{candidate_id}_seed_{seed}"
        json_path, npz_path = stem.with_suffix(".json"), stem.with_suffix(".npz")
        payload = _json(json_path)
        if payload["arrays"]["sha256"] != _sha256(npz_path):
            raise RuntimeError(f"worker array hash changed: {npz_path.name}")
        with np.load(npz_path, allow_pickle=False) as loaded:
            worker_names = np.asarray(loaded["contact_names"]).astype(str)
            if not np.array_equal(worker_names, names.astype(str)):
                raise RuntimeError(f"contact order changed: {npz_path.name}")
            onsets, ranks = np.asarray(loaded["onsets"], float), np.asarray(loaded["ranks"], float)
            block = {
                "seed": int(seed), "onsets": onsets, "ranks": ranks,
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
        records.extend({
            "seed": int(seed), "local_index": local, "global_index": cursor + local,
        } for local in range(len(onsets)))
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
    labels, ood = np.asarray(assigned["labels"], int), np.asarray(assigned["ood"], bool)
    clean = formal_clean_mask(
        onsets, labels, ood, groups, event_returned=event_returned,
    )
    clean_counts = np.bincount(labels[clean], minlength=2)
    with np.load(target_path, allow_pickle=False) as target:
        patient = {key: np.asarray(target[key]) for key in (
            "patient_train_ranks", "patient_train_old_labels",
            "patient_train_block_ids",
        )}
    return {
        "config": config, "config_path": config_path, "output_root": output_root,
        "phase": phase, "network_seed_key": seed_key,
        "manifest": manifest, "manifest_path": manifest_path,
        "summary": summary, "summary_path": summary_path,
        "candidate_id": candidate_id, "candidate": candidate,
        "groups": groups, "blocks": blocks, "records": records,
        "static": static, "onsets": onsets, "ranks": ranks,
        "labels": labels, "ood": ood,
        "embedding": np.asarray(assigned["embedding"], float),
        "event_returned": event_returned,
        "clean": clean, "clean_counts": clean_counts,
        "required_per_mode": 6, "patient": patient,
        "worker_inputs": worker_inputs, "target_path": target_path,
        "contract_path": contract_path,
    }


def _same_network_pair(bundle):
    seed_key = bundle.get("network_seed_key", "fit_network_seeds")
    for seed in bundle["config"]["search"][seed_key]:
        by_mode = []
        for mode in (0, 1):
            by_mode.append([
                index for index in np.flatnonzero(
                    bundle["clean"] & (bundle["labels"] == mode)
                ) if bundle["records"][int(index)]["seed"] == seed
            ])
        if by_mode[0] and by_mode[1]:
            return int(by_mode[0][0]), int(by_mode[1][0]), next(
                row for row in bundle["blocks"] if row["seed"] == seed
            )
    return None


def _plot_contacts(ax, bundle):
    xy = bundle["static"]["contact_xy_mm"]
    for shaft in ("ICL", "SCL"):
        selected = bundle["static"]["shaft_ids"] == shaft
        ax.plot(xy[selected, 0], xy[selected, 1], color=SHAFT_COLORS[shaft], lw=1.1)
        ax.scatter(xy[selected, 0], xy[selected, 1], s=31,
                   color=SHAFT_COLORS[shaft], edgecolor="white", linewidth=0.6, zorder=6)


def _field_grid(positions, values, *, size=84, limit=20.0):
    axis = np.linspace(0.0, float(limit), int(size))
    xx, yy = np.meshgrid(axis, axis)
    zz = griddata(positions, values, (xx, yy), method="linear")
    if np.isnan(zz).any():
        nearest = griddata(positions, values, (xx, yy), method="nearest")
        zz = np.where(np.isfinite(zz), zz, nearest)
    return xx, yy, zz


def _plot_landscape(ax, bundle):
    static = bundle["static"]
    positions, h = static["positions_E"], static["h"]
    xx, yy, hh = _field_grid(positions, h)
    vmax = max(float(np.quantile(h, 0.995)), 1e-6)
    surface = ax.plot_surface(
        xx, yy, np.minimum(hh, vmax), cmap="plasma", vmin=0.0, vmax=vmax,
        linewidth=0, antialiased=True, shade=False, alpha=0.97,
        rasterized=True,
    )
    ax.contour(xx, yy, hh, zdir="z", offset=0.0, levels=7, cmap="plasma",
               linewidths=0.55, alpha=0.75)
    contact_h = griddata(
        positions, h, static["contact_xy_mm"], method="linear",
    )
    contact_h = np.nan_to_num(contact_h, nan=0.0) + 0.025 * vmax
    for shaft in ("ICL", "SCL"):
        selected = static["shaft_ids"] == shaft
        xy = static["contact_xy_mm"][selected]
        z = contact_h[selected]
        ax.plot(xy[:, 0], xy[:, 1], z, color=SHAFT_COLORS[shaft], lw=1.15)
        ax.scatter(xy[:, 0], xy[:, 1], z, s=27,
                   color=SHAFT_COLORS[shaft], edgecolor="white",
                   linewidth=0.55, depthshade=False)
    ax.set(xlim=(0, 20), ylim=(0, 20), zlim=(0, 1.12 * vmax),
           xlabel="sheet x (mm)", ylabel="sheet y (mm)", zlabel="h")
    ax.set_title("continuous field landscape", weight="bold", pad=7)
    ax.view_init(elev=31, azim=-58)
    ax.set_box_aspect((1.0, 1.0, 0.58))
    ax.tick_params(labelsize=7.2, pad=1)
    colorbar = plt.colorbar(surface, ax=ax, fraction=0.04, pad=0.01, shrink=0.72)
    colorbar.set_label("pathology field h", fontsize=8)


def _plot_flow(ax, bundle):
    sys.path.insert(0, str(ROOT))
    from src.topic4_spatial_edge_flow import spatial_vector_field  # noqa: E402

    axis = np.linspace(1.0, 19.0, 19)
    xx, yy = np.meshgrid(axis, axis)
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    vector = spatial_vector_field(xy, bundle["candidate"]["coefficients"], L=20.0)
    magnitude = np.linalg.norm(vector, axis=1).reshape(xx.shape)
    image = ax.contourf(xx, yy, magnitude, levels=12, cmap="cividis", alpha=0.88)
    ax.quiver(xx, yy, vector[:, 0].reshape(xx.shape), vector[:, 1].reshape(xx.shape),
              color="white", alpha=0.82, scale=None, width=0.0032)
    _plot_contacts(ax, bundle)
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)", ylabel="sheet y (mm)")
    ax.set_aspect("equal")
    ax.set_title("continuous E-to-E route field", weight="bold")
    plt.colorbar(image, ax=ax, fraction=0.047, pad=0.025, label="flow magnitude")


def _plot_delta(ax, bundle):
    static = bundle["static"]
    delta = static["delta_vtheta"]
    vmax = max(float(np.quantile(np.abs(delta), 0.995)), 1e-6)
    image = ax.scatter(static["positions_E"][:, 0], static["positions_E"][:, 1],
                       c=delta, s=2.6, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       lw=0, alpha=0.78, rasterized=True)
    _plot_contacts(ax, bundle)
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)")
    ax.set_aspect("equal")
    ax.set_title(r"frozen Node $\Delta V_\theta=-hd$", weight="bold")
    plt.colorbar(image, ax=ax, fraction=0.047, pad=0.025, label="mV")


def _plot_mode(ax, bundle, global_index, mode):
    xy = bundle["static"]["contact_xy_mm"]
    if _is_spatial_ou(bundle):
        xx, yy, hh = _field_grid(
            bundle["static"]["positions_E"], bundle["static"]["h"],
        )
        levels = np.unique(np.quantile(bundle["static"]["h"], (0.75, 0.90, 0.98)))
        ax.contour(xx, yy, hh, levels=levels, colors="#303030",
                   linestyles="--", linewidths=0.65, alpha=0.55)
    if global_index is None:
        _plot_contacts(ax, bundle)
        ax.text(0.5, 0.5, "same-network\nmode pair unavailable",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="#9B2F2A", weight="bold")
    else:
        onset = bundle["onsets"][global_index]
        finite = np.isfinite(onset)
        relative = onset[finite] - np.nanmin(onset)
        span = max(float(np.max(relative, initial=0.0)), 1.0)
        ax.scatter(xy[~finite, 0], xy[~finite, 1], s=34, facecolor="white",
                   edgecolor="#B9B9B9", linewidth=0.8)
        image = ax.scatter(xy[finite, 0], xy[finite, 1], c=relative / span, s=70,
                           cmap="viridis", vmin=0, vmax=1, edgecolor="white",
                           linewidth=0.75, zorder=7)
        earliest = finite & np.isclose(onset, np.nanmin(onset))
        ax.scatter(xy[earliest, 0], xy[earliest, 1], marker="*", s=120,
                   color="#111111", edgecolor="white", linewidth=0.5, zorder=8)
        selected = bundle["onsets"][
            bundle["clean"] & (bundle["labels"] == mode)
        ]
        density = np.zeros(len(xy), float)
        for event_onset in selected:
            event_finite = np.isfinite(event_onset)
            if np.any(event_finite):
                event_earliest = event_finite & np.isclose(
                    event_onset, np.nanmin(event_onset),
                )
                density[event_earliest] += 1.0 / int(np.sum(event_earliest))
        density /= max(1, len(selected))
        present = density > 0
        ax.scatter(xy[present, 0], xy[present, 1],
                   s=52 + 150 * density[present], facecolor="none",
                   edgecolor=MODE_COLORS[mode], linewidth=1.1,
                   alpha=0.78, zorder=9)
        plt.colorbar(image, ax=ax, fraction=0.047, pad=0.025,
                     ticks=(0, 1), label="early to late")
    _plot_contacts(ax, bundle)
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)")
    ax.set_aspect("equal")
    ax.set_title(f"model mode {'AB'[mode]}", weight="bold")
    ax.text(0.03, 0.96, f"formal clean n={int(bundle['clean_counts'][mode])}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
            color=MODE_COLORS[mode], weight="bold")


def _plot_readout(ax, bundle, pair):
    if pair is None:
        ax.text(0.5, 0.52, "direct A/B readout unavailable",
                transform=ax.transAxes, ha="center", va="center",
                color="#9B2F2A", weight="bold", fontsize=11)
        ax.text(0.5, 0.40, "no single fit network produced both formal modes",
                transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return {"same_network_pair": False}
    a_index, b_index, block = pair
    records = bundle["records"]
    centers = []
    for global_index in (a_index, b_index):
        local = records[global_index]["local_index"]
        centers.append(0.5 * (block["event_t_on_ms"][local] + block["event_t_off_ms"][local]))
    width = max(1000.0, abs(centers[1] - centers[0]) + 320.0)
    center = np.mean(centers)
    duration = block["contact_envelope"].shape[1] * block["contact_envelope_dt_ms"]
    start = max(0.0, min(duration - width, center - width / 2.0))
    stop, dt = min(duration, start + width), block["contact_envelope_dt_ms"]
    indices = np.arange(block["contact_envelope"].shape[1])
    sample = (indices * dt >= start) & (indices * dt <= stop)
    trace = block["contact_envelope"][:, sample]
    t = indices[sample] * dt - start
    union = np.isfinite(bundle["onsets"][a_index]) | np.isfinite(bundle["onsets"][b_index])
    contacts = np.flatnonzero(union)
    scale = max(float(np.quantile(trace[contacts], 0.99)), 1e-9)
    offsets = np.arange(len(contacts)) * 1.15
    for mode, global_index in ((0, a_index), (1, b_index)):
        local = records[global_index]["local_index"]
        ax.axvspan(block["event_t_on_ms"][local] - start,
                   block["event_t_off_ms"][local] - start,
                   color=MODE_COLORS[mode], alpha=0.14, lw=0)
    for row, contact in enumerate(contacts):
        shaft = bundle["static"]["shaft_ids"][contact]
        ax.plot(t, trace[contact] * 0.68 / scale + offsets[row],
                color=SHAFT_COLORS[shaft], lw=0.82)
    bar_x = 0.045 * max(stop - start, 1.0)
    bar_y = -0.40
    ax.plot([bar_x, bar_x], [bar_y, bar_y + 0.68], color="#222222", lw=1.5)
    ax.text(bar_x + 0.015 * max(stop - start, 1.0), bar_y + 0.34,
            f"{scale:.2g} a.u.", ha="left", va="center", fontsize=7.2)
    ax.set_yticks(offsets, bundle["static"]["contact_names"][contacts], fontsize=7.5)
    ax.set_xlim(0, stop - start)
    ax.set_ylim(-0.6, offsets[-1] + 1.0)
    ax.invert_yaxis()
    ax.set_xlabel("simulation time (ms)")
    ax.set_ylabel("30-80 Hz model-current envelope")
    ax.set_title("direct electrode readout", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=MODE_COLORS[0], alpha=0.2, label="model mode A"),
        Patch(facecolor=MODE_COLORS[1], alpha=0.2, label="model mode B"),
    ], frameon=False, fontsize=7.5, ncol=2, loc="lower right")
    return {"same_network_pair": True, "seed": block["seed"],
            "mode_A_global_index": a_index, "mode_B_global_index": b_index}


def _save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white",
                bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return {suffix: {"path": str(stem.with_suffix(f".{suffix}")),
                     "sha256": _sha256(stem.with_suffix(f".{suffix}"))}
            for suffix in ("png", "pdf")}


def _metadata(bundle, figure):
    return {
        "figure": figure, "plotting_only": True,
        "candidate_id": bundle["candidate_id"],
        "candidate_role": (
            "pre-network frozen nonzero confirmation candidate"
            if bundle["phase"] == "confirmation"
            else f"frozen diagnostic best from equal-network {bundle['phase']} screen"
        ),
        "phase_diagnostic_best_candidate_id": bundle["summary"][
            "diagnostic_best_candidate_id"
        ],
        "candidate_is_phase_diagnostic_best": bool(
            bundle["candidate_id"]
            == bundle["summary"]["diagnostic_best_candidate_id"]
        ),
        "source_status": bundle["summary"]["status"],
        "formal_clean_mode_counts": {
            "A": int(bundle["clean_counts"][0]),
            "B": int(bundle["clean_counts"][1]),
        },
        "detected_event_count": int(len(bundle["event_returned"])),
        "returned_event_count": int(np.sum(bundle["event_returned"])),
        "nonreturned_event_count_excluded": int(np.sum(~bundle["event_returned"])),
        "required_events_per_mode": bundle["required_per_mode"],
        "inputs": {
            "config": {"path": str(bundle["config_path"]),
                       "sha256": _sha256(bundle["config_path"])},
            "manifest": {"path": str(bundle["manifest_path"]),
                         "sha256": _sha256(bundle["manifest_path"])},
            "summary": {"path": str(bundle["summary_path"]),
                        "sha256": _sha256(bundle["summary_path"])},
            "workers": bundle["worker_inputs"],
        },
        "claim_boundary": (
            f"returned detector events only; development-only {bundle['phase']}-network "
            "visualization; no patient-blind "
            "generalization, causal core, or ictal lifecycle claim"
        ),
    }


def _render_direct(bundle, output_dir):
    pair = _same_network_pair(bundle)
    fig = plt.figure(figsize=(22.0, 4.9), facecolor="white")
    grid = fig.add_gridspec(
        1, 5, width_ratios=(1.1, 1.0, 1.0, 1.0, 2.35),
        left=0.036, right=0.992, bottom=0.17, top=0.86, wspace=0.31,
    )
    axes = [
        fig.add_subplot(grid[0, 0], projection="3d")
        if _is_spatial_ou(bundle) else fig.add_subplot(grid[0, 0]),
        *[fig.add_subplot(grid[0, index]) for index in range(1, 5)],
    ]
    if _is_spatial_ou(bundle):
        _plot_landscape(axes[0], bundle)
    else:
        _plot_flow(axes[0], bundle)
    _plot_delta(axes[1], bundle)
    _plot_mode(axes[2], bundle, None if pair is None else pair[0], 0)
    _plot_mode(axes[3], bundle, None if pair is None else pair[1], 1)
    readout = _plot_readout(axes[4], bundle, pair)
    qualifier = "same-network A/B available" if pair else "same-network A/B unavailable"
    mechanism = (
        "Continuous field with spatial OU accessibility"
        if _is_spatial_ou(bundle) else "Continuous spatial edge field"
    )
    fig.suptitle(
        f"{mechanism}: direct model readout  |  {qualifier}",
        fontsize=13, weight="bold", y=0.985,
    )
    stem = Path(output_dir) / (
        "fig4a_spatial_ou_direct_readout" if _is_spatial_ou(bundle)
        else "fig4a_spatial_edge_flow_direct_readout"
    )
    metadata = _metadata(bundle, (
        "Fig4A spatial-OU direct readout" if _is_spatial_ou(bundle)
        else "Fig4A spatial edge-flow direct readout"
    ))
    metadata.update({"files": _save(fig, stem), "direct_readout": readout})
    Path(str(stem) + "_metadata.json").write_text(json.dumps(metadata, indent=2))
    return stem


def _canonicalize_kmeans(labels, z):
    means = [float(np.mean(z[labels == cluster, 0])) for cluster in (0, 1)]
    mapping = {int(old): int(new) for new, old in enumerate(np.argsort(means))}
    return np.asarray([mapping[int(value)] for value in labels], int)


def _kmeans(clean_z, direction):
    if len(clean_z) < 4 or np.unique(clean_z, axis=0).shape[0] < 2:
        return None, {"status": "INSUFFICIENT_EVENTS", "n_events": int(len(clean_z))}
    with threadpool_limits(limits=1):
        label_sets = []
        for seed in range(12):
            raw = KMeans(n_clusters=2, n_init=100 if seed == 0 else 20,
                         random_state=seed).fit_predict(clean_z)
            label_sets.append(_canonicalize_kmeans(raw, clean_z))
    pairwise = [
        adjusted_mutual_info_score(label_sets[left], label_sets[right])
        for left in range(len(label_sets)) for right in range(left + 1, len(label_sets))
    ]
    labels = label_sets[0]
    return labels, {
        "status": "OK", "n_events": int(len(labels)),
        "cluster_counts": np.bincount(labels, minlength=2).tolist(),
        "pairwise_seed_ami_median": float(np.median(pairwise)),
        "pairwise_seed_ami_min": float(np.min(pairwise)),
        "ami_with_supervised_direction": float(
            adjusted_mutual_info_score(labels, direction)
        ),
    }


def _direction_purity(labels, direction):
    labels = np.asarray(labels, int)
    direction = np.asarray(direction, int)
    contingency = np.zeros((2, 2), int)
    for cluster, mode in zip(labels, direction):
        contingency[int(cluster), int(mode)] += 1
    identity = int(contingency[0, 0] + contingency[1, 1])
    swapped = int(contingency[0, 1] + contingency[1, 0])
    return float(max(identity, swapped) / max(1, contingency.sum())), contingency


def _canonical_rank_kmeans(bundle, min_shared_contacts=3):
    """Apply the canonical Fig.4C masked-rank KMeans contract."""
    sys.path.insert(0, str(ROOT))
    from src.interictal_propagation import (  # noqa: E402
        compute_adaptive_cluster_stereotypy,
    )

    clean_index = np.flatnonzero(bundle["clean"])
    rank_matrix = np.asarray(bundle["ranks"][clean_index], float).T
    participation = np.isfinite(rank_matrix)
    valid = participation.sum(axis=0) >= int(min_shared_contacts)
    result = compute_adaptive_cluster_stereotypy(
        rank_matrix, participation, bundle["static"]["contact_names"].tolist(),
        k_range=(2, 2), use_masked_features=True,
        min_shared_channels=int(min_shared_contacts),
        min_participation=int(min_shared_contacts), n_sample=100,
        n_tau_seeds=5,
    )
    labels = np.asarray(result.get("labels", []), int)
    if labels.shape != (int(np.sum(valid)),):
        raise RuntimeError("canonical rank KMeans event subset changed")
    selected_index = clean_index[valid]
    direction = np.asarray(bundle["labels"][selected_index], int)
    purity, contingency = _direction_purity(labels, direction)
    scan = result.get("scan", [{}])[0]
    return {
        "clean_global_index": selected_index,
        "labels": labels,
        "direction": direction,
        "direction_purity": purity,
        "direction_contingency": contingency,
        "cluster_counts": np.bincount(labels, minlength=2),
        "within_cluster_tau_mean": result.get("within_cluster_tau_mean"),
        "inter_cluster_corr_matrix": result.get("inter_cluster_corr_matrix"),
        "candidate_forward_reverse_pairs": result.get(
            "candidate_forward_reverse_pairs", []
        ),
        "stability_ami_median": scan.get("median_ami"),
        "silhouette_median": scan.get("median_silhouette"),
        "result": result,
    }


def _patient_profiles(bundle):
    ranks = normalize_event_ranks(bundle["patient"]["patient_train_ranks"])
    labels = bundle["patient"]["patient_train_old_labels"].astype(int)
    blocks = bundle["patient"]["patient_train_block_ids"]
    profiles, lows, highs = [], [], []
    for mode in (0, 1):
        profiles.append(_column_stats(ranks[labels == mode])[0])
        block_profiles = np.asarray([
            _column_stats(ranks[(labels == mode) & (blocks == block)])[0]
            for block in np.unique(blocks[labels == mode])
        ])
        lows.append(_column_quantile(block_profiles, 0.05))
        highs.append(_column_quantile(block_profiles, 0.95))
    return np.asarray(profiles), np.asarray(lows), np.asarray(highs)


def _similarity(model, patient):
    matrix = np.full((2, 2), np.nan)
    for row in range(2):
        for column in range(2):
            finite = np.isfinite(model[row]) & np.isfinite(patient[column])
            if np.sum(finite) >= 3:
                matrix[row, column] = spearmanr(
                    model[row, finite], patient[column, finite]
                ).statistic
    return matrix


def _render_kmeans(bundle, output_dir):
    canonical = _canonical_rank_kmeans(bundle)
    clean_index = canonical["clean_global_index"]
    ranks = normalize_event_ranks(bundle["ranks"][clean_index])
    direction = canonical["direction"]
    labels = canonical["labels"]
    audit = {
        "status": "OK", "feature_contract": "masked normalized event ranks",
        "min_shared_contacts": 3,
        "n_events": int(len(labels)),
        "cluster_counts": canonical["cluster_counts"].tolist(),
        "direction_contingency": canonical[
            "direction_contingency"
        ].tolist(),
        "direction_purity": canonical["direction_purity"],
        "within_cluster_tau_mean": canonical[
            "within_cluster_tau_mean"
        ],
        "inter_cluster_corr_matrix": canonical[
            "inter_cluster_corr_matrix"
        ],
        "candidate_forward_reverse_pairs": canonical[
            "candidate_forward_reverse_pairs"
        ],
        "kmeans_stability_ami_median": canonical[
            "stability_ami_median"
        ],
        "silhouette_median": canonical["silhouette_median"],
    }
    names = bundle["static"]["contact_names"]
    fig = plt.figure(figsize=(18.8, 5.2), facecolor="white")
    grid = fig.add_gridspec(1, 4, width_ratios=(3.0, 1.0, 1.55, 1.25),
                            left=0.052, right=0.985, bottom=0.18, top=0.84, wspace=0.38)
    axes = [fig.add_subplot(grid[0, index]) for index in range(4)]
    if labels is None:
        for ax in axes:
            ax.text(0.5, 0.5, "insufficient formal clean events",
                    transform=ax.transAxes, ha="center", va="center",
                    color="#9B2F2A", weight="bold")
            ax.set_axis_off()
        matrix_valid = False
        matrix = np.full((2, 2), np.nan)
    else:
        slopes = np.nanmean(ranks, axis=1)
        order = np.lexsort((slopes, labels))
        cmap = plt.cm.viridis.copy()
        cmap.set_bad("#D7D7D7")
        image = axes[0].imshow(np.ma.masked_invalid(ranks[order].T), aspect="auto",
                               origin="upper", interpolation="nearest", cmap=cmap,
                               vmin=0, vmax=1)
        split = int(np.sum(labels[order] == 0))
        if 0 < split < len(order):
            axes[0].axvline(split - 0.5, color="#B22222", lw=1.3)
        axes[0].set_yticks(np.arange(len(names)), names, fontsize=8)
        axes[0].set(xlabel="formal clean events", ylabel="fixed contact identity")
        axes[0].set_title("KMeans event heatmap", weight="bold")
        plt.colorbar(image, ax=axes[0], fraction=0.025, pad=0.02,
                     ticks=(0, 1), label="first to last")

        values, positions = [], []
        for contact in range(len(names)):
            finite = ranks[:, contact]
            finite = finite[np.isfinite(finite)]
            if len(finite):
                values.append(finite)
                positions.append(contact)
        violin = axes[1].violinplot(values, positions=positions, vert=False,
                                    widths=0.78, showmedians=True, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor("#727272")
            body.set_alpha(0.45)
        axes[1].set(xlim=(-0.04, 1.04), ylim=axes[0].get_ylim(), xlabel="rank")
        axes[1].set_yticks(axes[0].get_yticks(), [])
        axes[1].set_title("rank distribution", weight="bold")

        patient, patient_low, patient_high = _patient_profiles(bundle)
        cluster_profiles = np.asarray([
            _column_stats(ranks[labels == group])[0] for group in (0, 1)
        ])
        majority_mode = np.argmax(
            canonical["direction_contingency"], axis=1,
        )
        y = np.arange(len(names))
        for group in (0, 1):
            finite = np.isfinite(cluster_profiles[group])
            axes[2].plot(cluster_profiles[group, finite], y[finite], "-o",
                         color=GROUP_COLORS[group], lw=1.8, ms=3.5,
                         label=f"group {group + 1} ({'AB'[majority_mode[group]]}-majority)")
        for mode in (0, 1):
            finite = np.isfinite(patient[mode])
            axes[2].fill_betweenx(y[finite], patient_low[mode, finite],
                                  patient_high[mode, finite],
                                  color=MODE_COLORS[mode], alpha=0.06)
            axes[2].plot(patient[mode, finite], y[finite], "--",
                         color=MODE_COLORS[mode], lw=1.3,
                         label=f"patient {'AB'[mode]}")
        axes[2].set(xlim=(-0.08, 1.08), ylim=axes[0].get_ylim(),
                    xlabel="mean normalized rank")
        axes[2].set_yticks(axes[0].get_yticks(), [])
        axes[2].set_title("cluster rank profile", weight="bold")
        axes[2].legend(frameon=False, fontsize=7, ncol=2)
        supervised_model = np.asarray([
            _column_stats(ranks[direction == mode])[0] for mode in (0, 1)
        ])
        matrix = _similarity(supervised_model, patient)
        matrix_valid = bool(np.all(bundle["clean_counts"] >= bundle["required_per_mode"]))
        axes[3].set_xticks((0, 1), ("patient A", "patient B"), fontsize=8)
        axes[3].set_yticks((0, 1), ("model A", "model B"), fontsize=8)
        axes[3].set_aspect("equal")
        axes[3].set_title("model vs patient", weight="bold")
        if matrix_valid:
            im = axes[3].imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
            for row in range(2):
                for column in range(2):
                    axes[3].text(column, row, f"{matrix[row, column]:+.2f}",
                                 ha="center", va="center", fontsize=10, weight="bold")
            plt.colorbar(im, ax=axes[3], fraction=0.047, pad=0.04, label="Spearman rho")
        else:
            axes[3].set_facecolor("#EFEFEF")
            axes[3].text(0.5, 0.55, "N/A", transform=axes[3].transAxes,
                         ha="center", va="center", fontsize=21,
                         color="#9B2F2A", weight="bold")
            axes[3].text(0.5, 0.32,
                         f"formal A/B={bundle['clean_counts'][0]}/{bundle['clean_counts'][1]} < 6",
                         transform=axes[3].transAxes, ha="center", va="center",
                         fontsize=8, color="#9B2F2A")
    corrected_path = bundle["output_root"] / "confirmation_verdict.json"
    corrected = _json(corrected_path) if corrected_path.exists() else {}
    benchmark = corrected.get("patient_matched_kmeans_direction_purity", {})
    q05 = benchmark.get("q05")
    if q05 is not None:
        qualifier = (
            f"direction purity={audit['direction_purity']:.2f} < "
            f"patient q05={q05:.2f}"
        )
    else:
        qualifier = (
            f"direction purity={audit['direction_purity']:.2f}; "
            + ("patient matrix evaluable" if matrix_valid else "patient matrix N/A")
        )
    fig.suptitle(f"KMeans modes against patient data  |  {qualifier}",
                 fontsize=13, weight="bold", y=0.985)
    fig.text(
        0.985, 0.045,
        f"clusters={audit['cluster_counts'][0]}/{audit['cluster_counts'][1]}"
        f"   stable AMI={audit['kmeans_stability_ami_median']:.2f}"
        f"   direction purity={audit['direction_purity']:.2f}",
        ha="right", va="bottom", fontsize=8.5, color="0.32",
    )
    stem = Path(output_dir) / (
        "fig4b_spatial_ou_kmeans_consistency" if _is_spatial_ou(bundle)
        else "fig4b_spatial_edge_flow_kmeans_consistency"
    )
    metadata = _metadata(bundle, (
        "Fig4B spatial-OU KMeans consistency" if _is_spatial_ou(bundle)
        else "Fig4B spatial edge-flow KMeans consistency"
    ))
    metadata.update({
        "files": _save(fig, stem), "kmeans": audit,
        "matrix_valid": matrix_valid,
        "matrix_status": "EVALUABLE" if matrix_valid else "NOT_EVALUABLE_SUPPORT",
        "supervised_direction_vs_patient_spearman": matrix.tolist(),
        "matrix_rows": "frozen supervised model A/B, not KMeans cluster labels",
        "corrected_confirmation_verdict": (
            {"path": str(corrected_path), "sha256": _sha256(corrected_path)}
            if corrected_path.exists() else None
        ),
    })
    Path(str(stem) + "_metadata.json").write_text(json.dumps(metadata, indent=2))
    return stem


def _write_readme(output_dir, bundle):
    path = Path(output_dir) / "README.md"
    candidate_context = (
        "selection 阶段在读取确认网络前冻结的非零候选"
        if bundle["phase"] == "confirmation"
        else "等网络 fit screen 冻结的 diagnostic best"
    )
    if _is_spatial_ou(bundle):
        path.write_text(f"""### fig4a_spatial_ou_direct_readout

这张图展示 fresh-network confirmation 中预先冻结的连续场候选 `{bundle['candidate_id']}`：三维 `h` landscape、神经元实际承受的 signed `Delta Vtheta`、同一网络内自发 A/B 的逐触点传播，以及连续 30-80 Hz model-current envelope。局部 OU 只提供全片、零均值、平移不变的连续随机可达性，不使用 contact、shaft、患者事件或 D4 source 坐标；圆环汇总所有 formal clean events 的 earliest-contact density。

**关注点**：同一冻结连续场能否在未见网络中自发访问 A/B 两条传播路径；直接模型电流包络不是临床 SEEG 电压。

### fig4b_spatial_ou_kmeans_consistency

这张图只使用 returned、双杆、patient-support 内的 formal clean events。KMeans 严格复用 Fig.4C 的 masked normalized rank 特征；方向 purity 判断两个自然簇与 frozen A/B 的关联强度。最右矩阵直接由 frozen model A/B 事件构建，而不是把 KMeans 簇强行改名为 A/B。

**关注点**：患者 rank 几何可恢复并不等于 KMeans 离散性达到患者水平；当前应同时看正对角/负交叉矩阵、direction purity 及 patient-matched purity 区间。这是 development confirmation，不是 patient blind generalization。
""")
        return
    path.write_text(f"""### fig4a_spatial_edge_flow_direct_readout

这张图展示 returned-only {candidate_context} `{bundle['candidate_id']}`：均匀连续 E-to-E vector field、冻结 Node 的 signed Delta Vtheta、同一网络 A/B 逐触点传播和连续 30-80 Hz model-current envelope。若没有单张网络同时产生两种 formal clean 模式，模式图与波形区会明确显示 unavailable，不跨网络拼接代表事件。

**关注点**：连续连接场是否在不增加 core、contact-conditioned 参数或 topology 的前提下补回同网络 mode A，同时保留 mode B。

### fig4b_spatial_edge_flow_kmeans_consistency

这张图只使用同一候选中 returned、双杆、patient-support 内的 formal clean events，展示固定 contact heatmap、KMeans、rank distribution、患者 prototype 和 model-patient 相关矩阵。KMeans 数值稳定与患者 A/B 一致性分开判定；任一 supervised mode 少于冻结的 6 个事件时，矩阵显示 N/A。

**关注点**：先看 formal A/B 支持和同网络共存，再看 KMeans AMI 与患者 prototype；不能用 pooled 两簇代替缺失的患者模式。
""")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root")
    parser.add_argument("--candidate-id")
    args = parser.parse_args()
    config = _json(args.config)
    output_root = Path(args.output_root or ROOT / config["output_root"])
    bundle = _load_bundle(args.config, output_root, args.candidate_id)
    figure_dir = output_root / "figures"
    direct = _render_direct(bundle, figure_dir)
    kmeans = _render_kmeans(bundle, figure_dir)
    _write_readme(figure_dir, bundle)
    print(json.dumps({
        "status": (
            "REV10D5_2_FIG4_VALIDATION_COMPLETE" if _is_spatial_ou(bundle)
            else "REV10R2_FIG4_VALIDATION_COMPLETE"
        ),
        "candidate_id": bundle["candidate_id"],
        "figures": [str(direct), str(kmeans)],
    }, indent=2))


if __name__ == "__main__":
    main()
