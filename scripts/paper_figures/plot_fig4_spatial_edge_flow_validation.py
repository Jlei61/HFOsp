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


def formal_clean_mask(onsets, labels, ood, groups):
    onsets = np.asarray(onsets, float)
    icl = np.isfinite(onsets[:, np.asarray(groups["ICL"], int)]).any(axis=1)
    scl = np.isfinite(onsets[:, np.asarray(groups["SCL"], int)]).any(axis=1)
    return icl & scl & ~np.asarray(ood, bool)


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


def _load_bundle(config_path, output_root, candidate_id=None):
    sys.path.insert(0, str(ROOT))
    from scripts.rescore_topic4_rev10_sa_historical_artifacts import (  # noqa: E402
        load_scoring_contract,
    )
    from src.topic4_shaft_aware import contract_groups  # noqa: E402
    from src.topic4_shaft_aware_direction import assign_direction_modes  # noqa: E402

    config_path, output_root = Path(config_path).resolve(), Path(output_root).resolve()
    config = _json(config_path)
    manifest_path = output_root / "candidate_manifest.json"
    summary_path = output_root / "fit_screen_summary.json"
    manifest, summary = _json(manifest_path), _json(summary_path)
    if manifest["config"]["sha256"] != _sha256(config_path):
        raise RuntimeError("manifest and rev10-R2 config do not match")
    frozen_id = summary["diagnostic_best_candidate_id"]
    if candidate_id is not None and candidate_id != frozen_id:
        raise RuntimeError("figure candidate must equal frozen diagnostic best")
    candidate_id = frozen_id
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
    for seed in config["search"]["fit_network_seeds"]:
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
    assigned = assign_direction_modes(
        onsets, groups=groups, embedding=embedding, classifier=classifier,
    )
    labels, ood = np.asarray(assigned["labels"], int), np.asarray(assigned["ood"], bool)
    clean = formal_clean_mask(onsets, labels, ood, groups)
    clean_counts = np.bincount(labels[clean], minlength=2)
    with np.load(target_path, allow_pickle=False) as target:
        patient = {key: np.asarray(target[key]) for key in (
            "patient_train_ranks", "patient_train_old_labels",
            "patient_train_block_ids",
        )}
    return {
        "config": config, "config_path": config_path, "output_root": output_root,
        "manifest": manifest, "manifest_path": manifest_path,
        "summary": summary, "summary_path": summary_path,
        "candidate_id": candidate_id, "candidate": candidate,
        "groups": groups, "blocks": blocks, "records": records,
        "static": static, "onsets": onsets, "ranks": ranks,
        "labels": labels, "ood": ood,
        "embedding": np.asarray(assigned["embedding"], float),
        "clean": clean, "clean_counts": clean_counts,
        "required_per_mode": 6, "patient": patient,
        "worker_inputs": worker_inputs, "target_path": target_path,
        "contract_path": contract_path,
    }


def _same_network_pair(bundle):
    for seed in bundle["config"]["search"]["fit_network_seeds"]:
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
    ], frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    return {"same_network_pair": True, "seed": block["seed"],
            "mode_A_global_index": a_index, "mode_B_global_index": b_index}


def _save(fig, stem):
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white",
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
        "candidate_role": "frozen diagnostic best from equal-network fit screen",
        "source_status": bundle["summary"]["status"],
        "formal_clean_mode_counts": {
            "A": int(bundle["clean_counts"][0]),
            "B": int(bundle["clean_counts"][1]),
        },
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
            "development-only fit-network visualization; no patient-blind "
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
    axes = [fig.add_subplot(grid[0, index]) for index in range(5)]
    _plot_flow(axes[0], bundle)
    _plot_delta(axes[1], bundle)
    _plot_mode(axes[2], bundle, None if pair is None else pair[0], 0)
    _plot_mode(axes[3], bundle, None if pair is None else pair[1], 1)
    readout = _plot_readout(axes[4], bundle, pair)
    qualifier = "same-network A/B available" if pair else "same-network A/B unavailable"
    fig.suptitle(
        f"Continuous spatial edge field: direct model readout  |  {qualifier}",
        fontsize=13, weight="bold", y=0.985,
    )
    stem = Path(output_dir) / "fig4a_spatial_edge_flow_direct_readout"
    metadata = _metadata(bundle, "Fig4A spatial edge-flow direct readout")
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
    clean_index = np.flatnonzero(bundle["clean"])
    ranks = normalize_event_ranks(bundle["ranks"][clean_index])
    direction = bundle["labels"][clean_index]
    labels, audit = _kmeans(bundle["embedding"][clean_index], direction)
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
        model = np.asarray([_column_stats(ranks[labels == group])[0] for group in (0, 1)])
        y = np.arange(len(names))
        for group in (0, 1):
            finite = np.isfinite(model[group])
            axes[2].plot(model[group, finite], y[finite], "-o",
                         color=GROUP_COLORS[group], lw=1.8, ms=3.5,
                         label=f"group {group + 1}")
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
        matrix = _similarity(model, patient)
        matrix_valid = bool(np.all(bundle["clean_counts"] >= bundle["required_per_mode"]))
        axes[3].set_xticks((0, 1), ("patient A", "patient B"), fontsize=8)
        axes[3].set_yticks((0, 1), ("group 1", "group 2"), fontsize=8)
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
    qualifier = "patient matrix evaluable" if matrix_valid else "patient matrix not evaluable"
    fig.suptitle(f"KMeans modes against patient data  |  {qualifier}",
                 fontsize=13, weight="bold", y=0.985)
    stem = Path(output_dir) / "fig4b_spatial_edge_flow_kmeans_consistency"
    metadata = _metadata(bundle, "Fig4B spatial edge-flow KMeans consistency")
    metadata.update({
        "files": _save(fig, stem), "kmeans": audit,
        "matrix_valid": matrix_valid,
        "matrix_status": "EVALUABLE" if matrix_valid else "NOT_EVALUABLE_SUPPORT",
        "descriptive_cluster_vs_patient_spearman": matrix.tolist(),
    })
    Path(str(stem) + "_metadata.json").write_text(json.dumps(metadata, indent=2))
    return stem


def _write_readme(output_dir, bundle):
    path = Path(output_dir) / "README.md"
    path.write_text(f"""### fig4a_spatial_edge_flow_direct_readout

这张图展示等网络 fit screen 冻结的 diagnostic best `{bundle['candidate_id']}`：均匀连续 E-to-E vector field、冻结 Node 的 signed Delta Vtheta、同一网络 A/B 逐触点传播和连续 30-80 Hz model-current envelope。若没有单张网络同时产生两种 formal clean 模式，模式图与波形区会明确显示 unavailable，不跨网络拼接代表事件。

**关注点**：连续连接场是否在不增加 core、contact-conditioned 参数或 topology 的前提下补回同网络 mode A，同时保留 mode B。

### fig4b_spatial_edge_flow_kmeans_consistency

这张图使用同一候选的 formal clean events 展示固定 contact heatmap、KMeans、rank distribution、患者 prototype 和 model-patient 相关矩阵。KMeans 数值稳定与患者 A/B 一致性分开判定；任一 supervised mode 少于冻结的 6 个事件时，矩阵显示 N/A。

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
        "status": "REV10R2_FIG4_VALIDATION_COMPLETE",
        "candidate_id": bundle["candidate_id"],
        "figures": [str(direct), str(kmeans)],
    }, indent=2))


if __name__ == "__main__":
    main()
