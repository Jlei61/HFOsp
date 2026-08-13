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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
JOINT_CONTINUOUS_SURFACE_ROLE = (
    "development_only_continuous_field_joint_direction_surface"
)
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_r2_spatial_edge_flow.json"
TA_MODE = 1
TB_MODE = 0
SEMANTIC_MODE_ORDER = (TA_MODE, TB_MODE)
TA_COLOR = "#C43C39"
TB_COLOR = "#277DA1"
MODE_COLORS = {TA_MODE: TA_COLOR, TB_MODE: TB_COLOR}
DISPLAY_MODE_COLORS = (TA_COLOR, TB_COLOR)
SHAFT_COLORS = {"ICL": "#E67E22", "SCL": "#159EAE"}
TRACE_BAND_HZ = (30.0, 80.0)


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
        "development_only_observation_invariant_continuous_field_kmeans_screen",
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
    if (config.get("scientific_role") == JOINT_CONTINUOUS_SURFACE_ROLE
            and (candidate_id is None or not allow_exploratory_candidate)):
        verdict = _json(output_root / "confirmation_verdict.json")
        frozen_id = verdict["diagnostic_display_candidate_id"]
        figure_candidate_selection = "post-run preregistered diagnostic display rule"
    elif config.get("scientific_role") == JOINT_CONTINUOUS_SURFACE_ROLE:
        # The auditor explicitly traverses every frozen candidate before the
        # verdict exists. This path never selects the final display candidate.
        frozen_id = candidate_id
        figure_candidate_selection = "explicit frozen-library audit candidate"
    else:
        frozen_id = (
            manifest["selection_freeze"]["selected_nonzero_candidate_id"]
            if phase == "confirmation"
            else summary["diagnostic_best_candidate_id"]
        )
        figure_candidate_selection = (
            "pre-network frozen candidate" if phase == "confirmation"
            else "phase diagnostic best"
        )
    if (candidate_id is not None and candidate_id != frozen_id
            and not allow_exploratory_candidate):
        raise RuntimeError("figure candidate violates the frozen display contract")
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
        "figure_candidate_selection": figure_candidate_selection,
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
    candidates = []
    for seed in bundle["config"]["search"][seed_key]:
        by_mode = {}
        for mode in SEMANTIC_MODE_ORDER:
            by_mode[mode] = [
                index for index in np.flatnonzero(
                    bundle["clean"] & (bundle["labels"] == mode)
                ) if bundle["records"][int(index)]["seed"] == seed
            ]
        if not by_mode[TA_MODE] or not by_mode[TB_MODE]:
            continue
        block = next(row for row in bundle["blocks"] if row["seed"] == seed)
        for ta_index in by_mode[TA_MODE]:
            ta_local = bundle["records"][ta_index]["local_index"]
            ta_center = 0.5 * (
                block["event_t_on_ms"][ta_local] + block["event_t_off_ms"][ta_local]
            )
            for tb_index in by_mode[TB_MODE]:
                tb_local = bundle["records"][tb_index]["local_index"]
                tb_center = 0.5 * (
                    block["event_t_on_ms"][tb_local] + block["event_t_off_ms"][tb_local]
                )
                gap = abs(ta_center - tb_center)
                onsets = bundle.get("onsets")
                if onsets is None:
                    ta_support = tb_support = 0
                else:
                    ta_support = int(np.sum(np.isfinite(onsets[ta_index])))
                    tb_support = int(np.sum(np.isfinite(onsets[tb_index])))
                candidates.append({
                    "gap_ms": float(gap), "seed": int(seed),
                    "ta_index": int(ta_index), "tb_index": int(tb_index),
                    "ta_support": ta_support, "tb_support": tb_support,
                    "block": block,
                })
    if not candidates:
        return None
    separated = [row for row in candidates if 250.0 <= row["gap_ms"] <= 1200.0]
    pool = separated or candidates
    selected = min(pool, key=lambda row: (
        -min(row["ta_support"], row["tb_support"]),
        -(row["ta_support"] + row["tb_support"]),
        abs(row["gap_ms"] - 550.0), row["seed"],
        row["ta_index"], row["tb_index"],
    ))
    return selected["ta_index"], selected["tb_index"], selected["block"]


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


def _figure2a_context_geometry(bundle):
    """Express the exact Figure 2A ICL/SCL implantation in the SNN sheet."""
    from scripts.paper_figures import (  # noqa: E402
        plot_fig2_e1146_template_projection_composite as figure2a,
    )

    record, dat_a, _, names, coords, coord_space = figure2a._load_case()
    basis = figure2a._basis(record, int(dat_a["transverse_sign"]))
    basis_coords = figure2a._to_basis(coords, basis)
    names = np.asarray(names, dtype=str)
    shafts = np.asarray([
        "".join(character for character in name if not character.isdigit())
        for name in names
    ], dtype=str)
    local = np.isin(shafts, ("ICL", "SCL"))
    names = names[local]
    shafts = shafts[local]
    measured = basis_coords[local]
    fitted, fit_audit = figure2a._fit_straight_shaft_points(
        measured, names.tolist(), shafts.tolist(),
    )

    static_names = np.asarray(bundle["static"]["contact_names"], dtype=str)
    measured_lookup = {name: index for index, name in enumerate(names)}
    selected_plane_xy = np.asarray([
        measured[measured_lookup[name], :2] for name in static_names
    ])
    design = np.column_stack((selected_plane_xy, np.ones(len(selected_plane_xy))))
    affine = np.linalg.lstsq(
        design, bundle["static"]["contact_xy_mm"], rcond=None,
    )[0]
    registration_error = float(np.max(np.abs(
        design @ affine - bundle["static"]["contact_xy_mm"]
    )))
    if registration_error > 1e-6:
        raise RuntimeError("Figure 2A and SNN sheet geometries no longer register")

    def to_sheet(xy):
        return np.column_stack((xy, np.ones(len(xy)))) @ affine

    selected = np.isin(names, static_names)
    return {
        "names": names,
        "shafts": shafts,
        "measured_sheet_xy": to_sheet(measured[:, :2]),
        "rod_sheet_xy": to_sheet(fitted[:, :2]),
        "rod_normal_residual": fitted[:, 2]
        * float(figure2a.NORMAL_DISPLAY_EXAGGERATION),
        "selected": selected,
        "groups": figure2a._shaft_indices(names.tolist(), shafts.tolist()),
        "coord_space": coord_space,
        "source_artifact": str(figure2a.INPUT_ARTIFACT),
        "source_artifact_sha256": _sha256(figure2a.INPUT_ARTIFACT),
        "n_context_contacts": int(len(names)),
        "n_selected_contacts": int(np.sum(selected)),
        "context_not_selected": names[~selected].tolist(),
        "registration_max_abs_error_mm": registration_error,
        "straight_shaft_fit": fit_audit,
        "camera": {"elevation_deg": 32.0, "azimuth_deg": -95.0},
    }


def _plot_landscape(ax, bundle):
    static = bundle["static"]
    positions, h = static["positions_E"], static["h"]
    xx, yy, hh = _field_grid(positions, h)
    vmax = max(float(np.quantile(h, 0.995)), 1e-6)
    ax.computed_zorder = False
    surface = ax.plot_surface(
        xx, yy, np.minimum(hh, vmax), cmap="plasma", vmin=0.0, vmax=vmax,
        linewidth=0, antialiased=True, shade=False, alpha=0.97,
        rasterized=True, zorder=0.5,
    )
    ax.contour(xx, yy, hh, zdir="z", offset=0.0, levels=7, cmap="plasma",
               linewidths=0.55, alpha=0.75)
    context = _figure2a_context_geometry(bundle)
    plane_z = 1.055 * vmax
    corners = np.asarray([
        [0.0, 0.0, plane_z], [20.0, 0.0, plane_z],
        [20.0, 20.0, plane_z], [0.0, 20.0, plane_z],
    ])
    ax.add_collection3d(
        Poly3DCollection(
            [corners], facecolor="#EEF2F3", edgecolor="#849196",
            linewidth=0.7, alpha=0.12, zorder=2.0,
        )
    )
    residual = np.asarray(context["rod_normal_residual"], float)
    residual_scale = 0.095 * vmax / max(float(np.max(np.abs(residual))), 1e-9)
    rod_z = plane_z + residual * residual_scale
    footprint_xy = context["measured_sheet_xy"]
    rod_xy = context["rod_sheet_xy"]
    for group in context["groups"]:
        ax.plot(
            rod_xy[group, 0], rod_xy[group, 1], rod_z[group],
            color="#A9B1B5", lw=4.0, alpha=0.96, zorder=3.2,
            solid_capstyle="round",
        )
        ax.plot(
            rod_xy[group, 0], rod_xy[group, 1], rod_z[group],
            color="#59666C", lw=0.9, alpha=0.94, zorder=3.3,
            solid_capstyle="round",
        )
    for index in range(len(rod_xy)):
        ax.plot(
            [rod_xy[index, 0], footprint_xy[index, 0]],
            [rod_xy[index, 1], footprint_xy[index, 1]],
            [rod_z[index], plane_z], color="#899398", lw=0.42,
            alpha=0.32, zorder=2.7,
        )
    ax.scatter(
        footprint_xy[:, 0], footprint_xy[:, 1], np.full(len(rod_xy), plane_z),
        s=10, facecolor="#7F8B90", edgecolor="none", alpha=0.36,
        depthshade=False, zorder=2.8,
    )
    unselected = ~context["selected"]
    ax.scatter(
        rod_xy[unselected, 0], rod_xy[unselected, 1], rod_z[unselected],
        s=34, facecolor="#C3C9CC", edgecolor="#59666C", linewidth=0.7,
        depthshade=False, zorder=4.0,
    )
    for shaft in ("ICL", "SCL"):
        selected = context["selected"] & (context["shafts"] == shaft)
        ax.scatter(
            rod_xy[selected, 0], rod_xy[selected, 1], rod_z[selected],
            s=42, facecolor=SHAFT_COLORS[shaft], edgecolor="white",
            linewidth=0.8, depthshade=False, zorder=4.2,
        )
    ax.set(xlim=(0, 20), ylim=(0, 20), zlim=(0, 1.18 * vmax))
    ax.set_proj_type("ortho")
    ax.view_init(elev=32.0, azim=-95.0)
    ax.set_box_aspect((1.0, 1.0, 0.46), zoom=1.03)
    ax.set_axis_off()
    colorbar = plt.colorbar(surface, ax=ax, fraction=0.042, pad=0.035, shrink=0.78)
    colorbar.set_label("field h", fontsize=10.5)
    colorbar.ax.tick_params(labelsize=9)
    return context


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


def _mode_onset_density(bundle, mode, xx, yy, sigma_mm=1.2):
    """Smooth all earliest-contact locations for one frozen mode."""
    xy = bundle["static"]["contact_xy_mm"]
    selected = bundle["onsets"][bundle["clean"] & (bundle["labels"] == mode)]
    density = np.zeros_like(xx, dtype=float)
    contact_mass = np.zeros(len(xy), dtype=float)
    for onset in selected:
        finite = np.isfinite(onset)
        if not np.any(finite):
            continue
        earliest = finite & np.isclose(onset, np.nanmin(onset))
        contact_mass[earliest] += 1.0 / int(np.sum(earliest))
    for mass, center in zip(contact_mass, xy):
        if mass <= 0.0:
            continue
        radius2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        density += mass * np.exp(-0.5 * radius2 / float(sigma_mm) ** 2)
    if np.max(density, initial=0.0) > 0.0:
        density /= np.max(density)
    return density, contact_mass / max(1, len(selected))


def _mode_mean_direction(bundle, mode):
    xy = bundle["static"]["contact_xy_mm"]
    selected = bundle["ranks"][bundle["clean"] & (bundle["labels"] == mode)]
    early_centers, late_centers = [], []
    for event_ranks in normalize_event_ranks(selected):
        finite = np.isfinite(event_ranks)
        if np.sum(finite) < 3:
            continue
        early = finite & (event_ranks <= 0.34)
        late = finite & (event_ranks >= 0.66)
        if np.any(early) and np.any(late):
            early_centers.append(np.mean(xy[early], axis=0))
            late_centers.append(np.mean(xy[late], axis=0))
    if not early_centers:
        return None
    return np.mean(early_centers, axis=0), np.mean(late_centers, axis=0)


def _plot_mode_density(ax, bundle, mode, display_name, *, show_ylabel):
    static = bundle["static"]
    xx, yy, hh = _field_grid(static["positions_E"], static["h"], size=110)
    vmax = max(float(np.quantile(static["h"], 0.995)), 1e-9)
    ax.contourf(xx, yy, np.minimum(hh, vmax), levels=18, cmap="plasma",
                vmin=0.0, vmax=vmax, alpha=0.94)
    density, contact_mass = _mode_onset_density(bundle, mode, xx, yy)
    if np.max(density, initial=0.0) > 0.0:
        levels = (0.22, 0.45, 0.70)
        ax.contour(xx, yy, density, levels=levels, colors=MODE_COLORS[mode],
                   linewidths=(1.1, 1.6, 2.2), alpha=0.98)
        ax.contourf(xx, yy, density, levels=(0.45, 0.70, 1.01),
                    colors=[MODE_COLORS[mode], MODE_COLORS[mode]],
                    alpha=0.10)
    _plot_contacts(ax, bundle)
    xy = static["contact_xy_mm"]
    present = contact_mass > 0.0
    ax.scatter(xy[present, 0], xy[present, 1],
               s=45 + 220 * contact_mass[present], facecolor="white",
               edgecolor=MODE_COLORS[mode], linewidth=1.3, alpha=0.92, zorder=8)
    direction = _mode_mean_direction(bundle, mode)
    if direction is not None:
        start, stop = direction
        delta = stop - start
        for color, linewidth in (("white", 5.2), (MODE_COLORS[mode], 2.7)):
            ax.annotate(
                "", xy=stop, xytext=start,
                arrowprops={"arrowstyle": "-|>", "color": color,
                            "lw": linewidth, "mutation_scale": 15,
                            "shrinkA": 0, "shrinkB": 0},
                zorder=10,
            )
    ax.set(xlim=(0, 20), ylim=(0, 20), xlabel="sheet x (mm)")
    ax.set_ylabel("sheet y (mm)" if show_ylabel else "")
    if not show_ylabel:
        ax.tick_params(axis="y", labelleft=False)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9.5)
    ax.xaxis.label.set_size(11)
    ax.yaxis.label.set_size(11)
    ax.set_title(display_name, fontsize=14,
                 color=MODE_COLORS[mode], weight="bold", pad=8)
    ax.text(0.035, 0.955, f"clean events  n={int(bundle['clean_counts'][mode])}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            color="white", weight="bold",
            bbox={"facecolor": MODE_COLORS[mode], "edgecolor": "none",
                  "alpha": 0.86, "pad": 2.2})


def _bandpass_contact_activity(envelope, dt_ms, band_hz=TRACE_BAND_HZ):
    envelope = np.asarray(envelope, dtype=float)
    fs_hz = 1000.0 / float(dt_ms)
    if envelope.ndim != 2 or envelope.shape[1] < 20:
        raise ValueError("contact envelope must be contact x time with >=20 samples")
    if not (0.0 < band_hz[0] < band_hz[1] < 0.5 * fs_hz):
        raise ValueError("readout band must lie below the envelope Nyquist frequency")
    sos = butter(4, band_hz, btype="bandpass", fs=fs_hz, output="sos")
    return sosfiltfilt(sos, envelope, axis=1)


def _plot_readout(ax, bundle, pair):
    if pair is None:
        ax.text(0.5, 0.52, "MTA/MTB readout unavailable",
                transform=ax.transAxes, ha="center", va="center",
                color="#9B2F2A", weight="bold", fontsize=11)
        ax.text(0.5, 0.40, "no single fit network produced both formal modes",
                transform=ax.transAxes, ha="center", va="center", fontsize=8)
        ax.set_axis_off()
        return {"same_network_pair": False}
    ta_index, tb_index, block = pair
    records = bundle["records"]
    centers = []
    for global_index in (ta_index, tb_index):
        local = records[global_index]["local_index"]
        centers.append(0.5 * (block["event_t_on_ms"][local] + block["event_t_off_ms"][local]))
    width = max(760.0, abs(centers[1] - centers[0]) + 280.0)
    center = np.mean(centers)
    duration = block["contact_envelope"].shape[1] * block["contact_envelope_dt_ms"]
    start = max(0.0, min(duration - width, center - width / 2.0))
    stop, dt = min(duration, start + width), block["contact_envelope_dt_ms"]
    indices = np.arange(block["contact_envelope"].shape[1])
    sample = (indices * dt >= start) & (indices * dt <= stop)
    filtered = _bandpass_contact_activity(
        block["contact_envelope"], block["contact_envelope_dt_ms"],
    )
    trace = filtered[:, sample]
    t = indices[sample] * dt - start
    patient_ranks = np.asarray(bundle["patient"]["patient_train_ranks"], float).T
    sys.path.insert(0, str(ROOT))
    from scripts import plot_interictal_propagation as propagation_plot  # noqa: E402
    patient_order = propagation_plot._fixed_channel_order(
        patient_ranks, np.isfinite(patient_ranks),
    )
    contacts = patient_order
    scale = max(float(np.quantile(np.abs(trace[contacts]), 0.99)), 1e-9)
    offsets = np.arange(len(contacts), dtype=float) * 1.25
    for mode, global_index in ((TA_MODE, ta_index), (TB_MODE, tb_index)):
        local = records[global_index]["local_index"]
        shade_pad_ms = 18.0
        ax.axvspan(max(start, block["event_t_on_ms"][local] - shade_pad_ms) - start,
                   min(stop, block["event_t_off_ms"][local] + shade_pad_ms) - start,
                   color=MODE_COLORS[mode], alpha=0.14, lw=0)
    for row, contact in enumerate(contacts):
        shaft = bundle["static"]["shaft_ids"][contact]
        ax.plot(t, trace[contact] * 0.72 / scale + offsets[row],
                color=SHAFT_COLORS[shaft], lw=0.95, alpha=0.96)
    for mode, global_index in ((TA_MODE, ta_index), (TB_MODE, tb_index)):
        onset = bundle["onsets"][global_index]
        xs, ys = [], []
        for row, contact in enumerate(contacts):
            if not np.isfinite(onset[contact]):
                continue
            x = float(onset[contact] - start)
            sample_index = int(np.clip(round(onset[contact] / dt), 0,
                                       filtered.shape[1] - 1))
            xs.append(x)
            ys.append(offsets[row] + filtered[contact, sample_index] * 0.72 / scale)
        ax.scatter(xs, ys, s=9, color="#222222", edgecolor="white",
                   linewidth=0.25, zorder=8)
    bar_x = 0.045 * max(stop - start, 1.0)
    bar_y = offsets[-1] + 0.05
    ax.plot([bar_x, bar_x], [bar_y - 0.36, bar_y + 0.36],
            color="#222222", lw=1.6, clip_on=False)
    ax.text(bar_x + 0.015 * max(stop - start, 1.0), bar_y,
            f"{scale:.2g} a.u.", ha="left", va="center", fontsize=9)
    ax.set_yticks(offsets, bundle["static"]["contact_names"][contacts], fontsize=9.5)
    for tick, contact in zip(ax.get_yticklabels(), contacts):
        tick.set_color(SHAFT_COLORS[bundle["static"]["shaft_ids"][contact]])
    ax.set_xlim(0, stop - start)
    ax.set_ylim(-0.65, offsets[-1] + 0.95)
    ax.set_xlabel("simulation time (ms)", fontsize=11.5)
    ax.set_ylabel("30-80 Hz virtual-contact activity", fontsize=11.5)
    ax.tick_params(axis="x", labelsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        Patch(facecolor=TA_COLOR, alpha=0.2, label="MTA"),
        Patch(facecolor=TB_COLOR, alpha=0.2, label="MTB"),
    ], frameon=False, fontsize=9.5, ncol=2, loc="lower right",
       bbox_to_anchor=(1.0, 1.01))
    return {"same_network_pair": True, "seed": block["seed"],
            "MTA_global_index": ta_index, "MTB_global_index": tb_index,
            "pair_contract": (
                "highest shared contact support among temporally separated clean "
                "MTA/MTB pairs from one same-network run"
            ),
            "signal_contract": "30-80 Hz bandpass of virtual-contact firing-density envelope",
            "not_current_lfp_or_clinical_seeg": True,
            "band_hz": list(TRACE_BAND_HZ), "common_scale_au": scale,
            "display_shading_pad_ms": 18.0,
            "displayed_contact_count": int(len(contacts)),
            "contact_order": bundle["static"]["contact_names"][contacts].tolist(),
            "display_window_ms": [float(start), float(stop)]}


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
    primary = bundle.get("manifest", {}).get("selection_freeze", {}).get(
        "primary_candidate_id",
    )
    return {
        "figure": figure, "plotting_only": True,
        "candidate_id": bundle["candidate_id"],
        "candidate_role": bundle["figure_candidate_selection"],
        "phase_diagnostic_best_candidate_id": bundle["summary"][
            "diagnostic_best_candidate_id"
        ],
        "candidate_is_phase_diagnostic_best": bool(
            bundle["candidate_id"]
            == bundle["summary"]["diagnostic_best_candidate_id"]
        ),
        "candidate_is_prefrozen_primary": bool(
            primary is not None and bundle["candidate_id"] == primary
        ),
        "source_status": bundle["summary"]["status"],
        "formal_clean_mode_counts": {
            "MTA": int(bundle["clean_counts"][TA_MODE]),
            "MTB": int(bundle["clean_counts"][TB_MODE]),
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


def _patient_semantic_audit(bundle):
    """Map frozen numeric labels to manuscript TA/TB using Figure 2 fields."""
    from scripts.paper_figures import (  # noqa: E402
        plot_fig2_e1146_template_projection_composite as figure2a,
    )

    field = _json(figure2a.INPUT_ARTIFACT)["interictal_field"]
    field_names = [str(value) for value in field["contact_order"]]
    model_names = bundle["static"]["contact_names"].astype(str).tolist()
    reorder = np.asarray([model_names.index(name) for name in field_names], int)
    ranks = normalize_event_ranks(bundle["patient"]["patient_train_ranks"])
    labels = bundle["patient"]["patient_train_old_labels"].astype(int)
    old_profiles = np.asarray([
        _column_stats(ranks[labels == mode])[0][reorder] for mode in (0, 1)
    ])
    field_profiles = np.asarray([
        field["rank_a"], field["rank_b"],
    ], float)
    matrix = _similarity(old_profiles, field_profiles)
    if not (
        matrix[TA_MODE, 0] > matrix[TA_MODE, 1]
        and matrix[TB_MODE, 1] > matrix[TB_MODE, 0]
    ):
        raise RuntimeError("frozen numeric modes no longer map to TA=1 and TB=0")
    return {
        "numeric_label_to_semantic_model_mode": {"0": "MTB", "1": "MTA"},
        "field_columns": ["TA", "TB"],
        "old_numeric_mode_vs_figure2_field_spearman": matrix.tolist(),
        "patient_training_counts": {
            "TA": int(np.sum(labels == TA_MODE)),
            "TB": int(np.sum(labels == TB_MODE)),
        },
        "figure2_field_artifact": {
            "path": str(figure2a.INPUT_ARTIFACT),
            "sha256": _sha256(figure2a.INPUT_ARTIFACT),
        },
    }


def _render_direct(bundle, output_dir):
    pair = _same_network_pair(bundle)
    semantic_audit = _patient_semantic_audit(bundle)
    fig = plt.figure(figsize=(19.2, 4.75), facecolor="white")
    grid = fig.add_gridspec(
        1, 4, width_ratios=(1.48, 1.0, 1.0, 2.42),
        left=0.035, right=0.992, bottom=0.16, top=0.88, wspace=0.24,
    )
    axes = [
        fig.add_subplot(grid[0, 0], projection="3d")
        if _is_spatial_ou(bundle) else fig.add_subplot(grid[0, 0]),
        *[fig.add_subplot(grid[0, index]) for index in range(1, 4)],
    ]
    context = None
    if _is_spatial_ou(bundle):
        context = _plot_landscape(axes[0], bundle)
    else:
        _plot_flow(axes[0], bundle)
    _plot_mode_density(
        axes[1], bundle, TA_MODE, "Model TA", show_ylabel=True,
    )
    _plot_mode_density(
        axes[2], bundle, TB_MODE, "Model TB", show_ylabel=False,
    )
    readout = _plot_readout(axes[3], bundle, pair)
    stem = Path(output_dir) / (
        "fig4a_spatial_ou_direct_readout" if _is_spatial_ou(bundle)
        else "fig4a_spatial_edge_flow_direct_readout"
    )
    metadata = _metadata(bundle, (
        "Fig4A spatial-OU direct readout" if _is_spatial_ou(bundle)
        else "Fig4A spatial edge-flow direct readout"
    ))
    direction_meta = {}
    for mode, label in ((TA_MODE, "MTA"), (TB_MODE, "MTB")):
        direction = _mode_mean_direction(bundle, mode)
        direction_meta[label] = (
            None if direction is None else [row.tolist() for row in direction]
        )
    metadata.update({
        "files": _save(fig, stem), "direct_readout": readout,
        "layout": [
            "continuous_field_landscape", "MTA_all_event_onset_density",
            "MTB_all_event_onset_density", "direct_contact_readout",
        ],
        "mode_map_contract": (
            "frozen numeric label 1 is MTA and label 0 is MTB, established against "
            "the canonical Figure 2 TA/TB fields; continuous h background plus "
            "all-clean-event earliest-contact density and mean lower-third to "
            "upper-third normalized-rank direction"
        ),
        "semantic_mode_audit": semantic_audit,
        "mode_mean_direction_xy_mm": direction_meta,
        "removed_redundant_panel": "frozen signed delta-Vtheta node map",
    })
    if context is not None:
        metadata["figure2a_geometry_context"] = {
            key: context[key] for key in (
                "coord_space", "source_artifact", "source_artifact_sha256",
                "n_context_contacts",
                "n_selected_contacts", "context_not_selected",
                "registration_max_abs_error_mm", "camera",
            )
        }
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
    for mode in SEMANTIC_MODE_ORDER:
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


def _map_kmeans_clusters_to_modes(labels, contingency):
    """Choose the two-cluster display permutation with maximal A/B agreement."""
    contingency = np.asarray(contingency, dtype=int)
    if contingency.shape != (2, 2):
        raise ValueError("KMeans direction contingency must be 2 x 2")
    identity = int(contingency[0, 0] + contingency[1, 1])
    swapped = int(contingency[0, 1] + contingency[1, 0])
    cluster_to_mode = np.array([0, 1] if identity >= swapped else [1, 0], int)
    return cluster_to_mode[np.asarray(labels, int)], cluster_to_mode


def _render_kmeans(bundle, output_dir):
    sys.path.insert(0, str(ROOT))
    from scripts import plot_interictal_propagation as propagation_plot  # noqa: E402
    from scripts.paper_figures.plot_fig1_interictal_hfo_temporal_scaffold import (  # noqa: E402
        _draw_fig1e_cluster_row,
    )

    canonical = _canonical_rank_kmeans(bundle)
    clean_index = canonical["clean_global_index"]
    n_contacts = len(bundle["static"]["contact_names"])
    display_ranks = normalize_event_ranks(bundle["ranks"][clean_index]) * (
        n_contacts - 1
    )
    frozen_direction = canonical["direction"]
    frozen_labels, cluster_to_mode = _map_kmeans_clusters_to_modes(
        canonical["labels"], canonical["direction_contingency"],
    )
    labels = np.where(frozen_labels == TA_MODE, 0, 1).astype(int)
    direction = np.where(frozen_direction == TA_MODE, 0, 1).astype(int)
    semantic_audit = _patient_semantic_audit(bundle)
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
        "display_cluster_to_frozen_mode": {
            str(cluster): "MTA" if mode == TA_MODE else "MTB"
            for cluster, mode in enumerate(cluster_to_mode.tolist())
        },
    }
    names = bundle["static"]["contact_names"]
    patient, patient_low, patient_high = _patient_profiles(bundle)
    patient *= n_contacts - 1
    patient_low *= n_contacts - 1
    patient_high *= n_contacts - 1
    patient_rank_matrix = np.asarray(
        bundle["patient"]["patient_train_ranks"], float,
    ).T
    channel_order = propagation_plot._fixed_channel_order(
        patient_rank_matrix, np.isfinite(patient_rank_matrix),
    )
    ranks = display_ranks.T
    bools = np.isfinite(ranks)
    valid_events = np.arange(len(labels), dtype=int)
    clustered_order = np.argsort(labels, kind="stable")
    arr = {
        "ranks": ranks,
        "bools": bools,
        "channel_order": channel_order,
        "ordered_names": names[channel_order].tolist(),
        "clustered_events_all": clustered_order,
        "clustered_labels_all": labels[clustered_order],
        "valid_events": valid_events,
        "labels": labels,
        "channel_names": names.tolist(),
    }

    fig = plt.figure(figsize=(19.2, 4.95), facecolor="white")
    outer = fig.add_gridspec(
        1, 5, width_ratios=(4.35, 0.16, 1.05, 1.82, 1.42),
        left=0.047, right=0.975, bottom=0.15, top=0.91, wspace=0.34,
    )
    draw = _draw_fig1e_cluster_row(
        fig, outer, 0, arr,
        column_indices=(0, 1, 3),
        gap_half_width_events=max(1, int(round(0.012 * len(labels)))),
        cluster_label_names=["MTA", "MTB"],
        cluster_colors=list(DISPLAY_MODE_COLORS),
        mean_profile_label_names=["MTA", "MTB"],
        heatmap_ytick_fontsize=11.5,
        cluster_label_fontsize=12.5,
        mean_label_fontsize=13,
        mean_xtick_fontsize=11,
    )
    heatmap_ax = draw["axes"]["heatmap"]
    profile_ax = draw["axes"]["mean_rank"]
    heatmap_ax.set_title("")
    heatmap_ax.set_ylabel("electrode contact", fontsize=13)
    heatmap_ax.tick_params(axis="x", labelsize=11)
    profile_ax.set_title("cluster rank profile", fontsize=14, weight="bold", pad=8)
    profile_ax.set_xlabel("normalized rank", fontsize=13)
    profile_ax.tick_params(axis="x", labelsize=11)

    rank_grid = outer[0, 2].subgridspec(2, 1, height_ratios=(20, 1), hspace=0.06)
    rank_ax = fig.add_subplot(rank_grid[0])
    fig.add_subplot(rank_grid[1]).axis("off")
    propagation_plot._plot_rank_histogram(
        rank_ax, ranks, bools, valid_events, channel_order, names.tolist(),
        title="rank distribution", show_ylabels=False,
        label_fontsize=13, title_fontsize=14, xtick_fontsize=11,
        ridge_spacing=0.10, smooth_sigma_bins=0.72,
        smooth_ridge_height=0.12,
    )

    y = np.arange(n_contacts, dtype=float)
    for mode in (0, 1):
        ordered = channel_order
        finite = np.isfinite(patient[mode, ordered])
        profile_ax.fill_betweenx(
            y[finite], patient_low[mode, ordered][finite],
            patient_high[mode, ordered][finite],
            color=DISPLAY_MODE_COLORS[mode], alpha=0.08, linewidth=0,
        )
        profile_ax.plot(
            patient[mode, ordered][finite], y[finite], "--",
            color=DISPLAY_MODE_COLORS[mode], lw=1.7,
            label=f"T{'AB'[mode]}",
        )
    profile_ax.legend(
        handles=[
            Line2D([0], [0], color=TA_COLOR, lw=2.3,
                   marker="o", ms=4.5, label="MTA"),
            Line2D([0], [0], color=TB_COLOR, lw=2.3,
                   marker="o", ms=4.5, label="MTB"),
            Line2D([0], [0], color=TA_COLOR, lw=1.7,
                   ls="--", label="TA"),
            Line2D([0], [0], color=TB_COLOR, lw=1.7,
                   ls="--", label="TB"),
        ],
        frameon=False, fontsize=10, ncol=2, loc="upper right",
        bbox_to_anchor=(1.0, 1.0), columnspacing=0.8,
        handlelength=1.5, borderaxespad=0.2,
    )

    cluster_profiles = np.asarray([
        _column_stats(display_ranks[labels == mode])[0] for mode in (0, 1)
    ])
    cluster_matrix = _similarity(cluster_profiles, patient)
    matrix = cluster_matrix
    displayed_matrix_contract = "pooled KMeans cluster vs patient profile"
    d61_row = None
    if bundle["config"].get("scientific_role") in {
            "development_only_continuous_field_natural_kmeans_fresh_closeout",
            JOINT_CONTINUOUS_SURFACE_ROLE,
            "development_only_continuous_field_joint_direction_replication"}:
        verdict_path = bundle["output_root"] / "confirmation_verdict.json"
        verdict = _json(verdict_path)
        d61_row = next(
            row for row in verdict["candidate_rows"]
            if row["candidate_id"] == bundle["candidate_id"]
        )
        numeric_matrices = np.asarray([
            seed_row["crossfit_patient_readout"]["matrix"]
            for seed_row in d61_row["natural_kmeans_by_network"].values()
        ], float)
        numeric_matrix = np.nanmean(numeric_matrices, axis=0)
        matrix = numeric_matrix[np.ix_(SEMANTIC_MODE_ORDER, SEMANTIC_MODE_ORDER)]
        displayed_matrix_contract = (
            "equal-network mean contact-split patient readout; assign on one "
            "within-shaft alternating contact fold and evaluate on the disjoint fold"
        )
    supervised_profiles = np.asarray([
        _column_stats(display_ranks[direction == mode])[0] for mode in (0, 1)
    ])
    supervised_matrix = _similarity(supervised_profiles, patient)
    display_counts = np.bincount(labels, minlength=2)
    matrix_valid = bool(np.all(display_counts >= bundle["required_per_mode"]))
    matrix_grid = outer[0, 4].subgridspec(2, 1, height_ratios=(20, 1), hspace=0.06)
    matrix_ax = fig.add_subplot(matrix_grid[0])
    fig.add_subplot(matrix_grid[1]).axis("off")
    matrix_ax.set_xticks((0, 1), ("TA", "TB"), fontsize=13, weight="bold")
    matrix_ax.set_yticks((0, 1), ("MTA", "MTB"), fontsize=13, weight="bold")
    for tick, color in zip(matrix_ax.get_xticklabels(), DISPLAY_MODE_COLORS):
        tick.set_color(color)
    for tick, color in zip(matrix_ax.get_yticklabels(), DISPLAY_MODE_COLORS):
        tick.set_color(color)
    matrix_ax.tick_params(axis="x", labelrotation=0, pad=5)
    matrix_ax.set_aspect("equal")
    matrix_ax.set_title(
        "cross-fit vs patient" if d61_row is not None else "cluster vs patient",
        fontsize=14, weight="bold", pad=8,
    )
    matrix_cbar_ax = matrix_ax.inset_axes([1.08, 0.0, 0.065, 1.0])
    if matrix_valid:
        image = matrix_ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        for row in range(2):
            for column in range(2):
                color = "white" if abs(matrix[row, column]) >= 0.55 else "#111111"
                matrix_ax.text(column, row, f"{matrix[row, column]:+.2f}",
                               ha="center", va="center", fontsize=13,
                               color=color, weight="bold")
        cbar = fig.colorbar(image, cax=matrix_cbar_ax)
        cbar.set_label("Spearman rho", fontsize=12)
        cbar.ax.tick_params(labelsize=10.5)
    else:
        matrix_cbar_ax.axis("off")
        matrix_ax.text(0.5, 0.5, "N/A", transform=matrix_ax.transAxes,
                       ha="center", va="center", fontsize=20,
                       color="#9B2F2A", weight="bold")
    corrected_path = bundle["output_root"] / "confirmation_verdict.json"
    corrected = _json(corrected_path) if corrected_path.exists() else {}
    benchmark = corrected.get("patient_matched_kmeans_direction_purity", {})
    q05 = benchmark.get("q05")
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
        "displayed_matrix": matrix.tolist(),
        "displayed_matrix_contract": displayed_matrix_contract,
        "pooled_kmeans_cluster_vs_patient_spearman_descriptive": (
            cluster_matrix.tolist()
        ),
        "d6_1_equal_network_natural_kmeans": (
            None if d61_row is None else d61_row[
                "natural_balanced_alignment_equal_network"
            ]
        ),
        "d6_1_equal_network_crossfit_margin": (
            None if d61_row is None else d61_row[
                "crossfit_margin_equal_network"
            ]
        ),
        "supervised_MTA_MTB_vs_patient_TA_TB_spearman": supervised_matrix.tolist(),
        "matrix_rows": (
            "MTA/MTB are KMeans clusters ordered by maximal agreement with frozen "
            "numeric direction labels after the Figure 2 semantic audit"
        ),
        "display_cluster_counts_MTA_MTB": display_counts.tolist(),
        "patient_columns": ["TA", "TB"],
        "semantic_mode_audit": semantic_audit,
        "display_contact_order": names[channel_order].tolist(),
        "visible_qualifier_removed": {
            "direction_purity": audit["direction_purity"],
            "patient_q05": q05,
        },
        "contact_order_contract": "Figure 1E fixed order from patient-training ranks",
        "figure1e_shared_painter": (
            "scripts/paper_figures/plot_fig1_interictal_hfo_temporal_scaffold.py"
            "::_draw_fig1e_cluster_row"
        ),
        "corrected_confirmation_verdict": (
            {"path": str(corrected_path), "sha256": _sha256(corrected_path)}
            if corrected_path.exists() else None
        ),
    })
    Path(str(stem) + "_metadata.json").write_text(json.dumps(metadata, indent=2))
    return stem


def _write_readme(output_dir, bundle):
    path = Path(output_dir) / "README.md"
    primary = bundle.get("manifest", {}).get("selection_freeze", {}).get(
        "primary_candidate_id",
    )
    role = bundle.get("config", {}).get("scientific_role")
    if role == JOINT_CONTINUOUS_SURFACE_ROLE:
        spatial_candidate_context = (
            "按运行前写定的 display rule 事后选择、仅用于诊断的候选"
        )
    else:
        spatial_candidate_context = (
            "fresh networks 运行前冻结的 primary"
            if primary is None or bundle["candidate_id"] == primary
            else "预冻结候选库中按 fresh 结果事后选择的描述性候选；不替代 primary"
        )
    candidate_context = (
        "selection 阶段在读取确认网络前冻结的非零候选"
        if bundle["phase"] == "confirmation"
        else "等网络 fit screen 冻结的 diagnostic best"
    )
    if _is_spatial_ou(bundle):
        d61 = bundle["config"].get("scientific_role") in {
            "development_only_continuous_field_natural_kmeans_fresh_closeout",
            JOINT_CONTINUOUS_SURFACE_ROLE,
            "development_only_continuous_field_joint_direction_replication",
        }
        matrix_text = (
            "最右矩阵是 6 张网络等权的 contact-split cross-fit patient readout："
            "在一组交替触点上分配模式，在互斥触点上评价，再交换两组。"
            if d61 else
            "最右矩阵沿用相同语义色。"
        )
        path.write_text(f"""### fig4a_spatial_ou_direct_readout

这张图展示 {spatial_candidate_context} `{bundle['candidate_id']}`。左侧直接复用 Figure 2A 的 E1146 ICL/SCL 三维几何、正交相机和触点投影语法：20 个局部植入触点全部显示，其中未进入 SNN readout 的 SCL1-SCL5 为灰色；透明投影平面覆盖连续 `h` landscape。中间按 Figure 2 冻结模板核正为 Model TA（数值标签 1）和 Model TB（数值标签 0），分别汇总所有 formal clean 事件的 earliest-contact density 与平均传播方向；右侧显示同一网络中触点支持最高且时间上分离的 MTA/MTB 事件对，并按 Figure 1E 的固定顺序展示全部 15 个 readout 电极。

**关注点**：两种模式的起始可达区域是否在同一连续场上形成可辨认的空间差异。右侧信号是 contact firing-density envelope 的带通结果，不是 current-LFP 或临床 SEEG 电压。

### fig4b_spatial_ou_kmeans_consistency

这张图只使用 returned、双杆、patient-support 内的 formal clean events。热图、masked cells、白色斜线分隔、色条、电极顺序和 rank summary 直接复用 Figure 1E painter；rank distribution 使用同一共享函数的平滑紧凑 ridgeline 显示。KMeans 两簇先按 frozen 数值方向标签排列，再经过 Figure 2 模板审计统一命名为 MTA/MTB；实线表示模型 MTA/MTB，虚线表示患者 TA/TB。{matrix_text}

**关注点**：同时看 cluster-vs-patient 正对角/负交叉、direction purity 与 patient-matched purity 区间；患者 rank 几何接近不等于 KMeans 离散性达标。这是 development confirmation，不是 patient blind generalization。
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
    parser.add_argument("--allow-exploratory-candidate", action="store_true")
    parser.add_argument("--figure-output-dir")
    parser.add_argument("--expected-commit")
    args = parser.parse_args()
    config = _json(args.config)
    output_root = Path(args.output_root or ROOT / config["output_root"])
    bundle = _load_bundle(
        args.config, output_root, args.candidate_id,
        allow_exploratory_candidate=args.allow_exploratory_candidate,
    )
    provenance = None
    if args.expected_commit is not None:
        # Load the shared Figure 1E and rank-distribution painters before the
        # provenance snapshot so their exact source files are covered.
        from scripts import plot_interictal_propagation  # noqa: F401
        from scripts.paper_figures import (  # noqa: F401
            plot_fig1_interictal_hfo_temporal_scaffold,
        )
        from scripts.run_topic4_rev9l_forced_source_worker import (
            _runtime_provenance,
        )

        provenance = _runtime_provenance(args.expected_commit)
        if (provenance["runtime_modules_dirty"]
                or not provenance["runtime_modules_match_expected_commit"]):
            raise RuntimeError("Fig.4 validation producer modules are not frozen")
    figure_dir = Path(args.figure_output_dir or output_root / "figures")
    direct = _render_direct(bundle, figure_dir)
    kmeans = _render_kmeans(bundle, figure_dir)
    _write_readme(figure_dir, bundle)
    if provenance is not None:
        (figure_dir / "figure_provenance.json").write_text(json.dumps({
            "status": "FIG4_DATA_DRIVEN_SNN_PRODUCER_FROZEN",
            "candidate_id": bundle["candidate_id"],
            "provenance": provenance,
            "figures": [str(direct), str(kmeans)],
        }, indent=2))
    print(json.dumps({
        "status": (
            "REV10D6_2_FIG4_DIAGNOSTIC_COMPLETE"
            if config.get("scientific_role") == JOINT_CONTINUOUS_SURFACE_ROLE
            else
            "REV10D6_1_FIG4_DIAGNOSTIC_COMPLETE"
            if config.get("scientific_role") == (
                "development_only_continuous_field_natural_kmeans_fresh_closeout"
            ) else
            "REV10D5_2_FIG4_VALIDATION_COMPLETE" if _is_spatial_ou(bundle)
            else "REV10R2_FIG4_VALIDATION_COMPLETE"
        ),
        "candidate_id": bundle["candidate_id"],
        "figures": [str(direct), str(kmeans)],
        "producer_frozen": provenance is not None,
    }, indent=2))


if __name__ == "__main__":
    main()
