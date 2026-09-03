#!/usr/bin/env python3
"""Render rev22-DCI Task 12 figures from frozen aggregate files only.

No simulation, patient artifact, worker artifact, or statistical recomputation is
allowed here.  The producer verifies the hash chain before creating its output
directory and renders the already-frozen descriptive and paired-bootstrap results.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402


PARAMS = ("g_LEE", "g_LEI", "theta_FT_deg", "AR_FT")
PARAM_LABELS = ("learned E->E dose", "learned E->I dose", "axis offset (deg)", "axis ratio")
PRIMARY = (
    ("D_support", "Held-out support", "lower"),
    ("D_order", "Held-out order", "lower"),
    ("D_time_ms", "Held-out timing (ms)", "lower"),
    ("recall", "Recall", "higher"),
    ("kmeans_alignment", "KMeans alignment", "higher"),
    ("ood", "OOD fraction", "lower"),
)
TRAINING = (
    ("D_support", "Training support"),
    ("D_order", "Training order"),
    ("D_lag", "Training timing"),
    ("D_cover", "Training coverage"),
)
SECONDARY = (
    ("D_cloud_composite", "Composite", "lower"),
    ("yield_total", "Yield", "higher"),
    ("c2st_separability", "C2ST separability", "lower"),
)
SCHEMAS = {
    "validation": "topic4_rev22_dci_validation_aggregate_v1",
    "validation_surface": "topic4_rev22_dci_validation_response_surface_v1",
    "fit": "topic4_rev22_dci_training_only_fit_aggregate_v2",
    "response": "topic4_rev22_dci_response_fit_v1",
    "frozen": "topic4_rev22_dci_frozen_candidates_v1",
}
FORBIDDEN = ("patient_ictal", "seizure", "fig3", "fig5", "worker", ".npz")
FORMATS = ("png", "pdf", "svg")
COLORS = {"blue": "#2878B5", "red": "#C43C39", "teal": "#159D8C", "gray": "#8A8A8A"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _guard_input(path: Path, role: str) -> Path:
    resolved = path.expanduser().resolve()
    lowered = str(resolved).lower().replace("interictal", "")
    if resolved.suffix.lower() not in {".json", ".csv"}:
        raise RuntimeError(f"plot input must be frozen JSON/CSV: {role}: {resolved}")
    if any(marker in lowered for marker in FORBIDDEN):
        raise RuntimeError(f"forbidden non-aggregate input for {role}: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {role}: {resolved}")
    return resolved


def _read_json(path: Path, role: str, schema: str) -> tuple[dict, str, Path]:
    resolved = _guard_input(path, role)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_id") != schema:
        raise RuntimeError(f"unexpected {role} schema: {payload.get('schema_id')}")
    return payload, _sha256(resolved), resolved


def _finite(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _hash_entry(payload: Mapping, *keys: str) -> str | None:
    value: Any = payload
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    if isinstance(value, Mapping):
        value = value.get("sha256")
    return str(value) if value else None


def load_inputs(validation_path: Path, validation_surface_path: Path, fit_path: Path,
                response_path: Path, frozen_path: Path) -> dict:
    validation, validation_hash, vp = _read_json(
        validation_path, "validation aggregate", SCHEMAS["validation"])
    validation_surface, validation_surface_hash, sp = _read_json(
        validation_surface_path, "validation response surfaces", SCHEMAS["validation_surface"])
    fit, fit_hash, fp = _read_json(fit_path, "fit aggregate", SCHEMAS["fit"])
    response, response_hash, rp = _read_json(response_path, "response fit", SCHEMAS["response"])
    frozen, frozen_hash, zp = _read_json(frozen_path, "frozen candidates", SCHEMAS["frozen"])
    if validation.get("status") != "VALIDATION_AGGREGATE_COMPLETE":
        raise RuntimeError("validation aggregate is not complete")
    if fit.get("status") != "FIT_AGGREGATE_COMPLETE":
        raise RuntimeError("fit aggregate is not complete")
    if response.get("status") != "RESPONSE_FIT_COMPLETE" or response.get("training_only") is not True:
        raise RuntimeError("response fit is not a complete training-only freeze")
    if validation.get("snn_simulation_run") is not False or validation.get("patient_ictal_input_read") is not False:
        raise RuntimeError("validation aggregate violates the pure plotting boundary")
    fit_descriptive = validation.get("fit_descriptive")
    if not isinstance(fit_descriptive, Mapping):
        raise RuntimeError("validation aggregate lacks fit_descriptive")
    if fit_descriptive.get("descriptive_only") is not True or fit_descriptive.get("cannot_select") is not True:
        raise RuntimeError("fit_descriptive is not locked descriptive-only/cannot-select")
    fit_rows = fit_descriptive.get("candidates")
    if int(fit_descriptive.get("candidate_count", -1)) != 96 or not isinstance(fit_rows, list) or len(fit_rows) != 96:
        raise RuntimeError("fit_descriptive must contain exactly 96 candidates")
    if (validation_surface.get("status") != "DESCRIPTIVE_VALIDATION_RESPONSE_COMPLETE"
            or validation_surface.get("descriptive_only") is not True
            or validation_surface.get("cannot_select") is not True
            or validation_surface.get("selection_permitted") is not False):
        raise RuntimeError("validation response surfaces are not descriptive-only/cannot-select")
    if int(validation_surface.get("design_point_count", -1)) != 96:
        raise RuntimeError("validation response surfaces must retain all 96 design points")
    bindings = (
        (_hash_entry(validation, "input_hashes", "frozen_candidates"), frozen_hash,
         "validation -> frozen candidates"),
        (_hash_entry(validation_surface, "input_hashes", "validation_aggregate"), validation_hash,
         "validation response surfaces -> validation aggregate"),
        (_hash_entry(validation_surface, "input_hashes", "frozen_candidates"), frozen_hash,
         "validation response surfaces -> frozen candidates"),
        (_hash_entry(response, "input_hashes", "fit_aggregate"), fit_hash,
         "response fit -> fit aggregate"),
        (_hash_entry(frozen, "input_hashes", "fit_aggregate"), fit_hash,
         "frozen candidates -> fit aggregate"),
        (_hash_entry(frozen, "input_hashes", "response_fit"), response_hash,
         "frozen candidates -> response fit"),
    )
    for recorded, observed, label in bindings:
        if recorded is None or recorded != observed:
            raise RuntimeError(f"broken frozen hash binding: {label}")
    confirmation = validation.get("phases", {}).get("confirmation")
    qualification = validation.get("phases", {}).get("qualification")
    if not isinstance(confirmation, list) or not isinstance(qualification, list):
        raise RuntimeError("validation phases are missing")
    frozen_ids = [str(value) for value in frozen.get("candidate_ids", [])]
    observed_ids = [str(row.get("candidate_id")) for row in confirmation]
    if not frozen_ids or len(frozen_ids) != len(set(frozen_ids)):
        raise RuntimeError("frozen candidate ids are empty or duplicated")
    if set(frozen_ids) != set(observed_ids):
        raise RuntimeError("confirmation candidates differ from the frozen candidate set")
    descriptive_ids = [str(row.get("candidate_id")) for row in fit_rows]
    surface_ids = [str(row.get("candidate_id"))
                   for row in validation_surface.get("original_design_points", [])]
    if len(set(descriptive_ids)) != 96 or set(descriptive_ids) != set(surface_ids):
        raise RuntimeError("validation surface points differ from fit_descriptive candidates")
    return {
        "validation": validation, "validation_surface": validation_surface,
        "fit": fit, "response": response, "frozen": frozen,
        "hashes": {"validation": validation_hash, "fit": fit_hash,
                   "validation_surface": validation_surface_hash,
                   "response": response_hash, "frozen": frozen_hash},
        "paths": {"validation": str(vp), "validation_surface": str(sp),
                  "fit": str(fp), "response": str(rp),
                  "frozen": str(zp)},
    }


def _physical_map(data: Mapping) -> dict[str, dict[str, float]]:
    mapping: dict[str, dict[str, float]] = {}
    for row in data["fit"].get("candidates", []):
        physical = row.get("physical") or {}
        if all(_finite(physical.get(name)) is not None for name in PARAMS):
            mapping[str(row["candidate_id"])] = {name: float(physical[name]) for name in PARAMS}
    for record in (data["response"].get("proposals") or {}).values():
        for point in record.get("frozen_points", []):
            candidate_id = point.get("execution_candidate_id") or point.get("existing_candidate_id")
            x = point.get("x")
            if candidate_id and isinstance(x, Sequence) and len(x) == len(PARAMS):
                mapping[str(candidate_id)] = {name: float(value) for name, value in zip(PARAMS, x)}
    missing = sorted(set(map(str, data["frozen"]["candidate_ids"])) - set(mapping))
    if missing:
        raise RuntimeError(f"frozen candidates lack physical parameters: {missing}")
    return mapping


def _family_map(data: Mapping) -> dict[str, str]:
    output = {}
    mask_map = data["frozen"].get("mask_to_candidates") or {}
    for mask, ids in mask_map.items():
        if len(str(mask)) != 5 or not str(mask).startswith("M"):
            raise RuntimeError(f"invalid family mask: {mask}")
        for candidate_id in ids:
            if candidate_id is not None:
                output.setdefault(str(candidate_id), str(mask))
    missing = sorted(set(map(str, data["frozen"]["candidate_ids"])) - set(output))
    if missing:
        raise RuntimeError(f"frozen candidates lack family mapping: {missing}")
    return output


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.0, "axes.titlesize": 8.0,
        "axes.labelsize": 7.0, "xtick.labelsize": 6.2, "ytick.labelsize": 6.2,
        "legend.fontsize": 6.2, "axes.linewidth": 0.7, "lines.linewidth": 1.1,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "svg.fonttype": "none", "pdf.fonttype": 42, "savefig.transparent": False,
    })


def _save(fig: plt.Figure, root: Path, stem: str) -> list[Path]:
    paths = []
    for suffix in FORMATS:
        path = root / f"{stem}.{suffix}"
        fig.savefig(path, dpi=300 if suffix == "png" else None, bbox_inches="tight",
                    facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _confirmation_rows(data: Mapping) -> list[dict]:
    order = list(map(str, data["frozen"]["candidate_ids"]))
    rows = {str(row["candidate_id"]): row
            for row in data["validation"]["phases"]["confirmation"]}
    return [rows[candidate_id] for candidate_id in order]


def _yield(row: Mapping) -> float:
    return float((row.get("secondary") or {}).get("yield_total") or 0.0)


def plot_validation_response(data: Mapping, out: Path) -> list[Path]:
    response = data["validation_surface"]
    rows = data["validation"]["fit_descriptive"]["candidates"]
    yields = np.asarray([_yield(row) for row in rows], float)
    sizes = 10.0 + 35.0 * np.sqrt(yields / max(float(yields.max()), 1.0))
    fig, axes = plt.subplots(len(PRIMARY), len(PARAMS), figsize=(8.0, 9.2), squeeze=False)
    for i, (endpoint, label, direction) in enumerate(PRIMARY):
        surface = (response.get("surfaces") or {}).get(endpoint)
        if not isinstance(surface, Mapping):
            raise RuntimeError(f"validation response surface missing endpoint: {endpoint}")
        for j, parameter in enumerate(PARAMS):
            ax = axes[i, j]
            good_x, good_y, good_s, bad_x = [], [], [], []
            for size, row in zip(sizes, rows):
                x = _finite((row.get("physical") or {}).get(parameter))
                value = _finite((row.get("primary_endpoints") or {}).get(endpoint))
                if x is None:
                    raise RuntimeError(f"descriptive point lacks parameter {parameter}")
                if row.get("primary_status") == "OK" and value is not None:
                    good_x.append(x); good_y.append(value); good_s.append(size)
                else:
                    bad_x.append(x)
            if good_x:
                ax.scatter(good_x, good_y, s=good_s, color=COLORS["gray"], alpha=0.18,
                           edgecolor="none", zorder=1)
            slices = surface.get("conditional_slices")
            record = slices.get(parameter) if isinstance(slices, Mapping) else None
            if surface.get("status") == "OK" and isinstance(record, Mapping):
                axis = np.asarray(record.get("axis"), float)
                mean = np.asarray(record.get("mean"), float)
                lo = np.asarray(record.get("lo90"), float)
                hi = np.asarray(record.get("hi90"), float)
                if not (axis.ndim == mean.ndim == lo.ndim == hi.ndim == 1
                        and len(axis) >= 2 and len(axis) == len(mean) == len(lo) == len(hi)
                        and np.isfinite(np.concatenate([axis, mean, lo, hi])).all()):
                    raise RuntimeError(f"malformed conditional slice: {endpoint}/{parameter}")
                adequate = bool((surface.get("cv") or {}).get("adequate"))
                color = COLORS["blue"] if adequate else COLORS["gray"]
                ax.fill_between(axis, lo, hi, color=color, alpha=0.15, linewidth=0)
                ax.plot(axis, mean, color=color, linestyle="-" if adequate else "--", zorder=3)
            else:
                ax.text(0.5, 0.5, "not estimable", ha="center", va="center",
                        transform=ax.transAxes, color=COLORS["gray"])
            if bad_x:
                top = ax.get_ylim()[1]
                ax.scatter(bad_x, [top] * len(bad_x), marker="x", s=20,
                           color=COLORS["red"], linewidth=0.9, clip_on=False, zorder=4)
            if i == 0:
                ax.set_title(PARAM_LABELS[j])
            if j == 0:
                ax.set_ylabel(label + ("\n(lower is better)" if direction == "lower" else
                                       "\n(higher is better)"))
            if i == len(PRIMARY) - 1:
                ax.set_xlabel(parameter)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    fig.suptitle("Selection-blind validation response", y=0.995, fontsize=9, fontweight="bold")
    fig.text(0.995, 0.005, "Line and band: conditional slice and 90% CI; pale point area: yield; x: not estimable", ha="right",
             va="bottom", fontsize=6.2, color="#555555")
    fig.tight_layout(rect=(0, 0.012, 1, 0.985))
    return _save(fig, out, "rev22_dci_validation_response_atlas")


def _secondary_value(row: Mapping, key: str) -> float | None:
    secondary = row.get("secondary") or {}
    if key == "c2st_separability":
        return _finite((secondary.get("c2st") or {}).get("separability"))
    return _finite(secondary.get(key))


def _contrast_by_candidate(validation: Mapping) -> dict[str, Mapping]:
    output = {}
    for row in validation.get("paired_pareto_contrasts", []):
        output[str(row.get("locked_candidate_id"))] = row
    for row in validation.get("paired_reference_contrasts", []):
        output.setdefault(str(row.get("candidate_id")), row)
    return output


def plot_family_matrix(data: Mapping, families: Mapping[str, str], out: Path) -> list[Path]:
    rows = _confirmation_rows(data)
    rows.sort(key=lambda row: (families[str(row["candidate_id"])], str(row["candidate_id"])))
    n = len(rows)
    fig = plt.figure(figsize=(10.3, max(3.6, 0.34 * n + 1.7)))
    grid = fig.add_gridspec(1, 10, width_ratios=[1.7, *([1.0] * 9)], wspace=0.22)
    mask_ax = fig.add_subplot(grid[0, 0])
    masks = np.asarray([[int(bit) for bit in families[str(row["candidate_id"])][1:]] for row in rows])
    mask_ax.imshow(masks, aspect="auto", cmap=matplotlib.colors.ListedColormap(["#F0F0F0", "#303030"]),
                   vmin=0, vmax=1)
    mask_ax.set_xticks(range(4), ["EE", "EI", "axis", "ratio"], rotation=45, ha="right")
    labels = [f"{families[str(row['candidate_id'])]}  {row['candidate_id']}" for row in rows]
    mask_ax.set_yticks(range(n), labels)
    mask_ax.set_title("Free parameters")
    mask_ax.tick_params(length=0)
    contrast = _contrast_by_candidate(data["validation"])
    for col, (endpoint, label, _) in enumerate(PRIMARY, start=1):
        ax = fig.add_subplot(grid[0, col], sharey=mask_ax)
        for y, row in enumerate(rows):
            record = contrast.get(str(row["candidate_id"]), {})
            item = (record.get("endpoints") or {}).get(endpoint) or {}
            point, lo, hi = (_finite(item.get(key)) for key in ("delta", "lo", "hi"))
            if point is None or lo is None or hi is None:
                ax.scatter([0], [y], marker="x", color=COLORS["gray"], s=13)
            else:
                ax.errorbar(point, y, xerr=[[point - lo], [hi - point]], fmt="o", ms=3.1,
                            color=COLORS["blue"], ecolor=COLORS["blue"], elinewidth=0.8,
                            capsize=1.5)
        ax.axvline(0, color="#AAAAAA", linewidth=0.7)
        ax.set_title(label, fontsize=7)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#ECECEC", linewidth=0.45)
    for offset, (key, label, _) in enumerate(SECONDARY, start=7):
        ax = fig.add_subplot(grid[0, offset], sharey=mask_ax)
        values = [_secondary_value(row, key) for row in rows]
        finite = [value for value in values if value is not None]
        lo, hi = (min(finite), max(finite)) if finite else (0.0, 1.0)
        span = max(hi - lo, 1e-12)
        for y, value in enumerate(values):
            if value is None:
                ax.scatter([0.5], [y], marker="x", color=COLORS["gray"], s=13)
            else:
                ax.scatter([(value - lo) / span], [y], s=16, color=COLORS["teal"])
        ax.set_xlim(-0.08, 1.08)
        ax.set_xticks([0, 1], [f"{lo:.2g}", f"{hi:.2g}"])
        ax.set_title(label, fontsize=7)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Nested-family comparison", y=0.995, fontsize=9, fontweight="bold")
    fig.text(0.58, 0.012, "Primary columns: paired improvement (point, 90% CI); x: not estimable",
             ha="center", fontsize=6.2, color="#555555")
    fig.subplots_adjust(left=0.16, right=0.99, bottom=0.09, top=0.91, wspace=0.34)
    return _save(fig, out, "rev22_dci_nested_family_matrix")


def plot_pareto(data: Mapping, families: Mapping[str, str], out: Path) -> list[Path]:
    rows = _confirmation_rows(data)
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    usable = []
    for row in rows:
        endpoints = row.get("primary_endpoints") or {}
        x, y, ood = (_finite(endpoints.get(key)) for key in ("D_order", "kmeans_alignment", "ood"))
        if row.get("primary_status") == "OK" and None not in (x, y, ood):
            usable.append((row, x, y, ood))
    if usable:
        oods = np.asarray([item[3] for item in usable], float)
        norm = Normalize(vmin=float(oods.min()), vmax=float(oods.max()) if oods.max() > oods.min() else float(oods.min() + 1))
        for row, x, y, ood in usable:
            size = 16 + 45 * math.sqrt(max(_yield(row), 0) / max(max(_yield(r[0]) for r in usable), 1))
            ax.scatter(x, y, c=[ood], cmap="magma_r", norm=norm, s=size,
                       edgecolor="white", linewidth=0.5)
            ax.annotate(families[str(row["candidate_id"])], (x, y), xytext=(3, 2),
                        textcoords="offset points", fontsize=5.6)
        scalar = matplotlib.cm.ScalarMappable(norm=norm, cmap="magma_r")
        fig.colorbar(scalar, ax=ax, pad=0.02, label="OOD fraction")
    else:
        ax.text(0.5, 0.5, "No estimable candidate", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Held-out order distance (lower)")
    ax.set_ylabel("KMeans alignment (higher)")
    ax.set_title("Validation trade-off", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#EAEAEA", linewidth=0.5)
    fig.tight_layout()
    return _save(fig, out, "rev22_dci_validation_pareto")


def plot_training_response(data: Mapping, out: Path) -> list[Path]:
    candidates = data["fit"].get("candidates") or []
    plane_specs = [("dose_plane", "g_LEE", "g_LEI", "Learned-pattern dose plane")]
    if data["response"].get("branch") == "PRIMARY_4D_BRANCH":
        plane_specs.append(("geometry_plane", "theta_FT_deg", "AR_FT", "Geometry plane"))
    fig, axes = plt.subplots(len(TRAINING), len(plane_specs),
                             figsize=(4.2 * len(plane_specs), 7.4), squeeze=False)
    row_mappables = []
    for i, (component, label) in enumerate(TRAINING):
        component_values = [
            _finite((row.get("standardized_Z") or {}).get(component))
            for row in candidates if row.get("block") in {spec[0] for spec in plane_specs}
        ]
        finite_component = [value for value in component_values if value is not None]
        if not finite_component:
            raise RuntimeError(f"training plane lacks estimable component: {component}")
        norm = Normalize(vmin=min(finite_component), vmax=(max(finite_component)
                         if max(finite_component) > min(finite_component)
                         else min(finite_component) + 1.0))
        row_mappable = None
        for j, (block, x_name, y_name, title) in enumerate(plane_specs):
            ax = axes[i, j]
            good, bad = [], []
            plane_rows = [row for row in candidates if row.get("block") == block]
            if len(plane_rows) < 3:
                raise RuntimeError(f"training response plane {block} has fewer than three points")
            for row in plane_rows:
                x = _finite((row.get("physical") or {}).get(x_name))
                y = _finite((row.get("physical") or {}).get(y_name))
                value = _finite((row.get("standardized_Z") or {}).get(component))
                if x is None or y is None:
                    raise RuntimeError(f"fit candidate lacks coordinates for {block}")
                if row.get("continuous_surface_eligible") and value is not None:
                    good.append((x, y, value))
                else:
                    bad.append((x, y))
            if len(good) >= 3:
                x = np.asarray([row[0] for row in good])
                y = np.asarray([row[1] for row in good])
                z = np.asarray([row[2] for row in good])
                centered = np.column_stack([x - x.mean(), y - y.mean()])
                if np.linalg.matrix_rank(centered) >= 2:
                    ax.tricontourf(x, y, z, levels=12, cmap="viridis", norm=norm, alpha=0.72)
                row_mappable = ax.scatter(x, y, c=z, cmap="viridis", norm=norm, s=22,
                                          edgecolor="white", linewidth=0.4, zorder=3)
            if bad:
                ax.scatter([row[0] for row in bad], [row[1] for row in bad], marker="x",
                           s=18, color=COLORS["gray"], zorder=4)
            if i == 0:
                ax.set_title(title)
            if j == 0:
                ax.text(-0.28, 0.5, label, rotation=90, ha="center", va="center",
                        transform=ax.transAxes)
            ax.set_xlabel(x_name)
            ax.set_ylabel(y_name)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(color="#ECECEC", linewidth=0.4)
        row_mappables.append(row_mappable)
    fig.suptitle("Training-only two-dimensional response planes", y=0.995,
                 fontsize=9, fontweight="bold")
    fig.subplots_adjust(left=0.12, right=0.87, bottom=0.07, top=0.95, hspace=0.5, wspace=0.38)
    for i, mappable in enumerate(row_mappables):
        if mappable is None:
            continue
        box = axes[i, -1].get_position()
        color_ax = fig.add_axes([0.895, box.y0, 0.012, box.height])
        colorbar = fig.colorbar(mappable, cax=color_ax)
        colorbar.set_label("standardized excess", fontsize=6.2)
        colorbar.ax.tick_params(labelsize=5.8, width=0.5)
    return _save(fig, out, "rev22_dci_training_response_atlas")


def failure_sidecar(data: Mapping) -> dict:
    validation_counts: dict[str, Counter] = {}
    for phase in ("qualification", "confirmation"):
        counter = Counter()
        for unit in data["validation"].get("unit_inventory", []):
            if unit.get("phase") != phase:
                continue
            reasons = unit.get("failure_reasons") or []
            if reasons:
                counter.update(map(str, reasons))
            else:
                counter["eligible"] += 1
        validation_counts[phase] = counter
    fit_counter = Counter()
    for row in data["fit"].get("candidates", []):
        if row.get("joint_feasibility"):
            fit_counter["feasible"] += 1
        else:
            reasons = row.get("candidate_failure_reasons") or ["not_feasible_unspecified"]
            fit_counter.update(map(str, reasons))
    return {
        "schema_id": "topic4_rev22_dci_failure_feasibility_v1",
        "fit_candidates": dict(sorted(fit_counter.items())),
        "validation_units": {phase: dict(sorted(counter.items()))
                             for phase, counter in validation_counts.items()},
        "fit_inventory": data["fit"].get("inventory"),
        "note": "Failures remain explicit; missing or invalid units are not converted to scientific scores.",
    }


def plot_failures(sidecar: Mapping, out: Path) -> list[Path]:
    panels = [("Fit candidates", sidecar["fit_candidates"]),
              ("Qualification units", sidecar["validation_units"]["qualification"]),
              ("Confirmation units", sidecar["validation_units"]["confirmation"])]
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 2.8))
    for ax, (title, counts) in zip(axes, panels):
        labels, values = list(counts), list(counts.values())
        if values:
            positions = np.arange(len(values))
            colors = [COLORS["teal"] if label in {"eligible", "feasible"} else COLORS["red"]
                      for label in labels]
            ax.barh(positions, values, color=colors, height=0.65)
            ax.set_yticks(positions, [label.replace("_", " ") for label in labels])
            ax.invert_yaxis()
            for y, value in zip(positions, values):
                ax.text(value, y, f" {value}", va="center", fontsize=6.2)
        else:
            ax.text(0.5, 0.5, "No units", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel("Count")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", color="#ECECEC", linewidth=0.45)
    fig.suptitle("Feasibility and artifact inventory", y=1.01, fontsize=9, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out, "rev22_dci_failure_feasibility")


def _atomic_json(path: Path, payload: Mapping) -> None:
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                   encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_readme(path: Path) -> None:
    path.write_text(
        """### rev22_dci_validation_response_atlas
六个 selection-blind 间期端点随四个连接参数的条件响应。实线及阴影来自冻结后 descriptive validation surface 在其他三个参数固定于 full-model 坐标时的一维切片及 90% 区间；96 个设计点仅以浅色散点显示，点面积表示返回事件产量，叉号表示不可估。该图是 descriptive-only 且 cannot-select，不是边际因果效应，也不能反向改变候选。

**关注点**：距离、recall、双模板一致性和 OOD 是否出现一致改善，而非由低产量换取。

### rev22_dci_nested_family_matrix
左侧标出每个嵌套家族放开或锁定的参数；六个主端点显示相对 full/reference 的配对改善及 90% 区间。右侧三列为 composite、yield 和 C2ST separability 的冻结绝对值，只作辅助描述。

**关注点**：锁回某个参数后，改善是否在配对网络层稳定消失。

### rev22_dci_validation_pareto
横轴为 held-out order 距离，纵轴为 KMeans alignment，颜色为 OOD，点面积为事件产量。该图展示验证指标间的权衡，不参与候选选择。

**关注点**：是否存在同时向左上移动、OOD 不升且不靠少出事件的候选。

### rev22_dci_training_response_atlas
96 点训练设计中预留的两组二维设计面：learned E→E×E→I dose，以及 primary branch 的轴向偏移×长短轴比。每行对应一个预注册训练目标分量，颜色为 standardized excess；灰叉保留不可行或不可估候选，不再把 Sobol unique-x 投影连接成折线。

**关注点**：哪些参数在训练目标中可辨识，以及训练改善是否与验证端一致。

### rev22_dci_failure_feasibility
分别汇总 fit 候选、qualification 单元和 confirmation 单元的可行性及失败原因。失败不会被替换为数值分数。

**关注点**：低产、runaway、非有限值、缺失产物或 provenance 异常是否集中在特定阶段。
""",
        encoding="utf-8",
    )


def render(data: Mapping, output_dir: Path) -> dict:
    _physical_map(data)
    families = _family_map(data)
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    _setup_style()
    with tempfile.TemporaryDirectory(dir=output_dir.parent,
                                     prefix=f".{output_dir.name}.") as temporary:
        staging = Path(temporary)
        outputs: list[Path] = []
        outputs += plot_validation_response(data, staging)
        outputs += plot_family_matrix(data, families, staging)
        outputs += plot_pareto(data, families, staging)
        outputs += plot_training_response(data, staging)
        sidecar = failure_sidecar(data)
        sidecar_path = staging / "rev22_dci_failure_feasibility.json"
        _atomic_json(sidecar_path, sidecar)
        outputs.append(sidecar_path)
        outputs += plot_failures(sidecar, staging)
        readme = staging / "README.md"
        _write_readme(readme)
        outputs.append(readme)
        metadata = {
            "schema_id": "topic4_rev22_dci_task12_figure_metadata_v1",
            "status": "FIGURES_RENDERED",
            "snn_simulation_run": False,
            "patient_ictal_input_read": False,
            "statistical_recomputation": False,
            "input_paths": data["paths"],
            "input_sha256": data["hashes"],
            "candidate_count": len(data["frozen"]["candidate_ids"]),
            "output_sha256": {path.name: _sha256(path) for path in outputs},
            "claim_boundary": (
                "Pure rendering of frozen training and selection-blind interictal aggregates; "
                "no SNN, patient ictal data, or post-hoc statistical recomputation."
            ),
        }
        _atomic_json(staging / "metadata.json", metadata)
        os.replace(staging, output_dir)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-aggregate", type=Path, required=True)
    parser.add_argument("--validation-response-surfaces", type=Path, required=True)
    parser.add_argument("--fit-aggregate", type=Path, required=True)
    parser.add_argument("--response-fit", type=Path, required=True)
    parser.add_argument("--frozen-candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    data = load_inputs(args.validation_aggregate, args.validation_response_surfaces,
                       args.fit_aggregate,
                       args.response_fit, args.frozen_candidates)
    metadata = render(data, args.output_dir)
    print(json.dumps({"status": metadata["status"], "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
