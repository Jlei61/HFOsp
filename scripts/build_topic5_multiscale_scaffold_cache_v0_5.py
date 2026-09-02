#!/usr/bin/env python3
"""Build the target-free v0.5 full-parent spatial cache and routing inventory.

The builder scans every patient in the frozen 34-patient masked-rank dataset.
It does not accept a hand-written cohort list and it never imports a target
reader.  Early-ictal routing metadata are copied with an explicit ``usecols``
allow-list; no activation, score, correlation, or energy column is loaded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CANONICAL_ROOT = ROOT.parents[1] if (ROOT.parents[1] / "results").exists() else ROOT

from src.topic5_lbss_rnn_v0_2 import build_pool_contract, strong_component_audit  # noqa: E402
from src.topic5_virtual_seeg_operator import kernel_sigma_mm, resolve_full_tissue_layout  # noqa: E402


DATASET_ROOT = CANONICAL_ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4"
FIELD_ROOT = CANONICAL_ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject"
FIG3_EVENT_TABLE = (
    CANONICAL_ROOT
    / "results/topic5_ictal_recruitment/tspectral_field_concordance"
    / "clinical_onset_gradient_field_cohort_stat_event.csv"
)
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OLD_CACHE_ROOT = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3/cache"
MIN_JOINT_CONTACTS = 6
NODE_SEED = 20260812
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
ROUTING_COLUMNS = (
    "dataset", "subject", "seizure_idx", "group_id", "phenotype", "band",
    "permutation_seed",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def densify_groups(groups: np.ndarray) -> np.ndarray:
    values = np.asarray(groups, dtype=np.int16)
    if values.ndim != 2:
        raise ValueError("groups must be event x contact")
    out = np.full_like(values, -1)
    for event_index, event in enumerate(values):
        present = np.unique(event[event >= 0])
        mapping = {int(old): new for new, old in enumerate(present)}
        for contact_index, old in enumerate(event):
            if old >= 0:
                out[event_index, contact_index] = mapping[int(old)]
    return out


def development_split(event_split: np.ndarray) -> np.ndarray:
    """Reproduce the v0.3 train/validation/test split inside frozen train80."""
    frozen = np.asarray(event_split, dtype=np.uint8)
    pool = np.flatnonzero(frozen == 0)
    train_cut = int(np.floor((1.0 - VALIDATION_FRACTION - TEST_FRACTION) * len(pool)))
    validation_cut = int(np.floor((1.0 - TEST_FRACTION) * len(pool)))
    if train_cut < 1 or validation_cut <= train_cut or validation_cut >= len(pool):
        raise RuntimeError("development split would create an empty partition")
    split = np.full(len(frozen), -1, dtype=np.int8)
    split[pool[:train_cut]] = 0
    split[pool[train_cut:validation_cut]] = 1
    split[pool[validation_cut:]] = 2
    return split


def solved_scopes(field: dict) -> tuple[dict[str, dict], str | None]:
    if field.get("status") != "ok" or "contact_order" not in field:
        return {}, str(field.get("status", "FIELD_CONTRACT_MISSING"))
    planes = field.get("planes", {})
    shared = planes.get("shared") or {}
    if shared.get("status") == "ok":
        return {"shared": shared}, None
    output = {}
    for scope in ("own_a", "own_b"):
        plane = planes.get(scope) or {}
        if plane.get("status") != "ok":
            return {}, f"UNSOLVED_{scope.upper()}"
        output[scope] = plane
    return output, None


def geometry_dimension(points: np.ndarray) -> tuple[int, float, float]:
    values = np.asarray(points, dtype=float)
    centered = values - np.mean(values, axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    rank = int(np.sum(singular > max(1e-8, singular[0] * 1e-6))) if singular.size else 0
    line_ratio = float(singular[1] / singular[0]) if len(singular) > 1 and singular[0] > 0 else 0.0
    try:
        area = float(ConvexHull(values).volume) if len(values) >= 3 else 0.0
    except QhullError:
        area = 0.0
    return rank, line_ratio, area


def effective_rank(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(np.asarray(matrix, float), compute_uv=False)
    total = float(np.sum(singular ** 2))
    return float((np.sum(singular) ** 2) / total) if total > 0 else 0.0


def prefix_distance_spread(ranks: np.ndarray, contacts: np.ndarray, split: np.ndarray) -> tuple[float, float]:
    distance = np.linalg.norm(contacts[:, None, :] - contacts[None, :, :], axis=-1)
    samples: list[float] = []
    for event in np.flatnonzero(split == 0):
        prefix = np.flatnonzero((ranks[event] >= 0) & (ranks[event] <= 2))
        if not len(prefix):
            continue
        nearest = distance[:, prefix].min(axis=1)
        samples.extend(nearest.tolist())
    if not samples:
        return float("nan"), float("nan")
    return float(np.quantile(samples, 0.1)), float(np.quantile(samples, 0.9))


def build_fit(
    subject: str,
    scope: str,
    plane: dict,
    field: dict,
    dataset_path: Path,
    out: Path,
) -> dict:
    with np.load(dataset_path, allow_pickle=False) as data:
        names = [str(value) for value in data["contact_names"]]
        order = [str(value) for value in field["contact_order"]]
        points = np.asarray(plane["points"], dtype=float)
        if points.shape != (len(order), 2):
            raise RuntimeError(f"{subject}/{scope}: plane/contact-order mismatch")
        finite = np.isfinite(points).all(axis=1)
        by_name = {
            order[index]: points[index] * float(plane["scale_mm"])
            for index in range(len(order)) if finite[index]
        }
        joint = [name for name in names if name in by_name]
        if len(joint) < MIN_JOINT_CONTACTS:
            raise RuntimeError(f"{subject}/{scope}: only {len(joint)} joint contacts")
        columns = np.asarray([names.index(name) for name in joint], dtype=int)
        # v0.3 constructed the full-tissue mesh from the float32 contact cache.
        # Quantise here before geometry generation so the 31 overlapping fits
        # reproduce the frozen nodes/H bit for bit.
        contacts = np.stack([by_name[name] for name in joint]).astype(np.float32).astype(float)
        ranks = densify_groups(np.asarray(data["event_group_ids"])[:, columns])
        lag = np.asarray(data["event_lag_raw"], dtype=np.float32)[:, columns]
        group_count = np.asarray([
            len(np.unique(event[event >= 0])) for event in ranks
        ], dtype=np.int16)
        split = development_split(np.asarray(data["event_split"]))
        rank_eligible = group_count >= 2
        split[~rank_eligible] = -1
        # Preserve the frozen v0.3 geometry exactly for the 31 overlapping
        # fits.  A 2-mm floor or a new node seed would silently turn the
        # follow-up into a different latent-domain experiment.
        fit_id = f"{subject}__{scope}"
        old_plane_path = OLD_CACHE_ROOT / fit_id / "plane.npz"
        if old_plane_path.exists():
            old_plane = np.load(old_plane_path, allow_pickle=False)
            if not np.array_equal(
                contacts.astype(np.float32), old_plane["contacts_xy_mm"]
            ):
                raise RuntimeError(f"{fit_id}: frozen v0.3 contact geometry changed")
            sigma = float(old_plane["sigma_mm"][0])
            nodes = np.asarray(old_plane["nodes_xy_mm"], dtype=float)
            H = np.asarray(old_plane["H"], dtype=float)
            distance = np.asarray(old_plane["D_mm"], dtype=float)
            geometry_source = "EXACT_FROZEN_V0_3"
            n_zero_h_nodes = int(np.sum(H.sum(axis=0) <= 1e-12))
            zero_h_fraction = float(np.mean(H.sum(axis=0) <= 1e-12))
        else:
            sigma = float(np.float32(kernel_sigma_mm(contacts, floor_mm=0.0)))
            layout = resolve_full_tissue_layout(contacts, sigma, seed=NODE_SEED)
            nodes = np.asarray(layout.nodes_xy, dtype=float)
            H = np.asarray(layout.H, dtype=float)
            distance = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
            geometry_source = "NEW_V0_5_USING_V0_3_RULE"
            n_zero_h_nodes = int(layout.n_zero_h_nodes)
            zero_h_fraction = float(layout.zero_h_fraction)
        pools = build_pool_contract(distance)
        supported = H.sum(axis=0) > 1e-12
        graph = strong_component_audit(pools.local_mask, supported=supported)
        valid = bool(
            graph["all_nodes_one_strong_component"]
            and graph["contact_supported_pairwise_reachability"] == 1.0
            and graph["minimum_in_degree"] >= 1
            and graph["minimum_out_degree"] >= 1
            and int(pools.extra_local_pool.sum()) >= int(pools.k_added)
            and int(pools.nonlocal_pool.sum()) >= int(pools.k_added)
            and n_zero_h_nodes >= 1
        )
        target = out / "cache" / fit_id
        target.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target / "plane.npz",
            contacts_xy_mm=contacts.astype(np.float32),
            nodes_xy_mm=nodes.astype(np.float32),
            H=H.astype(np.float32),
            D_mm=distance.astype(np.float32),
            sigma_mm=np.asarray([sigma], np.float32),
            scale_mm=np.asarray([float(plane["scale_mm"])], np.float32),
            latent_domain_version=np.asarray(["FULL_TISSUE_OFFSET_HULL_V0_5"]),
        )
        np.savez_compressed(
            target / "events_raw.npz",
            ranks=ranks.astype(np.int16),
            base_split=split.astype(np.int8),
            event_group_count=group_count,
            event_lag_raw=lag,
            event_abs_time=np.asarray(data["event_abs_time"], dtype=np.float64),
            event_source_index=np.asarray(data["event_source_index"], dtype=np.int64),
            event_dataset_index=np.arange(len(ranks), dtype=np.int64),
            contact_names=np.asarray(joint, dtype=str),
        )
    dim, line_ratio, hull_area = geometry_dimension(contacts)
    h_mass = H / np.maximum(H.sum(axis=1, keepdims=True), 1e-12)
    h_concentration = np.sum(h_mass ** 2, axis=1)
    q10, q90 = prefix_distance_spread(ranks, contacts, split)
    row = {
        "fit_id": fit_id,
        "subject": subject,
        "scope": scope,
        "n_joint_contacts": int(len(joint)),
        "joint_contacts": joint,
        "n_events": int(len(ranks)),
        "n_train": int(np.sum(split == 0)),
        "n_validation": int(np.sum(split == 1)),
        "n_test": int(np.sum(split == 2)),
        "n_nodes": int(len(nodes)),
        "n_zero_h_nodes": n_zero_h_nodes,
        "zero_h_fraction": zero_h_fraction,
        "geometry_source": geometry_source,
        "local_edges": int(pools.local_mask.sum()),
        "extra_local_pool_edges": int(pools.extra_local_pool.sum()),
        "nonlocal_pool_edges": int(pools.nonlocal_pool.sum()),
        "k_added": int(pools.k_added),
        "r_local_mm": float(pools.r_local_mm),
        "strongly_connected": bool(graph["all_nodes_one_strong_component"]),
        "contact_supported_pairwise_reachability": float(
            graph["contact_supported_pairwise_reachability"]
        ),
        "minimum_in_degree": int(graph["minimum_in_degree"]),
        "minimum_out_degree": int(graph["minimum_out_degree"]),
        "effective_rank_H": effective_rank(H),
        "median_H_support_concentration": float(np.median(h_concentration)),
        "contact_geometry_rank": dim,
        "contact_second_to_first_singular_ratio": line_ratio,
        "contact_convex_hull_area_mm2": hull_area,
        "geometry_class": "TWO_DIMENSIONAL" if dim >= 2 and line_ratio >= 0.05 else "DEGENERATE_ONE_DIMENSIONAL",
        "prefix_distance_q10_mm": q10,
        "prefix_distance_q90_mm": q90,
        "prefix_distance_spread_mm": q90 - q10,
        "exact_spatial_permutation_count": str(math.factorial(len(joint))),
        "dataset_sha256": sha256(dataset_path),
        "field_sha256": sha256(FIELD_ROOT / f"{subject}.json"),
        "plane_sha256": sha256(target / "plane.npz"),
        "events_raw_sha256": sha256(target / "events_raw.npz"),
        "target_values_read": False,
        "valid": valid,
    }
    write_json(target / "provenance.json", row)
    return row


def copy_routing_metadata(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    routing = pd.read_csv(FIG3_EVENT_TABLE, usecols=list(ROUTING_COLUMNS))
    routing = routing[routing.group_id.eq("all_phenotype_matched")].copy()
    routing = routing.sort_values(["subject", "seizure_idx"]).reset_index(drop=True)
    if routing.subject.nunique() != 17 or len(routing) != 167:
        raise RuntimeError("Figure 3 routing denominator is not 17 patients/167 seizures")
    path = out / "EARLY_ICTAL_ROUTING_METADATA.csv"
    routing.to_csv(path, index=False)
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "patients": int(routing.subject.nunique()),
        "seizures": int(len(routing)),
        "columns_deserialized": list(ROUTING_COLUMNS),
        "target_numeric_values_read": False,
    }


def render_stage_a(fits: pd.DataFrame, patients: pd.DataFrame, out: Path) -> None:
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelsize": 12, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    red, blue, grey = "#b23b45", "#3274a1", "#aeb4b8"
    figure, axes = plt.subplots(1, 3, figsize=(10.6, 3.2), gridspec_kw={"wspace": 0.48})

    axis = axes[0]
    axis.bar([0, 1], [21, patients.included.sum()], color=[grey, blue], width=0.62)
    axis.set_xticks([0, 1], ["v0.3", "v0.5"])
    axis.set_ylabel("Spatial patients")
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes[1]
    order = fits.sort_values(["n_joint_contacts", "fit_id"]).reset_index(drop=True)
    colors = np.where(order.geometry_class.eq("TWO_DIMENSIONAL"), blue, red)
    axis.scatter(np.arange(len(order)), order.n_joint_contacts, c=colors, s=24)
    axis.axhline(MIN_JOINT_CONTACTS, color="#333333", lw=0.8, ls="--")
    axis.set_xlabel("Spatial fits")
    axis.set_ylabel("Joint contacts")
    axis.set_xticks([])
    axis.spines[["top", "right"]].set_visible(False)

    axis = axes[2]
    fit_colors = np.where(fits.geometry_class.eq("TWO_DIMENSIONAL"), blue, red)
    axis.scatter(
        fits.effective_rank_H, fits.zero_h_fraction,
        c=fit_colors, s=25, alpha=0.9,
    )
    axis.set_xlabel("Effective rank of H")
    axis.set_ylabel("Unobserved latent nodes")
    axis.spines[["top", "right"]].set_visible(False)

    for label, axis in zip("ABC", axes):
        axis.text(-0.18, 1.12, label, transform=axis.transAxes, fontsize=13,
                  fontweight="bold", va="top")
    stem = figures / "stage_a_v0_5_full_parent_spatial_census"
    figure.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)
    readme = figures / "README.md"
    section = (
        "### stage_a_v0_5_full_parent_spatial_census.png\n\n"
        "A 对比旧手工窄队列与按统一规则自动扫描全部 34 位 K=2 患者后的 spatial 分母。"
        "B 显示每个 fit 的 exact joint-contact 数，红色表示接近一维的单杆/退化几何。"
        "C 同时显示 contact readout H 的可观测秩和组织平面中无法被任何 contact 直接读取的 latent-node 比例。\n\n"
        "**关注点**：本图只使用冻结间期数据与 routing metadata；没有读取 early-ictal energy。\n"
    )
    existing = readme.read_text() if readme.exists() else ""
    marker = "### stage_a_v0_5_full_parent_spatial_census.png"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n"
    readme.write_text((existing.rstrip() + "\n\n" + section).lstrip())


def main() -> None:
    global FIELD_ROOT, FIG3_EVENT_TABLE
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--field-root", type=Path, default=FIELD_ROOT)
    parser.add_argument("--fig3-events", type=Path, default=FIG3_EVENT_TABLE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    FIELD_ROOT = args.field_root.resolve()
    FIG3_EVENT_TABLE = args.fig3_events.resolve()
    dataset = args.dataset_root.resolve()
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    subjects = list(map(str, manifest.get("cohort_subjects", [])))
    if len(subjects) != 34:
        raise RuntimeError(f"expected 34 frozen K=2 patients, found {len(subjects)}")

    fit_rows: list[dict] = []
    patient_rows: list[dict] = []
    for subject in subjects:
        dataset_path = dataset / "per_subject" / f"{subject}.npz"
        field_path = FIELD_ROOT / f"{subject}.json"
        row = {"subject": subject, "included": False, "n_fits": 0, "reason": ""}
        if not field_path.exists():
            row["reason"] = "FIELD_FILE_MISSING"
            patient_rows.append(row)
            continue
        field = json.loads(field_path.read_text()).get("interictal_field", {})
        scopes, reason = solved_scopes(field)
        if reason:
            row["reason"] = reason
            patient_rows.append(row)
            continue
        built = []
        for scope, plane in scopes.items():
            try:
                built.append(build_fit(subject, scope, plane, field, dataset_path, out))
            except RuntimeError as error:
                row["reason"] = str(error)
                built = []
                break
        if built and all(item["valid"] for item in built):
            fit_rows.extend(built)
            row.update(included=True, n_fits=len(built), reason="INCLUDED")
        elif built:
            row["reason"] = "GEOMETRY_OR_POOL_CONTRACT_FAILED"
        patient_rows.append(row)

    fits = pd.DataFrame(fit_rows).sort_values("fit_id").reset_index(drop=True)
    patients = pd.DataFrame(patient_rows).sort_values("subject").reset_index(drop=True)
    fits.to_csv(out / "FULL_PARENT_FIT_CENSUS.csv", index=False)
    patients.to_csv(out / "FULL_PARENT_PATIENT_ATTRITION.csv", index=False)
    routing = copy_routing_metadata(out)
    early_subjects = set(pd.read_csv(routing["path"]).subject.astype(str))
    spatial_subjects = set(fits.subject.astype(str))
    intersection = sorted(early_subjects & spatial_subjects)
    early_table = pd.read_csv(routing["path"])
    n_early = int(early_table.subject.isin(intersection).sum())

    cache_files = []
    for path in sorted((out / "cache").glob("*/*")):
        if path.is_file():
            cache_files.append({"path": str(path.resolve()), "sha256": sha256(path)})
    summary = {
        "contract": "topic5_multiscale_effective_scaffold_full_parent_cache_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
        "target_reader_imported": False,
        "parent_patients": 34,
        "spatial_patients": int(fits.subject.nunique()),
        "spatial_fits": int(fits.fit_id.nunique()),
        "excluded_patients": patients.loc[~patients.included, "subject"].tolist(),
        "degenerate_one_dimensional_patients": sorted(
            fits.loc[fits.geometry_class.eq("DEGENERATE_ONE_DIMENSIONAL"), "subject"].unique()
        ),
        "early_routing_parent": routing,
        "early_spatial_intersection_patients": len(intersection),
        "early_spatial_intersection_seizures": n_early,
        "early_spatial_intersection": intersection,
        "dataset_manifest_sha256": sha256(dataset / "dataset_manifest.json"),
        "cache_files": cache_files,
    }
    write_json(out / "FULL_PARENT_CACHE_MANIFEST.json", summary)
    render_stage_a(fits, patients, out)
    write_json(out / "STAGE_A_COMPLETE.json", {
        "status": "PASS",
        "target_values_read": False,
        "spatial_patients": summary["spatial_patients"],
        "spatial_fits": summary["spatial_fits"],
        "early_spatial_patients": len(intersection),
        "early_spatial_seizures": n_early,
    })


if __name__ == "__main__":
    main()
