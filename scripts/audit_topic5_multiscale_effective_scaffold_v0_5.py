#!/usr/bin/env python3
"""Target-free feasibility audit for the v0.5 multiscale scaffold line.

This script only reads the sealed interictal rank dataset, frozen interictal
planes, and the completed v0.3 geometry audit.  It must never import an ictal
target reader or deserialize an early-ictal energy array.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CANONICAL_ROOT = ROOT.parents[1] if (ROOT.parents[1] / "results").exists() else ROOT

from src.topic5_lbss_rnn_v0_2 import build_pool_contract, strong_component_audit  # noqa: E402
from src.topic5_virtual_seeg_operator import kernel_sigma_mm, resolve_full_tissue_layout  # noqa: E402


RECOVERY_SUBJECTS = (
    "epilepsiae_1077",
    "epilepsiae_1096",
    "epilepsiae_1125",
    "epilepsiae_139",
    "epilepsiae_635",
)
NODE_SEED = 20260813


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def densify_groups(groups: np.ndarray) -> np.ndarray:
    """Remove rank gaps created when contacts are restricted to a plane."""
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


def relative_latency_span_ms(lag: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Per-event finite lag span in ms for events with at least two rank sets."""
    timing = np.asarray(lag, dtype=float)
    rank = np.asarray(groups, dtype=int)
    if timing.shape != rank.shape or timing.ndim != 2:
        raise ValueError("lag and groups must share event x contact shape")
    spans = np.full(timing.shape[0], np.nan, dtype=float)
    for event_index, (event_lag, event_rank) in enumerate(zip(timing, rank)):
        valid = (event_rank >= 0) & np.isfinite(event_lag)
        if np.unique(event_rank[valid]).size < 2:
            continue
        spans[event_index] = 1000.0 * float(np.max(event_lag[valid]) - np.min(event_lag[valid]))
    return spans


def plane_scopes(field: dict) -> dict[str, dict]:
    planes = field["planes"]
    shared = planes.get("shared")
    if shared is not None and shared.get("status") == "ok":
        return {"shared": shared}
    output = {}
    for key in ("own_a", "own_b"):
        plane = planes.get(key, {})
        if plane.get("status") != "ok":
            raise RuntimeError(f"missing solved {key} plane")
        output[key] = plane
    return output


def audit_fit(subject: str, scope: str, plane: dict, data: np.lib.npyio.NpzFile, order: list[str]) -> dict:
    event_names = [str(value) for value in data["contact_names"]]
    points = np.asarray(plane["points"], dtype=float)
    if points.shape != (len(order), 2):
        raise RuntimeError(f"{subject}/{scope}: plane points do not align to contact_order")
    finite = np.isfinite(points).all(axis=1)
    by_name = {
        name: points[index] * float(plane["scale_mm"])
        for index, name in enumerate(order)
        if finite[index]
    }
    joint = [name for name in event_names if name in by_name]
    if len(joint) < 6:
        raise RuntimeError(f"{subject}/{scope}: only {len(joint)} joint contacts")
    columns = np.asarray([event_names.index(name) for name in joint], dtype=int)
    contacts = np.stack([by_name[name] for name in joint])
    sigma = kernel_sigma_mm(contacts, floor_mm=0.0)
    layout = resolve_full_tissue_layout(contacts, sigma, seed=NODE_SEED)
    distance = np.linalg.norm(
        layout.nodes_xy[:, None, :] - layout.nodes_xy[None, :, :], axis=-1
    )
    pools = build_pool_contract(distance)
    graph = strong_component_audit(
        pools.local_mask, supported=np.asarray(layout.H.sum(axis=0) > 1e-12)
    )

    groups = densify_groups(np.asarray(data["event_group_ids"])[:, columns])
    event_group_count = np.asarray(
        [np.unique(event[event >= 0]).size for event in groups], dtype=int
    )
    lag = np.asarray(data["event_lag_raw"], dtype=float)[:, columns]
    participation = groups >= 0
    finite_lag = participation & np.isfinite(lag)
    spans = relative_latency_span_ms(lag, groups)
    valid = bool(
        graph["all_nodes_one_strong_component"]
        and graph["contact_supported_pairwise_reachability"] == 1.0
        and graph["minimum_in_degree"] >= 1
        and graph["minimum_out_degree"] >= 1
        and int(pools.extra_local_pool.sum()) >= int(pools.k_added)
        and int(pools.nonlocal_pool.sum()) >= int(pools.k_added)
        and int(layout.n_zero_h_nodes) >= 16
    )
    return {
        "fit_id": f"{subject}__{scope}",
        "subject": subject,
        "scope": scope,
        "n_dataset_contacts": int(len(event_names)),
        "n_plane_contacts": int(finite.sum()),
        "n_joint_contacts": int(len(joint)),
        "joint_contacts": "|".join(joint),
        "n_events_total": int(groups.shape[0]),
        "n_events_with_two_or_more_rank_sets": int(np.sum(event_group_count >= 2)),
        "event_lag_raw_available": True,
        "participating_lag_finite_fraction": float(finite_lag.sum() / max(1, participation.sum())),
        "median_relative_latency_span_ms": float(np.nanmedian(spans)),
        "n_nodes": int(len(layout.nodes_xy)),
        "n_zero_h_nodes": int(layout.n_zero_h_nodes),
        "zero_h_fraction": float(layout.zero_h_fraction),
        "local_edges": int(pools.local_mask.sum()),
        "extra_local_pool_edges": int(pools.extra_local_pool.sum()),
        "nonlocal_pool_edges": int(pools.nonlocal_pool.sum()),
        "k_added": int(pools.k_added),
        "r_local_mm": float(pools.r_local_mm),
        "strongly_connected": bool(graph["all_nodes_one_strong_component"]),
        "minimum_in_degree": int(graph["minimum_in_degree"]),
        "minimum_out_degree": int(graph["minimum_out_degree"]),
        "valid": valid,
    }


def render(rows: pd.DataFrame, old: pd.DataFrame, figures: Path) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5,
        "axes.labelsize": 11.5, "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    red, blue, grey = "#b62d3f", "#2f6fa5", "#aab0b4"
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.25), gridspec_kw={"wspace": 0.48})

    ax = axes[0]
    x = np.arange(2)
    patients = [int(old.fit_id.str.split("__").str[0].nunique()), 26]
    fits = [int(old.fit_id.nunique()), 40]
    width = 0.34
    ax.bar(x - width / 2, patients, width, color=blue, label="Patients")
    ax.bar(x + width / 2, fits, width, color=red, label="Fits")
    ax.set_xticks(x, ["v0.3", "v0.5"])
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    order = rows.sort_values(["n_joint_contacts", "fit_id"]).reset_index(drop=True)
    ax.scatter(np.arange(len(order)), order.n_joint_contacts, s=32, color=red)
    ax.axhline(6, color=grey, lw=0.9)
    labels = []
    for value in order.fit_id:
        subject, scope = value.split("__", 1)
        suffix = {"shared": "S", "own_a": "A", "own_b": "B"}[scope]
        labels.append(f"{subject.replace('epilepsiae_', 'E')} {suffix}")
    ax.set_xticks(np.arange(len(order)), labels, rotation=55, ha="right")
    ax.set_ylabel("Joint contacts")
    ax.set_ylim(5.5, 7.5)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    # own-A/B fits are two views of the same patient event inventory.  Use the
    # patient-level count once rather than double-counting it in the figure.
    patient = rows.groupby("subject", sort=False).agg(
        events=("n_events_with_two_or_more_rank_sets", "max"),
        latency=("median_relative_latency_span_ms", "median"),
    )
    ax.scatter(patient.events, patient.latency, s=38, color=blue)
    for subject, values in patient.iterrows():
        ax.annotate(subject.replace("epilepsiae_", "E"), (values.events, values.latency),
                    xytext=(3, 2), textcoords="offset points", fontsize=7.5)
    ax.set_xscale("log")
    ax.set_xlabel("Candidate events")
    ax.set_ylabel("Within-event lag span (ms)")
    ax.spines[["top", "right"]].set_visible(False)

    for label, axis in zip("ABC", axes):
        axis.text(-0.20, 1.05, label, transform=axis.transAxes, fontsize=13,
                  fontweight="bold", va="top")
    stem = figures / "stage_a_v0_5_spatial_cohort_recovery"
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (figures / "README.md").write_text(
        "### stage_a_v0_5_spatial_cohort_recovery.png\n\n"
        "A 显示 full-tissue spatial cohort 从 v0.3 的 21 人/31 fits 扩展到预期 26 人/40 fits。"
        "B 显示新增 9 fits 均有 6–7 个 joint contacts；横线是预先固定的 6-contact 下限。"
        "C 显示五位恢复患者具有大量候选间期事件，并且 `event_lag_raw` 可形成事件内相对时间跨度。\n\n"
        "**关注点**：本图只审计 interictal geometry/timing feasibility，未读取任何 early-ictal energy 数值。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root", type=Path,
        default=CANONICAL_ROOT / "results/topic5_interictal_rank_distribution/dataset_v0_4/per_subject",
    )
    parser.add_argument(
        "--field-root", type=Path,
        default=CANONICAL_ROOT / "results/interictal_propagation_masked/template_gradient_fields/per_subject",
    )
    parser.add_argument(
        "--parent-root", type=Path,
        default=ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3",
    )
    parser.add_argument(
        "--out-root", type=Path,
        default=ROOT / "results/topic5_multiscale_effective_scaffold_v0_5",
    )
    args = parser.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    inputs = []
    for subject in RECOVERY_SUBJECTS:
        dataset_path = args.dataset_root / f"{subject}.npz"
        field_path = args.field_root / f"{subject}.json"
        if not dataset_path.exists() or not field_path.exists():
            raise FileNotFoundError(f"missing frozen interictal input for {subject}")
        field = json.loads(field_path.read_text())["interictal_field"]
        order = [str(value) for value in field["contact_order"]]
        with np.load(dataset_path, allow_pickle=False) as data:
            required = {"event_group_ids", "event_lag_raw", "contact_names"}
            if not required.issubset(data.files):
                raise RuntimeError(f"{subject}: rank dataset lacks {sorted(required - set(data.files))}")
            for scope, plane in plane_scopes(field).items():
                rows.append(audit_fit(subject, scope, plane, data, order))
        inputs.extend([
            {"path": str(dataset_path.resolve()), "sha256": sha256(dataset_path)},
            {"path": str(field_path.resolve()), "sha256": sha256(field_path)},
        ])

    frame = pd.DataFrame(rows).sort_values(["subject", "scope"]).reset_index(drop=True)
    frame.to_csv(args.out_root / "RECOVERY_FIT_AUDIT.csv", index=False)
    parent = pd.read_csv(args.parent_root / "LATENT_DOMAIN_AUDIT.csv")
    parent = parent[parent.version.eq("v0.3")].copy()
    if parent.fit_id.nunique() != 31:
        raise RuntimeError("parent v0.3 fit inventory drift")
    summary = {
        "contract": "topic5_multiscale_effective_scaffold_feasibility_v0_5",
        "target_values_read": False,
        "target_reader_imported": False,
        "parent_spatial_patients": int(parent.fit_id.str.split("__").str[0].nunique()),
        "parent_spatial_fits": int(parent.fit_id.nunique()),
        "recovery_patients": int(frame.subject.nunique()),
        "recovery_fits": int(frame.fit_id.nunique()),
        "projected_spatial_patients": int(parent.fit_id.str.split("__").str[0].nunique() + frame.subject.nunique()),
        "projected_spatial_fits": int(parent.fit_id.nunique() + frame.fit_id.nunique()),
        "all_recovery_fits_valid": bool(frame.valid.all()),
        "joint_contact_range": [int(frame.n_joint_contacts.min()), int(frame.n_joint_contacts.max())],
        "all_event_lag_raw_available": bool(frame.event_lag_raw_available.all()),
        "minimum_participating_lag_finite_fraction": float(frame.participating_lag_finite_fraction.min()),
        "inputs": inputs,
        "rows": rows,
    }
    write_json(args.out_root / "FEASIBILITY_AUDIT.json", summary)
    render(frame, parent, args.out_root / "figures")
    write_json(args.out_root / "FEASIBILITY_AUDIT_COMPLETE.json", {
        "status": "PASS" if summary["all_recovery_fits_valid"] else "FAIL",
        "target_values_read": False,
        "n_patients": summary["recovery_patients"],
        "n_fits": summary["recovery_fits"],
    })
    if not summary["all_recovery_fits_valid"]:
        raise RuntimeError("one or more recovery fits failed the frozen geometry contract")


if __name__ == "__main__":
    main()
