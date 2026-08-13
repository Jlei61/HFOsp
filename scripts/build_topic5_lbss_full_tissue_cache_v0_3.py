#!/usr/bin/env python3
"""Build the target-free full-tissue LBSS v0.3 geometry/cache.

Only the latent plane changes.  Event arrays, contact order, splits and sigma are
copied from the frozen v0.2 cache.  No early-ictal artifact is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic5_lbss_rnn_v0_2 import build_pool_contract, strong_component_audit
from src.topic5_virtual_seeg_operator import SUPPORT_SIGMA, resolve_full_tissue_layout


DEFAULT_SOURCE = Path("results/topic5_lbss_rnn_v0_2")
DEFAULT_OUT = Path("results/topic5_lbss_full_tissue_rnn_v0_3")
NODE_SEED = 20260812


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def durable_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise RuntimeError(f"refusing to replace non-identical cache file: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def gap_crossing_fraction(
    contacts: np.ndarray,
    nodes: np.ndarray,
    local_mask: np.ndarray,
    sigma_mm: float,
) -> float:
    """Fraction of local edges whose segment leaves every direct contact footprint."""
    edges = np.argwhere(np.asarray(local_mask, bool))
    if not len(edges):
        return float("nan")
    alpha = np.linspace(0.0, 1.0, 21)[:, None]
    crossing = []
    for target, source in edges:
        segment = nodes[source][None, :] * (1.0 - alpha) + nodes[target][None, :] * alpha
        distance = np.linalg.norm(
            segment[:, None, :] - contacts[None, :, :], axis=-1
        ).min(axis=1)
        crossing.append(bool(distance.max() > SUPPORT_SIGMA * sigma_mm + 1e-6))
    return float(np.mean(crossing))


def fit_geometry_row(
    fit_id: str,
    contacts: np.ndarray,
    nodes: np.ndarray,
    H: np.ndarray,
    sigma: float,
    version: str,
) -> dict:
    distance = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)
    pools = build_pool_contract(distance)
    audit = strong_component_audit(pools.local_mask)
    support_count = (H > 0).sum(axis=0)
    local_lengths = distance[pools.local_mask.astype(bool)]
    return {
        "fit_id": fit_id,
        "version": version,
        "n_contacts": int(len(contacts)),
        "n_nodes": int(len(nodes)),
        "n_zero_h_nodes": int(np.sum(support_count == 0)),
        "zero_h_fraction": float(np.mean(support_count == 0)),
        "contact_private_node_fraction": float(np.mean(support_count == 1)),
        "min_nodes_per_contact": int((H > 0).sum(axis=1).min()),
        "median_nodes_per_contact": float(np.median((H > 0).sum(axis=1))),
        "local_edges": int(pools.local_mask.sum()),
        "local_k": int(pools.k_neighbors),
        "local_length_median_mm": float(np.median(local_lengths)),
        "local_length_p95_mm": float(np.quantile(local_lengths, 0.95)),
        "local_gap_crossing_fraction": gap_crossing_fraction(
            contacts, nodes, pools.local_mask, sigma
        ),
        "extra_local_pool_edges": int(pools.extra_local_pool.sum()),
        "nonlocal_pool_edges": int(pools.nonlocal_pool.sum()),
        "k_added": int(pools.k_added),
        "all_nodes_one_strong_component": bool(audit["all_nodes_one_strong_component"]),
    }


def plot_geometry_audit(
    table: pd.DataFrame,
    old_plane: dict[str, np.ndarray],
    new_plane: dict[str, np.ndarray],
    out_root: Path,
) -> None:
    figures = out_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 10.5,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(10.2, 3.25))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.08], wspace=0.42)
    colors = {"observed": "#3f7f93", "latent": "#c9ced1", "contact": "#171717"}

    for axis, plane, title in [
        (fig.add_subplot(grid[0, 0]), old_plane, "Contact-dilated"),
        (fig.add_subplot(grid[0, 1]), new_plane, "Full-tissue"),
    ]:
        H = plane["H"]
        observed = H.sum(axis=0) > 1e-12
        axis.scatter(
            plane["nodes"][:, 0], plane["nodes"][:, 1], s=9,
            c=np.where(observed, colors["observed"], colors["latent"]),
            edgecolors="none", zorder=1,
        )
        axis.scatter(
            plane["contacts"][:, 0], plane["contacts"][:, 1], s=31,
            facecolors="white", edgecolors=colors["contact"], linewidths=1.0, zorder=3,
        )
        axis.set_aspect("equal")
        axis.set_title(title)
        axis.set_xlabel("Propagation axis (mm)")
        axis.spines[["top", "right"]].set_visible(False)
    fig.axes[0].set_ylabel("Transverse axis (mm)")
    fig.axes[1].set_ylabel("")

    axis = fig.add_subplot(grid[0, 2])
    pivot = table.pivot(index="fit_id", columns="version", values="zero_h_fraction")
    x = np.array([0.0, 1.0])
    for _, row in pivot.iterrows():
        axis.plot(x, [row["v0.2"], row["v0.3"]], color="#aeb4b8", lw=0.65, alpha=0.7)
    axis.scatter(
        np.zeros(len(pivot)), pivot["v0.2"], s=17, color="#858b90", zorder=3,
    )
    axis.scatter(
        np.ones(len(pivot)), pivot["v0.3"], s=17, color="#3f7f93", zorder=3,
    )
    medians = pivot.median(axis=0)
    axis.plot(x, [medians["v0.2"], medians["v0.3"]], color="#171717", lw=2.1, zorder=4)
    axis.set_xlim(-0.35, 1.35)
    axis.set_ylim(-0.03, 1.0)
    axis.set_xticks(x, ["v0.2", "v0.3"])
    axis.set_ylabel("Unobserved latent nodes")
    axis.spines[["top", "right"]].set_visible(False)

    for label, axis in zip("ABC", fig.axes):
        axis.text(-0.18, 1.06, label, transform=axis.transAxes, fontsize=13,
                  fontweight="bold", va="top")
    fig.savefig(figures / "stage_a_b_latent_domain_correction.png", dpi=600,
                bbox_inches="tight", facecolor="white")
    fig.savefig(figures / "stage_a_b_latent_domain_correction.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    (figures / "README.md").write_text(
        "### stage_a_b_latent_domain_correction.png\n\n"
        "A–B 以 E1146 展示旧 contact-dilated domain 与 full-tissue latent domain。"
        "空心圆是 SEEG contacts，蓝色 nodes 可被 contact 直接读出，灰色 nodes 只能通过 recurrent propagation 参与。"
        "C 为 31 fits 的配对覆盖审计。\n\n"
        "**关注点**：v0.2 不存在真正未观测 latent state；v0.3 明确把 contact 恢复为局部读出端口。\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--representative-fit", default="epilepsiae_1146__shared")
    args = parser.parse_args()
    source = args.source_root.resolve()
    out = args.out_root.resolve()
    out.mkdir(parents=True, exist_ok=True)

    fit_dirs = sorted(
        path for path in (source / "cache").iterdir()
        if path.is_dir() and (path / "plane.npz").exists()
    )
    if len(fit_dirs) != 31:
        raise RuntimeError(f"expected 31 frozen physical-geometry fits, found {len(fit_dirs)}")

    rows: list[dict] = []
    fits: list[dict] = []
    cache_files: list[dict] = []
    representative_old = representative_new = None
    for source_fit in fit_dirs:
        fit_id = source_fit.name
        target_fit = out / "cache" / fit_id
        old = np.load(source_fit / "plane.npz", allow_pickle=False)
        contacts = np.asarray(old["contacts_xy_mm"], float)
        sigma = float(old["sigma_mm"][0])
        layout = resolve_full_tissue_layout(contacts, sigma, seed=NODE_SEED)
        nodes = layout.nodes_xy
        H = layout.H
        distance = np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=-1)

        target_fit.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            target_fit / "plane.npz",
            contacts_xy_mm=contacts.astype(np.float32),
            nodes_xy_mm=nodes.astype(np.float32),
            H=H.astype(np.float32),
            D_mm=distance.astype(np.float32),
            sigma_mm=np.asarray(old["sigma_mm"], np.float32),
            scale_mm=np.asarray(old["scale_mm"], np.float32),
            latent_domain_version=np.asarray(["FULL_TISSUE_OFFSET_HULL_V0_3"]),
        )
        durable_link(source_fit / "events.npz", target_fit / "events.npz")
        old_provenance = json.loads((source_fit / "provenance.json").read_text())
        provenance = {
            **old_provenance,
            "latent_domain_version": "FULL_TISSUE_OFFSET_HULL_V0_3",
            "source_contact_dilated_plane_sha256": sha256(source_fit / "plane.npz"),
            "source_events_sha256": sha256(source_fit / "events.npz"),
            "n_nodes_v0_2": int(old["nodes_xy_mm"].shape[0]),
            "n_nodes": int(len(nodes)),
            "node_seed": NODE_SEED,
            "domain_area_mm2": layout.domain_area_mm2,
            "domain_margin_mm": layout.domain_margin_mm,
            "background_spacing_mm": layout.background_spacing_mm,
            "candidate_step_mm": layout.candidate_step_mm,
            "n_background_nodes": layout.n_background_nodes,
            "n_support_nodes_added": layout.n_support_nodes_added,
            "n_zero_h_nodes": layout.n_zero_h_nodes,
            "zero_h_fraction": layout.zero_h_fraction,
            "contact_pitch_mm": layout.contact_pitch_mm,
            "envelope_kind": layout.envelope_kind,
            "target_values_read": False,
        }
        write_json(target_fit / "provenance.json", provenance)
        for name in ("plane.npz", "events.npz", "provenance.json"):
            cache_files.append({
                "fit_id": fit_id,
                "file": name,
                "local": str((target_fit / name).resolve()),
                "sha256": sha256(target_fit / name),
            })

        rows.append(fit_geometry_row(
            fit_id, contacts, np.asarray(old["nodes_xy_mm"], float),
            np.asarray(old["H"], float), sigma, "v0.2"
        ))
        new_row = fit_geometry_row(fit_id, contacts, nodes, H, sigma, "v0.3")
        new_row.update({
            "domain_area_mm2": layout.domain_area_mm2,
            "n_background_nodes": layout.n_background_nodes,
            "n_support_nodes_added": layout.n_support_nodes_added,
        })
        rows.append(new_row)
        fits.append(provenance)
        if fit_id == args.representative_fit:
            representative_old = {
                "contacts": contacts,
                "nodes": np.asarray(old["nodes_xy_mm"], float),
                "H": np.asarray(old["H"], float),
            }
            representative_new = {"contacts": contacts, "nodes": nodes, "H": H}

    table = pd.DataFrame(rows)
    table.to_csv(out / "LATENT_DOMAIN_AUDIT.csv", index=False)
    new = table[table.version == "v0.3"]
    manifest = {
        "contract": "topic5_lbss_full_tissue_cache_v0_3",
        "source_root": str(source),
        "source_status": "CONTACT_DILATED_DOMAIN_SENSITIVITY",
        "target_values_read": False,
        "node_seed": NODE_SEED,
        "n_patients": int(len({fit["subject"] for fit in fits})),
        "n_fits": len(fits),
        "zero_h_nodes_min": int(new.n_zero_h_nodes.min()),
        "zero_h_fraction_median": float(new.zero_h_fraction.median()),
        "all_graphs_strongly_connected": bool(new.all_nodes_one_strong_component.all()),
        "fits": fits,
        "audit_sha256": sha256(out / "LATENT_DOMAIN_AUDIT.csv"),
    }
    write_json(out / "INPUT_MANIFEST.json", manifest)
    write_json(out / "INPUT_CACHE_MANIFEST.json", {
        "contract": "topic5_lbss_full_tissue_input_cache_manifest_v0_3",
        "source_root": str(source),
        "target_values_read": False,
        "files": cache_files,
    })
    spec_path = ROOT / "docs/superpowers/specs/2026-08-12-topic5-lbss-full-tissue-rnn-v0-3-design.md"
    plan_path = ROOT / "docs/superpowers/plans/2026-08-12-topic5-lbss-full-tissue-rnn-v0-3.md"
    write_json(out / "RUN_CONTRACT.json", {
        "contract": "topic5_lbss_full_tissue_rnn_v0_3",
        "geometry_status": "RETROSPECTIVE_TEST_INFORMED_PROPAGATION_PLANE",
        "latent_domain": "OFFSET_CONVEX_HULL_WITH_EXPLICIT_ZERO_H_STATE",
        "edge_time_status": "ORDINAL_NO_PHYSICAL_DELAY",
        "target_values_read": False,
        "n_patients": manifest["n_patients"],
        "n_fits": manifest["n_fits"],
        "arms": [
            "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
            "L2_LOCAL_PLUS_RANDOM_LR", "L3_LOCAL_PLUS_LEARNED_LR",
            "C_L3_ORDER_SHUFFLED",
        ],
        "spec_sha256": sha256(spec_path),
        "plan_sha256": sha256(plan_path),
        "input_manifest_sha256": sha256(out / "INPUT_CACHE_MANIFEST.json"),
        "target_access_count": 0,
    })
    if representative_old is None or representative_new is None:
        raise RuntimeError("representative E1146 fit is missing")
    plot_geometry_audit(table, representative_old, representative_new, out)
    write_json(out / "FULL_TISSUE_CACHE_COMPLETE.json", {
        "status": "PASS",
        "n_fits": len(fits),
        "n_patients": manifest["n_patients"],
        "target_values_read": False,
        "manifest_sha256": sha256(out / "INPUT_MANIFEST.json"),
    })
    print(
        f"built {len(fits)} full-tissue fits; median nodes={int(new.n_nodes.median())}; "
        f"median zero-H={new.zero_h_fraction.median():.3f}"
    )


if __name__ == "__main__":
    main()
