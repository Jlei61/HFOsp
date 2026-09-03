"""rev22-DCI Task 4: freeze the branch-specific response design and the seed manifest.

Reads the Task 3 geometry-domain JSON (or a synthetic range for dry runs), chooses the
branch, generates the deterministic augmented maximin design, writes the candidate
manifest, the seed manifest, their SHA256 sidecars and the coverage figure. No simulation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.topic4_rev22_response_design import (  # noqa: E402
    PARAMS, REFERENCE, STATUS_PRIMARY, build_seed_manifest, design_rows_to_manifest,
    domain_from_geometry, generate_design, nearest_to_centre, point_table_sha256,
    synthetic_domain, validate_formal_geometry_contract,
)

DEFAULT_DOMAIN = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
                      "data_driven_dual_core_interictal_identifiability/geometry_domain/geometry_domain.json")
DEFAULT_OUT = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
                   "data_driven_dual_core_interictal_identifiability/response_design")
DEFAULT_REV20_MANIFEST = Path("/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/"
                              "data_driven_dual_core_mechanism_atlas/candidate_manifest.json")
AXIS_LABELS = {"g_LEE": "learned E→E pattern dose", "g_LEI": "learned E→I pattern dose",
               "theta_FT_deg": "fixed-topology long-axis angle (°)", "AR_FT": "fixed-topology aspect ratio"}
BLOCK_COLORS = {"reference": "#000000", "full4d": "#4C72B0", "lock_g_LEE": "#DD8452",
                "lock_g_LEI": "#55A868", "lock_theta": "#C44E52", "lock_AR": "#8172B3",
                "dose_plane": "#937860", "geometry_plane": "#DA8BC3"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(args: list[str]) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _write_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    digest = _sha256(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n")
    return digest


def _coverage_figure(out_dir: Path, rows: list[dict], bounds) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(combinations(range(4), 2))
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.6))
    blocks_present = []
    for row in rows:
        if row["block"] not in blocks_present:
            blocks_present.append(row["block"])
    for ax, (i, j) in zip(axes.ravel(), pairs):
        for block in blocks_present:
            if block == "reference":
                continue
            pts = np.asarray([[r["physical"][PARAMS[i]], r["physical"][PARAMS[j]]]
                              for r in rows if r["block"] == block], float)
            if len(pts):
                ax.scatter(pts[:, 0], pts[:, 1], s=18, color=BLOCK_COLORS[block], alpha=0.85,
                           linewidths=0, label=block)
        ax.scatter([REFERENCE[i]], [REFERENCE[j]], s=70, marker="*", color=BLOCK_COLORS["reference"],
                   zorder=5, label="reference")
        lo_i, hi_i = bounds[i]
        lo_j, hi_j = bounds[j]
        pad_i = 0.04 * (hi_i - lo_i if hi_i > lo_i else 1.0)
        pad_j = 0.04 * (hi_j - lo_j if hi_j > lo_j else 1.0)
        ax.set_xlim(lo_i - pad_i, hi_i + pad_i)
        ax.set_ylim(lo_j - pad_j, hi_j + pad_j)
        ax.set_xlabel(AXIS_LABELS[PARAMS[i]], fontsize=8)
        ax.set_ylabel(AXIS_LABELS[PARAMS[j]], fontsize=8)
        ax.tick_params(labelsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 8), fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    png = out_dir / "response_design_coverage.png"
    pdf = out_dir / "response_design_coverage.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def _write_readme(out_dir: Path, branch: str, n_points: int, bounds) -> None:
    if branch == STATUS_PRIMARY:
        body = (f"rev22 响应设计的六个二维投影，共 {n_points} 个候选点：一个参考点（星号）、"
                f"47 个四维空间填充点、四组各 8 个「锁一个坐标」的三维平面点、8 个剂量平面点和 8 个几何平面点。"
                f"几何域来自结构审计冻结的矩形：角度 {bounds[2][0]:g}°–{bounds[2][1]:g}°、"
                f"长短轴比 {bounds[3][0]:g}–{bounds[3][1]:g}。颜色区分设计块，每个块都以参考点为锚做最大最小距离填充。")
    else:
        body = (f"几何坐标被结构审计判为不可估计，退回剂量平面：共 {n_points} 个候选点，"
                f"一个参考点（星号）加 31 个在 learned E→E / E→I 剂量平面上的最大最小距离填充点，"
                f"角度和长短轴比固定在参考值。")
    text = ("### response_design_coverage.png\n\n" + body +
            "\n\n**关注点**：各投影里的点是否均匀铺满域、平面块是否严格落在参考值直线上、参考点是否只出现一次。\n")
    (out_dir / "README.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain-json", type=Path, default=DEFAULT_DOMAIN)
    parser.add_argument("--theta-range", type=float, nargs=2, default=None,
                        help="synthetic geometry range (dry run only; never writes to results/)")
    parser.add_argument("--ar-range", type=float, nargs=2, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--rev20-manifest", type=Path, default=DEFAULT_REV20_MANIFEST)
    parser.add_argument("--design-seed", type=int, default=20260905)
    args = parser.parse_args()

    synthetic = args.theta_range is not None or args.ar_range is not None
    if synthetic:
        if args.theta_range is None or args.ar_range is None:
            raise SystemExit("synthetic runs need both --theta-range and --ar-range")
        if str(args.out_dir.resolve()).startswith(str(DEFAULT_OUT.parents[1].resolve())):
            raise SystemExit("synthetic runs must not write into results/")
        domain_json = synthetic_domain(args.theta_range, args.ar_range)
        domain_source = {"kind": "synthetic", "sha256": None, "path": None}
    else:
        if not args.domain_json.is_file():
            raise SystemExit(f"geometry domain not found: {args.domain_json}")
        domain_json = json.loads(args.domain_json.read_text())
        domain_source = {"kind": "task3_geometry_domain", "sha256": _sha256(args.domain_json),
                         "path": str(args.domain_json)}

    rev20_manifest = json.loads(args.rev20_manifest.read_text())
    reference = [c for c in rev20_manifest["candidates"] if c.get("is_reference")][0]
    node_field = reference["node_field"]
    analysis = json.loads((ROOT / "config/topic4_rev22_dci_dual_core_interictal_identifiability.json").read_text())
    expected_reference = (float(analysis["reference"]["ellipse_angle_deg"]),
                          float(analysis["reference"]["ellipse_aspect_ratio"]))
    if not synthetic:
        rev20_config = ROOT / "config/topic4_rev20_dc_dual_core_mechanism_atlas.json"
        validate_formal_geometry_contract(
            domain_json, node_field_sha256=node_field.get("field_sha256"),
            rev20_config_sha256=_sha256(rev20_config), expected_reference=expected_reference,
        )
        grid_table = args.domain_json.with_name("geometry_audit_grid.csv")
        if not grid_table.is_file() or _sha256(grid_table) != domain_json.get("grid_table_sha256"):
            raise RuntimeError("geometry grid table is missing or changed")

    domain = domain_from_geometry(domain_json)
    design = generate_design(domain, seed=args.design_seed)
    node_mapping = {"node_gain": 1.0, "signed_depth_shrinkage": 1.0}
    rows = design_rows_to_manifest(design, node_field=node_field, node_mapping=node_mapping)
    table_sha = point_table_sha256(rows)
    reference_id = [r["candidate_id"] for r in rows if r["is_reference"]][0]
    decomposition_id = nearest_to_centre(rows)
    seed_manifest = build_seed_manifest(reference_id, decomposition_id)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    figure = _coverage_figure(out_dir / "figures", rows, design["bounds"])
    _write_readme(out_dir / "figures", design["branch"], len(rows), design["bounds"])
    manifest = {
        "schema_id": "topic4_rev22_dci_response_design_manifest_v1",
        "branch": design["branch"],
        "fallback_reason": domain["fallback_reason"],
        "geometry_status": domain["geometry_status"],
        "bounds": {name: design["bounds"][d] for d, name in enumerate(PARAMS)},
        "reference": {name: REFERENCE[d] for d, name in enumerate(PARAMS)},
        "geometry_reference": {
            "absolute_angle_deg": design["absolute_reference"]["angle_deg"],
            "absolute_aspect_ratio": design["absolute_reference"]["aspect_ratio"],
            "theta_FT_deg_is_offset_from_absolute_angle": True,
            "source": "rev22 amendment v5.1: registered patient axis (graph kernel long axis) and engine AR",
        },
        "design_seed": design["seed"],
        "regenerations": design["regenerations"],
        "design_quality": design["design_quality"],
        "candidate_count": len(rows),
        "block_counts": {b: sum(1 for r in rows if r["block"] == b) for b in dict.fromkeys(r["block"] for r in rows)},
        "point_table_sha256": table_sha,
        "domain_source": domain_source,
        "rev20_manifest_sha256": _sha256(args.rev20_manifest),
        "node_field_sha256": node_field.get("field_sha256"),
        "git_commit": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "decomposition_point_id": decomposition_id,
        "figure": figure,
        "candidates": rows,
        "claim_boundary": ("Design only: no simulation, no patient data; membership lists which nested "
                           "family subspaces contain each point."),
    }
    manifest_sha = _write_json(out_dir / "response_design_manifest.json", manifest)
    seed_manifest["response_design_manifest_sha256"] = manifest_sha
    seed_sha = _write_json(out_dir / "seed_manifest.json", seed_manifest)
    print(json.dumps({"branch": design["branch"], "candidates": len(rows),
                      "block_counts": manifest["block_counts"], "point_table_sha256": table_sha,
                      "response_design_manifest_sha256": manifest_sha, "seed_manifest_sha256": seed_sha,
                      "decomposition_point_id": decomposition_id, "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
