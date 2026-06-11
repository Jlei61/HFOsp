#!/usr/bin/env python3
"""Integrate propagation geometry outputs under Topic 3.

This is a packaging step only. It does not recompute scientific metrics.

Canonical output:
  results/spatial_modulation/propagation_geometry/

Inputs:
  results/topic4_sef_hfo/skeleton_geometry/
  results/propagation_entry_dispersion/

Rationale:
  The analysis is a spatial/where result: a real 3D propagation pathway and its
  per-event entry variability. SEF-HFO modeling consumes these numbers, but the
  empirical result belongs to Topic 3 rather than Topic 4.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "results/spatial_modulation/propagation_geometry"
CANON_PATH_AXIS_ROOT = OUT_ROOT / "components/path_axis"
CANON_ENTRY_ROOT = OUT_ROOT / "components/entry_variability"
LEGACY_PATH_AXIS_ROOT = ROOT / "results/topic4_sef_hfo/skeleton_geometry"
LEGACY_ENTRY_ROOT = ROOT / "results/propagation_entry_dispersion"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def copytree_refresh(src: Path, dst: Path) -> None:
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def choose_component_root(canonical: Path, legacy: Path) -> Path:
    """Prefer the new Topic 3 component root, fall back to legacy archive roots."""
    if (canonical / "cohort_summary.json").exists():
        return canonical
    return legacy


def finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and x == x


def compact_path_axis_record(d: dict[str, Any] | None) -> dict[str, Any]:
    if d is None:
        return {
            "status": "missing_or_excluded_from_path_axis",
            "eligibility_tier": None,
            "coord_space": None,
        }
    split = d.get("split_half_validation") or {}
    src_null = d.get("source_radius_null") or {}
    sink_null = d.get("sink_radius_null") or {}
    return {
        "status": d.get("status"),
        "eligibility_tier": d.get("eligibility_tier"),
        "coord_space": d.get("coord_space"),
        "n_eff": d.get("n_eff"),
        "k_used": d.get("k_used"),
        "swap_class": d.get("swap_class"),
        "template_source": d.get("template_source"),
        "clustering_provenance": d.get("clustering_provenance"),
        "axis_length_mm": d.get("axis_length_mm"),
        "weak_axis": d.get("weak_axis"),
        "sampling_geometry": d.get("sampling_geometry"),
        "source_radius": d.get("source_radius"),
        "sink_radius": d.get("sink_radius"),
        "perp_spread": d.get("perp_spread"),
        "perp_width_measurable": d.get("perp_width_measurable"),
        "axes_cos_angle": d.get("axes_cos_angle"),
        "heldout_spearman_rho": split.get("spearman_rho"),
        "heldout_kendall_tau": split.get("kendall_tau"),
        "heldout_rho_ci": [split.get("rho_ci_lo"), split.get("rho_ci_hi")],
        "heldout_n_channels": split.get("n_channels"),
        "source_radius_null_p": src_null.get("p_value"),
        "sink_radius_null_p": sink_null.get("p_value"),
        "phantom_core_violation": d.get("phantom_core_violation"),
    }


def compact_entry_record(d: dict[str, Any] | None) -> dict[str, Any]:
    if d is None:
        return {"status": "missing"}
    clusters = d.get("clusters", [])
    return {
        "status": "ok",
        "chosen_k": d.get("chosen_k"),
        "stable_k": d.get("stable_k"),
        "chosen_reason": d.get("chosen_reason"),
        "n_channels": d.get("n_channels"),
        "n_mapped_channels": d.get("n_mapped_channels"),
        "min_inter_cluster_corr": d.get("min_inter_cluster_corr"),
        "silhouette_chosen_k": d.get("silhouette_chosen_k"),
        "is_mixture": d.get("is_mixture"),
        "clusters": [
            {
                "cluster_id": c.get("cluster_id"),
                "n_events": c.get("n_events"),
                "fraction": c.get("fraction"),
                "neff": c.get("neff"),
                "top_share": c.get("top_share"),
                "verdict": entry_verdict(c),
                "p_neff_excess_pooled": (c.get("null") or {}).get("p_neff_excess"),
                "p_neff_excess_gauss": (c.get("null") or {}).get("p_neff_excess_gauss"),
                "p_neff_concentrated_pooled": (c.get("null") or {}).get("p_neff_concentrated"),
                "shape_spearman_prob_vs_trank": (c.get("shape") or {}).get("spearman_prob_vs_trank"),
                "downstream_cross_entry_rho": (c.get("downstream_invariance") or {}).get(
                    "median_cross_entry_spearman"
                ),
                "centered_tau_delta": (c.get("stability") or {}).get("centered_tau_delta"),
                "template_raw_tau": c.get("template_raw_tau"),
            }
            for c in clusters
        ],
    }


def entry_verdict(c: dict[str, Any]) -> str:
    n = c.get("null") or {}
    p_pool = n.get("p_neff_excess")
    p_gauss = n.get("p_neff_excess_gauss")
    p_con = n.get("p_neff_concentrated")
    if p_pool is not None and p_pool < 0.05 and p_gauss is not None and p_gauss < 0.05:
        return "robust_excess"
    if p_pool is not None and p_pool < 0.05:
        return "fragile_excess"
    if p_con is not None and p_con < 0.05:
        return "concentrated"
    return "consistent_one_template"


def build_subject_records(
    path_axis_files: dict[str, Path], entry_files: dict[str, Path], entry_overlap: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_subject_rows: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    overlap_by_key = {
        f"{r.get('dataset')}_{r.get('subject')}": r
        for r in entry_overlap.get("subjects", [])
    }

    all_stems = sorted(set(path_axis_files) | set(entry_files))
    for stem in all_stems:
        path_axis_full = read_json(path_axis_files[stem]) if stem in path_axis_files else None
        entry_full = read_json(entry_files[stem]) if stem in entry_files else None
        src = path_axis_full or entry_full
        if src is None:
            continue
        dataset = src.get("dataset")
        subject = src.get("subject")
        key = f"{dataset}_{subject}"

        compact_path = compact_path_axis_record(path_axis_full)
        compact_entry = compact_entry_record(entry_full)
        overlap = overlap_by_key.get(key)

        full_record = {
            "schema_version": "topic3_propagation_geometry_subject_v1",
            "topic": "topic3_spatial_soz_modulation",
            "dataset": dataset,
            "subject": subject,
            "scientific_object": "3D propagation pathway geometry plus per-event entry variability",
            "path_axis": path_axis_full or {"status": "missing_or_excluded_from_path_axis"},
            "entry_variability": entry_full or {"status": "missing"},
            "entry_overlap": overlap or {"status": "missing"},
            "component_paths": {
                "path_axis": (
                    f"components/path_axis/per_subject/{stem}.json"
                    if stem in path_axis_files
                    else None
                ),
                "entry_variability": f"components/entry_variability/per_subject/{stem}.json"
                if stem in entry_files
                else None,
            },
        }
        per_subject_rows.append(full_record)

        compact = {
            "dataset": dataset,
            "subject": subject,
            "path_axis": compact_path,
            "entry_variability": compact_entry,
            "entry_overlap": overlap or {"status": "missing"},
            "per_subject_json": f"per_subject/{stem}.json",
        }
        compact_rows.append(compact)

    return per_subject_rows, compact_rows


def summarize_integrated(
    path_axis_summary: dict[str, Any],
    entry_summary: dict[str, Any],
    overlap_summary: dict[str, Any],
    subjects: list[dict[str, Any]],
) -> dict[str, Any]:
    axis_subjects = [
        s for s in subjects
        if s["path_axis"].get("eligibility_tier") in ("primary", "fallback")
        and s["path_axis"].get("status") == "ok"
    ]
    rhos = [
        s["path_axis"].get("heldout_spearman_rho")
        for s in axis_subjects
        if finite_number(s["path_axis"].get("heldout_spearman_rho"))
    ]
    source_p = [
        s["path_axis"].get("source_radius_null_p")
        for s in axis_subjects
        if finite_number(s["path_axis"].get("source_radius_null_p"))
    ]
    sink_p = [
        s["path_axis"].get("sink_radius_null_p")
        for s in axis_subjects
        if finite_number(s["path_axis"].get("sink_radius_null_p"))
    ]
    axes_cos = [
        s["path_axis"].get("axes_cos_angle")
        for s in axis_subjects
        if finite_number(s["path_axis"].get("axes_cos_angle"))
    ]
    entry_clusters = entry_summary.get("clusters", [])

    return {
        "schema_version": "topic3_propagation_geometry_cohort_v1",
        "topic": "topic3_spatial_soz_modulation",
        "status": "canonical_integrated_packaging",
        "scientific_scope": (
            "Empirical 3D geometry of interictal HFO propagation templates: "
            "path-axis validation, source/sink core geometry, and per-event "
            "entry variability. This is a spatial/where result; SEF-HFO "
            "modeling consumes it downstream but does not define the result."
        ),
        "component_sources": {
            "path_axis": "components/path_axis/",
            "entry_variability": "components/entry_variability/",
            "legacy_path_axis_root": "results/topic4_sef_hfo/skeleton_geometry/",
            "legacy_entry_root": "results/propagation_entry_dispersion/",
        },
        "n_subject_records": len(subjects),
        "path_axis": {
            "n_processed": path_axis_summary.get("n_processed"),
            "n_ok": path_axis_summary.get("n_ok"),
            "tiers": path_axis_summary.get("tiers"),
            "sampling_geometry": path_axis_summary.get("sampling_geometry"),
            "weak_axis": path_axis_summary.get("weak_axis"),
            "phantom_core_violations": path_axis_summary.get("phantom_core_violations"),
            "heldout_validation": {
                "n_axis_subjects": len(rhos),
                "median_spearman_rho": median(rhos) if rhos else None,
                "n_strong_rho_ge_0_7": sum(x >= 0.7 for x in rhos),
                "n_moderate_0_4_to_0_7": sum(0.4 <= x < 0.7 for x in rhos),
                "n_weak_lt_0_4": sum(x < 0.4 for x in rhos),
            },
            "core_compactness_null": {
                "n_source_p_lt_0_05": sum(x < 0.05 for x in source_p),
                "n_sink_p_lt_0_05": sum(x < 0.05 for x in sink_p),
                "n_tested_source": len(source_p),
                "n_tested_sink": len(sink_p),
            },
            "reverse_template_shared_axis": {
                "n_axes_with_cos": len(axes_cos),
                "n_same_axis_reverse_cos_le_minus_0_85": sum(x <= -0.85 for x in axes_cos),
                "median_cos": median(axes_cos) if axes_cos else None,
            },
        },
        "entry_variability": {
            "summary": entry_summary.get("summary"),
            "n_clusters": len(entry_clusters),
        },
        "entry_overlap": {
            "summary": overlap_summary.get("summary"),
        },
        "interpretation_guardrails": [
            "source/sink core is template-level mean-rank endpoint; entry leader is per-event earliest detected participant.",
            "Do not report entry leader as a physiological source or as PR-6 endpoint anchoring.",
            "Yuquan native RAS mm and Epilepsiae MNI152 mm are not pooled as point clouds.",
            "Existing component figures are audit/diagnostic figures, not paper-grade main figures.",
        ],
        "subjects": subjects,
    }


def write_csv(path: Path, subjects: list[dict[str, Any]]) -> None:
    cols = [
        "dataset",
        "subject",
        "axis_status",
        "axis_tier",
        "coord_space",
        "axis_length_mm",
        "heldout_spearman_rho",
        "weak_axis",
        "source_radius_null_p",
        "sink_radius_null_p",
        "entry_chosen_k",
        "entry_n_clusters",
        "entry_median_top_share",
        "entry_median_neff",
        "entry_overlap_jaccard",
        "entry_overlap_centroid_dist_mm",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for s in subjects:
            clusters = (s["entry_variability"] or {}).get("clusters") or []
            top = [c.get("top_share") for c in clusters if finite_number(c.get("top_share"))]
            neff = [c.get("neff") for c in clusters if finite_number(c.get("neff"))]
            ov = s.get("entry_overlap") or {}
            row = {
                "dataset": s["dataset"],
                "subject": s["subject"],
                "axis_status": s["path_axis"].get("status"),
                "axis_tier": s["path_axis"].get("eligibility_tier"),
                "coord_space": s["path_axis"].get("coord_space"),
                "axis_length_mm": s["path_axis"].get("axis_length_mm"),
                "heldout_spearman_rho": s["path_axis"].get("heldout_spearman_rho"),
                "weak_axis": s["path_axis"].get("weak_axis"),
                "source_radius_null_p": s["path_axis"].get("source_radius_null_p"),
                "sink_radius_null_p": s["path_axis"].get("sink_radius_null_p"),
                "entry_chosen_k": s["entry_variability"].get("chosen_k"),
                "entry_n_clusters": len(clusters),
                "entry_median_top_share": median(top) if top else None,
                "entry_median_neff": median(neff) if neff else None,
                "entry_overlap_jaccard": ov.get("jaccard"),
                "entry_overlap_centroid_dist_mm": ov.get("centroid_dist_mm"),
            }
            w.writerow(row)


def write_readmes() -> None:
    (OUT_ROOT / "README.md").write_text(
        "# Topic 3 Propagation Geometry\n\n"
        "Canonical integrated result root for the empirical spatial geometry of "
        "interictal HFO propagation templates.\n\n"
        "This directory intentionally moves the result under Topic 3 because the "
        "scientific object is a spatial/where result: 3D propagation axis, "
        "source/sink core geometry, and per-event entry variability. Topic 4 "
        "models may consume these numbers, but they are not the canonical owner "
        "of the empirical result.\n\n"
        "## Files\n\n"
        "- `cohort_summary.json`: integrated cohort schema.\n"
        "- `cohort_summary.csv`: subject-level compact table for plotting.\n"
        "- `per_subject/`: integrated per-subject JSON, retaining the full component records.\n"
        "- `components/path_axis/`: former `results/topic4_sef_hfo/skeleton_geometry/` output.\n"
        "- `components/entry_variability/`: former `results/propagation_entry_dispersion/` output.\n\n"
        "## Guardrail\n\n"
        "`source/sink core` means a template-level mean-rank endpoint. "
        "`entry leader` means a per-event earliest detected participant. "
        "Do not merge these terms in text or figures.\n",
        encoding="utf-8",
    )
    fig = OUT_ROOT / "figures"
    fig.mkdir(parents=True, exist_ok=True)
    (fig / "README.md").write_text(
        "# Propagation Geometry Figures\n\n"
        "Current figures are component-level audit/diagnostic figures copied from "
        "the two source analyses. They are not paper-grade main figures.\n\n"
        "- `components/path_axis/figures/`: path-axis skeleton, held-out path cards, scalar diagnostics.\n"
        "- `components/entry_variability/figures/`: entry-dispersion diagnostics and entry-overlap maps.\n\n"
        "The future paper figure should read the integrated `cohort_summary.json` "
        "and `per_subject/*.json`, not the legacy component roots.\n",
        encoding="utf-8",
    )


def main() -> None:
    path_axis_root = choose_component_root(CANON_PATH_AXIS_ROOT, LEGACY_PATH_AXIS_ROOT)
    entry_root = choose_component_root(CANON_ENTRY_ROOT, LEGACY_ENTRY_ROOT)
    if not path_axis_root.exists():
        raise FileNotFoundError(path_axis_root)
    if not entry_root.exists():
        raise FileNotFoundError(entry_root)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Copy component artifacts into the Topic 3 canonical tree. This preserves
    # audit provenance while making the new root self-contained.
    copytree_refresh(path_axis_root, CANON_PATH_AXIS_ROOT)
    copytree_refresh(entry_root, CANON_ENTRY_ROOT)

    path_axis_summary = read_json(CANON_PATH_AXIS_ROOT / "cohort_summary.json")
    entry_summary = read_json(CANON_ENTRY_ROOT / "cohort_summary.json")
    overlap_summary = read_json(CANON_ENTRY_ROOT / "entry_overlap_summary.json")
    path_axis_files = {p.stem: p for p in (CANON_PATH_AXIS_ROOT / "per_subject").glob("*.json")}
    entry_files = {p.stem: p for p in (CANON_ENTRY_ROOT / "per_subject").glob("*.json")}

    full_subjects, compact_subjects = build_subject_records(
        path_axis_files, entry_files, overlap_summary
    )
    for rec in full_subjects:
        stem = f"{rec['dataset']}_{rec['subject']}"
        write_json(OUT_ROOT / "per_subject" / f"{stem}.json", rec)

    integrated = summarize_integrated(
        path_axis_summary, entry_summary, overlap_summary, compact_subjects
    )
    write_json(OUT_ROOT / "cohort_summary.json", integrated)
    write_csv(OUT_ROOT / "cohort_summary.csv", compact_subjects)
    write_readmes()

    print(f"wrote integrated Topic 3 propagation geometry root: {OUT_ROOT}")
    print(f"subjects: {len(full_subjects)}")


if __name__ == "__main__":
    main()
