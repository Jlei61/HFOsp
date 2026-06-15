#!/usr/bin/env python3
"""Build the Axis-A A5-real per-subject feature-coverage inventory.

Turns the prose "covered X%" coverage statement in the Axis-A propagation
plan into a reproducible per-subject artifact. Descriptive only: this script
emits which feature layers (coord-free rank template / mm propagation axis /
pathway width) are *available* per subject. It carries NO mechanism label and
makes NO scientific claim — it is a coverage inventory, nothing else.

Inputs (read-only):
  results/spatial_modulation/propagation_geometry/cohort_summary.json
      -> per-subject path_axis (status, weak_axis, axis_length_mm, coord_space,
         perp_width_measurable, sampling_geometry); also cohort rollups for
         reconciliation anchors.
  results/interictal_propagation_masked/per_subject/<dataset>_<subject>.json
      -> coord-free rank-space template layer. rank_available is true wherever
         a masked adaptive_cluster / propagation_stereotypy exists. This layer
         survives even when 3D coords are missing.

Outputs (the artifact):
  results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/fingerprint/
      real_feature_coverage.csv
      real_feature_coverage.json

Derivation contract (per AGENTS.md / plan field map):
  - rank_available           : a masked adaptive_cluster (or propagation_stereotypy)
                               exists for the subject in interictal_propagation_masked.
  - axis_available           : path_axis status == 'ok' AND not degenerate_axis,
                               where degenerate_axis == (axis_length_mm is None)
                               (the 4 descriptive_only subjects have no usable mm
                               vector). This is the mm propagation-axis layer.
  - pathway_width_available  : perp_width_measurable AND coords present. 1D single
                               shaft cannot measure perpendicular spread.
  - mm-scale coverage is reported PER DATASET (epi=mni152_1mm, yuquan=fs_native_ras_mm);
    mm values are never pooled across datasets.

pengzihang is excluded from path_axis upstream
('missing_or_excluded_from_path_axis'); it is kept as a row with exclusion_reason
set, and stays rank_available (its masked template exists).
"""

import csv
import json
import os
from collections import OrderedDict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COHORT_SUMMARY = os.path.join(
    REPO,
    "results/spatial_modulation/propagation_geometry/cohort_summary.json",
)
MASKED_DIR = os.path.join(
    REPO, "results/interictal_propagation_masked/per_subject"
)
OUT_DIR = os.path.join(
    REPO,
    "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/fingerprint",
)

# --- INTERIM broad-lagPat patch (2026-06-15) -------------------------------
# huangwanling + zhaojinrui have a 4-channel SINGLE-shaft narrow montage
# (H2-H5 / F5-F8) -> degenerate propagation axis -> axis_available=false. Their
# broad-lagPat re-derivation (top_n=20, multi-shaft; built by
# scripts/build_broad_lagpat_patch.py) yields a usable multi-shaft axis. Until
# the full-cohort broad re-derivation runs, overlay the broad path_axis
# COMPONENT record for exactly these two subjects. The broad component rec is
# itself the full path_axis dict (same field names this audit reads), so it
# drops straight into `pa`. Remove this block (and re-run) once the full broad
# cohort lands.
BROAD_PATCH = {("yuquan", "huangwanling"), ("yuquan", "zhaojinrui")}
BROAD_GEO_COMPONENT = os.path.join(
    REPO,
    "results/spatial_modulation/propagation_geometry_broad/"
    "components/path_axis/per_subject",
)


def broad_path_axis(dataset, subject):
    """Broad path_axis component rec for a patched subject, else None."""
    if (dataset, subject) not in BROAD_PATCH:
        return None
    path = os.path.join(BROAD_GEO_COMPONENT, f"{dataset}_{subject}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


# Known reconciliation anchors from the Axis-A A5-real coverage prose.
ANCHORS = {
    "rank_available": 40,
    "axis_available": 26,
    "pathway_width_available": 23,
    "weak_axis": 6,
    "entry_overlap_subjects": 26,
    "path_axis_processed": 39,  # 40 minus pengzihang (excluded upstream)
}


def load_rank_availability():
    """Map (dataset, subject) -> rank_available from the masked rank-space layer.

    Only real <dataset>_<subject>.json files count; pr5a/pr5b subdirs and any
    non-subject entry are skipped.
    """
    rank = {}
    for fn in sorted(os.listdir(MASKED_DIR)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(MASKED_DIR, fn)
        if not os.path.isfile(path):
            continue
        stem = fn[:-len(".json")]
        if "_" not in stem:
            continue
        dataset, subject = stem.split("_", 1)
        if dataset not in ("epilepsiae", "yuquan"):
            continue
        with open(path) as f:
            d = json.load(f)
        # Coord-free rank template exists if a masked adaptive_cluster or the
        # propagation_stereotypy block is present (survives missing coords).
        rank_available = (
            d.get("adaptive_cluster") is not None
            or d.get("propagation_stereotypy") is not None
        )
        rank[(dataset, subject)] = bool(rank_available)
    return rank


def build_rows():
    with open(COHORT_SUMMARY) as f:
        cohort = json.load(f)

    rank_avail_map = load_rank_availability()

    rows = []
    for s in cohort["subjects"]:
        dataset = s["dataset"]
        subject = s["subject"]
        pa = s.get("path_axis", {}) or {}
        broad_pa = broad_path_axis(dataset, subject)
        is_broad_patched = broad_pa is not None
        if is_broad_patched:
            pa = broad_pa
        status = pa.get("status")

        is_ok = status == "ok"
        # coord-missing: yuquan 'error: ... coord file not found' or excluded.
        coord_space = pa.get("coord_space")
        coords_present = coord_space not in (None, "missing")
        coord_space_out = coord_space if coords_present else "missing"

        weak_axis = pa.get("weak_axis") is True
        # No explicit degenerate_axis field upstream: the operational signal for
        # "ok status but no usable mm axis vector" is axis_length_mm is None,
        # which coincides exactly with the descriptive_only tier.
        degenerate_axis = is_ok and (pa.get("axis_length_mm") is None)

        rank_available = rank_avail_map.get((dataset, subject), False)
        axis_available = is_ok and not degenerate_axis
        perp_measurable = pa.get("perp_width_measurable") is True
        pathway_width_available = perp_measurable and coords_present

        sg = pa.get("sampling_geometry")
        if isinstance(sg, dict):
            sampling_geometry = sg.get("geometry") or "none"
            n_shafts = sg.get("n_shafts")
        else:
            sampling_geometry = "none"
            n_shafts = None

        # exclusion_reason: '' if fully usable (all three layers), else the most
        # specific reason this subject loses the mm / width layers.
        fully_usable = (
            rank_available and axis_available and pathway_width_available
        )
        if fully_usable:
            exclusion_reason = ""
        elif status == "missing_or_excluded_from_path_axis":
            exclusion_reason = "missing_or_excluded_from_path_axis"
        elif not coords_present:
            exclusion_reason = "coord file not found"
        elif degenerate_axis:
            exclusion_reason = "degenerate_axis"
        elif sampling_geometry == "1D" and not perp_measurable:
            exclusion_reason = "1D single shaft (perp unmeasurable)"
        elif weak_axis:
            exclusion_reason = "weak_axis"
        elif not pathway_width_available:
            exclusion_reason = "perp width unmeasurable"
        else:
            exclusion_reason = ""

        rows.append(OrderedDict([
            ("dataset", dataset),
            ("subject", subject),
            ("rank_available", rank_available),
            ("axis_available", axis_available),
            ("pathway_width_available", pathway_width_available),
            ("coord_space", coord_space_out),
            ("weak_axis", weak_axis),
            ("degenerate_axis", degenerate_axis),
            ("sampling_geometry", sampling_geometry),
            ("n_shafts", n_shafts),
            ("exclusion_reason", exclusion_reason),
            ("broad_patched", is_broad_patched),
        ]))

    rows.sort(key=lambda r: (r["dataset"], r["subject"]))
    return rows


def feature_rollup(rows, feature):
    """Per-feature available count, split by dataset and by sampling_geometry.

    Reported on the feature's OWN availability mask — never one global cohort.
    """
    by_dataset = {}
    by_geometry = {}
    total_available = 0
    for r in rows:
        if not r[feature]:
            continue
        total_available += 1
        by_dataset[r["dataset"]] = by_dataset.get(r["dataset"], 0) + 1
        g = r["sampling_geometry"]
        by_geometry[g] = by_geometry.get(g, 0) + 1
    return OrderedDict([
        ("n_available", total_available),
        ("n_total_subjects", len(rows)),
        ("by_dataset", by_dataset),
        ("by_sampling_geometry", by_geometry),
    ])


def build_reconciliation(rows):
    computed = {
        "rank_available": sum(1 for r in rows if r["rank_available"]),
        "axis_available": sum(1 for r in rows if r["axis_available"]),
        "pathway_width_available": sum(
            1 for r in rows if r["pathway_width_available"]
        ),
        "weak_axis": sum(1 for r in rows if r["weak_axis"]),
        "path_axis_processed": sum(
            1 for r in rows
            if r["exclusion_reason"] != "missing_or_excluded_from_path_axis"
        ),
    }
    # Broad-patch delta: the anchors above are the pre-patch narrow-cohort prose
    # numbers. The interim broad patch flips exactly the patched subjects, so the
    # rigorous check is computed == narrow_anchor + (patched rows now True). This
    # confirms ONLY the intended subjects changed and nothing else drifted.
    patch_delta = {
        "axis_available": sum(
            1 for r in rows if r["broad_patched"] and r["axis_available"]),
        "pathway_width_available": sum(
            1 for r in rows if r["broad_patched"] and r["pathway_width_available"]),
    }
    recon = OrderedDict()
    for key, anchor in ANCHORS.items():
        if key == "entry_overlap_subjects":
            # External anchor (entry_overlap_summary.csv = 26 rows); not a
            # column we recompute here, recorded as a note for traceability.
            recon[key] = OrderedDict([
                ("computed", None),
                ("anchor", anchor),
                ("match", None),
                ("note",
                 "external: entry_overlap_summary.csv has 26 subject rows; "
                 "spatial-onset overlap layer is not a column in this artifact"),
            ])
            continue
        comp = computed.get(key)
        delta = patch_delta.get(key, 0)
        recon[key] = OrderedDict([
            ("computed", comp),
            ("anchor", anchor),
            ("broad_patch_delta", delta),
            ("match", comp == anchor + delta),
        ])
    recon["rank_vs_path_axis_note"] = (
        "interictal_masked rank layer = 40 subjects vs path_axis processed = 39 "
        "because pengzihang (yuquan) is excluded from path_axis "
        "(missing_or_excluded_from_path_axis) but still has a masked rank "
        "template, so it stays rank_available."
    )
    return recon


def main():
    rows = build_rows()
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "figures"), exist_ok=True)

    # CSV
    csv_path = os.path.join(OUT_DIR, "real_feature_coverage.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = dict(r)
            for k, v in out.items():
                if isinstance(v, bool):
                    out[k] = "true" if v else "false"
                elif v is None:
                    out[k] = ""
            w.writerow(out)

    # JSON
    payload = OrderedDict([
        ("schema_version", "topic4_sef_hfo_axisA_real_feature_coverage_v1"),
        ("topic", "topic4_sef_hfo"),
        ("scope",
         "Axis-A A5-real per-subject feature-coverage inventory. Descriptive "
         "coverage only: which feature layers are available per subject. NO "
         "mechanism label, NO scientific claim."),
        ("feature_definitions", OrderedDict([
            ("rank_available",
             "coord-free rank-space template exists in "
             "interictal_propagation_masked (adaptive_cluster / "
             "propagation_stereotypy); survives missing 3D coords."),
            ("axis_available",
             "mm propagation-axis vector usable: path_axis status ok AND not "
             "degenerate_axis (degenerate_axis == axis_length_mm is None == "
             "descriptive_only tier)."),
            ("pathway_width_available",
             "perp_width_measurable AND coords present; a 1D single shaft "
             "cannot measure perpendicular spread."),
        ])),
        ("mm_coverage_discipline",
         "mm-scale coverage is per-dataset: epilepsiae=mni152_1mm, "
         "yuquan=fs_native_ras_mm. mm values are NEVER pooled across datasets."),
        ("n_subjects", len(rows)),
        ("per_feature_coverage", OrderedDict([
            ("rank_available", feature_rollup(rows, "rank_available")),
            ("axis_available", feature_rollup(rows, "axis_available")),
            ("pathway_width_available",
             feature_rollup(rows, "pathway_width_available")),
        ])),
        ("weak_axis_count", sum(1 for r in rows if r["weak_axis"])),
        ("degenerate_axis_count",
         sum(1 for r in rows if r["degenerate_axis"])),
        ("broad_patch", OrderedDict([
            ("status",
             "INTERIM: broad-lagPat path_axis overlaid for single-shaft "
             "4-channel subjects; superseded when the full-cohort broad "
             "re-derivation lands"),
            ("subjects",
             sorted(f"{d}:{s}" for (d, s) in BROAD_PATCH)),
            ("reason",
             "narrow montage = 4ch on ONE shaft -> degenerate axis "
             "(axis_available=false). Broad lagPat (top_n=20, multi-shaft) "
             "yields a usable axis."),
            ("source",
             "results/spatial_modulation/propagation_geometry_broad/"
             "components/path_axis/per_subject/"),
            ("built_by", "scripts/build_broad_lagpat_patch.py"),
            ("n_patched_rows", sum(1 for r in rows if r["broad_patched"])),
        ])),
        ("reconciliation", build_reconciliation(rows)),
        ("per_subject", rows),
    ])
    json_path = os.path.join(OUT_DIR, "real_feature_coverage.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("wrote", csv_path)
    print("wrote", json_path)
    print("n_subjects:", len(rows))
    for feat in ("rank_available", "axis_available", "pathway_width_available"):
        roll = payload["per_feature_coverage"][feat]
        print(f"  {feat}: {roll['n_available']}/{roll['n_total_subjects']} "
              f"by_dataset={dict(roll['by_dataset'])}")
    print("weak_axis:", payload["weak_axis_count"])
    print("reconciliation:")
    for k, v in payload["reconciliation"].items():
        if isinstance(v, dict) and "match" in v:
            print(f"  {k}: computed={v['computed']} anchor={v['anchor']} "
                  f"match={v['match']}")


if __name__ == "__main__":
    main()
