#!/usr/bin/env python3
"""Write the paper-facing old-versus-refresh Figure 3 change report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def run(args: argparse.Namespace) -> dict:
    legacy_field = _load(args.legacy_field_summary)
    timing_field = _load(args.timing_field_summary)
    space_field = _load(args.space_field_summary)
    fig3d = _load(args.fig3d_comparison)
    old_c = _load(args.legacy_fig3c)
    new_c = _load(args.space_fig3c)
    old_e = _load(args.legacy_fig3e)
    new_e = _load(args.space_fig3e)
    old_f = _load(args.legacy_fig3f)
    new_f = _load(args.space_fig3f)

    def fig3c_score(payload: dict) -> float:
        return float(payload["score_audit"]["observed"]["shared_maxab"])

    payload = {
        "scientific_question": (
            "How do Figure 3 template-dependent results change when interictal "
            "templates are clustered with timing plus real-coordinate event direction?"
        ),
        "template_geometry": {
            "legacy": legacy_field.get("denominators"),
            "matched_timing_only": timing_field.get("denominators"),
            "timing_plus_space": space_field.get("denominators"),
            "timing_plus_space_geometry_change": space_field.get("geometry_change"),
            "timing_plus_space_shared_field_change": space_field.get("shared_field_change"),
        },
        "fig3c_representative_field": {
            "subject": "epilepsiae_1146",
            "legacy_shared_maxab_r": fig3c_score(old_c),
            "timing_plus_space_shared_maxab_r": fig3c_score(new_c),
            "delta": fig3c_score(new_c) - fig3c_score(old_c),
            "legacy_fingerprint": old_c.get("frozen_fingerprint"),
            "timing_plus_space_fingerprint": new_c.get("frozen_fingerprint"),
        },
        "fig3d_cohort": {
            "variant_statistics": fig3d["fig3d_cohort_statistics"],
            "paired_comparisons": fig3d["fig3d_paired_comparisons"],
            "safe_conclusion": (
                "The legacy Pooled/Broadband Data>Null result is not retained after "
                "the QC-clean full-template refresh. Timing+Space is not detectably "
                "different from the matched Timing-only refresh, so the loss cannot "
                "be attributed specifically to the spatial view."
            ),
        },
        "fig3e_representative_trajectory": {
            "subject": "epilepsiae_1146",
            "legacy_n_seizures": old_e.get("n_seizures_processed"),
            "timing_plus_space_n_seizures": new_e.get("n_seizures_processed"),
            "legacy_projection": old_e.get("template_projection_z"),
            "timing_plus_space_projection": new_e.get("template_projection_z"),
            "legacy_pre_similarity": old_e.get("pre_m120_0"),
            "timing_plus_space_pre_similarity": new_e.get("pre_m120_0"),
            "legacy_early_similarity": old_e.get("early_0_30"),
            "timing_plus_space_early_similarity": new_e.get("early_0_30"),
        },
        "fig3f_ab_dominance": {
            "legacy_primary": old_f["primary_cohort_hierarchical_time_null"],
            "timing_plus_space_primary": new_f[
                "primary_cohort_hierarchical_time_null"
            ],
            "legacy_wilcoxon": old_f["primary_wilcoxon_greater"],
            "timing_plus_space_wilcoxon": new_f["primary_wilcoxon_greater"],
            "legacy_locked_count": old_f["primary_subject_locked_count"],
            "timing_plus_space_locked_count": new_f[
                "primary_subject_locked_count"
            ],
            "safe_conclusion": (
                "The full 17-subject near-onset increase in absolute A/B dominance "
                "is retained and slightly stronger under Timing+Space templates."
            ),
        },
        "claim_boundary": (
            "Held-out evidence for adding spatial information remains the interictal "
            "recording-block cross-fit direction score. Figure 3 uses all-interictal "
            "refits and is a downstream robustness analysis, not a second held-out proof."
        ),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "figure3_timing_plus_space_refresh_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )

    d_rows = {
        row["group_id"]: row
        for row in fig3d["fig3d_cohort_statistics"]
        if row["variant"] == "timing_plus_space"
    }
    md = f"""# Figure 3 Timing+Space 更新结论

## 模板与二维场

- Timing+Space 仍有 26 名患者支持二维场，没有新增二维患者。
- 可建立 shared field 的患者为 {space_field['denominators']['shared_fields_ready']} 名；matched Timing-only 为 {timing_field['denominators']['shared_fields_ready']} 名。

## Figure 3C/E

- E1146 seizure 2 的 shared maxAB：{fig3c_score(old_c):.3f} → {fig3c_score(new_c):.3f}。
- E1146 onset 后 q 中位数：{old_e['template_projection_z']['early_0_30_median']:.3f} → {new_e['template_projection_z']['early_0_30_median']:.3f}。

## Figure 3D

- Pooled：n={d_rows['all_phenotype_matched']['n_subjects']}，Data={d_rows['all_phenotype_matched']['data_median']:.3f}，Null={d_rows['all_phenotype_matched']['null_median']:.3f}，单侧 P={d_rows['all_phenotype_matched']['wilcoxon_one_sided_data_gt_null_p']:.3f}。
- Broadband：n={d_rows['strict_broadband']['n_subjects']}，Data={d_rows['strict_broadband']['data_median']:.3f}，Null={d_rows['strict_broadband']['null_median']:.3f}，单侧 P={d_rows['strict_broadband']['wilcoxon_one_sided_data_gt_null_p']:.3f}。
- Gamma：n={d_rows['gamma_nonbroadband']['n_subjects']}，Data={d_rows['gamma_nonbroadband']['data_median']:.3f}，Null={d_rows['gamma_nonbroadband']['null_median']:.3f}，单侧 P={d_rows['gamma_nonbroadband']['wilcoxon_one_sided_data_gt_null_p']:.3f}。
- 三组均不显著。matched Timing-only 新流程同样不显著，Timing+Space 与 matched Timing-only 的患者内变化也不显著；因此不能把旧 Fig. 3D 结论的消失单独归因于空间信息。

## Figure 3F

- 仍为 n={new_f['n_primary_eligible']}；cohort median delta：{old_f['primary_cohort_hierarchical_time_null']['median_delta']:.3f} → {new_f['primary_cohort_hierarchical_time_null']['median_delta']:.3f}。
- 层级时间 null P：{old_f['primary_cohort_hierarchical_time_null']['p_one_sided']:.4f} → {new_f['primary_cohort_hierarchical_time_null']['p_one_sided']:.4f}。
- 结论保留：临近 onset 的 A/B 相对优势增强。
"""
    (args.out_dir / "figure3_timing_plus_space_refresh_summary.md").write_text(md)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "legacy_field_summary", "timing_field_summary", "space_field_summary",
        "fig3d_comparison", "legacy_fig3c", "space_fig3c",
        "legacy_fig3e", "space_fig3e", "legacy_fig3f", "space_fig3f",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({
        "template_geometry": result["template_geometry"],
        "fig3c": result["fig3c_representative_field"],
        "fig3f": result["fig3f_ab_dominance"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
