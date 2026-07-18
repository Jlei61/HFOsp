#!/usr/bin/env python3
"""Cross-panel cohort-matched sensitivity for the fixed-sigma gradient F2.

This runner does not rescore windows.  It starts from the canonical n=19
subject-fixed-sigma artifact, freezes the independent cross-panel denominator
to ``group_id=all_phenotype_matched``, and then recomputes the cohort median and
seven-band Westfall--Young max-T FWER on those exact 17 subjects.  The selector
must exclude only E583 and Yuquan zhangkexuan; E139 and E1146 are required to
remain in the sensitivity cohort.

The output is a sensitivity sidecar.  It neither replaces the n=19 primary
analysis nor interprets the two exclusions as a geometry or single-shaft gate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_v2_phase1_figures import plot_null_per_band_figure  # noqa: E402
from scripts.run_topic5_gradient_multiband_original_f2 import (  # noqa: E402
    BASE_SEED,
    build_cohort_table,
    load_primary_band_contract,
)
from src.topic5_tspectral_field_concordance import jsonable  # noqa: E402


SOURCE = ROOT / "results/topic5_ictal_recruitment/gradient_multiband_original_f2_fixed_sigma"
SOURCE_STEM = "gradient_multiband_significance_original_f2_fixed_sigma"
SELECTOR_SOURCE = (
    ROOT
    / "results/topic5_ictal_recruitment/tspectral_field_concordance/"
      "clinical_onset_gradient_field_cohort_stat_subject.csv"
)
SELECTOR_DROP_SOURCE = (
    ROOT
    / "results/topic5_ictal_recruitment/tspectral_field_concordance/"
      "clinical_onset_gradient_field_cohort_stat_drop_inventory.csv"
)
OUT = (
    ROOT
    / "results/topic5_ictal_recruitment/"
      "gradient_multiband_original_f2_fixed_sigma_cohort_matched"
)
PAPER = ROOT / "results/paper-ready-figure/fig3_gradient_multiband_significance"
PAPER_FIGURES = PAPER / "figures"

SOURCE_CONTRACT = "topic5_gradient_shared_else_own_original_f2_subject_fixed_sigma_v1"
CONTRACT = (
    "topic5_gradient_original_f2_subject_fixed_sigma_"
    "cross_panel_cohort_matched_sensitivity_v1"
)
STEM = "gradient_multiband_significance_original_f2_fixed_sigma_cohort_matched_n17"
GROUP_ID = "all_phenotype_matched"
EXPECTED_SOURCE_N = 19
EXPECTED_MATCHED_N = 17
MIN_PERM = 1000
EXPECTED_EXCLUSIONS = frozenset({"epilepsiae_583", "yuquan_zhangkexuan"})
REQUIRED_INCLUDED = frozenset({"epilepsiae_139", "epilepsiae_1146"})
EXPECTED_SELECTOR_DROP_REASON = "no_strict_broadband_or_gamma_nonbroadband_event"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source_summary(
    summary: Mapping[str, object],
    *,
    expected_contract: str = SOURCE_CONTRACT,
    expected_n: int = EXPECTED_SOURCE_N,
) -> None:
    """Fail closed unless the input is the canonical n=19 fixed-sigma run."""
    if summary.get("contract") != expected_contract:
        raise ValueError(f"unexpected fixed-sigma source contract:{summary.get('contract')}")
    counts = summary.get("counts") or {}
    if int(counts.get("analysis_subjects", -1)) != int(expected_n):
        raise ValueError(f"fixed-sigma source denominator drifted from n={expected_n}")
    smoothing = summary.get("smoothing") or {}
    if smoothing.get("policy") != "subject_fixed":
        raise ValueError("source is not the subject-fixed-sigma analysis")


def cohort_matched_subject_ids(
    selector: pd.DataFrame,
    *,
    group_id: str = GROUP_ID,
    expected_n: int = EXPECTED_MATCHED_N,
    required_included: Sequence[str] = tuple(REQUIRED_INCLUDED),
) -> list[str]:
    """Return the exact cross-panel denominator from the independent selector."""
    required_columns = {"subject", "group_id"}
    missing = required_columns - set(selector.columns)
    if missing:
        raise ValueError(f"selector lacks fields:{sorted(missing)}")
    matched = selector.loc[
        selector["group_id"].astype(str).eq(str(group_id)), "subject"
    ].astype(str)
    if matched.duplicated().any():
        duplicated = sorted(matched[matched.duplicated(keep=False)].unique())
        raise ValueError(f"duplicate cohort-matched selector subjects:{duplicated}")
    subjects = sorted(matched.tolist())
    if len(subjects) != int(expected_n):
        raise ValueError(f"cohort-matched denominator drifted from n={expected_n}:{subjects}")
    missing_required = set(map(str, required_included)) - set(subjects)
    if missing_required:
        raise ValueError(
            "required cohort-matched subjects are absent:"
            f"{sorted(missing_required)}"
        )
    return subjects


def validate_exact_cohort_difference(
    canonical_subjects: Sequence[str],
    matched_subjects: Sequence[str],
    *,
    expected_source_n: int = EXPECTED_SOURCE_N,
    expected_matched_n: int = EXPECTED_MATCHED_N,
    expected_exclusions: Sequence[str] = tuple(EXPECTED_EXCLUSIONS),
) -> list[str]:
    """Require the n=19 minus n=17 difference to be exactly the two locked IDs."""
    canonical = set(map(str, canonical_subjects))
    matched = set(map(str, matched_subjects))
    if len(canonical) != int(expected_source_n):
        raise ValueError(f"canonical subject set drifted from n={expected_source_n}")
    if len(matched) != int(expected_matched_n):
        raise ValueError(f"matched subject set drifted from n={expected_matched_n}")
    extra = matched - canonical
    if extra:
        raise ValueError(f"selector contains subjects absent from canonical n=19:{sorted(extra)}")
    excluded = canonical - matched
    expected = set(map(str, expected_exclusions))
    if excluded != expected:
        raise ValueError(
            "canonical-minus-matched exclusions drifted:"
            f"expected={sorted(expected)},actual={sorted(excluded)}"
        )
    return sorted(excluded)


def selector_exclusion_reasons(
    drop_inventory: pd.DataFrame,
    excluded_subjects: Sequence[str],
    *,
    group_id: str = GROUP_ID,
    expected_reason: str = EXPECTED_SELECTOR_DROP_REASON,
) -> dict[str, str]:
    """Validate the selector's own recorded reason for the locked exclusions."""
    required = {"subject", "group_id", "drop_reason"}
    missing = required - set(drop_inventory.columns)
    if missing:
        raise ValueError(f"selector drop inventory lacks fields:{sorted(missing)}")
    excluded = set(map(str, excluded_subjects))
    rows = drop_inventory[
        drop_inventory["subject"].astype(str).isin(excluded)
        & drop_inventory["group_id"].astype(str).eq(str(group_id))
    ].copy()
    if rows["subject"].astype(str).duplicated().any():
        raise ValueError("duplicate selector exclusion reason rows")
    reasons = dict(zip(rows["subject"].astype(str), rows["drop_reason"].astype(str)))
    if set(reasons) != excluded:
        raise ValueError("selector drop inventory is incomplete for locked exclusions")
    unexpected = {
        subject: reason for subject, reason in reasons.items()
        if reason != str(expected_reason)
    }
    if unexpected:
        raise ValueError(f"unexpected selector exclusion reasons:{unexpected}")
    return reasons


def filter_cohort_matched_inputs(
    subjects: pd.DataFrame,
    perm_rows: pd.DataFrame,
    routing: pd.DataFrame,
    matched_subjects: Sequence[str],
    *,
    bands: Sequence[str],
    expected_n: int = EXPECTED_MATCHED_N,
    n_perm: int = MIN_PERM,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter every statistical input before recomputing the cohort statistic."""
    for label, frame, required in (
        ("subject", subjects, {"subject", "band", "field_plane", "delta"}),
        ("permutation", perm_rows, {"subject", "band", "perm_id", "perm_subject_median"}),
        ("routing", routing, {"subject", "smoothing_policy", "sigma_common"}),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{label} input lacks fields:{sorted(missing)}")

    keep = set(map(str, matched_subjects))
    if len(keep) != int(expected_n):
        raise ValueError(f"matched filter set is not n={expected_n}")
    subject_out = subjects[subjects["subject"].astype(str).isin(keep)].copy()
    perm_out = perm_rows[perm_rows["subject"].astype(str).isin(keep)].copy()
    routing_out = routing[routing["subject"].astype(str).isin(keep)].copy()

    for label, frame in (
        ("subject", subject_out), ("permutation", perm_out), ("routing", routing_out)
    ):
        actual = set(frame["subject"].astype(str))
        if actual != keep:
            raise ValueError(f"{label} input is incomplete for cohort-matched subjects")
    if not routing_out["subject"].is_unique:
        raise ValueError("cohort-matched routing subjects are not unique")
    if not routing_out["smoothing_policy"].eq("subject_fixed").all():
        raise ValueError("non-fixed-sigma routing row leaked into cohort-matched inputs")
    if not pd.to_numeric(routing_out["sigma_common"], errors="coerce").notna().all():
        raise ValueError("cohort-matched routing has missing common sigma")

    expected_bands = set(map(str, bands))
    by_subject = subject_out.groupby("subject")["band"].agg(
        lambda values: set(map(str, values))
    )
    if len(by_subject) != int(expected_n) or not all(
        value == expected_bands for value in by_subject
    ):
        raise ValueError("subject summary does not contain exactly one complete band family")
    if subject_out.duplicated(["subject", "band"]).any():
        raise ValueError("duplicate cohort-matched subject-band summary row")

    expected_perm_ids = set(range(-1, int(n_perm)))
    perm_id_sets = perm_out.groupby(["subject", "band"])["perm_id"].agg(
        lambda values: set(pd.to_numeric(values, errors="raise").astype(int))
    )
    if len(perm_id_sets) != int(expected_n) * len(expected_bands) or not all(
        value == expected_perm_ids for value in perm_id_sets
    ):
        raise ValueError("cohort-matched permutation draws are incomplete")
    return subject_out, perm_out, routing_out


def _plot(
    subjects: pd.DataFrame,
    cohort: pd.DataFrame,
    band_contract: Sequence[Mapping[str, object]],
    output: Path,
    *,
    seed: int,
) -> Path:
    bands = [str(row["band"]) for row in band_contract]
    labels = {str(row["band"]): str(row["label"]) for row in band_contract}
    values = {
        band: subjects.loc[subjects["band"] == band, "delta"].to_numpy(float)
        for band in bands
    }
    by_band = cohort.set_index("band")
    passed = int(cohort["passes_fwer_0p05"].sum())
    return plot_null_per_band_figure(
        bands,
        labels,
        values,
        by_band["cohort_perm_delta_spatial"].to_dict(),
        by_band["max_over_bands_p"].to_dict(),
        by_band["n_subjects"].to_dict(),
        f"F2 · fixed-sigma gradient · cohort-matched sensitivity "
        f"(n=17) · {passed}/7 pass FWER",
        output,
        ylabel="Gradient-field alignment − spatial-null median\n(subject-level Δ)",
        save_pdf=True,
        seed=seed,
        figsize=(11.8, 6.8),
        show_exact_annotations=False,
        significance_legend="band passes 7-band FWER",
        nonsignificance_legend="n.s. band",
        cohort_legend="cohort-matched Δ (tested)",
        subject_legend="per-subject Δ",
        title_mode="figure",
        layout_rect=(0.01, 0.01, 0.82, 0.95),
        xtick_fontsize=12.5,
        ytick_fontsize=11.5,
        ylabel_fontsize=13,
        title_fontsize=14,
        legend_fontsize=10.5,
    )


def _write_readme(cohort: pd.DataFrame) -> Path:
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    path = PAPER_FIGURES / "README.md"
    existing = path.read_text() if path.exists() else "# Topic 5 gradient 多频带显著性图\n"
    marker = f"### {STEM}.png / {STEM}.pdf"
    if marker in existing:
        existing = existing.split(marker, 1)[0].rstrip() + "\n\n"
    passed = int(cohort["passes_fwer_0p05"].sum())
    detail = "; ".join(
        f"{row.band}: pFWER={row.max_over_bands_p:.4g}, "
        f"Δ={row.cohort_perm_delta_spatial:.3f}"
        for row in cohort.itertuples()
    )
    addition = f"""### {STEM}.png / {STEM}.pdf

这是 canonical n=19 fixed-sigma gradient F2 的 cross-panel cohort-matched sensitivity。分母在读取七频带 cohort 结果前，由 `clinical_onset_gradient_field_cohort_stat_subject.csv` 的 `group_id=all_phenotype_matched` 独立固定为 n=17；随后仅在这 17 人内重新计算 cohort median 和七频带 Westfall–Young max-T FWER。E583 与 Yuquan zhangkexuan 因不在该 cross-panel phenotype-matched group 而排除；E139 与 E1146 明确保留。

**关注点**：{passed}/7 个频带通过 cohort-matched FWER。{detail}。这不是把两名排除者描述成“单杆/单 shaft”患者的几何剔除，名单也不是按本图 multiband concordance 结果挑选；它由预定义的 T_spectral 频谱表型合同给出，仅用于跨 panel 同分母敏感性，不替代 n=19 主分析。
"""
    path.write_text(existing.rstrip() + "\n\n" + addition)
    return path


def run(args: argparse.Namespace) -> dict[str, object]:
    band_contract = load_primary_band_contract()
    bands = [str(row["band"]) for row in band_contract]
    inputs = {
        "event": SOURCE / f"{SOURCE_STEM}_event.csv",
        "subject": SOURCE / f"{SOURCE_STEM}_subject.csv",
        "routing": SOURCE / f"{SOURCE_STEM}_field_routing.csv",
        "null_draws": SOURCE / f"{SOURCE_STEM}_subject_spatial_null_draws.parquet",
        "summary": SOURCE / f"{SOURCE_STEM}_summary.json",
        "selector": SELECTOR_SOURCE,
        "selector_drop_inventory": SELECTOR_DROP_SOURCE,
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cohort-matched sensitivity inputs missing:{missing}")

    source_summary = json.loads(inputs["summary"].read_text())
    validate_source_summary(source_summary)
    selector = pd.read_csv(inputs["selector"])
    selector_drops = pd.read_csv(inputs["selector_drop_inventory"])
    matched = cohort_matched_subject_ids(selector)
    events = pd.read_csv(inputs["event"])
    subjects = pd.read_csv(inputs["subject"])
    routing = pd.read_csv(inputs["routing"])
    perm_rows = pd.read_parquet(inputs["null_draws"])

    canonical_subjects = sorted(subjects["subject"].astype(str).unique())
    excluded = validate_exact_cohort_difference(canonical_subjects, matched)
    exclusion_reasons = selector_exclusion_reasons(selector_drops, excluded)
    subject_matched, perm_matched, routing_matched = filter_cohort_matched_inputs(
        subjects,
        perm_rows,
        routing,
        matched,
        bands=bands,
    )
    keep = set(matched)
    event_matched = events[events["subject"].astype(str).isin(keep)].copy()
    if set(event_matched["subject"].astype(str)) != keep:
        raise ValueError("event input is incomplete for cohort-matched subjects")
    cohort = build_cohort_table(
        subject_matched,
        perm_matched.assign(feature="raw_gradient"),
        band_contract,
    )
    if not (cohort["n_subjects"] == EXPECTED_MATCHED_N).all():
        raise RuntimeError("cohort-matched denominator is not n=17 in every band")

    exclusion_inventory = routing[
        routing["subject"].astype(str).isin(excluded)
    ][["dataset", "subject", "field_plane", "score_key"]].copy()
    if set(exclusion_inventory["subject"].astype(str)) != set(EXPECTED_EXCLUSIONS):
        raise RuntimeError("exclusion inventory does not contain the locked two subjects")
    exclusion_inventory["required_group_id"] = GROUP_ID
    exclusion_inventory["selector_drop_reason"] = exclusion_inventory["subject"].map(
        exclusion_reasons
    )
    exclusion_inventory["exclusion_reason"] = (
        "absent_from_cross_panel_all_phenotype_matched_group:"
        + exclusion_inventory["selector_drop_reason"].astype(str)
    )
    exclusion_inventory[
        "selection_predefined_and_independent_of_current_multiband_concordance_outcomes"
    ] = True
    exclusion_inventory["single_shaft_or_geometry_exclusion_claim"] = False

    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    outputs = {
        "event": OUT / f"{STEM}_event.csv",
        "subject": OUT / f"{STEM}_subject.csv",
        "cohort": OUT / f"{STEM}_cohort.csv",
        "routing": OUT / f"{STEM}_field_routing.csv",
        "exclusions": OUT / f"{STEM}_exclusion_inventory.csv",
        "null_draws": OUT / f"{STEM}_subject_spatial_null_draws.parquet",
        "manifest": OUT / f"{STEM}_manifest.json",
        "summary": OUT / f"{STEM}_summary.json",
    }
    event_matched.sort_values(
        ["band", "subject", "seizure_idx", "win_start_rel"]
    ).to_csv(outputs["event"], index=False)
    subject_matched.sort_values(["band", "subject"]).to_csv(
        outputs["subject"], index=False
    )
    cohort.to_csv(outputs["cohort"], index=False)
    routing_matched.sort_values("subject").to_csv(outputs["routing"], index=False)
    exclusion_inventory.sort_values("subject").to_csv(outputs["exclusions"], index=False)
    perm_matched.sort_values(["band", "subject", "perm_id"]).to_parquet(
        outputs["null_draws"], index=False
    )

    png = PAPER_FIGURES / f"{STEM}.png"
    _plot(subject_matched, cohort, band_contract, png, seed=args.seed)
    readme = _write_readme(cohort)

    input_manifest = {
        key: {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
        }
        for key, path in inputs.items()
    }
    selection_contract = {
        "selector_source_path": str(SELECTOR_SOURCE.relative_to(ROOT)),
        "selector_source_sha256": input_manifest["selector"]["sha256"],
        "selector_drop_inventory_path": str(SELECTOR_DROP_SOURCE.relative_to(ROOT)),
        "selector_drop_inventory_sha256": input_manifest["selector_drop_inventory"]["sha256"],
        "selector_group_id": GROUP_ID,
        "selector_basis": "predefined T_spectral spectral phenotype contract",
        "selection_predefined_and_independent_of_current_multiband_concordance_outcomes": True,
        "exact_included_subject_ids": matched,
        "exact_excluded_subject_ids": excluded,
        "required_included_subject_ids": sorted(REQUIRED_INCLUDED),
        "expected_excluded_subject_ids": sorted(EXPECTED_EXCLUSIONS),
        "canonical_minus_matched_verified_exact": True,
    }
    output_paths = {
        key: str(path.relative_to(ROOT)) for key, path in outputs.items()
    } | {
        "figure_png": str(png.relative_to(ROOT)),
        "figure_pdf": str(png.with_suffix(".pdf").relative_to(ROOT)),
        "figure_readme": str(readme.relative_to(ROOT)),
    }
    counts = {
        "canonical_fixed_sigma_subjects": len(canonical_subjects),
        "cohort_matched_subjects": len(matched),
        "excluded_subjects": len(excluded),
        "epilepsiae_subjects": int(
            subject_matched.drop_duplicates("subject")["dataset"].eq("epilepsiae").sum()
        ),
        "yuquan_subjects": int(
            subject_matched.drop_duplicates("subject")["dataset"].eq("yuquan").sum()
        ),
        "shared_subjects": int(
            subject_matched.drop_duplicates("subject")["field_plane"].eq("shared").sum()
        ),
        "own_fallback_subjects": int(
            subject_matched.drop_duplicates("subject")["field_plane"].eq("own_fallback").sum()
        ),
        "bands_passing_fwer": int(cohort["passes_fwer_0p05"].sum()),
        "event_window_rows": int(len(event_matched)),
        "subject_band_rows": int(len(subject_matched)),
        "permutation_rows": int(len(perm_matched)),
    }
    manifest = {
        "contract": CONTRACT,
        "analysis_role": "cross-panel cohort-matched sensitivity",
        "replaces_canonical_n19_primary": False,
        "source_contract": SOURCE_CONTRACT,
        "input_artifacts": input_manifest,
        "selection_contract": selection_contract,
        "statistical_contract": {
            "subject_level_inputs_reused_without_rescoring": True,
            "cohort_statistic": "median over included subjects",
            "fwer": "null-centered Westfall-Young max-T across seven primary bands",
            "n_permutations": MIN_PERM,
            "bands": band_contract,
        },
        "counts": counts,
        "outputs": output_paths,
    }
    outputs["manifest"].write_text(
        json.dumps(jsonable(manifest), ensure_ascii=False, indent=2) + "\n"
    )
    summary = {
        "contract": CONTRACT,
        "analysis_role": "cross-panel cohort-matched sensitivity",
        "replaces_canonical_n19_primary": False,
        "source_contract": SOURCE_CONTRACT,
        "selection_contract": selection_contract,
        "manifest": {
            "path": str(outputs["manifest"].relative_to(ROOT)),
            "sha256": _sha256(outputs["manifest"]),
        },
        "counts": counts,
        "cohort_statistics": cohort.to_dict("records"),
        "exclusions": exclusion_inventory.to_dict("records"),
        "outputs": output_paths,
    }
    outputs["summary"].write_text(
        json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n"
    )
    for source in (
        outputs["cohort"],
        outputs["subject"],
        outputs["exclusions"],
        outputs["manifest"],
        outputs["summary"],
    ):
        (PAPER / source.name).write_text(source.read_text())

    print(cohort.to_string(index=False), flush=True)
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {png}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
