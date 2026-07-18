#!/usr/bin/env python3
"""Original narrow F2 with gradient fields and one fixed sigma per subject.

This is the load-bearing controlled replacement requested after the smoothing
audit.  It inherits the original F2 windows, folds, endpoint-plane spatial-null
groups and FWER implementation from ``run_topic5_gradient_multiband_original_f2``.
The gradient field is rebuilt with one selected-plane A bandwidth per subject;
that same bandwidth is used for TA, TB, every ictal window and every null draw.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.plot_topic5_v2_phase1_figures import plot_null_per_band_figure  # noqa: E402
from scripts.run_topic5_gradient_multiband_original_f2 import (  # noqa: E402
    BASE_SEED,
    EXPECTED_ANALYSIS_N,
    EXPECTED_REFERENCE_N,
    EXPECTED_WINDOWS,
    MIN_CONTACTS,
    MIN_PERM,
    REFERENCE_WINDOWS,
    build_cohort_table,
    load_primary_band_contract,
    score_cohort,
    select_original_f2_rows,
)
from src.topic5_tspectral_field_concordance import jsonable  # noqa: E402


OUT = ROOT / "results/topic5_ictal_recruitment/gradient_multiband_original_f2_fixed_sigma"
PAPER = ROOT / "results/paper-ready-figure/fig3_gradient_multiband_significance"
PAPER_FIGURES = PAPER / "figures"
CONTRACT = "topic5_gradient_shared_else_own_original_f2_subject_fixed_sigma_v1"
STEM = "gradient_multiband_significance_original_f2_fixed_sigma"
SMOOTHING_POLICY = "subject_fixed"


def _plot(subjects: pd.DataFrame, cohort: pd.DataFrame, band_contract, output: Path,
          *, seed: int) -> Path:
    bands = [str(row["band"]) for row in band_contract]
    labels = {str(row["band"]): str(row["label"]) for row in band_contract}
    values = {
        band: subjects.loc[subjects["band"] == band, "delta"].to_numpy(float)
        for band in bands
    }
    by_band = cohort.set_index("band")
    passed = int(cohort["passes_fwer_0p05"].sum())
    return plot_null_per_band_figure(
        bands, labels, values,
        by_band["cohort_perm_delta_spatial"].to_dict(),
        by_band["max_over_bands_p"].to_dict(),
        by_band["n_subjects"].to_dict(),
        f"F2 · gradient field · fixed subject sigma (n=19) · {passed}/7 pass FWER",
        output,
        ylabel="Gradient-field alignment − spatial-null median\n(subject-level Δ)",
        save_pdf=True,
        seed=seed,
        figsize=(11.8, 6.8),
        show_exact_annotations=False,
        significance_legend="band passes 7-band FWER",
        nonsignificance_legend="n.s. band",
        cohort_legend="cohort Δ (tested)",
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

这是旧 narrow F2 的 fixed-sigma gradient-field 替换版。五类原始重叠窗、`ictal_fraction≥0.5`、window→seizure→subject 折叠、旧 endpoint 二维坐标上的 spatial-null 分组、1000 次置换及七频带 FWER 全部保留。每名患者只从结果无关的 selected A plane 读取一次最近邻带宽；shared 患者使用 shared-plane sigma，own fallback 患者使用 own-A sigma，并强制 TA、TB、全部频带、窗口与 null draw 共用该值。

**关注点**：n=19（E916 gradient axis 不可用），{passed}/7 个频带通过 FWER。{detail}。
"""
    path.write_text(existing.rstrip() + "\n\n" + addition)
    return path


def run(args: argparse.Namespace) -> dict:
    if args.n_perm < MIN_PERM:
        raise ValueError(f"n_perm must be >= {MIN_PERM}")
    band_contract = load_primary_band_contract()
    bands = [str(row["band"]) for row in band_contract]
    reference = select_original_f2_rows(pd.read_csv(REFERENCE_WINDOWS), bands)
    if reference["subject"].nunique() != EXPECTED_REFERENCE_N:
        raise RuntimeError("original F2 reference denominator drifted from n=20")

    events, subjects, perm_rows, routing, drops = score_cohort(
        reference, bands, n_perm=args.n_perm, seed=args.seed,
        smoothing_policy=SMOOTHING_POLICY,
    )
    if subjects.empty:
        raise RuntimeError("no fixed-sigma gradient original-F2 results")
    cohort = build_cohort_table(
        subjects, perm_rows.assign(feature="raw_gradient"), band_contract,
    )
    if not (cohort["n_subjects"] == EXPECTED_ANALYSIS_N).all():
        counts = dict(zip(cohort["band"], cohort["n_subjects"]))
        raise RuntimeError(f"fixed-sigma gradient denominator drifted from n=19:{counts}")

    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    paths = {
        "event": OUT / f"{STEM}_event.csv",
        "subject": OUT / f"{STEM}_subject.csv",
        "cohort": OUT / f"{STEM}_cohort.csv",
        "drops": OUT / f"{STEM}_drop_inventory.csv",
        "routing": OUT / f"{STEM}_field_routing.csv",
        "null_draws": OUT / f"{STEM}_subject_spatial_null_draws.parquet",
    }
    events.sort_values(["band", "subject", "seizure_idx", "win_start_rel"]).to_csv(
        paths["event"], index=False
    )
    subjects.sort_values(["band", "subject"]).to_csv(paths["subject"], index=False)
    cohort.to_csv(paths["cohort"], index=False)
    drops.to_csv(paths["drops"], index=False)
    routing.sort_values("subject").to_csv(paths["routing"], index=False)
    perm_rows.to_parquet(paths["null_draws"], index=False)

    png = PAPER_FIGURES / f"{STEM}.png"
    _plot(subjects, cohort, band_contract, png, seed=args.seed)
    readme = _write_readme(cohort)
    unique_subjects = subjects.drop_duplicates("subject")
    counts = {
        "reference_subjects": int(reference["subject"].nunique()),
        "analysis_subjects": int(unique_subjects.shape[0]),
        "epilepsiae_subjects": int(unique_subjects["dataset"].eq("epilepsiae").sum()),
        "yuquan_subjects": int(unique_subjects["dataset"].eq("yuquan").sum()),
        "shared_subjects": int(unique_subjects["field_plane"].eq("shared").sum()),
        "own_fallback_subjects": int(unique_subjects["field_plane"].eq("own_fallback").sum()),
        "bands_passing_fwer": int(cohort["passes_fwer_0p05"].sum()),
        "window_rows": int(len(events)),
    }
    summary = {
        "contract": CONTRACT,
        "analysis_role": "controlled gradient replacement with original F2 subject-fixed smoothing",
        "intended_changed_input_only": "gradient shared-else-own axis and field coordinates",
        "unavoidable_denominator_change": "E916 gradient axis_not_available; n=20 to n=19",
        "smoothing": {
            "policy": SMOOTHING_POLICY,
            "rule": "one selected-A-plane median-nearest-neighbor sigma per subject",
            "shared": "shared sigma reused for shared A and B",
            "own_fallback": "own-A sigma reused for own A and B",
            "reused_for": ["TA", "TB", "all bands", "all windows", "all seizures", "all null draws"],
        },
        "retained_original_f2_contract": {
            "source_rows": str(REFERENCE_WINDOWS.relative_to(ROOT)),
            "windows_sec": [list(window) for window in EXPECTED_WINDOWS],
            "ictal_fraction_min": 0.5,
            "folding": "window median within seizure; seizure median within subject; cohort median over subjects",
            "minimum_finite_contacts": MIN_CONTACTS,
            "spatial_null": "within shaft then endpoint-plane distance-bin then subject-wide fallback",
            "n_permutations": int(args.n_perm),
            "seed": int(args.seed),
            "fwer": "null-centered Westfall-Young max-T across seven primary bands",
        },
        "counts": counts,
        "cohort_statistics": cohort.to_dict("records"),
        "routing": routing.to_dict("records"),
        "drops": drops.to_dict("records"),
        "outputs": {
            key: str(path.relative_to(ROOT)) for key, path in paths.items()
        } | {
            "figure_png": str(png.relative_to(ROOT)),
            "figure_pdf": str(png.with_suffix(".pdf").relative_to(ROOT)),
            "figure_readme": str(readme.relative_to(ROOT)),
        },
    }
    summary_path = OUT / f"{STEM}_summary.json"
    summary_path.write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2) + "\n")
    for source in (paths["cohort"], paths["subject"], summary_path):
        (PAPER / source.name).write_text(source.read_text())
    print(cohort.to_string(index=False), flush=True)
    print(json.dumps(counts, ensure_ascii=False, indent=2), flush=True)
    print(f"[done] {png}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-perm", type=int, default=MIN_PERM)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
