#!/usr/bin/env python3
"""Render the formal onset 0--10 s gradient-field cohort panel."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.paper_figures.plot_fig3_field_concordance_cohort_stat import (
    plot_paired_data_null_groups,
)


ANALYSIS = ROOT / "results/topic5_ictal_recruitment/tspectral_field_concordance"
FIGURES = ROOT / "results/paper-ready-figure/fig3-sup-tspectral-field-concordance/figures"
STEM = "clinical_onset_gradient_field_cohort_stat"
GROUP_ORDER = (
    "all_phenotype_matched",
    "strict_broadband",
    "gamma_nonbroadband",
)
SHORT_LABELS = {
    "all_phenotype_matched": "Pooled",
    "strict_broadband": "Broadband",
    "gamma_nonbroadband": "Gamma",
}


def plot_clinical_onset_gradient_field_cohort(
    subjects: pd.DataFrame,
    cohort: pd.DataFrame,
    out_png: Path,
    out_pdf: Path,
) -> None:
    """Reuse the accepted Fig3 paired Data--Null statistical grammar."""
    groups = []
    cohort_by_group = cohort.set_index("group_id")
    for group_id in GROUP_ORDER:
        frame = subjects[subjects.group_id == group_id].sort_values("subject")
        if frame.empty:
            continue
        stat = cohort_by_group.loc[group_id]
        groups.append({
            "label": str(stat.group_label),
            "rows": [{
                "subject_id": row.subject,
                "data": float(row.data),
                "null": float(row.channel_null_median),
                "n_seizures": int(row.n_seizures),
            } for row in frame.itertuples()],
            "summary": {"n": int(stat.n_subjects)},
            "display_p": float(stat.wilcoxon_one_sided_data_gt_null_p),
            "p_label": "one-sided p",
            "x_label": f"{SHORT_LABELS[group_id]}\n$n={int(stat.n_subjects)}$",
        })
    plot_paired_data_null_groups(
        groups,
        out_png,
        out_pdf,
        ylabel="Field concordance |r|",
        seed=20260718,
        xaxis_mode="group",
        figsize=(5.8, 3.45),
        pair_gap=0.62,
        group_gap=1.75,
        ylabel_fontsize=14.0,
        tick_fontsize=11.5,
        group_tick_fontsize=11.5,
        legend_fontsize=11.5,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subject-csv", type=Path, default=ANALYSIS / f"{STEM}_subject.csv"
    )
    parser.add_argument(
        "--cohort-csv", type=Path, default=ANALYSIS / f"{STEM}_cohort.csv"
    )
    parser.add_argument("--out-dir", type=Path, default=FIGURES)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_clinical_onset_gradient_field_cohort(
        pd.read_csv(args.subject_csv),
        pd.read_csv(args.cohort_csv),
        args.out_dir / f"{STEM}.png",
        args.out_dir / f"{STEM}.pdf",
    )


if __name__ == "__main__":
    main()
