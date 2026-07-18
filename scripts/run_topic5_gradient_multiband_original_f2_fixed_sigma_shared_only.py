#!/usr/bin/env python3
"""Recompute the fixed-sigma gradient F2 statistic on shared subjects only.

This is a closed subgroup re-analysis of the canonical n=19 fixed-sigma run.
It filters by the pre-declared field-availability route (``field_plane=shared``)
before looking at any band result, then recomputes the cohort medians and the
seven-band Westfall--Young max-T FWER distribution on those subjects alone.
No own-fallback subject contributes to the displayed deltas or permutation
nulls.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

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
OUT = ROOT / "results/topic5_ictal_recruitment/gradient_multiband_original_f2_fixed_sigma_shared_only"
PAPER = ROOT / "results/paper-ready-figure/fig3_gradient_multiband_significance"
PAPER_FIGURES = PAPER / "figures"
STEM = "gradient_multiband_significance_original_f2_fixed_sigma_shared_only"
CONTRACT = "topic5_gradient_shared_only_original_f2_subject_fixed_sigma_v1"
EXPECTED_SHARED_N = 8
MIN_PERM = 1000


def shared_subject_ids(routing: pd.DataFrame, *, expected_n: int = EXPECTED_SHARED_N) -> list[str]:
    """Return the result-independent shared-field denominator."""
    required = {"subject", "field_plane", "smoothing_policy", "sigma_common"}
    missing = required - set(routing.columns)
    if missing:
        raise ValueError(f"routing table lacks fields:{sorted(missing)}")
    if not routing["subject"].is_unique:
        raise ValueError("routing subjects are not unique")
    shared = routing.loc[routing["field_plane"].eq("shared"), "subject"].astype(str).tolist()
    if len(shared) != int(expected_n):
        raise ValueError(f"shared denominator drifted from n={expected_n}:{shared}")
    sub = routing[routing["subject"].astype(str).isin(shared)]
    if not sub["smoothing_policy"].eq("subject_fixed").all():
        raise ValueError("shared subjects do not come from the fixed-sigma run")
    return sorted(shared)


def filter_shared_inputs(
    subjects: pd.DataFrame,
    perm_rows: pd.DataFrame,
    routing: pd.DataFrame,
    *,
    expected_n: int = EXPECTED_SHARED_N,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter all statistical inputs by routing before cohort recomputation."""
    shared = shared_subject_ids(routing, expected_n=expected_n)
    keep = set(shared)
    subject_shared = subjects[subjects["subject"].astype(str).isin(keep)].copy()
    perm_shared = perm_rows[perm_rows["subject"].astype(str).isin(keep)].copy()
    routing_shared = routing[routing["subject"].astype(str).isin(keep)].copy()
    if subject_shared["subject"].nunique() != int(expected_n):
        raise ValueError("subject summary is incomplete for shared denominator")
    if perm_shared["subject"].nunique() != int(expected_n):
        raise ValueError("permutation table is incomplete for shared denominator")
    if not subject_shared["field_plane"].eq("shared").all():
        raise ValueError("own-fallback row leaked into shared subject summary")
    sizes = perm_shared.groupby(["subject", "band"]).size().unique()
    if len(sizes) != 1 or int(sizes[0]) != MIN_PERM + 1:
        raise ValueError(f"shared permutation rows are incomplete:{sizes}")
    return subject_shared, perm_shared, routing_shared


def _plot(subjects: pd.DataFrame, cohort: pd.DataFrame, band_contract: Sequence[dict],
          output: Path, *, seed: int) -> Path:
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
        f"F2 · shared gradient field · fixed sigma (n=8) · {passed}/7 pass FWER",
        output,
        ylabel="Shared gradient-field alignment − spatial-null median\n(subject-level Δ)",
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

这是 fixed-sigma gradient F2 的 shared-only subgroup 版本。分母在读取频带结果前由 frozen routing 的 `field_plane=shared` 固定为 n=8；own fallback 11 人不进入 delta、cohort median 或任何 FWER null draw。原五类窗口、window→seizure→subject 折叠、endpoint-plane spatial-null 分组及 1000 次七频带 max-T FWER 均继承自 canonical fixed-sigma run。

**关注点**：{passed}/7 个频带通过 shared-only FWER。{detail}。该图是预先按 field availability 定义的 subgroup/sensitivity，不替代 n=19 主分析。
"""
    path.write_text(existing.rstrip() + "\n\n" + addition)
    return path


def run(args: argparse.Namespace) -> dict:
    band_contract = load_primary_band_contract()
    bands = [str(row["band"]) for row in band_contract]
    paths_in = {
        "event": SOURCE / f"{SOURCE_STEM}_event.csv",
        "subject": SOURCE / f"{SOURCE_STEM}_subject.csv",
        "routing": SOURCE / f"{SOURCE_STEM}_field_routing.csv",
        "null_draws": SOURCE / f"{SOURCE_STEM}_subject_spatial_null_draws.parquet",
        "summary": SOURCE / f"{SOURCE_STEM}_summary.json",
    }
    missing = [str(path) for path in paths_in.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"fixed-sigma source artifacts missing:{missing}")
    source_summary = json.loads(paths_in["summary"].read_text())
    if source_summary.get("contract") != "topic5_gradient_shared_else_own_original_f2_subject_fixed_sigma_v1":
        raise ValueError("unexpected fixed-sigma source contract")

    events = pd.read_csv(paths_in["event"])
    subjects = pd.read_csv(paths_in["subject"])
    routing = pd.read_csv(paths_in["routing"])
    perm_rows = pd.read_parquet(paths_in["null_draws"])
    subject_shared, perm_shared, routing_shared = filter_shared_inputs(
        subjects, perm_rows, routing,
    )
    shared = set(routing_shared["subject"].astype(str))
    event_shared = events[events["subject"].astype(str).isin(shared)].copy()
    cohort = build_cohort_table(
        subject_shared,
        perm_shared.assign(feature="raw_gradient"),
        band_contract,
    )
    if not (cohort["n_subjects"] == EXPECTED_SHARED_N).all():
        raise RuntimeError("shared-only cohort denominator is not n=8 in every band")

    OUT.mkdir(parents=True, exist_ok=True)
    PAPER.mkdir(parents=True, exist_ok=True)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    exclusions = routing[~routing["subject"].astype(str).isin(shared)][
        ["dataset", "subject", "field_plane"]
    ].copy()
    exclusions["exclusion_reason"] = "own_fallback_not_in_shared_only_subgroup"
    source_drops = source_summary.get("drops") or []
    if source_drops:
        source_drop_frame = pd.DataFrame(source_drops)
        source_drop_frame["dataset"] = source_drop_frame["subject"].astype(str).str.split("_", n=1).str[0]
        source_drop_frame["field_plane"] = "unavailable"
        source_drop_frame["exclusion_reason"] = source_drop_frame["drop_reason"]
        exclusions = pd.concat([
            exclusions,
            source_drop_frame[["dataset", "subject", "field_plane", "exclusion_reason"]],
        ], ignore_index=True)

    paths = {
        "event": OUT / f"{STEM}_event.csv",
        "subject": OUT / f"{STEM}_subject.csv",
        "cohort": OUT / f"{STEM}_cohort.csv",
        "routing": OUT / f"{STEM}_field_routing.csv",
        "exclusions": OUT / f"{STEM}_exclusion_inventory.csv",
        "null_draws": OUT / f"{STEM}_subject_spatial_null_draws.parquet",
    }
    event_shared.sort_values(["band", "subject", "seizure_idx", "win_start_rel"]).to_csv(
        paths["event"], index=False
    )
    subject_shared.sort_values(["band", "subject"]).to_csv(paths["subject"], index=False)
    cohort.to_csv(paths["cohort"], index=False)
    routing_shared.sort_values("subject").to_csv(paths["routing"], index=False)
    exclusions.sort_values("subject").to_csv(paths["exclusions"], index=False)
    perm_shared.to_parquet(paths["null_draws"], index=False)

    png = PAPER_FIGURES / f"{STEM}.png"
    _plot(subject_shared, cohort, band_contract, png, seed=args.seed)
    readme = _write_readme(cohort)
    counts = {
        "shared_subjects": int(subject_shared["subject"].nunique()),
        "shared_window_rows": int(len(event_shared)),
        "excluded_own_fallback_subjects": int((exclusions["field_plane"] == "own_fallback").sum()),
        "excluded_axis_unavailable_subjects": int((exclusions["field_plane"] == "unavailable").sum()),
        "bands_passing_fwer": int(cohort["passes_fwer_0p05"].sum()),
    }
    summary = {
        "contract": CONTRACT,
        "analysis_role": "shared-field availability subgroup sensitivity",
        "source_contract": source_summary["contract"],
        "selection": "field_plane=shared before band outcomes are read",
        "own_fallback_in_statistics": False,
        "smoothing": "one shared-plane sigma per subject, reused for A/B and all observed/null scores",
        "fwer": "recomputed within n=8 shared subjects across seven primary bands",
        "counts": counts,
        "cohort_statistics": cohort.to_dict("records"),
        "shared_subjects": sorted(shared),
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
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
