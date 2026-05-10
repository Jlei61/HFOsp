"""Topic 1 × Topic 5 Bridge CLI driver."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Topic 1 × Topic 5 Bridge — Q1 + Q1b + Q3 driver",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup", help="Freeze T0/T1 convention + audit-rerun marker")
    p_ps = sub.add_parser("per-subject", help="Run per-subject Q1 statistics")
    p_ps.add_argument(
        "--cohort",
        default="default",
        help="Comma-separated subject IDs, or 'default' for spec cohort + 442 + 1084",
    )
    p_ps.add_argument(
        "--bands",
        nargs="+",
        default=["gamma_ER"],
        help="Band(s) to process (default: gamma_ER)",
    )
    sub.add_parser("cohort", help="Aggregate cohort + 3-state verdict")
    sub.add_parser("sentinel-442", help="Run Q1b 442 binary-outlier exact tests")
    sub.add_parser("figures", help="Render the 5 locked figures")
    p_q1p = sub.add_parser("q1prime", help="Run Q1' per-subject (channel-rank correspondence)")
    p_q1p.add_argument("--cohort", default="default", help="default = 4 strict + 548 sentinel + 442 (descriptive)")
    p_q1p.add_argument("--band", default="gamma_ER")
    sub.add_parser("q1prime-cohort", help="Aggregate Q1' cohort + verdict")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    results_root = repo / "results"
    artifact_root = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
    out_root = results_root / "topic1_topic5_bridge"

    if args.cmd == "setup":
        from src.topic1_topic5_bridge import (
            COHORT_GAMMA, SENTINEL_442, SENSITIVITY_BROAD_1084,
            freeze_bridge_setup,
        )
        cohort = list(COHORT_GAMMA) + [SENTINEL_442, SENSITIVITY_BROAD_1084]
        payload = freeze_bridge_setup(
            cohort=cohort,
            results_root=results_root,
            artifact_root=artifact_root,
            out_path=out_root / "bridge_setup.json",
        )
        n_ok = len(payload["subjects"])
        n_dropped = len(payload["dropped_subjects"])
        print(f"setup frozen for {n_ok} subjects → {out_root / 'bridge_setup.json'}"
              + (f" ({n_dropped} dropped: {list(payload['dropped_subjects'].keys())})" if n_dropped else ""))
        return

    if args.cmd == "per-subject":
        from src.topic1_topic5_bridge import (
            COHORT_GAMMA, SENTINEL_442, SENSITIVITY_BROAD_1084,
            WINDOWS_MIN, run_per_subject,
        )
        if args.cohort == "default":
            cohort = list(COHORT_GAMMA) + [SENTINEL_442, SENSITIVITY_BROAD_1084]
        else:
            cohort = args.cohort.split(",")
        run_per_subject(
            cohort=cohort,
            bands=args.bands,
            windows_min=WINDOWS_MIN,
            results_root=results_root,
            artifact_root=artifact_root,
            out_dir=out_root / "per_subject",
        )
        print(f"per-subject done; {len(cohort)} subjects → {out_root / 'per_subject'}")
        return

    if args.cmd == "cohort":
        from src.topic1_topic5_bridge import (
            COHORT_GAMMA, WINDOWS_MIN, aggregate_cohort_summary,
        )
        payload = aggregate_cohort_summary(
            per_subject_dir=out_root / "per_subject",
            band="gamma_ER",
            windows_min=WINDOWS_MIN,
            cohort=COHORT_GAMMA,
            out_path=out_root / "cohort_summary.json",
        )
        print("verdict:", payload["cohort_judgement"])
        for wk, w in payload["windows"].items():
            print(f"  {wk}: {w['n_positive']}/{w['denom']} positive (p={w['binomial_p']:.4f}, pass={w['per_window_pass']})")
        return

    if args.cmd == "sentinel-442":
        from src.topic1_topic5_bridge import WINDOWS_MIN, q1b_sentinel_442
        p = q1b_sentinel_442(
            results_root=results_root,
            artifact_root=artifact_root,
            windows_min=WINDOWS_MIN,
            out_path=out_root / "q1b_442_sentinel.json",
        )
        for wk, w in p["windows"].items():
            print(f"  {wk}: outlier={w['n_outlier']}, main={w['n_main']}, "
                  f"frac_T0 p={w['frac_T0'].get('p',1.0):.4f}, eff={w['frac_T0'].get('effect',0.0):+.3f}")
        return

    if args.cmd == "figures":
        from src.topic1_topic5_bridge import (
            COHORT_GAMMA, PRIMARY_WINDOW,
            figure_q1_cohort_count_x_window, figure_q1_effect_distribution,
            figure_q1_per_subject_strip, figure_q1b_442_sentinel, figure_q3_stratifier,
        )
        fig_dir = out_root / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        cohort_summary = out_root / "cohort_summary.json"
        figure_q1_cohort_count_x_window(cohort_summary, fig_dir / "q1_cohort_count_x_window.png")
        figure_q1_effect_distribution(cohort_summary, fig_dir / "q1_effect_distribution.png")
        figure_q1_per_subject_strip(
            per_subject_dir=out_root / "per_subject",
            cohort=COHORT_GAMMA, band="gamma_ER",
            primary_window=PRIMARY_WINDOW,
            out_path=fig_dir / "q1_per_subject_strip.png",
        )
        figure_q1b_442_sentinel(
            out_root / "q1b_442_sentinel.json",
            fig_dir / "q1b_442_sentinel.png",
        )
        figure_q3_stratifier(cohort_summary, fig_dir / "q1_stratified_swap_silhouette.png")
        print(f"5 figures → {fig_dir}")
        return

    if args.cmd == "q1prime":
        from src.topic1_topic5_bridge import run_q1prime_per_subject
        if args.cohort == "default":
            cohort = ["1073", "1146", "635", "958", "548", "442"]
        else:
            cohort = args.cohort.split(",")
        run_q1prime_per_subject(
            cohort=cohort,
            band=args.band,
            results_root=results_root,
            artifact_root=artifact_root,
            out_dir=out_root / "q1prime_per_subject",
        )
        print(f"q1prime per-subject done; {len(cohort)} subjects → {out_root / 'q1prime_per_subject'}")
        return

    if args.cmd == "q1prime-cohort":
        from src.topic1_topic5_bridge import aggregate_q1prime_cohort
        payload = aggregate_q1prime_cohort(
            per_subject_dir=out_root / "q1prime_per_subject",
            cohort=["1073", "1146", "635", "958", "548", "442"],
            out_path=out_root / "q1prime_cohort_summary.json",
        )
        print(f"verdict: {payload['cohort_judgement']}")
        print(f"  strict positive: {payload['n_strict_positive']}/{payload['n_strict_total']}")
        print(f"  median Cramér V (strict): {payload['median_cramer_v_strict']:.3f}")
        print(f"  median AMI (strict): {payload['median_ami_strict']:.3f}")
        return

    raise NotImplementedError(f"subcommand {args.cmd} pending")


if __name__ == "__main__":
    main()
