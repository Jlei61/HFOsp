"""Topic 1 × Topic 5 Bridge CLI driver."""

from __future__ import annotations

import argparse
from pathlib import Path


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
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    results_root = repo / "results"
    artifact_root = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
    out_root = results_root / "topic1_topic5_bridge"

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

    raise NotImplementedError(f"subcommand {args.cmd} pending")


if __name__ == "__main__":
    main()
