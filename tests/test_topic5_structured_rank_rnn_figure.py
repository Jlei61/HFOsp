from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_six_panel_figure_renders_for_sealed_negative_gate(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    (analysis / "formal_gate_summary.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "formal_interictal_gate_pass": False,
                "comparison_gate_pass": False,
                "structure_gate_pass": False,
            }
        )
    )
    node_rows = []
    for seed in (1, 2, 3):
        for contact in range(8):
            row = {
                    "subject": "epilepsiae_1146",
                    "seed": seed,
                    "contact_name": f"c{contact}",
                    "observed_participation": 0.2 + 0.08 * contact,
                    "generated_participation": 0.21 + 0.075 * contact,
                    "observed_mean_rank": contact / 7,
                    "generated_mean_rank": min(1.0, contact / 7 + 0.02),
            }
            observed = np.exp(
                -((np.linspace(0, 1, 10) - contact / 7) ** 2) / 0.04
            )
            generated = np.exp(
                -(
                    (
                        np.linspace(0, 1, 10)
                        - min(1.0, contact / 7 + 0.02)
                    )
                    ** 2
                )
                / 0.04
            )
            observed /= observed.sum()
            generated /= generated.sum()
            for bin_index in range(10):
                row[f"observed_rank_bin_{bin_index}"] = observed[bin_index]
                row[f"generated_rank_bin_{bin_index}"] = generated[bin_index]
            node_rows.append(row)
    pd.DataFrame(node_rows).to_csv(
        analysis / "intact_k2_contact_distributions.csv", index=False
    )

    def benefit_rows(groups: list[str], column: str) -> pd.DataFrame:
        rows = []
        for group in groups:
            for metric in ("participation_mae", "rank_wasserstein"):
                for subject in range(34):
                    rows.append(
                        {
                            column: group,
                            "metric": metric,
                            "subject": f"s{subject}",
                            "seed_median_benefit": (
                                0.01 * np.sin(subject + len(group))
                            ),
                        }
                    )
        return pd.DataFrame(rows)

    baselines = ["no_history", "merged_path", "weight_shuffle", "mode_shuffle"]
    lesions = [
        "graph",
        "mode_collapse",
        "inhibition",
        "drop_forward",
        "drop_reverse",
        "drop_dominant_mode",
    ]
    benefit_rows(baselines, "baseline").to_csv(
        analysis / "comparison_patient_seed_medians.csv", index=False
    )
    benefit_rows(lesions, "lesion").to_csv(
        analysis / "lesion_patient_seed_medians.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "baseline": group,
                "metric": metric,
                "pass": False,
            }
            for group in baselines
            for metric in ("participation_mae", "rank_wasserstein")
        ]
    ).to_csv(analysis / "comparison_primary_statistics.csv", index=False)
    pd.DataFrame(
        [
            {"lesion": group, "metric": metric, "pass": False}
            for group in lesions
            for metric in ("participation_mae", "rank_wasserstein")
        ]
    ).to_csv(analysis / "lesion_primary_statistics.csv", index=False)

    dynamics = []
    for metric in (
        "posterior_entropy_normalized",
        "posterior_weighted_excitation",
        "posterior_weighted_inhibition",
    ):
        for progress in np.linspace(0, 1, 11):
            dynamics.append(
                {
                    "progress_bin": progress,
                    "metric": metric,
                    "n_patients": 34,
                    "median": 0.2 + 0.5 * progress,
                    "q25": 0.1 + 0.5 * progress,
                    "q75": 0.3 + 0.5 * progress,
                }
            )
    pd.DataFrame(dynamics).to_csv(
        analysis / "intact_k2_internal_dynamics_cohort.csv", index=False
    )

    out = tmp_path / "figures"
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    subprocess.run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts/paper_figures/plot_fig6_structured_rank_rnn.py"
            ),
            "--root",
            str(tmp_path),
            "--out-dir",
            str(out),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (out / "fig6_structured_rank_rnn.png").exists()
    assert (out / "fig6_structured_rank_rnn.pdf").exists()
    assert (out / "README.md").exists()
