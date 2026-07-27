import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/paper_figures/plot_fig6_symmetric_axis_propagation_state_v2_2.py"
)
SPEC = importlib.util.spec_from_file_location("fig6_v22", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_result_panels_render_with_frozen_schemas(tmp_path):
    claim2_patient = tmp_path / "claim2.csv"
    pd.DataFrame(
        {
            "seed_median_next_benefit": [0.1, -0.02, 0.04],
            "seed_median_future_benefit": [0.2, 0.01, -0.03],
        }
    ).to_csv(claim2_patient, index=False)
    claim2 = {
        "status": "complete",
        "claim2_next": "PASS",
        "claim2_future": "PASS",
        "endpoints": [
            {
                "endpoint": "next_set",
                "median_benefit": 0.04,
                "median_ci95_low": 0.01,
                "median_ci95_high": 0.08,
                "bh_fdr_q": 0.02,
                "pass": True,
            },
            {
                "endpoint": "future_first_arrival",
                "median_benefit": 0.01,
                "median_ci95_low": 0.001,
                "median_ci95_high": 0.10,
                "bh_fdr_q": 0.03,
                "pass": True,
            },
        ],
    }

    random_patient = tmp_path / "claim3.csv"
    pd.DataFrame(
        {"seed_median_delta_random_minus_learned": [0.1, -0.01, 0.05]}
    ).to_csv(random_patient, index=False)
    claim3 = {
        "status": "complete",
        "claim3_random_axis": "PASS",
        "median_ci95_low": 0.001,
        "median_ci95_high": 0.10,
    }
    readback_patient = tmp_path / "readback.csv"
    pd.DataFrame(
        {
            "status": ["estimable", "estimable"],
            "abs_axis_cosine": [0.4, 0.8],
        }
    ).to_csv(readback_patient, index=False)
    readback = {"status": "complete"}

    claim4_patient = tmp_path / "claim4.csv"
    pd.DataFrame(
        {
            "seed_median_left_axis_benefit": [0.1, 0.2, -0.01],
            "seed_median_right_axis_benefit": [0.1, 0.05, 0.01],
            "seed_median_M": [-0.02, -0.03, 0.01],
        }
    ).to_csv(claim4_patient, index=False)
    claim4 = {"status": "complete", "claim4_shared_scaffold": "PASS"}

    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    MODULE._panel_c(
        axes[0],
        status=claim2,
        patient_path=claim2_patient,
        preview=False,
    )
    MODULE._panel_d(
        axes[1],
        claim3=claim3,
        random_path=random_patient,
        readback=readback,
        readback_path=readback_patient,
        preview=False,
    )
    MODULE._panel_e(
        axes[2],
        status=claim4,
        patient_path=claim4_patient,
        preview=False,
    )
    MODULE._panel_f(
        axes[3],
        summary={"early_ictal_values_unlocked": False},
        target={
            "endpoint_denominators": {
                "energy_metadata": {"patients": 13, "seizures": 71}
            }
        },
        transfer_path=tmp_path / "missing.csv",
        preview=False,
    )
    output = tmp_path / "panels.png"
    fig.savefig(output)
    plt.close(fig)
    assert output.is_file()
