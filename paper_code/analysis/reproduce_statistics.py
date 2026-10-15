"""Recompute paper statistics from the independently copied derived inputs."""
import json
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from paper_paths import INPUT_ROOT
from analysis.src.lagpat_rank_audit import mask_phantom_ranks


def figure1_ranks():
    path = INPUT_ROOT / "results/paper-ready-figure/fig1/data/ce/display_ranks_18.npz"
    with np.load(path) as z:
        ranks, participants, labels = z["ranks"], z["participating"], z["labels"]
    masked = mask_phantom_ranks(ranks, participants, normalize=False)
    np.testing.assert_array_equal(np.isfinite(masked), participants)
    np.testing.assert_allclose(masked[participants] + 1, ranks[participants])
    assert np.bincount(labels).tolist() == [13160, 5030]
    return {"events": ranks.shape[1], "display_channels": ranks.shape[0],
            "cluster_counts": np.bincount(labels).tolist(), "participant_mask_verified": True}


def figure2_gain():
    folder = INPUT_ROOT / "results/interictal_propagation_masked/spatial_information_gain_all_events"
    rows = pd.read_csv(folder / "subject_spatial_information_gain.csv")
    recorded = json.loads((folder / "spatial_information_gain_summary.json").read_text())
    assert len(rows) == 26 and (rows["status"] == "ok").all()
    delta = rows["timing_plus_space_score"] - rows["timing_only_score"]
    np.testing.assert_allclose(delta, rows["spatial_information_gain"], atol=1e-14)
    result = {"patients": len(rows), "positive_gain": int((delta > 0).sum()),
              "median_gain": float(delta.median()),
              "paired_wilcoxon_greater_p": float(wilcoxon(rows["timing_plus_space_score"],
                                                         rows["timing_only_score"], alternative="greater").pvalue)}
    expected = recorded["cohort_statistics"]
    np.testing.assert_allclose(result["median_gain"], expected["median_gain"], atol=1e-14)
    np.testing.assert_allclose(result["paired_wilcoxon_greater_p"], expected["paired_wilcoxon_greater_p"], atol=1e-14)
    assert result["positive_gain"] == expected["n_positive_gain"]
    return result


def figure4_core_count():
    from modeling.scripts.topic4_core_count_optimization.report import trajectory
    folder = INPUT_ROOT / "results/paper-ready-figure/fig4/source"
    snapshot = json.loads((folder / "core_count_input_snapshot.json").read_text())
    arrays = {}
    for case in (snapshot["cases"][0], snapshot["three_core_case"], snapshot["four_core_case"]):
        for key, runs in case["records"].items():
            k = int(key)
            arrays[k] = np.asarray([trajectory(rows, 96)[0] for rows in runs])
    with np.load(folder / "core_count_plot_arrays.npz") as frozen:
        for k, values in arrays.items():
            np.testing.assert_allclose(values, frozen[f"core{k}"], rtol=0, atol=0, equal_nan=True)
    summary = pd.read_csv(folder / "core_count_mean_std_curves.csv")
    for k, values in arrays.items():
        expected = summary[(summary.core_count == k) &
                           (summary.figure == "e1146_one_two_three_four_core_loss")].sort_values("epoch")
        mean = np.mean(values[:, :31, 0], axis=0)
        sd = np.std(values[:, :31, 0], axis=0, ddof=1)
        np.testing.assert_allclose(mean, expected["mean"], atol=1e-12)
        np.testing.assert_allclose(sd, expected["std"], atol=1e-12)
    case = {"arrays": arrays, "summary": {"common_epoch": 31, "display_id": "E1146",
            "subject": "epilepsiae_1146", "prefix_by_restart": {str(k): [31]*4 for k in arrays}}}
    return case, {"core_counts": sorted(arrays), "restarts_per_count": 4, "common_epochs": 31,
                  "raw_score_reduction_matches_frozen_arrays": True, "sample_sd_ddof": 1}
