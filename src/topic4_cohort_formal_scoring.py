"""Selection, confirmation and cohort inference for the formal Topic 4 cohort.

The endpoints this module computes are frozen in
`config/topic4_data_driven_snn_cohort_formal_v1.json` before any formal patient
score exists.  Nothing here may be re-tuned against formal results.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, spearmanr, wilcoxon

from src.topic4_data_driven_cohort_formal import score_model_ranks_shaft_balanced

# Every loss component is a mean absolute difference between quantities in
# [0, 1], so a loss can never exceed 1.  A readout that cannot be scored at all
# is charged that ceiling instead of being dropped, otherwise a null would be
# built only from its own lucky draws.
UNSCORABLE_LOSS = 1.0


def endpoint_statistic(score: dict) -> float:
    """Weakest-mode loss, with unscorable readouts charged the ceiling."""
    if score.get("status") != "EVALUABLE":
        return float(UNSCORABLE_LOSS)
    return float(score["weakest_mode_loss"])


def natural_k2_verdict(score: dict, *, minimum_events: int,
                       minimum_seed_ami: float) -> dict:
    """Did one network hold two reproducible clusters that both match a mode?

    Kept apart from the supervised endpoint on purpose: assigning model events
    to patient centres can produce a clean two-mode table even when the network
    itself has no two-cluster structure.
    """
    if score.get("status") != "EVALUABLE":
        return {"status": score.get("status"), "same_network_k2": False}
    natural = score["natural_kmeans"]
    counts = np.asarray(natural["cluster_counts"], int)
    matrix = np.asarray(natural["patient_profile_matrix"], float)
    margins = [float(matrix[mode, mode] - matrix[mode, 1 - mode]) for mode in (0, 1)]
    both_populated = bool(np.all(counts >= int(minimum_events)))
    both_positive = bool(np.all(np.asarray(margins) > 0.0) and np.isfinite(margins).all())
    reproducible = float(natural["seed_ami_median"]) >= float(minimum_seed_ami)
    return {
        "status": "EVALUABLE",
        "cluster_counts": counts.tolist(),
        "aligned_template_margins": margins,
        "seed_ami_median": float(natural["seed_ami_median"]),
        "silhouette": natural["silhouette"],
        "both_clusters_populated": both_populated,
        "both_clusters_match_a_distinct_mode": both_positive,
        "clusters_reproducible_across_seeds": reproducible,
        "same_network_k2": bool(both_populated and both_positive and reproducible),
    }


def score_readout(model_ranks: np.ndarray, *, target: dict, contact_names: list[str],
                  patient_centers: np.ndarray, ood_threshold: float,
                  minimum_contacts: int, minimum_events_per_mode: int,
                  kmeans_seed: int, kmeans_n_init: int,
                  include_natural_kmeans: bool = True) -> dict:
    return score_model_ranks_shaft_balanced(
        model_ranks, patient_centers=patient_centers, target=target,
        contact_names=contact_names, patient_ood_threshold=float(ood_threshold),
        minimum_contacts=int(minimum_contacts),
        minimum_events_per_mode=int(minimum_events_per_mode),
        kmeans_seed=int(kmeans_seed), kmeans_n_init=int(kmeans_n_init),
        include_natural_kmeans=bool(include_natural_kmeans),
    )


def within_shaft_null_losses(model_ranks: np.ndarray, permutations: np.ndarray,
                             **kwargs) -> np.ndarray:
    """Re-score the same events after permuting model contact identity."""
    model_ranks = np.asarray(model_ranks, float)
    permutations = np.asarray(permutations, int)
    if permutations.ndim != 2 or permutations.shape[1] != model_ranks.shape[1]:
        raise ValueError("null permutations do not span the model contacts")
    kwargs = {**kwargs, "include_natural_kmeans": False}
    return np.asarray([
        endpoint_statistic(score_readout(model_ranks[:, row], **kwargs))
        for row in permutations
    ], float)


def permutation_p_value(observed: float, null: np.ndarray) -> float:
    null = np.asarray(null, float)
    return float((1 + int(np.sum(null <= float(observed)))) / (1 + len(null)))


def confirm_subject(model_ranks: np.ndarray, *, permutations: np.ndarray,
                    minimum_events: int, minimum_seed_ami: float,
                    **kwargs) -> dict:
    """One subject, one layout, one confirmation network."""
    observed_score = score_readout(model_ranks, **kwargs)
    observed = endpoint_statistic(observed_score)
    null = within_shaft_null_losses(model_ranks, permutations, **kwargs)
    null_median = float(np.median(null))
    return {
        "status": observed_score.get("status"),
        "observed_weakest_mode_loss": observed,
        "null_median": null_median,
        "null_minimum": float(np.min(null)),
        "null_size": int(len(null)),
        "delta_null_median_minus_observed": null_median - observed,
        "permutation_p": permutation_p_value(observed, null),
        "minimum_reachable_p": 1.0 / float(len(null) + 1),
        "subject_endpoint_pass": bool(
            observed_score.get("status") == "EVALUABLE" and observed < null_median
        ),
        "n_readable_events": observed_score.get("n_readable_events"),
        "n_in_distribution_events": observed_score.get("n_in_distribution_events"),
        "ood_fraction": observed_score.get("ood_fraction"),
        "supervised_mode_counts": (
            np.asarray(observed_score["supervised_mode_counts"], int).tolist()
            if "supervised_mode_counts" in observed_score else None
        ),
        "natural_kmeans": natural_k2_verdict(
            observed_score, minimum_events=minimum_events,
            minimum_seed_ami=minimum_seed_ami,
        ),
    }


def _paired_test(deltas: np.ndarray) -> dict:
    deltas = np.asarray(deltas, float)
    finite = deltas[np.isfinite(deltas)]
    positive = int(np.sum(finite > 0.0))
    nonzero = int(np.sum(finite != 0.0))
    result = {
        "n": int(len(finite)),
        "n_positive": positive,
        "median_delta": float(np.median(finite)) if len(finite) else float("nan"),
        "sign_test_p": (
            float(binomtest(positive, nonzero, 0.5).pvalue) if nonzero else float("nan")
        ),
    }
    if nonzero >= 6:
        result["wilcoxon_p"] = float(
            wilcoxon(finite, alternative="two-sided", zero_method="wilcox").pvalue
        )
    else:
        result["wilcoxon_p"] = float("nan")
        result["wilcoxon_note"] = "fewer than six non-zero subjects"
    return result


def cohort_summary(canonical_rows: list[dict], real_rows: list[dict], *,
                   pass_fraction_min: float, alpha: float,
                   per_seed_pass: dict[str, list[bool]] | None = None) -> dict:
    """Subject is the unit; events and network seeds are nested replicates."""
    if not canonical_rows:
        raise ValueError("cohort summary needs at least one canonical subject")
    deltas = np.asarray(
        [row["delta_null_median_minus_observed"] for row in canonical_rows], float,
    )
    passes = np.asarray([bool(row["subject_endpoint_pass"]) for row in canonical_rows])
    events = np.asarray([
        row["n_in_distribution_events"] if row["n_in_distribution_events"] is not None
        else 0 for row in canonical_rows
    ], float)
    primary = _paired_test(deltas)
    pass_fraction = float(passes.mean())

    kept = events <= np.quantile(events, 0.75)
    without_high_yield = _paired_test(deltas[kept])

    def _confound(values: np.ndarray) -> dict:
        if len(np.unique(values)) > 1 and len(values) > 2:
            correlation = spearmanr(deltas, values)
            return {
                "spearman_rho": float(correlation.statistic),
                "p": float(correlation.pvalue),
            }
        return {"spearman_rho": float("nan"), "p": float("nan")}

    event_confound = _confound(events)
    # The canary showed the raw same-minus-crossed margin rising as montages
    # shrink, which is what a rank correlation over six contacts does on its
    # own.  The endpoint here is null-relative, so that inflation should cancel
    # inside each subject; this reports whether it actually did.
    contacts = np.asarray([
        row.get("n_contacts") or 0 for row in canonical_rows
    ], float)
    contact_confound = _confound(contacts)

    # Every subject uses all confirmation seeds, so the honest single-network
    # check is the cohort pass fraction computed inside each seed on its own.
    seed_contribution = {
        seed: {
            "n_pass": int(sum(bool(value) for value in values)),
            "n": len(values),
            "pass_fraction": float(np.mean([bool(value) for value in values]))
            if values else float("nan"),
        }
        for seed, values in sorted((per_seed_pass or {}).items())
    }
    seed_fractions = [
        value["pass_fraction"] for value in seed_contribution.values()
        if np.isfinite(value["pass_fraction"])
    ]

    sensitivity = None
    if real_rows:
        real_lookup = {row["subject_id"]: row for row in real_rows}
        shared = [row for row in canonical_rows if row["subject_id"] in real_lookup]
        paired_canonical = np.asarray(
            [row["delta_null_median_minus_observed"] for row in shared], float,
        )
        paired_real = np.asarray([
            real_lookup[row["subject_id"]]["delta_null_median_minus_observed"]
            for row in shared
        ], float)
        canonical_median = float(np.median(paired_canonical))
        real_median = float(np.median(paired_real))
        sensitivity = {
            "n_paired_subjects": len(shared),
            "canonical_median_delta": canonical_median,
            "real_geometry_median_delta": real_median,
            "real_geometry_pass_fraction": float(np.mean([
                bool(real_lookup[row["subject_id"]]["subject_endpoint_pass"])
                for row in shared
            ])),
            "directions_agree": bool(
                np.sign(canonical_median) == np.sign(real_median)
                and canonical_median != 0.0
            ),
            "real_geometry_test": _paired_test(paired_real),
        }

    natural = [row["natural_kmeans"] for row in canonical_rows]
    same_network = float(np.mean([bool(row.get("same_network_k2")) for row in natural]))

    return {
        "n_subjects": len(canonical_rows),
        "pass_fraction": pass_fraction,
        "pass_fraction_min": float(pass_fraction_min),
        "pass_fraction_met": bool(pass_fraction >= float(pass_fraction_min)),
        "primary_test": primary,
        "primary_significant": bool(
            np.isfinite(primary["wilcoxon_p"]) and primary["wilcoxon_p"] < float(alpha)
            and primary["median_delta"] > 0.0
        ),
        "same_network_k2_fraction": same_network,
        "robustness": {
            "event_count_confound": event_confound,
            "contact_count_confound": contact_confound,
            "without_top_event_quartile": without_high_yield,
            "pass_by_confirmation_seed": seed_contribution,
            "worst_single_seed_pass_fraction": (
                float(min(seed_fractions)) if seed_fractions else None
            ),
            "cohort_survives_every_single_seed": (
                bool(min(seed_fractions) >= float(pass_fraction_min))
                if seed_fractions else None
            ),
        },
        "sensitivity": sensitivity,
    }


def adjudicate(summary: dict, *, same_network_k2_min: float) -> dict:
    """Turn the frozen gates into one status; never soften a failed gate."""
    reasons = []
    if not summary["pass_fraction_met"]:
        reasons.append(
            f"only {summary['pass_fraction']:.0%} of patients beat their own "
            f"shuffled-contact control, below the {summary['pass_fraction_min']:.0%} "
            f"required"
        )
    if not summary["primary_significant"]:
        reasons.append(
            "across patients the model is not reliably closer to held-out data "
            "than the shuffled-contact control"
        )
    if reasons:
        return {"status": "COHORT_MODEL_SUPPORT_INSUFFICIENT", "reasons": reasons}
    if summary["same_network_k2_fraction"] < float(same_network_k2_min):
        return {
            "status": "SAME_NETWORK_K2_INSUFFICIENT",
            "reasons": [
                f"only {summary['same_network_k2_fraction']:.0%} of patients had one "
                f"network hold both propagation modes, below the "
                f"{float(same_network_k2_min):.0%} required"
            ],
        }
    sensitivity = summary.get("sensitivity")
    if sensitivity is not None and not sensitivity["directions_agree"]:
        return {
            "status": "OBSERVATION_LAYOUT_DEPENDENCE_UNRESOLVED",
            "reasons": [
                "the contact-order readout and the real implant geometry point in "
                "opposite directions"
            ],
        }
    return {"status": "COHORT_MODEL_SUPPORT_SUPPORTED", "reasons": []}
