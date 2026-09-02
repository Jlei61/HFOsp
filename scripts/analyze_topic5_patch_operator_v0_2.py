#!/usr/bin/env python3
"""Topology-consensus finite-time propagation operator for Topic 5.2 v0.2.

The C4 result compares response fields produced by perturbing each network along
its own *fitted* future-field hidden direction.  Every arm's direction is fitted
to the same patient-level future-field label, so part of that convergence is
guaranteed by construction.  This analysis removes that dependence.

The operator is built from the frozen Gaussian tissue-patch central differences,
which are geometry-only: no fitted hidden axis, no future-field label, no
progress orientation enters the perturbation direction.

    K_{p,a}(c, i) = mean over frozen reference states of
                    [ l^{+i}_{k+tau, c} - l^{-i}_{k+tau, c} ] / (2 * dose)

with `i` a tissue patch centre and `c` a future contact.  Deviation from the
review's formula: the accumulator stores the mean over states rather than the
median, because the frozen R0 stage already stores state means and the linearity
of the projection then makes the R1 operator exactly reproduce R0's scores.

Four questions, in order:

1. is the operator the same at different event phases, or phase-specific?
2. do real-order topologies converge on it more than the order-shuffled arm?
3. does a consensus built from three topologies predict a held-out topology?
4. does that consensus match the patient's held-out empirical propagation
   transitions, against spatial and cross-patient nulls?
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.propagation_skeleton_geometry import parse_shaft  # noqa: E402
from src.topic5_latent_landscape_v0_2 import atomic_write_csv, atomic_write_json, sha256_file  # noqa: E402
from src.topic5_latent_perturbation_v0_2 import stable_seed  # noqa: E402
from scripts.audit_topic5_latent_control_referenced_v0_2 import _kernel_transport  # noqa: E402
from scripts.freeze_topic5_cross_patient_geometry_mappings_v0_2 import MAPPING  # noqa: E402
from scripts.freeze_topic5_spatial_patch_contract_v0_2 import patch_dir  # noqa: E402
from scripts.run_topic5_latent_pass1_v0_2 import OUT, PARENT  # noqa: E402
from scripts.run_topic5_spatial_patch_response_v0_2 import (  # noqa: E402
    OPERATOR, PATCH_OPERATOR_REVISION, operator_dir,
)
from scripts.summarize_topic5_latent_geometry_v0_2 import holm_adjust, one_sided_summary  # noqa: E402


ANALYSIS_REVISION = "PATCH_OPERATOR_ANALYSIS_R0_TOPOLOGY_CONSENSUS_AND_DATA_LINK"
REAL_ARMS = ("L0", "L1", "L2m", "L3")
CONTROL_ARM = "C-suffix"
PRIMARY_DOSE_INDEX = 1
FUTURE_TAU = (1, 2, 3)
TRANSITION_LAGS = (1, 2, 3)
N_SPATIAL_NULL = 512
MIN_CONTACTS = 5


def pattern_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Centred cosine over the flattened operator, i.e. a Pearson correlation."""
    a, b = np.asarray(left, float).ravel(), np.asarray(right, float).ravel()
    use = np.isfinite(a) & np.isfinite(b)
    if int(use.sum()) < 8:
        return float("nan")
    a, b = a[use] - a[use].mean(), b[use] - b[use].mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator > 1e-12 else float("nan")


def spearman_brown(reliability: float) -> float:
    """Half-length reliability corrected to the full-length operator."""
    if not np.isfinite(reliability) or reliability <= -0.999:
        return float("nan")
    return float(2.0 * reliability / (1.0 + reliability))


def load_cell(row: pd.Series) -> dict[str, np.ndarray]:
    with np.load(operator_dir(row) / "patch_operator.npz", allow_pickle=False) as source:
        mean_operator = np.asarray(source["mean_contact_operator"], float)
        half_operator = np.asarray(source["half_contact_operator"], float)
        doses = np.asarray(source["doses"], float)
    with np.errstate(invalid="ignore"):
        by_phase = np.nanmean(mean_operator[:, :, PRIMARY_DOSE_INDEX][:, :, FUTURE_TAU, :], axis=2)
        pooled = np.nanmean(by_phase, axis=0)
        halves = np.nanmean(
            np.nanmean(half_operator[:, :, :, PRIMARY_DOSE_INDEX][:, :, :, FUTURE_TAU, :], axis=3),
            axis=1,
        )
        low_dose = np.nanmean(np.nanmean(mean_operator[:, :, 0][:, :, FUTURE_TAU, :], axis=2), axis=0)
    return {
        # (contact, centre) so rows are future contacts and columns are tissue patches.
        "operator": pooled.T, "by_phase": np.transpose(by_phase, (0, 2, 1)),
        "halves": np.transpose(halves, (0, 2, 1)), "low_dose": low_dose.T,
        "doses": doses,
    }


def contact_space_operator(operator: np.ndarray, fit_id: str, arm: str, seed: int) -> np.ndarray:
    """Map the tissue-patch axis into contact space with the frozen observation operator.

    `read_weight[i, c]` is how much contact `c` observes the tissue that patch `i`
    perturbs, so the result answers "perturb the tissue contact c' reads from, what
    happens at future contact c".  Only frozen geometry is used.
    """
    row = pd.Series({"fit_id": fit_id, "public_arm": arm, "seed": seed})
    with np.load(patch_dir(row) / "patch_contract.npz", allow_pickle=False) as source:
        directions = np.asarray(source["patch_directions"], float)
    with np.load(PARENT / "cache" / fit_id / "plane.npz", allow_pickle=False) as source:
        observation = np.asarray(source["H"], float)
    read_weight = directions @ observation.T
    read_weight = np.clip(read_weight, 0.0, None)
    total = read_weight.sum(axis=0, keepdims=True)
    normalized = np.divide(read_weight, total, out=np.zeros_like(read_weight), where=total > 1e-12)
    return operator @ normalized


def empirical_transition_operator(fit_id: str) -> tuple[np.ndarray, int]:
    """Held-out follow-within-lag frequency minus a within-event rank-shuffle null.

    The null permutes which contact receives which rank inside an event, so every
    ordered pair of contacts present in that event has the same follow probability:
    the fraction of ordered position pairs in the event's own rank multiset whose
    difference lands in the lag window.  That expectation is exact, so it is
    computed in closed form instead of by resampling.
    """
    with np.load(PARENT / "cache" / fit_id / "events.npz", allow_pickle=False) as source:
        ranks = np.asarray(source["ranks"])
        split = np.asarray(source["split"])
    test = ranks[split == 2]
    n_contacts = test.shape[1]
    follow = np.zeros((n_contacts, n_contacts), float)
    together = np.zeros((n_contacts, n_contacts), float)
    null_sum = np.zeros((n_contacts, n_contacts), float)
    for event in test:
        present = np.flatnonzero(event >= 0)
        if len(present) < 2:
            continue
        values = event[present].astype(int)
        difference = values[None, :] - values[:, None]
        hit = np.isin(difference, TRANSITION_LAGS)
        block = np.ix_(present, present)
        together[block] += 1.0
        follow[block] += hit.astype(float)
        count = len(present)
        expectation = float(hit.sum()) / float(count * (count - 1))
        null_sum[block] += expectation
    observed = np.divide(follow, together, out=np.full_like(follow, np.nan), where=together > 0)
    null_mean = np.divide(null_sum, together, out=np.full_like(null_sum, np.nan), where=together > 0)
    # Rows are the future contact and columns the prefix contact, matching the operator.
    return (observed - null_mean).T, int(np.count_nonzero(split == 2))


def contact_permutations(fit_id: str, names: list[str], xy: np.ndarray) -> dict[str, np.ndarray]:
    """Relabelling nulls applied identically to both operator axes."""
    n = len(names)
    rng_all = np.random.default_rng(stable_seed(fit_id, "operator_all"))
    shafts: dict[str, list[int]] = {}
    for index, name in enumerate(names):
        shafts.setdefault(str(parse_shaft(name)[0]), []).append(index)
    shaft_groups = [np.asarray(values, int) for _, values in sorted(shafts.items())]
    order = np.argsort(np.asarray(xy, float)[:, 0])
    distance_groups = [part for part in np.array_split(order, min(4, max(1, n // 2))) if len(part)]
    families = {"ALL_CONTACT": [np.arange(n)], "WITHIN_SHAFT": shaft_groups,
                "DISTANCE_BIN": distance_groups}
    output: dict[str, np.ndarray] = {}
    for family, groups in families.items():
        rng = rng_all if family == "ALL_CONTACT" else np.random.default_rng(stable_seed(fit_id, family))
        draws = np.tile(np.arange(n), (N_SPATIAL_NULL, 1))
        movable = sum(len(group) for group in groups if len(group) > 1)
        for draw in range(N_SPATIAL_NULL):
            for group in groups:
                if len(group) > 1:
                    draws[draw, group] = rng.permutation(group)
        output[family] = draws
        output[f"{family}_movable"] = np.asarray([movable])
    return output


def off_diagonal_alignment(operator: np.ndarray, transition: np.ndarray) -> float:
    n = operator.shape[0]
    mask = ~np.eye(n, dtype=bool)
    a, b = operator[mask], transition[mask]
    use = np.isfinite(a) & np.isfinite(b)
    if int(use.sum()) < 8 or np.ptp(a[use]) < 1e-12 or np.ptp(b[use]) < 1e-12:
        return float("nan")
    return float(spearmanr(a[use], b[use]).statistic)


def main() -> None:
    status = json.loads((OPERATOR / "PATCH_OPERATOR_STATUS.json").read_text())
    if status.get("status") != "PASS" or status.get("patch_operator_revision") != PATCH_OPERATOR_REVISION:
        raise RuntimeError("the full 630-cell operator extraction must pass first")
    manifest = pd.read_csv(OUT / "CHECKPOINT_MANIFEST.csv")
    mapping = pd.read_csv(MAPPING / "CROSS_PATIENT_GEOMETRY_MAPPING_MANIFEST.csv")

    fit_operators: dict[tuple[str, str], np.ndarray] = {}
    fit_halves: dict[tuple[str, str], np.ndarray] = {}
    phase_rows: list[dict[str, object]] = []
    dose_rows: list[dict[str, object]] = []
    for (fit_id, arm), group in manifest.groupby(["fit_id", "public_arm"], sort=False):
        cells = [load_cell(row) for _, row in group.iterrows()]
        fit_operators[(fit_id, arm)] = np.nanmedian(np.stack([c["operator"] for c in cells]), axis=0)
        fit_halves[(fit_id, arm)] = np.nanmedian(np.stack([c["halves"] for c in cells]), axis=0)
        patient = str(group["patient"].iloc[0])
        for cell, (_, row) in zip(cells, group.iterrows()):
            phases = cell["by_phase"]
            pairs = [
                pattern_similarity(phases[left], phases[right])
                for left in range(len(phases)) for right in range(left + 1, len(phases))
            ]
            phase_rows.append({
                "patient": patient, "fit_id": fit_id, "public_arm": arm, "seed": int(row.seed),
                "phase_pair_similarity": float(np.nanmedian(pairs)),
            })
            dose_rows.append({
                "patient": patient, "fit_id": fit_id, "public_arm": arm, "seed": int(row.seed),
                "dose_similarity": pattern_similarity(cell["operator"], cell["low_dose"]),
            })

    reliability = {
        key: spearman_brown(pattern_similarity(halves[0], halves[1]))
        for key, halves in fit_halves.items()
    }
    fit_ids = sorted({fit for fit, _ in fit_operators})
    patient_of = manifest.drop_duplicates("fit_id").set_index("fit_id")["patient"].to_dict()

    convergence_rows: list[dict[str, object]] = []
    consensus_rows: list[dict[str, object]] = []
    consensus_operator: dict[str, np.ndarray] = {}
    for fit_id in fit_ids:
        real_pairs, real_control = [], []
        for left in range(len(REAL_ARMS)):
            for right in range(left + 1, len(REAL_ARMS)):
                a, b = REAL_ARMS[left], REAL_ARMS[right]
                raw = pattern_similarity(fit_operators[(fit_id, a)], fit_operators[(fit_id, b)])
                denominator = np.sqrt(reliability[(fit_id, a)] * reliability[(fit_id, b)])
                real_pairs.append((raw, raw / denominator if denominator > 1e-6 else np.nan))
        for arm in REAL_ARMS:
            raw = pattern_similarity(fit_operators[(fit_id, arm)], fit_operators[(fit_id, CONTROL_ARM)])
            denominator = np.sqrt(reliability[(fit_id, arm)] * reliability[(fit_id, CONTROL_ARM)])
            real_control.append((raw, raw / denominator if denominator > 1e-6 else np.nan))
        convergence_rows.append({
            "patient": patient_of[fit_id], "fit_id": fit_id,
            "real_pair_similarity": float(np.nanmedian([value[0] for value in real_pairs])),
            "real_to_shuffled_similarity": float(np.nanmedian([value[0] for value in real_control])),
            "real_pair_similarity_corrected": float(np.nanmedian([value[1] for value in real_pairs])),
            "real_to_shuffled_similarity_corrected": float(np.nanmedian([value[1] for value in real_control])),
            "median_reliability_real": float(np.nanmedian([reliability[(fit_id, arm)] for arm in REAL_ARMS])),
            "reliability_shuffled": float(reliability[(fit_id, CONTROL_ARM)]),
        })
        held_out, control_predicted = [], []
        for arm in REAL_ARMS:
            others = [fit_operators[(fit_id, other)] for other in REAL_ARMS if other != arm]
            consensus = np.nanmean(np.stack(others), axis=0)
            held_out.append(pattern_similarity(consensus, fit_operators[(fit_id, arm)]))
            control_predicted.append(pattern_similarity(consensus, fit_operators[(fit_id, CONTROL_ARM)]))
        consensus_rows.append({
            "patient": patient_of[fit_id], "fit_id": fit_id,
            "consensus_predicts_heldout_real": float(np.nanmedian(held_out)),
            "consensus_predicts_shuffled": float(np.nanmedian(control_predicted)),
            "leave_one_out_margin": float(np.nanmedian(held_out) - np.nanmedian(control_predicted)),
        })
        consensus_operator[fit_id] = np.nanmean(
            np.stack([fit_operators[(fit_id, arm)] for arm in REAL_ARMS]), axis=0
        )

    alignment_rows: list[dict[str, object]] = []
    contact_operators: dict[str, np.ndarray] = {}
    transitions: dict[str, np.ndarray] = {}
    for fit_id in fit_ids:
        provenance = json.loads((PARENT / "cache" / fit_id / "provenance.json").read_text())
        names = [str(value) for value in provenance["joint_contacts"]]
        if len(names) < MIN_CONTACTS:
            continue
        with np.load(PARENT / "cache" / fit_id / "plane.npz", allow_pickle=False) as source:
            xy = np.asarray(source["contacts_xy_mm"], float)
        consensus_contact = contact_space_operator(consensus_operator[fit_id], fit_id, REAL_ARMS[0], 0)
        shuffled_contact = contact_space_operator(
            fit_operators[(fit_id, CONTROL_ARM)], fit_id, CONTROL_ARM, 0
        )
        transition, n_test = empirical_transition_operator(fit_id)
        contact_operators[fit_id] = consensus_contact
        transitions[fit_id] = transition
        observed = off_diagonal_alignment(consensus_contact, transition)
        shuffled_observed = off_diagonal_alignment(shuffled_contact, transition)
        permutations = contact_permutations(fit_id, names, xy)
        row: dict[str, object] = {
            "patient": patient_of[fit_id], "fit_id": fit_id, "n_contacts": len(names),
            "n_test_events": n_test, "consensus_alignment": observed,
            "shuffled_arm_alignment": shuffled_observed,
            "consensus_minus_shuffled_arm": observed - shuffled_observed,
        }
        for family in ("ALL_CONTACT", "WITHIN_SHAFT", "DISTANCE_BIN"):
            draws = permutations[family]
            values = [
                off_diagonal_alignment(consensus_contact, transition[np.ix_(order, order)])
                for order in draws
            ]
            row[f"{family.lower()}_null_median"] = float(np.nanmedian(values))
            row[f"{family.lower()}_margin"] = float(observed - np.nanmedian(values))
            row[f"{family.lower()}_movable"] = int(permutations[f"{family}_movable"][0])
        alignment_rows.append(row)

    for row in alignment_rows:
        fit_id = str(row["fit_id"])
        candidates = mapping[mapping.target_fit_id.eq(fit_id)]
        cross, matched_self = [], []
        for candidate in candidates.itertuples(index=False):
            source_fit = str(candidate.source_fit_id)
            if source_fit not in transitions:
                continue
            with np.load(ROOT / str(candidate.mapping_path), allow_pickle=False) as handle:
                weights = np.asarray(handle["weights"], float)
                target_axis = np.asarray(handle["target_axis"], float)
            transported = weights @ transitions[source_fit] @ weights.T
            cross.append({
                "source_patient": candidate.source_patient,
                "score": off_diagonal_alignment(contact_operators[fit_id], transported),
            })
            # The cross-patient arm passes through a one-dimensional kernel on both
            # axes; the same-patient arm must get the same smoothing before the two
            # are differenced, otherwise the margin also measures resolution loss.
            self_weights = _kernel_transport(
                target_axis, target_axis, float(candidate.bandwidth_normalized_axis)
            )
            matched_self.append({
                "source_patient": candidate.source_patient,
                "score": off_diagonal_alignment(
                    contact_operators[fit_id], self_weights @ transitions[fit_id] @ self_weights.T
                ),
            })
        if cross:
            frame = pd.DataFrame(cross).groupby("source_patient", as_index=False).score.median()
            matched = pd.DataFrame(matched_self).groupby("source_patient", as_index=False).score.median()
            cross_median = float(np.nanmedian(frame.score.to_numpy(float)))
            matched_median = float(np.nanmedian(matched.score.to_numpy(float)))
            row["cross_patient_median"] = cross_median
            row["identity_margin"] = float(row["consensus_alignment"] - cross_median)
            row["smoothing_matched_same_patient"] = matched_median
            row["smoothing_matched_identity_margin"] = float(matched_median - cross_median)
            row["n_cross_patients"] = int(np.isfinite(frame.score.to_numpy(float)).sum())
        else:
            row["cross_patient_median"] = float("nan")
            row["identity_margin"] = float("nan")
            row["smoothing_matched_same_patient"] = float("nan")
            row["smoothing_matched_identity_margin"] = float("nan")
            row["n_cross_patients"] = 0

    phase_frame = pd.DataFrame(phase_rows)
    dose_frame = pd.DataFrame(dose_rows)
    convergence = pd.DataFrame(convergence_rows)
    consensus = pd.DataFrame(consensus_rows)
    alignment = pd.DataFrame(alignment_rows)

    def patient_median(frame: pd.DataFrame, column: str) -> np.ndarray:
        return frame.groupby("patient")[column].median().to_numpy(float)

    convergence_endpoints = {
        "raw_margin": one_sided_summary(
            patient_median(convergence.assign(
                raw_margin=convergence.real_pair_similarity - convergence.real_to_shuffled_similarity
            ), "raw_margin"), stable_seed(ANALYSIS_REVISION, "raw_margin"),
        ),
        "reliability_corrected_margin": one_sided_summary(
            patient_median(convergence.assign(
                corrected_margin=convergence.real_pair_similarity_corrected
                - convergence.real_to_shuffled_similarity_corrected
            ), "corrected_margin"), stable_seed(ANALYSIS_REVISION, "corrected_margin"),
        ),
        "leave_one_topology_out_margin": one_sided_summary(
            patient_median(consensus, "leave_one_out_margin"),
            stable_seed(ANALYSIS_REVISION, "leave_one_out"),
        ),
    }
    convergence_adjusted = holm_adjust(
        {key: value["p_one_sided"] for key, value in convergence_endpoints.items()}
    )
    for key, value in convergence_endpoints.items():
        value["p_holm"] = convergence_adjusted[key]
        value["ci95_median_excludes_zero"] = bool(value["ci95_median"][0] > 0.0)
        value["status"] = (
            "SUPPORTED"
            if value["median"] > 0 and value["p_holm"] < 0.05 and value["ci95_median_excludes_zero"]
            else "UNSUPPORTED"
        )

    alignment_endpoints = {
        name: one_sided_summary(
            patient_median(alignment, column), stable_seed(ANALYSIS_REVISION, name),
        )
        for name, column in (
            ("all_contact_margin", "all_contact_margin"),
            ("within_shaft_margin", "within_shaft_margin"),
            ("distance_bin_margin", "distance_bin_margin"),
            ("identity_margin", "identity_margin"),
            ("smoothing_matched_identity_margin", "smoothing_matched_identity_margin"),
            ("consensus_minus_shuffled_arm", "consensus_minus_shuffled_arm"),
        )
    }
    # The primary family is the spatially-controlled margin and the smoothing-matched
    # identity margin, not the two headline numbers with the loosest controls.
    alignment_adjusted = holm_adjust({
        key: alignment_endpoints[key]["p_one_sided"]
        for key in ("within_shaft_margin", "smoothing_matched_identity_margin")
    })
    for key, value in alignment_endpoints.items():
        value["ci95_median_excludes_zero"] = bool(value["ci95_median"][0] > 0.0)
    for key, value in alignment_adjusted.items():
        # Same bar the topology family is held to: a median above zero, a corrected
        # P below 0.05, and a bootstrap median interval that does not cross zero.
        alignment_endpoints[key]["p_holm_primary_family"] = value
        alignment_endpoints[key]["status"] = (
            "SUPPORTED"
            if alignment_endpoints[key]["median"] > 0 and value < 0.05
            and alignment_endpoints[key]["ci95_median_excludes_zero"]
            else "UNSUPPORTED"
        )

    payload = {
        "contract": "topic5_patch_operator_topology_consensus_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_revision": ANALYSIS_REVISION,
        "patch_operator_revision": PATCH_OPERATOR_REVISION,
        "status": "COMPLETE",
        "operator_definition": {
            "perturbation": "GEOMETRY_ONLY_GAUSSIAN_TISSUE_PATCH_CENTRAL_DIFFERENCE",
            "independent_of": "FITTED_HIDDEN_AXES_AND_FUTURE_FIELD_LABEL",
            "state_aggregation": "MEAN_OVER_FROZEN_REFERENCE_STATES",
            "dose": "PRIMARY_0P5_LOCAL_SD", "tau": list(FUTURE_TAU),
            "residual_axis_dependence": (
                "The per-state support gate is evaluated in each arm's own fitted "
                "coordinate space, so which state-patch pairs are eligible is weakly "
                "arm-dependent; the response quantity itself uses no fitted axis."
            ),
        },
        "phase_invariance": {
            "median_phase_pair_similarity": float(phase_frame["phase_pair_similarity"].median()),
            "patient_median": float(
                phase_frame.groupby("patient")["phase_pair_similarity"].median().median()
            ),
            "interpretation_boundary": (
                "A high value means the operator is close to one phase-independent motif; "
                "a low value means it must be reported per phase."
            ),
        },
        "dose_consistency": {
            "median_similarity_half_dose_vs_primary": float(dose_frame["dose_similarity"].median()),
        },
        "topology_convergence": {
            "n_patients": int(convergence["patient"].nunique()),
            "median_real_pair_similarity": float(convergence["real_pair_similarity"].median()),
            "median_real_to_shuffled_similarity": float(convergence["real_to_shuffled_similarity"].median()),
            "median_reliability_real": float(convergence["median_reliability_real"].median()),
            "median_reliability_shuffled": float(convergence["reliability_shuffled"].median()),
            "endpoints": convergence_endpoints,
        },
        "data_link": {
            "n_patients": int(alignment["patient"].nunique()) if len(alignment) else 0,
            "transition_operator": (
                "HELDOUT_TEST_EVENT_FOLLOW_WITHIN_LAG_1_TO_3_MINUS_WITHIN_EVENT_RANK_SHUFFLE_NULL"
            ),
            "contact_space_mapping": "FROZEN_OBSERVATION_OPERATOR_PATCH_READ_WEIGHTS",
            "median_consensus_alignment": float(alignment["consensus_alignment"].median()) if len(alignment) else None,
            "endpoints": alignment_endpoints,
        },
        "not_attempted": {
            "consensus_component_erasure": (
                "Needs a new hidden-space direction and fresh rollouts, which is outside "
                "this frozen-artifact analysis; deferred with its rationale."
            ),
        },
        "target_values_read": False,
    }
    atomic_write_csv(OPERATOR / "OPERATOR_PHASE_INVARIANCE.csv", phase_frame)
    atomic_write_csv(OPERATOR / "OPERATOR_DOSE_CONSISTENCY.csv", dose_frame)
    atomic_write_csv(OPERATOR / "OPERATOR_TOPOLOGY_CONVERGENCE.csv", convergence)
    atomic_write_csv(OPERATOR / "OPERATOR_LEAVE_ONE_OUT_CONSENSUS.csv", consensus)
    atomic_write_csv(OPERATOR / "OPERATOR_DATA_ALIGNMENT.csv", alignment)
    atomic_write_json(OPERATOR / "PATCH_OPERATOR_SUMMARY.json", payload)
    payload["artifact_hashes"] = {
        name: sha256_file(OPERATOR / name) for name in (
            "OPERATOR_PHASE_INVARIANCE.csv", "OPERATOR_DOSE_CONSISTENCY.csv",
            "OPERATOR_TOPOLOGY_CONVERGENCE.csv", "OPERATOR_LEAVE_ONE_OUT_CONSENSUS.csv",
            "OPERATOR_DATA_ALIGNMENT.csv",
        )
    }
    atomic_write_json(OPERATOR / "PATCH_OPERATOR_SUMMARY.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
