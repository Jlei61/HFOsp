#!/usr/bin/env python3
"""Patient-first target-free analysis for Topic 5.1 v0.5.

The formal question is whether the distal-prediction benefit of the
task-selected nonlocal arm over a fully macro-matched random-nonlocal refit
increases with a patient's independently cross-fitted nonlocality index.
Seeds are aggregated inside fit and A/B geometry fits are aggregated inside
patient before any inferential statistic is computed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from src.topic5_lbss_rnn_v0_2 import build_pool_contract


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
OLD_ROOT = ROOT / "results/topic5_lbss_full_tissue_rnn_v0_3"
ARMS = (
    "L0_LOCAL_ONLY",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
    "L2M_MACRO_MATCHED_RANDOM_LR",
    "L3_LOCAL_PLUS_LEARNED_LR",
    "C_L3_ORDER_SHUFFLED",
)
LABELS = {
    "L0_LOCAL_ONLY": "Local",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL": "+ nearby",
    "L2M_MACRO_MATCHED_RANDOM_LR": "+ matched nonlocal",
    "L3_LOCAL_PLUS_LEARNED_LR": "+ selected nonlocal",
    "C_L3_ORDER_SHUFFLED": "Suffix null",
}
COLORS = {
    "L0_LOCAL_ONLY": "#72787c",
    "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL": "#6f9aaa",
    "L2M_MACRO_MATCHED_RANDOM_LR": "#b48b55",
    "L3_LOCAL_PLUS_LEARNED_LR": "#b53e4a",
    "C_L3_ORDER_SHUFFLED": "#b9b9b9",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def paired_summary(values: np.ndarray) -> dict:
    data = np.asarray(values, float)
    data = data[np.isfinite(data)]
    tolerance = 1e-9
    nonzero = data[np.abs(data) > tolerance]
    p_greater = 1.0 if len(nonzero) == 0 else float(
        wilcoxon(nonzero, alternative="greater", method="auto").pvalue
    )
    return {
        "n": int(len(data)),
        "median": float(np.median(data)) if len(data) else float("nan"),
        "mean": float(np.mean(data)) if len(data) else float("nan"),
        "n_positive": int(np.sum(data > tolerance)),
        "n_negative": int(np.sum(data < -tolerance)),
        "n_tied": int(np.sum(np.abs(data) <= tolerance)),
        "wilcoxon_p_greater": p_greater,
    }


def holm(raw: list[float]) -> list[float]:
    values = np.asarray(raw, float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(values) - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def seed_removed_sequence_agreement(observed: np.ndarray, generated: list[list[int]]) -> float:
    """Spearman agreement after deleting the supplied rank 0 from both sides."""
    observed = np.asarray(observed, dtype=int)
    generated_order = {
        int(contact): rank
        for rank, rank_set in enumerate(generated[1:])
        for contact in rank_set
    }
    shared = [
        int(contact) for contact in np.flatnonzero(observed > 0)
        if int(contact) in generated_order
    ]
    if len(shared) < 3:
        return float("nan")
    observed_rank = np.asarray([observed[contact] - 1 for contact in shared], float)
    generated_rank = np.asarray([generated_order[contact] for contact in shared], float)
    if np.unique(observed_rank).size < 2 or np.unique(generated_rank).size < 2:
        return float("nan")
    value = spearmanr(observed_rank, generated_rank).statistic
    return float(value) if np.isfinite(value) else float("nan")


def _contract_distance_metrics(out: Path, metrics: dict, path: Path) -> dict:
    """Recompute the locked local/nonlocal estimand from raw held-out rows.

    The inherited v0.3 unit files also contain descriptive q50/q80 bins.  The
    v0.5 primary contract is different: a decision is nonlocal exactly when
    its observed frontier distance exceeds the pre-target ``r_local`` used to
    define the model's candidate pools.  Recomputing here keeps the original
    training outputs immutable while preventing the q80 diagnostic from
    silently becoming the primary denominator.
    """
    plane = np.load(out / "cache" / metrics["fit_id"] / "plane.npz", allow_pickle=False)
    cfg = metrics["config"]
    pools = build_pool_contract(
        plane["D_mm"], float(cfg["density"]), float(cfg["added_fraction"]),
        float(cfg.get("r_local_multiplier", 2.0)),
    )
    rows = json.loads((path.parent / "distance_decisions.json").read_text())
    finite = [row for row in rows if np.isfinite(row["frontier_distance_mm"])]
    local = [row for row in finite if row["frontier_distance_mm"] <= pools.r_local_mm]
    distal = [row for row in finite if row["frontier_distance_mm"] > pools.r_local_mm]

    def summarize(selected: list[dict]) -> tuple[float, float, int]:
        if not selected:
            return float("nan"), float("nan"), 0
        return (
            float(np.mean([row["contact_nll"] for row in selected])),
            float(np.mean([row["top1"] for row in selected])),
            len(selected),
        )

    local_nll, local_top1, local_n = summarize(local)
    distal_nll, distal_top1, distal_n = summarize(distal)
    support = np.asarray([
        (int(row["event_index"]), int(row["rank_index"]),
         round(float(row["frontier_distance_mm"]), 8))
        for row in finite
    ], dtype=np.float64)
    support_sha256 = hashlib.sha256(np.ascontiguousarray(support).view(np.uint8)).hexdigest()
    return {
        "r_local_mm": float(pools.r_local_mm),
        "local_contact_nll": local_nll,
        "local_top1": local_top1,
        "local_n": local_n,
        "distal_contact_nll": distal_nll,
        "distal_top1": distal_top1,
        "distal_n": distal_n,
        "distal_inferential_eligible": bool(distal_n >= 20),
        "distance_decision_support_sha256": support_sha256,
        "q80_distal_contact_nll_descriptive": metrics["distance_bins"]["distal"]["contact_nll"],
        "q80_distal_n_descriptive": metrics["distance_bins"]["distal"]["n"],
    }


def _strict_seed_removed_rollout(out: Path, metrics: dict, path: Path) -> dict:
    """Recompute the accepted post-seed rollout correlation from raw records.

    The inherited training JSON used a legacy helper whose field name said
    ``seed_removed`` even though rank 0 was still included.  Training and
    checkpoint selection never consume that diagnostic, so v0.5 keeps the
    frozen models intact and repairs only the post-training observable here.
    This matches the accepted v0.4 contract: delete the supplied first rank
    from both sequences, then correlate post-seed ranks on common contacts.
    """
    cache = out / "cache" / metrics["fit_id"]
    events_npz = np.load(cache / "events.npz", allow_pickle=False)
    keep = events_npz["split"] >= 0
    ranks = events_npz["ranks"][keep]
    source_index = events_npz["event_source_index"][keep]
    with gzip.open(path.parent / "heldout_rollouts.json.gz", "rt", encoding="utf-8") as stream:
        records = json.load(stream)
    values = []
    for record in records:
        event_index = int(record["kept_event_index"])
        if event_index < 0 or event_index >= len(ranks):
            raise RuntimeError(f"rollout event index outside v0.5 cache: {path}")
        if int(record["event_source_index"]) != int(source_index[event_index]):
            raise RuntimeError(f"rollout/cache event identity mismatch: {path}")
        observed = np.asarray(ranks[event_index], dtype=int)
        value = seed_removed_sequence_agreement(observed, record["generated_rank_sets"])
        if np.isfinite(value):
            values.append(float(value))
    return {
        "median": float(np.median(values)) if values else float("nan"),
        "n_events": int(len(values)),
        "n_rollouts": int(len(records)),
        "contract": "rank0_deleted_common_postseed_contacts_v0_4_accepted",
    }


def _row(out: Path, metrics: dict, path: Path, source: str) -> dict:
    if metrics.get("target_values_read") is not False:
        raise RuntimeError(f"target marker is not false: {path}")
    if not metrics.get("best_checkpoint_eligible"):
        raise RuntimeError(f"checkpoint before structural freeze: {path}")
    bins = metrics["distance_bins"]
    contract_distance = _contract_distance_metrics(out, metrics, path)
    strict_rollout = _strict_seed_removed_rollout(out, metrics, path)
    return {
        "fit_id": metrics["fit_id"], "subject": metrics["subject"],
        "scope": metrics["scope"], "arm": metrics["arm"],
        "seed": int(metrics["seed"]), "source": source,
        "test_contact_nll": metrics["test"]["contact_nll"],
        "test_top1": metrics["test"]["top1"],
        "local_contact_nll": contract_distance["local_contact_nll"],
        "intermediate_contact_nll": bins["intermediate"]["contact_nll"],
        "distal_contact_nll": contract_distance["distal_contact_nll"],
        "local_n": contract_distance["local_n"],
        "intermediate_n": bins["intermediate"]["n"],
        "distal_n": contract_distance["distal_n"],
        "distal_inferential_eligible": contract_distance["distal_inferential_eligible"],
        "r_local_mm": contract_distance["r_local_mm"],
        "distance_decision_support_sha256": contract_distance["distance_decision_support_sha256"],
        "q80_distal_contact_nll_descriptive": contract_distance["q80_distal_contact_nll_descriptive"],
        "q80_distal_n_descriptive": contract_distance["q80_distal_n_descriptive"],
        "rollout_spearman": strict_rollout["median"],
        "rollout_events_strict_n": strict_rollout["n_events"],
        "rollout_events_total_n": strict_rollout["n_rollouts"],
        "rollout_spearman_legacy_includes_seed": metrics["rollout"]["seed_removed_spearman_median"],
        "rollout_length_ratio": metrics["rollout"]["length_ratio_median"],
        "converged": bool(metrics.get("converged", False)),
        "hit_ceiling": bool(metrics.get("hit_ceiling", False)),
        "n_epochs": int(metrics["n_epochs"]),
        "seconds": float(metrics["seconds"]),
        "metrics_sha256": sha256_file(path),
    }


def load_units(out: Path, old: Path) -> pd.DataFrame:
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(out / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    reused = set(reuse.loc[reuse.checkpoint_reuse_eligible.astype(bool), "fit_id"].astype(str))
    records = []
    for fit_id in census.fit_id.astype(str):
        for arm in ARMS:
            for seed in range(3):
                if fit_id in reused and arm in {
                    "L0_LOCAL_ONLY", "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
                    "L3_LOCAL_PLUS_LEARNED_LR",
                }:
                    path = old / "per_fit" / fit_id / arm / f"seed{seed}" / "metrics.json"
                    source = "EXACT_V0_3_REUSE"
                else:
                    path = out / "formal_units" / fit_id / arm / f"seed{seed}" / "metrics.json"
                    source = "V0_5_FORMAL"
                if not path.exists():
                    raise FileNotFoundError(path)
                records.append(_row(out, json.loads(path.read_text()), path, source))
    units = pd.DataFrame(records)
    expected = len(census) * len(ARMS) * 3
    if len(units) != expected or expected != 630:
        raise RuntimeError(f"analysis matrix must contain 630 fit-arm-seeds, found {len(units)}")
    if units.groupby(["fit_id", "arm"]).seed.nunique().ne(3).any():
        raise RuntimeError("a fit-arm does not contain exactly three seeds")
    if units.groupby("fit_id").distance_decision_support_sha256.nunique().ne(1).any():
        raise RuntimeError("arms/seeds do not use the same held-out distance decisions")
    return units


def aggregate_patient(units: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "test_contact_nll", "test_top1", "local_contact_nll",
        "intermediate_contact_nll", "distal_contact_nll", "local_n",
        "intermediate_n", "distal_n", "rollout_spearman",
        "rollout_events_strict_n", "rollout_events_total_n",
        "rollout_spearman_legacy_includes_seed", "rollout_length_ratio", "n_epochs", "seconds",
        "r_local_mm", "q80_distal_contact_nll_descriptive", "q80_distal_n_descriptive",
    ]
    fit = units.groupby(["subject", "fit_id", "scope", "arm"], sort=False)[metrics].median().reset_index()
    patient = fit.groupby(["subject", "arm"], sort=False)[metrics].mean().reset_index()
    if patient.groupby("subject").arm.nunique().ne(len(ARMS)).any():
        raise RuntimeError("patient aggregation lost one or more arms")
    return fit, patient


def posttrain_candidate_exposure(out: Path, old: Path) -> tuple[pd.DataFrame, dict]:
    """Close the frozen L1/L3 opportunity audit using final training graphs."""
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    reuse = pd.read_csv(out / "V0_3_CHECKPOINT_REUSE_AUDIT.csv")
    reused = set(reuse.loc[reuse.checkpoint_reuse_eligible.astype(bool), "fit_id"].astype(str))
    arm_map = {
        "L1": "L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL",
        "L3": "L3_LOCAL_PLUS_LEARNED_LR",
    }
    rows = []
    for fit in census.itertuples():
        for label, arm in arm_map.items():
            for seed in range(3):
                root = (old / "per_fit" / fit.fit_id / arm / f"seed{seed}"
                        if fit.fit_id in reused else
                        out / "formal_units" / fit.fit_id / arm / f"seed{seed}")
                graph = np.load(root / "graph.npz", allow_pickle=False)
                pool = graph["candidate_pool"].astype(bool)
                seen = (
                    graph["initial_added_mask"].astype(bool)
                    | graph["added_mask"].astype(bool)
                    | (graph["exposure_count"] > 0)
                    | (graph["proposal_count"] > 0)
                ) & pool
                proposal_by_source = graph["proposal_count"].sum(axis=0).astype(float)
                eligible_sources = pool.sum(axis=0) > 0
                source_values = proposal_by_source[eligible_sources]
                rows.append({
                    "subject": fit.subject, "fit_id": fit.fit_id, "scope": fit.scope,
                    "arm": label, "seed": seed, "candidate_pool_size": int(pool.sum()),
                    "unique_candidates_ever_activated": int(seen.sum()),
                    "candidate_exposure_fraction": float(seen.sum() / max(1, int(pool.sum()))),
                    "candidate_proposal_total": int(graph["proposal_count"].sum()),
                    "source_proposal_min": float(source_values.min()) if len(source_values) else float("nan"),
                    "source_proposal_max": float(source_values.max()) if len(source_values) else float("nan"),
                    "source_proposal_cv": float(source_values.std() / source_values.mean())
                    if len(source_values) and source_values.mean() > 0 else float("nan"),
                    "graph_sha256": sha256_file(root / "graph.npz"),
                })
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "CANDIDATE_EXPOSURE_POSTTRAIN_FIT_SEED.csv", index=False)
    fit = frame.groupby(["subject", "fit_id", "scope", "arm"], as_index=False).agg(
        candidate_pool_size=("candidate_pool_size", "median"),
        unique_candidates_ever_activated=("unique_candidates_ever_activated", "median"),
        candidate_exposure_fraction=("candidate_exposure_fraction", "median"),
        candidate_proposal_total=("candidate_proposal_total", "median"),
        source_proposal_cv=("source_proposal_cv", "median"),
    )
    pivot = fit.pivot(index=["subject", "fit_id", "scope"], columns="arm")
    ratio = (
        pivot["candidate_exposure_fraction"]["L3"]
        / pivot["candidate_exposure_fraction"]["L1"].clip(lower=1e-12)
    )
    candidate = pd.read_csv(out / "CANDIDATE_CAPACITY_AUDIT.csv").set_index("fit_id")
    audit = pd.DataFrame({
        "subject": ratio.index.get_level_values("subject"),
        "fit_id": ratio.index.get_level_values("fit_id"),
        "scope": ratio.index.get_level_values("scope"),
        "exposure_fraction_ratio_L3_over_L1": ratio.to_numpy(),
    })
    audit["opportunity_severe"] = audit.fit_id.map(
        candidate.candidate_opportunity_severe.astype(bool)
    )
    audit["exposure_severe"] = ~audit.exposure_fraction_ratio_L3_over_L1.between(0.5, 2.0)
    audit["L3_minus_L1_mechanism_eligible"] = ~(audit.opportunity_severe | audit.exposure_severe)
    audit.to_csv(out / "CANDIDATE_EXPOSURE_POSTTRAIN_AUDIT.csv", index=False)
    summary = {
        "fits": len(audit),
        "opportunity_severe_fits": int(audit.opportunity_severe.sum()),
        "exposure_severe_fits": int(audit.exposure_severe.sum()),
        "L3_minus_L1_mechanism_eligible_fits": int(audit.L3_minus_L1_mechanism_eligible.sum()),
        "exposure_ratio_median": float(audit.exposure_fraction_ratio_L3_over_L1.median()),
        "target_values_read": False,
    }
    return audit, summary


def interaction(J: np.ndarray, gain: np.ndarray, *, seed: int = 20260813) -> dict:
    J = np.asarray(J, float)
    gain = np.asarray(gain, float)
    valid = np.isfinite(J) & np.isfinite(gain)
    J, gain = J[valid], gain[valid]
    observed = float(spearmanr(J, gain).statistic)
    rng = np.random.default_rng(seed)
    permutations = 100_000
    null = np.empty(permutations, dtype=np.float32)
    for index in range(permutations):
        null[index] = spearmanr(J, rng.permutation(gain)).statistic
    p_greater = float((1 + np.sum(null >= observed)) / (permutations + 1))
    bootstrap = np.empty(10_000, dtype=np.float32)
    n = len(J)
    for index in range(len(bootstrap)):
        sample = rng.integers(0, n, size=n)
        value = spearmanr(J[sample], gain[sample]).statistic
        bootstrap[index] = value if np.isfinite(value) else np.nan
    leave_one_out = []
    for heldout in range(n):
        keep = np.arange(n) != heldout
        leave_one_out.append(float(spearmanr(J[keep], gain[keep]).statistic))
    return {
        "n": int(n), "spearman_rho": observed,
        "permutation_p_greater": p_greater,
        "permutations": permutations,
        "bootstrap_95_ci": [float(np.nanpercentile(bootstrap, 2.5)), float(np.nanpercentile(bootstrap, 97.5))],
        "leave_one_patient_out_rho_range": [float(np.nanmin(leave_one_out)), float(np.nanmax(leave_one_out))],
    }


def patient_contrasts(patient: pd.DataFrame, J_table: pd.DataFrame, census: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    pivot = patient.pivot(index="subject", columns="arm")
    definitions = {
        "L3_vs_L0_all": pivot["test_contact_nll"]["L0_LOCAL_ONLY"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L1_all": pivot["test_contact_nll"]["L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L2m_all": pivot["test_contact_nll"]["L2M_MACRO_MATCHED_RANDOM_LR"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_suffix_all": pivot["test_contact_nll"]["C_L3_ORDER_SHUFFLED"] - pivot["test_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L0_distal": pivot["distal_contact_nll"]["L0_LOCAL_ONLY"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L1_distal": pivot["distal_contact_nll"]["L1_LOCAL_PLUS_LEARNED_EXTRA_LOCAL"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_L2m_distal": pivot["distal_contact_nll"]["L2M_MACRO_MATCHED_RANDOM_LR"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
        "L3_vs_suffix_distal": pivot["distal_contact_nll"]["C_L3_ORDER_SHUFFLED"] - pivot["distal_contact_nll"]["L3_LOCAL_PLUS_LEARNED_LR"],
    }
    rows, summaries = [], {}
    for label, series in definitions.items():
        summaries[label] = paired_summary(series.to_numpy())
        rows.extend({"subject": subject, "contrast": label, "gain_nats": value}
                    for subject, value in series.items())
    for family in (("L3_vs_L0_all", "L3_vs_L1_all", "L3_vs_L2m_all"),
                   ("L3_vs_L0_distal", "L3_vs_L1_distal", "L3_vs_L2m_distal")):
        adjusted = holm([summaries[key]["wilcoxon_p_greater"] for key in family])
        for key, value in zip(family, adjusted):
            summaries[key]["holm_p_greater_within_claim"] = value
    contrast = pd.DataFrame(rows)
    distal_support = patient.loc[
        patient.arm == "L3_LOCAL_PLUS_LEARNED_LR", ["subject", "distal_n"]
    ].rename(columns={"distal_n": "primary_distal_n"})
    primary = contrast.loc[contrast.contrast == "L3_vs_L2m_distal"].merge(
        J_table[["subject", "J_lat_exceedance_burden"]], on="subject", validate="one_to_one"
    ).merge(distal_support, on="subject", validate="one_to_one")
    primary["distal_inferential_eligible"] = primary.primary_distal_n >= 20
    primary = primary.loc[primary.distal_inferential_eligible].copy()
    primary["geometry_2d"] = primary.subject.map(
        census.groupby("subject").geometry_class.apply(lambda x: bool(np.all(x == "TWO_DIMENSIONAL")))
    )
    summaries["primary_nonlocality_interaction_all"] = interaction(
        primary.J_lat_exceedance_burden, primary.gain_nats
    )
    subset = primary.loc[primary.geometry_2d]
    summaries["primary_nonlocality_interaction_2d_sensitivity"] = interaction(
        subset.J_lat_exceedance_burden, subset.gain_nats, seed=20260814
    )
    return contrast, {"comparisons": summaries, "primary_rows": primary.to_dict("records")}


def plot(patient: pd.DataFrame, contrasts: pd.DataFrame, summary: dict, out: Path) -> None:
    pivot = patient.pivot(index="subject", columns="arm")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.labelsize": 12,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "pdf.fonttype": 42,
        "svg.fonttype": "none",
    })
    figure, axes = plt.subplots(1, 4, figsize=(14.2, 3.15), gridspec_kw={"wspace": 0.62})
    x = np.arange(len(ARMS))
    for subject in pivot.index:
        axes[0].plot(x, [pivot["test_contact_nll"][arm][subject] for arm in ARMS],
                     color="#c8ccce", lw=0.55, alpha=0.65)
    medians = [pivot["test_contact_nll"][arm].median() for arm in ARMS]
    axes[0].plot(x, medians, color="#171717", lw=1.8)
    axes[0].scatter(x, medians, c=[COLORS[arm] for arm in ARMS], s=28, zorder=3)
    axes[0].set_xticks(x, [LABELS[arm] for arm in ARMS], rotation=34, ha="right")
    axes[0].set_ylabel("Held-out contact NLL")

    labels = ("L3_vs_L0_distal", "L3_vs_L1_distal", "L3_vs_L2m_distal")
    for position, label in enumerate(labels):
        values = contrasts.loc[contrasts.contrast == label, "gain_nats"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(values))
        axes[1].scatter(position + jitter, values, s=15, color="#9ca3a6", alpha=0.75)
        axes[1].plot([position - 0.18, position + 0.18], [np.median(values)] * 2,
                     color=COLORS["L3_LOCAL_PLUS_LEARNED_LR"], lw=2.2)
    axes[1].axhline(0, color="#555555", lw=0.8, ls="--")
    axes[1].set_xticks(range(3), ["Local", "+ nearby", "Matched\nnonlocal"])
    axes[1].set_ylabel("L3 distal gain (nats)")

    primary = pd.DataFrame(summary["primary_rows"])
    axes[2].scatter(primary.J_lat_exceedance_burden, primary.gain_nats,
                    s=28, c=np.where(primary.geometry_2d, "#315f8a", "#d28b2d"),
                    edgecolors="white", linewidths=0.35)
    axes[2].axhline(0, color="#777777", lw=0.7, ls="--")
    axes[2].set_xscale("symlog", linthresh=1e-4)
    axes[2].set_xlabel("Cross-fitted nonlocality J")
    axes[2].set_ylabel("L3 − matched nonlocal\n(distal gain, nats)")

    suffix = contrasts.loc[contrasts.contrast == "L3_vs_suffix_all"].gain_nats.to_numpy()
    axes[3].scatter(np.zeros(len(suffix)) + np.linspace(-0.12, 0.12, len(suffix)), suffix,
                    s=17, color="#8f9699", alpha=0.8)
    axes[3].plot([-0.2, 0.2], [np.median(suffix)] * 2, color="#171717", lw=2.2)
    axes[3].axhline(0, color="#555555", lw=0.8, ls="--")
    axes[3].set_xlim(-0.35, 0.35)
    axes[3].set_xticks([0], ["True suffix\nvs reassigned"])
    axes[3].set_ylabel("Order-specific gain (nats)")
    for label, axis in zip("ABCD", axes):
        axis.spines[["top", "right"]].set_visible(False)
        axis.text(-0.18, 1.08, label, transform=axis.transAxes, fontsize=14,
                  fontweight="bold", va="top")
    stem = out / "figures/stage_e_v0_5_interictal_multiscale_scaffold"
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix(f".{suffix}"), dpi=600,
                       bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--old-root", type=Path, default=OLD_ROOT)
    args = parser.parse_args()
    out, old = args.out_root.resolve(), args.old_root.resolve()
    marker = json.loads((out / "STAGE_E_TRAINING_COMPLETE.json").read_text())
    if marker.get("status") != "PASS" or marker.get("formal_units") != 531:
        raise RuntimeError("formal training is not complete")
    units = load_units(out, old)
    _, exposure_summary = posttrain_candidate_exposure(out, old)
    fit, patient = aggregate_patient(units)
    J_table = pd.read_csv(out / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv")
    census = pd.read_csv(out / "FULL_PARENT_FIT_CENSUS.csv")
    contrasts, summary = patient_contrasts(patient, J_table, census)
    units.to_csv(out / "INTERICTAL_PER_FIT_SEED.csv", index=False)
    fit.to_csv(out / "INTERICTAL_PER_FIT.csv", index=False)
    patient.to_csv(out / "INTERICTAL_PER_PATIENT.csv", index=False)
    contrasts.to_csv(out / "INTERICTAL_PATIENT_CONTRASTS.csv", index=False)
    payload = {
        "contract": "topic5_multiscale_interictal_analysis_v0_5",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
        "patients": int(patient.subject.nunique()), "fits": int(fit.fit_id.nunique()),
        "formal_units": 531, "analysis_rows_with_reuse": len(units),
        "unconverged_units": int((~units.converged).sum()),
        "hit_ceiling_units": int(units.hit_ceiling.sum()),
        "candidate_exposure_audit": exposure_summary,
        **summary,
    }
    write_json(out / "INTERICTAL_V0_5_SUMMARY.json", payload)
    plot(patient, contrasts, summary, out)
    with (out / "figures/README.md").open("a") as stream:
        stream.write(
            "\n### stage_e_v0_5_interictal_multiscale_scaffold.png\n\n"
            "A 为 28 位患者五个模型的 held-out contact NLL；B 比较 task-selected nonlocal 相对 local、等容量 nearby 与 macro-matched random nonlocal 的 distal 增量。C 是预注册主检验：cross-fitted nonlocality J 与 L3−L2m distal benefit 的患者级关系；橙色为近一维几何 sensitivity。D 检验真实 suffix 相对跨事件 reassignment null 的增量。\n\n"
            "**关注点**：seed 先在 fit 内聚合，A/B fits 再在患者内聚合；统计单位始终是患者。只有 C 的正向 interaction 支持患者 nonlocality 对 selected-shortcut benefit 的调节。\n"
        )
    write_json(out / "STAGE_E_INTERICTAL_ANALYSIS_COMPLETE.json", {
        "status": "PASS", "target_values_read": False,
        "patients": int(patient.subject.nunique()), "fits": int(fit.fit_id.nunique()),
        "summary_sha256": sha256_file(out / "INTERICTAL_V0_5_SUMMARY.json"),
    })


if __name__ == "__main__":
    main()
