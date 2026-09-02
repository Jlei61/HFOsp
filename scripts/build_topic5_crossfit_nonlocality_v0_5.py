#!/usr/bin/env python3
"""Build the target-free, cross-fitted patient nonlocality index J."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_topic5_train_only_modes_suffix_null_v0_5 import train_only_modes  # noqa: E402
from src.topic5_lbss_rnn_v0_2 import build_pool_contract  # noqa: E402


OUT_ROOT = ROOT / "results/topic5_multiscale_effective_scaffold_v0_5"
PREFIX_RANKS = 3
INNER_FOLDS = 5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    values = np.asarray(values, dtype=float).ravel()
    weights = np.asarray(weights, dtype=float).ravel()
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float("nan")
    values, weights = values[finite], weights[finite]
    order = np.argsort(values, kind="stable")
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights) / weights.sum()
    return float(values[min(int(np.searchsorted(cumulative, quantile, side="left")), len(values) - 1)])


def top_mass_support(row: np.ndarray, mass: float = 0.90) -> tuple[np.ndarray, np.ndarray]:
    weight = np.abs(np.asarray(row, dtype=float))
    if not np.isfinite(weight).all() or weight.sum() <= 0:
        raise ValueError("H row has no finite positive mass")
    order = np.argsort(-weight, kind="stable")
    cumulative = np.cumsum(weight[order]) / weight.sum()
    count = max(1, int(np.searchsorted(cumulative, mass, side="left")) + 1)
    nodes = order[:count]
    selected = weight[nodes]
    return nodes.astype(int), selected / selected.sum()


def contact_path_distance_matrix(H: np.ndarray, local_mask: np.ndarray, D_mm: np.ndarray) -> np.ndarray:
    local = np.asarray(local_mask, dtype=bool)
    distance = np.asarray(D_mm, dtype=float)
    graph = np.where(local, distance, 0.0)
    node_path = shortest_path(csr_matrix(graph), directed=True, unweighted=False)
    if not np.isfinite(node_path).all():
        raise RuntimeError("local backbone does not provide finite directed reachability")
    supports = [top_mass_support(row) for row in np.asarray(H)]
    n_contacts = len(supports)
    output = np.zeros((n_contacts, n_contacts), dtype=np.float32)
    for source in range(n_contacts):
        source_nodes, source_weight = supports[source]
        for target in range(n_contacts):
            target_nodes, target_weight = supports[target]
            values = node_path[np.ix_(source_nodes, target_nodes)]
            weights = source_weight[:, None] * target_weight[None, :]
            output[target, source] = weighted_quantile(values, weights, 0.10)
    return output


def fit_nonnegative_beta(
    path: np.ndarray,
    relative_latency: np.ndarray,
    event: np.ndarray | None = None,
) -> float:
    """Fit a non-negative distance slope with an unconstrained event intercept."""
    x = np.asarray(path, dtype=float)
    y = np.asarray(relative_latency, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if event is None:
        x = x - x.mean()
        y = y - y.mean()
    else:
        groups = np.asarray(event)[finite]
        for label in np.unique(groups):
            selected = groups == label
            x[selected] -= x[selected].mean()
            y[selected] -= y[selected].mean()
    denominator = float(np.dot(x, x))
    if denominator <= 1e-12:
        return 0.0
    return max(0.0, float(np.dot(x, y) / denominator))


def robust_scale(residual: np.ndarray) -> float:
    value = np.asarray(residual, dtype=float)
    value = value[np.isfinite(value)]
    if not len(value):
        return float("nan")
    center = float(np.median(value))
    scale = 1.4826 * float(np.median(np.abs(value - center)))
    if scale <= 1e-9:
        scale = float(np.std(value))
    return max(scale, 1e-6)


def event_observations(
    ranks: np.ndarray,
    lag_raw: np.ndarray,
    event_indices: np.ndarray,
    contact_xy: np.ndarray,
    contact_path: np.ndarray,
    r_local_mm: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    for event in np.asarray(event_indices, dtype=int):
        row = ranks[event]
        valid = np.flatnonzero(row >= 0)
        prefix = np.flatnonzero((row >= 0) & (row < PREFIX_RANKS))
        suffix = np.flatnonzero(row >= PREFIX_RANKS)
        if prefix.size == 0 or suffix.size == 0:
            continue
        lag = np.asarray(lag_raw[event], dtype=float)
        lag = lag - float(np.min(lag[valid]))
        prefix_latency = float(np.median(lag[prefix]))
        for contact in suffix:
            front = float(np.min(np.linalg.norm(
                contact_xy[prefix] - contact_xy[int(contact)], axis=1
            )))
            path = float(np.min(contact_path[int(contact), prefix]))
            rows.append({
                "event": int(event),
                "contact": int(contact),
                "rank": int(row[contact]),
                "relative_latency": float(lag[contact] - prefix_latency),
                "path_distance_mm": path,
                "front_distance_mm": front,
                "distal": bool(front > float(r_local_mm)),
            })
    return pd.DataFrame(rows)


def chronological_folds(event_indices: np.ndarray, event_time: np.ndarray, n_folds: int) -> dict[int, int]:
    ordered = np.asarray(event_indices, dtype=int)[np.argsort(event_time[event_indices], kind="stable")]
    mapping: dict[int, int] = {}
    for fold, chunk in enumerate(np.array_split(ordered, n_folds)):
        for event in chunk:
            mapping[int(event)] = int(fold)
    return mapping


def event_sensitivities(group: pd.DataFrame) -> tuple[float, float]:
    if len(group) < 2:
        return float("nan"), float("nan")
    path = group.path_distance_mm.to_numpy(float)
    rank = group["rank"].to_numpy(float)
    latency = group.relative_latency.to_numpy(float)
    upper = np.triu_indices(len(path), 1)
    dx = (path[:, None] - path[None, :])[upper]
    dr = (rank[:, None] - rank[None, :])[upper]
    product = dx * dr
    concordant = int(np.sum(product > 0))
    discordant = int(np.sum(product < 0))
    tie_x = int(np.sum((dx == 0) & (dr != 0)))
    tie_y = int(np.sum((dx != 0) & (dr == 0)))
    denominator = np.sqrt(
        (concordant + discordant + tie_x) * (concordant + discordant + tie_y)
    )
    tau = (concordant - discordant) / denominator if denominator else float("nan")
    dy = (latency[:, None] - latency[None, :])[upper]
    comparable = dx != 0
    violations = int(np.sum((dx * dy < 0) & comparable))
    pairs = int(np.sum(comparable))
    return (1.0 - float(tau) if np.isfinite(tau) else float("nan"),
            violations / pairs if pairs else float("nan"))


def crossfit_fit(fit_id: str) -> tuple[dict, pd.DataFrame]:
    cache = OUT_ROOT / "cache" / fit_id
    events = np.load(cache / "events.npz", allow_pickle=False)
    raw = np.load(cache / "events_raw.npz", allow_pickle=False)
    plane = np.load(cache / "plane.npz", allow_pickle=False)
    ranks = events["ranks"]
    split = events["split"]
    if not np.array_equal(ranks, raw["ranks"]):
        raise RuntimeError(f"{fit_id}: raw and Stage-B ranks differ")
    pools = build_pool_contract(plane["D_mm"])
    contact_path = contact_path_distance_matrix(plane["H"], pools.local_mask, plane["D_mm"])
    outer_train = np.flatnonzero(split == 0)
    observations = event_observations(
        ranks, raw["event_lag_raw"], outer_train,
        plane["contacts_xy_mm"], contact_path, pools.r_local_mm,
    )
    if observations.empty:
        raise RuntimeError(f"{fit_id}: no suffix observations")
    fold_of = chronological_folds(outer_train, events["event_abs_time"], INNER_FOLDS)
    observations["inner_fold"] = observations.event.map(fold_of).astype(int)
    scored: list[pd.DataFrame] = []
    beta_rows: list[dict] = []
    for fold in range(INNER_FOLDS):
        eval_events = np.asarray([event for event, value in fold_of.items() if value == fold], dtype=int)
        fit_events = np.asarray([event for event, value in fold_of.items() if value != fold], dtype=int)
        inner_split = np.full(len(ranks), -1, dtype=np.int8)
        inner_split[fit_events] = 0
        inner_split[eval_events] = 1
        modes = train_only_modes(ranks, inner_split)
        train_obs = observations[observations.inner_fold != fold].copy()
        eval_obs = observations[observations.inner_fold == fold].copy()
        train_obs["mode"] = train_obs.event.map(
            {event: int(modes["full_train_mode"][event]) for event in fit_events}
        ).astype(int)
        eval_obs["mode"] = eval_obs.event.map(
            {event: int(modes["prefix_mode"][event]) for event in eval_events}
        ).astype(int)
        pooled = fit_nonnegative_beta(
            train_obs.path_distance_mm, train_obs.relative_latency, train_obs.event
        )
        betas = {0: pooled, 1: pooled}
        mode_specific = True
        for mode in (0, 1):
            subset = train_obs[train_obs["mode"] == mode]
            if subset.event.nunique() < 20 or len(subset) < 40:
                mode_specific = False
                continue
            betas[mode] = fit_nonnegative_beta(
                subset.path_distance_mm, subset.relative_latency, subset.event
            )
        train_prediction = np.asarray([
            betas[int(mode)] * path for mode, path in zip(train_obs["mode"], train_obs.path_distance_mm)
        ])
        sigma = robust_scale(train_obs.relative_latency.to_numpy() - train_prediction)
        eval_obs["beta"] = eval_obs["mode"].map(betas).astype(float)
        eval_obs["prediction"] = eval_obs.beta * eval_obs.path_distance_mm
        eval_obs["residual"] = eval_obs.relative_latency - eval_obs.prediction
        eval_obs["z_early"] = -eval_obs.residual / sigma
        eval_obs["exceedance"] = np.maximum(eval_obs.z_early - 1.0, 0.0)
        scored.append(eval_obs)
        beta_rows.append({
            "fold": fold, "beta_pooled": pooled, "beta_mode0": betas[0],
            "beta_mode1": betas[1], "mode_specific": mode_specific, "sigma": sigma,
        })
    scored_table = pd.concat(scored, ignore_index=True)
    distal = scored_table[scored_table.distal]
    event_burden = distal.groupby("event").exceedance.mean()
    event_old = distal.groupby("event").z_early.median().clip(lower=0)
    sensitivity = [event_sensitivities(group) for _, group in scored_table.groupby("event")]
    rank_tau = np.asarray([value[0] for value in sensitivity], dtype=float)
    violation = np.asarray([value[1] for value in sensitivity], dtype=float)
    front = observations.front_distance_mm.to_numpy(float)
    distal_count = int(observations.distal.sum())
    spread = float(np.quantile(front, 0.9) - np.quantile(front, 0.1))
    x = observations.path_distance_mm.to_numpy(float)
    if np.std(x) <= 1e-12:
        condition = float("inf")
    else:
        standardized = (x - x.mean()) / x.std()
        condition = float(np.linalg.cond(np.column_stack([np.ones(len(x)), standardized])))
    reasons = []
    if distal_count < 40:
        reasons.append("FEWER_THAN_40_FINITE_DISTAL_OBSERVATIONS")
    if spread < 2.0:
        reasons.append("FRONT_DISTANCE_SPREAD_BELOW_2MM")
    if condition > 1e6:
        reasons.append("DESIGN_CONDITION_ABOVE_1E6")
    betas = pd.DataFrame(beta_rows)
    local_unsupported = bool(np.allclose(betas[["beta_pooled", "beta_mode0", "beta_mode1"]], 0.0))
    result = {
        "fit_id": fit_id,
        "status": "NOT_IDENTIFIABLE" if reasons else (
            "LOCAL_WAVE_UNSUPPORTED" if local_unsupported else "IDENTIFIABLE"
        ),
        "not_identifiable_reasons": ";".join(reasons),
        "J_lat_exceedance_burden": float(np.median(event_burden)) if len(event_burden) else float("nan"),
        "J_old_median_sensitivity": float(np.median(event_old)) if len(event_old) else float("nan"),
        "J_rank_1_minus_tau": float(np.nanmedian(rank_tau)),
        "J_pairwise_violation": float(np.nanmedian(violation)),
        "n_outer_train_events": int(len(outer_train)),
        "n_oof_scored_events": int(scored_table.event.nunique()),
        "n_oof_distal_events": int(distal.event.nunique()),
        "n_finite_distal_observations": distal_count,
        "front_distance_q10_mm": float(np.quantile(front, 0.1)),
        "front_distance_q90_mm": float(np.quantile(front, 0.9)),
        "front_distance_spread_mm": spread,
        "design_condition_number": condition,
        "beta_pooled_median": float(betas.beta_pooled.median()),
        "beta_mode0_median": float(betas.beta_mode0.median()),
        "beta_mode1_median": float(betas.beta_mode1.median()),
        "mode_specific_all_folds": bool(betas.mode_specific.all()),
        "robust_sigma_median": float(betas.sigma.median()),
        "r_local_mm": float(pools.r_local_mm),
        "H_mass_support": 0.90,
        "path_support_quantile": 0.10,
        "latency_semantics": "EVENT_INTERNAL_RELATIVE_LAG_PROXY_NOT_CLINICAL_RECRUITMENT",
        "target_values_read": False,
        "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    scored_table["fit_id"] = fit_id
    return result, scored_table


def run_and_persist_fit(fit_id: str) -> dict:
    # Each fit gets a bounded BLAS/OpenMP pool; process-level parallelism is
    # controlled in main so a 140k-event fit cannot monopolise the node.
    with threadpool_limits(limits=4):
        result, scored = crossfit_fit(fit_id)
    root = OUT_ROOT / "nonlocality_oof"
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / f"{fit_id}.csv.gz.tmp"
    scored.to_csv(temporary, index=False, compression="gzip")
    temporary.replace(root / f"{fit_id}.csv.gz")
    sidecar = root / f"{fit_id}.json.tmp"
    sidecar.write_text(json.dumps(result, indent=2, allow_nan=True))
    sidecar.replace(root / f"{fit_id}.json")
    return result


def plot_stage_d(summary: pd.DataFrame, example: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.1), constrained_layout=True)
    axis = axes[0]
    finite = example[np.isfinite(example.path_distance_mm) & np.isfinite(example.relative_latency)]
    color = np.where(finite.distal, "#b74349", "#487fa8")
    axis.scatter(finite.path_distance_mm, finite.relative_latency, s=8, alpha=0.28, c=color)
    axis.set(xlabel="Local-graph path (mm)", ylabel="Relative event lag")
    axis.text(-0.20, 1.05, "A", transform=axis.transAxes, fontsize=15, fontweight="bold")

    axis = axes[1]
    values = summary.J_lat_exceedance_burden.dropna().to_numpy()
    axis.hist(values, bins=min(10, max(4, len(values) // 2)), color="#668fb5", edgecolor="white")
    axis.set(xlabel="Cross-fitted nonlocality J", ylabel="Spatial fits")
    axis.text(-0.20, 1.05, "B", transform=axis.transAxes, fontsize=15, fontweight="bold")

    axis = axes[2]
    colors = summary.status.map({
        "IDENTIFIABLE": "#315b8a", "LOCAL_WAVE_UNSUPPORTED": "#c17b36",
        "NOT_IDENTIFIABLE": "#a7adb2",
    })
    axis.scatter(summary.n_finite_distal_observations, summary.J_lat_exceedance_burden, c=colors, s=25)
    axis.set(xscale="log", xlabel="Distal observations", ylabel="Nonlocality J")
    axis.text(-0.20, 1.05, "C", transform=axis.transAxes, fontsize=15, fontweight="bold")
    output = OUT_ROOT / "figures/stage_d_v0_5_crossfit_nonlocality.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output


def main() -> None:
    if not (OUT_ROOT / "TARGET_PHYSICAL_EMBARGO_ACTIVE.json").exists():
        raise RuntimeError("target physical embargo is not active")
    census = pd.read_csv(OUT_ROOT / "FULL_PARENT_FIT_CENSUS.csv")
    rows = []
    oof_root = OUT_ROOT / "nonlocality_oof"
    oof_root.mkdir(parents=True, exist_ok=True)
    completed = []
    pending = []
    for fit_id in census.fit_id:
        sidecar = oof_root / f"{fit_id}.json"
        table = oof_root / f"{fit_id}.csv.gz"
        if sidecar.exists() and table.exists():
            payload = json.loads(sidecar.read_text())
            if payload.get("producer_sha256") == sha256_file(Path(__file__).resolve()):
                completed.append(payload)
                continue
        pending.append(fit_id)
    rows.extend(completed)
    with ProcessPoolExecutor(max_workers=min(4, max(1, len(pending)))) as executor:
        futures = {executor.submit(run_and_persist_fit, fit_id): fit_id for fit_id in pending}
        for number, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rows.append(result)
            print(json.dumps({
                "completed_fits": len(completed) + number, "total_fits": len(census),
                "fit_id": result["fit_id"], "status": result["status"],
            }), flush=True)
    summary = pd.DataFrame(rows)
    subject_scope = census[["fit_id", "subject", "scope", "geometry_class"]]
    summary = summary.merge(subject_scope, on="fit_id", how="left", validate="one_to_one")
    summary.to_csv(OUT_ROOT / "CROSSFIT_NONLOCALITY_FIT_SUMMARY.csv", index=False)
    patient = summary.groupby("subject", as_index=False).agg(
        J_lat_exceedance_burden=("J_lat_exceedance_burden", "mean"),
        J_old_median_sensitivity=("J_old_median_sensitivity", "mean"),
        J_rank_1_minus_tau=("J_rank_1_minus_tau", "mean"),
        J_pairwise_violation=("J_pairwise_violation", "mean"),
        n_fits=("fit_id", "nunique"),
        all_fits_identifiable=("status", lambda x: bool(np.all(np.asarray(x) != "NOT_IDENTIFIABLE"))),
        any_local_wave_unsupported=("status", lambda x: bool(np.any(np.asarray(x) == "LOCAL_WAVE_UNSUPPORTED"))),
    )
    patient.to_csv(OUT_ROOT / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv", index=False)
    example_path = oof_root / "epilepsiae_1146__shared.csv.gz"
    example = pd.read_csv(example_path if example_path.exists() else next(oof_root.glob("*.csv.gz")))
    figure = plot_stage_d(summary, example)
    payload = {
        "status": "PASS_J_FROZEN_TEMPLATE_ADVANTAGE_PENDING_FORMAL_RNN",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_values_read": False,
        "fits": int(len(summary)),
        "patients": int(patient.subject.nunique()),
        "identifiable_fits": int((summary.status != "NOT_IDENTIFIABLE").sum()),
        "not_identifiable_fits": int((summary.status == "NOT_IDENTIFIABLE").sum()),
        "local_wave_unsupported_fits": int((summary.status == "LOCAL_WAVE_UNSUPPORTED").sum()),
        "primary_J": "median_event_mean_distal_positive_z_exceedance_above_1",
        "crossfit": "five_contiguous_inner_folds_with_train_only_modes",
        "target_values_read": False,
        "figure": str(figure),
        "fit_summary_sha256": sha256_file(OUT_ROOT / "CROSSFIT_NONLOCALITY_FIT_SUMMARY.csv"),
        "patient_summary_sha256": sha256_file(OUT_ROOT / "CROSSFIT_NONLOCALITY_PATIENT_SUMMARY.csv"),
        "producer_sha256": sha256_file(Path(__file__).resolve()),
    }
    temporary = OUT_ROOT / "STAGE_D_J_COMPLETE.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(OUT_ROOT / "STAGE_D_J_COMPLETE.json")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
