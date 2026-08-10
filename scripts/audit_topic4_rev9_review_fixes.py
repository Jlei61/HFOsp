"""Zero-simulation audit of the completed rev9 exploratory experiment.

The producer preserves the original artifacts and writes a separate review
sidecar.  It fixes interpretation and statistical-unit problems without
pretending that post-hoc diagnostics were preregistered acceptance gates.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score

sys.path.insert(0, os.getcwd())
from scripts.aggregate_topic4_rev9_edge_alpha_selection import _load_bundle  # noqa: E402
from scripts.calibrate_topic4_core_field_stage3_joint_observable import (  # noqa: E402
    HELD_OUT_FRAC,
    SPLIT_SEED,
)
from scripts.run_topic4_core_field_stage3_joint_fit import load_reference  # noqa: E402
from scripts.run_topic4_core_field_stage3_profile_round1 import (  # noqa: E402
    PATIENT,
    axial_map,
)
from src.interictal_propagation import load_subject_propagation_events  # noqa: E402
from src.topic4_core_field_profile import (  # noqa: E402
    fit_profile_modes,
    normalized_rank_curve,
    profile_template_similarity,
    sliced_embedding_distance,
    split_by_block,
    transform_rank_curves,
)
from src.topic4_core_field_runner import atomic_write_json  # noqa: E402
from src.topic4_rev9_factorial import (  # noqa: E402
    ARM_ORDER,
    factorial_effects,
    normalized_event_ranks,
    pairwise_precedence,
)
from src.topic4_rev9_local_response import fit_response_slope  # noqa: E402
from src.topic4_rev9_review_audit import (  # noqa: E402
    common_detector_metrics,
    finite_interval,
    mode_evaluability,
    network_mode_summary,
    pareto_minimize_maximize,
    response_map_spearman,
    response_site_adjudication,
)


ROOT = Path("results/topic4_sef_hfo/data_driven_core_field_rev9")
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "review_audit_20260810"
DEFAULT_FACTORIAL = ROOT / "node_edge_factorial/factorial_summary.json"
DEFAULT_FACTORIAL_NPZ = ROOT / "node_edge_factorial/factorial_summary.npz"
DEFAULT_SELECTION = ROOT / "node_edge_calibration/edge_alpha_selection_summary.json"
DEFAULT_OOS = ROOT / "node_edge_calibration/alpha_star_out_of_selection_summary.json"
DEFAULT_CONFIG = Path("config/topic4_rev9_exploratory.json")
DEFAULT_FACTORIAL_CONFIG = Path("config/topic4_rev9_factorial.json")
DEFAULT_LEARNABILITY_DECISION = Path(
    "results/topic4_sef_hfo/data_driven_core_field_rev9_learnability/decision.json")
FIT_CHECKPOINTS = (
    Path("results/topic4_sef_hfo/data_driven_core_field_stage3/"
         "joint_fit_kmeans_rev8/checkpoint_K2_r0.json"),
    Path("results/topic4_sef_hfo/data_driven_core_field_stage3/"
         "joint_fit_kmeans_rev8/checkpoint_K3_r0.json"),
    Path("results/topic4_sef_hfo/data_driven_core_field_stage3/"
         "joint_fit_kmeans_rev8_1/checkpoint_K3_r0.json"),
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return default


def _runtime_provenance():
    """Hash repository modules actually imported by this zero-simulation producer."""
    paths = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py":
            continue
        try:
            relative = path.relative_to(REPO_ROOT)
        except ValueError:
            continue
        paths.add(str(relative))
    paths.add(str(Path(__file__).resolve().relative_to(REPO_ROOT)))
    paths = sorted(paths)
    dirty = _git("status", "--porcelain", "--", *paths)
    return dict(
        git_commit=_git("rev-parse", "HEAD"),
        runtime_modules_dirty=bool(dirty.strip()),
        runtime_module_sha256={path: _sha256(REPO_ROOT / path) for path in paths},
        python_executable=sys.executable,
        python_version=platform.python_version(),
        numpy_version=np.__version__,
        scipy_version=importlib.metadata.version("scipy"),
        sklearn_version=importlib.metadata.version("scikit-learn"),
    )


def _atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json_input(path):
    path = Path(path)
    return json.loads(path.read_text())


def _manifest(paths, worker_dir):
    entries = [dict(path=str(path), sha256=_sha256(path)) for path in paths]
    workers = sorted(Path(worker_dir).glob("*.npz"))
    worker_hashes = [f"{path.name}:{_sha256(path)}" for path in workers]
    digest = hashlib.sha256("\n".join(worker_hashes).encode()).hexdigest()
    return dict(
        immutable_source_artifacts=entries,
        worker_npz_count=int(len(workers)),
        worker_npz_manifest_sha256=digest,
        policy="source artifacts are read-only; corrected interpretations live in this sidecar",
    )


def _patient_input_paths(subject_dir):
    """Mirror the loader preference and retain hashes for every consumed block."""
    subject_dir = Path(subject_dir)
    lagpat = sorted(subject_dir.glob("*_lagPat_withFreqCent.npz"))
    if not lagpat:
        lagpat = sorted(subject_dir.glob("*_lagPat.npz"))
    paths = list(lagpat)
    for path in lagpat:
        if path.name.endswith("_lagPat_withFreqCent.npz"):
            packed = path.with_name(path.name.replace(
                "_lagPat_withFreqCent.npz", "_packedTimes_withFreqCent.npy"))
        else:
            packed = path.with_name(path.name.replace(
                "_lagPat.npz", "_packedTimes.npy"))
        if packed.exists():
            paths.append(packed)
    return sorted(paths)


def _positive_slope(amplitudes, values):
    slope = fit_response_slope(amplitudes, values)["slope"]
    return np.nan if slope is None else float(slope)


def _response_audit(config, *, alpha=0.75):
    instrument = config["small_kick_instrument"]
    amplitudes = np.asarray(instrument["amplitude_multipliers"], float)
    primary_window = np.asarray(
        instrument["primary_window_after_pulse_end_ms"], float)
    shape_amplitude = float(instrument["window_selection_amplitude_multiplier"])
    seeds = [int(value) for value in
             config["edge"]["out_of_selection_and_factorial_seeds"]]
    node_dir = ROOT / "node_edge_calibration/node_out_of_selection_references"
    edge_dir = ROOT / "node_edge_calibration/edge_alpha_star_out_of_selection"
    all_rows = []
    for seed in seeds:
        node_json = node_dir / f"node_alpha_0_seed_{seed}.json"
        node_npz = node_json.with_suffix(".npz")
        edge_json = edge_dir / f"edge_alpha_0p75_seed_{seed}.json"
        edge_npz = edge_json.with_suffix(".npz")
        _, node = _load_bundle(node_json, node_npz, expected_arm="Node")
        edge_payload, edge = _load_bundle(edge_json, edge_npz, expected_arm="Edge")
        if not np.isclose(float(edge_payload["alpha"]), alpha):
            raise RuntimeError("frozen alpha changed in response audit")
        windows = np.asarray(node["windows_after_pulse_end_ms"], float)
        window_index = np.flatnonzero(np.all(np.isclose(
            windows, primary_window), axis=1))
        amplitude_index = np.flatnonzero(np.isclose(
            node["amplitude_multipliers"], shape_amplitude))
        if len(window_index) != 1 or len(amplitude_index) != 1:
            raise RuntimeError("response instrument indices are ambiguous")
        wi, ai = int(window_index[0]), int(amplitude_index[0])
        for site_index, site_id in enumerate(node["site_ids"].astype(str)):
            node_eligible = bool(
                node["site_seed_window_linear_eligible"][0, site_index, wi])
            edge_eligible = bool(
                edge["site_seed_window_linear_eligible"][0, site_index, wi])
            paired = bool(node_eligible and edge_eligible)
            row = dict(
                seed=int(seed), site_id=str(site_id),
                role=str(node["site_roles"][site_index]),
                node_eligible=node_eligible, edge_eligible=edge_eligible,
                paired_eligible=paired,
            )
            if paired:
                source_node = _positive_slope(
                    amplitudes, node["source_positive_per_cell"][
                        0, site_index, :, wi])
                source_edge = _positive_slope(
                    amplitudes, edge["source_positive_per_cell"][
                        0, site_index, :, wi])
                downstream_node = _positive_slope(
                    amplitudes, node["downstream_positive_per_cell"][
                        0, site_index, :, wi])
                downstream_edge = _positive_slope(
                    amplitudes, edge["downstream_positive_per_cell"][
                        0, site_index, :, wi])
                row.update(
                    positive_source_slope_node=source_node,
                    positive_source_slope_edge=source_edge,
                    positive_downstream_slope_node=downstream_node,
                    positive_downstream_slope_edge=downstream_edge,
                    source_gain_ratio=(
                        source_edge / source_node if source_node > 1e-12 else None),
                    downstream_gain_ratio=(
                        downstream_edge / downstream_node
                        if downstream_node > 1e-12 else None),
                    signed_source_slope_node=float(
                        node["source_slopes"][0, site_index, wi]),
                    signed_source_slope_edge=float(
                        edge["source_slopes"][0, site_index, wi]),
                    signed_downstream_slope_node=float(
                        node["downstream_slopes"][0, site_index, wi]),
                    signed_downstream_slope_edge=float(
                        edge["downstream_slopes"][0, site_index, wi]),
                    r90_delta_mm=float(
                        edge["r90_mm"][0, site_index, ai, wi]
                        - node["r90_mm"][0, site_index, ai, wi]),
                    map_rho=response_map_spearman(
                        node["positive_maps_per_cell"][0, site_index, ai, wi],
                        edge["positive_maps_per_cell"][0, site_index, ai, wi]),
                )
            all_rows.append(row)

    table = []
    site_ids = [row["id"] for row in instrument["origins"]]
    for site_id in site_ids:
        rows = [row for row in all_rows if row["site_id"] == site_id]
        adjudication = response_site_adjudication(
            rows, minimum_valid_pairs=10, gain_bounds=(0.8, 1.25),
            maximum_abs_r90_delta_mm=1.0, minimum_map_rho=0.8)
        table.append(dict(
            site_id=site_id, role=rows[0]["role"],
            n_node_eligible=int(sum(row["node_eligible"] for row in rows)),
            n_edge_eligible=int(sum(row["edge_eligible"] for row in rows)),
            **adjudication,
        ))
    field = [row for row in table if row["role"] == "field_component"]
    patterns = [row["diagnostic_pattern"] for row in field]
    return dict(
        alpha=float(alpha),
        formal_status="LOCAL_RESPONSE_EQUIVALENCE_UNRESOLVED",
        reason=(
            "positive-response gain ratio and map-rho adjudication were not frozen "
            "before alpha selection; pooled J and Spearman do not establish equivalence"),
        metric_semantics=dict(
            formal_gain_input="positive kick-minus-sham response over three amplitudes",
            signed_slopes="retained as diagnostics only",
            map_rho="Spearman over union of non-zero positive-response cells",
            posthoc_reference_bands=dict(
                gain_ratio=[0.8, 1.25], abs_r90_delta_mm=1.0, map_rho=0.8,
                minimum_valid_pairs_per_site=10)),
        field_site_diagnostic=(
            "SOURCE_NUCLEATION_FAIL_DOWNSTREAM_PARTIAL_MATCH"
            if any("SOURCE_NUCLEATION_FAIL" in value for value in patterns)
            else "MIXED_LOCAL_RESPONSE_MISMATCH"),
        site_table=table,
        pair_rows=all_rows,
    )


def _structure_audit(selection):
    rows = []
    for summary in sorted(selection["summaries"], key=lambda row: row["alpha"]):
        ratio = summary["structure"]["edge_ratio"]
        admissible = bool(ratio["min"] >= 0.25 and ratio["max"] <= 4.0)
        rows.append(dict(
            alpha=float(summary["alpha"]),
            min_ratio=float(ratio["min"]), max_ratio=float(ratio["max"]),
            p01_ratio=float(ratio["p01"]), p99_ratio=float(ratio["p99"]),
            status=("STRUCTURALLY_ADMISSIBLE" if admissible else
                    "STRUCTURALLY_INADMISSIBLE_EXPLORATORY_ONLY"),
            entered_objective=summary["J_cal"] is not None,
            selected=bool(np.isclose(
                summary["alpha"], selection["selection"]["alpha_star"])),
        ))
    return dict(
        alpha_reference_role="RESPONSE_OBJECTIVE_SELECTED_CANDIDATE",
        alpha_reference=float(selection["selection"]["alpha_star"]),
        candidate_table=rows,
        correction=(
            "alpha=4 was simulated under the exploratory warning policy but violates "
            "the 0.25-4 reference band; it is retained only as a gray diagnostic and "
            "did not determine alpha=0.75"),
    )


def _trace_threshold_free(trace, bin_width_ms):
    trace = np.asarray(trace, float)
    return dict(
        mean_active_fraction=float(trace.mean()),
        p95_active_fraction=float(np.quantile(trace, 0.95)),
        peak_active_fraction=float(trace.max(initial=0.0)),
        integrated_activity_fraction_ms=float(trace.sum() * bin_width_ms),
    )


def _detector_audit(factorial, factorial_config):
    seeds = [int(value) for value in factorial["seeds"]]
    worker_dir = Path(factorial_config["output_root"]) / "workers"
    records = {arm: [] for arm in ARM_ORDER}
    node_thresholds = []
    for arm in ARM_ORDER:
        slug = arm.lower().replace("+", "_")
        for seed in seeds:
            payload = _json_input(worker_dir / f"{slug}_seed{seed}.json")
            with np.load(worker_dir / f"{slug}_seed{seed}.npz", allow_pickle=False) as z:
                trace = np.asarray(z["active_fraction"], float)
                bin_width = float(z["bin_width_ms"])
            threshold = float(payload["simulation"]["detector_threshold"])
            if arm == "Node":
                node_thresholds.append(threshold)
            records[arm].append(dict(
                seed=int(seed), trace=trace, bin_width_ms=bin_width,
                original_threshold=threshold,
                threshold_free=_trace_threshold_free(trace, bin_width)))
    central = float(np.median(node_thresholds))
    sensitivities = []
    arrays = dict()
    for multiplier in (0.8, 1.0, 1.2):
        threshold = central * multiplier
        per_arm = {}
        rates = {}
        for arm in ARM_ORDER:
            rows = [common_detector_metrics(
                row["trace"], row["bin_width_ms"], threshold)
                    for row in records[arm]]
            values = np.asarray([row["event_rate_hz"] for row in rows], float)
            rates[arm] = values
            per_arm[arm] = dict(
                event_rate_hz=finite_interval(
                    values, seed=20260812 + int(100 * multiplier), repeats=2000),
                total_events=int(sum(row["n_events"] for row in rows)),
                time_above_fraction=finite_interval(
                    [row["time_above_fraction"] for row in rows],
                    seed=20260822 + int(100 * multiplier), repeats=2000),
                integrated_excess_fraction_ms=finite_interval(
                    [row["integrated_excess_fraction_ms"] for row in rows],
                    seed=20260832 + int(100 * multiplier), repeats=2000),
            )
        effects = factorial_effects(
            rates, seed=20260842 + int(100 * multiplier), repeats=2000)
        tag = f"m{multiplier:g}".replace(".", "p")
        arrays[f"common_detector_{tag}_event_rate_hz"] = np.stack(
            [rates[arm] for arm in ARM_ORDER]).astype(np.float32)
        sensitivities.append(dict(
            multiplier=float(multiplier), threshold=float(threshold),
            arm_summaries=per_arm, paired_factorial_event_rate=effects))
    threshold_free = {}
    for arm in ARM_ORDER:
        threshold_free[arm] = {}
        for key in records[arm][0]["threshold_free"]:
            threshold_free[arm][key] = finite_interval(
                [row["threshold_free"][key] for row in records[arm]],
                seed=20260900 + len(threshold_free[arm]), repeats=2000)
    return dict(
        status="POSTHOC_COMMON_DETECTOR_SENSITIVITY_COMPLETE",
        limitation=(
            "the absolute threshold was not frozen before the four-arm run; this "
            "zero-simulation re-detection supports scalar burden comparisons only, "
            "because new event-wise electrode ranks require rerunning the SNN"),
        central_threshold=central,
        central_threshold_origin=(
            "posthoc median of the 12 Node arm-specific thresholds; applied unchanged "
            "to all arms"),
        sensitivities=sensitivities,
        threshold_free=threshold_free,
    ), arrays


def _patient_training_events(reference, grid, patient_prototypes, contact_names):
    data = load_subject_propagation_events(PATIENT)
    names = np.asarray(data["channel_names"]).astype(str)
    name_to_index = {name: index for index, name in enumerate(names)}
    missing = [name for name in contact_names if name not in name_to_index]
    if missing:
        raise RuntimeError(f"patient events miss model contacts: {missing}")
    axial = axial_map()
    grid = np.asarray(grid, float)
    curves, ranks, blocks = [], [], []
    bools = np.asarray(data["bools"], bool)
    raw_ranks = np.asarray(data["ranks"], float)
    block_ids = np.asarray(data["block_ids"])
    for event_index in range(raw_ranks.shape[1]):
        participating = np.flatnonzero(bools[:, event_index])
        rank_dict = {
            names[index]: float(raw_ranks[index, event_index])
            for index in participating
        }
        curve = normalized_rank_curve(rank_dict, axial, grid=grid)
        if curve is None:
            continue
        row = np.full(len(contact_names), np.nan)
        for contact_index, name in enumerate(contact_names):
            source_index = name_to_index[name]
            if bools[source_index, event_index]:
                row[contact_index] = raw_ranks[source_index, event_index]
        curves.append(curve)
        ranks.append(row)
        blocks.append(block_ids[event_index])
    curves = np.asarray(curves, float)
    ranks = np.asarray(ranks, float)
    blocks = np.asarray(blocks)
    train_index, _ = split_by_block(blocks, HELD_OUT_FRAC, SPLIT_SEED)
    train_curves = curves[train_index]
    train_ranks = ranks[train_index]
    modes = fit_profile_modes(train_curves, reference)
    similarity = profile_template_similarity(modes["prototypes"], patient_prototypes)
    raw, target = linear_sum_assignment(-similarity)
    raw_for_target = np.empty(2, int)
    raw_for_target[target] = raw
    raw_to_target = np.empty(2, int)
    raw_to_target[raw_for_target] = np.arange(2)
    labels = raw_to_target[np.asarray(modes["labels"], int)]
    aligned = np.asarray(modes["prototypes"], float)[raw_for_target]
    if not np.allclose(aligned, patient_prototypes, atol=1e-7, rtol=1e-7):
        raise RuntimeError("reconstructed patient-training modes changed")
    return dict(
        curves=train_curves, ranks=train_ranks, labels=labels,
        counts=np.bincount(labels, minlength=2),
        n_train_blocks=int(len(np.unique(blocks[train_index]))),
    )


def _safe_spearman(left, right):
    left = np.asarray(left, float)
    right = np.asarray(right, float)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or np.ptp(left[valid]) <= 0 or np.ptp(right[valid]) <= 0:
        return None
    value = float(spearmanr(left[valid], right[valid]).statistic)
    return None if not np.isfinite(value) else value


def _mode_descriptors(curves, ranks):
    curves = np.asarray(curves, float)
    ranks = np.asarray(ranks, float)
    normalized = normalized_event_ranks(ranks)
    recruitment = np.mean(np.isfinite(ranks), axis=0)
    mean_rank = np.full(ranks.shape[1], np.nan)
    for index in range(ranks.shape[1]):
        finite = normalized[:, index][np.isfinite(normalized[:, index])]
        if len(finite):
            mean_rank[index] = finite.mean()
    precedence, support = pairwise_precedence(ranks)
    return dict(
        prototype=np.mean(curves, axis=0), recruitment=recruitment,
        mean_rank=mean_rank, precedence=precedence,
        precedence_support=support,
    )


def _descriptor_comparison(model_curves, model_ranks, patient_curves,
                           patient_ranks, reference, *, seed):
    if len(model_curves) < 2:
        return dict(status="INSUFFICIENT", n_model_events=int(len(model_curves)))
    model = _mode_descriptors(model_curves, model_ranks)
    patient = _mode_descriptors(patient_curves, patient_ranks)
    recruitment_valid = np.isfinite(model["recruitment"]) & np.isfinite(patient["recruitment"])
    rank_valid = np.isfinite(model["mean_rank"]) & np.isfinite(patient["mean_rank"])
    upper = np.triu(np.ones_like(model["precedence"], bool), 1)
    precedence_valid = (
        upper & np.isfinite(model["precedence"]) & np.isfinite(patient["precedence"])
        & (model["precedence_support"] >= 3)
        & (patient["precedence_support"] >= 20))
    model_z = transform_rank_curves(model_curves, reference)
    patient_z = transform_rank_curves(patient_curves, reference)
    distance = sliced_embedding_distance(
        model_z, patient_z, reference["directions"])
    rng = np.random.default_rng(int(seed))
    floor = []
    for _ in range(200):
        selected = rng.choice(
            len(patient_z), size=len(model_z), replace=len(model_z) > len(patient_z))
        floor.append(sliced_embedding_distance(
            patient_z[selected], patient_z, reference["directions"]))
    return dict(
        status="DESCRIPTIVE",
        n_model_events=int(len(model_curves)),
        n_patient_train_events=int(len(patient_curves)),
        recruitment_rmse=float(np.sqrt(np.mean(
            (model["recruitment"][recruitment_valid]
             - patient["recruitment"][recruitment_valid]) ** 2))),
        mean_rank_mae=float(np.mean(np.abs(
            model["mean_rank"][rank_valid] - patient["mean_rank"][rank_valid]))),
        mean_rank_spearman=_safe_spearman(
            model["mean_rank"], patient["mean_rank"]),
        precedence_mae=(None if not precedence_valid.any() else float(np.mean(np.abs(
            model["precedence"][precedence_valid]
            - patient["precedence"][precedence_valid])))),
        n_precedence_pairs=int(precedence_valid.sum()),
        mean_profile_spearman=_safe_spearman(
            model["prototype"], patient["prototype"]),
        event_cloud_distance=float(distance),
        patient_matched_floor_95=[
            float(value) for value in np.quantile(floor, [0.025, 0.5, 0.975])],
    )


def _hierarchical_profile_interval(curves, labels, ood, seed_ids, seeds,
                                   patient_prototypes, *, seed, repeats=500):
    curves = np.asarray(curves, float)
    labels = np.asarray(labels, int)
    ood = np.asarray(ood, bool)
    seed_ids = np.asarray(seed_ids, int)
    rng = np.random.default_rng(int(seed))
    observed, draws = [], [[], []]
    for mode in (0, 1):
        selected = (labels == mode) & ~ood
        observed.append(_safe_spearman(
            curves[selected].mean(axis=0) if selected.any() else np.full(curves.shape[1], np.nan),
            patient_prototypes[mode]))
    for _ in range(int(repeats)):
        sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
        for mode in (0, 1):
            pieces = []
            for network_seed in sampled_seeds:
                selected = np.flatnonzero(
                    (seed_ids == network_seed) & (labels == mode) & ~ood)
                if len(selected):
                    pieces.append(curves[rng.choice(
                        selected, size=len(selected), replace=True)])
            value = (None if not pieces else _safe_spearman(
                np.concatenate(pieces).mean(axis=0), patient_prototypes[mode]))
            draws[mode].append(np.nan if value is None else value)
    output = []
    for mode in (0, 1):
        values = np.asarray(draws[mode], float)
        values = values[np.isfinite(values)]
        output.append(dict(
            estimate=observed[mode], n_bootstrap_valid=int(len(values)),
            interval_95=([None, None] if not len(values) else
                         np.quantile(values, [0.025, 0.975]).tolist())))
    return output


def _leave_one_network_out_kmeans(curves, seed_ids, frozen_labels, ood,
                                  seeds, patient_prototypes, reference):
    curves = np.asarray(curves, float)
    embedded = transform_rank_curves(curves, reference)
    rows = []
    for held_seed in seeds:
        train = (seed_ids != held_seed) & ~ood
        test = (seed_ids == held_seed) & ~ood
        if train.sum() < 4 or test.sum() < 2:
            continue
        model = KMeans(n_clusters=2, random_state=0, n_init=100).fit(embedded[train])
        train_labels = model.labels_
        raw_prototypes = np.asarray([
            curves[train][train_labels == mode].mean(axis=0) for mode in (0, 1)])
        similarity = profile_template_similarity(raw_prototypes, patient_prototypes)
        raw, target = linear_sum_assignment(-similarity)
        raw_to_target = np.empty(2, int)
        raw_to_target[raw] = target
        predicted = raw_to_target[model.predict(embedded[test])]
        observed = frozen_labels[test]
        rows.append(dict(
            held_seed=int(held_seed), n_test=int(test.sum()),
            accuracy=float(np.mean(predicted == observed)),
            ami=float(adjusted_mutual_info_score(observed, predicted)),
            observed_counts=np.bincount(observed, minlength=2).tolist(),
            predicted_counts=np.bincount(predicted, minlength=2).tolist(),
        ))
    return dict(
        n_held_networks=int(len(rows)), per_network=rows,
        accuracy_median=(None if not rows else float(np.median(
            [row["accuracy"] for row in rows]))),
        ami_median=(None if not rows else float(np.median(
            [row["ami"] for row in rows]))),
    )


def _factorial_mode_audit(factorial, arrays, reference, patient):
    seeds = np.asarray(factorial["seeds"], int)
    patient_counts = np.asarray(patient["counts"], int)
    patient_b = float(patient_counts[1] / patient_counts.sum())
    patient_prototypes = np.asarray(arrays["patient_train_mode_prototypes"], float)
    output = {}
    for arm_index, arm in enumerate(ARM_ORDER):
        slug = arm.lower().replace("+", "_")
        curves = np.asarray(arrays[f"{slug}_curves"], float)
        ranks = np.asarray(arrays[f"{slug}_ranks"], float)
        seed_ids = np.asarray(arrays[f"{slug}_seed_ids"], int)
        labels = np.asarray(arrays[f"{slug}_frozen_labels"], int)
        ood = np.asarray(arrays[f"{slug}_frozen_ood"], bool)
        network = network_mode_summary(
            labels, ood, seed_ids, seeds, duration_s=8.0,
            patient_mode_b_fraction=patient_b,
            bootstrap_seed=20261000 + 100 * arm_index,
            bootstrap_repeats=2000)
        id_counts = np.bincount(labels[~ood], minlength=2)
        evaluability = mode_evaluability(id_counts, float(np.mean(ood)))
        descriptors = []
        for mode in (0, 1):
            model_selected = (labels == mode) & ~ood
            patient_selected = patient["labels"] == mode
            descriptors.append(_descriptor_comparison(
                curves[model_selected], ranks[model_selected],
                patient["curves"][patient_selected], patient["ranks"][patient_selected],
                reference, seed=20261100 + 10 * arm_index + mode))
        in_distribution = ~ood
        identity_ami = (None if in_distribution.sum() < 2 else float(
            adjusted_mutual_info_score(seed_ids[in_distribution], labels[in_distribution])))
        output[arm] = dict(
            patient_mode_matrix_status=evaluability,
            network_mode_repertoire=network,
            cluster_label_network_identity_ami=identity_ami,
            leave_one_network_out_kmeans=_leave_one_network_out_kmeans(
                curves, seed_ids, labels, ood, seeds,
                patient_prototypes, reference),
            hierarchical_profile_spearman_95=(
                _hierarchical_profile_interval(
                    curves, labels, ood, seed_ids, seeds, patient_prototypes,
                    seed=20261200 + arm_index)
                if evaluability["status"] == "EVALUABLE" else None),
            mode_conditioned_readout=dict(
                mode_a=descriptors[0], mode_b=descriptors[1],
                layers=["recruitment", "precedence", "mean_profile", "event_cloud"]),
        )
    return dict(
        patient_training_reference=dict(
            n_events=int(patient_counts.sum()), counts=patient_counts.tolist(),
            mode_b_fraction=patient_b,
            n_recording_blocks=int(patient["n_train_blocks"]),
            heldout_readout_computed=False),
        arms=output,
    )


def _candidate_landscape(selection_record, final_confirmation,
                         learnability_decision=None):
    candidates, executions = [], []
    for phase_index, path in enumerate(FIT_CHECKPOINTS):
        checkpoint = _json_input(path)
        history = checkpoint.get("history", [])
        generations = sorted({int(row["generation"]) for row in history})
        executions.append(dict(
            phase_index=int(phase_index), path=str(path), sha256=_sha256(path),
            K=int(checkpoint["K"]), restart=int(checkpoint["restart"]),
            popsize=int(checkpoint["popsize"]),
            n_candidates=int(len(history)), generations=generations,
            stop_reason=checkpoint.get("stop_reason"),
            sigma0=(None if not checkpoint.get("generation_summary") else
                    float(checkpoint["generation_summary"][0]["sigma"])),
            sigma_final=float(checkpoint["optimizer"]["sigma"]),
            rotating_network_seed_block=bool(len({
                tuple(row["seeds"]) for row in checkpoint.get("generation_summary", [])
            }) > 1),
        ))
        for candidate_index, row in enumerate(history):
            mode = row.get("mode", {})
            if mode.get("status") != "ok" or mode.get("matched_correlations") is None:
                continue
            matched = np.asarray(mode["matched_correlations"], float)
            candidates.append(dict(
                phase_index=int(phase_index), K=int(checkpoint["K"]),
                restart=int(checkpoint["restart"]),
                candidate_index=int(candidate_index), generation=int(row["generation"]),
                support_eligible=bool(mode.get("support_eligible", False)),
                sign_consistent=bool(mode.get("matrix_sign_consistent", False)),
                mode_a=float(matched[0]), mode_b=float(matched[1]),
                worst_mode=float(np.min(matched)), matched_mean=float(np.mean(matched)),
                distance=None if row.get("distance") is None else float(row["distance"]),
                mode_loss=(None if mode.get("mode_matrix_loss") is None else
                           float(mode["mode_matrix_loss"])),
                joint_loss=(None if row.get("joint_loss") is None else
                            float(row["joint_loss"])),
                min_cluster_count=int(mode.get("min_cluster_count", 0)),
                minority_fraction=float(mode.get("minority_fraction", 0.0)),
                n_usable=int(row.get("n_usable", 0)),
            ))
    eligible_index = [index for index, row in enumerate(candidates)
                      if row["support_eligible"] and row["distance"] is not None]
    costs = np.asarray([candidates[index]["distance"] for index in eligible_index], float)
    benefits = np.asarray([candidates[index]["worst_mode"] for index in eligible_index], float)
    pareto = pareto_minimize_maximize(costs, benefits)
    for index, value in zip(eligible_index, pareto):
        candidates[index]["pareto_distance_worst_mode"] = bool(value)
    for index, row in enumerate(candidates):
        row.setdefault("pareto_distance_worst_mode", False)
    supported = [row for row in candidates if row["support_eligible"]]
    best_worst = max(supported, key=lambda row: row["worst_mode"])
    selected_mode = selection_record["selected_candidate"]["selection_metrics"]["mode"]
    selected_matched = np.asarray(selected_mode["matched_correlations"], float)
    final_candidate = final_confirmation["candidates"][0]
    final_mode = final_candidate["confirm"]["kmeans_data_consistency"]
    final_matched = np.asarray(final_mode["matched_correlations"], float)
    controls = final_confirmation["kmeans_controls"]
    control_rows = {}
    for name in ("hand_placed_two_cores", "stage2_filament"):
        row = controls[name]
        matched = np.asarray(row["matched_correlations"], float)
        distance_key = ("hand_placed_two_cores" if name == "hand_placed_two_cores"
                        else "stage2_filament")
        control_rows[name] = dict(
            mode_a=float(matched[0]), mode_b=float(matched[1]),
            worst_mode=float(matched.min()),
            distance=float(final_confirmation["optimization_controls_n20"][
                distance_key]["median"]),
        )
    l0 = None if learnability_decision is None else learnability_decision.get(
        "target_objective")
    if l0 is None:
        objective_evidence = None
        objective_text = (
            "SUPPORTED_PROBLEM: the scalar matrix RMSE averages modes and does not "
            "explicitly protect the weakest mode")
    else:
        mode_a = l0["old_objective_associations"]["mode_a_loss"]
        objective_evidence = l0
        objective_text = (
            "SUPPORTED_PROBLEM: L0 found no association between old joint loss and "
            f"mode-A loss (Spearman rho={mode_a['rho']:.3f}, p={mode_a['pvalue']:.3g}); "
            "no fit-library or selection-evaluated candidate dominated the selected "
            "candidate, so a selection miss is not established")
    return dict(
        status="REV8_CANDIDATE_LANDSCAPE_RESCORING_COMPLETE",
        executions=executions, candidates=candidates,
        n_mode_evaluable_candidates=int(len(candidates)),
        n_supported_candidates=int(len(supported)),
        n_supported_pareto_candidates=int(sum(
            row["pareto_distance_worst_mode"] for row in supported)),
        best_supported_worst_mode_candidate=best_worst,
        selected_candidate=dict(
            mode_a=float(selected_matched[0]), mode_b=float(selected_matched[1]),
            worst_mode=float(selected_matched.min()),
            distance=float(selection_record["selected_candidate"][
                "selection_metrics"]["distance"])),
        final_unseen_candidate=dict(
            mode_a=float(final_matched[0]), mode_b=float(final_matched[1]),
            worst_mode=float(final_matched.min()),
            distance=float(final_candidate["confirm"][
                "bootstrap_distance_patient_train"]["median"])),
        rigid_controls=control_rows,
        patient_floor_p95=float(final_confirmation["patient_floor_train"]["p95"]),
        diagnosis=dict(
            primary=(
                "OBJECTIVE_DOES_NOT_PROTECT_MODE_A_WITH_UNRESOLVED_SEARCH_AND_"
                "POSSIBLE_FAMILY_LIMIT"),
            optimizer=(
                "UNRESOLVED: every K used only restart 0, all phases stopped at "
                "max_generations, and each generation used a rotating four-network block"),
            objective=objective_text,
            objective_replay_evidence=objective_evidence,
            archive_context=(
                "a fit-only supported candidate reached worst-mode %.3f versus selected "
                "selection %.3f and final unseen %.3f; fit-only performance does not "
                "establish selection-seed dominance" % (
                    best_worst["worst_mode"], selected_matched.min(), final_matched.min())),
            model_family=(
                "POSSIBLE_LIMIT_NOT_PROVEN: no supported archived candidate reached the "
                "rigid hand-core worst-mode benchmark, but the search is too sparse to "
                "call structural impossibility"),
        ),
        next_discriminating_experiment=(
            "forced-initiation capacity first separates ignition from propagation; only "
            "after a good forced or oracle solution exists should matched-compute "
            "worst-mode objective and multi-restart optimizer experiments be run"),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--factorial-config", default=str(DEFAULT_FACTORIAL_CONFIG))
    parser.add_argument("--selection", default=str(DEFAULT_SELECTION))
    parser.add_argument("--out-of-selection", default=str(DEFAULT_OOS))
    parser.add_argument("--factorial", default=str(DEFAULT_FACTORIAL))
    parser.add_argument("--factorial-npz", default=str(DEFAULT_FACTORIAL_NPZ))
    parser.add_argument(
        "--learnability-decision", default=str(DEFAULT_LEARNABILITY_DECISION))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    # Snapshot code identity before any long analysis. Concurrent Topic 4 work
    # may advance HEAD while this producer is running.
    runtime_provenance = _runtime_provenance()
    worktree_status_at_start = _git("status", "--porcelain")
    producer_sha256_at_start = _sha256(__file__)

    config = _json_input(args.config)
    factorial_config = _json_input(args.factorial_config)
    selection = _json_input(args.selection)
    out_of_selection = _json_input(args.out_of_selection)
    factorial = _json_input(args.factorial)
    frozen = _json_input(ROOT / "frozen_readouts.json")
    with np.load(args.factorial_npz, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    reference_path = factorial["inputs"]["rank_curve_reference"]["path"]
    reference = load_reference(reference_path)
    contact_names = np.asarray(arrays["node_contact_names"]).astype(str)
    patient_prototypes = np.asarray(arrays["patient_train_mode_prototypes"], float)
    patient = _patient_training_events(
        reference, arrays["grid"], patient_prototypes, contact_names)
    selection_record = _json_input(config["inputs"]["selection"])
    final_confirmation = _json_input(config["inputs"]["confirmation"])
    learnability_decision = (
        _json_input(args.learnability_decision)
        if Path(args.learnability_decision).exists() else None)

    detector, detector_arrays = _detector_audit(factorial, factorial_config)
    response = _response_audit(config, alpha=selection["selection"]["alpha_star"])
    structure = _structure_audit(selection)
    mode_audit = _factorial_mode_audit(factorial, arrays, reference, patient)
    candidate_landscape = _candidate_landscape(
        selection_record, final_confirmation, learnability_decision)

    source_paths = [
        Path(args.config), Path(args.factorial_config),
        ROOT / "frozen_readouts.json", ROOT / "frozen_readouts.npz",
        ROOT / "edge_structure_audit.json", Path(args.selection),
        Path(args.out_of_selection), Path(args.factorial), Path(args.factorial_npz),
        Path(config["inputs"]["selection"]), Path(config["inputs"]["confirmation"]),
        Path(reference_path),
        Path("results/topic4_sef_hfo/data_driven_core_field/config/stage_config.json"),
        *FIT_CHECKPOINTS,
        *_patient_input_paths(PATIENT),
    ]
    if Path(args.learnability_decision).exists():
        source_paths.append(Path(args.learnability_decision))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz = out_dir / "rev9_review_audit_arrays.npz"
    _atomic_npz(
        out_npz,
        arms=np.asarray(ARM_ORDER, dtype="U16"),
        patient_mode_counts=np.asarray(patient["counts"], np.int64),
        **detector_arrays,
    )
    node_preflight = frozen.get("node_reconstruction_preflight", {})
    payload = dict(
        status="REV9_REVIEW_AUDIT_COMPLETE",
        scientific_state=(
            "NODE_REPERTOIRE_PARTIAL_PASS / EDGE_STANDALONE_INSUFFICIENT / "
            "EDGE_MODULATOR_CANDIDATE / LOCAL_RESPONSE_EQUIVALENCE_UNRESOLVED / "
            "CORE_CAUSALITY_NOT_TESTED"),
        safe_claim=(
            "the frozen patient-data-constrained Node field generates a stable, low-OOD "
            "two-cluster propagation repertoire on new networks, with strong mode-B and "
            "weak mode-A agreement; complete patient interictal activity is not reproduced"),
        frozen_execution_manifest=_manifest(
            source_paths, Path(factorial_config["output_root"]) / "workers"),
        node_reconstruction=dict(
            status=("PASS" if node_preflight.get("exact_at_frozen_dtype") else "FAIL"),
            evidence=node_preflight),
        edge_structure=structure,
        local_response=response,
        out_of_selection_pooled_reference=dict(
            status=out_of_selection["status"], observed=out_of_selection["observed"],
            interpretation="descriptive pooled replication, not equivalence confirmation"),
        common_detector=detector,
        factorial_mode_audit=mode_audit,
        optimization_diagnosis=candidate_landscape,
        arrays=dict(path=str(out_npz), sha256=_sha256(out_npz)),
        decision=dict(
            beta="DO_NOT_OPEN_YET",
            beta_reason=(
                "beta only adds radial concentration; current unresolved source gain and "
                "weak mode-A geometry are not an isolated radial-width defect"),
            next_causal_work=(
                "finish forced-initiation capacity, then component lesion, matched relocation, "
                "and multi-permutation d audit before any radial beta expansion")),
        provenance=dict(
            **runtime_provenance,
            producer_sha256=producer_sha256_at_start,
            worktree_status_porcelain_at_start=worktree_status_at_start,
            patient_readout="training blocks only; held-out values not computed"),
    )
    out_json = out_dir / "rev9_review_audit.json"
    atomic_write_json(payload, out_json)
    print(json.dumps(dict(
        status=payload["status"], scientific_state=payload["scientific_state"],
        node_reconstruction=payload["node_reconstruction"]["status"],
        local_response=payload["local_response"]["formal_status"],
        alpha_role=payload["edge_structure"]["alpha_reference_role"],
        patient_mode_status={
            arm: mode_audit["arms"][arm]["patient_mode_matrix_status"]["status"]
            for arm in ARM_ORDER},
        arrays_sha256=payload["arrays"]["sha256"]), indent=2))


if __name__ == "__main__":
    main()
