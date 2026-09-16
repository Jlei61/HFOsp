"""Off-diagonal multi-event distribution objective for Topic 4 v2.1.

This is intentionally a new import path so that the historical v2 pickle keeps
its D16 meaning when unpickled.
"""
from __future__ import annotations

import numpy as np

from src.topic4_interictal_repaired_evaluation import rank_features
from src.topic4_joint_xy_kernel import event_kernel_features, kernel_map
from src.topic4_multievent_distribution_objective import matched_mean_distance


OBJECTIVE_VERSION = "topic4_multievent_distribution_D_off_v2_1"


def off_diagonal_mean_distance(features, target):
    """Return ||mean(X)-t||^2 - mean_i||X_i-mean(X)||^2/(N-1)."""
    x = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    if (x.ndim != 2 or target.shape != (x.shape[1],)
            or not np.isfinite(x).all() or not np.isfinite(target).all()):
        raise ValueError("finite events x features and matching target required")
    if len(x) < 2:
        return None
    mean = x.mean(axis=0)
    variance = np.mean(np.sum((x - mean) ** 2, axis=1))
    return float(np.sum((mean - target) ** 2) - variance / (len(x) - 1))


def explicit_off_diagonal_mean_distance(features, target):
    """O(N^2) identity used only by tests and qualification audits."""
    x = np.asarray(features, dtype=float)
    target = np.asarray(target, dtype=float)
    if len(x) < 2:
        return None
    centered = x - np.asarray(target, dtype=float)
    cross = centered @ centered.T
    return float((cross.sum() - np.trace(cross)) / (len(x) * (len(x) - 1)))


def a_b_from_matched_and_off_diagonal(d_matched, d_off, n_events,
                                      matched_count=16):
    """Recover A and B per run/component; never apply after pooling runs."""
    if n_events < matched_count or matched_count < 1:
        raise ValueError("A/B diagnostic requires N >= matched_count >= 1")
    subtraction = matched_count / n_events * (d_matched - d_off)
    return float(d_off + subtraction), float(subtraction)


class MultieventDistributionObjectiveV21:
    """Frozen patient feature map with per-network D_off and D16 diagnostics."""

    def __init__(self, frozen_evaluator, *, matched_count=16,
                 normalizers=None):
        ev = frozen_evaluator
        self.km = ev.km
        self.xy = ev.xy
        self.groups = ev.groups
        self.scale = ev.scale
        self.maps = ev.maps
        self.k = ev.k
        self.matched_count = int(matched_count)
        self.proportions = (
            np.bincount(ev.fit_labels, minlength=self.k) / len(ev.fit_labels)
        )
        if np.any(self.proportions <= 0):
            raise ValueError("empty patient FIT mode")
        phi = self.embedding(ev.fit)
        self.target_global = phi.mean(axis=0)
        self.target_modes = np.concatenate([
            np.r_[1.0, phi[ev.fit_labels == mode].mean(axis=0)]
            for mode in range(self.k)
        ]) / np.sqrt(self.k)
        self.normalizers = None if normalizers is None else {
            "global": float(normalizers["global"]),
            "balanced_modes": float(normalizers["balanced_modes"]),
        }
        if self.normalizers is not None and not all(
                np.isfinite(v) and v > 0 for v in self.normalizers.values()):
            raise ValueError("fixed normalizers must be finite and positive")

    def embedding(self, times):
        return kernel_map(
            event_kernel_features(times, self.xy, self.groups, self.scale)["joint"],
            self.maps["joint"],
        ).astype(float)

    def balanced_embedding(self, phi, labels):
        z = np.column_stack([np.ones(len(phi)), phi])
        return np.column_stack([
            z * ((labels == mode) / self.proportions[mode])[:, None]
            for mode in range(self.k)
        ]) / np.sqrt(self.k)

    def components(self, times):
        table = np.asarray(times, dtype=float)
        if (table.ndim != 2 or table.shape[1] != len(self.xy)
                or np.isinf(table).any()):
            raise ValueError("events x contacts required")
        if np.any(np.isfinite(table).sum(axis=1) < 2):
            raise ValueError("unreadable event in training table")
        n_events = len(table)
        if n_events < self.matched_count:
            return {
                "status": "INSUFFICIENT_EVENTS", "n_events": n_events,
                "minimum_events_for_ranking": self.matched_count,
                "mode_counts": None, "D_off": None, "D16": None,
            }
        phi = self.embedding(table)
        labels = self.km.predict(rank_features(table))
        balanced = self.balanced_embedding(phi, labels)
        d_off = {
            "global": off_diagonal_mean_distance(phi, self.target_global),
            "balanced_modes": off_diagonal_mean_distance(
                balanced, self.target_modes),
        }
        d16 = {
            "global": matched_mean_distance(
                phi, self.target_global, self.matched_count),
            "balanced_modes": matched_mean_distance(
                balanced, self.target_modes, self.matched_count),
        }
        # For matched_count=m, D_m = A + (N-m)/(m(N-1))*V_N and
        # D_off = A - V_N/(N-1), hence B=V_N/(N-1)=m/N*(D_m-D_off).
        # This is a diagnostic decomposition only; ranking remains D_off.
        decomposition = {}
        for component in ("global", "balanced_modes"):
            mean_distance, subtraction = a_b_from_matched_and_off_diagonal(
                d16[component], d_off[component], n_events,
                self.matched_count,
            )
            decomposition[component] = {
                "A_mean_target_squared_distance": mean_distance,
                "B_finite_event_subtraction": subtraction,
                "D_off_equals_A_minus_B": float(
                    d_off[component]
                ),
            }
        return {
            "status": "ESTIMABLE", "n_events": n_events,
            "minimum_events_for_ranking": self.matched_count,
            "mode_counts": np.bincount(labels, minlength=self.k).astype(int).tolist(),
            "D_off": d_off,
            "D16": d16,
            "A_B_decomposition": decomposition,
            "A_B_diagnostic_only_not_ranking_or_gate": True,
        }

    def score_network(self, times):
        result = self.components(times)
        if result["status"] != "ESTIMABLE":
            return {**result, "loss_off": None, "loss_D16": None}
        if self.normalizers is None:
            raise RuntimeError("frozen positive v2 normalizers are not attached")
        losses = {}
        for statistic, output_key in (("D_off", "loss_off"), ("D16", "loss_D16")):
            raw = result[statistic]
            losses[output_key] = float(
                0.5 * raw["global"] / self.normalizers["global"]
                + 0.5 * raw["balanced_modes"]
                / self.normalizers["balanced_modes"]
            )
        a_normalized = float(sum(
            0.5 * result["A_B_decomposition"][component][
                "A_mean_target_squared_distance"
            ] / self.normalizers[component]
            for component in ("global", "balanced_modes")
        ))
        b_normalized = float(sum(
            0.5 * result["A_B_decomposition"][component][
                "B_finite_event_subtraction"
            ] / self.normalizers[component]
            for component in ("global", "balanced_modes")
        ))
        losses["loss_off_A_component"] = a_normalized
        losses["loss_off_B_subtraction"] = b_normalized
        losses["loss_off_A_minus_B_reconstruction_error"] = float(
            losses["loss_off"] - (a_normalized - b_normalized)
        )
        return {**result, **losses}

    def score_candidate(self, networks):
        units = {str(key): self.score_network(value)
                 for key, value in networks.items()}
        eligible = bool(units) and all(
            unit["loss_off"] is not None for unit in units.values()
        )
        return {
            "status": "ESTIMABLE" if eligible else "INSUFFICIENT_EVENTS",
            "objective_version": OBJECTIVE_VERSION,
            "per_network": units,
            "loss_off": (float(np.mean([
                unit["loss_off"] for unit in units.values()
            ])) if eligible else None),
            "loss_D16": (float(np.mean([
                unit["loss_D16"] for unit in units.values()
            ])) if eligible else None),
            "loss_off_A_component": (float(np.mean([
                unit["loss_off_A_component"] for unit in units.values()
            ])) if eligible else None),
            "loss_off_B_subtraction": (float(np.mean([
                unit["loss_off_B_subtraction"] for unit in units.values()
            ])) if eligible else None),
            "A_B_aggregation": (
                "computed per run and per component before equal-run averaging"
            ),
            "aggregation": "equal weight per fixed network; no event pooling",
            "partial_unit_average_forbidden": True,
        }
