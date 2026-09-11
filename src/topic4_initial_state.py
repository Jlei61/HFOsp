"""Initial-state (t=0 membrane voltage) conditioned propagation, v1.

Plan   : docs/archive/topic4/sef_hfo/initial_state_conditioned_propagation_plan_v1_2026-09-07.md
Design : config/topic4_initial_state_conditioned_propagation_v1.json

Everything here is deterministic and consumes no simulation RNG. Contract clause
labels (E*, M*, V*, D*, O*, S*, Q*) refer to the plan sections enumerated in the
implementation notes; each public function names the clauses it honours.
"""
from __future__ import annotations

import hashlib
import itertools

import numpy as np

ARMS = ("B0", "B1", "B2")
PERTURBED_CORE = {"B0": None, "B1": 0, "B2": 1}
HISTOGRAM_EDGES_MV = np.arange(0.0, 25.0 + 0.25, 0.25)


class ProbeNotApplicable(RuntimeError):
    """The fixed 1 mV cold-start probe cannot be applied on this substrate."""


def array_sha256(values):
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


# --------------------------------------------------------------------------
# membership (plan §3; clauses M1-M5)
# --------------------------------------------------------------------------
def core_membership(positions_e, h_e, centers_mm):
    """Equal-size nearest-member sets S_A, S_B inside the actual h>0 E cells.

    M1: partition h>0 E cells by nearest candidate center (ties -> lower core
        index, i.e. the candidate order; no mode / patient remapping, M4).
    M2: within each group order by (distance, neuron index).
    M3: K = min(n_A, n_B); S_A / S_B are the first K members of each group.
    M5: K == 0 raises ProbeNotApplicable (never re-selected by patient route).
    """
    positions = np.asarray(positions_e, dtype=np.float64)
    h = np.asarray(h_e, dtype=np.float64)
    centers = np.asarray(centers_mm, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions_e must have shape (n_E, 2)")
    if h.shape != (positions.shape[0],):
        raise ValueError("h_e must align with positions_e")
    if centers.shape != (2, 2) or not np.isfinite(centers).all():
        raise ValueError("two finite core centers are required")
    members = np.flatnonzero(h > 0.0)
    if members.size == 0:
        raise ProbeNotApplicable("no h>0 excitatory cell on this substrate")
    distance = np.linalg.norm(
        positions[members][:, None, :] - centers[None, :, :], axis=2)
    nearest = np.argmin(distance, axis=1)          # first index on ties (M1)
    groups = []
    for core in range(2):
        take = nearest == core
        selected = members[take]
        dist = distance[take, core]
        order = np.lexsort((selected, dist))        # primary distance, then index (M2)
        groups.append((selected[order], dist[order]))
    counts = [int(len(group[0])) for group in groups]
    k = min(counts)
    if k == 0:
        raise ProbeNotApplicable("one core has no h>0 member; K=0")
    s_a = np.asarray(groups[0][0][:k], np.int64)
    s_b = np.asarray(groups[1][0][:k], np.int64)
    if np.intersect1d(s_a, s_b).size:
        raise RuntimeError("core member sets overlap")
    return {
        "S_A": s_a, "S_B": s_b, "K": int(k),
        "n_A": counts[0], "n_B": counts[1],
        "n_h_positive": int(members.size),
        "distance_A_mm": np.asarray(groups[0][1][:k], np.float64),
        "distance_B_mm": np.asarray(groups[1][1][:k], np.float64),
        "maximum_member_distance_mm": [
            float(groups[0][1][:k].max()), float(groups[1][1][:k].max())],
        "dropped_from_larger_group": int(max(counts) - k),
        "centers_mm": centers.tolist(),
        "rule": ("actual h>0 E cells, nearest candidate center (ties -> lower "
                 "core index), per-core (distance, index) order, equal K nearest"),
        "S_A_sha256": array_sha256(s_a), "S_B_sha256": array_sha256(s_b),
    }


# --------------------------------------------------------------------------
# initial voltage arms (plan §3; clauses V1-V4)
# --------------------------------------------------------------------------
def build_initial_voltage(n_total, v_reset, membership, increment_mv, arm):
    """V1: B0 all V_reset; B1 adds `increment_mv` on S_A only; B2 on S_B only."""
    if arm not in PERTURBED_CORE:
        raise ValueError(f"unknown initial-state arm {arm!r}")
    n_total = int(n_total)
    voltage = np.full(n_total, float(v_reset), dtype=np.float64)
    core = PERTURBED_CORE[arm]
    if core is not None:
        index = np.asarray(membership["S_A" if core == 0 else "S_B"], np.int64)
        if index.size == 0 or index.min() < 0 or index.max() >= n_total:
            raise ValueError("member indices lie outside the network")
        voltage[index] += float(increment_mv)
    return voltage


def validate_initial_voltage(voltage, vtheta, v_reset, required_margin_mv):
    """V3: finite and below the actual per-neuron threshold by >= margin.

    Returns the audit; raises ProbeNotApplicable when the fixed probe violates
    the margin (no clipping, no amplitude change).
    """
    voltage = np.asarray(voltage, np.float64)
    vtheta = np.asarray(vtheta, np.float64)
    if voltage.shape != vtheta.shape:
        raise ValueError("initial voltage and threshold arrays must align")
    if not np.isfinite(voltage).all() or not np.isfinite(vtheta).all():
        raise ProbeNotApplicable("non-finite initial voltage or threshold")
    margin = vtheta - voltage
    perturbed = np.flatnonzero(voltage != float(v_reset))
    audit = {
        "required_margin_mV": float(required_margin_mv),
        "minimum_margin_all_mV": float(margin.min()),
        "minimum_margin_perturbed_mV": (
            float(margin[perturbed].min()) if perturbed.size else None),
        "n_perturbed": int(perturbed.size),
        "n_below_required_margin": int(np.sum(margin < required_margin_mv)),
        "pass": bool(np.all(margin >= required_margin_mv)),
    }
    if not audit["pass"]:
        raise ProbeNotApplicable(
            f"{audit['n_below_required_margin']} cells violate the "
            f"{required_margin_mv} mV threshold margin")
    return audit


def initial_voltage_summary(voltage, v_reset):
    voltage = np.asarray(voltage, np.float64)
    delta = voltage - float(v_reset)
    hist, _ = np.histogram(voltage, bins=HISTOGRAM_EDGES_MV)
    return {
        "n_total": int(voltage.size),
        "n_perturbed": int(np.count_nonzero(delta)),
        "total_increment_mV": float(delta.sum()),
        "l2_increment_mV": float(np.sqrt(np.sum(delta ** 2))),
        "maximum_abs_increment_mV": float(np.abs(delta).max()),
        "minimum_mV": float(voltage.min()), "maximum_mV": float(voltage.max()),
        "histogram_counts": hist.astype(int).tolist(),
        "histogram_edges_mV": HISTOGRAM_EDGES_MV.tolist(),
        "sha256": array_sha256(voltage),
    }


def check_arm_symmetry(summary_b1, summary_b2):
    """V2: B1 and B2 share K, total increment, L2 and histogram; only location differs."""
    same = all(summary_b1[key] == summary_b2[key] for key in (
        "n_perturbed", "total_increment_mV", "l2_increment_mV",
        "maximum_abs_increment_mV", "histogram_counts"))
    if not same:
        raise RuntimeError("B1/B2 initial voltages are not increment-matched")
    if summary_b1["sha256"] == summary_b2["sha256"]:
        raise RuntimeError("B1 and B2 initial voltages are identical")
    return True


# --------------------------------------------------------------------------
# read-only per-step observer (plan §5.3, §8; clauses D1, X10, E7)
# --------------------------------------------------------------------------
class RunObserver:
    """Streaming external-input digest and 1 ms state trace.

    D1: per segment (default 1000 ms) a sha256 over the per-step stream of the
        global OU value `xi`, the clipped rate `nu_now`, the exact Poisson counts
        and the spatial OU field sampled every `trace_ms`; plus exact per-neuron
        segment sums of the Poisson counts and of the spatial OU field (every
        step). No whole-run tape is stored.
    X10: every `trace_ms` the mean V, spike count, mean I_E and mean I_I per
        named neuron group (core A, core B, surround E, I) and global rates.
    Reads only; never draws from any RNG.
    """

    def __init__(self, *, dt_ms, n_e, n_total, groups, n_steps,
                 segment_ms=1000.0, trace_ms=1.0):
        self.dt_ms = float(dt_ms)
        self.n_e, self.n_total = int(n_e), int(n_total)
        self.segment_steps = int(round(segment_ms / self.dt_ms))
        self.trace_steps = int(round(trace_ms / self.dt_ms))
        if self.segment_steps < 1 or self.trace_steps < 1:
            raise ValueError("segment and trace strides must be >= 1 step")
        if self.segment_steps % self.trace_steps:
            raise ValueError("segment length must be a multiple of the trace stride")
        self.group_names = list(groups)
        self.groups = {name: np.asarray(index, np.int64) for name, index in groups.items()}
        for index in self.groups.values():
            if index.size and (index.min() < 0 or index.max() >= self.n_total):
                raise ValueError("group indices lie outside the network")
        self.segment_ms = float(segment_ms)
        self.trace_ms = float(trace_ms)
        self.n_steps = int(n_steps)
        n_trace = (self.n_steps + self.trace_steps - 1) // self.trace_steps
        self._trace_count = 0
        self.trace = {
            "time_ms": np.zeros(n_trace, np.float64),
            "xi": np.zeros(n_trace, np.float32),
            "nu_now": np.zeros(n_trace, np.float32),
            "E_spikes": np.zeros(n_trace, np.int32),
            "I_spikes": np.zeros(n_trace, np.int32),
            "E_mean_V": np.zeros(n_trace, np.float32),
        }
        for name in self.group_names:
            for field in ("mean_V", "spikes", "mean_I_E", "mean_I_I"):
                self.trace[f"{name}_{field}"] = np.zeros(
                    n_trace, np.int32 if field == "spikes" else np.float32)
        self._spike_accumulator = np.zeros(self.n_total, np.int32)
        self._reset_segment(0)
        self.segments = []
        self.ext_segment_sums = []
        self.delta_segment_sums = []
        self.n_observed_steps = 0
        self.maximum_poisson_count = 0

    def _reset_segment(self, index):
        self._segment_index = index
        self._segment_hash = hashlib.sha256()
        self._segment_steps_seen = 0
        self._segment_ext_sum = np.zeros(self.n_total, np.int64)
        self._segment_delta_sum = np.zeros(self.n_e, np.float64)
        self._segment_xi_sum = 0.0
        self._segment_xi_sumsq = 0.0
        self._segment_delta_present = False

    def _finalize_segment(self):
        if self._segment_steps_seen == 0:
            return
        start = self._segment_index * self.segment_steps * self.dt_ms
        self.segments.append({
            "index": int(self._segment_index),
            "start_ms": float(start),
            "end_ms": float(start + self._segment_steps_seen * self.dt_ms),
            "n_steps": int(self._segment_steps_seen),
            "complete": bool(self._segment_steps_seen == self.segment_steps),
            "stream_sha256": self._segment_hash.hexdigest(),
            "ext_sum_sha256": array_sha256(self._segment_ext_sum),
            "delta_sum_sha256": (array_sha256(self._segment_delta_sum)
                                 if self._segment_delta_present else None),
            "ext_total": int(self._segment_ext_sum.sum()),
            "delta_abs_total": (float(np.abs(self._segment_delta_sum).sum())
                                if self._segment_delta_present else None),
            "xi_sum": float(self._segment_xi_sum),
            "xi_sumsq": float(self._segment_xi_sumsq),
        })
        self.ext_segment_sums.append(self._segment_ext_sum.astype(np.int32))
        self.delta_segment_sums.append(self._segment_delta_sum.copy())

    def observe(self, t, tm, xi, nu_now, ext, delta_rate, V, I_E, I_I, spk):
        if t != self.n_observed_steps:
            raise RuntimeError("observer steps must be contiguous from 0")
        if (t // self.segment_steps) != self._segment_index:
            self._finalize_segment()
            self._reset_segment(t // self.segment_steps)
        # ---- streaming digest (D1) ----
        ext_counts = np.asarray(ext)
        maximum = int(ext_counts.max()) if ext_counts.size else 0
        if maximum > np.iinfo(np.uint16).max:
            raise RuntimeError("Poisson count exceeds uint16 digest storage")
        self.maximum_poisson_count = max(self.maximum_poisson_count, maximum)
        counts = ext_counts.astype(np.uint16)
        digest = self._segment_hash
        digest.update(np.float64(xi).tobytes())
        digest.update(np.float64(nu_now).tobytes())
        digest.update(counts.tobytes())
        self._segment_ext_sum += counts
        self._segment_xi_sum += float(xi)
        self._segment_xi_sumsq += float(xi) ** 2
        if delta_rate is not None:
            delta = np.asarray(delta_rate, np.float64)
            self._segment_delta_present = True
            self._segment_delta_sum += delta
            if t % self.trace_steps == 0:
                digest.update(np.ascontiguousarray(delta).tobytes())
        self._segment_steps_seen += 1
        # ---- state trace (X10) ----
        self._spike_accumulator += spk
        if (t + 1) % self.trace_steps == 0 or t + 1 == self.n_steps:
            k = self._trace_count
            trace = self.trace
            trace["time_ms"][k] = tm
            trace["xi"][k] = xi
            trace["nu_now"][k] = nu_now
            trace["E_spikes"][k] = int(self._spike_accumulator[:self.n_e].sum())
            trace["I_spikes"][k] = int(self._spike_accumulator[self.n_e:].sum())
            trace["E_mean_V"][k] = float(V[:self.n_e].mean())
            for name, index in self.groups.items():
                if index.size:
                    trace[f"{name}_mean_V"][k] = float(V[index].mean())
                    trace[f"{name}_spikes"][k] = int(self._spike_accumulator[index].sum())
                    trace[f"{name}_mean_I_E"][k] = float(I_E[index].mean())
                    trace[f"{name}_mean_I_I"][k] = float(I_I[index].mean())
            self._spike_accumulator[:] = 0
            self._trace_count += 1
        self.n_observed_steps += 1

    def finish(self):
        self._finalize_segment()
        self._segment_steps_seen = 0
        n = self._trace_count
        return {
            "trace": {key: value[:n] for key, value in self.trace.items()},
            "segments": list(self.segments),
            "ext_segment_sums": (np.stack(self.ext_segment_sums)
                                 if self.ext_segment_sums
                                 else np.zeros((0, self.n_total), np.int32)),
            "delta_segment_sums": (np.stack(self.delta_segment_sums)
                                   if self.delta_segment_sums
                                   else np.zeros((0, self.n_e), np.float64)),
            "n_observed_steps": int(self.n_observed_steps),
            "segment_ms": self.segment_ms, "trace_ms": self.trace_ms,
            "maximum_poisson_count": int(self.maximum_poisson_count),
            "digest_definition": (
                "sha256 per segment over the per-step stream [xi float64, "
                "nu_now float64, Poisson counts uint16 (all neurons)] plus the "
                "spatial OU field (float64, E neurons) sampled every trace "
                "stride; exact per-neuron per-segment sums of the Poisson counts "
                "and of the spatial OU field hashed separately"),
        }


def compare_input_digests(segments_a, segments_b):
    """D2/D4: compare the common prefix of two per-segment digest lists."""
    n = min(len(segments_a), len(segments_b))
    rows = []
    for index in range(n):
        a, b = segments_a[index], segments_b[index]
        common_steps = min(int(a["n_steps"]), int(b["n_steps"]))
        full = a["n_steps"] == b["n_steps"]
        rows.append({
            "index": int(index),
            "steps_compared": int(common_steps),
            "same_length": bool(full),
            "stream_equal": bool(full and a["stream_sha256"] == b["stream_sha256"]),
            "ext_sum_equal": bool(full and a["ext_sum_sha256"] == b["ext_sum_sha256"]),
            "delta_sum_equal": bool(full and a["delta_sum_sha256"] == b["delta_sum_sha256"]),
        })
    complete = [row for row in rows if row["same_length"]]
    return {
        "n_segments_compared": n,
        "n_complete_segments_compared": len(complete),
        "all_complete_segments_equal": bool(complete) and all(
            row["stream_equal"] and row["ext_sum_equal"] and row["delta_sum_equal"]
            for row in complete),
        "prefix_only": bool(len(segments_a) != len(segments_b)
                            or any(not row["same_length"] for row in rows)),
        "rows": rows,
    }


# --------------------------------------------------------------------------
# windows and proportions (plan §6; clauses O2-O4)
# --------------------------------------------------------------------------
def event_times_ms(observation):
    """O2: primary-eligible events located at the midpoint of qualifying_interval_ms."""
    events = observation["events"]
    primary = [int(i) for i in observation["primary_event_indices"]]
    times = np.full(len(events), np.nan)
    for index, event in enumerate(events):
        a, b = event["qualifying_interval_ms"]
        times[index] = 0.5 * (float(a) + float(b))
    return times, primary


def window_mode_counts(times_ms, labels, primary_indices, window_ms, n_modes=2):
    """O3/O4: counts per mode over classifiable primary events in [lo, hi)."""
    lo, hi = float(window_ms[0]), float(window_ms[1])
    counts = np.zeros(n_modes, int)
    n_unclassifiable = 0
    n_primary_in_window = 0
    for index in primary_indices:
        t = times_ms[index]
        if not (lo <= t < hi):
            continue
        n_primary_in_window += 1
        label = int(labels[index])
        if label < 0:
            n_unclassifiable += 1
        else:
            counts[label] += 1
    total = int(counts.sum())
    return {
        "window_ms": [lo, hi],
        "mode_counts": counts.tolist(),
        "n_classified": total,
        "n_unclassifiable": int(n_unclassifiable),
        "n_primary_in_window": int(n_primary_in_window),
        "mode0_proportion": (None if total == 0 else float(counts[0] / total)),
    }


# --------------------------------------------------------------------------
# paired statistics (plan §6; clauses S1-S6)
# --------------------------------------------------------------------------
def exchange_test_p(differences):
    """S3: two-sided exact paired sign-exchange p over all 2^n assignments (ties included)."""
    d = np.asarray(differences, np.float64)
    n = len(d)
    observed = abs(d.mean())
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    means = np.abs(signs @ d) / n
    hits = int(np.sum(means >= observed - 1e-12))
    return {"p_two_sided": float(hits / len(signs)), "n_assignments": int(len(signs)),
            "observed_abs_mean": float(observed)}


def bootstrap_mean_ci(differences, *, n_resamples, seed, level=0.95):
    """S2: percentile bootstrap of the mean over pairs (pairs resampled as units)."""
    d = np.asarray(differences, np.float64)
    n = len(d)
    rng = np.random.default_rng(int(seed))
    index = rng.integers(0, n, size=(int(n_resamples), n))
    means = d[index].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return {"ci": [float(lo), float(hi)], "n_resamples": int(n_resamples),
            "seed": int(seed), "level": float(level), "method": "percentile"}


def instability_flags(differences, ci, p_value, alpha=0.05):
    """S4: single-pair dominance, leave-one-out sign flip, bootstrap/exchange disagreement."""
    d = np.asarray(differences, np.float64)
    total_abs = float(np.abs(d).sum())
    contributions = (np.abs(d) / total_abs) if total_abs > 0 else np.zeros_like(d)
    full_mean = float(d.mean())
    loo = np.array([np.delete(d, i).mean() for i in range(len(d))])
    sign_flip = bool(np.any(np.sign(loo) != np.sign(full_mean))) if full_mean != 0 else bool(np.any(loo != 0))
    ci_excludes_zero = bool(ci[0] > 0.0 or ci[1] < 0.0)
    significant = bool(p_value <= alpha)
    return {
        "max_single_pair_contribution": float(contributions.max()) if len(d) else 0.0,
        "single_pair_dominates": bool(contributions.max() > 0.5) if len(d) else False,
        "leave_one_out_means": loo.tolist(),
        "leave_one_out_sign_flip": sign_flip,
        "bootstrap_ci_excludes_zero": ci_excludes_zero,
        "exchange_significant": significant,
        "bootstrap_exchange_disagree": bool(ci_excludes_zero != significant),
        "unstable": bool((contributions.max() > 0.5 if len(d) else False)
                         or sign_flip or (ci_excludes_zero != significant)),
    }


def paired_effect(differences, *, n_resamples, seed, alpha=0.05):
    """S1-S4 bundled for one contrast on estimable pairs."""
    d = np.asarray(differences, np.float64)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all():
        raise ValueError("at least two finite paired differences are required")
    boot = bootstrap_mean_ci(d, n_resamples=n_resamples, seed=seed)
    exchange = exchange_test_p(d)
    flags = instability_flags(d, boot["ci"], exchange["p_two_sided"], alpha)
    return {
        "n_pairs": int(len(d)), "differences": d.tolist(),
        "mean": float(d.mean()), "median": float(np.median(d)),
        "bootstrap": boot, "exchange": exchange, "instability": flags,
    }


def missing_pair_bounds(observed_differences, n_missing, n_total=12, bounds=(-1.0, 1.0)):
    """S6: full-n mean bounds with each missing pair at the extreme values."""
    d = np.asarray(observed_differences, np.float64)
    if len(d) + int(n_missing) != n_total:
        raise ValueError("observed plus missing pairs must equal the design size")
    return [float((d.sum() + n_missing * bounds[0]) / n_total),
            float((d.sum() + n_missing * bounds[1]) / n_total)]


def screen_verdict(effect, *, n_total=12, threshold=0.10, alpha=0.05, early_effect=None):
    """S5: the pre-registered decision table for one graph.

    `effect` is None when the primary contrast is NOT_ESTIMABLE; `early_effect`
    (optional) is the same statistic on the 0.5-6 s window used only to name
    STARTUP_SENSITIVITY_ONLY when the late window fails.
    """
    if effect is None or effect["n_pairs"] != n_total:
        return "NOT_ESTIMABLE"
    ci = effect["bootstrap"]["ci"]
    p = effect["exchange"]["p_two_sided"]
    mean = effect["mean"]
    unstable = effect["instability"]["unstable"]
    excludes = ci[0] > 0.0 or ci[1] < 0.0
    if (p <= alpha and excludes and abs(mean) >= threshold and not unstable):
        return "PERSISTENT_INITIAL_STATE_EFFECT_SINGLE_GRAPH"
    if ci[0] >= -threshold and ci[1] <= threshold:
        return "SMALL_EFFECT_BOUNDED_FOR_TESTED_PROBE"
    if early_effect is not None and early_effect.get("n_pairs") == n_total:
        early_ci = early_effect["bootstrap"]["ci"]
        early_p = early_effect["exchange"]["p_two_sided"]
        if (early_p <= alpha and (early_ci[0] > 0.0 or early_ci[1] < 0.0)
                and not early_effect["instability"]["unstable"]):
            return "STARTUP_SENSITIVITY_ONLY"
    return "INCONCLUSIVE"


def holm_adjust(p_values):
    """S7: Holm step-down adjusted p-values (monotone)."""
    p = np.asarray(p_values, np.float64)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (m - rank) * p[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


# --------------------------------------------------------------------------
# conditional propagation quality (plan §7; clause Q3)
# --------------------------------------------------------------------------
def off_diagonal_terms(features, target):
    """Q3: D_off = A - B with A = ||mean - target||^2, B = mean_i||x_i - mean||^2/(N-1)."""
    x = np.asarray(features, np.float64)
    target = np.asarray(target, np.float64)
    if x.ndim != 2 or target.shape != (x.shape[1],):
        raise ValueError("events x features and a matching target are required")
    n = len(x)
    if n < 2:
        return {"status": "NOT_ESTIMABLE", "n_events": int(n), "D_off": None,
                "A_mean_target_squared_distance": None,
                "B_finite_event_subtraction": None, "within_dispersion": None}
    mean = x.mean(axis=0)
    a = float(np.sum((mean - target) ** 2))
    dispersion = float(np.mean(np.sum((x - mean) ** 2, axis=1)))
    b = dispersion / (n - 1)
    return {"status": "ESTIMABLE", "n_events": int(n), "D_off": float(a - b),
            "A_mean_target_squared_distance": a,
            "B_finite_event_subtraction": float(b),
            "within_dispersion": dispersion}


def block_matched_reference(features, blocks, target, *, n_model, n_resamples, seed):
    """Q5: FIT-only two-stage block resampling matched to the model event count.

    Each draw picks a block uniformly among blocks holding this mode, then one
    event uniformly inside that block; N = n_model draws with replacement.
    Returns quantiles of D_off / A / B and the block coverage of the draws.
    """
    x = np.asarray(features, np.float64)
    blocks = np.asarray(blocks)
    unique = np.unique(blocks)
    if n_model < 2:
        return {"status": "NOT_ESTIMABLE", "n_model": int(n_model),
                "n_blocks_available": int(len(unique))}
    rng = np.random.default_rng(int(seed))
    members = {block: np.flatnonzero(blocks == block) for block in unique}
    d_off, a_term, b_term, coverage = [], [], [], []
    for _ in range(int(n_resamples)):
        chosen_blocks = rng.choice(unique, size=int(n_model), replace=True)
        rows = np.array([rng.choice(members[block]) for block in chosen_blocks])
        terms = off_diagonal_terms(x[rows], target)
        d_off.append(terms["D_off"])
        a_term.append(terms["A_mean_target_squared_distance"])
        b_term.append(terms["B_finite_event_subtraction"])
        coverage.append(len(np.unique(chosen_blocks)))
    quantiles = [0.025, 0.5, 0.975]

    def q(values):
        return np.quantile(np.asarray(values, float), quantiles).tolist()
    return {"status": "ESTIMABLE", "n_model": int(n_model), "n_resamples": int(n_resamples),
            "seed": int(seed), "n_blocks_available": int(len(unique)),
            "quantiles": quantiles, "D_off_quantiles": q(d_off),
            "A_quantiles": q(a_term), "B_quantiles": q(b_term),
            "distinct_blocks_per_draw_median": float(np.median(coverage)),
            "rule": "uniform block, then uniform event within block, N draws with replacement"}


def per_block_reference(features, blocks, target, minimum_n=2):
    """Q5: natural per-block D_off of the FIT events of one mode."""
    x = np.asarray(features, np.float64)
    blocks = np.asarray(blocks)
    rows = []
    for block in np.unique(blocks):
        take = np.flatnonzero(blocks == block)
        if len(take) < minimum_n:
            rows.append({"block": int(block), "n_events": int(len(take)),
                         "status": "NOT_ESTIMABLE", "D_off": None})
            continue
        terms = off_diagonal_terms(x[take], target)
        rows.append({"block": int(block), **terms})
    return rows
