"""Probe geometry, descendant-only response, in-window regime flags, splices.

Two separations are load-bearing here and neither may be relaxed:

1. The response counts DESCENDANT spikes only. The forced packet's own spikes
   ride the ordinary recorder, so a 256-cell packet contributes 256 excess
   spikes with zero recursive amplification -- enough on its own to clear any
   threshold on the order of a few hundred.
2. An "event" only counts when the probe branch has it and the paired sham
   branch does not. The unperturbed network is above the common detector 41 %
   of the time, so "an event occurred" is not evidence of anything.
"""
from __future__ import annotations

import copy

import numpy as np

from src.topic4_forced_source_capacity import exclude_injected_packet_frame

# (host, z source, m source)
_SPLICE = {
    "native_baseline":  ("baseline",  "baseline",  "baseline"),
    "native_pre_ictal": ("pre_ictal", "pre_ictal", "pre_ictal"),
    "reset_z":          ("pre_ictal", "baseline",  "pre_ictal"),
    "reset_m":          ("pre_ictal", "pre_ictal", "baseline"),
    "reset_zm":         ("pre_ictal", "baseline",  "baseline"),
    "slow_only":        ("baseline",  "pre_ictal", "pre_ictal"),
}
SPLICE_MODES = tuple(_SPLICE)


def frozen_sites(substrate, config, *, kind):
    """Site sets fixed from geometry alone, never from any run's output."""
    perturbation = config["perturbation"]
    if kind == "grid":
        lo, hi = perturbation["grid_extent_mm"]
        n = int(perturbation["grid_n"])
        axis = np.linspace(float(lo), float(hi), n)
        xy = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(-1, 2)
        return [{"site_id": f"g{i:02d}", "xy_mm": point, "kind": "grid"}
                for i, point in enumerate(xy)]
    if kind == "representative":
        src = np.asarray(substrate.axis_source_xy, float)
        snk = np.asarray(substrate.axis_sink_xy, float)
        mid = 0.5 * (src + snk)
        normal = np.array([-substrate.axis_unit[1], substrate.axis_unit[0]])
        centre = float(substrate.engine["L"]) / 2.0
        points = [src, snk, mid, mid + 4.0 * normal, mid - 4.0 * normal,
                  np.array([centre, centre])]
        names = ["source", "sink", "midpoint", "normal_plus", "normal_minus", "centre"]
        return [{"site_id": name, "xy_mm": np.asarray(point, float),
                 "kind": "representative"}
                for name, point in zip(names, points)]
    raise ValueError(f"unknown site kind {kind!r}")


def select_packet(positions_e, site_xy, *, n_cells, radius_mm):
    positions = np.asarray(positions_e, float)
    distance = np.linalg.norm(positions - np.asarray(site_xy, float), axis=1)
    inside = np.flatnonzero(distance <= float(radius_mm))
    if inside.size < int(n_cells):
        raise ValueError(
            f"insufficient E neurons within the packet radius: {inside.size} < {n_cells}")
    chosen = inside[np.argsort(distance[inside], kind="stable")[:int(n_cells)]]
    mask = np.zeros(len(positions), bool)
    mask[chosen] = True
    return mask


def _descendant(probe, sham, packet_mask, inject_step):
    return exclude_injected_packet_frame(
        np.asarray(probe["E_spk_bool"], bool), np.asarray(sham["E_spk_bool"], bool),
        np.asarray(packet_mask, bool), trigger_step=int(inject_step))


def response_metrics(probe, sham, *, dt_ms, positions_e, packet_mask, packet_xy,
                     envelope_probe, envelope_sham, envelope_dt_ms,
                     inject_step, split_ms, window_ms):
    sham_spikes = np.asarray(sham["E_spk_bool"], bool)
    probe_desc = _descendant(probe, sham, packet_mask, inject_step)
    stop = min(probe_desc.shape[0], int(inject_step) + int(round(window_ms / dt_ms)))
    split = int(inject_step) + int(round(split_ms / dt_ms))
    excess = (probe_desc[inject_step:stop].sum(axis=0).astype(float)
              - sham_spikes[inject_step:stop].sum(axis=0).astype(float))
    early = (probe_desc[inject_step:min(split, stop)].sum()
             - sham_spikes[inject_step:min(split, stop)].sum())
    late = (probe_desc[min(split, stop):stop].sum()
            - sham_spikes[min(split, stop):stop].sum())

    positive = np.clip(excess, 0.0, None)
    if positive.sum() > 0:
        distance = np.linalg.norm(np.asarray(positions_e, float)
                                  - np.asarray(packet_xy, float), axis=1)
        order = np.argsort(distance, kind="stable")
        cumulative = np.cumsum(positive[order])
        index = int(np.searchsorted(cumulative, 0.9 * cumulative[-1]))
        r90 = float(distance[order][min(index, len(order) - 1)])
    else:
        r90 = float("nan")

    frames = int(round(window_ms / envelope_dt_ms))
    frame0 = int(round(inject_step * dt_ms / envelope_dt_ms))
    contact = np.clip(np.asarray(envelope_probe, float)[:, frame0:frame0 + frames]
                      - np.asarray(envelope_sham, float)[:, frame0:frame0 + frames],
                      0.0, None)
    return {"susceptibility": float(excess.sum()),
            "excess_spikes_early": float(early),
            "excess_spikes_late": float(late),
            "r90_mm": r90,
            "contact_excess_energy": float(contact.sum()),
            "excess_per_neuron": excess.astype(np.float32)}


def _overlaps(a, b):
    return not (a[1] <= b[0] or b[1] <= a[0])


def in_window_ignition(probe_active, sham_active, *, active_dt_ms,
                       detector_threshold, inject_ms, window_ms,
                       probe_rate_hz=None, es_thresh_hz=120.0, es_dur_ms=100.0,
                       dt_ms=None):
    """Regime flags for EVERY E1 site, grid included.

    Events come from the FROZEN detector (src/sef_hfo_events.detect_events), not
    from a local threshold crossing: that detector requires MIN_DUR_MS = 8 ms,
    merges intervals closer than MERGE_GAP_MS = 12 ms, and carries the return
    semantics. Calling a single 1 ms excursion an event would over-detect
    ignition, invalidate E1 sites that are fine, and could manufacture a
    NO_SUBEVENT_PROBE_REGIME verdict out of nothing.

    Freezing the dose on baseline checkpoints guarantees nothing at the
    pre-ictal checkpoint, which is where excitability is hypothesised to be
    higher -- hence these flags travel with every E1 row.
    """
    from src.sef_hfo_events import detect_events

    probe_active = np.asarray(probe_active, float)
    sham_active = np.asarray(sham_active, float)
    lo = int(round(float(inject_ms) / float(active_dt_ms)))
    hi = lo + int(round(float(window_ms) / float(active_dt_ms)))
    probe_window = probe_active[lo:hi]
    sham_window = sham_active[lo:hi]

    probe_events = detect_events(probe_window, float(active_dt_ms),
                                 event_on_frac=float(detector_threshold))
    sham_events = detect_events(sham_window, float(active_dt_ms),
                                event_on_frac=float(detector_threshold))
    sham_spans = [(float(e["t_on"]), float(e["t_off"])) for e in sham_events]
    attributable = [e for e in probe_events
                    if not any(_overlaps((float(e["t_on"]), float(e["t_off"])), span)
                               for span in sham_spans)]

    # a bare threshold excursion is recorded, but it is NOT an event
    brief = bool(np.any(probe_window > float(detector_threshold))
                 and not probe_events)

    reached = False
    if probe_rate_hz is not None and dt_ms is not None:
        alpha = 1.0 - np.exp(-float(dt_ms) / 20.0)
        need = int(round(float(es_dur_ms) / float(dt_ms)))
        ema, run = 0.0, 0
        for value in np.asarray(probe_rate_hz, float):
            ema += alpha * (value - ema)
            run = run + 1 if ema >= float(es_thresh_hz) else 0
            if run >= need:
                reached = True
                break
    return {"probe_attributable_event_200ms": bool(attributable),
            "n_probe_events": int(len(probe_events)),
            "n_sham_events": int(len(sham_events)),
            "brief_threshold_excursion": brief,
            "reached_model_ictal_200ms": bool(reached),
            "e1_evaluable": bool(not attributable and not reached),
            "detector_contract": "src.sef_hfo_events.detect_events (MIN_DUR_MS=8, MERGE_GAP_MS=12)"}


def ignition_metrics(probe_active, sham_active, *, active_dt_ms, detector_threshold,
                     inject_ms, window_ms, probe_onset_ms, sham_onset_ms, **kwargs):
    out = in_window_ignition(probe_active, sham_active, active_dt_ms=active_dt_ms,
                             detector_threshold=detector_threshold,
                             inject_ms=inject_ms, window_ms=window_ms, **kwargs)
    censored = probe_onset_ms is None or sham_onset_ms is None
    out["onset_advance_ms"] = (float("nan") if censored
                               else float(sham_onset_ms) - float(probe_onset_ms))
    out["onset_censored"] = bool(censored)
    return out


def splice_checkpoint(pre_ictal_state, baseline_state, *, mode):
    """Counterfactual states for the attribution block.

    A spliced state is OFF-MANIFOLD: the dynamics never visit "pre-ictal fast
    state with baseline z". These answer which variable is consistent with
    carrying the elevated responsiveness, not what would have happened. Both
    donors are deep-copied because the same state object is reused across every
    site at a checkpoint.
    """
    if mode not in _SPLICE:
        raise ValueError(f"unknown splice mode {mode!r}")
    host_name, z_name, m_name = _SPLICE[mode]
    donors = {"pre_ictal": pre_ictal_state, "baseline": baseline_state}
    out = copy.deepcopy(donors[host_name])
    if out.get("slow") is None:
        raise ValueError("splice needs a slow payload in the host checkpoint")
    out["slow"]["z"] = np.array(donors[z_name]["slow"]["z"], copy=True)
    out["slow"]["m"] = np.array(donors[m_name]["slow"]["m"], copy=True)
    out["splice_mode"] = mode
    out["off_manifold"] = mode not in ("native_baseline", "native_pre_ictal")
    return out


def hotspot_compactness(sites_xy, values, *, quantile, n_null, seed):
    xy = np.asarray(sites_xy, float)
    values = np.asarray(values, float)
    finite = np.isfinite(values)
    xy, values = xy[finite], values[finite]
    if len(values) < 4:
        return {"status": "NOT_EVALUABLE", "n_sites": int(len(values))}
    cut = np.quantile(values, float(quantile))
    selected = np.flatnonzero(values >= cut)
    if selected.size < 2:
        return {"status": "NOT_EVALUABLE", "n_sites": int(len(values))}

    def _mean_pairwise(index):
        points = xy[index]
        diff = points[:, None, :] - points[None, :, :]
        distance = np.linalg.norm(diff, axis=-1)
        upper = distance[np.triu_indices(len(points), k=1)]
        return float(upper.mean())

    observed = _mean_pairwise(selected)
    rng = np.random.default_rng(int(seed))
    null = np.array([_mean_pairwise(rng.choice(len(values), selected.size, replace=False))
                     for _ in range(int(n_null))])
    return {"status": "OK", "n_sites": int(len(values)),
            "n_hotspot_sites": int(selected.size),
            "observed_mean_pairwise_mm": observed,
            "null_mean_pairwise_mm": float(null.mean()),
            "p_value": float((np.sum(null <= observed) + 1) / (len(null) + 1))}
