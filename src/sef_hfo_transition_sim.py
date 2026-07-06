"""M3A-v2.2 approach-criticality transition sim: factored no-GIF entry + handoff adapter (Task 1).

`run_transition()` is the no-GIF entry point for the M3A-v2.2 q_I build-up -> runaway
transition sim. It WRAPS the canonical integration loop that lives in the paper-figure module
`scripts/paper_figures/plot_fig_m3a_v2_2_hG_runaway_transition_gif.py` (`_build` +
`_simulate_continuous`) -- one source of truth, no duplicated SNN loop -- and adds the two
things the M3A->M3B handoff needs but the raw sim does not emit: a deterministic `events`
list (the runaway onset the figure already detects in `_activity_metrics`) and `dt_ms`.

`sim_dict_for_handoff()` adapts the v2.2 trace keys to the v1 tank-name schema that
`src/sef_hfo_m3a_export.build_handoff_from_sim` reads.

`default_transition_config()` returns the config-of-record as a plain dict (one type);
`run_transition` maps it back to the figure's `ProtocolConfig` dataclass internally.

Byte-parity: run_transition never touches `_simulate_continuous`'s body or RNG order, so its
sim outputs are byte-identical to the figure code (guarded by
tests/test_topic4_crit_integration.py against tests/fixtures/topic4_m3v2_2_transition_golden.*).
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
_FIGURE_MODULE = "scripts.paper_figures.plot_fig_m3a_v2_2_hG_runaway_transition_gif"


def _figure_module():
    """Lazily import the paper-figure module that owns the canonical sim body.

    Lazy so importing this module never triggers the figure's heavy imports
    (imageio/matplotlib/SNN engine) until a transition is actually run.
    """
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    return importlib.import_module(_FIGURE_MODULE)


def default_transition_config(layout: str = "subject1146", top: str = "qI") -> dict:
    """Config-of-record for the v2.2 transition sim as a plain dict (one type).

    Full `ProtocolConfig` field set (via asdict) so `run_transition` round-trips it exactly;
    carries `use_gK`, which the interface export reads as `cfg["use_gK"]`.
    """
    G = _figure_module()
    return asdict(G.ProtocolConfig(layout=layout, top=top))


def _detect_events(G, res: dict, dt_ms: float) -> list:
    """Deterministic event list = the runaway onset the v2.2 figure already relies on.

    Reuses the figure's own `_smooth_rate` + `_first_sustained` (the SAME detector
    `_activity_metrics` uses for `runaway_start_ms`, §6.1 helper-reuse), so the handoff's
    event notion cannot drift from what the figure reports. t_off = sim end;
    t_peak = argmax of the smoothed rate. Empty list when no runaway is detected.
    """
    rate_s = G._smooth_rate(res["rate_E"], dt_ms, 20.0)
    onset = G._first_sustained(rate_s, dt_ms)
    if onset is None:
        return []
    times = res["times"]
    return [{
        "t_on": float(onset),
        "t_off": float(times[-1]),
        "t_peak": float(times[int(np.argmax(rate_s))]),
    }]


def run_transition(cfg: dict) -> dict:
    """Run the no-GIF v2.2 transition sim from a config dict; add `events` + `dt_ms`.

    Wraps the figure's `_build` + `_simulate_continuous(..., record_gif=False)` unchanged, so
    the sim outputs stay byte-identical to the figure code. `dt_ms` is taken from the built
    network (`S["p"].dt`), NOT from the cfg dict. Returns the raw sim dict plus `dt_ms` and a
    deterministic `events` list.
    """
    G = _figure_module()
    pcfg = G.ProtocolConfig(**cfg)
    S = G._build(pcfg)
    res = G._simulate_continuous(S, pcfg, record_gif=False)
    res["dt_ms"] = float(S["p"].dt)
    res["events"] = _detect_events(G, res, res["dt_ms"])
    return res


def sim_dict_for_handoff(res: dict) -> dict:
    """Adapt v2.2 traces to the v1 tank-name schema `build_handoff_from_sim` reads.

    trace_core   <- trace_qI_min   (most-depleted spot ~ core disinhibition)
    trace_global <- trace_qI_mean  (sheet-mean inhibitory resource)
    trace_gk     <- trace_gK_axial (axial fatigue field)

    spk/posE are intentionally omitted: the T1 export calls `build_handoff_from_sim` without
    af/bin_w/L, so `event_metrics` is None and spk/posE are never accessed.
    """
    return {
        "trace_core": res["trace_qI_min"],
        "trace_global": res["trace_qI_mean"],
        "trace_gk": res["trace_gK_axial"],
    }
