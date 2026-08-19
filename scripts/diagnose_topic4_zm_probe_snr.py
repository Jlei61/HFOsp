#!/usr/bin/env python
"""Can the registered probe endpoint see the probe at all?

The dose endpoint is `susceptibility`: descendant-only excess spikes summed over
a 200 ms window. The injected packet's own frame is already excluded. What is
NOT excluded is the network's own spontaneous activity: if the probe advances,
delays or prevents one spontaneous event inside the window, the difference
against the sham is the size of that whole event.

This script puts the two magnitudes side by side:

  signal  the probe's direct effect, read from the first 50 ms where its
          descendants dominate (`excess_spikes_early`, already a registered
          reported quantity)
  noise   the size of one spontaneous event, estimated from the same run's
          own activity trace, not assumed

It changes nothing. Switching the dose endpoint from the 200 ms sum to the
50 ms window would be an amendment to a pre-registered endpoint and is a
decision for the round owner, not for a diagnostic.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic4_zm_ictal_transition import load_round_config  # noqa: E402


def spontaneous_event_scale(active_fraction, bin_ms, n_e, *, window_ms,
                            quantile=99.0):
    """Spikes contributed by one spontaneous event, from the run's own trace.

    Taken as the largest contiguous excursion the trace actually shows, not as
    peak x duration of an idealised event: the point is the magnitude a single
    event can move the 200 ms difference by, and that is what the trace holds.
    """
    active = np.asarray(active_fraction, float)
    width = max(1, int(round(float(window_ms) / float(bin_ms))))
    if width > active.size:
        # mode="valid" would return an empty array here and every statistic
        # below would come back nan without raising -- a noise floor of nan
        # reads as "no noise" to anything that compares against it.
        raise ValueError(f"window of {window_ms} ms needs {width} bins but the "
                         f"trace has {active.size}")
    counts = np.convolve(active * n_e, np.ones(width), mode="valid")
    return {"window_ms": float(window_ms),
            "median_spikes_in_window": float(np.median(counts)),
            "q99_spikes_in_window": float(np.percentile(counts, quantile)),
            "max_spikes_in_window": float(counts.max()),
            "n_windows": int(counts.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--replay-seed", type=int, default=1801)
    ap.add_argument("--candidate-id", default="joint_04_control")
    ap.add_argument("--baseline-window-ms", type=float, nargs=2, default=[500.0, 1000.0])
    args = ap.parse_args()

    config = load_round_config(args.config)
    output_root = ROOT / config["output_root"]
    perturbation = config["perturbation"]
    window_ms = float(perturbation["response_window_ms"])
    split_ms = float(perturbation["response_split_ms"])

    by_rung = collections.defaultdict(lambda: collections.defaultdict(list))
    for path in sorted(glob.glob(str(output_root / "dose" / "*.json"))):
        match = re.search(r"seed_(\d+)_low_activity_n(\d+)\.json$", path)
        if not match:
            continue
        rung = int(match.group(2))
        for row in json.loads(Path(path).read_text())["rows"]:
            by_rung[rung]["full"].append(row["susceptibility"])
            by_rung[rung]["early"].append(row["excess_spikes_early"])
            by_rung[rung]["late"].append(row["excess_spikes_late"])
            by_rung[rung]["event"].append(row["probe_attributable_event_200ms"])

    # noise scale, from the run's own trace over the baseline window
    replay = (output_root / "fig5_replay"
              / f"{args.candidate_id}_seed_{args.replay_seed}_frames.npz")
    with np.load(replay, allow_pickle=False) as z:
        active = np.asarray(z["active_fraction"], float)
        bin_ms = float(z["active_fraction_bin_ms"])
    lo, hi = args.baseline_window_ms
    segment = active[int(lo / bin_ms):int(hi / bin_ms)]
    noise_full = spontaneous_event_scale(segment, bin_ms, 32000, window_ms=window_ms)
    noise_early = spontaneous_event_scale(segment, bin_ms, 32000, window_ms=split_ms)

    ladder = {}
    for rung in sorted(by_rung):
        entry = by_rung[rung]
        ladder[rung] = {
            "n_units": len(entry["full"]),
            "full_window": {"median": float(np.median(entry["full"])),
                            "abs_max": float(np.abs(entry["full"]).max())},
            "early_window": {"median": float(np.median(entry["early"])),
                             "abs_max": float(np.abs(entry["early"]).max())},
            "late_window": {"median": float(np.median(entry["late"])),
                            "abs_max": float(np.abs(entry["late"]).max())},
            "n_probe_attributable_events": int(sum(entry["event"]))}

    def _ratios(key):
        rungs = sorted(ladder)
        out = {}
        for a, b in zip(rungs[:-1], rungs[1:]):
            lo_v = ladder[a][key]["median"]
            out[f"{a}->{b}"] = (float(ladder[b][key]["median"] / lo_v)
                                if abs(lo_v) > 1e-9 else None)
        return out

    report = {
        "status": "ZM_PROBE_SNR_DIAGNOSTIC",
        "question": "can the registered 200 ms endpoint see the probe at all?",
        "ladder": ladder,
        "rung_to_rung_ratio": {"full_window": _ratios("full_window"),
                               "early_window": _ratios("early_window")},
        "spontaneous_event_scale": {
            "measured_from": str(replay.relative_to(ROOT)),
            "baseline_window_ms": [lo, hi],
            f"{int(window_ms)}ms": noise_full,
            f"{int(split_ms)}ms": noise_early},
        "reading": [
            "The probe's own effect is confined to the first 50 ms and is monotone "
            "in dose there. The 150 ms remainder has a median near zero and "
            "excursions of the same magnitude as one spontaneous event.",
            "The 200 ms endpoint therefore carries a noise term that does not "
            "shrink with a smaller packet, because it is the network's activity "
            "and not the probe's.",
            "The median over site units survives this, because triggering an event "
            "is rare. A PER-SITE spatial map would not: at one unit per site the "
            "event term is the whole measurement.",
        ],
        "not_done_here": (
            "Switching the dose endpoint from the 200 ms sum to the 50 ms window "
            "would amend a pre-registered endpoint. This script does not do that, "
            "and the dose gate still runs on the registered quantity."),
    }
    out = output_root / "probe_snr_diagnostic.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"ladder": {k: {"n": v["n_units"],
                                     "full_median": v["full_window"]["median"],
                                     "early_median": v["early_window"]["median"],
                                     "late_abs_max": v["late_window"]["abs_max"],
                                     "events": v["n_probe_attributable_events"]}
                                 for k, v in ladder.items()},
                      "one_event_scale_200ms_q99": noise_full["q99_spikes_in_window"],
                      "one_event_scale_200ms_max": noise_full["max_spikes_in_window"]},
                     indent=2))


if __name__ == "__main__":
    main()
