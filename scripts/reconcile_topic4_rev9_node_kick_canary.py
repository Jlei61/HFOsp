"""Apply the frozen event-exclusion contract to an existing raw Node canary."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402
from src.topic4_rev9_local_response import fit_response_slope  # noqa: E402


ROOT = Path("results/topic4_sef_hfo/data_driven_core_field_rev9/node_edge_calibration")


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-json", default=str(ROOT / "node_kick_canary.json"))
    parser.add_argument("--raw-npz", default=str(ROOT / "node_kick_canary.npz"))
    parser.add_argument(
        "--out", default=str(ROOT / "node_kick_canary_eligibility.json"))
    args = parser.parse_args()

    raw = json.loads(Path(args.raw_json).read_text())
    if raw["arrays"]["sha256"] != _sha256(args.raw_npz):
        raise RuntimeError("raw JSON/NPZ hash mismatch")
    arrays = np.load(args.raw_npz)
    event = np.asarray(arrays["response_interval_event"], bool)
    runaway = np.asarray(arrays["runaway_early_stop_ms"], float)
    eligible = ~event.any(axis=2) & ~np.isfinite(runaway).any(axis=2)
    multipliers = np.asarray(arrays["amplitude_multipliers"], float)
    amplitudes = np.asarray(arrays["kick_boost_1_per_ms"], float)
    selection_multiplier = float(raw["window_selection"]["amplitude_multiplier"])
    match = np.flatnonzero(np.isclose(multipliers, selection_multiplier))
    if len(match) != 1:
        raise RuntimeError("selection amplitude is absent or ambiguous")
    selection_values = np.asarray(
        arrays["downstream_positive_per_cell"][:, :, int(match[0]), :], float)
    if eligible.any():
        medians = np.nanmedian(
            np.where(eligible[:, :, None], selection_values, np.nan), axis=(0, 1))
        best = np.nanmax(medians)
        selected = int(np.flatnonzero(medians == best)[0])
    else:
        medians = np.full(selection_values.shape[-1], np.nan)
        selected = None

    slope_rows = []
    site_ids = [str(value) for value in arrays["site_ids"]]
    seeds = [int(value) for value in arrays["seeds"]]
    windows = np.asarray(arrays["windows_after_pulse_end_ms"], float)
    for seed_index, seed in enumerate(seeds):
        for site_index, site_id in enumerate(site_ids):
            for window_index in range(len(windows)):
                valid = bool(eligible[seed_index, site_index])
                source = fit_response_slope(
                    amplitudes if valid else [],
                    (arrays["source_signed_per_cell"][
                        seed_index, site_index, :, window_index] if valid else []))
                downstream = fit_response_slope(
                    amplitudes if valid else [],
                    (arrays["downstream_signed_per_cell"][
                        seed_index, site_index, :, window_index] if valid else []))
                slope_rows.append(dict(
                    seed=seed, site_id=site_id, window_index=window_index,
                    eligible=valid, source=source, downstream=downstream))

    status = ("REV9_NODE_CANARY_ELIGIBLE_WINDOW_FROZEN" if selected is not None
              else "REV9_NODE_CANARY_NO_ELIGIBLE_WINDOW")
    payload = dict(
        status=status,
        scientific_interpretation=(
            "event-contaminated pairs remain raw exploratory observations but are "
            "excluded from window selection and response-amplitude slopes"),
        raw_status=raw["status"],
        raw_selected_window_invalidated=bool(
            raw["window_selection"]["selected_index"] is not None and selected is None),
        eligibility=dict(
            contract="site-seed excluded if any amplitude or paired sham has a response-interval detector event, truncation, or runaway",
            n_eligible_site_seed=int(eligible.sum()),
            n_total_site_seed=int(eligible.size),
            matrix=eligible.tolist(),
            n_event_contaminated_kick_pairs=int(event.sum()),
            n_total_kick_pairs=int(event.size),
            n_runaway_kick_pairs=int(np.isfinite(runaway).sum())),
        window_selection=dict(
            amplitude_multiplier=selection_multiplier,
            eligible_downstream_positive_per_cell_median=[
                None if not np.isfinite(value) else float(value) for value in medians],
            selected_index=selected,
            selected_window=(None if selected is None else windows[selected].tolist())),
        slopes=slope_rows,
        next_action=(
            "run sham-only onset scan and freeze a global quiet onset before any edge alpha simulation"
            if selected is None else "continue to edge response surface"),
        inputs=dict(
            raw_json=dict(path=args.raw_json, sha256=_sha256(args.raw_json)),
            raw_npz=dict(path=args.raw_npz, sha256=_sha256(args.raw_npz))),
        provenance=dict(
            **provenance(), git_status_porcelain=_git("status", "--porcelain"),
            producer_sha256=_sha256(__file__), python_executable=sys.executable,
            python_version=platform.python_version()),
    )
    atomic_write_json(payload, args.out)
    print(json.dumps(dict(
        status=status, eligibility=payload["eligibility"],
        window_selection=payload["window_selection"],
        next_action=payload["next_action"]), indent=2))


if __name__ == "__main__":
    main()
