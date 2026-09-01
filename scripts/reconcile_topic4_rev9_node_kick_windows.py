"""Reconcile rev9 canary eligibility against each response window."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
from src.topic4_core_field_runner import atomic_write_json, provenance  # noqa: E402
from src.topic4_rev9_local_response import (  # noqa: E402
    event_window_overlap,
    fit_response_slope,
)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(
            ["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return default


def _git_file(commit, path):
    return subprocess.check_output(["git", "show", f"{commit}:{path}"])


def _atomic_npz(path, arrays):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/topic4_rev9_exploratory.json")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--input-npz", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--source-execution-commit")
    parser.add_argument("--source-wrapper")
    args = parser.parse_args()

    package_lock = "requirements.txt"
    execution_provenance = dict(
        **provenance(), git_status_porcelain=_git("status", "--porcelain"),
        producer_sha256=_sha256(__file__), config_sha256=_sha256(args.config),
        python_executable=sys.executable,
        python_version=platform.python_version(),
        package_lock=dict(path=package_lock, sha256=_sha256(package_lock)),
        systemd_unit=os.environ.get("REV9_SYSTEMD_UNIT"),
    )
    config = json.loads(Path(args.config).read_text())
    instrument = config["small_kick_instrument"]
    source = json.loads(Path(args.input_json).read_text())
    with np.load(args.input_npz, allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}

    seeds = arrays["seeds"].astype(int)
    site_ids = arrays["site_ids"].astype(str)
    amplitudes = arrays["amplitude_multipliers"].astype(float)
    windows = arrays["windows_after_pulse_end_ms"].astype(float)
    pulse_end = float(source["pulse"]["end_ms"])
    shape = (len(seeds), len(site_ids), len(amplitudes), len(windows))
    overlaps = np.zeros(shape, bool)

    seed_index = {int(value): index for index, value in enumerate(seeds)}
    site_index = {str(value): index for index, value in enumerate(site_ids)}
    for run in source["runs"]:
        si = seed_index[int(run["seed"])]
        ji = site_index[str(run["site_id"])]
        matches = np.flatnonzero(np.isclose(
            amplitudes, float(run["amplitude_multiplier"])))
        if len(matches) != 1:
            raise RuntimeError("run amplitude is absent or ambiguous")
        ai = int(matches[0])
        overlaps[si, ji, ai] = event_window_overlap(
            run["event"]["events_in_response_interval"], pulse_end, windows)
        if run.get("paired_sham_response_event", False):
            sham = next(row for row in source["event_diagnostics"]["sham"]
                        if int(row["seed"]) == int(run["seed"]))
            overlaps[si, ji, ai] |= event_window_overlap(
                sham["events_in_response_interval"], pulse_end, windows)

    runaway = np.asarray(arrays["runaway_early_stop_ms"], float)
    eligible = (
        ~overlaps.any(axis=2)
        & ~np.isfinite(runaway).any(axis=2)[:, :, None])
    primary = np.asarray(instrument["primary_window_after_pulse_end_ms"], float)
    matches = np.flatnonzero(np.all(np.isclose(windows, primary), axis=1))
    if len(matches) != 1:
        raise RuntimeError("primary response window is absent or ambiguous")
    selected = int(matches[0])

    source_slopes = np.full((len(seeds), len(site_ids), len(windows)), np.nan)
    downstream_slopes = np.full_like(source_slopes, np.nan)
    slope_records = []
    for si, seed in enumerate(seeds):
        for ji, site_id in enumerate(site_ids):
            for wi in range(len(windows)):
                use = bool(eligible[si, ji, wi])
                source_fit = fit_response_slope(
                    arrays["kick_boost_1_per_ms"] if use else [],
                    arrays["source_signed_per_cell"][si, ji, :, wi] if use else [])
                downstream_fit = fit_response_slope(
                    arrays["kick_boost_1_per_ms"] if use else [],
                    arrays["downstream_signed_per_cell"][si, ji, :, wi]
                    if use else [])
                if source_fit["slope"] is not None:
                    source_slopes[si, ji, wi] = source_fit["slope"]
                if downstream_fit["slope"] is not None:
                    downstream_slopes[si, ji, wi] = downstream_fit["slope"]
                slope_records.append(dict(
                    seed=int(seed), site_id=str(site_id), window_index=wi,
                    window_linear_eligible=use,
                    source=source_fit, downstream=downstream_fit))

    paired = np.asarray(arrays["downstream_signed_per_cell"], float)
    eligible_paired = np.where(eligible[:, :, None, :], paired, np.nan)
    paired_median = np.nanmedian(eligible_paired, axis=0)
    paired_mad = np.nanmedian(
        np.abs(eligible_paired - paired_median[None, ...]), axis=0)
    downstream_snr = np.abs(paired_median) / (1.4826 * paired_mad + 1e-6)

    arrays["broad_interval_site_seed_linear_eligible"] = arrays[
        "site_seed_linear_eligible"]
    arrays["event_window_overlap"] = overlaps
    arrays["site_seed_window_linear_eligible"] = eligible
    arrays["site_seed_linear_eligible"] = eligible[:, :, selected]
    arrays["selected_window_index"] = np.asarray(selected, np.int64)
    arrays["source_slopes"] = source_slopes
    arrays["downstream_slopes"] = downstream_slopes
    arrays["downstream_snr"] = downstream_snr
    _atomic_npz(args.out_npz, arrays)

    selected_eligible = eligible[:, :, selected]
    eligible_seed_mask = selected_eligible.any(axis=1)
    source_execution = None
    if args.source_execution_commit:
        commit = _git("rev-parse", args.source_execution_commit)
        producer_path = "scripts/run_topic4_rev9_node_kick_canary.py"
        config_path = "config/topic4_rev9_exploratory.json"
        producer_blob = _git_file(commit, producer_path)
        config_blob = _git_file(commit, config_path)
        frozen_config = json.loads(config_blob)
        frozen_instrument = frozen_config["small_kick_instrument"]
        semantic_config_match = bool(
            source["seeds"] == frozen_instrument["canary_seeds"]
            and np.allclose(
                source["amplitude_multipliers"],
                frozen_instrument["amplitude_multipliers"])
            and np.allclose(
                source["windows_after_pulse_end_ms"],
                frozen_instrument["candidate_windows_after_pulse_end_ms"])
            and np.isclose(
                source["pulse"]["onset_ms"],
                frozen_instrument["kick_onset_ms"])
            and np.isclose(
                source["pulse"]["duration_ms"],
                frozen_instrument["kick_duration_ms"])
            and np.isclose(
                source["pulse"]["radius_mm"],
                frozen_instrument["radius_mm"])
            and source["sites"] == frozen_instrument["origins"])
        wrapper = None
        wrapper_commit_match = None
        if args.source_wrapper:
            wrapper_text = Path(args.source_wrapper).read_text()
            wrapper = dict(path=args.source_wrapper, sha256=_sha256(args.source_wrapper))
            wrapper_commit_match = bool(
                f"commit={commit[:8]}" in wrapper_text
                and commit[:8] in source["provenance"].get("systemd_unit", ""))
        if not semantic_config_match or wrapper_commit_match is False:
            raise RuntimeError("source execution evidence does not match frozen commit")
        expected_producer_sha = hashlib.sha256(producer_blob).hexdigest()
        expected_config_sha = hashlib.sha256(config_blob).hexdigest()
        payload_provenance_consistent = bool(
            source["provenance"].get("git_commit") == commit
            and source["provenance"].get("producer_sha256")
            == expected_producer_sha
            and source["provenance"].get("config_sha256")
            == expected_config_sha)
        source_execution = dict(
            commit=commit,
            producer=dict(path=producer_path, sha256=expected_producer_sha),
            config=dict(path=config_path, sha256=expected_config_sha),
            wrapper=wrapper,
            wrapper_commit_match=wrapper_commit_match,
            semantic_config_match=semantic_config_match,
            payload_end_of_run_provenance_consistent=payload_provenance_consistent,
            note=(
                "Raw payload captured file hashes at process end; use this "
                "start-commit reconstruction when consistency is false"))
    payload = dict(
        status="REV9_NODE_KICK_WINDOW_RECONCILED",
        scientific_role=(
            "Window-specific eligibility correction; no simulation, alpha "
            "selection, equivalence claim, or patient validation"),
        source=dict(
            json=dict(path=args.input_json, sha256=_sha256(args.input_json)),
            npz=dict(path=args.input_npz, sha256=_sha256(args.input_npz)),
            status=source["status"], provenance=source["provenance"],
            execution=source_execution),
        pulse=source["pulse"], seeds=seeds.tolist(), sites=source["sites"],
        amplitude_multipliers=amplitudes.tolist(),
        kick_boost_1_per_ms=arrays["kick_boost_1_per_ms"].tolist(),
        windows_after_pulse_end_ms=windows.tolist(),
        window_selection=dict(
            selection_rule="predefined_first_generation_window",
            selected_index=selected, selected_window=windows[selected].tolist(),
            eligible_site_seed_by_window=eligible.sum(axis=(0, 1)).tolist(),
            n_eligible_seeds=int(eligible_seed_mask.sum()),
            eligible_seeds=seeds[eligible_seed_mask].tolist(),
            cross_network_support=bool(eligible_seed_mask.sum() >= 2),
            support_role="diagnostic only; not an execution gate"),
        event_diagnostics=dict(
            n_event_overlap_by_window=overlaps.sum(axis=(0, 1, 2)).tolist(),
            n_runaway=int(np.isfinite(runaway).sum())),
        slopes=slope_records,
        arrays=dict(path=args.out_npz, sha256=_sha256(args.out_npz)),
        provenance=dict(
            **execution_provenance,
            network_seed=seeds.tolist(), readout_seed=None),
    )
    atomic_write_json(payload, args.out_json)
    print(json.dumps(dict(
        status=payload["status"],
        window_selection=payload["window_selection"],
        event_diagnostics=payload["event_diagnostics"],
        arrays_sha256=payload["arrays"]["sha256"]), indent=2))


if __name__ == "__main__":
    main()
