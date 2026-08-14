#!/usr/bin/env python3
"""Exact-state continuation of the single preselected LC5v2.1 contained/finite candidate."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts", ROOT / "src/snn_engine"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import aggregate_topic4_fcxr_lc5v2p1_phase_map as AGG  # noqa: E402
import run_topic4_fcxr_lc5v2_natural_prefix as PREFIX  # noqa: E402
from src.topic4_fcxr_lc3 import run_fcxr_loop, state_hash  # noqa: E402
from src.topic4_fcxr_lc5 import (  # noqa: E402
    AtomicStageBundle,
    ExactInputHasher,
    RecurrentDriveBlockObserver,
    SparseSpikeBinaryWriter,
    SparseSpikeStream,
    load_sparse_spike_stream,
)


U2 = PREFIX.U2
TARGET_POST_ONSET_MS = 20000.0
CLASSIFIER_AUDIT = U2.OUT / "lc1_classifier_snapshot_replay_audit.json"
TRACE_KEYS = (
    "D_mean", "H_mean", "H_source_mean", "gErec_mean", "u_mean", "u_max",
    "pump_phi_mean", "pump_current_mean", "pump_current_max", "clip_frac",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    PREFIX._write_json(path, payload)


def _combine_full(original, continuation):
    chunk_steps = int(round(U2.CHUNK_MS / U2.DT_MS))
    steps = [original.steps]
    cells = [original.cells]
    for index, stream in enumerate(continuation):
        steps.append(stream.steps + original.n_steps + index * chunk_steps)
        cells.append(stream.cells)
    return SparseSpikeStream(
        np.concatenate(steps), np.concatenate(cells),
        original.n_steps + len(continuation) * chunk_steps, original.n_cells,
    )


def _rate_from_stream(stream):
    counts = np.bincount(stream.steps, minlength=stream.n_steps).astype(float)
    return counts / stream.n_cells / U2.DT_MS * 1000.0


def chunk_mean_rate_hz(stream):
    duration_s = stream.n_steps * U2.DT_MS / 1000.0
    if duration_s <= 0.0:
        raise ValueError("chunk duration must be positive")
    return float(stream.steps.size / stream.n_cells / duration_s)


def _classifier_audit_contract():
    if not CLASSIFIER_AUDIT.is_file():
        raise RuntimeError("LC1 classifier snapshot replay audit is absent")
    audit = json.loads(CLASSIFIER_AUDIT.read_text())
    if audit.get("status") != "LC1_CLASSIFIER_SNAPSHOT_REPLAY_PASS":
        raise RuntimeError("LC1 classifier snapshot replay audit did not pass")
    if int(audit.get("n_pass", -1)) != int(audit.get("n_bundles", -2)):
        raise RuntimeError("LC1 classifier snapshot replay audit is incomplete")
    snapshot = ROOT / "config/topic4_fcxr_lc1_seed1_classifier_snapshot.json"
    if audit.get("snapshot_sha256") != _sha(snapshot):
        raise RuntimeError("LC1 classifier snapshot changed after its replay audit")
    return audit


def recovery_inventory(work, *, total_chunks):
    """Read the progress ledger without touching a live SNN state."""
    work = Path(work)
    progress_path = work / "progress.json"
    checkpoint = work / "rolling_checkpoint.npz"
    if not progress_path.is_file() or not checkpoint.is_file():
        raise RuntimeError("partial continuation lacks progress or rolling checkpoint")
    progress = json.loads(progress_path.read_text())
    completed = int(progress.get("completed_chunks", -1))
    if not 0 < completed < int(total_chunks):
        raise RuntimeError("partial continuation completed_chunks is outside the resumable range")
    spike_paths = [work / f"chunk_{index:02d}_spikes.npz" for index in range(completed)]
    missing_spikes = [str(path) for path in spike_paths if not path.is_file()]
    if missing_spikes:
        raise RuntimeError(f"partial continuation lacks spike chunks: {missing_spikes}")
    trace_paths = [work / f"chunk_{index:02d}_traces.npz" for index in range(completed)]
    input_paths = [work / f"chunk_{index:02d}_input.json" for index in range(completed)]
    return {
        "progress": progress,
        "completed_chunks": completed,
        "checkpoint": checkpoint,
        "spike_paths": spike_paths,
        "trace_paths": trace_paths,
        "input_paths": input_paths,
    }


def _archive_prior_failure(failed, tag):
    if not failed.is_file():
        return None
    payload = json.loads(failed.read_text())
    expected = "FileNotFoundError: [Errno 2] No such file or directory:"
    if not str(payload.get("error", "")).startswith(expected):
        raise RuntimeError("partial continuation has an unreviewed failure reason")
    superseded = U2.OUT / "superseded" / f"{tag}_baseline_path_failure_after_chunk1"
    superseded.mkdir(parents=True, exist_ok=True)
    target = superseded / failed.name
    if not target.exists():
        os.replace(failed, target)
    else:
        failed.unlink()
    _write_json(superseded / "recovery_note.json", {
        "status": "SUPERSEDED_INSTRUMENT_PATH_FAILURE",
        "scientific_chunks_retained": 1,
        "reason": "the simulation completed 25-26 s; only the read-only reducer used a deleted sibling path",
    })
    return target


def continuation_schedule(
    source_T_ms, onset_ms, target_post_onset_ms=TARGET_POST_ONSET_MS,
    *, target_total_ms=None,
):
    """Return an exact-state continuation horizon ending target time after onset."""
    source_T_ms = float(source_T_ms)
    if onset_ms is None:
        raise ValueError("candidate continuation requires an observed natural onset")
    onset_ms = float(onset_ms)
    if target_total_ms is None:
        target_total_ms = onset_ms + float(target_post_onset_ms)
    else:
        target_total_ms = float(target_total_ms)
    continuation_ms = target_total_ms - source_T_ms
    if continuation_ms <= 0:
        raise ValueError("source already reaches the locked post-onset target")
    n_chunks = continuation_ms / U2.CHUNK_MS
    if not np.isclose(n_chunks, round(n_chunks), rtol=0.0, atol=1e-9):
        raise ValueError("continuation horizon must be an integer number of chunks")
    return target_total_ms, continuation_ms


def _plot_extension(arm, result, rate10, traces):
    figures = Path(arm) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    t = np.arange(rate10.size) * U2.TRACE_DT_MS / 1000.0
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 7.4), sharex=True, constrained_layout=True)
    axes[0].plot(t, np.maximum(rate10, 1e-3), color="#333333", linewidth=.8)
    axes[0].set_yscale("log"); axes[0].set_ylabel("E rate (Hz)")
    axes[0].set_title("a  Natural entry and candidate continuation")
    axes[1].plot(t, traces["D_mean"], color="#B2182B", label="D = 1-z")
    axh = axes[1].twinx()
    axh.plot(t, traces["H_mean"], color="#2166AC", label="H")
    axes[1].set_ylabel("D"); axh.set_ylabel("H")
    axes[1].set_title("b  Entry coordinate and cooperative drive")
    axes[2].plot(t, traces["u_mean"], color="#E69F00", label="mean U load")
    axp = axes[2].twinx()
    axp.plot(t, traces["pump_current_mean"], color="#2CA25F", label="mean pump current")
    axes[2].set_ylabel("Mean U"); axp.set_ylabel("Pump current")
    axes[2].set_xlabel("Simulation time (s)")
    axes[2].set_title("c  Episode memory and delivered recovery current")
    for ax in axes:
        if result.get("onset_ms") is not None:
            ax.axvline(result["onset_ms"] / 1000.0, color="#B2182B", linestyle="--", linewidth=1)
        if result.get("offset_ms") is not None:
            ax.axvline(result["offset_ms"] / 1000.0, color="#2CA25F", linestyle="--", linewidth=1)
        ax.grid(alpha=.16)
    fig.suptitle(
        f"LC5v2.1 exact continuation: {result['outcome'].replace('_', ' ').lower()}", fontsize=13
    )
    png = figures / "lc5v2p1_candidate_extension.png"
    pdf = figures / "lc5v2p1_candidate_extension.pdf"
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    (figures / "README.md").write_text(
        "### lc5v2p1_candidate_extension.png\n\n"
        f"这张诊断图从自然轨迹 {result['source_T_ms']/1000:.1f} 秒的完整状态继续推进到 onset 后 "
        f"{result['target_post_onset_ms']/1000:.1f} 秒，不重置膜电位、突触、慢变量或随机数。a 看候选高态最后是终止、维持还是再次升级；b 看 D 与 H 的移动边界；c 看逐细胞 U 负荷及其恢复电流是否在活动下降后继续保留。\n\n"
        "**关注点**：只有绿色 offset 线之后同时出现低活动保护、D 恢复并最终恢复 returning IED，才构成完整 lifecycle；单独 containment 或 rate 下降不算。\n\n"
        "### lc5v2p1_candidate_extension.pdf\n\n与 PNG 相同的矢量版本。\n\n"
        "**关注点**：这是单 seed 机制诊断，不是鲁棒性或患者队列主张。\n"
    )
    _write_json(figures / "lc5v2p1_candidate_extension_metadata.json", {
        "summary": str(Path(arm) / "summary.json"), "outcome": result["outcome"],
        "onset_ms": result.get("onset_ms"), "offset_ms": result.get("offset_ms"),
        "png": str(png), "pdf": str(pdf),
    })


def _manifest_contract(path):
    path = Path(path).resolve()
    payload = json.loads(path.read_text())
    if payload.get("experiment_id") != "fcxr_lc6a_patient_axis_surround":
        raise RuntimeError("wrong LC6A execution manifest")
    contract = payload["lc5_continuation"]
    source_summary = ROOT / contract["source_summary"]
    if _sha(source_summary) != contract["source_summary_sha256"]:
        raise RuntimeError("LC5 source summary hash mismatch")
    source_state = source_summary.parent / "final_state.npz"
    if _sha(source_state) != contract["source_final_state_sha256"]:
        raise RuntimeError("LC5 source final-state artifact hash mismatch")
    source_payload = json.loads(source_summary.read_text())
    locked_fields = {
        "T_ms": "source_t_ms",
        "onset_ms": "source_onset_ms",
        "final_state_hash": "source_final_state_hash",
    }
    for source_key, contract_key in locked_fields.items():
        if source_payload.get(source_key) != contract.get(contract_key):
            raise RuntimeError(f"LC5 source {source_key} disagrees with execution manifest")
    for relative, expected in payload["blessed_engine_sha256"].items():
        if _sha(ROOT / relative) != expected:
            raise RuntimeError(f"blessed engine hash mismatch: {relative}")
    return path, payload, contract, source_summary


def _locked_config_from_summary(summary, p0, ne):
    """Rebuild the source arm's config without consulting deleted historical worktrees."""
    if "config_scalar" not in summary:
        raise RuntimeError("source summary lacks locked config_scalar")
    cfg = dict(summary["config_scalar"])
    scalar_checks = {
        "pump_Imax": float(summary["Imax"]),
        "pump_a_load": float(summary["a_load"]),
        "pump_tau_ms": float(summary["tau_ms"]),
    }
    for key, expected in scalar_checks.items():
        if not np.isclose(float(cfg[key]), expected, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"source summary config mismatch: {key}")
    cfg.update(
        pump_p0_E=np.asarray(p0, float).copy(),
        pump_u_init_E=np.zeros(int(ne), dtype=float),
        x_relay_frozen_E=np.ones(int(ne), dtype=float),
    )
    return cfg


def _fresh_system_from_locked_summary(summary, p0):
    S = U2.PP.build_substrate(U2.CONNECTION_SEED)
    U2.install_registered_noise_rng(S["net"])
    cfg = _locked_config_from_summary(summary, p0, S["NE"])
    slow = PREFIX.MZSlowVars(
        S["N"], 18.0, PREFIX.MZSlowVarsConfig(**cfg), NE=S["NE"],
        core_mask_E=U2.OLD_SLOW.build_core_masks(S),
    )
    S["net"]["rng"] = np.random.default_rng(U2.NOISE_SEED)
    return S, slow, cfg


def run(source_summary, *, target_total_ms=None, output_tag=None, execution_manifest=None):
    source_summary = Path(source_summary).resolve()
    source = source_summary.parent
    summary = json.loads(source_summary.read_text())
    execution_manifest_sha256_start = (
        _sha(execution_manifest) if execution_manifest is not None else None
    )
    if summary["outcome"] not in {"CONTAINED_HIGH_NO_OFFSET", "FINITE_EXCURSION_CANDIDATE"}:
        raise RuntimeError("source is not an extension-eligible candidate")
    target_total_ms, continuation_ms = continuation_schedule(
        summary["T_ms"], summary.get("onset_ms"), target_total_ms=target_total_ms,
    )
    tau_ms = float(summary["tau_ms"])
    gamma = float(summary["gamma_nominal_dose"])
    tag = output_tag or (
        f"lc5v2p1_candidate_extension_tau{int(tau_ms)}_gamma{int(round(gamma*1000)):04d}"
    )
    arm = U2.OUT / tag
    work = U2.OUT / f".{tag}.work"
    sentinel = tag.upper()
    if arm.is_dir():
        return json.loads((arm / "summary.json").read_text())
    lock = (U2.OUT / f".{tag}.lock").open("w")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise SystemExit(f"{tag} is already running") from exc
    running = U2.OUT / f"RUNNING_{sentinel}.json"
    failed = U2.OUT / f"FAILED_{sentinel}.json"
    done = U2.OUT / f"DONE_{sentinel}.json"
    try:
        classifier_audit = _classifier_audit_contract()
        resources = U2.GEO._meminfo()
        if resources["mem_available_gib"] < 96.0:
            raise RuntimeError("candidate continuation requires at least 96 GiB MemAvailable")
        baseline_swap = float(resources["swap_used_mib"])
        prelock, p0, imax = PREFIX._p0_contract(gamma, "q099", tau_ms)
        if not np.isclose(float(imax), float(summary["Imax"]), rtol=0.0, atol=1e-10):
            raise RuntimeError("candidate Imax drift")
        S, slow, _ = _fresh_system_from_locked_summary(summary, p0)
        template = U2.PM._seed_template(S, slow)
        state = U2.load_into(source / "final_state.npz", template)
        expected_step = int(round(float(summary["T_ms"]) / U2.DT_MS))
        if int(state.t) != expected_step or state_hash(state) != summary["final_state_hash"]:
            raise RuntimeError("source exact-state mismatch")
        source_state_hash = state_hash(state)
        original = load_sparse_spike_stream(source / "spikes.npz")
        total_chunks = int(round(continuation_ms / U2.CHUNK_MS))
        chunk_steps = int(round(U2.CHUNK_MS / U2.DT_MS))
        chunk_trace_n = int(round(U2.CHUNK_MS / U2.TRACE_DT_MS))
        trace_parts = {key: [] for key in TRACE_KEYS}
        streams, input_segments, missing_trace_intervals = [], [], []
        start_chunk = 0
        recovered_from_partial = False
        if work.exists():
            inventory = recovery_inventory(work, total_chunks=total_chunks)
            start_chunk = int(inventory["completed_chunks"])
            _archive_prior_failure(failed, tag)
            resumed = U2.load_into(inventory["checkpoint"], state)
            expected_resumed_step = expected_step + start_chunk * chunk_steps
            if int(resumed.t) != expected_resumed_step:
                raise RuntimeError("partial continuation checkpoint is at the wrong simulation step")
            if state_hash(resumed) != inventory["progress"].get("state_hash"):
                raise RuntimeError("partial continuation checkpoint disagrees with progress ledger")
            state = resumed
            streams = [load_sparse_spike_stream(path) for path in inventory["spike_paths"]]
            for index, (trace_path, input_path) in enumerate(
                zip(inventory["trace_paths"], inventory["input_paths"])
            ):
                if trace_path.is_file():
                    with np.load(trace_path, allow_pickle=False) as z:
                        for key in TRACE_KEYS:
                            trace_parts[key].append(np.asarray(z[key], np.float32))
                else:
                    for key in TRACE_KEYS:
                        trace_parts[key].append(np.full(chunk_trace_n, np.nan, np.float32))
                    missing_trace_intervals.append([
                        float(summary["T_ms"]) + index * U2.CHUNK_MS,
                        float(summary["T_ms"]) + (index + 1) * U2.CHUNK_MS,
                    ])
                if input_path.is_file():
                    input_segments.append(json.loads(input_path.read_text()))
                else:
                    input_segments.append({
                        "chunk": index,
                        "window_ms": [
                            float(summary["T_ms"]) + index * U2.CHUNK_MS,
                            float(summary["T_ms"]) + (index + 1) * U2.CHUNK_MS,
                        ],
                        "sha256": None,
                        "status": "UNAVAILABLE_AFTER_REDUCER_PATH_FAILURE",
                    })
            recovered_from_partial = True
        else:
            work.mkdir(parents=True)
        _write_json(running, {
            "status": "RUNNING", "pid": os.getpid(), "source_summary": str(source_summary),
            "source_summary_sha256": _sha(source_summary), "source_state_hash": source_state_hash,
            "tau_ms": tau_ms, "gamma": gamma, "continuation_ms": continuation_ms,
            "target_total_ms": target_total_ms,
            "resumed_from_partial": recovered_from_partial,
            "resumed_completed_chunks": start_chunk,
            "execution_manifest": str(execution_manifest) if execution_manifest else None,
            "execution_manifest_sha256_start": execution_manifest_sha256_start,
        })
        started = time.time()
        stride = int(round(U2.TRACE_DT_MS / U2.DT_MS))
        force_scale = float(state.slow.cfg.E_E - state.slow.cfg.v_match)
        state.slow.recurrent_drive_observer = RecurrentDriveBlockObserver(
            S["NE"], sample_every=stride,
            steps_per_block=int(round(1000.0 / U2.DT_MS)), force_scale=force_scale,
        )
        attrs = (
            "trace_z_mean", "trace_h_lc2_mean", "trace_gA_raw_lc2_mean",
            "trace_gErec_mean", "trace_u_mean", "trace_u_max", "trace_phi_pump_mean",
            "trace_pump_excess_mean", "trace_pump_excess_max", "trace_conductance_clip_frac",
        )
        early_stop_reason = None
        p = dataclasses.replace(S["p"], T=continuation_ms, dt=U2.DT_MS)
        for chunk in range(start_chunk, total_chunks):
            if (execution_manifest is not None and
                    _sha(execution_manifest) != execution_manifest_sha256_start):
                raise RuntimeError("execution manifest drifted during LC5 continuation")
            starts = {name: len(getattr(state.slow, name)) for name in attrs}
            binary = work / f"chunk_{chunk:02d}.bin"
            writer = SparseSpikeBinaryWriter(
                binary, step_origin=state.t, n_steps=chunk_steps, n_cells=S["NE"]
            )
            chunk_input_hasher = ExactInputHasher()
            run_out = run_fcxr_loop(
                p, S["net"], start=state, n_steps=chunk_steps, capture_final=True,
                store_spikes=False, spike_sink=writer, input_sink=chunk_input_hasher,
                v_th_per_neuron=S["vth"],
            )
            state = run_out["checkpoint"]
            stream = writer.finalize(work / f"chunk_{chunk:02d}_spikes.npz")
            binary.unlink(missing_ok=True)
            streams.append(stream)
            sliced = U2._trace_slice(state.slow, starts, stride)
            for key, value in sliced.items():
                trace_parts[key].append(value)
            PREFIX._npz_atomic(
                work / f"chunk_{chunk:02d}_traces.npz",
                **{key: np.asarray(value, np.float32) for key, value in sliced.items()},
            )
            input_row = {
                "chunk": chunk,
                "window_ms": [
                    float(summary["T_ms"]) + chunk * U2.CHUNK_MS,
                    float(summary["T_ms"]) + (chunk + 1) * U2.CHUNK_MS,
                ],
                "sha256": chunk_input_hasher.sha256,
                "status": "RECORDED",
            }
            _write_json(work / f"chunk_{chunk:02d}_input.json", input_row)
            input_segments.append(input_row)
            row = U2._resource_row(
                f"{sentinel}_CHUNK", baseline_swap, chunk=chunk + 1,
                completed_total_ms=state.t * U2.DT_MS, wall_s=time.time() - started,
            )
            U2.save_loop_state(str(work / "rolling_checkpoint.npz"), state)
            _write_json(work / "progress.json", {
                "status": "RUNNING", "completed_chunks": chunk + 1,
                "completed_total_ms": state.t * U2.DT_MS, "state_hash": state_hash(state),
                "resource_action": row["action"], "last_input_sha256": chunk_input_hasher.sha256,
            })
            if row["action"] == "TERMINATE_AFTER_CHECKPOINT":
                raise RuntimeError("RESOURCE_STOP_AFTER_CHECKPOINT")
            terminal = False
            if chunk_mean_rate_hz(stream) >= float(U2.SAT_CEILING_HZ):
                early_stop_reason = "REGISTERED_SATURATION_REACHED"
                terminal = True
            else:
                partial_full = _combine_full(original, streams)
                partial_rate = _rate_from_stream(partial_full)
                partial = PREFIX._adjudicate(partial_full, partial_rate)
                partial_offset = partial.get("offset_ms")
                partial_total_ms = partial_full.n_steps * U2.DT_MS
                if (partial_offset is not None and
                        partial_total_ms - float(partial_offset) >= 2000.0):
                    early_stop_reason = "OFFSET_PLUS_2S_LOW_OBSERVED"
                    terminal = True
            if terminal:
                break

        full = _combine_full(original, streams)
        rate = _rate_from_stream(full)
        adjudication = PREFIX._adjudicate(full, rate)
        with np.load(source / "traces.npz", allow_pickle=False) as z:
            old_traces = {name: np.asarray(z[name]) for name in z.files}
        new_traces = {name: np.concatenate(parts) for name, parts in trace_parts.items()}
        combined_traces = {}
        for name, old in old_traces.items():
            if name in {"rate_dt_ms", "af", "af_dt_ms", "rate_E"}:
                continue
            if name in new_traces:
                combined_traces[name] = np.concatenate([old, new_traces[name]])
        rate10 = rate[::stride].astype(np.float32)
        af, af_dt = adjudication["af"], adjudication["af_dt"]
        u_diag = PREFIX._u_tail_diagnostics(
            state.slow.u_pump_E, p0, tau_ms, combined_traces["u_mean"], U2.TRACE_DT_MS
        )
        result = {
            "status": "COMPLETE", "arm": tag, "outcome": adjudication["outcome"],
            "source_summary": str(source_summary), "source_summary_sha256": _sha(source_summary),
            "source_outcome": summary["outcome"], "source_state_hash": summary["final_state_hash"],
            "runtime_semantics": "exact_state_continuation_no_parameter_change",
            "tau_ms": tau_ms, "gamma_nominal_dose": gamma, "Imax": imax,
            "a_load": float(prelock["a_load"]), "p0_policy": "q099", "h": 3,
            "source_T_ms": float(summary["T_ms"]),
            "target_post_onset_ms": target_total_ms - float(summary["onset_ms"]),
            "target_total_ms": target_total_ms,
            "requested_continuation_ms": continuation_ms,
            "actual_continuation_ms": len(streams) * U2.CHUNK_MS,
            "recovered_from_partial_exact_checkpoint": recovered_from_partial,
            "recovered_completed_chunks": start_chunk,
            "missing_diagnostic_trace_intervals_ms": missing_trace_intervals,
            "early_stop_reason": early_stop_reason,
            "T_ms": full.n_steps * U2.DT_MS,
            "onset_ms": adjudication["onset_ms"], "offset_ms": adjudication["offset_ms"],
            "lifecycle": adjudication["lifecycle"], "n_events": len(adjudication["events"]),
            "n_returning": len(adjudication["returned"]),
            "per_second_mean_rate_hz": [float(x["mean_hz"]) for x in adjudication["reports"]],
            "mean_rate_hz": float(np.mean(rate)),
            "end_rate_hz": float(np.mean(rate[-int(round(1000.0/U2.DT_MS)):])),
            "u_tail_diagnostics": u_diag,
            "D_start_end": [summary["D_start_end"][0], float(combined_traces["D_mean"][-1])],
            "H_start_end": [summary["H_start_end"][0], float(combined_traces["H_mean"][-1])],
            "u_start_end": [summary["u_start_end"][0], float(combined_traces["u_mean"][-1])],
            "continuation_external_input_sha256": None,
            "continuation_external_input_hash_scope": "per_chunk_only",
            "continuation_external_input_segment_manifest_sha256": hashlib.sha256(
                json.dumps(input_segments, sort_keys=True).encode()
            ).hexdigest(),
            "continuation_external_input_segments": input_segments,
            "spike_sha256": full.sha256, "final_state_hash": state_hash(state),
            "clip_frac_max": float(np.nanmax(combined_traces["clip_frac"])),
            "clip_frac_max_observed": float(np.nanmax(combined_traces["clip_frac"])),
            "clip_trace_complete": not bool(missing_trace_intervals),
            "wall_s": time.time() - started,
            "claim_boundary": "extension tests containment/offset; recovery requires post-offset Z and returning-IED gates",
            "execution_manifest": str(execution_manifest) if execution_manifest else None,
            "execution_manifest_sha256": execution_manifest_sha256_start,
            "classifier_snapshot_replay_audit": str(CLASSIFIER_AUDIT),
            "classifier_snapshot_replay_audit_sha256": _sha(CLASSIFIER_AUDIT),
            "classifier_snapshot_replay_n_bundles": classifier_audit["n_bundles"],
        }
        with AtomicStageBundle(arm) as bundle:
            _write_json(bundle.path("summary.json"), result)
            PREFIX._npz_atomic(
                bundle.path("traces.npz"), rate_dt_ms=np.asarray([U2.TRACE_DT_MS], np.float32),
                rate_E=rate10, af=af.astype(np.float32), af_dt_ms=np.asarray([af_dt], np.float32),
                **{key: np.asarray(value, np.float32) for key, value in combined_traces.items()},
            )
            PREFIX._npz_atomic(
                bundle.path("spikes.npz"), steps=full.steps, cells=full.cells.astype(np.int32),
                n_steps=np.asarray([full.n_steps], np.int64), n_cells=np.asarray([full.n_cells], np.int64),
                sha256=np.asarray([full.sha256]),
            )
            U2.save_loop_state(str(bundle.path("final_state.npz")), state)
            bundle.commit(required=["summary.json", "traces.npz", "spikes.npz", "final_state.npz"])
        _plot_extension(arm, result, rate10, combined_traces)
        _write_json(done, {"status": "DONE", "arm": str(arm), "outcome": result["outcome"]})
        running.unlink(missing_ok=True)
        shutil.rmtree(work, ignore_errors=True)
        return result
    except BaseException as exc:
        _write_json(failed, {
            "status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
            "partial_work_preserved": bool(work.exists()),
        })
        running.unlink(missing_ok=True)
        raise
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path)
    parser.add_argument("--execution-manifest", type=Path)
    parser.add_argument("--target-total-ms", type=float)
    parser.add_argument("--output-tag")
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("candidate continuation requires --confirm-run")
    manifest_path = None
    if args.execution_manifest is not None:
        manifest_path, _, contract, source_summary = _manifest_contract(args.execution_manifest)
        target_total_ms = float(contract["target_total_ms"])
        output_tag = str(contract["output_tag"])
    else:
        if args.source_summary is None:
            parser.error("one of --source-summary or --execution-manifest is required")
        source_summary = args.source_summary
        target_total_ms = args.target_total_ms
        output_tag = args.output_tag
    print(json.dumps(PREFIX.json_sanitize(run(
        source_summary, target_total_ms=target_total_ms, output_tag=output_tag,
        execution_manifest=manifest_path,
    )), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
