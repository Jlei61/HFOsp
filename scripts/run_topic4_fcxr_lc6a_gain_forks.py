#!/usr/bin/env python3
"""Prelock, run, and aggregate the conditional FCXR-LC6A exact-state gain forks."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
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

import run_topic4_fcxr_lc6a_natural_trajectory as NAT  # noqa: E402
from src.topic4_fcxr_lc3 import clone_loop_state, state_hash  # noqa: E402
from src.topic4_fcxr_lc3_perturb import run_fcxr_perturbation  # noqa: E402
from src.topic4_fcxr_lc5 import (  # noqa: E402
    AtomicStageBundle, ExactInputHasher, SparseSpikeBinaryWriter,
    load_sparse_spike_stream,
)
from src.topic4_fcxr_lc6_functional import local_patch_pattern  # noqa: E402
from src.topic4_fcxr_lc6_gain import (  # noqa: E402
    active_area_mm2, binned_global_rate, paired_gain_readout,
)
from src.topic4_fcxr_lc6_trajectory import (  # noqa: E402
    cell_spatial_bins, local_saturation_readout, per_second_cell_rates,
    spatial_rate_maps,
)


OUT = NAT.OUT
PRELOCK = ROOT / "config/topic4_fcxr_lc6a_gain_fork_prelock.json"
FUNCTIONAL_PRELOCK = ROOT / "config/topic4_fcxr_lc6a_functional_probe_prelock.json"
FUNCTIONAL_LOCK = OUT / "functional_probe_lock.json"
PHENOTYPE_MAP = OUT / "phenotype_map.json"
LOCAL_LOCK = OUT / "local_classifier_manifest_addendum.json"
LOCK = OUT / "gain_fork_lock.json"
FORK_ROOT = OUT / "gain_forks"
FINAL = OUT / "gain_forks.json"
DONE = OUT / "DONE_LC6A_GAIN_FORKS.json"
FIGURES = OUT / "figures"
MECHANISM_FILES = (
    Path(__file__).resolve(), PRELOCK, FUNCTIONAL_PRELOCK,
    Path(NAT.__file__).resolve(),
    ROOT / "src/topic4_fcxr_lc3.py",
    ROOT / "src/topic4_fcxr_lc3_perturb.py",
    ROOT / "src/topic4_fcxr_lc3_statefork.py",
    ROOT / "src/topic4_fcxr_lc6_gain.py",
    ROOT / "src/topic4_fcxr_lc6_trajectory.py",
    ROOT / "src/snn_engine/mz_slow_vars.py",
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_hashes():
    return {str(path.relative_to(ROOT)): _sha(path) for path in MECHANISM_FILES}


def _write_json(path, payload):
    NAT._write_json(path, payload)


def _prelock():
    payload = json.loads(PRELOCK.read_text())
    if payload.get("experiment_id") != "fcxr_lc6a_gain_state_forks":
        raise RuntimeError("wrong gain-fork prelock")
    if payload["interpretation"]["gain_threshold_is_a_carrier_gate"] is not False:
        raise RuntimeError("gain must remain independent of boundedness")
    return payload


def _previous_second_checkpoint_load(stream, actual_ms, *, tau_ref_ms):
    hi = int(round(float(actual_ms) / NAT.U2.DT_MS))
    lo = hi - int(round(1000.0 / NAT.U2.DT_MS))
    if lo < 0 or hi > stream.n_steps:
        raise RuntimeError("checkpoint lacks a complete preceding 1 s load window")
    left = int(np.searchsorted(stream.steps, lo, side="left"))
    right = int(np.searchsorted(stream.steps, hi, side="left"))
    cells = stream.cells[left:right]
    rates = np.bincount(cells, minlength=stream.n_cells).astype(float)
    ceiling = 1000.0 / float(tau_ref_ms)
    return {
        "global_rate_hz": float(cells.size / stream.n_cells),
        "near_refractory_fraction": float(np.mean(rates >= .9 * ceiling)),
    }


def lock_selection():
    prelock = _prelock()
    if LOCK.is_file():
        return json.loads(LOCK.read_text())
    for path in (FUNCTIONAL_LOCK, PHENOTYPE_MAP, LOCAL_LOCK):
        if not path.is_file():
            raise RuntimeError(f"gain-fork prerequisite missing: {path}")
    functional = json.loads(FUNCTIONAL_LOCK.read_text())
    phenotype = json.loads(PHENOTYPE_MAP.read_text())
    if phenotype.get("status") != "COMPLETE":
        raise RuntimeError("five-arm phenotype map is not complete")
    candidates = phenotype.get("fork_candidates", [])
    maximum = int(prelock["selection"]["maximum_conditions"])
    if len(candidates) > maximum:
        raise RuntimeError("phenotype map selected too many gain-fork conditions")
    selected, skipped = [], []
    for order, candidate in enumerate(candidates):
        condition = candidate["condition"]
        arm = OUT / f"trajectories/{condition}"
        summary_path = arm / "summary.json"
        spike_path = arm / "spikes.npz"
        summary = json.loads(summary_path.read_text())
        stream = load_sparse_spike_stream(spike_path)
        checkpoint_rows = []
        for name in prelock["selection"]["checkpoint_names"]:
            record = summary.get("pinned_checkpoints", {}).get(name)
            if record is None:
                skipped.append({"condition": condition, "checkpoint": name, "reason": "NOT_SAVED"})
                continue
            checkpoint = arm / record["file"]
            if not checkpoint.is_file():
                raise RuntimeError(f"registered checkpoint is missing: {checkpoint}")
            load = _previous_second_checkpoint_load(
                stream, record["actual_ms"], tau_ref_ms=float(NAT.U2.Params().tau_ref_E),
            )
            reasons = []
            if load["global_rate_hz"] >= float(
                prelock["selection"]["skip_checkpoint_when_previous_1s_global_rate_hz_ge"]
            ):
                reasons.append("GLOBAL_SATURATION")
            if load["near_refractory_fraction"] >= float(
                prelock["selection"]["skip_checkpoint_when_previous_1s_near_refractory_fraction_ge"]
            ):
                reasons.append("LOCAL_REFRACTORY_SATURATION")
            if reasons:
                skipped.append({
                    "condition": condition, "checkpoint": name,
                    "reason": "+".join(reasons), "preceding_1s": load,
                })
                continue
            checkpoint_rows.append({
                "name": name, "path": str(checkpoint), "sha256": _sha(checkpoint),
                "state_hash": record["state_hash"], "actual_ms": record["actual_ms"],
                "preceding_1s": load,
                "duplicate_sham_required": False,
            })
        if checkpoint_rows:
            selected.append({
                "condition": condition, "selection_order": len(selected),
                "phenotype": candidate, "checkpoints": checkpoint_rows,
                "natural_summary": str(summary_path),
                "natural_summary_sha256": _sha(summary_path),
                "natural_spikes_sha256": _sha(spike_path),
            })
    if selected and prelock["determinism"][
        "duplicate_sham_for_first_selected_condition_at_each_eligible_checkpoint"
    ]:
        for checkpoint in selected[0]["checkpoints"]:
            checkpoint["duplicate_sham_required"] = True
    payload = {
        "status": "LOCKED", "stage": "LC6A_GAIN_FORK_SELECTION",
        "prelock": str(PRELOCK), "prelock_sha256": _sha(PRELOCK),
        "phenotype_map": str(PHENOTYPE_MAP), "phenotype_map_sha256": _sha(PHENOTYPE_MAP),
        "functional_lock": str(FUNCTIONAL_LOCK), "functional_lock_sha256": _sha(FUNCTIONAL_LOCK),
        "local_classifier_lock": str(LOCAL_LOCK), "local_classifier_lock_sha256": _sha(LOCAL_LOCK),
        "amplitude": float(functional["selected"]["amplitude"]),
        "patch_center": functional["patch_centers"][prelock["perturbation"]["location"]],
        "selected": selected, "skipped_checkpoints": skipped,
        "selection_used_only_prelocked_phenotype_rule": True,
        "responsiveness_is_independent_of_boundedness": True,
        "source_sha256": _source_hashes(),
    }
    _write_json(LOCK, payload)
    return payload


def _validate_lock():
    lock = json.loads(LOCK.read_text())
    if lock.get("status") != "LOCKED":
        raise RuntimeError("gain-fork selection is not locked")
    for key, path in (
        ("prelock_sha256", PRELOCK), ("phenotype_map_sha256", PHENOTYPE_MAP),
        ("functional_lock_sha256", FUNCTIONAL_LOCK),
        ("local_classifier_lock_sha256", LOCAL_LOCK),
    ):
        if lock[key] != _sha(path):
            raise RuntimeError(f"gain-fork locked artifact drift: {path}")
    if lock["source_sha256"] != _source_hashes():
        raise RuntimeError("gain-fork mechanism source drift")
    return lock


def _selected_condition(lock, condition):
    rows = [row for row in lock["selected"] if row["condition"] == condition]
    if len(rows) != 1:
        raise RuntimeError(f"condition is not uniquely selected for gain fork: {condition}")
    return rows[0]


def _system(condition, manifest_path):
    _path, _manifest, source_summary = NAT._validate_manifest(manifest_path, condition)
    graph, metadata = NAT._load_graph(OUT / f"graphs/{condition}.npz")
    summary_cfg = json.loads(source_summary.read_text())
    S, slow, _cfg = NAT._fresh_system(
        summary_cfg, graph, NAT.graph_sha256(graph), condition,
    )
    template = NAT.U2.PM._seed_template(S, slow)
    return S, template, metadata


def _one_arm(S, start, *, pattern, amplitude, work, tag, prelock):
    timing = prelock["perturbation"]
    readout = prelock["readout"]
    response_ms = float(timing["response_ms"])
    n_steps = int(round(response_ms / NAT.U2.DT_MS))
    p = dataclasses.replace(S["p"], T=response_ms, dt=NAT.U2.DT_MS)
    writer = SparseSpikeBinaryWriter(
        work / f"{tag}.bin", step_origin=0, n_steps=n_steps, n_cells=S["NE"],
    )
    hasher = ExactInputHasher()
    started = time.time()
    try:
        result = run_fcxr_perturbation(
            p, S["net"], start=clone_loop_state(start), n_steps=n_steps,
            current_pattern=pattern, amplitude=float(amplitude),
            pulse_steps=int(round(float(timing["pulse_ms"]) / NAT.U2.DT_MS)),
            capture_final=True, store_spikes=False, v_th_per_neuron=S["vth"],
            input_sink=hasher, spike_sink=writer,
        )
        stream = writer.finalize(work / f"{tag}_spikes.npz")
    finally:
        writer.close()
        (work / f"{tag}.bin").unlink(missing_ok=True)
    rate = binned_global_rate(
        stream.steps, n_steps=stream.n_steps, n_cells=stream.n_cells,
        dt_ms=NAT.U2.DT_MS, bin_ms=readout["global_rate_bin_ms"],
    )
    bins, occupancy = cell_spatial_bins(
        S["posE"], sheet_size_mm=S["L"],
        n_bins_axis=int(readout["local_map_bins_per_axis"]),
    )
    maps = spatial_rate_maps(
        stream.steps, stream.cells, bins, occupancy, n_steps=stream.n_steps,
        dt_ms=NAT.U2.DT_MS, window_ms=readout["spatial_area_bin_ms"],
    )
    local_lock = json.loads(LOCAL_LOCK.read_text())
    area = active_area_mm2(
        maps, occupancy, rate_threshold_hz=local_lock["thresholds"]["rate_threshold_hz"],
        sheet_size_mm=S["L"],
    )
    cell_rates = per_second_cell_rates(
        stream.steps, stream.cells, n_steps=stream.n_steps, n_cells=stream.n_cells,
        dt_ms=NAT.U2.DT_MS,
    )
    saturation = local_saturation_readout(
        cell_rates, refractory_ceiling_hz=1000.0 / float(S["p"].tau_ref_E),
    )
    global_1s = cell_rates.mean(axis=1)
    baseline_hi = float(NAT.U2._baseline()["band"]["roll_hi"])
    return {
        "stream": stream, "rate_10ms": rate, "area_100ms": area,
        "summary": {
            "tag": tag, "amplitude": float(amplitude),
            "external_input_sha256": hasher.sha256,
            "spike_sha256": stream.sha256,
            "final_state_hash": state_hash(result["checkpoint"]),
            "pulse_accounting": result["pulse_accounting"],
            "global_rate_1s_hz": global_1s.tolist(),
            "registered_global_saturation": bool(np.any(global_1s >= NAT.U2.SAT_CEILING_HZ)),
            "local_saturation": saturation,
            "offset_like_low_rate_last_1s": bool(global_1s[-1] <= baseline_hi),
            "diverged": False,
            "wall_s": time.time() - started,
        },
    }


def _duplicate_record(first, second):
    record = {
        "external_input_exact": first["summary"]["external_input_sha256"] == second["summary"]["external_input_sha256"],
        "spike_exact": first["summary"]["spike_sha256"] == second["summary"]["spike_sha256"],
        "final_state_exact": first["summary"]["final_state_hash"] == second["summary"]["final_state_hash"],
    }
    record["pass"] = bool(all(record.values()))
    return record


def run_condition(condition, manifest_path):
    prelock = _prelock()
    lock = _validate_lock()
    selected = _selected_condition(lock, condition)
    if _sha(selected["natural_summary"]) != selected["natural_summary_sha256"]:
        raise RuntimeError("selected natural-trajectory summary drift")
    spike_path = OUT / f"trajectories/{condition}/spikes.npz"
    if _sha(spike_path) != selected["natural_spikes_sha256"]:
        raise RuntimeError("selected natural-trajectory spikes drift")
    arm = FORK_ROOT / condition
    if arm.is_dir():
        return json.loads((arm / "summary.json").read_text())
    work = FORK_ROOT / f".{condition}.work"
    if work.exists():
        raise RuntimeError(f"stale gain-fork work directory requires inspection: {work}")
    work.mkdir(parents=True)
    S, template, metadata = _system(condition, manifest_path)
    functional_prelock = json.loads(FUNCTIONAL_PRELOCK.read_text())
    pattern = local_patch_pattern(
        S["posE"], lock["patch_center"],
        radius_mm=functional_prelock["geometry"]["patch_radius_mm"],
    )
    checkpoint_rows, arrays = [], {}
    try:
        for checkpoint in selected["checkpoints"]:
            if _sha(checkpoint["path"]) != checkpoint["sha256"]:
                raise RuntimeError("selected exact checkpoint drift")
            state = NAT.U2.load_into(checkpoint["path"], template)
            if state_hash(state) != checkpoint["state_hash"]:
                raise RuntimeError("loaded gain checkpoint state hash mismatch")
            name = checkpoint["name"]
            sham = _one_arm(
                S, state, pattern=pattern, amplitude=0.0, work=work,
                tag=f"{name}_sham", prelock=prelock,
            )
            probe = _one_arm(
                S, state, pattern=pattern, amplitude=lock["amplitude"], work=work,
                tag=f"{name}_probe", prelock=prelock,
            )
            if sham["summary"]["external_input_sha256"] != probe["summary"]["external_input_sha256"]:
                raise RuntimeError("gain sham/probe future external input differs")
            duplicate = None
            if checkpoint["duplicate_sham_required"]:
                repeated = _one_arm(
                    S, state, pattern=pattern, amplitude=0.0, work=work,
                    tag=f"{name}_sham_duplicate", prelock=prelock,
                )
                duplicate = _duplicate_record(sham, repeated)
                if not duplicate["pass"]:
                    raise RuntimeError("gain-fork exact duplicate determinism failed")
            paired = paired_gain_readout(
                sham["rate_10ms"], probe["rate_10ms"],
                sham["area_100ms"], probe["area_100ms"],
                pulse_l2_current=probe["summary"]["pulse_accounting"]["l2_current"],
                pulse_ms=prelock["perturbation"]["pulse_ms"],
                susceptibility_window_ms=prelock["readout"]["susceptibility_window_ms"],
                rate_bin_ms=prelock["readout"]["global_rate_bin_ms"],
                area_bin_ms=prelock["readout"]["spatial_area_bin_ms"],
                relaxation_fraction=prelock["readout"]["relaxation_fraction_of_peak"],
                relaxation_hold_ms=prelock["readout"]["relaxation_hold_ms"],
            )
            arrays[f"{name}__rate_sham_hz"] = sham["rate_10ms"].astype(np.float32)
            arrays[f"{name}__rate_probe_hz"] = probe["rate_10ms"].astype(np.float32)
            arrays[f"{name}__area_sham_mm2"] = sham["area_100ms"].astype(np.float32)
            arrays[f"{name}__area_probe_mm2"] = probe["area_100ms"].astype(np.float32)
            arrays[f"{name}__delta_rate_hz"] = paired.pop("delta_rate_hz").astype(np.float32)
            arrays[f"{name}__delta_area_mm2"] = paired.pop("delta_area_mm2").astype(np.float32)
            checkpoint_rows.append({
                "checkpoint": name, "start_state_hash": checkpoint["state_hash"],
                "actual_ms": checkpoint["actual_ms"], "preceding_1s": checkpoint["preceding_1s"],
                "sham": sham["summary"], "probe": probe["summary"],
                "duplicate_determinism": duplicate, "paired": paired,
                "response_detected_nonzero": bool(
                    paired["global_rate_l1_response_hz_s"] > 0
                    or paired["active_area_l1_deviation_mm2_s"] > 0
                ),
            })
        summary = {
            "status": "COMPLETE", "condition": condition,
            "graph_sha256": metadata["graph_sha256"],
            "graph_construction_q": metadata["construction_q"],
            "selection": selected["phenotype"], "checkpoints": checkpoint_rows,
            "responsiveness_is_independent_of_boundedness": True,
            "termination_tested": False, "lifecycle_tested": False,
            "lock_sha256": _sha(LOCK), "source_sha256": _source_hashes(),
        }
        with AtomicStageBundle(arm) as bundle:
            _write_json(bundle.path("summary.json"), summary)
            NAT._npz_atomic(bundle.path("responses.npz"), **arrays)
            bundle.commit(required=["summary.json", "responses.npz"])
        return summary
    finally:
        if work.exists():
            import shutil
            shutil.rmtree(work, ignore_errors=True)


def finalize():
    lock = _validate_lock()
    rows = []
    for selection in lock["selected"]:
        path = FORK_ROOT / selection["condition"] / "summary.json"
        if not path.is_file():
            raise RuntimeError(f"selected gain fork incomplete: {path}")
        rows.append(json.loads(path.read_text()))
    duplicate_rows = [
        row["duplicate_determinism"] for condition in rows
        for row in condition["checkpoints"] if row["duplicate_determinism"] is not None
    ]
    expected = int(_prelock()["determinism"]["expected_duplicate_checkpoints"])
    if lock["selected"] and len(duplicate_rows) != min(expected, len(lock["selected"][0]["checkpoints"])):
        raise RuntimeError("gain-fork duplicate checkpoint coverage differs from prelock")
    if any(not row["pass"] for row in duplicate_rows):
        raise RuntimeError("one or more gain-fork duplicate checkpoints failed")
    payload = {
        "status": "COMPLETE", "rows": rows,
        "n_selected_conditions": len(rows), "n_duplicate_checkpoints": len(duplicate_rows),
        "all_duplicates_exact": bool(all(row["pass"] for row in duplicate_rows)),
        "responsiveness_is_independent_of_boundedness": True,
        "gain_threshold_used_as_carrier_gate": False,
        "termination_tested": False, "lifecycle_tested": False,
        "claim_boundary": "Paired weak-patch response measures high-state susceptibility only.",
    }
    _write_json(FINAL, payload)
    if rows:
        labels, susceptibility, relaxation, rate_rms, area_l1 = [], [], [], [], []
        for condition in rows:
            for checkpoint in condition["checkpoints"]:
                labels.append(f"{condition['condition']}\n{checkpoint['checkpoint'].replace('onset_plus_', '+')}")
                paired = checkpoint["paired"]
                susceptibility.append(paired["susceptibility_hz_s_per_l2_current_s"])
                relaxation.append(paired["relaxation"]["relaxation_ms_after_pulse"])
                rate_rms.append(paired["global_rate_rms_deviation_hz"])
                area_l1.append(paired["active_area_l1_deviation_mm2_s"])
        x = np.arange(len(labels))
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
        axes[0, 0].bar(x, susceptibility, color="#3B7EA1")
        axes[0, 0].set_title("a  500 ms susceptibility"); axes[0, 0].set_ylabel("Hz·s / input L2·s")
        rel = [np.nan if value is None else value for value in relaxation]
        axes[0, 1].bar(x, rel, color="#8C6BB1")
        axes[0, 1].set_title("b  Relaxation after pulse"); axes[0, 1].set_ylabel("ms")
        axes[1, 0].bar(x, rate_rms, color="#D95F0E")
        axes[1, 0].set_title("c  Macroscopic rate deviation"); axes[1, 0].set_ylabel("RMS Hz")
        axes[1, 1].bar(x, area_l1, color="#2CA25F")
        axes[1, 1].set_title("d  Active-area deviation"); axes[1, 1].set_ylabel("mm²·s")
        for ax in axes.ravel():
            ax.set_xticks(x, labels, fontsize=8)
        fig.suptitle("FCXR-LC6A paired exact-state gain forks")
        FIGURES.mkdir(parents=True, exist_ok=True)
        png = FIGURES / "lc6a_gain_forks.png"
        pdf = FIGURES / "lc6a_gain_forks.pdf"
        fig.savefig(png, dpi=220, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        readme = FIGURES / "README.md"
        existing = readme.read_text() if readme.is_file() else ""
        section = f"""### {png.name}

这张图比较 phenotype map 预注册选出的最多两个高态，在 onset 后两个 exact checkpoint 对同一 50 ms 弱局部输入的配对响应。四格分别给出 500 ms susceptibility、回落时间、全局 rate 偏离和活动面积偏离；这些是独立响应性读数，不覆盖 boundedness 标签。

**关注点**：静止但仍能响应的高态可以保留为 carrier；本图不检验 termination 或完整 lifecycle。

### {pdf.name}

与 PNG 相同的矢量版本。

**关注点**：所有 sham/probe 共享 exact future input，首个候选的 checkpoint 另做 exact duplicate。
"""
        tmp = readme.with_name(readme.name + f".{os.getpid()}.tmp")
        tmp.write_text((existing.rstrip() + "\n\n" + section).lstrip())
        os.replace(tmp, readme)
    _write_json(DONE, {
        "status": "DONE", "gain_forks": str(FINAL), "gain_forks_sha256": _sha(FINAL),
        "n_selected_conditions": len(rows),
    })
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("lock", "run", "finalize"))
    parser.add_argument("--condition", choices=NAT.GRAPH_IDS)
    parser.add_argument(
        "--execution-manifest", type=Path,
        default=ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC6A gain forks require --confirm-run")
    if args.stage == "run" and args.condition is None:
        parser.error("run requires --condition")
    OUT.mkdir(parents=True, exist_ok=True)
    lock_name = ".gain_fork_global.lock" if args.stage != "run" else f".gain_fork_{args.condition}.lock"
    with (OUT / lock_name).open("w") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("requested LC6A gain-fork stage is already running") from exc
        label = args.stage.upper() if args.stage != "run" else str(args.condition)
        running = OUT / f"RUNNING_LC6A_GAIN_{label}.json"
        failed = OUT / f"FAILED_LC6A_GAIN_{label}.json"
        done = OUT / f"DONE_LC6A_GAIN_{label}.json"
        _write_json(running, {"status": "RUNNING", "pid": os.getpid(), "stage": args.stage})
        try:
            if args.stage == "lock":
                result = lock_selection()
            elif args.stage == "run":
                result = run_condition(args.condition, args.execution_manifest)
            else:
                result = finalize()
            _write_json(done, {"status": "DONE", "stage": args.stage, "condition": args.condition})
            failed.unlink(missing_ok=True)
            print(json.dumps(NAT._jsonable(result), indent=2, sort_keys=True))
        except BaseException as exc:
            _write_json(failed, {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"})
            raise
        finally:
            running.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
