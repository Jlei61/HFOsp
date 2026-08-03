#!/usr/bin/env python3
"""FCXR-LC3 E5 exact-landmark direct spatial response and projected SVD."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import dataclasses
import fcntl
import gc
import hashlib
import json
import resource
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "src", "snn_engine"))

import run_m4_phaseplane as PP  # noqa: E402
import run_topic4_fcxr_lc3 as E01  # noqa: E402
import run_topic4_fcxr_lc3_geometry as GEO  # noqa: E402
import run_topic4_fcxr_lc3_recon as RECON  # noqa: E402
from src.topic4_fcxr_lc3 import clone_loop_state  # noqa: E402
from src.topic4_fcxr_lc3_geometry import (  # noqa: E402
    H1_POINT_ID,
    load_prepared_checkpoint,
)
from src.topic4_fcxr_lc3_perturb import run_fcxr_perturbation  # noqa: E402
from src.topic4_fcxr_lc3_spatial import (  # noqa: E402
    RATE_WINDOW_MS,
    RESPONSE_TIMES_MS,
    SPATIAL_CONTROL_SEED,
    build_equal_local_masks,
    build_signed_basis,
    first_passage_ms,
    global_control_patterns,
    positive_patterns,
    projected_response_matrix,
    rate_fields,
    svd_summary,
)


OUT = E01.OUT
LOCK = os.path.join(OUT, "spatial_probe_lock.json")
MANIFEST = os.path.join(OUT, "spatial_probe_manifest.json")
PATTERNS = os.path.join(OUT, "spatial_probe_patterns.npz")
DIRECT = os.path.join(OUT, "spatial_direct_response.json")
OPERATOR = os.path.join(OUT, "projected_response_operator.json")
CELL_DIR = os.path.join(OUT, "spatial_probe_cells")
DT = E01.DT
PULSE_MS = 10.0
RESPONSE_MS = max(RESPONSE_TIMES_MS)
AMPLITUDE_FRACTIONS = (0.05, 0.10)
N_GRID = 16
SPATIAL_SOURCES = (
    "src/topic4_fcxr_lc3.py",
    "src/topic4_fcxr_lc3_geometry.py",
    "src/topic4_fcxr_lc3_perturb.py",
    "src/topic4_fcxr_lc3_spatial.py",
    "src/snn_engine/mz_slow_vars.py",
    "scripts/run_topic4_fcxr_lc3_spatial.py",
    "scripts/run_topic4_fcxr_lc3_spatial_autopilot.sh",
    "docs/superpowers/specs/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability-design.md",
    "docs/superpowers/plans/2026-08-03-topic4-fcxr-lc3-dx-spatial-instability.md",
)

_SPATIAL_SWAP_BASE_MIB = None


def _now():
    return datetime.now(timezone.utc).isoformat()


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _load(path):
    with open(path) as f:
        return json.load(f)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def _write_npz(path, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _meminfo():
    with open("/proc/meminfo") as f:
        d = {line.split(":", 1)[0]: float(line.split()[1]) for line in f}
    return dict(
        mem_available_gib=d["MemAvailable"] / 1024.0 / 1024.0,
        swap_used_mib=(d["SwapTotal"] - d["SwapFree"]) / 1024.0,
        self_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0 / 1024.0,
    )


def _wait_before_new_arm():
    """Honor the +256 MiB pause contract before starting another 40k arm."""

    if _SPATIAL_SWAP_BASE_MIB is None:
        raise RuntimeError("spatial swap baseline was not initialized")
    while True:
        mem = _meminfo()
        if (mem["swap_used_mib"] - _SPATIAL_SWAP_BASE_MIB < 256.0
                and mem["mem_available_gib"] >= 96.0):
            return
        time.sleep(30.0)


@contextmanager
def _stage_lock(name):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f".{name}.lock"), "a+") as fd:
        try:
            fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"spatial stage already running: {name}") from exc
        yield


def _checkpoint_record(path, *, source, state_id, scientific_role):
    if not os.path.isfile(path):
        raise RuntimeError(f"missing checkpoint {path}")
    payload = load_prepared_checkpoint(path)
    return dict(
        state_id=state_id, path=path, file_sha256=_sha(path), source=source,
        scientific_role=scientific_role,
        dynamic_state_hash=payload["dynamic_state_hash"],
        configured_state_hash=payload["configured_state_hash"],
    )


def _select_input_states():
    """Prefer real onset landmarks; otherwise use accepted frozen-map states."""

    recon_path = os.path.join(RECON.OUT, "recon_noise401.json")
    if not os.path.isfile(recon_path):
        raise RuntimeError("primary reconnaissance output is required")
    recon = _load(recon_path)
    exact = recon.get("exact_landmark_states", {})
    preferred = ("pre_onset", "onset", "early_high", "late_high_pre_offset")
    if recon.get("lifecycle", {}).get("bout") is not None and all(name in exact for name in preferred):
        rows = []
        for name in preferred:
            rec = exact[name]
            rows.append(_checkpoint_record(
                rec["path"], source="dynamic_reconnaissance_noise401",
                state_id=f"dynamic_{name}", scientific_role=name,
            ))
        return rows, "DYNAMIC_EXACT_LANDMARKS"

    rows = []
    for kind in ("low", "high"):
        pkl, js = GEO._prep_paths(H1_POINT_ID, kind)
        if not (os.path.isfile(pkl) and os.path.isfile(js)):
            continue
        meta = _load(js)
        if meta.get("status") != "ACCEPTED_CANONICAL_STATE":
            continue
        rows.append(_checkpoint_record(
            pkl, source="frozen_geometry_substitution",
            state_id=f"frozen_H1_{kind}", scientific_role=f"frozen_{kind}_branch",
        ))
    if rows:
        return rows, "FROZEN_MAP_SUBSTITUTION_NO_DYNAMIC_ONSET"

    # Engineering-only fallback: it preserves an exact real state but cannot
    # support the canonical spatial-mechanism verdict promised for a map landmark.
    if "no_onset_final" in exact:
        rec = exact["no_onset_final"]
        return [_checkpoint_record(
            rec["path"], source="dynamic_no_onset_fallback",
            state_id="dynamic_no_onset_final", scientific_role="noncanonical_fallback",
        )], "NONCANONICAL_NO_ONSET_FALLBACK"
    raise RuntimeError("no exact dynamic or accepted frozen state is available for E5")


def _healthy_reference_and_amplitudes(S):
    pkl, js = GEO._prep_paths(H1_POINT_ID, "low")
    meta = _load(js)
    if meta.get("status") != "ACCEPTED_CANONICAL_STATE":
        raise RuntimeError("accepted H1 healthy low state is required to derive I_ref")
    payload = load_prepared_checkpoint(pkl, expected_file_sha256=meta["checkpoint"]["file_sha256"])
    state = payload["state"]
    slow = clone_loop_state(state).slow
    drive, _g_rel, _g_rev = slow.membrane_terms(
        state.I_E, state.I_I, S["net"]["labels"], I_E_rec=state.I_E_rec)
    i_ref = float(np.quantile(np.abs(np.asarray(drive[:S["NE"]], float)), 0.95))
    if not np.isfinite(i_ref) or i_ref <= 0.0:
        raise RuntimeError("healthy additive-drive I_ref is invalid")
    return dict(
        definition="Q95_E_abs_additive_drive_at_accepted_H1_healthy_low_exact_state",
        value=i_ref, source_path=pkl, source_sha256=_sha(pkl),
        fractions=list(AMPLITUDE_FRACTIONS),
        amplitudes=[float(f * i_ref) for f in AMPLITUDE_FRACTIONS],
    )


def cmd_lock(_args):
    if not os.path.isfile(os.path.join(RECON.OUT, "aggregate.json")):
        raise SystemExit("completed reconnaissance aggregate is required")
    initial = _load(os.path.join(OUT, "execution_lock.json"))
    states, state_source = _select_input_states()
    S = PP.build_substrate(1)
    masks = build_equal_local_masks(
        S["posE"], S["src_xy"], S["snk_xy"], S["axis_unit"],
        core_r=PP.CORE_R, random_seed=SPATIAL_CONTROL_SEED)
    positive = positive_patterns(masks)
    globals_ = global_control_patterns(S["NE"], int(next(iter(masks.values())).sum()))
    basis = build_signed_basis(masks, random_seed=SPATIAL_CONTROL_SEED)
    arrays = {}
    arrays.update({f"mask_{name}": value.astype(np.uint8) for name, value in masks.items()})
    arrays.update({f"positive_{name}": value for name, value in {**positive, **globals_}.items()})
    arrays.update({f"basis_{name}": value for name, value in basis.items()})
    _write_npz(PATTERNS, **arrays)
    i_ref = _healthy_reference_and_amplitudes(S)
    payload = dict(
        status="LOCKED", schema="fcxr-lc3-spatial-lock-1.0",
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip(),
        state_source=state_source, states=states, healthy_I_ref=i_ref,
        pulse_ms=PULSE_MS, response_times_ms=list(RESPONSE_TIMES_MS),
        rate_window_ms=RATE_WINDOW_MS, spatial_control_seed=SPATIAL_CONTROL_SEED,
        n_local=int(next(iter(masks.values())).sum()), n_basis=len(basis),
        pattern_path=PATTERNS, pattern_sha256=_sha(PATTERNS),
        sources={rel: _sha(os.path.join(ROOT, rel)) for rel in SPATIAL_SOURCES},
        engine_hashes=initial["engine_hashes"], resource_at_lock=_meminfo(),
        claim_boundary=("direct response; dynamic landmarks preferred; frozen substitution explicitly "
                        "labelled; no lifecycle/eigenvalue claim"),
        locked_at=_now(),
    )
    _write_json(LOCK, payload)
    print(json.dumps(dict(status="LOCKED", n_states=len(states), state_source=state_source,
                          amplitudes=i_ref["amplitudes"], n_local=payload["n_local"]), indent=2))


def _assert_lock():
    if not os.path.isfile(LOCK):
        raise SystemExit("missing spatial probe lock")
    lock = _load(LOCK)
    if lock.get("status") != "LOCKED" or _sha(PATTERNS) != lock["pattern_sha256"]:
        raise SystemExit("invalid spatial probe lock or pattern drift")
    for state in lock["states"]:
        if _sha(state["path"]) != state["file_sha256"]:
            raise SystemExit(f"spatial checkpoint drift: {state['state_id']}")
    for rel, expected in lock["sources"].items():
        if _sha(os.path.join(ROOT, rel)) != expected:
            raise SystemExit(f"spatial source drift: {rel}")
    for rel, expected in lock["engine_hashes"].items():
        if _sha(os.path.join(ROOT, rel)) != expected:
            raise SystemExit(f"engine drift: {rel}")
    return lock


def cmd_manifest(_args):
    lock = _assert_lock()
    positive_names = ("core_A", "core_B", "axial", "transverse", "shuffled_axial",
                      "global_charge_matched", "global_rms_matched")
    basis_names = ("global_mode", "core_A", "core_B", "axial_symmetric",
                   "axial_antisymmetric", "transverse", "surround",
                   "random_731", "random_732")
    rows = []

    def add(state_id, kind, pattern, amplitude, **extra):
        tag = _arm_tag(kind, pattern, amplitude)
        stem = os.path.join(CELL_DIR, f"{state_id}__{tag}")
        rows.append(dict(
            index=len(rows), row_id=f"{state_id}__{tag}", state_id=state_id,
            kind=kind, pattern=pattern, amplitude=float(amplitude),
            output_json=stem + ".json", output_npz=stem + ".npz",
            done_path=stem + ".DONE.json", **extra,
        ))

    for state in lock["states"]:
        add(state["state_id"], "sham", "global_mode", 0.0)
        for amplitude in lock["healthy_I_ref"]["amplitudes"]:
            for pattern in positive_names:
                add(state["state_id"], "positive", pattern, amplitude)
            epsilon = float(amplitude * np.sqrt(lock["n_local"]))
            for pattern in basis_names:
                add(state["state_id"], "signed_minus", pattern, -epsilon,
                    amplitude_reference=amplitude)
                add(state["state_id"], "signed_plus", pattern, epsilon,
                    amplitude_reference=amplitude)
    payload = dict(
        status="LOCKED", schema="fcxr-lc3-spatial-manifest-1.0", rows=rows,
        n_rows=len(rows), n_states=len(lock["states"]),
        lock_sha256=_sha(LOCK), created=_now(),
    )
    _write_json(MANIFEST, payload)
    print(json.dumps(dict(status="LOCKED", n_rows=len(rows),
                          n_states=len(lock["states"])), indent=2))


def _assert_manifest():
    lock = _assert_lock()
    manifest = _load(MANIFEST)
    expected = len(lock["states"]) * (1 + 2 * (7 + 2 * 9))
    if manifest.get("status") != "LOCKED" or len(manifest.get("rows", [])) != expected:
        raise SystemExit("spatial manifest row matrix is incomplete")
    rows = manifest["rows"]
    if ([row.get("index") for row in rows] != list(range(expected))
            or len({row.get("row_id") for row in rows}) != expected
            or not all(row.get("done_path") for row in rows)):
        raise SystemExit("spatial manifest row identities/DONE paths are invalid")
    if manifest.get("lock_sha256") != _sha(LOCK):
        raise SystemExit("spatial manifest lock drift")
    return lock, manifest


def _load_patterns():
    with np.load(PATTERNS) as z:
        masks = {k[5:]: z[k].astype(bool) for k in z.files if k.startswith("mask_")}
        positive = {k[9:]: z[k].astype(float) for k in z.files if k.startswith("positive_")}
        basis = {k[6:]: z[k].astype(float) for k in z.files if k.startswith("basis_")}
    return masks, positive, basis


def _run_arm(S, state, pattern, amplitude):
    p = dataclasses.replace(S["p"], T=RESPONSE_MS, dt=DT)
    out = run_fcxr_perturbation(
        p, S["net"], start=clone_loop_state(state),
        n_steps=int(round(RESPONSE_MS / DT)), current_pattern=pattern,
        amplitude=float(amplitude), pulse_steps=int(round(PULSE_MS / DT)),
        capture_final=True, store_spikes=True, v_th_per_neuron=S["vth"],
    )
    if not (np.all(np.isfinite(out["rate_E"])) and np.all(np.isfinite(out["rate_I"]))):
        raise RuntimeError("spatial response produced non-finite population rate")
    slow = out["checkpoint"].slow
    clip = max(slow.trace_conductance_clip_frac[-out["n_steps"]:] or [0.0])
    if clip > 0.0:
        raise RuntimeError(f"spatial response hit conductance hard clip: {clip}")
    return out


def _arm_tag(kind, pattern_name, amplitude):
    amp = f"{float(amplitude):+.10g}".replace("+", "p").replace("-", "m").replace(".", "p")
    return f"{kind}__{pattern_name}__{amp}"


def _run_or_load_reduced(S, state, *, state_id, kind, pattern_name,
                         pattern, amplitude):
    """Run one manifest arm atomically, or resume its hash-checked reduction."""

    stem = os.path.join(CELL_DIR, f"{state_id}__{_arm_tag(kind, pattern_name, amplitude)}")
    json_path = stem + ".json"
    npz_path = stem + ".npz"
    done_path = stem + ".DONE.json"
    if all(os.path.isfile(path) for path in (json_path, npz_path, done_path)):
        record = _load(json_path)
        done = _load(done_path)
        if (record.get("status") == "COMPLETE"
                and done.get("output_sha256") == _sha(json_path)
                and record.get("array_sha256") == _sha(npz_path)):
            with np.load(npz_path) as z:
                return dict(
                    fields={float(t): z["rate_fields"][i].astype(float)
                            for i, t in enumerate(RESPONSE_TIMES_MS)},
                    active=z["active"].astype(bool),
                    first_passage=z["first_passage_ms"].astype(float),
                    accounting=record["accounting"],
                    max_population_rate_hz=record["max_population_rate_hz"],
                    refractory_ceiling_fraction=record["refractory_ceiling_fraction"],
                    artifact=dict(json_path=json_path, npz_path=npz_path,
                                  resumed=True),
                )

    _wait_before_new_arm()
    out = _run_arm(S, state, pattern, amplitude)
    fields = rate_fields(out["E_spk_bool"], dt_ms=DT)
    active = out["E_spk_bool"].any(axis=0)
    fp = first_passage_ms(out["E_spk_bool"], dt_ms=DT)
    per_cell_hz = out["E_spk_bool"].sum(axis=0) / (RESPONSE_MS * 1e-3)
    ceiling_frac = float(np.mean(per_cell_hz >= 0.8 * 1000.0 / S["p"].tau_ref_E))
    _write_npz(
        npz_path,
        response_times_ms=np.asarray(RESPONSE_TIMES_MS, np.float32),
        rate_fields=np.stack([fields[t] for t in RESPONSE_TIMES_MS]).astype(np.float64),
        active=active.astype(np.uint8), first_passage_ms=fp.astype(np.float64),
    )
    record = dict(
        status="COMPLETE", state_id=state_id, kind=kind,
        pattern=pattern_name, amplitude=float(amplitude),
        accounting=out["pulse_accounting"],
        max_population_rate_hz=float(np.max(out["rate_E"])),
        refractory_ceiling_fraction=ceiling_frac,
        array_path=npz_path, array_sha256=_sha(npz_path),
        source_lock_sha256=_sha(LOCK), finished=_now(),
    )
    _write_json(json_path, record)
    _write_json(done_path, dict(status="DONE", output_sha256=_sha(json_path), finished=_now()))
    del out
    gc.collect()
    return dict(
        fields=fields, active=active, first_passage=fp,
        accounting=record["accounting"],
        max_population_rate_hz=record["max_population_rate_hz"],
        refractory_ceiling_fraction=record["refractory_ceiling_fraction"],
        artifact=dict(json_path=json_path, npz_path=npz_path, resumed=False),
    )


def _region_rate_deltas(arm_fields, sham_fields, regions):
    return {
        str(t): {name: float(np.mean(arm_fields[t][mask] - sham_fields[t][mask]))
                 for name, mask in regions.items()}
        for t in RESPONSE_TIMES_MS
    }


def _positive_metrics(S, arm, sham, pattern, amplitude, regions):
    arm_fields = arm["fields"]
    sham_fields = sham["fields"]
    arm_active = arm["active"]
    sham_active = sham["active"]
    newly = arm_active & ~sham_active
    grid_xy = np.floor(np.asarray(S["posE"]) / float(S["L"]) * N_GRID).astype(int)
    np.clip(grid_xy, 0, N_GRID - 1, out=grid_xy)
    new_voxels = int(np.unique(grid_xy[newly], axis=0).shape[0]) if newly.any() else 0
    fp = arm["first_passage"]
    fp_regions = {}
    for name, mask in regions.items():
        values = fp[mask & np.isfinite(fp)]
        fp_regions[name] = float(np.median(values)) if values.size else None
    current = float(amplitude) * np.asarray(pattern, float)
    gains = {
        str(t): float(np.linalg.norm(arm_fields[t] - sham_fields[t])
                      / max(np.linalg.norm(current), 1e-12))
        for t in RESPONSE_TIMES_MS
    }
    return dict(
        accounting=arm["accounting"], finite_time_gain=gains,
        newly_recruited_cells=int(newly.sum()), newly_recruited_voxels=new_voxels,
        newly_recruited_area_mm2=float(new_voxels * (S["L"] / N_GRID) ** 2),
        arm_active_cells=int(arm_active.sum()), sham_active_cells=int(sham_active.sum()),
        first_passage_region_median_ms=fp_regions,
        regional_rate_delta_hz=_region_rate_deltas(arm_fields, sham_fields, regions),
        max_population_rate_hz=arm["max_population_rate_hz"],
        refractory_ceiling_fraction=arm["refractory_ceiling_fraction"],
        arm_artifact=arm["artifact"],
    )


def _state_spatial_label(rows):
    by = {(r["pattern"], r["amplitude"]): r for r in rows}
    amps = sorted({r["amplitude"] for r in rows})
    axial_ok = []
    transverse_ok = []
    global_ok = []
    polarity_ok = []
    for amp in amps:
        gain = lambda name: by[(name, amp)]["metrics"]["finite_time_gain"]["300.0"]
        axial_ok.append(gain("axial") > 1.2 * max(gain("transverse"), gain("shuffled_axial")))
        transverse_ok.append(gain("transverse") > 1.2 * gain("axial"))
        global_ok.append(gain("global_rms_matched") > 1.2 * gain("axial"))
        a = by[("core_A", amp)]["metrics"]["regional_rate_delta_hz"]["150.0"]
        b = by[("core_B", amp)]["metrics"]["regional_rate_delta_hz"]["150.0"]
        polarity_ok.append(a["core_A"] > a["core_B"] and b["core_B"] > b["core_A"])
    if all(axial_ok) and all(polarity_ok):
        return "AXIAL_LOCAL_DIRECT_RESPONSE"
    if all(transverse_ok):
        return "TRANSVERSE_DIRECT_RESPONSE"
    if all(global_ok):
        return "GLOBAL_DIRECT_RESPONSE"
    return "SPATIAL_DIRECT_RESPONSE_UNRESOLVED"


def _run_state(S, state_record, lock, masks, positive, basis):
    payload = load_prepared_checkpoint(
        state_record["path"], expected_file_sha256=state_record["file_sha256"])
    base = payload["state"]
    regions = GEO._region_masks(S)
    sham = _run_or_load_reduced(
        S, base, state_id=state_record["state_id"], kind="sham",
        pattern_name="global_mode", pattern=np.ones(S["NE"]), amplitude=0.0)
    sham_fields = sham["fields"]
    positive_rows = []
    signed_by_amp = []
    arrays = {}

    for amplitude in lock["healthy_I_ref"]["amplitudes"]:
        for name, pattern in positive.items():
            arm = _run_or_load_reduced(
                S, base, state_id=state_record["state_id"], kind="positive",
                pattern_name=name, pattern=pattern, amplitude=amplitude)
            metrics = _positive_metrics(S, arm, sham, pattern, amplitude, regions)
            positive_rows.append(dict(pattern=name, amplitude=float(amplitude), metrics=metrics))
            del arm
            gc.collect()

        epsilon = float(amplitude * np.sqrt(lock["n_local"]))
        plus_fields, minus_fields, nonlinear, signed_ceiling = {}, {}, {}, {}
        for name, vector in basis.items():
            plus = _run_or_load_reduced(
                S, base, state_id=state_record["state_id"], kind="signed_plus",
                pattern_name=name, pattern=vector, amplitude=epsilon)
            minus = _run_or_load_reduced(
                S, base, state_id=state_record["state_id"], kind="signed_minus",
                pattern_name=name, pattern=vector, amplitude=-epsilon)
            plus_fields[name] = plus["fields"]
            minus_fields[name] = minus["fields"]
            signed_ceiling[name] = max(
                plus["refractory_ceiling_fraction"],
                minus["refractory_ceiling_fraction"])
            nonlinear[name] = {}
            for t in RESPONSE_TIMES_MS:
                midpoint = 0.5 * (plus_fields[name][t] + minus_fields[name][t])
                nonlinear[name][str(t)] = float(
                    np.linalg.norm(midpoint - sham_fields[t])
                    / max(np.linalg.norm(plus_fields[name][t] - minus_fields[name][t]), 1e-12))
            del plus, minus
            gc.collect()
        basis_names, matrices = projected_response_matrix(
            plus_fields, minus_fields, basis, epsilon_l2=epsilon)
        svd = {str(t): svd_summary(matrices[t], basis_names) for t in RESPONSE_TIMES_MS}
        for t, matrix in matrices.items():
            arrays[f"A{amplitude:.8g}_T{int(t)}"] = matrix
        signed_by_amp.append(dict(
            amplitude_reference=float(amplitude), epsilon_l2=epsilon,
            basis_names=basis_names, svd=svd, nonlinearity_ratio=nonlinear,
            refractory_ceiling_fraction_by_input=signed_ceiling,
        ))

    ceiling = max(
        [sham["refractory_ceiling_fraction"]]
        + [row["metrics"].get("refractory_ceiling_fraction", 0.0)
           for row in positive_rows]
        + [value for arm in signed_by_amp
           for value in arm["refractory_ceiling_fraction_by_input"].values()]
    )
    label = ("SPATIAL_SATURATED_TONIC_BAD_DATA" if ceiling >= 0.05
             else _state_spatial_label(positive_rows))
    npz_path = os.path.join(CELL_DIR, f"{state_record['state_id']}_projected_matrices.npz")
    _write_npz(npz_path, **arrays)
    record = dict(
        status="COMPLETE", state=state_record, spatial_label=label,
        positive_rows=positive_rows, signed_response=signed_by_amp,
        refractory_ceiling_fraction_max=ceiling,
        projected_matrix_path=npz_path, projected_matrix_sha256=_sha(npz_path),
        sham_max_rate_hz=sham["max_population_rate_hz"],
        sham_artifact=sham["artifact"],
        resource=_meminfo(), finished=_now(),
    )
    out = os.path.join(CELL_DIR, f"{state_record['state_id']}.json")
    _write_json(out, record)
    _write_json(out.replace(".json", ".DONE.json"),
                dict(status="DONE", output_sha256=_sha(out), finished=_now()))
    del sham, base
    gc.collect()
    return record


def cmd_all(args):
    global _SPATIAL_SWAP_BASE_MIB
    if not args.confirm_run:
        raise SystemExit("40k spatial response requires --confirm-run")
    lock, _manifest = _assert_manifest()
    if _meminfo()["mem_available_gib"] < 128.0:
        raise SystemExit("spatial response requires 128 GiB MemAvailable")
    masks, positive, basis = _load_patterns()
    S = PP.build_substrate(1)
    _SPATIAL_SWAP_BASE_MIB = _meminfo()["swap_used_mib"]
    os.makedirs(CELL_DIR, exist_ok=True)
    t0 = time.time()
    with _stage_lock("spatial_all"):
        rows = []
        for state in lock["states"]:
            rows.append(_run_state(S, state, lock, masks, positive, basis))
        counts = {label: sum(r["spatial_label"] == label for r in rows)
                  for label in sorted({r["spatial_label"] for r in rows})}
        canonical = lock["state_source"] != "NONCANONICAL_NO_ONSET_FALLBACK"
        direct = dict(
            status="COMPLETE", state_source=lock["state_source"],
            n_states=len(rows), spatial_label_counts=counts,
            state_labels={r["state"]["state_id"]: r["spatial_label"] for r in rows},
            canonical_spatial_interpretation_authorized=canonical,
            overall_label=("DIRECT_RESPONSE_COMPLETED" if canonical
                           else "DIRECT_RESPONSE_NONCANONICAL_FALLBACK"),
            positive_accounting_contract=("local equal cell/charge/RMS; global charge and RMS "
                                          "controls separated"),
            wall_s=time.time() - t0, resource=_meminfo(), completed=_now(),
        )
        operator = dict(
            status="COMPLETE", n_states=len(rows), response_times_ms=list(RESPONSE_TIMES_MS),
            basis_names=list(basis), two_amplitudes=lock["healthy_I_ref"]["amplitudes"],
            states=[dict(
                state_id=r["state"]["state_id"],
                matrix_path=r["projected_matrix_path"],
                matrix_sha256=r["projected_matrix_sha256"],
                signed_response=r["signed_response"],
            ) for r in rows],
            claim_boundary="projected finite-time response SVD; not a Jacobian eigenmode or DMD",
            completed=_now(),
        )
        _write_json(DIRECT, direct)
        _write_json(OPERATOR, operator)
        _write_json(os.path.join(OUT, "SPATIAL_PROBE_DONE.json"),
                    dict(status="DONE", direct_sha256=_sha(DIRECT),
                         operator_sha256=_sha(OPERATOR), finished=_now()))
    print(json.dumps(direct, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lock")
    sub.add_parser("manifest")
    allp = sub.add_parser("all")
    allp.add_argument("--confirm-run", action="store_true")
    args = ap.parse_args()
    if args.cmd == "lock": cmd_lock(args)
    elif args.cmd == "manifest": cmd_manifest(args)
    elif args.cmd == "all": cmd_all(args)


if __name__ == "__main__":
    main()
