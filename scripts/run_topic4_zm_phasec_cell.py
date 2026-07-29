#!/usr/bin/env python
"""Atomic Phase-C SNN cells on the accepted Z/M+S_G checkpoint substrate.

The script intentionally imports the accepted branch-decision builders instead
of duplicating the SNN configuration.  It never edits the guarded engine.  A
single invocation writes one hash-addressable identity or gain part and then
exits, so a coordinator can use high safe concurrency without retaining several
full 40k-neuron rasters in one parent process.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import os
import resource
import sys
import tempfile
import time

import numpy as np
from scipy import signal

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ENGINE = os.path.join(ROOT, "src", "snn_engine")
for path in (ROOT, HERE, ENGINE):
    if path not in sys.path:
        sys.path.insert(0, path)
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import scripts.run_topic4_zm_branch_decision as R  # noqa: E402
import src.topic4_zm_checkpoint as CK  # noqa: E402
import src.topic4_zm_noise_bank as NB  # noqa: E402
import src.topic4_zm_phasec_contract as PCC  # noqa: E402
import src.topic4_zm_phasec_resources as PRES  # noqa: E402
import src.topic4_zm_phasec_metrics as PCM  # noqa: E402
import src.topic4_zm_phasec_observation as PCO  # noqa: E402
import src.topic4_zm_source_rhythm as SR  # noqa: E402
import src.topic4_zm_ictal_carrier as CG  # noqa: E402


OUT = os.path.join(
    ROOT, "results", "topic4_sef_hfo", "zm_phase_c_tonic_identity"
)
IDENTITY_BURN_MS = 500.0
IDENTITY_MEASURE_MS = 8000.0
GAIN_BURN_MS = 500.0
GAIN_MEASURE_MS = 1000.0
FINE_BIN_MS = 2.0
CURRENT_STRIDE_MS = 1.0
GAIN_DELTAS_MV = (0.05, 0.10)
TIME_BLOCK_MS = 500.0
MANIFEST_PATH = os.path.join(OUT, "phasec_manifest.json")
C1_COORDINATE_MANIFEST_PATH = os.path.join(
    OUT, "phasec1_coordinate_manifest_dt.json"
)
C1_GAIN_TRIGGER_MANIFEST_PATH = os.path.join(
    OUT, "c1_gain_trigger_manifest.json"
)
C1_DT2_CONFIRMATION_MANIFEST_PATH = os.path.join(
    OUT, "phasec1_dt2_confirmation_manifest.json"
)
C1_BASE_PART_SCHEMA = "zm_phasec1_base_part_v1_2026-07-28"
C1_OBSERVABLES_SCHEMA = "zm_phasec1_observables_v1_2026-07-28"
C1_GAIN_PART_SCHEMA = "zm_phasec1_conditional_gain_part_v1_2026-07-28"
HIERARCHICAL_ARRAY_FIELDS = (
    "rho80_active_core_by_block_window",
    "block_isi_cv2_by_panel_neuron",
    "block_refractory_isi_numerator_by_stratum",
    "block_refractory_isi_denominator_by_stratum",
    "pair_corr_by_block_and_pair",
    "pair_null_median_by_block_and_draw",
    "active_area_fraction_by_block_window",
)


def _rss_gb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2, 3)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_bytes(payload):
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _object_sha(payload):
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _read_json(path):
    with open(path, encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"{path}: expected a JSON object")
    return value


def _publish_json_once(path, payload):
    """Durably publish one JSON object without ever replacing an old result."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp",
        dir=os.path.dirname(path),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                payload, handle, indent=2, sort_keys=True, ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _manifest(path=MANIFEST_PATH):
    manifest = _read_json(path)
    PCC.require_production_manifest(manifest)
    return manifest


def _reuse_or_fail(path, expected):
    """Return True for an exact completed cell; never overwrite a conflicting part."""
    if not os.path.exists(path):
        return False
    with open(path) as handle:
        old = json.load(handle)
    mismatches = {
        key: (old.get(key), value)
        for key, value in expected.items()
        if old.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"existing Phase-C part conflicts with request: {mismatches}")
    if old.get("status") not in {"complete", "scientific_failure"}:
        raise RuntimeError(f"existing Phase-C part is not terminal: {path}")
    print(f"[phasec cell] reused exact terminal part -> {path}", flush=True)
    return True


def _publish_npz_once(path, **arrays):
    """Durably publish one NPZ without overwriting an old or orphaned part."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        raise FileExistsError(f"refusing to overwrite existing result: {path}")
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp.npz",
        dir=os.path.dirname(path),
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.link(tmp, path)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _publish_content_addressed_npz(directory, stem, arrays):
    """Publish immutable NPZ bytes without making an orphan block resume.

    The cell JSON is the commit marker.  If a process dies after linking this
    NPZ but before publishing JSON, an exact rerun can safely reuse equal bytes
    or publish unequal bytes under a different SHA-addressed filename.
    """
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=f".{stem}.", suffix=".tmp.npz", dir=directory
    )
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        digest = _sha(tmp)
        path = os.path.join(directory, f"{stem}.{digest}.npz")
        try:
            os.link(tmp, path)
        except FileExistsError:
            if _sha(path) != digest:
                raise RuntimeError(
                    "content-addressed NPZ path has conflicting bytes"
                )
        return path, digest
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _semantic_npz_sha(path):
    """Hash NPZ array semantics independently of ZIP metadata."""
    h = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for name in sorted(data.files):
            array = np.ascontiguousarray(np.asarray(data[name]))
            h.update(
                f"{name}|{array.dtype.str}|{array.shape}|".encode("utf-8")
            )
            h.update(array.tobytes())
    return h.hexdigest()


def _validate_fixed_panels(seed_row, seed, ctx):
    """Resolve only the panel contract embedded in the locked v1.3 production manifest."""
    row = seed_row.get("fixed_panels")
    if not isinstance(row, dict):
        raise RuntimeError(f"Phase-C manifest lacks fixed panels for seed {seed}")
    if row.get("activity_independent") is not True:
        raise RuntimeError(f"seed {seed} panel is not activity-independent")
    if not isinstance(row.get("panel_sha256"), str):
        raise RuntimeError(f"seed {seed} panel SHA is absent")
    if _object_sha({
        key: value for key, value in row.items() if key != "panel_sha256"
    }) != row["panel_sha256"]:
        raise RuntimeError(f"seed {seed} embedded panel self-hash mismatch")
    analysis = np.asarray(row.get("analysis_panel_E_ids"), int)
    pairwise = np.asarray(row.get("pairwise_panel_E_ids"), int)
    for name, ids in (("analysis", analysis), ("pairwise", pairwise)):
        if (
            ids.ndim != 1
            or ids.size == 0
            or np.any(ids < 0)
            or np.any(ids >= ctx["S"]["NE"])
            or np.unique(ids).size != ids.size
        ):
            raise RuntimeError(f"seed {seed} {name} panel is invalid")
    analysis_n_core = int(row.get("analysis_panel_n_core", -1))
    pairwise_n_core = int(row.get("pairwise_panel_n_core", -1))
    if analysis_n_core + int(
        row.get("analysis_panel_n_surround", -1)
    ) != analysis.size:
        raise RuntimeError("analysis-panel stratum counts do not match IDs")
    if pairwise_n_core + int(
        row.get("pairwise_panel_n_surround", -1)
    ) != pairwise.size:
        raise RuntimeError("pairwise-panel stratum counts do not match IDs")
    core = np.asarray(ctx["core"], bool)
    if (
        not np.all(core[analysis[:analysis_n_core]])
        or np.any(core[analysis[analysis_n_core:]])
        or not np.all(core[pairwise[:pairwise_n_core]])
        or np.any(core[pairwise[pairwise_n_core:]])
    ):
        raise RuntimeError("fixed-panel IDs do not match locked core/surround order")
    return row, analysis, pairwise


def _resolution_seed_row(manifest, seed, resolution):
    """Return one resolution-local input row with explicit parent lineage.

    ``dt2`` is not a numerical conversion of the native row.  It is a separate
    canonical configuration/anchor/state family generated upstream.  Anatomy
    panels remain those selected from the parent native configuration.
    """
    native = manifest["per_seed"][str(int(seed))]
    if resolution == "dt":
        return native, native
    if resolution != "dt2":
        raise RuntimeError(f"unsupported Phase-C resolution: {resolution}")
    row = native.get("resolution_confirmations", {}).get("dt2")
    if not isinstance(row, dict):
        raise RuntimeError(
            f"final Phase-C manifest lacks independent dt2 inputs for seed {seed}"
        )
    if (
        int(seed) not in (1, 3)
        or row.get("resolution") != "dt2"
        or row.get("parent_config_sha") != native.get("canonical_config_sha")
        or row.get("panel_selection_config_sha")
        != native.get("canonical_config_sha")
        or row.get("panel_selection_resolution") != "parent_native_dt"
        or row.get("fixed_panels") != native.get("fixed_panels")
    ):
        raise RuntimeError(
            f"seed {seed} dt2 parent-config/anatomy-panel lineage mismatch"
        )
    return native, row


def _locked_refs(manifest, seed, state_tag, replicate, *, resolution="dt"):
    _native, seed_row = _resolution_seed_row(manifest, seed, resolution)
    if state_tag == "pre_entry__natural":
        family = seed_row["c0_pre_entry_gain_control"]
    elif state_tag in {"bounded_mid__rising", "bounded_mid__peak"}:
        family = seed_row["c0_carrier_states"][state_tag.rsplit("__", 1)[1]]
    else:
        raise RuntimeError(f"{state_tag!r} is not a locked C0 state")
    banks = {row["replicate"]: row for row in family["noise_banks"]}
    if replicate not in banks:
        raise RuntimeError(f"{state_tag} lacks locked noise bank {replicate}")
    return seed_row, family["state"], banks[replicate]


def _load_state(ctx, state_ref):
    path = os.path.join(ROOT, state_ref["path"])
    if _sha(path) != state_ref["file_sha256"]:
        raise RuntimeError(f"locked state file SHA mismatch: {state_ref['path']}")
    engine_sha = ctx["cfg_locked"]["engine_sha256"]["src/snn_engine/kick_probe.py"]
    state, manifest = CK.load_state_npz(
        path,
        expected_config_sha=ctx["cfg_sha"],
        expected_engine_sha=engine_sha,
        expected_dt=ctx["dt"],
    )
    if manifest.get("state_hash") != state_ref["state_hash"]:
        raise RuntimeError(f"locked semantic state hash mismatch: {state_ref['path']}")
    if int(np.asarray(state["t"])) != int(state_ref["t_step"]):
        raise RuntimeError(f"locked state step mismatch: {state_ref['path']}")
    return state, manifest


def _locked_inputs(args, ctx, *, state_tag=None):
    manifest = _manifest(args.manifest)
    provenance = manifest.get("provenance", {})
    input_lock = {
        "path": provenance.get("phasec_input_manifest_path"),
        "file_sha256": provenance.get(
            "phasec_input_manifest_file_sha256"
        ),
        "manifest_sha256": provenance.get(
            "phasec_input_manifest_manifest_sha256"
        ),
    }
    if not all(
        isinstance(input_lock[key], str) and input_lock[key]
        for key in input_lock
    ):
        raise RuntimeError("final Phase-C manifest lacks input-manifest provenance")
    input_path = os.path.join(ROOT, input_lock["path"])
    if not os.path.isfile(input_path) or _sha(input_path) != input_lock[
        "file_sha256"
    ]:
        raise RuntimeError("Phase-C input-manifest file provenance mismatch")
    input_manifest = _read_json(input_path)
    _validate_self_hash(input_manifest, label="Phase-C input manifest")
    if (
        input_manifest["manifest_sha256"] != input_lock["manifest_sha256"]
        or input_manifest.get("production_authorized") is not False
    ):
        raise RuntimeError("Phase-C input manifest is not the locked bootstrap")
    producer_locks = manifest.get("provenance", {}).get(
        "producer_file_sha256"
    )
    if not isinstance(producer_locks, dict) or not producer_locks:
        raise RuntimeError("Phase-C manifest lacks producer-file hash locks")
    for relative_path, expected_sha in sorted(producer_locks.items()):
        live_path = os.path.join(ROOT, relative_path)
        if not os.path.isfile(live_path) or _sha(live_path) != expected_sha:
            raise RuntimeError(
                f"Phase-C live producer hash mismatch: {relative_path}"
            )
    selected_state_tag = args.state_tag if state_tag is None else state_tag
    native_seed_row, resolution_seed_row = _resolution_seed_row(
        manifest, args.seed, args.resolution
    )
    _locked_seed_row, state_ref, bank_ref = _locked_refs(
        manifest, args.seed, selected_state_tag, args.replicate,
        resolution=args.resolution,
    )
    _native_locked_row, native_state_ref, _native_bank_ref = _locked_refs(
        manifest, args.seed, selected_state_tag, args.replicate,
        resolution="dt",
    )
    expected_config_sha = (
        native_seed_row["canonical_config_sha"]
        if args.resolution == "dt"
        else resolution_seed_row["config_sha"]
    )
    if expected_config_sha != ctx["cfg_sha"]:
        raise RuntimeError("Phase-C manifest/canonical config mismatch")
    if args.resolution == "dt2":
        config_path = os.path.join(ROOT, resolution_seed_row["config_path"])
        if (
            not os.path.isfile(config_path)
            or _sha(config_path)
            != resolution_seed_row["config_file_sha256"]
        ):
            raise RuntimeError("locked independent dt2 config file SHA mismatch")
        config_lock = _read_json(config_path)
        if (
            config_lock.get("resolution") != "dt2"
            or config_lock.get("config_sha") != ctx["cfg_sha"]
            or config_lock.get("parent_config_sha")
            != native_seed_row["canonical_config_sha"]
            or float(config_lock.get("dt", np.nan)) != float(ctx["dt"])
        ):
            raise RuntimeError("live dt2 config/parent/native lineage mismatch")
    anchor_path = os.path.join(ROOT, resolution_seed_row["anchor_path"])
    if _sha(anchor_path) != resolution_seed_row["anchor_file_sha256"]:
        raise RuntimeError("locked anchor file SHA mismatch")
    anchor = _read_json(anchor_path)
    if (
        not isinstance(anchor.get("locks"), dict)
        or anchor.get("config_sha") != ctx["cfg_sha"]
        or anchor.get("resolution", "dt") != args.resolution
        or float(anchor.get("dt", np.nan)) != float(ctx["dt"])
    ):
        raise RuntimeError("locked anchor lacks the rest-reference contract")
    panel_row, analysis_ids, pairwise_ids = _validate_fixed_panels(
        native_seed_row, args.seed, ctx
    )
    state, state_manifest = _load_state(ctx, state_ref)
    bank = NB.build_noise_bank(
        ctx["cfg_sha"], int(args.seed), int(state_ref["t_step"]), args.replicate
    )
    for key in ("bank_sha", "replicate", "start_step", "is_paired", "ext_mean_only"):
        if bank.get(key) != bank_ref.get(key):
            raise RuntimeError(f"locked future-noise field mismatch: {key}")
    return {
        "manifest": manifest,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": _sha(args.manifest),
        "manifest_path": os.path.relpath(args.manifest, ROOT),
        "seed_row": resolution_seed_row,
        "native_seed_row": native_seed_row,
        "anchor": anchor,
        "state_ref": state_ref,
        "state": state,
        "state_manifest": state_manifest,
        "bank": bank,
        "bank_ref": bank_ref,
        "native_homolog_state_ref": native_state_ref,
        "panel_row": panel_row,
        "analysis_ids": analysis_ids,
        "pairwise_ids": pairwise_ids,
    }


def _validate_self_hash(payload, *, label):
    claimed = payload.get("manifest_sha256")
    body = {key: value for key, value in payload.items()
            if key != "manifest_sha256"}
    if not isinstance(claimed, str) or _object_sha(body) != claimed:
        raise RuntimeError(f"{label} self-hash mismatch")


def _slow_state_sha(z, m, S_G):
    """Hash the exact arrays that will be restored, not an unsaved precursor."""
    h = hashlib.sha256()
    for key, value in (("z", z), ("m", m)):
        array = np.ascontiguousarray(np.asarray(value))
        h.update(f"{key}|{array.dtype.str}|{array.shape}|".encode())
        h.update(array.tobytes())
    h.update(f"S_G|{float(S_G):.17g}".encode())
    return h.hexdigest()


def _coordinate_contract(manifest, coordinate_path, *, resolution="dt"):
    """Validate the final Phase-C -> coordinate edge without a hash cycle.

    The coordinate builder may record the pre-coordinate bootstrap manifest as
    construction provenance.  Production authority is exclusively the final
    Phase-C manifest's forward locks below.
    """
    c1 = manifest.get("c1")
    if not isinstance(c1, dict):
        raise RuntimeError("final Phase-C manifest lacks C1 contract")
    by_resolution = c1.get("coordinate_manifests")
    if isinstance(by_resolution, dict):
        forward = by_resolution.get(str(resolution))
    else:
        forward = c1.get("coordinate_manifest") if resolution == "dt" else None
    if not isinstance(forward, dict):
        raise RuntimeError(
            "final Phase-C manifest lacks c1.coordinate_manifest"
        )
    expected_path = forward.get("path")
    expected_file_sha = forward.get("file_sha256")
    expected_manifest_sha = forward.get("manifest_sha256")
    expected_semantic_sha = forward.get("semantic_sha256")
    if not all(isinstance(value, str) and value for value in (
        expected_path, expected_file_sha, expected_manifest_sha,
        expected_semantic_sha,
    )):
        raise RuntimeError(
            "final Phase-C manifest lacks immutable coordinate-manifest locks"
        )
    actual = os.path.abspath(coordinate_path)
    locked = os.path.abspath(os.path.join(ROOT, expected_path))
    if actual != locked:
        raise RuntimeError("coordinate manifest path differs from final Phase-C lock")
    if not os.path.isfile(actual) or _sha(actual) != expected_file_sha:
        raise RuntimeError("coordinate manifest file SHA mismatch")
    coordinate = _read_json(actual)
    _validate_self_hash(coordinate, label="C1 coordinate manifest")
    semantic_body = {
        key: value for key, value in coordinate.items()
        if key not in {"manifest_sha256", "semantic_sha256"}
    }
    if (
        coordinate["manifest_sha256"] != expected_manifest_sha
        or coordinate.get("semantic_sha256") != expected_semantic_sha
        or _object_sha(semantic_body) != expected_semantic_sha
    ):
        raise RuntimeError("coordinate manifest/semantic SHA mismatch")
    input_provenance = manifest.get("provenance", {})
    if (
        coordinate.get("parent_phasec_input_manifest_path")
        != input_provenance.get("phasec_input_manifest_path")
        or coordinate.get("parent_phasec_input_manifest_file_sha256")
        != input_provenance.get("phasec_input_manifest_file_sha256")
        or coordinate.get("parent_phasec_input_manifest_sha256")
        != input_provenance.get("phasec_input_manifest_manifest_sha256")
    ):
        raise RuntimeError(
            "coordinate parent Phase-C input-manifest provenance mismatch"
        )
    producer_locks = coordinate.get("producer_file_sha256")
    if not isinstance(producer_locks, dict) or not producer_locks:
        raise RuntimeError("coordinate manifest lacks producer locks")
    for relative_path, expected_sha in sorted(producer_locks.items()):
        path = os.path.join(ROOT, relative_path)
        if not os.path.isfile(path) or _sha(path) != expected_sha:
            raise RuntimeError(
                f"C1 coordinate live producer hash mismatch: {relative_path}"
            )
    return coordinate


def _load_coordinate_state(
    manifest, coordinate_path, *, seed, tier, cell_id, ctx
):
    coordinate = _coordinate_contract(
        manifest, coordinate_path, resolution=ctx.get("resolution", "dt")
    )
    seed_row = coordinate.get("seeds", {}).get(str(int(seed)))
    if not isinstance(seed_row, dict):
        raise RuntimeError(f"coordinate manifest lacks seed {seed}")
    if seed_row.get("config_sha") != ctx["cfg_sha"]:
        raise RuntimeError("coordinate/canonical config SHA mismatch")
    matches = [
        row for row in seed_row.get("cells", [])
        if row.get("cell_id") == cell_id and row.get("tier") == tier
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"coordinate cell must resolve exactly once: {seed}/{tier}/{cell_id}"
        )
    cell = matches[0]
    if cell.get("status") != "valid":
        raise RuntimeError(
            f"invalid_physical_cell cannot be simulated: {seed}/{tier}/{cell_id}"
        )
    npz_path = os.path.join(ROOT, seed_row["npz_path"])
    resolution = ctx.get("resolution", "dt")
    file_by_resolution = manifest["c1"].get(
        "coordinate_npz_file_sha256_by_seed_by_resolution"
    )
    semantic_by_resolution = manifest["c1"].get(
        "coordinate_npz_semantic_sha256_by_seed_by_resolution"
    )
    final_npz_locks = (
        file_by_resolution.get(resolution)
        if isinstance(file_by_resolution, dict) else None
    )
    final_semantic_locks = (
        semantic_by_resolution.get(resolution)
        if isinstance(semantic_by_resolution, dict) else None
    )
    coordinate_file_sha = seed_row.get("npz_file_sha256")
    coordinate_semantic_sha = seed_row.get("npz_semantic_sha256")
    if (
        not isinstance(final_npz_locks, dict)
        or not isinstance(final_semantic_locks, dict)
        or final_npz_locks.get(str(int(seed))) != coordinate_file_sha
        or final_semantic_locks.get(str(int(seed))) != coordinate_semantic_sha
    ):
        raise RuntimeError("final Phase-C manifest/coordinate NPZ lock mismatch")
    if (
        not os.path.isfile(npz_path)
        or _sha(npz_path) != coordinate_file_sha
        or _semantic_npz_sha(npz_path) != coordinate_semantic_sha
    ):
        raise RuntimeError("coordinate NPZ SHA mismatch")
    row_index = int(cell.get("array_row", -1))
    with np.load(npz_path, allow_pickle=False) as data:
        required = {"cell_ids", "tiers", "status", "z", "m", "S_G"}
        if not required.issubset(data.files):
            raise RuntimeError("coordinate NPZ lacks required full slow fields")
        if row_index < 0 or row_index >= len(data["cell_ids"]):
            raise RuntimeError("coordinate array_row is out of bounds")
        if (
            str(data["cell_ids"][row_index]) != cell_id
            or str(data["tiers"][row_index]) != tier
            or str(data["status"][row_index]) != "valid"
        ):
            raise RuntimeError("coordinate NPZ row identity mismatch")
        z = np.array(data["z"][row_index], copy=True)
        m = np.array(data["m"][row_index], copy=True)
        S_G = float(np.asarray(data["S_G"][row_index]))
    if z.shape != (ctx["S"]["NE"],) or m.shape != z.shape:
        raise RuntimeError("coordinate slow fields do not match NE")
    actual_state_sha = _slow_state_sha(z, m, S_G)
    if actual_state_sha != cell.get("state_sha256"):
        raise RuntimeError(
            "coordinate semantic state SHA does not match serialized fields"
        )
    return {
        "coordinate_manifest": coordinate,
        "coordinate_manifest_file_sha256": _sha(coordinate_path),
        "coordinate_cell": cell,
        "coordinate_seed": seed_row,
        "coordinate_npz_path": os.path.relpath(npz_path, ROOT),
        "coordinate_npz_file_sha256": coordinate_file_sha,
        "coordinate_npz_semantic_sha256": coordinate_semantic_sha,
        "z": z,
        "m": m,
        "S_G": S_G,
    }


def _restore_coordinate(base_state, coordinate, ctx):
    """Replace every current-affecting C1 slow coordinate and nothing fast."""
    state = copy.deepcopy(base_state)
    n_e = int(ctx["S"]["NE"])
    z = np.asarray(state.get("slow.z"))
    m = np.asarray(state.get("slow.m"))
    if z.ndim != 1 or m.shape != z.shape or z.size < n_e:
        raise RuntimeError("base checkpoint lacks aligned full Z/M state")
    state["slow.z"] = np.array(z, copy=True)
    state["slow.m"] = np.array(m, copy=True)
    state["slow.z"][:n_e] = coordinate["z"]
    state["slow.m"][:n_e] = coordinate["m"]
    state["slow.S_G"] = np.asarray(float(coordinate["S_G"]))
    # I cells are not Z/M dynamical variables in this substrate.  Requiring
    # their canonical constants prevents a partial-field restore from hiding.
    if (
        state["slow.z"].size > n_e
        and (
            not np.all(state["slow.z"][n_e:] == 1.0)
            or not np.all(state["slow.m"][n_e:] == 0.0)
        )
    ):
        raise RuntimeError("canonical I-cell Z/M constants are corrupted")
    return state


def _c1_locked_inputs(args, ctx, *, require_trigger=False):
    phase = args.phase
    locks = _locked_inputs(
        args, ctx, state_tag=f"bounded_mid__{phase}"
    )
    coordinate = _load_coordinate_state(
        locks["manifest"], args.coordinate_manifest,
        seed=args.seed, tier=args.tier, cell_id=args.cell_id, ctx=ctx,
    )
    locks.update(coordinate)
    locks["state"] = _restore_coordinate(locks["state"], coordinate, ctx)
    if args.resolution == "dt2":
        if require_trigger:
            raise RuntimeError(
                "C1 dt2 is non-tonic confirmation-only; conditional AI gain "
                "is neither required nor authorized"
            )
        selection_path = os.path.abspath(args.dt2_confirmation_manifest)
        if selection_path != os.path.abspath(
            C1_DT2_CONFIRMATION_MANIFEST_PATH
        ):
            raise RuntimeError(
                "dt2 C1 must consume the canonical write-once confirmation lock"
            )
        selection = _read_json(selection_path)
        _validate_self_hash(
            selection, label="C1 dt2 confirmation manifest"
        )
        if (
            selection.get("schema")
            != "zm_phasec1_dt2_confirmation_manifest_v1_2026-07-28"
            or selection.get("resolution") != "dt2"
            or selection.get("selection_is_closed") is not True
            or selection.get("final_phasec", {}).get("manifest_sha256")
            != locks["manifest_sha256"]
            or selection.get("final_phasec", {}).get("file_sha256")
            != locks["manifest_file_sha256"]
            or selection.get("coordinate_manifests", {}).get(
                "dt2", {}
            ).get("manifest_sha256")
            != coordinate["coordinate_manifest"]["manifest_sha256"]
            or selection.get("coordinate_manifests", {}).get(
                "dt2", {}
            ).get("semantic_sha256")
            != coordinate["coordinate_manifest"]["semantic_sha256"]
            or selection.get("coordinate_producer_file_sha256")
            != coordinate["coordinate_manifest"]["producer_file_sha256"]
        ):
            raise RuntimeError("C1 dt2 confirmation parent/coordinate drift")
        expected_path = _c1_base_relative_path(
            "dt2", args.seed, args.tier, args.cell_id,
            args.phase, args.replicate,
        )
        matches = [
            row for row in selection.get("expected_base_arms", [])
            if (
                int(row.get("seed", -1)) == int(args.seed)
                and row.get("tier") == args.tier
                and row.get("cell_id") == args.cell_id
                and row.get("phase") == args.phase
                and row.get("noise") == args.replicate
                and row.get("path") == expected_path
            )
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "requested dt2 C1 arm is absent/duplicated in closed selection"
            )
        selected_arm = matches[0]
        exact = {
            "schema": C1_BASE_PART_SCHEMA,
            "phasec_manifest_sha256": locks["manifest_sha256"],
            "phasec_manifest_file_sha256": locks["manifest_file_sha256"],
            "coordinate_manifest_sha256": coordinate[
                "coordinate_manifest"
            ]["manifest_sha256"],
            "coordinate_manifest_semantic_sha256": coordinate[
                "coordinate_manifest"
            ]["semantic_sha256"],
            "coordinate_manifest_file_sha256": coordinate[
                "coordinate_manifest_file_sha256"
            ],
            "coordinate_npz_file_sha256": coordinate[
                "coordinate_npz_file_sha256"
            ],
            "coordinate_npz_semantic_sha256": coordinate[
                "coordinate_npz_semantic_sha256"
            ],
            "config_sha": ctx["cfg_sha"],
            "fast_base_state_hash": locks["state_manifest"]["state_hash"],
            "state_file_sha256": locks["state_ref"]["file_sha256"],
            "noise_bank_sha": locks["bank"]["bank_sha"],
            "slow_state_sha256": coordinate[
                "coordinate_cell"
            ]["state_sha256"],
            "trajectory_id": coordinate["coordinate_cell"]["trajectory_id"],
            "path_index": int(
                coordinate["coordinate_cell"]["path_index"]
            ),
            "path_direction": coordinate[
                "coordinate_cell"
            ]["path_direction"],
            "resolution": "dt2",
            "seed": int(args.seed),
            "tier": args.tier,
            "cell_id": args.cell_id,
            "phase": args.phase,
            "noise": args.replicate,
            "path": expected_path,
            "burn_in_ms": float(
                locks["manifest"]["c0"]["protocols"]["identity"][
                    "burn_in_ms"
                ]
            ),
            "measure_ms": float(
                locks["manifest"]["c0"]["protocols"]["identity"][
                    "measure_ms"
                ]
            ),
        }
        if any(selected_arm.get(key) != value for key, value in exact.items()):
            raise RuntimeError("C1 dt2 expected-arm provenance mismatch")
        locks["dt2_confirmation_manifest"] = selection
        locks["dt2_confirmation_manifest_file_sha256"] = _sha(
            selection_path
        )
    if require_trigger:
        trigger = _read_json(args.trigger_manifest)
        _validate_self_hash(trigger, label="C1 conditional-gain trigger manifest")
        if (
            trigger.get("phasec_manifest_sha256") != locks["manifest_sha256"]
            or trigger.get("phasec_manifest_file_sha256")
            != locks["manifest_file_sha256"]
            or trigger.get("coordinate_manifest_sha256")
            != coordinate["coordinate_manifest"]["manifest_sha256"]
            or trigger.get("coordinate_manifest_file_sha256")
            != coordinate["coordinate_manifest_file_sha256"]
            or trigger.get("resolution") != args.resolution
        ):
            raise RuntimeError("conditional-gain trigger parent hash mismatch")
        if os.path.abspath(args.trigger_manifest) != os.path.abspath(
            C1_GAIN_TRIGGER_MANIFEST_PATH
        ):
            raise RuntimeError(
                "conditional gain must consume the canonical write-once trigger"
            )
        key = (int(args.seed), args.tier, args.cell_id)
        candidates = [
            row for row in trigger.get("triggered_cells", [])
            if (int(row["seed"]), row["tier"], row["cell_id"]) == key
        ]
        if len(candidates) != 1:
            raise RuntimeError("cell is absent or duplicated in write-once trigger")
        trigger_row = candidates[0]
        if trigger_row.get("slow_state_sha256") != coordinate[
            "coordinate_cell"
        ]["state_sha256"]:
            raise RuntimeError("trigger/coordinate slow-state SHA mismatch")
        arm_path = _c1_gain_relative_path(
            args.resolution, args.seed, args.tier, args.cell_id,
            args.phase, args.replicate, args.sign * args.delta_mV,
        )
        expected_arms = {
            row["path"] for row in trigger_row.get(
                "expected_carrier_gain_arms", []
            )
        }
        if arm_path not in expected_arms:
            raise RuntimeError("requested arm is absent from write-once trigger")
        for row in trigger_row.get("triggering_base_parts", []):
            path = os.path.join(ROOT, row["part_path"])
            if not os.path.isfile(path) or _sha(path) != row["part_sha256"]:
                raise RuntimeError("triggering base-part provenance drift")
        locks["trigger_manifest"] = trigger
        locks["trigger_manifest_file_sha256"] = _sha(args.trigger_manifest)
        locks["trigger_row"] = trigger_row
    return locks


def _run(ctx, state, bank, duration_ms, *,
         dump_i_spikes=False, recorder=None, perturb=None,
         return_final_state=False):
    base_slow = R.make_slow(ctx, freeze_arm=None)
    effective_observer = PCO.PhaseCEffectiveSlowObserver(
        base_slow,
        ctx["core"],
        stride_steps=max(1, int(round(CURRENT_STRIDE_MS / ctx["dt"]))),
    )
    slow = R.FS.FreezeWrapper(
        effective_observer, R.FS.FreezePolicy.for_arm("freeze_all")
    )
    ckpt = CK.ZMCheckpoint(
        initial_state=state,
        return_final_state=return_final_state,
        rng_state=bank["rng_state"],
        ext_mean_only=bank["ext_mean_only"],
    )
    substrate = ctx["S"]
    params = dataclasses.replace(substrate["p"], T=float(duration_ms))
    substrate["net"]["rng"] = np.random.default_rng(substrate["seed"])
    result = R.simulate_kick(
        params,
        substrate["net"],
        0.0,
        slow=slow,
        kick_center=list(substrate["src_xy"]),
        r_kick=R.PP.R_KICK,
        t_kick=1e9,
        V_th_per_neuron=substrate["vth"],
        verbose=False,
        lfp_recorder=recorder,
        early_stop_runaway=True,
        es_thresh_hz=R.ES_THRESH_HZ,
        es_dur_ms=100.0,
        dump_i_spikes=dump_i_spikes,
        perturb=perturb,
        zm_ckpt=ckpt,
    )
    return result, ckpt, effective_observer


def _finite_summary(values):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0, "mean": None, "median": None, "q05": None, "q95": None}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "q05": float(np.percentile(x, 5)),
        "q95": float(np.percentile(x, 95)),
    }


def _effective_summary(state, ctx):
    if state is None:
        return None
    cfg = ctx["cfg_locked"]["slow_field"]
    components = PCO.reconstruct_effective_snapshot(
        state,
        nE=ctx["S"]["NE"],
        alpha_G=float(cfg["alpha_G"]),
        eta_m=float(cfg["eta_m"]),
        alpha_H=float(cfg.get("alpha_H", 0.0)),
        beta_SG=float(cfg.get("beta_SG", 0.0)),
    )
    core = np.asarray(ctx["core"], bool)
    keys = (
        "raw_ampa_mV", "raw_gaba_mV", "recurrent_ampa_removed_by_SG_mV",
        "effective_excitation_mV", "effective_inhibition_z_mV",
        "adaptation_m_current_mV", "effective_outward_total_mV",
        "effective_net_drive_mV",
    )
    return {
        "evidence_label": components["evidence_label"],
        "S_G": components["S_G"],
        "divisive_load": components["divisive_load"],
        "identity_max_abs_error_mV": components["identity_max_abs_error_mV"],
        "components": {
            key: {
                "all_E": _finite_summary(components[key]),
                "core_E": _finite_summary(components[key][core]),
                "surround_E": _finite_summary(components[key][~core]),
            }
            for key in keys
        },
        "claim_boundary": components["claim_boundary"],
    }


def _margin_summary(state, ctx):
    if state is None:
        return None
    return PCO.free_e_threshold_margin_snapshot(
        state,
        ctx["S"]["vth"],
        nE=ctx["S"]["NE"],
        core_mask_E=ctx["core"],
    )


def _spectral_1d(trace, fs_hz):
    x = np.asarray(trace, float)
    if x.size < 128 or not np.any(np.isfinite(x)):
        return {"status": "insufficient_trace"}
    x = signal.detrend(np.nan_to_num(x))
    freq, power = signal.welch(x, fs=float(fs_hz), nperseg=min(x.size, 2048))
    band = (freq >= 5.0) & (freq <= min(150.0, 0.45 * fs_hz))
    p = power[band]
    if not p.size or p.sum() <= 0:
        return {"status": "zero_power"}
    pn = p / p.sum()
    entropy = -np.sum(pn * np.log(pn + 1e-30)) / np.log(pn.size)
    peak = int(np.argmax(p))
    peak_fraction = float(p[peak] / p.sum())
    return {
        "status": "ok",
        "band_hz": [5.0, float(min(150.0, 0.45 * fs_hz))],
        "dominant_frequency_hz": float(freq[band][peak]),
        "spectral_entropy": float(entropy),
        "dominant_bin_fraction": peak_fraction,
        "broadband_continuity": float(1.0 - peak_fraction),
        "claim_boundary": "current-based raw-synaptic proxy; no empirical ictal match",
    }


def _spectral_summary(trace, fs_hz, contact_names):
    x = np.asarray(trace, float)
    if x.ndim != 2 or x.shape[1] != len(contact_names):
        return {
            "status": "invalid_contact_trace",
            "n_contacts_expected": int(len(contact_names)),
        }
    per_contact = {
        str(name): _spectral_1d(x[:, index], fs_hz)
        for index, name in enumerate(contact_names)
    }
    return {
        "status": "ok",
        "proxy_definition": (
            "spatially weighted |I_E|+|I_I| on E cells before Z/M/S_G; "
            "not a transmembrane current"
        ),
        "per_contact": per_contact,
        "spatial_contact_mean_secondary": _spectral_1d(
            np.mean(x, axis=1), fs_hz
        ),
        "claim_boundary": (
            "raw-synaptic virtual-SEEG proxy only; per-contact spectra preserve "
            "phase-relayed activity and do not establish an empirical ictal match"
        ),
    }


def _postburn_trace(trace_payload, burn_ms):
    time_ms = np.asarray(trace_payload.get("sample_time_ms", []), float)
    keep = time_ms >= float(burn_ms)
    out = {}
    for key, value in trace_payload.items():
        array = np.asarray(value) if isinstance(value, np.ndarray) else None
        if array is not None and array.shape == time_ms.shape:
            out[key] = array[keep]
        else:
            out[key] = value
    out["sample_time_ms"] = time_ms[keep] - float(burn_ms)
    return out


def _trace_stats(trace_payload):
    rows = {}
    for key, value in trace_payload.items():
        if not key.endswith("_mean_mV"):
            continue
        rows[key] = _finite_summary(value)
    return {
        "evidence_label": trace_payload.get("evidence_label"),
        "n_recorded": int(len(np.asarray(trace_payload.get("sample_time_ms", [])))),
        "sample_dt_ms": trace_payload.get("sample_dt_ms"),
        "components": rows,
        "claim_boundary": trace_payload.get("claim_boundary"),
    }


def _first_sustained(mask, n_required):
    run = 0
    for index, value in enumerate(np.asarray(mask, bool)):
        run = run + 1 if value else 0
        if run >= int(n_required):
            return index - int(n_required) + 1
    return None


def _carrier_gate_evidence(ctx, result, burn_ms, anchor_locks, area_threshold):
    """Reapply the upstream rest and whole-sheet gates after burn-in."""
    bin_ms = 25.0
    metrics = R.segment_metrics(ctx, result, bin_ms=bin_ms)
    b0 = int(round(float(burn_ms) / bin_ms))
    if metrics["n_bins"] <= b0:
        if result.get("runaway_early_stop_ms") is None:
            raise RuntimeError("continuation is too short to evaluate carrier gates")
        return (
            {
                "runaway": True,
                "whole_sheet_plateau": False,
                "empirical_rest_dwell": False,
            },
            {
                "bin_ms": bin_ms,
                "burn_bin": b0,
                "postburn_bins": 0,
                "not_evaluable_reason": "runaway_before_postburn_window",
            },
            {
                "carrier_gate_d_rest": np.asarray([], np.float32),
                "carrier_gate_A_active": np.asarray([], np.float32),
                "carrier_gate_r_all_hz": np.asarray([], np.float32),
                "carrier_gate_r_core_hz": np.asarray([], np.float32),
                "carrier_gate_bin_ms": np.asarray(bin_ms),
            },
        )
    distance = R.MC.rest_distance(metrics, anchor_locks["rest_reference"])[b0:]
    area = np.asarray(metrics["A_active"], float)[b0:]
    rate_all = np.asarray(metrics["r_all"], float)[b0:]
    rate_core = np.asarray(metrics["r_core"], float)[b0:]
    if not (distance.size == area.size == rate_all.size == rate_core.size):
        raise RuntimeError("carrier-gate trace length mismatch")
    rest_index = R.MC.first_rest_return(
        distance,
        bin_ms,
        float(anchor_locks["d_rest_thresh"]),
        float(anchor_locks["rest_dwell_ms"]),
    )
    plateau_need = max(1, int(round(500.0 / bin_ms)))
    plateau_index = _first_sustained(area >= float(area_threshold), plateau_need)
    gates = {
        "runaway": result.get("runaway_early_stop_ms") is not None,
        "whole_sheet_plateau": plateau_index is not None,
        "empirical_rest_dwell": rest_index is not None,
    }
    evidence = {
        "bin_ms": bin_ms,
        "burn_bin": b0,
        "d_rest_threshold": float(anchor_locks["d_rest_thresh"]),
        "rest_dwell_ms": float(anchor_locks["rest_dwell_ms"]),
        "first_rest_dwell_postburn_ms": (
            None if rest_index is None else float(rest_index * bin_ms)
        ),
        "whole_sheet_area_threshold": float(area_threshold),
        "whole_sheet_plateau_dwell_ms": 500.0,
        "first_whole_sheet_plateau_postburn_ms": (
            None if plateau_index is None else float(plateau_index * bin_ms)
        ),
        "A_active": _finite_summary(area),
        "r_all_hz": _finite_summary(rate_all),
        "r_core_hz": _finite_summary(rate_core),
    }
    arrays = {
        "carrier_gate_d_rest": np.asarray(distance, np.float32),
        "carrier_gate_A_active": np.asarray(area, np.float32),
        "carrier_gate_r_all_hz": np.asarray(rate_all, np.float32),
        "carrier_gate_r_core_hz": np.asarray(rate_core, np.float32),
        "carrier_gate_bin_ms": np.asarray(bin_ms),
    }
    return gates, evidence, arrays


def _gain_block_metrics(e, ctx, block_ms=500.0):
    steps = int(round(float(block_ms) / ctx["dt"]))
    n_blocks = e.shape[0] // steps
    if n_blocks < 2:
        raise RuntimeError("gain continuation lacks two complete 500 ms blocks")
    block = e[:n_blocks * steps].reshape(
        n_blocks, steps, e.shape[1]
    )
    duration_s = float(block_ms) * 1e-3
    core = np.asarray(ctx["core"], bool)
    core_rate = block[:, :, core].sum(axis=(1, 2)) / core.sum() / duration_s
    all_rate = block.sum(axis=(1, 2)) / e.shape[1] / duration_s
    spatial = PCM.activity_and_spatial_entropy(
        e, ctx["dt"], bin_ms=5.0, positions=ctx["S"]["posE"], L=ctx["S"]["L"]
    )
    firing, _rates = PCM.firing_and_ceiling_metrics(
        e,
        ctx["dt"],
        ctx["S"]["p"].tau_ref_E,
        core_mask=ctx["core"],
    )
    return {
        "block_ms": float(block_ms),
        "core_rate_hz": np.asarray(core_rate, float).tolist(),
        "all_E_rate_hz": np.asarray(all_rate, float).tolist(),
        "active_grid_fraction_5ms": spatial["active_grid_fraction"],
        "rho80_active_core_median": firing["rho80_active_core_median"],
    }


def _runtime_provenance(args, ctx, locks):
    sources = tuple(sorted(
        locks["manifest"]["provenance"]["producer_file_sha256"]
    ))
    coordinator = PRES.coordinator_identity_from_env()
    return {
        "manifest_path": locks["manifest_path"],
        "manifest_sha256": locks["manifest_sha256"],
        "manifest_file_sha256": locks["manifest_file_sha256"],
        "runtime_git_sha": ctx["runtime_git_sha"],
        "runtime_started_at": ctx["runtime_started_at"],
        "self_pid_at_publish": os.getpid(),
        "self_vm_swap_kb_at_publish": PRES.process_swap_kb(os.getpid()),
        "self_vm_swap_sample_semantics": (
            "pre-publish child self snapshot; not a kernel peak"
        ),
        **coordinator,
        "command": [sys.executable, *sys.argv],
        "cwd": os.path.abspath(os.getcwd()),
        "producer_sha256": {
            path: _sha(os.path.join(ROOT, path)) for path in sources
        },
        "panel_sha256": locks["panel_row"]["panel_sha256"],
        "state_file_sha256": locks["state_ref"]["file_sha256"],
        "noise_bank_sha": locks["bank"]["bank_sha"],
        "homologous_anchor_validated": bool(args.resolution == "dt2"),
        "homologous_native_state_hash": (
            locks["native_homolog_state_ref"]["state_hash"]
            if args.resolution == "dt2" else None
        ),
        "homologous_parent_config_sha": (
            locks["native_seed_row"]["canonical_config_sha"]
            if args.resolution == "dt2" else None
        ),
    }


def _c0_cell_root(args, kind):
    if kind not in {"identity", "gain"}:
        raise ValueError(f"unsupported C0 cell kind: {kind}")
    return os.path.join(
        OUT,
        "smoke" if args.smoke else "parts",
        f"c0_{kind}",
        args.resolution,
        f"seed{args.seed}",
        args.state_tag,
        args.replicate,
    )


def _hierarchical_observables_complete(arrays, *, min_blocks=2):
    """Fail closed on the compact hierarchical artifact shape."""
    if not isinstance(arrays, dict):
        return False
    if any(key not in arrays for key in HIERARCHICAL_ARRAY_FIELDS):
        return False
    try:
        rho = np.asarray(arrays["rho80_active_core_by_block_window"])
        cv2 = np.asarray(arrays["block_isi_cv2_by_panel_neuron"])
        ref_numerator = np.asarray(
            arrays["block_refractory_isi_numerator_by_stratum"]
        )
        ref_denominator = np.asarray(
            arrays["block_refractory_isi_denominator_by_stratum"]
        )
        pair = np.asarray(arrays["pair_corr_by_block_and_pair"])
        null = np.asarray(
            arrays["pair_null_median_by_block_and_draw"]
        )
        area = np.asarray(
            arrays["active_area_fraction_by_block_window"]
        )
        names = tuple(
            str(value) for value in np.asarray(
                arrays["pair_null_stratum_names"]
            ).ravel()
        )
        refractory_names = tuple(
            str(value) for value in np.asarray(
                arrays["refractory_isi_stratum_names"]
            ).ravel()
        )
    except (KeyError, TypeError, ValueError):
        return False
    n_block = rho.shape[0] if rho.ndim == 2 else 0
    return bool(
        n_block >= int(min_blocks)
        and rho.shape[1:] == (6,)
        and cv2.ndim == 2
        and ref_numerator.shape == (n_block, 2)
        and ref_denominator.shape == ref_numerator.shape
        and np.issubdtype(ref_numerator.dtype, np.integer)
        and np.issubdtype(ref_denominator.dtype, np.integer)
        and np.all(ref_numerator >= 0)
        and np.all(ref_numerator <= ref_denominator)
        and pair.ndim == 2
        and area.ndim == 2
        and area.shape[1] == 20
        and all(value.shape[0] == n_block for value in (
            cv2, ref_numerator, ref_denominator, pair, null, area
        ))
        and null.shape == (n_block, 3, 100)
        and names == PCM.PAIR_NULL_STRATUM_NAMES
        and refractory_names == PCM.REFRACTORY_ISI_STRATUM_NAMES
    )


def _smoke_observables_complete(
    arrays,
    *,
    c1=False,
    analysis_ids,
    pairwise_ids,
    thresholds,
):
    """Require locked hierarchical, current, and spatial smoke schemas."""
    fine = (
        "source_rate_hz", "rest_mask", "active_area_fraction",
        "kymograph", "axis_positions",
    ) if c1 else (
        "global_E_rate_hz", "global_I_rate_hz", "fine_bin_ms",
    )
    required = (
        "hierarchical_schema", "E_rate_grid", "I_rate_grid",
        "spatial_grid_n_occupied_E",
        "spatial_area_denominator", "raw_sample_time_ms",
        "effective_sample_time_ms", "analysis_panel_E_ids",
        "pairwise_panel_E_ids", "block_ms", "ceiling_window_ms",
        "ceiling_stride_ms", "active_area_window_ms", "pairwise_bin_ms",
        "pairwise_null_draws", "spatial_grid_n",
        "raw_raw_ampa_core_mean_mV", "raw_raw_gaba_core_mean_mV",
        "effective_effective_excitation_core_mean_mV",
        "effective_effective_inhibition_z_core_mean_mV",
        "effective_adaptation_m_core_mean_mV",
        "effective_effective_outward_total_core_mean_mV",
        *fine,
    )
    if (
        not _hierarchical_observables_complete(arrays, min_blocks=2)
        or any(key not in arrays for key in required)
    ):
        return False
    try:
        expected_scalars = {
            "block_ms": float(thresholds["time_block_ms"]),
            "ceiling_window_ms": float(
                thresholds["sliding_rate_window_ms"]
            ),
            "ceiling_stride_ms": float(
                thresholds["sliding_rate_window_stride_ms"]
            ),
            "active_area_window_ms": float(
                thresholds["active_area_window_ms"]
            ),
            "pairwise_bin_ms": float(thresholds["pairwise_bin_ms"]),
            "pairwise_null_draws": float(
                thresholds["pairwise_shift_null_draws"]
            ),
            "spatial_grid_n": float(thresholds["spatial_grid_n"]),
        }
        if any(
            np.asarray(arrays[key]).size != 1
            or not np.isclose(
                float(np.asarray(arrays[key]).item()), expected
            )
            for key, expected in expected_scalars.items()
        ):
            return False
        if str(np.asarray(
            arrays["hierarchical_schema"]
        ).reshape(()).item()) != PCM.HIERARCHICAL_STATS_VERSION:
            return False
        if not np.array_equal(
            np.asarray(arrays["analysis_panel_E_ids"], int),
            np.asarray(analysis_ids, int),
        ) or not np.array_equal(
            np.asarray(arrays["pairwise_panel_E_ids"], int),
            np.asarray(pairwise_ids, int),
        ):
            return False
        if np.asarray(arrays["spatial_area_denominator"]).item() != (
            "anatomy_occupied_E_grid_bins"
        ):
            return False

        raw_time = np.asarray(arrays["raw_sample_time_ms"], float)
        effective_time = np.asarray(
            arrays["effective_sample_time_ms"], float
        )
        if (
            raw_time.ndim != 1
            or raw_time.size == 0
            or effective_time.shape != raw_time.shape
            or not np.isfinite(raw_time).all()
            or not np.isfinite(effective_time).all()
            or not np.allclose(raw_time, effective_time)
        ):
            return False
        current_keys = (
            "raw_raw_ampa_core_mean_mV",
            "raw_raw_gaba_core_mean_mV",
            "effective_effective_excitation_core_mean_mV",
            "effective_effective_inhibition_z_core_mean_mV",
            "effective_adaptation_m_core_mean_mV",
            "effective_effective_outward_total_core_mean_mV",
        )
        currents = {
            key: np.asarray(arrays[key], float) for key in current_keys
        }
        if any(
            value.shape != raw_time.shape or not np.isfinite(value).all()
            for value in currents.values()
        ):
            return False
        outward_keys = (
            "raw_raw_gaba_core_mean_mV",
            "effective_effective_inhibition_z_core_mean_mV",
            "effective_adaptation_m_core_mean_mV",
            "effective_effective_outward_total_core_mean_mV",
        )
        if any(np.any(currents[key] < -1e-6) for key in outward_keys):
            return False

        e_grid = np.asarray(arrays["E_rate_grid"])
        i_grid = np.asarray(arrays["I_rate_grid"])
        grid_n = int(thresholds["spatial_grid_n"])
        if (
            e_grid.ndim != 3
            or e_grid.shape != i_grid.shape
            or e_grid.shape[0] == 0
            or e_grid.shape[1:] != (grid_n, grid_n)
            or not np.isfinite(e_grid).all()
            or not np.isfinite(i_grid).all()
        ):
            return False
        if c1:
            n_time = e_grid.shape[0]
            source = np.asarray(arrays["source_rate_hz"])
            rest = np.asarray(arrays["rest_mask"])
            area = np.asarray(arrays["active_area_fraction"])
            kymo = np.asarray(arrays["kymograph"])
            axis = np.asarray(arrays["axis_positions"])
            return bool(
                source.shape == (n_time,)
                and rest.shape == (n_time,)
                and rest.dtype == np.dtype(bool)
                and area.shape == (n_time,)
                and kymo.ndim == 2
                and kymo.shape[0] == n_time
                and axis.shape == (kymo.shape[1],)
                and np.isfinite(source).all()
                and np.isfinite(area).all()
                and np.isfinite(kymo).all()
                and np.isfinite(axis).all()
                and np.asarray(arrays["bin_ms"]).size == 1
                and np.isclose(
                    float(np.asarray(arrays["bin_ms"]).item()),
                    FINE_BIN_MS,
                )
            )
        global_e = np.asarray(arrays["global_E_rate_hz"])
        global_i = np.asarray(arrays["global_I_rate_hz"])
        return bool(
            global_e.shape == (e_grid.shape[0],)
            and global_i.shape == global_e.shape
            and np.isfinite(global_e).all()
            and np.isfinite(global_i).all()
            and np.asarray(arrays["fine_bin_ms"]).size == 1
            and np.isclose(
                float(np.asarray(arrays["fine_bin_ms"]).item()),
                FINE_BIN_MS,
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def _build_hierarchical_observables(
    e, ctx, locks, manifest, pairwise_seed, *, technical_complete
):
    """Build identical hierarchical units for smoke and production runs."""
    if not technical_complete:
        return None
    return PCM.phasec_bootstrap_units(
        e,
        ctx["dt"],
        ctx["S"]["p"].tau_ref_E,
        core_mask=ctx["core"],
        analysis_panel_ids=locks["analysis_ids"],
        pairwise_panel_ids=locks["pairwise_ids"],
        positions=ctx["S"]["posE"],
        L=ctx["S"]["L"],
        block_ms=float(manifest["thresholds"]["time_block_ms"]),
        pairwise_bin_ms=float(
            manifest["thresholds"]["pairwise_bin_ms"]
        ),
        pairwise_n_null=int(
            manifest["thresholds"]["pairwise_shift_null_draws"]
        ),
        ceiling_window_ms=float(
            manifest["thresholds"]["sliding_rate_window_ms"]
        ),
        ceiling_stride_ms=float(
            manifest["thresholds"]["sliding_rate_window_stride_ms"]
        ),
        active_area_window_ms=float(
            manifest["thresholds"]["active_area_window_ms"]
        ),
        spatial_active_floor_hz=float(
            manifest["thresholds"]["local_active_floor_hz"]
        ),
        n_grid=int(manifest["thresholds"]["spatial_grid_n"]),
        pairwise_null_seed=pairwise_seed,
    )


def run_identity(args):
    ctx = R.build_context(args.seed, resolution=args.resolution)
    locks = _locked_inputs(args, ctx)
    manifest = locks["manifest"]
    root = _c0_cell_root(args, "identity")
    path = os.path.join(root, "identity.json")
    if not args.smoke and _reuse_or_fail(path, {
        "schema": "zm_phasec_identity_cell_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "seed": int(args.seed),
        "resolution": args.resolution,
        "state_tag": args.state_tag,
        "replicate": args.replicate,
    }):
        return
    stride = max(1, int(round(CURRENT_STRIDE_MS / ctx["dt"])))
    recorder = PCO.PhaseCCurrentRecorder(ctx["rec"], ctx["core"], stride_steps=stride)
    burn_ms = 250.0 if args.smoke else IDENTITY_BURN_MS
    # One second retains two complete 500-ms blocks, satisfying the existing
    # hierarchical producer without weakening its production minimum.
    measure_ms = 1000.0 if args.smoke else IDENTITY_MEASURE_MS
    if not args.smoke:
        if (
            burn_ms != manifest["c0"]["protocols"]["identity"]["burn_in_ms"]
            or measure_ms != manifest["c0"]["duration_ms"]
            or measure_ms != manifest["c0"]["protocols"]["identity"]["measure_ms"]
        ):
            raise RuntimeError("identity cell does not match immutable Phase-C manifest")
    total_ms = burn_ms + measure_ms
    t0 = time.time()
    result, ckpt, effective_observer = _run(
        ctx, locks["state"], locks["bank"], total_ms,
        dump_i_spikes=True, recorder=recorder, return_final_state=True,
    )
    stop = result.get("runaway_early_stop_ms")
    burn_step = int(round(burn_ms / ctx["dt"]))
    technical_complete = (
        result["E_spk_bool"].shape[0] >= int(round(total_ms / ctx["dt"]))
        and result.get("I_spk_bool") is not None
        and ckpt.final_state is not None
        and not ckpt.final_truncated
    )
    e = result["E_spk_bool"][min(burn_step, result["E_spk_bool"].shape[0]):]
    i = (
        result["I_spk_bool"][min(burn_step, result["I_spk_bool"].shape[0]):]
        if result.get("I_spk_bool") is not None else None
    )
    spike_metrics = None
    fine_metrics = None
    fields = None
    bootstrap_units = None
    pairwise_seed = int(
        args.seed * 1_000_000
        + int(locks["state_ref"]["t_step"]) % 100_000
        + NB.PAIRED_REPLICATES.index(args.replicate)
    )
    if e.shape[0] >= int(round(250.0 / ctx["dt"])):
        spike_metrics = PCM.phasec_metrics_from_raster(
            e, ctx["dt"], ctx["S"]["p"].tau_ref_E,
            core_mask=ctx["core"], positions=ctx["S"]["posE"],
            L=ctx["S"]["L"],
            pairwise_null_seed=pairwise_seed,
            analysis_panel_ids=locks["analysis_ids"],
            pairwise_panel_ids=locks["pairwise_ids"],
        )
    bootstrap_units = _build_hierarchical_observables(
        e, ctx, locks, manifest, pairwise_seed,
        technical_complete=technical_complete,
    )
    if i is not None and e.shape[0] >= int(round(512.0 / ctx["dt"])):
        fields = SR.bin_spikes_to_grid(
            e, i, ctx["S"]["posE"], ctx["S"]["posI"],
            L=ctx["S"]["L"], dt_ms=ctx["dt"], bin_ms=FINE_BIN_MS,
            n_grid=int(manifest["thresholds"]["spatial_grid_n"]),
        )
        fine_metrics = SR.characterize_source_rhythm(
            fields["E_rate_grid"], fields["I_rate_grid"], bin_ms=FINE_BIN_MS
        )

    raw_post = _postburn_trace(recorder.traces(dt_ms=ctx["dt"]), burn_ms)
    effective_post = _postburn_trace(
        effective_observer.traces(dt_ms=ctx["dt"]), burn_ms
    )
    lag = PCO.raw_synaptic_lag(raw_post, region="core")
    gates, gate_evidence, gate_arrays = _carrier_gate_evidence(
        ctx,
        result,
        burn_ms,
        locks["anchor"]["locks"],
        manifest["thresholds"]["whole_sheet_plateau_area_frac"],
    )

    lfp = result.get("lfp_trace")
    lfp_post = (
        np.asarray(lfp)[min(burn_step, len(lfp)):]
        if lfp is not None else np.zeros((0, len(ctx["contacts"])))
    )
    spectral = _spectral_summary(
        lfp_post, 1000.0 / ctx["dt"], ctx["contacts"]
    )

    arrays = {
        "hierarchical_schema": np.asarray(
            PCM.HIERARCHICAL_STATS_VERSION
        ),
        "panel_sha256": np.asarray(locks["panel_row"]["panel_sha256"]),
        "raw_sample_time_ms": np.asarray(raw_post["sample_time_ms"], np.float32),
        "lfp_raw_synaptic_proxy": np.asarray(lfp_post, np.float32),
        "lfp_fs_hz": np.asarray(1000.0 / ctx["dt"]),
    }
    for prefix, traces in (("raw", raw_post), ("effective", effective_post)):
        for key, value in traces.items():
            if isinstance(value, np.ndarray):
                arrays[f"{prefix}_{key}"] = value
    arrays.update(gate_arrays)
    if bootstrap_units is not None:
        arrays.update(bootstrap_units)
    if fields is not None:
        arrays.update(
            E_rate_grid=np.asarray(fields["E_rate_grid"], np.float32),
            I_rate_grid=np.asarray(fields["I_rate_grid"], np.float32),
            global_E_rate_hz=np.asarray(fields["global_E_rate_hz"], np.float32),
            global_I_rate_hz=np.asarray(fields["global_I_rate_hz"], np.float32),
            nE_per_cell=np.asarray(fields["nE_per_cell"], np.int32),
            nI_per_cell=np.asarray(fields["nI_per_cell"], np.int32),
            fine_bin_ms=np.asarray(FINE_BIN_MS),
        )
    arrays_path, arrays_sha = _publish_content_addressed_npz(
        root, "observables", arrays
    )
    if stop is not None:
        status = "scientific_failure"
        scientific_reason = "runaway"
        technical_reason = None
    elif (
        not technical_complete
        or not _hierarchical_observables_complete(arrays, min_blocks=2)
        or (
            args.smoke
            and not _smoke_observables_complete(
                arrays,
                analysis_ids=locks["analysis_ids"],
                pairwise_ids=locks["pairwise_ids"],
                thresholds=manifest["thresholds"],
            )
        )
    ):
        status = "technical_invalid"
        scientific_reason = None
        technical_reason = "truncated_or_missing_observable"
    else:
        status = "complete"
        scientific_reason = None
        technical_reason = None
    payload = {
        "schema": "zm_phasec_identity_cell_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": locks["manifest_file_sha256"],
        "panel_sha256": locks["panel_row"]["panel_sha256"],
        "status": status,
        "scientific_end_reason": scientific_reason,
        "technical_end_reason": technical_reason,
        "seed": int(args.seed),
        "resolution": args.resolution,
        "state_tag": args.state_tag,
        "replicate": args.replicate,
        "config_sha": ctx["cfg_sha"],
        "producer_git_sha": ctx["runtime_git_sha"],
        "engine_sha": ctx["cfg_locked"]["engine_sha256"]["src/snn_engine/kick_probe.py"],
        "state_hash": locks["state_manifest"]["state_hash"],
        "state_path": locks["state_ref"]["path"],
        "state_file_sha256": locks["state_ref"]["file_sha256"],
        "noise_bank_sha": locks["bank"]["bank_sha"],
        "homologous_anchor_validated": bool(args.resolution == "dt2"),
        "homologous_native_state_hash": (
            locks["native_homolog_state_ref"]["state_hash"]
            if args.resolution == "dt2" else None
        ),
        "homologous_parent_config_sha": (
            locks["native_seed_row"]["canonical_config_sha"]
            if args.resolution == "dt2" else None
        ),
        "dt_ms": ctx["dt"],
        "burn_in_ms": burn_ms,
        "measure_ms": measure_ms,
        "evidence_value": "none_smoke" if args.smoke else "production",
        "runaway_early_stop_ms": stop,
        "carrier_gates": gates,
        "carrier_gate_evidence": gate_evidence,
        "spike_metrics": spike_metrics,
        "fine_source_metrics": fine_metrics,
        "raw_synaptic_lag": lag,
        "raw_synaptic_vseeg_proxy_spectral": spectral,
        "raw_synaptic_trace_summary": _trace_stats(raw_post),
        "effective_membrane_drive_summary": _trace_stats(effective_post),
        "effective_snapshot_initial": _effective_summary(locks["state"], ctx),
        "effective_snapshot_final": _effective_summary(ckpt.final_state, ctx),
        "threshold_margin_initial": _margin_summary(locks["state"], ctx),
        "threshold_margin_final": _margin_summary(ckpt.final_state, ctx),
        "observables_path": os.path.relpath(arrays_path, ROOT),
        "observables_sha256": arrays_sha,
        "runtime_provenance": _runtime_provenance(args, ctx, locks),
        "wall_s": round(time.time() - t0, 2),
        "peak_rss_gb": _rss_gb(),
        "claim_boundary": (
            "source-space tonic identity only; not observation-matched ictal "
            "activity, entry, offset, recovery, or lifecycle"
        ),
    }
    _publish_json_once(path, payload)
    print(
        f"[phasec identity] seed={args.seed} {args.state_tag} {args.replicate} "
        f"status={payload['status']} wall={payload['wall_s']}s rss={payload['peak_rss_gb']}GB "
        f"-> {path}",
        flush=True,
    )


def run_gain(args):
    is_zero = args.sign == 0 and args.delta_mV == 0.0
    if (args.sign == 0) != (args.delta_mV == 0.0):
        raise SystemExit("gain zero arm requires both --sign 0 and --delta-mV 0")
    if not is_zero and (
        args.delta_mV not in GAIN_DELTAS_MV or args.sign not in (-1, 1)
    ):
        raise SystemExit("gain is locked to 0 or delta 0.05/0.10 mV and sign +/-1")
    ctx = R.build_context(args.seed, resolution=args.resolution)
    locks = _locked_inputs(args, ctx)
    manifest = locks["manifest"]
    label = (
        "d0_zero" if is_zero
        else f"d{args.delta_mV:g}_{'plus' if args.sign > 0 else 'minus'}"
    )
    root = os.path.join(_c0_cell_root(args, "gain"), label)
    path = os.path.join(root, "gain.json")
    if not args.smoke and _reuse_or_fail(path, {
        "schema": "zm_phasec_gain_cell_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "seed": int(args.seed),
        "resolution": args.resolution,
        "state_tag": args.state_tag,
        "replicate": args.replicate,
        "delta_mV": float(args.delta_mV),
        "sign": int(args.sign),
    }):
        return
    burn_ms = 250.0 if args.smoke else GAIN_BURN_MS
    measure_ms = 250.0 if args.smoke else GAIN_MEASURE_MS
    if not args.smoke:
        offset = float(args.sign * args.delta_mV)
        if (
            offset not in manifest["c0"]["threshold_perturbation"]["values"]
            or burn_ms != manifest["c0"]["protocols"]["gain"]["burn_in_ms"]
            or measure_ms != manifest["c0"]["protocols"]["gain"]["measure_ms"]
        ):
            raise RuntimeError("gain cell does not match immutable Phase-C manifest")
    start_ms = int(np.asarray(locks["state"]["t"])) * ctx["dt"]
    perturb = None if is_zero else {
        "kind": "inhibitory_pulse",
        "t0": start_ms + burn_ms,
        "t1": start_ms + burn_ms + measure_ms,
        "val": float(args.sign * args.delta_mV),
        "target_mask": np.concatenate([
            np.asarray(ctx["core"], bool),
            np.zeros(ctx["S"]["NI"], dtype=bool),
        ]),
    }
    t0 = time.time()
    result, _ckpt, _effective_observer = _run(
        ctx, locks["state"], locks["bank"],
        burn_ms + measure_ms,
        dump_i_spikes=False, recorder=None, perturb=perturb,
        return_final_state=False,
    )
    burn_step = int(round(burn_ms / ctx["dt"]))
    e = result["E_spk_bool"][min(burn_step, result["E_spk_bool"].shape[0]):]
    duration_s = e.shape[0] * ctx["dt"] * 1e-3
    stop = result.get("runaway_early_stop_ms")
    full_measure = duration_s >= 0.99 * measure_ms * 1e-3
    block_metrics = None
    if full_measure and not args.smoke:
        block_metrics = _gain_block_metrics(
            e, ctx, block_ms=float(manifest["thresholds"]["time_block_ms"])
        )
    elif duration_s > 0:
        core_rate = float(
            e[:, ctx["core"]].sum() / ctx["core"].sum() / duration_s
        )
        block_metrics = {
            "block_ms": float(manifest["thresholds"]["time_block_ms"]),
            "core_rate_hz": [core_rate],
            "all_E_rate_hz": [],
            "active_grid_fraction_5ms": None,
            "rho80_active_core_median": None,
        }
    gates, gate_evidence, _gate_arrays = _carrier_gate_evidence(
        ctx,
        result,
        burn_ms,
        locks["anchor"]["locks"],
        manifest["thresholds"]["whole_sheet_plateau_area_frac"],
    )
    if stop is not None:
        status = "scientific_failure"
        scientific_reason = "runaway"
        technical_reason = None
    elif not full_measure or block_metrics is None:
        status = "technical_invalid"
        scientific_reason = None
        technical_reason = "short_gain_continuation"
    else:
        status = "complete"
        scientific_reason = None
        technical_reason = None
    core_rate_500ms = block_metrics["core_rate_hz"]
    core_rate = float(np.mean(core_rate_500ms)) if core_rate_500ms else None
    gain_plateau_gate_pass = bool(
        not gates["runaway"] and not gates["whole_sheet_plateau"]
    )
    payload = {
        "schema": "zm_phasec_gain_cell_v1",
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_file_sha256": locks["manifest_file_sha256"],
        "panel_sha256": locks["panel_row"]["panel_sha256"],
        "status": status,
        "scientific_end_reason": scientific_reason,
        "technical_end_reason": technical_reason,
        "seed": int(args.seed),
        "resolution": args.resolution,
        "state_tag": args.state_tag,
        "replicate": args.replicate,
        "delta_mV": float(args.delta_mV),
        "sign": int(args.sign),
        "threshold_offset_mV": float(args.sign * args.delta_mV),
        "core_rate_hz": core_rate,
        "core_rate_500ms_hz": core_rate_500ms,
        "gain_block_metrics": block_metrics,
        "gain_plateau_gate_pass": gain_plateau_gate_pass,
        "carrier_gates": gates,
        "carrier_gate_evidence": gate_evidence,
        "config_sha": ctx["cfg_sha"],
        "producer_git_sha": ctx["runtime_git_sha"],
        "engine_sha": ctx["cfg_locked"]["engine_sha256"]["src/snn_engine/kick_probe.py"],
        "state_hash": locks["state_manifest"]["state_hash"],
        "state_path": locks["state_ref"]["path"],
        "state_file_sha256": locks["state_ref"]["file_sha256"],
        "noise_bank_sha": locks["bank"]["bank_sha"],
        "homologous_anchor_validated": bool(args.resolution == "dt2"),
        "homologous_native_state_hash": (
            locks["native_homolog_state_ref"]["state_hash"]
            if args.resolution == "dt2" else None
        ),
        "homologous_parent_config_sha": (
            locks["native_seed_row"]["canonical_config_sha"]
            if args.resolution == "dt2" else None
        ),
        "dt_ms": ctx["dt"],
        "burn_in_ms": burn_ms,
        "measure_ms": measure_ms,
        "evidence_value": "none_smoke" if args.smoke else "production",
        "runaway_early_stop_ms": stop,
        "runtime_provenance": _runtime_provenance(args, ctx, locks),
        "wall_s": round(time.time() - t0, 2),
        "peak_rss_gb": _rss_gb(),
        "claim_boundary": "paired source-core threshold susceptibility; diagnostic only",
    }
    _publish_json_once(path, payload)
    print(
        f"[phasec gain] seed={args.seed} {args.state_tag} {args.replicate} "
        f"{label} rate={core_rate} status={payload['status']} "
        f"wall={payload['wall_s']}s rss={payload['peak_rss_gb']}GB -> {path}",
        flush=True,
    )


def _c1_base_relative_path(
    resolution, seed, tier, cell_id, phase, noise, *, smoke=False
):
    return os.path.join(
        "results", "topic4_sef_hfo", "zm_phase_c_tonic_identity",
        "smoke" if smoke else "parts",
        "c1_base", resolution, f"seed{int(seed)}", tier, cell_id,
        phase, noise, "phenotype.json",
    )


def _c1_gain_relative_path(
    resolution, seed, tier, cell_id, phase, noise, delta_mV, *, smoke=False
):
    delta = float(delta_mV)
    label = (
        "d0_zero" if delta == 0.0
        else f"d{abs(delta):g}_{'plus' if delta > 0 else 'minus'}"
    )
    return os.path.join(
        "results", "topic4_sef_hfo", "zm_phase_c_tonic_identity",
        "smoke" if smoke else "parts",
        "c1_conditional_gain", resolution, f"seed{int(seed)}",
        tier, cell_id, phase, noise, label, "gain.json",
    )


def _fine_c1_observables(
    e, i, ctx, gate_arrays, anchor_locks, *,
    spatial_grid_n, spatial_active_floor_hz,
):
    fields = SR.bin_spikes_to_grid(
        e, i, ctx["S"]["posE"], ctx["S"]["posI"],
        L=ctx["S"]["L"], dt_ms=ctx["dt"], bin_ms=FINE_BIN_MS,
        n_grid=int(spatial_grid_n),
    )
    source = R.MC.source_metrics(
        e, ctx["core"], ctx["S"]["posE"], ctx["S"]["L"], ctx["dt"],
        bin_ms=FINE_BIN_MS, axis_coord=ctx["axis"], n_axial=24,
    )
    n = min(
        int(fields["E_rate_grid"].shape[0]),
        int(source["n_bins"]),
    )
    if n < 16:
        raise RuntimeError("C1 fine observables contain fewer than 16 bins")
    distance = np.asarray(gate_arrays["carrier_gate_d_rest"], float)
    if not distance.size:
        rest_mask = np.zeros(n, bool)
    else:
        fine_time = np.arange(n, dtype=float) * FINE_BIN_MS
        index = np.clip(
            np.floor(fine_time / float(R.MC.BIN_MS)).astype(int),
            0, distance.size - 1,
        )
        rest_mask = distance[index] < float(anchor_locks["d_rest_thresh"])
    axis_edges = np.linspace(
        float(np.min(ctx["axis"])),
        float(np.max(ctx["axis"])) + 1e-9,
        25,
    )
    axis_positions = 0.5 * (axis_edges[:-1] + axis_edges[1:])
    kymograph = np.asarray(source["kymo_axial"], float).T[:n]
    if kymograph.shape != (n, 24):
        raise RuntimeError("C1 axial kymograph shape mismatch")
    e_grid = np.asarray(fields["E_rate_grid"][:n], float)
    occupied_e_grid = np.asarray(fields["nE_per_cell"], int) > 0
    if occupied_e_grid.shape != e_grid.shape[1:] or not np.any(occupied_e_grid):
        raise RuntimeError("C1 E-rate grid lacks anatomy-occupied bins")
    active_area_fraction = PCM.spatial_active_area_from_rate_grid(
        e_grid,
        fields["nE_per_cell"],
        active_floor_hz=spatial_active_floor_hz,
    )
    return {
        "fields": fields,
        "E_rate_grid": np.asarray(e_grid, np.float32),
        "I_rate_grid": np.asarray(fields["I_rate_grid"][:n], np.float32),
        "source_rate_hz": np.asarray(source["r_core"][:n], np.float32),
        "rest_mask": np.asarray(rest_mask, bool),
        "active_area_fraction": np.asarray(
            active_area_fraction, np.float32
        ),
        "spatial_grid_n_occupied_E": np.asarray(
            int(np.sum(occupied_e_grid))
        ),
        "spatial_grid_all_E_bins_occupied": np.asarray(
            bool(np.all(occupied_e_grid))
        ),
        "spatial_active_floor_hz": np.asarray(
            float(spatial_active_floor_hz)
        ),
        "spatial_area_denominator": np.asarray(
            "anatomy_occupied_E_grid_bins"
        ),
        "kymograph": np.asarray(kymograph, np.float32),
        "axis_positions": np.asarray(axis_positions, np.float32),
    }


def _c1_runtime_provenance(args, ctx, locks):
    out = _runtime_provenance(args, ctx, locks)
    out.update({
        "coordinate_manifest_sha256": locks[
            "coordinate_manifest"
        ]["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": locks[
            "coordinate_manifest"
        ]["semantic_sha256"],
        "coordinate_manifest_file_sha256": locks[
            "coordinate_manifest_file_sha256"
        ],
        "coordinate_npz_file_sha256": locks[
            "coordinate_npz_file_sha256"
        ],
        "coordinate_npz_semantic_sha256": locks[
            "coordinate_npz_semantic_sha256"
        ],
        "coordinate_producer_sha256": locks[
            "coordinate_manifest"
        ]["producer_file_sha256"],
    })
    if "trigger_manifest" in locks:
        out.update({
            "trigger_manifest_sha256": locks[
                "trigger_manifest"
            ]["manifest_sha256"],
            "trigger_manifest_file_sha256": locks[
                "trigger_manifest_file_sha256"
            ],
            "trigger_producer_sha256": locks[
                "trigger_manifest"
            ]["producer_file_sha256"],
        })
    if "dt2_confirmation_manifest" in locks:
        out.update({
            "dt2_confirmation_manifest_sha256": locks[
                "dt2_confirmation_manifest"
            ]["manifest_sha256"],
            "dt2_confirmation_manifest_file_sha256": locks[
                "dt2_confirmation_manifest_file_sha256"
            ],
        })
    return out


def run_c1_base(args):
    ctx = R.build_context(args.seed, resolution=args.resolution)
    locks = _c1_locked_inputs(args, ctx)
    manifest = locks["manifest"]
    coordinate = locks["coordinate_cell"]
    relative = _c1_base_relative_path(
        args.resolution, args.seed, args.tier, args.cell_id,
        args.phase, args.replicate, smoke=args.smoke,
    )
    path = os.path.join(ROOT, relative)
    root = os.path.dirname(path)
    expected = {
        "schema": C1_BASE_PART_SCHEMA,
        "phasec_manifest_sha256": manifest["manifest_sha256"],
        "coordinate_manifest_sha256": locks[
            "coordinate_manifest"
        ]["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": locks[
            "coordinate_manifest"
        ]["semantic_sha256"],
        "coordinate_npz_file_sha256": locks[
            "coordinate_npz_file_sha256"
        ],
        "coordinate_npz_semantic_sha256": locks[
            "coordinate_npz_semantic_sha256"
        ],
        "seed": int(args.seed),
        "cell_id": args.cell_id,
        "tier": args.tier,
        "phase": args.phase,
        "noise": args.replicate,
        "resolution": args.resolution,
        "slow_state_sha256": coordinate["state_sha256"],
        "config_sha": ctx["cfg_sha"],
        "fast_base_state_hash": locks["state_manifest"]["state_hash"],
        "state_file_sha256": locks["state_ref"]["file_sha256"],
        "noise_bank_sha": locks["bank"]["bank_sha"],
    }
    if "dt2_confirmation_manifest" in locks:
        expected.update({
            "dt2_confirmation_manifest_sha256": locks[
                "dt2_confirmation_manifest"
            ]["manifest_sha256"],
            "dt2_confirmation_manifest_file_sha256": locks[
                "dt2_confirmation_manifest_file_sha256"
            ],
        })
    if not args.smoke and _reuse_or_fail(path, expected):
        return
    burn_ms = 250.0 if args.smoke else IDENTITY_BURN_MS
    measure_ms = 1000.0 if args.smoke else IDENTITY_MEASURE_MS
    if not args.smoke and (
        burn_ms != manifest["c0"]["protocols"]["identity"]["burn_in_ms"]
        or measure_ms != manifest["c0"]["protocols"]["identity"]["measure_ms"]
    ):
        raise RuntimeError("C1 base protocol differs from immutable Phase-C lock")
    stride = max(1, int(round(CURRENT_STRIDE_MS / ctx["dt"])))
    recorder = PCO.PhaseCCurrentRecorder(
        ctx["rec"], ctx["core"], stride_steps=stride
    )
    total_ms = burn_ms + measure_ms
    t0 = time.time()
    result, ckpt, effective_observer = _run(
        ctx, locks["state"], locks["bank"], total_ms,
        dump_i_spikes=True, recorder=recorder, return_final_state=True,
    )
    stop = result.get("runaway_early_stop_ms")
    burn_step = int(round(burn_ms / ctx["dt"]))
    expected_steps = int(round(total_ms / ctx["dt"]))
    technical_complete = bool(
        result["E_spk_bool"].shape[0] >= expected_steps
        and result.get("I_spk_bool") is not None
        and ckpt.final_state is not None
        and not ckpt.final_truncated
    )
    e = result["E_spk_bool"][min(burn_step, len(result["E_spk_bool"])):]
    i = (
        None if result.get("I_spk_bool") is None
        else result["I_spk_bool"][min(burn_step, len(result["I_spk_bool"])):]
    )
    pairwise_seed = int(
        args.seed * 1_000_000
        + int(locks["state_ref"]["t_step"]) % 100_000
        + NB.PAIRED_REPLICATES.index(args.replicate)
    )
    spike_metrics = None
    bootstrap_units = None
    if e.shape[0] >= int(round(250.0 / ctx["dt"])):
        spike_metrics = PCM.phasec_metrics_from_raster(
            e, ctx["dt"], ctx["S"]["p"].tau_ref_E,
            core_mask=ctx["core"], positions=ctx["S"]["posE"],
            L=ctx["S"]["L"], pairwise_null_seed=pairwise_seed,
            analysis_panel_ids=locks["analysis_ids"],
            pairwise_panel_ids=locks["pairwise_ids"],
        )
    bootstrap_units = _build_hierarchical_observables(
        e, ctx, locks, manifest, pairwise_seed,
        technical_complete=technical_complete,
    )
    raw_post = _postburn_trace(recorder.traces(dt_ms=ctx["dt"]), burn_ms)
    effective_post = _postburn_trace(
        effective_observer.traces(dt_ms=ctx["dt"]), burn_ms
    )
    gates, gate_evidence, gate_arrays = _carrier_gate_evidence(
        ctx, result, burn_ms, locks["anchor"]["locks"],
        manifest["thresholds"]["whole_sheet_plateau_area_frac"],
    )
    fine = None
    if i is not None and e.shape[0] >= int(round(512.0 / ctx["dt"])):
        fine = _fine_c1_observables(
            e, i, ctx, gate_arrays, locks["anchor"]["locks"],
            spatial_grid_n=int(
                manifest["thresholds"]["spatial_grid_n"]
            ),
            spatial_active_floor_hz=float(
                manifest["thresholds"]["local_active_floor_hz"]
            ),
        )
    if fine is None and stop is None:
        technical_complete = False

    arrays = {
        "phasec1_observables_schema": np.asarray(C1_OBSERVABLES_SCHEMA),
        "hierarchical_schema": np.asarray(
            PCM.HIERARCHICAL_STATS_VERSION
        ),
        "panel_sha256": np.asarray(locks["panel_row"]["panel_sha256"]),
        "bin_ms": np.asarray(FINE_BIN_MS),
        "readout_kernel_width_mm": np.asarray(
            float(ctx["S"]["p"].Rr)
        ),
        "raw_sample_time_ms": np.asarray(raw_post["sample_time_ms"], np.float32),
    }
    for prefix, traces in (("raw", raw_post), ("effective", effective_post)):
        for key, value in traces.items():
            if isinstance(value, np.ndarray):
                arrays[f"{prefix}_{key}"] = value
    arrays.update(gate_arrays)
    if bootstrap_units is not None:
        arrays.update(bootstrap_units)
    if fine is not None:
        for key in (
            "E_rate_grid", "I_rate_grid", "source_rate_hz", "rest_mask",
            "active_area_fraction", "spatial_grid_n_occupied_E",
            "spatial_grid_all_E_bins_occupied", "spatial_active_floor_hz",
            "spatial_area_denominator", "kymograph", "axis_positions",
        ):
            arrays[key] = fine[key]
    lfp = result.get("lfp_trace")
    lfp_post = (
        np.asarray(lfp)[min(burn_step, len(lfp)):]
        if lfp is not None else np.zeros((0, len(ctx["contacts"])))
    )
    arrays["lfp_raw_synaptic_proxy"] = np.asarray(lfp_post, np.float32)
    arrays["lfp_fs_hz"] = np.asarray(1000.0 / ctx["dt"])
    arrays_path, arrays_sha = _publish_content_addressed_npz(
        root, "observables", arrays
    )

    if gates["runaway"]:
        status, scientific_reason, technical_reason = (
            "scientific_failure", "runaway", None
        )
    elif gates["whole_sheet_plateau"]:
        status, scientific_reason, technical_reason = (
            "scientific_failure", "whole_sheet_plateau", None
        )
    elif gates["empirical_rest_dwell"]:
        status, scientific_reason, technical_reason = (
            "scientific_failure", "empirical_rest_dwell", None
        )
    elif (
        not technical_complete
        or fine is None
        or not _hierarchical_observables_complete(arrays, min_blocks=2)
        or (
            args.smoke
            and not _smoke_observables_complete(
                arrays,
                c1=True,
                analysis_ids=locks["analysis_ids"],
                pairwise_ids=locks["pairwise_ids"],
                thresholds=manifest["thresholds"],
            )
        )
    ):
        status, scientific_reason, technical_reason = (
            "technical_invalid", None, "truncated_or_missing_C1_observable"
        )
    else:
        status, scientific_reason, technical_reason = "complete", None, None
    saturation_fraction = (
        None if spike_metrics is None else spike_metrics[
            "firing"
        ]["rho80_active_core_median"]
    )
    payload = {
        **expected,
        "phasec_manifest_file_sha256": locks["manifest_file_sha256"],
        "coordinate_manifest_file_sha256": locks[
            "coordinate_manifest_file_sha256"
        ],
        "trajectory_id": coordinate["trajectory_id"],
        "path_index": int(coordinate["path_index"]),
        "path_direction": coordinate["path_direction"],
        "status": status,
        "scientific_end_reason": scientific_reason,
        "technical_end_reason": technical_reason,
        "config_sha": ctx["cfg_sha"],
        "noise_bank_sha": locks["bank"]["bank_sha"],
        "state_hash": CK.state_hash(locks["state"]),
        "fast_base_state_hash": locks["state_manifest"]["state_hash"],
        "state_file_sha256": locks["state_ref"]["file_sha256"],
        "coordinate_npz_file_sha256": locks[
            "coordinate_npz_file_sha256"
        ],
        "coordinate_npz_semantic_sha256": locks[
            "coordinate_npz_semantic_sha256"
        ],
        "dt_ms": ctx["dt"],
        "burn_in_ms": burn_ms,
        "measure_ms": measure_ms,
        "runaway_early_stop_ms": stop,
        "saturation_fraction": saturation_fraction,
        "carrier_gates": gates,
        "carrier_gate_evidence": gate_evidence,
        "spike_metrics": spike_metrics,
        "raw_synaptic_vseeg_proxy_spectral": _spectral_summary(
            lfp_post, 1000.0 / ctx["dt"], ctx["contacts"]
        ),
        "raw_synaptic_lag": PCO.raw_synaptic_lag(raw_post, region="core"),
        "raw_synaptic_trace_summary": _trace_stats(raw_post),
        "effective_membrane_drive_summary": _trace_stats(effective_post),
        "effective_snapshot_initial": _effective_summary(locks["state"], ctx),
        "effective_snapshot_final": _effective_summary(ckpt.final_state, ctx),
        "threshold_margin_initial": _margin_summary(locks["state"], ctx),
        "threshold_margin_final": _margin_summary(ckpt.final_state, ctx),
        "observables_path": os.path.relpath(arrays_path, ROOT),
        "observables_sha256": arrays_sha,
        "runtime_provenance": _c1_runtime_provenance(args, ctx, locks),
        "evidence_value": "none_smoke" if args.smoke else "production",
        "wall_s": round(time.time() - t0, 2),
        "peak_rss_gb": _rss_gb(),
        "claim_boundary": (
            "frozen source-space identity/maturation only; not entry, "
            "offset, recovery, observation match, actuator, or lifecycle"
        ),
    }
    _publish_json_once(path, payload)
    print(
        f"[phasec1 base] seed={args.seed} {args.tier}/{args.cell_id} "
        f"{args.phase}/{args.replicate} status={status} "
        f"wall={payload['wall_s']}s rss={payload['peak_rss_gb']}GB -> {path}",
        flush=True,
    )


def run_c1_gain(args):
    is_zero = args.sign == 0 and args.delta_mV == 0.0
    if (args.sign == 0) != (args.delta_mV == 0.0):
        raise SystemExit("conditional gain zero requires delta=0 and sign=0")
    if not is_zero and (
        args.delta_mV not in GAIN_DELTAS_MV or args.sign not in (-1, 1)
    ):
        raise SystemExit("conditional gain is locked to 0 or +/-0.05/0.10mV")
    ctx = R.build_context(args.seed, resolution=args.resolution)
    locks = _c1_locked_inputs(args, ctx, require_trigger=True)
    coordinate = locks["coordinate_cell"]
    delta = float(args.sign * args.delta_mV)
    relative = _c1_gain_relative_path(
        args.resolution, args.seed, args.tier, args.cell_id,
        args.phase, args.replicate, delta, smoke=args.smoke,
    )
    path = os.path.join(ROOT, relative)
    expected = {
        "schema": C1_GAIN_PART_SCHEMA,
        "trigger_manifest_sha256": locks[
            "trigger_manifest"
        ]["manifest_sha256"],
        "phasec_manifest_sha256": locks["manifest_sha256"],
        "coordinate_manifest_sha256": locks[
            "coordinate_manifest"
        ]["manifest_sha256"],
        "coordinate_manifest_semantic_sha256": locks[
            "coordinate_manifest"
        ]["semantic_sha256"],
        "coordinate_npz_file_sha256": locks[
            "coordinate_npz_file_sha256"
        ],
        "coordinate_npz_semantic_sha256": locks[
            "coordinate_npz_semantic_sha256"
        ],
        "seed": int(args.seed),
        "tier": args.tier,
        "cell_id": args.cell_id,
        "phase": args.phase,
        "noise": args.replicate,
        "resolution": args.resolution,
        "slow_state_sha256": coordinate["state_sha256"],
        "delta_mV": delta,
    }
    if not args.smoke and _reuse_or_fail(path, expected):
        return
    burn_ms = 250.0 if args.smoke else GAIN_BURN_MS
    measure_ms = 250.0 if args.smoke else GAIN_MEASURE_MS
    start_ms = int(np.asarray(locks["state"]["t"])) * ctx["dt"]
    perturb = None if is_zero else {
        "kind": "inhibitory_pulse",
        "t0": start_ms + burn_ms,
        "t1": start_ms + burn_ms + measure_ms,
        "val": delta,
        "target_mask": np.concatenate([
            np.asarray(ctx["core"], bool),
            np.zeros(ctx["S"]["NI"], dtype=bool),
        ]),
    }
    t0 = time.time()
    result, _ckpt, _observer = _run(
        ctx, locks["state"], locks["bank"], burn_ms + measure_ms,
        perturb=perturb,
    )
    burn_step = int(round(burn_ms / ctx["dt"]))
    e = result["E_spk_bool"][min(burn_step, len(result["E_spk_bool"])):]
    duration_ms = len(e) * ctx["dt"]
    full = duration_ms >= 0.99 * measure_ms
    block_metrics = (
        _gain_block_metrics(
            e, ctx, block_ms=float(
                locks["manifest"]["thresholds"]["time_block_ms"]
            )
        )
        if full and not args.smoke else None
    )
    gates, gate_evidence, _ = _carrier_gate_evidence(
        ctx, result, burn_ms, locks["anchor"]["locks"],
        locks["manifest"]["thresholds"]["whole_sheet_plateau_area_frac"],
    )
    if gates["runaway"]:
        status, scientific_reason, technical_reason = (
            "scientific_failure", "runaway", None
        )
    elif gates["whole_sheet_plateau"]:
        status, scientific_reason, technical_reason = (
            "scientific_failure", "whole_sheet_plateau", None
        )
    elif not full or block_metrics is None:
        status, scientific_reason, technical_reason = (
            "technical_invalid", None, "short_conditional_gain"
        )
    else:
        status, scientific_reason, technical_reason = "complete", None, None
    rates = [] if block_metrics is None else block_metrics["core_rate_hz"]
    payload = {
        **expected,
        "phasec_manifest_file_sha256": locks["manifest_file_sha256"],
        "coordinate_manifest_file_sha256": locks[
            "coordinate_manifest_file_sha256"
        ],
        "trigger_manifest_file_sha256": locks[
            "trigger_manifest_file_sha256"
        ],
        "trajectory_id": coordinate["trajectory_id"],
        "path_index": int(coordinate["path_index"]),
        "path_direction": coordinate["path_direction"],
        "status": status,
        "scientific_end_reason": scientific_reason,
        "technical_end_reason": technical_reason,
        "config_sha": ctx["cfg_sha"],
        "noise_bank_sha": locks["bank"]["bank_sha"],
        "state_hash": CK.state_hash(locks["state"]),
        "fast_base_state_hash": locks["state_manifest"]["state_hash"],
        "state_file_sha256": locks["state_ref"]["file_sha256"],
        "coordinate_npz_file_sha256": locks[
            "coordinate_npz_file_sha256"
        ],
        "coordinate_npz_semantic_sha256": locks[
            "coordinate_npz_semantic_sha256"
        ],
        "threshold_offset_mV": delta,
        "core_rate_hz": float(np.mean(rates)) if rates else None,
        "core_rate_500ms_hz": rates,
        "gain_block_metrics": block_metrics,
        "gain_plateau_gate_pass": bool(
            not gates["runaway"] and not gates["whole_sheet_plateau"]
        ),
        "carrier_gates": gates,
        "carrier_gate_evidence": gate_evidence,
        "runaway_early_stop_ms": result.get("runaway_early_stop_ms"),
        "dt_ms": ctx["dt"],
        "burn_in_ms": burn_ms,
        "measure_ms": measure_ms,
        "runtime_provenance": _c1_runtime_provenance(args, ctx, locks),
        "evidence_value": "none_smoke" if args.smoke else "production",
        "wall_s": round(time.time() - t0, 2),
        "peak_rss_gb": _rss_gb(),
        "claim_boundary": (
            "conditional frozen-state susceptibility only; not entry, "
            "offset, recovery, actuator efficacy, or lifecycle"
        ),
    }
    _publish_json_once(path, payload)
    print(
        f"[phasec1 gain] seed={args.seed} {args.tier}/{args.cell_id} "
        f"{args.phase}/{args.replicate} delta={delta:+g} status={status} "
        f"wall={payload['wall_s']}s rss={payload['peak_rss_gb']}GB -> {path}",
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("identity", "gain", "c1_base", "c1_gain"),
        required=True,
    )
    parser.add_argument("--seed", type=int, choices=(1, 3, 4), required=True)
    parser.add_argument("--resolution", choices=("dt", "dt2"), default="dt")
    parser.add_argument("--state-tag")
    parser.add_argument("--phase", choices=("rising", "peak"))
    parser.add_argument(
        "--tier", choices=("primary_convex", "secondary_shell")
    )
    parser.add_argument("--cell-id")
    parser.add_argument("--replicate", choices=NB.PAIRED_REPLICATES, required=True)
    parser.add_argument("--delta-mV", type=float)
    parser.add_argument("--sign", type=int, choices=(-1, 0, 1))
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument(
        "--coordinate-manifest", default=C1_COORDINATE_MANIFEST_PATH
    )
    parser.add_argument(
        "--trigger-manifest", default=C1_GAIN_TRIGGER_MANIFEST_PATH
    )
    parser.add_argument(
        "--dt2-confirmation-manifest",
        default=C1_DT2_CONFIRMATION_MANIFEST_PATH,
    )
    parser.add_argument("--confirm-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("production cell requires --confirm-run")
    if args.mode == "gain" and (args.delta_mV is None or args.sign is None):
        parser.error("gain mode requires --delta-mV and --sign")
    if args.mode in {"identity", "gain"} and not args.state_tag:
        parser.error(f"{args.mode} mode requires --state-tag")
    if args.mode in {"c1_base", "c1_gain"} and (
        args.phase is None or args.tier is None or not args.cell_id
    ):
        parser.error(f"{args.mode} mode requires --phase --tier --cell-id")
    if args.mode == "c1_gain" and (
        args.delta_mV is None or args.sign is None
    ):
        parser.error("c1_gain mode requires --delta-mV and --sign")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.mode == "identity":
        run_identity(arguments)
    elif arguments.mode == "gain":
        run_gain(arguments)
    elif arguments.mode == "c1_base":
        run_c1_base(arguments)
    else:
        run_c1_gain(arguments)
