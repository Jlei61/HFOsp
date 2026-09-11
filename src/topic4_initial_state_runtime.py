"""Shared runtime for the initial-state conditioned propagation v1 scripts.

Substrate construction mirrors scripts/run_topic4_multidimensional_worker.py
argument-for-argument so that the physical model is the mainline's; only the
t=0 membrane voltage and the read-only observer are new.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (str(ROOT), str(ROOT / "src" / "snn_engine")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

DESIGN_PATH = ROOT / "config/topic4_initial_state_conditioned_propagation_v1.json"
ARTIFACT_ROOT = Path("/home/honglab/leijiaxin/HFOsp")
PYTHON = "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python"
ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": "/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib",
    "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    "CUDA_VISIBLE_DEVICES": "",
}
MAINLINE_WORKER_NAMES = {
    "run_topic4_rev12_node_worker.py", "run_topic4_multidimensional_worker.py",
    "run_topic4_xy_fig5_worker.py",
}
BRANCH_WORKER_NAME = "run_topic4_initial_state_worker.py"


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(handle, "w") as stream:
        json.dump(json_safe(payload), stream, indent=1, sort_keys=False)
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
    os.close(handle)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_design(path=DESIGN_PATH):
    design = read(path)
    if design["design_id"] != "topic4_initial_state_conditioned_propagation_v1":
        raise RuntimeError("unexpected design id")
    return design


def output_root(design):
    return Path(design["output_root"])


def verify_design_sources(design):
    """Hash every frozen source the design records; return the audit."""
    rows = {}
    for key, record in design["sources"].items():
        path = Path(record["path"])
        observed = sha(path) if path.exists() else None
        rows[key] = {"path": str(path), "expected": record["sha256"],
                     "observed": observed, "match": observed == record["sha256"]}
    return rows


def git_commit(root=ROOT):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def git_dirty_paths(root=ROOT):
    out = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True)
    return [line[3:] for line in out.splitlines() if line.strip()]


def loaded_source_hashes():
    """sha256 of every imported .py module under ROOT (like _runtime_provenance)."""
    paths = set()
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != ".py":
            continue
        try:
            paths.add(str(path.relative_to(ROOT)))
        except ValueError:
            continue
    return {path: sha(ROOT / path) for path in sorted(paths)}


def verify_source_snapshot(manifest):
    """Abort when any frozen source module differs from the frozen manifest."""
    for relative, expected in manifest["source_hashes"].items():
        observed = sha(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"frozen source changed: {relative}")
    return True


def candidate_record(design):
    manifest_path = Path(design["sources"]["candidate_manifest"]["path"])
    if sha(manifest_path) != design["sources"]["candidate_manifest"]["sha256"]:
        raise RuntimeError("candidate manifest hash changed")
    manifest = read(manifest_path)
    rows = [row for row in manifest["candidates"] if row["candidate_id"] == design["candidate_id"]]
    if len(rows) != 1:
        raise RuntimeError("candidate is not unique in the frozen manifest")
    candidate = rows[0]
    canonical = json.dumps(candidate, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    if digest != design["candidate_canonical_json_sha256"]:
        raise RuntimeError("candidate canonical JSON hash changed")
    return candidate


def execution_config(design):
    path = Path(design["sources"]["execution_config"]["path"])
    if sha(path) != design["sources"]["execution_config"]["sha256"]:
        raise RuntimeError("execution config hash changed")
    return read(path)


def transition_config(design):
    from src.topic4_initial_state_substrate import load_round_config
    execution = execution_config(design)
    record = execution["inputs"]["transition_config"]
    local = ROOT / record["path"]
    path = local if local.exists() else ARTIFACT_ROOT / record["path"]
    if sha(path) != record["sha256"]:
        raise RuntimeError("transition config hash changed")
    return load_round_config(path), path


def network_record(design, topology_seed, *, frozen_manifest=None):
    """Corrected-graph record for one topology seed (2511 from the mainline
    execution config; other seeds from the frozen manifest of this branch)."""
    execution = execution_config(design)
    record = execution.get("corrected_networks", {}).get(str(int(topology_seed)))
    if record is None and frozen_manifest is not None:
        record = frozen_manifest.get("replication_networks", {}).get(str(int(topology_seed)))
    if record is None:
        raise RuntimeError(f"no corrected graph bound for topology seed {topology_seed}")
    if sha(record["path"]) != record["sha256"]:
        raise RuntimeError("corrected graph hash changed")
    return record


def build_frozen_substrate(design, topology_seed, dynamics_seed, *, frozen_manifest=None):
    """Exactly the mainline worker's construction for the frozen candidate."""
    from src.topic4_multidimensional_parameters import apply_parameters
    from src.topic4_initial_state_substrate import build_substrate
    candidate = candidate_record(design)
    transition, _ = transition_config(design)
    execution = execution_config(design)
    record = network_record(design, topology_seed, frozen_manifest=frozen_manifest)
    node_mapping = candidate.get("node_mapping", {})
    mechanisms = candidate.get("mechanisms", {})
    if candidate.get("field_transform") not in (None, "none"):
        raise RuntimeError("the frozen candidate must not carry a field transform")
    if candidate.get("edge_coefficients_override") is not None or candidate.get("topology_override") is not None:
        raise RuntimeError("the frozen candidate must not carry overrides")
    base_candidate_id = str(execution.get("reference", {}).get(
        "base_substrate_candidate_id", "node_baseline"))
    substrate = build_substrate(
        transition, base_candidate_id, int(topology_seed),
        cache_dir=str(execution["network_cache"]),
        field_transform=None,
        ee_dose=float(mechanisms.get("g_EE", 0.0)),
        etoi_dose=float(mechanisms.get("g_EtoI", 0.0)),
        node_candidate_override=candidate["node_field"],
        node_depth_shrinkage=float(node_mapping.get("signed_depth_shrinkage", 1.0)),
        node_gain=float(node_mapping.get("node_gain", 1.0)),
        node_dispersion_candidate_override=candidate.get("node_dispersion_field"),
        edge_coefficients_override=None,
        graph_aspect_ratio_override=None,
        ee_ellipse_angle_deg=float(mechanisms.get("ellipse_angle_deg", 45.0)),
        ee_ellipse_aspect_ratio=float(mechanisms.get("ellipse_aspect_ratio", 2.0)),
        ee_ellipse_reference_angle_deg=(
            None if mechanisms.get("ellipse_reference_angle_deg") is None
            else float(mechanisms["ellipse_reference_angle_deg"])),
        ee_ellipse_reference_aspect_ratio=(
            None if mechanisms.get("ellipse_reference_aspect_ratio") is None
            else float(mechanisms["ellipse_reference_aspect_ratio"])),
        artifact_root=ARTIFACT_ROOT,
        topology_seed=int(topology_seed), dynamics_seed=int(dynamics_seed),
        network_cache_record=record,
    )
    if np.any(substrate.edge_coefficients != 0):
        raise RuntimeError("frozen candidate has nonzero learned connectivity")
    parameter_audit = apply_parameters(substrate, candidate["dynamic_parameters"])
    return substrate, candidate, transition, execution, parameter_audit, record


def static_identity(substrate, parameter_audit):
    from src.topic4_initial_state import array_sha256
    return {
        "positions_E_sha256": array_sha256(np.asarray(substrate.positions_e, np.float32)),
        "h_sha256": array_sha256(np.asarray(substrate.h_e, np.float32)),
        "delta_vtheta_sha256": array_sha256(np.asarray(substrate.delta_vtheta, np.float32)),
        "vtheta_sha256": array_sha256(np.asarray(substrate.vtheta, np.float32)),
        "positions_E_float64_sha256": array_sha256(np.asarray(substrate.positions_e, np.float64)),
        "h_float64_sha256": array_sha256(np.asarray(substrate.h_e, np.float64)),
        "vtheta_float64_sha256": array_sha256(np.asarray(substrate.vtheta, np.float64)),
        "ampa_topology_sha256": parameter_audit["sparse_pathways"]["ampa_by_delay"]["topology_sha256"],
        "ampa_values_sha256": parameter_audit["sparse_pathways"]["ampa_by_delay"]["after_sha256"],
        "gaba_topology_sha256": parameter_audit["sparse_pathways"]["gaba_by_delay"]["topology_sha256"],
        "gaba_values_sha256": parameter_audit["sparse_pathways"]["gaba_by_delay"]["after_sha256"],
        "identity_scope": ("positions, E/I partition, delay-binned edges/weights, "
                           "core mask, signed threshold offsets and thresholds"),
    }


def load_evaluator(design):
    record = design["sources"]["evaluator"]
    if sha(record["path"]) != record["sha256"]:
        raise RuntimeError("frozen evaluator changed")
    with open(record["path"], "rb") as stream:
        evaluator = pickle.load(stream)
    if evaluator.k != 2 or not evaluator.modes_stable:
        raise RuntimeError("frozen evaluator does not carry two stable modes")
    evaluator.cache_probe = None
    return evaluator


def load_objective(design):
    record = design["sources"]["objective"]
    if sha(record["path"]) != record["sha256"]:
        raise RuntimeError("frozen objective changed")
    with open(record["path"], "rb") as stream:
        return pickle.load(stream)


def load_observation_contract(design):
    record = design["sources"]["observation_contract"]
    if sha(record["path"]) != record["sha256"]:
        raise RuntimeError("frozen observation contract changed")
    qualification = design["sources"]["evaluator_qualification"]
    if sha(qualification["path"]) != qualification["sha256"]:
        raise RuntimeError("frozen observer qualification changed")
    if read(qualification["path"])["observer_sha256"] != record["sha256"]:
        raise RuntimeError("observer contract is not the qualified one")
    return read(record["path"])


def classify_with_both_modes(evaluator, table):
    """Frozen classifier output plus the distance to EVERY patient mode (Q2).

    Labels, support state and the assigned-mode distance are identical to
    evaluator.classify(); nothing is refit (Q1).
    """
    from src.topic4_interictal_repaired_evaluation import rank_features, validate
    from src.topic4_joint_xy_kernel import event_kernel_features
    t = validate(table)
    if t.shape[1] != evaluator.patient.shape[1]:
        raise ValueError("contact count mismatch")
    n = len(t)
    readable = np.isfinite(t).sum(1) >= 2
    labels = np.full(n, -1, int)
    state = np.zeros(n, int)
    distance_modes = np.full((n, evaluator.k), np.nan)
    if readable.any():
        ix = np.flatnonzero(readable)
        labels[ix] = evaluator.km.predict(rank_features(t[ix]))
        x = event_kernel_features(t[ix], evaluator.xy, evaluator.groups, evaluator.scale)["joint"]
        for mode in range(evaluator.k):
            distance_modes[ix, mode] = evaluator.trees[mode].query(x, k=5)[0].mean(1)
            local = np.flatnonzero(labels[ix] == mode)
            selected = ix[local]
            d = distance_modes[selected, mode]
            state[selected[d <= evaluator.radii[mode, 0]]] = 1
            state[selected[d > evaluator.radii[mode, 1]]] = -1
    reference_labels, reference_state, reference_distance = evaluator.classify(t)
    assigned = np.where(labels >= 0, distance_modes[np.arange(n), np.maximum(labels, 0)], np.nan)
    if (not np.array_equal(labels, reference_labels) or not np.array_equal(state, reference_state)
            or not np.array_equal(np.isnan(assigned), np.isnan(reference_distance))
            or not np.allclose(np.nan_to_num(assigned), np.nan_to_num(reference_distance))):
        raise RuntimeError("classification helper diverged from the frozen classifier")
    return labels, state, distance_modes


def process_tree(pid):
    children = {}
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            stat = (proc / "stat").read_text()
            parent = int(stat.rsplit(")", 1)[1].split()[1])
            children.setdefault(parent, []).append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    stack, seen = [int(pid)], []
    while stack:
        current = stack.pop()
        seen.append(current)
        stack.extend(children.get(current, []))
    return seen


def running_worker_pids():
    """PIDs of mainline SNN workers and of this branch's workers, separately."""
    mainline, branch = [], []
    for proc in Path("/proc").glob("[0-9]*"):
        try:
            args = (proc / "cmdline").read_bytes().split(b"\0")
            names = {Path(os.fsdecode(arg)).name for arg in args if arg}
            if names & MAINLINE_WORKER_NAMES:
                mainline.append(int(proc.name))
            elif BRANCH_WORKER_NAME in names:
                branch.append(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError, ValueError):
            continue
    return {"mainline": sorted(set(mainline)), "branch": sorted(set(branch))}


def available_gib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return float(line.split()[1]) / 1024 ** 2
    raise RuntimeError("MemAvailable missing")


def peak_rss_gib(pid=None):
    path = Path(f"/proc/{os.getpid() if pid is None else pid}/status")
    for line in path.read_text().splitlines():
        if line.startswith("VmHWM:"):
            return float(line.split()[1]) / 1024 ** 2
    return None
