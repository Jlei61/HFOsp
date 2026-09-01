"""Shared runtime contract for data-driven Topic 4 SNN field searches."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "config/topic4_data_driven_snn_baseline_zm_v1.json"
EXPECTED_SCHEMA = "topic4_data_driven_snn_baseline_zm_v1"
EXPECTED_ID = "data_driven_snn_h_spatial_ou_zm_reference_v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data_driven_snn_baseline(path=DEFAULT_BASELINE, *, root=ROOT):
    path = Path(path).resolve()
    root = Path(root).resolve()
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != EXPECTED_SCHEMA:
        raise RuntimeError("data-driven SNN baseline schema changed")
    if payload.get("baseline_id") != EXPECTED_ID:
        raise RuntimeError("data-driven SNN baseline identity changed")
    for record in payload["inputs"].values():
        input_path = root / record["path"]
        if sha256(input_path) != record["sha256"]:
            raise RuntimeError(f"baseline input hash changed: {record['path']}")
    consumer = payload["consumer_contract"]
    if consumer.get("default_runtime_mode") is not None:
        raise RuntimeError("unsafe implicit data-driven SNN runtime mode")
    if not consumer.get("runtime_mode_must_be_explicit"):
        raise RuntimeError("data-driven SNN runtime mode must remain explicit")
    if float(consumer["minimum_simulation_duration_ms"]) < 20000.0:
        raise RuntimeError("baseline duration cannot detect delayed runaway")
    if not consumer.get("late_runaway_is_invalid"):
        raise RuntimeError("late runaway must invalidate a field candidate")
    return payload


def baseline_record(path=DEFAULT_BASELINE):
    path = Path(path).resolve()
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256(path),
        "baseline_id": EXPECTED_ID,
    }


def apply_data_driven_snn_baseline(candidate, baseline, *, runtime_mode):
    """Attach one explicit shared runtime to a frozen field candidate."""
    allowed = baseline["consumer_contract"]["allowed_runtime_modes"]
    if runtime_mode not in allowed:
        raise ValueError(
            f"runtime_mode must be explicit and one of {allowed}, got {runtime_mode!r}"
        )
    output = copy.deepcopy(candidate)
    spatial_ou = copy.deepcopy(baseline["spatial_ou"])
    if "spatial_ou" in output and output["spatial_ou"] != spatial_ou:
        raise RuntimeError("candidate spatial OU differs from shared baseline")
    output["spatial_ou"] = spatial_ou
    mz = copy.deepcopy(
        baseline["active_slow_state"]
        if runtime_mode == "active_z_plus_m"
        else baseline["paired_slow_off"]
    )
    mz.pop("reference_status", None)
    mz.pop("usage", None)
    if "mz" in output and output["mz"] != mz:
        raise RuntimeError("candidate Z/M state differs from shared baseline")
    for key in ("adaptation", "inhibitory_resource", "ee_std"):
        if output.get(key, {}).get("mode", "off") != "off":
            raise RuntimeError(f"shared Z/M baseline cannot combine with active {key}")
    output["mz"] = mz
    output["data_driven_snn_baseline"] = {
        "baseline_id": baseline["baseline_id"],
        "runtime_mode": runtime_mode,
        "active_reference_status": baseline["active_slow_state"][
            "reference_status"
        ],
        "minimum_simulation_duration_ms": baseline["consumer_contract"][
            "minimum_simulation_duration_ms"
        ],
        "late_runaway_is_invalid": True,
    }
    return output
