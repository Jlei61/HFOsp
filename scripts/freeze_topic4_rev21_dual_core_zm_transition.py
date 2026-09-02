#!/usr/bin/env python3
"""Freeze the rev21 dual-core Z/M coarse library from rev20 evidence."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve(artifact_root: Path, path: str) -> Path:
    local = ROOT / path
    return local if local.exists() else artifact_root / path


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".json.tmp")
    os.close(handle)
    try:
        Path(temporary).write_text(json.dumps(
            payload, indent=2, sort_keys=True, allow_nan=False,
        ) + "\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _token(value: float) -> str:
    return f"{float(value):.2f}".rstrip("0").rstrip(".").replace(".", "p")


def build_candidates(config: dict, rev20_candidate: dict) -> list[dict]:
    expected = config["reference"]
    mechanisms = rev20_candidate["mechanisms"]
    checks = {
        "field_sha256": rev20_candidate["node_field"]["field_sha256"],
        "node_gain": rev20_candidate["node_mapping"]["node_gain"],
        "signed_depth_shrinkage": rev20_candidate["node_mapping"][
            "signed_depth_shrinkage"
        ],
        "g_EE": mechanisms["g_EE"],
        "g_EtoI": mechanisms["g_EtoI"],
        "ellipse_angle_deg": mechanisms["ellipse_angle_deg"],
        "ellipse_aspect_ratio": mechanisms["ellipse_aspect_ratio"],
    }
    if any(checks[key] != expected[key] for key in checks):
        raise RuntimeError(f"rev20 winner drifted from rev21 reference: {checks}")

    base = copy.deepcopy(rev20_candidate)
    base["source_rev20_candidate_id"] = base.pop("candidate_id")
    base["family"] = "zm_off_reference"
    base["level"] = "off"
    base["is_reference"] = True
    base["candidate_id"] = "rev21_zm_off"
    base["mechanisms"]["Z_M"] = "off"
    base["slow_variables"] = {"mode": "off"}
    rows = [base]

    reference = config["zm_reference"]
    integrated = float(reference["eta_m"]) * float(reference["tau_adp_ms"])
    for s_i in config["coarse_grid"]["I_th_EI_scales"]:
        for s_m in config["coarse_grid"]["integrated_M_scales"]:
            row = copy.deepcopy(base)
            row["candidate_id"] = f"rev21_si_{_token(s_i)}_sm_{_token(s_m)}"
            row["family"] = "coarse_zm"
            row["level"] = {"I_th_EI_scale": s_i, "integrated_M_scale": s_m}
            row["is_reference"] = False
            row["mechanisms"]["Z_M"] = "z_plus_m"
            tau_adp = float(reference["tau_adp_ms"])
            row["slow_variables"] = {
                "mode": "z_plus_m",
                "use_z": True,
                "use_m": True,
                "I_th_EI": float(reference["I_th_EI"]) * float(s_i),
                "tau_z": float(reference["tau_z_ms"]),
                "tau_adp": tau_adp,
                "eta_m": integrated * float(s_m) / tau_adp,
                "I_th_EI_scale": float(s_i),
                "integrated_M_scale": float(s_m),
                "trace_stride_steps": int(reference["trace_stride_steps"]),
            }
            rows.append(row)
    if len(rows) != 17 or len({row["candidate_id"] for row in rows}) != 17:
        raise RuntimeError("rev21 coarse library must contain 17 unique candidates")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--artifact-root", type=Path,
        default=Path("/home/honglab/leijiaxin/HFOsp"),
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    artifact_root = args.artifact_root.resolve()
    config = json.loads(config_path.read_text())
    for name, record in config["inputs"].items():
        path = _resolve(artifact_root, record["path"])
        if _sha256(path) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {name}")

    manifest_path = _resolve(
        artifact_root, config["inputs"]["rev20_candidate_manifest"]["path"],
    )
    rev20 = json.loads(manifest_path.read_text())
    selected_path = _resolve(
        artifact_root, config["inputs"]["rev20_selection"]["path"],
    )
    selected = json.loads(selected_path.read_text())
    candidate_id = config["reference"]["rev20_candidate_id"]
    if candidate_id not in selected["candidate_ids"]:
        raise RuntimeError("rev21 substrate was not frozen by rev20 selection")
    matches = [row for row in rev20["candidates"]
               if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise RuntimeError("rev20 substrate candidate is missing or duplicated")

    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    payload = {
        "schema_id": "topic4_rev21_dual_core_zm_coarse_manifest_v1",
        "status": "REV21_DUAL_CORE_ZM_COARSE_LIBRARY_FROZEN",
        "git_commit": commit,
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": _sha256(config_path),
        "rev20_manifest_sha256": _sha256(manifest_path),
        "rev20_selection_sha256": _sha256(selected_path),
        "substrate_candidate_id": candidate_id,
        "candidates": build_candidates(config, matches[0]),
        "patient_ictal_inputs_read": False,
        "claim_boundary": config["claim_boundary"],
    }
    output = artifact_root / config["candidate_manifest"]
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output),
                      "candidate_count": len(payload["candidates"])}, indent=2))


if __name__ == "__main__":
    main()
