#!/usr/bin/env python3
"""Issue the LC5 terminal-negative authorization for LC6A 40k dynamics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_topic4_fcxr_lc5v2_u2 as U2  # noqa: E402


OUT = ROOT / "results/topic4_sef_hfo/fcxr_lc6a_patient_axis_surround"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def adjudicate(summary: dict) -> dict:
    checks = {
        "status_complete": summary.get("status") == "COMPLETE",
        "registered_saturation_outcome": summary.get("outcome") == "ESCALATING_SATURATION",
        "registered_saturation_stop": summary.get("early_stop_reason") == "REGISTERED_SATURATION_REACHED",
        "no_offset": summary.get("offset_ms") is None,
        "runaway_label": summary.get("lifecycle", {}).get("label") == "RUNAWAY",
        "end_rate_reaches_registered_ceiling": float(summary.get("end_rate_hz", 0.0)) >= float(U2.SAT_CEILING_HZ),
        "no_conductance_clip": float(summary.get("clip_frac_max_observed", 1.0)) == 0.0,
        "classifier_replay_complete": int(summary.get("classifier_snapshot_replay_n_bundles", 0)) == 28,
    }
    authorized = all(checks.values())
    return {
        "authorize_lc6a_40k_dynamics": bool(authorized),
        "decision": (
            "LC5_LEGACY_SUBSTRATE_ESCALATING_SATURATION"
            if authorized else "LC5_RESULT_DOES_NOT_AUTHORIZE_LC6A"
        ),
        "checks": checks,
        "scientific_boundary": (
            "cell-local U did not open a bounded carrier or offset on the legacy substrate; "
            "this does not reject U on a repaired fast spatial substrate"
        ),
    }


def run(manifest_path: Path) -> dict:
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    contract = manifest["lc5_continuation"]
    summary_path = (
        ROOT / "results/topic4_sef_hfo/fcxr_lc5v2_finite_episode"
        / contract["output_tag"] / "summary.json"
    )
    if not summary_path.is_file():
        raise RuntimeError("LC5 continuation summary is missing")
    summary = json.loads(summary_path.read_text())
    result = adjudicate(summary)
    result.update({
        "status": "COMPLETE",
        "stage": "LC5_TO_LC6A_AUTHORIZATION",
        "lc5_summary": str(summary_path),
        "lc5_summary_sha256": _sha(summary_path),
        "execution_manifest": str(manifest_path),
        "execution_manifest_sha256": _sha(manifest_path),
        "lc5_outcome": summary["outcome"],
        "lc5_onset_ms": summary["onset_ms"],
        "lc5_terminal_ms": summary["T_ms"],
        "lc5_end_rate_hz": summary["end_rate_hz"],
        "lc5_D_end": summary["D_start_end"][1],
        "lc5_H_end": summary["H_start_end"][1],
    })
    _write_json(OUT / "lc5_to_lc6a_authorization.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-manifest", type=Path,
        default=ROOT / "config/topic4_fcxr_lc6a_patient_axis_surround.json",
    )
    parser.add_argument("--confirm-run", action="store_true")
    args = parser.parse_args()
    if not args.confirm_run:
        raise SystemExit("LC5-to-LC6A adjudication requires --confirm-run")
    print(json.dumps(run(args.execution_manifest), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
