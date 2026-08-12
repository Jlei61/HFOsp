"""Freeze D4.1 source-dose confirmation inputs before fresh networks run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d4_1_packet_dose_confirmation.json"


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(payload, indent=2, sort_keys=True))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_contract(config):
    if config["scientific_role"] != (
        "development_only_fresh_network_forced_route_dose_confirmation"
    ):
        raise RuntimeError("rev10-D4.1 scientific role changed")
    sources = config["sources"]
    if [(row["xy_mm"], row["expected_mode"]) for row in sources] != [
        ([18.0, 6.0], "A"), ([2.0, 14.0], "B"),
    ]:
        raise RuntimeError("D4.1 source identity changed")
    fractions = list(map(float, config["packet_fractions_of_E"]))
    if fractions != sorted(set(fractions)) or fractions[-1] != 0.005:
        raise RuntimeError("D4.1 packet-dose ladder changed")
    if len(set(map(int, config["network_seeds"]))) != 6:
        raise RuntimeError("D4.1 requires six distinct fresh networks")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    validate_contract(config)
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")

    d4_verdict = json.loads((ROOT / config["inputs"]["d4_verdict"]["path"]).read_text())
    if not (
        d4_verdict["status"]
        == "REV10D4_UNIFORM_FORCED_MODE_A_ROUTE_CAPACITY_OBSERVED"
        and d4_verdict["selected_source_id"] == "grid_x18_y06"
    ):
        raise RuntimeError("D4.1 A source is not supported by the frozen D4 verdict")
    d4_manifest = json.loads(
        (ROOT / config["inputs"]["d4_source_manifest"]["path"]).read_text()
    )
    payload = {
        "status": "REV10D4_1_PACKET_DOSE_CONFIRMATION_FROZEN",
        "scientific_role": config["scientific_role"],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "sources": config["sources"],
        "packet_fractions_of_E": list(map(float, config["packet_fractions_of_E"])),
        "network_seeds": list(map(int, config["network_seeds"])),
        "direction_classifier": d4_manifest["direction_classifier"],
        "direction_classifier_source": {
            "path": config["inputs"]["d4_source_manifest"]["path"],
            "sha256": config["inputs"]["d4_source_manifest"]["sha256"],
            "copied_without_refit": True,
        },
        "d4_source_selection": {
            "path": config["inputs"]["d4_verdict"]["path"],
            "sha256": config["inputs"]["d4_verdict"]["sha256"],
            "selected_before_fresh_networks": True,
            "development_only": True,
        },
        "git_commit_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
    }
    output = ROOT / config["output_root"] / "packet_dose_manifest.json"
    _atomic_json(output, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
