"""Freeze the observation-invariant uniform forced-source grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config/topic4_rev10_d4_uniform_forced_source_map.json"


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


def uniform_sources(config):
    coordinates = [float(value) for value in config["source_grid"]["coordinates_mm"]]
    return [{
        "source_id": f"grid_x{int(x):02d}_y{int(y):02d}",
        "xy_mm": [x, y],
    } for y in coordinates for x in coordinates]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config["scientific_role"] != "development_only_uniform_forced_route_localization":
        raise RuntimeError("rev10-D4 scientific role changed")
    for record in config["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"input hash changed: {record['path']}")
    sources = uniform_sources(config)
    if len(sources) != 25 or len({row["source_id"] for row in sources}) != 25:
        raise RuntimeError("uniform source grid changed")
    classifier_record = config["inputs"]["frozen_direction_classifier_manifest"]
    classifier_source = json.loads((ROOT / classifier_record["path"]).read_text())
    payload = {
        "status": "REV10D4_UNIFORM_SOURCE_GRID_FROZEN",
        "scientific_role": config["scientific_role"],
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": _sha256(config_path),
        },
        "source_grid": {
            "n_sources": len(sources),
            "sources": sources,
            "packet_fraction_of_E": float(
                config["source_grid"]["packet_fraction_of_E"]
            ),
            "selection": config["source_grid"]["selection"],
        },
        "direction_classifier": classifier_source["direction_classifier"],
        "direction_classifier_source": {
            "path": classifier_record["path"],
            "sha256": classifier_record["sha256"],
            "copied_without_refit": True,
        },
        "network_seeds": list(map(int, config["network_seeds"])),
        "git_commit_at_freeze": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip(),
    }
    output = ROOT / config["output_root"] / "source_grid_manifest.json"
    _atomic_json(output, payload)
    print(json.dumps({
        "status": payload["status"],
        "n_sources": len(sources),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
