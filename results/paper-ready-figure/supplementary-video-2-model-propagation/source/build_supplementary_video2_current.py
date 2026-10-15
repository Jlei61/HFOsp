#!/usr/bin/env python3
"""Verify the accepted Supplementary Video 2, or copy its frozen exports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "results/paper-ready-figure/supplementary-video-2-model-propagation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Copy frozen GIF and preview to an empty directory")
    args = parser.parse_args()
    pointer = json.loads((PACKAGE / "current_version.json").read_text())
    metadata_path = ROOT / "results/paper-ready-figure/supplementary-video-2_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    for record in (pointer, metadata):
        assert record["paper_slot"] == "Supplementary Video 2"
        assert record["status"] == "AUTHOR_ACCEPTED_FINAL"
        assert record["version"] == "paired_rows_20261015"
        assert record["human_visual_acceptance"] == "AUTHOR_CONFIRMED"
    manifest = json.loads((PACKAGE / "release_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert sha(PACKAGE / name) == expected, name
    for name, expected in pointer["outputs"].items():
        assert sha(PACKAGE / "figures" / name) == expected, name
    canonical = ROOT / pointer["canonical_path"]
    assert sha(canonical) == pointer["outputs"]["supplementary-video-2.gif"] == metadata["video"]["sha256"]
    with Image.open(canonical) as video:
        assert video.size == (1600, 1091) and video.n_frames == 125
        for i in range(video.n_frames):
            video.seek(i)
            video.load()
            assert video.info["duration"] == 50
    registry = json.loads((ROOT / "config/paper_figure_source_registry.json").read_text())
    assert registry["current_supplementary_video2"]["pointer"] == str((PACKAGE / "current_version.json").relative_to(ROOT))
    assert registry["current_supplementary_video2"]["asset_id"] == pointer["asset_id"]
    assert metadata_path.read_bytes() == (PACKAGE / "metadata.json").read_bytes()
    if args.output_dir:
        output = args.output_dir.resolve()
        assert output != PACKAGE and PACKAGE not in output.parents, "Use a separate output directory"
        assert not output.exists() or not any(output.iterdir()), "Output directory must be empty"
        output.mkdir(parents=True, exist_ok=True)
        for name, expected in pointer["outputs"].items():
            shutil.copyfile(PACKAGE / "figures" / name, output / name)
            assert sha(output / name) == expected
    print(json.dumps(dict(status="PASS", paper_slot=pointer["paper_slot"],
                          version=pointer["version"], author_acceptance="CONFIRMED",
                          frame_count=125, all_frames_decoded=True,
                          release_files_verified=len(manifest["files"])), indent=2))


if __name__ == "__main__":
    main()
