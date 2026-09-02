#!/usr/bin/env python3
"""Mechanical QA sidecar for the repaired Topic 5 motif-RNN figure."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (ROOT / "results/paper-ready-figure/fig6_dynamical_motif_rnn_v0_2"
           / "figures")
STEM = FIGURES / "topic5_figure6_dynamical_motif_rnn_v0_2"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual-verdict", choices=("PASS", "FAIL"), required=True)
    parser.add_argument("--manual-note", required=True)
    args = parser.parse_args()
    paths = {suffix: STEM.with_suffix(f".{suffix}") for suffix in ("png", "pdf", "svg")}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(missing)

    with Image.open(paths["png"]) as image:
        width, height = image.size
    pdf_info = subprocess.run(
        ["pdfinfo", str(paths["pdf"])], check=True, capture_output=True, text=True).stdout
    pages = int(next(line.split(":", 1)[1] for line in pdf_info.splitlines()
                     if line.startswith("Pages:")).strip())
    bbox = subprocess.run(
        ["pdftotext", "-bbox", str(paths["pdf"]), "-"],
        check=True, capture_output=True, text=True).stdout
    bbox_root = ET.fromstring(bbox)
    word_heights = [float(word.attrib["yMax"]) - float(word.attrib["yMin"])
                    for word in bbox_root.iter() if word.tag.endswith("word")]
    svg_root = ET.parse(paths["svg"]).getroot()
    svg_text = sum(1 for element in svg_root.iter() if element.tag.endswith("text"))
    payload = {
        "asset_id": STEM.name,
        "manual_visual_verdict": args.manual_verdict,
        "manual_visual_note": args.manual_note,
        "png_pixels": [width, height],
        "pdf_pages": pages,
        "pdf_min_word_box_height_pt": min(word_heights) if word_heights else None,
        "pdf_word_boxes": len(word_heights),
        "svg_text_nodes": svg_text,
        "svg_text_preserved": svg_text > 0,
        "sha256": {suffix: digest(path) for suffix, path in paths.items()},
        "source_data_present": (FIGURES / "source_data").is_dir(),
        "readme_present": (FIGURES / "README.md").is_file(),
        "metadata_present": (FIGURES / "FIGURE6_METADATA.json").is_file(),
    }
    (FIGURES / "FIGURE_VISUAL_QA.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
