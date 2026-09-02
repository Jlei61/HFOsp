#!/usr/bin/env python3
"""Record same-state visual QA for the Topic 5.2 candidate figure."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.topic5_latent_landscape_v0_2 import atomic_write_json, sha256_file  # noqa: E402


FIGURES = ROOT / "results/topic5_latent_propagation_landscape_v0_2/paper-ready-figure/latent_landscape_candidate/figures"
STEM = "topic5_latent_landscape_v0_2_candidate"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png-pass", action="store_true")
    parser.add_argument("--pdf-pass", action="store_true")
    parser.add_argument("--svg-pass", action="store_true")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    paths = {suffix: FIGURES / f"{STEM}.{suffix}" for suffix in ("png", "pdf", "svg")}
    metadata_path = FIGURES / f"{STEM}_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    image = Image.open(paths["png"])
    checks = {
        "png_operator_visual_pass": args.png_pass,
        "pdf_operator_visual_pass": args.pdf_pass,
        "svg_operator_visual_pass": args.svg_pass,
        "png_large_enough": image.width >= 3000 and image.height >= 2400,
        "png_rgb_or_rgba": image.mode in {"RGB", "RGBA"},
        "pdf_signature": paths["pdf"].read_bytes()[:5] == b"%PDF-",
        "svg_signature": b"<svg" in paths["svg"].read_bytes()[:1000],
        "metadata_hashes_match": all(
            metadata["outputs"].get(suffix) == sha256_file(path) for suffix, path in paths.items()
        ),
        "all_panel_source_contracts": len(list((FIGURES / "source_data").glob("panel_*"))) == 10,
    }
    payload = {
        "contract": "topic5_latent_landscape_figure_visual_QA_v0_2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "png_dimensions": [image.width, image.height],
        "review_notes": args.notes,
        "same_state_hashes": {suffix: sha256_file(path) for suffix, path in paths.items()},
        "claim_boundary": "CANDIDATE_CLOSEOUT_FIGURE_NOT_CANONICAL_PAPER_SLOT",
    }
    atomic_write_json(FIGURES / "FIGURE_VISUAL_QA.json", payload)
    if payload["status"] != "PASS":
        raise RuntimeError(f"figure QA failed: {[name for name, passed in checks.items() if not passed]}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
