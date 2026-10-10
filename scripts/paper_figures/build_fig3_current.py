#!/usr/bin/env python3
"""Verify or rebuild the author-accepted A–E Figure 3 from native plot layers.

The default verifies the current package. --output-dir writes a new, empty
directory without touching the accepted figure. Raw recordings are not needed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

# Load the scientific environment before MuPDF on the project's workstation.
import matplotlib
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "results/paper-ready-figure/fig3"
VERSION = "visual_alignment_compact_rows_20261010"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verify(package):
    pointer = read(package / "current_revision.json")
    registry = read(package / "figure3_panel_registry.json")
    manifest = read(package / "release_manifest.json")
    for record in (pointer, registry, manifest):
        if record.get("status") != "AUTHOR_ACCEPTED_FINAL" or record.get("version") != VERSION:
            raise ValueError("Figure 3 pointer, registry and manifest must agree on the accepted A–E version.")
    if set(registry["panels"]) != set("abcde") or pointer["layout"] != "A–E":
        raise ValueError("The current Figure 3 has five panels A–E; legacy A–F is not a fallback.")
    for filename, expected in manifest["files"].items():
        path = package / filename
        if not path.is_file() or digest(path) != expected["sha256"]:
            raise ValueError(f"Missing or changed Figure 3 release file: {filename}")
    if any((package / "figures").glob("fig3-panelf.*")):
        raise ValueError("Obsolete panel F found in the current Figure 3 output directory.")
    b = registry["panels"]["b"]
    if b["identity"] != "Y1 | SZ6" or b["time_window_sec"] != [0, 10] or b["shared_axis"]:
        raise ValueError("The accepted Y1/SZ6 fixed-window case contract changed.")
    return {"status": "PASS", "version": VERSION, "panels": list("ABCDE"),
            "files_verified": len(manifest["files"]), "author_acceptance": "CONFIRMED"}


def compose(package, output, dpi):
    spec = read(package / "source/layout_spec.json")
    doc = pymupdf.open()
    width, height = spec["canvas_inches"]
    page = doc.new_page(width=width * 72, height=height * 72)
    for key in "abcde":
        placement = spec["placements"][key]
        with pymupdf.open(output / f"figures/fig3-panel{key}.pdf") as panel:
            page.show_pdf_page(pymupdf.Rect(placement["destination_points"]), panel, 0,
                               clip=pymupdf.Rect(placement["clip_points"]))
        x, y = placement["letter_inches"]
        page.insert_text((x * 72, y * 72), key.upper(), fontsize=19, fontname="hebo")
    stem = output / "figures/fig3-complete-layout"
    doc.save(stem.with_suffix(".pdf"), deflate=True, garbage=4)
    page.get_pixmap(dpi=dpi).save(stem.with_suffix(".png"))
    page.get_pixmap(dpi=130).save(output / "figures/fig3-complete-layout-preview.png")
    stem.with_suffix(".svg").write_text(page.get_svg_image(text_as_path=True), encoding="utf-8")
    doc.close()


def compare_pdf_pixels(first, second, dpi=130):
    with pymupdf.open(first) as a, pymupdf.open(second) as b:
        pa, pb = a[0].get_pixmap(dpi=dpi), b[0].get_pixmap(dpi=dpi)
        return (pa.width, pa.height, pa.n) == (pb.width, pb.height, pb.n) and pa.samples == pb.samples


def rebuild(package, output, dpi=600):
    if output.resolve() == package.resolve() or (output.exists() and any(output.iterdir())):
        raise ValueError("Use a separate empty output directory; the accepted package is read-only.")
    (output / "figures").mkdir(parents=True, exist_ok=True)
    for key in "abcde":
        for extension in ("png", "pdf", "svg"):
            name = f"fig3-panel{key}.{extension}"
            shutil.copy2(package / "source/native_panels" / name, output / "figures" / name)
            assert digest(output / "figures" / name) == digest(package / "figures" / name)
    compose(package, output, dpi)
    same = compare_pdf_pixels(output / "figures/fig3-complete-layout.pdf",
                              package / "figures/fig3-complete-layout.pdf")
    if not same:
        raise ValueError("Rebuilt Figure 3 differs from the author-accepted composite.")
    shutil.copy2(package / "figures/README.md", output / "figures/README.md")
    report = {"status": "PASS", "single_panels_byte_identical": True,
              "complete_pdf_render_pixel_identical": same, "comparison_dpi": 130,
              "method": "Native frozen panel vectors at 1:1 scale and recorded placements; no raw analysis rerun."}
    (output / "rebuild_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--output-dir", type=Path, help="Empty directory for a verified layout rebuild.")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()
    report = verify(args.package)
    if args.output_dir:
        report["rebuild"] = rebuild(args.package, args.output_dir, args.dpi)
        report["output"] = str(args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
