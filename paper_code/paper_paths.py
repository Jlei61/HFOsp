"""Portable paths: code is local; data and generated outputs live separately."""
from pathlib import Path
import hashlib
import json
import os

CODE_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path(os.environ.get("HFO_PAPER_INPUTS", CODE_ROOT.parent / "paper_inputs")).resolve()
OUTPUT_ROOT = Path(os.environ.get("HFO_PAPER_OUTPUTS", CODE_ROOT.parent / "paper_outputs")).resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_inputs():
    path = INPUT_ROOT / "input_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing derived input pack: {path}. Set HFO_PAPER_INPUTS before starting Python.")
    expected = (CODE_ROOT / "config/inputs.sha256").read_text().strip()
    if sha256(path) != expected:
        raise ValueError("The input manifest does not match this code package's frozen input set")
    manifest = json.loads(path.read_text())
    for name, digest in manifest["files"].items():
        if sha256(INPUT_ROOT / name) != digest:
            raise ValueError(f"Input differs from the recorded snapshot: {name}")
    return {"files_verified": len(manifest["files"]), "input_root": str(INPUT_ROOT)}


def new_output(name):
    path = OUTPUT_ROOT / name
    if path == INPUT_ROOT or INPUT_ROOT in path.parents or path == CODE_ROOT or CODE_ROOT in path.parents:
        raise ValueError("Outputs must be outside the code and input directories")
    path.mkdir(parents=True, exist_ok=True)
    return path
