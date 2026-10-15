"""Portable paths: code is local; data and generated outputs live separately."""
from pathlib import Path
import os

CODE_ROOT = Path(__file__).resolve().parent
INPUT_ROOT = Path(os.environ.get("HFO_PAPER_INPUTS", CODE_ROOT.parent / "paper_inputs")).resolve()
OUTPUT_ROOT = Path(os.environ.get("HFO_PAPER_OUTPUTS", CODE_ROOT.parent / "paper_outputs")).resolve()


def new_output(name):
    path = OUTPUT_ROOT / name
    if path == INPUT_ROOT or INPUT_ROOT in path.parents or path == CODE_ROOT or CODE_ROOT in path.parents:
        raise ValueError("Outputs must be outside the code and input directories")
    path.mkdir(parents=True, exist_ok=True)
    return path
