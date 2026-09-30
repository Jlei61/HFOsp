"""Paths and units for the rate-model restart; no density state is imported."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/topic4_sef_hfo/fig5_spatial_rate_model_20260917'
SOURCE = ROOT / 'results/topic4_sef_hfo/fig5_current_network_z_state_v1'
GRID = SOURCE / 'approx/coarse_20'
DT = .1  # ms; all internal firing rates are spikes per ms per neuron


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x))
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n')
