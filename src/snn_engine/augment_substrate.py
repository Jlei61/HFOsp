"""Add the resting E/I neuron layout (allE_xy red, allI_xy blue) to the panel-c/d npz.

The substrate = neuron POSITIONS by type — no simulation needed (place_neurons is
deterministic at seed 1, matching the placement used in the kick/front runs).
Run from repo root: python src/snn_engine/augment_substrate.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params import Params                       # noqa: E402
from connectivity import place_neurons          # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

p = Params(g=3.6, L=3.0, density=1800.0, seed=1)
pos, labels, NE, NI = place_neurons(p, np.random.default_rng(p.seed))
allE = pos[labels == 0]
allI = pos[labels == 1]

for fn in ["kick_snapshots.npz", "front_shapes.npz"]:
    path = os.path.join(DATA, fn)
    d = dict(np.load(path, allow_pickle=True))
    d["allE_xy"] = allE
    d["allI_xy"] = allI
    np.savez(path, **d)
    print(f"augmented {fn}: allE={allE.shape[0]} allI={allI.shape[0]}")
