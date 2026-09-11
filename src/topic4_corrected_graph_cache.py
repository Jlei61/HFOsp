"""Explicit corrected graph cache for historical initial-state studies."""
import os
import pickle
import tempfile
import numpy as np
from src.topic4_core_field_runner import cache_key, atomic_write_json
from src.topic4_xy_search import connectivity_config

def get_network(p, theta_deg, ar, cache_dir, *, git_commit=None):
    """Build or load the connectivity graph.

    Field-independent, so ONE build per (seed, theta) serves every arm. Written
    via a temp file plus atomic rename: Stage 1 parallelises over seeds precisely
    so two workers never race here, and the rename makes a partial file
    impossible even if that assumption is ever broken.
    """
    from src.topic4_corrected_connectivity import place_neurons
    from src.topic4_corrected_connectivity_rot import build_connectivity_rot

    cfg = connectivity_config(p, theta_deg, ar, git_commit=git_commit)
    key = cache_key(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        return payload["net"], payload["NE"], payload["NI"], True

    rng = np.random.default_rng(p.seed)
    pos, labels, NE, NI = place_neurons(p, rng)
    net = build_connectivity_rot(p, pos, labels, NE, NI, rng,
                                 theta_EE=np.deg2rad(theta_deg), AR=ar, verbose=False)
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    with os.fdopen(fd, "wb") as fh:
        pickle.dump({"net": net, "NE": NE, "NI": NI, "config": cfg},
                    fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return net, NE, NI, False
