"""Simulation glue: network cache, provenance, and one arm run.

Calls the blessed engine and the existing read-out chain; changes neither.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import tempfile

import numpy as np

CONNECTIVITY_FIELDS = (
    "L", "density", "f_E", "seed", "g",
    "C_EE", "C_IE", "C_EI", "C_II",
    "l_EE", "l_IE", "l_EI", "l_II",
    "rho_EE", "rho_IE", "rho_EI", "rho_II",
    "tau0", "v_axon", "delay_dt",
)
TRACKED_MODULES = (
    "src/topic4_core_field.py",
    "src/topic4_core_field_scoring.py",
    "src/topic4_core_field_report.py",
    "src/topic4_core_field_runner.py",
)


def _git(*args, default="unknown"):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return default


def canonical_checksum(obj, drop=("checksum",)):
    """SHA256 of the canonical JSON with `drop` fields removed.

    Verification must recompute from content, never compare a stored string with
    itself (third-review P0-7).
    """
    if isinstance(obj, dict):
        obj = {k: v for k, v in obj.items() if k not in drop}
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def provenance():
    """What code actually produced an artifact."""
    dirty = _git("status", "--porcelain", *TRACKED_MODULES, default="?")
    return dict(
        git_commit=_git("rev-parse", "HEAD"),
        tracked_modules_dirty=(bool(dirty.strip()) if dirty != "?" else None),
        module_sha256={m: hashlib.sha256(open(m, "rb").read()).hexdigest()
                       for m in TRACKED_MODULES if os.path.exists(m)},
        numpy_version=np.__version__,
    )


def connectivity_config(p, theta_deg, ar):
    """Every field that can change the connectivity graph. Keying on
    (seed, theta, L, density, AR) alone would silently hit a stale cache."""
    cfg = {f: getattr(p, f) for f in CONNECTIVITY_FIELDS}
    cfg["theta_EE_deg"] = float(theta_deg)
    cfg["AR"] = float(ar)
    cfg["numpy_version"] = np.__version__
    cfg["rng_bit_generator"] = "PCG64"
    cfg["git_commit"] = _git("rev-parse", "HEAD")
    return cfg


def cache_key(config):
    return canonical_checksum(config, drop=())


def get_network(p, theta_deg, ar, cache_dir):
    """Build or load the connectivity graph.

    Field-independent, so ONE build per (seed, theta) serves every arm. Written
    via a temp file plus atomic rename: Stage 1 parallelises over seeds precisely
    so two workers never race here, and the rename makes a partial file
    impossible even if that assumption is ever broken.
    """
    import sys
    eng = os.path.join("src", "snn_engine")
    for path in (eng, os.getcwd()):
        if path not in sys.path:
            sys.path.insert(0, path)
    from connectivity import place_neurons
    from connectivity_rot import build_connectivity_rot

    cfg = connectivity_config(p, theta_deg, ar)
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


def atomic_write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    with os.fdopen(fd, "w") as fh:
        json.dump(obj, fh)
    os.replace(tmp, path)
