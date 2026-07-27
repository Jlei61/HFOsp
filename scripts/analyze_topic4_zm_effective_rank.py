#!/usr/bin/env python
"""Aggregate completed Task-9A probe manifests across seeds and microstates."""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime

import numpy as np

_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SCRIPTS)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.topic4_zm_effective_rank as ER  # noqa: E402

OUT = os.path.join(_ROOT, "results", "topic4_sef_hfo", "zm_branch_decision")


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".effective_rank.", suffix=".json",
                               dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _matrices(manifest, state_tag):
    rows = [
        r for r in manifest["rows"]
        if r["state_tag"] == state_tag and r.get("response_valid")
    ]
    coords = tuple(manifest["coordinate_order"])
    if len(rows) != 2 * len(coords):
        raise ValueError(f"seed {manifest['seed']} {state_tag}: incomplete central pairs")
    q = np.asarray(manifest["q_scales"], float)
    y = np.asarray(manifest["y_scales"], float)
    static = ER.assemble_paired_sensitivity(
        [{**r, "y": r["y_static"]} for r in rows], coords
    )
    impulse = ER.assemble_paired_sensitivity(
        [{**r, "y": r["y_impulse"]} for r in rows], coords
    )
    n_time = int(rows[0]["n_time_bins"])
    return (
        ER.standardize_sensitivity(static, q, y),
        ER.standardize_sensitivity(impulse, q, np.tile(y, n_time)),
    )


def main():
    paths = sorted(glob.glob(os.path.join(
        OUT, "effective_rank", "seed*", "rank_probes.json"
    )))
    manifests = [json.load(open(p)) for p in paths]
    manifests = [
        m for m in manifests
        if m.get("complete") is True
        and m.get("effective_rank_version") == ER.EFFECTIVE_RANK_VERSION
    ]
    if len(manifests) < 2:
        raise SystemExit("effective-rank aggregation requires two complete seeds")
    state_tags = tuple(manifests[0]["state_tags"])
    if any(tuple(m["state_tags"]) != state_tags for m in manifests):
        raise RuntimeError("effective-rank manifests use different state sets")

    static_by_seed, impulse_by_seed = [], []
    per_state = {}
    per_seed_state = {}
    for manifest in manifests:
        seed = int(manifest["seed"])
        per_seed_state[seed] = {}
        for state_tag in state_tags:
            per_seed_state[seed][state_tag] = _matrices(
                manifest, state_tag
            )

    for state_tag in state_tags:
        st, it = [], []
        for manifest in manifests:
            a, b = per_seed_state[int(manifest["seed"])][state_tag]
            st.append(a)
            it.append(b)
        per_state[state_tag] = {
            "static_seed_matrices": [x.tolist() for x in st],
            "impulse_matrix_shape": list(it[0].shape),
            "static_point": ER.rank_summary(np.mean(st, axis=0)),
            "impulse_point": ER.rank_summary(np.mean(it, axis=0)),
        }

    for manifest in manifests:
        seed = int(manifest["seed"])
        static_by_seed.append([
            per_seed_state[seed][state_tag][0]
            for state_tag in state_tags
        ])
        impulse_by_seed.append([
            per_seed_state[seed][state_tag][1]
            for state_tag in state_tags
        ])

    static_boot = ER.hierarchical_bootstrap_rank(
        np.asarray(static_by_seed), n_boot=2000, seed=271
    )
    impulse_boot = ER.hierarchical_bootstrap_rank(
        np.asarray(impulse_by_seed), n_boot=2000, seed=272
    )
    if static_boot["rank1_supported"] and impulse_boot["rank1_supported"]:
        verdict = "near_rank1_local_functional_collinearity"
    elif static_boot["rank1_supported"] != impulse_boot["rank1_supported"]:
        verdict = "mixed_static_impulse_rank"
    else:
        verdict = "rank2_or_higher_local_directions"

    payload = {
        "effective_rank_version": ER.EFFECTIVE_RANK_VERSION,
        "analysis_git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT, text=True
        ).strip(),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "verdict": verdict,
        "n_seeds": len(manifests),
        "seeds": sorted(int(m["seed"]) for m in manifests),
        "n_seed_microstates": len(manifests) * len(state_tags),
        "bootstrap_structure": "hierarchical_seed_then_microstate",
        "static_bootstrap": static_boot,
        "impulse_bootstrap": impulse_boot,
        "per_state": per_state,
        "sources": [
            {"path": os.path.relpath(path, _ROOT), "sha256": _sha(path)}
            for path in paths
            if any(int(m["seed"]) == int(os.path.basename(os.path.dirname(path))[4:])
                   for m in manifests)
        ],
        "claim_boundary": (
            "local standardized functional rank near confirmed carrier states; "
            "not global slow-manifold dimensionality"
        ),
    }
    out = os.path.join(OUT, "effective_rank", "effective_rank_summary.json")
    _atomic(out, payload)
    print(
        f"[effective_rank] verdict={verdict} "
        f"samples={len(manifests) * len(state_tags)} "
        f"-> {os.path.relpath(out, _ROOT)}"
    )


if __name__ == "__main__":
    main()
