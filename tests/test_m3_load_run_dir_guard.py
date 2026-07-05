"""P1-2: load_run_dir() must FAIL CLOSED against stale/mixed artifacts.

The 4x4->5x5 grid bug (2026-06-24) slipped in because n_bins lives in thresholds.json,
not config.sweep_parameters. The assembly side had no guard, so --assemble-only could
silently ingest a stale 4x4 run or a mixed npz/CSV. These tests pin the guards:
  - thresholds.json n_bins == expected (25 = the 5x5 working point)
  - npz n_bins == thresholds n_bins (consistency)
  - ea_net_bins.shape[2] == n_bins
  - npz kicks == sorted CSV kicks
  - src_bin_idx == argmin(bin_centers, kick_xy)
"""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import sef_hfo_mini_w_event as mwe  # noqa: E402


def _grid_centers(nb, L=20.0):
    c = (np.arange(nb) + 0.5) * (L / nb)
    return np.array([[c[j], c[i]] for i in range(nb) for j in range(nb)], dtype=float)


def _make_run(path, *, thr_n_bins=25, npz_n_bins=25, ea_bins=25, grid_nb=5,
              npz_kicks=(0.8, 1.0, 1.2), csv_kicks=(0.8, 1.0, 1.2),
              src_bin_idx=12, kick_xy=(10.0, 10.0), n_seed=2):
    os.makedirs(path, exist_ok=True)
    # per_seed_metrics.csv (minimal columns _per_seed_ea + _per_seed_core_only need)
    cols = ["kick_boost", "win_lo", "win_hi", "seed", "r95_mm_ea", "far_field_frac_ea",
            "returned", "runaway", "t0_ms", "core_only_downstream_resp",
            "no_core_no_kick_downstream", "seed_local_returned"]
    rows = ["\t".join(cols).replace("\t", ",")]
    for k in csv_kicks:
        for s in range(n_seed):
            rows.append(",".join(str(x) for x in
                        [k, 20.0, 28.0, s, 3.0, 0.1, 1, 0, 120.0, 5.0, 4.0, 1]))
    with open(os.path.join(path, "per_seed_metrics.csv"), "w") as f:
        f.write("\n".join(rows) + "\n")
    with open(os.path.join(path, "thresholds.json"), "w") as f:
        json.dump({"n_bins": thr_n_bins, "L": 20.0}, f)
    np.savez(os.path.join(path, "ea_net_bins.npz"),
             kicks=np.asarray(npz_kicks, float),
             seeds=np.arange(n_seed),
             ea_net_bins=np.ones((len(npz_kicks), n_seed, ea_bins), float),
             bin_idx=src_bin_idx, src_bin_idx=src_bin_idx,
             bin_centers=_grid_centers(grid_nb), n_bins=npz_n_bins,
             core_mean=17.6, kick_xy=np.asarray(kick_xy, float))


def test_valid_5x5_run_loads(tmp_path):
    d = str(tmp_path / "ok")
    _make_run(d)
    out = mwe.load_run_dir(d)               # default expected_n_bins=25
    assert out["src_bin_idx"] == 12
    assert out["ea_net_bins"].shape[2] == 25


def test_stale_4x4_run_raises(tmp_path):
    d = str(tmp_path / "stale4x4")
    _make_run(d, thr_n_bins=16, npz_n_bins=16, ea_bins=16, grid_nb=4, src_bin_idx=10)
    with pytest.raises(ValueError, match="n_bins"):
        mwe.load_run_dir(d)


def test_mixed_npz_thresholds_nbins_raises(tmp_path):
    d = str(tmp_path / "mixed")
    _make_run(d, thr_n_bins=25, npz_n_bins=16, ea_bins=16, grid_nb=5)
    with pytest.raises(ValueError):
        mwe.load_run_dir(d)


def test_ea_shape_mismatch_raises(tmp_path):
    d = str(tmp_path / "shape")
    _make_run(d, thr_n_bins=25, npz_n_bins=25, ea_bins=16, grid_nb=5)
    with pytest.raises(ValueError, match="shape"):
        mwe.load_run_dir(d)


def test_npz_kicks_disagree_with_csv_raises(tmp_path):
    d = str(tmp_path / "kicks")
    _make_run(d, npz_kicks=(0.8, 1.0), csv_kicks=(0.8, 1.0, 1.2))
    with pytest.raises(ValueError, match="kick"):
        mwe.load_run_dir(d)


def test_src_bin_not_argmin_of_kick_xy_raises(tmp_path):
    d = str(tmp_path / "srcbin")
    _make_run(d, src_bin_idx=0, kick_xy=(10.0, 10.0))   # argmin([10,10]) is bin 12, not 0
    with pytest.raises(ValueError, match="src_bin"):
        mwe.load_run_dir(d)
