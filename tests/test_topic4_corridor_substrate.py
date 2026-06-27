import numpy as np
from src.topic4_corridor_substrate import corridor_regions, hub_mask_E


def test_regions_deterministic_and_partition():
    posE = np.c_[np.linspace(-10, 10, 50), np.zeros(50)]
    out = corridor_regions(posE, center=np.array([0., 0]), axis_unit=np.array([1., 0]),
                           half=10.0, corridor_half_frac=0.6, hub_frac=0.1)
    # corridor (along<=6) and global (along>6) partition all 50 cells
    assert set(out["corridor_idx"]) | set(out["global_idx"]) == set(range(50))
    assert set(out["corridor_idx"]) & set(out["global_idx"]) == set()
    # hub is a subset of corridor, near the +edge (split point s=6)
    assert set(out["hub_idx"]).issubset(set(out["corridor_idx"]))
    assert min(out["along"][i] for i in out["hub_idx"]) > 4.0
    # determinism: same inputs -> identical hub
    out2 = corridor_regions(posE, np.array([0., 0]), np.array([1., 0]), 10.0, 0.6, 0.1)
    assert np.array_equal(out["hub_idx"], out2["hub_idx"])


def test_hub_mask():
    m = hub_mask_E(10, [2, 5, 7])
    assert m.sum() == 3 and m[2] and m[5] and m[7] and not m[0]
    assert hub_mask_E(5, []).sum() == 0


def test_axis_unit_normalized_and_centered():
    # non-unit axis vector still works (normalized internally); center at world x=10
    posE = np.c_[np.linspace(0, 20, 41), np.zeros(41)]
    out = corridor_regions(posE, np.array([10., 0]), np.array([3., 0]), 10.0,
                           corridor_half_frac=0.6, hub_frac=0.1)
    # split at along=6 from center 10 -> world x <= 16 is corridor
    assert all(posE[i, 0] <= 16.0 + 1e-9 for i in out["corridor_idx"])
    assert all(posE[i, 0] > 16.0 - 1e-9 for i in out["global_idx"])


def test_at_least_one_hub_when_corridor_nonempty():
    posE = np.c_[np.linspace(-10, 10, 50), np.zeros(50)]
    out = corridor_regions(posE, np.array([0., 0]), np.array([1., 0]), 10.0,
                           corridor_half_frac=0.6, hub_frac=0.0)  # frac 0 -> still >=1
    assert out["hub_idx"].size >= 1


def test_global_gap_creates_buffer_band():
    posE = np.c_[np.linspace(-10, 10, 100), np.zeros(100)]
    out = corridor_regions(posE, np.array([0., 0]), np.array([1., 0]), 10.0,
                           corridor_half_frac=0.5, hub_frac=0.1, global_gap_frac=0.2)
    cor = set(out["corridor_idx"]); glo = set(out["global_idx"])
    assert cor & glo == set()                 # disjoint
    buffer = set(range(100)) - cor - glo      # gap leaves a buffer in neither set
    assert len(buffer) > 0
    # corridor along<=5, global along>7, buffer in (5,7]
    assert all(5.0 < out["along"][i] <= 7.0 + 1e-9 for i in buffer)
    assert max(out["along"][i] for i in out["corridor_idx"]) <= 5.0 + 1e-9
    assert min(out["along"][i] for i in out["global_idx"]) > 7.0 - 1e-9
