# tests/test_propagation_contact_plane_readout.py
import sys
from pathlib import Path
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src import propagation_contact_plane_readout as R


def _perp_from_coords(coords, src_idx, snk_idx):
    """Helper: replicate compute_axis_frame 的轴外残差 perp_vec (n,3)."""
    src_c = np.nanmean(coords[src_idx], axis=0)
    snk_c = np.nanmean(coords[snk_idx], axis=0)
    u = (snk_c - src_c) / np.linalg.norm(snk_c - src_c)
    rel = coords - src_c
    along = rel @ u
    return rel - np.outer(along, u)


def test_signed_transverse_determinism_and_mirror():
    # 触点沿 x 排开，y 方向左右各半（带符号才能分开）
    coords = np.array([[0,0,0],[1,0,0],[2,0,0],[1,1.0,0],[1,-1.0,0]], float)
    part = np.array([True]*5)
    perp = _perp_from_coords(coords, [0], [2])
    out = R.signed_transverse_axis(perp, part)
    st = out["signed_transverse"]
    # 左右两触点符号相反
    assert np.sign(st[3]) == -np.sign(st[4])
    # 确定性：再跑一次一致
    out2 = R.signed_transverse_axis(perp, part)
    assert np.allclose(out["signed_transverse"], out2["signed_transverse"], equal_nan=True)
    # 镜像输入 -> 整体变号（不是随机翻面）
    coords_m = coords.copy(); coords_m[:, 1] *= -1
    perp_m = _perp_from_coords(coords_m, [0], [2])
    out_m = R.signed_transverse_axis(perp_m, part)
    assert np.allclose(np.abs(out_m["signed_transverse"]), np.abs(st), equal_nan=True)


def test_signed_transverse_pc1_variance_and_degenerate():
    # 近一维残差 -> pc1 方差解释率高
    coords = np.array([[0,0,0],[1,0,0],[2,0,0],[1,1,0],[1,-1,0]], float)
    perp = _perp_from_coords(coords, [0], [2])
    out = R.signed_transverse_axis(perp, np.array([True]*5))
    assert out["pc1_variance_explained"] > 0.95
    # 真二维散布 -> pc1 解释率明显 < 1（残差在 y、z 两个方向等量铺开；
    # 注意：轴 = x，所以横向点必须在 y 和 z 上都有分量，否则残差仍共线）
    coords2 = np.array([[0,0,0],[1,0,0],[2,0,0],
                        [1,1,0],[1,-1,0],[1,0,1.0],[1,0,-1.0]], float)
    perp2 = _perp_from_coords(coords2, [0], [2])
    out2 = R.signed_transverse_axis(perp2, np.array([True]*7))
    assert out2["pc1_variance_explained"] < 0.95
    # <3 参与 -> 全 NaN
    out3 = R.signed_transverse_axis(perp, np.array([True, True, False, False, False]))
    assert np.isnan(out3["signed_transverse"]).all()


def test_plane_grid_symmetry():
    X, Y = R.make_plane_grid()
    # y 对称：flip 行后 Y == -Y
    assert np.allclose(np.flip(Y, axis=0), -Y)
    assert X.shape == Y.shape == (R.GRID_N, R.GRID_N)


def test_contact_aggregates_rank_event_size_invariant():
    # 同一空间顺序，两组 event size 不同 -> typical_rank 必须一致（不被 size 污染）
    from src.lagpat_rank_audit import mask_phantom_ranks
    # 3-ch event 与 5-ch event 对同 3 个核心触点给相同顺序
    n_ch = 5
    # event A: 触点 0,1,2 参与，顺序 0<1<2
    # event B: 全 5 个参与，顺序 0<1<2<3<4
    bools = np.zeros((n_ch, 2), bool)
    bools[[0,1,2], 0] = True
    bools[:, 1] = True
    ranks = np.full((n_ch, 2), np.nan)
    ranks[[0,1,2], 0] = [0, 1, 2]
    ranks[:, 1] = [0, 1, 2, 3, 4]
    masked = mask_phantom_ranks(ranks, bools, normalize=True)
    agg = R.contact_aggregates(masked, lag_raw=np.where(bools, ranks, np.nan), bools=bools)
    # 触点 0 在两事件里都是"最早"(归一化 rank=0) -> typical_rank≈0
    assert agg["typical_rank"][0] == pytest.approx(0.0, abs=1e-9)
    # support：触点 3 只在 event B 参与 -> 0.5
    assert agg["support"][3] == pytest.approx(0.5)
    assert agg["support"][0] == pytest.approx(1.0)


def test_contact_aggregates_time_unit_invariant():
    # lag_raw 一份秒、一份 ms(×1000) -> typical_time 场一致（事件内归一化消单位）
    n_ch = 4
    bools = np.ones((n_ch, 1), bool)
    masked = np.array([[0.0],[0.333],[0.667],[1.0]])
    lag_sec = np.array([[0.0],[0.010],[0.020],[0.030]])
    lag_ms = lag_sec * 1000.0
    a = R.contact_aggregates(masked, lag_sec, bools)
    b = R.contact_aggregates(masked, lag_ms, bools)
    assert np.allclose(a["typical_time"], b["typical_time"], equal_nan=True)
    # 归一化后 0..1
    assert a["typical_time"][0] == pytest.approx(0.0)
    assert a["typical_time"][3] == pytest.approx(1.0)
