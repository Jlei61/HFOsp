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


def test_build_readout_record_normalized_coords_and_flags():
    # along_axis_mm + axis_length -> x_norm = along/axis_length; signed_transverse -> y_norm
    n_ch = 7
    names = [f"A{i}" for i in range(n_ch)]
    coords = np.array([[i, 0, 0] for i in range(n_ch)], float)
    coords[3, 1] = 1.0; coords[4, 1] = -1.0   # 两侧横向
    along = np.array([float(i) for i in range(n_ch)])     # 0..6 mm
    axis_length = 6.0
    signed_t = np.array([0,0,0, 1.0, -1.0, 0, 0])
    masked = np.tile(np.linspace(0, 1, n_ch)[:, None], (1, 4))
    bools = np.ones((n_ch, 4), bool)
    rec = R.build_readout_record(
        dataset="yuquan", subject="s1", template_id="t0", names=names,
        along_axis_mm=along, axis_length_mm=axis_length, off_axis_mm=np.zeros(n_ch),
        signed_transverse=signed_t, pc1_variance_explained=0.99,
        masked=masked, lag_raw=masked.copy(), bools=bools,
        soz_first_contacts=set(), lag_time_unit="ms",
        one_dimensional_sampling=False)
    ch = {c["name"]: c for c in rec["channels"]}
    assert ch["A6"]["x_norm"] == pytest.approx(1.0)         # along/axis_length
    assert ch["A3"]["y_norm"] == pytest.approx(1.0 / 6.0)   # signed_t/axis_length
    assert ch["A4"]["y_norm"] == pytest.approx(-1.0 / 6.0)
    assert rec["lag_time_unit"] == "ms"
    assert rec["flags"]["poor_planarity"] is False


def test_build_readout_record_poor_planarity_and_low_contact():
    n_ch = 7
    names = [f"A{i}" for i in range(n_ch)]
    along = np.arange(n_ch, dtype=float)
    masked = np.tile(np.linspace(0, 1, n_ch)[:, None], (1, 4))
    bools = np.ones((n_ch, 4), bool)
    rec = R.build_readout_record(
        dataset="yuquan", subject="s2", template_id="t0", names=names,
        along_axis_mm=along, axis_length_mm=6.0, off_axis_mm=np.zeros(n_ch),
        signed_transverse=np.zeros(n_ch),
        pc1_variance_explained=0.5,                 # < POOR_PLANARITY_PC1
        masked=masked, lag_raw=masked.copy(), bools=bools,
        soz_first_contacts=set(), lag_time_unit="s",
        one_dimensional_sampling=False)
    assert rec["flags"]["poor_planarity"] is True
    # 仅 4 参与触点 (< MIN_CONTACTS=6) -> low_contact_count
    bools_few = np.zeros((n_ch, 4), bool); bools_few[:4, :] = True
    rec2 = R.build_readout_record(
        dataset="yuquan", subject="s3", template_id="t0", names=names,
        along_axis_mm=along, axis_length_mm=6.0, off_axis_mm=np.zeros(n_ch),
        signed_transverse=np.zeros(n_ch), pc1_variance_explained=0.99,
        masked=masked, lag_raw=masked.copy(), bools=bools_few,
        soz_first_contacts=set(), lag_time_unit="s",
        one_dimensional_sampling=False)
    assert rec2["flags"]["low_contact_count"] is True


def test_smooth_field_support_weight_and_gate():
    # 两触点 (x_norm,y_norm,typical_rank,support)，sigma 小 -> 各自附近显色
    rec = {"channels": [
        {"x_norm": 0.1, "y_norm": 0.0, "typical_rank": 0.0, "support": 1.0,
         "uncertainty_rank": 0.0},
        {"x_norm": 0.9, "y_norm": 0.0, "typical_rank": 1.0, "support": 1.0,
         "uncertainty_rank": 0.0},
    ]}
    X, Y = R.make_plane_grid()
    fld = R.smooth_field(rec, X, Y, sigma_xy=0.05, scalar="rank")
    assert fld["T"].shape == X.shape
    # S 是支撑权重（>=0），不是事件率
    assert (fld["S"] >= 0).all()
    # 远离两触点的像素 S 很低 -> 被 gate 掉（mask=False）
    far = (np.abs(X - 0.5) < 0.02) & (np.abs(Y) < 0.02)   # 网格中央、无触点
    assert fld["mask"][far].sum() == 0
    # 触点 0 附近 T≈0，触点 1 附近 T≈1
    near0 = np.unravel_index(np.argmin((X-0.1)**2 + (Y-0)**2), X.shape)
    near1 = np.unravel_index(np.argmin((X-0.9)**2 + (Y-0)**2), X.shape)
    assert fld["T"][near0] == pytest.approx(0.0, abs=0.2)
    assert fld["T"][near1] == pytest.approx(1.0, abs=0.2)


def test_smooth_field_sigma_default_nn_spacing():
    rec = {"channels": [
        {"x_norm": 0.0, "y_norm": 0.0, "typical_rank": 0.0, "support": 1.0,
         "uncertainty_rank": 0.0},
        {"x_norm": 0.3, "y_norm": 0.0, "typical_rank": 1.0, "support": 1.0,
         "uncertainty_rank": 0.0},
    ]}
    X, Y = R.make_plane_grid()
    fld = R.smooth_field(rec, X, Y, sigma_xy=None, scalar="rank")
    # 默认 sigma = 最近邻间距中位数 = 0.3
    assert fld["sigma_xy"] == pytest.approx(0.3, abs=1e-9)


def test_corr_pair_mirror_invariant():
    X, Y = R.make_plane_grid()
    # 一个沿 +y 偏的场
    F1 = np.exp(-((X-0.5)**2 + (Y-0.4)**2)/0.05)
    S1 = F1.copy()
    # F2 = F1 沿 y 翻面（即镜像副本）
    F2 = np.flip(F1, axis=0)
    S2 = np.flip(S1, axis=0)
    out = R.corr_pair_mirror_invariant(F1, S1, F2, S2, s_thresh=0.1, overlap_min=5)
    # 镜像不变：max-over-mirror 必须把翻面的 F2 对回去 -> corr≈1
    assert out["corr"] == pytest.approx(1.0, abs=1e-6)
    assert out["insufficient_overlap"] is False


def test_corr_pair_low_overlap_flag():
    X, Y = R.make_plane_grid()
    # 两个支撑几乎不交叠的场
    F1 = np.exp(-((X-0.1)**2 + Y**2)/0.01); S1 = F1.copy()
    F2 = np.exp(-((X-1.4)**2 + Y**2)/0.01); S2 = F2.copy()
    out = R.corr_pair_mirror_invariant(F1, S1, F2, S2, s_thresh=0.5, overlap_min=25)
    assert out["insufficient_overlap"] is True
    assert out["corr"] is None
