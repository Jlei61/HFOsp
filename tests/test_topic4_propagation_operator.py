import numpy as np
from src.topic4_propagation_operator import (make_step_operator, make_shape_operator,
    spectral_radius, h_field, principal_axis, ordering_predictivity)


def test_h_from_unnormalized_resp_not_flat():
    # W_resp 未归一 -> h_post 反映真实招募强度差异 (不被 row-norm 抹平到 1)
    # 行和 [6,1,1]: row-norm 实现会产生 [1,1,1]/median≈[1,1,1], 无法通过此断言
    W = np.array([[0, 4., 2.], [1., 0, 0.], [0, 1., 0]])    # 行和 = [6,1,1]
    h = h_field(W, "post")
    assert h[0] != h[1]                              # 不抹平
    assert np.allclose(h, np.array([6., 1., 1.]) / np.median([6., 1., 1.]))


def test_step_operator_scales_with_gain_not_rownorm():
    # W_step 按源活动归一; 整体放大 W_resp -> ρ 增大 (若是 row-norm 则 ρ 恒 1 = bug)
    # 必须用「有向环」recurrent 矩阵 (非前馈/幂零, 否则特征值恒 0、branching_ratio 测不出增益)
    W = np.array([[0, 2., 0], [0, 0, 2.], [2., 1., 0]]); sm = np.array([1., 1, 1])
    rho1 = spectral_radius(make_step_operator(W, sm))
    rho2 = spectral_radius(make_step_operator(2 * W, sm))
    assert rho2 > rho1 + 1e-6                        # ρ 跟增益走 (非 row-norm 恒定)


def test_shape_operator_rownormalized():
    W = np.array([[0, 1., 3.], [0, 0, 0.], [0, 0, 0]])
    S = make_shape_operator(W)
    rs = S.sum(1); assert np.allclose(rs[rs > 0], 1.0) and np.allclose(np.diag(S), 0)


def test_ordering_predictivity_W_beats_distance():
    centers = np.array([[0, 0], [1, 5], [2, 0.], [3, 5]])
    # [target, source] 约定: 源 0->目标 1->目标 2->目标 3 的链 => W[b,a]>0 下三角
    Wshape = np.array([[0, 0, 0, 0], [1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 1., 0]])
    out = ordering_predictivity(Wshape, centers, [0, 1, 2, 3], rates=np.ones(4))
    assert out["rho_W"] >= out["rho_dist"]


def test_ordering_predictivity_direction_convention():
    # 合同: W_resp[p,q] = target p <- source q. 源 0->目标 1, 源 1->目标 2 的纯链;
    # 事件顺序 [0,1,2] 必须被 W 完美预测 (rho_W == 1). 把 row 当 source (方向反) 会失败.
    W = np.zeros((3, 3)); W[1, 0] = 1.; W[2, 1] = 1.        # [target, source]
    centers = np.array([[0, 0], [1, 0.], [2, 0]])
    out = ordering_predictivity(make_shape_operator(W), centers, [0, 1, 2], rates=np.ones(3))
    assert out["rho_W"] == 1.0


def test_ordering_predictivity_transposed_W_loses():
    # 方向敏感: 正向 W 完美预测 [0,1,2]; 转置 W (方向反了) 不应再完美预测.
    Wf = np.zeros((3, 3)); Wf[1, 0] = 1.; Wf[2, 1] = 1.
    centers = np.array([[0, 0], [1, 0.], [2, 0]])
    rho_fwd = ordering_predictivity(make_shape_operator(Wf), centers, [0, 1, 2], rates=np.ones(3))["rho_W"]
    rho_rev = ordering_predictivity(make_shape_operator(Wf.T), centers, [0, 1, 2], rates=np.ones(3))["rho_W"]
    assert rho_fwd == 1.0 and rho_rev < 1.0


def test_step_operator_excludes_low_src_mass():
    # 防爆 (审阅 P1): 低 src_mass 源 bin -> 整列置 0, 不被小分母放大成假高增益
    W = np.array([[0, 2., 0], [0, 0, 2.], [0, 0, 0]])
    sm = np.array([1., 1e-9, 1.])                    # bin 1 = 不可靠源
    S = make_step_operator(W, sm, src_mass_floor=1e-3)
    assert np.allclose(S[:, 1], 0.0)                 # 排除, 而非 2/1e-9 爆掉


def test_step_operator_injected_mass_sensitivity():
    # sensitivity 口径: 除以注入期望 spike mass, 不是 src_mass
    W = np.array([[0, 2., 0], [0, 0, 0.], [0, 0, 0]]); sm = np.array([1., 1, 1]); inj = np.array([4., 4, 4])
    S = make_step_operator(W, sm, injected_mass=inj)
    assert np.isclose(S[0, 1], 0.5)                   # 2/4, 用 injected 分母


def test_principal_axis_aligns_with_dominant_displacement():
    # 边界: 所有传播位移沿 x -> 主轴 ~ [±1, 0]
    centers = np.array([[0, 0], [1, 0.], [2, 0]])
    W = np.zeros((3, 3)); W[1, 0] = 1.; W[2, 1] = 1.   # 位移都是 [1,0]
    axis = principal_axis(W, centers)
    assert abs(axis[0]) > 0.99 and abs(axis[1]) < 0.01


def test_h_field_degenerate_all_zero_no_crash():
    # 边界: 全零 W -> h 全零, 不崩 (median 退化保护)
    assert np.allclose(h_field(np.zeros((3, 3)), "post"), 0.0)


def test_step_operator_all_below_floor_zeroed():
    # 边界: 全部源都低于 floor -> W_step 全零 (不放大任何假增益)
    W = np.array([[0, 2., 0], [0, 0, 2.], [2., 0, 0]]); sm = np.full(3, 1e-9)
    assert np.allclose(make_step_operator(W, sm, src_mass_floor=1e-3), 0.0)
