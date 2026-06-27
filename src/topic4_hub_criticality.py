"""Topic 4 M3 — structural criticality probe (no dynamics run).

朴素版：从一张已经建好的 spiking 网络"接线图"里，不跑任何放电动力学，只用线性代数
估一个"分支比"——把"现在着火的细胞"映射到"下一步会被它们点着的细胞"的线性算子的
最大特征值。> 1 意味着一个活跃细胞平均点着多于一个邻居（活动会扩散/超临界），< 1
意味着活动会熄灭（亚临界）。我们额外沿一条 corridor -> hub -> global 的通路强制活动
"必须穿过 hub"才能到 global，量化这条跨区通路的有效分支比。

输入是 build 出来的连接矩阵（AMPA E->E 子块）+ 每个细胞离阈值的距离；离阈值越近的目标
细胞越容易被点着（gap_factor 越大）。整模块纯 numpy + scipy.sparse，不加载任何引擎。
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

_EPS = 1e-6


def recruitment_operator(net, V_th, NE, drive_rest, link_cap=10.0):
    """Linearized recruitment operator M (csr, NE x NE), x_{t+1} ~ M @ x_t.

    M[j, i] = W[j, i] * gap_factor(V_th[j]), W = (sum of ampa_by_delay)[:NE, :NE]
    (row=target j, col=source i). gap_factor(vth) = clip(1/max(vth-drive_rest, eps), 0, cap):
    a target closer to its threshold is recruited more easily by an active source.
    """
    mats = net["ampa_by_delay"]
    A = mats[0]
    for m in mats[1:]:
        A = A + m
    A = sp.csr_matrix(A)
    W = A[:NE, :NE].tocsr()

    vth_e = np.asarray(V_th, float)[:NE]
    gap = 1.0 / np.maximum(vth_e - drive_rest, _EPS)
    gap = np.clip(gap, 0.0, link_cap)  # gap_factor per target j

    # M[j, i] = W[j, i] * gap[j]; scale each row j by gap[j], preserve W's sparsity.
    D = sp.diags(gap)
    M = (D @ W).tocsr()
    return M


def _largest_real_eig(Msub):
    """max real-part eigenvalue of a sparse submatrix; dense for tiny, eigs for larger."""
    n = Msub.shape[0]
    if n == 0:
        return 0.0
    if Msub.nnz == 0:
        return 0.0
    if n <= 3:
        vals = np.linalg.eigvals(Msub.toarray())
        return float(np.max(np.real(vals)))
    try:
        # k=1 LM eigenvalue; eigs needs k < n-1 (n > 3 guaranteed here).
        vals = spla.eigs(Msub.astype(float), k=1, which="LM", return_eigenvectors=False)
        return float(np.real(vals[0]))
    except Exception:
        vals = np.linalg.eigvals(Msub.toarray())
        return float(np.max(np.real(vals)))


def branching_ratio(M, idx=None):
    """Largest-magnitude eigenvalue's real part of M restricted to idx (or all if None).

    Returns 0.0 if the submatrix is empty or all-zero.
    """
    M = sp.csr_matrix(M)
    if idx is None:
        Msub = M
    else:
        ix = np.asarray(idx, dtype=int)
        if ix.size == 0:
            return 0.0
        Msub = M[ix][:, ix]
    return _largest_real_eig(sp.csr_matrix(Msub))


def crossing_branching(M, corridor_idx, hub_idx, global_idx):
    """Effective branching ratio along corridor -> hub -> global, forcing the hub path.

    Restrict M to ix = sorted(corridor | hub | global), then zero every direct
    corridor->global edge (M_r[a, b]=0 when global-index(a) in global_idx and
    global-index(b) in corridor_idx) so global cells are only recruitable via the hub.
    Returns 0.0 if hub_idx is empty.
    """
    hub_idx = list(hub_idx)
    if len(hub_idx) == 0:
        return 0.0

    corridor_set = set(int(c) for c in corridor_idx)
    global_set = set(int(g) for g in global_idx)

    ix = sorted(set(int(c) for c in corridor_idx)
                | set(int(h) for h in hub_idx)
                | global_set)
    M = sp.csr_matrix(M)
    Msub = sp.csr_matrix(M[np.asarray(ix, dtype=int)][:, np.asarray(ix, dtype=int)]).tolil()

    # zero direct corridor-source -> global-target edges (a=target, b=source).
    for a_local, a_global in enumerate(ix):
        if a_global in global_set:
            for b_local, b_global in enumerate(ix):
                if b_global in corridor_set:
                    Msub[a_local, b_local] = 0.0

    return branching_ratio(sp.csr_matrix(Msub))


def sigma_phase_map(build_fn, alpha_grid, gain_grid, regions, V_th0, NE, drive_rest, degnorm_fn):
    """(alpha x gain) phase map of corridor and crossing branching ratios.

    For each (alpha, gain): net=build_fn(gain); V_th=V_th0+degnorm_fn(net, alpha);
    M=recruitment_operator(net, V_th, NE, drive_rest); record branching_ratio over the
    corridor and crossing_branching over corridor->hub->global.
    """
    alpha_grid = np.asarray(alpha_grid)
    gain_grid = np.asarray(gain_grid)
    V_th0 = np.asarray(V_th0, float)

    na, ng = len(alpha_grid), len(gain_grid)
    sigma_corridor = np.zeros((na, ng))
    sigma_crossing = np.zeros((na, ng))

    corridor_idx = regions["corridor_idx"]
    hub_idx = regions["hub_idx"]
    global_idx = regions["global_idx"]

    for ai, alpha in enumerate(alpha_grid):
        for gi, gain in enumerate(gain_grid):
            net = build_fn(gain)
            V_th = V_th0 + degnorm_fn(net, alpha)
            M = recruitment_operator(net, V_th, NE, drive_rest)
            sigma_corridor[ai, gi] = branching_ratio(M, corridor_idx)
            sigma_crossing[ai, gi] = crossing_branching(M, corridor_idx, hub_idx, global_idx)

    return dict(alpha_grid=alpha_grid, gain_grid=gain_grid,
                sigma_corridor=sigma_corridor, sigma_crossing=sigma_crossing)


def crossing_path_gain(M, corridor_idx, hub_idx, global_idx):
    """Two-stage corridor->hub->global PATH gain (directional reachability, NOT an eigenvalue).

    朴素版：corridor 内部的递归很强，所以"corridor∪hub∪global 整体的最大特征值"几乎就等于
    corridor 自己的分支比，量不出 hub 这个瓶颈。"能不能跨过 hub"是一个**有方向**的问题：
    假设 corridor 已经着火，活动能不能经过 hub 漏到 global。跨越需要两段都通：
      hub_recruit   = 流进 hub 细胞、来自 corridor 的 gap 加权驱动的均值（hub 阈值越高 = degnorm
                      alpha 越大，这段越关）
      hub_broadcast = 流进 global 细胞、来自 hub 的 gap 加权驱动的均值（hub 长程广播增益越大越通）
      gain          = hub_recruit * hub_broadcast（两段都开才跨得过去）
    M[j,i] 已经把目标 j 的 gap_factor 烘进去了，所以某区"收到的 gap 加权驱动"就是对应子块之和。
    Returns dict(gain, hub_recruit, hub_broadcast)."""
    Mc = M.tocsr()
    cor = np.asarray(corridor_idx, int)
    hub = np.asarray(hub_idx, int)
    glo = np.asarray(global_idx, int)
    if hub.size == 0 or cor.size == 0 or glo.size == 0:
        return dict(gain=0.0, hub_recruit=0.0, hub_broadcast=0.0)
    hub_recruit = float(Mc[hub][:, cor].sum()) / hub.size
    hub_broadcast = float(Mc[glo][:, hub].sum()) / glo.size
    return dict(gain=hub_recruit * hub_broadcast, hub_recruit=hub_recruit, hub_broadcast=hub_broadcast)
