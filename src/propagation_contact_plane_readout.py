# src/propagation_contact_plane_readout.py
"""Subject-specific 2D 传播触点平面读出（real ↔ model 共同语言）。

Spec: docs/superpowers/specs/2026-06-11-propagation-contact-plane-readout-design.md
新增 only: signed-transverse 轴 / normalized 2D 场 / 镜像不变相关 / model-vs-cohort 比较 /
SOZ overlay 匹配。轴框架/端点核/stereotypy/数据加载复用其它模块。
"""
from __future__ import annotations

from typing import Dict, Sequence, Optional, List

import numpy as np

GRID_N = 81
X_LO, X_HI = -0.5, 1.5
Y_EXT = 1.0
S_THRESH = 0.15
OVERLAP_MIN = 25
POOR_PLANARITY_PC1 = 0.80
MIN_CONTACTS = 6
LOW_SUPPORT_FRAC = 0.25


def make_plane_grid(n: int = GRID_N):
    """对称 normalized 平面网格。Y 关于 0 对称（奇数 n 含 y=0 行），
    np.flip(F, axis=0) 精确实现 y -> -y。返回 (X, Y) 各 (n, n)。"""
    x = np.linspace(X_LO, X_HI, n)
    y = np.linspace(-Y_EXT, Y_EXT, n)
    # row index = y, col index = x；flip(axis=0) 翻 y
    Y, X = np.meshgrid(y, x, indexing="ij")
    return X, Y


def signed_transverse_axis(perp_vec: np.ndarray,
                           participating_mask: np.ndarray) -> Dict[str, object]:
    """带符号横向坐标 = 轴外残差在其第一主方向上的投影。

    perp_vec : (n_ch, 3) 轴外残差（= compute_axis_frame 内部 rel - along*u），
               NaN 行允许（非 mapped / 非参与）。
    participating_mask : (n_ch,) bool。

    符号约定（B1，仅供画图）：令参与触点里 |投影| 最大者为正，确定性。
    返回 signed_transverse (n_ch,)（非参与/退化 = NaN）、v_perp (3,)、
    pc1_variance_explained (float)、n_used (int)。
    """
    perp = np.asarray(perp_vec, float)
    part = np.asarray(participating_mask, bool)
    n_ch = perp.shape[0]
    st = np.full(n_ch, np.nan)
    use = part & ~np.isnan(perp).any(axis=1)
    idx = np.where(use)[0]
    if idx.size < 3:
        return {"signed_transverse": st, "v_perp": np.array([np.nan]*3),
                "pc1_variance_explained": float("nan"), "n_used": int(idx.size)}
    P = perp[idx]
    Pc = P - P.mean(axis=0)
    # SVD of centered residuals; 第一主方向 + 方差解释率
    U, S, Vt = np.linalg.svd(Pc, full_matrices=False)
    v1 = Vt[0]
    var_total = float((S ** 2).sum())
    pc1_var = float((S[0] ** 2) / var_total) if var_total > 1e-12 else float("nan")
    proj = P @ v1
    # B1 确定性符号：|proj| 最大触点为正
    anchor = idx[np.argmax(np.abs(proj))]
    sign = 1.0 if (P[np.argmax(np.abs(proj))] @ v1) >= 0 else -1.0
    v1 = v1 * sign
    st[idx] = P @ v1
    return {"signed_transverse": st, "v_perp": v1,
            "pc1_variance_explained": pc1_var, "n_used": int(idx.size)}
