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


def contact_aggregates(masked: np.ndarray, lag_raw: np.ndarray,
                       bools: np.ndarray) -> Dict[str, np.ndarray]:
    """每触点时序聚合。

    masked : (n_ch, n_ev) 事件内归一化 masked rank（= mask_phantom_ranks(normalize=True)），
             非参与 = NaN。这就是 spec 的 rank_norm，不再二次归一化。
    lag_raw : (n_ch, n_ev) 原始 lag（秒或 ms，单位无关），非参与 = NaN。
    bools : (n_ch, n_ev) bool。

    返回各 (n_ch,) 数组：
      typical_rank   = nanmedian_e masked            （主，进比较）
      typical_time   = nanmedian_e lag_norm          （副，仅画图）
      support        = 参与事件数 / 总事件数
      uncertainty_rank = nan-IQR_e masked
      uncertainty_time = nan-IQR_e lag_norm
    其中 lag_norm(c,e) = (lag-min)/max(lag-min) 在每事件参与触点内（min/max 消单位）。
    """
    masked = np.asarray(masked, float)
    lag_raw = np.asarray(lag_raw, float)
    bools = np.asarray(bools, bool)
    n_ch, n_ev = masked.shape
    # 事件内 lag 归一化（仅参与触点）
    lag_norm = np.full_like(lag_raw, np.nan)
    for e in range(n_ev):
        idx = np.where(bools[:, e])[0]
        if idx.size == 0:
            continue
        v = lag_raw[idx, e]
        vmin = np.nanmin(v)
        rel = v - vmin
        rmax = np.nanmax(rel)
        lag_norm[idx, e] = rel / rmax if rmax > 1e-12 else 0.0

    def _nan_iqr(a, axis):
        with np.errstate(invalid="ignore"):
            q75 = np.nanpercentile(a, 75, axis=axis)
            q25 = np.nanpercentile(a, 25, axis=axis)
        return q75 - q25

    with np.errstate(invalid="ignore"):
        typ_rank = np.nanmedian(masked, axis=1)
        typ_time = np.nanmedian(lag_norm, axis=1)
    support = bools.sum(axis=1).astype(float) / max(n_ev, 1)
    return {
        "typical_rank": typ_rank,
        "typical_time": typ_time,
        "support": support,
        "uncertainty_rank": _nan_iqr(masked, 1),
        "uncertainty_time": _nan_iqr(lag_norm, 1),
    }


def build_readout_record(
    *, dataset: str, subject: str, template_id: str, names: Sequence[str],
    along_axis_mm: np.ndarray, axis_length_mm: float, off_axis_mm: np.ndarray,
    signed_transverse: np.ndarray, pc1_variance_explained: float,
    masked: np.ndarray, lag_raw: np.ndarray, bools: np.ndarray,
    soz_first_contacts: set, lag_time_unit: str,
    one_dimensional_sampling: bool,
) -> Dict[str, object]:
    """组装一份标准化 readout record（real / model 同构）。

    along_axis_mm / off_axis_mm 来自 compute_axis_frame；signed_transverse 来自
    signed_transverse_axis；masked = mask_phantom_ranks(normalize=True)。
    x_norm = along/axis_length，y_norm = signed_transverse/axis_length（spec §3 双坐标系）。
    每触点一条（仅 along/signed 非 NaN 的参与触点）。flags 见 spec §10。
    SOZ overlay 仅描述性：标 is_soz（first-contact alias 在 soz_first_contacts 内）。
    """
    from src.propagation_skeleton_geometry import parse_shaft
    from src.sef_hfo_soz_localization import _first_contact

    along = np.asarray(along_axis_mm, float)
    st = np.asarray(signed_transverse, float)
    agg = contact_aggregates(masked, lag_raw, bools)
    L = float(axis_length_mm)
    channels: List[dict] = []
    for i, nm in enumerate(names):
        a_i, s_i = float(along[i]), float(st[i])
        if not (np.isfinite(a_i) and np.isfinite(s_i)) or L < 1e-9:
            continue
        if agg["support"][i] <= 0:
            continue
        channels.append({
            "name": str(nm),
            "shaft": str(parse_shaft(nm)[0]),
            "along_axis_mm": a_i,
            "signed_transverse_mm": s_i,
            "off_axis_mm": float(off_axis_mm[i]),
            "x_norm": a_i / L,
            "y_norm": s_i / L,
            "typical_rank": float(agg["typical_rank"][i]),
            "typical_time": float(agg["typical_time"][i]),
            "support": float(agg["support"][i]),
            "uncertainty_rank": float(agg["uncertainty_rank"][i]),
            "is_soz": _first_contact(str(nm)) in soz_first_contacts,
        })
    med_support = float(np.median([c["support"] for c in channels])) if channels else 0.0
    flags = {
        "one_dimensional_sampling": bool(one_dimensional_sampling),
        "poor_planarity": bool(np.isfinite(pc1_variance_explained)
                               and pc1_variance_explained < POOR_PLANARITY_PC1),
        "low_contact_count": len(channels) < MIN_CONTACTS,
        "low_support": med_support < LOW_SUPPORT_FRAC,
        "weak_axis": L < 1e-9,
    }
    return {
        "dataset": dataset, "subject": subject, "template_id": template_id,
        "axis_length_mm": L,
        "transverse_pc1_variance_explained": float(pc1_variance_explained),
        "lag_time_unit": lag_time_unit,
        "channels": channels,
        "flags": flags,
        "n_channels": len(channels),
    }
