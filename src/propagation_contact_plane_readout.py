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


def _median_nn_spacing(pts: np.ndarray) -> float:
    """最近邻间距中位数（normalized plane）。"""
    if pts.shape[0] < 2:
        return 0.1
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(axis=1)))


def smooth_field(record: Dict, X: np.ndarray, Y: np.ndarray,
                 sigma_xy: Optional[float] = None, scalar: str = "rank",
                 s_thresh: float = S_THRESH) -> Dict[str, object]:
    """normalized 平面上的 support 加权 kernel regression。

    每触点提供 (x_norm, y_norm, value, support)，value = typical_rank(scalar='rank')
    或 typical_time('time')。
      w_i(x,y) = support_i * exp(-((x-x_i)^2+(y-y_i)^2)/(2 sigma^2))
      T = Σ w_i value_i / Σ w_i ；S = Σ w_i（支撑权重，NOT 事件率）；
      U = Σ w_i unc_i / Σ w_i ；mask = S >= s_thresh。
    返回 T/S/U/mask（均 (n,n)）+ sigma_xy。
    """
    val_key = "typical_rank" if scalar == "rank" else "typical_time"
    # 过滤：坐标有限 AND 该标量有限 AND support>0。否则 NaN-value 通道仍会把权重
    # 加进 S 分母却不进 T 分子 -> 稀释 T（reviewer P1）。
    chans = [c for c in record["channels"]
             if np.isfinite(c["x_norm"]) and np.isfinite(c["y_norm"])
             and np.isfinite(c.get(val_key, np.nan)) and c.get("support", 0) > 0]
    pts = np.array([[c["x_norm"], c["y_norm"]] for c in chans], float).reshape(-1, 2)
    vals = np.array([c[val_key] for c in chans], float)
    sup = np.array([c["support"] for c in chans], float)
    unc = np.array([c.get("uncertainty_rank", 0.0) for c in chans], float)
    if sigma_xy is None:
        sigma_xy = _median_nn_spacing(pts)
    sig2 = 2.0 * sigma_xy ** 2
    gx = X.ravel(); gy = Y.ravel()
    S = np.zeros(gx.shape); WT = np.zeros(gx.shape); WU = np.zeros(gx.shape)
    for k in range(pts.shape[0]):
        d2 = (gx - pts[k, 0]) ** 2 + (gy - pts[k, 1]) ** 2
        w = sup[k] * np.exp(-d2 / sig2)
        S += w
        if np.isfinite(vals[k]):
            WT += w * vals[k]
        WU += w * unc[k]
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(S > 1e-12, WT / S, np.nan).reshape(X.shape)
        U = np.where(S > 1e-12, WU / S, np.nan).reshape(X.shape)
    S = S.reshape(X.shape)
    mask = S >= s_thresh
    return {"T": T, "S": S, "U": U, "mask": mask, "sigma_xy": float(sigma_xy)}


def _support_corr(F1, F2, S1, S2, s_thresh):
    """两场在双方 support>=thresh 像素交集上的 Pearson 相关 + 交集像素数。"""
    m = (S1 >= s_thresh) & (S2 >= s_thresh) & np.isfinite(F1) & np.isfinite(F2)
    n = int(m.sum())
    if n < 2:
        return float("nan"), n
    a = F1[m]; b = F2[m]
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan"), n
    return float(np.corrcoef(a, b)[0, 1]), n


def corr_pair_mirror_invariant(F1, S1, F2, S2, s_thresh: float = S_THRESH,
                               overlap_min: int = OVERLAP_MIN) -> Dict[str, object]:
    """y-reflection invariant、support-gated 场相关（spec §4 P0 lock）。

    signed-y 只是图上稳定坐标，不是解剖左右；所以取
    corr = max(corr(F1,F2), corr(F1, flip_y(F2)))。
    仅在双方 S>=s_thresh 像素交集上算；交集 < overlap_min -> corr=None +
    insufficient_overlap=True。网格须 y 对称（make_plane_grid 保证），故
    np.flip(axis=0) == y -> -y。
    """
    c_id, n_id = _support_corr(F1, F2, S1, S2, s_thresh)
    F2m = np.flip(F2, axis=0); S2m = np.flip(S2, axis=0)
    c_mir, n_mir = _support_corr(F1, F2m, S1, S2m, s_thresh)
    n_overlap = max(n_id, n_mir)
    if n_overlap < overlap_min:
        return {"corr": None, "n_overlap": n_overlap, "insufficient_overlap": True}
    cands = [c for c in (c_id, c_mir) if np.isfinite(c)]
    if not cands:
        return {"corr": None, "n_overlap": n_overlap, "insufficient_overlap": True}
    return {"corr": float(max(cands)), "n_overlap": n_overlap,
            "insufficient_overlap": False}


def placement_in_distribution(value: float, dist: Sequence[float]) -> Dict[str, float]:
    """value 落在 dist 的 percentile + robust z（median + MAD）。不报 p 值（spec §9）。"""
    d = np.asarray([x for x in dist if np.isfinite(x)], float)
    if d.size == 0 or not np.isfinite(value):
        return {"percentile": float("nan"), "robust_z": float("nan"), "n": int(d.size)}
    pct = float((d < value).mean() * 100.0)
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med)))
    rz = float((value - med) / (1.4826 * mad)) if mad > 1e-12 else float("nan")
    return {"percentile": pct, "robust_z": rz, "n": int(d.size)}


def subject_first_fold(records: Sequence[dict], key: str) -> List[float]:
    """多模板 subject 先折叠为一个代表值（同 subject 的 key 取中位），再返回每 subject 一个值。
    防 cohort 汇总里模板多的 subject 被重复计数（spec §9 聚合纪律）。"""
    by_subj: Dict[tuple, list] = {}
    for r in records:
        v = r.get(key)
        if v is None or not np.isfinite(v):
            continue
        by_subj.setdefault((r["dataset"], r["subject"]), []).append(float(v))
    return [float(np.median(vs)) for vs in by_subj.values()]


def compare_model_to_cohort(model_record: dict, real_records: Sequence[dict],
                            X: np.ndarray, Y: np.ndarray,
                            sigma_xy: Optional[float] = None,
                            s_thresh: float = S_THRESH,
                            overlap_min: int = OVERLAP_MIN) -> Dict[str, object]:
    """spec §9 描述性 posterior-predictive 比较。

    (a) 标量：对每个 real-vs-model cohort scalar，real subject-first 折叠成分布，
        model 取 placement_in_distribution（percentile + robust z）。
    (b) field：先给每个 record 算 rank 场；real_to_real(i)=median_{j≠i} corr_pair；
        model_to_real=median_i corr_pair(model,i)；报 model 值落在
        {real_to_real} 分布的 percentile/z。镜像不变 + support-gated。
    禁报 p 值。
    """
    SCALARS = ["axis_length_mm", "transverse_width_mm", "early_zone_spread",
               "late_zone_spread", "early_late_centroid_distance_norm",
               "rank_vs_xnorm_spearman"]
    # (a) 标量
    scalar_placement = {}
    for s in SCALARS:
        dist = subject_first_fold(
            [{"dataset": r["dataset"], "subject": r["subject"],
              "scalar": r.get("scalars", {}).get(s)} for r in real_records],
            key="scalar")
        mv = model_record.get("scalars", {}).get(s)
        if mv is not None and dist:
            scalar_placement[s] = placement_in_distribution(float(mv), dist)
    # (b) field —— subject-first（reviewer P1：同 subject 多模板不得自我膨胀/过度加权）
    def fld(rec):
        return R_smooth_rank(rec, X, Y, sigma_xy, s_thresh)
    def _subj(r):
        return (r["dataset"], r["subject"])
    real_flds = [fld(r) for r in real_records]
    model_fld = fld(model_record)
    # real-to-real：同一 (dataset,subject)（含其它模板）跳过；每 record 取对【其他 subject】
    # 的 median，再 subject_first_fold 折成每 subject 一个值
    r2r = []
    for i, fi in enumerate(real_flds):
        cs = []
        for j, fj in enumerate(real_flds):
            if _subj(real_records[i]) == _subj(real_records[j]):
                continue
            c = corr_pair_mirror_invariant(fi["T"], fi["S"], fj["T"], fj["S"],
                                           s_thresh, overlap_min)["corr"]
            if c is not None:
                cs.append(c)
        if cs:
            r2r.append({"dataset": real_records[i]["dataset"],
                        "subject": real_records[i]["subject"],
                        "scalar": float(np.median(cs))})
    r2r_dist = subject_first_fold(r2r, key="scalar")
    # model-to-real：先按 real subject 折叠 corr（同 subject 多模板取 median），再跨 subject median
    m2r_by_subj: Dict[tuple, list] = {}
    for fi, ri in zip(real_flds, real_records):
        c = corr_pair_mirror_invariant(model_fld["T"], model_fld["S"],
                                       fi["T"], fi["S"], s_thresh, overlap_min)["corr"]
        if c is not None:
            m2r_by_subj.setdefault(_subj(ri), []).append(c)
    m2r_subj = [float(np.median(v)) for v in m2r_by_subj.values()]
    m2r_med = float(np.median(m2r_subj)) if m2r_subj else float("nan")
    field_placement = {
        "model_to_real_median_corr": m2r_med,
        "real_to_real_distribution_n": len(r2r_dist),
        "placement": placement_in_distribution(m2r_med, r2r_dist),
    }
    return {"scalar_placement": scalar_placement, "field_placement": field_placement,
            "note": "descriptive posterior-predictive; no p-value; SOZ not a metric"}


def R_smooth_rank(rec, X, Y, sigma_xy, s_thresh):
    """compare 内部用：取 rank 场（薄封装 smooth_field）。"""
    return smooth_field(rec, X, Y, sigma_xy=sigma_xy, scalar="rank", s_thresh=s_thresh)


def resolve_soz_overlay(names: Sequence[str], soz_core: set,
                        montage: str) -> Dict[str, object]:
    """SOZ overlay 匹配，政策锁 = first-contact alias（spec §7）。

    single montage：montage 名即 contact，精确匹配 soz_core。
    bipolar montage：每通道取 _first_contact（'E11-E12'->'E11'），匹配 soz_core；
      一个 SOZ contact 映射到 >=2 个通道 -> ambiguous（记录，不强配）。
    返回 soz_first_contacts（用于 record is_soz）+ soz_ambiguous（图注标注）。
    仅描述性，不进任何 metric。
    """
    from src.sef_hfo_soz_localization import _first_contact
    if montage == "single":
        hit = {n for n in names if n in soz_core}
        return {"soz_first_contacts": hit, "soz_ambiguous": []}
    first_map: Dict[str, list] = {}
    for n in names:
        first_map.setdefault(_first_contact(n), []).append(n)
    hit, ambig = set(), []
    for contact in soz_core:
        pairs = first_map.get(contact, [])
        if len(pairs) == 1:
            hit.add(contact)
        elif len(pairs) >= 2:
            ambig.append(contact)
    return {"soz_first_contacts": hit, "soz_ambiguous": sorted(ambig)}


def compute_cohort_scalars(record: dict) -> Dict[str, float]:
    """real-vs-model cohort scalar（spec §9 (a)，均真实数据自身有定义；NO theta_ref）。"""
    from scipy.stats import spearmanr
    chans = record["channels"]
    x = np.array([c["x_norm"] for c in chans], float)
    rk = np.array([c["typical_rank"] for c in chans], float)
    st = np.array([c["signed_transverse_mm"] for c in chans], float)
    ok = np.isfinite(x) & np.isfinite(rk)
    rho = float(spearmanr(x[ok], rk[ok]).correlation) if ok.sum() >= 3 else float("nan")
    # transverse_width：signed_transverse 的稳健展宽（p90-p10），mm
    tw = float(np.nanpercentile(st, 90) - np.nanpercentile(st, 10)) if st.size else float("nan")
    # 早/晚端：按 typical_rank 取前/后 1/3 的 signed_transverse 展宽 + 沿轴中心距
    order = np.argsort(rk)
    k = max(1, len(order) // 3)
    early, late = order[:k], order[-k:]
    along = np.array([c["along_axis_mm"] for c in chans], float)
    early_late_dist = abs(float(np.nanmean(along[late]) - np.nanmean(along[early])))
    L = record.get("axis_length_mm", float("nan"))
    return {
        "axis_length_mm": float(L),
        "transverse_width_mm": tw,
        "early_zone_spread": float(np.nanstd(st[early])) if k else float("nan"),
        "late_zone_spread": float(np.nanstd(st[late])) if k else float("nan"),
        "early_late_centroid_distance_norm":
            early_late_dist / L if L and np.isfinite(L) and L > 1e-9 else float("nan"),
        "rank_vs_xnorm_spearman": rho,
    }
