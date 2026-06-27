"""Topic 5 发作早期方向无监督两类聚类 ↔ 间期 A/B 方向（纯函数, TDD）。

设计 spec: docs/superpowers/specs/2026-06-27-topic5-ictal-direction-clustering-design.md
口径: ictal-only k=2 (盲于间期) -> 三道预锁门 (P0 两类资格 / P1 旋转对齐 null / P1 轴质量) -> 描述性分档。
圆周统计复用 src.topic5_axis_direction; 这里只放 geometry / clustering / null / gate。
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from src.topic5_axis_direction import (circular_mean, resultant_length,
                                        axial_resultant_length)

TWO_PI = 2.0 * np.pi
SEED = 20260627


# ---------- geometry ----------
def plane_fit_direction(x, y, values):
    """方向(值增长, [0,2pi)) + 梯度范数 + 拟合 R² + 有限点数, 经最小二乘平面拟合。"""
    x = np.asarray(x, float); y = np.asarray(y, float); v = np.asarray(values, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(v)
    n_valid = int(ok.sum())
    if n_valid < 3 or np.nanstd(v[ok]) < 1e-12:
        return (np.nan, 0.0, 0.0, n_valid)
    X = np.column_stack([x[ok] - x[ok].mean(), y[ok] - y[ok].mean()])
    vv = v[ok] - v[ok].mean()
    beta, *_ = np.linalg.lstsq(X, vv, rcond=None)
    grad_norm = float(np.linalg.norm(beta))
    if grad_norm < 1e-12:
        return (np.nan, 0.0, 0.0, n_valid)
    pred = X @ beta
    ss_res = float(np.sum((vv - pred) ** 2)); ss_tot = float(np.sum(vv ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    angle = float(np.mod(np.arctan2(beta[1], beta[0]), TWO_PI))
    return (angle, grad_norm, r2, n_valid)


def coord_aspect(x, y):
    """触点云 PCA 次/主奇异值比 ∈ [0,1]; 近一维≈0。"""
    P = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    P = P[np.isfinite(P).all(1)]
    if len(P) < 3:
        return np.nan
    P = P - P.mean(0)
    ev = np.linalg.svd(P, compute_uv=False)
    return float(ev[1] / ev[0]) if ev[0] > 0 else np.nan


# ---------- clustering ----------
def cluster_directions_k2(angles, seed=0):
    """ictal-only k=2 on [cosθ,sinθ]; 返回 labels/means/sizes/class_R + 全体 R_dir/R_axial。"""
    a = np.asarray(angles, float); a = a[np.isfinite(a)]
    out = {"n": int(a.size),
           "R_dir": float(resultant_length(a)) if a.size else np.nan,
           "R_axial": float(axial_resultant_length(a)) if a.size else np.nan,
           "angles": a}
    if a.size < 2:
        out.update(labels=np.zeros(a.size, int), means=[np.nan, np.nan],
                   sizes=[int(a.size), 0], class_R=[np.nan, np.nan])
        return out
    V = np.column_stack([np.cos(a), np.sin(a)])
    labels = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(V)
    means, sizes, class_R = [], [], []
    for c in (0, 1):
        ac = a[labels == c]
        means.append(circular_mean(ac) if ac.size else np.nan)
        sizes.append(int(ac.size))
        class_R.append(float(resultant_length(ac)) if ac.size else np.nan)
    out.update(labels=labels, means=means, sizes=sizes, class_R=class_R)
    return out


def silhouette_unit(angles, labels):
    """silhouette on [cosθ,sinθ]; <2 distinct labels -> -1.0。"""
    a = np.asarray(angles, float); labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2 or a.size < 3:
        return -1.0
    V = np.column_stack([np.cos(a), np.sin(a)])
    return float(silhouette_score(V, labels))


# ---------- P0 gate: is there a real second concentrated mode? ----------
def kappa_from_R(R):
    """von Mises 浓度 κ = A⁻¹(R) (Mardia & Jupp 1999 分段近似)。"""
    R = float(R)
    if R < 1e-8:
        return 0.0
    if R < 0.53:
        return 2 * R + R ** 3 + 5 * R ** 5 / 6.0
    if R < 0.85:
        return -0.4 + 1.39 * R + 0.43 / (1 - R)
    denom = R ** 3 - 4 * R ** 2 + 3 * R
    return 1.0 / denom if denom > 1e-9 else 1e6


def unimodal_null_pvalue(angles, B=2000, seed=SEED):
    """p_bimodal: H0 = 一个集中主方向 + 均匀背景散点; 信号 = 第二个集中模。

    先 k=2 分多数簇, 对多数簇拟合 von Mises(mu, kappa), 少数比例 f=n_minor/n;
    每次模拟 n 个角度(以概率 1-f 抽 von Mises, f 抽 [0,2pi) 均匀)再 k=2 取 silhouette。
    p 小 = 观测 silhouette 超过 '主方向+散点' 能产生的水平 = 真有第二个集中模。
    纯单峰 null 太弱(主方向+少数散点会被误判两类); 见 spec §4.1 P1 修复 2026-06-27。
    """
    a = np.asarray(angles, float); a = a[np.isfinite(a)]
    n = a.size
    if n < 4:
        return (1.0, np.nan)
    clus = cluster_directions_k2(a, seed=0)
    s_obs = silhouette_unit(a, clus["labels"])
    labels = clus["labels"]
    n0 = int((labels == 0).sum())
    maj = a[labels == (0 if n0 >= n - n0 else 1)]
    mu = circular_mean(maj)
    kappa = max(kappa_from_R(resultant_length(maj)), 1e-6)
    f = 1.0 - maj.size / n
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(B):
        u = rng.random(n)
        sim = np.where(u < f, rng.uniform(0, TWO_PI, n), rng.vonmises(mu, kappa, n))
        s = silhouette_unit(sim, cluster_directions_k2(sim, seed=0)["labels"])
        if s >= s_obs:
            ge += 1
    return ((1 + ge) / (B + 1), float(s_obs))


def bootstrap_label_stability(angles, B=500, seed=SEED):
    """中位 bootstrap ARI: 重抽→重聚类→全点指派最近质心→与原始标签 ARI。

    注意: 仅测 '划分可复现', 对固定散点会给假高; 反散点主门是 unimodal_null_pvalue。
    """
    a = np.asarray(angles, float); a = a[np.isfinite(a)]
    n = a.size
    if n < 4:
        return np.nan
    base = cluster_directions_k2(a, seed=0)["labels"]
    V = np.column_stack([np.cos(a), np.sin(a)])
    rng = np.random.default_rng(seed)
    aris = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        boot = cluster_directions_k2(a[idx], seed=0)
        cents = []
        for c in (0, 1):
            ac = a[idx][boot["labels"] == c]
            if ac.size == 0:
                cents = None; break
            cents.append([np.cos(ac).mean(), np.sin(ac).mean()])
        if cents is None:
            continue
        cents = np.asarray(cents)
        d = np.linalg.norm(V[:, None, :] - cents[None, :, :], axis=2)
        pred = d.argmin(axis=1)
        aris.append(adjusted_rand_score(base, pred))
    return float(np.median(aris)) if aris else np.nan


def two_class_eligible(n_sz, sizes, p_bimodal, stability, *,
                       bimodal_alpha=0.05, stab_min=0.5):
    """P0 三条 AND 门; 返回 (eligible, reasons)。未过门只准 '主方向' 措辞, 禁止 '两类'。"""
    reasons = []
    if n_sz < 6:
        reasons.append("n_sz<6")
    if min(sizes) < 3:
        reasons.append("min_class<3")
    if not (p_bimodal is not None and np.isfinite(p_bimodal) and p_bimodal < bimodal_alpha):
        reasons.append("p_bimodal>=alpha")
    if not (stability is not None and np.isfinite(stability) and stability >= stab_min):
        reasons.append("stability<min")
    return (len(reasons) == 0, reasons)


# ---------- P1 gate: interictal A/B template-axis quality ----------
def axis_quality_tier(delta_ab_rad, n_valid_a, n_valid_b, *,
                      interp_min_deg=120, weak_min_deg=60, min_valid=6):
    """间期 A/B 轴质量分档 (按 Δ_AB 度数, 预锁阈值); 模板触点不足强降 diagnostic。"""
    if n_valid_a < min_valid or n_valid_b < min_valid or not np.isfinite(delta_ab_rad):
        return "diagnostic_only"
    deg = float(np.degrees(delta_ab_rad))
    if deg >= interp_min_deg - 1e-9:        # tolerance: degrees<->radians round-trip
        return "interpretable"
    if deg >= weak_min_deg - 1e-9:
        return "weak_axis"
    return "diagnostic_only"


# ---------- P1 gate: ictal-vs-interictal alignment with rotation null ----------
def angular_distance(a, b):
    """全圆角距 ∈ [0,pi]; d(0,pi)=pi, d(0,2pi)=0。"""
    d = abs(float(a) - float(b)) % TWO_PI
    return float(min(d, TWO_PI - d))


def best_pair_residual(class_means, template_dirs):
    """{c1,c2}->{A,B} 取 straight/crossed 角距和最小; dict{sum,mean,pairing,matched}(rad); 任一 NaN -> None。"""
    c1, c2 = class_means; A, B = template_dirs
    if not all(np.isfinite(v) for v in (c1, c2, A, B)):
        return None
    straight = (angular_distance(c1, A), angular_distance(c2, B))
    crossed = (angular_distance(c1, B), angular_distance(c2, A))
    matched, pairing = (straight, "straight") if sum(straight) <= sum(crossed) else (crossed, "crossed")
    return {"sum": float(sum(matched)), "mean": float(np.mean(matched)),
            "pairing": pairing, "matched": [float(matched[0]), float(matched[1])]}


def nearest_template_gap(main_dir, theta_a, theta_b):
    """角距(rad, [0,pi]) 从一个方向到两条模板方向里**较近**的那条 (sign-free 轴对齐)。"""
    return min(angular_distance(main_dir, theta_a), angular_distance(main_dir, theta_b))


def cohort_alignment_rotation_test(mains, ab_pairs, B=10000, seed=SEED, stat="median"):
    """队列检验: 发作主方向是否比随机方向更贴近其**较近**的间期模板方向 (SIGN-FREE 轴对齐)。

    mains: 每被试发作主方向 (rad)。ab_pairs: 每被试 (theta_a, theta_b) (rad)。
    每被试 gap = 到两模板方向较近者的角距; cohort 统计量 = 各 gap 的 median/mean。
    null: 每被试 main 方向换成独立均匀随机方向(模板固定), 重算 gap 与 cohort 统计量;
    p = P(stat_null <= stat_obs) 单侧。返回 dict(gaps, T_obs, p, per_pct, null_lo/md/hi) (rad)。
    """
    mains = np.asarray(mains, float)
    A = np.asarray([p[0] for p in ab_pairs], float)
    Bv = np.asarray([p[1] for p in ab_pairs], float)
    ok = np.isfinite(mains) & np.isfinite(A) & np.isfinite(Bv)
    mains, A, Bv = mains[ok], A[ok], Bv[ok]
    n = mains.size
    agg = np.median if stat == "median" else np.mean

    def _gap(m):
        da = np.abs(m - A) % TWO_PI; da = np.minimum(da, TWO_PI - da)
        db = np.abs(m - Bv) % TWO_PI; db = np.minimum(db, TWO_PI - db)
        return np.minimum(da, db)

    gaps = _gap(mains)
    T_obs = float(agg(gaps))
    rng = np.random.default_rng(seed)
    null_stat = np.empty(B)
    per_null = np.empty((B, n))
    for b in range(B):
        g = _gap(rng.uniform(0, TWO_PI, n))
        per_null[b] = g
        null_stat[b] = agg(g)
    p = float((1 + np.sum(null_stat <= T_obs)) / (B + 1))
    per_pct = np.array([(1 + np.sum(per_null[:, i] <= gaps[i])) / (B + 1) for i in range(n)])
    return dict(gaps=gaps, T_obs=T_obs, p=p, per_pct=per_pct,
                null_lo=np.percentile(per_null, 5, axis=0),
                null_md=np.percentile(per_null, 50, axis=0),
                null_hi=np.percentile(per_null, 95, axis=0))


def best_pair_rotation_null(class_means, template_dirs, B=2000, seed=SEED):
    """旋转 null: 共同旋转两类均向 φ~U[0,2pi), 对固定模板重做 best-pair; p_align=resid_sum<=obs 比例。"""
    c1, c2 = class_means
    if not all(np.isfinite(v) for v in (c1, c2, *template_dirs)):
        return np.nan
    obs = best_pair_residual(class_means, template_dirs)["sum"]
    rng = np.random.default_rng(seed)
    le = 0
    for _ in range(B):
        phi = rng.uniform(0, TWO_PI)
        r = best_pair_residual([c1 + phi, c2 + phi], template_dirs)
        if r["sum"] <= obs:
            le += 1
    return (1 + le) / (B + 1)
