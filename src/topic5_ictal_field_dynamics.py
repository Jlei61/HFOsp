"""Topic 5 发作内 field 动力学 — 纯数学（source-core 定位、轴四分区、各 field 指标）。
设计: docs/superpowers/specs/2026-06-28-topic5-ictal-field-dynamics-design.md。无 scripts.* 依赖。"""
from __future__ import annotations
import numpy as np

GROUPS = ("source_core", "axis_end_noncore", "axial_mid", "non_axial")


def source_core(order_names, pos, compact_mm=15.0):
    """order_names: 触点名 earliest-first（已过滤到 valid）。pos: {name:(x,y)} mm。
    top2 间距 < compact_mm → 双点 core；否则单点 + uncertain。返回 (core, uncertain, top2_dist)。"""
    mapped = [n for n in order_names if n in pos]
    if not mapped:
        return [], True, float("nan")
    if len(mapped) == 1:
        return [mapped[0]], True, float("nan")
    p0, p1 = np.asarray(pos[mapped[0]], float), np.asarray(pos[mapped[1]], float)
    d = float(np.hypot(*(p1 - p0)))
    if d < compact_mm:
        return [mapped[0], mapped[1]], False, d
    return [mapped[0]], True, d


def axis_partition(names, pos, core_a, core_b, *, mid_band=(0.25, 0.75), degen_frac=0.15):
    """names: 全部 mapped 名。core_a/core_b: 两侧 source-core 名。投影到 P_A->P_B 线段，
    按 d(垂距) 中位 + t(沿轴位置) 划 4 组 MECE。"""
    P = {n: np.asarray(pos[n], float) for n in names if n in pos}
    pts = np.array(list(P.values()))
    bbox_diag = float(np.hypot(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))) if len(pts) > 1 else float("nan")
    PA = np.mean([P[n] for n in core_a if n in P], axis=0)
    PB = np.mean([P[n] for n in core_b if n in P], axis=0)
    u = PB - PA
    L = float(np.hypot(*u))
    axis_degenerate = (not np.isfinite(L)) or (np.isfinite(bbox_diag) and L < degen_frac * bbox_diag)
    src = set(core_a) | set(core_b)
    t, d = {}, {}
    for n, p in P.items():
        if L > 0:
            tt = float(np.dot(p - PA, u) / (L * L))
            t[n] = tt
            d[n] = float(np.hypot(*((p - PA) - tt * u)))
        else:
            t[n] = float("nan"); d[n] = float("nan")
    nonsrc = [n for n in P if n not in src]
    med_d = float(np.median([d[n] for n in nonsrc])) if nonsrc else float("nan")
    groups = {}
    for n in P:
        if n in src:
            groups[n] = "source_core"
        elif np.isfinite(d[n]) and d[n] > med_d:
            groups[n] = "non_axial"
        elif np.isfinite(t[n]) and mid_band[0] <= t[n] <= mid_band[1]:
            groups[n] = "axial_mid"
        else:
            groups[n] = "axis_end_noncore"
    return dict(groups=groups, t=t, d=d, P_A=PA.tolist(), P_B=PB.tolist(),
                L=L, bbox_diag=bbox_diag, axis_degenerate=bool(axis_degenerate), med_d=med_d)


def positive_mass_share(zmean, groups):
    """Rectified-mass fraction per group. A group with ZERO finite-z contacts returns NaN
    (= not measurable, e.g. no corridor), NOT 0.0 (which would read as 'present but inactive')."""
    rect = {n: max(float(v), 0.0) for n, v in zmean.items() if np.isfinite(v)}
    present = {groups[n] for n in zmean if n in groups and np.isfinite(zmean[n])}
    out = {g: (0.0 if g in present else float("nan")) for g in GROUPS}
    total = sum(rect.values())
    if total <= 0:
        return out
    for n, r in rect.items():
        g = groups.get(n)
        if g in present:
            out[g] += r / total
    return out


def group_mean(zmean, groups, group):
    vals = [float(v) for n, v in zmean.items() if groups.get(n) == group and np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def field_gradient(zmean, pos):
    items = [(pos[n][0], pos[n][1], float(v)) for n, v in zmean.items()
             if n in pos and np.isfinite(v)]
    if len(items) < 3:
        return float("nan"), float("nan")
    A = np.array([[x, y, 1.0] for x, y, _ in items], float)
    z = np.array([v for _, _, v in items], float)
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return float(np.degrees(np.arctan2(b, a))), float(np.hypot(a, b))


def fold_angle_deg(a_deg, b_deg):
    if not (np.isfinite(a_deg) and np.isfinite(b_deg)):
        return float("nan")
    diff = abs((a_deg - b_deg) % 180.0)
    return float(min(diff, 180.0 - diff))


def field_synchrony(ztraces):
    names = [n for n, tr in ztraces.items()
             if np.isfinite(tr).sum() >= 2 and np.nanstd(tr) > 0]
    corrs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = np.asarray(ztraces[names[i]], float), np.asarray(ztraces[names[j]], float)
            ok = np.isfinite(a) & np.isfinite(b)
            if ok.sum() >= 2 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                corrs.append(np.corrcoef(a[ok], b[ok])[0, 1])
    return float(np.median(corrs)) if corrs else float("nan")


def participation(zmean, thresh=2.0):
    vals = [float(v) for v in zmean.values() if np.isfinite(v)]
    return float(np.mean([v > thresh for v in vals])) if vals else float("nan")


def offset_pre_onset_overlap(win_lo_rel_offset, eeg_offset_rel):
    return bool((eeg_offset_rel + win_lo_rel_offset) < 0.0)


def parity_max_abs_diff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.max(np.abs(a[m] - b[m]))) if m.any() else float("inf")
