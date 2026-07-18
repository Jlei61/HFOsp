"""Topic 5 间期单事件包络场 — 纯数学。无绘图、无 scripts.* 依赖。

一次间期群体高频事件里，各触点一个接一个亮起来；这里把那道波从原始信号里还原出来，
画到**冻结的 TA/TB 共享平面**上，看 TA 事件和 TB 事件是不是沿共享轴反向位移。

三条承重合同（错了会静默出一张看着对但其实是错的图）：

1. **打包窗是尺子，不是测量结果。** 窗宽必须逐事件从 `packedTimes` 读取（例如
   E1146 为 250 ms、E139 为 200 ms）；动画窗按本次事件实际质心跨度收紧。

2. **包络是 amplitude，不是 energy。** `return_hil_enve_norm` = 各子带 `|hilbert(x)|` 求和，
   没有平方。所有面向读者的字样必须写 "envelope amplitude"，不能写 energy/power。

3. **主图只用本次事件的参与触点。** 模板 participation support 会把模板的空间签名重新
   画进示例事件，属于循环论证；全部冻结触点和模板加权只允许作为 QC。
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

MS = 1000.0
ENVELOPE_QUANTITY = "envelope amplitude"   # 合同 2：不是 energy/power


# ============================================================ 事件跨度 / 代表事件
def event_spreads_ms(lag_raw, bools, *, min_participating=3):
    """每个事件的最早→最晚跨度（毫秒）。参与通道不足 -> NaN（= 测不了，不是 0）。"""
    lag, bo = np.asarray(lag_raw, float), np.asarray(bools, bool)
    out = np.full(lag.shape[1], np.nan)
    for i in range(lag.shape[1]):
        v = lag[bo[:, i], i]
        v = v[np.isfinite(v)]
        if v.size >= min_participating:
            out[i] = float((np.max(v) - np.min(v)) * MS)
    return out


def stored_lag_rel_ms(lag_raw_col, bool_col):
    """存档的每通道点火时刻 -> 相对最早参与通道的毫秒数（非参与 = NaN）。

    存档绝对值在一条拼接时间轴上，只有**同一事件内的相对差**有意义
    （见 src/group_event_analysis.py 的 lagPatRaw 说明），所以减去参与通道的最小值。
    """
    lag = np.asarray(lag_raw_col, float).copy()
    lag[~np.asarray(bool_col, bool)] = np.nan
    if not np.isfinite(lag).any():
        raise ValueError("event has no participating channel with a finite stored lag")
    return (lag - np.nanmin(lag)) * MS


def template_likeness(masked_col, template_rank, bool_col, *, min_common=5):
    """这个事件的点火顺序，跟该模板的平均顺序有多像（Spearman）。不够共同通道 -> NaN。"""
    m = (np.asarray(bool_col, bool) & np.isfinite(np.asarray(template_rank, float))
         & np.isfinite(np.asarray(masked_col, float)))
    if m.sum() < min_common:
        return float("nan")
    r = spearmanr(np.asarray(masked_col, float)[m], np.asarray(template_rank, float)[m]).correlation
    return float(r) if np.isfinite(r) else float("nan")


def rank_candidates(masked, bools, labels, label, template_rank, spreads_ms,
                    spread_target, npart_target, *, min_participating=5, top_k=25,
                    rho_decimals=3):
    """第一步（便宜，不碰原始信号）：门 + 排序 -> top_k 候选位置。最终挑选交给 select_medoid_event。

    **并列打破是承重的**：真实被试里成千上万个事件的顺序跟模板**完全一致**（epilepsiae_958
    实测 8 万事件、top-25 全是 rho=1.000）。此时按 rho 排序全是并列，稳定排序会让 top_k 恒取
    **事件序号最小的一批 = 录制最早期的事件**，候选池带隐藏的时间偏差。
    所以 rho 先四舍五入到 rho_decimals 位分档，同档内按「便宜特征（跨度、参与数）离典型值多远」
    再排 —— 候选池代表**典型事件**，而不是代表录制的开头。
    """
    tr = np.asarray(template_rank, float)
    sp = np.asarray(spreads_ms, float)
    scored = []
    for i in np.where(np.asarray(labels, int) == int(label))[0]:
        npart = int(np.asarray(bools[:, i], bool).sum())
        if npart < min_participating or not np.isfinite(sp[i]):
            continue
        r = template_likeness(masked[:, i], tr, bools[:, i], min_common=min_participating)
        if np.isfinite(r):
            scored.append((r, int(i), npart))
    if not scored:
        raise ValueError(f"label {label}: no event passes the participation/spread gate "
                         f"(n_labelled={int((np.asarray(labels)==label).sum())})")
    sp_scale = 1.4826 * np.median(np.abs(sp[[i for _, i, _ in scored]] - spread_target)) or 1.0
    np_scale = 1.4826 * np.median(np.abs(np.array([n for _, _, n in scored], float)
                                         - npart_target)) or 1.0
    def _key(item):
        r, i, npart = item
        cheap_d = abs(sp[i] - spread_target) / sp_scale + abs(npart - npart_target) / np_scale
        return (-round(float(r), int(rho_decimals)), cheap_d)
    scored.sort(key=_key)
    n_tied = sum(1 for r, _, _ in scored
                 if round(float(r), int(rho_decimals)) == round(float(scored[0][0]),
                                                                int(rho_decimals)))
    return ([i for _, i, _ in scored[:int(top_k)]], {i: r for r, i, _ in scored},
            dict(n_eligible=len(scored), n_tied_at_top=n_tied,
                 top_rho=float(scored[0][0])))


def select_medoid_event(cand, feats, target, *, scale=None):
    """第二步：多维 medoid —— 挑各维都最接近典型值的那个事件，不是最漂亮的那个。

    feats: {event_pos: [f1, f2, ...]}；target: [t1, t2, ...] 该模板的典型值。
    每维用 scale（默认各维 MAD，退化时用 1）标准化后取 L1 距离最小者。

    为什么不用「最像模板的那一次」：那条规则会系统性挑到长事件（事件越长、先后越拉得开、
    顺序越干净越容易跟模板对上；epilepsiae_139 实测 相关 +0.277），挑出来的是长尾而不是
    典型事件，动画窗被撑长、图跟自己标题里的中位跨度打架。
    **挑选过程绝不读取最终图上的传播斜率** —— 那样就是 cherry-pick。
    """
    if not cand:
        raise ValueError("empty candidate pool")
    F = np.array([feats[i] for i in cand], float)
    tg = np.asarray(target, float)
    if F.shape[1] != tg.size:
        raise ValueError(f"feature dim {F.shape[1]} != target dim {tg.size}")
    if scale is None:
        med = np.median(F, axis=0)
        scale = 1.4826 * np.median(np.abs(F - med), axis=0)
    scale = np.where(np.asarray(scale, float) > 1e-12, scale, 1.0)
    d = np.abs((F - tg) / scale).sum(axis=1)
    return int(cand[int(np.argmin(d))]), float(d.min())


# ============================================================ 包络 / 基线 / 时间零点
def baseline_robust_z(env, t, ev_lo, ev_hi, *, guard_sec=0.05, min_baseline_samples=64):
    """每通道对「事件窗外的安静段」做 median/MAD 稳健 z。

    基线 = t < ev_lo - guard 或 t > ev_hi + guard。guard 防事件肩部漏进基线（漏进去
    会把 MAD 撑大、把事件自己压扁）。MAD=0 的死通道 -> 全 NaN，不假装成有信号。
    """
    t = np.asarray(t, float)
    m = (t < ev_lo - guard_sec) | (t > ev_hi + guard_sec)
    if int(m.sum()) < min_baseline_samples:
        raise ValueError(f"baseline too short: {int(m.sum())} samples < {min_baseline_samples}; "
                         f"widen the pad around the event")
    base = np.asarray(env, float)[:, m]
    med = np.median(base, axis=1, keepdims=True)
    scale = 1.4826 * np.median(np.abs(base - med), axis=1, keepdims=True)
    scale = np.where(scale <= 0, np.nan, scale)
    return (np.asarray(env, float) - med) / scale


# ---------------- 探测器自己的事件定义（bqk_utils.find_high_enveTimes 同一套规则）
def detector_excursions(env, t, *, rel_thresh=2.0, abs_thresh=2.0, min_gap_ms=20.0,
                        min_last_ms=50.0, max_last_ms=200.0):
    """按**探测器自己的规则**把每通道的包络切成一段段"事件"，返回 per-channel [(t0,t1), ...]。

    照抄 src/utils/bqk_utils.py::find_high_enveTimes 的判据（legacy yuquan 口径见
    config/default.yaml: rel=abs=2.0, min_gap=20ms, min_last=50ms, max_last=200ms）：
      包络 > rel_thresh x 本通道中位  AND  > abs_thresh x 全通道中位
      -> 间隔 < min_gap 的合并 -> 只留时长在 [min_last, max_last] 内的。

    **为什么必须用它而不是在整个 200ms 打包窗里取 argmax**：一个 200ms 窗里可以装下不止一次
    事件（被 min_gap 分开的就是两次）。实测 epilepsiae_1146 TB：低轴端触点在 ~120ms 还有一团，
    那是**另一次事件**，不属于这个群体事件；把它当成"这次的峰"会让传播读出被拖长一倍。
    """
    e = np.asarray(env, float)
    t = np.asarray(t, float)
    gmed = float(np.median(e))
    out = []
    for i in range(e.shape[0]):
        cmed = float(np.median(e[i]))
        hot = (e[i] > rel_thresh * cmed) & (e[i] > abs_thresh * gmed)
        runs = _runs_from_mask(hot, t)
        runs = _merge_runs(runs, min_gap_ms / MS)
        out.append([(a, b) for a, b in runs
                    if (b - a) > min_last_ms / MS
                    and (max_last_ms is None or (b - a) < max_last_ms / MS)])
    return out


def _runs_from_mask(mask, t):
    m = np.asarray(mask, bool).astype(int)
    if not m.any():
        return []
    d = np.diff(np.concatenate([[0], m, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return [(float(t[a]), float(t[b])) for a, b in zip(starts, ends)]


def _merge_runs(runs, min_gap_sec):
    if not runs:
        return []
    out = [list(runs[0])]
    for a, b in runs[1:]:
        if a - out[-1][1] < min_gap_sec:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [tuple(x) for x in out]


def group_event_interval(excursions, t):
    """群体事件 = **最多通道同时在放电**的那一段。返回 (t0, t1)；没有则 None。

    群体事件本来就是这么定义的（多通道同时超阈才被打包成一个事件），所以"这次事件是哪一段"
    应该由并发度决定，而不是由某个通道最响的那一刻决定。
    """
    t = np.asarray(t, float)
    conc = np.zeros(t.size, int)
    for ch in excursions:
        for a, b in ch:
            conc[(t >= a) & (t <= b)] += 1
    if conc.max() == 0:
        return None
    runs = _runs_from_mask(conc >= max(conc.max(), 1), t)
    return max(runs, key=lambda r: r[1] - r[0]) if runs else None


def peak_times_in_event(env, t, excursions, interval):
    """每通道取**跟群体事件那一段重叠**的那次放电里的峰时刻。不重叠 -> NaN（= 该通道没参与这次）。"""
    e, t = np.asarray(env, float), np.asarray(t, float)
    lo, hi = interval
    out = np.full(e.shape[0], np.nan)
    for i, ch in enumerate(excursions):
        hits = [(a, b) for a, b in ch if b >= lo and a <= hi]
        if not hits:
            continue
        a, b = max(hits, key=lambda r: min(r[1], hi) - max(r[0], lo))   # 重叠最多的那一次
        m = (t >= a) & (t <= b)
        if m.any() and np.isfinite(e[i, m]).any():
            out[i] = float(t[m][int(np.nanargmax(e[i, m]))])
    return out


def envelope_peak_times(z, t, lo, hi):
    """每通道在 [lo, hi] 内包络峰值出现的时刻（全 NaN 的通道 -> NaN）。"""
    t = np.asarray(t, float)
    m = (t >= lo) & (t <= hi)
    if not m.any():
        raise ValueError(f"empty search window [{lo}, {hi}]")
    sub, ts = np.asarray(z, float)[:, m], t[m]
    out = np.full(sub.shape[0], np.nan)
    for i in range(sub.shape[0]):
        if np.isfinite(sub[i]).any():
            out[i] = ts[int(np.nanargmax(sub[i]))]
    return out


def envelope_peak_z(z, t, lo, hi):
    """每通道在 [lo, hi] 内的包络峰值高度（倍基线）。用作信噪门 + medoid 的一维。"""
    t = np.asarray(t, float)
    m = (t >= lo) & (t <= hi)
    sub = np.asarray(z, float)[:, m]
    out = np.full(sub.shape[0], np.nan)
    for i in range(sub.shape[0]):
        if np.isfinite(sub[i]).any():
            out[i] = float(np.nanmax(sub[i]))
    return out


def _first_run_start(mask, need):
    mask = np.asarray(mask, bool)
    if need <= 1:
        w = np.where(mask)[0]
        return int(w[0]) if w.size else None
    if mask.size < need:
        return None
    c = np.convolve(mask.astype(int), np.ones(int(need), int), mode="valid")
    w = np.where(c == int(need))[0]
    return int(w[0]) if w.size else None


def event_level_t0(z, t, part_mask, lo, hi, *, frac_of_peak=0.25, min_z=5.0, sustain_ms=5.0):
    """t=0 = 参与触点的**合奏包络**（逐时刻取参与触点的最大值）第一次持续越过
    `max(frac_of_peak × 该合奏自身的峰, min_z)` 的时刻。

    两条都是被真实数据打脸后定的：

    1. **不能锚「存档说最早的那个触点」**：存档说最早的可能是个弱通道，包络根本没有真峰
       （epilepsiae_139 的 HRB1：探测器算它参与，z 峰值只有 3，锚上去就锚到噪声）。

    2. **不能用绝对低阈值（如 z>=3 持续 5ms）**：z=3 只是 3 倍 MAD，噪声里本来就常见，
       撑 5ms 毫无难度。epilepsiae_1146 实测：该规则把 t0 钉在打包窗左缘，而参与触点的
       包络峰其实在 +96..+145 ms —— 整个显示窗画的是事件前的爬升，不是事件。
       所以阈值必须**对事件自身的幅度自标定**（默认 25% 峰高），min_z 只作噪声地板。

    逐时刻取参与触点的**最大值**（不是均值）：行波里每个触点各自在不同时刻达峰，
    最大值才跟着波前走；均值会被还没点亮的触点稀释。

    ⚠️ 这是**滤波后包络的 onset**，不是精确生理起始，措辞不能升级。
    """
    t = np.asarray(t, float)
    m = (t >= lo) & (t <= hi)
    ts = t[m]
    if ts.size < 2:
        raise ValueError(f"empty t0 search window [{lo}, {hi}]")
    part = np.asarray(part_mask, bool)
    if not part.any():
        raise ValueError("event has no participating contact")
    zz = np.asarray(z, float)[np.ix_(part, m)]
    if not np.isfinite(zz).any():
        raise ValueError("participating contacts have no finite envelope in the event window")
    with np.errstate(invalid="ignore"):
        ens = np.nanmax(zz, axis=0)
    peak = float(np.nanmax(ens))
    thresh = max(float(frac_of_peak) * peak, float(min_z))
    if peak < thresh:
        raise ValueError(f"event never reaches the onset threshold (peak z={peak:.1f} < "
                         f"{thresh:.1f}); no usable envelope onset")
    dt = float(np.median(np.diff(ts)))
    need = max(int(round(sustain_ms / MS / dt)), 1)
    k = _first_run_start(np.nan_to_num(ens, nan=-np.inf) >= thresh, need)
    if k is None:
        raise ValueError(f"ensemble envelope never sustains z>={thresh:.1f} for >={sustain_ms} ms; "
                         f"no usable envelope onset")
    return float(ts[k])


def estimator_agreement(peak_ms, stored_ms):
    """白拿的验证数：包络峰值顺序 vs 存档顺序（存档用频谱能量重心，这里画包络峰值）。

    两者各自减去自身最早值后比（都是相对量）。返回 (rho, 中位绝对时间差 ms, n)。
    ⚠️ 读法：这个时间尺度上十几个触点挤在几十毫秒内，**一个没有真峰的弱通道就能把 rho 翻盘**
    （epilepsiae_139 实测：9/10 通道一致到 0.3-3.9 ms，一个 HRB1 把 rho 拉到 0.469）。
    中位时间差比 rho 更可信。
    """
    a, b = np.asarray(peak_ms, float), np.asarray(stored_ms, float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 3:
        return float("nan"), float("nan"), int(m.sum())
    rho = spearmanr(a[m], b[m]).correlation
    d = (a[m] - np.min(a[m])) - (b[m] - np.min(b[m]))
    return float(rho), float(np.median(np.abs(d))), int(m.sum())


# ============================================================ 时间窗 / 取帧 / 色标
def display_window_ms(peak_ms_per_event, *, pre_ms=10.0, margin_ms=15.0,
                      min_post_ms=40.0, max_post_ms=190.0):
    """TA/TB 共用的同一条时间轴，由右侧实际显示的 Fig1a 质心定。

    不用模板的存档跨度定窗：右侧采用 Fig1a 的主增强连通区质心，它与 lagPat 的全局频谱
    重心不是同一个估计量。显示什么就用什么定窗，避免把实际质心轨迹裁掉。

    peak_ms_per_event: 每个事件一个数组 = 参与触点质心时刻（相对各自 t=0，毫秒）。
    """
    v = np.concatenate([np.asarray(x, float).ravel() for x in peak_ms_per_event])
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -abs(float(pre_ms)), float(min_post_ms)
    lo = min(-abs(float(pre_ms)), float(np.min(v)) - float(margin_ms))
    hi = float(np.clip(np.max(v) + float(margin_ms), min_post_ms, max_post_ms))
    return float(lo), hi


def frame_times_ms(t_lo, t_hi, n):
    return np.linspace(float(t_lo), float(t_hi), int(n))


def values_at(z, t_sec, t0, t_query_ms, *, avg_ms=3.0):
    """取一帧：中心前后 avg_ms/2 内取平均，压掉单采样点的闪烁（审阅 §6.4）。"""
    t = np.asarray(t_sec, float)
    c = float(t0) + float(t_query_ms) / MS
    h = float(avg_ms) / MS / 2.0
    m = (t >= c - h) & (t <= c + h)
    if not m.any():
        m = np.zeros(t.shape, bool)
        m[int(np.argmin(np.abs(t - c)))] = True
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.asarray(z, float)[:, m], axis=1)


def pooled_vmax(z_list, *, q=99.0):
    """TA/TB 两事件、全部显示时间和触点 pooled 分位 -> 一个共享固定色标。

    不逐行、不逐帧、不逐事件归一化 —— 否则 A/B 的强弱不可比，"反向"也可能是归一化artifact。
    """
    v = np.concatenate([np.asarray(x, float).ravel() for x in z_list])
    v = v[np.isfinite(v)]
    return max(float(np.percentile(v, q)), 1.0) if v.size else 1.0
