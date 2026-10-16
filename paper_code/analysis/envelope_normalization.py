"""Copied from topic5_interictal_event_field.py."""
import numpy as np

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

MS=1000.
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
