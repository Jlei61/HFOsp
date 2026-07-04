"""Topic 5 间期传播场外推到发作隐身 territory。

发作侧 = z-ER 招募顺序 (Layer A r_sz，低=早=源)；间期侧 = broad 轴场 typical_rank (低=早)。
同向 → 正相关。F = 用 broad 间期顺序场 (support 加权) 留一预测隐身电极位置的间期顺序，
对发作 z-ER 序的带符号 Spearman；C = 隐身电极自己那条噪的 broad rank 对发作序的带符号 Spearman。
F 赢 C = 可信核心场把噪电极补活了。

Spec: docs/superpowers/specs/2026-06-30-topic5-interictal-field-broad-extrapolation-design.md
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import spearmanr, rankdata

DEF_AXIS_DIR = "results/spatial_modulation/propagation_geometry_broad/observation_readout/real_subjects"
DEF_BROAD_POOL = "results/interictal_propagation_masked_broad/per_subject"
DEF_NARROW_POOL = "results/interictal_propagation_masked/per_subject"
DEF_LAYER_A = "results/data_driven_soz/layer_a_ictal_er_rank/per_subject"
DEF_T0 = "results/topic5_ictal_recruitment/t0_feature_cache"


# ---------------------------------------------------------------- Task 1: loaders
def load_broad_axis_record(ds_sid: str, axis_dir: str = DEF_AXIS_DIR, template: str = "t_a"):
    """template='t_a' 或 't_b'（两间期模板，无优劣）。文件不存在返回 None。"""
    p = Path(axis_dir) / f"{ds_sid}_{template}.json"
    return json.load(open(p)) if p.exists() else None


def channel_names_from_pool(ds_sid: str, pool_dir: str) -> List[str]:
    d = json.load(open(Path(pool_dir) / f"{ds_sid}.json"))
    return list(d["channel_names"])


def broad_minus_narrow(broad_names: Sequence[str], narrow_names: Sequence[str]) -> List[str]:
    nset = set(narrow_names)
    return sorted(n for n in set(broad_names) if n not in nset)


def ictal_reliability(ds_sid: str, layer_a_dir: str = DEF_LAYER_A,
                      er_config: str = "broad_ER", *, min_s_sz: float = 0.3,
                      min_n_ok: int = 5) -> dict:
    """发作招募排序是否稳定到能当 ground-truth。

    producer_health + s_sz (跨发作 rank 向量两两 Spearman 中位) + n_seizures_ok。
    s_sz 低 = 招募顺序跨发作不一致 → 没有稳定"发作方向"可供间期场预测 (= 本检验前提失败)。
    """
    d = json.load(open(Path(layer_a_dir) / f"{ds_sid}.json"))
    er = d["per_er"][er_config]
    health = d.get("producer_health", {}).get(er_config)
    n_ok = er.get("n_seizures_ok")
    s_sz = er.get("s_sz")
    reliable = (health in ("stable", "moderate")
                and isinstance(s_sz, (int, float)) and s_sz >= min_s_sz
                and isinstance(n_ok, int) and n_ok >= min_n_ok)
    return {"health": health, "n_seizures_ok": n_ok, "s_sz": s_sz, "reliable": bool(reliable)}


def ictal_zer_ranks(ds_sid: str, layer_a_dir: str = DEF_LAYER_A,
                    er_config: str = "broad_ER", min_valid_count: int = 3) -> Dict[str, float]:
    d = json.load(open(Path(layer_a_dir) / f"{ds_sid}.json"))
    er = d["per_er"][er_config]
    r_sz = er["r_sz"]
    vc = er.get("r_sz_valid_count", {})
    out: Dict[str, float] = {}
    for ch, r in r_sz.items():
        if r is None:
            continue
        if int(vc.get(ch, 0)) < min_valid_count:
            continue
        out[ch] = float(r)
    return out


# ------------------------------------------------- Task 2: field prediction (LOO)
def _median_nn(pts: np.ndarray) -> float:
    if pts.shape[0] < 2:
        return 0.1
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return float(np.median(d.min(1)))


def field_predict_at_points(record_channels, eval_xy, exclude_name: Optional[str] = None,
                            sigma_xy: Optional[float] = None) -> np.ndarray:
    chans = [c for c in record_channels
             if c["name"] != exclude_name
             and np.isfinite(c["x_norm"]) and np.isfinite(c["y_norm"])
             and np.isfinite(c.get("typical_rank", np.nan)) and c.get("support", 0) > 0]
    pts = np.array([[c["x_norm"], c["y_norm"]] for c in chans], float).reshape(-1, 2)
    vals = np.array([c["typical_rank"] for c in chans], float)
    sup = np.array([c["support"] for c in chans], float)
    if sigma_xy is None:
        sigma_xy = _median_nn(pts)
    sig2 = 2.0 * sigma_xy ** 2
    eval_xy = np.asarray(eval_xy, float).reshape(-1, 2)
    out = np.full(eval_xy.shape[0], np.nan)
    for i in range(eval_xy.shape[0]):
        x, y = eval_xy[i]
        w = sup * np.exp(-(((pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2) / sig2))
        sw = w.sum()
        if sw > 1e-12:
            out[i] = float((w * vals).sum() / sw)
    return out


def predicted_interictal_order(record, target_names, loo: bool = True,
                               sigma_xy: Optional[float] = None,
                               core_names: Optional[Sequence[str]] = None) -> Dict[str, float]:
    """场在 target 电极位置的预测。
    core_names=None → 场用全 record 通道（LOO 只排目标，其它 hidden 仍参与平滑）= broad-LOO 版。
    core_names=set → 场**只用** core 通道建（如 narrow/core），证"预测来自核心非 hidden 互借"
      = F_core_only contract 版（review P1）。"""
    chans = record["channels"]
    if core_names is not None:
        cset = set(core_names)
        chans = [c for c in chans if c["name"] in cset]
    by_name = {c["name"]: c for c in record["channels"]}
    out: Dict[str, float] = {}
    for nm in target_names:
        if nm not in by_name:
            continue
        c = by_name[nm]
        if not (np.isfinite(c["x_norm"]) and np.isfinite(c["y_norm"])):
            continue
        xy = np.array([[c["x_norm"], c["y_norm"]]], float)
        pred = field_predict_at_points(chans, xy,
                                       exclude_name=nm if loo else None, sigma_xy=sigma_xy)
        out[nm] = float(pred[0])
    return out


# ----------------------------------------------- Task 3: F / C / null / radial
def signed_spearman(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2 or np.std(x[m]) < 1e-12 or np.std(y[m]) < 1e-12:
        return float("nan")
    return float(spearmanr(x[m], y[m]).correlation)


def _align(record, hidden_names, ictal_ranks):
    by_name = {c["name"]: c for c in record["channels"]}
    names = [n for n in hidden_names if n in by_name and n in ictal_ranks
             and np.isfinite(by_name[n].get("typical_rank", np.nan))]
    return names, by_name


def compute_f_c(record, hidden_names, ictal_ranks, loo: bool = True, sigma_xy=None) -> dict:
    names, by_name = _align(record, hidden_names, ictal_ranks)
    pred = predicted_interictal_order(record, names, loo=loo, sigma_xy=sigma_xy)
    names = [n for n in names if np.isfinite(pred.get(n, np.nan))]
    p = np.array([pred[n] for n in names])
    own = np.array([by_name[n]["typical_rank"] for n in names])
    ict = np.array([ictal_ranks[n] for n in names])
    return {"F": signed_spearman(p, ict), "C": signed_spearman(own, ict),
            "n_hidden": len(names), "names": names,
            "predicted": p.tolist(), "own_rank": own.tolist(), "ictal": ict.tolist()}


def null_F(record, hidden_names, ictal_ranks, n: int = 2000, seed: int = 0,
           loo: bool = True, sigma_xy=None) -> dict:
    base = compute_f_c(record, hidden_names, ictal_ranks, loo=loo, sigma_xy=sigma_xy)
    names = base["names"]
    F_obs = base["F"]
    p = np.array(base["predicted"])
    ict = np.array(base["ictal"])
    if len(names) < 3 or not np.isfinite(F_obs):
        return {"F_obs": F_obs, "p_value": float("nan"), "p95": float("nan"),
                "null_median": float("nan"), "n_hidden": len(names)}
    rng = np.random.default_rng(seed)
    null = np.array([signed_spearman(p, rng.permutation(ict)) for _ in range(n)])
    null = null[np.isfinite(null)]
    if null.size == 0:
        return {"F_obs": F_obs, "p_value": float("nan"), "p95": float("nan"),
                "null_median": float("nan"), "n_hidden": len(names)}
    p_value = float((1 + (null >= F_obs).sum()) / (1 + null.size))
    return {"F_obs": F_obs, "p_value": p_value, "p95": float(np.percentile(null, 95)),
            "null_median": float(np.median(null)), "n_hidden": len(names)}


def radial_baseline_corr(record, hidden_names, ictal_ranks) -> float:
    names, by_name = _align(record, hidden_names, ictal_ranks)
    src = min((c["along_axis_mm"] for c in record["channels"]
               if np.isfinite(c.get("along_axis_mm", np.nan))), default=0.0)
    dist = np.array([by_name[n]["along_axis_mm"] - src for n in names])
    ict = np.array([ictal_ranks[n] for n in names])
    return signed_spearman(dist, ict)


# -------------------------------- bb_auc activation basis (= field_concordance 显著口径)
# 发作侧 = bb_auc (broadband 1-45Hz baseline-robust-z power, [0,10]s 均值, 整体能量非 z-ER)。
# 每发作算 |corr| → 对发作取中位数 (镜像 axis_alignment 的 per-seizure→median 稳健聚合)。
def ictal_bb_auc_by_seizure(ds_sid: str, t0_dir: str = DEF_T0, activation: str = "bb_auc"):
    """返回 (cache_channels: list[str], sz_arrays: list[np.ndarray])，只取 eligible_idxs。"""
    npz = np.load(Path(t0_dir) / f"{ds_sid}.npz", allow_pickle=True)
    meta = json.load(open(Path(t0_dir) / f"{ds_sid}.json"))
    chans = list(npz["channels"])
    out = []
    for idx in meta.get("eligible_idxs", []):
        key = f"{activation}__{idx}"
        if key in npz.files:
            out.append(np.asarray(npz[key], float))
    return chans, out


def _abs_spear(x, y) -> float:
    """|spearman| = |pearson(rank x, rank y)|（rankdata 平均秩，与 scipy spearman 等价但更快）。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    c = np.corrcoef(rankdata(x), rankdata(y))[0, 1]
    return abs(float(c)) if np.isfinite(c) else float("nan")


def ictal_paired_features(ds_sid: str, key_pred: str, key_targ: str, t0_dir: str = DEF_T0):
    """返回 (cache_channels, list[(pred_arr, targ_arr)])，只取两 key 都在的 eligible 发作。"""
    npz = np.load(Path(t0_dir) / f"{ds_sid}.npz", allow_pickle=True)
    meta = json.load(open(Path(t0_dir) / f"{ds_sid}.json"))
    chans = list(npz["channels"])
    out = []
    for idx in meta.get("eligible_idxs", []):
        kp, kt = f"{key_pred}__{idx}", f"{key_targ}__{idx}"
        if kp in npz.files and kt in npz.files:
            out.append((np.asarray(npz[kp], float), np.asarray(npz[kt], float)))
    return chans, out


def per_seizure_paired_median_abscorr(names, cache_channels, paired_arrays):
    """每发作 |spearman(pred[names], targ[names])| (>=3 有限) → 对发作取中位数。"""
    cidx = {n: i for i, n in enumerate(cache_channels)}
    keep = [n for n in names if n in cidx]
    ci = np.array([cidx[n] for n in keep])
    vals = []
    for p_arr, t_arr in paired_arrays:
        p, t = p_arr[ci], t_arr[ci]
        m = np.isfinite(p) & np.isfinite(t)
        if m.sum() >= 3 and np.std(p[m]) > 1e-12 and np.std(t[m]) > 1e-12:
            vals.append(_abs_spear(p[m], t[m]))
    vals = [v for v in vals if np.isfinite(v)]
    return (float(np.median(vals)) if vals else float("nan")), len(vals), keep


def compute_c2_perchannel_energy(record, hidden_names, cache_channels, paired_arrays) -> dict:
    """C2 = 逐通道能量 fingerprint 基线：每发作 |spearman(隐身电极间期 baseline 活跃度 bact,
    其发作能量)| → 对发作取中位数。

    语义精确性（review P1）：这是"隐身电极的间期能量空间 fingerprint 能多大程度预测其发作能量
    空间 fingerprint"，**不是**单纯"活跃通道恒活跃"——后者只含正相关，而这里用 |corr| 把
    **反相关也算成强基线**（保守：让 C2 更难被场超过）。场赢 C2 = 间期传播几何带来的预测力
    超过"用通道自身能量指纹就能解释"。对应 axis_alignment 的 anchor/activity-matched 控制族。"""
    by_name = {c["name"]: c for c in record["channels"]}
    names = [n for n in hidden_names if n in by_name
             and np.isfinite(by_name[n].get("typical_rank", np.nan)) and n in set(cache_channels)]
    C2, nsz, keep = per_seizure_paired_median_abscorr(names, cache_channels, paired_arrays)
    return {"C2": C2, "n_hidden": len(keep), "n_seizures_used": nsz, "names": keep}


def _per_seizure_median_abscorr(x_by_name, names, cache_channels, sz_arrays):
    """对每个发作: |spearman(x[names], bb[names])| (要求>=3 有限) → 对发作取中位数。"""
    cidx = {n: i for i, n in enumerate(cache_channels)}
    keep = [n for n in names if n in cidx]
    ci = np.array([cidx[n] for n in keep])
    x = np.array([x_by_name[n] for n in keep], float)
    vals = []
    for bb in sz_arrays:
        b = bb[ci]
        if np.isfinite(b).sum() >= 3 and np.std(b[np.isfinite(b)]) > 1e-12:
            vals.append(_abs_spear(x, b))
    vals = [v for v in vals if np.isfinite(v)]
    return (float(np.median(vals)) if vals else float("nan"), len(vals), keep)


def maxabscorr_series(x_arrays, ci, sz_arrays):
    """每发作 max_t |spear(x_t, bb[ci])| 的**逐发作序列**（<3 有限或退化 → np.nan，保持发作对齐）。
    max_t = A/B 两模板取最好（review: AB 无优劣）。供 paired 检验用。"""
    out = []
    for bb in sz_arrays:
        b = bb[ci]
        m = np.isfinite(b)
        if m.sum() >= 3 and np.std(b[m]) > 1e-12:
            cs = [_abs_spear(x[m], b[m]) for x in x_arrays]
            cs = [c for c in cs if np.isfinite(c)]
            out.append(max(cs) if cs else float("nan"))
        else:
            out.append(float("nan"))
    return out


def _maxabscorr_over_seizures(x_arrays, ci, sz_arrays):
    s = [v for v in maxabscorr_series(x_arrays, ci, sz_arrays) if np.isfinite(v)]
    return (float(np.median(s)) if s else float("nan")), len(s)


def _permute_within_labels(pos, labels, rng):
    """pos = 有限位置(下标进 ci 空间)；labels = 长度 nh 的组标签；组内 permute pos。"""
    out = pos.copy()
    lp = labels[pos]
    for g in np.unique(lp):
        sel = pos[lp == g]
        if sel.size > 1:
            out[lp == g] = rng.permutation(sel)
    return out


def quantile_bin_labels(values, n_bins=4):
    """values → 分位箱标签（NaN→-1 单独一组，不参与 shuffle 混入）。"""
    v = np.asarray(values, float)
    lab = np.full(v.shape, -1, int)
    fin = np.isfinite(v)
    if fin.sum() == 0:
        return lab
    qs = np.quantile(v[fin], np.linspace(0, 1, n_bins + 1)[1:-1]) if n_bins > 1 else np.array([])
    lab[fin] = np.digitize(v[fin], qs)
    return lab


def field_null_p(pred_arrays, ci, sz_arrays, F_obs, labels_by_sz, n=2000, seed=0):
    """F(_core_only) 在给定 per-seizure shuffle 标签下的 per-subject permutation p。
    labels_by_sz[k] = 长度 nh 的组标签（channel:全 0；within_shaft:杆 id；anchor:bact 分位箱）。
    每发作组内 shuffle 发作能量 → max_t |corr| → 中位数；B 次 → p = P(null≥obs)。"""
    rng = np.random.default_rng(seed)
    # 预算每发作: 有限位置 pos、整条发作能量 b(full)、预测子秩(固定不随 permute 变)、组标签
    pre = []
    for k, bb in enumerate(sz_arrays):
        b = bb[ci]
        fin = np.isfinite(b)
        pos = np.where(fin)[0]
        if pos.size < 3 or np.std(b[pos]) < 1e-12:
            continue
        rxs = [rankdata(x[pos]) for x in pred_arrays if np.std(x[pos]) > 1e-12]
        if not rxs:
            continue
        pre.append((pos, b, rxs, np.asarray(labels_by_sz[k])))
    null = []
    for _ in range(n):
        per_sz = []
        for pos, b, rxs, labels in pre:
            perm = _permute_within_labels(pos, labels, rng)   # pos 的组内重排(绝对下标)
            ry = rankdata(b[perm])                            # 重排后发作能量(pos 序)
            cs = [abs(np.corrcoef(rx, ry)[0, 1]) for rx in rxs]
            cs = [c for c in cs if np.isfinite(c)]
            if cs:
                per_sz.append(max(cs))
        per_sz = [v for v in per_sz if np.isfinite(v)]
        if per_sz:
            null.append(float(np.median(per_sz)))
    null = np.array([v for v in null if np.isfinite(v)])
    if null.size == 0 or not np.isfinite(F_obs):
        return {"p_value": float("nan"), "p95": float("nan"), "null_median": float("nan")}
    return {"p_value": float((1 + (null >= F_obs).sum()) / (1 + null.size)),
            "p95": float(np.percentile(null, 95)), "null_median": float(np.median(null))}


def compute_f_c_activation(record, hidden_names, cache_channels, sz_arrays, record_b=None,
                           loo: bool = True, sigma_xy=None, core_names=None) -> dict:
    """F = 间期顺序场 LOO 预测 → 发作能量, per-seizure |corr|→中位数。
    record_b 给定 → A/B 两模板都建场, **per-seizure 取 max(A,B)** (镜像 axis_alignment max_ab)。
    core_names 给定 → F 场只用 core 通道建 (F_core_only contract, review P1)。"""
    recs = [record] + ([record_b] if record_b is not None else [])

    def _valid(rec):
        bn = {c["name"]: c for c in rec["channels"]}
        return {n for n in hidden_names if n in bn and np.isfinite(bn[n].get("typical_rank", np.nan))}

    cset = set(cache_channels)
    names = [n for n in hidden_names if n in cset and all(n in _valid(r) for r in recs)]
    preds = [predicted_interictal_order(r, names, loo=loo, sigma_xy=sigma_xy, core_names=core_names)
             for r in recs]
    names = [n for n in names if all(np.isfinite(p.get(n, np.nan)) for p in preds)]
    cidx = {n: i for i, n in enumerate(cache_channels)}
    ci = np.array([cidx[n] for n in names], int)
    pred_arrays = [np.array([p[n] for n in names], float) for p in preds]
    own_arrays = [np.array([{c["name"]: c for c in r["channels"]}[n]["typical_rank"] for n in names], float)
                  for r in recs]
    F, nF = _maxabscorr_over_seizures(pred_arrays, ci, sz_arrays)
    C, _ = _maxabscorr_over_seizures(own_arrays, ci, sz_arrays)
    return {"F": F, "C": C, "n_hidden": len(names), "n_seizures_used": nF, "names": names,
            "n_templates": len(recs),
            "predicted": [a.tolist() for a in pred_arrays], "own_rank": [a.tolist() for a in own_arrays]}


def null_F_activation(record, hidden_names, cache_channels, sz_arrays, record_b=None,
                      n: int = 2000, seed: int = 0, loo: bool = True, sigma_xy=None,
                      core_names=None) -> dict:
    base = compute_f_c_activation(record, hidden_names, cache_channels, sz_arrays,
                                  record_b=record_b, loo=loo, sigma_xy=sigma_xy, core_names=core_names)
    names = base["names"]
    F_obs = base["F"]
    if len(names) < 3 or not np.isfinite(F_obs) or base["n_seizures_used"] < 1:
        return {"F_obs": F_obs, "p_value": float("nan"), "p95": float("nan"),
                "null_median": float("nan"), "n_hidden": len(names)}
    cidx = {nm: i for i, nm in enumerate(cache_channels)}
    ci = np.array([cidx[n] for n in names], int)
    pred_arrays = [np.array(a, float) for a in base["predicted"]]   # 每模板一向量
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n):
        per_sz = []
        for bb in sz_arrays:
            b = bb[ci].copy()
            fin = np.isfinite(b)
            if fin.sum() >= 3 and np.std(b[fin]) > 1e-12:
                b[fin] = rng.permutation(b[fin])      # 同一 shuffle 给两模板 → 控 max selection
                cs = [_abs_spear(x[fin], b[fin]) for x in pred_arrays]
                cs = [c for c in cs if np.isfinite(c)]
                if cs:
                    per_sz.append(max(cs))
        per_sz = [v for v in per_sz if np.isfinite(v)]
        if per_sz:
            null.append(float(np.median(per_sz)))
    null = np.array(null)
    if null.size == 0:
        return {"F_obs": F_obs, "p_value": float("nan"), "p95": float("nan"),
                "null_median": float("nan"), "n_hidden": len(names)}
    p_value = float((1 + (null >= F_obs).sum()) / (1 + null.size))
    return {"F_obs": F_obs, "p_value": p_value, "p95": float(np.percentile(null, 95)),
            "null_median": float(np.median(null)), "n_hidden": len(names)}
