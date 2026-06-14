"""Stage 3 可读模板复核（描述性，C2）。先 dry-run 验证 54 个 readable_global 能精确对回
record 列，再走 masked 聚类。聚类结果只分三档：可复核 / 数据太少 / 源-方向歧义限制。
swap_class 是描述性模板复核，不是 Stage 3 证据。"""
import os, re, glob, json, argparse
import numpy as np, pandas as pd

import sys; sys.path.insert(0, os.getcwd())
from src.interictal_propagation import (
    compute_adaptive_cluster_stereotypy, build_cluster_templates)
from src.rank_displacement import compute_swap_score_sweep
from src.lagpat_rank_audit import build_masked_kmeans_features   # P1-3: distinct-pattern count

ROOT = "results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous"

required = {"cell", "seed", "tag", "event_id", "t_on", "source", "sign",
            "n_part", "propagation_class", "source_core_ignite_frac",
            "other_core_ignite_frac", "duration"}


def find_sidecar(tag):
    """递归唯一定位 sidecar（生成 event_types 行的源文件）。非唯一 → None（dry-run 会报）。"""
    hits = glob.glob(os.path.join(ROOT, "**", f"sidecar_{tag}.json"), recursive=True)
    return hits[0] if len(hits) == 1 else None


def find_record(tag):
    """P1-1: record 必须是 sidecar 的同级 record/<tag>/，不能全局 glob 取第一个。"""
    sc = find_sidecar(tag)
    if sc is None:
        return None
    hits = glob.glob(os.path.join(os.path.dirname(sc), "record", tag, "*_lagPat_withFreqCent.npz"))
    return hits[0] if len(hits) == 1 else None   # 要求唯一；重复/缺失 → None


def dry_run():
    rg = pd.read_csv(os.path.join(ROOT, "readable_global_events.csv"))
    assert required <= set(rg.columns), f"missing cols: {required - set(rg.columns)}"
    assert set(rg["propagation_class"]) == {"readable_global"}
    assert rg["source"].isin(["neg", "pos"]).all(), "readable_global source must be neg/pos"
    print(f"[dry-run] {len(rg)} readable_global events; "
          f"source={dict(rg.source.value_counts())}; sign={dict(rg.sign.value_counts())}")

    montage_ref, problems = None, []
    for tag, sub in rg.groupby("tag"):
        sc = find_sidecar(tag)
        if sc is None:
            n = len(glob.glob(os.path.join(ROOT, "**", f"sidecar_{tag}.json"), recursive=True))
            problems.append(f"{tag}: sidecar not unique (found {n})"); continue
        rec = find_record(tag)
        if rec is None:
            n = len(glob.glob(os.path.join(os.path.dirname(sc), "record", tag, "*_lagPat_withFreqCent.npz")))
            problems.append(f"{tag}: sibling record not unique under {os.path.dirname(sc)}/record (found {n})"); continue
        npz = np.load(rec, allow_pickle=True)
        n_events = npz["lagPatRank"].shape[1]
        names = [str(c) for c in npz["chnNames"]]
        if montage_ref is None: montage_ref = names
        elif names != montage_ref: problems.append(f"{tag}: montage mismatch")
        # event_id == lagPat 列号
        for eid in sub.event_id:
            if not (0 <= eid < n_events):
                problems.append(f"{tag}: event_id {eid} out of range {n_events}")
        # t_on 同源核对：sidecar 事件按 t_on 排序后第 event_id 个的 t_on == csv t_on
        sc_ev = sorted(json.load(open(sc)).get("events", []), key=lambda e: e["t_on"])
        for _, r in sub.iterrows():
            if int(r.event_id) < len(sc_ev):
                if abs(sc_ev[int(r.event_id)]["t_on"] - r.t_on) > 1e-6:
                    problems.append(f"{tag}: event_id {r.event_id} t_on misaligned")
    assert not problems, "DRY-RUN FAILED:\n" + "\n".join(problems)
    print(f"[dry-run] OK — all {len(rg)} events map to record columns; montage consistent "
          f"({len(montage_ref)} contacts)")
    return rg, montage_ref


def pool_subset(rg_subset, montage_ref):
    """把 rg_subset 里每个 (tag,event_id) 对应的 lagPat 列抽出来，跨 tag 拼成一个
    合成 subject（masked-loader-readable）。montage 必须一致（dry-run 已 assert）。"""
    ranks, bools, raws = [], [], []
    for tag, sub in rg_subset.groupby("tag"):
        npz = np.load(find_record(tag), allow_pickle=True)
        assert [str(c) for c in npz["chnNames"]] == montage_ref
        cols = sub.event_id.astype(int).to_numpy()
        ranks.append(npz["lagPatRank"][:, cols])
        bools.append(npz["eventsBool"][:, cols])
        raws.append(npz["lagPatRaw"][:, cols])
    R = np.concatenate(ranks, axis=1)
    B = np.concatenate(bools, axis=1)
    return R, B, montage_ref


def n_unique_patterns(R, B):
    """P1-3: 有效形态多样性 = masked 特征矩阵里 distinct 行数。distinct 行很少时
    stable_k 不能当真（54 个可读事件已复核：pooled=13, neg=6, pos=7）。"""
    f = np.asarray(build_masked_kmeans_features(R, B, impute='event_median'))
    return int(np.unique(np.round(f, 6), axis=0).shape[0])


def cluster_layer(R, B, names, label):
    n_ev = R.shape[1]
    nuq = n_unique_patterns(R, B) if n_ev else 0
    out = dict(layer=label, n_events=int(n_ev), n_unique_masked_patterns=nuq)
    if n_ev < 8:                       # 事件太少（k=2 下限不稳）
        out["verdict"] = "数据太少（事件数<8）"; return out
    pr2 = compute_adaptive_cluster_stereotypy(R, B, names, use_masked_features=True)
    chosen_k = int(pr2["chosen_k"])
    out.update(chosen_k=chosen_k, stable_k=pr2.get("stable_k"),
               diversity_limited=bool(nuq < 10))   # P1-3: <10 distinct -> stable_k 慎读
    if chosen_k == 2 and pr2.get("labels"):
        labels = np.array(pr2["labels"], int)
        t0, t1 = build_cluster_templates(R, B, labels, 2)
        vm0 = np.isfinite(t0) & (t0 >= 0); vm1 = np.isfinite(t1) & (t1 >= 0)
        try:
            swap = compute_swap_score_sweep(t0, t1, vm0, vm1, n_perm=1000, seed=0)
            out["swap_class_descriptive"] = swap.get("swap_class")   # C2: 描述性，非 Stage3 证据
            out["decision_k"] = swap.get("decision_k")
        except Exception as e:
            out["swap_error"] = repr(e)
    return out


def cluster_all(rg, montage_ref):
    layers = []
    # (a) 全池化 —— 仅 sanity
    R, B, names = pool_subset(rg, montage_ref)
    layers.append({**cluster_layer(R, B, names, "pooled_sanity"),
                   "note": "全池化仅 sanity；混了 cell/seed/source，不作模板结论"})
    # (b) 按 source 分层
    for src in ("neg", "pos"):
        sub = rg[rg.source == src]
        R, B, names = pool_subset(sub, montage_ref)
        layers.append(cluster_layer(R, B, names, f"source_{src}"))
    # (c) 按 top cell 分层（事件最多的 cell）
    top_cell = rg.cell.value_counts().idxmax()
    sub = rg[rg.cell == top_cell]
    R, B, names = pool_subset(sub, montage_ref)
    layers.append({**cluster_layer(R, B, names, f"top_cell:{top_cell}"),
                   "n_seeds": int(sub.seed.nunique())})
    return layers


def verdict_of(layer):
    if str(layer.get("verdict", "")).startswith("数据太少"): return layer["verdict"]
    if layer.get("diversity_limited"):
        return f"数据太少（形态多样性不足: n_unique={layer['n_unique_masked_patterns']}）"  # P1-3
    if "swap_class_descriptive" not in layer: return "数据太少"
    return "可复核"   # "源-方向歧义限制" 由人读 source_pos 层 + sign_by_source 判定，写进归档段


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="只跑 event_id->lagPat 列映射核对（C5），不进聚类")
    args = ap.parse_args()

    if args.dry_run:
        dry_run()
        return

    # 全模式：先 dry-run 焊死映射，再走 masked 三层聚类
    rg, montage_ref = dry_run()
    layers = cluster_all(rg, montage_ref)
    for layer in layers:
        layer["verdict"] = verdict_of(layer)

    summary = {
        "n_readable_global_events": int(len(rg)),
        "source_counts": {str(k): int(v) for k, v in rg.source.value_counts().items()},
        "sign_counts": {str(k): int(v) for k, v in rg.sign.value_counts().items()},
        "montage": montage_ref,
        "n_contacts": len(montage_ref),
        "layers": layers,
        "note": "swap_class 是描述性模板复核，不是 Stage 3 证据（C2）。",
    }
    out_path = os.path.join(ROOT, "stage3_readable_templates_summary.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print(f"[full] wrote {out_path}")
    for layer in layers:
        print(f"  {layer['layer']:>20}: n_events={layer['n_events']:>2} "
              f"n_unique_masked_patterns={layer['n_unique_masked_patterns']:>2} "
              f"chosen_k={layer.get('chosen_k')} "
              f"swap_class_descriptive={layer.get('swap_class_descriptive')} "
              f"| verdict={layer['verdict']}")


if __name__ == "__main__":
    main()
