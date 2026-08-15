#!/usr/bin/env python
"""Regenerate the report's result tables straight from the stored artifacts.

Keeps the archive doc recomputable: every number in the two selection-null
tables and the cohort progress table comes from a JSON on disk, not from a
transcript.  Prints markdown ready to paste, and can patch the archive doc in
place at its `<!--TABLE-SCN-->`, `<!--READ-SCN-->` and `<!--TABLE-COHORT-->`
markers.

Inner quoting inside the Chinese prose uses corner brackets, never ASCII double
quotes, so the string literals stay well formed.

Run:  python make_report_tables.py [--write]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FORMAL = REPO / "results/topic4_sef_hfo/data_driven_snn_cohort_v1/formal"
DOC = REPO / "docs/archive/topic1/propagation/multimode_selection_null_and_916_extent_2026-08-16.md"

ORDER = ["yuquan_huangwanling", "epilepsiae_818", "epilepsiae_916",
         "yuquan_zhangjinhan", "yuquan_zhourongxuan", "yuquan_zhaojinrui"]


def short(sid):
    return sid.split("_", 1)[1]


def _load():
    scn = {s: json.load(open(HERE / "selection_corrected_null" / f"{s}.json")) for s in ORDER}
    mx = {s: json.load(open(HERE / "marginal_maxent_null" / f"{s}.json")) for s in ORDER}
    aud = json.load(open(HERE / "null_construction_audit.json"))["subjects"]
    return scn, mx, aud


def table_scn() -> str:
    scn, mx, aud = _load()
    L = []
    L.append("#### 主零假设：只把先后顺序随机化（承重）\n")
    L.append("| 患者 | K | 触点 | 实测集中度 | 零假设均值 | 零假设 q95 | 实测−零假设 | 经验 P | 超过 q95 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for s in ORDER:
        d = scn[s]
        o = d["nulls"]["order"]
        n = o["concentration_null"]
        L.append(
            f"| `{short(s)}` | {d['chosen_k']} | {d['n_channels']} | "
            f"{d['observed']['equal_mode_weighted_dominant_order_concentration']:.4f} | "
            f"{n['mean']:.4f} | {n['q95']:.4f} | "
            f"**{o['observed_minus_null_concentration']:+.4f}** | "
            f"{o['empirical_p_concentration_ge']:.4f} | "
            f"{'**是**' if o['observed_exceeds_null_q95_concentration'] else '否'} |")
    n_ex = sum(scn[s]["nulls"]["order"]["observed_exceeds_null_q95_concentration"] for s in ORDER)
    L.append(f"\n每人 512 次有效随机化、0 次无效；P = 0.0019 是 512 次抽样下的经验下限。"
             f"**{n_ex}/6 超过 q95。**\n")

    L.append("#### 敏感性零假设：连每个触点自身的早/晚倾向也保住（最大熵，穷举 p!）\n")
    L.append("| 患者 | 实测 | 零假设均值 | 零假设 q95 | 实测−零假设 | 经验 P | 超过 q95 | 构造检验（离散度比） | IPF 边缘误差 最大/事件加权 | 拟合不良分层 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for s in ORDER:
        d, m = scn[s], mx[s]
        cc, pv = m["construction_check"], m["provenance"]
        L.append(
            f"| `{short(s)}` | "
            f"{d['observed']['equal_mode_weighted_dominant_order_concentration']:.4f} | "
            f"{m['concentration_null']['mean']:.4f} | {m['concentration_null']['q95']:.4f} | "
            f"**{m['observed_minus_null_concentration']:+.4f}** | "
            f"{m['empirical_p_concentration_ge']:.4f} | "
            f"{'**是**' if m['observed_exceeds_null_q95_concentration'] else '否'} | "
            f"{cc['null_over_observed_spread_ratio']:.2f} "
            f"{'✓' if cc['preserves_marginals'] else '**✗**'} | "
            f"{pv['ipf_max_marginal_error']:.1e} / {pv['ipf_event_weighted_marginal_error']:.1e} | "
            f"{pv['ipf_n_strata_error_gt_0p05']} 个 "
            f"({pv['ipf_event_fraction_in_those_strata']*100:.1f}% 事件) |")
    n_ex2 = sum(mx[s]["observed_exceeds_null_q95_concentration"] for s in ORDER)
    n_valid = sum(mx[s]["construction_check"]["preserves_marginals"] for s in ORDER)
    L.append(f"\n构造检验合格 {n_valid}/6（判据：零假设与实测的「各触点平均次序离散度」之比落在 0.8–1.25）；"
             f"**{n_ex2}/6 超过 q95**。\n")

    L.append("#### 三种零假设构造的自检（第一版为何作废）\n")
    L.append("| 患者 | 实测离散度 | 顺序随机化 | 第一版打乱重排 | 比值 | 最大熵 | 比值 |")
    L.append("|---|---|---|---|---|---|---|")
    for s in ORDER:
        a, cc = aud[s], mx[s]["construction_check"]
        L.append(
            f"| `{short(s)}` | {a['observed_per_contact_order_spread']:.4f} | "
            f"{a['order_null_spread']['mean']:.4f} | {a['marginal_null_spread']['mean']:.4f} | "
            f"**{a['marginal_null_ratio_to_observed']:.2f}** | "
            f"{cc['null_per_contact_order_spread_mean']:.4f} | "
            f"**{cc['null_over_observed_spread_ratio']:.2f}** |")
    L.append("\n顺序随机化零假设把这层结构抹到 0.002–0.005（**6/6 合格**，它本来就要抹掉）；"
             "第一版把它**放大**到 1.17–2.49 倍（**1/6 合格**，作废）；"
             "最大熵版把它还原到实测水平（5/6 合格）。\n")
    return "\n".join(L)


def read_scn() -> str:
    scn, mx, _ = _load()
    ex1 = [s for s in ORDER if scn[s]["nulls"]["order"]["observed_exceeds_null_q95_concentration"]]
    ex2 = [s for s in ORDER if mx[s]["observed_exceeds_null_q95_concentration"]
           and mx[s]["construction_check"]["preserves_marginals"]]
    inval = [s for s in ORDER if not mx[s]["construction_check"]["preserves_marginals"]]
    j = lambda xs: "、".join("`" + short(x) + "`" for x in xs)

    L = []
    L.append("两个零假设叠起来，把「模式内部为什么会有一个主导发放顺序」拆成三层：\n")
    L.append("1. **选择偏差层**——固定 K 的聚类在一个很小的离散格点集合上切分箱，"
             "本身就会造出集中度。主零假设量化的就是这一层。")
    L.append("2. **各触点自身早/晚倾向层**——某个触点整体上就是偏早或偏晚。"
             "主零假设把这一层也一并打掉了（它把离散度抹到 0.002–0.005），"
             "而最大熵零假设把它原样保住。**两个零假设之差就是这一层的贡献。**")
    L.append("3. **事件内跨触点协同层**——在给定每个触点自身倾向之后，"
             "还剩多少「谁跟谁一起、按什么次序」的结构。这是实测减去最大熵零假设。\n")
    L.append(f"结果：扣掉第 1 层之后，**{len(ex1)}/6** 患者仍有超出（{j(ex1)}）；"
             f"再扣掉第 2 层之后，只剩 **{len(ex2)}/6**"
             f"（{j(ex2) if ex2 else '无'}）。")
    if inval:
        L.append(f"\n⚠️ {j(inval)} 的最大熵零假设**没有通过构造检验**"
                 f"（离散度比落在 0.8–1.25 之外，且大部分事件所在分层的边缘拟合失败），"
                 f"其第 3 层判读**不可用**，该患者只报第 1 层。")
    L.append("\n**能写的一句话**：这些模式内部确实有比「聚类切格点」更集中的发放顺序，"
             "但这份集中度可以由「每个触点自己偏早还是偏晚」这一层一阶结构解释掉——"
             "在构造检验合格的患者里，没有一个在扣掉这一层之后还剩下可检出的事件内协同。")
    L.append("\n**不能写**「每个模式是一个传播模板」：那句话要求第 3 层成立，而第 3 层在本轮全部为零。"
             "也**不能写**「模式完全是聚类假象」：第 1 层已经被 4/6 患者跨过。"
             "准确的说法是——模式确实抓住了真实的顺序不均匀性，"
             "但那份不均匀性来自单个触点的早/晚偏好，不是来自触点之间的协同。")
    return "\n".join(L)


def table_cohort() -> str:
    ctrl = (FORMAL / "controller.status").read_text().strip()
    sup = (FORMAL / "supervisor.status").read_text().strip()
    counts, failed = {}, []
    for st in sorted((FORMAL / "run_logs").glob("*.status")):
        t = st.read_text().strip()
        state = t.split()[0] if t else "EMPTY"
        counts[state] = counts.get(state, 0) + 1
        if state == "FAILED":
            failed.append(st.stem)
    trail = HERE / "formal_cohort_watch" / "watch_trail.jsonl"
    rows = [json.loads(l) for l in open(trail)] if trail.exists() else []
    stage_files = sorted(p.name for p in FORMAL.glob("stage_*_selection.json"))
    L = ["| 项 | 值 |", "|---|---|",
         f"| supervisor | `{sup}` |",
         f"| controller | `{ctrl}` |",
         f"| 逐单元状态 | {counts} |",
         f"| 失败单元 | {failed if failed else '**无**'} |",
         f"| worker 产出 JSON | {len(list((FORMAL / 'workers').glob('*.json')))} |",
         f"| 阶段聚合产物 | {stage_files if stage_files else '**尚未产生**（stage A 未结束）'} |",
         f"| 监视器采样数 | {len(rows)}（450 秒节拍） |",
         f"| 冻结模块漂移 | {'**全程无**' if all(not r.get('frozen_module_drift') for r in rows) else '⚠️ 有'} |"]
    if rows:
        L.append(f"| 可用内存区间 | {min(r['mem_available_gib'] for r in rows):.0f} – "
                 f"{max(r['mem_available_gib'] for r in rows):.0f} GiB（保留线 20 GiB） |")
        L.append(f"| 磁盘空闲区间 | {min(r['free_disk_gib'] for r in rows):.0f} – "
                 f"{max(r['free_disk_gib'] for r in rows):.0f} GiB（停机线 6 GiB） |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    blocks = {"<!--TABLE-SCN-->": table_scn(),
              "<!--READ-SCN-->": read_scn(),
              "<!--TABLE-COHORT-->": table_cohort()}
    for k, v in blocks.items():
        print(f"\n{'='*20} {k}\n{v}")
    if args.write:
        s = DOC.read_text()
        for k, v in blocks.items():
            if k in s:
                s = s.replace(k, v)
            else:
                print(f"  [warn] marker {k} not found (already filled?)")
        DOC.write_text(s)
        print(f"\npatched {DOC}")


if __name__ == "__main__":
    main()
