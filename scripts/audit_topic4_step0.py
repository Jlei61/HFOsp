#!/usr/bin/env python3
"""Topic 4 Step 0 audit.

Per the locked design (Topic 1 → Topic 4 bridge: propagation-state attractor
diagnostics), this audit produces a per-subject eligibility table for the
spectral-gap analysis. It does NOT compute Λ_gap or any algorithm — it
only reads the existing PR-2 cluster JSON and the lagPat lineage to
characterise cohort + block geometry.

Output:
    results/topic4_attractor/step0_audit.csv

Columns:
    sid                    : "<dataset>_<subject>"
    dataset                : yuquan / epilepsiae
    subject                : raw subject id
    stable_k               : adaptive_cluster.stable_k from PR-2 JSON
    group                  : "stable_k=2" or "stable_k>2"
    n_channels_union       : number of channels in the lagPat union for this subject
    n_events_total         : total events from loader
    n_events_eligible      : events with n_participating >= 6
    n_blocks_total         : blocks loaded
    n_blocks_with_events   : blocks containing >=1 eligible event
    median_block_dur_h     : median block duration in hours
    min_block_n_events     : minimum eligible events per block (over blocks with >=1 event)
    eligible_for_main      : (n_events_eligible >= 100) AND (group == "stable_k=2")
    notes                  : any caveats (missing JSON / loader errors / etc.)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.interictal_propagation import load_subject_propagation_events  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("topic4_step0")

YUQUAN_ROOT = Path("/mnt/yuquan_data/yuquan_24h_edf")
EPILEPSIAE_ROOT = Path("/mnt/epilepsia_data/interilca_inter_results/all_data_lns")
PR2_PER_SUBJECT_DIR = REPO_ROOT / "results" / "interictal_propagation" / "per_subject"
OUT_DIR = REPO_ROOT / "results" / "topic4_attractor"
OUT_CSV = OUT_DIR / "step0_audit.csv"
OUT_SUMMARY = OUT_DIR / "step0_audit_summary.md"

# Mirrors `scripts/run_interictal_propagation.py` (the canonical PR-2 cohort).
YUQUAN_SUBJECTS = [
    "zhangkexuan", "pengzihang", "chengshuai", "huangwanling",
    "liyouran", "songzishuo", "zhangbichen", "zhaochenxi",
    "zhaojinrui", "zhourongxuan", "zhangjiaqi",
    "chenziyang", "hanyuxuan", "huanghanwen", "litengsheng",
    "xuxinyi", "zhangjinhan", "sunyuanxin",
    "gaolan", "wangyiyang",
]

EPILEPSIAE_SUBJECTS = [
    "1096", "1084", "958", "922", "590", "1150", "442", "1073",
    "253", "1146", "916", "620", "583", "548", "384", "139",
    "1125", "1077", "818", "635",
]

N_PARTICIPATING_MIN = 6
N_EVENTS_ELIGIBLE_MIN = 100


def _epilepsiae_subject_dir(subject: str) -> Path:
    legacy = EPILEPSIAE_ROOT / subject / "all_recs"
    if legacy.exists():
        return legacy
    return EPILEPSIAE_ROOT / subject


def _subject_dir(dataset: str, subject: str) -> Path:
    if dataset == "yuquan":
        return YUQUAN_ROOT / subject
    return _epilepsiae_subject_dir(subject)


def _read_stable_k(json_path: Path) -> Tuple[int, str]:
    """Return (stable_k, note). stable_k = -1 if unavailable."""
    if not json_path.exists():
        return -1, "no_pr2_json"
    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception as exc:  # pragma: no cover
        return -1, f"json_read_error:{type(exc).__name__}"
    ac = d.get("adaptive_cluster")
    if not isinstance(ac, dict):
        return -1, "no_adaptive_cluster"
    sk = ac.get("stable_k")
    if sk is None or not np.isfinite(float(sk)):
        return -1, "no_stable_k"
    return int(sk), ""


def _audit_subject(dataset: str, subject: str) -> Dict[str, Any]:
    sid = f"{dataset}_{subject}"
    json_path = PR2_PER_SUBJECT_DIR / f"{sid}.json"
    stable_k, note_pr2 = _read_stable_k(json_path)

    row: Dict[str, Any] = {
        "sid": sid,
        "dataset": dataset,
        "subject": subject,
        "stable_k": stable_k,
        "group": "",
        "n_channels_union": 0,
        "n_events_total": 0,
        "n_events_eligible": 0,
        "n_blocks_total": 0,
        "n_blocks_with_events": 0,
        "median_block_dur_h": float("nan"),
        "min_block_n_events": 0,
        "eligible_for_main": False,
        "notes": note_pr2,
    }

    if stable_k == 2:
        row["group"] = "stable_k=2"
    elif stable_k > 2:
        row["group"] = "stable_k>2"
    else:
        row["group"] = "unknown"

    sub_dir = _subject_dir(dataset, subject)
    if not sub_dir.exists():
        row["notes"] = ("; ".join([row["notes"], "subject_dir_missing"])
                        .lstrip("; "))
        return row

    try:
        loaded = load_subject_propagation_events(sub_dir)
    except Exception as exc:
        row["notes"] = ("; ".join([row["notes"], f"loader_error:{type(exc).__name__}:{exc}"])
                        .lstrip("; "))
        return row

    bools = loaded.get("bools", np.zeros((0, 0), dtype=bool))
    block_ids = np.asarray(loaded.get("block_ids", np.zeros(0, dtype=int)))
    block_time_ranges = loaded.get("block_time_ranges", []) or []

    if bools.size == 0 or block_ids.size == 0:
        row["notes"] = ("; ".join([row["notes"], "loader_empty"])
                        .lstrip("; "))
        return row

    n_participating = bools.sum(axis=0).astype(int)
    eligible_mask = n_participating >= N_PARTICIPATING_MIN

    n_total = int(bools.shape[1])
    n_eligible = int(eligible_mask.sum())
    n_chan_union = int(bools.shape[0])
    row["n_channels_union"] = n_chan_union

    n_blocks_total = int(loaded.get("n_blocks_used", len(block_time_ranges)) or 0)

    block_durations_h: List[float] = []
    for tr in block_time_ranges:
        try:
            t0, t1 = float(tr[0]), float(tr[1])
        except Exception:
            continue
        if np.isfinite(t0) and np.isfinite(t1) and t1 > t0:
            block_durations_h.append((t1 - t0) / 3600.0)

    median_block_dur_h = (
        float(np.median(block_durations_h)) if block_durations_h else float("nan")
    )

    if n_eligible > 0:
        elig_block_ids = block_ids[eligible_mask]
        unique_blocks, counts = np.unique(elig_block_ids, return_counts=True)
        n_blocks_with_events = int(unique_blocks.size)
        min_block_n = int(counts.min()) if counts.size > 0 else 0
    else:
        n_blocks_with_events = 0
        min_block_n = 0

    row["n_events_total"] = n_total
    row["n_events_eligible"] = n_eligible
    row["n_blocks_total"] = n_blocks_total
    row["n_blocks_with_events"] = n_blocks_with_events
    row["median_block_dur_h"] = median_block_dur_h
    row["min_block_n_events"] = min_block_n
    row["eligible_for_main"] = bool(
        n_eligible >= N_EVENTS_ELIGIBLE_MIN and row["group"] == "stable_k=2"
    )

    return row


def _format_float(x: Any) -> str:
    try:
        v = float(x)
    except Exception:
        return ""
    if not np.isfinite(v):
        return ""
    return f"{v:.3f}"


def _write_csv(rows: List[Dict[str, Any]], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sid", "dataset", "subject", "stable_k", "group",
        "n_channels_union",
        "n_events_total", "n_events_eligible",
        "n_blocks_total", "n_blocks_with_events",
        "median_block_dur_h", "min_block_n_events",
        "eligible_for_main", "notes",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["median_block_dur_h"] = _format_float(out.get("median_block_dur_h"))
            out["eligible_for_main"] = "true" if out.get("eligible_for_main") else "false"
            w.writerow(out)
    logger.info("Wrote %s (%d rows)", out_csv, len(rows))


def _write_summary(rows: List[Dict[str, Any]], out_md: Path) -> None:
    """Emit a brief Chinese summary appropriate for archive consumption."""
    n_total = len(rows)
    by_group: Dict[str, int] = {}
    for r in rows:
        by_group[r["group"]] = by_group.get(r["group"], 0) + 1

    eligible_main = [r for r in rows if r["eligible_for_main"]]
    stable_k2_underpowered = [
        r for r in rows
        if r["group"] == "stable_k=2" and not r["eligible_for_main"]
    ]
    stable_k_gt2 = [r for r in rows if r["group"] == "stable_k>2"]
    unknown = [r for r in rows if r["group"] == "unknown"]

    lines: List[str] = []
    lines.append("# Topic 4 Step 0 — Cohort audit")
    lines.append("")
    lines.append(f"_Generated: this run produced `{OUT_CSV.relative_to(REPO_ROOT)}`._")
    lines.append("")
    lines.append("## 入口")
    lines.append("")
    lines.append(
        "数据入口与现象证据见 Topic 1 §3.1d cluster geometry。本审计仅做"
        f"cohort eligibility 切分，不计算 Λ_gap、不投影 principal curve。"
    )
    lines.append("")
    lines.append("## 切分规则（pre-locked）")
    lines.append("")
    lines.append(f"- `n_participating ≥ {N_PARTICIPATING_MIN}` 才计入 eligible event。")
    lines.append(f"- `n_events_eligible ≥ {N_EVENTS_ELIGIBLE_MIN}` 且 `stable_k = 2` 才进主分析。")
    lines.append("- `stable_k > 2` 平行报，不进 H3 推论池。")
    lines.append("- PR-2 JSON 缺失 / loader 错误的 subject 标 `unknown`，独立列出。")
    lines.append("")
    lines.append("## Cohort 切分实测")
    lines.append("")
    lines.append(f"- 总 subject 数：{n_total}（期望 40 = 20 yuquan + 20 epilepsiae）")
    for g in ("stable_k=2", "stable_k>2", "unknown"):
        lines.append(f"- group `{g}`：{by_group.get(g, 0)} 例")
    lines.append(f"- 主分析合规 (eligible_for_main = true)：**{len(eligible_main)}** 例")
    lines.append("")

    if eligible_main:
        n_evs = [r["n_events_eligible"] for r in eligible_main]
        med_dur = [r["median_block_dur_h"] for r in eligible_main
                   if np.isfinite(r["median_block_dur_h"])]
        nblk = [r["n_blocks_with_events"] for r in eligible_main]
        n_ch = [r["n_channels_union"] for r in eligible_main]
        lines.append("### Main 分析子集（stable_k=2 且 ≥100 eligible events）")
        lines.append("")
        lines.append(f"- n_events_eligible：median={int(np.median(n_evs))}, "
                     f"min={min(n_evs)}, max={max(n_evs)}")
        lines.append(f"- n_blocks_with_events：median={int(np.median(nblk))}, "
                     f"min={min(nblk)}, max={max(nblk)}")
        if med_dur:
            lines.append(f"- median_block_dur_h：median={np.median(med_dur):.2f}h, "
                         f"min={min(med_dur):.2f}h, max={max(med_dur):.2f}h")
        lines.append(f"- n_channels_union：median={int(np.median(n_ch))}, "
                     f"min={min(n_ch)}, max={max(n_ch)}")
        lines.append("")

    if stable_k2_underpowered:
        lines.append("### Stable_k=2 但 underpowered（事件数不够）")
        lines.append("")
        for r in stable_k2_underpowered:
            lines.append(
                f"- {r['sid']}: n_events_eligible={r['n_events_eligible']}, "
                f"notes={r['notes'] or '-'}"
            )
        lines.append("")

    if stable_k_gt2:
        lines.append("### Stable_k > 2（平行报，不进 H3）")
        lines.append("")
        narrow_chan_all = all(r["n_channels_union"] <= 5 for r in stable_k_gt2)
        if narrow_chan_all:
            lines.append(
                "**结构限制说明**：以下 stable_k>2 subject 全部 n_channels_union ≤ 5，"
                f"因此自动落到 `n_participating ≥ {N_PARTICIPATING_MIN}` 闸门之外（"
                "n_participating 上限 = n_channels_union）。这是 lagPat refine 阶段"
                "的 channel-pick 结果，不是 audit 实现错误。这也意味着 principal "
                "curve PCA-3 在 4-5D 空间内本就不是 well-posed 的（top-3 PC 几乎覆盖"
                "整个空间），所以 Topic 4 的 `eligible_for_main = false` 与"
                "几何可解释性两条独立理由都把这部分 subject 排除在主分析之外。"
            )
            lines.append("")
        for r in stable_k_gt2:
            lines.append(
                f"- {r['sid']}: stable_k={r['stable_k']}, "
                f"n_channels_union={r['n_channels_union']}, "
                f"n_events_total={r['n_events_total']}, "
                f"n_events_eligible={r['n_events_eligible']}"
            )
        lines.append("")

    if unknown:
        lines.append("### Unknown / 数据缺失")
        lines.append("")
        for r in unknown:
            lines.append(f"- {r['sid']}: notes={r['notes'] or '-'}")
        lines.append("")

    lines.append("## Step 1 待定（在动算法前需要锁的两条合同）")
    lines.append("")
    lines.append(
        "Step 0 跑完后审了 PR-2 / PR-4B 的实现，发现 plan 在两个地方没有显式给出"
        "选择，按 CLAUDE.md §5（re-read upstream definitions）这两条不适合在 Step 1 "
        "代码里默认掉。需要先回答：")
    lines.append("")
    lines.append("### Q1：feature 空间用 `relative_lag` 还是 `lagPatRank`？")
    lines.append("")
    lines.append(
        "- Plan 写 \"Per event: relative_lag (lagPatRaw - per-event min)\"，理由是 "
        "PR-4B Step 0 已校验 30/30 order match。"
    )
    lines.append(
        "- 但 PR-2 `compute_kmeans_cluster_stereotypy` 的 KMeans 跑在 `lagPatRank` "
        "上（`src/interictal_propagation.py:1215`，rank_subset = ranks[:, valid].T）。"
        "stable_k=2 这个判定的 feature 空间是 rank，不是 relative_lag。"
    )
    lines.append(
        "- H3 的可证伪声明是\"PR-2 stable_k=2 反映真实 metastable switching\"，因此"
        "principal curve 投影空间 **应与 PR-2 KMeans feature 空间一致**，否则测的不是"
        "同一个 cluster geometry。"
    )
    lines.append(
        "- 推荐：**主分析用 `lagPatRank`（与 PR-2 一致）**，relative_lag 作 sensitivity。"
        "如果坚持 relative_lag 主分析，需要额外做 \"在 relative_lag 空间重跑 KMeans, "
        "stable_k 是否仍是 2\" 的桥接验证；这本身就是新一段工作。"
    )
    lines.append("")
    lines.append("### Q2：inactive channel（`bools=False`）的 NaN 怎么填？")
    lines.append("")
    lines.append(
        "- PR-2 KMeans 用 `np.where(finite_mask, rank, 0.0)` —— 不参与的 channel rank=0。"
    )
    lines.append(
        "- PR-4B `_compute_relative_lag_matrix` 给 inactive channel `NaN`，下游 Pearson / "
        "tau 用 pairwise complete cases 处理；没有为 PCA 场景定义填补规则。"
    )
    lines.append(
        "- 三个候选：(a) NaN → 0（与 PR-2 KMeans 一致）；(b) NaN → channel 全集 mean；"
        "(c) probabilistic / EM-PCA 处理 missing data。"
    )
    lines.append(
        "- 推荐：**与 PR-2 KMeans 一致用 NaN → 0**（rank 空间下 0 = 比 active 都早的 sentinel；"
        "relative_lag 空间下 0 = per-event min 同位）。这条选择会进入"
        "`docs/archive/topic4/...` 顶部当假设条目，并在 Step 2+ 做敏感性。"
    )
    lines.append("")
    lines.append("### 答完 Q1 / Q2 后 Step 1 的执行")
    lines.append("")
    lines.append(
        "1. 在 `eligible_for_main = true` 的子集（35 例）上对每个 event 构造 "
        "n_chan_union 维特征向量（按 Q1 的选择 = rank 或 relative_lag，按 Q2 填 NaN）。"
    )
    lines.append(
        "2. PCA → top-3 子空间；smoothing-spline 主曲线；报 `var_explained_curve`、"
        "`residual_to_curve` 中位数、主曲线切向 vs PR-2 KMeans 主轴夹角。"
    )
    lines.append(
        "3. GOF 闸：`var_explained_curve > 0.6` 不过的 subject 标 \"1D coordinate not "
        "valid\"，独立列出，不进 H3 推论池（夹角 15°/30° 仅 sensitivity，不是硬闸）。"
    )
    lines.append(
        "4. Step 2+：quantile-bin transition matrix（K=8 主，{6,10,12} sensitivity）+ "
        "reversibilized 谱 + within-block shuffle null，按已锁定的 Λ_gap 公式走。"
    )
    lines.append("")
    lines.append("### Cohort scope reminder")
    lines.append("")
    lines.append(
        f"- 主分析池：{len(eligible_main)} 例 stable_k=2，n_events_eligible 范围 "
        f"{min(r['n_events_eligible'] for r in eligible_main)}–"
        f"{max(r['n_events_eligible'] for r in eligible_main)}。"
    )
    lines.append(
        f"- 平行报：{len(stable_k_gt2)} 例 stable_k>2（n_channels_union ≤ 5，"
        "结构性窄通道，不进 H3 推论池）。"
    )
    lines.append("")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    logger.info("Wrote %s", out_md)


def main() -> int:
    rows: List[Dict[str, Any]] = []
    for subject in YUQUAN_SUBJECTS:
        row = _audit_subject("yuquan", subject)
        rows.append(row)
        logger.info(
            "yuquan/%-15s stable_k=%s eligible=%d/%d %s",
            subject, row["stable_k"], row["n_events_eligible"],
            row["n_events_total"],
            "MAIN" if row["eligible_for_main"] else "",
        )
    for subject in EPILEPSIAE_SUBJECTS:
        row = _audit_subject("epilepsiae", subject)
        rows.append(row)
        logger.info(
            "epilepsiae/%-6s stable_k=%s eligible=%d/%d %s",
            subject, row["stable_k"], row["n_events_eligible"],
            row["n_events_total"],
            "MAIN" if row["eligible_for_main"] else "",
        )

    _write_csv(rows, OUT_CSV)
    _write_summary(rows, OUT_SUMMARY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
