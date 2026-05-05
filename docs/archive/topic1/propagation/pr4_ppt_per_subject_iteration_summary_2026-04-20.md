# PR-4 PPT per-subject 综合图迭代总结（2026-04-20）

> 对应脚本：`scripts/plot_topic1_pr4_ppt.py`  
> 对应输出：`results/interictal_propagation/figures/ppt/per_subject/{ds}_{sub}.png`

---

## 1. 这轮对话到底改了什么

用户目标很明确：不是改科学结论，而是把 per-subject 综合图做成可直接上 PPT 的最终版，减少重复信息并提升可读性。

最终固定的版式合同：

- 上半：`a`（rate envelope）+ `b`（stacked count）+ 共享 day/night strip
- 下半：`c` raw rank heatmap、`d` per-channel ranks、`e` clustered heatmap、`f` cluster rank curves
- 不再给 `b` 独立 panel 字母；`a/c/d/e/f` 用统一 panel letter 对齐
- 全局字体上调到 `20/20/24/26/30`（tick/label/title/suptitle/panel-letter）
- `f` 移除 legend（避免与 `c/e` 重复）
- `f` 的 rank 轴按数据收紧，去除明显白边
- rank colorbar 从“e 下方”移动到“c 与 e 之间”，且长度保持旧版比例（左列宽度的 32%）

---

## 2. 从图里得到的关键观察（描述层）

### 2.1 rate 调制与 seizure-onset cluster 的关系

- 以 `enrich`（高 rate bin 内 seizure 富集）与 `ρ_dom`（dominant fraction vs 距离 seizure）做联合排序后，可见 cohort 内存在一个明确子群：
  - `enrich ≥ 1.5`：`15/25`
  - strict (`enrich ≥ 1.5` 且 `|ρ_dom| ≥ 0.15`)：`9/25`
- 这说明“rate burst 与 seizure onset 的群集共现”是真现象，但不是全体规律。

### 2.2 绝对率 vs 相对占比必须分开

- **绝对率**（dominant template events/h）在 post-ictal 升高：与 PR-4C/PR-5 主线一致。
- **相对占比**（dominant share）在 PR-5 合同下并不稳定复制：不能作为主结论，只能保留为描述层线索。

---

## 3. 记录下来的“有趣患者”

### 3.1 值得优先做 case-series 的 strict 组

- `yuquan:litengsheng`（ρ 高）
- `epilepsiae:139`
- `epilepsiae:1096`
- `epilepsiae:1125`
- `epilepsiae:916`
- `yuquan:sunyuanxin`

### 3.2 对照型/反例型患者

- `epilepsiae:818`、`epilepsiae:590`、`epilepsiae:253`：`enrich` 高但 `ρ_dom` 接近 0（loose）
- `epilepsiae:548`：视觉上容易误判“总在 burst 上发作”，但 `enrich≈1`，属于高 seizure 密度造成的图像错觉

---

## 4. 目前指标的短板与新 metric 需求

`enrich + ρ_dom` 能筛查，但不能区分“真实簇化”与“高密度均匀叠加”。下一步应定义新指标：

1. **SBCI（Seizure-onset Burst Concentration Index）**
   - 衡量 seizure onset 是否集中在少数 burst
   - 建议在 burst-level seizure count 上用 concentration 指标（Gini / HHI / 归一化熵）
2. **TRIS（Template Recruitment Imbalance Slope）**
   - 衡量 dominant 与 non-dominant 绝对招募差值随 `|Δt_to_seizure|` 的斜率
   - 直接回答“是全体一起升，还是 non-dominant 被额外招募”

建议先在 PR-5 后续扩展中做 metric 合同（阈值、失败合同、敏感性分析）再进主文档，避免再次出现“图像读感先行、统计定义滞后”。
