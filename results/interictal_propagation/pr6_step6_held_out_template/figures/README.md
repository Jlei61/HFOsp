# PR-6 Step 6 — Held-out time template stability 图说明

> Plan: `docs/archive/topic1/pr6_template_anchoring/pr6_step6_held_out_template_plan_2026-05-10.md`
> Cohort: n=35 stable_k=2 subject（含 PR-6 主 cohort 与 rank_displacement n=35 sensitivity 的并集，去重）
> 严格 train/test：first half 定 cluster + endpoint，second half 只投射验证。Endpoint 集合不在 second half 上重新发明。

### tier_distribution_bar

四 tier (strong/moderate/weak/fail) 的 cohort 计数 bar。tier 规则见 plan §7.1：strong = template_spearman > 0.7 AND endpoint_position_recall > 0.6 AND swap_class_concordant；其他按达标条数下降。

**关注点**：fail 计数应为 0、weak 应远少于 strong + moderate。当前 cohort 落在 strong 20 / moderate 13 / weak 2 / fail 0，端点几何在记录窗口内对 PR-2 + PR-6 same-events double-dipping 稳健。

### template_spearman_recall_box

两 panel 对照 box plot：左为 train vs test 模板的 Spearman ρ（first-half template 与 second-half projected template 之间），右为 endpoint position recall（direction-preserving，top→top + bottom→bottom）。两 panel 都按 §8 dual-tier swap_class（strict / candidate / none）分组着色。dashed line 是 strong-tier 阈值（Spearman 0.7、recall 0.6），dotted 是 recall 的 random baseline 0.30。

**关注点**：strict / candidate / none 三组的中位数差异是否窄；如果三组明显重叠且都高于阈值线，说明 hold-out 稳健性不依赖 §8 swap_class 标签。

### endpoint_position_recall_scatter

每个 subject 一点，x = template Spearman、y = endpoint position recall。点形按 dataset 区分（圆 = epilepsiae，方 = yuquan）；填充色按 tier；外圈描边按 §8 swap_class（rust = strict、dust = candidate、白 = none）。strong-tier zone（绿底浅 alpha）= 右上 [0.7, 1.0] × [0.6, 1.10]。

**关注点**：(a) 点是否大量落在 strong zone；(b) 散布是否沿对角线（两个量同向）还是十字分布（两个量独立失败）；(c) 边缘 case (litengsheng / zhaochenxi 等 weak tier subject) 的几何特征。

### swap_class_transitions

3×3 transition matrix（行 = first-half train 的 swap_class，列 = second-half projected 的 swap_class）。对角线 = 一致；off-diagonal = 跨 tier 漂移。底注给出 concordant 比例。

**关注点**：(a) 对角线占比是否 > 50%（plan §7.2 cohort acceptance 阈值）；(b) strict ↔ none 这种"远跨度"漂移的频次（如果显著则说明 swap_class label 在 hold-out 下不稳）；(c) 当前 cohort concordant = 24/35 = 68.6%，超过阈值。
