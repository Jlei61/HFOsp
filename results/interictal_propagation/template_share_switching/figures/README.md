# 模板占比 + 模板切换 in pre/post-seizure windows

> 队列：stable_k=2，PR-1 n40 cohort，去掉 8 个 yuquan 无发作 subject + 1 个 epi label 长度不匹配 subject。
> Share 队列 n=27（含原 PR-5-B 1096 patch），Transition 队列 n=26（剔除 huanghanwen 的 n_pairs<10 离群值）。
> 窗口定义：baseline = [−4h, −1h]，pre = [−1h, −0.25h]，post = [+0.25h, +1h]（PR-4C/PR-5 main config）。
> 数据源：`pr5b_recruitment_shift_extended.json`、`pr5b_recruitment_shift.json`（1096 patch）、`pr5_transition_windows.json`。
> 高亮 9 个 "rate-burst seizure-enrich" subject 来自 PR-4D `_score_rate_cluster_seizure` strict-match (enrich ≥ 1.5 AND |ρ(dom_frac, |Δt_sz|)| ≥ 0.15)：1125, 1096, 916, 635, 442, 1150, 139, sunyuanxin, litengsheng。

---

## fig_a_template_share.{png,pdf}

**Q1 — 两类模板的占比在发作前后是否变化？**

- (a) per-subject dominant template share，按 baseline / pre / post 三个窗口连线，每条线对应一个 subject。深色加粗 + 黑边圆点 = 9 个 PR-4D strict-match subject（rate burst 富集 sz 且 dom_frac 与 |Δt_sz| 显著相关）。淡色细线 = 其余 18 个 subject。Epi=赤陶色，Yuq=灰蓝色。0.5 灰虚线 = 双模板均衡参考。
- (b) cohort paired differences，n=27 paired Wilcoxon + sign-test。三组 Δ：pre−baseline、post−baseline、post−pre。

**关注点**：
- post−baseline median **+0.021**，Wilcoxon p=**0.005**\*\*，sign 20+/7−（p=0.019）→ **dominant 模板招募率在 post-ictal 整体抬升**
- post−pre median **+0.045**，Wilcoxon p=**0.004**\*\*，sign 20+/7−（p=0.019）→ **dominant 抬升相对 pre 更明显**，不是简单的"发作邻近"对称效应
- pre−baseline NULL（p=0.27）→ pre-ictal 本身 dominant 份额不变；变化集中在 post-ictal
- (a) 视觉上多数 strict-match subject post 高于 baseline，但也有 litengsheng（深蓝下降）这类反向 case → cohort 不是完全单向
- 与 PR-5-B archive 原 retained cohort (n=23) 结论方向一致；扩展到 stable_k=2 n=27 后仍稳定

## fig_b_template_switching.{png,pdf}

**Q2 — 两类模板的切换在发作前后是否变化？**

- (a) per-subject next-event transition lift，按 baseline / pre / post 三窗口连线。lift = 实际 P(next-event opposite-template) odds ÷ i.i.d. baseline odds；lift = 1 表示与独立抽样一致（无切换偏好），> 1 表示比独立更倾向切换，< 1 表示倾向同模板成簇。颜色规则同 Fig A。1.0 灰虚线 = 独立抽样基线。
- (b) cohort paired differences，同样三组 Δ + Wilcoxon。

**关注点**：
- 三组 Δ cohort 全部 NULL：pre−base p=0.44、post−base p=0.29、post−pre p=0.35
- median Δ 都接近 0（+0.007 ~ +0.024）；方向 sign 全部 15-16+/10-11−，无显著偏向
- **结论**：模板切换率在发作前后不变 → 与 PR-7 cohort NULL 一致，并且这次是直接在 baseline/pre/post 窗口下测，PR-7 §17 "未测 form (3) seizure-proximity switching" 这一栏现在可以加上 "已测，cohort 仍 NULL"
- 个体层面也极少有人显著偏离 lift=1，最低 ~0.72（922、548、sunyuanxin），最高 ~1.27（384、635），但跨三窗口稳定，**不是发作邻近的调制**

---

## 双图合在一起讲什么

间期模板的 **谁被招募**（share）在 post-ictal 整体抬升（dominant 模板更多被复用），但 **怎么排列**（next-event 切换概率）在发作前后不变。

= "post-ictal 把已有的 dominant 模板用得更多，但模板之间的接力顺序不重排"。

这跟 Topic 1 主结论一致：**间期刻板时序的几何在发作邻近不变形、不替换、不重组；变的只是被招募的频率，且集中在 post-ictal 时段**。

---

## 数据 / 代码

- 数据：
  - `results/interictal_propagation/pr5b_recruitment_shift_extended.json`（新跑，stable_k=2 n=35 中 27 个 OK）
  - `results/interictal_propagation/pr5b_recruitment_shift_extended.csv`（per-subject 表）
  - `results/interictal_propagation/pr5_transition_windows.json`（新跑，stable_k=2 n=35 中 26 个 OK）
  - `results/interictal_propagation/pr5_transition_windows.csv`
- 代码：
  - `scripts/run_pr5b_share_extended.py`（PR-5-B 在扩展 cohort 上重跑，绕过 PR-5-A gate；仅描述层，不替代 archive 的 retained cohort 主统计）
  - `scripts/run_pr5_transition_windows.py`（pre/post 窗口 transition_odds，pool n_pairs across seizures per state）
  - `scripts/plot_template_share_switching.py`（绘图）

## 范围声明

- 这是 **描述层** 输出，复用 PR-5-B 已验收的 dom_global_share 合同，并把 PR-7 transition_odds 第一次按 baseline/pre/post 窗口切片。
- 不替代 archive `pr5_template_recruitment_plan_2026-04-20.md` §4 的 PR-5-B retained cohort (n=23) 主统计 — 那条仍是 §4.5 主结论的单一来源。
- 不能升级为新的 cohort claim：PR-5-A novel-template gate 没有在 n=27 上重跑，绕过 gate 只用于描述性图；要升级到主结论必须先在扩展 cohort 上跑 gate。
