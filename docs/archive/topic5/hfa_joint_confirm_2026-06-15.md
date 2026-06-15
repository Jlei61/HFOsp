# Topic 5 A 线 · hfa×joint 冻结复验（split-half 稳健性 + 负对照）

> 日期：2026-06-15 · 上游：`docs/archive/topic5/axis_alignment_AB_result_2026-06-14.md`（A+B 主线）
> 代码：`scripts/run_topic5_hfa_joint_confirm.py`（复用 `_subject` / `_cohort_stats`）
> 结果：`results/topic5_ictal_recruitment/axis_alignment/hfa_joint_confirm.{json,log}`

---

## 0. 白话摘要（§8 三段式）

**测了什么。** A+B 主线里只有一个发现过了最严的随机对照——"快活动（60–100 Hz）"这个激活量、在
"同杆×同活跃度联合洗牌"这一层（最严）上，间期传播轴与发作激活仍然对得上。因为这个量是灵敏度档、不是
预注册的主指标，**不能拿同一遍探索就升成主结论**。所以我们把它单独拎出来、锁死参数复验一遍，并按
"奇数次发作 / 偶数次发作"把每个被试的发作分成两半各算一遍，再加一个"把对齐打乱、本该失败"的负对照。

**怎么测的。** 四臂，都看最严那层（joint）的队列 Wilcoxon p：full（全部发作）/ even（偶数次）/
odd（奇数次）/ negative control（把发作激活在通道间打乱）。如果这个细对齐是稳的，前三臂都该 <0.05、
负对照该 ≥0.05。

**揭示了什么。** **在全数据上够得着，但经不起分半。** full 干净复现了主表（joint Wilcoxon=0.022），
偶数半也显著（0.035，但留一临界 0.056），**奇数半就达不到显著（0.078）**——而且奇数半连"同杆"层都掉了
（0.209）。负对照把全部四层都打回非显著（joint=0.95，粗层 0/18），说明这检验不是"假阳性机器"、全数据上的
信号是真的。结论：**这条"细到活跃度无关的精细轴"是真的、但不够稳，不能升主结论；要升格必须有独立第二
队列，不能在同一份数据上挑。** 它继续留在灵敏度档。

（内部归档代号：metric=hfa，null=joint=within_shaft×anchor_bin，split_half_robust=False，B=2000。）

---

## 1. 设计（锁死）

- **被测对象**：仅 `hfa`（60–100 Hz）激活量 × `joint`（同杆×活跃度箱联合洗牌）null —— A+B 探索里
  唯一过 joint 的 metric×null 格（FINAL 表 hfa joint B2000 Wilcox=0.0221）。
- **四臂**（都跑全部四层 null，重点看 joint）：
  - `full` —— 全部合格发作，锁参（B=2000，seed=20260615）干净重跑。
  - `split_half_even` / `split_half_odd` —— 每被试按 eligible-seizure **位置**奇偶分两半
    （`src.topic5_axis_alignment.seizure_parity_subsets`，TDD），各自 per-subject 中位→队列。
  - `negative_control` —— 打分前把发作激活在通道间随机洗牌（`channel_shuffle`），对齐被破坏，
    队列应**回到非显著**（坏数据必 FAIL 门）。
- **稳健判据**：`split_half_robust = (full<0.05) ∧ (even<0.05) ∧ (odd<0.05) ∧ (neg≥0.05)`。
- **范围声明**：同一 18-被试 Epilepsiae 队列 —— 这是**稳健性**检查，**不是留出/独立验证**；
  升 primary 需要**第二队列**（当前不可得）。

## 2. 结果（joint 层，队列 n=18，B=2000）

| 臂 | joint n_pass | binom p | **joint Wilcoxon p** | LOSO-worst |
|---|---|---|---|---|
| full（锁参重跑） | 3/18 | 0.058 | **0.0221** ✓ | 0.0394 |
| split-half even | 3/18 | 0.058 | **0.0352** ✓ | 0.0559（临界） |
| split-half odd | 1/18 | 0.603 | **0.0778** ✗ | 0.1228 |
| negative control | 1/18 | 0.603 | **0.9506** ✓(应非显著) | 0.9847 |

奇数半的弱不是只在 joint：odd 臂 within-shaft Wilcox=0.209（fail）、coarse 0.019 / activity 0.010
（仍过），即**奇数半整体比偶数半弱，最严的 joint 层首当其冲**。负对照四层全部非显著
（coarse 0.248 / within 0.929 / activity 0.858 / joint 0.951；coarse n_pass 0/18）。

**verdict：`split_half_robust = False`。**

## 3. 解读与口径（写论文/主文档照搬）

- **允许**："hfa×joint 的精细、活跃度无关轴在全数据上可检出（Wilcoxon p=0.022，负对照干净），
  但**不**经按发作奇偶分半（奇数半 p=0.078）——因此维持灵敏度档，不升 primary。"
- **允许**："负对照把全部四层 null 打回非显著，说明 A 线检验不是假阳性机器，主线（粗骨架）结论不受影响。"
- **禁止**：把 hfa×joint 写成 cohort primary claim；把"full 显著"单独拿出来当稳健证据
  （split-half 已证明它不稳）。
- **不改变主线**：A+B 主线结论（**粗共享网络轴稳**）依旧成立——本复验只针对"最严层的精细轴"，
  且把它**降温**（real-but-not-robust），与"primary 只有 broadband×channel"的预注册一致。

## 4. 下一步

- 若将来拿到**独立第二队列**（另一中心 SEEG/ECoG，或 Yuquan 发作覆盖补齐），再预注册
  hfa×joint 做真正的留出确认；在此之前不动它的档位。
- 复验脚本已参数化（`_subject` 加 `sz_subset` / `negative_control`），第二队列直接复用。
