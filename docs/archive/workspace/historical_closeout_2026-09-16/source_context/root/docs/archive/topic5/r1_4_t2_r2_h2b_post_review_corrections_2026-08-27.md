# R1.4 / T2-R2.0 / H2b 阶段报告 代码复审更正记录（2026-08-27）

更正对象：`results/epi_prssm/continuous_marked_state/r1/final_reports/r1_4_t2_r2_h2b_{plain,technical}_2026-08-27.md`
（提交 `58ba340e`，分支 `codex/topic5-continuous-state-r1-4`）。
凡与本文件冲突处以本文件为准。两份报告由 `finalize_r1_4_t2_r2_reports.py` 生成，
更正已改在生成器里并重新出报，因此报告本身已经带上更正。

## 0. 一句话

五条科学结论**全部成立**，承重数字我逐项核对过（H2b 我从逐发作产生表独立重算，
完全复现）。但有一处把**本轮最有信息量的阴性写成了空白**：E958 的累积暴露边
3/3 seed 停在零边，而同一批行上的对照臂拟合出了边并胜过无边——这是支持度最好的
患者给出的一个 H3a 阴性回答，却被记成「不可估计」并整行打 `n/a`。

---

## 1. P1：把「搜索找不到边」和「测不了」写成了同一件事

### 测了什么

H3a 问：把最近一百次 IED 里「预测不到的那部分」累加起来，能不能在当前状态之外
再解释下一次 IED。做法是只训练一个事件后的状态跳变 B，其余全部冻结。

### 揭示了什么

`real_edge_estimable` 的定义里混进了一项结果：

```
estimable = gradient 有限 且 gradient≠0 且 exposure 满秩 且 spread 非退化
            且 edge_left_zero_initialisation        ← 这一项是结果，不是可识别性
```

最后一项等于「拟合出来的边在 TRAIN 内验证上胜过了零边」。把它并进「可估计」，
再用「可估计」去筛数值分母（提交 `58ba340e`），等于**按结果筛样本**：找到边的 seed 留下，
没找到的 seed 被记成缺测量。

实测被筛掉的正是零结果：E958 load 3/3、participation 3/3，E620 participation 2/3、load 1/3。
而这些 seed 的可识别性审计全都是健康的：`gradient_at_zero_norm` 0.13–0.88、
exposure 满秩、`exposure_sd` 12–21、TRAIN 事件对 13,437 / 80,723。

更关键的是**同排证据**。E958 / load 三个 seed 在完全相同的行上：

| 臂 | 选中 epoch | validation joint NLL 相对无边 |
|---|---:|---:|
| 累积暴露 real | 0 / 0 / 0 | 0.000000（三个 seed） |
| current-event 对照 | 10 / 12 / 26 | +0.00102 / **−0.00146** / **−0.00156** |

也就是说：这套机器在这批行上**确实能**拟合出一条边，只是累积暴露那条边在任何被检查
的 epoch 上都没有带来 held-out 收益。这是 H3a 想要的那种阴性，不是一格空白。

### 已改

- `real_edge_status` 三分：`FITTED` / `ZERO_EDGE_SELECTED` / `NOT_IDENTIFIABLE`。
  只有后者与 support-ineligible 才是缺测量。
- 数值中位**仍然**只取 FITTED seed——真实边为零时 `real − no_edge` 恒为零，
  `real − placebo` 只反映 placebo 更差（这正是 `c651a957` 修掉的假持续标签），
  两者都不该进中位。这一条原本就是对的，保留。
- 但零边 seed 不再整行打 `n/a`：新增 `zero_edge_selected_seeds`、
  `sibling_current_edge_fitted_seeds` 和一句 `zero_edge_evidence`，
  并在两份报告的表里加一列「同排对照臂拟合出边的 seed」。
- 同排证据由存档的 validation 数值**推算**，不依赖新写的标志位，
  所以修复前产出的 12 个 per-seed 结果也能正确归类。

### 更正后的读法

| 患者 / source | 拟合出边 | 零边 | 无支持 | 同排对照臂拟合出边 | 结论 |
|---|---:|---:|---:|---:|---|
| E620 / load | 2/3 | 1/3 | 0/3 | 1/3 | 有边但未过增量门 |
| E620 / participation | 1/3 | 2/3 | 0/3 | 3/3 | 多数 seed 找不到边 |
| **E958 / load** | 0/3 | **3/3** | 0/3 | **2/3** | **普通阴性**：设计能承载边，累积暴露没有 |
| E958 / participation | 0/3 | 3/3 | 0/3 | 0/3 | 零边，且无同排证据 |
| 黄瀚文 / 两源 | 0/3 | 0/3 | **3/3** | 0/3 | 真正的缺测量（没有一步事件对） |

结论方向不变（0 个组合获准扩尺度），但 H3a 现在有了**证据**而不只是「没测到」。

---

## 2. P1：10-donor 敏感性与主分析用了不同精度的观测嵌入

主 R1.4 拟合用 `materialize_embedding(..., use_amp=True)`（CUDA 上 fp16 autocast）；
10-donor 敏感性走 `load_fitted_explicit_t1`，那里写死 `use_amp=False`（fp32）。
checkpoint 相同，但**重新算出来的观测嵌入不同**，因此 `correct` 臂的基线也不同。
于是 5→10 donor 的对比同时变了 donor 数和嵌入精度，而报告承重的
「三位稳定患者 9/9 seed 正确时刻有利」效应量在 1e-4–1e-2，与 fp16 噪声同量级。

**已改**：主分析把精度写进结果 JSON；`load_fitted_explicit_t1` 与 R1.4 loader 接受
`use_amp` 并默认继承主分析的取值；产物记录 `embedding_precision_matches_primary`。

## 3. P1：陈旧的 10-donor 敏感性会被静默接受

`aggregate_r1_4.py` 只检查文件存在、`same_checkpoint_as_primary_5_donor is True`
和 `sealed_opened is False`。那个标志是产出时自己写的，**无法察觉主分析后来重拟合过**；
而流水线对任何已 COMPLETE 的文件都跳过重算，所以一次 OOM 重试或续跑之后，
旧的敏感性会作为新 checkpoint 的敏感性被报出来。

**已改**：新增 `status == COMPLETE` 检查、`source_checkpoint_sha256` 必须命中本次汇总的
主 checkpoint、以及嵌入精度一致性检查，任一不满足即报错而不是静默采用。

## 4. P1：报告生成器里写死了分母

`finalize_r1_4_t2_r2_reports.py` 的「34 位患者进入，27 位有可分析发作」是字面量，
而同一行后半段的数字是从 `h2b_lead30` 插值出来的。队列一变，同一句话的前后半段就会
自相矛盾，而脚本不会报错。**已改**为从 `step_1_cohort_patients` /
`step_2_patients_with_any_analysable_seizure` 取值。

## 5. P2：几处已记录但未改动数值的问题

- **暴露尺度不匹配**：`real_cumulative` 的 TRAIN 标准差是 12–21，`current_event_only` 是 1.0
  （逐事件 innovation 已按残差标准差归一，但指数累加器又把尺度乘了回去）。两臂共用
  同一套 AdamW 预算 / 学习率 / 权重衰减 / 梯度裁剪。我原以为这会系统性偏袒某一臂，
  **实测不成立**：AdamW 基本是尺度无关的，同一真值在 1× 和 18× 两个尺度下
  尺度校正后的边范数只差约 1%。真正会变的是**搜索停在第几个 epoch**（8 vs 12），
  而 epoch 正是两臂应当共享的东西。已加 `standardise_exposure`（纯重参数化，`B·x` 不变）
  把所有拟合臂放到同一 TRAIN 尺度，并加了锁住这一点的单元测试。
  现有 12 个人体产物是标准化之前跑的。
- **合成校准用的不是人体那套 placebo**：合成里 placebo 是 `np.roll(real, 1500)`，
  人体用的是 83 维最近邻、TRAIN-only donor 池、5N 不重叠规则的
  `state_matched_nonoverlap_placebo`。所以合成校准的是边拟合与符号恢复，
  **不是**人体的配对器。已写进合成产物的 `placebo_construction` 字段。
- **零真值缺一条判据**：合成检查了 `zero_does_not_beat_placebo`，但没有
  `zero_does_not_beat_current`，而人体 primary 要求同时胜过两者。
- **三列并非划分**：可估计 / 结构零 / 支持不足三列原本不保证加起来等于 3。
  已加 `unclassified_seeds` 并在表头写明三种结局的读法。
- **魔法字符串**：记录覆盖段 donor 约束由自由文本 `--experiment-label` 精确匹配触发。
  已加 `choices` 并改用导入的 `R1_4_REVISION` 常量。
- **韩宇轩的 raw「2/3 同向」中位是 −6.1e-7**，比 E620 的 +0.0023 小七个数量级。
  白话报告已经写明了这一点；技术报告的「raw joint 有利 1/6」应连同量级一起读。

## 6. 逐项核对通过的

| 报告说法 | 核对结果 |
|---|---|
| R1.4 六人表（六列） | 与 `r1_4_summary.json` 逐值一致 |
| 稳定 T1 = ≥2/3 seed（epoch>0 且 persistent<memless 且 correct<wrong） | 代码与合同一致 |
| 黄瀚文 validation 107 次（620=6,314、958=30,884） | 一致 |
| 10-donor 稳定三人 9/9 | 一致（但见 §2） |
| H2b +0.454373 / 21/27 / p=0.0059246123 / 339 of 361 | **我从产生表独立重算，完全复现** |
| 高可观测层 +0.266655 / 17/27 / p=0.2477886 / 201 of 203 | 同上 |
| 6 个假持续标签 | 一致（E620 participation 4、E958 participation 2） |
| `scale_expansion_candidates=[]` | 一致 |
| 313 tests / rc=0 | 一致；本工作树（多两个提交）为 323，加本轮 6 项后 **329** |
| 机器审计 9 输入 + 2 输出哈希 | 全部验证通过 |
| 所有臂共享完全相同的事件支持 | 代码对 next-event 与 H5/H10 各自断言 |
| 交叉拟合从不使用 validation 结局 | 折是按时间连续块切的，validation 由全 TRAIN 模型预测 |
| H5/H10 只注入一次 jump，之后不读新观测、不再跳变 | 代码一致 |

## 7. 与 H3a 结论的关系

修完之后 H3a 的结论**不变但变强**：不是「N=100 上没测出来」，而是
「在支持度最好的患者上，累积暴露边在任何被检查的 epoch 上都没有 held-out 收益，
而同一批行上的当前事件边有」。这句话可以直接写进论文口径。
