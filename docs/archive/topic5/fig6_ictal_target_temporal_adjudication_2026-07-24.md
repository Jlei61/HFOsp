# Figure 6 发作侧 target 的时间结构裁决

**日期**：2026-07-24
**状态**：完成；用于撤回 v0.2 的逐秒 ictal rollout，并冻结 v0.3 静态 readout
**问题**：现有数据是否支持把 clinical onset 后 0–10 s 拆成逐秒接触点传播序列？

## 1. 一句话结论

不支持。现有结果支持的是：

> 患者特异的间期传播 scaffold 与 peri-onset/0–10 s 静态能量空间场相关；
> 该空间关系在 0–10 s 内总体稳定，并且 clinical onset 前已经存在。

因此 0–10 s 应继续作为与既有主结果一致的**固定静态测量窗**，而不是被重新解释
为九步传播 target。RNN 的 recurrence 应只放在单个间期事件内部；发作侧改为
冻结 operator 到静态 susceptibility/rank field 的跨域 readout。

## 2. 已接受窗口扫描已经否定 ictal-specific emergence

来源：

- `docs/archive/topic5/axis_alignment_hardening_result_2026-06-15.md`
- `results/topic5_ictal_recruitment/axis_alignment/window/window_summary.md`

18 名 Epilepsiae 患者的既有结果：

| 窗口/比较 | 结果 |
|---|---:|
| broadband 0–5 s real median alignment | 0.561 |
| broadband 5–10 s real median alignment | 0.547 |
| broadband 0–10 s real median alignment | 0.554 |
| broadband distal −120 至 −90 s | 0.543 |
| patient-level post 0–10 minus distal median | −0.0256 |
| one-sided Wilcoxon，post > distal | P = 0.367 |

0–5 和 5–10 s 的读出相近；远端 preclinical window 也没有下降。原分析已将
verdict 冻结为 `persistent_scaffold_NOT_ictal_specific`。

## 3. strict-BB150 母队列上的直接逐秒场检查

母清单取自：

`results/topic5_state_conditioned_predictor/fit12_clinical_bb150/fit2/fig6_fit2_clinical_onset_scaffold_event.csv`

固定：

- `group_id == strict_broadband`；
- 13 名患者、71 次发作；
- 每次发作使用 frozen interictal-field `contact_order` 与 ictal cache 的 exact
  channel-name join；
- 中位触点数 11；
- 时序代理为现有 `t0_feature_cache_v2_windows` 中 1–45 Hz、0.1 s hop 的
  baseline-robust-z trace。

每次发作先算 contact-level Spearman，再按患者取 seizure median。患者级结果：

| 场稳定性指标 | n | median | IQR |
|---|---:|---:|---:|
| 每个 1 s bin vs 0–10 s 聚合场的中位相关 | 13 | **0.815** | [0.682, 0.929] |
| 相邻 1 s bins 的中位相关 | 13 | **0.750** | [0.691, 0.964] |
| 0–1 s vs 1–10 s 聚合场 | 13 | **0.733** | [0.464, 0.893] |
| 0–5 s vs 5–10 s | 13 | **0.762** | [0.464, 0.958] |
| −10–0 s vs 0–10 s | 13 | **0.643** | [0.527, 0.875] |

这不是说所有患者、所有秒都完全相同；例如 E1077、E1084、E384、E958 的最差
单秒更不稳定。但 cohort 的主要结构是重复的患者静态场，而不是一致的逐触点
传播顺序。

该检查使用 1–45 Hz 全时序 cache，只用于判断“是否存在稳定逐秒 target”的构造
有效性；它不替代 strict BB1–150 主统计。

## 4. 现有 1–150 Hz peri-onset 轨迹的独立支持

来源：

`results/topic5_ictal_recruitment/field_dynamics_signed/*_signed_broadband_1_150Hz_similarity_timecourse_m120_p20_10s_step2s_per_seizure.csv`

对 17 名有该轨迹的 Epilepsiae 患者、318 次发作，先在每名患者内跨 seizure
取每个窗口的 `maxAB_abs_corr` 中位数，再检查 −10 至 +10 s 的 10 s
sliding windows：

| 指标 | 患者级 median |
|---|---:|
| peri-onset maxAB alignment | **0.784** |
| peri-onset window-to-window SD | **0.056** |
| onset windows minus −20–0 s windows | **+0.016** |

这些轨迹是 10 s window / 2 s step 的 field-alignment diagnostic，不是逐秒
contact-field identity，也不是 strict-BB150 aggregate producer 的替代物。但它独立显示：
clinical onset 附近没有一致的大幅空间场跃迁，很多患者在 onset 前已经具有同一
scaffold readout。

## 5. strict BB150 新逐秒 cache 当前没有可接受 parity

既有 accepted `t0_feature_cache_bb150_1_150` 只保存 `[0,10] s` 聚合
`bb150_auc`，没有保存 BB150 全时序 trace。

使用当前 checkout 的 producer 逻辑对 E1084 seizure 0 做一次只读重提取时，
新聚合值未精确复现 accepted artifact：

- `max_abs_difference = 0.3865`
- Pearson `r = 0.8249`

该 sentinel 只说明当前重提取链与 2026-07-06 artifact 之间存在尚未裁决的
producer/extraction drift；它不推翻 Fit1 对 accepted artifact 的精确 parity。
在 root cause 和 fingerprint 未锁定前，不能把新生成的 BB150 逐秒 trace 冻结为
模型主 target。

## 6. 科学裁决

### 保留

- clinical onset；
- `[0,10] s`；
- strict BB 1–150 Hz；
- TA/TB `maxAB`；
- all-contact channel shuffle；
- seizure → subject → cohort fold。

### 撤回

- 0–1 s seizure seed；
- 1–10 s 九步 closed-loop rollout；
- contact × time rank 作为主 target；
- “发作早期沿间期顺序逐点重放”的解释；
- event pseudo-time 与 1 s physical time 共用同一 transition 的主假设。

### 新解释

`[0,10] s` 只是一个已验证、可比较的标准化静态读出窗。它不表示该空间模式在
clinical onset 才出现，也不表示窗内存在可监督的传播顺序。

## 7. 对新模型的直接要求

1. RNN 只从单个间期事件的 rank prefix 学 suffix/STOP/participation。
2. 冻结 RNN 后，从其内生的两个传播 mode 生成两张无序的患者特异静态
   susceptibility/rank fields。
3. 不输入 seizure seed，不对发作时间做 recurrence。
4. 用 accepted 0–10 s BB150 energy field 做 zero-shot cross-domain readout。
5. 主统计继续对两张 mode fields 做 A/B/mirror `maxAB`，每次 all-contact
   shuffle 重新承担相同选择成本。
6. −10–0、0–5、5–10 和 distal windows 只作冻结后的时间敏感性，预期回答
   persistent scaffold，而不是寻找“最显著窗口”。
