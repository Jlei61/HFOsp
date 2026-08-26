# Epi-PRSSM H2B / H3 修订 — 技术版

**日期：** 2026-08-20 · **分区：** development only；formal test partition 未打开
**图形后端：** Python / matplotlib（`nature_figure_backend.py set python`）
**样式合同来源：** `nature-figure` skill（core/contract.md + core/stance.md）→ `docs/figure_style_guide.md` §0 → `docs/superpowers/specs/2026-08-18-topic5-epi-prssm-figure-contract.md` → `docs/topic5_seizure_subtyping.md`

## 1. Seizure crosswalk（任务 A）

**脚本：** `scripts/topic5_epi_prssm/build_seizure_crosswalk.py`
**输出：** `results/epi_prssm/v0_1/seizure_crosswalk/{crosswalk__<layer>__<lead>.csv, CROSSWALK_SUMMARY__*.json}`

| 路径 | n | 说明 |
|---|---|---|
| `seizure_id` 直连 | 325 | Epilepsiae 两侧编号一致 |
| `record_code+index` | 34 | Yuquan：`FA0013KQ_0` → 记录码 `FA0013KQ` + 记录内第 0 次（按 onset 排序） |
| `unmatched` | 2 | `chenziyang`、`zhangbichen` 不在 `results/dataset_inventory/yuquan_seizure_inventory.csv` |

- 时间戳审计：`onset_difference_seconds` 中位 / p95 / max **全部 0.0**，容差 5.0 s
- 一对一：`n_ambiguous = 0`；`n_id_did_not_parse = 0`
- 亚型接入：`broad_ER` 192 / `gamma_ER` 180（来自 `results/data_driven_soz/layer_a_ictal_er_rank/seizure_clusters/per_subject/*__zer_binned.json` 的 `seizure_ids_kept` × `subtype_label` × `outlier_flag`）
- **禁止的连接方式**：字符串直连（丢全部 Yuquan）、数组位置（两表顺序与长度均不同）

## 2. Denominators（任务 A）

**脚本：** `build_h2b_denominators.py` → `results/epi_prssm/v0_1/h2b_denominators/`

| lead | cohort | 有可分析发作的患者 | eligible | premise-met | premise-met 患者 | 比例 |
|---|---|---|---|---|---|---|
| 5m | 34 | 27 | 363 | 154 | 25 | 0.424 |
| 15m | 34 | 27 | 363 | 182 | 27 | 0.501 |
| **30m** | 34 | 27 | **361** | **203** | 27 | 0.562 |
| 60m | 34 | 27 | 360 | 220 | 27 | 0.611 |

Strata（30m）：`none` 82 / `1to4` 24 / `5to19` 38 / `ge20` 217；premise-met 构成 `ge20` 180 + `5to19` 23，其余两层恒 0（前提要求 ≥5）。

## 3. H2B sensitivity（任务 B）

**脚本：** `run_h2b_sensitivity.py` → `results/epi_prssm/v0_1/h2b_sensitivity/`

- **端点分层**：primary `first_selection_entropy`（frozen decoder readout）；secondary `expected_load`, `resource`；**sensitivity-only** `state_norm`
- **统计单位**：patient；seizure 嵌套于 patient（先在患者内取中位）
- **预先指定的主格**：`linear_graph_recurrent × lead30m × all_eligible × open_loop_at_onset × first_selection_entropy`

| 主格 | median | 95% CI | 方向有利 | p |
|---|---|---|---|---|
| all_eligible / open_loop | **+0.446** | [0.128, 0.539] | 20/27 | **0.019** |
| all_eligible / filtered_at_onset | +0.316 | [0.146, 0.692] | 21/27 | 0.0059 |
| high_observability / open_loop | +0.288 | [−0.013, 0.469] | 18/27 | 0.122 |

- **留一**：LOPO 27 次重算 median-of-medians +0.422，范围 [+0.421, +0.447]，符号稳定；LOSO 339 次 +0.446，范围 [+0.396, +0.449]，符号稳定
- **连续可观测性**（患者内秩相关，再以患者为单位）：`n_ied_lookback` −0.074（10/22 正）、`anchor_gap` −0.058（9/22）、`coverage` −0.080（8/20）→ **无梯度**
- **亚型交互**（一个统计量，非 per-subtype 检验组）：`broad_ER` 6 位患者有两个 size≥3 的亚型，实测落差中位 0.932 vs 标签打乱零假设 0.673，3/6 超过自身零假设，sign p=1.0，Fisher 合并 p=0.531；`gamma_ER`（sensitivity）4 位患者，1/4，p=0.625
- **小亚型规则**（Topic5 §6.2 要求下游自行声明）：`size < 3` **排除出交互检验，但计数并报告**；`broad_ER` 有 7 位患者、`gamma_ER` 有 9 位患者含此类亚型
- **band 规则**：`broad_ER` 主、`gamma_ER` 敏感性，不并列解释

## 4. H3a 重构（任务 C）

**模块：** `src/topic5_epi_prssm/arrival.py`（`RenewalIntensity(markov_renewal=)` + `goodness_of_fit`）、`src/topic5_epi_prssm/rate_state.py`
**脚本：** `run_arrival_channel.py --markov-renewal`

诊断依据：log-interval lag-1 相关中位 **+0.300、33/34 为正**（与 Topic 2 独立记录的 30/30、r≈0.299 一致）。原强度形式（慢乘性项 × 固定更新形状）无法产生短程间隔相关。

| 模型 | rescaled mean | rescaled sd | 闸门 |
|---|---|---|---|
| modulated renewal | 1.006 | **0.704** | FAIL |
| **+ lag-1 条件项** | 0.975 | **0.838** | **PASS** |

`goodness_of_fit` 现同时输出 mean / sd / KS 统计量 / 残差 ACF(1,2,5) / QQ 最大偏离。GOF 不合格时 evidence card 标 `contracts_admissible: false`，所有 arm contrast 为 diagnostic-only。

**进度：** `renewal_only` 3/3 完成；`t0_exogenous_clock` / `t1_observer` / `t2_physical` 运行中（`manifests/plans/arrival_markov.json`）。

## 5. H3b 重构（任务 D）

**脚本：** `run_h3b_transition_coupling.py` → `results/epi_prssm/v0_1/h3b_transition/`

不再是 H2b × H3a 的 AND gate。定义为 patient-internal case-crossover 下的 exposure → preictal state，并检验 exposure × subtype 交互。

- exposure → state：21 位患者、327 次发作，within-patient 斜率中位 +0.237，**11/21 为正，sign p=1.0**，逐患者 outcome-shuffle 零假设中位 p=0.354
- exposure × subtype：`NO_PATIENT_WITH_TWO_USABLE_SUBTYPES`
- early recruitment leg：`NOT_RUNNABLE` —— 需要盲法裁定的 onset 触点，登记表 0/71；已锁定的盲法合同禁止用临床致痫区、患者级 focus、模板端点或最高能量触点顶替
- mediation：`NOT_RUN`（同一批发作上测量、无干预，系数只描述协方差）

## 6. 已失效结果与其归档路径

| 作废对象 | 原因 | 归档 |
|---|---|---|
| 到达通道第一批（18 runs） | `optimiser.step()` 在患者循环外，整个拟合 25 步；时间常数逐位等于初始化 | `_invalidated_one_step_per_epoch/` |
| 到达通道第二批（7 runs） | 生存补偿项使用区间**终点**状态，依赖被估计的到达时间 | `_invalidated_endpoint_compensator/` |
| 图零假设通路拆分 | 跨 5 个 package hash 比较；50 个 (arm,seed) 中 14 个重复被静默平均 | 聚合器现锁定单包并去重，旧 run 保留但不参与 |
| `event_rate_only_drift` 旧结果 | 生成器从未改动速率，与 `no_state` 逐位相同 | 已修生成器并重跑 |

## 7. 图形

**新建：** `asset_id=epi_prssm_core_evidence`，`paper_slot=TBD`
**路径：** `results/epi_prssm/v0_1/figures/revisions/20260820-0949/epi_prssm_core_evidence/`

- 面板：A 关闭观测后的 H5/H10/H20/H40；B 生成器/decoder 图通路 2×2（**当前为占位**，等重跑）；C correct-state vs matched-state swap；D 同前缀不同后缀
- 字号：本图自设 `FS_TICK/FS_AXIS/FS_TITLE/FS_LEGEND = 7.0/7.6/8.4/7.0`（项目默认 tick 6.8 pt **低于 7 pt 下限**，不能直接沿用）
- 导出：PNG 600 dpi + 单页矢量 PDF（171.6 × 107.3 mm，`/FontFile2` 嵌入 Type0 可编辑字体，无位图 XObject）+ metadata JSON + 中文 `figures/README.md`
- 目视验收：3 轮。抓到并修复——y 轴标签裁切、图例压数据、标题溢出、结构性零柱无标注（读作缺数据）、面板字母与长 y 标签碰撞
- **一次静默失败**：Panel C 的第一次 `str.replace` 未匹配而无声跳过，靠目视比对发现

**未完成：** 其余 4 个 exploratory asset 的重绘（architecture 缩为方法面板、seizure 改为 denominator flow + observability + subtype interaction、exposure 改名 H3 diagnostic 并灰显 GOF 不合格 contrast、event_distribution）尚未执行。

## 8. 复现命令

```bash
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
export OMP_NUM_THREADS=1
$PY scripts/topic5_epi_prssm/build_seizure_crosswalk.py --layer linear_graph_recurrent --lead lead30m
$PY scripts/topic5_epi_prssm/build_h2b_denominators.py
$PY scripts/topic5_epi_prssm/run_h2b_sensitivity.py
$PY scripts/topic5_epi_prssm/run_h3b_transition_coupling.py
$PY scripts/topic5_epi_prssm/run_arrival_channel.py --arm t1_observer --seed 11 --cohort all34 --max-epochs 30 --markov-renewal
$PY scripts/topic5_epi_prssm/aggregate_graph_null.py --cohort all34      # 锁单包并去重
$PY scripts/topic5_epi_prssm/make_figure_core_evidence.py --run-id 20260820-0949
```
