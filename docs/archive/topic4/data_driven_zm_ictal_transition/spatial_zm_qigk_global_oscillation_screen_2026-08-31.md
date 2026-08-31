# 空间 Z/qI–M/gK 持续全局振荡筛选：当前两变量线未通过

**Date:** 2026-08-31  
**Status:** `CLOSED_NEGATIVE_WITH_PARTIAL_SPATIAL_RHYTHM`  
**Figure decision:** Fig5A formal render **blocked**；没有生成或登记“通过版”图片。

## 一句话结论

合作者的质疑成立：原 data-driven Z/M 轨迹只进入持续高率态，不是持续高频、全局招募、
跨触点相锁的振荡态。把逐细胞 Z/M 连续推广为空间 `q_I(x,t)` 与局部 M/gK 后，确实能在
冻结 E1146 learned E→E/E→I 全开底物上产生稳健的 40–46 Hz 局部/传播性节律，但最佳仍只有
9/15（60%）触点在 4 个 250 ms 窗中至少通过 3 个；冻结门要求 12/15（80%）。全片 gK 会把
频率拖到 20–26 Hz。当前纯 Z/M–qI/gK 两变量线因此不能支撑目标 Fig5A。

## 1. 冻结问题与判据

目标不是旧的 `120 Hz for 100 ms` 操作性 runaway，而是同一条连续轨迹同时满足：

1. 转变前驻留至少 2 s，转变前 500 ms 中位率 ≤60 Hz、q95 <120 Hz；
2. 转变后 300 ms settling 之后，连续 1 s 中位率 ≥120 Hz，且至少为转变前 2 倍；
3. 20 ms 窗中“≥50% E 神经元活跃且 ≥50% 组织片被招募”的联合占空比 ≥0.75；
4. 15 个虚拟触点各分成 4 个 250 ms 窗；每触点至少 3/4 窗的峰在 30–80 Hz、峰功率占比
   ≥0.20、带功率相对低态 ≥2；
5. 至少 80% 触点持续有节律，触点峰频 MAD ≤8 Hz。

科学 onset 使用 `oscillatory_median_v1`：最早一个 300 ms 前向窗，其 20 ms block-rate 中位数
≥120 Hz；onset 是该窗内第一个 ≥120 Hz block。该定义允许振荡谷值，但拒绝孤立短 burst。

旧 qI forced-pulse E1146 轨迹只作为**形态学仪器正对照**：15/15 触点、36 Hz、带功率比 61.85、
联合全局占空比 1.0。它不提供自发驻留、机制或患者波形证据。

## 2. 机制合同

只保留一张抑制资源场与一个 M/gK 状态，不堆四个同义慢变量：

\[
\dot q(x)=\frac{1-q(x)}{\tau_q}-k_q(x)F_n[K_q*r(x)]q(x),
\]

\[
\dot m_i=-\frac{m_i}{\tau_M}+k_M\left[(1-\rho)s_i+\rho(K_M*r_E)(x_i)\right].
\]

- `n=1` 精确回到历史 qI saturation；`n>1` 连续逼近 hard Z；
- `rho=0` 是逐细胞 M，`rho=1` 是局部空间 gK-like drive；
- `q_init_h_gain`、`k_q_h_gain`、`q_floor_h_gain`、`eta_m_h_gain` 只消费冻结患者场 `h(x)`；
- 没有随机逐神经元参数场；learned E→E/E→I 剂量固定 1.0，权重、拓扑、delay 均不改；
- `m_build_gain`（gK 建立率）与 `eta_m`（膜电流耦合）分开，避免一个参数同时改变时标和幅度。

实现：`src/topic4_spatial_zm_qigk.py`。runner：
`scripts/run_topic4_spatial_zm_qigk_canary.py`。

## 3. 结果

### 3.1 原 Z/M 与连续动态混合线

- 纯 Z/M 最佳 full-edge 候选有广泛高率招募，但触点主频 16.0→13.9 Hz；不是目标振荡。
- 动态 q/M 候选覆盖 hard-Z、Hill q、q-floor 0.40–0.75、M tau 6.25–25 ms、线性/有界 M
  电流、状态上界、0.5–1.0 mm 局部 M 场。没有一个通过 7 项形态门。
- 统一按冻结 onset/gate 重分类后共 39 条 discovery：30 条进入但不满足全局节律、5 条未进入持续
  高态、4 条 post/onset 窗不完整；`n_hybrid_full_edge_pass=0`，没有 primary candidate。
- 典型失败是 post rate 200–455 Hz、全局占空比 0.75–1.0，但带功率比只有
  `7e-5`–`7.6e-2`：高率固定点，不是节律高态。

### 3.2 frozen-q 快子系统 atlas（23 个状态）

用途：固定 q 后只看末端 1 s，问冻结快网络是否存在可供慢变量落入的振荡支路；这不是转变或
驻留证据。正式结果：
`results/topic4_sef_hfo/data_driven_zm_ictal_transition/spatial_zqim_hybrid/frozen_q_fast_subsystem_atlas.{json,csv}`。

| 条件 | 末端率 | 节律触点 | 峰频 | 峰占比 | 相对 q=1 带功率 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| q=1 homogeneous | 28.8 Hz | 0/15 | 20 Hz | 0.346 | 1.00 | 低态参考 |
| q=0.75 homogeneous | 402.6 Hz | 4/15 | 52 Hz | 0.503 | 3.47 | 局部，不全局 |
| q=0.725, h-gain=0.4 | 378.1 Hz | 9/15 | 44 Hz | 0.475 | 4.21 | 最早空间候选 |
| q=0.730, h-gain=0.45 | 393.9 Hz | 9/15 | 40 Hz | 0.470 | 6.42 | 局部平台 |
| q=0.735, h-gain=0.40 | 397.3 Hz | 9/15 | 44 Hz | 0.504 | 5.83 | 局部平台 |
| q=0.725, h-gain=0.4, q-floor=0.45–0.55 | 402–403 Hz | 9/15 | 44–46 Hz | 0.482–0.505 | 5.73–6.01 | 裁底不救 |

23/23 无通过；空间异质性把覆盖从 homogeneous 的 4/15 提高到 9/15，但局部细化在多个相邻点
形成同一 60% 平台，没有跨过 80%。未通过的远端 ICL 接触并非未招募，而是前后窗在 20 Hz tonic
与 44–48 Hz 传播节律之间切换，无法满足至少 3/4 窗持续节律。

### 3.3 frozen-q + gK atlas（17 个状态）

正式结果：
`results/topic4_sef_hfo/data_driven_zm_ictal_transition/spatial_zqim_hybrid/frozen_q_gk_fast_subsystem_atlas.{json,csv}`。

| gK 条件 | 末端率 | 节律触点 | 峰频 | 相对 q=1 带功率 | 判定 |
|---|---:|---:|---:|---:|---|
| 全片，tau=12.5 ms，kM=0.05–0.20，eta=60–120 | 200–244 Hz | ≤5/15 | 20–24 Hz | 4.55–7.85 | 频率过慢 |
| 全片，tau=6.25 ms，kM=0.05–0.10，eta=60–120 | 235–248 Hz | 5/15 | 24 Hz | 5.64–6.59 | 缩 tau 不救 |
| 高-h 定向，sigma=0.5 mm，kM=0.01–0.02 | 249–252 Hz | 5/15 | 24–26 Hz | 5.53–5.81 | 仍过慢 |
| 同上，M threshold=0.015/0.020/0.025 | 403 Hz | 9/15 | 44 Hz | 6.15–6.52 | 回到 q-only 60% 平台 |

17/17 无通过。全片 gK 能压低 tonic 率并增强带功率，但把系统锁到 20–26 Hz；按测得 M 末段
分布加入激活阈值后，频率回到 44 Hz，却只恢复 q-only 的 9/15，并不增加全局覆盖。

## 4. Fig5A 决策

`scripts/paper_figures/plot_fig5a_spatial_zm_qigk_dynamics.py` 已实现 fail-closed：只有同一冻结参数
家族至少 3 个新 confirmation seed 全部通过，才允许按 seed 排序中位数渲染 PNG/PDF/SVG。
本轮 discovery 中没有合格参数家族，因此：

- 不启动 confirmation seeds；
- 不生成 paper-ready Fig5A；
- 不用 9/15 的局部节律轨迹冒充“持续全局振荡”；
- 旧 Z/M Fig5 只能保留为“操作性 runaway / 持续高率”诊断，不能写成目标表型复现。

## 5. 科学边界与下一步

当前证据支持的窄结论是：**在这一块冻结患者来源底物、完整 learned E→E/E→I、给定 q/M 空间
参数化与固定形态门下，空间 q 能产生局部 40–46 Hz 节律，但 Z/M–qI/gK 两变量反馈不能同时维持
30–80 Hz 和 ≥80% 触点持续相锁。**这不是对 Z/M、qI 或 gK 的普遍不可能性证明。

失败是结构性的二选一：无/门控 gK 时快网络滑向高率 tonic；连续 gK 时出现更慢的 20–26 Hz
共同周期。下一步若仍要求目标 Fig5A，需要显式加入一个**集体快重置机制**，而不是继续扫慢变量
小数位，例如：

1. 在不改 learned topology 的前提下，加入可消融的 E→E 短时突触抑制/恢复；或
2. 冻结并校准 E-I synaptic kinetics / delay，使高态由 PING-like collective reset 维持。

这两项都超出“纯 Z/M–qI/gK 两变量线”，必须另立合同、配 matched slow-off 与 full-edge 对照后再
决定是否重开 Fig5A。

## 6. 可复现入口

```bash
python scripts/analyze_topic4_frozen_q_atlas.py \
  --input-dir results/topic4_sef_hfo/data_driven_zm_ictal_transition/spatial_zqim_hybrid/frozen_q_atlas \
  --out results/topic4_sef_hfo/data_driven_zm_ictal_transition/spatial_zqim_hybrid/frozen_q_fast_subsystem_atlas.json

python scripts/analyze_topic4_frozen_q_atlas.py \
  --input-dir results/topic4_sef_hfo/data_driven_zm_ictal_transition/spatial_zqim_hybrid/frozen_q_gk_atlas \
  --reference-json results/topic4_sef_hfo/data_driven_zm_ictal_transition/spatial_zqim_hybrid/frozen_q_atlas/seed1801_q1000.json \
  --out results/topic4_sef_hfo/data_driven_zm_ictal_transition/spatial_zqim_hybrid/frozen_q_gk_fast_subsystem_atlas.json
```

相关测试：`tests/test_topic4_spatial_zm_qigk.py`、
`tests/test_topic4_global_recruited_oscillation.py`、
`tests/test_analyze_topic4_frozen_q_atlas.py`、
`tests/test_aggregate_topic4_spatial_zm_qigk.py`、
`tests/test_plot_fig5a_spatial_zm_qigk_dynamics.py`。
