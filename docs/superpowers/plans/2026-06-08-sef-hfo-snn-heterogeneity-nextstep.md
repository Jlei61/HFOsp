# SEF-HFO 放电网络异质性 — Next-Step 路线图（2026-06-08 review 收口）

> 承接 `docs/archive/topic4/sef_hfo/snn_heterogeneity_mechanism_2026-06-08.md`（主结果）+ review（2026-06-08）。本轮已做：2×2 轴（variance/mean/combined）、ignition 拆分、干净自限指标、provenance、science-guard 测试、传播图/异质性图分离。本文件记录**本轮不做、留作后续轮次**的工作，按 user 在 review 时的优先级排序。

## 优先级 1 — 多种子 + 位置/幅度 sweep（统计裁决前置）

当前除 mid 核 3 种子外都是单种子粗筛，"−0.01±0.05 / +0.51±0.04"只够机制筛查，不够论文统计。最低配置：

- **每位置 ≥10–20 个独立实现**：网络连接种子 × OU 噪声种子 × kick 种子分开扫，报 mean±CI（不是 std/3）。
- **位置系统扫**：along-axis（近种子→远端连续）/ off-axis / far-end，至少 5–7 个位置，确认 unmatched 的"位置依赖正负翻转"是连续规律还是个别格子噪声。
- **kick 幅度三档**（低/中/高 × nu_theta）：避免单一刺激强度决定结论；尤其确认 matched-null 在更强刺激下是否仍 null。
- **统计**：matched vs baseline 配对检验（每位置）；unmatched/mean_only 因自点火，改报 ignition 概率 + latency 分布（见优先级 4）。

## 优先级 2 — 前向 LFP/HFO 读出层（才敢说对应 SEEG）

当前读出是 firing-density envelope（**不是 LFP**），小尺度（3mm）、电极按比例缩小。要和临床 SEEG/HFO 对齐需要一层正式 forward model：

1. spike/rate → 突触电流 proxy（推广 `engine/lfp.py` 的 |I_E|+|I_I|，已有雏形）。
2. 空间平滑 / 体积传导核（distance-decay forward kernel）。
3. 虚拟电极读出（真实 SEEG 间距 3.5–5mm；需先把 SNN 放大到 cm 级，见框架 §13.1）。
4. 带限事件幅度 / HFO proxy（80–250 Hz envelope；注意 framework §8.1 红线：模型不产 carrier，HFO proxy 只能是 event-envelope 层）。

只有加完这层，"传播方向/同步"才从"模型内部机制"升级到"对应 SEEG 观测"。

## 优先级 3 — 异质性拓宽（不只阈值）

当前把异质性 operationalize 成**单一的点火门槛离散度**，这是 Rich 宽义异质性（细胞 f-I 多样性、突触多样性、抑制多样性、通道表达、微环路架构）的一个**窄 projection**。措辞已锁"We operationalized heterogeneity as local spike-threshold dispersion"，不写"tested epileptic heterogeneity"。后续轴：

- 突触权重 / 时间常数离散度；
- 抑制（I 细胞）异质性 / I→E 覆盖离散度；
- 连接架构异质性（局部 degree / 各向异性的空间变化）。
每条都走同一套 2×2（mean/variance 分离）+ ignition 拆分 + 自限纪律。

## 优先级 4（review 提出、user 未点选，记录在案）

- **率配平 / 转进函数配平控制**：matched 现在只是"门槛均值配平"，非"发放率配平 / F-I 斜率配平"，仍带 Jensen 残留。"纯方差轴"的干净控制需要：baseline-rate matched（调均值使自发率一致）+ evoked-response matched（调输入使 kick 后早期峰一致）+ transfer-function matched（直接估核内 F-I，使工作点附近 F₀/斜率一致）。**注**：本轮已加 mean-only 轴（std 固定、mean 下移），可与 unmatched 比，部分回答"是否纯 mean 效应"——但完整"纯方差轴"仍需上述配平。
- **evoked vs spontaneous 指标完全拆分**：runner 现在给每个 run 打 `evoked_clean`（base/core 都不 pre-kick 点火才算）+ `core_prekick_ignited` + ignition latency。下一步把 d_core_paf **只在 evoked_clean trials 上**作"诱发事件同步"统计；igniting conds 改报"自发点火"族指标（pre-kick burst probability / ignition latency / ignition location / pre-kick rate elevation），两类指标永不混用一个数。

## 不变量（跨所有后续轮次）

- gap-limit 不偷偷放宽（std 锁 1.5/0.5，mean 下移记录）；clinical SOZ 不作拟合；近临界→韧性措辞。
- matched-null 永远写"在这个工作点/尺度/基质没复现"，不写"证否 Rich"。
- B 侧"宽参差→自发爆发"是**独立的 tail-driven nucleation 机制**（低门槛尾造有限尺寸成核点），**不**并进"loss of heterogeneity"机制链。
- 引擎改动 provenance（git hash + checksum + config + seed + metric version）每个结果 JSON 必写。
