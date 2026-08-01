# FCXR-LC2-Phenotype — deferred waveform, spatial, and E1146 confirmation layer

日期：2026-08-01

状态：**DEFERRED — LOCKED OUT UNTIL CORE_LIFECYCLE_REPLICATED**

Core spec：`docs/superpowers/specs/2026-08-01-topic4-fcxr-lc2-core-design.md`

> 本文件只锁下一层的科学边界，不授权实现或仿真。没有
> `CORE_LIFECYCLE_REPLICATED`，不得写 implementation plan、不得建立患者阈值、不得用 M/空间机制
> 回救 Core。

---

## 1. 唯一目标

在冻结全部 H/X/Z Core 参数后，检验逐细胞 M_i 能否在**不破坏已复制 lifecycle**的条件下改善
high-state waveform；随后检验患者特异轴向 recruitment 和 E1146 observation matching。

这一层回答“像不像目标发作”，不再回答“核心生命周期能不能闭合”。

---

## 2. 解锁顺序

### P1：M morphology

比较：

- M-off；
- per-cell M_i；
- matched-total-load mean-field M。

H/X/Z、连接、noise protocol 和 lifecycle classifier 全冻结。允许结论：

```text
lifecycle preserved, morphology improved
lifecycle preserved, morphology negative
M_i destroys lifecycle
```

后二者不能改写为 Core negative。

### P2：空间 recruitment

只有 P1 保留 lifecycle 后，才检验 first-passage、newly recruited area、front velocity、axis/off-axis
latency 和 phase staggering。总 occupied-volume 只作描述，不投票。

### P3：E1146 observation matching

最后才锁 returning-interictal、early-ictal、offset、postictal/recovery windows 和 dynamotype。模型与数据
只比较各自 baseline-normalized 指标，不比较绝对电压。

---

## 3. 时间窗合同

同一 high-state episode 内允许不同的嵌套窗：

- spectral/broadband/harmonic-comb/coherence：连续 1–2 s established-high epoch；
- spatial phase、first passage、局部 burst order：嵌套 200–300 ms windows；
- 两层必须来自同一 episode，但不要求同一个 200–300 ms 微窗。

禁止用 200–300 ms 单窗稳定估计 3–8 Hz 频谱、spectral entropy 或跨接触 coherence。

---

## 4. 结果层级

| Core | P1 | P2 | P3 | 安全结论 |
|---|---|---|---|---|
| fail | — | — | — | lifecycle geometry negative |
| pass | fail | — | — | mechanism positive, morphology negative |
| pass | pass | fail | — | lifecycle/waveform positive, spatial negative |
| pass | pass | pass | fail | mechanistic-spatial candidate, patient matching negative |
| pass | pass | pass | pass | patient-phenotype candidate |

“patient-phenotype candidate”仍不是完整患者离子机制、通用 seizure dynamotype 或最终论文因果结论。

---

## 5. 明确延后

- E1146 empirical/dynamotype window builder；
- per-cell M 参数和 force-match；
- spatial first-passage 与轴旋转/isotropic controls；
- 15-contact virtual SEEG；
- replication/confirmatory phenotype seeds；
- full causal ablation、eigenmode 和 paper-ready figure。

这些内容将在 Core replicated 后另写独立 plan；本文件没有执行权限。
