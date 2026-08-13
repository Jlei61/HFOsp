# Topic 4 rev10-D7：active Z+M 连续场稳定性与模式可达性 canary

## 唯一科学问题

在冻结 Z/M 机制参考、spatial OU、总 field mass、网络拓扑和观察模型后，改变不使用电极位置构造的连续 `h(x,y)`，是否能同时：

1. 避免当前 warm field 在 20 s 内出现的晚期 runaway；
2. 保留同网络自然 KMeans 双模式；
3. 改善 patient-training A/B 几何，而不是只产生可监督分开的标签。

## 候选与方程

候选复用 D6 的 49 个 whole-sheet low-frequency Fourier residual 场，均投影到同一个 18×18 tensor cubic B-spline 表示并保持总 field mass。场构造禁止使用 contact 坐标、shaft identity、患者事件或标签，也没有 `K`、component 或 peak 参数。

所有候选显式继承共享 baseline：

```text
config/topic4_data_driven_snn_baseline_zm_v1.json
runtime_mode = active_z_plus_m
```

动力学为：

```text
V_th,i = V_th,0 - h_i d_i
tau_z dz_i/dt = H(I_th_EI - I_i^EI) - z_i
dm_i/dt = -m_i/tau_adp + sum_k delta(t - t_i^k)
I_net,i = I_i^E - z_i I_i^I - eta_m m_i
```

固定 `tau_z=5000 ms`、`tau_adp=500 ms`、`eta_m=0.0074516`。这是 unsafe mechanism reference，不是已通过的稳定 baseline。

## 执行合同

- fit seeds：`1421, 1422`，候选间 common random numbers；
- 49 candidates × 2 networks = 98 workers；
- 每 worker 20 s，允许 early-stop runaway，但任何早/晚 runaway 都使该 network 无效；
- common absolute detector 与 spatial OU 不变；
- Edge exact no-op，`beta`、topology growth、额外 adaptation/resource/EE-STD 关闭；
- worker 数由实测 RSS 和当前 `MemAvailable` 决定，单 worker cgroup `MemoryMax=24G`。

## 探索性读数

不新增大量 hard gate。先按以下顺序审计：

1. 2/2 网络无 runaway；
2. 每网络自然 KMeans 是否可评价及 balanced alignment；
3. patient contact-split cross-fit signed margin；
4. worst-mode recruitment error、OOD 和 detector occupancy。

标量 score 只用于排序，不能单独构成科学通过。只有 2/2 安全且 KMeans 可评价的候选才可进入新网络 selection；若没有候选满足，停止在 canary，不调 optimizer。

## 后续规则与结论边界

若出现安全候选，最多选 4 个，在新网络上成对运行 `active_z_plus_m` 与 `paired_slow_off`，区分 field 本身与 Z/M 交互。只有 selection 后再冻结单一候选做 confirmation，并生成标准两图：

- `fig4a_spatial_ou_direct_readout`
- `fig4b_spatial_ou_kmeans_consistency`

本轮使用 patient-training target，没有新 patient blind unit。阳性结果也只能说明 development-level active Z+M field feasibility；不能称为完整患者间期活动、patient generalization、core 因果、Edge 机制或发作生命周期。

