# Fig.5 dual-core transition story v2 审阅（2026-09-04）

## 2026-09-05 复核后的状态修正

本页最初生成的 A–F 只能保留为排版/语义原型，不能作为新版 Fig.5。原 A 是全区域高率的 tonic plateau，
不是持续跨 shaft 高频振荡；原 E 是 equilibrium branch atlas，原 F 是全部 run 都转变的 latency response
surface，二者都不是真正 dynamical-regime/phase map。

更精细的原生步长 fast-subsystem 分析找到 `s≈0.27, tau_GABA≈8.5–9 ms` 的窄 oscillatory ridge：25 格主网
1 格、15 格 zoom 2 格通过；`s=0.27,tau_GABA=9 ms` 延长 5 s 后保持 30 Hz/depth≈0.207，同参数 low/high
初值分别保持 low 与 oscillatory-recruited，说明 basin coexistence。但该 ridge 紧贴 localized→tonic 边界，尚未
跨 topology 或完整 SNN 成立。

新增含 4×/8× M 扩展的 77 格 native spatial-Z × dynamic-M 图后，low/pre-fold 初态为 58 low + 19 localized，
recruited 初态为 16 localized + 59 tonic + 2 oscillatory；两个 oscillatory 格仍只在 `s=0.27`，强 M 只把
该处 recruited boundary 推到约 `s=0.28`。fold 后 `s=0.34/0.36` 的
low/pre-fold 轨迹延长 5 s 仍为 localized，所以 fold 不是 global-runaway 相界。

40,000-cell `tau_GABA=8/4/2 ms` 及 `si=0.8,tau_GABA=9 ms` sensitivity 的全区域 rate 均进入高态，群体主峰为
52/59/66/54 Hz；但群体 depth 仅 0.064/0.079/0.097/0.0365。逐触点同时要求 250-ms 与 100-ms persistence 后，四者
SCL 均为 0/4。4-ms 的整段 PSD 或单独 250-ms 粗窗会产生跨 shaft 假阳性；SCL 只通过 2–4/10 个 100-ms
小窗，仍是间歇 burst，不能升级为持续全局振荡。
完整证据见 `dual_core_spatial_z_bifurcation_result_2026-09-04.md` §7。

## 一句话结论（原 v2 设计，已降级）

现有 artifact 已能生成一张语义闭合的 A–F 候选图：A/B/C 是同一条 continuous-OU、Z/M-active dual-core SNN 轨迹，D 是独立临床 cohort 空间桥，E 是 verified spatial-Z saddle-node 多分支图谱，F 是已有 `tau_z × tau_m` operational-runaway latency 网格。最大缺口不是排版，而是 F 尚未做 `depletion strength × tau_z` 专扫，C 也仍是单 realized trajectory。

## 1. A/B 为什么不画“上升的 broadband power”

冻结轨迹为 `rev21_ts_tz3000_ta500`、topology 2542、dynamics 2642；`dualcore_s39 + Joint=1.25`、continuous OU、Z/M 均开启。rev21 model-state onset 为 `2515.4 ms`，记录到 `3815.4 ms`。

这个工作点进入的是近饱和 tonic recruited plateau：population E rate 与 global recruitment 明显上升，但 10–250 Hz band-limited virtual-contact power 在 onset 后下降。后者如果画成上升只能靠改定义或混入 DC，会把 tonic plateau 冒充 broadband oscillatory energy。因此 v2 的 B 顶轨用直接 population E rate；A 保留 30–80 Hz contact 波形，并用独立 recruitment strip 显示全片高态。

允许写：慢抑制耗竭伴随群体率和招募比例上升。不能写：模型复现了 onset 后 broadband oscillatory-power 增强。

## 2. A/B/C 时间与空间合同

- A/B 共用 `0–3.815 s` 的同一 x 轴、同一 onset 竖线和同一窗口阴影。
- 间期代表事件按“latest returned event with complete 15-contact onset readout”选定，为 `1402–1674 ms`；不是按图形美观选择。
- pre-onset 窗为冻结 `2015.4–2515.4 ms`。
- early recruited 窗为冻结 early-ictal 起点后的前 400 ms，即 `2615.4–3015.4 ms`。
- C 三图统一用 0.5-mm bin 内 mean local E spike rate，显示 `log10(1+Hz)`，三图共用同一 normalization。

空间参考轴由所选 returned interictal event 的 onset map 做 least-squares plane gradient；每个状态图再对 `log(1+local E rate)` 拟合 gradient，读出绝对余弦：

| 窗口 | `|cos Δtheta|` |
|---|---:|
| interictal | 0.9947 |
| pre-onset | 0.9923 |
| early recruited | 0.9457 |

这支持“该代表轨迹三个时段的活动梯度保持同一轴”。interictal 自身值是参考内一致性；承重变化是 pre-onset 与 early-recruited 两个独立窗口。它不是多 seed 或 cohort 结论。

## 3. D 的指标与边界

D 不把临床数据写成 across-patient two-core simulation。它复用 frozen strict-broadband cohort：每患者将 interictal A/B propagation field 与 clinical onset 后 `0–10 s`、`1–150 Hz` early-ictal energy field 做绝对空间相关，先在患者内折叠，再与同一患者 all-contact channel-shuffle 的 null median 配对。

- n = 16 patients / 106 seizures；
- 12/16 的 observed 高于各自 null median；
- one-sided paired Wilcoxon observed > null median：`p=0.0193`；
- observed cohort median `0.7845`，null median `0.7802`。

这个 panel 支持 cohort-level 的粗空间 concordance。它不支持逐点 replay，也不能写成“16 个患者的 individualized SNN 预测成功”。all-contact channel shuffle 保留触点坐标但比 within-shaft null 弱；若 D 要独立承重模型空间泛化，必须补更强 null 或将本 panel 明确保留为临床 bridge。

## 4. E 与 F 分别回答什么

E 回答 frozen fast subsystem 的状态几何：spatial-Z continuation 中存在 low outer root、tonic outer root 与至少两条实际续接的空间 family；fold 必须同时满足 pseudo-arclength 折返和 fixed-point Jacobian 实零特征值穿零。所有 branch loci 用实线，线型不编码 delay-aware stability。OU-on 中位竖线只表示已测试工作截面，不证明整条高支稳定。

F 回答现有 SNN library 中控制参数怎样改变 operational runaway timing。`tau_z × tau_m = 3×3`，每格 2 topology × 2 dynamics，全部 4/4 触发 operational detector；cell median latency 随 `tau_z=3/5/8 s` 的行中位约为 `2.11/3.60/5.25 s`。这说明 inhibitory recovery timescale 是当前网格里的主要 latency lever；`tau_m` 效应较弱且不单调。

F 不能写成 clinical seizure latency，也不能掩盖 rev21 原 formal result：9 个 timescale cells 中只有单个 run 过完整 model-ictal eligibility，timescale grid 没有冻结正式 cross-state workpoint。用户当前接受 tonic runaway 后，4/4 operational transition 可用于 runaway-latency 描述，但不能追溯升级旧 formal gate。

## 5. 最小补分析路线

1. **F 的承重专扫**：固定同一 two-core substrate，扫 `depletion strength × tau_z`；至少 5×6 参数、3 topology × 4 dynamics。输出 transition probability 与 right-censor-aware latency（优先 RMST），未转变 run 不得从中位数分母删除。
2. **C 的多 seed 确认**：在不按图选 run 的冻结 seed 集上，复用相同三个相对时间窗和相同 gradient estimator；报告每 run 的 pre-onset / early-recruited `|cos Δtheta|` 与 axis-shuffle null。
3. **D 的位置选择**：若留主图，改用 within-shaft null 或在 caption 明写 all-contact null；若 Fig.3 已充分承载该临床结果，则 Fig.5D 可替换成 across-seed model alignment，避免重复。
4. **E 的尺度敏感性**：至少补 3 topology 和第二 coarse resolution；此前不要把 2-mm single-topology fold 数写成 patient-general law。

## 6. 产出

- producer：`scripts/paper_figures/plot_fig5_dual_core_transition_story.py`
- 图：`results/paper-ready-figure/fig5_dual_core_transition_story/figures/fig5-dual-core-transition-story-v2.{png,pdf,svg}`
- metadata：同目录 `fig5-dual-core-transition-story-v2-metadata.json`
- 图说明：同目录 `README.md`
- 针对性测试：`tests/test_fig5_dual_core_transition_story.py`

当前状态：`CANDIDATE`，不是 author-locked Fig.5。
