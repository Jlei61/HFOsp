# Figure 6：G 与 M 共用 logistic 的原生验证

科学问题是：把两处 clip 同时改成平滑 sigmoid 后，是否仍能经历高活动、退出、恢复间期短事件并再次进入。统计单位为同一固定网络、同一噪声种子的配对轨迹；比较对象为此前 Figure 6 的双 clip 版本。

$$
S(R;R_0,k)=\frac{1}{1+\exp[-(R-R_0)/k]},\quad
\tau_G\dot G=G_{\max}S-G,\quad
\dot M_i=-\frac{M_i}{\tau_M(R)}+\frac{a_0/\tau_M(R)+a_1S}{\eta_M}\sum_n\delta(t-t_i^n).
$$

## 实际结果

| 门控 | 观察时长 | 前 8 s 短事件数 | 进入 → 退出 → 两 core Z 恢复 → 再进入 (s) | 恢复后的短事件 / 覆盖时长 | 判断 |
|---|---|---:|---|---|---|
| 旧 clip | 120 s | 38 | 9.39 → 17.04 → 24.38 → 60.22 | 45 次 / 12.05 s | 完整循环 |
| logistic，k=25 Hz | 120 s | 34 | 17.06 → 18.99 → 24.08 → 45.16 | 65 次 / 19.33 s | 完整循环；最初一次退出后快速反跳，随后周期才完整恢复 |
| logistic，k=40 Hz | 120 s | 34 | 22.04 → 27.09 → 34.40 → 71.81 | 48 次 / 14.54 s | 完整循环 |
| logistic，k=50 Hz | 60 s | 8 | — | — | 60 s 内未进入；之后未计算 |
| logistic，k=75 Hz | 60 s | 0 | — | — | 60 s 内未进入；之后未计算 |

采用 k=40 Hz 的完整第一周期更新 Figure 6 A–D。k=25 Hz 也能产生完整循环，但第一次退出后先出现快速反跳；k=50/75 Hz 在完整 60 s 筛查内没有进入高活动，保留可恢复的检查点，不能据此推断更长时间也不会进入。所有运行均报告，宽度选择是基于开发轨迹的有界调整，没有更换噪声种子。

较宽 sigmoid 在低活动区具有较大的非零尾部，因此可能提前募集反馈并抑制间期活动。这与本次不同宽度的现象一致，但仅凭这四条轨迹不能分别确定 G 和 M 的独立贡献。结论是共享 logistic 存在实现完整过程的参数设置，不是任意参数均成立，也不是跨网络或跨噪声的稳健性证明。

## 观察量与一致性检查

- 高活动：10 ms 窗口全体 E 发放率持续 200 ms 不低于 200 Hz。
- 退出：沿用旧 observer，两段连续 1 s 窗口中全体 E 与两个 core 的均值均低于 50 Hz，静默占比达到 0.2。退出时刻与“所有活动完全静默”的时刻不是同一观察量。
- 恢复：两个 core 的 Z 均恢复至各自初始 8 s 参考值，此后至少 10 次完整 20–200 ms 事件覆盖至少 5 s，再出现新的持续高活动。
- 参数替换：G 和 M 使用同一 logistic，保存的每个门控采样与公式逐值相等；G=Gmax·s、每次放电的 M 增量与方程一致。
- 对照一致：网络身份不变，配对种子相同，已记录的背景输入采样与旧 clip 前缀逐值相等；全局、区域与空间网格的原生放电计数守恒。
- 原有 Z 负荷阈值和低活动 τM 保留规则未改变。本文的“平滑”专指两处活动门控，不表示完整放电网络处处可微。

具体参数见 [effective_protocol.json](effective_protocol.json)，完整结果见 [comparison.json](comparison.json)。[连续轨迹对照](figures/paired_clip_logistic_validation.png)与[门控曲线](figures/logistic_gate_comparison.png)来自同批结果。原始连续放电、虚拟触点电流与检查点保留在 `/data/hfosp/topic4_sef_hfo/logistic_feedback_20261102/`。

## 复现与来源

共享机器上，使用冻结网络、原生/临床数据及已安装的 NumPy、SciPy、Numba CUDA 环境：

```bash
python scripts/run_topic4_logistic_feedback.py worker --name logistic_r350_k40_s9108404 --midpoint 350 --width 40 --device 0
python scripts/analyze_topic4_logistic_feedback.py
python scripts/paper_figures/build_fig6_logistic_feedback.py results/paper-ready-figure/fig6/candidates/shared_logistic_feedback_20261102/display_configuration.json
```

不要并行重启同名已在运行的任务。冻结 runner 的初始默认宽度 75 Hz 对应首个筛查设置；上面的显式 40 Hz 才是当前图的数据来源。运行所需的历史网络与原始数据不包含在 Git 包中；源码及依赖快照保存在此处 `source/` 和 Figure 6 候选 `source/python/`，路径和哈希见 effective protocol。`source/inherited_protocol.json` 仅保存历史参数链，其中旧 clip/K 字段不定义当前方程。

新版 A–D 来自同一条 120 s 共享 logistic 轨迹；E 的 13 患者、140 次发作统计保持原历史模型，尚未用新模型重算。Figure 6 为待作者目视检查的候选，正式指针未改。
