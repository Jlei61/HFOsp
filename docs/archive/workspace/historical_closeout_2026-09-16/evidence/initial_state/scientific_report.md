# 同一结构下的初态条件传播 v1：科学报告（候选版）

状态：`INITIAL_STATE_ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW`；生成时间 2026-09-08 05:19。工作树快照 `0431f8cea2b9`（/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-initial-state-propagation-v1）。

## 1. 先回答问题

- **初态效应**：`SMALL_EFFECT_BOUNDED_FOR_TESTED_PROBE`。
- **患者传播结构是否改善**：`STRUCTURE_GAP_UNRESOLVED_ALL_ARM_MODE_OUTSIDE_PATIENT_BAND_NO_SEPARATED_IMPROVEMENT_VS_B0`。

## 2. 测了什么、怎么测的

把同一张固定网络（连接、细胞位置、两个病灶核、逐细胞阈值、所有时间常数、外源随机输入规律全部不变）从三种起点各跑 24 秒：
B0（共同基线：全体 V_reset）；B1（core A 的 733 个 E 细胞 +1 mV）；B2（core B 的 733 个 E 细胞 +1 mV）。同一噪声 seed 的三臂收到逐步完全相同的全局 OU、空间 OU 与 Poisson 输入（用每秒一段的流式摘要逐段核验，不是只凭同一个整数 seed）。
然后用冻结的患者观察器（0.5 s burn-in 不变）检测事件，用冻结的患者分类器给每个 primary 事件贴 M0/M1 标签；
主要读出只有一个：12–24 s 窗内 M0 占比的 B1−B2 差，以 12 个噪声配对为统计单位，等权平均，不池化事件。

## 3. 资格检查（I0）

`qualification.json`：I0_QUALIFICATION_PASS，20/20 项通过；B1/B2 与 B0 轨迹分离时刻 {'B1_vs_B0_ms': 29.1, 'B2_vs_B0_ms': 22.0, 'B1_vs_B2_ms': 22.0}；短程单元估算正式单元约 44 分钟、进程峰值 3.3 GiB。

## 4. 主要结果

- I1 筛查（拓扑 2511，dynamics 820101–820112）：12/12 对可估计。B1−B2 的 12–24 s M0 比例差 Δ = -0.9 个百分点，95% 配对 bootstrap 区间 [-9.0, 6.3]，双侧配对交换 p = 0.8340（4096 种符号分配）；预注册不稳标志：leave_one_out_sign_flip（去一对后均值变号只是均值≈0 时的必然现象，不改变'区间落在 ±10 点内'的判定）；最大单对贡献 27.7%。判定 SMALL_EFFECT_BOUNDED_FOR_TESTED_PROBE。
- I2 重复：未触发（I1 判定 SMALL_EFFECT_BOUNDED_FOR_TESTED_PROBE 未达到持续效应门槛）。

次要窗口（screen）B1−B2 平均差（百分点，95% 区间；描述性）：
  - 0.5-6 s: -2.6 [-17.4, 11.7]，12/12 对
  - 6-12 s: 3.5 [-9.2, 15.6]，12/12 对
  - 12-18 s: 5.6 [-4.2, 14.6]，12/12 对
  - 18-24 s: -10.6 [-25.7, 5.0]，12/12 对
  - 0.5-24 s: -0.8 [-7.4, 5.3]，12/12 对
晚期窗口相对 B0（Holm 校正两项）：B1-B0: -9.0 点，Holm p = 0.2070；B2-B0: -8.1 点，Holm p = 0.2070

## 5. 模式条件传播质量（独立结果层）

患者 FIT 参考：M0 6605 事件、M1 13165 事件；各块自然 D_off 中位数 M0 0.0110、M1 0.0130。

screen：
  - B0_M0: 可估计运行 12，run 级 D_off 均值 0.0911（区间 [0.0502, 0.1401]），患者匹配带 [-0.0157, 0.0018, 0.0456]，参与率 MAE 0.130，成对时差残差 MAE 17.2 ms，顺序 TV 0.240，支持/OOD 比例 53.2%/13.7%
  - B1_M0: 可估计运行 12，run 级 D_off 均值 0.0918（区间 [0.0639, 0.1204]），患者匹配带 [-0.0198, 0.0012, 0.0502]，参与率 MAE 0.105，成对时差残差 MAE 15.7 ms，顺序 TV 0.243，支持/OOD 比例 56.1%/12.2%
  - B2_M0: 可估计运行 12，run 级 D_off 均值 0.0733（区间 [0.0407, 0.1138]），患者匹配带 [-0.0201, 0.0016, 0.0506]，参与率 MAE 0.120，成对时差残差 MAE 13.6 ms，顺序 TV 0.211，支持/OOD 比例 58.7%/11.6%
  - B0_M1: 可估计运行 12，run 级 D_off 均值 0.0914（区间 [0.0653, 0.1337]），患者匹配带 [-0.0103, 0.0014, 0.0305]，参与率 MAE 0.219，成对时差残差 MAE 6.6 ms，顺序 TV 0.189，支持/OOD 比例 62.9%/21.1%
  - B1_M1: 可估计运行 12，run 级 D_off 均值 0.1090（区间 [0.082, 0.1391]），患者匹配带 [-0.0086, 0.0017, 0.026]，参与率 MAE 0.217，成对时差残差 MAE 6.8 ms，顺序 TV 0.165，支持/OOD 比例 55.0%/26.3%
  - B2_M1: 可估计运行 12，run 级 D_off 均值 0.0884（区间 [0.0706, 0.1068]），患者匹配带 [-0.009, 0.0016, 0.0284]，参与率 MAE 0.212，成对时差残差 MAE 6.7 ms，顺序 TV 0.177，支持/OOD 比例 53.7%/23.4%

相对共同基线 B0 的描述性比较（不是检验）：
  - screen B1 M0: D_off 0.0918 vs B0 0.0911（run 级区间重叠），参与率更近，成对时差更近，落入患者匹配带 False
  - screen B2 M0: D_off 0.0733 vs B0 0.0911（run 级区间重叠），参与率更近，成对时差更近，落入患者匹配带 False
  - screen B1 M1: D_off 0.1090 vs B0 0.0914（run 级区间重叠），参与率更近，成对时差未更近，落入患者匹配带 False
  - screen B2 M1: D_off 0.0884 vs B0 0.0914（run 级区间重叠），参与率更近，成对时差未更近，落入患者匹配带 False

## 6. 完成数与工程状态

正式 24 s 运行 36 次（设计 36），提前终止（runaway 门）0 次，单次墙钟中位 26.0 分钟。
跨臂输入摘要与静态数组身份核查：screen: 摘要全等 True，静态身份全等 True，与主线历史 worker 一致 True

## 7. 限制与边界

- a significant late-window difference on one or two graphs is not evidence of patient bistability
- two hand-picked initial states mixed at patient proportions are not a state-access mechanism
- the 1 mV cold-start voltage probe covers a small part of state space; negatives do not exclude synaptic, inhibitory, adaptive or slow-variable states
- route diagnostics are development checks on FIT only and feed nothing back into the probe
- 纯 V 初态只覆盖状态空间的一小部分；阴性只限制这个工作点上的 1 mV 冷启动探针。
- 12 对是有限筛查预算，没有独立初态方差估计，不承诺检出 10 个百分点。
- 所有图仍是候选版，需用户亲自目视检查后才算验收。

## 8. 下一步（不在本轮执行）

- 若初态效应持续且可重复：按方案第 9 节检验自主访问与状态记忆（自然静息期完整 checkpoint 延续、原位扰动/恢复）。
- 若阴性或仅瞬态：隐状态作为新的结构假设（固定标量 s 调制阈值），必须与 s=0 及单峰慢漂移模型对照。
- 若状态只改标签占比而不改善参与/时差结构：停止用状态选择解决传播质量缺口。