# FCXR-LC5v2.1 matched-dose episode-memory map：最终收口

日期：2026-08-14
分支：`codex/topic4-fcxr-lc3`

## 1. 最终判决

本轮完成基础 3×3、边界补点 11 格，以及两个受短观察窗限制条件的 exact-state 续跑。
逐细胞 episode-load `U_i` 在当前 ZM/RC1/H 底座上形成了一条很窄的“继续饱和—阻断自然进入”边界，
但没有观察到自主 offset，更没有 postictal protection、Z 恢复或 returning IED recovery。

因此本轮判为：

```text
BOUNDED_NEGATIVE_NO_OFFSET_WINDOW
```

这关闭的是当前 `q99-p0 + matched early-episode dose + cell-local U_i` 参数族，不是否定所有逐细胞
代谢负荷、pump 或 adaptation 终止机制。

## 2. 实验到底测了什么

所有 E 细胞使用同一套方程和参数，只积分自身 spike history；没有 population seizure sensor、空间
mask、recruited-area、Gaussian sharing 或人工 core 权重。比较的是：在 baseline leakage 和早期发作
积分剂量配平后，`tau_U={3,8,15} s` 与 `Gamma_U` 的组合能否把自然进入后的升级轨迹变成有限高态并
自主退出。

所有科学臂均从 fresh `t=0` 开始，U 始终在线，Z/H 动态，`X=1`、`M=0`，无 kick、reset 或参数 step。

## 3. 数字结果

20 格初筛原始标签为：

- 10 格 `ESCALATING_SATURATION`；
- 8 格 `ENTRY_BLOCKED_WITH_IED`；
- 2 格短窗 `CONTAINED_HIGH_NO_OFFSET`；
- 0 格 offset。

两格短窗读数不能并列当作成功：

1. `tau=3 s, Gamma=0.060` 在 18 s 时看似 contained，但 exact-state 续跑下一秒即达到 308.68 Hz，
   触发注册饱和线，最终改判 `ESCALATING_SATURATION`；当时 `D=0.511`、`H=19.822`，U 仍以
   `+5.04/s` 上升，估算释放时间中位数约 38 s。
2. `tau=15 s, Gamma=0.003` 到 23 s 才 onset，初始记录在 25 s 结束，只观察到 onset 后 2 s；
   exact-state 续跑到 27 s 后，末端 E 率达到 405.86 Hz，同样触发注册饱和线；当时 `D=0.573`、
   `H=25.763`，U 仍以 `+5.60/s` 上升，未见 offset。

合并续跑后的最终证据分层：

- 12 格升级至饱和；
- 8 格保持 IED 但阻断自然进入；
- 0 格仍属右删失；
- 0 格自主 offset；
- 0 格完整 lifecycle。

## 4. 科学解释

`tau_U` 确实移动了进入边界：记忆越长，较小剂量就足以延迟或阻断自然 onset。但在当前机制家族中，
这个移动没有打开一个可重复的中间动力学区。较弱 U 追不上不断上升的 D/H 正反馈，较强 U 则在 onset
前就降低易感性；目前没有条件同时做到“保留自然进入—形成有限高态—自主退出”。

这说明当前失败不只是时间窗太短。两个一度最有希望的短窗条件都从完整状态继续，并明确回到饱和类。
其中第二条续跑因早先 reducer-only 失败，25--26 s 的逐块输入 digest 无法回溯，但恢复使用的是已落盘
exact state，26--27 s 输入 digest 已记录；这不影响其 405.86 Hz 注册饱和判决，但该 provenance 限制
保留在原 summary 中。

允许声称：

- cell-local episode memory 可以强烈移动自然进入概率/时刻；
- 当前相图存在陡峭的 saturation-to-entry-blocked 边界；
- 当前锁定家族没有展示 autonomous offset。

禁止声称：

- 逐细胞 pump/adaptation 在一般意义上不能终止发作；
- 两个短窗 contained 标签已经形成 bounded ictal carrier；
- 获得 postictal protection、Z recovery、returning IED recovery 或完整 lifecycle；
- 当前结果已经回答是否需要修改连接结构。

## 5. 工程与产物

最终聚合器保留原始 screen label，并另写 `adjudicated_outcome` 与
`final_evidence_class`，避免把续跑后的饱和点继续画成候选。核心产物：

- `results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/lc5v2p1_joint_phase_map/phase_map.json`
- `results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/lc5v2p1_joint_phase_map/phase_map.csv`
- `results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/lc5v2p1_joint_phase_map/figures/lc5v2p1_joint_phase_map.png`
- `results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/lc5v2p1_candidate_extension_tau3000_gamma0060/summary.json`
- `results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/lc5v2p1_right_censor_extension_tau15000_gamma0003/summary.json`
- `results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/DONE_LC5V2P1_CANDIDATE_EXTENSION_TAU3000_GAMMA0060.json`
- `results/topic4_sef_hfo/fcxr_lc5v2_finite_episode/DONE_LC5V2P1_RIGHT_CENSOR_EXTENSION_TAU15000_GAMMA0003.json`

所有长运行已结束，无残留 LC5v2.1 仿真进程。本轮不再补格、不再续跑候选、不解锁 recovery、M、
multi-seed、eigenmode 或论文主张。

代码收口后 LC3--LC5 全链回归为 `420 passed`，`git diff --check` 干净；聚合器的续跑合并与右删失
分层另有定向回归测试。

## 6. 对下一代设计的边界

下一轮若讨论修改连接，必须把问题写成新的因果比较：当前失败究竟来自 H/D 正反馈过陡、缺少局部
抑制塑形（例如 Mexican-hat/WTA 型竞争），还是 U 的作用通道没有削弱真正承重的 carrier mode。
不得在本相图后继续加零散 `tau/Gamma` 点来替代结构假设。
