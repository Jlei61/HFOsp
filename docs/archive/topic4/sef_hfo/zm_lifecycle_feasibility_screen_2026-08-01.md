# Z/M 生命周期设计诊断复核（2026-08-01）

**状态：** `ACCEPT_AS_DESIGN_DIAGNOSTICS_ONLY`
**拒绝的旧语义：** `infeasible_registered_substrate` / “4/4 算术证明不可能”

## 1. 一句话判断

这组计算正确地暴露了四个值得在新 spec 中显式处理的量级风险，但它们不是
ictal carrier 或 lifecycle 的数学必要条件，不能汇总成“已注册 Z/M 衬底不可能
建立 lifecycle”。本次复核保留数值、provenance 和回归测试，把顶层结果改成
`diagnostic_risks_present`；真正的 GO/NO-GO 仍必须来自 baseline preservation、
frozen-state carrier、扰动返回和 observation gate。

## 2. 为什么原 4/4 NO-GO 不能验收

### 2.1 (m) 的 0.25 mV 是量级诊断，不是 termination 必要条件

对当前

\[
\dot m=-m/\tau_m+\sum_k\delta(t-t_k),\qquad I_M=\eta_m m,
\]

用 (1/\tau_{ref}) 给 firing-rate ceiling，确实得到

\[
I_{M,max}=\eta_m\tau_m/\tau_{ref}=0.25\ \mathrm{mV}.
\]

它只占 reset-to-threshold gap 6.5 mV 的 3.85%，说明当前 (m) 对**单细胞硬静默**
的权限很小。但是 recurrent network 可以在离分岔很近时被远小于 6.5 mV 的电流
推过稳定性边界。因此“必须跨过完整 reset-to-threshold gap”不是必要条件。
正确用途是：在可信 carrier 建立后，用 frozen-state continuation / clamp-(m) / gain
ablation 测量 (m) 是否足以移动 offset boundary。

### 2.2 (z) 是 entry latch，不等于完整 lifecycle 不可逆

在 elevated inhibitory-current state 中 (z_\infty=0)，所以 (z) 自己不会在 ictus
期间折返，这个方向判断是对的。但新科学分工本来就是：(z) 负责 entry，(m) 或另一个
独立负反馈负责 offset；activity 下降后 (I_I<I_{th,EI})，(z) 才恢复。于是该结果只能写：

> (z) 不能同时承担 entry 和 sole-offset coordinate。

不能写“entry 在完整系统中不可逆”。

### 2.3 \(\tau_S=80\) ms 不限制它只能终止 80 ms 状态

(S_G) 在活动存在时持续被驱动。一个 80 ms feedback loop 可以参与稳定或破坏一个
持续数秒的 attractor；其时间常数不是“可终止状态的最大持续时间”。正确的结论是：

> 活动撤除后 (S_G) 的 post-activity memory 很短，单独承担 postictal hold 的能力弱。

此前真实仿真已经显示它把 runaway 整形成 burst train，这一经验结果仍然成立；但原因不能
被提升成普适的 \(\tau_S/T_{seizure}\ge1\) 必要门。

### 2.4 439 Hz 限制向上 headroom，但不排除向下 population modulation

Phase C 核心 435–443 Hz 相对 500 Hz refractory ceiling 的 occupancy 为 0.87–0.89，
平均 ISI 只比 2 ms 硬地板多约 0.28 ms。它清楚说明当前 tonic branch 很靠近高率边界，
不适合直接叫真实 ictal carrier。但 20% population modulation 可以通过 firing-rate trough、
部分神经元退出、空间 phase staggering 或转入较低均值 carrier 实现，不需要继续向 500 Hz
上方摆动。因此该量是 saturation-risk diagnostic，不是 carrier non-existence proof。

## 3. 保留的四项定量诊断

| 诊断 | 当前值 | 安全解释 |
|---|---:|---|
| (m) refractory-capped current scale | 0.25 mV；reset-gap ratio 0.0385 | 当前参数下 intrinsic adaptation 较弱；是否足以跨网络 offset boundary 未测 |
| (z) ictal target direction | (0.75\rightarrow0.30\rightarrow0) | (z) 是 entry latch，不能独自终止；允许另一个变量负责 offset |
| (S_G) post-drive decay | 80 ms；baseline IED 间隔约 567 ms | 能在 IED 间复位；缺少长 postictal memory，但可参与 driven fast dynamics |
| Phase-C core refractory occupancy | median 0.878；ISI headroom 0.277 ms | 当前 branch 高率且低 headroom；新 carrier 应离开该 branch，但 carrier 并未被解析排除 |

机器输出：
`results/topic4_sef_hfo/zm_lifecycle_feasibility/feasibility_verdict.json`。
版本：`topic4_lifecycle_design_diagnostics_v2_2026-08-01`。

顶层只允许：

- `diagnostic_risks_present`；
- `no_diagnostic_flags`。

禁止输出 `infeasible`、`carrier_impossible` 或 lifecycle scientific GO/NO-GO。

## 4. Phase C 与 Phase D 的最终验收关系

### Phase C

验收为 `ACCEPT_PHASEC_POST_RESULT_FUTILITY_STOP`：C0 153/153 完成；C1 seed-1
primary 59/60 全为 corrected `tonic_non_AI`，原 registered primary GO 逻辑不可达。
这排除了“继续扩大相同 frozen Z/M morphology 网格”的价值，但不是三-seed bounded
negative，也不是 carrier non-existence proof。

### Phase D

验收为 `NO_GO_baseline_calibration_failed_zero_spike_dominance`：conductance replacement
及 dynamic-threshold 基础设施已实现，状态迁移和 default-off parity 通过；但 conductance
membrane 从 (t=0) 消除了原 returning IED generator，B/C/D carrier arms 未获准运行。
该结果只否定已注册 reversals/anchor/scale 下的全程 conductance replacement，不否定
conductance inhibition，也没有检验 dynamic threshold 本身。

## 5. 对下一版设计的约束

1. 不再扩大 Phase-C slow morphology atlas。
2. 不在当前 tonic branch 上提前做 termination/stimulation。
3. 下一轮只测试一个尚未真正运行过、default-off parity 已通过且可直接接受 baseline
   preservation 检验的 fast mechanism：current-based Z/M substrate 上的 per-neuron
   E-only dynamic-threshold feedback。
4. (z,m) 在 carrier existence stage 冻结；(z) 的 entry 与 (m) 的 offset 只在后续
   独立 gates 中释放。
5. primary carrier gate 是持续、bounded、spatially structured 且通过 virtual-SEEG
   observation contract；AI/tonic/refractory 分类只作 secondary mechanism endpoint。
6. 任何新机制先过真实 dynamic interictal baseline preservation；结构恒等或量级诊断
   不能代替该行为门。

## 6. 验证

- `src/topic4_lifecycle_feasibility.py` 已降级为 quantitative diagnostics；
- runner 不再生成“最大可终止持续时间”等错误字段；
- tests 明确断言 risk flag 不得变成 infeasibility proof；
- 本轮复核不启动新的 SNN 仿真。
