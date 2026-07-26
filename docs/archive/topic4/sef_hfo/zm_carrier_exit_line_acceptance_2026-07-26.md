# Z/M carrier–exit 机制线验收（2026-07-26）

## 0. 一句话判决

**验收既有线路为一个边界清楚、工程证据完整的机制性 NO-GO；不验收为 ictal
lifecycle。**

这条线证明了：

1. 裸 Z/M 能把原始各向异性 SNN 从 returning interictal events 推到 runaway；
2. \(S_G\) 能 containment，但产物是由慢反馈反复 reset 的低占空比 burst train，
   不是持续的 ictal carrier；
3. persistence-based \(H\) 在这个 bursty 状态上没有稳定可积累的驱动，不能完成退出；
4. 把抑制池局部化，在已测试的双池 rate-field 上也不能把同步振荡横向失稳成接力载体。

因此，当前缺口不是“再调一个终止器”，而是需要先分开验证两个动力学对象：

\[
\boxed{
\text{最小动力学子系统中的 stable/metastable carrier}
\quad+\quad
\text{与 entry 区分的 joint offset geometry}
}
\]

下一步设计已修订为：
`docs/superpowers/specs/2026-07-26-topic4-zm-minimal-carrier-branch-decision-design.md`。

---

## 1. 本次验收覆盖什么

### 1.1 正确 Z/M SNN 迁移

- substrate：E1146 `twoend_equal`，二维各向异性 E/I SNN，\(N=40000\)
  （\(N_E=32000,N_I=8000\)）；
- 慢变量：`use_z=True, use_m=True, use_qI=False, use_gK=False`；
- lockpoint：`zA_q75_tz5000__mA0p001_tau500`；
- Z/M 移植通过逐位 spike parity；
- 本线不修改 E→E 拓扑，与并行 E→E 机制线独立。

详细实现与种子 1 诊断：
`docs/archive/topic4/sef_hfo/m4_snn_native_exit_execution_2026-07-21.md`
顶部“Z/M-native rebuild”。

### 1.2 carrier gate

三条臂：

- `bare`：Z/M；
- `sg`：Z/M+\(S_G\)；
- `sgh`：Z/M+\(S_G\)+\(H\)。

源空间和 E1146 虚拟 SEEG 使用 revised carrier protocol v2.1 判定，结果归档：
`docs/archive/topic4/sef_hfo/zm_ictal_carrier_gate_2026-07-24.md`。

### 1.3 局部抑制空间秩 screen

在 matched mean-field inhibition budget 下，比较全局、局部、混合抑制池。对预注册
五档工作点和 \(n=32\) 网格的全部 513 个独立非 DC 模态计算横向 Floquet 增长率。
结果归档：
`docs/archive/topic4/sef_hfo/zm_reduced_field_screen_2026-07-25.md`。

### 1.4 不纳入验收的历史线

旧 q_I+\(S_G\)+p/H 沙盒不是当前 Z/M 衬底。它只保留工程方法与机制提示，不提供
Z/M lifecycle 证据。其“持续态可被 H 终止”“没有 interictal attractor”等结论不得
迁移到 Z/M。

## 2. 科学结果

### 2.1 Entry：成立到 runaway，但不是 lifecycle

裸 Z/M 在 seed 1 上出现：

\[
\text{returning interictal events}
\rightarrow\text{事件加密/空间招募}
\rightarrow\text{runaway}.
\]

这说明当前 Z/M 的 onset/recruitment 路线是可工作的。它没有证明一个 bounded ictal
attractor，因为仿真终点仍是 runaway。

### 2.2 Containment：成立，但产物不是 ictal carrier

\(S_G\) 把 runaway 压成一个长期 recurrent focal burst train：

- 全场均值低；
- 核区瞬时峰值很高；
- burst 之间活动回落；
- \(z_\mathrm{core}\) 继续耗竭；
- \(S_G\) 随 burst 升高、在间隙塌落。

revised v2.1 观测门的关键结论：

- `sg` 源空间没有持续 \(\ge100\) ms 的 onset；
- 活跃虚拟触点可有较高 occupancy，但最长持续簇约 0.6 s；
- 0 个触点同时满足 occupancy \(\ge0.8\) 且持续 \(\ge2\) s；
- verdict = `fail_hfo_like_train`。

因此它应解释为：

\[
\text{local burst}
\rightarrow S_G/M\text{ reset}
\rightarrow\text{gap}
\rightarrow\text{re-ignition},
\]

即 slow-feedback-supported relaxation burst train。高峰值或跨频带功率不能单独证明
broadband ictal carrier；尖锐脉冲列本身可产生 harmonic pseudo-broadband。

### 2.3 Exit：H 的负结果是机制错配，不是生命周期否定

在当前 burst train 上，三种 \(H\) sensor 的 \(H_{\max}\le0.035\)。burst 之间的低活动
使持续性场被时间/空间平均稀释，因此 persistence-based \(H\) 无法建立足够的慢记忆。

安全结论：

- 当前 \(H\) 设计不适合由低占空比 burst train 驱动；
- 尚未测试一个“已形成的 frozen ictal carrier”能否被 \(H\) 或其他 exit coordinate
  终止；
- 不能据此声称 Z/M 不可能恢复，因为系统还没有先到达一个合格 carrier。

### 2.4 局部抑制：空间秩变化不足以产生 carrier

reduced-field screen 的完整 513-mode 结果：

- 每一预注册工作点、每一条有效抑制臂、每一非 DC 模态增长率均为负；
- 目标 `global_stable_local_unstable` 窗口为空；
- verdict = `both_stable`；
- 局部臂最接近 0 的模态总是盒子允许的最小波数，其接近中性方向是 \(k\to0\) 的结构
  极限，不是“离分岔只差一点”；
- 因此不迁移到 SNN 是正确 stop decision。

这个结果只排除“在该双池降阶工作点上，局部化抑制会让同步轨道自发横向失稳”这一条
机制。它不普遍否定空间局部抑制，也不直接代表原始 Z/M SNN。

## 3. 更深一层的动力学解释

当前变量形式上不止一个，但可能主要沿同一有效兴奋性方向作用：

\[
\eta_{\mathrm{eff}}
=\eta_0+c_z(1-z)-c_m m-c_GS_G-c_HH.
\]

当 \(z,m,S_G,H\) 都由相近的活动统计量驱动，又主要通过同一增益方向反馈时，慢系统
可以功能上接近 rank-1：

- \(Z\) 长期向高兴奋性推进；
- \(M/S_G/H\) 在 burst 时暂时向低兴奋性拉回；
- 活动一降，负反馈的 drive 同时消失；
- 低 \(z\) 再次点燃网络。

问题不是“activity-dependent 变量不能共存”。真正需要检验的是：

1. 慢变量冻结时，快子系统是否已经拥有 bounded carrier；
2. entry 与 exit 是否跨越不同边界；
3. exit 变量是否积分不同 observable，并作用于不同动力学/空间方向。

因此下一阶段的合理角色分工是：

\[
\boxed{
Z=\text{entry/recruitment},\qquad
\text{fast E/I}=\text{carrier},\qquad
M\text{ 或 }P/A=\text{exit}.
}
\]

\(M\) 仍优先测试，但它可能更适合 burst shaping、局部 refractory wake 或
tonic–clonic 转换；是否能承担 offset 必须由 frozen-\(M\) continuation 决定。

## 4. 当前可以写与不能写

### 可以写

- Z/M 能在原始各向异性 SNN 上组织自发的 interictal-to-runaway entry。
- \(S_G\) 能抑制 runaway，但把它整形成 recurrent focal burst train，不是通过正式门
  的持续 ictal carrier。
- persistence-based \(H\) 与该低占空比载体存在驱动错配。
- 在已测试的 matched-budget 双池降阶场中，局部抑制没有产生横向失稳，完整可表示
  波数谱均稳定。
- 当前瓶颈被定位为“carrier 是否存在”先于“termination 怎么实现”。

### 不能写

- 已经存在可控 ictal lifecycle；
- `sg` 或 `sgh` 是 seizure attractor、limit cycle 或真实 tonic–clonic seizure；
- 高峰值/跨频带能量等于持续 broadband ictal activity；
- Z/M substrate 没有可恢复 interictal basin；
- 局部抑制普遍不能产生时空间发作模式；
- 再加一个 H/P/A 就一定能终止；
- reduced field 的负结果等于 SNN 层的普遍定理。

## 5. 工程验收

### 5.1 通过

- Z/M parity 与 slow-off baseline guard 保持；
- carrier protocol 的已知 P0/P1 问题已修到 v2.1：
  onset 重估、真实四维 separation、plateau/tail 判据、A7/A8 fixture、provenance；
- reduced-field screen 完成 fail-closed lock、参数编码 cache、完整 513-mode closure；
- 105 个相关测试在既有交付时通过；
- 既有长仿真无残留进程，资源记录未出现 OOM/swap 增长；
- 线路没有修改 E→E，符合与并行路线独立的约束。

### 5.2 本轮拒收的未提交原型

旧版 excitable-wave Phase-0 原型把 kick offset 的第一个采样点计入“刺激后峰值”，会把
刺激撤除后的被动衰减误标为 `excitable`。其候选响应峰值发生在撤除瞬间，随后单调衰减，
且随 kick 幅度呈 graded growth，没有证明 regenerative/all-or-none excursion。

处理：

- 不纳入结果；
- 不写进 archive 作为科学证据；
- 不提交其代码/测试；
- 其设计只作为 conditional Branch F 的反面检查表保留在修订 spec §7。

## 6. 验收 verdict

| 层级 | verdict | 含义 |
|---|---|---|
| Z/M entry | `supported_to_runaway_seed1` | onset 链路存在，终点仍失控 |
| \(S_G\) containment | `bounded_hfo_like_relaxation_train` | 有界，但不是持续 carrier |
| H exit | `driver_mismatch_no_exit` | 当前 burst train 喂不饱 persistence memory |
| local-inhibition field | `both_stable_no_migration` | 没有横向失稳带 |
| full lifecycle | `not_established` | carrier、offset、recovery 均未闭合 |

**总 verdict：`ACCEPT_BOUNDED_MECHANISM_NO_GO`。**

这不是放弃 Z/M，而是把下一步从“继续给 burst train 调终止器”纠正为“先确认快子系统
是否有 carrier，再决定 exit 坐标”。

## 7. 下一步决策树

1. 在 canonical SNN 上加入 parity-locked checkpoint，做 exact-state capture。
2. 以自然 fast phase × paired future noise，比较 E/I、E/I+\(M\)、E/I+\(S_G\)、
   E/I+\(M+S_G\) 的 probabilistic stable/metastable carrier。
3. 若 visited states 无 carrier，先审计 coarse + full-field/pathology-axis slow neighbourhood：
   邻域有 carrier → Branch T；邻域仍无 → Branch F。
4. 若有 carrier，先做 slow-coordinate functional rank 与 trajectory-conditioned modal audit。
5. 分别估计 \(Z\)-entry 和 existing slow-coordinate offset：
   \(M\) alone、\(M+S_G\)、\(M+Z\)-recovery。
6. 只有所有现有 offset 坐标均不足，才离线比较 local cumulative load \(P\) 与
   recruited-area/flux \(A\)，并另写 spec 实现其中一条。
7. 只有完成 carrier → offset → refractory/recovery → returning interictal events 后，才恢复
   Figure-5 lifecycle 与三条下游 workflow。
