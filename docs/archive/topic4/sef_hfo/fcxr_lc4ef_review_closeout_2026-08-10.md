# FCXR-LC4e/f 审阅收口：当前 X 实现阴性，不把空间覆盖写成唯一原因

日期：2026-08-10

状态：**LC4e/f CLOSED；当前 cell-local `y→X` 与 closed-loop shared-execution 实现停止。完整 lifecycle 未建立。**

## 1. 最终科学判决

LC4f 是一个有效的预注册阴性：40k E1146 网络在无 kick 条件下经历 29 个 returning IED，于 11 s 自然进入高态，但在预设的 onset 后 1–5 s offset 窗内没有自主退出，高态持续到 22 s 记录边界。

安全结论只有：

> 当前 cell-local `y→X` 以及它的 closed-loop shared-execution 变体，在自然闭环中没有表现出足够的终止权威。

不能写成：

> X 失败的唯一原因是没有覆盖全场；只要把全场都压到 `X=0.380` 就一定终止。

## 2. 已经成立的四点

1. **自然进入成立。** 29 个 returning IED 后无外部刺激进入，支持 repeated IED → `Z` depletion / `D=1-z` increase → recruitment escalation。
2. **late-bout 深度不能直接搬到自然轨迹。** 自然轨迹的群体平均 X 最低 0.488、末端 0.501；旧 late-bout fork 的 0.380 只能称 state-conditioned reference。
3. **空间追逐确实存在。** LC4e 的 shared execution 消除了 local 臂的“核心塌陷、轴外残留”形态。
4. **完整生命周期仍未完成。** 无 offset，因而没有 postictal protection、Z recovery 或 returning-IED recovery 的证据。

## 3. 为什么不把原因归到单一空间覆盖

当前有四个耦合因素没有被拆开：

- cell-local X 与 surviving carrier support 的空间分布不匹配；
- shared execution 压低活动后也压低自己的 `y` 输入，峰值执行剂量从 51.4 降到 20.3；
- 发作中 `D` 从约 0.066 升到 0.54、`H` 从约 0.29 升到 2.12，终止面随慢状态移动；
- population-mean X 未必是承载 carrier 的 recurrent mode 所看到的有效 X。

更相关的诊断量是当前 recurrent carrier 的加权有效 relay：

\[
g_{EE}^{(v)}=
\frac{v^T W_{EE}\operatorname{diag}(x)v}{v^T W_{EE}v},
\]

而不是区域均值或全体均值本身。本轮没有保存足以重建该量的全状态和 mode，因此不补算一个伪精确数字。

## 4. LC4e/f 图和措辞的订正

- LC4e 不再称“matched spatial sharing”；只称 **closed-loop spatial sharing**。两臂共享同一初始合同和 causal prefix，但整个发作期累计剂量不匹配。
- LC4f Panel B 的 0.380 改称 **archived late-bout reference**，不称 universal offset boundary。
- LC4f Panel D 改称 **X field 与 surviving carrier support 空间失配**，不称“缺少 population-wide coverage”。
- “never reaches offset depth”收窄为“在预注册 target window 内未达到 archived reference，也未 offset”。

## 5. X 方向以后只保留的诊断

X 暂停作为 lifecycle 主退出变量。若为了论文机制归因继续分析，只允许以下短诊断：

1. 从同一 exact onset snapshot 做累计剂量完全相同的 local / Gaussian / global yoked replay；
2. 在 onset+1 s、+4 s、late 三个真实 snapshot 上冻结 D/H，缩放 X field，估计 trajectory-conditioned offset surface；
3. 输出 `I_H`、有效 recurrent E drive 和 mode-weighted X，检查 H 是否绕开 relay depression；
4. 对 22 s 轨迹只做末端斜率/渐近值诊断，区分“太浅”与“太慢”，但不改变 1–5 s 阴性判决。

这些诊断不构成新的 lifecycle 路线，也不授权 spatial mask、global seizure sensor 或 recruited-area field。

## 6. 工程订正

后续长仿真必须把 NPZ、summary JSON、event ledger、exact state 与 RNG/noise provenance 作为一个事务：先写临时文件，完成 schema/hash 校验后 atomic rename。所有 JSON 在写出前递归转换 `np.generic → item()`、小数组 → list、`Path → str`。每 1–2 s 仿真时间写 checkpoint 与增量 ledger，避免再次出现仿真完成而 summary 序列化失败。

## 7. 下一路线

不再把空间 sharing/mask 作为主机制；连接和实际 spike history 是唯一空间组织来源。生命周期退出改由独立的逐细胞 episode-load recovery coordinate `U_i` 承担，详见：

- `docs/superpowers/specs/2026-08-10-topic4-fcxr-lc5-per-cell-episode-pump-design.md`
- `docs/superpowers/plans/2026-08-10-topic4-fcxr-lc5-per-cell-episode-pump.md`
