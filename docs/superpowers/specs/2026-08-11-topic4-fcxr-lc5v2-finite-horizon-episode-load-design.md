# FCXR-LC5v2 — finite-horizon per-cell episode-load interception

状态：**DESIGN LOCK — CORE EXPLORATION**
日期：2026-08-11

## 1. 唯一科学问题

在当前可由 returning IED 自然点火、但随后继续升级到 refractory plateau 的
`Z -> H` SNN 中，一个始终逐细胞、只读取自身 spike history 的 episode-load current，
能否在 onset 后及时建立，把升级轨迹改造成：

1. 非饱和的受控高态；或
2. 驻留 0.5--5 s 后自主 offset、并保留至少 2 s post-offset memory 的有限 excursion。

本轮不要求 pump-off substrate 自己先具有稳定高分支。`U_i` 允许同时承担 containment 与
episode-level offset；这正是完整正反馈/慢负反馈系统的被测对象。

## 2. 冻结与禁止

冻结：E1146 connectivity、两个低阈值 core、RC1 recurrent saturation、dynamic Z、dynamic H、
`X=1`、`M=0`、connection seed 1、noise seed 401、`dt=0.05 ms`、`h_U=3`。

方程不改：

\[
\Phi(u_i)=\frac{u_i^3}{1+u_i^3},\qquad
u_i(t+dt)=\max\{0,u_i+a_U N_i^{spk}-dt\,\Phi(u_i)/\tau_U\},
\]

\[
I_{U,i}=I_{max}[\Phi(u_i)-p_{0,i}]_+.
\]

禁止：population sensor、spatial mask/shared load、删病理细胞、重调 Z/H、打开 M、修改
clearance 方程、使用 late refractory plateau 定标、先跑完整 3x3。

## 3. 数据源与有限时段标定

唯一源为已封存的 U1a bundle；不得重跑或改写：

`results/topic4_sef_hfo/fcxr_lc5_episode_pump/u1_capture/`

自然 onset 为 `t_on=11 s`。锁定：

- baseline window `W_B=[t_on-4s,t_on)=[7,11)s`；
- early-episode window `W_E=[t_on+1s,t_on+3s)=[12,14)s`。

对 `tau_U in {3,8,15}s`，从 `u_i(0)=0` 离线重放完整 sparse spike stream。用确定性二分求
`a_U`，使：

\[
median_{i,t\in W_E}\Phi(u_i)=0.5.
\]

这里 0.5 是有限时段 episode-load 标尺，不是稳态平衡点。旧的 `a_U r_i tau_U<1` 门退出
LC5v2，不得继续阻断前向实验。

离线 calibration 使用锁定的 `dt_cal=1 ms`、每 `5 ms` 取一次联合 cell-time activation
样本；同一 1 ms bin 内先按上一 bin 末负荷计算清除，再加入该 bin 的 spike count。该近似只用于
离线尺度与 onset load 初始化；必须在合成轨迹和 U1 首秒上与 `dt=0.05 ms` 原方程比较并报告误差。
正式 U2 membrane/load dynamics 仍使用引擎的 `dt=0.05 ms`，不得把 calibration 步长带进 SNN。

最终 `a_U` 锁定后，用同一 replay 计算：

\[
p_{0,i}=Shrink\left(mean_{t\in W_B}\Phi(u_i(t))\right).
\]

shrinkage 规则必须在查看 U2 outcome 前锁定；第一版复用既有 pump shrinkage 实现。保存 onset、
onset+1 s、onset+4 s 的 `u_i`。禁止先生成 load state 后再改 `a_U`。

## 4. 有限时段剂量

对 `W_E` 使用逐细胞时间积分：

\[
\Gamma_U^{dose}=
\frac{median_i\int_{W_E} I_{U,i}(t)dt}
{median_i\int_{W_E} I_{EE,i}^{force}(t)dt}.
\]

按下式解析锁 `Imax`：

\[
I_{max}=\Gamma_U
\frac{median_i\int_{W_E} I_{EE,i}^{force}dt}
{median_i\int_{W_E}[\Phi(u_i)-p_{0,i}]_+dt}.
\]

首轮 `Gamma_U in {0,0.10,0.25,0.40}`，`tau_U=8 s`。所有分母、支持集、窗和 hash 必须落盘。

## 5. U2a：顺序强度探索

从原始 onset exact state 分叉，将离线 replay 得到的 onset `u_i` 附着到 pump-enabled template；
Z/H 动态，X=1，M=0，未来外源输入由 checkpoint RNG 连续产生。每臂目标 7 s：

1. control `Gamma=0`；
2. `Gamma=0.10`；
3. `Gamma=0.25`；
4. `Gamma=0.40`。

先运行 control 与 `Gamma=0.25`。只有二者表现出可解释分离，才补 0.10/0.40。允许一次单边扩展：

- 0.40 仍等同 control saturation，且 achieved dose 单调、数值安全：追加 0.60；
- 0.10 已 <0.5 s 立即压灭：追加 0.05。

两边不得同时扩展。

## 6. U2 标签与核心 gate

- `ESCALATING_SATURATION`：继续进入既有 refractory/saturation class；
- `CONTAINED_HIGH_NO_OFFSET`：阻止饱和并保持高态，但 7 s 内未 offset；
- `FINITE_EXCURSION_OFFSET`：高态 0.5--5 s，随后低于 interictal upper band >=2 s；
- `OFFSET_WITH_REBOUND`：offset 后再次进入；
- `IMMEDIATE_SUPPRESSION`：<0.5 s 压灭；
- `BURST_SILENCE_LOOP`；
- `NUMERICAL_FAIL`。

`FINITE_EXCURSION_OFFSET` 还要求：无 saturation；`I_U` 上升先于 rate 下降；随后
`I_EE_force` 与 H source/H 下降；offset 后 `I_U` 仍高于 baseline。

`CONTAINED_HIGH_NO_OFFSET` 是核心正结果：证明 U 已把 escalating source 变成 bounded carrier，
不按机制阴性处理。

## 7. 后续只由结果解锁

若 U2a 出现 contained/finite 点，才在 1--2 个 Gamma 上比较 `tau_U={3,8,15}s`。只对 primary
补一对 dynamic-D / frozen-D 短诊断。late snapshot 是 stress test，不阻断候选。

U3 只在出现 finite excursion 后启动；从 t=0 `u=0`、U 始终在线，按 5--10 s simulation chunk
顺序续跑。每个 chunk 使用 rolling exact checkpoint、连续 RNG、atomic ledger。满足完整 lifecycle
后可提前结束；onset 后 15--20 s 仍 sustained/saturation 可提前失败。

## 8. 资源与授权

40k 严格单 worker；线程数全 1；每次提交前检查 sibling、MemAvailable 和 swap。swap 相对该 stage
基线 +256 MiB 停止新提交，+512 MiB 保存 rolling checkpoint 后终止当前 worker。每个 U2 7 s arm
使用实测 cost 生成 wall guard，初始上限 5 h，可由首臂实测更新；不得把整个 U3 塞进 12 h 单任务。

所有长任务 `setsid nohup`，stage-scoped flock，PID 与 RUNNING/DONE/FAILED/RESOURCE_STOP sentinel。

## 9. 当前授权边界

本锁授权：finite calibration、候选锁、U2a control 与首个 `Gamma=0.25`，以及在核心分离出现后的
0.10/0.40 顺序补点。未出现核心分离前不做多 seed、形态、eigenmode 或论文图。
