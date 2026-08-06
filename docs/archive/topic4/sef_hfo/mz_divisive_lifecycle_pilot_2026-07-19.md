# MZ divisive lifecycle：独立机制线最终报告

日期：2026-07-19--20

分支：`codex/topic4-mz-divisive-lifecycle`
状态：v1/v2/v3 已执行；exact composition scoped no-go；三条下游 workflow 不释放

## 1. 一句话结论

本线第一次把原来的 delayed runaway 改造成了**无 kick、自主进入、约 5 Hz 的有限窗 recruited
bursting**，这是一个真实的动力学 opening；但它不是 settled ictal state，也没有自终止。随后锁定的线性 M
ladder 全部在进入前就把高态拆碎或压住，没有一格完成“先进入，再退出，再回到间期”。

因此当前最安全判决是：

> high-state-gated recurrent-E slow divisor 能提供 containment 和 bursting transient；原线性 M 仍然是
> entry brake，不是 exit variable。这个精确组合结束，不再细调慢负反馈。

## 2. 与并行 conductance 线的非冲突边界

本线保留 current-based membrane，只测试：

\[
I^{net}_{E,i}=I^{ff}_{E,i}+
\frac{I^{rec}_{E,i}}{1+\alpha_fS_G+\alpha_TT_G}
-z_iI^I_i-\eta_m m_i,
\]

其中 `T_G` 只在持续广泛招募后启动，是一个抽象的 global slow gain brake；它不是 GABA conductance、
sAHP、pump 或 Abbott/Liou reproduction。

并行 `topic4-mz-conductance` 已把下一节点收紧为：先做 E-cell full AMPA/GABA conductance 的 frozen
fast-branch gate，再考虑 persistence-gated presynaptic E→E relay `x_j`。它明确不使用本线的
`1/(1+alpha_f*S_G+alpha_T*T_G)`。截至本报告收口，它的 Stage 0A engine/parity/re-bless 已 ACCEPT，
Stage 0B `c_E` workpoint 正在运行；诊断初见 `c_E=0.85` 近静息、`c_E=1.0` 过热，正式 workpoint 尚未判定，
fast high-branch gate 也尚未开始。两条线没有重复，也不应现在混合。

## 3. v1：fast recurrent divisor clean no-go

`S_G` observer 能区分 slow-off 与 Z-runaway，因此 negative 不是 sensor 没看见招募。可是固定
`p={1,3}`、`alpha_G` 边界与 `tau_S={40,80,120,200} ms` 后，只出现 amplified IED trains 或 delayed
runaway；没有 settled m-off high state。

两个最接近边界的 20 s 结果：

| cell | runaway | final rolling-1 s | last-3 s slope |
|---|---:|---:|---:|
| `p=1, alpha=2` | 14639.7 ms | 90.4 Hz | +16.1 Hz/s |
| `p=3, alpha=1` | 12979.2 ms | 103.9 Hz | +18.6 Hz/s |

所以 v1 只是延迟失控，不允许直接加 M。证据：
`results/topic4_sef_hfo/mz_divisive_lifecycle/runs/20260719T143832.483454Z_6ce230e_b7055b17e8_long_check/summary.json`。

## 4. v2：slow-gated divisor 打开有限窗 bursting，但不闭环

v2 新增：

\[
U_T(A_G)=\frac{[A_G-A_0]_+^4}{A_{50}^4+[A_G-A_0]_+^4},\qquad
\tau_T\dot T_G=-T_G+U_T(A_G).
\]

`A0=0.15` 高于 slow-off sensor ceiling，所以普通 IED 不驱动 `T_G`。锁定 5-cell screen 的结果：

| cell | 20 s outcome | recruited | rhythm | return |
|---|---|---:|---:|---:|
| `alpha_T=0` | delayed runaway | -- | -- | -- |
| `alpha_T=4, tau_T=750 ms` | finite-window bursting | 6.14 s | 5.05 Hz | no |
| `alpha_T=4, tau_T=2000 ms` | finite-window bursting | 8.30 s | 4.21 Hz | no |
| `alpha_T=6, tau_T=750 ms` | bursting then fragmented high tail | 1.38 s | 5.06 Hz | no |
| `alpha_T=6, tau_T=2000 ms` | bursting then fragmented high tail | 3.10 s | 4.19 Hz | no |

最佳格有明确节律，不是 classifier 幻觉；严格谱峰功率比 0.315。但其最后 3 s 仍有
`z_mean=-0.0218/s`、`TG=+0.0378/s`、`AG=+0.00677/s` 漂移。接近零的 rate slope 是两股慢漂移暂时
抵消，不能写成稳定支、attractor 或 limit cycle。

证据：

- `results/topic4_sef_hfo/mz_divisive_lifecycle/runs/20260719T162035.230785Z_6ce230e_e1acc35592_slow_gate/summary.json`
- `results/topic4_sef_hfo/mz_divisive_lifecycle/runs/20260719T162035.230785Z_6ce230e_e1acc35592_slow_gate/strict_audit.json`

## 5. v3：原线性 M 改变 entry，不产生 exit

v3 固定 v2 最佳格，只复用 simulation 前已经注册的
`eta_m=[0,0.00186,0.00373,0.00745,0.01118,0.01863]`、`tau_m=2 s`，seed 1，全部完整跑 25 s。

| eta_m | strict state | 最长 recruited shoulder | final 2 s mean |
|---:|---|---:|---:|
| 0 | finite-window recruited bursting | 11.14 s，延伸到 endpoint | 78.44 Hz |
| 0.00186 | no recruited macrostate | 500 ms | 21.75 Hz |
| 0.00373 | no recruited macrostate | 220 ms | 13.90 Hz |
| 0.00745 | no recruited macrostate | 190 ms | 9.19 Hz |
| 0.01118 | no recruited macrostate | 120 ms | 6.37 Hz |
| 0.01863 | no recruited macrostate | 0 ms | 5.33 Hz |

m-off 到 25 s 仍未 operational runaway，但 endpoint rate 为 79.3 Hz，末段仍 `+1.63 Hz/s`；同时
`z_mean=-0.0175/s`、`TG=+0.0506/s`，因此只把 v2 的 bounded window 延长到了 25 s，仍不是 settled
state。所有 M-on cells 都没有跨过预注册的 1 s recruited-state gate：最弱 M 只留下 500 ms shoulder，
更强 M 逐渐变成 shorter fragments / elevated trains / prevention。

这排除了“原线性 M 在 v2 高态形成后自然承担 exit”的解释。它从 t=0 就累积，先改变了 onset corridor；没有
观察到 `entry -> M build-up -> exit`，所以 seeds 3/4、clamped-M 与 retrigger 按 stop rule 不启动。

证据：

- `results/topic4_sef_hfo/mz_divisive_lifecycle/runs/20260719T172358.336529Z_6ce230e_80a127d772_slow_gate_m/summary.json`
- `results/topic4_sef_hfo/mz_divisive_lifecycle/runs/20260719T172358.336529Z_6ce230e_80a127d772_slow_gate_m/strict_audit.json`
- 启动时原始 config：同目录 `launch_config_snapshot.yaml`，SHA256 前 12 位 `9487232751d3`

## 6. 判据审计与修复

旧在线 classifier 有两个会阻断正结论的问题：

1. return 使用 `baseline+5 Hz`，平坦 4 Hz 尾巴也可能被当成恢复；
2. 100 ms gap/250 ms episode 规则会漏短 burst train，而谱峰功率比虽计算却没进入 bursting gate。

新增 strict post-hoc 合同：

- 250 ms envelope 定义 recruited macrostate，至少持续 1 s；
- bursting 额外要求 0.5--20 Hz 谱峰功率比至少 0.10；
- return 必须落入 paired seed-1 slow-off 的 2 s rolling Q99、末端仍在该带、2 s 后重新出现短 returning
  event，且无 rebound macrostate；
- 同时导出 rate、Z/M/SG/AG/TG 的 last-3-s slope。

4 Hz clonic pulse train、高噪声 plateau、4 Hz flat false-return 与真实 returning-event tail 均已有固定测试。
旧 label 只保留为 screen descriptor，最终 verdict 来自 `strict_audit.json`。

## 7. 文献校正

Liou/Abbott 模型确实同时使用 excitatory、inhibitory 和 sAHP conductances；local Gaussian 与
distance-independent inhibition 一起进入 GABA conductance。其基础模型由短暂局部输入触发，不是天然自发；STDP
重塑后才展示 spontaneous seizure。文中 inhibition exhaustion 更接近 Z 的 entry/recruitment 角色，sAHP 主要推动
tonic-to-clonic 并加速终止；过强 sAHP 同样会使 onset failure，这与本轮 M 的 prevention 现象一致。

Jirsa/Epileptor 的关键启发也不是“慢变量越多越好”，而是 frozen fast system 先要有正常/发作动力学对象，slow
permittivity 再把轨迹带过不同 entry/exit boundaries。因此 v2 的 finite-window bursting 是有价值的 opening，但不能
替代 fast-branch/initial-condition/frozen-state 验证。

参考：Liou et al., eLife 2020, https://doi.org/10.7554/eLife.50927；Jirsa et al., Brain 2014,
https://doi.org/10.1093/brain/awu133。

## 8. 工程与资源验收

- 方程审阅未见符号、更新顺序或 I-cell 污染问题；只除 recurrent E，feed-forward E 与 I cells 不变。
- `use_SG=False`、`alpha_G=0`、active fast pool + `alpha_TG=0` parity 已覆盖；相关回归 **85 passed**。
- runner 补齐 YAML threshold/tau_S wiring、dt assertion、source hashes、dirty state、trace stride 与 BLAS=1 强制合同。
- v3 使用 6 workers；启动时可用内存 214.61 GiB，最高 worker RSS 13.16 GiB；全过程 swap 维持约
  0.638 GiB，无 OOM、NaN 或 worker failure。
- v3 启动后才补 provenance/判据字段；这些修改不改变数值积分。launch manifest + exact config snapshot 锁定
  原始积分，strict audit 作为独立后处理记录自身 source hash。

## 9. 最终 Go / No-Go

- **NO-GO**：本线继续细化 `eta_m/tau_m/T_G/alpha_T`，或再叠第三个 global slow brake。
- **NO-GO**：把 v2/v3 写成 ictal attractor、limit cycle、自终止 seizure，或恢复三条下游 workflow。
- **GO**：让独立 conductance/FCXR 线先完成正在运行的 Stage 0B workpoint；只有保留 slow-off 工作点，才进入
  frozen low/high initial-condition map。只有找到 finite high branch/orbit，才启动 persistence sensor 与 local
  relay `x_j`。
- 若 full conductance 仍只有 low 与 refractory-ceiling saturation 两端，则停止 X，回到 reduced rate/field
  continuation 设计真正能承载有限高支/振荡的 fast E-I subsystem；不要再用慢负反馈替代缺失的 fast topology。

机器可读总表：`results/topic4_sef_hfo/mz_divisive_lifecycle/pilot_summary.json`。
