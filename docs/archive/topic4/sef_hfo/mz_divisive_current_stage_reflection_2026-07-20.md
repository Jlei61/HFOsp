# 审阅结论：MZ current-based 阶段反思与 topology-first 更新路线

日期：2026-07-20
分支：`codex/topic4-mz-divisive-lifecycle`
状态：current-based v1/v2/v3 与代表性时空 capture 已收口；paper-ready current-stage diagnostic 已生成；Stage 0A 分析链正对照已 PASS；Stage 0B 原始六变量 fast topology 已限定范围 clean no-go；Stage 0C 动态除法快池打开了有限振荡线索；Stage 0D v1.1 未通过预注册窗口分类复现门，但 Stage 0E 已在 `alpha_G={15,16}` 直接解析出闭合 period-1 orbit 和强有限扰动回归；Floquet 导数的 epsilon 平台仍未解析，Stage 1/空间层仍关闭

## 1. 一句话判断

**完整 SNN 已做到“间期事件逐渐增强并自主跨入 recruited bursting”，约化快系统也已找到闭合且能吸引有限扰动的 period-1 周期；但还没有建立可靠的 Floquet 稳定性证书、entry/exit boundary，更没有由局部恢复场实现同一间期 basin 的时空闭环。**

问题不能简单归结为“电流型或 additive 方程一定不能发作”。当前 Z 是局部乘性抑制，`T_G/S_G` 是 recurrent-E 的乘性/除法增益，只有 M/候选 recovery 是加性负向输入。Stage 0E 说明 delayed mean divisive feedback 确实能创建有限快周期，所以“只是加电流”并不是准确诊断。现在真正的阻断点是：周期轨道的导数级稳定性仍受 LUT 数值底噪限制，而 entry、maintenance、exit 尚未由不同变量承担；同时 `T_G/S_G` 仍把整个空间场压缩成一个全局数，自身没有位置记忆，尚不能生成或解释前沿、penumbra 与 refractory wake。

## 2. 完成程度

> **完成度：52/100**

已经完成：

- 保留了 slow-off 的 returning interictal events；
- Z 能在无外加 kick 条件下逐步把系统推过 recruited-state 操作阈值；
- high-state-gated `T_G` 第一次把 delayed runaway 改成约 5 Hz 的有限窗自主 bursting；
- strict audit 已区分短暂 rate 平台、真正 return、谱峰与慢变量收敛；
- exact linear-M ladder 给出了有边界的 negative：M-on 改变 entry/containment，而没有终止一个已建立的 high state。
- 锁定 seed-1 的同一条 20 s 轨迹已完成 virtual-SEEG、逐神经元空间场和 source-to-sink 时空场捕获；它显示当前模型确有快速轴向 event wave，但没有显示慢速 tissue-recruitment front、wake 或 offset。
- Stage 0B 已证明原始六变量 fast block 在锁定网格上只有 low 与 `>100 Hz` saturation cliff；Stage 0C 则证明延迟的 recurrent-E mean divisive feedback 能打开 relaxation-like waveform，但尚未打开稳健 basin。
- transfer-support v1.1 已排除原 LUT 下界造成的主要数值混淆；唯一 survivor 在 confirm 与 `dt/2` 下均为约 1.665 Hz，但同点只有 1/17 个初始历史进入，因而不能称 attractor/orbit。
- Stage 0D v1.1 的固定窗口分类器给出 175 条 unresolved + 5 条 survivor，且 survivor 只集中在 `z=0.85, alpha_G=15, phase_050`；该结果没有证明开 basin，但暴露了 FFT/窗口分类对起始相位与尾窗的敏感性。
- Stage 0E 不再用 FFT：Poincare shooting 在 `alpha_G=15/16` 分别解出 `604.898/608.519 ms` 的 period-1 orbit，closure 约 `10^-15`，`dt/2` 波形残差为 `0.01495/0.01563`。四个相位、fast/pool 各 8 条有限扰动全部回归，8 次 return 后 median distance ratio 约 `10^-9`。

尚未完成：

- 轨道的 Floquet/variational 稳定性证书，以及它与 low interictal branch 之间的 entry/exit boundary；
- 不依赖记录终点的自发 offset；
- 回到同一个间期 basin 以及 early/late retrigger；
- 由动态状态变量生成的局部 nucleation、有限速 wavefront、stall/annihilation 与 wake；
- 跨 seed、空间 ablation 和三条下游 workflow 所需的稳定动力学对象。

扣分的核心不是图形“不像 seizure”，而是 entry、maintenance、exit 和 spatial propagation 仍未被拆成可验证的动力学职责。

## 3. P0 / P1 关键问题

### P0：约化周期已闭合，但稳定证书与 entry/exit 拓扑尚未闭合

**问题是什么**
完整 SNN 的 v2 最佳格虽然在 13.856 s 后进入 recruited bursting，但最后 3 s 中 `z_mean=-0.0218/s`、`T_G=+0.0378/s`、`A_G=+0.00677/s`，慢状态仍持续漂移。25 s 的 M-off 延长轨迹也继续向高率方向走。约化 Stage 0E 虽已直接解出两条闭合周期并观察到强有限扰动收缩，但有限差分 Floquet Jacobian 在三档 epsilon 下没有形成平台。

**为什么严重**
接近零的短窗 rate slope 仍不能证明完整 SNN 中有冻结 ictal object。Stage 0E 已经将“可能只是长瞬态”显著收紧为“约化系统中有闭合周期”，但尚不能用未收敛的局部导数估计声称 stable Floquet attractor，也尚未证明 Z/recovery 能把轨迹带入再带出该周期窗口。

**怎么改**
只开放一次 Stage 0F 导数级复核：在同两个点上用 smooth/exact transfer 与 variational/event-sensitivity 求 Poincare Jacobian，不再扫生物参数，不用 FFT 替代稳定性。只有导数证书与 Stage 0E 的闭合/扰动回归一致，才开放冻结 `r` continuation 去求 `r_sep(z)`、entry/exit 边界和 low-basin 共存窗口。

### P0：当前 M 把 entry 与 exit 混在一起

**问题是什么**
线性 M 从第一颗 E spike 起累积，因此正常 IED、onset 前 repeated events 和 recruited state 都驱动它。最弱非零 M 已把最长 recruited shoulder 压到 500 ms，更强 M 直接转成 prevention。

**为什么严重**
这不是“终止更强”，而是 ictal state 尚未建立就改变了进入 corridor。继续扫 `eta_m/tau_m` 无法区分 timing 错误与 exit leverage 缺失。

**怎么改**
停止 broad M grid。current branch 只保留一次 established-high-state state fork：先用完全相同轨迹建立 high state，再在匹配状态下释放或 clamp 一个 high-state-gated recovery。它只用于回答“原结构有无退出杠杆”，不作为主机制继续细调。

### P0：global scalar 不能承载空间相序

**问题是什么**
`A_G/T_G` 只保留“全网活动有多大”，丢失“活动在哪里、形状如何”。空间位置被任意置换、只要值分布相同，就得到相同 global brake；同一个 `T_G` 又均匀缩放所有 E 细胞的 recurrent excitation。

**为什么严重**
它可以做 recruited-area containment，却不能区分 core、penumbra、front ahead、recruited tissue 与 refractory wake，也不能解释同样 recruited area、不同空间形状为何有不同结局。固定 scaffold 最多决定哪里更易点火，不能替代动态传播与退出机制。

**怎么改**
把慢恢复改成局部场 `r(x,t)`，并预注册逐位置相序：`z decline -> local recruitment -> r rise -> local offset`。检验较宽抑制核能否形成 front 前方 penumbra，让局部 `r` 在 front 后方留下 wake；全局/非局部抑制若保留，只能作为 extent-dependent soft containment 候选，不能在没有局部状态和空间对照时独自承担 pattern 解释。

锁定 capture 还把“没有空间相序”收紧成了更准确的结论：操作性 onset 附近，48/48 个轴向 bin 在因果 50-ms activity 门下于 60 ms 内依次跨阈，轴向 Spearman `rho=+0.963`，按 19.66 mm 轴长折算的描述性表观速度约 326 mm/s。它是清楚的 **fast axial event sweep**，不是全场严格同时点燃；但这个速度和时间尺度对应一次快速传播事件，不是秒级局部组织从 interictal basin 切到 ictal basin 的慢 wavefront。当前结构因此不是“没有空间 pattern”，而是**只有快波的空间相位，尚无慢状态前沿与其后的恢复尾迹**。

### P1：conductance 是重要改进，但不是动力学保证

**问题是什么**
full conductance 会补上 reversal dependence、总膜电导分母与有效膜时间常数，因而可能重塑高态分支与饱和行为；但仅把 current 改成 conductance 并不自动产生双稳态、limit cycle、front 或 self-termination。

**为什么严重**
如果 frozen fast system 仍只有低支和 ceiling saturation，conductance 后继续叠慢资源也会重复当前的 prevention/plateau/runaway 三分法。

**怎么改**
并行 FCXR 线继续 bottom-up 检查 full-conductance fast branch；本线独立做 topology-first reduced field。两条线用同一组 branch/front/return 指标会合，不提前拼方程。

### P1：local wake 不能预先写成 front stall

**问题是什么**
局部 `r` 可以让已经招募的位置退出，但即使 `D_r=0`，均匀 excitable medium 仍可能形成恒速 traveling
pulse，一直跑到边界。较宽 `K_I` 也只是候选，不保证自然 stall。

**怎么改**
Stage 2 必须把 `intrinsic stall/annihilation`、`constant-speed pulse to boundary` 和
`near-synchronous ignition` 分开。第二类只能写 wake PASS / containment FAIL；只在这一类下开放一个
`K_I=(1-gamma)G_sigma+gamma U` 的 fast nonlocal-inhibition sensitivity。不存在有限有效窗口就停止，不能用
边界或患者异质性制造假终止。

### P1：数值可信不等于已进入生理工作区

**问题是什么**
Stage 0C 唯一 survivor 虽然经过 extra-fine/direct-exact 复核，但其输入 moment 约覆盖 `mu_E=-130...58 mV`、`mu_I=-64...41 mV`，E-rate peak 约 96 Hz，紧贴当前 100 Hz 分类门。

**为什么严重**
transfer 在数学上有定义、步长收敛，不代表这个 orbit 已位于 HFOsp 校准过的生理工作域。即使 Stage 0D basin 复现通过，当前也只能先称 **finite mathematical relaxation orbit**。

**怎么改**
Stage 0E 已完成 operating-envelope 审计：两周期 peak 为 `98.16/96.35 Hz`，`>80 Hz` occupancy 仅 `1.65%/1.52%`，无 `>=100 Hz` 占用，但 `mu_E` 仍约覆盖 `-130...59 mV`。Stage 0F 必须分开回答“导数是否可靠”与“工作域是否可接受”；前者通过也只能称 finite mathematical relaxation orbit，不能直接升级为 biophysical seizure state。

### P1：Z 与 inhibitory penumbra 可能互相破坏

**问题是什么**
当前 Z 由 raw postsynaptic inhibitory barrage 驱动耗竭。但 front 前方 penumbra 恰好应该是“高抑制输入、低本地放电”；如果局部和较宽程 GABA 都被同一 Z 耗竭，penumbra 可能在 front 到达前就被削弱。

**为什么严重**
这会把本来用于 spatial containment 的机制反过来变成预先招募机制，使全场快速点燃更容易。

**怎么改**
1D 启动前锁定一个小 sensitivity：Z 作用于全部 GABA，对照 Z 只作用于 local GABA/较宽程 GABA 受保护。这不能提前在 0D 中调参，也不能与并行 conductance 线的方程实现混在一起。

### P1：`W_AB` 还不是可执行数据合同

**问题是什么**
当前只写了“患者 scaffold”，尚未锁 canonical producer/artifact、subject fingerprint、矩阵方向、单位、
归一化，也未说明如何避免与 `K_E` 重复计算同一耦合。

**怎么改**
Stage 3 前必须锁定这些字段，用唯一 `kappa_W` 显式 blend，并保留 `W_AB off` 对照；在这之前 2D
patient-specific field 仍关闭。

## 4. 科学性问题

### 4.1 current-based / additive 能否产生真正切换

可以，**原则上可以**。带非线性 transfer、阈值/reset、E/I 延迟和合适 recurrent topology 的 current-based 系统一样能产生双稳态、Hopf、极限环或 excitable transition。慢变量以加性偏置、阈值移动或耦合变化进入，也都可能把轨迹推过 bifurcation。

因此当前 negative 不能写成“additive 模型不可能产生 seizure”。更准确的结论是：

> 完整 SNN 中的 v2 仍只支持 **operational sustained-recruitment phenotype**：centered 250-ms envelope 跨过 20 Hz 并持续至少 1 s，同时有约 5 Hz 主导调制。它目前最符合“非平稳慢漂移中的 transient”，不是已证实的 SNN dynamical regime/attractor；centered envelope 还可能相对因果 onset 提前约半个窗口。但在冻结的九维 reduced fast system 中，Stage 0E 已经把 delayed divisive loop 生成的波形解析成两条闭合 period-1 orbit；当前不确定性已从“是否有周期轨道”转移到“局部线性稳定性如何可靠定量”。

### 4.2 当前设计中正确的部分

- Z 保持逐细胞、由局部 inhibitory use 驱动，作为 onset/recruitment susceptibility 是合理的；
- recurrent-E 与 feed-forward E 分开，使 global brake 的因果职责较干净；
- `T_G` 只在持续高状态后启动，避免普通 IED 持续污染它；
- strict classifier 把 engineering completion、有限窗 opening、settled state、return 分开；
- v3 negative 及时停止了无信息的大参数网格。

### 4.3 正确但还不够的部分

- `T_G` 不是简单 additive current，而是改变 recurrent gain/Jacobian；这能 containment，却仍缺 reversal dependence、`tau_eff` 和空间记忆；
- Z/M 都是逐细胞变量，所以模型并非“没有空间”。但当前没有证明它们形成 front 前后有方向的相序；空间图样可能主要继承固定 `W`、core geometry 和噪声；
- 约 1.65 Hz reduced relaxation cycle 已不只是 waveform；Poincare closure 和多相位有限扰动回归支持其为真实周期对象。但 Floquet epsilon 平台未过，且这一 reduced orbit 尚未映射回完整 SNN；
- global inhibition 与局部兴奋、延迟和非线性结合时可以参与分岔、pattern selection 与终止；但当前 permutation-blind `T_G` 没有空间记忆，尚未生成或解释可分辨的 front/wake。

### 4.4 真实 capture 显示当前究竟产生了什么

同一锁定轨迹（E1146、seed 1、20 s、无 kick）给出了四层可以安全陈述的事实：

图证据：`results/paper-ready-figure/fig5_mz_divisive_current_stage/figures/fig5_candidate_E1146_mz_divisive_current_stage.png`；对应的 aggregate failure summary 为同目录下 `fig5_candidate_E1146_mz_divisive_failure_summary.png`。两者都是阶段诊断，不是锁定的 Figure 5 主张。

1. strict 250-ms population envelope 在 `13.8539 s` 跨过 20 Hz，并持续到 20 s 记录终点；因此只有 observed opening，没有 observed offset；
2. 机器选择的 onset 前 returning event 持续 `82.6 ms`，有 `11,040/32,000` 个 E 神经元参与；神经元 first-spike latency 对轴坐标为 `rho=-0.440`，但 11 个可读 virtual-contact 的轴向 timing 为 `rho=+0.045, p=0.894`，所以不能把这一个事件升级成 electrode-readable template；
3. 操作性 onset 处出现上面所述的 60-ms source-to-sink fast sweep；它证明现有 scaffold/快耦合保留空间传播能力，但没有证明局部 ictal tissue recruitment；
4. onset 后首个 1 s 仅 `47.4%` 的 E 神经元至少发放一次，而全体 E 中 `18.3%` 超过 100 Hz；全体神经元率的 P95/P99 分别约 `266/365 Hz`。因此约 60 Hz 的 population mean 不是均匀、低率、有界 ictal branch，而是被大量 silent cells 稀释的高率空间带。

综合起来，当前阶段最准确的 phenotype 是：

> returning fast spatial events -> 一次有序快速轴向 sweep -> 空间带状高率持续招募，同时 Z/T_G 继续漂移；记录窗内无 wake、front stall、annihilation 或 return。

这正好解释了为什么现有三条 workflow 都不能继续：时间上缺 exit boundary，空间上缺慢状态 front/wake，early-ictal bridge 目前只能看到 event rate/extent 连续增大，而不是逼近一个已定义 ictal basin 的临界轨迹。

### 4.5 新的主假设

本线锁定为 **topology-first finite-domain Z–recovery E/I field**。第一版 fast block 直接复用 M3B 的
`rE/rI/sEE/sEI/sIE/sII` 与 LIF population transfer，不另造任意 sigmoid。Z 保留当前模型的
postsynaptic inhibitory-efficacy 语义，关键 E-cell 输入为：

\[
\mu_E=\tau_{mE}\left(C_{EE}w_{EE}s_{EE}
-C_{EI}z(x,t)w_{EI}s_{EI}\right)
+\tau_{mE}J_{XE}\nu_{ext}+\mu_{core}-r(x,t),
\]

\[
\sigma_E^2=\tau_{mE}\left(C_{EE}w_{EE}^2s_{EE}
+C_{EI}[z(x,t)w_{EI}]^2s_{EI}\right)
+\tau_{mE}J_{XE}^2\nu_{ext},
\]

\[
\tau_E\partial_t e=-e+\Phi_E(\mu_E,\sigma_E;V_{th,E}+\phi).
\]

Z sensor 读取未被 Z 缩放前的 raw postsynaptic inhibitory drive，而不是改成 presynaptic resource：

\[
I_I^{raw}=\tau_{mE}C_{EI}w_{EI}s_{EI},\qquad
\tau_z\partial_tz=1-S_\epsilon(I_I^{raw}-I_{th})-z.
\]

\[
\tau_p\partial_t p=-p+e,
\qquad
\tau_r\partial_t r=-r+r_{\max}\,
\mathrm{Hill}([p-\vartheta_r]_+)+D_r\nabla^2r.
\]

这里 `r` 只从 `mu_E` 减一次，`phi` 只进入 threshold；Z 在抑制均值中乘一次、方差中平方一次。`r` 在
reduced equation 里仍是 activation input 的加性减项；**成功关键不在把
additive 换一个名字**。关键是 fast E/I block 先独立证明 low/high topology，`r` 只在 established
recruitment 后启动，而且作为局部场保存招募历史。若只有 bounded tonic branch 而无节律，才开放一个
80--150 ms dynamic-threshold `phi` arm，且 `phi` 只负责 within-bout rhythm，不负责最终退出。

慢回路还必须在 `(z,r)` 平面满足 post-offset safety：先由 frozen state fork 求
`r_sep(z)` 和 `z_safe`，然后要求 offset 后到 `z>=z_safe` 的整个窗口中
`r-r_sep(z)>=delta_r>0`。否则 `r` 先衰减、低 Z 立即重燃，仍不算 closed lifecycle。

职责严格分开：

- `z(x,t)`：front 前方的 local entry permission；
- fast E/I subsystem：有限 ictal branch/orbit 与 within-bout dynamics；
- persistence gate `p`：防止普通短 IED 驱动 recovery；
- `r(x,t)`：recruitment 后才升高的局部 exit field 与 refractory wake；
- broader `K_I`：待检验的 front 外 inhibitory-penumbra mechanism；
- `W_AB`：患者 scaffold，只调传播方向/易感性，不替代 local bistability。

Jirsa 等强调正常态与 ictal dynamics 的共存及由慢变量跨越不同 onset/offset bifurcation；Proix 等进一步把局部动力学嵌入 neural field，区分慢 ictal wavefront 与更快的 spike-wave dynamics。这支持“先锁 fast topology，再放空间 front”的顺序，而不是要求整个 seizure 是永久极限环：[Jirsa et al., Brain 2014](https://academic.oup.com/brain/article/137/8/2210/2847958)，[Proix et al., Nature Communications 2018](https://www.nature.com/articles/s41467-018-02973-y)。人体记录中的 ictal core 与 inhibitory penumbra 分离也说明只看全局 rate/LFP 不足以判定局部 recruitment：[Schevon et al., Nature Communications 2012](https://www.nature.com/articles/ncomms2056)。

## 5. 工程性问题

- 本轮没有修改并行 `.worktrees/topic4-mz-conductance`，两线没有代码冲突；
- 2026-07-20 再核并行线状态（HEAD `b0df4e5`）：full-conductance engineering/parity 已 ACCEPT；seed 1 与
  seed 3 的 workpoint 均在锁定 `c_E={0.85,1.0,1.15}` 合同内同向 NO-GO（0.85 静息，1.0/1.15
  过热并触 cap），所以 fast high-branch 与 presynaptic relay 均未启动。该线回答 bottom-up conductance
  workpoint，本线回答 reduced fast topology；两线没有重复实现；
- 代表性重放固定 seed、20 s、无 kick、锁定 v2 参数和 strict onset，不再人工选“好看事件”；
- spatial capture 只保存 rate、slow traces、virtual-SEEG、逐神经元派生场、24x24 movie 与 source-to-sink space-time map，不保存完整 `E_spk_bool`；NPZ 为 8.73 MB、SHA256 `4d3ebbc7cb0f...`，20 s 单进程仿真 wall time 2475.38 s、峰值 RSS 11.666 GiB；
- LFP readout 是既有 `LFPRecorder` 的原始突触电流代理，发生在 slow/divisive 处理前，不能解释为 divisor 后的有效膜电流；
- capture 强制单进程、BLAS=1，并在 build 前和 simulation 前保证预测启动后仍至少有 96 GiB 可用内存；
- 新增 capture 与 lifecycle/divisive 定向回归为 26/26 passed；完整模型相关回归此前为 85 passed。Stage 0B/0C/transfer-support 并入后的本轮定向联合回归为 73/73 passed（仅有 exact-Siegert quadrature roundoff warning）。

Stage 0B 已把 continuation/state-fork、ceiling detector 与 exact-Siegert sensitivity 收到同一分析合同中。锁定的 126 个参数点共得到 200 个 roots；排除 exact-root 初态后，1786 条 dynamical forks 只有 450 条回到 low 和 1336 条进入 `>100 Hz` 高支。额外 504 条 off-manifold probes 仍只有 123 条 low 与 381 条 `>100 Hz`；0 个有限候选进入 confirm。exact-Siegert 局部复核 200/200 收敛，37/37 个 sub-100-Hz unstable roots 仍不稳定，111/111 个 stable high roots 仍稳定且全部 `>100 Hz`。因此 Stage 0B 的工程与科学 verdict 均已锁为 `CLEAN_NO_GO_LOW_OR_SATURATION_CLIFF_ONLY`。

这个 no-go 有明确范围：dense multistart 加单向 warm scan 不是严格的双向连续 continuation，四类 off-manifold 初态也不可能穷尽六维 basin；exact-Siegert 只局部复核已发现 roots。它足以关闭当前注册参数合同下的 Stage 1--3，却不能升级成“六维系统数学上不存在任何极限环”或“所有 current-based/additive 模型都失败”。

复用审计进一步收紧了 Stage 0B：

- `src/sef_hfo_lif.py` 的 LIF transfer/参数、`topic4_m3b_spectral_phase.py` 的六变量 operator/Jacobian、
  `topic4_criticality.solve_branches` 的多初值与 continuation 框架可以复用；
- 但既有 criticality 只追低支，`solve_branches` 未透传 `w_ee_mult/ratio`，其 branch 聚类尺度也过粗，不能把
  `n_branches_found` 当成多个 fast attractor；
- M3B `field_rhs` 为局部线性分析冻结了 `sigma`，不能用于大振幅 state fork；Stage 0B 必须在同一 RHS 中
  自洽更新 `mu/sigma`；
- `explore_liou_bolton_rate.py` 只能复用 dynamic-threshold 形式，不能复用 lifecycle/front 结论；其旧
  wavefront 代码对已阈值化 bool 再执行一次 `r>thr`，会使非 silent 轨迹的 edge 恒为零。

Stage 0B 的正式结果确认了 spot-check 暗示的拓扑：稳定低支越过不稳定 finite separator 后，注册网格内落入的是约 179--460 Hz 的 stable saturation branch，而不是有限 ictal object。这是节省后续 SNN/空间资源的有效 clean no-go，不是失败后继续叠加慢变量的理由。证据见
`results/topic4_sef_hfo/spatial_slowfast_topology/stage0b_ei_fast/stage0b_summary.json` 与同目录
`stage0b_brief_cn.md`；正式运行单进程 5 分 16.74 秒、峰值 RSS 0.221 GiB，Stage 0A+0B 定向测试 18/18 passed。

Stage 0C 的 coarse 9D screen 随后在 189 个参数点、3564 条 forks 中找到 23 条表面的 oscillatory candidates，但它们全部越过原 transfer LUT 的低 `mu` 支持域，因而 coarse verdict 只能是 `INCONCLUSIVE_NO_CONFIRMED_FINITE_FAST_OBJECT`。独立 v1.1 数值修复用 no-clip/no-extrapolation 的 extra-fine transfer 重放 6 点、102 条锁定 histories：17 条进入 `>100 Hz`，84 条长瞬态/分类未决，1 条在 `z=0.85, alpha_G=16` 存活。该轨迹 12 s confirm 为 mean 5.969 Hz、peak 96.34 Hz、frequency 1.665 Hz；`dt/2` 为 mean 5.927 Hz、peak 95.46 Hz 且同频，直接 exact-transfer 审计也通过。但同点仅 1/17 histories 存活，0 个参数点通过两初态 basin 门，所以 Stage 0C 是 **numerically repaired but scientifically inconclusive**，不是 pass，也不是 clean no-go。证据见 `results/topic4_sef_hfo/spatial_slowfast_topology/stage0c_transfer_support_audit_v1_1/stage0c_transfer_support_summary_v1_1.json`。

五臂 ablation 只支持一个受限的机制线索：dynamic 和 mean-only 均保留 oscillation，instantaneous 变为约 6.04 Hz tonic，clamped 回到约 0.457 Hz low，matched-subtractive 数值发散并 fail-closed。因此 **delayed mean divisive feedback** 值得保留，但方差的 `/D_G^2` 缩放不是这条轨迹振荡的必要条件；单轨迹不能支持机制特异性或 ictal attractor 主张。

Stage 0D v1.1 已修正 confirm--`dt/2` 频率门和空 Figure B，且与 v1 的科学结果完全一致：180 条历史中 175 条 `numerical_unresolved`、5 条 `candidate_survives`，后者全部集中在 `z=0.85, alpha_G=15, phase_050`，中心 `alpha_G=16` 无 survivor，因而预注册 open-basin/邻域门不通过。v1.1 对 v1 的 180 条 fork 分类改变为 0，`scientific_result_changed=false`；它将 Stage 0D 锁为“窗口分类不复现且有大量未决轨迹”，而不是“不存在周期”。证据见 `results/topic4_sef_hfo/spatial_slowfast_topology/stage0d_local_basin_replication_v1_1/stage0d_local_basin_summary_v1_1.json`。

Stage 0E 用截面 return 与 shooting 取代 FFT 分类，直接证实 `z=0.85, alpha_G=15/16` 存在闭合 period-1 轨道。两点周期为 `604.898/608.519 ms`，shooting closure 约 `10^-15`，base--`dt/2` 周期差为 `0.525/0.539 ms`，相位对齐波形残差为 `0.01495/0.01563`。每点四个 phase anchors、8 条 fast 和 8 条 pool 有限扰动经 8 次 return 全部收缩，median final/first distance 约 `10^-9`。但三档 epsilon 的 Jacobian 相对差为 `0.22--1.07`，未形成导数平台；虽然所有估计谱半径都 `<=0.00304`，仍必须按锁定规则判为 `STAGE0E_NUMERICAL_UNRESOLVED`。这是强的闭合/非线性回归证据，不是已认证 stable Floquet attractor。证据见 `results/topic4_sef_hfo/spatial_slowfast_topology/stage0e_poincare_floquet_audit/stage0e_poincare_floquet_summary.json`。

Stage 0A topology oracle 已完成并 **PASS**：canonical quintic subcritical-Hopf normal form 的数值
entry bracket 为 `[0, 0.025]`（解析边界 `0`），exit bracket 为 `[-0.275, -0.25]`（解析边界
`-0.25`）；同一 classifier 正确拒绝 oscillatory ceiling 和 long transient。constructed closed toy 也能无 reset
完成 entry、13.3 cycles、exit、return 以及 early/late retrigger sanity。它只证明分析链能识别已知拓扑，**不支持
HFOsp 的 Z/recovery 机制已经成立**。toy 中提高兴奋性的慢量已命名为 `permissivity`，不与本项目“下降才促发”的
抑制资源 Z 混用。证据：
`results/topic4_sef_hfo/spatial_slowfast_topology/stage0a_oracle/stage0a_oracle_summary.json`；新增测试
6/6、相关联合回归 32/32 passed，单线程峰值 RSS 0.061 GiB。

## 6. 最小修改路线

1. **如实画当前阶段（已完成）**：同一轨迹已对齐 continuous virtual-SEEG、population rate、Z/`T_G`，并展示 returning event、recruited window、完整 source-to-sink field 与 onset zoom。图明确把 60-ms 有序 sweep 标成 fast event wave，而不是慢 ictal wavefront。
2. **关闭 broad linear-M 搜索**：只留一次 established-state gated-recovery fork，回答 timing 与 leverage 的诊断问题。
3. **Stage 0A topology oracle（已 PASS）**：已用已知有低态/有限振荡态/进入退出边界的 canonical normal form 验证 continuation、classifier、retrigger 和 slow-loop 分析链；它只是正对照，不作生物机制 claim。
4. **Stage 0B frozen E/I topology（已 clean no-go）**：锁定网格只有 low、unstable separator 与 `>100 Hz` saturation cliff，因而关闭直接接 slow loop/space 的路线。
5. **Stage 0C 最小 fast-topology surgery（已收口，inconclusive）**：延迟 mean divisive feedback 打开了一条数值可信的 relaxation-like 轨迹，但 0/6 点通过 basin-level finite-object 门。它不是 conductance，也不负责秒级退出。
6. **Stage 0D 局部 basin/邻域复现（已完成，窗口门未过）**：180 条锁定历史只有一个 phase/邻点保留 5 条 survivor，不支持预注册 open basin；不据此排除周期。
7. **Stage 0E Poincare/Floquet（已完成，数值未解析）**：两条 period-1 orbit 的闭合和有限扰动回归已强支持；Floquet epsilon 平台未过，不开放下游。
8. **Stage 0F smooth/variational certificate（唯一开放）**：固定同两个点、同截面和 base/half `dt`，用可微 transfer 与离散 tangent/event projection 复核 Floquet；不扫参数，不加新状态。
9. **Stage 1 0D closed lifecycle（关闭）**：只有 Stage 0F 通过并找到 low/cycle 与 `r_sep(z)` 边界才开放；让 Z 负责进入、p-gated local recovery 负责建立后退出，验证 same-basin return 与 early/late retrigger。
10. **Stage 2 1D finite domain（关闭）**：固定通过 0D gate 的参数后，才测试 local nucleation、有限速 front、penumbra、wake、stall/annihilation；front 必须按 local branch/orbit membership + 至少 2--3 周期 dwell 定义，不用首次 spike/activity crossing，并排除撞边界假终止。
11. **Stage 3 2D E1146（关闭）**：最后才加入 anisotropic kernel、患者 scaffold、double source 与 electrode readout；做 isotropic、rotated-axis、spatial-shuffle 对照。
12. **Stage 4 两线会合**：topology-first 线提供 branch/front 必要条件，FCXR 线检验 conductance/relay 能否在 40k SNN 中实现它；未过各自 gate 前不合并。

## 7. 下一步的建议

核心目标保持不变：

> 在同一组参数、无 reset 的轨迹中，得到 returning interictal event -> 局部 ictal nucleation -> 有限速 patterned recruitment -> spatial stall/annihilation -> 回到同一 interictal basin，并让这些阶段都可被 frozen-state 与空间指标分析。

本轮建议的执行判决：

- **GO**：完成 current-stage 真实时空图并把它当作边界诊断；
- **DONE / 仅分析链**：低成本 topology oracle 已 PASS；
- **DONE / CLEAN NO-GO**：原始六变量 0D frozen E/I fast-branch screen 已关闭直接下游；
- **DONE / INCONCLUSIVE**：9D M4 动态除法快池仅留下 1 条 basin 未证实的数值可信轨迹，不允许直接进入 Stage 1；
- **DONE / 操作门未过**：Stage 0D v1.1 未复现预注册 open basin，但不排除周期；
- **DONE / 周期闭合强支持，Floquet 未解析**：Stage 0E 已解出两条闭合周期和强有限扰动回归，但按锁定门仍不能称 stable attractor；
- **GO / 唯一开放复核**：Stage 0F 只解析 smooth/variational Floquet 证书，不扫新参数；
- **NO-GO**：继续扫 `eta_m/tau_m/T_G/alpha_T`；
- **NO-GO**：在 finite high branch 与 exit boundary 出现前恢复三条下游 workflow；
- **NO-GO**：把 conductance、更多慢变量或更大算力本身当作成功标准。

详细设计合同见：`docs/superpowers/specs/2026-07-20-topic4-topology-first-spatial-slow-fast-field-design.md`。
