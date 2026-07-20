# Fig4/5 SNN 核心公式与逻辑链
---

## A. Fig4/5 自发传播衬底

### A1. E/I LIF 网络

$$
\tau_m^a\dot V_i=-V_i+I_i^E-I_i^I,\qquad a\in\{E,I\}
$$

$$
V_i\ge V_{\theta,i}\Rightarrow \text{spike},\quad V_i\leftarrow V_r,\quad \text{refractory}
$$

突触电流为延迟双指数：

$$
\tau_r^X\dot s_i^X=-s_i^X+\sum_jJ_{ij}^X\delta(t-t_j-d_{ij}),\qquad
\tau_d^X\dot I_i^X=-I_i^X+s_i^X
$$

$$
d_{ij}=\tau_0+\|x_i-x_j\|/v_{\text{axon}}
$$

背景输入：

$$
\nu_{\text{ext}}(t)=\big[r_{\text{ext}}\nu_\theta+\xi(t)\big]_+,\qquad
d\xi=-\xi/\tau_n\,dt+\sigma_\xi\,dW_t
$$

**逻辑**：背景 OU/Poisson 噪声给自发点火机会；E/I 和 refractory 让事件可自限；Fig4/5 不加外部 kick。

变量/参数：

- $a$：神经元类型，$E$ 或 $I$。
- $V_{\theta}$ / $V_r$：发放阈值 / reset，当前基线为 18 / 11 mV。
- $\tau_m^{E/I}$、$\tau_{\text{ref}}^{E/I}$：膜时间常数和不应期；E 比 I 慢。
- $I^E,I^I$：AMPA 兴奋性电流和 GABA 抑制性电流。
- $r_{\text{ext}}$：背景 drive 比例；Fig4/5 用亚阈工作点，surround 安静。

### A2. E->E 长轴 scaffold

只有 E->E 连接做旋转各向异性。位移 $\Delta x=(z_1,z_2)$：

$$
u=\cos\theta_{EE}z_1+\sin\theta_{EE}z_2,\qquad
v=-\sin\theta_{EE}z_1+\cos\theta_{EE}z_2
$$

$$
P(j\to i)\propto
\exp\left[-\sqrt{(u/\ell_\parallel)^2+(v/\ell_\perp)^2}\right],
\qquad
\ell_\parallel=\ell_{EE}\sqrt{\mathrm{AR}},\quad
\ell_\perp=\ell_{EE}/\sqrt{\mathrm{AR}}
$$

**逻辑**：不是硬编码一条线，而是让 recurrent E->E 更容易沿 $\theta_{EE}$ 接力传播。Fig5 用 $45^\circ$、AR=2；Fig4 用病人 source->sink 轴。

变量/参数：

- $\theta_{EE}$：E->E 长轴方向；Fig5 固定 45 度，Fig4 来自病人传播轴。
- $\ell_{EE}$：E->E 空间尺度，当前基线 0.380 mm。
- $\mathrm{AR}$：长短轴比例；AR=2 表示沿轴连接更远。
- $\ell_\parallel,\ell_\perp$：沿轴 / 垂轴的有效连接长度。
- $C_{EE}$：每个 E 细胞接收的 E 输入数；实现里固定 in-degree，不是全连接。

### A3. 双低阈值 core

灶外阈值保持基线；灶内 E 阈值降低：

$$
V_{\theta,i}=
\begin{cases}
18, & i\notin\text{core 或 }i\in I\\
v,\quad v\sim\mathcal N(\mu_{\text{core}},\sigma_{\text{core}}^2),\ v\ge V_r, & i\in\text{core E}
\end{cases}
$$

双灶放在长轴两端：

$$
c_\pm=c_0\pm s_{\text{sep}}\frac{L}{2}\hat u_{EE}
$$

`twoend_equal`：两端同一组 $(\mu_{\text{core}},\sigma_{\text{core}})$；哪端先点火来自阈值抽样、连接抽样、噪声和读出几何，不是人为指定。

变量/参数：

- $\mu_{\text{core}}$：core 内 E 阈值均值；越低越容易自发点火。
- $\sigma_{\text{core}}$：core 内阈值异质性；决定低阈值尾部和点火不均匀性。
- $V_r$：截断下界，避免出现低于 reset 的非物理阈值。
- $s_{\text{sep}}$：两灶沿长轴分离比例。
- core radius：core 空间半径；当前 Fig4/5 系列通常用 1.5 mm。

### A4. 自发事件

$$
\text{KICK\_BOOST}=0,\qquad t_{\text{kick}}\to\infty,\qquad \text{slow=None}
$$

$$
\text{noise}\rightarrow
\text{one core ignites}\rightarrow
\text{E->E axial relay}\rightarrow
\text{forward/reverse event}\rightarrow
\text{self-termination}
$$

**当前边界**：支持“同轴两端可自发成核，并产生少数可读正/反传播事件”；不支持“平衡、独立、长时序双源列车”。

变量/参数：

- `KICK_BOOST=0`：没有外部触发。
- `slow=None`：M3A 慢变量关闭，等价于只看固定 E/I 衬底。
- `forward/reverse`：由虚拟 SEEG 读出的传播方向，不等同于真实病因标签。
- `returned`：事件能否回到基线，是自限性判断的一部分。

### A5. 虚拟 SEEG readout

方向判定用 spike-derived envelope：

$$
\mathrm{env}_k(t)=\sum_{j\in E}w_{kj}r_j(t),\qquad
w_{kj}\propto\exp(-d_{kj}^2/2\sigma_w^2)
$$

参与触点：

$$
\mathrm{peak}_k>\text{floor}+\text{margin}
$$

触点 onset：

$$
t_k=\min\{t:\mathrm{env}_k(t)\ge0.5\,\mathrm{peak}_k\}
$$

方向：

$$
a_{\text{event}}=
\mathrm{centroid}(\text{late }k_{\text{dir}})
-
\mathrm{centroid}(\text{early }k_{\text{dir}}),
\qquad
\mathrm{sign}=\mathrm{sign}(a_{\text{event}}\cdot\hat u_{EE})
$$

**逻辑**：模型内部的 hidden source 是哪端 core 先点火；论文图展示的是同一套虚拟 SEEG 能否把事件读成 forward / reverse。

变量/参数：

- $\mathrm{env}_k(t)$：第 $k$ 个虚拟触点的 spike-derived envelope。
- $r_j(t)$：E 神经元 $j$ 的局部发放密度。
- $\sigma_w$：神经元到触点的空间平滑宽度。
- $k_{\text{dir}}$：用于估方向的早/晚端触点数；Fig5 通常 3，Fig4 因真实电极稀疏可用 2。
- $\hat u_{EE}$：source->sink 参考轴；sign 只表示沿该轴正/反。

### A6. Fig4/5 讲法

图形链：

$$
\text{mechanism}\rightarrow
\text{tempA source}\rightarrow
\text{tempB source}\rightarrow
\text{electrode readout}
$$

安全表述：

> 一个带 E->E 长轴 scaffold 的 E/I SNN，在轴两端加入低阈值易激 core 后，可以在无外部触发的背景噪声下产生自发正/反传播事件，并可被同一虚拟 SEEG montage 读出。

禁止表述：

> 真实病人的正反模板已经被证明来自两个病灶独立随机点火。

---

## B. M3A 慢变量扩展层

M3A 问题：

$$
\text{同一 scaffold 上，慢状态能否解释事件为什么被推大、怎样被推大、以及怎样恢复？}
$$

当前边界：同轴变大的事件只算 expanded axial recruitment；只有主导活动模式从轴向传播转向离轴、全局或低 $k$ 招募，并且能恢复，才算 ictal-like candidate。

慢变量打开时：

$$
\tau_m^a\dot V_i=-V_i+I_i^E-s_i^{\text{inh}}I_i^I-g_{K,i}
$$

### B1. 三个单细胞慢变量

去抑制：

$$
\tau_z\dot z=z_\infty-z,\qquad s_i^{\text{inh}}=z_i
$$

自适应阈值：

$$
\dot\phi_i=-\frac{\phi_i-\phi_0}{\tau_\phi}+\Delta\phi\,S_i(t),\qquad
V_{\theta,i}\to\phi_i
$$

sAHP / 慢恢复电流：

$$
\dot g_{K,i}=-\frac{g_{K,i}}{\tau_K}+g_K^{\max}S_i(t),\qquad
I_i^{\text{net}}\mathrel{-}=g_{K,i}
$$

**逻辑**：$z$ 让抑制变弱，$\phi$ 和 $g_K$ 提供活动后的恢复/自限。

变量/参数：

- $z$：去抑制变量；越小表示有效抑制越弱。
- $\phi$：动态阈值；spike 后升高，随后恢复。
- $g_K$：慢外向恢复电流；活动越多越强。
- $S_i(t)$：第 $i$ 个神经元的 spike train。
- $\tau_z,\tau_\phi,\tau_K$：对应慢变量的恢复时间尺度；当前属于 pilot 参数，未生物标定。

### B2. 区域抑制资源 q

区域油箱 $q_r\in[q_{\min},1]$ 缩放抑制输入：

$$
q_r\leftarrow q_r+
dt\left[
\frac{1-q_r}{\tau_{\text{rec}}}
-
k_{\text{use}}\bar a_r q_r
\right],
\qquad
\bar a_r\leftarrow\bar a_r+\alpha_a(a_r-\bar a_r)
$$

稳态：

$$
q_r^\ast=\frac{1}{1+k_{\text{use}}\bar a_r\tau_{\text{rec}}}
$$

施加到 E 细胞抑制：

$$
s_i^{\text{inh}}=
\begin{cases}
q_{\text{global}}q_{\text{core}}, & i\in\text{core E}\\
q_{\text{global}}, & i\in\text{background E}
\end{cases}
$$

**逻辑**：活动多 $\rightarrow q$ 漏 $\rightarrow$ 去抑制增强；安静期 $q$ 回灌。

变量/参数：

- $q_r$：区域 $r$ 的抑制资源；1 为满箱，越低表示抑制越耗竭。
- $q_{\text{global}}$：全局抑制资源，作用所有 E 细胞。
- $q_{\text{core}}$：core 额外抑制资源，只作用 core E。
- $a_r,\bar a_r$：区域即时活动和 EMA 平滑活动。
- $\tau_{\text{rec}}$：资源恢复时间常数。
- $k_{\text{use}}$：活动使用资源的强度，是 M3A-A2 主扫旋钮。

### B2.1. M4 regional q–p–M hybrid lifecycle（R4 rate-patch experimental node）

> **状态（2026-07-21）**：本节记录 `topic4-mz-divisive-lifecycle` 独立线已经数值闭合的完整
> R4 center equation。它是 P=3 core–annulus–bath rate-patch mechanism screen，**不是**本文件
> A 节的 full SNN；没有修改 E→E weight/kernel/delay、conductance membrane 或 presynaptic relay。
> 代码字段仍叫 `z`，本节为避免和空间坐标混淆统一记为 inhibitory resource $q$。

#### B2.1.1 状态与空间投影

每个区域 $r\in\{c=\mathrm{core},a=\mathrm{annulus},b=\mathrm{bath}\}$ 有 10 个连续状态：

$$
x_r=(r_{E,r},r_{I,r},s_{EE,r},s_{EI,r},s_{IE,r},s_{II,r},
r^f_{E,r},q_r,p_r,m_r),
$$

全域再共享 $(\mu_G,S_G)$，故总维数为 $3\times10+2=32$。$K_{EE}$ 与 $K_I$ 是从 locked
$48\times48$、12-mm periodic field 对 core、equal-area annulus、far bath 做 block-average 得到的
row-stochastic spatial operators；row 是 target region、column 是 source region。三区面积权重为

$$
(w_c,w_a,w_b)=(113,112,2079)/2304.
$$

#### B2.1.2 快 rate–synapse 子系统

$$
\tau_E\dot r_{E,r}=-r_{E,r}+\Phi_E(\mu_{E,r},\sigma_{E,r}),
\qquad \tau_E=20\ \mathrm{ms},
$$

$$
\tau_I\dot r_{I,r}=-r_{I,r}+\Phi_I(\mu_{I,r},\sigma_{I,r}),
\qquad \tau_I=10\ \mathrm{ms}.
$$

$\Phi_{E/I}$ 是 LIF Siegert transfer（输出单位 kHz）：

$$
\Phi_x(\mu,\sigma)=
\left[
\tau_{\mathrm{ref},x}+\tau_x\sqrt\pi
\int_{(V_R-\mu)/\sigma}^{(V_\theta-\mu)/\sigma}
e^{u^2}(1+\operatorname{erf}u)\,du
\right]^{-1}.
$$

四个空间突触场为：

$$
\tau_A\dot s_{EE}=K_{EE}r_E-s_{EE},\qquad
\tau_G\dot s_{EI}=K_Ir_I-s_{EI},
$$

$$
\tau_A\dot s_{IE}=K_Ir_E-s_{IE},\qquad
\tau_G\dot s_{II}=K_Ir_I-s_{II},
$$

其中 $\tau_A=3.5$ ms，$\tau_G=18$ ms。定义 shared recurrent-E divisor

$$
D(t)=1+\alpha_GS_G(t),\qquad \alpha_G=15,
$$

以及 $W_{EE}^{\mathrm{eff}}=1.1W_{EE}$、$W_{EI,r}^{\mathrm{eff}}=q_rW_{EI}$，输入矩为：

$$
\mu_{E,r}=
\frac{\tau_EC_{EE}W_{EE}^{\mathrm{eff}}s_{EE,r}}{D}
-\tau_EC_{EI}q_rW_{EI}s_{EI,r}
+\tau_EJ_E\nu_{\mathrm{ext}}
-A_{\max}m_r+I_r^{\mathrm{pulse}},
$$

$$
\sigma_{E,r}^2=
\frac{\tau_EC_{EE}(W_{EE}^{\mathrm{eff}})^2s_{EE,r}}{D^2}
+\tau_EC_{EI}(q_rW_{EI})^2s_{EI,r}
+\tau_EJ_E^2\nu_{\mathrm{ext}},
$$

$$
\mu_{I,r}=\tau_I(C_{IE}W_{IE}s_{IE,r}-C_{II}W_{II}s_{II,r})
+\tau_IJ_I\nu_{\mathrm{ext}},
$$

$$
\sigma_{I,r}^2=
\tau_I(C_{IE}W_{IE}^2s_{IE,r}+C_{II}W_{II}^2s_{II,r})
+\tau_IJ_I^2\nu_{\mathrm{ext}}.
$$

所以 R4 并非只有一个 additive current：$q$ 乘性改变 I→E mean/variance，$S_G$ 除性改变
recurrent-E mean/variance，只有 $A_M=A_{\max}m$ 是 subtractive exit current。

#### B2.1.3 inherited shared-pool containment 内环

$$
15\ \mathrm{ms}\,\dot r^f_{E,r}=r_{E,r}-r^f_{E,r},
$$

$$
\psi(r)=\frac{[r-.005]_+^2}{.015^2+[r-.005]_+^2},
\qquad a_G=\sum_rw_r\psi(r^f_{E,r}),
$$

$$
30\ \mathrm{ms}\,\dot\mu_G=-\mu_G+a_G,
\qquad
80\ \mathrm{ms}\,\dot S_G=-S_G+\mu_G.
$$

低态中 $r_E^f<5$ Hz，故 $\psi=0$，shared pool 不制造 entry fold；fold 后它在几十毫秒内建立
负反馈并把高支约束为 bounded core–annulus oscillation（CCO）。因此当前分工是：$q$ 打开入口，
shared pool 形成有界 bursting 内环，M 负责退出。

#### B2.1.4 slow sensors

所有传感器使用 C1 compact smoothstep：

$$
H(x;\theta,w)=h^2(3-2h),\qquad
h=\operatorname{clip}\left(\frac{x-\theta}{w},0,1\right).
$$

抑制使用量先按区域计算：

$$
U_r^{\mathrm{local}}=d_r
H(s_{EI,r}-s^0_{EI,r};.004,.004),
\qquad d=(1,1,0),
$$

再把 core/annulus 面积加权 pooling：

$$
U_c=U_a=
\frac{w_cU_c^{\mathrm{local}}+w_aU_a^{\mathrm{local}}}{w_c+w_a}.
$$

fast occupancy 与 neighborhood recruitment 为：

$$
O_r=H(r^f_{E,r};.020,.010),\qquad R=K_{EE}O.
$$

$U$ 测量到达该区的 inhibitory synaptic use；$O$ 测量局部高活动；$R$ 要求该活动还通过既有
空间 E kernel 招募邻域。三个 rate threshold 的单位均为 kHz。

#### B2.1.5 inhibitory resource q

core 与 annulus 的资源方程为：

$$
\dot q_r=
r_{\mathrm{rec}}(\bar m)(q_0-q_r)
-\frac{U_r}{\tau_D}(q_r-q_{\mathrm{res}}),
\qquad r\in\{c,a\},
$$

$$
r_{\mathrm{rec}}(\bar m)=
\frac{1-\bar m}{\tau_{\mathrm{slow}}}
+\frac{\bar m}{\tau_{\mathrm{fast}}}.
$$

其中

$$
q_0=.90,\quad q_{\mathrm{res}}=.8415605,\quad
\tau_D=.278354\ \mathrm{s},\quad
\tau_{\mathrm{slow}}=90\ \mathrm{s},\quad
\tau_{\mathrm{fast}}=20\ \mathrm{s}.
$$

第一项是把资源拉回 $q_0$ 的正向恢复力；第二项是使用依赖、把资源拉向 $q_{\mathrm{res}}$ 的负向
耗竭力。bath 在当前 diagnostic 被强制 $\dot q_b=0$。冻结 $(\bar m,U)$ 后只有一个稳定 nullcline：

$$
q^*(\bar m,U)=
\frac{r_{\mathrm{rec}}(\bar m)q_0+(U/\tau_D)q_{\mathrm{res}}}
{r_{\mathrm{rec}}(\bar m)+U/\tau_D},
\qquad
\lambda_q=-\left[r_{\mathrm{rec}}(\bar m)+\frac{U}{\tau_D}\right]<0.
$$

所以 q 方程本身不产生 Hopf 或第二个稳定点；它只让慢轨迹跨越快子系统的 fold。

#### B2.1.6 persistence、hybrid latch 与 M

$$
\tau_p\dot p_r=O_r-p_r,\qquad \tau_p=.75\ \mathrm{s}.
$$

$O_r$ 是建立 persistence 的正向力，$-p_r$ 是静息时的遗忘力。离散 latch
$L\in\{0,1\}$ 使用 hysteretic guards：

$$
L:0\to1
\quad\Longleftrightarrow\quad
p_c,p_a\ge.115\ \text{且}\ R_c,R_a\ge.60,
$$

$$
L:1\to0
\quad\Longleftrightarrow\quad
r^f_{E,c},r^f_{E,a}\le.005,\quad
q_c,q_a\ge.885,\quad p_c,p_a\le.03.
$$

core/annulus pooled M 为：

$$
\dot{\bar m}=
L\,O_cO_a\frac{1-\bar m}{.225\ \mathrm{s}}
-(1-L)\frac{\bar m}{12\ \mathrm{s}},
\qquad A_M=1.6\bar m\ \mathrm{mV}.
$$

当 $L=1$ 且两区都活跃时，M 向 1 建立；当 $L=1$ 但 bursting 已停止时 $\dot m=0$，M 被保持，
形成 protected recovery window；只有 $L=0$ 后 M 才以 12 s 时间常数自然释放。这一中性保持段是
R4 属于 hybrid controller、而不是平滑恢复电流的关键。

#### B2.1.7 触发协议、fold 关系与验收边界

登记的 interictal-like challenge 是 six-pulse core drive：

$$
I_r^{\mathrm{pulse}}(t)=3\ \mathrm{mV}
\sum_{k=1}^{6}\mathbf 1_{[t_k,t_k+20\ \mathrm{ms})}(t)(1,0,0)_r,
$$

$$
t_k=(1000,3122,5044,6321,7531,10915)\ \mathrm{ms}.
$$

该电流只加到 E-input mean，不加到 variance。第五次事件使 regional resource 穿过已求得的 localized
real fold

$$
q_R^{SN}=.8558315843,
$$

随后出现 4 次 response-excluded core/annulus paired returns；latch 在约 10.234 s 打开，M 完成 finite
fast exit。$q$ 恢复到 `.885` 后 latch 在约 74.117 s reset，M 再释放，约 311.7 s 回到原 LLL basin
邻域。早 challenge 不重建 autonomous lifecycle，same-basin late challenge 又在 event 5 重建 4-return
lifecycle。

此前的 fold 图是冻结 $(q,A_M)$ 后快子系统的地形图；R4 是慢状态在该地形上实际走出的一条
entry–burst–exit–reset loop。R4 没有创造新 fold，也没有证明 Hopf、SNIC、torus 或平滑 full-system
limit cycle。精确 regional exit continuation 目前只有 $q=.855/.850$ 两点，实际 dynamic exit 发生在更深
$q\approx.845$，因此只能写 **finite additive exit**，不能写“已精确穿过完整 exit-fold curve”。此外，
20–312 s 恢复含 analytic slow bridges 与 full-fast sentinels，late challenge 是 same-basin state fork；
结果不是零输入 spontaneous onset、连续空间 wavefront、Virtual-SEEG 或 full-SNN seizure lifecycle。
完整验收见 `docs/archive/topic4/sef_hfo/mz_actual_entry_lifecycle_closure_2026-07-21.md`。

### B3. 状态坐标

静态 E/I 比例：

$$
\mathrm{lgr}=\frac{c_{ee}/c_{ei}}{G}
$$

动态化：

$$
\rho(t)=
\frac{c_{ee}/(c_{ei}q_{\text{core}})}
{Gq_{\text{global}}}
=
\frac{\mathrm{lgr}}{q_{\text{core}}q_{\text{global}}}
$$

慢变量逻辑链：

$$
\text{spontaneous events}
\rightarrow
\bar a_r\uparrow
\rightarrow
q_{\text{core/global}}\downarrow
\rightarrow
\rho(t)\uparrow
\rightarrow
\text{larger recruitment?}
$$

判定边界：$\rho$ 是模型坐标，不是生理量。M3A 先看慢状态轨迹是否合理，再看 phenotype 是否真的发作样。单纯事件率升高、$r95$ 变大、contact 看起来更同步，或旧 collision 读数变化，都不能自动算 ictal-like。

变量/参数：

- $c_{ee}$：core 内 E->E 增益。
- $c_{ei}$：core E 接收抑制的静态缩放。
- $G$：全局 E 接收抑制的静态缩放。
- $\mathrm{lgr}$：静态“局部兴奋 / 全局抑制”坐标。
- $\rho(t)$：慢变量动态化后的状态坐标；$q$ 漏时 $\rho$ 上升。

### B4. M3A phenotype gate

当前 M3A 验收分四类：

1. **Interictal axial event**：局部、短时、自限，source-space onset 主要沿 E->E 长轴推进。
2. **Expanded axial recruitment**：范围更大、招募更多，但仍沿同一长轴相干传播；这是间期轴向传播的放大版，不自动叫发作。
3. **Ictal-like recruitment candidate**：大范围招募同时出现轴向主导下降、离轴/全局/低 $k$ 成分上升，并且有恢复。
4. **Runaway**：tonic pinned high 或 tail 不回落；不是成功发作模型。

因此 A1b 的 local-global 地形只支持“局部化易激性优势 / 全局 restraint”作为状态坐标；A1c 只说明全局动态刹车有 timing 作用但空间上不干净；A2 当前支持 $q_{\text{core}}$ 耗竭 + $g_K$ 恢复可以产生 expanded axial recruitment 候选，还没有证明完整的 interictal -> ictal-like -> recovery 转换。

## B5. M3A-v2 空间慢变量场（spatial slow-variable field）

> **本节状态**：公式 + 计划锁定（2026-06-28），实现走 red-TDD，见
> `docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2-spatial-slowvar-field-plan.md`。
> 仍是 **mechanism screen**：下面的 `ictal-like recruitment candidate` 是一个**检出标签**，
> **不是**发作主张。破轴（轴向主导被打破）是否真的发生是**经验问题**，移交延后的 ablation（§B5.8）。
> v2 造的是**探测器 + 机制载体**，不预设破轴一定发生。

### B5.0 为什么从两标量油箱升级到空间场

v1（§B2 `q_core`·`q_global`，两个**标量**油箱）能把事件沿轴推大（expanded axial recruitment），
但**结构上无法表示破轴**：两个全局标量没有**空间历史**——“轴向疲劳的同时周边许可度上升”这件事
没有自由度去承载，只要 E→E scaffold 还在，全局去抑制通常只会继续**加强**轴向，而不是让离轴/全局模式追上来。

v2 给每个位置一份自己的慢状态：抑制资源场 $q_I(x,t)$ 与疲劳/恢复场 $g_K(x,t)$。于是机制链

$$
\underbrace{\text{局部资源耗竭 } q_I\!\downarrow}_{\text{推轴向扩大}}
\;\rightarrow\;
\underbrace{\text{轴向通道疲劳 } g_{K,\text{axis}}\!\uparrow \;+\; \text{周边许可度上升 } q_{I,\text{offaxis}}\!\downarrow}_{\text{降低轴向优势}}
\;\rightarrow\;
\underbrace{\text{破轴 / 离轴·全局·低-}k\text{ 招募}}_{\text{ictal-like candidate}}
\;\rightarrow\;
\underbrace{\text{恢复变量终止 + }q\text{ 回灌}}_{\text{returned}}
$$

**可被表示、可被检出**。最小 v2 = $q_I + g_K$；$D_{EE}$（§B5.4）是延后的可选第二阶段。

### B5.1 空间活动场（firing-rate fields）

把 E / I 的 spike 平滑成空间率场（卷积核 $K_r$ + 时间 EMA $\tau_a$）：

$$
r_E(x,t)=K_r * \sum_{i\in E}S_i(t)\,\delta(x-x_i),\qquad
r_I(x,t)=K_r * \sum_{i\in I}S_i(t)\,\delta(x-x_i)
$$

**实现**：每 $dt$ 把当步 spike bin 到 $n_{\text{grid}}\times n_{\text{grid}}$ 网格，空间卷积各向同性高斯 $\sigma_r$，
对结果做时间 EMA（系数 $\alpha_a=1-e^{-dt/\tau_a}$）。复用 `src/sef_hfo_field.py` 的
`isotropic_gaussian` + `convolve_periodic`（FFT 周期卷积）。

- $K_r,\sigma_r$：spike→率场的空间平滑核 / 宽度（mm）。
- $\tau_a$：率场时间 EMA，是慢变量看到的“活动”平滑尺度。
- $x_i$：神经元 $i$ 在 $L\times L$ 薄片上的连续坐标（mm）。

### B5.2 抑制资源场 $q_I(x,t)$

$$
\partial_t q_I(x,t)=\frac{1-q_I(x,t)}{\tau_q}-k_q\,f\!\big(a_q(x,t)\big)\,q_I(x,t),
\qquad q_{\min}\le q_I\le 1
$$

驱动项（抑制资源主要随**抑制使用**耗竭，故 $\eta_I\ge\eta_E$）：

$$
a_q(x,t)=K_q*\big[\eta_E\,r_E(x,t)+\eta_I\,r_I(x,t)\big]
$$

饱和函数（避免单个 spike 就把油箱抽干）：

$$
f(a)=\frac{[a-a_0]_+}{a_{50}+[a-a_0]_+},\qquad
f(a_0)=0,\;\; f(a_0+a_{50})=\tfrac12,\;\; f\to1\ (a\to\infty)
$$

施加到 E 细胞抑制（$I$ 细胞 $q_I\equiv1$）：

$$
s_i^{\text{inh}}=q_I(x_i,t)\quad(i\in E)
$$

**归约**：$q_I\equiv1\Rightarrow s_i^{\text{inh}}=1$（无去抑制，回到 §A 基线）；$q_I$ **空间均匀** $=q_{\text{global}}$
则复现 §B2 的标量 `RegionalResource`（global-only，$q_{\text{core}}=1$）。$k_q=0$ 即关（off-by-default 字节奇偶）。

- $\tau_q,k_q,q_{\min}$：恢复时间常数 / 耗竭率（主旋钮）/ 下界。
- $K_q,\sigma_q$：去抑制活动感知核 / 宽度；**结构不变量 $\sigma_q>\sigma_K$**（去抑制足迹**宽**）。
- $\eta_E,\eta_I$：率场权重，默认 $\eta_E=0.3,\eta_I=1.0$（抑制资源跟抑制使用）。
- $a_0,a_{50}$：饱和起点 / 半饱和点。

### B5.3 疲劳 / 恢复场 $g_K(x,t)$

$$
\partial_t g_K(x,t)=-\frac{g_K(x,t)}{\tau_K}+k_K\,f\!\big(a_K(x,t)\big)\,\big(g_K^{\max}-g_K(x,t)\big),
\qquad 0\le g_K\le g_K^{\max}
$$

$k_K$ 是 build-rate **强度旋钮**（不是 on/off 开关）：$(g_K^{\max}-g_K)$ 因子让 build 内禀有界、不靠 clip 兜底，
定点 $g_K^{*}=g_K^{\max}\,k_K f\tau_K/(1+k_K f\tau_K)<g_K^{\max}$；同样活动下 $k_K$ 越大 $g_K$ 升得越快、稳态越高。
$k_K=0\Rightarrow$ build 项恒为 0，$g_K$ 只衰减、从 0 出发恒为 0（off-by-default 字节奇偶）。

驱动（第一版只随 **E 活动**累积、只作用 E 细胞）：

$$
a_K(x,t)=K_K*r_E(x,t)
$$

施加：

$$
I_i^{\text{net}}\mathrel{-}=\eta_K\,g_K(x_i,t)\quad(i\in E)
$$

**关键核宽关系（承重，TDD 锁）**：$\sigma(K_q)>\sigma(K_K)$——去抑制足迹**宽**、疲劳足迹**窄**。
直觉：$q_I$ 代表局部网络状态 / 抑制可用度，可影响活动区**周围**（让周边许可度上升）；
$g_K$ 是使用依赖性恢复，应更**局部**，只压真正高放电的轴向通道。注意 $g_K$ **单独**通常只让轴向活动**停**、
不让离轴接管；它必须与 $q_{I,\text{offaxis}}\!\downarrow$ 配合才能破轴（§B5.0 机制链）。

- $\tau_K,k_K,g_K^{\max}$：衰减时间常数 / **build-rate 强度旋钮**（$k_K=0$ 即关 → 字节奇偶）/ 上界。
- $K_K,\sigma_K$：疲劳活动感知核 / 宽度（**窄**）。
- $\eta_K$：$g_K$ 进膜的耦合强度。

### B5.4 可选 E→E depression $D_{EE}(x,t)$（stage-2，**本轮只写公式不实现**）

$$
\partial_t D_{EE}(x,t)=\frac{1-D_{EE}(x,t)}{\tau_D}-k_D\,f\!\big(a_D(x,t)\big)\,D_{EE}(x,t),
\qquad D_{\min}\le D_{EE}\le 1
$$

$a_D=K_D*r_E$；施加到 E→E 突触（presynaptic 位置疲劳→relay 减弱）：

$$
J_{ij}^{EE,\text{eff}}(t)=D_{EE}(x_j,t)\,J_{ij}^{EE}
$$

**口径**：$D_{\min}\approx0.5\sim0.8$（**不要**掉到 0，否则只是把通道关掉、不是发作样招募）。
只有当 $q_I+g_K$ 不足以降 axis score 时才开。本轮**无实现、无 TDD**。

### B5.5 慢变量全开时的膜方程

$$
\tau_m^a\dot V_i=-V_i+I_i^E-q_I(x_i,t)\,I_i^I-\eta_K\,g_K(x_i,t)
\qquad(\text{+ stage-2 时 }D_{EE}\text{ 改 }J^{EE})
$$

基线 $q_I\equiv1,\,g_K\equiv0$ 退回 §A1 的 $\tau_m^a\dot V_i=-V_i+I_i^E-I_i^I$（**字节奇偶**：等价 `slow=None`）。

### B5.6 四类状态 operational 判据（验收 gate）

**承重纪律**：$S_{\text{axis}}$ 必须用**源空间逐细胞 onset 梯度**算（§4 方法学锁；复用
`src/sef_hfo_snn_metrics.py::onset_axis`），**不是**触点空间方向、**不是** collision、**不是**放电拉伸。

每个事件算 6 个量：

$$
R_{\text{area}}=\tfrac1N\textstyle\sum_x\mathbf 1[A(x)>\theta_A]
\quad(\text{招募面积}),\qquad
T_{\text{event}}=t_{\text{off}}-t_{\text{on}}\quad(\text{时长})
$$

$$
S_{\text{axis}}=\frac{|v_{\text{event}}\cdot\hat u_{EE}|}{\|v_{\text{event}}\|},
\;\; v_{\text{event}}=\texttt{onset\_axis}(\text{pos}_E,\text{onset})
\qquad
F_{\text{offaxis}}=\frac{\sum_{x\notin\text{corridor}}A(x)}{\sum_x A(x)}
$$

$$
G_{\text{PR}}=\frac{\big(\sum_x A(x)\big)^2}{N\sum_x A(x)^2}\;(\text{globality}),\qquad
\text{recovery}=\big[\bar r_E(\text{post})\le r_{\text{base}}+m\sigma_{\text{base}}\big]
$$

四类（结构锁定，阈值待标定）：

| 状态 | $R_{\text{area}}$ | $S_{\text{axis}}$ | $F_{\text{offaxis}}/G_{\text{PR}}$ | recovery |
| --- | --- | --- | --- | --- |
| **interictal axial event** | 小 | 高 | 低 | 是 |
| **expanded axial recruitment** | 中–大 | **仍高** | 低–中 | 是 |
| **ictal-like recruitment candidate** | 大 | **明显↓** | **明显↑** | 是 |
| **runaway** | 大 | 任意 | 任意 | **否** |

**最关键边界**：expanded axial $\neq$ ictal-like。ictal-like 是**四条件 AND**：$R_{\text{area}}$ **大**（$\ge$ `area_large`）
$\wedge$ $S_{\text{axis}}$ **明显↓**（$<$ `axis_broken`）$\wedge$（$F_{\text{offaxis}}$↑ $\vee$ $G_{\text{PR}}$↑）$\wedge$ recovery；
缺任意一条都不是。即使事件很强，只要 $S_{\text{axis}}$ 仍高 **且** $F_{\text{offaxis}}$ 仍低 **且** $G_{\text{PR}}$ 没明显升，
它就只是 expanded axial。**尤其：小事件（$R_{\text{area}}$ 小）即使破轴/离轴也不是 ictal-like**——size 是必要条件，
光破轴不够（否则一个局部偶发的离轴噪声 blip 会被误读成发作样）。
**坏数据回归**：有限 onset 数 $<\texttt{min\_onsets}$ 或 $S_{\text{axis}}=$ NaN $\Rightarrow$ `INSUFFICIENT`，
**不分类**（绝不默认成 ictal-like）。

### B5.7 相图（两层）

**proxy（在线，从场轨迹）**：区域 $R\in\{\text{axis},\text{offaxis},\text{global}\}$ 的有效招募压力

$$
P_R(t)=\log(\mathrm{lgr}_R)-\big\langle\log(q_I(x,t)+\epsilon)\big\rangle_R-\beta_K\big\langle g_K(x,t)\big\rangle_R,
\qquad
X=P_{\text{axis}}-P_{\text{offaxis}},\;\; Y=P_{\text{global}}
$$

**符号约定**：$q_I\!\downarrow$（去抑制）$\Rightarrow -\langle\log q_I\rangle\!\uparrow\Rightarrow$ 该区 pressure $P\!\uparrow$。故 **axis-dominant**
（轴更去抑制、$q_{I,\text{axis}}$ 低）$\Rightarrow X=P_{\text{axis}}-P_{\text{offaxis}}>0$；**off-axis 追上**（$q_{I,\text{offaxis}}\!\downarrow$）
$\Rightarrow X\!\downarrow$（可转负）= 破轴。$Y$ 取 **global** region pressure（与下面 spectral $Y_{\text{spec}}=\alpha_{\text{global}}$ 对齐、
便于 overlay；用 global mask，非 dead arg）；$Y\!\uparrow$ 且不回 = runaway 风险。

**spectral（冻结 Jacobian，复用 §B 线 `src/topic4_m3b_spectral_phase.py`）**：

$$
X_{\text{spec}}=\alpha_{\text{axis}}-\alpha_{\text{global}},\quad
Y_{\text{spec}}=\alpha_{\text{global}},\qquad
\alpha_\bullet=\max_{m\in\bullet}\mathrm{Re}(\lambda_m)
$$

B 线相图 overlay 仍受 `src/sef_hfo_m3_interface.py`（D1 归一化轴 / D2 5%越界 / D3 recovery 无损投影）
fail-closed 合同约束；空间场的接口扩展**延后**（合同 §9 deferred）。

### B5.8 红线 / 口径 / 延后

- **仍是 mechanism screen**；ictal-like 是检出标签，**禁**“已证明发作机制 / Abbott 成立 / v2 过间期-发作两态”。
- **破轴是否发生 = 经验问题**，移交延后的 **ablation**（A=固定 scaffold 无慢变量；B=只 $q_I$；C=$q_I+g_K$ 主模型；
  D=+$D_{EE}$）。本轮**不建 ablation runner、不写机制主张**。
- **风险（用户 §10）**：$q_I$ 耗竭最强处往往**就是** axis，可能只**放大** axis 而不破轴。破轴靠
  $\sigma_q>\sigma_K$ 的核宽差 + $g_{K,\text{axis}}\!\uparrow$ 的 balance；调参序：加宽 $K_q$ → 降 $q_{\min}$ →
  加强 $g_K$ build-up →（仍不够才）开 $D_{EE}$。**若只见 expanded axial，不是模型失败，是 balance 不足。**
- 模块：场动力学 `src/snn_engine/slow_field.py::SpatialSlowField`（实现 `simulate_kick` 的 slow 协议）；
  事件读出 + 四类分类器 + proxy 相图 `src/topic4_m3a_v2_phenotype.py`。

---

## B6. M3A-v2.2 全局抑制性恢复变量 `h_G(t)`（global inhibitory recovery）

> **本节状态**：公式锁定（2026-06-28）。实验结构 / 硬合同（C1–C9）见
> `docs/superpowers/specs/2026-06-28-sef-hfo-m3a-v2.2-global-recovery-design.md`；
> 实现计划见 `docs/superpowers/plans/2026-06-28-sef-hfo-m3a-v2.2-global-recovery-plan.md`。
> 仍是 **mechanism screen**：`h_G` 是 global recovery/restraint 变量，**非**发作机制 validation。
> OFF-by-default：`use_hG=False` 把 `h_G` 对膜电流与 step 的耦合**硬门控为零** ⇒ **engine-output 与
> `slow=None` 字节一致**（当 `q_I/g_K` 也中性时）。注意是**有效耦合为零**、非内部 state 数学恒等 0
> （`hG_init`≠0 时标量 `h_G` 本身非 0，但不进任何输出）。

把"回来"从局部 `g_K`（§B5.3）分出去：`h_G(t)` 是**全局标量**，只看**网络整体活动**（非轴/旁分区）。
机制链 `ignite/expand (q_I↓) → redirect/limit (g_K↑) → terminate/recover (h_G↑)` 里它只管最后一步。

### B6.1 传感器（快 EMA 率场 `r̃_E`，时间常数 `τ_s`，独立于 §B5.1 的 `τ_a`）

$$
M(t)=\langle \tilde r_E\rangle_x,\qquad
B(t)=\Big\langle \sigma\!\big(\tfrac{\tilde r_E-r_A}{\Delta_A}\big)\Big\rangle_x,\qquad
\Pi(t)=\frac{\big(\sum_x\tilde r_E\big)^2}{N_x\sum_x\tilde r_E^2+\epsilon}
$$

`M`=总活动强度，`B`=软参与面积（`σ`=logistic，非硬阈），`Π`=空间 participation/globality
（单热点→低、大范围均匀→高）。

### B6.2 平滑 AND 触发 + 有界 build ODE + 膜耦合（仅 E）

$$
\chi_G=H_{n_M}(M;M_{50})\,H_{n_B}(B;B_{50})\,H_{n_\Pi}(\Pi;\Pi_{50}),\qquad
H_n(z;z_{50})=\frac{z^n}{z^n+z_{50}^n}
$$

$$
\dot h_G=-\frac{h_G}{\tau_G}+k_G\,\chi_G\,(h_G^{\max}-h_G),\quad 0\le h_G\le h_G^{\max},\qquad
I^{\text{net}}_i\mathrel{-}=\eta_G\,h_G\ \ (i\in E)
$$

`k_G=0` 即**不 build**（不是不 decay：`-h_G/τ_G` 衰减项始终在）。小局部轴向事件 `χ_G≈0`（不触发）；
近失控时 `M/B/Π` 都上来、`χ_G` 自然变大、`h_G` 接管。`I` 细胞先不加 `h_G`。

### B6.3 可选 q 回灌（仅 arm F，单独消融）+ 相图新 Y + clamp/surrogate

$$
\partial_t q_I \mathrel{+}= \lambda_G\,h_G\,(1-q_I)\quad(\text{arm E: }\lambda_G=0),\qquad
Y^{\text{new}}=P_{\text{global}}-\beta_G h_G
$$

`X` 不变（`-β_G h_G` 对 axis/offaxis/global 一视同仁、差分抵消）→ **`h_G` 不伪造破轴**。
`hG_script` 非空时跳过 ODE、`h_G=\text{clip}(hG\_script(t),0,h_G^{\max})`（恒定钳制相图 / onset-gated 假 `h_G`）。

- `τ_s,r_A,Δ_A`：快 EMA 时间常数 / 软面积参考 / 斜率。`M50/B50/Π50,n_*`：Hill 半触发 / 指数。
- `τ_G,k_G,h_G^max,η_G`：衰减时间常数 / build 强度旋钮（`k_G=0`→不 build → 字节奇偶）/ 上界 / 膜耦合强度。
- `λ_G`：q 回灌强度（arm F secondary）。`β_G`：相图 Y 中 `h_G` 权重。
- 模块：`src/snn_engine/slow_field.py::SpatialSlowField`（`h_G` 态 + `hG_script`）；传感器纯函数
  `src/topic4_m3a_v2_2_sensors.py`；持续驱动 `src/topic4_m3a_v2_2_protocol.py`（runner 级 `nu_signal_fn`，不碰引擎）。

---

## 出处

- 底层 SNN：`src/snn_engine/{model,params,connectivity_rot,kick_probe}.py`
- 阈值 core：`src/sef_hfo_heterogeneity.py::sample_core_field`
- 自发读出：`scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`
- Fig4/5：`scripts/paper_figures/{plot_fig_subject_snn,plot_fig5_core_model_s3_brakeoff}.py`
- M3A v1：`src/snn_engine/slow_vars.py` / `RegionalResource`（标量 $q_{\text{core}}/q_{\text{global}}$）
- M3A-v2 空间场：`src/snn_engine/slow_field.py` / `SpatialSlowField`；读出 `src/topic4_m3a_v2_phenotype.py`；
  源空间 onset 仪器 `src/sef_hfo_snn_metrics.py`；率场卷积 `src/sef_hfo_field.py`；谱相图 `src/topic4_m3b_spectral_phase.py`
