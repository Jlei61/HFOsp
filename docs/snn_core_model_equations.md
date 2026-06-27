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

---

## 出处

- 底层 SNN：`src/snn_engine/{model,params,connectivity_rot,kick_probe}.py`
- 阈值 core：`src/sef_hfo_heterogeneity.py::sample_core_field`
- 自发读出：`scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`
- Fig4/5：`scripts/paper_figures/{plot_fig_subject_snn,plot_fig5_core_model_s3_brakeoff}.py`
- M3A：`src/snn_engine/slow_vars.py` / `RegionalResource`
