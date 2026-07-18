# 空间易激场 SNN 模型与虚拟 SEEG 读出

为评估患者内稳定传播轴能否由局部易激性和各向异性连接共同读出，我们构建二维兴奋性-抑制性 spiking neural network（SNN）模型。模型输入为空间连接参数、局部阈值场和背景噪声驱动，输出为神经元 spike、虚拟 SEEG envelope、参与触点 onset 和事件传播方向。

## 模型规模与神经元组成

SNN 定义在边长为 \(L\) 的二维薄片上，总神经元数由薄片面积和神经元空间密度决定。本文默认工作点为 \(L=20\) mm、\(\rho=100\) neurons/mm\(^2\)，总神经元数和 E/I 组成定义为：

\[
N=\mathrm{round}(\rho L^2),
\]

\[
N_E=\mathrm{round}(0.8N),\qquad N_I=N-N_E.
\]

在该工作点下，\(N=40{,}000\)，其中兴奋性神经元 \(N_E=32{,}000\)，抑制性神经元 \(N_I=8{,}000\)。神经元位置在二维薄片内均匀随机采样；兴奋性神经元和抑制性神经元分别按 80% 和 20% 的比例分配。

局部异质性由核心区内兴奋性神经元的阈值分布实现。低阈值核心定义为空间上选定并赋予低阈值分布的一组兴奋性神经元；在默认核心半径 \(r_{\mathrm{core}}=1.5\) mm 和 \(\rho=100\) neurons/mm\(^2\) 下，单个核心区的低阈值兴奋性神经元期望数量为：

\[
\mathbb{E}[n_{\mathrm{core},E}]
\approx 0.8\rho\pi r_{\mathrm{core}}^2
\approx 565.
\]

双核心条件下，两个核心区合计约 \(1.1\times10^3\) 个低阈值兴奋性神经元；实际数量由随机空间采样和核心位置确定。

## 单神经元动力学

每个神经元采用 current-based leaky integrate-and-fire 动力学估计膜电位变化。兴奋性和抑制性神经元共享同一形式的膜方程，但使用不同膜时间常数和不应期：

\[
\tau_m^a\frac{dV_i}{dt}=-V_i+I_i^E-I_i^I,
\qquad a\in\{E,I\}.
\]

当 \(V_i\) 达到阈值 \(V_{\theta,i}\) 时，神经元发放 spike，随后 \(V_i\) 重置为 \(V_r\) 并进入不应期。基线参数为 \(V_\theta=18\) mV、\(V_r=11\) mV；兴奋性神经元使用 \(\tau_m^E=20\) ms、\(\tau_{\mathrm{ref}}^E=2\) ms，抑制性神经元使用 \(\tau_m^I=10\) ms、\(\tau_{\mathrm{ref}}^I=1\) ms。

突触输入由延迟双指数电流估计。对突触类型 \(X\)，突触门控变量 \(s_i^X\) 和突触电流 \(I_i^X\) 满足：

\[
\tau_r^X\frac{ds_i^X}{dt}
=-s_i^X+\sum_j J_{ij}^X\delta(t-t_j-d_{ij}),
\]

\[
\tau_d^X\frac{dI_i^X}{dt}=-I_i^X+s_i^X.
\]

其中 \(J_{ij}^X\) 为突触权重，\(t_j\) 为突触前神经元 spike 时间。传导延迟由神经元间距离估计：

\[
d_{ij}=\tau_0+\frac{\|x_i-x_j\|}{v_{\mathrm{axon}}}.
\]

## 背景驱动

自发事件由空间均一的背景输入和 Ornstein-Uhlenbeck 噪声共同驱动。外部输入率定义为：

\[
\nu_{\mathrm{ext}}(t)=\left[r_{\mathrm{ext}}\nu_\theta+\xi(t)\right]_+,
\]

\[
d\xi=-\frac{\xi}{\tau_n}\,dt+\sigma_\xi\,dW_t.
\]

其中 \(\nu_\theta\) 为无 recurrent input 时达到阈值所需的参考输入率，\(r_{\mathrm{ext}}\) 为背景驱动比例。默认工作点使用 \(r_{\mathrm{ext}}=0.6\)，使背景输入处于亚阈值范围，并由噪声提供自发点火机会。

## 空间连接结构

空间传播结构由固定入度连接和 E-to-E 各向异性连接核共同定义。每个目标神经元从不同源群体接收固定数量的输入连接，默认入度为：

\[
C_{EE}=800,\qquad C_{IE}=800,\qquad C_{EI}=200,\qquad C_{II}=200.
\]

其中 \(C_{ab}\) 表示目标群体 \(a\) 从源群体 \(b\) 接收的平均连接数。E-to-E 连接使用旋转椭圆核；其他连接使用各向同性核。

E-to-E 连接核先将神经元间位移 \(\Delta x=(z_1,z_2)\) 投影到连接长轴和垂轴方向。给定连接长轴方向 \(\theta_{EE}\)，定义：

\[
u=\cos\theta_{EE}z_1+\sin\theta_{EE}z_2,
\]

\[
v=-\sin\theta_{EE}z_1+\cos\theta_{EE}z_2.
\]

E-to-E 连接概率随椭圆距离衰减：

\[
P(j\to i)\propto
\exp\left[
-\sqrt{\left(\frac{u}{\ell_\parallel}\right)^2+
\left(\frac{v}{\ell_\perp}\right)^2}
\right],
\]

\[
\ell_\parallel=\ell_{EE}\sqrt{\mathrm{AR}},
\qquad
\ell_\perp=\frac{\ell_{EE}}{\sqrt{\mathrm{AR}}}.
\]

默认参数为 \(\ell_{EE}=0.380\) mm、\(\mathrm{AR}=2\)。患者特异性仿真中，\(\theta_{EE}\) 由患者内传播模板的共同空间轴确定；示意性仿真中，\(\theta_{EE}=45^\circ\)。

## 双低阈值核心

局部易激性通过两个空间低阈值核心区定义。两个核心区的中心放置在 E-to-E 连接长轴两端：

\[
c_\pm=c_0\pm s_{\mathrm{sep}}\frac{L}{2}\hat u_{EE}.
\]

默认分离比例为 \(s_{\mathrm{sep}}=0.7\)。每个核心区选择半径 \(r_{\mathrm{core}}=1.5\) mm 内的兴奋性神经元，并对这些神经元重新抽样阈值；核心区外兴奋性神经元和所有抑制性神经元保持基线阈值。

核心区兴奋性神经元的阈值由截断正态分布估计。阈值规则为：

\[
V_{\theta,i}=
\begin{cases}
18\ \mathrm{mV}, & i\notin \mathrm{core}\ \mathrm{or}\ i\in I,\\
v,\quad v\sim\mathcal{N}(\mu_{\mathrm{core}},\sigma_{\mathrm{core}}^2),\ v\ge V_r,
& i\in \mathrm{core}\ E.
\end{cases}
\]

默认核心参数为 \(\mu_{\mathrm{core}}=17.5\) mV、\(\sigma_{\mathrm{core}}=1.0\) mV。两个核心区使用相同的 \(\mu_{\mathrm{core}}\) 和 \(\sigma_{\mathrm{core}}\)，但阈值抽样彼此独立。

## 虚拟 SEEG 读出

虚拟 SEEG 读出将兴奋性神经元 spike 转换为触点级 envelope。虚拟触点按患者坐标或规则电极杆坐标放置；规则读出使用 4 mm contact pitch，并包括与 E-to-E 长轴平行和垂直的电极杆。第 \(k\) 个触点的 envelope 定义为：

\[
\mathrm{env}_k(t)=\sum_{j\in E} w_{kj}r_j(t).
\]

其中 \(r_j(t)\) 为兴奋性神经元 \(j\) 的局部发放密度，\(w_{kj}\) 为触点 \(k\) 对神经元 \(j\) 的空间权重。权重随神经元到触点的距离衰减：

\[
w_{kj}\propto
\exp\left(-\frac{d_{kj}^2}{2\sigma_w^2}\right).
\]

参与触点由 envelope 峰值阈值确定。对每个事件窗口，若触点 envelope 峰值超过背景 floor 加 10% 动态范围 margin，则该触点记为参与触点；参与触点 onset 定义为 envelope 首次达到自身峰值 50% 的时间：

\[
t_k=\min\{t:\mathrm{env}_k(t)\ge0.5\,\mathrm{peak}_k\}.
\]

## 传播方向估计

事件传播方向由早参与触点和晚参与触点的空间质心差估计。对每个事件，先按触点 onset 从早到晚排序，再取最早 \(k_{\mathrm{dir}}\) 个触点和最晚 \(k_{\mathrm{dir}}\) 个触点：

\[
a_{\mathrm{event}}
=
\mathrm{centroid}(\mathrm{late}\ k_{\mathrm{dir}}\ \mathrm{contacts})
-
\mathrm{centroid}(\mathrm{early}\ k_{\mathrm{dir}}\ \mathrm{contacts}).
\]

默认 \(k_{\mathrm{dir}}=3\)，因此方向估计要求事件至少包含 7 个参与触点。事件方向随后投影到 E-to-E 连接长轴，得到沿轴方向的正向或反向读出标签。

## 空间慢变量场

空间慢变量场用于把 spike 活动转换为同一二维薄片上的状态变量。薄片被离散为 \(n_g\times n_g\) 个格点，默认 \(n_g=32\)。每个时间步先将兴奋性和抑制性 spike 按空间位置计数，再用宽度为 \(\sigma_r\) 的二维高斯核平滑，并用时间常数 \(\tau_a\) 做指数滑动平均：

\[
\tilde r_E(x,t)=K_{\sigma_r}*
\sum_{i\in E}S_i(t)\delta(x-x_i),
\qquad
\tilde r_I(x,t)=K_{\sigma_r}*
\sum_{i\in I}S_i(t)\delta(x-x_i),
\]

\[
r_X(x,t+\Delta t)
=r_X(x,t)+\alpha_a\{\tilde r_X(x,t)-r_X(x,t)\},
\qquad
\alpha_a=1-\exp(-\Delta t/\tau_a),
\quad X\in\{E,I\}.
\]

默认空间平滑宽度为 \(\sigma_r=0.5\) mm，默认慢场读出使用 \(\tau_a=100\) ms；连续 runaway 转换分析使用 \(\tau_a=20\) ms，以保留重复脉冲之间的短时活动累积。

慢变量开启时，兴奋性神经元的净输入电流由局部抑制资源 \(q_I(x,t)\) 和恢复电流 \(g_K(x,t)\) 调制。兴奋性神经元使用：

\[
\tau_m^E\frac{dV_i}{dt}
=-V_i+I_i^E-q_I(x_i,t)I_i^I-\eta_K g_K(x_i,t),
\qquad i\in E.
\]

抑制性神经元保持基线电流形式：

\[
\tau_m^I\frac{dV_i}{dt}
=-V_i+I_i^E-I_i^I,
\qquad i\in I.
\]

## 抑制资源场

抑制资源场用于估计局部活动对有效抑制许可度的慢性改变。资源变量 \(q_I(x,t)\) 初始为 \(q_I(x,0)=1\)，并被约束在 \([q_{\min},1]\) 内；对应的 permissivity 读出定义为：

\[
p_I(x,t)=1-q_I(x,t).
\]

资源耗竭驱动由兴奋性和抑制性发放率场加权后再空间平滑得到：

\[
a_q(x,t)
=K_{\sigma_q}*
\left[\eta_E r_E(x,t)+\eta_I r_I(x,t)\right].
\]

耗竭函数采用有下限的饱和形式：

\[
f_q(a)=
\frac{[a-a_{0,q}]_+}{a_{50,q}+[a-a_{0,q}]_+}.
\]

抑制资源场的动力学定义为：

\[
\frac{\partial q_I(x,t)}{\partial t}
=
\frac{1-q_I(x,t)}{\tau_q}
-k_q f_q(a_q(x,t))q_I(x,t),
\qquad q_{\min}\le q_I(x,t)\le 1.
\]

默认资源参数为 \(\tau_q=5000\) ms、\(\sigma_q=1.5\) mm、\(q_{\min}=0.25\)、\(\eta_E=0.3\)、\(\eta_I=1.0\)、\(a_{0,q}=0\)、\(a_{50,q}=1\)。参数 \(k_q\) 控制活动依赖耗竭强度；连续 runaway 转换轨迹使用 \(k_q=0.3\) 和 \(q_{\min}=0.25\)。

## 恢复电流场

恢复电流场用于估计活动后的局部慢外向电流。恢复驱动由兴奋性发放率场平滑得到：

\[
a_K(x,t)=K_{\sigma_K}*r_E(x,t).
\]

对应的饱和函数为：

\[
f_K(a)=
\frac{[a-a_{0,K}]_+}{a_{50,K}+[a-a_{0,K}]_+}.
\]

恢复电流场的动力学定义为：

\[
\frac{\partial g_K(x,t)}{\partial t}
=
-\frac{g_K(x,t)}{\tau_K}
+k_K f_K(a_K(x,t))(g_K^{\max}-g_K(x,t)),
\qquad 0\le g_K(x,t)\le g_K^{\max}.
\]

默认恢复参数为 \(g_K(x,0)=0\)、\(\tau_K=5000\) ms、\(\sigma_K=0.5\) mm、\(g_K^{\max}=1\)、\(a_{0,K}=0\)、\(a_{50,K}=1\)。参数 \(k_K\) 控制恢复电流建立速度，\(\eta_K\) 控制恢复电流进入膜方程的强度。连续 runaway 轨迹的轴向 fatigue 读出使用 \(k_K=1.0\)、\(\eta_K=0\) 来记录 \(\bar g_{K,\mathrm{axis}}(t)\)；耦合型 qI/gK 条件通过 \(\eta_K>0\) 评估恢复电流反馈。慢变量分析固定 \(\sigma_q>\sigma_K\)，使抑制资源变化具有较宽空间 footprint，而恢复电流具有较局部的 footprint。

## 慢变量轨迹读出

慢变量轨迹读出用于比较同一连续仿真内的全局资源、局部资源下限和轴向恢复强度。每个时间点记录全场平均抑制资源、全场最低抑制资源和轴向恢复电流：

\[
\bar q_I(t)=\frac{1}{|\Omega|}\sum_{x\in\Omega}q_I(x,t),
\qquad
q_I^{\min}(t)=\min_{x\in\Omega}q_I(x,t),
\]

\[
\bar g_{K,\mathrm{axis}}(t)
=
\frac{1}{|\Omega_{\mathrm{axis}}|}
\sum_{x\in\Omega_{\mathrm{axis}}}g_K(x,t).
\]

其中 \(\Omega\) 为整个二维薄片格点集合，\(\Omega_{\mathrm{axis}}\) 为沿 E-to-E 长轴定义的轴向走廊。事件前慢状态用脉冲开始前的 \(\bar q_I(t)\)、\(q_I^{\min}(t)\) 和 \(\bar g_{K,\mathrm{axis}}(t)\) 估计；事件后慢状态用响应窗口末端后的同一组变量估计。

## 连续 runaway 转换协议

连续 runaway 转换协议用于估计重复局部输入下，同一 SNN scaffold 的短暂局部事件、慢变量累积和持续高放电状态。该协议使用缩放薄片 \(L=10\) mm 和 \(\rho=100\) neurons/mm\(^2\)，对应 \(N=10{,}000\)、\(N_E=8{,}000\)、\(N_I=2{,}000\)。仿真在一条连续时间轨迹中进行，两个低阈值核心按顺序交替接受局部脉冲输入。第 \(m\) 个脉冲的起止时间定义为：

\[
t_m=t_{\mathrm{start}}+m\Delta t_{\mathrm{pulse}},
\qquad
t\in[t_m,t_m+\tau_{\mathrm{pulse}}),
\qquad m=0,\ldots,n_{\mathrm{pulse}}-1.
\]

脉冲中心按两个核心交替指定：

\[
c_m=
\begin{cases}
c_-, & m\ \mathrm{even},\\
c_+, & m\ \mathrm{odd}.
\end{cases}
\]

默认连续轨迹使用 \(t_{\mathrm{start}}=130\) ms、\(\Delta t_{\mathrm{pulse}}=135\) ms、\(\tau_{\mathrm{pulse}}=18\) ms、\(n_{\mathrm{pulse}}=9\)。脉冲作用于当前核心中心 \(c_m\) 周围半径 \(R_{\mathrm{pulse}}\) 内的兴奋性神经元，并在脉冲窗口内提高其外部输入率：

\[
\nu_{\mathrm{ext},i}^{\mathrm{pulse}}(t)
=
\nu_{\mathrm{ext}}(t)
+r_{\mathrm{pulse}}\,
\mathbf 1\!\left[
\|x_i-c_m\|\le R_{\mathrm{pulse}}
\right]
\mathbf 1\!\left[
t_m\le t<t_m+\tau_{\mathrm{pulse}}
\right],
\qquad i\in E.
\]

runaway 转换分析固定慢资源参数 \(k_q=0.3\)、\(q_{\min}=0.25\)，并使用 \(r_{\mathrm{pulse}}=1.6\) 定义局部驱动强度。每次脉冲后的响应窗口为 85 ms；窗口内估计平滑兴奋性群体发放率峰值、参与兴奋性神经元比例、二维活动场和虚拟 SEEG envelope。

runaway onset 由平滑兴奋性群体发放率定义。先用 20 ms 窗口平滑全体兴奋性神经元发放率 \(R_E(t)\)，得到 \(\tilde R_E(t)\)；若 \(\tilde R_E(t)\) 在长度为 \(T_{\mathrm{sustain}}=100\) ms 的窗口内至少 80% 时间点超过 \(R_{\mathrm{run}}=120\) Hz，则该窗口起点定义为：

\[
t_{\mathrm{run}}
=
\min_t
\left\{
t:
\frac{1}{T_{\mathrm{sustain}}}
\int_t^{t+T_{\mathrm{sustain}}}
\mathbf 1[\tilde R_E(s)\ge R_{\mathrm{run}}]\,ds
\ge 0.8
\right\}.
\]

pre-runaway 局部事件由 \(t_{\mathrm{run}}\) 前的脉冲响应窗口估计。若窗口峰值位于 10--120 Hz，且至少 2% 的兴奋性神经元在窗口内发放，则该窗口计为一次局部响应；这些响应用于估计事件大小与脉冲前 permissivity 的关系：

\[
\mathrm{size}_m=
\frac{1}{N_E}
\sum_{i\in E}
\mathbf 1\!\left[
\exists t\in[t_m,t_m+85\ \mathrm{ms}):S_i(t)=1
\right],
\qquad
p_{I,m}^{\mathrm{pre}}=
1-\bar q_I(t_m^-).
\]

## 事件形态分类

仿真事件按源空间招募范围、轴向主导性、离轴或全局成分以及恢复情况分类。每个事件计算招募面积、事件时长、source-space onset 轴向分量、离轴招募比例、globality 和恢复指标。

事件分类采用四类标签。interictal axial event 表示局部、短时、沿 E-to-E 长轴传播并恢复的事件；expanded axial recruitment 表示招募范围扩大但轴向主导仍强的事件；ictal-like recruitment candidate 表示大范围招募、轴向主导下降、离轴或全局成分增强并恢复的事件；runaway 表示持续高放电且未恢复的事件。该分类结果用于比较不同连接、阈值和慢变量参数下的模型状态。
