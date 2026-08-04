# Topic 5 / Figure 6：source-conditioned shared-scaffold RNN v0.3

**状态**：implementation-ready amendment  
**继承**：v0.2 的患者内切分、target seal、loss 分解、ordinary/static 对照、统计与 Figure A--E 合同。

## 1. 为什么替代 v0.2

v0.2 把对称 scaffold 直接当成 next-contact transition operator。首批完整患者显示该模型
接近 static，且真实顺序与 rank-shuffle 几乎重合。这不是论文假设要求的映射：底层共享结构可以
近似对称，但观测转移还需要由事件起点决定方向，并允许对候选 contact 产生正负竞争。

该批结果只保留为 `symmetric-only diagnostic`，不得进入正式 Figure 6。

## 2. 唯一结构模型

每位患者学习一个有符号的一维 contact coordinate (s_i)，中心化并标准化。它产生两个连续端点
membership：

\[
a_i=\sigma(-s_i/T),\qquad b_i=\sigma(s_i/T).
\]

同一对 membership 同时构造对称与反对称的 rank-2 operator：

\[
K^S_{\mathrm{axis}}=ba^\top+ab^\top,
\qquad
K^A_{\mathrm{axis}}=ba^\top-ab^\top.
\]

其中 (K^S=(K^S)^\top)、(K^A=-(K^A)^\top)，且二者不是两套独立路径参数。固定 shaft-local
graph 只加入 (K^S)。用同一个 symmetric degree normalization 得到 (W^S,W^A)。

每场事件第一 rank set (x_0) 只依赖已观察 source，因果地产生方向状态：

\[
d_e=\tanh\left[\kappa\,\operatorname{mean}_{i\in x_0}(a_i-b_i)\right].
\]

source 在 (a) 端时 (d_e>0)，(W^S+d_eW^A) 偏向 (b) 端；source 在 (b) 端时符号反转，
但共享 scaffold 与全部参数不变。递归固定为：

\[
P_{t+1}=\rho_PP_t+W^Sx_t+\lambda_A d_eW^Ax_t,
\]

\[
R_{t+1}=\rho_RR_t+W^Sx_t,
\]

\[
z_{t+1}=b+\beta_PP_{t+1}-\beta_RR_{t+1}+m_t.
\]

禁止 dense contact decoder、独立 forward/reverse operator、A/B label、mean rank 和 ictal target。
STOP/cardinality heads仍只读 permutation-invariant summaries。

## 3. 解释边界

- (W^S) 是患者特异 effective scaffold，不是解剖连接矩阵。
- (W^A) 是由同一 rank-2 端点基底派生的 source-conditioned flow，不是第二条路径。
- (P/R) 是 rank-step propagation/restraint state，不解释为细胞级 E/I 或秒级时间常数。
- 经验 A/B 只在冻结后 read-back；它不是训练金标准。

## 4. 训练与确认

先仅用三位既定 development patients 的 validation20 比较少量学习率
`3e-4 / 1e-3 / 3e-3`；模型、loss、rank=2 与其余配置不变。按三人 validation contact NLL
中位数冻结一个 learning rate。随后 34 人完整训练；31 位 development-excluded 为正式确认，
34 位为全队列描述。不得根据 early-ictal 结果选择学习率或模型。

## 5. Figure 6 不变的主顺序

A 为本模型结构；B 为同一模型两端 source 的 A-like/B-like observed-vs-rollout 时序；C 为 34 人
next-contact 结果并同时标 31 人确认统计；D 为预冻结 E1146 的两张 RNN 场与患者中位 early-ictal
场；E 为 15 人 target-free correspondence 及 structured-vs-ordinary 配对结果。
