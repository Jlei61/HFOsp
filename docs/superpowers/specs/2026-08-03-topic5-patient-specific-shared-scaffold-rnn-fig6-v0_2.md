# Topic 5 / Figure 6：patient-specific shared-scaffold propagation RNN v0.2

**状态**：superseded by the source-conditioned v0.3 amendment below  
**替代**：`2026-08-03-topic5-patient-specific-target-free-rnn-bridge-v0_1.md`

## 1. 唯一科学问题

本合同检验：

\[
\boxed{
\text{患者内间期 contact-rank sequences}
\rightarrow
\text{共享传播 scaffold + event state}
\rightarrow
\text{患者特异传播结构}
}
\]

以及冻结模型派生的同一组双向空间场，是否与同一患者 clinical onset 后早期
broadband energy field 存在 target-free 跨状态对应。

这里的 `target-free` 指：模型、超参数、checkpoint、source pools、rollout horizon
和两张方向场都必须在读取 early-ictal energy values 之前冻结。它不表示每名患者都必须
阳性，也不表示经验 A/B 是训练金标准。

主张层级固定为：

1. 患者内自监督模型能否预测 held-out 间期 next contact；
2. structured model 能否比 static 和 ordinary model 更好地恢复 held-out 传播结构；
3. structured model 的冻结双向场能否在患者层面对应 early-ictal energy field。

不检验跨患者读出，不从一个患者学习另一个患者的参数，也不寻找唯一细胞级 E/I 机制。

## 2. 数据、切分与防泄漏

### 2.1 患者内切分

- 间期主队列为现有 masked rank dataset 的 34 人；非参与 contact 必须保持 masked。
- 每位患者按事件发生时间严格切为 `fit60 / validation20 / test20`。
- `fit60`：拟合 static prior 和模型参数。
- `validation20`：在冻结的训练配置下选择 checkpoint；不得与 test 合并，也不重新选择学习率。
- `test20`：只作一次冻结后的最终间期评价。
- 每位患者独立训练；任何其他患者的数据都不得进入其参数估计。

### 2.2 Primary discovery 禁止输入

以下变量不得进入 static、ordinary 或 structured 模型的训练输入：

- empirical A/B template、A/B label 或 A/B axis；
- contact empirical mean rank、conditional mean rank 或其变换；
- ictal energy、clinical onset contact、SOZ 或 seizure label；
- test20 的 participation、rank distribution 或 transition statistics；
- 最终事件长度、未来参与 contact、`t/T_event` 等 future information。

允许输入只有：当前及过去 rank-set multi-hot、已参与 mask、由 fit60 估计并冻结的 contact
participation bias，以及不依赖生理 target 的 shaft topology。A/B 只可在模型冻结后作为外部
read-back，不参与模型选择或 Figure 6 患者选择。

### 2.3 Early-ictal target seal

主发作队列固定为 15 位 clinical-onset eligible 患者；`epilepsiae_1146` 单列
supportive，不并入 primary cohort statistic。主 target 固定为：

- clinical onset 后 `[0,10] s`；
- `1--150 Hz` broadband contact energy；
- exact contact-name join；
- seizure-first、patient-first 汇总。

训练前可以读取 target metadata 以冻结患者、seizure、contact denominator，但不得读取
energy values。只有下列内容全部写入 immutable manifest 后才解除 seal：三个模型的
checkpoints、由 structured operator 定义且两模型共用的 source pools、rollout horizon、
seed ensemble、两方向场和
相应 SHA256。间期结果无论阳性或阴性都不阻止 target-free scoring。

## 3. 公平模型比较

三种模型使用完全相同的 events、prefix decisions、candidate mask、static bias、STOP/
cardinality heads、loss、batch schedule、seed 和训练更新上限。

### 3.1 Static no-history baseline

对 contact \(i\)，从 fit60 continuation decisions 估计它在 eligible 时进入下一
rank set 的条件 hazard；其中 \(n_i^{+}\) 是它成为 next-contact 的次数，
\(n_i^{\mathrm{eligible}}\) 是它尚未参与、因而可被选择的次数：

\[
\hat h_i=\frac{n_i^{+}+1/2}{n_i^{\mathrm{eligible}}+1},
\qquad b_i=\operatorname{logit}(\hat h_i).
\]

Static 模型在每一步只使用 \(b_i\) 和已参与 mask。STOP 概率与 next-set cardinality
distribution 也只从 fit60 估计。它不接收 rank history。

### 3.2 Ordinary dense GRU

Ordinary 模型是患者特异的 dense GRU。输入为当前 rank-set multi-hot，hidden state 在每场
事件开始时清零；contact decoder 为 unrestricted dense linear readout。它不读取 A/B、mean
rank、geometry 或 ictal target，且与 structured model 使用相同的 static bias、decision
mask、STOP/cardinality 分解、split、batch order 和训练预算。

该模型是“普通 RNN 能否完成同一任务”的直接对照。它允许任意 hidden/contact mixing，参数量
可以高于 structured model；因此 structured model 若优于它属于保守比较。一个同状态方程但
使用 directed low-rank operator 的模型只可作为 Supplementary mechanism sensitivity，不进入
Figure 6 主比较。

### 3.3 Patient-specific shared-scaffold propagation RNN

固定 same-shaft local adjacency \(K_{\mathrm{shaft}}\) 只编码同杆相邻 contact；若某患者
shaft metadata 缺失，该项置零并在 inventory 报告，不排除该患者。患者特异非负低秩因子
\(B_p\in\mathbb R_+^{N_p\times 2}\) 产生共同 scaffold：

\[
A_p=(1-\gamma_p)\bar K_{\mathrm{shaft},p}
+\gamma_p\overline{B_pB_p^\top},
\]

\[
W_p=g_pD_p^{-1/2}A_pD_p^{-1/2},
\qquad W_p=W_p^\top.
\]

横线表示 Frobenius normalization；\(B_p=\operatorname{softplus}(V_p)\)、
\(\gamma_p=\sigma(a_p)\)、\(g_p=\operatorname{softplus}(c_p)\)。所有可学习的
cross-contact mixing 只能经过 \(W_p\)，不得增加 dense decoder、MLP contact mixer 或独立
forward/reverse operator。

structured dynamics 固定为：

\[
P_{t+1}=\rho_PP_t+W_px_t,
\qquad
R_{t+1}=\rho_RR_t+W_px_t,
\]

\[
z_{t+1}=b+\beta_PP_{t+1}-\beta_RR_{t+1}+m_t.
\]

其中 \(m_{i,t}=-\infty\) 屏蔽已参与 contact；每场事件开始时 \(P=R=0\)。
\(\rho_P,\rho_R\) 只能解释为 rank-step persistence，不是秒级生物时间常数。

## 4. 统一训练目标

所有模型把输出拆成三项，避免把事件长度误写成传播预测：

1. `contact set | continue, k`：给定下一 rank-set 大小 \(k\)，使用 eligible contacts 上的
   exact conditional \(k\)-subset likelihood；其概率与集合内 contact logits 之和的指数成正比，
   normalization 覆盖全部 eligible \(k\)-subsets；
2. `cardinality | continue`：预测下一 rank set 大小；
3. `STOP`：预测事件是否结束。

\[
\mathcal L=
\mathcal L_{\mathrm{contact}|\mathrm{continue}}
+0.25\mathcal L_{\mathrm{cardinality}}
+0.25\mathcal L_{\mathrm{STOP}}.
\]

Primary interictal endpoint 是 held-out
\(\mathcal L_{\mathrm{contact}|\mathrm{continue}}\)；joint loss、STOP 和 cardinality 单独
报告。所有比较必须使用相同 held-out decisions 和 eligible-contact denominator。

训练配置直接继承已完成的训练充分性审计：三个 seeds `11,29,47`；batch 256；AdamW、
weight decay 0、gradient clip 1、learning rate `3e-4`；固定 7 次 fit60 完整覆盖，每次覆盖
分为 32 次 optimizer updates。每次完整覆盖后计算 validation contact-set NLL；七次覆盖中
该值最低的 checkpoint 冻结。这里不再扩学习率或 architecture grid。同一患者/seed/model
使用同一预生成 batch order。

## 5. 间期评价与统计

34 人均报告：

- contact-choice NLL（nats/continue decision）与 top-1 next-contact accuracy；
- structured 相对 static、ordinary 的 paired \(\Delta\)NLL；
- model rollout 与 test20 的 contact participation correlation；
- model rollout 与 test20 的 pairwise precedence correlation；
- expected-rank Wasserstein distance。

Primary patient-level statistic 为 structured vs ordinary 的 test20 contact NLL；
structured vs static 为必要参照。报告患者级配对差、bootstrap 95% CI、精确 Wilcoxon
和正/负/并列计数。rollout consistency 为第二独立 endpoint，不与 contact NLL 合并成总
gate。empirical fit60 precedence \(\rightarrow\) test20 作为数据上限参照，但不属于训练模型。

## 6. 唯一冻结的 two-direction rollout field

每个 recurrent checkpoint 只允许产生一对方向场，不允许从 participation、early、late、
endpoint 等候选场中用 ictal target 挑赢家。

1. 从 structured model 的 seed-ensemble effective operator 构造 normalized graph
   Laplacian；取首个非平凡 diffusion coordinate \(q_p\)，其符号任意。该坐标只定义 source
   interventions，不读取 empirical A/B。
2. \(q_p\) 最低和最高四分位分别定义 source pool \(S^-_p,S^+_p\)；每侧至少一个 contact。
3. ordinary 与 structured 使用完全相同的 \(S^-_p,S^+_p\)；从各 source pool 初始化各自
   已冻结的 recurrent model。rollout horizon
   \(H_p\) 固定为 fit60 事件 rank 长度的 90th percentile，并截在 `[3,12]`。
4. 唯一方向场定义为 participation-weighted first-arrival earliness：

\[
F^d_{p,i}=\sum_{t=1}^{H_p}
\left(1-\frac{t}{H_p}\right)
\Pr(T_i=t\mid S^d_p),
\qquad d\in\{-,+\}.
\]

source pools、\(H_p\)、\(F^-_p,F^+_p\) 全部在 target seal 解除前冻结。经验 A/B 可以在
事后检查是否与这对方向场对应，但不能替换、旋转或重选它们。

## 7. Early-ictal target-free correspondence

对 seizure \(s\) 的 exact-joined contacts：

\[
C_{p,s}=\max_{d\in\{-,+\}}
\left|\rho_{\mathrm{Spearman}}(F^d_p,Y^{0-10s,1-150Hz}_{p,s})\right|.
\]

- primary null：5000 次 **all-contact label permutation**；每次都重新执行绝对值和双方向
  max；
- sensitivity：5000 次 **within-shaft permutation**，同样重新 max；
- seizure 内先评分，患者内取 seizure median，最后做 patient-first cohort statistics；
- primary 15 人与自己的 null 比较，并比较 structured vs ordinary 的 null-corrected
  margin；
- `epilepsiae_1146` 只显示 supportive point，不进入 primary P value。

Static baseline 使用其唯一 fit60 participation field；它没有人为复制出的方向分支。所有
模型使用相同 seizure/contact denominator。不能要求每名患者超过个人 95% null 才报告 cohort
结果。

## 8. Figure 6 A--E 冻结含义

### A｜为什么要用 shared-scaffold RNN

画出 rank-set input、固定 static participation bias、对称低秩 shared scaffold、
propagation/restraint states 以及 next-contact/STOP/cardinality 输出。用同一个 \(W_p\) 从
两个 source pools 初始化得到相反方向 rollout。图内明确标注：`patient-specific training`、
`no A/B or ictal target input`、`same W, two observed source sides`。

### B｜模型复现患者内双向间期传播

对预先冻结的 illustrative patient `epilepsiae_1146` 上下放置 source-minus 与
source-plus 两组。每组都包含：test20 observed contact-by-rank
heatmap、structured rollout 的对应 first-arrival heatmap，以及从 rank 1 到末 rank 的小型时序
分解。contacts 按冻结 diffusion coordinate \(q_p\) 排序。E1146 是论文既有代表病例并在
本分析中始终标为 supportive；不得再根据新 structured 或 ictal 结果更换病例。经验 A/B 只可
在图注中作为外部 read-back，不得用于训练、重排或选择 model mode。

### C｜34 人间期 next-contact 预测与传播一致性

左侧显示 static、ordinary、structured 的 patient-level held-out contact NLL 配对点；右侧
显示 rollout-vs-test20 pairwise precedence correlation。主标注为 structured vs ordinary
的配对效应、95% CI、P 值和正/负/并列计数。不得只画单个代表患者或只画训练 loss。

### D｜冻结模型场与发作早期场的直观对应

继续使用预先冻结的 E1146，在同一 contact layout 上画 \(F^-\)、\(F^+\) 和其患者内两次
seizure 的中位 early-ictal broadband energy field；方向配色、contact 顺序和色标规则固定。
标题明确两张 model fields 在读取 target 前已经冻结，E1146 为 illustrative/supportive，
不承担 Panel E 的 primary cohort statistic。

### E｜患者层面的跨状态统计

显示 15 位 primary 患者的 null-corrected early-ictal correspondence：ordinary 与
structured paired dots，并叠加 structured 的 cohort summary。标注 all-contact primary
null 的 cohort effect、structured-vs-ordinary paired effect、个人超过 null p95 的人数；
within-shaft 结果用小型 sensitivity inset。E1146 用空心符号单列，不进入统计。

## 9. 解释边界与执行规则

若 structured 在间期和 ictal correspondence 上均占优，可写：结构先验使患者内自监督
RNN 从间期 rank sequences 恢复了可迁移的患者特异传播 scaffold。

若只在间期占优，可写：structured RNN 改善了患者内传播恢复，但未建立跨状态对应。
若跨状态对应主要由 static baseline 承担，应明确写成稳定 contact recruitment scaffold，
不把它归因于 recurrent dynamics。

本合同不设置连锁 `hard_gate_pass`。除数据泄漏、损坏 artifact、NaN/OOM 或 checkpoint 不完整
等工程错误外，所有预定模型和评分均执行完成；每个 claim 独立报告阳性、阴性或不确定结果。
