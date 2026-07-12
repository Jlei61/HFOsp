# Methods 中文工作母稿

> 状态：working draft，2026-07-12  
> 写作目标：说明稳定的患者内间期 HFO 传播结构如何被提取、做空间约束，并在各向异性 E–I SNN 中接受机制可行性检验。  
> 当前边界：二维间期场—发作场比较尚有 producer 合同冲突，暂不并入已执行主方法；慢变量扩展是探索性分析，不属于当前主 SNN 结论。

## 术语与统计层级

| 本稿统一用语 | 定义与边界 |
|---|---|
| 单通道 HFO event | 单触点上通过 80–250 Hz 检测合同的候选事件 |
| group event | 短时间窗内多个 HFO-rich 触点共同参与的事件 |
| participating channel | 在该 group event 窗口内实际检测到 HFO 的触点 |
| rank pattern | 仅对 participating channels 的时间质心排序；未参与触点为缺失 |
| propagation template | masked rank features 经患者内聚类得到的重复传播模式 |
| source / sink endpoint | 给定模板或 rank-displacement 对中最早 / 最晚的有效触点集合 |
| propagation axis | source centroid 指向 sink centroid 的空间方向 |
| scientific tier | 主分析、敏感性分析和探索性分析分开报告，不由显著性反推层级 |

## 伦理批准与知情同意

本研究使用两组既有的人类颅内脑电数据。Yuquan 数据来自清华大学玉泉医院癫痫中心接受术前立体定向脑电图（stereoelectroencephalography, SEEG）评估的药物难治性癫痫患者。作者提供的伦理批准号为 2016005。研究方法依照适用的伦理指南和法规执行，并在数据采集和研究使用前取得书面知情同意。

> **TBC-E1**：提交前需用伦理批件或原始发表记录核对伦理委员会法定名称、批准号、批准范围，以及未成年人是否由法定监护人签署；在核对前不统一写成“患者本人或法定监护人”。

EPILEPSIAE 是 EU FP7 EPILEPSIAE 项目建立的去标识化癫痫数据库。本研究仅使用其中已去标识化的颅内脑电记录和临床标注。既有数据库论文报告，数据采集分别获得 Freiburg、Pitié-Salpêtrière 和 Coimbra 参与中心伦理委员会批准，参与者提供了将临床数据用于研究的书面同意（正式稿补数据库原始论文引用）。

> **TBC-E2**：按 EPILEPSIAE 原始论文逐字核对三家委员会名称；当前来源只支持“患者书面同意”，不自动扩写为“患者本人或法定监护人”。

## 研究队列与数据来源

Yuquan 数据由术前 SEEG 长程记录、术前 T1 加权 MRI、术后 CT、临床发作标注和 SOZ 信息构成。原始 SEEG 以 2 kHz 采集。论文队列已锁定为 20 人，并已在不入库的 private crosswalk 中与 artifact folder 一对一对齐。仓库可追踪的第 21 名 detector artifact 不在当前 40-subject masked primary per-subject summary 中，也缺临床 SOZ/病因定义，因此仅作 detector/lineage artifact 保留，不进入投稿 cohort。

EPILEPSIAE 包含长期颅内脑电记录、SQL 临床元数据和发作标注，植入形式包括深部、条带和网格电极。因此本文统一称为 intracranial EEG (iEEG)，不把该队列整体称为 SEEG。当前挂载数据含 27 名 SQL 受试者，其中 20 名具有既有间期分析 artifact；“原始 30 名中按采样率筛到 20 名”的表述尚缺本地 subject-level 证据，暂不作为正式纳入流程。

论文总体队列为 40 人（Yuquan 20 人、EPILEPSIAE 20 人），40 人均进入 masked temporal primary analysis。旧 n=33 same-lineage 集合保留为 sensitivity。后续分析依赖的输入不同，每一项分析分别报告实际纳入的患者数、记录数、事件数、触点数或发作数；总体 cohort、artifact existence、analysis eligibility 与下游科学分母不合并。

> **TBC-C1**：冻结主队列及各分析的 flow table，至少包含 dataset、subject ID、included、exclusion reason、sampling rate、artifact source、coordinate available、seizure available 和 analysis tier。

## iEEG 预处理与电极定位

非颅内辅助通道在分析前排除。Yuquan 数据采用患者特异性的固定通道排除表和双极重参考，并从 2 kHz 重采样到 800 Hz。EPILEPSIAE 数据采用 common average reference，保持原始采样率；其 v2 检测流程不沿用旧版手工 sub_dropChns 列表。两套数据只处理低于 Nyquist 频率的 50 Hz 及其谐波。

Yuquan 电极坐标来自术后 CT 与术前 T1 MRI 配准后的触点定位。EPILEPSIAE 使用数据库提供的 MRI 空间触点坐标，并按可用坐标变换映射到对应解剖空间。所有空间分析在触点名称和坐标顺序显式核对后进行；名称或顺序不能唯一对齐的记录不进入空间分析。

发作期信号与间期 HFO artifact 来自不同 producer 链。除非后续逐项核实滤波、参考和通道表完全一致，本文不写“发作期和间期采用相同预处理”。

## 单通道 HFO 检测

单通道 HFO 采用当前 detector v2 合同。信号在 80–250 Hz 范围内按 20 Hz 子带处理，并以 Hilbert 包络汇总宽频高频活动。对触点 \(c\)，检测阈值为

\[
\theta_c=\max(\alpha_c m_c,\alpha_g m_g),
\]

其中 \(m_c\) 是该触点包络中位数，\(m_g\) 是同一记录所有颅内触点的全局中位数，\(\alpha_c=\alpha_g=2\)。相隔不足 20 ms 的候选段合并，只保留严格满足 \(50<duration<200\) ms 的事件。事件平均能量还需高于相邻背景侧窗；Yuquan 和 EPILEPSIAE 的侧窗倍率分别为 1.5 和 2.0。

EPILEPSIAE v2 使用 200 s、无重叠分块，FIR-801 线噪声陷波和 FIR-201 子带滤波。所有滤波频带均经 Nyquist hard gate；低采样率记录不被默默裁剪后继续作为完整 80–250 Hz 分析。

## Group event 构建

对每名患者，先根据触点事件计数分布识别 HFO-rich 触点。阈值由患者内计数均值加患者特异性的 \(k\) 倍标准差确定。单通道事件向两侧各扩展 30 ms，并在统一时间轴上统计同步参与触点数；达到候选触点集合预设比例的时间段进入 group-event 候选。

候选中心随后扩展为患者特异性的固定 packing window。相互重叠的候选窗均删除，长度超过 2 s 的候选不保留；在最终窗口内重新计算参与触点。每个事件保存时间窗、参与掩码和触点级事件。患者特异性的 pick_k、pack_win_sec、pack_top_n 及其他 legacy-compatible 参数必须在 Supplementary Table 中逐例列出。

## 事件内时间质心与 masked rank pattern

对每个 group event 的每个 participating channel，在 50–300 Hz（受 Nyquist 限制）范围计算 spectrogram。使用 50 ms Hamming 窗、40 ms 重叠，并对时频矩阵作 \(\sigma=1.5\) 的高斯平滑。设平滑后的非负强度为 \(S_{ij}\)，采用三次幂权重

\[
w_{ij}=\frac{S_{ij}^3}{\sum_{ij}S_{ij}^3},\qquad
t_M=\sum_{ij}w_{ij}t_i.
\]

\(t_M\) 是事件窗口内的相对高频活动质心，不解释为神经元放电或病理传播的精确生物学起始时刻。同一事件内仅对 participating channels 的 \(t_M\) 从早到晚重新排序；未参与触点保持缺失。任何聚类输入均通过 build_masked_kmeans_features(..., impute='event_median') 构造，不能使用 legacy lagPatRank 中非参与触点的伪 rank。

## 患者内模板估计与重复性

对每名患者，在 masked rank-feature 空间内评估 \(k=2,\ldots,8\)。每个 \(k\) 使用 10 个外层随机种子，每次 KMeans 使用 n_init=10。候选 \(k\) 需同时满足跨种子 adjusted mutual information 中位数不低于 0.70，以及所有种子中的最小簇比例不低于 0.10；在通过者中选择 silhouette 中位数最高的 \(k\)。

模板的时间稳定性采用两种独立切分：按真实绝对事件时间的前后半记录，以及奇偶记录块。每个切分内独立拟合模板，再用模板间 Spearman 相关构成代价矩阵并通过 Hungarian assignment 配对。平均模板相关不低于 0.80 且事件分配一致率不低于 0.70 定义为 strong；两者分别不低于 0.50 定义为 moderate；其余为 weak。

两个模板的平均顺序 Spearman \(r<-0.5\) 时标记为候选 forward/reverse pair。只要前后半或奇偶块任一切分复现该关系，即记为 forward_reverse_reproduced。

模板分解的解释增益以整体事件对相似性与簇内事件对相似性的差值汇总。Matching Index 现仅在每个事件的共同 participating channels 上计算（masked：非参与触点在 raw lagPatRank 中携带的 phantom rank 不进入统计），与文字定义一致；置换零分布在每个事件的参与触点集合内重排秩后重算。全队列 40/40 患者的 masked MI 高于其置换零分布（cohort median 0.228）。纳入非参与 phantom rank 的历史全通道实现保留为 unmasked_sensitivity 敏感性字段（同为 40/40 显著，median 0.188），仅供历史可比性。

## 三维传播端点与传播轴

空间输入使用 masked rank-displacement 的 variable-\(k\) swap_sweep 输出，而不是直接从自适应 KMeans 的 template_rank 取极端值。对具有 strict 或 candidate swap label 的患者，使用 decision_k 定义两个方向模式的 source 和 sink 触点；joint_valid 和参与掩码用于排除从未参与模板的触点。

当有效空间触点数不少于 7 时，端点核心取最早或最晚 3 个触点；有效触点数为 5–6 时取 2 个并作为 fallback；少于 5 个仅作描述或排除。端点集合 \(G\) 的主要紧凑性指标为触点到质心的 RMS 半径：

\[
R_G=\sqrt{\frac{1}{|G|}\sum_{i\in G}\|x_i-\mu_G\|^2}.
\]

患者内空间零分布从所有有坐标、但不属于该 swap endpoint 的 SEEG 触点中按相同集合大小随机抽样，重复 2,000 次。source 和 sink 紧凑性分别检验；其统计层级按预注册合同报告。

给定 source 和 sink 质心，传播轴定义为

\[
\hat{u}=\frac{\mu_{\rm sink}-\mu_{\rm source}}
{\|\mu_{\rm sink}-\mu_{\rm source}\|}.
\]

轴的 out-of-sample 稳定性通过事件层 held-out/bootstrap 重估评估（200 次）。若存在少数模板，则独立构建其轴，并用两轴余弦相似度描述患者内双向或反向几何关系。

## 二维 contact-plane 与发作场分析：待锁定

附件提出了“模板 A 与 B 的早端连线作为共同轴、PCA 残差作为横轴、两个模板和发作均投影到同一平面”的定义。当前已执行的 run_contact_plane_readout.py 则分别为模板 A 和 B 构建各自的 source–sink 轴和平面；两者不是同一个 producer 合同。因此，本节在锁定并重跑前不能写成已完成方法。

发作场还存在以下未锁定分支：正式 Topic 5 A-line 的主频带为 1–45 Hz，而附件使用 1–150 Hz；当前 producer 以 EEG onset 为主，而附件写 clinical onset；平滑窗口存在 5 s 与 10 s 两个版本。maxAB 的 identity/mirror-invariant 相关与 all-contact、within-shaft null 可以保留，但必须在上述输入合同锁定后整链重放。

> **TBC-F1**：决定采用“每模板独立平面”还是“共同 A–B 平面”。  
> **TBC-F2**：锁定主频带、onset 定义、基线窗、平滑窗、支持阈值、置换次数和统计层级。  
> **TBC-F3**：重新生成 per-subject、cohort summary、图和方法字段后，再把本节升级为正式方法。

## 空间易激场 SNN

### 主模型与科学边界

主 SNN 用于检验一个有限的机制可行性命题：局部低阈值易激核心与各向异性 E-to-E 连接共同存在时，是否能够产生并被稀疏虚拟 SEEG 读出为稳定的轴向传播事件。它不用于证明患者的两个传播模板由两个独立病灶因果产生。

模型定义在边长 20 mm 的二维薄片上，神经元密度为 100 neurons/mm²，共 40,000 个神经元，其中 80% 为兴奋性、20% 为抑制性。神经元采用 current-based leaky integrate-and-fire 动力学：

\[
\tau_m^a\frac{dV_i}{dt}=-V_i+I_i^E-I_i^I,\qquad a\in\{E,I\}.
\]

基线阈值和重置电位分别为 18 和 11 mV；兴奋性神经元的膜时间常数/不应期为 20/2 ms，抑制性神经元为 10/1 ms。空间均一背景输入叠加 Ornstein–Uhlenbeck 噪声，默认背景驱动比例为 0.6，不额外施加触发 kick。

固定入度为 \(C_{EE}=800\)、\(C_{IE}=800\)、\(C_{EI}=200\)、\(C_{II}=200\)。E-to-E 连接采用旋转椭圆核，默认尺度 \(l_{EE}=0.380\) mm、长宽比 \(AR=2\)；其余连接采用各向同性核。两个半径 1.5 mm 的低阈值兴奋性核心沿 E-to-E 长轴放置，中心间分离比例为 0.7。核心阈值独立抽样自下截断于重置电位的正态分布，默认均值 17.5 mV、标准差 1.0 mV；核心外神经元保持 18 mV。

### 虚拟 SEEG 读出

虚拟触点按患者坐标或规则杆坐标放置。规则读出使用 4 mm contact pitch，并同时布置平行与垂直于连接长轴的电极杆。每个触点的 envelope 是邻近兴奋性神经元发放密度的高斯距离加权和。超过背景 floor 加 10% 动态范围的触点记为参与；onset 定义为 envelope 首次达到自身峰值 50% 的时间。

事件方向由最早和最晚 \(k_{\rm dir}\) 个参与触点的空间质心差估计。规则密集读出默认 \(k_{\rm dir}=3\)，至少需要 7 个参与触点；患者稀疏读出允许 \(k_{\rm dir}=2\)，并单独标记为不同读出层级。

### 慢变量扩展：探索性、非主模型

抑制资源场 \(q_I(x,t)\) 与恢复电流场 \(g_K(x,t)\) 属于 M3A-v2 机制筛查，不是当前 Fig. 4/5 主模型的一部分；主模型运行时 slow=None。探索性膜方程为

\[
\tau_m^E\frac{dV_i}{dt}
=-V_i+I_i^E-q_I(x_i,t)I_i^I-\eta_K g_K(x_i,t).
\]

现有结果只支持边界性阴性结论：该参数化未获得“可控的离轴/全局招募后恢复”，\(g_K\) 更倾向抑制而非重定向传播。因此，不把慢变量写成已验证的 ictal transition 机制，也不把四类事件标签写成临床发作分类器。完整方程、脉冲协议和参数扫描保留在 Supplementary exploratory methods，正文只在讨论模型局限时引用。

## 可复现性与报告

最终 Methods 和 Supplementary 必须同步冻结：

1. 两数据集 subject-level 纳入排除表和私有 ID crosswalk；
2. Yuquan 患者特异性的检测与 packing 参数；
3. detector、masked propagation、rank-displacement、geometry、field 和 SNN 的 config hash、seed、软件版本与 artifact 路径；
4. 每项分析的实际患者数、事件数、触点数及失败原因；
5. 主分析、敏感性分析、机制 sanity 与探索性分析的层级；
6. 所有图对应的 producer 与输入 artifact。
