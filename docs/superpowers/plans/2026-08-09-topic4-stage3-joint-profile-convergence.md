# Topic 4 Stage 3 联合剖面与收敛计划（rev1，2026-08-09）

**Spec：** `docs/superpowers/specs/2026-08-06-topic4-axis-constrained-data-driven-core-field-design.md` rev6 §9.3c。

**核心问题：** 在不把双向验收门写入目标的前提下，完整逐事件剖面的联合分布能否从患者数据约束出一个
跨网络稳定的病理场；随后同一空间场能否由局部连接性调制替代外加阈值场，并接入既有相图与有限发作生命周期合同。

## 冻结边界

- 训练目标只用 rev6 `D_curve`；`k_dir`、TA/TB 标签、两模板相关与 §10.3 门不进入优化。
- 每个候选的主目标固定使用 20 条 usable events；患者同构地板为 `0.287 [0.236, 0.358]`。
  少于 20 条只走 feasibility key，超过 20 条不得凭更多事件降低经验 Wasserstein 偏差。
- 患者侧按 recording block 留出；模型侧确认 seed 不得进入拟合。
- 第一轮旧的一维 TV 结果只作历史对照，不作 warm start 的科学证据。
- 当前 core 是 `V_th_per_neuron` 的静态**带符号阈值调制场**：冻结异质性中约 69.5% E 细胞的
  depth 降低阈值、30.5% 提高阈值。连接性等效版本是独立机制层，只有阈值场先稳定后才能启动。
- 固定耦合下已有 readable 与 self-limited tradeoff；局部连接增强不得以“更多事件”为成功，必须同时检查
  size、duration、空间自限、退出和恢复。

## Task 1：联合观测量冻结（零仿真）

- [x] 同码构造 31 点 normalized rank curve。
- [x] 患者训练段拟合 8 维固定 embedding；64 投影 sliced Wasserstein。
- [x] 已有患者/手放双核/Stage 2/Stage 3 标定。
- [x] 四臂共同配平到 n=18；优化合同固定 n=20，并生成同构患者地板。
- [x] Leg A 留一网络位置敏感性。
- [x] 将 reference NPZ、summary、图和 README 纳入可复现产物检查；reference SHA256
  `987e6b4bdfdc9b3d31485852e2eb71cd8df3ee7d92ff6eb0456c8fa04925adc1`，producer 在
  `18c64806` 上 `tracked_modules_dirty: false` 且逐值复现。

**进入 Task 2 的门：** 四项 calibration gate 全过；患者留出必须最接近；Stage 3 正相关失败必须被距离自然惩罚。

## Task 2：3D landscape

- [x] 左侧 7×7 网格改为真实 `x-y-score` 3D bars；缺事件格保持灰色底面，不插成虚假高地。
- [x] 在底面保留真实电极位置，并投影学得场的 90% 质量等值线。
- [x] 视角固定并保留原有二维 diagnostic 版本，避免透视遮掉局部最优区。
- [x] PNG/PDF 像素、标签、遮挡和 metadata 输入合同检查。

## Task 3：优化器收敛修复

- [x] 先审计参数尺度：中心 mm、log-sigma、角度、weight logits 不能共用未预条件化的单位协方差。
- [x] 新一轮使用标准化 latent decoder；禁止继续使用 clip 造成的大块平坦区。
- [x] 对 `<20` usable events 的候选使用可审计 feasibility key（usable count、near-readable participant credit），
  不把所有 0-event 候选压成同一个分数。
- [ ] `K=1/2/3` 各自优化，至少 3 次独立重启；不得靠 K=3 权重饱和冒充正规嵌套比较。
- [ ] cheap pilot 先证明 sigma 不持续发散、每代死区比例下降、best/median 有稳定改进，再开长跑。
- [x] 把训练 global best、最后一代 best 与 CMA mean 在预先冻结的 6 张未见网络上并列重评；确认池不再选参数。

**Pilot-1（工程结果，不进入科学证据）**：K=3、1 代、16 候选、2 网络。空间覆盖初始化把
zero-usable fraction 从旧 pilot 的 62.5% 降到 12.5%，median usable events 从 0 提到 11，feasible
fraction 从 12.5% 提到 25%，sigma `0.650→0.627`；但 best 固定-n 距离为 0.820，尚差于已有 Stage 3
校准值 0.740。裁定为**解除死区阻断，但未证明目标收敛**；下一步只能做小规模多代/多 restart，不得直接称恢复场。

**Pilot-1--3（clean convergence pilot，仍不进入科学接受）**：commit `63eeab79`，K=3 restart 0，
每代 16 候选、2 张同代共享网络。三代 `sigma=0.627/0.607/0.584`，feasible fraction
`25%/25%/38%`，zero-usable `12%/31%/19%`，median usable `11/9/16`，best `D_curve`
`0.820/0.634/0.702`。第一代与 dirty engineering pilot 逐值一致，证明重建可复现；尺度和死区工程问题
明显改善，但目标随网络池波动，尚不满足稳定收敛门。全局训练 best 的有效结构是一主一次两个紧凑团块，
第三分量权重很小且都偏在板的一侧，**不是两端 core 恢复证据**。下一步先做固定未见网络重评，不继续堆代数。

**未见网络确认（6 网络，commit `6bd7bbc8`，不再选参数）**：三候选均无执行错误且 usable events
分别为 59/35/67。训练 global best 的患者训练/held-out 距离为 0.673/0.702，两簇 23/36 但相关
`+0.951`；最后一代 best 为 0.640/0.663，两簇 2/33；CMA mean 为 0.776/0.795，两簇 1/66，
其 `r=-0.464` 由单事件小簇造成，不能读成双簇恢复。按每簇至少 10 个事件的原充分性门，三者分别为
`OPPOSITION_FAIL / TWO_CLUSTER_SUPPORT_FAIL / TWO_CLUSTER_SUPPORT_FAIL`。事件 bootstrap 的训练距离中位数
为 0.675/0.706/0.789，均未优于 Stage 2 控制 0.627，也远高于患者地板 0.284。裁定：**死区修复有效，
但当前三代优化没有恢复“接近患者且两个簇相反”的联合结构；不得进入连接性功能等效或 lifecycle 动态验证。**

**收敛门：** 至少两次重启的 held-out `D_curve` 落入彼此 bootstrap 区间；场的主要质量分量跨重启可匹配；
独立网络确认不回退到 rigid-family best。

## Task 4：阈值 core 到连接性等效 core

- [x] 记录阈值场自由参数与派生量：中心、长短轴、角度、质量、`h_i`、`V_th` 深度分布；core radius 不是单独自由参数，
  而是混合分量 sigma 与预算投影后的派生尺度。
- [x] 首个等效版本只调制已有 E→E recurrent gain：同一 `h_i` 调制局部 E→E，保持 postsynaptic incoming-E 总量归一，
  避免把全局增益变化误写成局部 core。
- [ ] 与阈值场做 matched-local-gain 校准：局部线性响应相等、全局均值连接增益不变、噪声和 RNG 配对。
- [ ] 局部起核通过后只生成 hash-locked `h_i + alpha_EE` 交接产物，不在本分支直接改正在并行执行的
  FCXR-LC3/LC4。历史 E→E STD 单独终止器已经是 3-seed clean no-go，只能作受限负对照，不能再作为默认恢复解。

## Task 5：相图与有限生命周期接口

- [x] 映射到现有合同但不混变量：静态 `h_i/alpha_EE` 是 substrate/core-support coordinate；动态
  `D_i=1-Z_i` 是 FCXR-LC3 entry field，presynaptic `a_X` 是 relay/offset coordinate，逐细胞 adaptation 是
  当前 LC3/LC4 termination 候选。旧 `q_core/q_global` 只保留为历史 M3/M4 坐标，不与新静态 core 等同。
- [ ] 相图候选点必须再过 finite-pulse 生命周期：entry、bounded carrier、exit、postictal protection、return/recovery。
- [ ] runaway、tonic plateau、只短暂下降分别保留独立状态，不合并成“发作样”。
- [ ] synthetic readout 继续走真实事件/电极 pipeline；只作为 mechanism screen，不作患者机制证明。

## 跨工作树边界

- 当前分支只拥有 data-driven field、联合目标和 field→connectivity 适配器。
- `main` 已锁定完整 lifecycle 尚未 PASS；`codex/topic4-fcxr-lc3` 工作树仍有未提交 LC3/LC4 执行改动，禁止在本线
  修改、合并或重写。
- 后续交接的最小合同是：field/reference/config checksum、网络位置顺序、连续 `h_i`、`alpha_EE`、逐 target
  incoming-E 守恒审计和 paired local-response 标定。由 lifecycle 线显式消费后，才允许讨论进入、退出和恢复。

## 停止规则

- 新联合观测量若不能稳定区分单中段生成器，停止优化并回到观测量设计。
- cheap pilot 若仍有 >50% 候选完全无 usable event 且连续三代不降，停止长跑。
- 连接性版本若只提高 rate、不改善 size/duration/self-limitation/return，按历史负对照关闭，不继续扫静态增益。
- 生命周期任一退出或恢复门失败，不进入发作机制表述。
