# RNN 双向性、两类模型与 clinical-onset 静态迁移审计 v2.5

> 日期：2026-07-28
> 状态：完整执行；22 人双侧 heldout 审计、strict clinical-onset 16 人/106 次发作
> 静态迁移、5000 次全通道 coherent shuffle、代表患者图均完成。

## 一、总判断

这次修正了 v2.4 的三个关键偏差：

1. 不再把经验 A/B 轴当作 RNN 必须恢复的金标准；
2. 不再把两名只有 EEG onset 的 Yuquan 混写成 clinical-onset 队列；
3. 不再只用一个跨患者 ridge 读出，而是直接比较模型生成的 contact-rank distribution
   与每次发作的 early-ictal static energy field。

结果有一条明确阳性和一条明确边界。

**阳性**：只用间期 rank sequence 自监督训练的 full-history GRU，生成的 contact-level
分布与 clinical onset 后 `[0,10] s`、1–150 Hz 静态能量场存在中等绝对相似度。严格
16 人中，患者级 absolute similarity 中位数为 `0.577`；相对5000次患者内全通道打乱，
margin 中位数为 `0.218`，15/16 为正。它相对 static contact hazard 的 absolute
similarity 增量为 `0.077`，相对 null margin 增量为 `0.037`。

**边界**：这个迁移不能归因于“结构化病理轴动力学”。结构化 RNN 在共同 11 人中也有
绝对相似度（中位 `0.536`）和正 all-contact margin（中位 `0.143`），但 full、
no-history、isotropic、axis-no-source 与 node-only 之间没有稳定的患者级增量。普通
GRU 相对 rank-shuffle 的 absolute similarity 较高，但在更关键的 all-contact-null
margin 上不再有显著优势。因此当前证据首先支持：

> 间期事件包含可迁移的患者特异 contact scaffold；full-history GRU 对这个 scaffold
> 做了有效平滑/重构。现有结果尚不能证明迁移依赖有序历史，也不能证明 RNN 内部恢复了
> 唯一物理病理轴。

这条线可以进入论文，但应作为**间期病理 contact distribution 向 early-ictal static
field 的自监督计算桥接**，不能写成 latent mechanism、发作动态传播预测或 A/B 轴自动
发现。

---

## 二、两类 RNN 到底是什么

### 2.1 普通 full-history GRU

普通模型没有固定物理图。

- 输入是截至当前时刻的有序 rank-set prefix；
- 自监督任务是预测下一 rank set 和 STOP；
- hidden size 为 32，允许自由 hidden mixing；
- 外层按患者 LOSO：heldout 患者不进入共享模型训练；
- 对 heldout 患者只使用其 train80 做 local calibration，最终评分只用 chronological
  heldout20；
- free rollout 5000 个事件，输出每个 contact 的参与概率、条件 rank 分布、平均 rank、
  rank variance 和 pairwise precedence。

它回答的是：

> 一个容量相对自由的序列模型，能否从大量间期事件中重构每个 contact 的完整 rank
> distribution。

它不是 KMeans，也不以 A/B 标签为目标。A/B 可以自然包含在完整分布中，但模型不需要
先把事件分成两类。

### 2.2 结构化 competitive-propagation RNN

结构化模型是患者内的小模型。

- 每名患者先从 train80 transition residual 的32个无符号候选方向中选择一个方向；
- 该方向不是 RNN 端到端自由学出的 3D 向量；
- RNN 中只有对称局部/轴向核、传播 trace、较慢竞争 trace、source-direction term 和
  contact node bias；
- 禁止额外 dense contact-to-contact mixing；
- fit60 拟合、validation20 选 epoch、heldout20 评分；
- 每名患者3个 seed。

它回答的是：

> 一个受限的局部/轴向传播系统，能否用少量可解释参数重现下一 contact 及长期
> contact-rank distribution。

### 2.3 两类模型不能混称

普通 GRU 的阳性不能解释成物理轴机制；结构化 RNN 的阴性也不能否定间期 rank
distribution 的可迁移性。两者的科学分工是：

- 普通 GRU：检验数据里是否有可学习、可迁移的信息；
- 结构化 RNN：检验目前提出的“对称轴 + 传播/竞争 + source”是否足以解释这些信息。

本轮结果是前者部分成立，后者未成立。

---

## 三、对照模型分别是什么意思

### 3.1 普通 GRU 对照

| 对照 | 看到了什么 | 排除了什么 |
|---|---|---|
| `static_contact_hazard` | 每个 contact 的基础出现频率 | 完全不看事件 prefix 或顺序 |
| `unordered_prefix` | 当前已经出现过哪些 contact | 不使用这些 contact 的先后顺序 |
| `last_set_first_order` | 只看最后一个 rank set | 不保留更早历史 |
| `rank_shuffle_gru` | 与 GRU 相同容量，但训练 rank 顺序被破坏 | 检验阳性是否真依赖有序 rank |
| `empirical_rank_distribution` | heldout 真实事件直接汇总 | 不经过模型，是数据上限/噪声参照 |

`static_contact_hazard` 不是随机模型。它保留每个 contact 在患者中的病理参与频率，所以
它本身可以与 early-ictal energy 相似。它检验的是 RNN 是否带来超出静态节点偏好的
信息。

### 3.2 结构化 RNN 对照

| 对照 | 改动 | 正确解释 |
|---|---|---|
| `node_only` | 只保留 contact node bias | 静态节点频率 |
| `no_history` | 两个 persistence 均设为0 | 只看当前 rank，不累积历史 |
| `local_isotropic` | `gamma=0`，只保留局部各向同性核 | 普通局部传播 |
| `axis_no_source` | 轴向核 + 两状态，但无 source-direction term | 轴向结构本身 |
| `full` | 轴向核 + 两状态 + source-direction | 完整结构模型 |

旧 v2.4 把 `full - local_isotropic` 直接叫 axis contribution，这是不严格的，因为两者
同时改变了 axis 和 source 项。本轮新增 `axis_no_source`，真正的轴对照是：

```text
axis_no_source − local_isotropic
```

source 对照是：

```text
full − axis_no_source
```

history 对照是：

```text
full − no_history
```

---

## 四、A/B-independent 双向性

### 4.1 怎么测

队列是结构化模型冻结的22名 geometry-complete、development-excluded 患者。

每名患者：

1. 使用 train80 transition residual 选出的无符号方向；
2. heldout20 事件按第一 rank set 的轴投影分为 negative-side source 和
   positive-side source；
3. 每侧至少20个事件才可进入；22/22全部满足；
4. 分别计算 source 到后续 contact 的 inward displacement；
5. 患者级双向分数取两侧较弱值，防止一侧强、一侧无效仍被称为双向；
6. 在全部32个候选方向上重复同一测量，以“选中方向 − 候选方向中位数”检验方向特异性；
7. A/B relation 完成后才读回作描述，不进入方向选择或主要统计。

### 4.2 结果

| 指标 | n | median [95% bootstrap CI] | 正向患者 | P |
|---|---:|---:|---:|---:|
| 两侧较弱的 inward displacement | 22 | 0.180 [0.110, 0.236] | 21/22 | 4.77×10⁻⁷ |
| 选中方向 − 候选方向中位数 | 22 | 0.0238 [−0.0441, 0.0502] | 13/22 | 0.475 |
| 两侧较弱的 axis NLL benefit | 22 | 0.000013 [−0.000202, 0.000424] | 11/22 | 0.262 |
| 两侧较弱的 source-term benefit | 22 | −0.000177 [−0.00124, −0.000077] | 2/22 | 1.000 |
| full − isotropic 两侧较弱 benefit | 22 | −0.000212 [−0.000972, 0.000405] | 10/22 | 0.563 |

### 4.3 该怎么解释

“从两端起点向中间走”在22人中很常见，但选中的方向并不比其他候选方向更能产生该
现象。最可能的替代解释是：

- contact cloud 有边界；
- source 在投影端点时，后续被招募 contact 自然更靠内部；
- 任意穿过 contact cloud 的方向都可能产生一定 inward regression。

因此现在可以说：

> heldout 间期事件在两侧 source 条件下都呈现双侧扩展形态。

不能说：

> RNN 在更多患者中自动恢复了正确的患者病理轴。

而且 source-direction term 在两侧同时成立的检验中反而没有收益，说明当前 source
项的数学形式没有抓住双向传播所需的信息。

### 4.4 A/B relation 只作描述

当前可读 relation 中：

- same：3/22；
- reversed：4/22；
- different：2/22；
- 其余13人没有同源 relation 记录。

各组都能出现双侧 inward displacement，没有证据表明只有 strict same/reversed
患者才有该现象。由于组很小、缺失多，不做组间显著性，也不设置角度阈值。这与本轮
修正一致：A/B 是一种数据可见的强表型，不是所有患者真实病理轴的金标准。

---

## 五、clinical-onset 静态迁移

### 5.1 迁移预测的到底是什么

当前迁移是**source-free static prediction**，不是发作传播序列预测。

target 固定为：

- Epilepsiae；
- clinical onset；
- `[0,10] s`；
- 1–150 Hz baseline-robust-z broadband energy；
- strict broadband phenotype；
- 16名患者、106次发作。

每个间期模型给出一个患者级、contact-wise distribution。固定导出五个静态场：

1. participation probability；
2. early joint mass；
3. late joint mass；
4. early + late endpoint mass；
5. participation-weighted earliness。

每次发作分别计算五个场与发作能量的 Spearman correlation，再取最大绝对值。
5000次全通道置换中每次都重新选择最大场，因此已经支付“五选一”的 readout 选择成本。
同一次 contact mapping 贯穿该患者的全部 seizure 和 seed。最后先折叠 seizure，再折叠
seed，患者是独立统计单位。

这比旧 v2.4 更接近论文主结果：不要求每位患者单独越过其95%随机线，而是对每位患者
的 observed 与 all-contact null median 做患者级配对统计。

### 5.2 绝对相似度与 all-contact null

#### 普通模型：strict clinical-onset 16人

| 模型 | absolute similarity median | all-contact margin median | margin > 0 |
|---|---:|---:|---:|
| empirical heldout rank distribution | 0.568 | 0.137 | 13/16 |
| full-history GRU | **0.577** | **0.218** | **15/16** |
| static contact hazard | 0.502 | 0.153 | 13/16 |
| unordered prefix | 0.563 | 0.154 | 13/16 |
| last-set first-order | 0.579 | 0.167 | 12/16 |
| rank-shuffle GRU | 0.522 | 0.168 | 14/16 |

full-history GRU 自身：

- absolute similarity median `0.577`，95% CI `[0.500, 0.750]`；
- all-contact margin median `0.218`，95% CI `[0.0595, 0.357]`；
- margin 15/16 为正；
- observed > null 的患者级 one-sided Wilcoxon `P=3.27×10⁻⁴`。

#### 结构化模型：strict clinical 与 physical-axis 交集11人

| 模型 | absolute similarity median | all-contact margin median | margin > 0 |
|---|---:|---:|---:|
| empirical train80 distribution | 0.607 | 0.108 | 10/11 |
| structured full | **0.536** | **0.143** | 7/11 |
| no-history | 0.536 | 0.133 | 8/11 |
| local isotropic | 0.525 | 0.143 | 7/11 |
| axis no-source | 0.525 | 0.126 | 7/11 |
| node-only | 0.475 | 0.050 | 7/11 |

structured full 自身的 all-contact margin one-sided `P=0.0273`，但该值不能单独解释
为轴机制，因为其所有结构消融几乎相同。

### 5.3 普通 GRU 的增量来自哪里

预先列出的五个普通模型比较做 BH-FDR。

#### Absolute similarity

| 比较 | median Δ | 正向患者 | raw P | BH-FDR q |
|---|---:|---:|---:|---:|
| full GRU − static hazard | +0.0774 | 15/16 | 3.05×10⁻⁵ | 1.53×10⁻⁴ |
| full GRU − unordered prefix | +0.0261 | 10/16 | 0.0434 | 0.0450 |
| full GRU − last-set first-order | +0.0223 | 9/16 | 0.0450 | 0.0450 |
| full GRU − rank-shuffle GRU | +0.0733 | 10/16 | 0.00928 | 0.0232 |
| full GRU − empirical distribution | +0.0589 | 11/16 | 0.0346 | 0.0450 |

#### All-contact-null margin

| 比较 | median Δmargin | 正向患者 | raw P | BH-FDR q |
|---|---:|---:|---:|---:|
| full GRU − static hazard | +0.0373 | 13/16 | 0.00157 | 0.00673 |
| full GRU − unordered prefix | +0.0240 | 9/16 | 0.0778 | 0.130 |
| full GRU − last-set first-order | +0.00357 | 8/16 | 0.165 | 0.207 |
| full GRU − rank-shuffle GRU | +0.0181 | 9/16 | 0.469 | 0.469 |
| full GRU − empirical distribution | +0.0661 | 13/16 | 0.00269 | 0.00673 |

最重要的区别是：

- full GRU 稳定超过完全静态的 contact hazard；
- 但当评价量改成“高于每个模型自己 all-contact null 的多少”时，它没有稳定超过
  unordered、first-order 或 rank-shuffle。

因此可以说 full GRU 做了有价值的非线性重构/去噪；不能说这一迁移已经被证明来自
有序长历史。

### 5.4 结构化模型没有独立结构增量

共同11人中：

- full − no-history：absolute similarity median `0`，P=0.313；
- axis-no-source − isotropic：median `0`，P=0.688；
- full − axis-no-source：median `0`，P=0.219；
- full − node-only：median `0`，P=0.150。

all-contact margin 的四个差值中位数同样均为 `0`。

所以结构化模型的静态阳性主要来自 contact bias 和粗粒度 distribution，而不是：

- persistence/history；
- 轴向 anisotropy；
- source-direction term。

### 5.5 哪种 contact field 在驱动相似性

各 field 的患者级 absolute rho 中位数：

| 模型 | participation | endpoint mass | weighted earliness | early mass | late mass |
|---|---:|---:|---:|---:|---:|
| empirical | 0.358 | 0.351 | 0.307 | 0.256 | 0.250 |
| full-history GRU | **0.415** | 0.402 | 0.381 | 0.343 | 0.285 |
| structured full | **0.467** | 0.350 | 0.359 | 0.238 | 0.300 |
| structured node-only | **0.442** | 0.356 | 0.333 | 0.286 | 0.300 |

participation 是最稳定的单一 readout；endpoint mass 和 weighted earliness 提供额外形态，
但结构化 full 与 node-only 很接近。这再次表明当前迁移首先是“哪些病理 contact 经常
参与”的静态 scaffold，而不是完整传播顺序的重放。

---

## 六、same / reversed / different 患者

在 strict clinical 16人中，现有 relation 记录为：

- same n=4；
- reversed n=5；
- different n=4；
- unavailable n=3。

full-history GRU 的 all-contact margin 中位数分别为：

- same `0.339`；
- reversed `0.142`；
- different `0.218`；
- unavailable `0.038`。

结构化共同队列中：

- same n=3，margin `0.143`；
- reversed n=4，margin `0.092`；
- different n=2，margin `0.191`；
- unavailable n=2，margin `−0.075`。

这些数字只作描述，不能比较组间显著性。它们至少说明：

> 静态迁移并不只出现在严格 reversed 或严格共线患者；different 组同样可见。

这符合“宽病理 scaffold 可能比可稳定拟合的 A/B 轴更普遍”的解释，但还不能把所有
患者都命名为有一个真实宽病理轴。

---

## 七、代表患者图

固定患者为：

- E1084：发作数多；
- E958：既有双向传播背景；
- E1096：模型和数据形态较一致。

没有根据本轮 target correlation 选择患者。

图中每行按 empirical mean rank 排 contact；该排序完全来自间期数据。三块 rank
distribution 分别是：

1. heldout 真实间期；
2. full-history GRU；
3. structured full RNN。

最右是 strict clinical-onset seizure 的 early-ictal energy rank，仅用于形态对照。

图：

- `results/topic5_rnn_bidirectional_cross_model_audit_v2_5/figures/representative_rank_distribution_comparison.png`
- `results/topic5_rnn_bidirectional_cross_model_audit_v2_5/figures/cohort_bidirectional_static_transfer_summary.png`

代表图支持的是模型确实保留了 contact-level rank distribution 的粗结构；它不支持
逐次事件 replay。

---

## 八、对核心科学目标有没有偏移

### 8.1 对齐的部分

- 输入始终是 masked raw contact-rank events；
- 没有用 A/B label 训练模型；
- 发作 target 回到 clinical onset、`[0,10] s`、1–150 Hz；
- 使用论文接受的 all-contact coherent shuffle；
- 每次发作先评分，再 patient-first；
- 报告绝对相似度，不再只报告是否超过单患者严格随机线；
- A/B relation 只作外部描述；
- 普通和结构化两类模型都进入同一迁移审计。

### 8.2 仍未完成的部分

- 没有证明一个唯一物理轴在多数患者中可辨识；
- 没有证明 source term 能同时解释两侧传播；
- 没有证明 early-ictal static similarity 来自 ordered history；
- 没有 exact per-seizure clinical-onset contact sets，因此不能做
  source-conditioned dynamic ictal rollout；
- 当前迁移是静态 contact field，不是 seizure recruitment sequence。

### 8.3 结论没有回到“证明 RNN 有用”

普通 GRU 的阳性有科学价值，因为它说明 interictal rank distribution 可迁移；结构化
模型的消融阴性同样重要，因为它限制了机制解释。报告不以 AUC 或模型胜负作为终点，
而是区分：

```text
数据中存在的可迁移信息
≠
当前结构化模型已解释其机制
```

---

## 九、这部分怎样放进论文

### 9.1 推荐主文口径

如果需要把 RNN 放入主文，最安全的一句话是：

> A self-supervised recurrent model trained only on interictal contact-rank
> sequences reconstructed patient-specific contact distributions that showed
> above-null spatial correspondence with early-ictal broadband energy fields.

紧接着必须加边界：

> This transfer was not specifically enhanced by the current symmetric-axis
> propagation constraint, indicating that the shared signal was primarily a
> patient-specific contact scaffold rather than a uniquely identified
> directional mechanism.

中文含义：

> 自监督间期序列模型保留了可迁移到发作早期的患者特异 contact scaffold；但当前对称轴
> 结构没有带来额外增量，因此尚不能把它解释成被模型辨识出的方向性机制。

### 9.2 不建议的写法

不能写：

- RNN 自动恢复了 A/B 病理轴；
- RNN 预测了发作传播顺序；
- latent state 对应真实兴奋/抑制；
- history-dependent replay 已得到证明；
- 结构化 RNN 优于所有简单对照；
- 34人都做了 clinical-onset transfer。

### 9.3 Figure 6 的可行分工

如果作为主文计算补充，可考虑：

| Panel | 科学问题 |
|---|---|
| A | 间期 contact-rank 自监督任务与两类模型 |
| B | 真实 vs full-history GRU contact-rank distribution |
| C | strict 16人 full GRU 与 all-contact null 的 static transfer |
| D | full GRU vs static/unordered/first-order/rank-shuffle |
| E | 22人双侧 source 有传播形态，但方向相对候选不特异 |
| F | 结构化 full/no-history/isotropic/axis-no-source 无独立增量 |

这个结构的中心不是“结构化 RNN 证明机制”，而是：

> 数据中有可迁移的 sequence-derived scaffold；最小轴向机制仍不足。

如果论文主线不希望主图包含这么强的 bounded-negative，建议把 C 放主文，把 D/E/F
放 Supplementary。

---

## 十、下一步 internal-state reduction v0.1

下一步不应立刻换新 GRU 或增加 rank。应先冻结现有 full-history GRU，检查其相对 static
hazard 的增量到底是什么。

已写独立 spec：

`docs/superpowers/specs/2026-07-28-topic5-rnn-internal-state-reduction-v0_1.md`

核心步骤：

1. 保存 heldout event 每个 prefix 的 hidden trajectory；
2. 只在间期 train/validation 上做 PCA/effective-rank 与跨 seed、split-half 稳定性；
3. 用低维 state 预测 next contact、future participation 和 remaining rank；
4. 与 last-set、rank-shuffle hidden state 做严格比较；
5. 沿稳定 hidden direction 做小幅 perturbation，观察 contact distribution 如何改变；
6. direction 完全冻结后才读 strict clinical-onset static target。

只有 full hidden state 同时超过 rank-shuffle 和 last-set，且 perturbation 的 contact field
效应跨 seed 稳定，才有资格写“latent transition”。否则收口为：

> GRU 的迁移优势主要是对患者静态 contact scaffold 的非线性平滑，而不是可辨识的
> ordered-history dynamics。

---

## 十一、产物

### 合同

- `docs/superpowers/specs/2026-07-28-topic5-rnn-bidirectional-cross-model-audit-v2_5.md`
- `docs/superpowers/specs/2026-07-28-topic5-rnn-internal-state-reduction-v0_1.md`

### 结果

- `results/topic5_rnn_bidirectional_cross_model_audit_v2_5/BIDIRECTIONAL_SUMMARY.json`
- `results/topic5_rnn_bidirectional_cross_model_audit_v2_5/bidirectional_patient_metrics.csv`
- `results/topic5_rnn_bidirectional_cross_model_audit_v2_5/STATIC_TRANSFER_SUMMARY.json`
- `results/topic5_rnn_bidirectional_cross_model_audit_v2_5/static_transfer_patient_metrics.csv`
- `results/topic5_rnn_bidirectional_cross_model_audit_v2_5/static_transfer_field_metrics.csv`
- `results/topic5_rnn_bidirectional_cross_model_audit_v2_5/static_transfer_paired_comparisons.csv`

### 图

- `results/topic5_rnn_bidirectional_cross_model_audit_v2_5/figures/cohort_bidirectional_static_transfer_summary.png`
- `results/topic5_rnn_bidirectional_cross_model_audit_v2_5/figures/representative_rank_distribution_comparison.png`
- 同名 PDF；
- `figures/README.md`。

### 复现入口

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/analyze_topic5_rnn_bidirectional_cross_model_v2_5.py \
  --section bidirectional

/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/analyze_topic5_rnn_bidirectional_cross_model_v2_5.py \
  --section static

/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python \
  scripts/plot_topic5_rnn_bidirectional_cross_model_v2_5.py
```
