# Axis A — 病灶内部机制 → 传播指纹 → 真实数据对应（plan）

**Date:** 2026-06-14 · **Status:** PLAN（review-corrected r1+r2 2026-06-14），A0 GO / 其余 gated · **Topic:** Topic 4 SEF-HFO 观测层
**Origin:** 用户 2026-06-14 定向——"主要做 A：先复刻 Stage-2 单灶，然后双灶看不同异质性导致的传播特征差别，有差别再看能不能和部分 subject 数据对应"。B 轴（长记录标签独立性）与并行工作重复，本轮只落文档不再探索。

---

## §0 一句话目标 + 口径（朴素）

**做什么** —— 用模型问一个问题：**病理核内部"长什么样"（兴奋/抑制失衡 + 阈值离散程度）会不会改变间期 HFO 在网络上传播时留下的"指纹"**（往哪传、传多快、入口抖不抖、通路多宽、招募多远）；**如果会，这些不同的指纹能不能在真实病人的传播特征分布里找到对应的位置**。

**口径锁定（先钉死，避免越界）：**
- 这是**机制充分性 + 对应性**探索，**不是因果断言**。允许说"模型里把抑制刹车调坏，传播指纹朝 X 方向变；真实队列里确实有一批病人落在 X 区"；**禁止**说"病人甲是抑制塌陷型"。
- **对应 = 描述性落位**（真实指纹是否落在模型机制扫出来的指纹空间里），不是分类/反演。
- 传播指纹是**观测层**特征（透过虚拟 SEEG 读出来的），和病人数据可比，因为病人数据也是同样的传播读出。

**我做的一个解读选择（请用户确认）** —— 用户说"不同异质性"。我把"病灶内部机制"读成**两个旋钮族**：(1) **阈值离散程度**（`core_std`，便宜、不改引擎）；(2) **E/I 失衡**（抑制刹车失灵 `w_EI↓` / 自激加粗 `w_EE↑` / 抑制延迟 `τ_I`，需要改守护引擎，见 axis-A spec）。A 轴主线是 (2)，(1) 是便宜的先导。若用户只想要 (1) 或只想要 (2)，把对应 stage 删掉即可。

### §0.1 审阅修复后状态（2026-06-14，按 review P0/P1）

本轮按审阅报告改了 6 处，每处都在代码/spec 里核实过站得住：

- **P0 指纹 schema（§1 + A0）**：当前读出只落 `n_part/axis_err/sign/ranks`、`rep_npz` 是单代表事件——不够支撑跨事件 jitter/speed。A0 目标从"写抽取函数"改成 **event-level schema freeze**（落每事件一行 + 单位 + primary/secondary 分层 + engine 签名/seed/rate/collision/excluded 计数）。
- **P0 A4 同发（§A4）**：删掉"同发不致命"。同发=混合波形，不是单灶指纹。A4 主分析改成 **paired single-focus**；true two-focus 只用 sidecar `hidden_source_label` 非 collision 的 clean events，不足则 no-go。
- **P1 A1 工作点混淆（§A1）**：拆成 **0a ignition feasibility**（不写科学差异）+ 正式 A1（**fixed-mean 或 matched-rate/matched-drive**，pathology mapping spec §5.0 合同 2）。窄档只能靠降 mean 点着 → 只写"匹配工作点下的窄档指纹"。
- **P1 A5 过解释（§A5）**：先做 **real-fingerprint 可靠性审计**（split-half/odd-even 稳定才进落位、按采样几何分层、geometry-matched null），输出写覆盖率，**禁止** subject-level 机制 label。
- **特征分层（§1）**：`speed` / `event_size` **deferred 不进 primary**（毫秒 latency / 幅度可比性不满足）；primary = `pathway_width` + `onset_jitter` + `axis_dir` 门。
- **工程（A2/A3/各 stage）**：`w_EI↓`=target-in-core、`w_EE↑`=source∧target-in-core、`tau_I`=global 后置（axis-A spec §1）；事件率匹配带 A1/A3/A4 共用；每个出图目录补 `figures/README.md`；summary JSON 落 engine 签名等。

**round-2 审阅又补了 3 处（2026-06-14）**：
- **P1-lite `n_min_events` 门**（§1 + A1/A3/A4 门）：每 condition/seed 够数的 clean events 才进汇总，不足报 INSUFFICIENT，防"率匹配了但统计靠几条事件撑"。
- **P2 `amplitude proxy` 标 provenance/diagnostic**（§1）：落盘但不进主比较，防它暗中替 deferred 的 `event_size` 回到 primary。
- **§5 决策锁定**：两异质性都要但本轮只跑 A0+A1+A5-prep（E/I 只 Step 0 read-only）；A4 只做 paired single-focus；`speed`/`event_size` deferred 确认。

**Go / No-Go（round-2 §4）**：A0 = **GO**（可立刻开跑）；A1 = CONDITIONAL（A0 schema 冻结后，先 0a 再 matched）；A5-real audit = CONDITIONAL（先 inventory，正式抽 fingerprint 等 A0 schema）；A4 paired = WAIT（等 A0+A1 初步）；A2/A3 = WAIT（只允许 read-only engine Step 0，不改 engine，直到 Stage 3 单网结构函数 + 索引契约钉死）。

**本轮并行 agent 拓扑（按 round-2 §5）**：Agent 1 = A0 Schema Lead（阻断主线，写 `src/sef_hfo_fingerprint.py` + TDD + baseline）；Agent 2 = Real-Data Audit Prep（只读 inventory）；Agent 3 = Engine Step 0 Auditor（只读 `build_connectivity_rot`，不改 engine）；+ QA Gatekeeper（查 figures/README、summary JSON 字段、deferred 纪律、无 subject-level 机制 label）。A1 Feasibility Runner 等 A0 schema 冻结 + 用户审阅后才启动。

---

## §1 传播指纹 = 测量仪器（先冻结 schema，再跑任何科学 sweep）

每个**干净事件**从读出里抽一行 event-level 记录，跨事件聚合成"传播指纹"。全部复用已有读出机器（`src/sef_hfo_observation.py` 的 `endpoint_centroid_axis` / 每触点 onset / `direction_readability`，以及 `src/sef_hfo_stage3.py` 的 `entry_jitter_stats`）。

**关键纠正（P0：指纹仪器还没有完整数据合同）**：当前读出 `read_event` 只落 `n_part / axis_err / sign / ranks`，代表事件 `rep_npz` 只是单个代表事件——**不足以**支撑跨事件的 jitter / speed 统计。所以 A0 的第一件事**不是**"写个 `extract_fingerprint` 函数"，而是**冻结一个 event-level 测量合同**（schema freeze）：字段、单位、primary/secondary 分层全部先钉死，避免事后挑特征。

**event-level 记录合同（每个干净事件一行，A0 落盘 schema）**：source label（双灶时来自 sidecar hidden label，**非** direction sign）、template/direction、每触点 rank、每触点 onset 时刻、每触点坐标、participation mask、事件窗口、peak 时刻、amplitude proxy（**provenance/diagnostic，落盘但不进 A1/A4/A5 主比较**，P2）、engine 版本签名、seed、是否 collision/ambiguous。

| 特征 | 朴素含义 | 来源 | 层级 |
|---|---|---|---|
| `axis_dir` | 往哪个方向传（轴角，对几何轴的偏差） | `endpoint_centroid_axis` / `axis_err` | primary（同时是可读性门） |
| `pathway_width` | 波垂直于传播轴有多宽（⊥ 杆参与度 / ⊥ 空间展宽） | ⊥ 杆触点参与 + 坐标 | **primary** |
| `onset_jitter` | 每次事件"最早入口"漂多少（跨事件的入口触点离散） | `entry_jitter_stats`（top1/top3 fraction、n_unique） | **primary** |
| `latency_jitter` | 每触点达峰时刻跨事件抖多少 | 每事件每触点 onset 的跨事件方差 | secondary（先证仪器不从稀疏触点/坏轴造信号） |
| `recruit_extent` | 传多远 / 点亮几个触点 | `n_part` + 空间跨度 | secondary |
| `speed` | 传多快（沿轴 onset→peak 的时间 ÷ 空间跨度，mm/ms） | 每触点 onset | **deferred（不进 primary）**：需真实毫秒 latency 对齐，模型 dt-bin onset 与真实 HFO 时间分辨率不可直接比 |
| `event_size` | 事件大小（幅度 / 参与触点数） | envelope 幅度 + `n_part` | **deferred（不进 primary）**：模型 LFP/firing proxy 幅度与真实 HFO 幅度不可直接比 |

**承重纪律**：
- 这个 schema 在跑任何病灶变体之前**先冻结**（避免事后挑特征）；primary = `pathway_width` + `onset_jitter`（+ `axis_dir` 作可读性门），其余 secondary/deferred 只描述、不进主比较；`amplitude proxy` 仅 provenance/diagnostic，不进主比较。
- **最低可用事件数门（`n_min_events`，P1-lite）**：每个 condition / seed 至少 `n_min_events` 个 clean readable events 才进 fingerprint 汇总，且 pooled clean events 要够稳定估计 `onset_jitter` / `pathway_width`；不足报 **INSUFFICIENT**，**不进入组间比较**（防"事件率匹配了、但统计靠几条事件撑起来"）。`n_min_events` 的具体值在 A0 schema freeze 时定死并写进 summary。
- **坏数据回归对照**：拿一个"已知没有方向"的事件（⊥ 杆同步达峰）喂进去，`axis_dir` 应判不可读、`speed` 应为 NaN——确认仪器不会无中生有；同时验 `pathway_width` / `onset_jitter` 不从稀疏触点或坏轴里造信号。

---

## §2 分阶段计划

### Stage A0 — 冻结指纹 schema + 验证抽取器（schema freeze，先于一切科学 sweep）
- **目标改成 schema freeze + extractor validation**（P0）：A0 没冻结字段 / 单位 / primary-secondary 分层之前，**后面任何科学 sweep 都不许跑**。
- 复用现成单灶 run（`oneend_neg` / `oneend_pos`，工作点 `mean=17.0 wide`，即已验证的 Stage-2 main）→ 干净方向模板 + 它的传播指纹 = **参考指纹**。
- 实现 §1 的 event-level 抽取器 `src/sef_hfo_fingerprint.py`，产出**稳定 artifact schema**（不是只写 `extract_fingerprint(readout_json, rep_npz) -> dict`）：每个干净事件落一行 event-level table（§1 合同字段）+ 跨事件聚合的指纹向量。TDD（用合成事件测每个 primary 特征 + 坏数据回归）。
- **summary JSON 必须落**：engine 版本签名、seed、config、clean-event rate、collision rate、excluded-event counts（collision/ambiguous 剔除计数）。
- **门**：单灶参考指纹能稳定抽出（`axis_dir` 可读、`pathway_width`/`onset_jitter` 有限值且跨 seed 一致）；坏数据事件被正确判不可读。
- 产出：`results/topic4_sef_hfo/observation_layer/snn_cm_spontaneous/fingerprint/baseline_oneend.json` + `figures/README.md`（中文逐图说明）+ 一张 Stage-2 风格图（复用 train + mechanism plot）。

### Stage A1 — 阈值离散程度 → 指纹（拆成 feasibility + matched-control，便宜，不改引擎）
- **0a（ignition feasibility，不写科学差异，P1）**：先扫哪些 `(mean, std)` 组合能点着（窄档 0.5 在 mean=17.0 已知 gate-fail）。这一层**只报可点火性**，不下"离散度导致指纹差异"的结论。
- **正式 A1（mean-/event-rate-matched control，P1）**：正式比较**不允许**靠单独降 mean 把窄档点亮后直接比——否则 `core_std` 效应和 `V_th mean` 工作点效应混掉。每个离散档要么 **fixed-mean**，要么作为 **mean-rate-matched / mean-drive-matched control** 报告（重定均值使收窄方差不顺带移动平均工作点——pathology mapping spec §5.0 合同 2）。若窄档只有降 mean 才点着，只能写"**匹配工作点下的窄档指纹**"，**禁止**写"阈值离散度本身导致 fingerprint 差异"。
- 在匹配工作点下各跑 ≥3 seed，每档抽指纹，问：**离散程度变了，primary 指纹（`pathway_width` / `onset_jitter`）有没有显著差别**。统计：跨 seed 的指纹分布，三档之间非参检验 + 效应量；主读数预注册 `pathway_width` + `onset_jitter`。
- **门**：匹配工作点（fixed-mean 或 matched-rate）前提下，三档主读数是否分得开（效应量 + 坏数据对照：同 seed 同工作点重复跑应给同档内一致指纹）。**每档先过 §1 `n_min_events` 门，不足报 INSUFFICIENT 不进组间比较。**
- 产出：`fingerprint/heterogeneity_sweep.{csv,png}` + `figures/README.md`（三档指纹对比，T4 模型 3 行风格，见 `docs/figure_style_guide.md`）。

### Stage A2 — 实现 E/I 局部病灶（改守护引擎，按 axis-A spec）
- **前置依赖**：A2/A3 在 **Stage 3 single-network structure function ship + 本 Step 0 索引契约钉死之后**才动（axis-A spec §4/§6：axis A 直接消费 Stage 3 抽出的单网结构函数）。
- 按 `docs/superpowers/specs/2026-06-13-sef-hfo-snn-axisA-ei-local-lesion-design.md`：给 `engine/connectivity*.py` 的建连循环加一个**按神经元的空间权重缩放场**，在核内对 `w_EI`（抑制刹车）/`w_EE`（自激）缩放。
- **Step 0（先做，索引契约是科学决定、非代码细节，spec §1）**：读 `build_connectivity_rot` 端到端，钉死——`w_EI↓`（perisomatic 抑制塌陷）缩放**target E 神经元在核内**的连接（被刹的是 target，**不是** presynaptic `local_scale[i]`）；`w_EE↑`（复发兴奋簇）缩放 **source∧target 都在核内**（至少 source-in-core，Step 0 定）。`tau_I` 是**全局** intrinsic 轴、不是空间场 → **不进首批局部双灶异质性变量**（后置；Step 0 先判断局部版是否有意义）。
- TDD：建连权重场的小单元测试（核内权重被正确缩放、核外不变；区分 target-indexed vs source-indexed 两种缩放路径）。
- 改完**重新签名引擎**（`engine_versions.json`，logged diff）。
- 病灶模式：`twoend_inhib`（`w_EI↓`）/ `twoend_recur`（`w_EE↑`）/ `twoend_combined`，先做单灶版 `oneend_inhib` / `oneend_recur`。
- **门**：引擎守护通过（改后能跑）；E/I 病灶单灶能点着并产生事件。

### Stage A3 — E/I 机制 → 指纹（事件率匹配，主线）
- 单灶 E/I 病灶（抑制塌陷 / 自激 / 合并），**调病灶强度让事件率落在 baseline 的 0.8×–1.25× 带内**（axis-A spec 锁定的事件率匹配门——不能靠"更容易点火"冒充机制差异）。**这条 0.8×–1.25× 带是 A1（matched control）/ A4（双灶对比）共用的同一条事件率匹配纪律**——任何"更容易点火"都不许冒充"机制指纹不同"。
- 问两件事：(a) E/I 病灶**是不是也能产生干净方向模板**（A 轴原始问题：更病理的机制也能给双向模板吗）；(b) 它的**指纹和 V_th↓ / 和阈值离散档**比，有没有可区分的差别。
- **门**：事件率匹配前提下，E/I 病灶产生可读方向模板（`axis_dir` 可读、`stable_k` 结构在）；指纹差别有效应量。**禁止**未匹配事件率就比指纹。**每 condition 先过 §1 `n_min_events` 门，不足报 INSUFFICIENT。**
- 产出：`fingerprint/ei_mechanism.{csv,png}` + `figures/README.md` + 每个 E/I flavor 的 Stage-2 风格图。

### Stage A4 — 双灶不同内部属性 → 受控对比（先 paired single-focus，再 true two-focus）
- **同发不是无害噪声（P0 纠正）**：两个灶同发时，读出的是**混合波形 / 混合传播**，不是某个灶自己的 fingerprint。Stage 3 已定 source label 要看 sidecar hidden label、**不能用 direction sign 代替**；double-focus clean-source gate 没过时，**不能**抽"每个灶各自指纹"。
- **A4 主分析 = paired single-focus**：先做"分别单灶、同 seed / 同 geometry / 同工作点"的成对对比——阈值离散对（一端宽 `std=1.5`、一端中 `std=1.0`，**不要宽+窄**，窄会被饿死）；再做 E/I 对（一端抑制塌陷、一端自激）。这是最干净的"内部属性 → 指纹差异"读法。
- **true two-focus = Stage-3 extension**：真正同网双灶**只允许**用 sidecar `hidden_source_label ∈ {neg, pos}` 且**非 collision** 的 clean events，collision/ambiguous 全部排除。若 clean source 事件不足（参考 Stage-3 pilot：L=20 `twoend_equal` 一核独大 + 38% 同 bin 伪同发），A4 报 **no-go**，**不做**双灶 fingerprint 结论（不靠 L=40 大网硬上，除非另立 spec）。
- 问：**两个不同内部属性的灶，传播指纹是不是系统性地不同**（同 seed / 同工作点，唯一变量是病灶内部属性；事件率按 A3 的 0.8×–1.25× 带匹配）。
- **门**：两灶指纹差别 > 同属性重复跑的差别（within-property 噪声做对照）；每个灶 / 每个属性先过 §1 `n_min_events` 门（不足报 INSUFFICIENT）；true two-focus 额外门 = sidecar clean-source 事件数足够。
- 产出：`fingerprint/two_focus_contrast.{csv,png}` + `figures/README.md`。

### Stage A5 — 和真实 subject 数据找对应（capstone；先做 real-fingerprint 可靠性审计）
- **先做 A5-real feasibility/reliability audit（P1，纯数据、不依赖模型）**：真实队列每个指纹特征的**可得性 + split-half/odd-even 稳定性 + 采样几何依赖**。**只有可靠（split-half/odd-even 稳定）的特征才进落位**；`speed` / 幅度类不进（受采样几何 / 模态 / 触点密度污染）。Topic 4 目前最稳的桥接是 geometry scalars 约束模型、不是机制分类——A5 守这条。
- 取真实间期传播指纹分布：复用 `results/spatial_modulation/propagation_geometry/`（已建好的真实 per-subject 轴 + 二维触点平面）+ `results/interictal_propagation_masked/per_subject/`（masked 模板），抽**同一组通过可靠性门的 primary 指纹特征**（轴向、入口散度、通路宽度…）。
- **按 Yuquan / Epilepsiae 或采样几何分层**；必须有 **geometry-matched null / shuffle null**（三层对照同 B 轴归档纪律：坏数据 / 打乱 / 几何匹配 null）。
- 问：模型按内部属性扫出来的指纹空间，能不能**覆盖**真实队列的指纹变异。方法：把模型指纹和真实指纹投到同一标准化特征空间，看真实点是否落在模型流形内 + 每个真实 subject 最近的模型机制档（描述性最近邻，**不做分类断言**）。
- **门 + 口径**：输出写"真实队列的传播指纹变异，被模型的内部属性轴张成的空间**覆盖了 X%**"；**禁止**写 subject-level mechanism label（"病人甲=抑制塌陷"）。复用 `corr_pair_mirror_invariant` / `compare_model_to_cohort` / `placement_in_distribution`（topic3↔4 几何读出的 real-vs-model 落位方法，A5 原样换成 fingerprint 空间）。
- 产出：`fingerprint/model_vs_real_placement.{csv,png}` + `figures/README.md` + archive doc。

---

## §3 复用清单（不重造）
- 读出 + 指纹来源：`src/sef_hfo_observation.py`、`src/sef_hfo_stage3.py::entry_jitter_stats`、读出 runner `scripts/run_sef_hfo_snn_cm_spontaneous_readout.py`（已有 `--core-std` / `--sep-frac` / `--drive`）。
- E/I 引擎改动：`docs/superpowers/specs/2026-06-13-sef-hfo-snn-axisA-ei-local-lesion-design.md`（A2/A3 直接照它做）。
- 真实数据 + real-vs-model 落位：`results/spatial_modulation/propagation_geometry/` + `src/propagation_contact_plane_readout.py`（`corr_pair_mirror_invariant` / `compare_model_to_cohort` / `placement_in_distribution`）。
- 图风格：`docs/figure_style_guide.md`（T4 模型 3 行 = locked exemplar）。
- 资源纪律：所有 sim 走 RAM 预检 + 低并行（B 轴归档里 OOM 教训）。

## §4 执行顺序（schema 先冻结、便宜先行、改引擎后置）
1. **A0（schema freeze + 抽取器验证，TDD）**——立即；**schema 没冻结前不跑任何科学 sweep**。
2. **A1-0a（ignition feasibility）→ A1（matched-control）**——便宜，先看离散度在匹配工作点下有没有指纹效应（若没有，E/I 那条更值得投）。
3. **A5-real feasibility/reliability audit**——便宜、纯数据，决定 A4/A5 对应有没有可靠信号空间（哪些特征 split-half 稳定、是否依赖采样几何）。
4. **A2→A3（E/I 引擎 + 机制指纹）**——改引擎、重头；**前置依赖 = Stage 3 single-network structure function ship + engine Step 0 索引契约钉死**。
5. **A4（先 paired single-focus；true two-focus 需 Stage-3 sidecar clean source）→ A5（落位，仅用过可靠性门的 primary 特征）**——capstone。

## §5 本轮决策（锁定，按 review round-2 §6 default + 用户授权"按建议执行"）
1. **异质性范围**：两者都要（阈值离散 + E/I），但**本轮只跑 A0 + A1 + A5-real prep**，E/I 只做 read-only Step 0（不改 engine）。
2. **Stage A4**：本轮**只做 paired single-focus**，**不**尝试 true two-focus（true two-focus 留待 A0+A1 出初步结果后、且 Stage-3 sidecar clean-source 足够才考虑）。
3. **特征分层**：确认 **primary = `pathway_width` + `onset_jitter`**（+ `axis_dir` 作可读性门）、`speed` / `event_size` **deferred 不进 primary**、`amplitude proxy` 仅 provenance/diagnostic。

（内部归档代号：axis A = E/I 局部病灶 + 阈值异质性 → propagation fingerprint → real-cohort placement；fingerprint primary = {axis_dir, pathway_width, onset_jitter}，secondary = {latency_jitter, recruit_extent}，deferred = {speed, event_size}；event-level schema freeze = A0 落盘合同（source label 走 sidecar hidden label）；引擎扩展见 axisA-ei-local-lesion-design spec（`w_EI↓`=target-in-core / `w_EE↑`=source∧target-in-core / `tau_I`=global 后置）；real 数据 = propagation_geometry + interictal_propagation_masked；落位 = corr_pair_mirror_invariant / compare_model_to_cohort / placement_in_distribution；事件率匹配带 0.8×–1.25×（A1/A3/A4 共用）。）
