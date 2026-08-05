# 给执行 agent 的无人值守 Prompt：Topic 5 shared-axis structured RNN v0.4

下面代码块可直接复制给执行 agent。它的目标是在 8--10 小时内完成实现、训练、统计、Figure 6 和报告，而不是只把任务挂起。

```text
你现在负责完成 Topic 5 / Figure 6 的 interictal–ictal shared-axis structured RNN v0.4。请设置一个本对话 goal，并持续执行到完整结果、图、报告和逻辑 commit 全部完成。不要在启动训练后停止回复“任务已挂起”；你必须通过 watcher 持续监控、断点续跑，并在训练结束后自动推进分析、target-free field freeze、early-ictal scoring、制图和验收。

工作目录：
/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-structured-rnn-fig6

分支：
codex/topic5-structured-rnn-fig6

必须完整阅读并严格执行：
1. docs/superpowers/specs/2026-08-05-topic5-interictal-ictal-shared-axis-rnn-v0_4.md
2. docs/archive/topic5/source_conditioned_shared_scaffold_rnn_v0_3_result_2026-08-04.md
3. docs/topic0_methodology_audits.md，尤其 lagPatRank phantom-rank 合同
4. docs/topic5_seizure_subtyping.md 中 early-ictal field、clinical onset 与 A/B axis 的当前口径
5. docs/figure_style_guide.md
6. AGENTS.md

唯一科学目标：
患者内 structured RNN 仅用间期 contact-rank sequences 学出一个稳定、可双向读取的 effective propagation axis。所有模型、source pools、rollout fields 和 hashes 冻结后，才读取同患者 clinical-onset 后 0--10 s、1--150 Hz control-normalized broadband energy，检验冻结间期轴与发作早期场的 target-free 对应。经验 A/B 只作冻结后的外部 read-back，不是训练标签或金标准。

不要把任务改成：
- 跨患者共享权重或 LOSO readout；
- 用 RNN 预测完整发作过程；
- 用 A/B 分类作为训练目标；
- 追求 structured 在纯 NLL 上超过 65 倍容量的 dense GRU；
- 用 early-ictal target 调模型、挑 checkpoint、挑患者或挑代表场；
- 再扩 architecture zoo。

开始时必须做：
1. `git status --short --branch` 和 `git diff -- scripts/paper_figures/plot_topic5_figure6_source_conditioned_rnn.py`。
2. 当前工作树已有用户/前序 agent 的未提交制图改动。必须保留，禁止 reset、checkout、clean、stash pop 或覆盖；本轮 commit 不得误收这处改动，除非你明确审计并说明它属于 v0.4。
3. 检查 GPU/CPU/RAM、现有 tmux/nohup 和残留训练进程；不要杀无关任务。
4. 新建且只使用结果根：`results/topic5_interictal_ictal_shared_axis_rnn_v0_4/`。绝不覆盖 v0.3 final。
5. 创建 `RUN_MANIFEST.json`，记录 git HEAD、dirty diff hash、spec/config/core/runner SHA256、Python/torch/CUDA、GPU、seeds、subject inventory、split fingerprints 与 target seal 状态。

数据与泄漏纪律：
- 间期使用 masked participant-only rank；任何 consumer 都必须显式证明没有消费 phantom finite ranks。
- 每位患者独立 fit60/validation20/test20；patient weights 不共享。
- 只允许三位 development patients 用于 interictal validation 超参数选择：epilepsiae_1073、epilepsiae_1146、yuquan_chenziyang。
- 在新的 `axis_freeze/FROZEN_AXIS_FIELD_MANIFEST.json` 完成以前，不得打开、反序列化或绘制 primary 患者的 ictal energy values，也不得浏览旧 early_ictal CSV 来做选择。
- E1146 继承既有 seal incident，只能 supportive；不能进入 primary P 值。
- target unseal 后一律禁止修改模型、loss、source pool、horizon、checkpoint、patient set 或 figure representative。

P0 实现任务：
1. 从 v0.3 复制出明确命名的 v0.4 config/core/runner/rollout/analysis 入口；不要在同一 output root 混 schema。
2. 实现 spec 中的 continuous-axis advection operator：
   - centered unit-RMS `s`；
   - symmetric `W_S`；
   - `W_Aij = W_Sij * tanh((s_i-s_j)/delta)`，严格反对称；
   - direction 只由 first rank 产生并在事件内冻结；
   - rank-set input 按 active contact count 归一；
   - same-shaft axis smoothness；
   - 不得有 dense contact bypass 或独立 forward/reverse matrices。
3. 保留 v0.3 的 contact-choice、STOP、cardinality likelihood 分解。主要间期终点必须是 `contact identity | continue,k`。
4. 新增/更新单测，至少覆盖：
   - W_S 对称、W_A 反对称、对角为零；
   - source 在两端时 direction 符号相反；
   - direction 不读取后续 rank；
   - event reset 完整；
   - sign flip `s -> -s` 在交换方向标签后预测等价；
   - source-excluded rollout 不含强制 source；
   - batch/scalar、CPU/CUDA、checkpoint reload 一致；
   - no dense contact mixer；
   - masked nonparticipants 不进入 rank loss；
   - target seal 与 manifest 顺序。
5. 使用 `/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python` 跑相关测试。测试不绿不得进入训练。

训练与时间预算：
- 目标总墙钟 8--10 小时；优先保证主链完整，不要把时间耗在次要 architecture sweep。
- 当前机器基线是一张 RTX 3090 24 GB、80 CPU、约 251 GB RAM；开始时重新探测。
- 所有 worker 设置 `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`。
- 先 4-worker smoke，再 12 workers；观察前 10--20 units 后最多升到 14。旧实测约 14 已饱和，禁止盲目升到 24；除非你有实时吞吐证据，且 GPU memory <20 GB、RAM <70%。
- 每 unit 独立 log、checkpoint、DONE.json/FAILED.json；atomic write；`--resume` 必须安全。
- OOM 后自动降低 workers 或 chunk，保留已完成 units 并续跑；不得删除整个结果根。
- 训练主任务：102 main structured + 34 rank-shuffle seed11 + 68 split-half seed11。ordinary GRU checkpoint 若 hash/split/schema 兼容则复用，只用 v0.4 source pools 重新 rollout；不兼容才重训。

无人值守调度：
- 建议建立 `tmux` 会话 `topic5_v04_train` 和 `topic5_v04_watch`。
- watcher 每 300 秒写 `PROGRESS.json`：expected/completed/running/failed/OOM/nonfinite、recent throughput、ETA、GPU memory/utilization、RAM、当前阶段。
- 驱动器必须按阶段自动推进：tests -> smoke -> dev audit -> formal train -> interictal summary -> split/axis audit -> field freeze -> target unseal -> ictal score -> Figure -> report -> acceptance。
- 任一步非零退出必须写 `PIPELINE_FAILED.json`，包含 stage、command、return code、log、已完成比例和恢复命令；完整链才写 `PIPELINE_COMPLETE.json`。
- 每 30--60 分钟在对话中给一次简短状态；不要只报告 GPU 占用，要报告科学阶段和完成分母。

development 选择只能读取 interictal validation：
1. lr = 0.01 / 0.03 / 0.1，三人、seed11，选 median validation contact NLL 最低者。
2. 正式预算固定为 28x32 updates；三人另跑 84x32 只作收敛 sensitivity。若长预算 median improvement >=0.005 nats/decision，记录训练预算 caveat，并只在总墙钟仍允许时追加长预算确认，不得阻止 28x32 primary 全链完成。
3. 选择结果写 `development/SELECTION.json`，明确 `ictal_target_values_read=false`、`test20_used=false`。
4. 正式 config 冻结后重新写 hash；formal train 期间不得编辑 core/runner/config。若必须修代码，停止、换新 output root suffix 并从头冻结，不能混 hash。

间期分析必须产出：
- 31 位 development-excluded confirmation 和 34 位 description；
- patient-level contact-choice NLL、top1、structured-static、structured-ordinary、true-rankshuffle；
- 两个 learned source sides 各自的 source-excluded observed-vs-rollout expected-rank Spearman；
- matched-minus-swapped bidirectional margin；
- 三 seed sign-aligned axis stability、split-half stability、endpoint Jaccard；
- all eligible 与完全 target-independent 的 axis-identifiable subset；
- masked train-only empirical A/B read-back，仅作 external validation；
- 不得把不同起点造成的两个 rollout 直接写成“模型自发发现了两类”。

field freeze：
1. 用三 seed ensemble 的 learned `s` 两端各 `max(2, ceil(0.2N))` contacts 定义 source pools。
2. structured full、flow lesion、ordinary GRU 共用 source pools、horizon、exact-k sampler、rollout seeds 和 denominator。
3. 每方向每 seed 5000 rollouts；冻结唯一 first-arrival earliness F-/F+ 与 axis contrast G。
4. 生成至少 256 个 shaft-preserving axis-permutation fields/nulls；不得根据 ictal target 选择轴。
5. `FROZEN_AXIS_FIELD_MANIFEST.json` 必须列出每位患者/模型/seed/checkpoint/contact order/source pools/F-/F+/G 的 SHA256，并写 `target_values_read=false`。
6. manifest 写完并通过独立 validator 后，才允许 target unseal。

early-ictal 评分：
- primary：clinical onset 后 0--10 s、1--150 Hz、control-normalized broadband energy；exact contact join；seizure-first/patient-first。
- 每 seizure 的主 score 是 max(|rho(F-,Y)|, |rho(F+,Y)|)。
- axis-specific score 是 |rho(G,Y)|。
- primary null 是论文当前冻结的 all-contact contact-label permutation，5000 次；每次重做 absolute 和 two-direction max。within-shaft 是 sensitivity。
- 比较 structured full、flow lesion、axis permutation、ordinary GRU、static participation、empirical A/B。
- exact Wilcoxon 前用 1e-9 tie band，报告正/负/并列、bootstrap 95% CI、individual p95 count。
- all eligible 和 axis-identifiable 两层都展示；不能只保留更好看的层。
- 结果阴性也必须完整收口；target unseal 后不准再调参。

Figure 6：
- 按 spec 的 A--F 六块制作，读取 `docs/figure_style_guide.md` 并复用当前 paper-ready figure 的字体、线宽、留白、患者配对点和统计语法。
- A 是真实 structured RNN 计算结构，不是流程框图。
- B 是 E1146 两个 source-conditioned directions 的 observed-vs-rollout rank heatmaps，必须排除 source 后展示 profile。
- C 是 cohort interictal prediction + bidirectional recovery。
- D 是 axis seed/split stability 与 flow lesion necessity。
- E 是同一几何上的 F-/F+/G/clinical-onset early field，所有模型场共享尺度语义。
- F 是 patient-level cross-state full vs lesion vs GRU vs static vs empirical A/B；E1146 空心 supportive。
- 输出 PDF、SVG、600-dpi PNG、每 panel source CSV、statistics JSON、中文 figures/README.md。
- 图生成后必须逐 panel 目视 QA，核对文字遮挡、色标、contact 顺序、患者分母、P 值和 source-data 一致性；不能只说脚本成功。

验收与报告：
1. 运行相关测试与 artifact audit；全仓测试若超过时间预算，可给出 targeted tests 全绿和未跑全仓的明确边界，不能谎称全仓 green。
2. 写详细中文白话报告到 `docs/archive/topic5/`：测了什么、怎么测、间期轴是否学到、哪些患者轴稳定、early field 是否对应、full 是否超过 lesion/GRU/static、两个数据集是否同向、限制是什么、能写/不能写什么。
3. 更新 `docs/archive/topic5/INDEX.md` 和图 README。
4. 按实现与测试 / 正式结果 / Figure 与 source data / 文档验收分逻辑 commit。不要把开始时已有的无关 dirty plot 改动误提交。
5. 不 push；等待用户明确要求。

最终回答必须直接给：
- goal 是否完成；
- 完成 units/失败/OOM/NaN；
- 间期 NLL 与双向 source-excluded 指标；
- axis seed/split stability 与 axis-identifiable 人数；
- clinical-onset early-ictal full/null/lesion/GRU/static/empirical A/B 的 patient-level 结果；
- 当前证据达到 spec 的 Level 0/1/2/3 哪一级；
- Figure 6 与中文报告的绝对路径；
- commits；
- 未完成事项和为什么。
```
