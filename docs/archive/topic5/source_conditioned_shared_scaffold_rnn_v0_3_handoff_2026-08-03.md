# Topic 5 source-conditioned shared-scaffold RNN v0.3 handoff

**日期**：2026-08-03  
**工作树**：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-structured-rnn-fig6`  
**分支**：`codex/topic5-structured-rnn-fig6`  
**基线提交**：`5cb65936`  
**当前状态**：未提交；v0.3 core 为半成品，不可直接训练；无 tmux 任务在运行。

## 1. 唯一科学目标

需要完成并检验的是：

> 患者内有结构的 RNN 能否从间期 contact-rank sequences 中恢复患者特异的传播结构；冻结模型派生的双向空间场能否与同一患者发作早期 1--150 Hz broadband energy field 形成 target-free 跨状态对应，并与 ordinary dense GRU 和 static baseline 比较。

这里不做跨患者读出，不把经验 A/B 当训练标签，不追求普通 GRU 的最高 AUC，也不把 RNN hidden unit 解释成真实 E/I 神经元。

Figure 6 的目标顺序已经冻结：

1. **A**：新 structured RNN 的结构与为什么它能用同一 scaffold 表达双向传播；
2. **B**：代表患者中，观察到的两类相反间期传播与模型 rollout 的逐 rank 对应；
3. **C**：全 cohort 的 held-out next-contact prediction 和 rollout consistency；
4. **D**：冻结 structured fields 与同患者 early-ictal field 的直观空间对应；
5. **E**：15 位 primary 患者的跨状态统计，以及 structured、ordinary、static 的比较。

## 2. 已完成内容

### 2.1 v0.2 工程链

已经实现并测试过：

- 患者内 chronological `fit60 / validation20 / test20`；
- static、ordinary dense GRU、structured、structured rank-shuffle；
- contact-choice、STOP、cardinality 分解；
- exact conditional k-subset likelihood；
- resume-safe 多进程训练、日志、checkpoint 和 monitor；
- exact-k rollout sampler；
- target seal、field manifest、all-contact/within-shaft permutation scoring；
- patient-first 间期汇总脚本。

v0.2 修改前的相关测试曾达到 38 项全通过；field/readout 与 runner 单测也已存在。所有相关文件目前都是该工作树中的未跟踪文件，不能使用 `git clean` 或覆盖式恢复。

### 2.2 数据和 target metadata 审计

- 间期 rank dataset：34 位患者，864,163 场事件，单患者 6--52 个 contacts；
- early-ictal primary：15 位；`epilepsiae_1146` 为 supportive；
- 共 33 次 seizures；exact join 后每位 6--16 个 contacts，中位 9；
- target values 至今没有读取；seal 仍保持关闭状态。

### 2.3 v0.2 bounded diagnostic

v0.2 正式 clean run 在 18/306 units 时被主动停止，0 FAILED、0 OOM、0 NaN。结果保存在：

`results/topic5_patient_specific_shared_scaffold_rnn_v0_2_final/`

诊断说明：

`results/topic5_patient_specific_shared_scaffold_rnn_v0_2_final/DIAGNOSTIC_ONLY_NOT_FOR_FIGURE6.md`

首个完整患者的代表数字：

- static test contact NLL：1.794252；
- symmetric-only structured：1.794758；
- structured rank-shuffle：1.794854；
- ordinary GRU：1.760245。

因此旧 structured 几乎退化到 static，true order 与 rank shuffle 只差约 0.0001。它再次把“近对称底层 scaffold”错误地直接等同成“观察到的单步 transition operator”。该结果只作诊断，不得进入 Figure 6，也不得与 v0.3 混池。

旧 mixed-hash root 与 clean root 的训练数学已经逐位审计为等价；这只用于 provenance，不改变 v0.2 的科学停跑决定。

## 3. v0.3 新 structured RNN 的精确合同

每位患者只学习一条有符号 contact coordinate `s_i`。先中心化并用 population RMS 标准化，再生成两端连续 membership：

\[
a_i=\sigma(-s_i/T),\qquad b_i=\sigma(s_i/T).
\]

同一组 membership 同时产生对称 scaffold 与反对称 flow：

\[
K^S=ba^\top+ab^\top,\qquad K^A=ba^\top-ab^\top.
\]

其中 `K^S` 对称、`K^A` 反对称，二者 rank 均不超过 2。固定 same-shaft local graph 只能加入 `K^S`。用同一 symmetric degree scaling 得到 `W_S` 与 `W_A`，保持：

\[
W_S=W_S^\top,\qquad W_A=-W_A^\top.
\]

每场事件只用已经观察到的第一 rank set `x_0` 确定方向：

\[
d_e=\tanh\left[\kappa\,\operatorname{mean}_{i\in x_0}(a_i-b_i)\right].
\]

`d_e` 在该事件剩余 rank steps 中冻结。状态更新固定为：

\[
P_{t+1}=\rho_PP_t+W_Sx_t+\lambda_A d_eW_Ax_t,
\]

\[
R_{t+1}=\rho_RR_t+W_Sx_t,
\]

\[
z_{t+1}=b+\beta_PP_{t+1}-\beta_RR_{t+1}+m_t.
\]

推荐初始化：`betaP≈1`、`betaR≈0.25`、`skew_gain=lambda_A≈0.5`、`direction_gain=kappa≈2`；正值参数用 softplus，persistence 用 sigmoid。这样避免 v0.2 初始时 propagation 与 restraint 几乎完全抵消。

### 严禁绕行

- 不得加入 dense contact decoder 或 MLP contact mixer；
- 不得为 forward/reverse 学两套独立 operator；
- 不得输入经验 A/B、mean rank、SOZ、clinical label 或 ictal target；
- 不得按 early-ictal 结果选择 axis、field、checkpoint、学习率或代表患者；
- `W_A` 必须由与 `W_S` 相同的 `a/b` membership 派生；
- ordinary GRU 只作同任务对照，不向 structured 模型复制 dense mixing。

## 4. 当前代码的精确停点

`src/topic5_shared_scaffold_rnn.py` 只完成了 operator helper 的第一步：

- 已新增 `source_conditioned_shared_scaffold(...)`；
- 已实现 signed coordinate、`a/b`、`K^S/K^A`、`W/W_skew`；
- 已返回 `axis_coordinate`、`endpoint_minus`、`endpoint_plus` 等组件。

但是 `SharedScaffoldPropagationRNN` 类仍引用已经删除的 `symmetric_shared_scaffold` 和旧 `low_rank_raw`。因此当前 core 预计无法导入或运行。

下一位 agent 必须先完成：

1. 将类参数从 `low_rank_raw` 改成 `axis_coordinate_raw`；如需 CLI 兼容，可保留 `low_rank=2`，但其他值必须报错；
2. `PropagationRestraintState` 增加 `direction` 与 `source_initialized`；
3. 第一 rank 因果计算 `d_e`，之后冻结；event reset 时一起清零；
4. 接入 `W_S + d_e W_A` 的 propagation 更新；restraint 只走 `W_S`；
5. 修正参数初始化，避免 `P/R` 抵消；
6. 更新 `operator_components()`、checkpoint metadata 与 `__all__`；
7. 更新 rollout/field 读取，使其识别 v0.3 operator 字段；
8. 更新测试并确保 runner/config hash 与 checkpoint schema 一致。

必须补齐的 core 测试：

- `W_S` 对称、`W_A` 反对称；
- analytic components rank 不超过 2；
- 两端 source 得到相反 `d_e` 和镜像 flow；
- direction 只由 first rank 决定，后续不重算；
- event reset 清除全部状态；
- batch/scalar 一致；
- CPU/CUDA 在容差内一致；
- structured contact logits 不存在任何绕过 operator 的 dense path。

## 5. 现有文件

### 冻结文档与配置

- `docs/superpowers/specs/2026-08-03-topic5-source-conditioned-shared-scaffold-rnn-fig6-v0_3.md`
- `docs/superpowers/plans/2026-08-03-topic5-source-conditioned-shared-scaffold-rnn-fig6-v0_3.md`
- `config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml`
- `config/topic5_source_conditioned_ictal_readout_v0_3.yaml`

### Core、runner、rollout、readout

- `src/topic5_shared_scaffold_rnn.py`
- `src/topic5_shared_scaffold_rollout.py`
- `src/topic5_shared_scaffold_field_readout.py`
- `scripts/run_topic5_shared_scaffold_rnn_unit_v0_2.py`
- `scripts/launch_topic5_shared_scaffold_rnn_v0_2.py`
- `scripts/watch_topic5_shared_scaffold_rnn_v0_2.py`
- `scripts/analyze_topic5_shared_scaffold_interictal_v0_2.py`
- `scripts/freeze_topic5_shared_scaffold_rollout_subject_v0_2.py`
- `scripts/launch_topic5_shared_scaffold_rollouts_v0_2.py`
- `scripts/freeze_topic5_shared_scaffold_field_manifest_v0_2.py`
- `scripts/audit_topic5_shared_scaffold_ictal_metadata_v0_2.py`
- `scripts/score_topic5_shared_scaffold_early_ictal_v0_2.py`
- `scripts/run_topic5_source_conditioned_lr_audit_v0_3.py`

脚本文件名仍含 `v0_2`，但多数支持显式传入 v0.3 config。不要仅凭文件名判断合同；运行前必须检查输出 JSON 中的 contract、config SHA、runner SHA 和 core SHA。field manifest 的 contract/schema 如仍硬编码 v0.2，必须在正式冻结前参数化或升级到 v0.3。

### 测试

- `tests/test_topic5_shared_scaffold_rnn.py`
- `tests/test_topic5_shared_scaffold_rollout.py`
- `tests/test_topic5_shared_scaffold_field_readout.py`
- `tests/test_topic5_shared_scaffold_runner_v0_2.py`

## 6. 推荐执行顺序

### Phase 0：把半成品 core 修到 green

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic5-structured-rnn-fig6
PY=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python
$PY -m pytest \
  tests/test_topic5_shared_scaffold_rnn.py \
  tests/test_topic5_shared_scaffold_rollout.py \
  tests/test_topic5_shared_scaffold_field_readout.py \
  tests/test_topic5_shared_scaffold_runner_v0_2.py -q
```

在测试全绿以前，不启动 smoke 或 GPU 正式训练。

### Phase 1：只读数据与 target metadata 审计

```bash
$PY scripts/audit_topic5_shared_scaffold_inputs_v0_2.py \
  --config config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml \
  --output-root results/topic5_patient_specific_source_conditioned_rnn_v0_3/input_audit

$PY scripts/audit_topic5_shared_scaffold_ictal_metadata_v0_2.py \
  --readout-config config/topic5_source_conditioned_ictal_readout_v0_3.yaml \
  --training-config config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml \
  --output-root results/topic5_patient_specific_source_conditioned_rnn_v0_3
```

这一步只能读取患者、seizure、contact join 等 metadata；检查 `target_values_read=false`。

### Phase 2：smoke

```bash
$PY scripts/launch_topic5_shared_scaffold_rnn_v0_2.py \
  --config config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml \
  --smoke --workers 4 --resume
```

smoke 必须覆盖 structured、ordinary、至少一个 structured rank-shuffle，并检查：DONE、NaN、OOM、显存、checkpoint 可重载、两端 source 的方向反号。

### Phase 3：三患者、只用间期 validation20 的学习率审计

```bash
$PY scripts/run_topic5_source_conditioned_lr_audit_v0_3.py
```

固定患者：`epilepsiae_1073`、`epilepsiae_1146`、`yuquan_chenziyang`；固定学习率：`3e-4 / 1e-3 / 3e-3`；只比较三人的 median validation contact NLL。不得读取 early-ictal values。选择完成后，只允许把 v0.3 config 的 `training.learning_rate` 改一次并记录选择表。

### Phase 4：冻结代码/config hash 后正式训练

正式单位数：34 patients × 3 models × 3 seeds = 306。31 位 development-excluded 患者是正式确认；34 位是完整队列描述。

在启动前生成 freeze manifest。formal run 期间严禁编辑 core、runner 或 config；任何必要修改都必须停止、换新 output root 并重新开始，不能混合 hash。

```bash
OUT=results/topic5_patient_specific_source_conditioned_rnn_v0_3
tmux new-session -d -s topic5_v03_train \
  "cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic5-structured-rnn-fig6 && \
   exec $PY scripts/launch_topic5_shared_scaffold_rnn_v0_2.py \
   --config config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml \
   --workers 24 --resume > $OUT/formal_tmux.log 2>&1"

tmux new-session -d -s topic5_v03_watch \
  "cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic5-structured-rnn-fig6 && \
   exec $PY scripts/watch_topic5_shared_scaffold_rnn_v0_2.py \
   --config config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml \
   --output-root $OUT --watch --interval 300 > $OUT/watcher_tmux.log 2>&1"
```

24 workers 是上轮实测的高并发上限；先看 smoke 和前 10--20 units 的 GPU/CPU/RAM，再决定是否维持。每个 unit 都必须有独立 log、DONE/FAILED 和 resume-safe checkpoint。若吞吐在约 14 workers 饱和，可以降到 14，不需要为了占满 GPU 增加 OOM 风险。

### Phase 5：间期汇总和 Figure C 数据

```bash
$PY scripts/analyze_topic5_shared_scaffold_interictal_v0_2.py \
  --config config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml \
  --output-root results/topic5_patient_specific_source_conditioned_rnn_v0_3
```

必须报告：

- test20 contact-choice NLL，明确排除 STOP/cardinality；
- structured vs ordinary、structured vs static 的患者级差；
- 31 位 development-excluded 的确认统计；
- 34 位全 cohort 描述；
- top-1 next contact；
- rollout-vs-test20 participation、pairwise precedence 和 expected-rank distance；
- structured true vs structured rank-shuffle。

统计单位是患者；seed 先在患者内汇总，再做 exact Wilcoxon、bootstrap CI、正/负/并列计数。不得以“structured 必须胜过 ordinary”为继续下游的 hard gate；各项独立报告。

### Phase 6：target-free rollout fields

structured seed ensemble 的 scaffold 定义两端 source pools；ordinary 与 structured 必须共用同一 source pools、horizon、rollout seeds 和 denominator。

```bash
$PY scripts/launch_topic5_shared_scaffold_rollouts_v0_2.py \
  --config config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml \
  --models structured ordinary_gru \
  --workers 8 --resume

$PY scripts/freeze_topic5_shared_scaffold_field_manifest_v0_2.py \
  --config config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml \
  --models structured ordinary_gru
```

正式 field 固定为 first-arrival earliness：

\[
F_i^d=\sum_{t=1}^{H}\left(1-\frac{t}{H}\right)P(T_i=t\mid S^d),\quad d\in\{-,+\}.
\]

每侧每 seed 5000 rollouts；使用 exact elementary-symmetric-DP k-subset sampler。不得在 participation、early、late、signed 或 absolute 等候选场中用 ictal target 选赢家。

### Phase 7：manifest 后才解封 early-ictal target

只有所有需要的 checkpoints、source pools、horizon、两方向 fields 与 SHA256 已进入 immutable manifest 后，才运行：

```bash
$PY scripts/score_topic5_shared_scaffold_early_ictal_v0_2.py \
  --readout-config config/topic5_source_conditioned_ictal_readout_v0_3.yaml \
  --training-config config/topic5_source_conditioned_shared_scaffold_rnn_v0_3.yaml \
  --output-root results/topic5_patient_specific_source_conditioned_rnn_v0_3
```

评分合同：clinical onset 后 0--10 s、1--150 Hz、exact contact join、seizure-first/patient-first；每 seizure 取两方向场的最大绝对 Spearman。primary null 是 5000 次 all-contact permutation，每次重新做 absolute 和 two-direction max；within-shaft 5000 次为 sensitivity。15 位 primary 进入统计，E1146 只作 supportive。

## 7. Figure 6 制图合同

正式制图前读 `docs/figure_style_guide.md`，并参考当前 paper-ready figures 的字号、线宽、留白和颜色。创建新 `figures/` 目录时必须同步写中文 `figures/README.md`。

### A｜structured RNN 示意图

左上角。画清：rank-set input、signed coordinate、同一 `a/b` membership 派生 `W_S/W_A`、first-rank source 产生方向状态 `d_e`、propagation/restraint state、contact/STOP/cardinality 输出。用两端 source 的小插图解释“同一 scaffold，方向符号翻转”。标注 `patient-specific`、`no A/B labels`、`target sealed`。

### B｜代表患者的间期复现

固定 `epilepsiae_1146`，不能根据结果换病例。上下两行分别为 learned source-minus 与 source-plus 事件；每行左侧是 test20 observed contact-by-rank heatmap，右侧是 structured rollout first-arrival heatmap，并配 rank 1→末 rank 的简短时序条。contact 只按冻结 learned coordinate 排序。事件分组只由观察到的 first-rank source membership 决定；经验 A/B 只作事后 read-back，不能参与分组、排序或 model mode 选择。

### C｜全 cohort 间期预测

同时显示 patient-level test contact NLL 和 rollout consistency。至少包含 static、ordinary、structured；标注 31 人确认统计和 34 人完整描述。以配对点/雨云图为主，不能只画 bar 或单例。主 endpoint 是 `contact identity | continue,k`，不把 STOP 混进去。

### D｜冻结场与 early-ictal 场

继续用 E1146，在同一 contact layout 上并列 `F^-`、`F^+`、患者内 seizure-median early-ictal broadband field。两张 model fields 都必须显示，不能只展示事后更像 target 的一个方向。统一 contact 布局和色标语义。

### E｜跨状态 cohort 统计

画 15 位 primary 患者的 all-contact-null-corrected correspondence，structured、ordinary、static 用同患者配对连线；同时给 structured vs ordinary 的 paired effect、CI、P 值、正/负/并列和个人超过 null p95 的人数。within-shaft 作为小型 sensitivity inset。E1146 用空心点单列，不进入 primary P value。

需要输出：PDF、SVG、600 dpi PNG、source-data CSV、统计 JSON 和中文 README。完成后必须逐 panel 目视 QA，检查文字遮挡、色标含义、患者数、统计数字和图中数据是否由同一 source-data 自动派生。

## 8. Definition of done

只有以下都完成，才可把本分支提交给用户验收：

- v0.3 core 全接通，新增解析测试和现有回归测试全绿；
- input/metadata audit 通过且 target 仍 sealed；
- 三患者 LR audit 完成，选择只基于 interictal validation；
- 306/306 formal units 完成，0 OOM/NaN，所有 hash 单一一致；
- 31 人确认与 34 人描述统计均生成；
- structured/ordinary 的 target-free fields 和 manifest 冻结；
- 之后才完成 15 人 primary early-ictal scoring；
- Figure 6 A--E、source data、README、复现 manifest 和详细中文报告完成；
- 结果按实际数据报告，不使用 hard gate 删除阴性结果，也不把普通预测性能升级成生物机制；
- 分逻辑批次 commit；只有用户明确要求后才 push。

## 9. 给下一位 agent 的第一条指令

> 不要直接启动 GPU 训练。先检查当前工作树的未跟踪文件和 `src/topic5_shared_scaffold_rnn.py` 的半完成状态，完成 v0.3 source-conditioned core 与解析测试；测试全绿后执行 metadata audit、smoke 和三患者 LR audit。正式训练开始前冻结 core/runner/config SHA，训练期间不要编辑这些文件。early-ictal target values 必须一直保持 sealed，直到 structured/ordinary checkpoints、source pools、horizon、双向 fields 和 immutable manifest 全部冻结。最终 Figure 6 按 A--E 合同完成，不得回到 symmetric-only v0.2，也不得用经验 A/B 或 ictal target 选择模型结构。
