# Epi-PRSSM v0.1 自主实验 Agent Prompt

下面代码块可以直接复制给新的 Codex agent。它是执行入口，不替代 scientific spec、implementation plan 或 figure contract。

```text
你现在负责在以下仓库中自主实现并运行 Epi-PRSSM v0.1：

/home/honglab/leijiaxin/HFOsp

你的任务不是写一份建议，而是：审计现状、补齐实现、运行尽可能完整的探索性实验矩阵、持续监控后台任务、汇总全部阳性和阴性结果、生成图，并交付白话版与技术版两份报告。

除非遇到真正需要新权限、数据挂载缺失或不可逆外部操作的问题，不要频繁向用户确认。对 spec 已覆盖的选择，采用最简单、可逆、可记录的方案继续推进。

## 一、开始前必须完整阅读

按顺序阅读：

1. /home/honglab/leijiaxin/HFOsp/AGENTS.md
2. /home/honglab/leijiaxin/HFOsp/docs/topic0_methodology_audits.md
3. /home/honglab/leijiaxin/HFOsp/docs/topic5_seizure_subtyping.md
4. /home/honglab/leijiaxin/HFOsp/docs/superpowers/specs/2026-08-18-topic5-epi-prssm-v0_1.md
5. /home/honglab/leijiaxin/HFOsp/docs/superpowers/plans/2026-08-18-topic5-epi-prssm-v0_1.md
6. /home/honglab/leijiaxin/HFOsp/docs/superpowers/specs/2026-08-18-topic5-epi-prssm-figure-contract.md
7. /home/honglab/leijiaxin/HFOsp/docs/figure_style_guide.md
8. /home/honglab/leijiaxin/HFOsp/docs/paper_figure_registry.md

以上三份 Epi-PRSSM 文档是本任务的直接合同：

- scientific spec 决定科学对象、模型接口和 claim boundary；
- implementation plan 决定 Goal 0–5、实验矩阵、输出和三个完整性硬门；
- figure contract 决定 Figure A–E 的语义、配色、统计图语法和包装。

若旧文档与这三份新合同冲突，新合同只覆盖 Epi-PRSSM v0.1；不得回改旧冻结结果。

## 二、核心科学问题

必须分别回答，而不是压成一个总 PASS/FAIL：

1. H1：是否存在 correction-off 后仍能预测 future IED repertoire 的慢状态？数据支持 G0、G1、G2、G3 中哪一层？
2. H2a：慢状态是否改变完整事件的 masked rank/order/STOP，以及支持充分患者的 ambiguous-prefix suffix？
3. H2b：只用间期数据学习并冻结的状态，是否在发作前移动，并预测 early-ictal recruitment？
4. H3a：IED exposure 是否更新间期功能状态，而不只是帮助 observer 追踪状态？
5. H3b：只有 H3a 和 H2b 都有支持且方向一致时，才问 exposure-related update 是否与发作转换一致。

H3 很重要，但绝不是 H1/H2 的 gate。即使 H3 完全阴性，只要 H1/H2a/H2b 有独立证据，主体工作仍然完整。

## 三、不要过度防御

本项目是探索性实验，默认多做实验、并行比较、完整记录。

只有以下三类完整性要求是硬门：

1. 数据/泄漏完整性：source/session、channel mapping、mask、tied rank、chronology、forbidden inputs 必须正确；
2. 读取 seizure labels 前冻结 interictal model family/checkpoints/endpoints；
3. 正式主张只能来自一次 untouched test；如果 test 被用于继续调参，后续结果自动标 exploratory。

以下情况都不是停止整个项目的理由：

- G1/G2/G3 未超过 G0；
- open-loop H20 阴性；
- ambiguous-prefix 支持不足；
- H2a、H2b、H3a 或 H3b 任一阴性；
- R2 impulse 阴性；
- R3 integrated exposure 阴性；
- 某个 null 或某个 dataset stratum 阴性；
- resource 参数塌缩；
- 某个模型发生 OOM、NaN 或 numerical failure。

正确做法是：继续其它独立 Goal，追加针对性诊断，并降低对应 evidence card 的 claim。不得因为一个科学结果不漂亮就停止核心实验矩阵。

只有以下情况需要真正停下受影响部分并向用户报告：

- 关键数据挂载或 artifact 缺失且 repo 内无法恢复；
- channel/patient 映射无法确定；
- 需要读取未授权临床标签或外部系统；
- 需要删除、覆盖现有结果或执行不可逆操作；
- 同一 schema/systematic bug 使整批结果失去解释资格。

即使发生这些问题，也只暂停受影响 Goal；其它可独立运行的 Goal 继续。

## 四、启动前现场审计：先查重，再运行

第一步只读检查：

- git worktree list --porcelain
- git status --short
- git branch --show-current
- 当前 commit
- tmux ls
- 与 topic5_epi_prssm、epi_prssm、slow_state、topic5 相关的运行进程
- results/epi_prssm/v0_1/ 下已有 manifest、controller.status、PID、job status 和 logs
- GPU 型号、总/空闲显存、正在运行的 GPU 进程
- CPU cores、load、MemAvailable、swap、磁盘余量
- 当前 Python/conda 环境、PyTorch/CUDA 版本

先判断是否已有 controller 或 workers：

- 已存在且 provenance 匹配：接管/恢复，不重复启动；
- 已存在但 provenance 不明：只读审计 PID、command、working directory、config hash 和 output root，不能直接 kill；
- status 写 RUNNING 但 PID 不存在：标记 STALE，保留旧日志，然后只重启未完成 jobs；
- output 已 COMPLETE：校验 artifact 后跳过，不重复跑；
- output PARTIAL/FAILED/OOM：按 job manifest 精确恢复。

不要杀死、覆盖或重复当前用户/其它 agent 的 worker。不要清理不属于本任务的 worktree 和结果。

把审计写入：

results/epi_prssm/v0_1/manifests/LIVE_EXECUTION_AUDIT.json

## 五、实现一个可恢复的 supervisor，而不是一次性 shell 循环

如果 repo 中还没有 Epi-PRSSM controller，优先实现：

- scripts/topic5_epi_prssm/launch_autonomous.py
- scripts/topic5_epi_prssm/monitor_autonomous.py
- src/topic5_epi_prssm/run_registry.py

controller 必须维护：

results/epi_prssm/v0_1/logs/controller.status
results/epi_prssm/v0_1/logs/controller.log
results/epi_prssm/v0_1/manifests/JOB_MANIFEST.json
results/epi_prssm/v0_1/manifests/RESOURCE_AUDIT.json
results/epi_prssm/v0_1/jobs/<job_id>.status.json
results/epi_prssm/v0_1/logs/<job_id>.log

每个 job 的唯一键至少包括：

- goal；
- dataset/patient；
- model family；
- adapter/resource arm；
- seed；
- split；
- config hash；
- code revision；
- input hash。

状态只允许：

PENDING / RUNNING / COMPLETE / FAILED / OOM / NAN / INVALID_INPUT / SKIPPED_EXISTING

status 写入必须原子化。controller 每 60 秒写 heartbeat、active worker 数、pending/completed/failed 数、MemAvailable、GPU free memory 和 free disk。

恢复时以 job key + config/input/code hash 判断是否可以复用，不能只看文件名存在。

## 六、长任务必须脱离当前网络会话

网络和 SSH 经常波动。所有预计超过 10 分钟的任务必须在 tmux 或 nohup 下运行，不能依赖当前交互 shell。

优先方案：

1. 使用唯一 tmux session，例如 epi_prssm_v01；
2. tmux 内启动 controller；
3. controller 再用有界 worker pool 调度独立 jobs；
4. controller 和 workers 全部使用绝对路径、已解析 Python executable、独立 log 和 manifest。

在 detach 前完成依赖和输入校验。长任务不得在运行中依赖 pip/conda 下载、远端 API 或在线数据拉取；网络断开时本地计算应继续。

nohup fallback：

- stdin 指向 /dev/null；
- stdout/stderr 写独立 log；
- PID 写入 task-specific pid 文件；
- 使用绝对 working directory 和 Python path；
- 不依赖远端网络、当前 shell alias 或临时环境变量。

若用户级 systemd-run 可用，可以使用：

systemd-run --user -> nohup -> controller/worker

但 tmux/nohup 已足够时不要为了工程形式阻塞实验。

每次启动后必须立即验证：

- session/PID 存在；
- controller.status 从 STARTING 进入 RUNNING；
- log 有 heartbeat；
- 一个 smoke job 真正写出 artifact；
- 断开当前 shell 后任务仍存活。

网络断开不是 job failure。agent 恢复后首先读取 controller.status 和 job manifest，不要从头重跑。

## 七、无 OOM 前提下尽量多用 workers

目标是在不发生 OOM、swap thrashing、GPU 抢占和输出冲突的前提下，把独立的 patient × model × seed jobs 尽量并行铺开。

### 7.1 先测真实峰值

每一种 workload class 至少跑一个非空 sentinel：

- CPU data/inventory worker；
- CPU synthetic/analysis worker；
- GPU training worker；
- GPU rollout worker（若峰值与训练明显不同）。

记录：

- peak RSS；
- peak GPU memory；
- wall time；
- CPU threads；
- data read bandwidth；
- temporary disk growth。

不要根据模型参数量猜内存。

### 7.2 自动计算 worker 上限

使用 sentinel 实测值，预留：

- 系统 RAM reserve：至少 20 GiB，或总 RAM 的 20%，取较大者；
- GPU reserve：至少 2 GiB，或单卡显存的 10%，取较大者；
- 磁盘低水位：free disk < 6 GiB 时暂停启动新 job；
- CPU reserve：至少留 2 个逻辑核给系统和 controller。

内存估算对 sentinel 峰值乘 1.25 安全因子。worker 上限取以下最小值：

- pending jobs；
- CPU cores/每 worker threads；
- RAM headroom/单 worker 安全 RSS；
- 各 GPU free VRAM/单 worker安全 VRAM 之和；
- config 中允许的最大 worker 数。

当内存和显存有明显余量、连续运行稳定且没有 OOM 时，主动把 workers 增加到计算出的安全上限，不要长期保守地只跑 1–2 个 worker。

对于独立 CPU jobs，默认：

OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1

避免每个 worker 自己再开满 CPU 造成 oversubscription。

对于 GPU jobs：

- 优先一个 job 明确绑定一张 GPU 或一个显存配额；
- 只有 sentinel 证明一张卡能安全容纳多个小模型时，才在同卡并发；
- 不得占用其它活跃项目已使用的 GPU memory；
- 数据加载 workers 与 model workers 分开计算，不叠加猜测。

### 7.3 动态扩缩容

- 每 300–600 秒检查一次资源和进度；
- MemAvailable/GPU free memory 高于 reserve 且队列有 pending：补满 worker；
- 接近 reserve：停止启动新 worker，让 active jobs 自然完成；
- 不因单次瞬时 load spike 杀 active worker；
- 发现 swap 快速增长、持续 I/O wait 或 GPU OOM：下调相应 workload class 的并发上限。

### 7.4 OOM 处理

单个 job OOM 不等于科学失败，也不等于整个 Goal 停止。

处理顺序：

1. 保存完整 traceback、peak RSS/VRAM、batch/chunk 和 worker concurrency；
2. 将该 job 标为 OOM；
3. 降低同类 worker 并发 1 档；
4. 优先减小 microbatch/chunk，并用 gradient accumulation 保持 optimizer update 和科学 batch 语义；
5. 只重试该 job；
6. 第二次仍 OOM，记录为 unresolved engineering cell，继续其它 jobs；
7. 不得偷偷减短序列、删患者、降模型维数或改变 endpoint 来伪装完成。

不得 kill 不属于本任务的进程。不得通过无限重试制造重复输出。

## 八、实验执行顺序：并行而非超级串行 gate

### Goal 0：数据与 baseline

立即执行：

- immutable data/split/forbidden-input manifest；
- v4.0 reconciliation；
- tied rank、mask、channel order、real delta-t audit；
- train-only patient baseline mu_p；
- static/dynamic variance decomposition；
- patient/source/time-scale/ambiguous-prefix support inventory；
- existing contact-RNN parity。

Goal 0 的 engineering/data checks 通过后，Goal 1、Goal 2 准备和 Goal 4 interictal 可以并行。

### Goal 1 / H1：G0–G3 generator ladder

运行：

- static repertoire；
- event-index EWMA；
- CT-EWMA/G0；
- persistent unconstrained GRU；
- G1 graph-CLDS；
- G2 graph-GRU-ODE；
- G3 graph recurrent + autonomous resource；
- G3 flexible observer-resource correction control。

先跑每个数据源 smoke，然后 6–8 名 support-stratified development patients × 3 seeds breadth pilot；稳定模型扩到全部 eligible development patients；每个结构层代表再做至少 5 seeds formal confirmation。

所有模型至少报告：

- filtered prediction；
- correction-off H5/H10/H20/H40；
- state reset curve；
- delta-t shuffle；
- correction energy；
- time constants/stability margin；
- wall time、RSS、VRAM、NaN/OOM。

如果 G1/G2/G3 不超过 G0，继续 Goal 2–4；只把 H1 evidence card 写成 leaky tracking。

### Goal 2 / H2a：state-conditioned event distribution

inventory、synthetic 和 caching 与 Goal 1 并行准备。比较：

- no state；
- initial-state adapter；
- Node FiLM；
- restricted low-rank graph edge gate。

每个 adapter 至少配 static/G0/G1/G2/G3 代表，避免把 adapter capacity 写成 state effect。

全队列 primary：

- masked rank/order；
- STOP；
- participation-residualized repertoire；
- correct-state vs patient-internal matched-state swap。

ambiguous-prefix 只在 support-rich patients 做 targeted analysis。支持不足记 not eligible，不算阴性，不阻止其它患者和 full-event analysis。

### Goal 3 / H2b：interictal-to-ictal link

先写并冻结：

results/epi_prssm/v0_1/manifests/INTERICTAL_MODEL_FREEZE.json

允许冻结多个预定义结构层代表，不需要只留一个“赢家”。冻结后才释放 seizure labels。

执行：

- last-observation 后 observer-off，自主积分到 clinical onset；
- seizure-aligned state trajectory；
- matched pseudo-onsets；
- leave-seizure-out；
- rate、IEI、source、time-of-day、可用 sleep/vigilance controls；
- time-in-warning；
- onset state -> early-ictal order/field/extent。

H2b 阴性只关闭 seizure-link claim，不停止 H3a interictal 或其它 Goal。

### Goal 4 / H3：R0–R3 exposure ladder

在纯 interictal 数据上可与 Goal 2 并行：

- R0 no resource；
- R1 autonomous resource，先冻结 tau_r；
- R2 single-event depletion；
- R3 integrated exposure，tau_r 冻结后比较 fast/medium/slow；
- 完整 5/15/30/60/120 min 和 5/10/20/40/80 events 作 sensitivity；
- frozen-T1 cross-fitted expected-load innovation；
- state-matched shuffle、time reversal、event-count control、hidden-common-cause synthetic。

H3a primary 至少包含一个与 load 不同义的 outcome：masked order/rank、suffix、direction 或 participation-residualized repertoire。participation/extent 只作 secondary。

H3a 独立报告。H3b 只能读取冻结的 H2b endpoints 做方向一致性，不能用 H2b 反向选择 T2/tau。

### Goal 5：learned event encoder

显式 marks 的 breadth ladder 稳定后，按资源余量依次运行：

- explicit marks；
- frozen event encoder；
- frozen encoder + state model；
- low-learning-rate joint fine-tuning；
- raw waveform encoder 最后。

它不是 gate；若资源足够就运行，用于判断 representation 是否限制前面结果。

## 九、just-in-time synthetic

不要先跑一个数月级巨型 synthetic 网格。按 Goal 即时校准：

- Goal 1 前：no-state、leaky、graph recurrent、observer-overpowering；
- Goal 2 前：state-conditioned ambiguous suffix、no-state false adapter、state swap；
- Goal 3 前：latent preictal drift、event-rate-only confound、last-observation gap；
- Goal 4 前：T1、R2、R3、hidden common cause、event-count-only、switching、observer-resource substitution。

synthetic 失败只限制对应模型解释。保留 truth version 和 holdout truth，不能在同一个 synthetic test 上无限调参。

## 十、监控和自主续跑

controller 每 60 秒 heartbeat；agent 每 300–600 秒检查一次：

- controller 是否活着；
- active/pending/completed/failed/OOM；
- CPU/RAM/swap/GPU/disk；
- 日志是否持续推进；
- 是否出现大量同类 failure；
- 是否存在输出冲突或重复 job。

前 10–20 分钟密切观察 sentinel 和第一批 workers；稳定后可以延长到 10–20 分钟一次，但不要超过 30 分钟完全不检查。

若当前 Codex 任务或网络会话将结束，而后台 worker 仍在运行：

1. 不要杀 worker；
2. 确认 tmux/nohup、heartbeat、manifest 和 logs 完整；
3. 写 CURRENT_HANDOFF.md，列 active/pending/completed、预计剩余时间、恢复命令；
4. 后续恢复时从 controller/status 接管；
5. 不把“已后台启动”写成“实验已完成”。

最终报告必须等计划实验完成、或每个未完成 cell 都有具体不可执行原因后再写。不能因为一次模型阴性提前结束。

## 十一、结果与图形纪律

所有统计 patient-first：

- seed 先在 patient 内聚合；
- patient 是 cohort 推断单位；
- event/window/seizure 是重复测量，不当独立样本；
- Epilepsiae、Yuquan 分层并给 combined；
- 报 denominator、effect size、interval、正向患者数和 paired test；
- 不只给 P 值；
- 所有阴性和失败模型保留在 model matrix 中。

按 figure contract 生成 Figure A–E 候选包：

- epi_prssm_architecture_ladder；
- epi_prssm_generator_evidence；
- epi_prssm_event_distribution；
- epi_prssm_seizure_link；
- epi_prssm_exposure_mechanism。

每个图必须同一次运行生成：

- PNG；
- PDF；
- metadata JSON；
- figures/README.md；
- 需要多 panel 时生成 complete-layout。

必须实际目视 PNG，并核对 PDF 单页、字体、白边、遮挡、颜色、denominator、panel/data 对齐。代码通过或文件存在不等于图验收完成。

不得覆盖 results/paper-ready-figure/fig1–fig4。新 paper slot 必须先更新 paper_figure_registry.md。

## 十二、最终交付：白话版、细节版、机器版

实验结束后必须生成三份互相一致的输出。

### 12.1 白话版报告

路径：

docs/archive/topic5/epi_prssm_v0_1_plain_chinese_report_YYYY-MM-DD.md

面向作者快速审阅，使用中文白话，建议结构：

1. 一句话：这批实验最重要的结果是什么；
2. 我们实际做了什么：患者数、模型数、seeds、主要实验；
3. H1 看到了什么、没看到什么；
4. H2a 看到了什么、没看到什么；
5. H2b 看到了什么、没看到什么；
6. H3a/H3b 看到了什么、没看到什么；
7. 最可信的三条发现；
8. 最重要的三条阴性或限制；
9. 目前论文可以怎么说、不能怎么说；
10. 下一步最值得做的实验。

不要堆公式、内部字段和 P 值墙。每个问题独立总结，不给整个项目一个总 PASS/FAIL，也不让 H3 覆盖 H1/H2。

### 12.2 技术细节报告

路径：

docs/archive/topic5/epi_prssm_v0_1_technical_report_YYYY-MM-DD.md

必须包含：

1. code/input/config hashes、环境、运行时间；
2. 完整 denominator flow；
3. data/split/forbidden-input audit；
4. G0–G3、adapter、R0–R3 的完整实验矩阵；
5. 每个模型、patient、seed 的完成/失败/OOM/NaN 状态；
6. worker 数选择依据、sentinel peak RSS/VRAM 和动态扩缩容记录；
7. H1 filtered/open-loop/reset/shuffle 结果；
8. H2a full-event/state-swap/ambiguous-prefix 结果；
9. H2b pseudo-onset/LOSO/nuisance/early-ictal 结果；
10. H3a predictive/innovation/directionality/timescale 结果；
11. H3b 与冻结 seizure endpoints 的一致性；
12. 所有 null、sensitivity 和 dataset/support strata；
13. numerical stability、correction energy、resource boundary；
14. 图形路径、metadata 和视觉验收；
15. 精确复现命令；
16. 未完成单元及具体原因；
17. claim boundary 和建议论文措辞。

技术报告不是只列成功项。阴性结果、失败模型和资源问题必须与阳性项同等可见。

### 12.3 机器可读总结

路径：

results/epi_prssm/v0_1/FINAL_RUN_SUMMARY.json

至少包含：

- status；
- run IDs/hashes；
- denominator；
- jobs planned/completed/failed/OOM/invalid；
- peak resources/max workers；
- H1/H2a/H2b/H3a/H3b evidence cards；
- figure paths；
- report paths；
- unresolved items；
- safe claims；
- forbidden claims。

同时更新 docs/archive/topic5/INDEX.md，加入两份报告链接。

## 十三、完成标准

以下条件满足后才可以说任务完成：

- Goal 0–5 中所有计划且数据支持的 breadth/development 实验均已运行；
- formal test 若被释放，严格按 frozen contract 完成；若未释放，明确保留 exploratory 状态；
- 每个 job 都有终态或具体不可执行原因；
- 没有活跃但无人管理的 worker；
- 所有汇总重新从 per-job artifacts 计算，不从日志手抄；
- Figure A–E 中有数据支持的图均生成并目检；
- 白话报告、技术报告、FINAL_RUN_SUMMARY.json 三者一致；
- 没有把单事件、单患者、pooled events 或 prediction improvement 写成机制因果；
- 没有把 H3 当 H1/H2 的 gate；
- 没有因为阴性结果删掉或停止其它独立实验。

## 十四、工作区和提交边界

- 保留用户和其它 agent 的未提交修改；
- 新结果写入独立 results/epi_prssm/v0_1/ 路径；
- 不覆盖旧 artifacts；
- 不删除旧结果；
- 不 commit/push，除非用户明确要求；
- 如果未来被要求 commit，必须做窄范围 staged audit，不能带入无关 worktree 改动。

现在开始：先完成只读现场审计和 LIVE_EXECUTION_AUDIT.json，然后实现/恢复 controller，完成 sentinel 资源测量，计算最大安全 workers，启动 Goal 0 与可并行准备项。不要只回复计划；持续推进到实验与最终报告真实完成。
```
