# 下一位 Agent 的执行 prompt：V4 双核 XY 搜索，完成一个可审阅里程碑

你接手的是 HFOsp 的 Topic 4 双核 SNN 基底搜索，最终服务于 Topic 5 / Fig.5。请先核实现场，再延续已启动的工作。用户安排两个 Agent 交替：你负责执行到下面限定的里程碑，整理证据交回上一位 Agent 审阅；不要未经审阅直接启动下一代搜索或机制实验。中文沟通，不反复确认已授权的读取、检查、修复和运行。

## 1. 科学目标与用户最新要求

用同一组固定双核 VTH 空间位置，在不同网络实现上恢复患者真实间期事件的参与接触点、masked rank、毫秒时差及空间联合分布。四个坐标 X1/Y1/X2/Y2 都需要 data-driven 搜索。不要把位置手工固定在底部，也不要仅凭两核连线符合 E 长轴、出现两种模式、两个动画看着像就宣布成功。

用户已经明确确认 ground-truth GIF 为 **Supplementary Video 1，E10 / Epilepsiae 1146，TA event 6344、TB event 937**。逐接触点身份、参与、顺序、时差是参照；插值 GIF 不是未采样组织内活动的真值。模拟与患者事件没有天然一一配对，不能每次挑最像的 TA/TB 作为总体恢复标准。患者全事件训练分布仍是主目标。

用户特别要求检查 readout 之前的原始 2D 场，避免电极投影掩盖多个波前或其他异常。**不要求每次事件两个 core 都激活，也不要求干净的往返传播。** 目标是患者的整个事件分布。合格基底之前不改 E/EI 来做机制归因，不进入 Fig.5；之后才研究参数稳健性、必要性及间期到 runaway 的变化。runaway 不等于已复现完整临床发作；活动跳变不等于已经证明数学分叉。

## 2. 工作现场：必须用这个 worktree

- 主仓库：`/home/honglab/leijiaxin/HFOsp`
- **实际执行工作树 WT：`/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix`**
- 分支：`codex/topic4-substrate-autapse-fix`
- 当前 HEAD：`336800aabfed51f0d2a8bf008b90fc1d94e0046c`
- WT 和主仓库均有未提交工作。保留全部现状，不 reset、不 clean、不回滚、不移除 worktree、不顺手 commit/push。
- Python：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`
- 环境：`LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib`；`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1`。
- 下文未写绝对路径的项目文件一律相对 WT。主仓库与 WT 的代码版本不同，不能混用 producer。
- 先读适用的 AGENTS.md；遵守来源链、masked rank、参与掩膜、接触点排序和图目录 README 的合同。优先沿下面的具体 artifact 查，避免重新扫描整个项目历史。

结果根：

- V3：`WT/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v3`
- V4：`WT/results/topic4_sef_hfo/joint_rank_space_dual_core_search_v4`
- 报告：`WT/docs/archive/topic4/sef_hfo/`

## 3. 交接瞬间的真实运行状态（2026-09-06 约 11:33 CST；必须刷新）

**V4 主服务已经启动：** `codex-t4-component-xy-20260906.service`，最近核实 MainPID=1529491，active/running，NRestarts=0。命令为 WT 下的 `python -u scripts/run_topic4_joint_xy_component_search.py`。不要重复启动第二个控制器。

当时 V4/status.json 仍是 PREPARED_KERNEL_SEARCH，尚无 baseline_scores.json 和第一轮 design；进程正在初始化/来源核验/旧轨迹处理阶段。**服务存活不等于 worker 已经开始，更不等于有新结果。** 先核实进程、CPU、日志及新文件，不因准备状态就杀掉重跑。

V3 主服务和原始诊断服务均正常结束，MainPID=0，Result=success。V3 完整周期不是 OOM 或异常退出。

**V4 自动图与原始传播 watcher 已写好，但尚未启动：**

- `scripts/render_topic4_xy_component_diagnostics.py`
- `scripts/watch_topic4_xy_component_diagnostics.py`
- 预定服务名：`codex-t4-component-diagnostics-20260906.service`

这是你的第一个待接上的工程动作：确认不存在重复进程后启动它。用同一 WT/Python/线程环境，建议 MemoryHigh=3G、MemoryMax=4G，单个串行诊断进程。它监控完成的轮次并生成整轮候选图及每个候选的原始 GIF、全部事件 CSV/PNG/PDF；它不会代替目视验收，也不会释放 Fig5 hold。若主服务已结束，它仍会先处理已完成分析再退出。

V4 主服务已设 MemoryHigh=210G、MemoryMax=230G，Restart=on-failure、RestartSec=60。dispatcher 最多 24 workers，保留 40 GiB，8 s 仿真预算至少 12 GiB/worker，20 s 确认至少 25 GiB/worker。实际并行数由可用内存决定。最近机器可用约 245 GiB，磁盘约 129 GiB；不要把这个快照当成当前值，也不要保证永不 OOM。若失败反复自动重启，应先读具体失败原因并处理，不能放任循环。

## 4. 按这个顺序读取关键证据

1. `docs/archive/topic4/sef_hfo/joint_xy_component_coverage_revision_2026-09-06.md`
2. `config/topic4_joint_xy_kernel_v4.json`
3. `scripts/run_topic4_joint_xy_component_search.py` 和 `src/topic4_xy_component_search.py`
4. V4 的 `objective_contract.json`、`analysis_input_lock.json`、`prelaunch_verification.json`
5. V3 的 `cycle_closure_review.json`、`component_coverage_review.json`
6. `docs/archive/topic4/sef_hfo/joint_xy_v3_round8_complete_review_2026-09-06.md`
7. `docs/archive/topic4/sef_hfo/joint_xy_v3_round8_raw_video_review_2026-09-06.md` 与 V3/rounds/008/raw_propagation_review.json
8. `docs/archive/topic4/sef_hfo/joint_xy_raw_propagation_acceptance_contract_2026-09-06.md`

第 8 项合同里的“当前最好候选”属于旧时点，最新排名以下面的 V3 完整结果为准。旧 handoff 的冻结旧解、不要 GIF 等指示已被用户后来的要求替代。

## 5. V3 已经查明了什么

V3 正常完成 8 轮，新增 192 个四维几何（90 global、102 local），加旧 155 个，共 347 个。54 个几何完成同样的 8 个训练网络，其中 42 个满足局部参考所需的事件数和几何条件。**全池无合格模型，未进入独立确认。** 672 是本周期新增网络运行数；1396 是来源核验的轨迹 artifact 数（包含 JSON/NPZ 和旧复用来源），不要误写成 1396 次新仿真。

初始联合最优 raw distance=0.08395897；最新 `replicated_r008_003` 为 0.05772181（改善约 31%），75 个事件，8 个训练网络。对应样本量的患者门槛仅 0.00838787，因此仍明显失败。它的 D_support=0.142293、D_order=0.046420、D_lag=13.634907 ms；相应门槛为 0.080070、0.006356、3.077518 ms。相比上一轮，时差有所改善，但参与、顺序等并未一起变好。

V3 第 8 轮 6 个候选共 441 个事件完成数值审计，每个候选最低 seed 的首个时间事件完成 125 帧原始场目视，共 750 帧，并检查了对应 PNG/PDF。新最优首事件仍有 core 外活动和分开的波前。native/readout 成对时差误差的事件中位数通常约 2 ms，但个别事件大得多；小中位数不代表 readout 保留了整个 2D 波前，更不代表患者复现。

V3 失败还不能证明当前模型绝对无解：发现提名和局部搜索有一个明确覆盖缺口。旧提名保留 rank/timing 分量最优，却不保留 participation 分量最优；局部参考只按联合分数。参与较好的位置因此未得到同等细化。

- 完整八网络参与核最优 `xy_whole_sheet_007`：support=0.024494，joint=0.119138。
- 完整八网络 rank 核最优 `replicated_r001_004`：rank_space=0.027580，joint=0.124271。
- 联合/时差核最优 `replicated_r008_003`：support=0.075420，rank=0.059188，timing=0.057583。

这些专长候选也不合格。V4 检验的是改善搜索覆盖能否减少分量取舍，不是证明它必然会成功。

## 6. V4 冻结方案和已验证项目

V4 是 optimizer-only revision，不改患者目标或模型。

- 局部参考最多四个：完整八训练网络中的联合、参与、rank、timing 最优候选；身份去重后不足四个，用空间不同的联合候选补足。低事件数、缺 seed、runaway、不完整或重叠双核不能作为局部参考。
- 每轮 24 个新四维位置。随机全局起点抽取概率 0.5，停滞时 0.85；其余来自局部参考。几何拒绝之后的“成功接受提案”不必一半全局，别混淆概率和计数。
- 两个初筛训练 seed：2511、2512，8 s；每轮从整个未扩展池保留参与、rank、timing 专长，再用联合分数和空间差异补到 6 个；补算共同训练 seed 2513–2518，同样 8 s。
- 旧 347 个几何及轨迹复用，不重新模拟充数。
- **四轮后自动停在 NEEDS_MODEL_CAPACITY_REVIEW**，这是本次交替审阅的自然边界。
- V4 master seed=3028316138；V3=2395812204。第一轮预演为 8 global、16 local，正式 design 需要再核对。
- 初始四个局部参考：r008_003、xy_whole_sheet_007、r001_004、r007_009（完整 ID 看 prelaunch_verification）。
- 6 个相关测试通过；347 候选的旧/新验收结果逐项完全一致；实际四维随机提案重放一致、与旧几何不重复；提前进入 Fig5 的路径做过隔离临时目录测试，仍保持 hold。
- 校准 SHA256：`cba9814d4ce89de5aaa903bedeae66fe6265ce16aa2988e12509fc0c8951b795`。
- 独立确认继续使用预提名、6 个新网络、20 s，至少 64 pooled events、每 seed 至少 5、至少 5/6 单网络 joint 达标。失败尝试记录，不能对下一候选重用同一确认 seed。连续适应性开发不是一次完全未触碰的验证。

**数值确认通过后，V4 只写 raw_review_queue.json 并停在 NUMERICALLY_CONFIRMED_PENDING_RAW_REVIEW。** `qualified_substrate.json` 这个沿用的文件名仅表示该阶段数值确认，不能绕过原始传播验收、自动宣布整个科学问题完成。不要恢复旧 Fig5 自动执行链。

当前 source/input hashes 已冻结。不要直接编辑运行中的控制器、模块、配置、测试、锁定报告或旧轨迹；如有阻断 bug，先让当前 worker 安全结束，另做可追溯修订。不要为了方便而更新哈希掩盖变化。

## 7. 必須知道的模型和观测定义

- 20×20 mm，32000 E + 8000 I，无 autapse。
- **E→E 各向异性为全场设置**：长轴角约 −22.805384°，长短轴比 2；不是仅两个 core 中间才有。E→I、I→E、I→I 当前各向同性。连接与传播延迟使用欧氏距离，并非 torus 连通。
- 双核是在 1499 个节点上施加固定 realization 的 VTH 变化，XY 改位置。不是所有被选节点都降低阈值：当前 signed realization 约 27–33% 提高阈值；与旧手放时的正值截断方案比较时必须说明差异。不能改 realization 后仍声称只改变 XY。
- Z/M 当前关闭；本轮不改 E/EI、OU、时间常数、VTH 深度和 readout。
- 15 个虚拟接触点的 firing-density proxy，不是验证过的 HFO/LFP 生成器。
- 固定 250 ms 事件窗口、2 ms readout 时间步、5 ms 时间平滑、0.25 mm 空间读出；burn-in 500 ms。按所有 15 点的参与选择事件，无 causal family 或 K=2 筛选。
- 训练目标来自 56 个训练块、30049 个可读事件，参与 masked，接触点身份和顺序已核对。训练时差单位换算 20.6328125 ms；不读 heldout 来调参。
- 联合目标为固定 Gaussian RFF，1024 features，包含参与、masked rank、未裁剪相对时差、固定空间矩。方向仅派生验证，不加角度损失。
- 样本量门槛取“下一个不小于 N 的校准 bin”，不是最近 bin。比如 N=61/64 用 64；N=69/75 用 128。跨 bin 的归一化柱状图不能当 raw loss 排名。
- `sheet_activity_counts` 为每 1 mm 格、每 2 ms 内不同活跃 E 神经元的数量，不是 spike 总数，也不含单神经元身份。不能从格点汇总反推出具体神经元的因果作用。

## 8. 患者 GIF 与图像核验

患者正式 GIF：
`/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/supplementary-video-1.gif`

SHA256：`5612da9fcc19ac20643b690117ba8313162285785b982789636a7d3fce09c9c5`

患者 cache：
`/home/honglab/leijiaxin/HFOsp/results/interictal_propagation_masked/event_envelope_fields/epilepsiae_1146_event_envelope_field_cache.npz`

SHA256：`d2fd193cbb30c7cd30a8c173c955ed300d08e657bac8eb43f0d29c7b9a0a9a8b`

原件 30 帧，相对 −8..50 ms、2 ms 生物学步长、80 ms 播放帧时长。TA/TB 的显示质心与训练 lagPatRaw 全窗质心不是同一估计器，不要静默替换。

**GIF 解码使用 cuda_env 的 Pillow 11.1.0。** 系统旧 Pillow 出现过局部调色板/透明背景伪影，不能将解码闪烁误判为模型活动。PNG 可直接看，PDF 用 pdftoppm 渲染后看。自动解码成功不是目视成功；实际调用 view_image 检查，工具输出截断的图片不能记作看过。

目视 QA 单独写 `figures/visual_qa.json`，不要修改 watcher 已锁定的 summary、completion、GIF、CSV、PNG/PDF 或 README。必须区分“全部事件数值检查”和“只看了某个事件动画”。保存完整场、全事件失败模式，不挑看起来最像患者的事件。

## 9. 你的下一里程碑（明确停止条件）

**完成 V4 这一轮四轮覆盖实验的可复核审阅包，并停下来交回审阅。** 如果更早出现独立数值确认通过的候选，则以“该候选确认来源＋原始传播检查＋是否可进入后续”的完整审阅包为提前里程碑；先交回，不自行进入 E/EI 或 Fig5。

执行顺序：

1. 刷新主服务/锁/进程/磁盘/内存，接上 V4 诊断 watcher。已有主控制器存活就继续等待同一个进程，不重复投递。
2. 每轮检查 24 新位置和 48 初筛运行、6 个预提名和 36 补算运行的完整性；核验实际独立 seed、真实事件数、轨迹 SHA、最终 combined 与 assessment。nomination 必须先于额外仿真。旧缓存能复用就复用。
3. 比较完整八网络候选的 joint、参与、rank、lag、方向分量及原始残差；特别回答新增 participation 提名是否得到补算、component anchors 是否真的贡献了新局部提案、是否缓解分量取舍。不能只汇报最优总分。
4. 四轮均数值审计；最终六候选至少按固定首事件完整看原始 GIF，其他轮保留完整自动诊断。对值得判断的候选扩大到不同 seed 和时间事件，先声明覆盖规则，记录未看的范围。若声称模型合格，必须升级为确认网络完整事件覆盖，不能沿用单首事件诊断当最终验收。
5. 若存在异常跨格跳跃等现象，先定位存储/读出/窗口/事件检测问题；不要仅凭看图改参数。必要的同 seed 回放需先验证与旧轨迹一致。
6. 将“事件不足”“患者分布失败”“网络间不稳定”“readout 局限”“优化覆盖不足”“容量嫌疑”分开。V4 无改善只能加强容量/模型设定需审查的证据，不能证明数学上的无解。
7. 达里程碑后，不启动 V5、不修改 loss/门槛、不调 E/EI、不发布或冻结论文 Fig.5。列出下一步最小实验与预期可区分的解释，等待用户携包返回上一位 Agent 审阅。

若脚本提前失败，修复可逆工程问题并继续；不可将失败或预算结束当作科学目标完成。当前尚无合格模型，总目标保持未完成。

## 10. 交回审阅的文件和最终消息

在 WT 的 `docs/archive/topic4/sef_hfo/` 写一个中文里程碑报告，例如 `joint_xy_v4_component_coverage_milestone_review_2026-09-06.md`（按实际日期）。结果目录保存相应 JSON/CSV 和 figures/README.md，报告给绝对路径链接。

报告至少包含：

- 运行快照与完整性：轮次、几何数、实际仿真数、复用数、OOM/失败/重启、耗时、磁盘与峰值内存（未测则明确）。
- 来源清单：代码/配置/校准/输入哈希、master/训练/确认 seeds、轨迹和图表的可追溯链。
- V3 对 V4 的完整八网络结果表，全部患者门槛逐项通过/失败，不仅最优者。
- 新增参与提名和局部参考的实际覆盖、贡献及局部几何散布；不把不同 seed 当不同患者做 cohort 统计。
- 目视记录：到底看过哪些候选、seed、事件、帧，PNG/PDF 是否一致；原始 2D 场和真实逐点时序是否相符，哪些结论仍不可下。
- P0/P1 科学性和工程性问题，当前模型是否支持目标。
- 明确结论：继续优化位置／先修观测或实现／需要模型容量实验；每条建议给最小可执行实验，标明哪些参数会变、什么保持固定、怎样判断结果。
- 是否满足进入 E/EI 与 Fig5 的前提，未满足就明确写“不进入”，不要输出看似完成的 Fig5。

最终只需概括里程碑结论、关键数值、未解决问题和审阅文件路径；让用户能直接把这份包交回上一位 Agent。不要以“服务启动了”或“脚本写好了”代替这个里程碑。
