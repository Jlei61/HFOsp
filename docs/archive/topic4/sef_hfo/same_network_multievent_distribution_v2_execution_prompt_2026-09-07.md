# 执行 prompt：多事件分布学习 v2

> **已由 [v2.1](same_network_multievent_distribution_v2_1_execution_prompt_2026-09-07.md) 替代（2026-09-07）**：下文保留历史记录。D16不再用于主排序；确认改为2拓扑×2动力学。执行请使用v2.1，不按本旧版启动。

请实施本任务已完成的 v2 设计，不再返回泛泛建议。工作点为 `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-substrate-autapse-fix`；保留其他未提交文件、旧结果和正在运行的任务，不自动 commit/push。

先读 `docs/archive/topic4/sef_hfo/same_network_multievent_distribution_design_v2_2026-09-07.md` 和 `config/topic4_multievent_distribution_search_v2.json`。用户明确要求：TB 从右下起始等具体路径必须由数据驱动学习后接受验证，不进入 loss、奖励、core 起始区域约束或事件专属刺激。整个目标是一个固定网络在多次自主事件中产生患者的多模式分布。

已有实物：

- `src/topic4_multievent_distribution_objective.py`：通用整体＋患者模式质量敏感核目标，等网络权重、16事件匹配样本量的精确期望。
- `scripts/prepare_topic4_multievent_distribution_search.py`：可重复准备入口，已实际运行，8项检查通过。
- `results/topic4_sef_hfo/multievent_distribution_search_v2/`：`training_objective.pkl`、`initial_candidate_manifest.json`（48条件，2个人口各24）、`loss_qualification.json`、`preparation.json`。
- 旧观察与模型接口：`src/topic4_observation_repaired.py`、`scripts/run_topic4_multidimensional_worker.py`；复用修复目录的固定 `observation_contract.json`。不使用worker内其他lineage onsets替代冻结质心事件表。

当前没有派发新 SNN，完整 v2 controller 尚未实现。不要把准备包当成已经开跑。完成以下工作并持续到本轮验证证据可审阅：

1. 新建 v2 专用 controller，接入已有 worker、冻结读出和训练目标。优化器只读训练 payload，不加载 route CSV、原生 GIF、PROBE 评价或支持/覆盖表作排名。增加与任务直接相关的测试：跨网络不能凑模式，缺事件不能平均掉，候选恢复不重复派发，训练提名不受验证文件内容影响，参数真实生效。
2. 先进行设计要求的24秒canary；记得旧18GiB预算只在12秒运行验证过。检测实际内存、参数剂量与归一化、原生数组及冻结观测行为。支持18GiB地址空间限制、40GiB系统余量、其他任务未来内存、磁盘预计用量，最多8worker。物理runaway与程序故障分别处理。canary成功的正式任务直接复用。
3. 完成48条件×2训练seed，生成每人口8个DE后代，共最多64条件/128次24秒训练任务。读现成候选清单中的随机master seed并持久化DE随机状态；不要重新随机生成这48个条件。DE只接收两网络各自的通用loss及可估计/失败状态。优化参数与本轮预算固定。
4. 每人口提名前两名，按设计去重与递补，写定 nomination 文件后才打开验证。提名条件、训练分数、参数、输入哈希及提名时间要可追溯。自动接最多4候选＋3历史基线、各4个新确认seed的24秒验证。先核对6101–6104未用于本任务；有冲突则在运行前分配并记录替代种子。
5. 并行于不依赖它的运行，恢复患者64事件验证小包的真实event ID和raw envelope。只能从原producer追溯行身份，不靠最近rank或相近相对时间猜测。复用主工作区 `scripts/plot_topic5_interictal_event_envelope_field.py` 的 `load_events/build_event` 等低层读取逻辑；不要调用默认方向筛选/挑示例入口，不覆盖正式视频缓存或Paper Ready Figure。按照设计预先随机抽样，不根据方向、波形或模型匹配程度挑患者事件。
6. 提名后输出每固定网络的模式分布和残差、原始场多事件GIF与患者参考。GIF必须按时间顺序覆盖预定seed的全部检测窗口，标记拟合排除和跳过的间隔，并附完整24秒轨迹概览、所有seed逐事件表。最早质心和包络10%时间都不是神经元起燃位置；具体路径检验不回灌本轮优化。
7. 自动做所有帧解码、时间/事件覆盖检查和Agent目视抽帧，生成中文审阅报告与每图README。解释训练进步是否伴随模式条件路径改善、是否只在旧seed成立，以及是否仍为多处活动/中部扩展/缺失SCL。人口级波形暂不可恢复时，只暂停该证据层，其他部分继续，不能以两个示例冒充完成群体动态验证。

最多156次正式24秒仿真（canary复用其中任务），不自动增加代数或改目标。结束在 `ROUND_COMPLETE_PENDING_SCIENTIFIC_REVIEW`，不自动冻结模型、重绘正式图或进入 Fig. 5。真有路线验证失败要如实报告，不把期望路线写进目标来获得正结果。所有患者块均为重复使用的开发数据，未放进loss不等于独立患者盲测。

环境：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`，`LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib`，BLAS/OMP均1线程。可重复准备命令（不启动SNN）：

```bash
LD_LIBRARY_PATH=/home/honglab/leijiaxin/anaconda3/envs/cuda_env/lib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/prepare_topic4_multievent_distribution_search.py
```

交付时明确实际完成数、物理失败数、工程失败数、验证缺口与未完成事项；不把“controller写好”或“两个标签都出现”当作科学任务完成。
