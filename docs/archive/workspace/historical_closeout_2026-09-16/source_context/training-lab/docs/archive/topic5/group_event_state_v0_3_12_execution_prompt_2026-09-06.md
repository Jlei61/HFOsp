# v0.3.12 下一窗口自主执行 prompt

状态：**该交接已执行并由 [窗口收口](group_event_state_v0_3_12_window_closeout_2026-09-07.md) 替代为最终记录。** 以下正文保留当时冻结的执行合同；其中每卡两个训练进程已由运行事故证明不安全，下一次计划固定为每卡一个训练进程。

---

在 `/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab` 执行已接受的 v0.3.12 合同，结果根 `/data/hfosp_group_event_state_observable_state_validation_v0312`。本次授权是一个 8–10 小时的科学运行窗口：持续监控并运行已登记的合法任务，8 小时开始收口，最迟 10 小时交付可核验结果和剩余任务。不要提交或推送，不覆盖 v0.3.11，不修改无关脏文件。读取当前工作区 AGENTS.md。

先读：

1. `docs/archive/topic5/group_event_state_v0_3_12_observable_state_validation_spec_2026-09-06.md`，第 10 节是实施冻结条款，在冲突处优先。
2. `docs/archive/topic5/group_event_state_v0_3_12_implementation_readiness_2026-09-06.md`。
3. 结果根下 `handoff.json`、`readiness/admission.json`、`next_window_plan.json`。

科学主线是同一患者、同一冻结 checkpoint 的证据链：实际可读的丰富 IED 历史 → 预测下一分钟及 5/30/120 分钟后那一分钟的数量、空间和条件形态 → 冻结细触点身份 → 可估发作关联与临床起点空间对应。优先回答输入和状态是否有功能增量。不要通过扩大模型、增加时间常数或重做非线性结构矩阵来绕开这条链。

此前 26 个短任务和三个四步合成世界只是工程验证；不得把它们混入正式人体结果。科学队列中的 67 项是跨窗口 backlog，不承诺全部在一晚结束；34 项是神经拟合，其余是数据、配方、冻结分析和基线。先首种子完整链，后 S-ID、仪器和已登记重复，按机器优先级执行。禁止按新的 outer 阳性程度换患者、选步数或改变目标。

先执行检查，检查通过再启动；这两步都是本 prompt 的已授权工作，无需再次询问：

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python /data/hfosp_group_event_state_observable_state_validation_v0312/launch_next_window.py --check-only --hours 8
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python /data/hfosp_group_event_state_observable_state_validation_v0312/launch_next_window.py --hours 8
```

启动器检查代码、数据依赖、计划哈希和 worker 锁；本次历史计划使用四个 GPU 训练 worker（每卡两个）加两个 CPU 消费者。运行后修订禁止同一卡并发两个训练进程；新计划只能每张 GPU 一个训练进程。不得只启动 GPU worker，否则配方／冻结依赖不会推进。已经在跑的窗口不要重复启动；读取 `window_logs/*/launch.json` 与任务锁找到当前进程，必要时继续原有工作。续跑同一合法 PAUSED 任务使用相同计划和启动器；配置、样本、源码改变时不能直接复用旧卡。

运行组织：

- 每 5–10 分钟核对 `queue_state`、`runs/*/progress.json`、worker 日志及 nvidia-smi。用户不需要固定频率的空状态播报；只有阶段完成、会影响结论的新事实、错误或需要真实外部信息时更新。
- GPU 有合格任务时持续供给。派发要求至少 8 GiB 空闲；训练剩余显存低于 6 GiB 保存暂停。不得通过缩短历史、改有效批量、减少路径数、删除失败患者来维持利用率。OOM 允许相同批量按更小 microbatch 重算，保留事故记录。
- 冻结消费者只用 CPU。首次冻结消费者和第一次正式训练各发生一次异步 CUDA indexing assertion；同步重放不能复现，潜动力学中唯一动态 `index_put` 已由数学／梯度等价的固定基底收缩替代，旧部分运行在 `superseded_cuda_indexing_assert_20260907T0006/`。不得说已证明唯一根因或硬件问题。训练进程若再出现 CUDA assert，停止该进程并保留日志；由新进程复现定位，不能在已污染的 CUDA context 内继续。
- 不在运行中的源码目录修科学算法。遇到会改变结果的错误，先暂停受影响任务，保存原因；使用新版本/并行目录重跑受影响依赖，再生成新计划及准入。独立且来源不受影响的任务可继续。不能仅改 hash 强行准入。
- 主模型持续过滤；反传至少覆盖批内最早 query 之前 2h，严格 H 的全部指定可用历史才反传。0.5h 逐 query 判可读性，不能全局禁用；临床 reset 前的历史永远不可跨。严格 H、跨视图和更多患者本窗口没有隐含全面扩展授权，应先完成登记主链。
- 两个 temporal INNER 冻结步数和 RELAX 配方，再 outer 重拟合；outer 不作训练选模。第 0 步可被选择，此时记未建立 learned-state，不能靠“跑了很多步”资格放行。
- 固定 1e-3 共同随机数平台规则并非收敛证明；保存两次降率、选中步、末段趋势与四组 MC 数值误差。预算停止记预算停止。近零结果若未分辨或仪器功效未校准，只能记未建立，不是科学阴性。
- S-A 的原始率含时钟，优先同时检查相同时钟下当前状态对 FIT RESET 的功能增量、可读近期率再次匹配后的结果及伪发作。S-B 是临床起点 0–10s 的描述性空间对应，不是 EEG 起点场／传播路径。S-C 风险不准入，IED 生理反馈不由本轮检验。

同一 split/seed/h/view 在同一物理目标上配对；参考封底是整个端点的保守 benchmark，不是每个测试目标上挑最强预测器。P_marks 对 P_stats 仍包含事件编码器带来的额外容量；只有重复稳定并胜过丰富历史参考才推进解释。报告数量、条件形态、粗空间与细身份各自结果，不将一个小数代替整个共同状态。

检查点冻结后立即运行已登记消费者，不等所有种子完成。CPU 与 GPU 同步推进，GPU 运算期间整理数据支持、训练曲线和逐目标结果；调用已有报告器：

```bash
/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python scripts/report_group_event_state_v0312.py --plan /data/hfosp_group_event_state_observable_state_validation_v0312/next_window_plan.json --out /data/hfosp_group_event_state_observable_state_validation_v0312/window_summary.json
```

该机器汇总只是配对分数与参考封底入口。还需从每份冻结 summary、adapter、calibration 和 source/query metadata 整理同 checkpoint 完整证据链，不能把缺失端点默认为零。按计划目录与 task id 纳入，排除 readiness 下的所有拟合。

8 小时时让训练按既定截止保存 PAUSED，CPU 可完成已开始的消费者并整理。若延长到 10 小时有具体价值，可以在余下时间恢复原队列，但不要到最后一分钟才启动需要长时间才能交付的消费者。没有足够时间完成的训练保留 optimizer/sampler/scheduler，给精确续跑命令；记录失败、不可估和待运行，而不是改成 COMPLETE。

交付中文报告、机器 summary、同 checkpoint 证据链、逐层参数／更新／优化表、逐物理目标配对、可读年龄/曝光/昼夜/episode 支持、临床起点空间映射和 S-A 匹配／伪发作台账。必要的科学图使用标准绘图工具，PNG/PDF/metadata/中文 README 成套，Agent 自查后明确尚待用户目视验收。不要为凑图数制造无法解释的 latent 散点。

最终明确回答：丰富输入是否增量、当前状态是否胜过丰富历史和常态、演化是否有额外价值、细身份与发作联系是否复用同一状态。任何层可以局部支持或未建立；不能提前承诺阳性，也不能在功效尚未校准时宣布原问题科学阴性。若合格参考、测量链或数值误差阻止判读，优先交付这一原因及最小下一项有区分力的实验，停止无依据的容量扩展。
