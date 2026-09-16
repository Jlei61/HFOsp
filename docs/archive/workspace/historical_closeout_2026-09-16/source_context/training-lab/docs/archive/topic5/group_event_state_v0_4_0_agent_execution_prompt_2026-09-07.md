# Group-Event State v0.4.0 Agent 执行 prompt

日期：2026-09-07，按审阅修订，版本保持 v0.4.0。以下是后续执行 Agent 的指令；本次文档会话没有启动训练。

---

在 `/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab` 实现并执行 v0.4.0 首个癫痫状态证据包。先完整读取：

1. [v0.4 科学 spec](/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab/docs/archive/topic5/group_event_state_v0_4_epilepsy_state_scientific_spec_2026-09-07.md)。
2. [v0.4.0 首包与预算合同](/home/honglab/leijiaxin/HFOsp/.worktrees/topic5-ges-v033-training-lab/docs/archive/topic5/group_event_state_v0_4_0_first_evidence_package_2026-09-07.md)。

目标是从丰富群体 IED 判断哪些内容有未来信息、持续状态保留了什么，以及同一冻结状态连接哪些间期表达和发作现象。保留五臂、q–f–g、首种子完整链。不要转为完整 DSR、通用合成校准或算法排行榜；不要求两个 INNER、全部视图或全部下游阳性。

保留其他修改、冲突、运行任务和全部 v0.3 产物，不提交、不推送；保持 v0.4/v0.4.0 版本。建立独立新入口/配置和结果目录 `/data/hfosp_group_event_state_epilepsy_state_v040`，复用有效实现，不直接把旧队列改名运行。

## G0：有限接口定义和局部修复

- 复核时间/字段发布/曝光。已有元数据触发 **2 小时主短历史**，30 分钟只作时延诊断，仍保持五臂；边界修复改变支持时按 spec 的同一预定规则在拟合前冻结。区分观察到的安静、部分发布、整段未发布及 count/marks 异步；支持末端与最后 IED 时刻分别记录。输出全支持及 memory_support。
- 检查训练截止：输入、标签、scaler、模板、统计量的 release/右上下文均不跨截止；晚发布标签只能事后评分。修正仅按包结束初筛不足之处，锁定全部种子的 query/target/mask。
- 导出实际 loss 表。保留已有 count/shaft composition/band_ratio/signed_xlag/delay_iqr/load 头及优化时距权重，当前包重建权重为 0；实现窗口等权的形态边缘评分和 **30 分钟组织 J 选模**。count/load 与旧 1min selection 只监测，不把整包 log-mixture 除事件数改名为新评分。
- 唯一出口 Q=(m24,P24×24,metadata)；H1 用后验，H2a/S-B head 只读 m，S-A 用冻结 g 的功能坐标，不导出 c64、不挑 latent 轴。静态 trait、未学习输入驱动表示、学习表示分开；共同 C 的确定性计时/支持字段给所有对应臂同等机会。
- 提前接好 H2a-A/B、S-A/S-B：最近合法分钟状态、真实前缀、精确条件集合归一、S-A 不用未来真实 K 的固定功能量、S-B 冻结间期 head 到临床 0–10s 宽带激活的 community 内 Spearman。首包不新增发作监督 head。
- 定位已有极端 NLL，拆均值/尺度/尾部/单位/曝光/MC 与基线拟合，只修依赖问题，不剪 final score。用小检查验证权重、集合归一、release 截止、重放和随机流。每次更新用当前权重重建旧前缀，grad_hours=2 不截断前向历史，不携带 stale hidden 跨更新，不扩八小时反传。
- 交付 support、training_objectives、state_export、consumer_routes 四张机器表与实际代码入口。独立有效部分立即继续，G0 不是全局资格认证。

## G1：首种子五臂与针对性诊断

五臂为 B_stats、B_marks、S_stats、S_marks、S_marks-short。优先完成 S_marks 的配方及最终冻结，尽早释放 G2；其余臂按资源继续。

主体固定 **25 次拟合**：seed=20260906 每臂两个 INNER 共 10 次；三个 seed 最终重拟合共 15 次。两个 INNER 独立模型/优化器，使用共同评估网格和由平均组织 J 驱动的共同 LR 里程碑，从实际共同评估点选步数，按工作包 Plateau/3200–6400 上限执行；不能拼两个 argmin 或不同 LR 轨迹。后两 seed=20260907/20260908 仅固定配方重复，不重跑 INNER、不看 OUTER 选 checkpoint。报告这是固定配方优化重复性。

合成两个任务各 3 个实现，仅 S_stats/S_marks/S_marks-short 三臂，共 18 次单验证段小型诊断。零关系在实际可读 H_stats/C 下成立；组织阳性让已发布的较早 marks 在相同近期 rich 之外提供信息。按 spec 固定效应、噪声和支持，提供合法旧汇总 oracle。隐藏 U 下独立不等于可读条件零；近期 rich 阳性不等于长历史校准；3 次重复不校准错误率，也不认证人体双 INNER 程序。

D-local 并行，第一份交付逐字段列右上下文、具体 producer、最早可验证 release、处理时延及可用/需重算/缺失结论；能重算的完成同原始时间比较。不能只写“仍需审计”，缺完整低延迟版本不阻断 D-delayed。

## G2：S_marks 冻结立即完成下游

不等待其余四臂结束或 H1 阳性；producer 仅由间期选定，发作不能回选它。

- H2a-A：C、C+H、C+S、C+H+S 四个匹配预算条件集合 head，事件等权，按工作包固定 400 步上限等规则，K_c/有效事件分母明确。
- 正确时刻：在固定 C+H+S head 上按 FIT 供体规则替换状态；recipient C/H、目标和 K_c 不变。报告匹配比例/供体重用/分布外影响；无供体局部不可估，不重训 producer。
- H2a-B：相同真正已见前缀有/无 m，分别报告 next-contact/同步组与 STOP、teacher-forced 与自由续接；A 不能替代 B，无合法前缀只阻断 B。
- S-A：固定空间/形态功能量的有符号病例差和整体描述量，率/方差辅助；标准/近期率匹配、共同病例、相同 query 分开，不挑轴。
- S-B：同一间期 head 与临床 0–10s 宽带空间作 community 内 Spearman 对应，不把能量代入集合似然，不新增监督 head。缺完整人时只阻断 S-C。

初始化被选中不声称学得转移，也不自动等同静态 trait；消费者按实际未学习输入驱动对象解释。

## G3/G4：重复、验收及收口

先完成首种子完整链，再补两个预定 seed 及同样消费者。H2a-A 最多 12 次小 head 拟合；合法 continuation 最多 6 次，既有配置/具体预算先锁定；S-A/S-B/wrong-time 无新 head。修复重跑、复用减少和未运行全部单列，不藏进 25 次主体。

逐项输出 R(B_stats)−R(B_marks)、R(S_stats)−R(S_marks)、R(B_marks)−R(S_marks)、R(S_marks-short)−R(S_marks)、R(C+H)−R(C+H+S)。rich 与旧历史同时有效不能自动推出旧 rich 有效；如需该主张，下包只加一次保留旧粗历史、移除旧额外 marks 的训练匹配实验，本轮不扩矩阵。

H1 窗口主权重、形态事件配套权重和逐族缺失支持均报告；H2a 事件等权。2h 时间块、4h 敏感性、逐日结果说明依赖，发作按簇，query/事件/seed 不当患者。已用 E1125 始终为开发资料。

每 GPU 一个训练进程，保留恢复能力；实测首臂成本后列机器任务与时长预算，不以 GPU 利用率决定科学扩展。预算停止、数值失败、不可估、未运行保留在分母，不无限调参直到阳性。

交付科学报告、逐比较表、四张合同表、实际配置/任务表和必要来源。分别判断丰富信息、持续历史/时间对应、H2a-A、H2a-B、S-A、S-B；说明会改变结论的替代解释。只按具体缺口决定独立确认、旧 marks、局部测量或一次结构解释，不加患者×latent×尺度×结构矩阵，不把工作包完成写成科学闭环完成。
