# RNN v0.3.8：状态到触点序列的冻结接口测试

本轮按用户要求补测试，检验状态网络与 contact-sequence 网络之间的预测联系。代码工作区独立于仍在运行的 v0.3.9，原 v0.3.8 与测量重建结果保留。

## 冻结范围

- 执行清单：`/data/hfosp_rnn_v038_contact_bridge_20260905/execution_manifest.json`，先于人体评分写入。
- 35 个既有 checkpoint/adapter 单元：E1096 event/grid/dual、E253 event/dual、E1125 dual、E958 dual，各 5 seeds。按用户点名病例与接口来源选取，不加入新架构、学习率或病例搜索。
- 采用第二轮修复汇总明确绑定的最终 overlay card；每个上游、adapter、decoder、冻结重放输入与实际 rank 输入均登记 hash。源码从已存档的 `h2a_budget_complete_v1` 复制为本任务快照。
- 重评已检查过的 SELECTION，保存全量逐事件 logits、STOP、配对 donor、原始 ranks、评分与物理窗明细。始终冻结全部权重，不打开 development/sealed。

## 主要问题与端点

观察同样前两组触点、给定下一组大小 K 后，真实状态是否提高**下一组触点身份**的条件概率。分叉字典只由 FIT 中至少 5 次出现、至少 2 个不同下一触点集合的 prefix+K 定义。所有臂使用同一批具有相同记录段、相同 prefix、相同 K 且相隔至少 2 h 的错时 donor 可配对事件。

下一组大小条件下的集合概率定义为

`p(S | |S|=K, available) = exp(sum(logit[S])) / sum_{|A|=K} exp(sum(logit[A]))`。

这是冻结 Bernoulli 触点读出在 K 上条件化后的分布。K=0 和 K=available_count 没有身份选择空间，记为身份端点不可评分；不以零损失加入均值。旧 contact_nll 的平均 log-softmax 保留用于父分数重放；K=1 时两种概率评分等价。

次要端点分别记录：全部有身份选择空间的逐步预测、严格后缀身份、前两组之后再预测两组的 teacher-forced 评分、整组命中率、前缀末端 STOP。多步 teacher forcing 不称为自由续写，合并 grammar 不代替身份或 STOP。STOP 主解释用全部前缀事件；只保留继续事件的身份条件子集不能评估 STOP 判别能力。

## 对照与归因

共同预测对照是 static、B_mark、FIT 常数和错时状态；各臂同一事件集合，不以受损 B_mark 取代更好的 static。另把原合并形态评分与 prefix-only 比较，检查旧阳性是否只是恢复更复杂基线引入的损害。

归因探针包括事件分支均值、背景分支均值、初始化事件矩阵、相同时间轴上按记录段及分区打乱 mark 后重放、单独错移事件或背景分支。初始化保持原 seed 构造顺序；权重变动与原 lineage 按 checkpoint hash 对齐。

这些探针**使用固定的已训练 adapter**，只能说明当前网络依赖什么输入；删除后变差、保留初始化变差或打乱历史变差，均不能独自证明已学习历史优于重新训练的替代模型，更不能证明事件导致生理反馈。打乱及错时是非因果诊断臂，不作为可部署预测器。

## 统计及边界

逐 seed 保留事件等权和绝对时间轴上 2 h 物理窗等权的配对差。正方向容差固定为 `1e-6`；四项比较必须在同一个 seed 同时为正。3/5 只描述优化重复，不是显著性检验；不把非重叠窗、共享历史的窗或 seed 当作独立患者，也不靠 bootstrap 制造确定性。

旧测量字典仍使用了全记录触点统计，本轮全部结果为回顾性接口诊断。新测量前瞻资格和重新拟合的分支对照依赖另一条 v0.3.9 工作，不能通过本轮旧权重消融补成。父分数无法在原容差内重放的模型保留来源限制，不进入阳性计数。

## 复现

使用 CUDA 环境 Python、对应 `LD_LIBRARY_PATH`，但本次只使用 2 个单线程 CPU worker。入口 `scripts/run_rnn_v038_contact_bridge_batch.py --root /data/hfosp_rnn_v038_contact_bridge_20260905`；评分器和依赖模块运行前复制为固定源码，已有输出只在 hash 与代码一致时复用。运行日志、清单、逐单元卡片与汇总全部位于独立输出根。
