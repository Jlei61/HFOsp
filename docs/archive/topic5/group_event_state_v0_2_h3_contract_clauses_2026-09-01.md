# Group-Event State v0.2-C：H3 契约条款枚举（实现前的 §6 仪式）

状态：**implementation contract checklist**；每条在代码中以 `# CLAUSE <id>` 注释锚定，并在
`tests/test_topic5_group_event_state_h3.py` 中对应至少一个断言（标记 `no-test` 的除外，需给理由）。

来源（逐条重读，不凭记忆）：
`group_event_state_v0_2_common_contract_2026-09-01.md`（下称 CC）、
`group_event_state_v0_2_engineering_invariants_2026-09-01.md`（EI）、
`group_event_state_v0_2_h3_spec_plan_2026-09-01.md`（H3）、
`group_event_state_v0_1_data_contract_2026-08-31.md`（DC）。

---

## A. 边界条款（exposure / target / state 不得静默跨界）

| id | 条款 | 出处 | 实现锚点 |
|---|---|---|---|
| A1 | coverage segment：相邻 block 时钟间隙 > 2.0 s 断开；"有记录但无 group-event 产物"的 block 断开 | DC §12、`contract.SEAM_TOLERANCE_SECONDS` | `support.build_coverage_segments()`；`block_time_ranges` 等价物为**必填**参数，无 `None` 默认 |
| A2 | exposure 与 target window 不跨 seizure onset | CC §7.4 | `support.cut_intervals_at_seizures()` |
| A3 | 发作后不静默桥接：从 offset + 60 min 起新 segment（primary；其他长度只作敏感性） | CC §7.5 | 同上，`postictal_exclusion_s=3600.0` 为 primary，0 s 为 sensitivity |
| A4 | target/exposure block 不跨 TRAIN / inner-validation / development-test 边界 | CC §7.1 | `support.split_by_physical_time()` 产生的边界进入 `usable_intervals` 切割 |
| A5 | 状态不跨 gap / segment 传播；segment 起点重置为可学习初值 | CC §7.3、EI §1.2 | `models.StateCore.rollout()` 每个 segment 独立初始化 |
| A6 | future block 不读取中间真实事件（no teacher forcing inside the block） | H3 §6、C prompt §3.8 | `models.decode_future_block()` 只吃 anchor 状态 |
| A7 | 统计分母 = 不重叠 physical blocks；滑窗数不得写成独立窗口数 | H3 §6、EI §2 | `support.tile_blocks()` 只产生 disjoint tile；`n_independent_blocks` 与 `n_sliding_windows` 分开落盘 |

## B. 配对 / 分母条款

| id | 条款 | 实现锚点 |
|---|---|---|
| B1 | 跨患者 M0/M1/M2 比较必须按 subject key 对齐；key 集合不等即 raise，不按数组位置对齐 | `analysis.paired_by_subject()` |
| B2 | 逐 block 配对必须按 `(subject, split, horizon, block_index)` 键对齐 | `analysis.align_block_scores()` |
| B3 | seed 是重复拟合不是样本量：先在患者内对 seed 取中位数，再做患者层统计 | `analysis.collapse_seeds()` |

## C. Stub / 缺失条款（缺失即 not_available，禁止 fallback）

| id | 条款 | 实现锚点 |
|---|---|---|
| C1 | registry producer 缺失 → `producer_status="not_available"`，**不得**静默回落到 C 自己的模型；C 内部诊断必须写在**不同的** key 下并显式标注 | `registry.load_producer()` raise / `innovation.py` 的 `source` 字段 |
| C2 | H2b frozen risk readout 为 secondary，缺失不阻断 H3 | `innovation.attach_h2b_readout()` 返回 `not_available` |

## D. 需要**向用户显式上报**的 signature-vs-prose 冲突（不静默修补）

- **D1**：C prompt §5 要求"承重人体结果必须绑定 registry 的真实 producer"，但承重图（§7）是 M0/M1/M2，
  而 M0/M1/M2 按 H3 §3 是 C 自己训练的模型族。交接时 shared registry **尚不存在**（A/B 刚起步）。
  → 处理：M0/M1/M2 走 C 自有 producer 并写入 **C-local additive registry**（不动 A 的整份 registry）；
  C1 functional innovation 的 registry-bound 版本在 A 的 producer 落地前记 `not_available`。
- **D2**：CC §7.1 要求按 recorded physical time 切分，v0.1 dataset 出厂的是**事件计数**切分，且 core 归 A 只读。
  → 处理：C 侧写独立 split adapter（纯函数、可复算、落 hash）；A 的 registry 出现后必须比对，
  不一致就报 mismatch，不静默采用任一方。
- **D3**：H3 §3 把 `M2` 写成 `G(·)+A_mark(...)`，但 §9 验收要求"M2 在**相同 count/time** 下超过 M1"。
  要让增量可归因于 mark 内容，M2 必须**嵌套**包含 M1 的 count 通路。
  → 处理：实现 `M2 = G + A_count + A_mark`，并在报告中明写这是嵌套解释。

## E. "Reported alongside" = 首轮必做（不得推迟）

| id | 条款 |
|---|---|
| E1 | future-block score 必须拆 `count/rate` 与 `conditional mark`，两者同时首轮实现 |
| E2 | `participation/extent`、`multiband expression` 端点首轮实现 |
| E3 | event-type-specific **signed** impulse response 是承重图的一半，首轮实现 |
| E4 | burden estimand 与 content estimand 分开报告 signed effect，首轮实现 |
| E5 | support inventory：每患者 TRAIN / inner-validation / development-test 的真实小时、block 数、有效独立分母 |
| E6 | `uses_background` 在 manifest 显式声明；禁止从 `a4/a5` 之类臂名推断语义 |

## F. 扰动 / surrogate 构造条款（每个子句一条测试）

| id | 条款 |
|---|---|
| F1 | `no_event_feedback`：同一冻结 checkpoint，**只在 exposure window 内**把 adapter 置零；同一 pre-state、同一 exposure window、同一 future target |
| F2 | `state_matched_mark_replacement`：**事件数与时刻逐位保留**，只替换 mark；donor 按 pre-state 匹配，donor 选择不得读 future target |
| F3 | burden estimand **不得**匹配或回归掉 exposure window 的 count/rate |
| F4 | content estimand 必须逐位保留 count/times（断言相等） |
| F5 | 所有扰动从同一 pre-state 出发；扰动后关闭真实未来 teacher forcing |
| F6 | constant / intercept / drift 零真值只作回归测试，不扩成人体主臂 |
| F7 | rate-preserving mark shuffle、burst thinning 为 secondary |
| F8 | 首轮主比较**只有** 3 条 arm（real / no-feedback / state-matched replacement）；不新增自造 arm |

## G. 工程不变量（EI）

| id | 条款 |
|---|---|
| G1 | `exp(clamp(log_tau))`，禁 `softplus(log tau)`；且不得把"可表达范围"写成"已识别尺度" |
| G2 | split pass + carry 必须与不间断 causal pass 逐位一致（单测） |
| G3 | 无状态臂的 head 不得偷看 carry（输入扰动验证） |
| G4 | source / config / checkpoint hash 锁定；重复 payload 直接报错 |
| G5 | 绝对时刻用 float64；禁止远历元 float32 |
| G6 | 线性探针按 Gram 尺度正规化；远坏于 intercept baseline 的拟合标为 `not_estimable` |
| G7 | seed 必须在 init / 训练顺序 / payload hash 上真的不同 |
| G8 | synthetic PASS 只证明实现符合合同，不证明 H3 |
| G9 | 原子写 + manifest 后置 + 按 payload hash 幂等跳过 |
| G10 | 单一 queue owner；禁 `pkill -f`；写 `shared/resource_leases/agent_c.json` |
| G11 | 所有 worker `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1` |

## H. §6.1 helper 复用的"问题匹配"检查

- **H1**：v0.1 `SubjectSequence.split_slice()` 回答的是"按**事件计数** 70/10/20 切"。
  H3 需要"按**记录物理时间**切"。→ **不复用** `split_slice()`；只复用其 memmap gather。
- **H2**：v0.1 `scripts/.../run_h3_exposure.py` 回答的是"在冻结状态上，滚动**事件计数**曝光能否线性地
  改进对**下一事件 size** 的预测"。H3 v0.2 问的是"事件→状态的显式边能否改进**未见的固定物理时间未来块**"。
  两者控制的东西与比较的对象都不同 → **新模块，不是扩展**。
- **H3**：v0.1 `run_sequence` 的 state 在**事件时刻**推进且 background 以**逐事件**加性修正进入。
  H3 的 `M0` 要求"事件不进入状态转移"，若 background 以逐事件脉冲进入，事件计数就会经由
  "施加了几次 background 脉冲"泄漏进 M0 的状态 → **不复用** `ContinuousState`；
  改写为"松弛目标（relaxation target）"形式，使 M0 的自由动力学对插入幻影事件**逐位不变**（单测 G/A 交叉项）。
