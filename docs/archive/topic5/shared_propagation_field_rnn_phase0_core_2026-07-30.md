# Topic 5 SPF-RNN：Phase 0 与 v2.0-core 实施报告（2026-07-30）

## 1. 一句话结论

RNNv2 的科学对象已经从 next-rank prediction 改成完整 suffix 的患者内自主生成，首个可运行实现已落地；人类数据接口通过，但 SNN identifiability 正对照尚未冻结，因此当前只能进入工程 pilot，不能解释 latent structure。

## 2. 完成度

**当前实施完成度：65/100。**

已完成：

- 科学合同和主要 claim boundary；
- 人类 Phase 0 全队列审计；
- target-blind 六患者 pilot；
- exact conditional `k`-subset likelihood 与精确 sampler；
- future-blind prior、training-only posterior、自主 latent rollout；
- frozen train-only static scaffold；
- M0 static 与 M1 first-order fixed-schedule baseline；
- prior-predictive、IWAE diagnostic 和 repertoire 输出；
- 单患者真实 artifact smoke run；
- 数值、mask、泄漏和自由生成单元测试。

未完成：

- 标准化 SNN positive-control dataset 与 G0；
- M2 mixture Markov、M3 latent template；
- 六患者 × 3 seeds 训练充分性审计；
- observable source-conditioned stability；
- real-vs-Markov surrogate 稳定性；
- G1–G3 正式判决；
- 独立确认集。

## 3. Phase 0 人类输入

输入：

`results/topic5_interictal_rank_distribution/dataset_v0_4`

manifest SHA-256：

`8bed46286219362d360a6f6f75232cc729bffea2a1b5824e87cc0cad0f27381d`

审计结果：

- 34/34 名患者通过 rank-event contract；
- Epilepsiae 18 人，Yuquan 16 人；
- 总事件数 864,163；
- 每患者 event 数：447–140,337，中位数 8,159.5；
- contact 数：6–52，中位数 11.5；
- 每事件 rank 数的患者中位数：3–33，cohort 中位数 7；
- rank-set size 的患者中位数全部为 1；
- tied-rank-set fraction 最大值约 `1.36e-4`；
- 34/34 名患者存在重复 first-rank condition；
- inner train 无 zero-support 或 full-support contact；
- old heldout20 状态为 `PREVIOUSLY_READ_NOT_CONFIRMATORY_FOR_RNNV2`。

输出：

- `results/topic5_shared_propagation_field/phase0/human_subject_audit.csv`
- `results/topic5_shared_propagation_field/phase0/human_input_audit.json`
- `results/topic5_shared_propagation_field/phase0/pilot_subjects_target_blind.csv`
- `results/topic5_shared_propagation_field/phase0/PHASE0_STATE.json`

## 4. target-blind pilot

选择仅使用 dataset、event 数和 train80 precedence entropy，不读取 A/B、axis、SOZ 或 heldout outcome：

- Epilepsiae：`epilepsiae_922`、`epilepsiae_620`、`epilepsiae_1096`
- Yuquan：`yuquan_zhangkexuan`、`yuquan_chenziyang`、`yuquan_zhangjiaqi`

这些 low/middle/high 标签只表示工程用 precedence-entropy strata，不是传播生物亚型。

## 5. legacy SNN inventory

在 `results/topic4_sef_hfo` 下找到并读取：

- 354/354 个 `*_lagPat_withFreqCent.npz`；
- 合计 4,053 场事件，但跨条件不可池化；
- 单 artifact 最大 123 场；
- 仅 1 个 artifact 至少 100 场。

`chnNames` 为 legacy object array，本审计不反序列化；只安全读取 numeric `lagPatRank/eventsBool`。这些文件缺少统一的 condition/lesion/observation manifest，不能替代 G0 数据集。`N_min` 尚未定义，必须由标准化 SNN learning curve 得到。

输出：

- `results/topic5_shared_propagation_field/phase0/snn_legacy_artifact_audit.csv`
- `results/topic5_shared_propagation_field/phase0/snn_positive_control_readiness.json`

## 6. 真实 artifact smoke run

运行范围：

- subject：`yuquan_chenziyang`
- old train80 内的 inner train/validation；
- target-blind 均匀抽样 512/128 events；
- latent dimension 4；
- 3 epochs；
- device CUDA；
- outer heldout20、ictal、A/B、axis、geometry 均未读取。

结果：

- 训练和 prior-predictive rollout 完成；
- 无 gradient clipping、NaN 或 candidate-mask 失败；
- SPF prior-predictive NLL/event：9.565；
- static NLL/event：9.568；
- Markov NLL/event：7.349；
- Markov precedence correlation：0.743；
- SPF precedence correlation：0.358。

这是三轮、子采样的工程 smoke，不是训练充分性比较。它只证明 runner 和完整 rollout 可工作；目前没有证据表明 SPF 超过 Markov，G1 仍为未判决。

输出：

`results/topic5_shared_propagation_field/development/smoke_yuquan_chenziyang_seed20260730`

## 7. 测试

命令：

`conda run --no-capture-output -n cuda_env pytest -q tests/test_topic5_shared_propagation_field.py`

结果：13 passed。

覆盖：

- rank id 连续和 masked contract；
- tie cardinality；
- exact subset normalizer 对 brute force；
- unordered likelihood；
- exact sampler 的枚举分布；
- train-only static bias；
- loss/gradient finite；
- future-blind prior；
- conditioned free rollout；
- M0/M1 schedule 保持；
- legacy object-array 不被误反序列化。

实现过程中发现并修复一个数值问题：PyTorch 对不可达 DP 状态执行 `logaddexp(-inf,-inf)` 时可能产生 NaN backward。现改为 dtype-scaled finite sentinel，其概率质量数值下溢为 0，同时保持有限梯度。

## 8. P0 / P1

### P0：G0 数据集未就绪

现有 SNN artifact 不能支持 ground-truth identifiability 或 lesion prediction。下一步应先冻结标准化 SNN producer，而不是先跑 34 人正式训练。

### P0：一般 contact intervention 在当前模型中未定义

contact 不回写 autonomous state，所以 raw weight 或删除 contact loading 不能解释为 lesion。v0.1 只报告 source-conditioned observable response；general intervention 延后到显式 closed-loop operator。

### P1：必要基线未齐

M3 latent template 是区分“低维时间模板”与“自主动力学”的关键对照；M2/M3 未实现前不能判 G1。

### P1：无独立确认集

旧 heldout20 已被多轮读取。若无新 cohort 或从未开启的新时间块，RNNv2 只能保持 exploratory。

## 9. 当前 Gate verdict

- Phase 0 human feasibility：PASS
- G0 SNN identifiability：LOCKED / NOT RUN
- G1 full-event generation：NOT RUN；单次 smoke 不判决
- G2 stable observable structure：LOCKED / NOT RUN
- G3 one structure, many trajectories：LOCKED / NOT RUN

当前允许写：

> 已建立一个不 teacher-force suffix 的患者内自主生成框架，并验证冻结 rank-event 数据可支持工程开发。

当前禁止写：

> 已恢复稳定有效连接、一个传播场已解释患者多样性、或 RNN 已预测 SNN lesion。
