# Group-Event State v0.2-C：Event feedback（H3）

开始前完整阅读共同科学合同和工程附录。本线不再把普通 RNN“看到事件后更新 hidden state”当作 IED 塑形，因为那也可能只是 observer 更新了对共同慢状态的估计。

## 1. 唯一核心问题

在控制 pre-event state、真实 `dt`、背景与近期统计后，未来间期事件块是否仍需要一条从 IED 数量或内容进入状态转移的显式反馈边？

最高允许结论为：**event-feedback-like predictive dependence**。人体观察数据不能仅凭模型比较写成 IED 因果改变脑网络。

## 2. 第一层：functional innovation trajectory

对每个观察事件记录：

\[
\Delta S^{func}_e=S^{func}(z_e^+)-S^{func}(\widetilde z_e^-),
\]

其中 `z_e+` 是 observer 读取当前事件后的状态，`z_tilde_e-` 只做时间/背景传播、不读取当前事件。`S_func` 使用冻结 future-block readout，不使用 raw latent 欧氏距离。

先描述：

- 哪类事件产生正/负 innovation；
- innovation 与未来 5/30/120 min 的实际 count、conditional mark、extent/multiband 变化是否相关；
- 累积 innovation 是否超过 `B_multiscale` 中的 rate/count。

这一步是轨迹测量和候选机制定位，不单独证明 feedback。

## 3. 三个必须显式比较的模型

### `M0_no_feedback`：common-drive/readout-only

\[
S_{e+1}=G(S_e,\Delta t_e,B_e),\qquad X_e\sim p(X_e\mid S_e).
\]

IED 是状态的读数，不能进入未来状态转移。

### `M1_count_rate_feedback`

\[
S_{e+1}=G(\cdot)+A_{count/rate}(X_e).
\]

只允许 event occurrence/count/rate/burden 进入低容量 signed feedback。

### `M2_mark_specific_feedback`

\[
S_{e+1}=G(\cdot)+A_{mark}(participation,extent,waveform,multiband).
\]

在 event times/count 相同条件下，检验空间—频带内容是否提供额外反馈。

三者共享 observer、base dynamics、decoder 和训练预算；新增 adapter 做容量配平。checkpoint 只按间期 inner-validation future-block objective 选择。

## 4. 两个 estimand 不得混在一起

### 4.1 Event burden effect

问题：在相同 pre-state/background 下，更多事件是否改善对之后状态/未来块的预测？这里不能匹配或回归掉 exposure window 的 event count/rate。

### 4.2 Event content effect

问题：在事件数和时刻相同时，mark 内容是否改善未来预测？这里使用 count/time-preserving mark replacement 或 shuffle。

两者分别报告 signed effect；不预设 IED 一定促发作。不同 event type 可以产生相反 impulse response。

## 5. 主要端点与时间窗

承重端点是完全未见 future block 的 held-out log score，拆：

- event count/rate；
- conditional mark/repertoire；
- participation/extent；
- multiband expression；
- H2b frozen seizure-risk/field readout 仅作 secondary，缺失不阻断 H3。

主 horizon 为 5/30/120 min fixed-time blocks；6 h 仅在连续 coverage 和有效不重叠窗足够时探索。event-count 100/1,000/10,000 只作映射/敏感性，不作为“长尺度”定义，也不做全笛卡尔网格。

## 6. 最小 perturbation 集合

首轮主比较只保留：

1. `real_sequence`；
2. `no_event_feedback`；
3. `state_matched_mark_replacement`（保留 event count/time）。

secondary：rate-preserving mark shuffle、预先定义 burst thinning/removal。intercept/count-matched、constant/drift zero-truth 继续作为回归测试和少量 sensitivity，不占一条完整人体主 arm。

所有 perturbation 从同一 pre-state、同一 exposure window、同一未来 target 开始；扰动后关闭真实未来 teacher forcing。窗口不跨 gap、split、seizure，统计分母是不重叠 physical blocks。

## 7. Background/common drive

`M0` 必须读取 manifest 中允许的 background/clock/multiscale covariates，避免把共同驱动误归为反馈。若 producer 不使用 background，明确 `uses_background=false`；不得把 a4/a5 名称当作语义。

可比较 event-only、background-only、combined 作为 observer 信息来源诊断，但它们不替代 M0/M1/M2，也不扩成主 arm 网格。

## 8. 执行计划

### C0：support 和 schema

1. 读取 checkpoint registry，验证 producer/trajectory/functional-readout hash 对齐。
2. 按真实 coverage segment 构造不重叠 5/30/120 min exposure/target blocks。
3. 报告每患者 TRAIN/inner-validation/development-test 的真实小时、block 数和有效独立分母。

### C1：functional innovation

先在固定 3 位长患者 × 3 seeds 画完整 trajectory，确认 pre/post、事件输入和 future outcomes 对齐；再扩所有有支持患者。只描述轨迹，不借此宣布 feedback。

### C2：M0/M1/M2

实现统一接口和低容量 signed adapters；用同一 optimizer/steps/early-stopping 规则训练。synthetic recovery 只校验三模型在已知零/非零边下可区分，不作为人体探索 gate。

### C3：人体主分析

运行 M0/M1/M2 的 5/30/120 min future-block score；burden 与 content estimand 分开。3 seeds 全跑；预先固定的主配置可增加 5 seeds，不按结果补 seed。

### C4：最小 perturbation 和收口

在固定 checkpoint 上做 real/no-feedback/state-matched replacement；再运行少量 secondary shuffle/burst。主图只呈 M0/M1/M2 gain 和 signed event-type impulse response。

## 9. 验收和允许结论

- M1 超过 M0：count/rate feedback-like predictive dependence。
- M2 在相同 count/time 下超过 M1：mark-specific feedback-like predictive dependence。
- 只有 observer innovation 或 hidden ablation 改变：event observation is informative，不称反馈。
- 只有 rolling exposure 与未来相关：antecedent association/common drive 未排除。
- 不重叠窗不足或模型未估计：instrument/data not estimable，不作 H3 阴性。
- signed effect 患者/事件型异质：如实报告促发作样、抗发作样或双向，不强行合并成单调恶化。
