# Topic 3：Where / SOZ 空间归因

> 状态：当前正式入口
> 范围：只讨论慢调制和时序差异在空间上发生在哪里，尤其是 SOZ / non-SOZ 的分离。

---

## 1. 这个 topic 只回答什么问题

本 topic 只回答：

1. lagPat 群体事件框架为什么回答不好 where。
2. per-channel relaxed-refine 框架下，SOZ 与 non-SOZ 是否真的不同。
3. 观察到的差异中，哪部分更像全局调制，哪部分更像 SOZ 的局部短程记忆。

它**不**回答：

- `~2 Hz` peak 是不是真的：那是 `docs/topic2_between_event_dynamics.md`
- 单个事件内部传播是否刻板：那是 `docs/topic1_within_event_dynamics.md`

---

## 2. 一句话当前结论

lagPat 群体事件框架中的 SOZ / non-SOZ 对比被结构性选择偏差严重污染。转到 per-channel relaxed-refine 后，raw serial correlation 的 SOZ 优势基本消失；更可信的信号是：**SOZ 通道在全局慢调制之上，可能额外保留了更强的局部短程记忆。**

---

## 3. 核心证据链

### 3.1 为什么旧 lagPat 框架不适合回答 where

旧框架有三层结构性偏差：

1. refine 先把通道宇宙截成 HFO 高发通道，天然富集 SOZ
2. 群体事件的 SOZ 标注是二值的，只要有一个 SOZ 通道参与就算 SOZ 事件
3. 群体事件本身就是多通道重叠现象，在 SOZ 富集通道宇宙里几乎必然被 SOZ 主导

因此旧 lagPat 结果里看到的 SOZ > non-SOZ，很可能混着：

- 真实空间差异
- 事件率差异
- 通道选择偏差
- group-event 定义本身的偏向

### 3.2 per-channel relaxed-refine 为什么更干净

新路线不是重写整条 pipeline，而是：

- 从 `*_gpu.npz` 加载 per-channel events
- 用 relaxed refine 扩大通道集
- 在通道层面计算 IEI / detrended metrics / event rate
- 再做 SOZ / non-SOZ 对比

这样统计单元从“群体事件类别”变成“subject 内的通道”，这是更像样的数据结构。

### 3.3 当前实际结果

Yuquan-only PR-1（Epilepsiae `gpu.npz` 全坏）得到：

- Yuquan 审计：`11/18` 有效，PASS
- Epilepsiae 审计：`0/20` 有效，FAIL
- relaxed refine 选 `k = 0.0`，中位通道数约 `33`
- 有效 SOZ/non-SOZ 配对：`9` subjects

主要结果：

- raw `iei_lag1_r`：SOZ 与 non-SOZ 无差异，`p = 1.000`
- detrended `iei_detrended_r`：SOZ 稍高，`7/9` 同方向，`p = 0.250`
- `detrend_fraction`：SOZ 更低，方向 `7/9` 指向“SOZ 慢漂移占比更低”
- `iei_median`：SOZ 更短，`p = 0.055`

### 3.4 这意味着什么

最有信息量的不是某个单独的 `p` 值，而是分解后的方向：

- raw serial correlation 无差异
- 但 detrended residual 倾向 SOZ 更高
- `detrend_fraction` 倾向 SOZ 更低

最合理的解释是：

- 全局慢调制对 SOZ / non-SOZ 都在起作用
- SOZ 通道之上还叠加了额外的局部短程记忆

这比“SOZ 整体更强”要精确得多，也比旧 lagPat 结果更可信。

---

## 4. 当前最可信的结果

- Epilepsiae 目前做不了正式 per-channel 空间归因，因为上游 `gpu.npz` 全是损坏桩
- 在 Yuquan-only 分析里，旧 lagPat 框架下的 SOZ raw-corr 优势并不稳
- 更可信的信号是 detrended 之后的残差方向，而不是 raw serial correlation
- `iei_median` 更短与 SOZ 更高事件率方向一致，但还只是边缘结果

---

## 5. 仍未解决的问题 / 风险点

- 最大现实问题不是统计，而是数据：Epilepsiae 还没解锁
- 当前 `n = 9` 配对太小，只能做探索性判断
- SOZ / non-SOZ 的 event-rate 支撑域不完全重叠，仍有混淆
- 当前结果最容易被误读成“SOZ 没差异”；其实更准确的说法是“差异主要出现在去趋势后的局部残差，而不是 raw 总量”

---

## 6. 代码与结果入口

- 主文档：`docs/archive/topic3/spatial_modulation_soz_analysis.md`
- 审计脚本：`scripts/audit_gpu_npz.py`
- 主脚本：`scripts/run_spatial_modulation.py`
- 作图脚本：`scripts/plot_spatial_modulation.py`
- 相关代码：`src/event_periodicity.py` 中的 per-channel / SOZ helpers，`src/group_event_analysis.py`
- 结果：`results/spatial_modulation/`、`results/refine_soz_validation/`

---

## 7. 与其他 topic 的边界

- 如果问题是“慢调制本身是否存在、是不是 oscillator”，去 `docs/topic2_between_event_dynamics.md`
- 如果问题是“SOZ 传播路径是否更刻板”，去 `docs/topic1_within_event_dynamics.md`
- Topic 3 关注的是 where，不是 whether

---

## 8. 历史文档索引

- `docs/archive/topic3/spatial_modulation_soz_analysis.md`
  - 保留完整的计划、执行过程、基础设施与阶段结果

当前正式口径以本文件为准。
