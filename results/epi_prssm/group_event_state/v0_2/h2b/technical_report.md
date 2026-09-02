# H2b（间期状态 → 发作转移）技术报告

状态：**科学结论全部撤回；本轮只保留工程层与分母层**
日期：2026-09-02（v2，取代 09-02 上午版）　分支：`codex/topic5-group-event-state-v02-b`
外部审阅：`shared/waiter/reviews/p0_b_cohort_report_consumes_invalid_and_unverifiable_state_registry_2026-09-02.md`

---

## 0. 撤回声明（先读这一条）

本报告 v1 曾写：「在 12–13 位病人、61–81 个留出发作事件上，没有看到间期状态超过静态
病人平均场的增量；无记忆对照表现一致。」

**这句话作为 H2b 结果撤回。** 三条独立理由，每条都单独足以使其不成立：

1. **读的不是合同定义的状态。** 共同合同 §4 明确承重对象是**冻结功能读出** `S_func`
   （未来事件数 / 条件 mark / participation / extent-STOP / multiband 的预测分布），
   并写明 `z_fast`/`z_slow` 只作诊断。B1/B2 实际读的是 `anchor_state.npz::state`——**原始 latent**。
   （补充精确性：cosine 对**正交旋转**是不变的，所以"latent 可旋转"这一条本身不构成反驳；
   真正成立的是——latent 可做**各维不同尺度的重参数化**，cosine 对此不不变；
   且合同本来就把 `S_func` 指定为承重对象。按合同，这是 compliance 问题，与度量不变性无关。）
2. **producer 不可验收。** 见 §2：两个递归 producer 的 cell 跨 27–29 种不同配置。
3. **关键对照从未执行。** `B_multiscale`、recent/current IED、以及合同 §6 规定的
   **主时间零假设 within-session block circular shift** 都没有实现。

因此当前 B2 只能记为：**在无效 producer 上、对非合同状态对象、缺主要对照的一次诊断**。

---

## 1. 本轮保留下来的（工程层，独立于 producer）

这一层不依赖任何模型，审阅亦认可保留：

| 项 | 数 |
|---|---:|
| 按录音编号 crosswalk、逐 onset 核对 | 274 匹配，**零** onset 落在自己录音外、零歧义、零重复 |
| 成簇归并后的独立事件 | **209** |
| rolling-origin 留出事件 | **99** |
| 有锚点：5min / 30min / 2h / 6h | 98 / 89 / 79 / 75 |
| 发作能量场（按**脑电起点**锚定） | 264 成功 / 10 丢弃（10 条全为窗口越出块边界） |
| 对拍已验收缓存（168 次 / 11 人） | 同锚点中位 **ρ=+0.9977**；通道顺序处处一致 |
| 脑电起点 vs 临床起点 | **145/168** 更早，中位 5.0 s，最大 86.2 s |
| 不应期敏感性 30/60/120 min | 留出 106/99/80（主口径不在悬崖上） |
| 锚点新鲜度 | 上一次间期事件中位早 5–14 s |

另修复两条 `block_id` 指向比自己发作早 ~14 h 结束的块，各救回 1 次发作。

**预注册基准仍然有效**（不依赖 producer）：静态"病人平均场"预测留出发作头 5 s
中位 **ρ=+0.41**、最高 +0.93。这是任何状态必须越过的线。

---

## 2. 新增：registry 的 fail-closed 校验（P0-2 的落地）

原 loader 只检查「文件存在且 `np.load` 成功」。现已加入默认开启的校验，
不通过即 `not_available`（诊断用途可显式 `verify=False`，输出带 `verified=False`）。

校验两条：**producer 内部 cell 配置是否同质**、**声明的 `checkpoint_sha256` 是否与盘上一致**。

对当前 registry 实测：

| producer | 可采纳 cell | 结论 |
|---|---:|---|
| `P_memoryless` | **22** | cell 配置同质（27 cell 共 1 个配置） |
| `P_local` | **0** | cell 跨 **29** 种配置（每个 seed 内部就有 10 种） |
| `P_slow` | **0** | cell 跨 **27** 种配置（每个 seed 内部 9 种） |

`checkpoint_sha256` 抽查 10/10 全部吻合 → **不是文件损坏，是配置异质**。
承载假设的两个递归 producer 恰好全部不可采纳，只剩无记忆对照可用——
**在这种情况下无法做有意义的对照比较**。

### 对审阅措辞的一处更正（使其结论更强而非更弱）

审阅称「`54845f4` 并不包含实际训练代码」。实测：该 commit **存在**，且**包含** 26 个
`topic5_group_event_state/v02` 下的文件。真正可验证的缺陷更锐利：

- artifact 写入时间：**2026-09-01 23:04**
- registry 声明的 `source_commit` `54845f4d` 提交时间：**2026-09-02 04:13**

**声明的源码 commit 比它所描述的 artifact 晚 5 小时**，因此不可能是产出它们的代码。
（更早一版 registry 声明的 `18027162` 提交于 09-01 23:08，也晚于 artifact 4 分钟。）

---

## 3. 新增：`B_multiscale` 存在未来标签泄漏（P0-3，已验证）

`.worktrees/topic5-group-event-state-v02-a/src/topic5_group_event_state/v02/baseline.py:251`

```python
to_next[ok] = np.clip(onsets[j[ok]] - grid.t_anchor[ok], 0.0, SEIZURE_TIME_CAP_SECONDS)
...
_add(np.log1p(np.minimum(since_prev, to_next)), ["log_time_to_nearest_seizure"])
```

`to_next` = 距**下一次**发作的时间。该维度在时刻 t 提前知道未来。

**后果**：本线原先提给 A 的「请把 111 维特征照样导出」的请求**作废**——照做会把一个
作弊基线引进来，反而把状态臂的增量压成假的负数。请求已修订为：先删除一切由 `to_next`
派生的维度，重建因果可用 baseline，并在 manifest 声明「不含前瞻量」。

---

## 4. 本轮修正的自身错误

| # | 错误 | 影响 | 状态 |
|---|---|---|---|
| 1 | `eval_events` 报的是 5 min 网格行，不是独立发作 | 916 上 1578 行仅对应 **40** 次发作；v1 报的 "987 events" 实为 **24** 次 → **41 倍虚报** | 已改为同时输出 `eval_event_rows` 与 `eval_distinct_seizures`，后者为正式分母 |
| 2 | 温度网格最大只到 4.0，**不含**均匀权重 | 文档称"基线是严格嵌套特例"在**模型选择中并不成立**（τ=4 距均匀 3%） | 已把 `inf` 放进网格；留一现在可以真正选中基线 |
| 3 | 首版 registry 校验规则过严 | 会把 cell-vs-producer 记账哈希差异误判为缺陷，导致 **全部 154 cell 被拒**，掩盖真实问题 | 已改为只判「producer 内部 cell 配置异质」+ 校验 checkpoint 哈希 |

加入 `inf` 之后，`P_memoryless` 在四个提前量上的中位增量**恰好为 0.0000**——
因为 TRAIN 内留一现在**直接选中了均匀权重基线**，即状态相似度加权在训练集上就没挣到位置。

---

## 5. 仍未做 / 未结清

- `S_func` 冻结功能读出（合同承重对象）**未导出**，H2b 主分析无法进行。
- `B_multiscale`（去泄漏后）、recent/current IED、**block circular shift 时间零假设**三条对照未实现。
- B1 缺 calibration、seizure-level ranking、episode 内先汇总再 patient-first。
- `epilepsiae_922` 的 8 次对拍偏低仍未查明。**已按要求预先声明排除敏感性**（§6）。

## 6. `epilepsiae_922` 预声明排除敏感性

在 v1（未加校验、含 `P_local`/`P_slow`）的数字上，排除 922 后各格中位变化 **−0.0025 ~ +0.0012**，
与效应本身同量级，但所有中位仍落在零附近 ±0.006 内，无实质改变。
该表随 v1 数字一并**降级为诊断**，待 producer 重新验收后重算。

## 7. 允许的措辞

仅允许：

- 「crosswalk / episode 合并 / lead-time 覆盖 / early-ictal 场 / patient-first 分母 / 不可估计分类
  这套**工程与分母层**已建立并可复现。」
- 「在当前 raw-latent 最近状态读出方式下、且在**不可采纳的 producer** 上，未观察到超过患者平均
  发作场的增量——**这是诊断，不是 H2b 阴性**。」
- 「B1 在当前估计量下 assay not estimable。」

禁止：❌ 任何形式的「H2b 阴性」；❌「间期状态与发作无关」；❌ 把 B1 的负增量读成状态有害；
❌ 引用 v1 的 `P_local`/`P_slow` 数字作为结果。
