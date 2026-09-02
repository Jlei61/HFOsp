# Group-Event State v0.2 — 全程序整体收口（A + B + C）

日期：2026-09-02　范围：三条线合并判读，供 v0.3 立项使用
机器状态：**`V0_2_CLOSED__ENGINEERING_COMPLETE__NO_HYPOTHESIS_ESTABLISHED_OR_REFUTED`**
依据：三份同日 P0 审阅 + 本线对 registry 的独立复核（`shared/waiter/reviews/`）

---

## 0. 一句话

v0.2 **工程上跑完了**（三条线全部满格执行、零失败），**科学上一条假设都没有立起来，也一条都没有被推翻**。
全程序只有**一个**读数是干净的：**跨事件携带历史能改善下一次事件的到达时间**——
这是短程 predictive memory，不是慢状态、不是 repertoire 状态、更不是生理状态。

**v0.2 不产出任何关于"间期状态 ↔ 发作"的结论。**

---

## 1. 三条线的执行与判读

| 线 | 执行 | 科学判读 | 阻断原因 |
|---|---|---|---|
| **A** H1 / H2a（建立状态） | 27/27 患者，producer / future-block / same-prefix 全部完成 | **不可判** | 主参照 `B_multiscale` 含未来发作变量；`P_slow` 按 anchor 而非物理时间等权；same-prefix 用整段 core 而非首 100 ms；wrong-time 用 `np.roll` 事件计数位移而非物理 block shift；24 个读出格 inner-validation 爆炸未被 gate |
| **B** H2b（发作转移） | 133 格 / 266 次运行，零失败；分母层完整 | **不可判** | 读的是 raw latent 而非合同 `S_func`；`B_multiscale`、recent IED、block circular shift 三条对照未执行；B1 assay not estimable |
| **C** H3（事件反馈） | 486/486 models、81/81 ×3 stage、29/29 测试，零 OOM/NaN | **不可识别** | `M0` 与 M1/M2 不共享看过同样历史的 pre-state，`M1−M0` 同时改变两件事；innovation 测的是 observer 后验更新；合成尺对零真值不居中且不随强度单调；人体对比全部低于 refit floor |

### 唯一干净的科学读数（全程序）

`P_local` vs `P_memoryless`——**同一 local objective**，且我独立复核确认二者的
**实际训练配置（`chunk_events` / `batch_segments`）在 27/27 位患者上逐患者匹配**：

> **跨事件携带历史改善下一次事件的到达时间**：中位 gain **+0.0774**，**21/27** 患者为正，
> 符号检验 **p = 0.0059**。

允许的措辞仅限：**短程 predictive memory 存在**。
禁止：慢状态、repertoire 状态、生理状态、任何跨任务含义。
（`P_slow` vs memoryless 改变了训练目标，**不能**用来隔离"是否携带历史"。）

---

## 2. 跨线共因缺陷（v0.3 必须先解决）

### 2.1 未来标签泄漏（影响 A 与 B 的主参照）

`v02/baseline.py:251`
```python
to_next[ok] = np.clip(onsets[j[ok]] - grid.t_anchor[ok], 0.0, SEIZURE_TIME_CAP_SECONDS)
_add(np.log1p(np.minimum(since_prev, to_next)), ["log_time_to_nearest_seizure"])
```
`to_next` = 距**下一次**发作的时间。`B_multiscale` 因此不是因果可用的纯间期基线。
→ **A 报告中一切"相对 `B_multiscale` 无增量"的人体结论暂不可采信；B 侧原"照样导出 111 维"的请求作废。**

### 2.2 provenance 不成立（影响全部三线的可复现性）

对 189 个训练 cell 的 `result.json` 逐个复核：

- **实际训练用了 13 个不同 commit**，而 registry 只声明一个（`54845f4d`，覆盖 27/189）。
- 声明的 `source_commit` 提交于 **2026-09-02 04:13**，而它描述的 artifact 写于 **2026-09-01 23:04**
  ——**声明源码晚于自身产物 5 小时**，不可能是产出它们的代码。
- `config_hash` 有 **57** 个不同值，而实际训练配置组合只有 **18** 种
  → 该字段并不表征训练配置，**不能**用它做验收判据（见 §4 本线自纠）。

### 2.3 固定时刻权重（影响 A 的慢状态定义）

`P_slow` 按 anchor 数等权训练，事件密集段获得更多 future-block 梯度；
合同要求的是按**记录物理时间**等权的固定时间慢状态 producer。**当前 `P_slow` 不是合同定义的那个对象。**

### 2.4 OOM 后配置漂移（局部，可定位）

189 个 cell 中 **12 个**被 OOM 降过 `chunk/batch`（9 个 (64,4)、3 个 (32,2)），
集中在 4 位 Yuquan 患者、以 `P_slow` 为主：
`hanyuxuan`、`sunyuanxin`、`zhangbichen`、`zhangkexuan`。
这 12 格不能与其余同 arm 并列；其余 **177/189 在预期配置 (128, 8) 上**。

### 2.5 "读出即状态"的混淆（影响 B 与 C）

合同 §4 把承重对象定为**冻结功能读出** `S_func`，并写明隐变量只作诊断。
B 读了 raw latent；C 的 innovation 量的是同一 RNN 读入事件后的后验更新。
两者都把**观测者更新**放到了**生理状态**的位置上。

---

## 3. v0.2 可以保留、v0.3 可以直接复用的东西

**数据与分母层（不依赖任何 producer，已验收）**

- 按录音编号的发作 crosswalk：274 次匹配，**逐 onset 零误差**，零歧义、零重复。
- 成簇归并：274 次发作 → **209 个独立事件** → rolling-origin 留出 **99 个**。
- 各提前量可用锚点：5min **98** / 30min **89** / 2h **79** / 6h **75**。
- 状态覆盖与监测覆盖分离；断监测 → right-censored，不冒充"没发作"。
- early-ictal 场按**脑电起点**重建，与已验收缓存同锚点对拍中位 **ρ = +0.9977**。
- 不应期敏感性 30/60/120 min → 留出 106/99/80，主口径不在悬崖上。

**方法学副产（可直接进 v0.3 的合同）**

- **脑电起点在 145/168 次发作中早于临床起点**，中位 **5.0 s**、最大 **86.2 s**。
  → 任何"发作最初 N 秒"的定义必须按脑电起点，否则最多可能从电活动开始后 86 秒才起算。
- **预注册基准**：静态"病人平均场"预测留出发作头 5 s 中位 **ρ = +0.41**、最高 **+0.93**。
  → **v0.3 的任何状态必须越过这条线，而不是越过零。**
- 单次发作自身可复现度中位 +0.30，**但它不是上限**——平均能消噪，静态基线在 9/12 人反超它。

**工程基础设施**：A 的训练/汇总/作图链、B 的 crosswalk/risk-grid/target/scoring/registry 校验、
C 的 486-cell 全链路执行，均可复用。

---

## 4. 本线（B）在收口过程中的自纠

除已归档的 8 项外，本次整体收口又发现并修正 **1 项**，且它曾直接影响 v0.2 的判读：

> **我曾用 `config_hash` 做 producer 验收判据，据此判定 `P_local`/`P_slow` 全部不可采纳。这是错的。**
> 复核 189 个 `result.json` 后：`config_hash` 有 57 个值而实际训练配置只有 18 种，
> 该字段不表征训练配置。改用**运行自身记录的 `chunk_events`/`batch_segments`** 判据后，
> **144/154 格可采纳**，仅 10 格因 OOM 降配被拒。
>
> 同时更正：审阅称 A 的"逐患者配置匹配"不成立——**在实质维度上 A 是对的**，
> `P_local` vs `P_memoryless` 在 (chunk, batch) 上 **27/27 匹配**，差异只在 commit。
>
> **但 H2b 的撤回结论不变**——它由 §2.5（读错对象）与 §2.1/对照缺失独立支撑，与本条无关。

---

## 5. v0.3 的准入条件（按依赖排序）

1. **删除一切前瞻量**，重建因果可用的 `B_multiscale`，并在 manifest 声明"不含前瞻量"。
2. **导出冻结功能读出 `S_func`**（带名称、TRAIN 标准化统计、checkpoint hash）；
   主分析只读它，raw latent 降为 sensitivity。
3. **`P_slow` 按记录物理时间等权重训**，使其成为合同定义的固定时间慢状态 producer。
4. **provenance 逐 cell 重建**：真实训练 commit、真实配置、payload hash；registry fail-closed。
5. **C 线三臂共享 pre-state / encoder / decoder 初始化**，并先通过合成尺
   （零真值居中、count 真值随强度单调且只被 M1 捕获、mark 真值只使 M2−M1 单调）。
6. **B 线补两条对照**：recent/current IED、**within-session block circular shift** 时间零假设；
   B1 补 calibration、seizure-level ranking、episode 内先汇总再 patient-first。
7. 统一 OOM 处理：降配即记 `resource_failed`，不与同 arm 并列。

**在 1–4 完成之前，v0.3 不得产出任何关于 H1 / H2a / H2b / H3 的人体结论。**

---

## 6. 允许与禁止的措辞（v0.2 对外口径）

**允许**：

- 「v0.2 完成了三条线的完整 development 执行；**没有任何假设被建立或被推翻**。」
- 「唯一干净读数：跨事件携带历史改善**下一次事件到达时间**（+0.0774，21/27，p=0.0059），
  即短程 predictive memory。」
- 「数据/分母/目标层已验收并可复用；脑电起点与静态基线两条方法学结论可直接进 v0.3 合同。」

**禁止**：

- ❌ 「H1 / H2a / H2b / H3 阴性」——四条没有一条被有效检验过。
- ❌ 「当前多时长模型没有慢状态」——`P_slow` 尚不是合同定义的那个对象。
- ❌ 「IED 反馈到慢状态」或把 observer 后验更新称作生理 feedback。
- ❌ 引用任何"相对 `B_multiscale` 无增量"的人体数字（参照含未来变量）。
- ❌ 任何临床部署性能说法。

---

## 7. 索引

- 本线阶段封存：`group_event_state_v0_2_h2b_stage_closeout_2026-09-02.md`
- 三份 P0 审阅：`/data/hfosp_group_event_state_v0_2/shared/waiter/reviews/p0_{a,b,c}_*_2026-09-02.md`
- 合同：`group_event_state_v0_2_{common_contract,engineering_invariants}_2026-09-01.md`
- 未触碰：formal/sealed 分区、paper-ready Fig1–Fig4。
