# mini-W_event 设计稿（Step 6，2026-06-23）— 待 review，未执行

**目的（一句话）**：验证 n17.6 是不是**只降低 finite-event 触发阈值 K_min，而不改变传播形状 W_event_shape**。
即测试 `K_min^{n17.6}(q) < K_min^{bare}(q)` 且 `W_shape^{n17.6}(p|q) ≈ W_shape^{bare}(p|q)`。
**这是 B1（B2 唯一缺口）的验证。** 不是全图、不是 μ 相图、不是 dynamic m。**PILOT-FIRST：本稿确认后才跑。**

前置已成立：B0 支持、B2 概率层支持（ceiling：K_min(n17.6)=1.1 vs bare=1.6，核 OR=4.49 [2.89,6.97]，
全程不失控）。污染处置：Step B 的 `spontaneous_ignition_flag` 逐 seed 剔除（B2 底物 bare/n17.6 自发点火 0/8、0/12，
mini-W 沿用同 flag）。

---

## 0. 工程现实（必须先解决，但不碰 engine）

当前 `run_m3_kick_calibration.py` 只在**核中心**施加 finite-pulse kick。mini-W 需要**多源 kick**（在 5 个不同 (x,y) 点着）。

- **改动落在 runner 层**：新增 `--kick-xy X Y`（或多源列表），由 runner 计算"哪些 bin 收到额外外部 Poisson 驱动"的 mask，
  传给 engine 的 per-bin 外部速率。**engine 不动**（它只接受 per-bin external rate）；保持 `M3_BASE_SHA=da5fc18c27d5340a` μ=0 bit-parity。
- **复用现有检测器**：EA 事件对齐窗、`returned/runaway`（raw core_kick 轨迹）、source-excluded 差分空间度量、
  `core_only_quiet` 多数票门，全部沿用，只是 kick 中心可变。
- **复用 Step B flag**：每个 (source, seed) 先过 `spontaneous_ignition_flag`，自发点火 seed 标记并在主估计中剔除（median 汇总）。

新分析助手放 `src/sef_hfo_mini_w_event.py`（W_shape 构造 / axis 拟合 / ordering predictivity），不混进 runner。

---

## 1. 5-source 几何（L=20，先 pilot 不做全图）

E→E 长轴方向当前 θ=45°；off-axis = 垂直方向（135°）。源相对 sheet/核中心偏移 `R_src`（建议 1 个电极间距 ≈ 4mm，
落在 sheet 内、远离边界 ≥ r95 典型值以免墙效应）：

| source q | 位置（相对中心） | 测什么 |
|----------|------------------|--------|
| `center` | (0,0) = 核所在 | 基准（= 当前 ceiling 的工作点） |
| `+axis` | +R_src·(cos45,sin45) | 沿 E→E 轴、偏离核 |
| `-axis` | −R_src·(cos45,sin45) | 沿轴另一侧 |
| `+offaxis` | +R_src·(cos135,sin135) | 垂直轴、偏离核 |
| `-offaxis` | −R_src·(cos135,sin135) | 垂直轴另一侧 |

回答：① 沿轴 source 是否更易点着 / 传得更远；② off-axis source 事件是否不同；
③ **n17.6 是否在所有 source 上降 K_min，还是只在中心核附近**（核效应是局域还是全局）；④ 传播形状是否沿同一主轴。

bare 因无核、近似各向同性（仅边界破缺），5 源主要测传播是否沿 E→E 轴各向异性；
n17.6 的 5 源测核的阈值下移空间范围 + 形状。

---

## 2. source-specific K50 / K_min 标定（不固定 kick 比较）

finite event 非线性响应，固定 kick 会把"哪里更易点着"和"点着后往哪传"混在一起。**每个 source q 各自标定**：

每个 (substrate, q) 扫 `kick = 0.8, 1.0, 1.1, 1.2, 1.3, 1.4, 1.6`（边界必要时加密），seeds ≥ 8（建议 12），定义：

- `K_min(q)` = 最小 kick s.t. `P_EA-local-returned ≥ 0.7`（沿用 ceiling 判据：r95_ea≤6 且 far_ea≤0.5 且 returned；EA primary）。
- `K50_returned(q)` = `P_EA-local-returned(q,K) ≈ 0.5` 的 kick（线性内插）。

**primary 用 source-specific K50（或 near-threshold K）估 W_shape**——这样比的是"点着以后怎么传"，不是"哪里更易点着"。
固定 K 作 sensitivity 备查。

---

## 3. 三个对象必须分开（不混成一个矩阵）

### (1) K_min(q) — 招募阈值图（B2 核心）
`K_min(q)` per (substrate, source)。回答"哪里更易被有限幅扰动招募"。bare vs n17.6 比，看核是否在多个 source 上压低 K_min。

### (2) W_event_shape(p|q) — 传播形状（B1 核心）
**只在成功的 finite returned events 中**算。对 source q 在 K50(q)（或 near-threshold）点着、回静、不失控的事件，取 EA 早期窗
`[t0, t0+Δ]`（Δ 沿用 EA_DELTA2=10ms），定义每个 bin p 的早期被招募强度（event-aligned early activity，source-excluded、
跨成功事件平均、过 Step B flag 剔除自发点火 seed）：
```
W_shape(p|q) = ⟨ p 在 q-发起的成功 returned 事件早期窗的归一化招募强度 ⟩
```
回答"点着以后沿哪里传"。**这是 phenotype 之外、真正的传播形状层。**

### (3) P_escape(q,K) — 逃逸概率（B3 前置）
`P_escape(q,K) = P(sustained recruitment | q,K)`（用 runaway / 未回静 判据）。ceiling 到 1.6 全 0，仍逐 source、逐 K 记录，
为后续 μ 提供 escape 基线。回答"同一 source 在更高 kick / 之后 μ 下是否转持续招募"。

**三件分别落盘**：`mini_w_kmin_map.csv` / `mini_w_shape_<substrate>_<source>.npy(+csv)` / `mini_w_pescape.csv`。不合并。

---

## 4. W_event 成功标准（B1a–d，draft 阈值，审计期可调）

### B1a — shape 可重复
同 substrate、同 source、不同 seed：`sim(W_shape^{seed_i}, W_shape^{seed_j})`（cosine 或 Spearman over bins）
**高于** shuffled-seed / shuffled-spatial-bin null。draft：观测相似度 ≥ null 95 分位（per source）。

### B1b — shape 沿 E→E 主轴
`W_shape` 的 principal axis（加权二阶矩的主特征向量）接近 θ=45°。**控制**：跑 AR=1（各向同性连接）或把 θ 旋转，
shape 主轴应相应**减弱 / 旋转**（证明形状由连接各向异性驱动，非 montage 伪影）。draft：|主轴角 − 45°| 小（如 ≤ 20°），
且 AR=1 时各向异性显著下降。

### B1c — shape 比 distance/rate 更能预测 event order
对每个成功事件，取实际 bin 招募先后顺序，比较四个预测子的排序相关（Spearman vs 实际 order）：
① W_event-predicted order；② pure distance（离 source）；③ local rate；④ K_min/susceptibility map。
**W_event 必须显著优于 distance 和 rate**，否则它不是有用的传播算子。draft：W_event 的 ρ 的 CI 下界 > distance/rate 的 ρ。

### B1d — 核改阈值不改形状（B2 完成 / B1 落地）
取 **matched finite returned events**：`bare @ K≈1.6`（其 K_min）vs `n17.6 @ K≈1.1–1.2`（其 K_min）。比较两者 W_shape：
```
W_shape^{bare}(K_min^bare) ≈ W_shape^{n17.6}(K_min^n17.6)  ?
```
draft：两者 W_shape 相似度高于跨-source / 跨-substrate-shuffle null。若 shape 相似 **且** K_min 下移 →
支持"核/易感性门控同一传播场，不创造新路径"。**这才是写 'B2 完成、同一 W' 的唯一充分条件。**

---

## 5. 进入 μ 相图的 gate（元素 7，硬条件）

mini-W 过了**也不立刻**进相图。进 μ 相图前需全部满足：
1. B1a W_shape 可重复（≥ null）；
2. B1c W_event ordering 胜过 distance 和 rate；
3. B1b shape 沿 E→E 主轴 + AR=1/θ-rotation 控制通过；
4. B1d bare-high-kick 与 n17.6-low-kick 的 matched shape 相似（核只改阈值不改形状）；
5. **合成结论过真实 masked lagPat/rank/KMeans pipeline**（原 M3 plan 硬条件：不是只看 raster，要把 SNN 事件读回真实
   propagation-rank/template 机制层）；
6. 污染 seed 全程用 Step B flag 剔除、core_only 用 median。

满足后相图横轴**改名**：不叫 `Λ0(W_small)`，叫 **`Λ_event`** 或更保守的 finite-event recruitment gain
`R_event = #newly-recruited-bins_{g+1} / #front-bins_g`。h 与 Λ 用 W_event 定义，不用已被否定的 W_small。

---

## 6. 失败口径（若 B1 不过，怎么写）

若 n17.6 确降 K_min 但 W_shape 不稳 / 不能预测 order：
> 局部低阈值核可提高有限事件发生概率，但当前 minimal local-W SNN 尚未形成稳定可读的 propagation operator。

即：可解释 event susceptibility，不能解释真实数据的稳定传播模板；**不进 μ 相图**；回头考虑更稳的结构异质性 /
multi-ignition field / 真实数据约束下的 W 估计。**这不是白做——它把模型边界说清。**

---

## 7. 现在仍不能写 / 不能做

- 不能写"同一 W 已证明" → 只能写"同 phenotype / 同类 finite returned event",直到 B1d 过。
- 不能进 μ 相图（B1 未过）；不能写"发作样机制已开始支持"（B3 未碰，ceiling 全回静不失控）。
- 不做全图、不做 dynamic m、不做 ablation —— pilot 先行。

---

## 8. 执行计划（Step D，等本稿确认后）

1. runner 加 `--kick-xy`（多源 kick mask，engine 不动，加 bit-parity 回归测试：单 source center == 现有行为）。
2. `src/sef_hfo_mini_w_event.py`：W_shape 构造 + axis 拟合 + ordering predictivity + B1a-d 助手（TDD）。
3. **pilot**：bare + n17.6 × 5 source × kick{0.8..1.6} × 12 seed，先出 K_min(q) 图 + 单 source(center) 的 W_shape 可重复性，
   **停下来给 review** —— 不一次跑完 B1a-d。
4. review 通过再补 off-axis source + B1b 控制（AR=1/θ-rotation）+ B1c ordering + B1d matched-shape。
5. 全过 → 进 §5 gate；任一不过 → §6 失败口径。

**数据/脚本沿用** `results/topic4_sef_hfo/m3_local_w/`；新结果目录 `mini_w_event/`。
关联：[[m3_finescan_recap_2026-06-23]]（B 分层现状）、Step B 审计（污染剔除）。
