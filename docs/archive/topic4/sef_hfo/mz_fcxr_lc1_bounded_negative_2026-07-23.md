# FCXR-LC1 — 动态慢反馈生命周期闭环（bounded-negative 收口）

日期：2026-07-23
分支：`codex/topic4-mz-fcxr-lc1`（worktree `.worktrees/topic4-mz-fcxr-lc1`），base = `6819643`（FCXR-RC1 Stage D bounded-negative）
设计合同：`docs/superpowers/plans/2026-07-22-topic4-mz-fcxr-lc1.md`
结果根：`results/topic4_sef_hfo/mz_full_conductance_spatial_relay/lifecycle_closure/`
机器判决：`candidate_verdict.json`；provenance：`run_manifest.json`
状态：**E0–E4 全跑完（seed1+seed3）。E4 bounded-negative：无 confirmed 闭环 + 新正结果（X 能终止一次持续发作）。**

## 0. 朴素话摘要（测了什么 / 怎么测的 / 揭示了什么）

**测了什么。** 想让一个模拟脑网络**自己**走完一遍："平时零星放小火（间期）→ 一次持续的大发作 → 自己停下来 →
回到平时零星放小火"，全程只靠两个慢慢变化的内部记忆开关驱动，不许外部踢一下、不许手动重置、不许拼接窗口。

**怎么测的。** 装两个慢开关：**第一个（松刹车）** 随一次次放电逐步松开抑制；**第二个（耗资源）** 在感到"持续密集
放电"时把参与传播的兴奋连接资源耗掉、延迟地掐断发作、再慢慢恢复。分五步，每步都跟"平时静息该长什么样"对比验收。

**揭示了什么。** 前四步都立住：静息干净；只松刹车就能让系统自己先稳住 ~9–10 秒间期、再沿一条平滑台阶（不是一步跳）
走进一串有界、会自发熄灭的密集事件；第二个开关的持续度探头分得开两态。第五步完整闭环**没走通**，原因很具体——
允许用的两档"松刹车速度"恰好把需要的中间档夹在中间：**松得温柔那档**，密集事件太短、刹车自己就把它们熄灭了，
第二个开关（要靠"持续"才触发）根本用不上；**松得狠那档**，确实出一次持续发作、第二个开关**被真正调用、把连接资源
耗到只剩三到五成、把发作掐断回到有界**（这一档单独放会失控到 ~450Hz，加了开关就不失控——**两套网络都复现、时间
顺序对**），但发作来得太快（前面只 ~0–1 秒间期，不够 8 秒）、掐断后刹车仍停在松开位（有一套还被掐得近乎全静默），
没有干净的统计恢复。净结论：零件都验证过了（含新拿到的正结果：第二个开关能终止一次真正的持续发作），闭环差的是
一个**介于两档之间、不在允许标定里的松刹车速度**。

## 1. 设置（锁定底座 + 两个慢变量）

- **快系统（RC1，本轮不动）**：full_conductance，external additive FF AMPA + recurrent E→E 电导朝 `E_E=58` +
  recurrent-only 平滑饱和 `g_sat=21.6`，`V_match=18`、`gaba_gain=1.125`、`dt=0.05`、`N=40000`、`L=20`，
  `epilepsiae_1146` narrow，seeds 1/3。
- **慢变量 1：Z（抑制效能）**，动态 `z_inf = H(I_th_EI − I_I)`、Euler `tau_z`。既有标定（`results/topic4_sef_hfo/mz_slowvars/calibration.json`）：
  **q75 主** `I_th_EI=95.198 / tau_z=5000`（mid depletion），**q50 sens** `I_th_EI=1.665 / tau_z=10000`（strong）。
- **慢变量 2：X（持续度门控 E→E relay）**，本已存在（`use_x`、`y_j` 传感器 `tau_y`、Hill `x_inf=1−(1−x_min)·Hill([y−y_gate]₊)`、
  因果 `ee_relay_send=x(t−)` 按源缩放外向 E→E）。**本轮唯一引擎改动 = 非对称 kinetics**：`tau_x_down`（耗尽）/`tau_x_up`（恢复），
  off-by-default 逐比特还原单-`tau_x`（契约 C1–C9 全绿，`tests/test_mz_full_conductance_spatial_relay.py`）。`mz_slow_vars.py` 非-blessed；
  blessed engine（`kick_probe.py` 等 6 个）未动、无 re-bless。
- **相图坐标**：`D_Z(t)=Σp_i(1−z_i)/Σp_i`、`D_X(t)=Σp_i(1−x_i)/Σp_i`（`p_i`=onset-depletion 权重，取 `zA_q75_tz5000` snapshot），从 z/x snapshots 后处理。
- **分类器**：`src/topic4_mz_fcxr_lifecycle.py`（纯逻辑，17 tests）。承重反作弊 L5：post-ictal 全静默（event_rate=0）→ `PERMANENT_SILENCE`，**永不** `RECOVERED`；孤立 burst 平滑、`DENSE_EVENT_TRAIN` 需 sustained。

## 2. E1 baseline（ACCEPTED）

seed1/3 各 24s slow-off。两 seed 都 `INTERICTAL_BASELINE`（分类器 hardening 后：忽略孤立 burst），零 clip，`tau_eff_min≈0.274/0.277ms ≫ 2dt`，
`n_returning=34/67`，`roll_hi=9.74/9.44Hz`。统计带（`baseline_contract_seed{1,3}.json`）建立。峰值 RSS 20.2GB。

## 3. E2 dynamic Z-only（ACCEPTED）

`use_z=on`，X/M/phi off，无 kick。
- **q75 主**：seed1 `DENSE_EVENT_TRAIN`（pre 10s、`D_Z→0.165`、`max_step=0.0085`=event-locked staircase 非硬阈值伪跳、end 8.9Hz 有界）；
  seed3 `TERMINATED_REFRACTORY`（pre 9s、`D_Z→0.186`、end 9.0Hz 有界）。→ **动态 Z 自驱 间期→有界亚稳密事件区，平滑台阶，两 seed 复现。**
- **q50 sens**：`ICTAL_LIKE_BOUNDED`、`D_Z→0.805`、`z→0.21`、end **452.8Hz bounded=False**（不熄的高态，I_th 太低 Z 不恢复）；单 run 4.3h（runtime 教训）。
- E2 gate 过（两档都进、q75 有界、无伪跳）。

## 4. E3 X sensor separation（ACCEPTED）

frozen-Z D=0 vs D=0.15，`use_x + x_min=1`（X 中性、byte-identical 放电，只录 y_j）。`y_gate = baseline y q99.9`。
- seed1：`separated=True`，y_gate 76.6，base_occ 0.001、dense_occ 0.254（**254×**），y_peak base 79.0/dense 127.1，dense 有界 11.5Hz。
- seed3：`separated=True`，y_gate 85.3，base_occ 0.001、dense_occ 0.191（**191×**），y_peak 88.9/129.8，5.7Hz。
- core+axis 同时招募（~400-600ms）、off 静（合 Stage-D 核结构）。→ **传感器分得开间期 vs 密态，两 seed 同向。**

## 5. E4a xcontrol（x_min 客观标定）

frozen D=0.15 + 动态 X（`tau_x_down=1000/tau_x_up=5000` 固定），扫 `x_min∈{1,0.7,0.5,0.3,0.1}`，量 dense occ 降幅（§10.1，非从 lifecycle 反选）：
neutral(x_min=1) occ 0.773 → 0.589/0.599/0.594/0.479（x_min 0.7/0.5/0.3/0.1）；`x_reached` 底 ~0.72（x 自限，从不到 floor）。
→ **X 有 authority（把 occ 压下 0.18-0.29）但在冻结 Z 下不 terminate（占比仍 ≥0.48，衬底上膛再点火，呼应 Stage-D STD 结论）；这低估了动态 Z 情形（Z 恢复也复原抑制）。bracket {0.1,0.3,0.5}。**

## 6. E4 lifecycle（BOUNDED-NEGATIVE + 新正结果）

动态 Z(q75/q50) + 非对称 X，无 kick。

### 6.1 q75 + X = X inert（robust）
- pilot（x_min=0.3, td=1000, tu=5000, 20s）：`DENSE_EVENT_TRAIN`、`D_X_max≈9e-6`、`x_relay_min=0.964`。
- 最激进（x_min=0.1, td=500, tu=5000, 16s）：`DENSE_EVENT_TRAIN`、`x_relay_min=0.921`、`y_peak=96.2`（>gate 76.6）但 **`y_occ_above_gate=0.039`**（只闪 3.9% 不持续）。
- **机理**：q75 密集事件是亚稳、Z 恢复自己就熄灭它们；episodes 太短 → y 只闪不持续 → 持续度门控的 X 积累不了耗尽 → **X inert，与 x_min/tau_x_down 无关（gate 上游卡住）**。q75 的"发作样"其实是 Z 单独驱动的 间期↔密态振荡，没有给 X 抓的持续发作。

### 6.2 q50 + X = X 终止一次持续发作（新正结果，两 seed 复现）
- seed1（x_min=0.1, td=1000, tu=10000, 12s）：`UNRESOLVED` **bounded=True** end 4.1Hz；`x_relay_min=0.493`、`D_X_max=0.128`、`y_peak=160.9`、`y_occ=0.137`、**`x_after_onset=True`**；bout=[6,7]、pre 1000ms、post_return 0ms。
- seed3（同 cfg）：`UNRESOLVED` **bounded=True** end 0.1Hz；`x_relay_min=0.322`、`D_X_max=0.214`、`y_occ=0.171`、`x_after_onset=True`；bout=[8,10]、pre 0ms、post_return 1000ms。
- **机理**：q50 撑出**持续**发作（y 持续高）→ X 被真正调用、x 耗到 0.32-0.49、把发作掐断且**有界**（q50 单独 452Hz runaway；q50+X 不失控，`D_Z_max` 只 0.29-0.35 而非 0.805，因 X 抑制活动反过来限制了 Z 耗尽）。**X 能终止一次持续发作、时间顺序对、两 seed 复现——这是本轮新正结果。** 但 q50 耗 Z 太快 → pre-ictal 只 0-1s（不够 8s）+ 掐断后 Z 停在耗尽位（seed3 甚至近全静默过抑制）→ 无干净统计恢复 → `UNRESOLVED`。

### 6.3 为什么闭环没走通
两档允许的 Z 失效速率**夹住**需要的中间档：
- q75 太温柔：留住长间期 + Z 自恢复，但 episodes 自灭、无持续发作 → X 无从发力。
- q50 太猛：撑出持续发作 + X 能终止（有界、causal），但 pre-ictal 太短 + Z 不恢复 → 无恢复。
缺的是一个**介于两者之间**的失效速率：慢到留 ≥8s 间期、又能撑出不自灭的持续发作、且事后能恢复。不在冻结的 {q75,q50} 标定里。

## 7. 判决 + 措辞纪律

> **bounded-negative**：用两档允许标定（q75/q50）未得到 confirmed 单发作闭环；它们夹住需要的中间档。
> **机制零件全部验证**，含**新正结果：持续度门控 relay X 能终止一次持续发作（有界、causal、两 seed 复现）**。

- **允许写**：bounded-negative；"两档失效速率夹住需要的中间档"；"X 能终止一次持续发作（有界、causal、两 seed）"；"零件验证过"。
- **禁止写**：seizure lifecycle 达成；limit cycle；bistability；可恢复发作；干净统计恢复；把 q75 的 Z-only 振荡写成"发作生命周期"。

分层：engineering **green** / baseline **accepted** / sensor **accepted** / lifecycle candidate **未** / lifecycle confirmed **未**。

## 8. 下一 sprint（建议，未执行）

- 主杠杆：一个**介于 q75 与 q50 之间**的抑制失效速率（或让 Z 失效速率本身随状态可调），慢到留 ≥8s 间期 + 撑出不自灭的持续发作 + 事后可恢复。
- 备选：专门的 **thresholded global recovery**（§15 预留）——在 X 终止后主动把 Z 拉回安全邻域，补上"恢复"这条缺腿。
- 本轮受限于冻结的 {q75, q50} 标定；上面两条都需要放开 Z 标定，属下一 sprint 决策。

## 9. Provenance / 资源

- 代码：`src/topic4_mz_fcxr_lifecycle.py`（分类器/reducer/D_Z-D_X）；`scripts/run_topic4_mz_fcxr_lifecycle.py`（baseline/zonly/sensor/xcontrol/lifecycle + 看门狗）；
  `scripts/plot_topic4_mz_fcxr_lifecycle_diagnostic.py`；引擎改动仅 `src/snn_engine/mz_slow_vars.py`（非对称 X + x/y snapshot）。
- 测试：`tests/test_topic4_mz_fcxr_lifecycle.py`（17）+ `tests/test_mz_full_conductance_spatial_relay.py`（含 C1–C9 + snapshot x/y；63 relay+slowvars 全绿）+ engine-bless gate 绿。
- 结果：`baseline_contract_seed{1,3}.json`、`z_only_summary_*`、`sensor_separation_seed{1,3}_ty120.json`、`xcontrol_seed1.json`、`lifecycle_seed{1,3}_*`、`candidate_verdict.json`、`run_manifest.json`；图 `figures/lifecycle_diagnostic.{png,pdf}` + `figures/README.md`。
- 引擎纪律：6 blessed engine 未动（bless 未变）。速度 ~3.88ms/step；24s ~30min + build；raster ~14.3GB。T≥20s 严格 1 worker、OMP=1、setsid nohup + launcher.pid + sentinels；MemAvailable 全程 215-258GB、swap 增量~0、无 OOM；q50 加 2400s wall runaway kill-guard。
- 分支 `codex/topic4-mz-fcxr-lc1`，base `6819643` 之上 16 commits，**未 push / 未 merge**。
