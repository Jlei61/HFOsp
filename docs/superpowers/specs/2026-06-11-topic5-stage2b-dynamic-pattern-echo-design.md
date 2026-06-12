# Topic 5 — Stage 2b Early-Ictal Dynamic Pattern Echo Design Spec (2026-06-11)

> **状态**：设计稿 **v1**（2026-06-11，user-directed pivot from Stage 2 onset-rank failure）。plan 待写。
> **Topic**：把"间期那条固定的高频传播通路（topic1 模板）"与"发作头 0–10s 的**动态激活模式**（不是首次越阈的逐触点 onset rank）"对齐——问"发作早期的激活/上升模式是否系统性地贴合间期模板的通道优先级"。
> **为什么 pivot（Stage 2 first-onset 仪器失败的诚实记录）**：Stage 2 用 band-power 变点（CUSUM）测逐触点首次招募时刻，detector repair 全扫（bias 0.5–2.0 × detrend{none,median,quantile} × {cusum,zcross}）**都不过门**。诊断确认 blocker 不是阈值标定，而是**最早被招募的触点在触点层面近乎同步、并列**——几十个触点在招募窗起点同帧越阈，"最早 3 个"在并列里随机挑，各特征挑各的（early_K_overlap 恒 0）；换 pass-1 onset 也救不了。**结论：发作早期在触点层面近乎同步，不能稳定定义"最早几个触点"。** 这**支持** Stage 1 谨慎口径（最稳的是病灶距离/早晚粗锚，不是具体路径重放），但**不是**"发作动态与间期模板无关"的负结论——只是"first-onset rank"这个量错了。Stage 2b 换量。详见 [[project_topic5_stage2_recruitment_2026-06-11]] + `results/topic5_ictal_recruitment/detector_sweep.csv`。
> **Owner**：topic5 拥有"真正发作 EEG 招募/动态层 + 间期↔发作桥接"（与 Topic 4 H5 不重复）。
> **前身**：Stage 2 spec `docs/superpowers/specs/2026-06-10-topic5-ictal-recruitment-stage2-design.md`（v2.1，仪器链路 + per-dataset montage + 缓存 + echo 复用，**全部继承**；只换"招募顺序量"这一层）。

---

## 0. 一个真正测什么的朴素话

每个病人有好几次发作。间期我们已知每个病人有一条形状固定、被高频事件反复扫到的传播通路（topic1"模板"——每个通道一个优先级排名）。Stage 2 想问"发作时谁先点亮、顺序像不像模板"，但发现**发作头一两秒里很多触点几乎同时亮**——"谁第一"测不准。

Stage 2b 换个问法，绕开"谁第一"：

> **发作头 0–10s 里，各通道的激活强弱 / 上升快慢这个整体模式，是不是和间期模板的通道优先级对得上？模板里排在前的通道，在发作早期是不是也更强、上升更快？**

具体三种朴素问法（互相印证）：
1. **模板对齐曲线**：每个时刻，把"此刻各通道的激活程度"和"间期模板的通道排名"算一个相关；得到一条 echo(t) 曲线，看 0–10s 里有没有系统性的正峰。——不依赖"谁第一"，依赖"整体哪些通道此刻更活跃"。
2. **上升快慢的潜伏期**：不用首次越阈，改用每个通道**上升最快的时刻**（或升到一半/八成的时刻）。同步起跳之后，传播差异体现在"谁升得更快/更早到峰"，比"谁先越阈"更能拆开。
3. **早期强度**：每个通道在 0–2s / 2–5s / 5–10s 的激活面积或斜率。问"模板里排在前的通道，是不是在早期窗里更强、更快"，而不是"是不是第一个"。

并列太多时，把同一杆 / 解剖区的触点聚合，先问"**区域**招募模式像不像模板"，再决定要不要回触点层面。

**揭示了什么（叙事）**：若这些动态量里有多条都显示"模板靠前的通道在发作早期系统性更强/更快"，就支持"间期刻板模板=被发作早期复用的网络骨架"；若都打平，则进一步坐实 Stage 1 的"共享粗锚为主"。

---

## 1. 一句话主张（locked framing）

在扩展的 epilepsiae(CAR) + yuquan(bipolar) 探索性 cohort 内，**发作头 0–10s 的动态激活模式（模板对齐曲线 echo(t) / 上升斜率潜伏期 / 早期强度）与该病人间期传播模板的相似度，系统性高于 within-seizure 通道身份打乱的零假设**（单侧），并在 within-shaft / anchor-matched null 下不完全消失。

- 判定语言只允许 "**像 / 不像 / 没看清**"；禁止 "predicts seizure" / "causes" / α-claim 升机制因果。
- **不再用 first-onset contact rank**（Stage 2 已证其在触点层面并列、测不稳）。改用 §3 的动态量。
- echo 统计复用 Stage 1 `src/topic5_echo_gate.py`；模板 = masked narrow（Main-A）为主、broad 为 §6 secondary。

---

## 2. 核心对象：Z(contact, time, feature) — 已建/已缓存

Stage 2 已经把每个发作的每个特征算成了逐触点逐帧的 robust-z 轨迹 `Z[feature][contact, frame]`（line-length / broadband / HFA / spectral-edge，+ ER held-out），并缓存到 `results/topic5_ictal_recruitment/sentinel_cache/`（raw feature traces + fs/channels/pre_sec/eeg_rel/montage/baseline-pool）。**Stage 2b 直接消费这个 Z**，不重读 EDF。

- 时间轴：帧中心相对临床 onset 的秒（`t_center = frame*hop + win/2 - pre`），hop=0.1s，per-feature win（HFA=0.5s 其余 1.0s）。
- 分析窗：early-ictal `[T0, T1]` = `[0, +10]s` rel 标注 onset（primary；sensitivity `[-2,+10]` / `[0,+5]`）。**不再需要 data-driven global onset 的 -2s 重测窄带**（那正是制造并列的根源）；Stage 2b 直接在 early-ictal 窗上算动态量。

---

## 3. 三族动态量（minimal sentinel set；互相印证）

> 所有"模板对齐"= Spearman(模板 rank, 动态量 over **common 通道**)，common = `template valid_mask==True` ∩ 该量非 NaN ∩ montage-aligned（per-dataset alias，§5）。模板 rank 方向：约定模板 rank 小=早/源（与 topic1 一致；实现期核对方向，§5）。

### 3.1 A — 模板对齐曲线 `echo_curve`（主量，绕开并列）

对每个特征 f、每个时刻 t（early-ictal 窗内逐帧或逐 0.5s 窗）：
- `activation_z[f][:, t]` = 此刻各通道的 robust-z。
- `dZdt[f][:, t]` = robust-z 的时间导数（平滑差分，§5）。
- `echo_act[f](t) = Spearman(template_rank, activation_z[f][:, t])`（模板靠前通道此刻是否更活跃；注意方向：模板 rank 小=源，activation 大=活跃 → 期望**负**相关，统一取 `-Spearman` 或翻转模板方向使"对齐"为正，实现期锁定，§5）。
- `echo_slope[f](t) = Spearman(template_rank, dZdt[f][:, t])`（模板靠前通道此刻是否升得更快）。

**判读量（per seizure, per feature）**：echo(t) 在 early-ictal 窗内的**峰值** `echo_peak`、**峰值时间** `t_peak_echo`、**窗内均值** `echo_mean`。主张 = 多特征的 echo(t) 在合理 early-ictal 时间出现**系统性正峰**（方向一致）。

### 3.2 B — 上升斜率潜伏期 `latency`（拆同步起跳后的传播差异）

每个通道、每个特征，在 early-ictal 窗内：
- `t_max_slope` = `dZdt` 最大的时刻（上升最快时刻）。
- `t50_rise` / `t80_rise` = robust-z 首次升到该通道窗内峰值 50% / 80% 的时刻。
- `t_peak` = robust-z 峰值时刻。

→ 每个量给一个逐触点的"潜伏期 rank"（小=早）→ `Spearman(template_rank, latency_rank)`（模板靠前通道是否更早到达上升峰/半高）。这比 first-crossing 更能在同步起跳后拆出传播顺序（峰/半高时刻不并列在窗起点）。

### 3.3 C — 早期强度 `ramp_strength`

每个通道、每个特征，分窗 `0–2s / 2–5s / 5–10s`：
- `AUC_w` = robust-z 在窗 w 的积分（早期累计激活）。
- `slope_w` = robust-z 在窗 w 的线性斜率。

→ `Spearman(template_rank, AUC_0_2)` 等（模板靠前通道是否在最早窗里更强/更陡）。

### 3.4 D — region / shaft 聚合（并列太多时的退路）

把 §3.1–3.3 的逐触点量按 **shaft（`parse_shaft`）/ 解剖 region** 聚合（median over contacts in group），得 region-level 量，对 region-level 模板优先级（模板 rank 的 region 聚合）做同样的 Spearman。先报 region-level echo（更稳），再回触点层面。

---

## 4. 判决合同（revised gate — 不再要求 early_K）

> Stage 2 的硬门 `feature_agreement_flag`（含 early_K_overlap≥0.3）**作废**（已证其卡在并列）。Stage 2b 改用以下门：

| 门 | 条件 |
|---|---|
| **方向一致** | §3 的多个动态量（A echo_act/echo_slope、B latency、C ramp）的 cohort 合并方向**同向为正**（模板对齐为正），不是单个量孤证 |
| **时间合理** | echo(t) 的系统性正峰落在 early-ictal 窗（`t_peak_echo ∈ [0, ~6]s`），不是窗尾偶发 |
| **null 不全消** | within-seizure channel-shuffle 下显著；且 **within-shaft 或 anchor-matched null 下不完全消失**（否则只是病灶距离粗锚，写"shared anchor"） |
| **construct validity** | 多个**独立**特征（line-length / broadband / HFA / spectral-edge）+ ER held-out 的动态量方向一致（替代失败的 early_K：现在看"整体动态模式"跨特征是否一致，而非"最早 3 个"） |
| **双数据集** | epi(CAR) + yuquan(bipolar) **都**能画出可解释的 echo(t) 曲线且方向一致（cross-dataset bridge 必须，sentinel 上先看） |

- **站住·动态 echo（含路径）**：上述全过 **AND** within-shaft 或 anchor-matched 仍正 → 发作早期动态系统性贴合模板具体通道优先级。
- **站住·稳定锚为主**：channel-shuffle 下正但 within-shaft/anchor-matched 打平 → shared ictal/interictal channel-priority anchor。
- **没看清 / 阴性**：动态量方向不一致或全打平 → 当前动态量也没看到 echo（与 Stage 1 粗锚口径一致，不升级为"无关"负结论）。
- 合并仍 subject-level 优先（Stage 1 `pool_echo_subject_level`），dataset 分层 sensitivity。

---

## 5. 复用 + 新代码

### 5.1 复用（不重造）

| 需要 | 复用 | 说明 |
|---|---|---|
| Z(contact,time,feature) 轨迹 + 缓存 | Stage 2 `_raw_traces`/`_z_from_traces` + `sentinel_cache/` | ✅ 直接消费，不重读 EDF |
| per-dataset montage + 对齐 | Stage 2 `ICTAL_REFERENCE`/`bipolar_alias_label`/`assert_channel_identity` | ✅ epi=CAR / yuquan=bipolar；模板 rank 方向核对 |
| masked narrow 模板（Main-A）| Stage 2 `_load_masked_template`（`rank_displacement` pairs[0]）| ✅ |
| echo 统计 + null（channel/within-shaft/anchor-matched）+ subject pooling | `src/topic5_echo_gate.py`（Stage 1）| ✅ 喂动态量 rank 而非 onset rank |
| shaft 解析（region 聚合）| `src.propagation_skeleton_geometry.parse_shaft` | ✅ §3.4 |

### 5.2 新代码

- **新建 `src/topic5_dynamic_echo.py`（纯数学，no I/O）**：
  - `activation_and_slope(z_trace, *, hop)` → `(activation_z, dZdt)`（平滑差分，方向锁）。
  - `echo_curve(template_rank, z_by_t, *, mode)` → `echo(t)` 数组（mode ∈ {activation, slope}），+ `echo_peak/t_peak/echo_mean`（early-ictal 窗内）。
  - `slope_latencies(z_trace, *, window, hop)` → `{t_max_slope, t50_rise, t80_rise, t_peak}` per contact。
  - `ramp_strength(z_trace, *, windows, hop)` → `{AUC_w, slope_w}` per contact。
  - `region_aggregate(per_contact, shafts)` → region-level（median）。
  - 方向约定 + common-channel masking helper（模板 valid_mask ∩ 非 NaN）。
- **新建 `scripts/run_topic5_dynamic_echo.py`**：`sentinel`（从 `sentinel_cache/` 算 §3 三族量 + echo(t) + null + 出曲线图）/ `per-subject` / `cohort`。**先 sentinel**，过门再 cohort。
- **新建 `scripts/plot_topic5_dynamic_echo.py`**：echo(t) 曲线（每特征一条 + null 带）、latency/ramp 的模板对齐散点、region-level 对照；paper-grade 自洽 + 中文 README。
- TDD（`tests/test_topic5_dynamic_echo.py`）：合成"模板靠前通道早期更强/升更快"的 Z → echo(t) 出正峰；合成"同步并列但无优先级" → echo(t) 平；slope-latency 在同步起跳+不同上升率下拆出顺序；region 聚合正确；方向约定回归（模板 rank 小=源 → 对齐为正）。

### 5.3 输出

```
results/topic5_dynamic_echo/
├── sentinel/                 # 每 sentinel seizure: echo_curve.json + echo(t) 曲线 PNG（A/B/C/region）
├── per_subject/<ds>_<sid>.json
├── cohort_dynamic_echo_summary.json   # subject-level verdict（§4）+ 每 null + dataset 分层 + 方向一致性
└── figures/ (README.md 中文)
```

---

## 6. Secondary line — broad / large lagPat（Stage 2b-2）

user 次线：发作早期参与的网络比 topic1 高-HI-index 窄模板大。所以 **secondary**：
- 模板换 broad lagPat（`results/lagpat_broad/` Yuquan 已封板；epi broad 未封板 → 只 sensitivity）。
- 在 early-ictal 窗内提取峰 / 尖波 / HFA burst / ramp peak，构 early-ictal lagPat-like pattern，与 interictal **broad** 模板做 §3 的动态相似度。
- 角色：扩展分析（更大网络是否更回声），**不替代** narrow Main-A；带 broad-unsealed caveat。

---

## 7. Caveats & NOT-DO

1. **Stage 2 first-onset 失败已记录**：发作早期触点层面近乎同步 → 不能稳定定义"最早几个触点"。Stage 2b 换动态量。
2. **不是负结论**：Stage 2 失败 ≠ 发作动态与模板无关；只是 onset-rank 量错。Stage 1 谨慎口径（粗锚为主）仍是最稳。
3. **方向约定是合同**：模板 rank 方向 vs activation/slope 方向必须实现期核对并 TDD 锁（否则 echo 符号反）。
4. **exploratory**：sensitivity（窗 / 特征 / null）+ user 视觉巡视 echo(t) 曲线全过前不写 paper-level claim。
5. **双数据集必须都过**（epi+yuquan），montage per-dataset。
6. **不进 cohort 直到 sentinel echo(t) 曲线 epi+yuquan 都可解释且方向一致**（staged gate）。
- NOT-DO：不用 first-onset contact rank；不重读 EDF（用缓存）；不在 within-subject 写 α-claim；broad 未封板不进 Main 估计。

---

## 8. Staged execution

1. 写 spec（本文件）→ user review。
2. 写 plan（TDD）→ user review。
3. 实现 `src/topic5_dynamic_echo.py` + runner（TDD 绿）。
4. **sentinel**（从已有 `sentinel_cache/` 的 epi 1146:2/5 + yuquan litengsheng:0，必要时多缓存几个）：算 §3 三族量 + echo(t) + null，出曲线图，**人工目视** echo(t) 是否有系统正峰、epi+yuquan 是否方向一致。
5. sentinel 过门 → per-subject + cohort（§4 verdict）。
6. figures + archive doc（`docs/archive/topic5/dynamic_echo/`）→ sensitivity 全过后回链主文档。

---

## 9. 来源 / 关联

- `docs/superpowers/specs/2026-06-10-topic5-ictal-recruitment-stage2-design.md`（Stage 2 仪器链路 + montage + 缓存，继承）
- `results/topic5_ictal_recruitment/detector_sweep.csv`（Stage 2 onset-rank 失败证据）
- `src/topic5_echo_gate.py`（echo 统计 + null，复用）
- `results/topic5_ictal_recruitment/sentinel_cache/`（Z 缓存，复用）
- `results/lagpat_broad/`（broad 模板，secondary §6）
- memory [[project_topic5_stage2_recruitment_2026-06-11]]、[[project_topic5_echo_gate_2026-06-08]]
