# FCXR-LC5：U1a 自然进入源采集完成，负荷标定判定无解（CALIBRATION-BLOCKED）

日期：2026-08-11
状态：**U0–U1a 完成；U1b 判定 `U_SCALE_NOT_IDENTIFIABLE`；U2 未运行且已被 STOP 哨兵机械阻断。**

上游：
- spec `docs/superpowers/specs/2026-08-10-topic4-fcxr-lc5-per-cell-episode-pump-design.md`
- plan `docs/superpowers/plans/2026-08-10-topic4-fcxr-lc5-per-cell-episode-pump.md`
- 状态页 `results/topic4_sef_hfo/fcxr_lc5_episode_pump/STATUS.md`

## 0. 朴素话摘要

我们想问的是：给网络里每一个细胞装一个"自己攒、慢慢放"的疲劳刹车——放电越多攒得越多，攒到一定量就往外
拉电流把自己按下去，而且活动降下来之后这个记忆还能保持几秒——这样一个刹车，能不能让一场自己烧起来的
高活动自己停下来。

这一轮**还没测刹车**。刹车从头到尾是关的。这一轮做的是两件事。

第一件：先录一条干净的、能一模一样重放的"自己烧起来"的过程。结果是网络在没有任何外部踢一脚的情况下，
先出现 29 次短促的小事件（每次都自己回到静息），第 11 秒自己进入高活动，然后一直到第 22 秒记录结束
都没有回到静息。

第二件：检查刹车的力度该定多大。刹车的道理是：每个细胞按自己的放电攒负荷，负荷高了往外拉的电流就大；
要让负荷停在一个固定值上，这个细胞的放电率就不能超过某个上限。事先定好的规矩是"把参考时段里中等活跃的
那个细胞的负荷定在半满"。问题出在这儿：那个参考时段（第 12–15 秒）里细胞平均每秒放电 46 次，看着很正常；
但同一场高活动到第 20–22 秒已经每秒 447 次了——**是参考时段的 9.7 倍**。细胞有 2 毫秒的不应期，物理上
最快也就每秒 500 次，所以这时候几乎所有细胞都贴在自己的天花板上了。

后果是：按参考时段定的那个力度，在参考时段上只有 5 个细胞（三万二千分之五）攒不住负荷；但在刹车真正要
面对的晚期高活动上，**每一个细胞都攒不住**。想靠把"半满"调低一点绕过去也不行——参考时段要求低于 0.48，
晚期高活动要求低于 0.11，而且窗口越往后要求越低，**没有任何一个固定值同时管用**。根子在于：这个定标办法
默认了高活动期间放电率是稳住不动的，而这条轨迹在刹车本身的时间尺度（3–15 秒）里就涨了将近 10 倍。

所以现在的处境是：**定标这一步不成立**，不是逐细胞刹车这个想法被否掉了（它一次都没被打开过），也不是
发作生命周期成功了。

（内部归档代号：U0/U1a/U1b/U2、`ICTAL_LIKE_BOUNDED`、`r_hi_ref`、`q_i*`、`target_activation`、
`U_SCALE_NOT_IDENTIFIABLE`、`tau_ref_E`、`SAT_CEILING_FRAC`。）

## 1. 本次审阅修复了什么

审阅（用户提供）把完成度定在 40/100，判 P0 两条、P1 两条。逐条复算后全部成立，另查出三条审阅未覆盖的
合同缺口。修复覆盖全部 P0/P1 + 三条新缺口。

| 编号 | 问题 | 来源 | 处置 |
|---|---|---|---|
| P0-1 | 源轨迹不能包装为合格 ictal carrier | 审阅 | 新增 `--stage adjudicate` → `u1_carrier_adjudication.json` |
| P0-2 | 当前尺度合同不允许启动 U2 | 审阅 | 新增 `--stage scale` → `u1b_scale_lock/` + `U_SCALE_NOT_IDENTIFIABLE.json` + runner 机械阻断 |
| P0-3 | 尺度规则在本源上是**病态**的，非 5 细胞离群 | 本次新查 | 逐窗 `admissible_target_activation` 扫描写入尺度判决 |
| P0-4 | plan T3.2 prefix validation **从未执行** | 本次新查 | 新增 `--stage prefix` → `u1_prefix_validation.json` |
| P1-1 | spec §12 swap 守卫**从未实现**（只记录不判决） | 审阅指出违约，本次查明根因 | `resource_stop_reason()` + `_enforce_resource_guard()` + 7 条回归 |
| P1-2 | 未收口（STATUS / manifest / 图 / README / archive） | 审阅 | 全部补齐 |
| P1-3 | spec §12 要求的 `c_wall` **从未落盘**，前向预算与 12 h 上限冲突 | 本次新查 | 新增 `--stage manifest` → `run_manifest.json` |

## 2. U1a 采集事实（复算确认）

canonical substrate：`Z dynamic, H dynamic, X≡1 from t=0, M≡0 from t=0, U actuator off, no kick/reset/parameter step`。
connection_seed=1，noise_seed=401，dt=0.05 ms，T=22 s，40k，严格单 worker。

- onset = 11000 ms（`bout=[11,21]`，跑到记录末端无自主终止）
- onset 前 returning IED = 29；总 events = 82
- 预锁窗：baseline `[7000,11000)`，high reference `[12000,15000)`
- `E_hi` = 32000/32000（高参考窗内全体 E 细胞至少发过 1 spike）
- `r_hi_ref` = 53.0 Hz（独立复算一致）
- `clip_frac_max` = 0.0，`finite` = true
- wall = 25309.8 s，peak RSS = 8.38 GiB

## 3. P0-1：载体判读（`u1_carrier_adjudication.json`）

判据复用底物**已注册**的两条线，不新造阈值：
- 硬物理上限 = `1000/tau_ref_E` = 500 Hz（`tau_ref_E=2.0 ms`，`PP.build_substrate` 不覆盖该常数）
- 注册饱和线 = `PP.SAT_CEILING_FRAC × 1000/tau_ref_E` = 250 Hz。其自身合同（`src/sef_hfo_m4_metrics.py`
  `finite_energy_ok`）即"peak rate < sat_ceiling 才算 finite energy、未被钉在 runaway ceiling"——问题同层，
  按 CLAUDE.md §6.1 属可复用。

| 窗 (ms) | 逐细胞率 mean / median / max (Hz) | /500 | /250 | ≥250 占比 | ≥450 占比 | D | H | gErec_eff |
|---|---|---|---|---|---|---|---|---|
| baseline 7000–11000 | 4.7 / 3.0 / 27.8 | 0.009 | 0.019 | 0.000 | 0.000 | 0.066 | 0.29 | 0.00 |
| entry 11000–12000 | 13.2 / 9.0 / 55.0 | 0.026 | 0.053 | 0.000 | 0.000 | 0.087 | 0.99 | 0.15 |
| **high ref 12000–15000** | **46.3 / 53.0 / 110.3** | 0.093 | 0.185 | 0.000 | 0.000 | 0.276 | 3.86 | 0.52 |
| late 15000–18000 | 206.6 / 222.7 / 313.3 | 0.413 | 0.827 | 0.126 | 0.000 | 0.529 | 23.26 | 18.38 |
| late 18000–21000 | 433.3 / 435.3 / 476.3 | 0.867 | 1.733 | **1.000** | 0.013 | 0.741 | 28.95 | 19.03 |
| late 21000–22000 | 450.8 / 452.0 / 483.0 | 0.902 | 1.803 | **1.000** | **0.616** | 0.788 | 29.39 | 19.10 |

- late/high-ref 均值比 = **9.737**
- `gErec_eff` 末值 19.10 对 `rec_sat_g` = 21.6 → 平滑饱和已走完 **88.4%** 行程
- `verdict: source_type = escalating_saturated_source`

采集摘要自身已写 `refractory_ceiling_not_adjudicated_here: true`；本阶段即该缺失判读。
`lifecycle.label = ICTAL_LIKE_BOUNDED` 来自共享的 lifecycle 形状分类器，只表示"数值有限、无 clip、
跑到记录末端"，**不是载体验收**。已发布的采集 bundle 未被改写（plan T3.3 禁止事后手工重建事务），
判读作为同级新产物落盘。

## 4. P0-2 + P0-3：尺度判决（`u1b_scale_lock/u1b_scale_verdict.json`）

复用 `src.topic4_fcxr_lc5.lock_load_scales`（已存在，未重写）。
`a_load(tau) = target/(r_hi_ref·tau)` ⟹ `q_i* = target·r_i/r_hi_ref`，与 `tau_U` 无关，故三个 tau 共用一门。
`Phi(u*) = q_i*` 仅在 `q_i* < 1` 时可解。

门（两半，不可互换）：

| 项 | 值 | 门 | 判 |
|---|---:|---:|---|
| `q99(q_i*)` | 0.7704 | < 0.90 | PASS |
| `max(q_i*)` | 1.0409 | < 1 | **FAIL** |
| divergent cells | 5 / 32000 | = 0 | **FAIL** |
| `admissible` | false | — | **`U_SCALE_NOT_IDENTIFIABLE`** |

`a_load`（未采用，仅记录）：tau 3/8/15 s → 3.14465e-3 / 1.17925e-3 / 6.2893e-4。

**逐窗可用目标扫描（P0-3，本次新增）**——每窗"让每个细胞都还有有限平衡"的目标上确界：

| 窗 | max rate (Hz) | sup(target) | target=0.5 下发散占比 |
|---|---:|---:|---:|
| baseline | 27.8 | 1.9099 | 0.0000 |
| entry | 55.0 | 0.9636 | 0.0000 |
| **high ref（锁定窗）** | 110.3 | **0.4804** | 0.0002 |
| late 15–18 s | 313.3 | 0.1691 | 0.9991 |
| late 18–21 s | 476.3 | 0.1113 | 1.0000 |
| late 21–22 s | 483.0 | **0.1097** | 1.0000 |

结论：只看锁定窗，把 0.5 降到 0.48 即可过门；但执行器实际面对的晚期高态要求 < 0.11，且随窗右移仍在下降。
`q_i*` 是**平衡记账量**，前提是率在 `tau_U`（3–15 s）尺度上定常；本源的率在同一尺度内涨约 9.7 倍，
该前提不成立。因此这是**尺度定义在本源上病态**，不是 5 个离群细胞的问题。

明确禁止（已写入 JSON `forbidden_next_actions`）：调低 target 过门而不重锁设计、把发散细胞移出支持集、
用 q99 顶替 max 门、启动 U2。

### 4.1 发散细胞身份（`u1b_divergent_cell_audit.json`）

| cell | v_th | 位置 (mm) | baseline (Hz) | high (Hz) | q* |
|---|---:|---|---:|---:|---:|
| 10046 | 14.350 | (2.56, 9.03) | 26.75 | 110.33 | 1.0409 |
| 10241 | 14.653 | (15.41, 8.88) | 24.75 | 107.67 | 1.0157 |
| 14427 | 14.559 | (18.01, 8.64) | 25.75 | 108.33 | 1.0220 |
| 17406 | 14.997 | (16.58, 8.70) | 24.50 | 106.67 | 1.0063 |
| 29460 | 14.575 | (16.27, 8.05) | 27.75 | 108.67 | 1.0252 |

基线阈值 18.0；全片低阈值细胞 760/32000 = 1.9%。**5/5 发散细胞全部落在低阈值病灶斑块内 → 52.6 倍富集**，
两个病灶核各有代表（一个在 x≈2.6，四个在 x≈15–18）。低阈值场即本底物的病理本身（AGENTS.md
`v_th_per_neuron` 条目），删除这些细胞等于删除让片子能点火的机制。"drop the 5 cells" 这条路已关闭。

## 5. P0-4：仪器纯度补检（`u1_prefix_validation.json`）

plan §5 T3.2 要求把 capture 与 accepted no-pump reference 比 external input hash / first onset /
pre-onset ledger / rate prefix / Z-H traces，不合则记 `CAPTURE_CONTAMINATES_TRAJECTORY`。原 `stage_capture()`
**没有任何此类比较**，该门从未执行——而整个 bundle 作为 *canonical* 源的资格正建立在它之上。

补检设计（直接检验风险本体）：同一配置的第一秒跑两遍，A 臂挂全套仪器（sparse spike sink、exact input
hasher、vSEEG observer、recurrent drive observer），B 臂裸跑；再把 A 臂与**已发布** spike stream 的首秒逐
spike 比对。

- 配置 hash 与已发布采集一致：`c30a0143…`（不一致即拒绝运行）
- A 臂 / B 臂 exact state hash：`f5f00919…` == `f5f00919…` → **逐字节一致**
- A 臂首秒 139962 个 E spike 与已发布流首秒 **逐 spike 完全一致**
- `verdict: CAPTURE_DOES_NOT_CONTAMINATE_TRAJECTORY`

范围声明：本检验只证明仪器不扰动轨迹、且轨迹可精确重放；对高态是否为合格载体不作任何陈述。

## 6. P1-1：资源合同（`resource_log.jsonl` + `run_manifest.json`）

根因：`_append_resource()` 只写日志，spec §12 的 ≥256 / ≥512 MiB 门与 `RESOURCE_STOP` **从未实现**，
所以 U1a 全程没有守卫。

U1a 实测：swap 705.0 → 峰值 2001.5 MiB（**+1296.5 MiB**，为 512 MiB 终止线的 2.5 倍）。
归因：本 worker `self_peak_rss_gib` 全程恒定 6.79 GiB；`MemAvailable` 最低 143.7 GiB；
`sibling_topic4_python_count` 在 0–37 间震荡，峰值时刻为 37。→ 增量不归因于本次采集，**已得数据未受损**；
但合同当时确实未被执行，且 U2/U3 更长。

修复：`resource_stop_reason()` 纯函数（阈值判定 + 归因字段）+ `_enforce_resource_guard()`（写 `RESOURCE_STOP.json`
并 `SystemExit`）。终止行（DONE/FAILED）显式 `enforce=False`，避免在成果已发布后反而把结果毁掉。
`stage_capture` 的顺序本就是 `save_loop_state → rolling_checkpoint → _append_resource`，故守卫触发点在
checkpoint 之后，符合 spec "超时先写 rolling checkpoint 再终止"。
回归 7 条，含用 U1a 实测值回放的 `test_swap_guard_would_have_fired_on_the_observed_u1a_growth`。

## 7. P1-3：实测机器成本与前向预算（`run_manifest.json`）

spec §12 要求 U1a 后实测 `c_wall = T_wall/T_sim`，此前未落盘。

- `c_wall` = 25309.8 / 22.0 = **1150.4 s wall per s sim**
- 逐秒成本随率强烈上升：interictal 段约 120–240 s/sim-s；17–21 s 段 2715 / 4808 / 5170 / 4484 s/sim-s
- U2 单臂覆盖的 8 秒（onset+1s=12 s → 20 s）**实测 4.51 h**；9+1 臂串行 **≈45 h**
- 按 spec 自身公式 `1.5·c_wall·T_target`：
  - U2 单臂 → 3.83 h（12 h 上限内）
  - U3 单条 70 s → **33.6 h > 12 h 硬上限**；primary + sensitivity 共 **67.1 h**

⚠️ **U3 按现行 spec 写法在这个成本下不可执行**，需更便宜的底物、更短目标、或显式重锁上限。
⚠️ 上述 wall 数字混合了模型成本与机器争用（同期最多 37 个 sibling 进程），不是干净的单 worker 基准；
`run_manifest.json` 内已明写该限制。

## 8. 工程变更

`src/topic4_fcxr_lc5.py`（新增 3 个纯函数）：
`resource_stop_reason` / `refractory_ceiling_report`（可选 `sat_ceiling_hz`）/ `admissible_target_activation`。

`scripts/run_topic4_fcxr_lc5.py`：
- `_append_resource(enforce=)` + `_enforce_resource_guard()`
- 新增 stage：`prefix` / `adjudicate` / `scale` / `cells` / `manifest`
- `_assert_no_stop()`：`U_SCALE_NOT_IDENTIFIABLE.json` 存在时拒绝任何不在 `STOP_EXEMPT_STAGES` 内的 stage。
  argparse choices（`STAGES`）与豁免名单（`STOP_EXEMPT_STAGES`）**刻意分成两个字面量**——否则将来加 U2 时
  必须把它写进 choices 才能被选中，而那一次编辑会同时把它豁免掉，守卫被"添加该阶段"这个动作本身击穿。
  配 4 条回归，其中一条对真实落盘哨兵验证 `u2` 被拒

`scripts/plot_topic4_fcxr_lc5_closeout.py` + `figures/README.md`（中文逐图说明）。

`tests/test_topic4_fcxr_lc5_closeout.py`：25 条。
plan §12 测试矩阵 `test_topic4_mz_fcxr_pump / test_mz_slow_vars / test_topic4_fcxr_lc4* / test_topic4_fcxr_lc5*`
**212 passed**。

未改：六个 blessed engine 文件；`mz_slow_vars.py`；已发布的 `u1_capture/` bundle；用户文件
`scripts/nohup_subject_capture.sh`。

`u0_lineage_audit.json` 认证的是采集当时的代码（git `40a5765`）。收口 stage 是之后加的，因此
`--stage capture` 现在会按合同以 source-drift 报错——预期行为，非回归。

## 9. 允许 / 禁止措辞

允许：

> LC5 拿到了一条无 kick、29 次 returning IED 后自然进入的可精确重放源轨迹；该源随后升级为逼近
> refractory ceiling 的饱和高态。逐细胞 episode-load 尚未接受任何终止测试，且预注册的 load scaling
> 在这条源上不存在有限平衡，因此 LC5 当前是 **calibration-blocked**。

禁止：pump 机制阴性 / lifecycle / carrier / recovery established / 把 `ICTAL_LIKE_BOUNDED` 当载体验收 /
调低 target 后直接重启 U2 / 删除发散细胞 / 用 q99 顶替 max。
