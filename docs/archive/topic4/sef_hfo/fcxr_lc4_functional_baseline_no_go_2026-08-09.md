# FCXR-LC4：协同终止器功能基线门（F0 bounded-negative）

日期：2026-08-09

分支：`codex/topic4-fcxr-lc3`

设计：`docs/superpowers/specs/2026-08-09-topic4-fcxr-lc4-functional-selectivity-design.md`

结果根：`results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/lc4_lifecycle_gate/`
机器判决：`NO_BASELINE_PRESERVING_HILL_CANDIDATE`

## 0. 白话结论

这轮想解决的是 LC4 已经暴露出的矛盾：刹车若足以终止已建立的高态，就会在间期提前漏出，把原本用于累积 D/Z 的稀疏 IED 压没。新方案不改负荷变量，只把执行器改成更陡的协同 Hill 曲线，希望它在间期近乎关闭、在发作期仍给出与已验收 n=4 终止臂相同的剂量。

结果只走到第一关。n=6 和 n=8 的实际间期最大电流分别只有 recurrent scale 的 **0.00498%** 和 **0.000889%**，都远低于 0.1% 泄漏上限；网络也都有限、零 clip、没有误入持续高态。可是功能统计没有一起保住：同噪声 control 在计分的 10 秒内有 8 个 returning IED，n=6 只剩 5 个且 IEI 更不规则，n=8 则增到 11 个。两条曲线朝相反方向越过同一条事件率门，因此没有候选获准进入 frozen-D onset surface。

最安全的解释是：**把平均泄漏电流做小，并不足以让间期事件生成邻域保持不变。** 这块噪声驱动、混沌的网络对极小但状态依赖的逐细胞反馈仍很敏感；n 改变的不只是剂量，还会重排哪些细胞在何时跨阈。后一句是机制推断，不是已完成的因果分解。

## 1. 测了什么

冻结上游 RC1、患者特异 E-to-E 轴、H1 carrier 点、D/Z/X 标定和所有事件判据。只比较三个同底座、同 connection seed=1、同 noise seed=401、同 `D_healthy=0`、relay=1 的 12 秒无 kick 轨迹：

1. actuator off 配对 control；
2. Hill n=6，慢关闭；
3. Hill n=8，慢关闭。

两条候选的 `K=45.5601`、`tau_m=1000 ms`、`tau_on=100 ms`、`tau_off=10000 ms` 不变，并把 ictal 电流严格配平到 n=4 已执行终止臂的 **44.86194**。前 2 秒 burn-in，后 10 秒按 returning-event rate、IEI CV、duration、participation 和数值安全联合验收。

## 2. F0 完整结果

| arm | returning IED | event rate | IEI CV | duration | participation | max current / recurrent | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| control | 8 | 0.80/s | 0.605 | 9 ms | 0.04544 | 0 | reference |
| Hill n=6 | 5 | 0.50/s | 1.044 | 8 ms | 0.04556 | 0.00498% | FAIL |
| Hill n=8 | 11 | 1.10/s | 0.556 | 11 ms | 0.04684 | 0.000889% | FAIL |

相对 control 的硬门：

- n=6：event-rate ratio=0.625（失败，门 `[0.80,1.25]`）；IEI-CV ratio=1.724（失败，门 `[2/3,1.5]`）；duration 和 participation 通过。
- n=8：event-rate ratio=1.375（失败）；IEI CV、duration 和 participation 通过。
- 两候选均：至少 3 个 returning IED、无 sustained bout、finite、zero clip、`tau_eff_min≈0.274 ms`、current leakage 通过。

因此这是一个**功能基线 bounded-negative**，不是数值失败，也不是执行器在间期误点燃高态。

## 3. 没有测什么

F0 失败后，两个 detached autopilot 按预注册规则写入 STOP sentinel：

- F1 `D_healthy/D10/D30/D50` frozen-D onset surface：未运行；
- F2 70 秒无 kick 动态 Z/H/X 轨迹：未运行；
- exact final-D 12 秒 returning-IED continuation：未运行。

因此本轮无权讨论新的 onset bracket、相图闭环、autonomous offset、postictal suppression、returning IED recovery 或完整 seizure lifecycle。上一阶段从真实高态 fork 得到的终止 authority 仍成立，但不能与本轮三个静息起跑臂拼接成生命周期。

## 4. 允许和禁止的科学措辞

允许写：

- 在锁定 `K/tau/dose` 下，n=6 与 n=8 都把间期执行器电流压到 0.005% 以下且数值安全；
- 这仍不足以同时保住配对的 returning-IED event rate 和 IEI statistics；
- 锁定的平滑 Hill 家族没有通过功能基线门。

禁止写：

- cooperative termination 被否定；
- D/Z 不再能进入发作；
- frozen-D onset surface 消失；
- lifecycle 或 recovery 失败（这些阶段根本未运行）；
- n=6/n=8 朝相反方向变化已证明某个具体单细胞机制。

## 5. 下一步边界

不能继续加大 n、延长 F0 或从 n=8 的“只差 10%”救援。当前 spec 已明确：若两候选都失败，应停止这个无死区的平滑 Hill 家族。下一设计若继续，最小结构变化应是一个**真正的低负荷 dead zone**，使间期执行器严格为零而不是仅仅很小；先重复同一配对 F0，再谈 D/Z onset。dead-zone 形状、阈值来源和坏数据回归需要重新预注册，不能从本轮 n=6/n=8 结果反选。

## 6. 图、工程与资源

- 图：`figures/lc4_functional_baseline_gate.{png,pdf}`；中文说明 `figures/README.md`。图只画实际测得的 F0，不为 F1/F2 造占位曲线。
- 判决：`baseline_verdict.json`；三个 run 的 JSON/NPZ 在 `runs/`。
- 回归：LC2/LC3/LC4/slow-variable 定向集合 **363 passed**。
- blessed engine：6/6 live sha256 与冻结记录一致；`mz_slow_vars.py` 是本机制模块，已另行锁哈希。
- 资源：严格单个 40k worker；峰值 self RSS **12.984 GiB**；最低记录 MemAvailable **191.35 GiB**；swap delta **0 MiB**；无残留本任务进程，未触碰 sibling worktree。
- 运行方式：F0、F0-to-F1、F1-to-F2 均为 `setsid nohup`，带 PID、stage lock、RUNNING/DONE/STOP sentinel 和 source hash lock。F0 NO-GO 后两个接力器均正常停止。

分层口径：engineering **green** · F0 functional baseline **bounded-negative** · F1 **not run** · F2 **not run** · lifecycle **not established**。
