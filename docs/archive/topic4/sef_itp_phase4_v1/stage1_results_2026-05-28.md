# SEF-ITP Phase 4 Stage 1 Results — Single-Node HR

> 状态：**PASS**（Stage 1 exit contract 满足）
> 日期：2026-05-28
> 上游 spec：`docs/superpowers/specs/2026-05-27-sef-itp-phase4-v1-design.md` v0.2
> 上游 plan：`docs/superpowers/plans/2026-05-27-sef-itp-phase4-stage1-hr-single-node.md` v3.3
> Framework banner：v1.0.7 → v1.0.8（commit c391806）

## 一句话朴素话（测了什么 / 怎么测的 / 揭示了什么）

**测了什么**：单个 Hindmarsh-Rose 神经元节点，在三个旋钮（输入电流 I、慢变量速率 r、
噪声强度 σ）上各扫一遍，找一个"平时安安静静、偶尔被噪声推一下打一个短暂的电活动、
然后又回到安静"的工作点（excitable regime）。这个工作点是后面 2D 网格里每个节点的
基础动力学参数。

**怎么测的**：每个 (I, r, σ) 组合跑一条 500 时间单位的轨迹（先 burn-in 100 单位丢掉
初始松弛瞬态），数它打了几次"事件"（spike-level 越阈），按事件数和间隔归类成
silent / excitable / repetitive-burst / unstable 四类。然后挑一个满足"σ=0 时安静 +
σ>0 时被噪声触发 + 噪声上下浮动 ±50% 仍然 excitable"的点。

**揭示了什么**：在 I ∈ [−1.5, 1.0]、σ ∈ {0, 0.2, 0.4, 0.6} 的网格上，excitable 工作区
确实存在且行为符合可激介质的标准图景——σ=0 时全网格安静，噪声越大可激区从高 I 向低 I
扩展。选中工作点 **(I*=1.0, r*=0.006, σ*=0.4)**，σ=0 安静、σ=0.2/0.6 两侧都 excitable，
噪声鲁棒。这说明 HR 单节点可以承担间期事件模型里"静息+偶发触发"的节点角色，Stage 2 可以
在此基础上搭 2D 网格。

## Stage 1 退出契约结果

- [x] 存在 excitable 参数子带：**PASS**（216 cell 中 59 个 excitable）
- [x] noise amplitude ±50% regime 不漂：**PASS**（baseline σ*=0.4，lower=0.2/upper=0.6 都 excitable）
- [x] 选中 baseline (I*, r*, σ*) = **(1.0, 0.006, 0.4)**
- [x] TDD 测试全 GREEN：**49 non-slow PASS**（11 hr_core + 6 ou_noise + 16 hr_dynamics + 11 hr_sweep + 4 hr_viz + 1 cli help）+ slow：3 hr_dynamics_integration（2 PASS + 1 XFAIL 经验边界）+ 1 cli exit-1 PASS
- [x] CLI exit code: **0**（baseline found + regime_map ok）

## Regime distribution（smoke config, 216 cells）

```
silent              148
excitable            59
repetitive-burst      9
unstable              0
```

## Artifacts

`results/topic4_sef_itp/phase4_modeling/stage1_hr_single/`
- `regime_map.png` — 三参数 regime 热图（已目视：σ=0 全灰、噪声增大绿区扩展、标准可激介质图景）
- `phase_portraits/baseline.png` — 静息 + 噪声触发大环（excitable 签名，已目视）
- `phase_portraits/baseline_zero_noise.png` — 纯静息无大环（已目视，确认 σ=0 silent）
- `phase_portraits/baseline_high_noise.png` — 2× 噪声下更频繁但仍离散事件
- `baseline.json` — (I*, r*, σ*) + 噪声鲁棒窗口
- `regime_summary.json` — 配置 + 计数 + exit-contract 标志
- `sweep_results.json` — per-cell 明细
- `README.md` — 中文逐图说明

## 关键过程发现：两轮 observation-layer 重标定（2026-05-28）

Stage 1 在 Task 4 之后、Task 5 之前的 sanity probe 暴露了一个会让 exit contract
**不可能满足**的问题：framework 时拍脑袋猜的检测阈值和扫描范围都和 HR 真实时间尺度
不匹配。两轮修复（都是 observation/experiment-design 层，不是结果调参）：

1. **detect_bursts 重标定**（commit 5f9450c）：旧默认 `min_burst_duration=5.0, bridge_gap=2.0,
   x_threshold=1.0` 假设"1 HR 单位≈1ms、burst≥5ms"。实测 HR fast spike 在 x>1.0 之上只有
   ~0.85 单位宽、intra-burst ISI~15 >> bridge_gap=2.0 → **全网格返回 0 事件 → 永远 silent**。
   改 spike-level：`x_threshold=0.0, min_burst_duration=0.3, bridge_gap=1.0`。检测单位 =
   spike-level excursion（见下方 caveat 3）。

2. **CLI 扫描范围重标定**（commit 76448c6, plan v3.3）：旧范围 I≤0.0、σ≤0.15 整个落在
   silent 区。改 I∈[−1.5,1.0]、σ 到 0.6，让网格**覆盖**实际存在的 excitable 带。picker 仍
   按客观判据选 baseline，没有 cherry-pick。

## Caveats（operational，写给 Stage 2+ reader）

1. **`fastmath=True`**：HR core / OU / trajectory 的 numba JIT 都开了 fastmath，允许编译器
   重排浮点运算。单步 JIT-vs-Python 1e-10 一致性有测试守着，但长轨迹上 JIT 与 Python 参考
   会因为混沌放大而逐渐分叉——这是预期的，不是 bug。若将来单步一致性测试偶发脆，先怀疑 fastmath。

2. **`n_steps = int(T/dt)` 截断**：非整除的 (T, dt) 会截掉零头。当前所有默认 (T, dt) 都整除，
   不影响；将来用奇怪的 T/dt 组合做 sensitivity 时注意。

3. **"excitable" 标签是 operational 不是动力系统严格分类**：本 Stage 的 regime classifier 按
   "事件数 + 事件间隔"打标签，单个短事件在长 T 里就算 excitable。这对 Stage 1 选 baseline 的
   目的是对的（picker 还会用 σ=0-silent 交叉校验），但**不**等于动力系统文献里严格的
   excitable 分类（这里没做严格分岔分析）。Stage 2+ 不要据此 over-claim。另外检测单位是
   spike-level excursion 而非 multi-spike burst envelope——burst-vs-spike 的区分要到 Stage 2-3
   讲传播顺序/参与时才 load-bearing，届时重新评估。

## 下一步

Stage 1 PASS → Stage 2 plan 立项（`docs/superpowers/plans/2026-05-XX-sef-itp-phase4-stage2-2d-homogeneous.md`）。
Stage 2 的每个节点用 baseline (I*=1.0, r*=0.006, σ*=0.4) 作基础动力学，加 2D anisotropic
diffusion 扫 D_x/D_y ∈ {1,2,3,5}，看均质网格上的 regime map。
