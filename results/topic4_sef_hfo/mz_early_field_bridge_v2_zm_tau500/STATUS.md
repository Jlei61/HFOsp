# MZ early-field bridge V2（z+m，τ_adp=500 ms）— STATUS

> 版本：**V2 z+m observational bridge — supported**，2026-07-20，分支 `codex/topic4-mz-early-bridge-v2`（local-only）
> 设计合同：`docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md`
> 完整归档：`docs/archive/topic4/sef_hfo/mz_early_field_bridge_v2_zm_tau500_2026-07-20.md`

## 一句话结论

在同一块 E1146 模型底物上，把中间轨迹从"只有去抑制"换成"去抑制＋快速适应 m"之后，用安静态间期事件排出的**双向时序轴仍然预测**系统失控前的早期触点能量分布——三个噪声实现的方向无关 `maxAB` 中位数约 0.90（0.89/0.97/0.90），杆内随机重排全部过线，胜出方向一致，且 m 把失控时刻推后 1.5–3.7 s。这支持的是**"同一支架、状态依赖读出"的观测层可行性（z+m 下仍成立）**，不是发作复现、因果机制或患者队列结论。

## 当前证据

- 分母：一块 E1146 模型底物 × seeds 1/3/4，不是 3 个患者；V1+V2 六次运行不是 6 个独立样本。
- **preflight 三个 seed 全 PASS，t120 delta=0.0 ms**（native z+m 精确复现提交的 onset 12956.2/11008.6/12676.6；同时证明 use_m/eta_m/tau_adp 进入仿真 + LFP 记录器不扰动动力学）。
- 主统计：held-out slow-off 双向模板与 `t_recruit` 后 0–50 ms、`t120` 前 contact energy 的 mirror-invariant `maxAB`。三个 seed = 0.8915 / 0.9691 / 0.9043；within-shaft null p = 0.0027 / 0.0001 / 0.0017（**3/3 过线**），胜出方向都是 B_to_A。
- source-grid（补充）：0.573 / 0.498 / 0.535，toroidal p = 0.069 / 0.045 / 0.047（**2/3 过线**，seed1 marginal），方向不矛盾；只作补充诊断，不与 contact 合并。
- local participation（V2-only 真值）：median 0.287 / 0.171 / 0.355，per-contact 0–1 混合 → 高能量部分对应局部招募、部分远场；诚实的混合。
- core-exclusion：三个 seed `n_kept=15`（一个触点没删）→ **uninformative**，不下"不依赖 core"结论。

## V1 z-only ↔ V2 z+m 配对

seed1 0.945→0.891（Δ−0.054）；seed3 0.735(未过线)→0.969（Δ+0.234，转显著）；seed4 0.924→0.904（Δ−0.020）。方向全一致；onset 全推后（+3663/+1509/+2919 ms）。**V1 里偏弱的 seed3 在 V2 里最强** → V1 的弱-seed3 结果在 z+m 下不再出现。加了 m 主要改变**点火时刻**，双向轴的可预测性**保留**。

## 可以写 / 不可以写

可以写：固定患者布局支架上，held-out 双向间期时序轴在三个噪声实现里预测了失控前虚拟触点能量分布，加入 m 并推后点火后仍成立——观测层"同一支架、状态依赖读出"可行性桥。

不可以写：clinical seizure / clinical broadband power / complete seizure cycle / m 稳住发作或产生恢复态 / `z_i` 唯一生物机制 / 某端固定发作灶 / 间期事件因果触发失控 / 局部 z 图案因果 / contact 热点＝局部招募 / 结果不依赖 core。

## 完成层级

- engineering complete ✔ ｜ numerically eligible ✔ ｜ scientific observation ✔（**本轮上限**，桥 supported）
- **causal mechanism：未完成** — CRN replay 非 checkpoint 后真实状态分叉；区分整体去抑制增益 vs 局部 z 图案、以及事件是否因果触发失控，都需逐位可续跑 snapshot/resume + native/uniform/shuffle/reset z 对照（design §11.2），本轮明确未做。

## 图

- 主图（Figure-5 语法，seed1）：`../../paper-ready-figure/fig_mz_early_bridge_v2_zm_tau500/figures/fig_mz_early_bridge_v2_zm_tau500.{png,pdf}`
- 三 seed 配对诊断图（非主图）：同目录 `fig_mz_v1_v2_paired_diagnostic.{png,pdf}`
