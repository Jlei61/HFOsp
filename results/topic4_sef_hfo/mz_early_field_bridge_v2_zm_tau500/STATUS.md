# MZ early-field bridge V2（z+m，τ_adp=500 ms）— STATUS

> 版本：**V2 z+m observational bridge — ACCEPTED / FROZEN 2026-07-20**（(A) 长轴在失控前被调用 supported / (B) 模板超越几何轴未确立），分支 `codex/topic4-mz-early-bridge-v2`（local-only）
> 设计合同：`docs/superpowers/specs/2026-07-19-topic4-mz-early-field-bridge-design.md`
> 完整归档：`docs/archive/topic4/sef_hfo/mz_early_field_bridge_v2_zm_tau500_2026-07-20.md`

## 一句话结论（分两层；见 archive §10 长轴几何对照）

在同一块 E1146 模型底物上，把中间轨迹从"只有去抑制"换成"去抑制＋快速适应 m"之后，系统失控前 0–50 ms 的早期触点能量场**沿固定长轴组织、并在失控前被调用**——而且**单凭触点的固定长轴坐标就已能显著预测这个能量场（3/3，p<0.01）**；安静期时序模板几乎就是这条长轴（相关 −0.95～−0.98），只在长轴之上多贡献 maxAB +0.10～0.13（控制长轴后残余关联全样本 2/3 显著，但 **leave-one-contact-out 后只 1/3 稳健**——seed3；seed1 单触点驱动）。所以安全结论分两层：**(A) 固定病理长轴在失控前被调用 = 3/3 成立、LOO 稳健；(B) 间期细粒度时序超越几何轴的额外信息 = LOO 后只 1/3 seed 稳健、未确立（至多单-seed 线索）。** 且这是**早期（0–50 ms）现象**——到 50–100 ms 减弱。m 主要推后失控时刻（+1.5～3.7 s），但也改变招募/动态范围/跨尺度表达（不是"只改点火时间"）。这是观测层可行性（一块底物×3 seed），不是发作复现、因果或队列结论。

## 当前证据

- 分母：一块 E1146 模型底物 × seeds 1/3/4，不是 3 个患者；V1+V2 六次运行不是 6 个独立样本。
- **preflight 三个 seed 全 PASS，t120 delta=0.0 ms**（native z+m 精确复现提交的 onset 12956.2/11008.6/12676.6；同时证明 use_m/eta_m/tau_adp 进入仿真 + LFP 记录器不扰动动力学）。
- 主统计：held-out slow-off 双向模板与 `t_recruit` 后 0–50 ms、`t120` 前 contact energy 的 mirror-invariant `maxAB`。三个 seed = 0.8915 / 0.9691 / 0.9043；within-shaft null p = 0.0027 / 0.0001 / 0.0017（**3/3 过线**），胜出方向都是 B_to_A。
- **长轴几何对照（archive §10.1，最重要）**：单凭触点长轴坐标 maxAB = 0.789/0.839/0.779（3/3，p<0.01）；模板与长轴相关 −0.95～−0.98、增量只 +0.10～0.13；控制长轴后残余 partial 全样本 2/3 显著（seed1 p=0.032、seed3 p=0.001、seed4 p=0.078），**但 LOO 后只 1/3 稳健**（seed3 最坏 p=0.022；seed1 单触点驱动、掉一个即 p=0.203）。→ (A) 长轴被调用=3/3 稳健；(B) 模板超越几何轴 **LOO 后只 1/3、未确立**。
- **时间局限（§10.2）**：0–25 / 25–50 ms 强，**50–100 ms 减弱**（p 0.133 / 0.054 / 0.013）；早期 0–50 ms 现象。
- source-grid（补充）：0.573 / 0.498 / 0.535，toroidal p 0.069 / 0.045 / 0.047（2/3 过线，seed1 marginal）；**seed1 名义方向 A→B ≠ contact B→A，且 source maxAB 三个全部相对 V1 下降** → 无稳定跨尺度方向一致。
- local participation（V2-only 真值）：**energy-participation Spearman 0.90 / 0.91 / 0.94**，热触点参与 ≈0.89–1.0、冷触点 ≈0（median 只因冷热各半被拉中）→ 高能量触点确有明显局部招募，但沿轴共变、**不能证明局部独占**（既非"热点=局部源"、也非"混合/不支持"）。
- held-out 不对称（§10.3）：seed1 A→B held-out med 0.361（含负分）vs B→A 0.995；三个 V2 都 B→A 胜出——只验证 B→A 分支，**winner 一致不作证据**。
- core-exclusion：三个 seed `n_kept=15` → **uninformative**，不下"不依赖 core"结论。

## V1 z-only ↔ V2 z+m 配对

seed1 0.945→0.891（Δ−0.054）；seed3 0.735(未过线)→0.969（Δ+0.234，转显著）；seed4 0.924→0.904（Δ−0.020）。onset 全推后（+3663/+1509/+2919 ms）。**V1 里偏弱的 seed3 在 V2 里最强**。但 m **不只推后点火**：它同时改变招募规模（seed3 recruited 0→5）、动态范围（seed3 dyn_range 3.88→22.06）、跨尺度表达（source maxAB 三个全部下降）。准确说法：m 推后点火**并保留最早 0–50 ms 的 contact 级轴匹配**，同时改变早期场的动态范围/招募/跨尺度——不是"只改点火时间"。三个都 B→A 胜出、A→B held-out 弱 → 只验证 B→A，winner 一致不作证据。

## 可以写 / 不可以写

可以写：固定患者布局支架上，失控前早期虚拟触点能量场**沿固定长轴组织、并在失控前被调用**（单凭长轴坐标即可预测，3/3；加入 m 推后点火后仍成立）——观测层"同一支架、状态依赖读出"可行性桥。

不可以写：clinical seizure / clinical broadband power / complete seizure cycle / m 稳住发作或产生恢复态 / m "只改点火时间" / `z_i` 唯一生物机制 / 某端固定发作灶 / 间期事件因果触发失控 / 局部 z 图案因果 / contact 热点＝局部独占源 / 结果不依赖 core / **间期模板超越几何长轴提供预测信息**（LOO 后只 1/3 seed 稳健、未确立）/ **双向轴都被稳定使用**（只 B→A）/ source 跨尺度方向一致。

## 完成层级

- engineering complete ✔ ｜ numerically eligible ✔ ｜ scientific observation ✔（**本轮上限**）：桥 **(A) 层 supported**（固定长轴在失控前被调用，3/3、LOO 稳健）；**(B) 层未确立**（间期模板超越几何轴 LOO 后只 1/3 稳健）；早期 0–50 ms 现象
- **causal mechanism：未完成** — CRN replay 非 checkpoint 后真实状态分叉；区分整体去抑制增益 vs 局部 z 图案、以及事件是否因果触发失控，都需逐位可续跑 snapshot/resume + native/uniform/shuffle/reset z 对照（design §11.2），本轮明确未做。

## 图

- 主图（Figure-5 语法，seed1）：`../../paper-ready-figure/fig_mz_early_bridge_v2_zm_tau500/figures/fig_mz_early_bridge_v2_zm_tau500.{png,pdf}`
- 三 seed 配对诊断图（非主图）：同目录 `fig_mz_v1_v2_paired_diagnostic.{png,pdf}`
