# LIF-derived rate field —— Topic 4 主模型数学路线（理论收口，2026-06-03 晚）

> User 写了完整的理论重新推导（把 generic sigmoid rate field 升级为 LIF-derived rate field）。本文记录该推导，**并折入一处经验强制的更正**（near-critical → excitable）。逻辑链 "population transfer → gain → linear stability → finite-pulse validation" 不变，与 v0.2 plan 的 gain-closed + pulse-validated + control-disciplined 一致；只换 transfer function 的具体实现。**这是 Topic 4 主模型数学路线的更新（minimal surgery：换 transfer，不动整体逻辑）。**

## 0. 一句话

主模型从「低异质性阈值分布驱动的 sigmoid rate field（`F_eff`）」**升级为「LIF-derived near-threshold noisy rate field（`Φ_LIF(μ,σ)`）」**。低异质性仍是重要 biological motivation，但**不再作 Step 0 硬编码入口**；它后置为「LIF 参数分布收窄」，检验是否真的通过 `Φ_LIF`、gain、`λ(k)` 改变局部易受激性。

## 1. 三层模型（钉死，防混）

- **第一层 真实 LIF spiking network**：`τ_m dV_i/dt = -(V_i-E_L) + I_i(t)`，到阈值发 spike、reset、refractory。
- **第二层 LIF mean-field transfer**：一群 LIF 收到均值 μ、噪声 σ 的输入 → 平均发放率 `ν = Φ_LIF(μ,σ)`（Siegert first-passage 曲线）。这是 population input-output 曲线。
- **第三层 LIF-derived rate field**：空间每点 `τ_a^r ∂r_a/∂t = -r_a + Φ_a(μ_a(x,t), σ_a(x,t))`，a∈{E,I}。仍是 rate field，但 transfer 是 LIF 平均场推出的 `Φ_LIF`，不是人为 sigmoid。与 Brunel/Bachschmid-Romano 的空间结构化 LIF E-I + mean-field/susceptibility/plane-wave 线性稳定性精神一致。

## 2. `F_eff` 发生了什么

旧 `F_eff(h)=∫f(h-φ)p(φ)dφ` 与新 `Φ_LIF(μ,σ)` **功能位置相同**（input → population transfer → output rate），但：`F_eff` 是人为 sigmoid-family 平均；`Φ_LIF` 是从 noisy LIF first-passage 推出的发放率曲线。**`F_eff` 降级为 biological heterogeneity 的一种 optional coarse-graining，不再是 Step 0 主公式。** 若重新放入低异质性，写 LIF 版 population averaging：`Φ_eff_LIF(μ,σ;x)=∫Φ_LIF(μ,σ;θ)p(θ;x)dθ`，θ={V_th,V_reset,τ_m,g_L,C_m,adaptation,rheobase}——比只分布一个抽象阈值 φ 更合理地接 biophysical heterogeneity（Cell Reports 2022 路线）。

## 3. 为什么 sigmoid 卡死、LIF 能活（核心数学变化）

sigmoid `G(h)=F'(h)=βF(1-F/Fmax)`：低静息率 `F(h0)≈0 ⇒ G(h0)≈0` ⇒ 低 gain ⇒ 扰动传不动 ⇒ finite pulse 点不燃邻居。但间期 HFO 要「平时低活动 + 被扰动时敏感」，sigmoid 难两全。
LIF `ν=Φ_LIF(μ,σ)`：即使 ν0 低，只要处在**噪声驱动近阈区**，μ 稍增就显著提高 crossing probability ⇒ `ν0 low ⇏ G_μ low`。**经验证实**（preflight）：sigmoid 低率 gain≈0、最高 loop gain 1.26（只在饱和）；LIF 在 1–10Hz 低率处 loop gain 2–5。「transfer 的斜率救了模型」判断准确。

## 4. 自洽稳态（σ 也自洽）

`μ_a^0 = μ_a,ext + Σ_b ±J_ab K̄_ab r_b^0`；`(σ_a^0)^2 = σ_a,ext^2 + Σ_b J_ab^2 K̄_ab r_b^0`（抑制输入方差也是**正**贡献）；`r_a^0 = Φ_a(μ_a^0,σ_a^0)`。这是自洽非线性方程组，**必须 root-find（fsolve）**——阻尼定点迭代在高 loop gain 下不收敛、会给偏高的假工作点（本轮踩过这个坑）。**已验证**：fsolve 解复现 coworker1 的低 E 率 SI 态。

## 5. Gain（两个）

`G_a^μ = ∂Φ_a/∂μ_a|_0`、`G_a^σ = ∂Φ_a/∂σ_a|_0`。**主文走简化版**（固定 σ、只用 `G^μ`：`δr_a≈G_a^μ δμ_a`），清楚稳易 debug。**完整版**（`δr_a≈G_a^μ δμ_a + G_a^σ δσ_a`）更接近 Brunel（mean + variance susceptibility），作 sensitivity——见下 §更正 (2)，σ-dynamics 可能是速度-尺度张力的钥匙。

## 6. 线性化 → 色散

`τ_a^r ∂δr_a/∂t = -δr_a + G_a^μ δμ_a`（忽略 δσ）；`δμ_E = J_EE K_EE*δr_E - J_EI K_EI*δr_I`，I 类似。Fourier：得 `M(λ,k)`，`det[I - A(λ,k)]=0` → `λ_j(k)`。承重参数：`G_E,G_I,J_ab,K̂_ab(k),τ_ab,D_ab`——其中 `G_E,G_I` 由 transfer 给定，**不是自由调参**。

## 9. 慢 GABA / synaptic filter

`H_GABA(λ)=1/(1+λτ_GABA)`（或双指数）。`λ=iω` 时 H 是复数 → **频率依赖相位滞后** → 让 E-I 负反馈环 `E→I⊣E` 的特征根获非零虚部 `Im λ`，把单纯空间扩散/放大变成**振荡传播模式**。**经验证实**：把 GABA 衰减从 18ms 换成 2ms（其他不动）→ 振荡消失、回静态 k=0。这就是「慢抑制给出移动振荡的时间结构」的数学说法。

## 10. η_lin + 目标

`α_max(k)=max_j Re λ_j(k)`、`α_*=max_k α_max(k)`、`η_lin = -α_*`。`η_lin>0` 稳、`≈0+` 近临界、`<0` 失稳。

## ⚠️ 经验强制的更正（near-critical → excitable）：本轮最重要的一处

User 推导把 Step-0a 目标写成 **near-critical `η_lin≈0+` / finite-k Hopf**。**但 fsolve 后的真工作点不在那里**：
- **全程稳健 STABLE**（max Re λ≈−0.05、loop gain≈0.58 且随 drive **下降**、最不稳模 **k=0**）——**不近临界、无 finite-k Hopf**（之前报的 finite-k Hopf 是阻尼迭代假象）。
- 但 finite-pulse **self_limited_propagation 照样发生**，且波前推进**与幅度无关**（A=6→24 都 6.9mm）= **全或无**。

这个签名是**非线性可激介质（稳定静息 + 阈值 + 再生脉冲）**，**不是**近临界线性失稳。脉冲靠**瞬态**把局部推进高 G 区（μ→阈值）点燃，不靠坐在 η≈0+。所以：
> **Step 0a 目标应改为「稳健稳定但可激」（η>0，有限脉冲触发全或无传播响应），不是「近临界 η≈0+」。** 色散 / finite-k-Hopf 仍值得算（诊断），但**不是闸门**——finite-pulse 才是。`Φ_LIF→G→λ(k)→η_lin→finite-pulse` 链条对，只是**η_lin 没接近 0，这没问题**——是 excitability 而非 near-criticality 在扛事件。
>
> **（请 user 复核此更正：基于 fsolve 工作点稳健稳定 + 幅度无关全或无脉冲两条 committed 证据；色散用了静态 gain + 单极突触 + 固定 σ 近似，动态 LIF H(ω) 可能移动 η，但「真低率工作点不在 finite-k Hopf 上、脉冲是非线性可激」定性稳。）**

## 11. finite-pulse 仍必须做

线性稳定只答「无限小扰动增不增长」；真实 HFO 是有限幅事件。Step 0c 模拟 `r_E(x,0)=r_E^0 + A exp(-|x-x0|^2/2s_p^2)`，看 `P_ignite, R_spread, T_event, θ_prop, self-limited vs runaway`。LIF transfer 让传播可能（low baseline + high G + anisotropic K_EE → 沿轴再生）；recovery/慢抑制/adaptation 让它停（`q↑ ⇒ μ_E^eff↓ ⇒ G/drive↓ ⇒ 终止`）。

## 12. 重构后的 Step 0（0a–0e）+ 与已做工作的对应

- **0a 同质 transfer 屏**：找工作点 `r_E^0 low, r_I^0 reasonable, G_E high`，**（更正）+ 稳健稳定可激，不是 η≈0+**。✅ 已做（preflight + fsolve 工作点）。
- **0b 空间色散屏**：加 `K_ab(x)`、`K̂_ab(k)`，算 `λ(k)`，看 `k*≠0`（finite-k）。✅ 已做（**结果：真工作点 k*=0、稳健稳定**——所以本步是诊断不是闸门，见更正）。
- **0c finite-pulse 验证**：near-critical-but-stable→**(更正)稳定但可激**参数做 finite pulse；合格 = 点着 + 传播 + 自终止（只爆炸 / 怎么都不传播 = 不合格）。✅ 已做（self_limited_propagation 找到，mechanism-scale gate 已过）。
- **0d 各向异性控制**：加各向异性 `K_EE`，检 `θ_prop≈θ_EE`；**旋转连接轴 ⇒ θ_prop 旋转；旋转电极杆 ⇏ θ_prop 根本改变**。✅ **已做 PASS（2026-06-03）**：θ_prop 随 θ_EE 转 <0.1°（测 0/30/60/90/135），isotropic 对照无优势轴 ratio 1.00；framework 的承重判别指标通过。`scripts/sef_hfo_step0d_anisotropy_control.py`（commit c183ed6 + canonical 化 e95af61）。
- **0e heterogeneity patch 后置**：`Φ_a^eff(μ,σ;x)=∫Φ_a(μ,σ;θ)p_a(θ;x)dθ`，扫 `Var(θ)↓` 看是否 `G_E(x)↑ / η_lin(x)↓`，只在此方向成立才说「low heterogeneity 让该 patch 更易受激」。❌ 未做（计算较重：Φ_LIF 在多维 θ 分布上积分；scope 为后置层，不入 Step 0 核心）。

## 13. 机制链条（收口）

`Φ_LIF(μ,σ) → G_μ → M(λ,k) → λ(k) → η_lin（诊断）→ finite-pulse ignition（闸门）→ anisotropic propagation → self-limited HFO envelope`。**不是** `σ_φ↓ → F_eff → easy excitation`。

## 由此推导出的新 Step-0 工作（状态更新 2026-06-03：1/2/4 已做，3 已初探，0e deferred）

1. ✅ **0d 各向异性旋转控制（已做 PASS，commit c183ed6 + e95af61）**：模板方向随连接各向异性轴转、不随电极杆转（<0.1°）；isotropic+aligned-shaft 对照不过（Step-1 噪声下重跑见 `step1_noise_contract_2026-06-03.md` §6）。v0.2 plan 承重判据通过。
2. ✅ **把 LIF transfer 收进 canonical src 模块（已做，commit 46f9040 + e95af61）**：`Φ_LIF` 落到 canonical `src/sef_hfo_lif.py`（Siegert + fsolve `mean_field` + `lif_gains` + `integrate_lif_field`（可旋各向异性 + recovery）+ `classify_response`，TDD）；exploration 脚本不再是唯一实现。w_ee_mult 贯穿场 + mean_field 多初值取最低-nuE root（暴露 wEE×1.4 双稳）已加固。
3. **σ-dynamics + G^σ 作速度-尺度张力 sensitivity**（Step 1/2）：finite-pulse 期间局部 σ²∝r 变化大，固定-σ 近似忽略了它，而它直接调 G → 传播速度；σ-dynamics 是解开"波速太快/reach 太大"的候选。**已初探（2D Φ_LIF(μ,σ) lut + 动态 σ(x,t)，在病理增益 wEE×1.4 regime）：σ-dynamics ON 把事件时长拉长（110→151ms，更贴 envelope），但 reach 仍 ~填满网格、波速没明显降——所以 σ-dynamics 是真实的"时长"杠杆，但 NOT 速度-尺度张力的解药。reach/速度张力仍开放（按 mechanism-scale 验收 = Step 1/2 sensitivity；要解可能得动空间核尺度/连接 reach，或接受"事件填满 SOZ patch"的 mechanism-scale 口径）。**
4. ✅ **Step-0a 目标措辞改"稳定可激"（已折入）**：本文 §⚠️更正 + framework §6.5/banner 的 near-critical 语言已收窄。

（内部归档代号：Φ_LIF Siegert transfer、F_eff demote、G^μ/G^σ susceptibility、self-consistent (μ,σ) fsolve、M(λ,k) dispersion、η_lin、finite-k Hopf vs k=0、excitable-from-stable-rest vs near-critical、all-or-none amplitude-independent pulse、slow-GABA phase lag、anisotropy rotation control θ_prop≈θ_EE、heterogeneity-as-LIF-param-distribution、speed-scale tension、σ-dynamics）
