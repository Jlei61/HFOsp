# Stage 0C 简报

## 结论

本轮是 **INCONCLUSIVE**，不是 GO，也不是 clean no-go。M4 的 `rE_fast → mu_G → S_G` 两级动态除法池在当前 LUT 实现的锁定网格内呈现重复有界活动，但所有达到候选门的轨迹都越过现有 LIF transfer LUT 的低输入边界，因此不能把它们验收成有限快态或 limit cycle。

## 实际看到什么

- `alpha_G <= 8` 只有 low / saturation 两类；`alpha_G >= 12` 出现 314 条 long transient、111 条 bounded-indeterminate 和 23 条 non-exact pre-audit oscillatory candidates。
- 最强点是 `z=0.80, alpha_G=12`：12 个不同 on-manifold / synaptic-history / pool-history / root-perturbation 初态给出一致的约 2.079 Hz 重复活动，tail mean 约 9.10 Hz、逐步 tail peak 约 99.56 Hz。
- 该点 tail 约 9.1% Euler states 的输入均值低于 LUT 下界；focused replay 中 `mu_E` 约到 -161 mV、`mu_I` 约到 -86 mV，而 sigma 仍在 LUT 支持内。按预注册 any-clip 规则，这 23 条全部标为 `audit_invalid_candidate`，所以 confirm=0、ablation=0。
- `alpha_G=0` 对 Stage0B 的 root、stability class、231 条共同 fork 首六维全部复刻；正反向 continuation 在 189/189 点 root-set 一致；所有 alpha 在 `z=1` 保留未裁剪稳定低态。

## 判断边界

当前安全说法是：**动态 recurrent-gain feedback 打开了疑似有限振荡窗口，但现有 LUT 无法判断该窗口是真实 9D orbit，还是 transfer floor 造成的数值结构。** 不能把低输入 clip 事后豁免，也不能把本轮写成 seizure、termination 或空间 pattern。

锁定 alpha 轴从 0 直接跳到 1，`0 < alpha_G < 1` 未采样。本轮没有扩 transfer、没有补 sub-unit alpha、没有打开 phi/slow/spatial，也没有运行五臂 ablation。

## 下一门（未启动）

若主线决定继续，应单独预注册两项，不和本轮混写：

1. 用扩展且数值验证过的低 `mu` transfer / exact-Siegert sensitivity 重放 6 个 LUT-blocked 参数点，先判候选是否仍存在；
2. 只有 transfer 复核通过后，才做 12 s confirm、相邻格/双初态门和 dynamic-vs-instantaneous/clamped/subtractive/mean-only ablation。

工程上，重复 full screen 前应把 `PoolParameters` 数组预计算移出 Euler 循环；当前实现虽只占 0.311 GiB，但 wall 约 30 min。
