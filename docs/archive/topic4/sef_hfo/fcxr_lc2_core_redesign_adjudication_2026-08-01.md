# FCXR-LC2 大设计审阅与 Core 重构裁决（2026-08-01）

## 结论

接受“不得按原 LC2 Phase 0–3 waterfall 执行”的核心审阅。旧 spec/plan 已标记 superseded；下一轮
只运行 LC2-Core：sensor separability → H/X reduced geometry → 40k frozen forks → dynamic Z/H/X
lifecycle → replication。M、患者表型和详细空间分析延后。

本次只改文档、核代码/产物，不跑仿真、不改引擎。

## 成立的四个 P0

1. **X 映射不同构**：旧 reduced `-g_Xx` 是加性负电流，真实 SNN X 是 presynaptic E→E release 的
   乘性 availability。Core 删除加性项，H 积分 post-X recurrent drive。
2. **tau_H 不是 equilibrium geometry 轴**：它只控制事件间残留和 trough bridge；theta/k/rho 与 RC1
   slope 决定固定点几何。Core 先做 sensor 可分离性，再由 loop gain 解 rho。
3. **M 不得阻塞 offset/recovery**：Core 全程 M=0；M 只在 lifecycle replicated 后进入 Phenotype。
4. **Reduced 不负责生成 IED**：它只证明 basin、offset surface 和平均慢路径；无 kick repeated-IED
   onset 只在 SNN 验收。

## 真实代码数据流核验

已直接核查：

```text
slow.step snapshots x_relay(t-)
 -> kick_probe E->E scatter multiplies edge weights
 -> delay ring / I_E_rec
 -> mz_slow_vars.membrane_terms
 -> gErec_raw
 -> cooperative/saturation position
 -> conductance membrane
```

所以 `gErec_raw` 已是 post-X 输入；`membrane_terms` 也确实控制 tanh 前的 recurrent conductance。当前
H vertical slice 可只改非 blessed `mz_slow_vars.py`。这里保留 blessed hashes 是最小改动选择，不再
写成科学公理；真正硬门是 H-off 对 RC1 逐位一致。

## 对审阅建议的两处事实性修正

1. **不能零计算完成 R1**：现有 HEO/LC1 NPZ 主要保存 rate/LFP/pooled hist，没有足够的逐细胞
   `gErec_raw(t)`；必须同配置短 sensor-only replay，并先复现原状态摘要。
2. **不需要为 H 改 blessed hook**：审阅提出这是待验证问题；代码核验已回答为“现有 hook 足够”。

## 新文件

- Core spec：`docs/superpowers/specs/2026-08-01-topic4-fcxr-lc2-core-design.md`
- Core plan：`docs/superpowers/plans/2026-08-01-topic4-fcxr-lc2-core.md`
- Deferred Phenotype：
  `docs/superpowers/specs/2026-08-01-topic4-fcxr-lc2-phenotype-deferred-design.md`

旧文件仅作审计：

- `docs/superpowers/specs/2026-08-01-topic4-fcxr-lc2-hysteretic-carrier-design.md`
- `docs/superpowers/plans/2026-08-01-topic4-fcxr-lc2-phase0-phase3.md`

## 当前边界

Core/plan 仍是 candidate，尚未授权执行。没有产生 H sensor、分支图、frozen forks 或 lifecycle；当前
允许的表述只到“设计已按同构机制与快速证伪顺序修订”。
