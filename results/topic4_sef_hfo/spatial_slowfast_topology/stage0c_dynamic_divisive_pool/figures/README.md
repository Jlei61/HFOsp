### stage0c_dynamic_pool_topology.png

这张图汇总冻结 z、均匀 9D 快系统中动态 recurrent-E 除法池的 root 与 state-fork 结果。左上圆点/叉号分别表示稳定/不稳定 root，颜色只编码 E rate；右上是非 exact-root 轨迹的参数格分类；下排展示 z=0.90 的同一 boundary probe 在代表性 alpha_G 下的 E rate 与未裁剪 S_G。它只回答是否存在有限快态对象，不代表发作、自发转换、恢复或空间传播。

**关注点**：是否存在低于 100 Hz、无 LUT/状态边界依赖、并通过 12 s confirm 的绿色 candidate 格。

### stage0c_clipped_orbit_diagnostic.png

这张图重放锁定网格内最强的未决点 `z=0.80, alpha_G=12`。上排显示 E rate 与动态池/除数的重复耦合，下排把 `mu_E/mu_I` 相对 LUT 下界 -40 mV 画出，并给出 `(S_G,r_E)` 投影；同一点在正式 screen 中有 12 个不同 non-exact histories 呈约 2.079 Hz、tail mean 约 9.10 Hz、逐步 tail peak 约 99.56 Hz 的相同行为，但 tail LUT 越界约 9.1%。这只是 `audit_invalid_candidate` 的诊断重放，不是 accepted orbit，也没有进入 confirm/ablation。

**关注点**：重复活动的每个低谷都越过 transfer LUT 的低 `mu` 边界，因此下一步必须先复核 transfer，不能直接把闭环轨迹解释成真实 limit cycle。
