# M3A-v2.2 approach-to-criticality — frozen-Jacobian verdict (PRELIMINARY)

**Output framing:** branch-aware frozen-Jacobian PRELIMINARY verdict, pending Milestone-2 spot-check / attribution / controls. This does NOT claim the model proves CSD exists or is absent.

## 测了什么
我们把一次会“跑飞”的仿真（抑制资源 q_I 慢慢耗尽、全局恢复变量 h_G 参与）沿时间抽了若干快照，在每个快照把网络当下的慢状态冻住，问一句：如果现在轻轻推一下，这个网络回弹得快不快？回弹越慢，说明它越靠近“一推就失稳”的临界点。

## 怎么测的
每个快照都在一个小的降维率场模型上求解当下的静止工作点（低放电支），再算冻结雅可比的主特征值 α₁（越接近 0 越临界；τ=−1/α₁ 是回弹时间）。只有通过质量门（收敛、非饱和、准静态、率不失配）的低支点才算数。抽了 48 个快照，其中 15 个合格；注入的慢变量：q_I, h_G。

## 揭示了什么
没看清，但原因很具体：抽到的合格快照上，回弹速率 α₁ 一直明显为负（离“一推就失稳”的临界点还有余量），所以够不上“平滑软着陆”的判据；但在最后一个合格快照和跑飞之间做二分补检时，发现低支的 α₁ 在两个相邻抽样点之间**穿过了 0**（补检里最高冲到 +0.189 per ms）——也就是说临界边界很可能就落在这段没被抽到的空隙里。既然确实存在一个被跳过的 α₁≈0 过渡，就不能判成“无预警硬跳”；又因为这个过渡没被合格快照采到，也不能确认“平滑软着陆”。所以当前正确结论是：transition 区间抽样密度不够、临界边界可能被漏采，判为 unresolved——这是采样分辨率的问题，不是求解器坏了、也不是工作点不可信。下一步（Milestone-2）在这段空隙里加密抽样即可定位这个穿零点，并判断穿零处的主模态是沿轴向、非轴向还是全局。

## Overlay 决策
- overlay_verdict = `refused`；overlay_drawn = `False`。
- phase_map overlay REFUSED (overlay_verdict=refused); uncalibrated slow->rate mapping -> mechanism_candidate_only, no atlas overlay (Hard-QC #7)

## 关键字段（内部归档代号，括号补注）
- verdict = `unresolved_operating_point`（∈ smooth_CSD / hard_jump_no_CSD / unresolved_operating_point）
- verdict_source = `actual_trajectory`（来自真实 3-D 仿真轨迹（M3A-v2.2 approach trajectory），不是 2-D atlas）
- operator_type = `continuous_jacobian`，alpha_units = `per_ms`
- tier = `model_side_ground_truth_preliminary`
- unresolved_subreason = `alpha0_crossing_between_sampled_trajectory_points`；continuation_source = `actual_slow_space`

阈值敏感性、每点 α₁/τ、mode-class、非正规放大（numerical_abscissa / directional_gain）见 `trajectory_verdict.json`；诊断图见 `figures/trajectory_criticality_verdict.png`。

## 实现说明（INTERIM_BRIDGE）
本里程碑为保证“单一仿真源 + 逐比特不走样（byte-parity）”，让库层临时包一层去调用旧的画图脚本作为仿真的唯一来源。这不是长期干净的 model API。Milestone-2 在做 SNN 扰动实验前，应把依赖方向倒过来、暴露一个干净的 仿真/扰动 API。