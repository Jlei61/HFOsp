# Stage-2 动态臂（persistence 恢复电流）诊断图

给 M4 有界态加一个"活动持续够久才积累的局部外向恢复电流" `p(x,t)`，看能不能终止+恢复。三行=E 率 / `q_I` / `S_G`(+`p`)。

### arms_s2cal_seed1.png
校准三臂：A_slow_off=间期基线（34 个短促自终止 IED、`q_I`=1、`S_G`=0，这是恢复目标）；B_m4_anchor=有界态（~80Hz 振荡持续、`q_I` 到地板、`S_G`~0.42）；C_sensor_on=B 加传感器（η_r=0，动力学与 B 一致确认 parity），`p` 平滑爬升到 p_max≈0.55、间期短事件几乎不充它。
**关注点**：确认 A/B/C 三态 + 传感器的持续时间选择性成立（慢升、短事件不触发）。

### arms_s2d_seed1.png
对称 `p`（单 τ_p）扫描 (τ_p, η_r)。强电流把宽持续态压到一个**更低的持续水平**（~25Hz，fragment，`q_I` 只回灌到 ~0.18），弱/慢电流（τ8000 η40）反而**饿死 S_G→失控**。都不终止。
**关注点**：persistence 电流是负反馈调速器不是终止器——压活动就切掉自己的传感器输入，稳到自洽低态。

### spatial_D_tau5000_eta40.png
上面 fragment 态的源空间活动帧：一个**大团四处游走**（压这里→那里冒→漂移），非局灶起始、非轴向招募、非终止波前。
**关注点**：电流是把活动空间搬家、不是熄灭它。

### arms_asymfix_seed1.png（真·不对称，修复后）
真·不对称 `p`（快充 τ_p=3000 + 慢放 τ_p_down=12000，`cfg_effective` 确认 tau_p_down=12000）。**不失控**：初段活动被压到 0 → 安静期 `q_I` 回灌到 0.7–0.8 → 之后**离散短促 burst**（伴 `q_I` 下凹 + `S_G` 瞬起）。
**关注点**：这是本线唯一像"活动→压住→恢复→返回式短事件"的轨迹，但**未确证**：τ_p=3000 快充约 3s 就介入、有界态还没到满态（可能 prevention 非 termination）；后段 burst ~26–198ms、~每 4s 一次（长于基线 IED ~22ms、疏于 ~0.3–0.5s），更像 rebound/breakthrough 而非返回间期事件；空间仍是宽条带。⚠️ 旧 `arms_s2dasym*`（此前误标"真不对称"、实为对称 τ_p=3000 的 P0-bug 跑）已移入 `../invalidated_pre_p0_fix/`。

### A_sensor_on / A_persist_act（selectivity + prevention 控制）
`A_sensor_on`=slow-off + 传感器(η_r=0)，测真间期事件充 `p` 到多少。`A_persist_act`=slow-off + **候选 actuator 开**（同候选参数 τ3000/τ_down12000/p50=0.15/η80,150），测真 34 个 IED 在候选电流下是否存活（prevention 检验）。
**关注点**：候选 `p50=0.15` 下 `Φ(0.084)≈0.09→7–13mV`，不可忽略——所以"候选只作用持续态不误伤 IED"必须用**候选参数匹配**的 `A_persist_act` 验证（`arms_prevctl_eta*`），不能用 `p50=0.25` 的 `A_sensor_on` 代替。
