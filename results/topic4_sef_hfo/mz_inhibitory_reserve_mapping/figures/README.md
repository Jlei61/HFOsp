### mz_inhibitory_reserve_mapping.png

这张图检查 R0b 的 fixed-q corridor 能否映射到 reserve 参数。A 用完整 return-to-return 窗口测量 bounded CCO 的 inhibitory-use；B 显示锁定背景事件的 sensor dose；C 将 q_res 明确标成参数而非安全边界；D 直接检验是否真的只有最后一次事件越过 entry fold；E 只检查平均标量 q-nullcline；F 给出 fail-closed gate。

本节点的 q trajectory 仍是离线 scalar replay，尚未与 fast state 自洽耦合。锁定 event schedule 同时参与 endpoint calibration，因此只用于暴露 timing conflict，不能作为 autonomous entry 证据。

**关注点**：如果前五次事件已经越 fold，结果必须登记为 clean no-go；不能换 seed、改 target 或把 q_res 当作 q(t) 的安全下界来救。
