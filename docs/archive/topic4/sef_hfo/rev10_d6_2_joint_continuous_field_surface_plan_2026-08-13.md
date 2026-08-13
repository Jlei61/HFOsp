# Topic 4 rev10-D6.2 执行计划

1. 冻结七个连续场坐标、六张新网络、固定 16 s 时长和 exact no-op Edge。
2. 先跑一个 `/usr/bin/time -v` sentinel，据实测 RSS 和当前 `MemAvailable` 决定并发；每 worker 设 `MemoryHigh=20G`、`MemoryMax=24G`。
3. 其余 41 个任务由 `systemd-run --user -> nohup` 托管；数值库每 worker 单线程，状态和 NPZ/JSON 原子落盘。
4. controller 以 180 s 间隔内部等待，完成后自动 aggregate、运行 D6.2 auditor 并发桌面通知；对话侧只挂一个长 waiter，不做轮询。
5. 审计通过后，按冻结的诊断展示规则生成 Fig.4 风格两图：同网络直接波形/空间模式，以及自然 KMeans/患者 cross-fit 一致性。图只作诊断，是否科学通过由网络级统计决定。
