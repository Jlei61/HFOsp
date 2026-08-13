# Topic 4 rev10-D6.3 执行计划

1. 在任何 1401–1412 网络运行前冻结 warm 与 `d62_a0p5_b0p5` 两臂。
2. 先跑一个候选 sentinel，按 `/usr/bin/time -v` RSS 和 `MemAvailable` 决定并发；每 worker 单数值线程并设 20/24 GiB cgroup 界限。
3. 24 个任务全部由 `systemd-run --user -> nohup` 托管；controller 内 180 s 低频等待，完成后自动 aggregate、audit 和通知。
4. 只用 12 张新网络作正式复制；D6.2 结果仅作背景。
5. 完成后生成冻结候选的同网络直接波形图与 KMeans/患者一致性图，图为诊断，网络级 verdict 为科学裁定。
