# Topic 4 rev10-D7 执行计划

1. 冻结共享 Z/M baseline hash、D6 连续场生成器和 49 候选 manifest。
2. 在一个 sentinel worker 上实测 RSS，再由 launcher 计算并发数；保留一半可用内存并设置 24 GiB cgroup 上限。
3. 用 `systemd-run --user` 包裹 `/usr/bin/nohup` 启动 98 个 worker；controller 每 180 s 检查原子 status，不在对话中轮询。
4. 自动聚合 returned-only events，运行自然 KMeans、patient cross-fit、recruitment、OOD 与 runaway 审计，并发送完成通知。
5. 若无 2/2 安全且可评价的候选，冻结 negative canary 并停止；若有，另写 selection spec，不沿 fit networks 直接宣称成功。
6. 最终 confirmation 只接受同网络直接波形图和 KMeans/患者一致性图，且必须同时通过 network-level 统计与 PNG/PDF QA。

