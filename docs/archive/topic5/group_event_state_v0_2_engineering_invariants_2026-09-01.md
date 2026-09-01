# Group-Event State v0.2：工程不变量与回归测试

状态：**所有 A/B/C agent 的强制工程附录；不是科学结论清单**。

## 1. 四条不可违反的科学实现原则

1. causal prefix：任何 anchor 只能读取该时刻及以前的信息。
2. session/gap/seizure boundary：状态、exposure 和 target 不得静默跨越合同边界。
3. train-only target construction：repertoire、归一化、超参数和 checkpoint 选择只用 TRAIN/inner-validation。
4. patient/seizure/block-first evaluation：事件行和滑窗不冒充独立样本。

## 2. 已关闭、必须长期回归的缺陷

- `softplus(log tau)` 曾把分钟初始化压到秒级；使用 `exp(clamp(log_tau))`，同时禁止把可表达范围写成已识别尺度。
- validation/test warm-up 终态曾被丢弃；split pass + carry 必须与 uninterrupted causal pass 对齐。
- no-state 臂的 STOP/participation head 曾偷看 state；用输入扰动证明所有 head 真看不到 carry。
- 图零假设曾混用五个代码包并平均重复 payload；比较必须锁 source/config/checkpoint hash，重复直接报错。
- Yuquan seizure ID 不能直接字符串连接；用 recording code crosswalk 并逐发作核对 onset。
- 长窗资格必须与最终 estimator 使用同一 coverage segment 逻辑；滑窗数不能写成独立窗口数。
- fixed jump 曾饱和成免费截距；截距/常数漂移零真值进入单元测试，主科学 arm 不为此膨胀。
- ridge 必须按 operator/Gram 尺度正规化；远坏于 intercept baseline 的拟合标为不可估计。
- seed 必须检查初始化、训练顺序和 payload hash 不同；byte-identical seed 记作一个拟合。
- synthetic/test PASS 只证明实现符合合同，不证明 H1/H2/H3。

## 3. Session-preserving training

- 主训练 batch 的并行维是不同 recorded sessions。
- 同 session 内 chunk 严格按时间顺序，边界 carry state；允许 detach graph，不允许重新初始化 state/counter。
- `n_streams=8` 人为切 session 的旧方式只作 compatibility sensitivity，不作主训练。
- state/split 使用 float64 或整数采样点保存绝对时刻，禁止远历元 float32 时间戳。

最少测试：单次整段 forward 与多 chunk carry 的输出/终态一致；跨 gap/seizure 必须 reset；打乱 session 内 chunk 顺序必须失败。

## 4. Manifest、原子输出和恢复

- producer、dataset、split、source、config、checkpoint、target-builder 均写 hash。
- result 先写临时文件，完整校验后原子 rename；manifest 只在所有必需文件存在后更新。
- rerun 按完整 payload hash 幂等跳过；缺文件、旧 hash、非有限值不可冒充 DONE。
- 单一 queue owner；监督进程不拥有或复制队列。
- `CURRENT_HANDOFF.md`、STATUS、PID、日志、失败类型和精确 resume 命令必须落盘。

## 5. 资源与 OOM 合同

- Python：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`。
- 所有 worker 设置 `OMP_NUM_THREADS=1`、`MKL_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`NUMEXPR_NUM_THREADS=1`。
- 新模型先在每张 GPU 各跑 1 个代表性 smoke，记录 peak allocated/reserved 和 wall time，再计算并发。
- GPU 并发上限：保留至少 4 GiB 安全余量，并只使用峰值显存的 80% 容量预算；GPU 已被其他队列持续高利用时不得仅因“尚有显存”继续堆作业。
- OOM 只对当前 job 依次降低 batch、chunk、slot；最多重试 3 次，仍失败记 `resource_failed`，不删患者、不改 endpoint。
- CPU 并发由 `MemAvailable`、实测 p95 RSS 和底层物理盘决定；共享缓存构建沿用已测安全上限 Epilepsiae 9、Yuquan 7，分析任务可从 8 worker 起逐步扩到 16。
- 大缓存、checkpoint 和临时 target 写 `/data`；根分区只放代码、小日志和最终索引。
- 禁止 `pkill -f`；用 queue owner 记录的精确 PID/PGID 管理。

建议共享资源租约：

`results/epi_prssm/group_event_state/v0_2/shared/resource_leases/<agent>.json`

至少写 agent、PID/PGID、GPU、slot、预计峰值、开始时间和心跳。租约用原子创建，失活租约经 PID 与时间双重核验后才能回收。

## 6. 首轮运行纪律

- 先 3 位固定长患者 × 3 seeds 做 smoke/资源/收敛检查，再运行预先定义的 development cohort。
- 所有核心 producer 和主任务至少 3 seeds；5 seeds 只用于预先指定的承重配置，不按结果追加。
- 不打开 formal/sealed，不生成或覆盖 paper-ready Fig1–Fig4。
- 图生成后同时提供 PNG、矢量 PDF、metadata 和 `figures/README.md`，并进行目视验收。

## 7. 每个 agent 的完成定义

工程完成、assay 可估计和科学支持必须分别给状态。最终至少交付：

- `plain_language_report.md`；
- `technical_report.md`；
- machine-readable JSON/CSV；
- `CURRENT_HANDOFF.md`；
- 复现命令、失败清单、资源使用和未触碰范围。
