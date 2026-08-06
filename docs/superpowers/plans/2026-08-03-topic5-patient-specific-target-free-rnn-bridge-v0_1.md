# Topic 5 patient-specific target-free RNN bridge v0.1 execution plan

## A. 输入冻结

1. 从 v0.4 masked rank dataset 读取 34 人 event/contact 数据。
2. 从已审计 clinical-onset cache 只读取 target metadata，冻结可用患者、发作和 exact
   contact join；训练阶段不反序列化 target 数值。
3. 保存 dataset、config、code 和 target-cache fingerprints。

## B. 患者内训练

1. 先在 E1084 跑一个 GRU seed smoke，核对 loss、GPU 内存、rollout 长度和接触点顺序。
2. 并行跑 strict clinical-onset 队列：patient × 3 seeds ×
   {GRU, linear-state, rank-shuffle GRU}。
3. 每个 unit 原子写 `DONE.json`、checkpoint、training log、heldout metrics、rollout 和
   contact-rank distribution；已有成功 unit 自动跳过。
4. watcher 汇总完成数、失败、GPU 内存、NaN/OOM，并支持网络重连后续跑。

## C. 患者内间期结果

1. 核对真实顺序模型相对 rank-shuffle 的 heldout NLL。
2. 核对模型 rollout 与 test20 的 participation、rank distribution 和 precedence。
3. 展示代表患者 observed-vs-generated contact-rank distribution。

## D. 跨状态 readout

1. 确认全部 target-free unit 冻结后再读取 early-ictal target。
2. 在每次发作 exact joined contacts 上计算 sign-free max-field score。
3. 运行 5000 次 all-contact null 和 5000 次 within-shaft sensitivity；每次重新最大化
   候选 field。
4. patient-first 汇总，并单列 `epilepsiae_1146` supportive。

## E. 交付

1. 生成六联 paper-ready 图和 `figures/README.md`。
2. 写白话结果报告、限制边界和论文推荐措辞。
3. 跑测试、manifest、图像目视 QA。
4. 按实现、结果/图、文档三个逻辑批次提交。
