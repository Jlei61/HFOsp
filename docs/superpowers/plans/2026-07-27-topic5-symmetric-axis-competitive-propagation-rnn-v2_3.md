# v2.3 execution plan

> 对应 spec：
> `2026-07-27-topic5-symmetric-axis-competitive-propagation-rnn-v2_3.md`

## Milestone A：冻结输入与 tied-rank denominator

1. 从 dataset v0.4 生成 34 人 tied-rank inventory。
2. primary 排除存在 non-source tied rank 的 25 个 events。
3. 核对 development、31 人 sequence 和 22 人 physical-axis locks。
4. 核对 transition decomposition 的 train-only selected axes 与 checksums。
5. 三位 development patients用同一 32-direction train-only procedure生成 axis。
6. 再次验证 target seal。

## Milestone B：实现最小 categorical recurrent core

1. symmetric normalization 和共同 axis kernel。
2. shared-axis skew basis。
3. \(P/C\) 两状态 recurrence，event reset。
4. fixed node bias、seen mask 和 categorical next-contact likelihood。
5. fixed LOSO STOP。
6. 禁止 dense bypass。

必须通过 toy tests：

- \(W^S=(W^S)^\top\) 且非负；
- \(W^A=-(W^A)^\top\)；
- \(\mathbf u\rightarrow-\mathbf u\) 不改变 source-conditioned product；
- seen contacts 概率为 0；
- categorical probability 在 eligible contacts 上和为 1；
- event reset；
- heldout 不进入 axis、bias 或 optimizer selection；
- one-state、no-source、local-isotropic 和 no-history ablations确实移除相应结构。

## Milestone C：三患者 development

只选择：

- persistence pair；
- learning rate；
- batch size；
- early-stopping epoch policy。

工程验收：

- 3 patients × 2 seeds；
- 无 NaN/Inf；
- train/validation loss 可下降；
- exact resume；
- checkpoint/config/log 完整；
- peak RSS/VRAM 记录；
- target values read=false。

不设置“2/3 患者科学阳性”门。

## Milestone D：正式纯间期训练

### D1 22 人 physical-axis primary

- 3 seeds；
- 8 个 frozen model conditions；
- chronological train80/heldout20；
- 每患者 train60/val20 只作 epoch selection；
- patient-first summary。

### D2 31 人 supportive

只跑 coordinate-free history controls和 categorical Markov benchmark；不池化进
physical-axis claim。

资源合同：

- subject/seed/condition level CPU/GPU并行；
- 每 GPU 只放一个可控 worker；
- 默认 batch 2048，OOM时按 2048→1024→512 退避并记录；
- 每个 worker 独立 log、checkpoint、resolved config；
- `nohup`/`tmux` supervisor + atomic run-state；
- 不覆盖已完成 checkpoint。

## Milestone E：按 Claim 顺序分析

1. Claim A full > node。
2. Claim B full > no-history / one-state。
3. Claim C full > local-isotropic。
   - 同时检查 matched `axis two-state no-source > local-isotropic`；
   - 两者都通过才允许 physical-axis interpretation。
4. Claim D full > no-source，独立 secondary。
5. full 相对 empirical ordered-history Markov 的 benefit recovery。

Claim A–C 未全部通过时，不生成 latent-state mechanism panel。

## Milestone F：latent-state dynamics

仅在 A–C 通过后：

- \(P/C\) rank-step trajectories；
- per-node cumulative drive / competition；
- autoregressive participation 与 first-arrival rank distribution；
- source-side rollout；
- A/B heldout read-back；
- mixed-sign axis residual 是否由 delayed competition 解释。

## Milestone G：clinical metadata 独立推进

继续使用
`results/topic5_clinical_onset_source_annotation_v0_1/annotation_registry.csv`
做双盲人工标注。模型运行与标注者隔离。

没有 exact source sets 时，不创建 ictal dataloader，不读取 energy values。

## Milestone H：交付

- development freeze manifest；
- formal run inventory；
- patient-level comparison table；
- calibration/benefit-recovery table；
- latent-state figures（仅 gate 后）；
- figures/README.md；
- target-seal audit；
- checksums；
- tests/logs；
- bounded claim / forbidden claim report。
