# Topic 5 RNN 全 cohort field-transfer 分母修复执行计划 v0.1

对应 spec：`docs/superpowers/specs/2026-08-11-topic5-rnn-full-cohort-field-transfer-correction-design.md`

## A｜输入与分母冻结

1. 审计 dataset_v0_4 的 34 人与 102 个 converged checkpoints/rollouts；
2. 从 Figure 3D subject/event CSV 冻结 17 人/167 seizures；
3. 写入 source/config/hash manifest；
4. 加回归检查，禁止旧 `outer_*` target cache。

## B｜target-free model field

1. 每患者只用 train events 冻结 K=2 mode read-back；
2. 对三 seed held-out native rollouts计算两个 mode 的 rank/support；
3. 在冻结 Figure 3 plane 上生成 model fields；
4. 写 17/17 `MODEL_FIELD_MANIFEST` 后才允许进入 C。

## C｜统一 early-ictal 评分

1. 复用 Figure 3 phenotype selector 和原始 0–10 s activation cache；
2. 覆盖 pooled 17/167、broadband 16/106、gamma 11/61；
3. 1000 次 synchronized all-contact shuffle；
4. 每个 draw 重做 mirror 与 maxAB；
5. seizure→patient→cohort。

## D｜两张图

1. E1146：empirical TA/TB、RNN two-mode fields、early-ictal field；
2. cohort：34 人间期 native rollout vs static；17 人 RNN field vs channel-null；
3. PNG 600 dpi、单页 PDF、SVG、source CSV/JSON、中文 README；
4. 逐图检查布局、字体、legend、色图和分母。

## E｜资源与运行

本轮复用冻结模型，不重新训练。field freeze 与 1000-draw scorer 为 CPU/低显存任务；用独立
`nohup` 进程、4 CPU threads、分阶段 DONE 文件运行。若后续发现 checkpoint 缺失，才补训对应
patient×seed，单进程显存上限 12%，最多 6 workers，不因扩大 early-ictal target 而重训已有模型。
