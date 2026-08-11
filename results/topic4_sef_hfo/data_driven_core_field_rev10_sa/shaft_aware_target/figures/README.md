# rev10-SA 图说明

### rev10_sa_shaft_aware_positive_controls

这张图不包含新 SNN 仿真。A 比较患者训练事件、人工删除全部 SCL、以及只改变跨杆 timing 后的新目标；B 检查从 0/4 到 4/4 逐步恢复 SCL contact 时误差是否连续下降；C 固定 15 个 contact 的二维位置和杆身份。

**关注点**：删除 SCL 必须显著恶化，跨杆 timing 操作必须主要改变 ICL-SCL precedence，且接触点坐标重合不能让 ICL/SCL 身份消失。

### rev10_sa_patient_mode_definition_audit

这张图在同一个 patient-training-only shaft-aware PCA 空间中比较旧 A/B 标签与新的 consensus KMeans。A/B 是同一批事件的两种着色；C 给出逐行归一化混淆矩阵；D 比较两套标签下 15 个固定 contact 的 recruitment prototype。

**关注点**：先看 KMeans 是否形成稳定分割，再看它是否仍对应原来的传播 A/B；稳定聚类与旧标签低 AMI 表示 target 定义改变，不能直接进入模型优化。


### rev10_sa_direction_extent_factorization

这张图只使用患者训练 blocks。A/B 比较旧传播方向标签与 shaft-aware KMeans 对事件招募范围的分割；C 在 `recording block × ICL 招募数 × SCL 招募数` 内严格配平 A/B 后，比较三类 precedence 与同层置换 null；D 显示方向与范围的四个交叉组合都有充足事件。

**关注点**：若 KMeans 几乎完全由事件范围预测，而范围配平后旧 A/B 的 ICL-ICL 和 ICL-SCL precedence 仍高于 null，则两者应作为不同因子进入 target，不能强迫一个平面 K=2 同时承担两种结构。


### rev10_sa_historical_artifact_rescore

这张图对保留逐事件 contact 数据的历史模型做零仿真重评分。A 显示各历史家族的 SCL recruitment，斜线灰柱表示 OOD 超过 50%、患者模式不可评价；B 将 mode A 的 ICL 内 precedence 误差与 SCL recruitment 误差分开；C/D 比较旧 shaft-blind objective 与新 shaft-aware score 在 L2/L3 候选中的排序。

**关注点**：没有逐事件 identity 的 48 个 field-fit candidates 不进图；FULL_TIMING 与 ORDINAL_COMPATIBLE 使用各自患者 floors，不跨语义比较绝对分数。
