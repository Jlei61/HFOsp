### interictal_spatial_information_gain.png / .pdf

左侧固定复用原 Fig. 2B 的 E1146 与 E548，每位患者只画一张 fold-0 held-out 全事件 rose；同色虚线是 Timing 训练模板轴，同色实线是 Timing + space 训练模板轴，两种方法严格共享事件集合和二维显示基底，并将空间模型的 Mode 1 红色实线固定为 0°。
右侧恢复原 Fig. 2B 的绝对 direction-score/零假设语法：底部同一行叠加蓝色 Timing、橙色 +Space 的 10,000 次患者 bootstrap cohort-median 分布，以及冻结 Timing + space 模型后在 held-out recording block 内打乱事件方向得到的灰色 cohort-median null；空间模型真实中位分数为 0.568，对 null 的经验 p=0.000999。
每条患者内连线从 Timing 指向 Timing + space，25 位可评估患者中 21 位提高，增益中位数为 0.028（95% bootstrap CI 0.001–0.063），单侧配对 Wilcoxon p=0.0009078；增益对 block-shuffle null 的 p=0.000999。
该结果只支持空间信息能提高可估计事件子集的跨 block 方向一致性，不代表未见患者泛化、真实组织轨迹、传播速度或因果机制。

**关注点**：先看单张 rose 内同色虚线和实线的方向差异，再看右侧绝对分数高于零假设且多数患者连线向右。底部分布区的长横括号检验 +Space 相对方向置换零模型，短横括号检验 +Space 相对 Timing；图内 p 值只显示星号，精确值保留在本说明和 metadata。

### interictal_spatial_information_gain_paired_violin.png / .pdf

该补充图把同一 25 位患者的 Timing 与 Timing + space held-out direction score 画成配对连线、violin、IQR 和中位数，显著性使用患者级单侧配对 Wilcoxon 检验。
Timing 中位数为 0.506，Timing + space 中位数为 0.568，21/25 位提高，p=0.0009078。
它用于显示配对统计的完整分布，不替代主图中的绝对零假设比较。

**关注点**：看每条患者内连线的方向和整体中位数移动，不要只比较两个 violin 的边际形状。
