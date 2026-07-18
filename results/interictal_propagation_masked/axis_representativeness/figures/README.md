# 模板轴对单事件主方向的代表性统计

### axis_representativeness_cohort.png

主图检验所有可拟合且具有足量二维 QC-clean 单事件的 gradient axis。以 subject 为统计单位，先在每个患者内等权折叠 TA/TB，再比较真实轴的 mean signed cosine 与同一 montage 上 template-rank shuffle 重建假轴的中位数；正值表示真实轴比几何匹配的假轴更能代表事件方向。
图形复用 paper-ready `fig3_field_concordance_cohort_stat` 的 Data-vs-Null 语法：violin + IQR box + subject points + 显著性括号。由于真实值与 null 已完全分离，正式版不画患者内连线；所有 subject 点固定在对应类别中心，不使用随机 jitter。Strict stability 不作为主分析纳入门；主分析 n=26，strict 子集仅作为 sensitivity。
箱体为四分位距、黑线为中位数；事件均向角度改善作为次要表达仅保留在 summary JSON 和 CSV，不重复占用正式画布。
主效应量是患者级 alignment margin：真实 template-gradient axis 与单事件传播方向的 mean signed cosine，减去同患者 montage-matched rank-shuffle null 的中位数。

**关注点**：看真实 gradient axis 是否在患者内系统高于 rank-shuffle null；这是同数据 descriptive representativeness，不是 held-out generalization。
