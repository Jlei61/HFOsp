> 2026-09-29: user discontinued the null branch; prior validation is secondary. Current I revision shows separate descriptive A/B similarities without nulls or P values.

### fig4-paneli.png
红、蓝分别展示TA–MTA、TB–MTB；Data的每个点为一位患者，粗线和竖段为中位数与IQR。Null灰色小提琴为25位患者各自独立杆内置换后计算的队列中位数分布，灰线为该分布中位数与IQR；它不是25人的原始null分数混池。每次两模型模板共用一个杆内触点置换并重做最佳一一配对，括号P为单侧条件性比较、两模式Holm校正。
**关注点**：这是训练选点后的触点错配参考，未校正整个拟合和搜索过程，不能称为独立恢复显著性；灰色null和Data点的汇总层次见上述定义。

### fig4-complete-layout.png
复用当前A–H，仅替换I；替换区域以外像素与当前图一致，同次导出PNG/PDF/SVG。仍使用上一版25人、417个已评分条件的冻结工作点，不读取后台新增搜索点。
**关注点**：这是待作者目视检查候选，current_version.json未更新；H仍是原例的触点分折读出。

### cohort_null_comparison.png
分别画A、B的队列中位数null分布，彩色竖线为实际队列中位数。每次置换中每位患者权重相同，独立生成杆内映射，共9999次联合抽样；小置换空间的单患者分布先完整枚举。
**关注点**：A/B的Holm校正条件性P分别为0.0202/0.0004，比较问题是固定拟合结果的触点对应，而非未见数据预测。

### per_subject_null_comparison.png
每行一位患者，左右分别为A/B，彩色点为实际相关，灰分布及横线为该患者的置换参考和中央95%区间，灰短竖线为null中位数。至多24种置换的小空间直接显示离散支撑点；右侧列出50项Holm校正P。
**关注点**：50项校正后没有单患者模式比较达到0.05，队列偏高不等于每位患者都显著；完整原始和校正P、置换空间与来源见source/per_subject.csv及分析目录。
