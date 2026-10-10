# Figure 3 重建输入

`native_panels/`保存作者认可的A–E原生PDF、SVG和PNG，与正式单panel文件逐字节一致。`layout_spec.json`保存画布、裁切、1:1放置和角标位置；`build_fig3_current.py`据此重建完整图，不需原始数据盘。

`layout_code/`保留本机生成本版的布局代码快照，`lineage.json`保留完整本机科学来源的路径与摘要。科学内容修改仍须使用原分析producer和数据盘，不能把矢量图层重建称为重新分析原始记录；远端包不包含原始记录及私有患者映射。
