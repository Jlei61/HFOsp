# Figure 1：恢复原始谱图与质心方法

来源布局：`/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig1/candidates/y1_local_rank_peak_profiles_20261010`。算法核对见上级spectrum_contract.json和method_consistency_audit.md。

### fig1-panela.png / .pdf

采用该布局来源中的Y1 A7/A9脑图与波形，文件逐字节保留。彩色中点表示相邻双极通道，波形示意不参与本次质心计算。

**关注点**：脑朝向、引线和作者已接受的几何对应不变。

### fig1-panelb.png / .pdf

右侧恢复原始800 Hz处理链、50 ms Hamming窗与40 ms重叠，先平滑完整谱再截50<f<300 Hz。中心为完整事件S³加权质心，显示为S/max(S)；采用Y1同一A杆A3–A9的1559/1562/1574三个实例，显示窗统一为±150 ms。左侧178段HFO谱与老脚本数值完全一致。

**关注点**：多峰时中心允许位于两峰之间；不能改成峰顶或70%峰团。

### fig1-panelc.png / .pdf

逐字节保留来源布局的Y1原序热图与rank分布。全部18,190事件及参与掩码不变。

**关注点**：18通道布局为显示内rank，26通道布局为原rank，不能混作重新聚类。

### fig1-paneld.png / .pdf

逐字节保留来源布局的permutation机制示意及40人MI统计。Null点、括号、统计值和坐标均未改动。

**关注点**：本次不重新计算MI或其null。

### fig1-panele.png / .pdf

逐字节保留来源布局中Y1的TA/TB热图及均值±总体标准差。冻结分组仍为13,160和5,030个事件。

**关注点**：与C共享相同事件全集及显示通道。

### fig1-panelf.png / .pdf

逐字节保留来源布局的40人overall/within-template MI与single/multi配对inset。字体和坐标排版保持。

**关注点**：谱图修复不改变队列结果。

### fig1-complete-layout.png / .pdf

仅替换完整拼版中B右侧谱图区域，其他区域PNG逐像素不变；PDF保留原矢量图层并叠加新谱图。标题和时间轴继续统一。

**关注点**：本次恢复的是计算方法，完整拼版待作者目视检查。
