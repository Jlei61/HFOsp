# Figure 1 方法一致性核对（2026-10-10）

## 已恢复并验证

- B右侧直接复用ReplayIED原始预处理函数：按EDF原200 s分段、相邻双极、800 Hz重采样、50–250 Hz谐波IIR陷波Q=30、3阶80–250 Hz Butterworth及filtfilt。先处理整段，再截取该段全部既有packed events并拼接，未单独过滤展示用短窗。
- Hamming 50 ms / overlap 40 ms / nfft等于窗长，幅度谱先Gaussian σ=1.5平滑再保留50<f<300 Hz。显示每通道每事件S/max(S)，中心使用完整事件S³/ΣS³；无70%阈值、连通区选择、边缘排除或逐通道移动。
- 同一输入下，新谱与中心逐项匹配原函数；亦通过src.group_event_analysis.compute_stitched_spectrogram_centroids_legacy独立实现验证。展示事件对已存lagPatRaw的相对质心最大偏差小于1e-10 ms，属于浮点舍入。
- 左侧178段HFO的原始平均谱和基线归一化谱，与p16_mechan_events_specComp.py数值逐项完全相同。它的1000 Hz、180 ms Hann窗及先截频段后平滑是该独立展示的原有定义，不能被右侧群体事件算法覆盖。
- A、C、D、E、F的独立PNG/PDF逐字节保留。完整图修改区域仅B右侧；C/E全部18,190事件、冻结TA/TB的13,160/5,030标签及D/F 40人统计未改动。

## Methods中仍需澄清的一处文字

methods_revised_draft.md第21行把线噪处理统称为FIR；实际原始检测脚本有FIR分支，但本次复现并与lagPatRaw完全对齐的质心支路调用highEvents_yuquan0910_utils.py的IIR notch + Butterworth/filtfilt。因此不能声称这一句对所有处理支路都准确。此次未改写Methods或改变原分析来迁就该文字；应在稿件中区分检测与质心计算支路。

## 显示及解释边界

主图±150 ms只是显示范围；另提供完整±250 ms窗口。三例按恢复后的方法检查并选取，仅作展示，不估计发生率或队列传播强度。质心是整段时频分布的代表时间，多峰时可能处于两峰之间，不是精确生物学起始时刻。

最新18通道C/E布局沿用作者另一轮的显示内重排名次；原始26通道rank和聚类未改写。当前修复同时提供26通道正式布局及18通道最新布局，未把后者自动宣称为人工验收通过。
