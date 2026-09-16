# 04:38阶段记录

前一轮progress，本轮progress+verified wait：新启动仅一个相同早期Z条件的数值/记录副本，不加任何生物参数条件或统计样本。goal仍active，09:40截止新增探索。

## 为何增加保存副本
原M40/early worker每0.5s有内存checkpoint回调，但只有每10s或终点才flush全部观察。高态散射极慢，数小时都不能落下一个10s块，阻碍Fig5读取实际过程。
run_topic4_early_Z_lookup_dense.py给源early_z_refill_s9108401做一个数值副本：同一10s完整父状态、同job(η=.005,τM=1)、同噪声、同2s早期refill/1s补充后释放/60s再进入观察规则；仅采用已通过200ms整个高态逐位一致QA的scatter_lookup，并在回调中只读输出每0.5s完整观测。未改原物理或端点源码。

- 副本worker PID211021。主体输出early_Z_lookup_dense/runs，完整0.5s观察供图目录early_Z_lookup_dense_figures/runs；最终完整候选在early_Z_lookup_dense/candidate。
- 原early两个worker30171/31978和supervisor28233全部继续。副本不是第三个噪声、不进入M40或Z147的F。
- 原source_progress在prepare时保存到source_progress_at_prepare.json。到同一时刻会自动核对Z、M电流、进入/返回tracker，写source_progress_parity.json；目前尚未到，不要说已经完成此项。与同段原空间/raster逐位对照须待原10s块落盘后才有，不提前声称。
- 启动时两个输出目录所复制engine均与10s源engine摘要严格一致，参数/完整RNG未重置。原始0–10s闭合chunk只链接，不重复计算/计数。
- 只读prefixwatcher现在211022，update动态读取window.json的additional_preview_sources。它会自动看到dense副本的前缀，但当前尚无新的高态图；第一块观测仍待验证。
- 源物理文件9个hash检查通过，原worker均未重启。新副本实际live，CPU/内存活动已核查(~2.3GiB)；50s具体handle211021等待后仍运行，不能因为尚未写第一块就判作终止。

## 其他状态
Z147已116完成、31运行、无pending。M40仍0完整/12运行/28pending，后续新Mworker会使用lookup wrapper；当前尚无新wrapper起跑。fullfast90s pilot到绝对120s(释放后43.5s)无再入，Mclear到327s无再入，均仍live。
注意high_state_M_gain_probe的checkpoint实际为下一个整数秒(75.5起→76、77、78…80.5)，不是每0.5s；其76.0之后长时间未写不表示仿真卡死。前次总结里每0.5s说法应更正。

## 下一步
1. 首先核查dense副本第一块0.5s观测/时间连续性、source_progress_parity，以及是否产生实际不同M条件的新Fig5。不要将numeric副本当重复样本。
2. 原early/η.2/fullfast继续等待实际完成端点，不自动缩短原科学观察窗；Z新完整格按固定规则自动汇总。
3. 新图逐图目视，完整1–5必须真有中间有限事件且⑤真实第二高态。1000s原Z-only完整对照已交付仍缺⑤，E2仍15/15未增强，不能改窗口冒充通过。
4. 09:40盘点要额外检查window列出的dense副本和其自动图；原goal supervisor4073736的snapshot不含全部后加支路。
