# Figure 5 自主回返：交付入口

本轮全部有界仿真与候选图已完成：18条主条件分支、16条结构条件分支、2条120秒自主结构对照。最终判断见[科学审阅](final_scientific_review.md)、[机器摘要](final_summary.json)及[完成核对](completion_audit.json)。所有图仍为候选，Agent自查与作者目视验收分开；正式分岔未认证。

## 先看同一条自主轨迹

|输出|科学用途|
|---|---|
|[A：完整raster与三个放大窗](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924/figures/fig5-panela-loop.png)|检查原间期、进入高活动、自主退出及第一次短事件返回；固定细胞与真实时间。|
|[B：Z/K/G/M轨迹](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924/figures/fig5-panelb-loop.png)|把Z恢复到参考与间期事件回来分开，不把低活动自动当作间期。|
|[C：原生空间帧](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924/figures/fig5-panelc-loop.png)|检查同一网络上的空间招募与返回事件。|
|[状态空间回返](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924/figures/fig5-state-return.png)|保留原Z–局部抑制–率投影，并给Z–K–率投影；没有人为首尾连线。|
|[候选说明与复现入口](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924/README.md)、[技术和目视记录](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924/publication_qa.json)|解释观察时刻、固定细胞和坐标含义；所有PNG与同版PDF已自查，人工待验收。|
|[恢复机制审阅](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/recovery_mechanism/scientific_review.md)|实际Z逐步收支、K衰减尾部及已有反馈时间配对对照。|

A–D来自同一成功轨迹，不把该轨迹内的四次返回当成四个独立种子。原正式图的旧ηM扫描和旧患者能量比较不能自动改称新反馈模型结果。自主回返已观察到，不等于认证极限环或某类分岔。

## 再看固定Z/K的条件响应

|输出|科学用途|
|---|---|
|[18条条件响应与漂移](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924/figures/fig5-zk-conditional.png)|完整空间场钳制Z/K，G/M及其他状态继续演化；两种内源历史共享未来输入。|
|[18条空间均值图](/home/honglab/leijiaxin/HFOsp/results/paper-ready-figure/fig5/candidates/autonomous_loop_20260924/figures/fig5-zk-spatial-fields.png)|发现全E均值掩盖的活动区域差异；均值图不能替代逐帧传播。|
|[主矩阵科学审阅](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/primary_scientific_review.md)、[完整表](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/conditional_table.md)、[完成核对](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/primary_completion_audit.json)|18/18完成；条件分支不计作自主闭环，不把有限窗历史差异称为双稳态。|
|[原生转换与网格覆盖](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/trajectory_grid_coverage.json)|网格没有包围实际第一轮的进入/退出轨迹，不声称找全转换边界。|
|[电导率闭合审阅](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/conductance_response/static_calibration_v1/scientific_review.md)|预定静态验证未通过；后续瞬态/全空间对应及正式分岔认证不成立。|

条件图的K/gL是当下状态，不是k100参数。箭头来自解除钳制时原方程的瞬时漂移平均，G/M仍动态，不能把图当作封闭二维自治向量场。

## 连接结构与传播组织

|输出|科学用途与状态|
|---|---|
|[高活动历史：三结构传播图](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/conditional_runs/event_comparison/high/figures/axis_highZ_lowK_propagation.png)|高Z、低K共同坐标；按时间顺序选首个完整短事件，缺失时用固定末300ms。PNG/同版PDF已自查。|
|[间期历史：三结构传播图](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/conditional_runs/event_comparison/interictal/figures/axis_highZ_lowK_propagation.png)|另一种完整内源历史，同一选择规则；PNG/同版PDF已自查。|
|[两历史的原生计数核对](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/conditional_runs/highZ_lowK_history_comparison.json)|原图/各向同性末10秒计数相同，旋转图仍不同；共享噪声，不能推断独立复现、全引擎收敛或双稳态。|
|[四坐标条件汇总](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/conditional_runs/comparison.md)|复用8条原结构结果，新增16条全部完成；[24条最终总图](axis_controls/conditional_runs/figures/axis_conditional_responses.png)及同版PDF已自查。|
|[自主120秒结构比较](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/native_runs/analysis.json)|两条新增轨迹均完成；三图使用共同120秒，完整解释见下。|
|[活动段及Z收支](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/native_runs/bout_mechanism.md)|三种结构完整轨迹的实际收支已齐备。|
|[输出度与兴奋性来源审计](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/outgoing_and_excitability_audit.json)|入度/权重等匹配不等于纯方向干预，输出度与低阈值来源强度仍有差异。|

在高Z、低K共同坐标，两种历史均显示原图/各向同性有分离短事件及正Z漂移，旋转结构有持续传播及负Z漂移；它支持空间连接参与活动分隔与资源负荷，不能单独归因于轴方向。四个条件坐标也不足以证明转换边界相同或不同的完整形状。

## 方程、运行与复现

[执行合同](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/execution_contract.md)界定本轮范围及率闭合失败后的原生条件分析路线；[持续科学审阅](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/scientific_review_live.md)记录当前证据和限制。[执行期代码与配置快照](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/reproducibility/execution_20260925/README.md)保留当时版本；最终分析和绘图版本另见[最终代码与配置快照](reproducibility/final_20260925_v2/README.md)，逐文件读回核对见完成审计。大型完整状态、图结构和观测保持在协议指定原路径，快照不是可移植的完整数据容器。

统一解释器：`/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python`。分析和绘图producer记录在各图metadata；原生科学runner冻结，未修改方程、增加参数点、种子或观察时长。

完整自主结构对照现已完成，详见[科学审阅](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/native_runs/scientific_review.md)、[共同120秒图](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/native_runs/full_comparison/figures/axis_native_120s.png)、[各向同性末段传播](/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924/axis_controls/native_runs/isotropic_completed_tail/figures/isotropic_110_120s_first_brief.png)。全部结构条件批次与最终总图现已完成；运行状态和完整性以最终完成核对为准。

最终核对通过：34条条件分支、3结构共同120秒读出、7组图元数据及185份快照文件均已核验。本轮仿真及调度进程已退出。逐项任务完成状态见[goal验收](goal_requirements_audit.json)，运行核对见[进程记录](final_process_check.json)。作者目视待验收；正式分岔类型仍未证明。
