### fig5-dual-core-transition-story-v2.png / .pdf / .svg

Fig.5 六联候选图。A/B 来自同一条 `dualcore_s39 + Joint=1.25`、topology 2542 / dynamics 2642、连续 OU 驱动且 Z/M 同时开启的 40,000-cell SNN 轨迹，并严格共用 0–3.815 s 时间轴。A 显示返回型间期事件、pre-onset 放大和持续招募态；B 显示同一时刻的 population E rate、core/surround 的 `Z` 与 core adaptation `A=eta_m*m`。当前工作点是 tonic plateau，10–250 Hz band-limited virtual-contact power 在 onset 后下降，因此没有把它伪装成 runaway magnitude；runaway 用直接的群体放电率和招募比例定义。

C 使用 A/B 中精确标出的三个时间窗，统一画 0.5-mm local-E rate 的 `log10(1+Hz)`。白色箭头由规则选中的返回型间期事件 onset map 拟合，并原样复制到 pre-onset 与 early recruited map；图下的 `|cos Δtheta|` 是各窗口 `log(1+rate)` 空间梯度与该间期 onset-gradient 的绝对余弦。橙/青轮廓是同一 realized network 的两个冻结 core，空心点是虚拟触点。

D 是独立的临床 cohort 桥，不是 16 个患者各自跑了 two-core SNN。指标是每名患者冻结间期 A/B propagation field 与 clinical onset 后 0–10 s、1–150 Hz early-ictal energy field 的绝对空间相关，先在患者内折叠，再与保留触点几何的 all-contact channel-shuffle 中位数配对比较；strict-broadband 组为 12/16 高于 null，单侧 paired Wilcoxon `p=0.0193`。该 null 比 within-shaft null 弱。

E 复用经过 pseudo-arclength 折返与 fixed-point Jacobian 零特征值共同核验的 spatial-Z 多分支图谱。线型不编码 delay-aware 稳定性；OU-on 竖线只投影 100 个 active runs 的中位工作截面。F 使用现有 `tau_z x tau_m` 3x3 网格，每格 2 topology x 2 dynamics；格内上行为 median operational runaway latency、下行为发生转变的次数。

**关注点**：这是 layout 与证据链都已落到真实 artifact 的 v2，不含示意数字。当前最需要补的是 F 的 `depletion strength x tau_z` 专扫，以及把 C 的空间轴一致性在多个 topology/dynamics seed 上确认；D 目前承担临床数据桥，不能写成跨患者模型预测。
