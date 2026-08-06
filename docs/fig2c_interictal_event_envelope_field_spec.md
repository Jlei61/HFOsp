# Fig2-C 间期单事件包络传播 frame / GIF 规范

> 合同：`fig2c_interictal_event_envelope_field_candidate_v1`  
> 状态：paper-ready candidate；后续所有“间期传播场 frame / event-envelope GIF”先读本文件。  
> Canonical paper producer：`scripts/paper_figures/plot_fig2c_interictal_event_envelope_field.py`  
> Core renderer：`scripts/plot_topic5_interictal_event_envelope_field.py`

## 1. 这套图回答什么

这套图把同一患者 TA/TB 两类间期群体事件的真实 HFO 包络定时，放回已经冻结的患者特异传播轴和平面中，展示两个代表事件在几十毫秒内的空间演化。

本合同只规范**单事件版本**：TA 行恰好一个 exemplar，TB 行恰好一个 exemplar，两行同步显示各自事件内相对时间。它不规范多事件 event train、事件间切换、跨事件时间拼接或事件平均 movie。后续多事件 GIF 必须另立 producer/spec，显式定义事件边界、事件间间隔、逐事件 t0、是否时间归一化和 exemplar/抽样规则；不得把多个事件直接拼进本合同后仍称 Fig2-C 单事件图。

它与 Fig2-A、Fig2-E 分工不同：

- Fig2-A 展示大量事件的长时序 rank 结构和无监督聚类；
- Fig2-C 展示两次真实代表事件的包络随时间演化；
- Fig2-E 展示冻结 TA/TB 模板本身的静态 rank field。

安全表述是：

> raw-EEG-derived envelope timing cross-check on a previously frozen interictal axis.

不得写成 template-free 验证、cohort 统计、跨未采样组织的 traveling wave 证明、发作重放或机制证明。

## 2. 输入与冻结合同

1. 传播轴、shared plane、contact order、support 和 fingerprint 只从  
   `results/interictal_propagation_masked/template_gradient_fields/per_subject/<dataset>_<subject>.json` 读取。
2. 必须经 `scorers_from_interictal_record()` 校验 fingerprint；失败立即停止。
3. 下游原始 EEG 只提供本次事件的包络和 Fig1a readout，不得回头重拟合轴、plane、support、sigma 或显示范围。
4. 触点必须按 artifact 的 exact contact-name order 对齐；缺失保持缺失，不缩小 contact set 后重建几何。
5. 主图场平滑只使用该次事件实际参与触点。all-contact 和 template-weighted 版本只能作为 QC。
6. 显示核固定 `sigma_display=6 mm`，只控制画布连续性；冻结分析 kernel 保持不变。

## 3. 原始信号与时间零点

- HFO 包络使用单带 `return_hil_enve`，频带 `80–250 Hz`，CAR 后做 50 Hz 谐波 notch。
- 显示量是 robust-z Hilbert amplitude envelope，不写成 power，也不写成跨子带归一化 energy。
- `return_hil_enve_norm()` 是探测用途的跨子带聚合，不用于本图的精细定时。
- Fig1a readout 必须复用 `scripts/paper_figures/fig1_spectrogram_utils.py`：Gaussian-smoothed magnitude、逐触点逐事件 max normalization、主增强连通区质心和真实 STFT cell edges。
- `t=0` 为第一个可用 Fig1a 质心；静态 field、GIF cursor、spectrogram x 轴必须使用同一时间基准。
- exemplar 可以读取模板标签用于分组和筛选，因此不能把这张图称为“无模板发现”。

## 4. Fig2-C 静态 candidate 布局

固定为两行：上 TA、下 TB。每行从左到右：

```text
single-event readout | magnitude colorbar | gap | 7 envelope frames | envelope colorbar |
gap | frozen template rank field | rank colorbar
```

### 4.1 左侧 readout

- readout 与 field 等高但横向更窄，`box_aspect=1.18`；只显示本次事件参与且质心可用的触点。
- 触点按 frozen shared axis 排序，y tick 只写触点名，不追加 `+10`、`+1` 等坐标。
- TA/TB 只作为最左列 y-label：TA `#B2182B`，TB `#2166AC`，粗体。
- readout 顶部明确写 `Sample from TA` / `Sample from TB`，强调它是一次单事件而非完整模板；不写 rho/n。
- 标题在 readout 内靠右对齐，不得伸入其右侧 colorbar 标签。
- 上下两行都写 `time (ms)`。
- 两行 x limits 取两次事件真实 STFT 记录窗的交集；不得用并集在 TA 左缘或 TB 右缘制造无数据白条，也不得裁掉任何已显示质心。
- 每行独立一条 `0–1` colorbar，标签固定为 `Normalized magnitude`。
- 质心轨迹 TA 用红、TB 用蓝，marker face 为金色 `#ffb000`。

### 4.2 右侧 field frames

- 所有 frame 使用同一 frozen shared plane、物理毫米坐标、相同 x/y limits、`aspect="equal"`。
- E1146 candidate 固定 7 帧：`−8, 0, +4, +15, +27, +38, +50 ms`；`0 ms` 必须显式出现，`+50 ms` 后不进入静态 panel。
- 其他患者使用 `_frame_window()`：前缘 8 ms、后缘最多 +50 ms；`_static_frame_times()` 保留六帧基础网格并插入 `0 ms`，总计 7 个不重复 frame。不得为了“更漂亮”手挑时刻。
- 每个 field 都保留 x ticks；只在下排中央 frame 写共享 xlabel `shared TA axis (mm)`。
- 左侧 field 只保留数值 y ticks，不重复 transverse 或 TA/TB y-label。
- 包络使用 `magma`、`vmin=0`。TA/TB 各自有一条与本行 field 等高的 colorbar，标签固定为 `HFO envelope (robust z)`；两条必须共享同一个 `vmax`。不得写成 energy/power。
- `vmax` 从完整显示时间窗的 participant-only 包络联合 99th percentile 计算，不能只看 6 个离散 frame。
- 参与触点画白色粗外圈；未参与触点只画灰色空心圈，不进入主图平滑。

### 4.3 最右冻结模板 rank field

- 每行最右各放一幅冻结群体模板场，上 `TA template`、下 `TB template`；这两幅不是单事件场。
- 必须调用 `build_interictal_ab_panel_payloads()` 和 `draw_interictal_rank_field_panel()`；不得从当前 exemplar 重拟合或复制平滑函数。
- 与中间 event frames 使用同一 shared plane、contact order、物理毫米范围和 6 mm display kernel。
- field 内部仍用冻结 rank 的线性 `viridis` 映射，但 colorbar 必须显示 artifact 中的实际 rank 数值（E1146 为 `0–14`），不得只报归一化 `0–1`；它不是毫秒时延。
- 每行 colorbar 顶部标题只写 `ranks`，不在右侧放长竖排标题；数值 tick 保留实际 rank，最低/最高端分别附 `early` / `late`，不用括号。
- TA/TB 两幅模板场都显示简写 y-label `y (mm)`；不得只给第一行，也不再写较长的 `transverse (mm)`。
- 中间 `magma` 与最右 `viridis` 是两种不同物理量；不得共享 colorbar 或互换标签。

### 4.4 标题与字号

- 全图标题只写匿名患者号，例如 `E1146`，15 pt、粗体、左对齐。
- frame 时间、TA/TB：12 pt；轴标签：12 pt；field/contact/colorbar ticks：8 pt；readout x ticks：9 pt。
- readout、event field、template field 和三类 colorbar 的上下边缘必须严格对齐。
- 当前 paper canvas 为 `16.2 × 4.9 inch`；PNG 150 dpi，同时输出矢量 PDF。

## 5. TA/TB 动态 GIF

GIF 必须复用静态 candidate 的同一对 exemplar、participant mask、shared plane、display limits、6 mm kernel、colormap 和共同 `vmax`。

- 生物学时间：与静态图同一窗口，E1146 为 `−8…+50 ms`；
- 生物学帧间隔：`2 ms`，共 30 帧；
- 播放速度：默认 `12 fps`，只为观看，不代表真实时间倍率；
- 每行左侧 readout 增加黑色虚线 cursor，必须与中间当前 envelope field 帧使用同一时间值；
- 最右 template rank field 在 GIF 中保持静态，只提供群体模板空间参照；
- field title 显示当前相对时间；
- GIF 循环播放，不在末尾加入新的数据帧；
- metadata 必须分别记录 biological step 和 playback fps，禁止把二者混为一谈。
- 本 GIF 始终是一对单事件 side-by-side comparison，不承担事件间状态切换；多事件 GIF 另立合同。

## 6. 标准输出与复现

```bash
python scripts/paper_figures/plot_fig2c_interictal_event_envelope_field.py \
  --subject epilepsiae_1146
```

输出目录：

```text
results/paper-ready-figure/fig2c_interictal_event_envelope_field/figures/
```

正式 candidate 文件：

```text
fig2c_candidate_E1146_interictal_event_envelope_field.png
fig2c_candidate_E1146_interictal_event_envelope_field.pdf
fig2c_candidate_E1146_interictal_event_envelope_field.gif
fig2c_candidate_E1146_interictal_event_envelope_field_metadata.json
README.md
```

### 6.1 更换单事件 exemplar 前的候选筛查

当默认 medoid 事件方向可测但视觉上不连续时，不得直接看成图后手工替换。先运行独立的候选筛查：

```bash
python scripts/paper_figures/screen_fig2c_tb_event_candidates.py \
  --subject epilepsiae_1146 --top-k 500 --n-candidates 4
```

筛查固定 canonical TA、frozen shared plane、participant-only support、6 mm display kernel、
`−8…+50 ms` frame window，并对当前 TB 与所有替代 TB 使用同一个联合 `vmax`。排序只读取原始
readout 数值，不读取渲染像素。cheap search 先在 top-500 template-concordant events 中要求
ICL≥6、SCL≥2 个参与触点、ICL 中段全部参与且 stored-lag `rho≤−0.50`，只对前 40 个幸存事件
回读原始 EEG。E1146 TB strict gate 进一步固定为：每根杆至少 2 个 participating 且 centroid
usable 触点、至少 1 个 peak-z≥5 的触点；沿轴质心-轴 `rho≤−0.75`、沿轴中间三分之一触点
全部可用、中段最低 envelope peak-z≥5、左端减右端质心时差≥8 ms、从右向左相邻质心单调
比例≥0.70。

输出放在：

```text
results/paper-ready-figure/fig2c_interictal_event_envelope_field/tb_candidate_screen/
```

候选筛查不得自动覆盖正式 Fig2-C。若最终用方向清晰度选中的事件替换默认 medoid，metadata 和
图注必须明确写作 `direction-qualified illustrative exemplar`，不能继续称为无条件
representative event，也不能用它估计 TA/TB 方向出现率或效应量。中段完整性优先于 recording
block 去重；同一 block 的不同事件允许并列进入候选屏，但最终主图仍只能选一次 TB 事件。

当前 Fig2-C 已锁定 E1146 TB `event_pos=829`（candidate 4）：ICL 11/11、SCL 4/4 均参与且
可用，沿 ICL 的质心-轴 Spearman 为 −0.900，中段 3/3 可用。它是经过上述门限筛查的
`direction-qualified illustrative exemplar`；canonical paper producer 对 E1146 默认读取该
event，只有显式传入 `--use-medoid-tb` 才回到旧 medoid。

对已经进入候选屏的单个 TB 生成锁尺度 GIF：

```bash
python scripts/paper_figures/screen_fig2c_tb_event_candidates.py \
  --subject epilepsiae_1146 --gif-event-pos 829 --gif-step-ms 2 --gif-fps 12 \
  --mark-selected-for-fig2c
```

此模式不得重跑或改写候选排序，只重建固定 TA 和指定 TB 的原始信号，并复用 candidate-screen
JSON 中的 frozen fingerprint、frame window 和 global `vmax`。

## 7. 强制验收

每次修改 renderer 后必须：

1. 运行 `tests/test_plot_topic5_interictal_event_envelope_field.py`、`tests/test_topic5_interictal_event_field.py` 和 `tests/test_plot_topic5_interictal_template_ab_fields.py`；
2. 目视 PNG：面板/色条高度、方形几何、字体、ticks、无裁切；
3. 逐帧检查 GIF 首帧、中间帧、末帧，确认 cursor 与 field 同步；
4. 核对 metadata 中 PNG/GIF 的 `vmax`、display sigma、exemplar 和 fingerprint 一致；
5. 更新输出目录 `figures/README.md`；
6. 不以“测试绿”替代视觉和科学合同验收。

## 8. 给后续 agent 的执行提示

```text
任务：生成或修改间期单事件传播场 frame/GIF。

先完整读取 docs/fig2c_interictal_event_envelope_field_spec.md。canonical producer 是
scripts/paper_figures/plot_fig2c_interictal_event_envelope_field.py，禁止复制 renderer
另写一套。必须复用 frozen interictal artifact、fingerprint、contact order、shared plane、
participant-only support、单带 return_hil_enve、Fig1a spectrogram helper、6 mm display kernel、
magma 和 TA/TB 共同 vmax。静态图按 single-event readout | cbar | 7 envelope frames | cbar |
frozen template rank field | cbar；必须显式包含 0 ms，三个 colorbar 写明物理量。GIF 用
同一 exemplar 和窗口，2 ms biological step，并记录独立 playback fps。输出后运行三组测试，
目视 PNG/GIF，并更新 figures/README.md。claim 只能是 representative raw-envelope timing
cross-check on a frozen interictal axis，不能写 template-free、cohort proof、traveling-wave proof
或 mechanism proof。
```
