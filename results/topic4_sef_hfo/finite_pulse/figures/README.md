# Step 0b 有限脉冲响应图（局部均匀斑块近似）

> 本步是空间均匀近似、单一刺激位置——中心/轴端/离轴的区分留到 Step 3 加空间 patch 后。所有幅度都是真实仿真（不是用粗扫标签近似）。
>
> **本次为 scaffold（占位参数）跑，结果是退化的**：所有工作点、所有脉冲（半径×时长×幅度 0.2–3.0）的响应**全部是 extinction（点不着就熄灭）**，`fraction_with_window = 0`（recovery off 与 on 都是 0）。这与 Step 0a 的发现一致——scaffold 连接强度下系统深度稳定、最不稳模在 k=0（全局），不存在沿轴有限波长的可激窗。因此：(1) `step0b_example_snapshots.png` **本次没有生成**——没有任何 self-limited / global-synchronous / runaway 事件可作样例；(2) 这是计划预期内的合法 scaffold 结论（"`fraction_with_window==0` 也是一个发现，Step 1 不启动"），**不是 bug**。正式结论须等数据锚定的工作点与单位（见 go/no-go 闸门 formal tier）。

### step0b_response_surface.png
代表性候选工作点上的**完整响应面**：横轴脉冲幅度、纵轴脉冲（半径×时长）组合、颜色是五类响应（熄灭/局部鼓包/自限传播/全局同步/失控），左右分"纯抑制"和"加恢复变量"。
**关注点**：绿色（自限传播）有没有形成一条夹在"局部鼓包"和"失控"之间的带——这条带存在、且上方"失控"出现得明显更晚，才说明有真正的间期工作窗；不能只挑一个好看的脉冲。（本 scaffold 跑：整面全灰=全 extinction，没有任何工作窗。）

### step0b_margin_waterfall.png
每个候选工作点的 `A_self_limited`（刚出现自限传播的幅度）、`A_runaway`（刚失控的幅度）和两者之间的安全余量带，左右 off/on。
**关注点**：余量带为正且不太窄的工作点有几个——这是 go/no-go 的直接依据；注意余量是 `A_runaway − A_self_limited`，**不含**局部鼓包/全局同步。（本 scaffold 跑：所有点 `A_self_limited` 与 `A_runaway` 均为无穷，余量带为空。）

### step0b_window_fraction.png
候选工作点里"存在自限窗+正余量"的**比例**，off / on **并列**（不自动选）。
**关注点**：off / on 哪套比例更高，且 JSON 里 `sensitivity`（半 dt、小 L）是否一致——不一致说明是数值/边界假象，不能信。（本 scaffold 跑：off=on=0，半 dt 与小 L 也都是 0 → 退化结论在数值上是稳定的，不是 dt/边界假象。）

### step0b_example_snapshots.png（本次未生成）
正式跑里：各挑一个自限传播 / 全局同步 / 失控的例子画时间快照，青色竖线是激活质心位置；自限传播那行的青线应**逐帧移动**（波前在走），全局同步那行青线基本**不动**（原地一起亮）——这是把"传播"和"同步闪光"肉眼分开的判据。本 scaffold 跑没有任何此三类事件，故此图缺省。
