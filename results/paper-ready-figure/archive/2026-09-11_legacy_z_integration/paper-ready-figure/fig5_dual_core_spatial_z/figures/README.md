### fig5-panel-c-core-a-bifurcation.png / .pdf / .svg

Fig.5C 候选。横轴是对称空间路径上的 core-A 失抑制 `D_A=1-Z_A`，纵轴是 core A 内每个 E 神经元的平均发放率。蓝/橙粗线是低态与 tonic 外根；棕色细线是从 tonic 外根实际续接的 global-recruited family，灰色细线是从低支零模配对根独立续接的 core-A-localized family，空心圆是全部 continuation folds。粉色线/带是 100 个 OU-on SNN 的 operational onset 中位数及 q10–q90。

这不是字面意义上的单个 LIF 神经元分岔：当前确定性模型最细是 2 mm E/I population unit。图中使用 core-A per-neuron mean，是现有证据允许的局部尺度。

**关注点**：主图没有 inset、没有手工补线，也没有用线型外推稳定性。global-recruited 与 core-A-localized 两族在相同参数处最近仍相差 7.29 Hz full-state RMS，所以保持分开；这不等于证明它们在未续接区间永不相连。OU-on 中位截面找到 6 个完整空间根，但投影到 core-A 均值只有 4 个高度。含全部实际 delay bins、mean gain 与 diffusion-variance gain 的稳定性和 nonlinear OU residence 只在这个工作截面报告，不扩展为整条分支定理。

### fig5-panel-d-state-response.png / .pdf / .svg

Fig.5D 候选。同一条冻结 dual-core SNN 轨迹上，在低态 1000 ms 与 early-ictal 2615.4 ms 使用完全相同的 16 个分层随机位置和相同 16-cell 弱脉冲。每个位置均做 exact-resume paired probe–sham，图中分别对 0–50 ms descendant-only signed response 做等权位置平均。

**关注点**：左右图比较同一网络两个时点的 incremental response，不是比较两个不同网络，也不丢弃强响应位置。两侧均由少数 hotspot 主导，且 early-ictal sham 已处于高态（0/16 可再作 ignition test），所以该图不是跨 seed 易感性或触发概率估计。

### fig5-panels-cd-dual-core-spatial-z.png / .pdf / .svg

C/D 同行 proof sheet，尺寸比例按 Fig.5 下排版准备。C 是多分支 fixed-point atlas，D 保持同位置扰动的 low/early-ictal 状态响应。只允许与同一 `dualcore_s39 + Joint=1.25` 底物重画的 A/B 合并；不能直接与旧 `joint_04_control seed1801` A/B 拼成同一实验。

**关注点**：C 回答 core-A 局部群体快系统有什么分支，D 回答相同局部扰动在 runaway 前后如何产生不同空间响应。

### fig5-supp-spatial-z-mechanism.png / .pdf / .svg

机制补图。左图是 runaway-entry fixed-point 零模在 20 mm 双核 sheet 上的位置；右图是固定 `Z_surround=0.80` 后独立扫描 `Z_A` 与 `Z_B` 的有限 multi-start root catalog。它们解释分岔如何落到空间，但不再冒充正式 Fig.5D。

**关注点**：零模 93.7% 的能量位于 core A；相图只报告有限 root catalog，不把未找到的根写成数学不存在。
