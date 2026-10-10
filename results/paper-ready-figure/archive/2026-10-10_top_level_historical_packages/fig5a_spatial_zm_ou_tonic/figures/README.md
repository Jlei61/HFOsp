### fig5a-spatial-zm-ou-tonic-global-runaway.png

正式 Fig5A 静态图。它展示 confirmation seed 1842 在持续平稳空间 OU 背景下，由低活动进入持续、全局招募的近饱和高态；四层依次为 Z/q 与 M/gK、群体率、全局招募和 15 个 virtual-contact current proxies。

**关注点**：这是 3/3 新 seed 确认后的模型内部高态，不是临床 SEEG 或患者机制复现。

### fig5a-spatial-zm-ou-tonic-global-runaway.gif

早期生成的曲线揭示版，不含逐神经元 SNN 空间放电。本文件仅保留为诊断记录，不是本次要求的动画交付。

**关注点**：需要看真实 SNN 空间放电时，请使用下面的 `fig5a-spatial-zm-ou-tonic-snn-activity.gif`。

### fig5a-spatial-zm-ou-tonic-snn-activity.gif

正式 seed 1842 的同步 SNN 动画。左侧逐帧显示空间 Z/q permissivity 与 M adaptation，中间显示过去 10 ms 内真实发过 spike 的 E 神经元在冻结组织网格上的活动比例，右侧同步播放群体率、全片招募和 15 个 virtual-contact current proxies；所有面板共用同一个模型时钟。为补齐旧归档未保存的逐神经元 spike，使用原 commit、原网络缓存、原空间 OU realization 精确重放，并确认 14 组归档轨迹逐值 bit-identical。GIF 含 111 个有效帧、90 ms/帧，末帧停留 1.17 s，总播放 11.07 s 并无限循环。

**关注点**：看 480 ms 附近局部空间放电如何迅速扩展为全片点亮，并与约 394 Hz 的 tonic plateau 及 15 个触点的持续高态同时出现；这是模型内部 near-saturated tonic global runaway，不是临床 SEEG。

### fig5a-spatial-zm-ou-tonic-snn-activity-final.png

上述 SNN 动画的末态静态快照，用于快速检查布局、固定色标和持续高态是否可读。

**关注点**：末态 E 活动为全片 100%，Z/q 达到 0.775 floor（即 permissivity 0.225），M 空间均值约 4.9；色标为整段固定值，不做逐帧归一化。
