# SNN cm-scale traveling-wave read — four-control bi-model validation (2026-06-08)

虚拟 SEEG 观测层的 **SNN 第二基底在厘米尺度（真实 4mm 电极间距）跑通** 的归档。配套 rate-field
runner-v2（用户的四对照已过）→ 现在 SNN 也过 → **双模态门齐活**。

Runner: `scripts/run_sef_hfo_snn_cm_wave_read.py`；engine `r_kick`/`t_kick` patch（patch-note
`scripts/engine_patches/kick_center.patch`）；结果 `results/topic4_sef_hfo/observation_layer/snn_cm_wave/`。

---

## 0. 朴素话三段

**测了什么**：把"戳一下→沿连接方向传播→虚拟电极读方向"这套,在能插真实 4mm 电极的厘米尺寸 SNN 上,
看四个对照是否都对:(C1) 连接方向取 0/45/90° 读出是否跟着转;(C2) 踢点挪位是否把方向带跑;
(C3) 电极杆整体旋转是否把方向带跑;(C4) 各向同性连接(无方向)是否如实读不出稳定方向。

**怎么测的**：reframe(用户 option A)——cm 事件不是"自限团"而是**持续行波**,所以读"第一遍前沿
扫过各触点的先后"(=传播方向),用 endpoint_centroid_axis(早到→晚到 centroid)。两种读法:发放包络
+ 电流型 LFP(真实记录形态)。同一套几何/估计子/阈值,只换基底(rate→SNN)。

**揭示了什么(精确口径,不放宽闸门)**：cm SNN 读出的方向**确实来自模型的连接,不是电极/踢点/脚本**——
但按**预注册闸门(两种读法都 ≥7 触点 且 <25°)**逐条说:**3/4 对照两种读法都干净(C1 连接跟踪、
C3 电极旋转不变、C4 各向同性负对照);C2(踢点挪位)方向正确但发放读法触点不足**(见 §2 caveat)。
**双模态 parity 的可比信号是发放包络**(rate runner-v2 读的就是发放密度,rate 场没有电流型 LFP 对应物);
电流型 LFP 是 SNN 独有的额外(更真实)读法。**所以 parity 口径: C1/C3/C4 在共同的发放信号上一致过,
C2 在发放信号上欠采样(方向对、触点 5<7)。** 不写"四对照全过"。

---

## 1. 关键 reframe + scale law（用户驱动）

**reframe (option A)**: cm 尺度 SNN 单端踢出来的是**持续推进的行波**(不是小尺度的自限团)。用"等事件
自终止"的窗(event_window_for_run)读 → n=0(波不自终止)。改读**第一遍前沿**(first-crossing onset,
固定窗),估计子不变(WF-A 垂直陷阱只坑 onset_front_axis,不坑 endpoint_centroid_axis)。

**scale law（验证后直接驱动调参,用户 2026-06-08 提示)**:
1. **mean-field 工作点对 N/density 不变**(固定入度 C_EE=800)→ g=3.6/drive=0.6 **跨尺度不重调**。
2. **有限尺寸噪声 ~1/√N**:小 N 噪声大→事件自限;大 N 噪声小→行波持续 + 周期边界 re-entry。
3. **前沿速度 v≈0.13–0.17mm/ms 是 intensive**(强度量)→ 读窗随 L 缩,不随 N。
4. **建连内存 ~N²**(cm 真正的墙:L=12 density=1000=144k 建连就 OOM>215GB)→ **density 是降 N 的
   旋钮**(不动工作点),且额外噪声**顺带把行波拉回自终止**(同时解决内存 + 持续不灭)。L=12 density=500
   (N=72k)≈50GB,可行。

**读法修复链**: 早踢(t_kick=20,默认 150 太晚→短 T 把前沿切断误判 runaway)+ 固定第一遍窗 +
endpoint_centroid_axis + density=500。

---

## 2. 验证（L=8 → L=12 → 四对照）

- **L=8 density=1000**: oracle 45.0°, current-LFP **3.3°/14c**, firing 6c(差1), returned=True,
  v=0.133mm/ms。读法在小尺度先验通。
- **L=12 density=500 (N=72k, 内存可行) θ=45**: oracle 45.0°, **current-LFP 13.0°/12c, firing 10.9°/8c,
  returned=True** → cm WAVE-READ PASS。
- **四对照 @ L=12 density=500**(下表; current-LFP 为主读, firing 为快速 proxy):

| 对照 | 测什么 | oracle(err) | current-LFP | firing | 判 |
|---|---|---|---|---|---|
| C1 θ=0 | 读跟连接转 | 1.4(1.4) | 1.1°/10c | 15.0°/8c | ✅ |
| C1 θ=45 | 同 | 45.0(0.0) | 13.0°/12c | 10.9°/8c | ✅ |
| C1 θ=90 | 同 | 90.7(0.7) | 15.0°/12c | 15.0°/8c | ✅ |
| C2 踢点 −perp | 种子位置≠方向 | 43.0(2.0) | 8.9°/9c | 5c | ✅(LFP+oracle 停 45°) |
| C2 踢点 +perp | 同 | 48.1(3.1) | 8.9°/9c | 5c | ✅(同) |
| C3 电极旋 30° | 电极角≠方向 | 45.0(0.0) | 3.7°/10c | 3.7°/10c | ✅ |
| C3 电极旋 60° | 同 | 45.0(0.0) | 13.0°/12c | 3.7°/8c | ✅ |
| C3 电极旋 90° | 同 | 45.0(0.0) | 3.7°/10c | 0.0°/10c | ✅ |
| C4 各向同性 AR=1 | 无连接方向→读不出 | 69(24) | 46.1° **rd 0.08** | 76.1° rd 0.09 | ✅(无可读轴) |

阈: axis_err<25°, n_part≥7, 各向同性 readability<TAU_FAIL=0.3。

**结论判读**:
- **C1（承重）**: 0/45/90 三个角度读出都跟着连接转(oracle err≤1.4°, LFP/firing<25°)。
- **C3（最强的反电极伪影对照)**: 电极杆转 30/60/90° 读出**仍停在连接的 45°**(LFP/firing≤13°)——
  证明读的是模型不是电极摆位。
- **C4（负对照)**: 各向同性事件无方向 → readability 塌到 **0.08**(<<0.3), err 46–76°(乱) →
  读出**找不到稳定方向**,正确。peak_inst=0.366(更接近全局)、v=0.078(更慢)——各向同性是大而慢的
  无向团,印证"各向异性是定向传播必需"。
- **C2**: 轴**停在 45°**(oracle 2–3°, LFP 8.9°)→ 踢点挪位不带偏方向。但 **firing proxy 只 5 触点**
  (<7): perp 偏移把行波路径挪开 → 居中 montage 的发放触点偏少。**current-LFP(空间积分,真实记录)
  仍 9 触点过线**。

**caveat(诚实,advisor 复核后收紧)**:
- **不写"四对照全过"**。按预注册闸门(两读都 ≥7 触点),**C2 在发放包络上 INSUFFICIENT(n=5<7)**——
  不能事后改口径说"LFP 为主读就算过"(=放宽闸门,CLAUDE.md 明确反对)。C2 方向是对的(三个独立信号:
  oracle 2–3°、current-LFP 8.9°、readability 0.894),只是**发放触点不足**。
- **C2 发放欠采样的根因 = sheet-vs-montage(不是基底失败,也不是简单换 montage 能补)**: nc=6 montage
  足迹 20mm > L=12 半片 ±6mm,本就外触点出片/绕回;perp 踢点把行波横移 → 居中、且过大的 montage 上
  发放(脉冲密度,本就比 rate 连续场稀)在片内触点更少。rate 场 C2 过是因为用了 **L=24 大片**(montage
  能装下横移的波);**SNN 因建连内存 ~N² 上不到 L≥20**(OOM)→ 同样几何在 L=12 装不下。加大 montage 更
  出片、跟随踢点又毁掉"踢点不变"对照逻辑 → 在 SNN 可行尺度上 C2 发放无法干净补到 ≥7。
- **parity 可比信号 = 发放包络**(rate 场无 current-LFP);所以 parity 口径 = **C1/C3/C4 共同信号一致过,
  C2 发放欠采样(方向对)**。current-LFP 全过是 SNN 独有 bonus,不拿它代替 parity。
- **尺度/密度显式**: cm 验证在 **density=500(N=72k,内存可行的 mean-field-等价点)**,非 L=8 用的 1000;
  oracle 在 0/45/90 都 err≤1.4° 正是"低密度仍保方向"的证据。density=500 比 1000 噪声大→误差 3–6°→10–15°(<25°)。
- **C4 两旋钮**: iso = AR=1 **且** center 踢(vs 单端),匹配 rate 设计;"读不出轴"主要是各向同性,但
  对称中心踢也贡献一部分(无源→汇梯度),严格说不是纯各向异性单因。

---

## 3. 工程教训（跨夜自动扫失败,已修)

跨夜 14-条件自动扫**零结果**: ①raw `nohup python &` 后台没扛过会话中断(`/compact` API 错)→ 5min 死;
②内存看门狗 `pgrep -f run_sef_hfo_snn_cm_explore` **匹配到自己的命令行**(命令里含该串)→ 自检失明,
进程死了没察觉没重启。**修复**: harness `run_in_background`(持久)+ **PID-file 看门狗**(按进程号杀,不按名)
+ 脚本内 `/proc/meminfo` 自检(<40GB 优雅退出)。**建连内存 ~N²** 是 cm 真墙(见 §1.4)。

---

## 4. 状态 + 下一步

- **双模态门: 实质满足、一处未完全闭合(诚实)**。共同信号(发放包络)上: **C1/C3/C4 两基底一致过;
  C2 SNN 发放欠采样(方向对、触点 5<7),受 SNN 可行尺度 L=12 的 sheet-vs-montage 限制**。即 cm SNN 作为
  独立第二基底**确实读出连接方向、且不被电极摆位/踢点带跑**(C3 旋转不变是最强证据),但严格 parity 在
  C2 发放上有一个覆盖缺口。**要完全闭合 C2 发放需 SNN 上到 L≥20**(需先解 ~N² 建连内存)= 单独工程。
  作 Step-3 验证模态够用(方向读出在两基底一致 + 反电极伪影对照过),但报告时不说"四对照全 PASS"。
- 阈值/估计子/几何/D6 未动。工作点 g=3.6/drive=0.6 跨尺度不变(scale law §1.1)。
- 可选 follow-up(非阻塞): firing proxy 在 perp 踢点下用稍大/跟随 montage 补到 ≥7(纯 proxy 完整性,
  不影响 LFP 验收); 更高密度(内存允许时)降噪把 cm 误差压回 <10°。
- **Step-3 异质核**(用户并行已在 engine 加 `V_th_per_neuron` hook): 现在有了可信的 cm 观测层 +
  scale law 调参框架,可以接异质 patch 读"哪里先点着/沿哪轴传/通道顺序稳不稳"。
