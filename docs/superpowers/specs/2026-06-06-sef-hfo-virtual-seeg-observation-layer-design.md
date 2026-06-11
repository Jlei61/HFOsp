# SEF-HFO 虚拟 SEEG 观测层 Design Spec

> **状态**：draft 2026-06-06。本 spec 设计**把源空间模型（SNN / rate field）产出的活动，透过虚拟 SEEG 电极、用与真实数据同一套分析流水线读回成传播模板**的观测层基础设施。
>
> **关系**：
> - **实现**（不替换）`docs/superpowers/specs/2026-06-06-sef-hfo-pathology-parameter-mapping-design.md` §5.0 调参合同 ④「虚拟 SEEG 必须走真实模板提取 pipeline（与真实数据同一套：通道序、rank template、forward/reverse），不另起炉灶」与 §7 Track 连接 验收（「isotropic + aligned-shaft 控制必须失败；旋转 θ_EE → 模板方向跟着转，旋转电极杆 → 模板方向不变」）。那个 spec **断言**了这条读出链，但没有实现它；本 spec 就是缺失的那一段。
> - **承接** `docs/topic4_sef_itp_framework.md` §6.5 Step 2/Step 4：「synthetic data 必须走真实 PR-2 / PR-2.5 / endpoint pipeline」「虚拟 SEEG 用突触电流 proxy + envelope」「方向先场后真实 pipeline」「全套 shaft controls」。
> - **保留**（不触碰）framework 已锁的 H1–H6 真实数据验收合同、cohort tier、phantom-rank 纪律、clinical SOZ 不作拟合标签红线、v0.2 核心边界（不解释 clinical seizure onset，不解释 HFO 80–250 Hz carrier）。
> - **不冲突** framework §8.1（HFO carrier 不分 E/I）：本观测层**不产生也不检测** 80–250 Hz 载波；它在 event-envelope 层工作，下游接入点是事件 + 通道时序，不是带通检测。

---

## 0. 一句话承诺（朴素话）

**测了什么**：我们让模型里那块组织发生一次「点着 → 定向传播 → 自终止」的群体事件，然后**不是上帝视角直接读源头**，而是像真实病人一样，**透过插进去的几根虚拟电极**，每个触点记一段活动包络，再用现成的那套分析（哪些通道参与、谁先谁后 → 拼成传播模板 → 比正反模板）把它读成结论。

**怎么测的**：核心是一个判别——如果传播方向是被**连接结构**定的，那把连接长轴转一个角度，读出来的通道先后方向应该跟着转、把电极杆转一个角度方向不该变；反过来，如果某个「方向」其实是**线性电极杆从一头排到另一头**造出来的假象，那即使连接各向同性（根本没有真方向），电极杆对齐时也会读出一个稳定方向——这种情况**必须读不出稳定方向**，否则整条链是垃圾进垃圾出。

**揭示了什么**：到目前为止，那条承重的方向判别一直是在连续场上直接量主轴（绕过电极），我们自己的归档已标成「被网格几何污染 / 未决」。这个观测层补上的就是「和真实数据同一种读法」的版本——只有它钉死了，后面拿模型去比病例参数（哪里容易点着、传播轴朝哪）才有地基。

（内部归档代号：framework §6.5 Step 2/4，pathology-mapping spec §5.0-④/§7 Track 连接，0d anisotropy control，`*_lagPat_withFreqCent.npz`，phantom-rank masked features。）

---

## 1. 五条已锁决定（2026-06-06 user 讨论敲定）

| # | 决定 | 理由 |
|---|---|---|
| **D1 接入层** | 合成数据从**事件 + 通道激活时序**层接入真实流水线（PR-2 吃的那一层）。**不走真实 HFO 检测 / legacy packer、不伪造载波**；但**仍要写一个兼容真实 loader 的 `*_packedTimes_withFreqCent.npy` companion**（per-事件起止时刻，由合成事件直接给，见 §3）——「不走 packer」= 不跑检测/打包算法，**不是**不写这个产物。 | 模型不产 80–250 Hz 载波（framework §8.1 红线），带里没有可检测的东西；伪造载波 = 把一个被明确排除的环节请回来，且检测器测的是我们捏载波的方式不是机制。但 loader 需要 packedTimes 提供事件起止时刻，故 companion 必写。 |
| **D2 电极几何** | **统一接口**：观测层只认「一组触点坐标」作为虚拟 montage。**参数化电极杆先行**（给朝向 + 触点间距，可旋转 / 抖动 / 打乱身份）；**真实病例几何 = 第二层**（投影到模型平面），第一版 **loud-fail stub**。 | 承重方向判别 + 全套对照只需要参数化杆；真实几何要先解决 3D→2D 配准，更重，且方向判别的对照反而不好在真实几何上构造。 |
| **D3 接入时刻定义** | 每个触点的「激活时刻」**锁为：包络首次越阈（前沿到达）** + tie tolerance Δt + 最小时间分辨率（= 采样 dt）；**越哪个阈见 §4.1**（participation = 噪底+margin、timing = per-contact 相对自身峰值 f，**非全局绝对阈**）。预注册，**不准看结果再换**。 | 平滑包络上 first-crossing / peak / centroid 给不同通道顺序；前沿到达对平滑包络最稳（键在最陡的前沿、不受各触点幅度 / 时长差异影响），且与 SNN 已有 onset-front 证据一致。**已知不对称见 §4**。 |
| **D4 构建顺序** | **增量 1 = 已知方向玩具波合同门（不上任何模型）**；**增量 2 = SNN 低驱动竖切（单端踢、只做单模板方向跟踪，第一科学客户）**。两端各踢 swap 强制读出 + rate field 接入延后（§7 / D5）。**不**一上来建「完整虚拟 SEEG 系统」。 | 先把「源空间传播能不能透过虚拟电极 + 真实流水线读回来」钉死；钉不死后面全是 GIGO。SNN 低驱动已有安静可激 + 前沿读出证据，是合适 first customer；简化同质 rate field 的噪声自发离散事件是诚实 NULL / 未稳（2026-06-04），不适合当第一条验收链。 |
| **D5 rate field 角色** | 同质 rate field 接入 = **工程适配 parity 测试**（同一条链能不能在场上跑通），**不是科学主验收**。科学 rate field 客户**等异质核版本**（pathology-mapping spec §5.2 `Var(V_th,E)` patch）。 | 真正要比病例参数，必须是「哪里容易点着」的异质核；同质场没有成核热点，比不了。 |
| **D6 方向估计子 + 2D montage** | 承重判据的「方向」= **pipeline 输出的 endpoint-centroid 轴**（单模板：最早-rank 通道质心 → 最晚-rank 通道质心；forward/reverse：rank-displacement swap-k 的 source→sink），用 **montage 触点坐标**算。读出 montage **必须 ≥2 根非平行电极杆（2D 张成）**。场梯度平面拟合（`t_i≈a+g·x_i`，方向 ĝ）**只作 cross-check，永不作闸门**。详见 §3.5。 | 模板是通道上的 rank 向量、**不是** 2D 方向；从模板到方向那个选择**正是上一版 field-space 网格污染所在**。单根线性杆触点共线，只能读「沿杆分量」，「旋转电极杆 → 方向不变」在单杆上 ill-posed（转 90° 到只读 1D 的波信号从满到零）。 |

---

## 2. 模块结构 —— 小纯函数，不写成大杂烩

`src/sef_hfo_observation.py` = **一组小纯函数**（无 stateful god-class）：

```text
montage  ──  build_parametric_shaft(angle, pitch, n_contacts, origin, ...) -> VirtualMontage
             from_real_geometry(coords3d, ...)  ->  raise NotImplementedError   # 第二层，loud-fail
             VirtualMontage = (contacts: (n_contact,2) coords in model frame, provenance: str)

sampler  ──  sample_envelopes(source_activity, montage, footprint) -> (n_contact, n_time) envelope
             # 推广 results/topic4_sef_hfo/lif_snn/engine/lfp.py 的距离 footprint 到任意坐标
             # footprint = 距离衰减核 f(r)（kernel 宽度是参数）；非满容积传导、非裸最近格点取样

extractor ── extract_lagpat(envelopes, event_windows, lag_def, gates) -> LagPatArtifact
             # 每个事件窗口内: 每触点 participation (越阈) + 激活时刻 (D3: first-crossing+tie+min-res)
             # 事件窗口复用 src.sef_hfo_events.detect_events (已锁检测器)
             # 不足 (见 §5 gates) -> 返回 INSUFFICIENT, 绝不强行聚类

artifact ──  LagPatArtifact = 内部对象(bools/ranks/lag_raw/contact_coords/event_rel_times/chn_names)
             write_legacy_npz(artifact, path)        # 写盘必须用 LEGACY key (lagPatRank/eventsBool/...), 见 §3
             write_montage_manifest(artifact, path)  # 同步写 <record>_montage.json (D6 endpoint axis 需坐标)
             write_packed_times(artifact, path)      # 同步写 <record>_packedTimes_withFreqCent.npy (per-event 起止)
             validate_artifact(artifact)             # 断言 legacy key 齐 + phantom-mask + 坐标顺序==chnNames
```

**模型适配器是薄的，且住在模型旁边、不进本模块**：
- rate field 适配器（在 `src/sef_hfo_lif.py` 或 step runner）：把 `r_E(grid, t)` 在触点坐标上取样 → 喂 `sample_envelopes`。
- SNN 适配器（在 `results/topic4_sef_hfo/lif_snn/engine/` 旁，推广 `lfp.py`）：突触电流 proxy `|I_E|+|I_I|` → 喂 `sample_envelopes`。

**从 `extract_lagpat` 往后全部共享**——两模型并行只多两个薄适配器，不写两套链。

---

## 3. 真实流水线接入点（artifact 合同，§6 deep-contract-verify）

下游真实分析 `src.interictal_propagation.load_subject_propagation_events`（→ PR-2 / `src.rank_displacement` / endpoint anchoring）**读盘用 legacy key，不是内部名**。**分两层，不可混**（写错 key → loader 直接 `KeyError` 或丢事件时间）：

**(a) 内部对象 `LagPatArtifact`**（模块内传递，名字随意）：`bools` / `ranks` / `lag_raw` / `contact_coords` / `event_rel_times` / `chn_names`。

**(b) 写盘 = 真实 loader 的精确 key（核对自 `src/interictal_propagation.py` L344–397）**：

| 写盘位置 | 精确 key / 文件名 | 形状 | 内部来源 | 必需性 |
|---|---|---|---|---|
| `<record>_lagPat_withFreqCent.npz` | `lagPatRank` | `(n_ch, n_ev)` float | ranks（参与内 argsort²；**非参与 = −1/NaN**）| **必需**（loader 直接 `lp["lagPatRank"]`，缺 → KeyError）|
| 同上 | `eventsBool` | `(n_ch, n_ev)` | bools（loader `>0` 转 bool）| **必需** |
| 同上 | `lagPatRaw` | `(n_ch, n_ev)` float | lag_raw（first-crossing；§4）| loader 可选（缺 → NaN）→ 我们**必写**（否则 relative_lag 全 NaN、rank 检验失据）|
| 同上 | `chnNames` | `(n_ch,)` | montage 触点名/序 | **必需** |
| 同上 | `start_t` | scalar | 合成块起始时刻（可 0）| 可选 → 写 |
| `<record>_packedTimes_withFreqCent.npy` | （.npy，无 key）| `(n_ev, ≥2)` | 每**事件** rel start/end 时刻 | loader 可选；**但缺则 event rel/abs times 全 NaN** → 任何**时间类** H（H3 窗口 / 率 / IEI）失据。方向读出（增量 1/2）用不上，但**写完整 artifact 时必写** |

- **时间单位 = 秒（on-disk legacy 口径）**：real loader 的 `lagPatRaw` / `packedTimes` / `start_t` 都是**秒**（实测 verified：`src/event_periodicity.py` `block_dur = 3600.0` s、detections `+ start_t` 秒）。内部 `LagPatArtifact` / 玩具波 / sim 用**毫秒**（sim 约定，对齐 `sef_hfo_events` DT/MIN_DUR_MS）；**writer 在写盘时 ms→s 转换（`MS_TO_S = 1e-3`）**。增量 1/2 只用 rank **顺序**、单位不影响；但 rate / IEI / window（H3/H4，后续）需要正确的秒，否则差 1000×。round-trip 测试断言 `loaded.lag_raw == artifact.lag_raw × MS_TO_S`。
- **packedTimes ≠ per-channel lag**：它是 per-**event** 起止时刻（`packed[:,0]`=rel start、`[:,1]`=rel end），喂时间分析；per-channel 时序在 `lagPatRaw`，两者别混。
- **坐标 manifest（D6 必需）**：legacy lagPat 文件**不带坐标**，但 §3.5 endpoint axis 要坐标。所以**同步写** `<record>_montage.json`：`{contact_coords: (n_ch,2), chn_names:[...]}`，`validate_artifact` 断言其顺序与 `chnNames` **逐一相同**（否则 endpoint axis silently 错配）。
- 非参与触点 rank 走 **phantom-mask** 路径（AGENTS.md `mask_phantom=True` / `build_masked_kmeans_features`）；`validate_artifact` 断言「非参与 rank ∈ {−1, NaN}」。
- 通道序：`chnNames` = montage 触点序，全程一致；下游索引前核对（framework `channel_names` ordering 纪律）。`_record_name_from_lagpat_path` 锁 `_lagPat_withFreqCent.npz` 变体。

---

## 3.5 方向估计子 —— 一个增量一个主门，全部预注册（承上一版网格污染教训）

模板是**通道上的 rank 向量**，不是 2D 方向。rank→方向的转换**正是上一版 field-space onset-front 里网格污染所在**，所以锁死，且**每个增量只有一个主门**（不许「Spearman 或夹角」二选一 = fishing 入口）：

- **增量 1 主门（唯一）= rank-vs-真方向投影 Spearman**：读出每事件参与触点的 rank 顺序 vs 触点沿已知 `n̂` 投影的顺序，取 Spearman ρ。endpoint-centroid 夹角在增量 1 **只作辅助旁证，不进门**。
- **增量 2 主门（唯一）= endpoint-centroid 轴 vs `θ_EE` 夹角**：用下面锁定的 endpoint axis（真实 pipeline 无 `n̂` ground-truth 可投，θ_EE 才是真值）；rank-vs-投影 Spearman 在增量 2 **只作辅助**。

**endpoint-centroid axis（全部参数预注册）**：
- `k_dir`（每端取几个通道）**锁 = 3**；要求 `n_participating ≥ 2·k_dir + 1 = 7`，否则该事件该方向 = **degenerate / INSUFFICIENT**（不强算）。
- 轴 = `centroid(最早 k_dir 个 rank 的通道) → centroid(最晚 k_dir 个)`，坐标用 montage 坐标。
- **退化条件**：`‖late_centroid − early_centroid‖ < ε_deg`（提议 ε_deg = 0.5×触点间距）**或**两端集合相交 → 判 **no-axis**（C1 必须落这里）。
- **符号约定**：与 `θ_EE` 比时按**无向轴（mod 180°）**——连接各向异性是双向的，单模板 early→late 朝向不对应固定 forward/reverse。
- **forward/reverse swap-k 轴 = 延后增量**（§7），不在本轮主门。

**2D montage 硬要求（D6）**：endpoint axis 要恢复 2D 方向，触点必须**张成 2D**。单根线性杆触点共线 → 质心轴被压到杆线上（只读沿杆分量），「旋转电极杆 → 方向不变」在单杆上 ill-posed。所以读出 montage **必须 ≥2 根非平行电极杆**（或真实多杆布局）；framework §6.5「multiple shaft orientations」从 control **升为主读出 montage**。

**方向可读性标量 `direction_readability`（C1/C2 负对照专用，2026-06-06 实现复核补定）**：= 在一圈候选轴（0–180°）上取 `rank-vs-该轴投影 Spearman` 的**最大值**（给源「找到一条方向」的最好机会）。真行波 → ≈ 主门 Spearman（高）；居中径向 / 同起同落 → rank 在**任何**轴上都不单调 → 低（全 tie → NaN）。**C1/C2 must-fail = `direction_readability < τ_fail`**。这比「endpoint 轴是否 None」更稳——rank 并列时 argsort 取子集会偏置出**伪轴**（实现复核实测：�in偏心 montage 对居中径向读出 ~67° 伪轴），可读性标量不受此影响。它是 rank 层的、与 §10 主门同源的量，**不是**被禁作闸门的场梯度平面拟合。**C1 协议**：径向 must-fail 只在 montage **中心对齐径向源中心**且**对称**时成立（偏心 montage 把径向波读成单调到达梯度 = 真实采样伪迹，不是估计子幻觉，不该算 C1 失败）——用对称双杆 + **偶数触点**（避免原点重合），montage 质心 = 径向中心。

**cross-check（永不作闸门）**：场梯度平面拟合 `t_i ≈ a + g·x_i`，方向 `ĝ`；部分绕回场捷径，只作一致性旁证，**不进任何门**。

---

## 4. 预注册的已知不对称：synthetic = first-crossing，real = centroid

`src.interictal_propagation._compute_relative_lag_matrix` 注释明确：真实 `lag_raw` 是 **stitched centroid times**（HFO 活动质心）。我们 D3 锁的合成侧是 **first-crossing**。这是**有意的、预注册的不对称**，不是 bug：

1. 下游只用 `lag_raw` 算**事件内 rank 顺序**（→ relative_lag → ranks），H1–H6 合同测的是**几何 / 顺序 / 稳定性**，**不**比较 lag_raw 的绝对数值。first-crossing 与 centroid 都是「该通道在事件里何时活跃」的单调代理，事件内顺序一致即足够。
2. 合成侧无载波，「HFO 活动质心」无直接对应物；包络 first-crossing 是 recruitment-onset 的诚实代理。
3. 平滑包络上 centroid 正是 D3 要避开的「抹平 lag」最严重者。

**纪律**：这条不对称在 spec 锁定，**不允许**因为「想更像真实 pipeline」事后改成 centroid（那会重新引入 D3 警告的抹平问题）。若将来要做严格 parity sensitivity，可**并列**加一个 envelope-centroid 读数作对照，但主口径锁 first-crossing。

---

## 4.1 越哪个阈：participation 阈 vs timing 阈分离（锁）

D3 锁了「首次越阈」，但**越哪个阈**会改通道顺序，必须分锁成两个**互不相同**的阈：

- **participation 阈（决定 `bools`，全局）**：触点在事件窗内活动是否超过 **噪声地板 + margin**（沿用 `src.sef_hfo_events.calibrate_detector` 的 per-operating-point bar 口径）→ 决定该触点**算不算参与**这次事件。
- **timing 阈（决定 `lag_raw` first-crossing，每触点相对自身峰值）**：在已判参与的触点上，激活时刻 = **首次越过 `f × 该触点本事件内峰值`** 的时刻（提议 `f = 0.5`，spec review 锁）。

**为什么 timing 阈必须 per-contact 相对峰值、不能用全局绝对阈**：这正是 C2（§5）的试金石——对「同起同落、只有幅度差」的源，全局绝对阈会让高幅触点先越阈 = **假时序**（C2 fail）；per-contact 相对峰值阈则所有触点在自身峰值同一比例处同时越阈 → **无假序**（C2 pass）。所以 C2 同时锁住此处的阈值选择。

## 4.2 rate-substrate 检测前端合同（事件窗源 + participation margin）—— Increment-3a 2026-06-07 锁

**背景**：Increment-3a rate-field 四对照在初次实现里，把事件窗从「触点 aggregate」改成「源场全局活跃比例 (`ext`)」、把 participation margin 从 0.5 改成 0.1，**且是在看到四对照失败之后改的**——这正是 §4.1/§10 红线「越哪个阈/事件窗会改通道顺序，不准看结果再换」的违规风险。本节把这两步**显式锁成 rate-substrate 检测前端合同**，并用 sensitivity 证据说明它们**不是**造结论的旋钮。

**关键 reframe（advisor 2026-06-07）**：事件窗源 + participation margin 是**事件检测前端**（决定「有没有一个事件窗 / 哪些触点算参与」= 覆盖）；**parity 主张 = 方向估计子 + 阈值**（`endpoint_centroid_axis`、25°、k_dir=3），这部分**两基底逐字相同、不动**。判定性 sensitivity 问题不是「换旋钮还过不过」，而是「换旋钮 axis_err 会不会变」——若 axis_err 不变、旋钮只移动覆盖，则旋钮是检测前端、不是造结论的拨盘。

**锁定合同（rate substrate）**：

- **事件窗源 = 源场活跃比例 `ext`（rate-substrate 锁，非触点 aggregate）**。理由：踢 (kick) 是模型人为刺激、真实数据**无对应物**；源场事件区间是**几何无关**的最干净时间门（触点 aggregate 在某些电极旋转下会被「直接受刺激 / 平行波的触点持续点亮」污染→检测器判 `returned=False`→无窗）。**代价坦白说明，不掩盖**：这用了真实 SEEG 没有的源场信息——但事件窗只是粗时间门，方向读出（被测对象）仍只用触点。SNN 侧用触点 aggregate（介观、踢盘 0.15mm 极少盖住触点）；rate 侧 2mm 踢盘相对 4mm 触点更大，故 rate 必须用 `ext`。
- **participation margin = 0.10 × (env_max − env_min)（rate-substrate 锁，非 0.5）**。理由：rate field 沿波方向有 3–10× 幅度梯度（近踢点触点 ≫ 波前触点），0.5（SNN 口径）会把波前触点全切掉。**0.10 是最大覆盖端**（更严的 margin 只会丢触点、降精度，不是 cherry-pick 的甜点）。

**Sensitivity 证据（`results/.../increment3a_rate_parity/rate_parity_sensitivity.json`，10 条件 × {margin 10/20/30%} × {window field/contact}）**：

1. **事件窗源不变 read**：在 contact-aggregate **也有窗**的条件（C1×3、C2:perp0、C3:rot0/rot60）上，field-ext vs contact 的 axis_err **差 = 0.0°**。即事件窗源移动的是**覆盖**（contact-aggregate 在 4 个几何脆弱条件 C2:perp±0.35、C3:rot30/rot90 根本无窗），**不是** read。这直接回答用户「事件窗口径会改通道顺序」的担忧——证据是 0.0°，不是散文。
2. **margin 从不把 pass 翻成 FAIL**：10/20/30% 下没有任何条件从 pass 翻成 fail，只会 pass→**insufficient**（覆盖丢失、诚实报）。axis_err 在 n_part≥7 处**始终 < 25° 门**。
3. **诚实 caveat**：axis_err 数值**会**随 margin 收紧漂移（最大 ~10°，如 C1:0deg 1.1°→11.3°）——因为 margin 越紧、触点越少、endpoint-centroid 越噪（覆盖→精度效应），**不掩盖**。但 (a) 它从不越 25° 门，(b) 0.10 是最大覆盖端故非 cherry-pick。
4. **诚实 caveat（C3 rot30/rot90 无对照）**：这两条在 contact-aggregate 下无窗（一根杆恰平行波→触点持续点亮→aggregate 不 return），field-ext 是唯一能给它们定窗的——故它们的 PASS 依赖「两种窗都有窗的条件上 read 一致(0.0°)」的归纳论证，明确写出。

**纪律**：本合同是 **rate substrate 专属**；**L=3 介观 SNN 自限 blob** 仍用触点 aggregate + 0.5（§4.1 原口径）。两基底**估计子 + 阈值相同**（parity 所在），**检测前端按基底特性分锁**（因为踢是 artifact、两基底踢盘/触点尺度比不同）。不允许把 rate 的 0.10/ext 口径反向套回 L=3 SNN，也不允许事后再调这两个值——已锁、已 sensitivity。（**注意**：放大到厘米尺度的 SNN 单端踢出来的是 **traveling front**、不是 L=3 自限 blob，是**第三个基底 regime**，其 participation margin 独立锁 0.10、独立 sensitivity，见 §4.3——那不是「把 rate 口径套回 SNN」，而是 cm 行波自身的幅度梯度独立论证得到的同一数值。）

---

## 4.3 cm-SNN substrate 检测前端合同（traveling-front participation margin）—— 2026-06-10 锁

**背景**：§4.1 把 SNN participation margin 锁成 0.5——但那是为 **L=3 介观自限团**（kick disk 0.15mm 勉强盖住触点、事件是小尺度自限 blob）刻画的。§13.1 + cm 归档（`snn_cm_traveling_wave_2026-06-08.md`）把 SNN 放大到**厘米尺度真实 4mm 电极**后，单端踢出来的是**持续推进的行波（traveling front）**，不是自限团——沿波方向有 3–10× 幅度梯度（近踢点触点 ≫ 波前触点），与 rate field（§4.2）**同形**、与 L=3 blob **不同形**。cm runner `run_sef_hfo_snn_cm_wave_read.py` 实测用 **0.10**。本节把 cm-SNN traveling-front substrate 的 participation margin **显式锁成第三个 substrate regime 的检测前端合同**，并用 sensitivity 证据说明它**不是**造结论的旋钮。

**先澄清两个 0.5 不是同一个东西**（防代号撞车）：
- **timing 阈 `f = 0.5`**（§4.1：first-crossing = 首次越过 `f × 该触点本事件内峰值`）—— 决定 `lag_raw` 时序。cm runner **保持 0.5 不动**（`extract_lagpat(..., margin, 0.5, ...)` 里那个 `0.5` 就是它）。
- **participation margin = 0.10 × (env_max − env_min)** —— 决定 `bools`（哪些触点算参与）。这才是 §4.1 原本对 L=3 SNN 锁 0.5、cm-SNN 改 0.10 的那个量。**本节只动 participation margin，不动 timing f。**

**关键 reframe（沿用 §4.2）**：participation margin 是**事件检测前端**（决定覆盖）；**parity 主张 = 方向估计子 + 阈值**（`endpoint_centroid_axis` / 25° / k_dir=3），三基底（玩具波 / rate / cm-SNN）逐字相同、不动。判定性 sensitivity 问题不是「换 margin 还过不过」，而是「换 margin axis_err 会不会变 / 方向会不会翻」。

**锁定合同（cm-SNN traveling-front substrate）**：
- **participation margin = 0.10 × (env_max − env_min)（cm-SNN 锁，非 0.5）**。理由：cm 行波沿波方向幅度梯度大，0.5（L=3 口径）会把低幅波前触点切掉。**0.10 是最大覆盖端**（更严只丢触点、不是甜点）。CLI `--margin-frac` 默认 0.10；output JSON `config.margin_frac` 记录实际值（可审计）。`--margin-scan` 复跑此 sensitivity。

**Sensitivity 证据（`results/.../snn_cm_wave/wave_read_MARGINSCAN_*.json`，L=20/d100，margin {0.10/0.20/0.30/0.50} × {firing, current-LFP} 两读法）**：

1. **高覆盖处 axis_err 对 margin 完全不变**：C1 θ=45（n_part=10/18）在 0.10→0.50 全程 axis_err **逐字 = 10.9°**（两读法）、n_part=10、readability=0.994/0.988 **一字不变**——覆盖充足时 margin 根本不动 read。
2. **margin 从不把方向翻错，只动覆盖**：coverage-marginal 的 C2 perp−（n_part 8–9）在 n_part≥7 处 axis_err 始终 ≤ 9.6°（≪ 25° 门）；margin 收紧只让 n_part 单调降（fir 8→8→7→6、lfp 9→8→8→6），**0.50 把两读法都降到 6 < PART_MIN=7 → INSUFFICIENT（轴 = None）**。即 0.50 会把一个在 0.10/0.20 干净 PASS 的承重正对照**变成覆盖不足**，不是把它修对。
3. **诚实结论**：0.50 = L=3 口径直接套到 cm 行波 → 切掉低幅波前 → C2 覆盖不足；0.10 = 最大覆盖端、承住波前触点、方向读出与 0.50 在高覆盖处逐字相同。**所以 cm-SNN 锁 0.10 是为承住 traveling-front 波前覆盖，不是为改方向结论**（方向结论 margin-invariant，证据是 §1 的逐字不变 + §2 的从不翻错）。

**纪律**：本合同是 **cm-SNN traveling-front substrate 专属（第三 regime）**。**L=3 介观 SNN 自限 blob 仍锁 0.5（§4.1 不动）**；**rate substrate 仍锁 0.10/ext（§4.2 不动）**。三基底 **估计子 + 阈值（`endpoint_centroid_axis`/25°/k_dir=3）逐字相同**（parity 所在），**检测前端按基底波形特性分锁**（自限 blob vs traveling front vs rate field）。0.10 已锁、已 sensitivity，之后不随结果再调。

---

## 5. 增量 1 —— 已知方向玩具波合同门（不上任何模型）

**目标**：在没有任何模型动力学的情况下，证明「montage + sampler + extractor + artifact」这一段能把**已知**的传播方向正确读回来。钉不死 = 停，后面全是 GIGO。

**构造**：解析合成一个平滑行波 `a(x, t) = g( n̂·x − c·t )`（`g` = 平滑钟形包络，`n̂` = 已知传播方向单位向量，`c` = 波速），方向取 **30° / 60°**（+ 0° / 90° / 135° 补充）。用 **2D montage（≥2 非平行参数化电极杆，D6）** 在若干朝向上采样 → extractor → artifact → 读出每个事件的通道 rank 顺序（主门 = §3.5 增量 1 = rank-vs-`n̂`-投影 Spearman；endpoint-centroid 轴只作辅助旁证）。

**主判据（唯一，编码结论非仅存在性）**：读出每事件参与触点 rank 顺序 vs 触点沿 `n̂` 投影顺序的 **Spearman ρ ≥ τ_pass**（提议 0.9，spec review 锁）。endpoint-centroid 夹角只作辅助旁证（§3.5，不进门）。

**两条 must-fail 负对照（增量 1 就要有，不留到后面）**：
- **C1 居中径向源（无优势方向）**：居中径向扩散鼓包，**montage 中心对齐径向源中心 + 对称（偶数触点）**。**must-fail = `direction_readability < τ_fail`（提议 0.3）**——rank(∝半径) 在任何轴上都不单调。**注意**：montage 必须中心对齐 + 对称；偏心采样会把径向波读成单调到达梯度（真实采样伪迹、不是估计子幻觉），那不该算 C1 失败（见 §3.5 C1 协议）。
- **C2 无到达时间梯度源（严格版，电极排列自造时序陷阱）**：`a(x,t) = b(x)·h(t)`——所有位置**同起同落**（共用一个 `h(t)`），只有空间幅度 `b(x)` 不同、**无任何到达时间梯度**。测的是「电极排列 + 越阈定义本身会不会从纯幅度差造出假时序」。**must-fail = `direction_readability < τ_fail` 或 NaN（全 tie）**。这正是 §4.1 timing 阈的试金石——全局绝对阈 → 高幅触点先越阈 = 假序（fail）；per-contact 相对峰值阈 → 同时越阈、全 tie、无假序（pass）。

**通过门**：主判据 ≥ τ_pass **且** C1、C2 都 < τ_fail。任一不满足 → 观测层污染，停。

**scope 红线（C1 ≠ 增量 2 承重对照）**：C1 的 `direction_readability`（居中 + 对称采样的无方向源不被瞎读出方向）**不替代**增量 2 的 `isotropic 连接 + aligned-shaft must-fail` 承重对照。**偏心采样把径向波读成单调梯度 = 虚拟电极采样几何能真实造方向的风险**（实现复核实测确有此事），C1 用居中对称 montage 是**回避**了这个风险来单测估计子、不是证明它不存在；该风险只有增量 2 的「连接各向同性 + 电极杆对齐 → 必须读不出方向」对照才真正检验。

**scope 声明（不可过度声称）**：增量 1 只运动 **artifact 构造 + lag→rank + 方向估计子**——单一固定方向波只产生**一个模板**，没有聚类多样性 / forward-reverse / swap。真实 pipeline 的**模板聚类 / swap 机制**到**增量 2** 才第一次被运动。**增量 1 通过 ≠「整条 pipeline 验证」**。

---

## 6. 增量 2 —— SNN 低驱动竖切（第一科学客户）

**目标**：把整条链接到真实模型 + 真实分析，复现承重方向判别。

**构造**：SNN 在**低驱动安静可激工作点**（spiking_gt_validation：驱动比 ≈ 0.6，非 1.0 自持振荡）→ 局部踢一下产生自限定向事件 → SNN 适配器（推广 `lfp.py` 突触电流 proxy + 包络）在 **2D montage（≥2 非平行参数化电极杆，D6）** 上采样 → 完整链 → **真实** `rank_displacement` / 模板分析。

**本轮锁定 = 单端踢、只做单模板方向跟踪（user review 拍板）**：0d/spiking 的踢是单 disk → 单方向；本增量**只**从一端踢，验「观测层能把单模板传播方向干净读回来」这一件事。forward/reverse swap **不在本增量**——它要从轴两端各踢，而那是**强制读出测试**（验「观测层能读回两种输入方向」），**不是**模型自然产生正反模板的机制证据，放延后增量（§7）。montage 触点坐标必须 plumb 进空间检验作坐标输入（endpoint axis 才算得出来）。

**主验收（= pathology-mapping spec §7 Track 连接，透过虚拟电极 + 真实 pipeline 版本）**：
- 旋转 E→E 连接长轴 `θ_EE`（0/45/90…）→ D6 读出方向**跟着转**（误差 < 提议 25°，承接 0d 已锁容差）。
- 旋转**整个 2D montage（≥2 杆一起转）** → 读出方向**不变**。
- **must-fail**：isotropic 连接核 + aligned-shaft → **读不出稳定方向**（方向各向异性比 < 提议 1.3，承接 0d/spiking 已用阈）。
- 全套 shaft controls（整组 montage 旋转 / jittered contacts / shuffled contact identity / 多组杆朝向）全部走同一条真实 pipeline。

**采样密度 gate（你点名的稀疏陷阱）**：本轮主问题是方向读出，所以**主门有效事件门槛直接锁 = 参与触点 ≥ 2·k_dir+1 = 7**（与 §3.5 endpoint 轴一致，**不留「进模板但不进方向」灰区**——`n_contact_min` 不设 4）；单事件参与触点 **< 7** → 不入主门；条件内有效事件 **< n_event_min**（待 SNN 事件产率定）→ 该条件报 **INSUFFICIENT**，**绝不强行聚类**造稳定模板。

---

## 7. 延后 / scoped（不进本轮）

- **两端各踢 swap 强制读出测试** = 延后增量：从传播轴两端各踢产生正反两套模板，验「观测层 + rank-displacement swap 能读回两种**输入**方向」。**报告口径 = 强制读出测试，不是模型自然产生 forward/reverse 的机制证据**（输入方向是人为给的）。先把单模板方向读干净（§6）再让 swap 进场。
- **同质 rate field 接入** = 工程 parity 测试（同一条链能否在 `r_E` 场上跑通），**不**作科学主验收（D5）。
- **科学 rate field 客户** = 异质核版本（pathology-mapping spec §5.2 `Var(V_th,E)` patch），等那一步。
- **真实病例几何 montage**（`from_real_geometry`，3D→2D 配准）= loud-fail stub，做 per-patient 异质性比对时才建。

---

## 8. 主要科学坑 → 已锁防护

| 坑 | 锁 |
|---|---|
| **循环论证**：用观测模板 / 端点反推模型斑块方向，再用「复现模板」自证 | 真实电极**几何** = 采样装置（合法 / 必需）；隐变量 patch / 轴**永不**从观测模板设定。观测空间只作**比对**，停在「高于几何采样对照」那一关。pathology-mapping spec §7 Track 异质性「非循环声明」同源。 |
| **电极杆造假方向** | **最硬负对照 = 各向同性连接 + 电极杆对齐必须读不出稳定方向**（C2 + 增量 2 must-fail），增量 1、增量 2 **都**要有。读出 montage **必须 ≥2 非平行杆（2D 张成，D6）**，否则单杆只读沿杆分量、「旋转杆方向不变」ill-posed。 |
| **模板→方向的转换选择**（= 上一版 field-space 网格污染所在） | 方向 = **§3.5 主估计子（pipeline endpoint-centroid 轴）**，唯一作闸门；场梯度平面拟合只作 cross-check、不进任何门。模板是 rank 向量不是方向，这个转换必须锁死。 |
| **时间估计改结论** | D3 first-crossing + tie tolerance + 最小分辨率（预注册）；**§4.1 锁「越哪个阈」**（participation = 噪底+margin；timing = per-contact 相对自身峰值 f，**非全局绝对阈**）；§4 first-crossing vs centroid 不对称锁定。 |
| **采样密度不足造模板** | `n_contact_min` + `n_event_min` gates；不足报 INSUFFICIENT，绝不强行聚类（§6）。 |
| **强制读出当自然机制**（两端各踢 swap）| 两端各踢的 forward/reverse = **强制读出测试**（输入方向人为给），报告**禁止**写成「模型自然产生正反模板」；延后增量（§7）。 |

## 9. 工程纪律

- `src/sef_hfo_observation.py` = 小纯函数（montage / sampler / extractor / artifact writer+validator）；模型适配器薄、住模型旁。
- 输出遵守 phantom-rank 纪律：非参与触点 `-1` / mask，下游走 masked-feature；`validate_artifact` 断言。
- `from_real_geometry` raise `NotImplementedError`（§6 stub loud-fail），不得悄悄返回参数化几何假装完成。
- TDD：增量 1 的已知方向玩具波 = 天然单元测试（已知 ground-truth 方向 → 读出顺序）。

## 10. 验收门（编码结论、非仅存在性 —— 参考 acceptance-gate 纪律）

- **增量 1 门**（主门 = rank-vs-真方向投影 Spearman，§3.5；2D 对称居中 montage）：wave 主判据 Spearman ρ ≥ τ_pass(0.9) **且** C1（居中径向源）`direction_readability` < τ_fail(0.3) **且** C2（严格无到达梯度源 `a=b(x)h(t)`）`direction_readability` < τ_fail 或 NaN。任一破 → 停。**endpoint 轴角误差是报告字段、不是增量 1 gate**（其精度 gate = Task5 单元 + 增量 2 vs θ_EE）——避免 §3.5/§10 两套口径。
- **增量 2 门**（单端踢、单模板；方向 = §3.5 增量 2 主门 = endpoint-centroid 轴 vs θ_EE，2D montage）：θ_EE 旋转→方向跟转(误差<25°) **且** 整组 2D montage 旋转→方向不变 **且** isotropic+aligned-shaft must-fail(各向异性比<1.3) **且** 有效事件 ≥ n_event_min 否则 INSUFFICIENT。**forward/reverse swap 不在本门**（延后增量，§7）。
- 所有提议阈值（τ_pass=Spearman 0.9 / τ_fail=0.3 / 25° / 各向异性比 1.3 / **参与触点门槛=7（=2·k_dir+1，锁，不设 4）** / n_event_min=待 SNN 事件产率定 / D3 tie tolerance Δt + 最小时间分辨率 / §4.1 timing 阈 f=0.5 + participation margin（L=3 SNN=0.5 / rate=0.10 §4.2 / **cm-SNN traveling-front=0.10 §4.3，已 sensitivity 锁 2026-06-10**） / §3.5 ε_deg=0.5×触点间距；**k_dir=3 已锁**）**在 spec review 锁定**，之后不随结果调整；锁定值同步写回本 §10 + §1 D3 + §3.5 + §4.1 + §4.3 + §5/§6。

## 11. 输出目录

```
results/topic4_sef_hfo/observation_layer/
├── figures/
│   └── README.md            ← 必须存在，中文，逐图说明 (AGENTS.md 规范)
├── increment1_toywave/      ← 玩具波合同门结果 + 通过/失败 JSON
├── increment2_snn_slice/    ← SNN 低驱动竖切结果 + 真实 pipeline 裁决 JSON
└── cohort_summary.json      ← 索引
```

## 12. 自检清单

- [x] D1 接入层 = 事件 + 通道时序（无载波）；D2 几何统一接口（参数化先 / 真实几何 stub）；D3 lag = first-crossing + tie + 最小分辨率（预注册）；D4 顺序 = 玩具波门 → SNN 竖切；D5 rate field = 工程 parity，科学客户等异质核
- [x] 模块 = 小纯函数（montage/sampler/extractor/artifact），适配器薄住模型旁
- [x] **artifact 写盘合同分两层**：内部名 vs 真实 loader **legacy key**（`lagPatRank`/`eventsBool`/`lagPatRaw`/`chnNames`/`start_t` + 同步 `*_packedTimes_withFreqCent.npy` + `<record>_montage.json` 坐标）；核对自 `interictal_propagation.py` L344–397；phantom-mask + 坐标顺序==chnNames 断言
- [x] §4 first-crossing vs centroid 已知不对称预注册、锁定、给出非-bug 理由
- [x] §4.1 锁「越哪个阈」：participation = 噪底+margin（决定 bools）/ timing = per-contact 相对自身峰值 f（决定 lag，非全局绝对阈）；C2 试金石
- [x] §3.5 方向估计子 **每增量一个主门**（增量 1 = rank-vs-投影 Spearman；增量 2 = endpoint-centroid 轴 vs θ_EE），**k_dir=3 / ε_deg / 无向轴 mod180° / degenerate 条件全预注册**；场梯度拟合只作 cross-check；读出 montage ≥2 非平行杆（2D）
- [x] 增量 1 两条 must-fail 用 `direction_readability`（max-axis Spearman）< τ_fail：C1 **居中对称** montage 径向源（偏心读成梯度=采样伪迹、不算失败）+ C2 **严格无到达梯度源** `a(x,t)=b(x)h(t)`（全 tie→NaN）
- [x] scope：增量 1 只验 artifact+lag→rank+方向估计子（单模板无 swap）；**增量 2 = 单端踢、只做单模板方向跟踪**；两端各踢 swap = 延后增量 + **强制读出口径**（不当自然机制）
- [x] 增量 2 = pathology-mapping spec §7 Track 连接的「透过虚拟电极 + 真实 pipeline」实现 + 2D montage + 采样密度 INSUFFICIENT gate
- [x] 六道科学坑（循环 / 杆造假 / 模板→方向转换 / 时间估计含越哪个阈 / 稀疏 / 强制读出当自然机制）逐条锁防护
- [x] 验收门编码结论非仅存在性；阈值 spec review 锁定不随结果调
- [x] 与 pathology-mapping spec（§5.0-④/§7）+ framework（§6.5 Step 2/4、§8.1/§8.4 红线）一致、不冲突

---

## 13. 观测真实性合同（electrode scale + figure discipline + LFP 两档，2026-06-07 user lock）

源于 user 的电极尺度锚（"虚拟 SEEG 应是真实电极间距；real 和 model 保持一致，成本很低"）+ 由此暴露的尺度不匹配（SNN 引擎是 2mm 介观 patch、一个 3.5mm 触点比整个 patch 还大）。五条锁定：

**13.1 真实电极尺度锁（先锁，real↔model 一致）**
- 虚拟电极**触点间距 = 真实 SEEG 间距**：Yuquan **3.5 mm**；Epilepsiae（UKLFR mount）间距**待从真实触点坐标算**（`src.seeg_coord_loader` 取一被试同杆相邻触点欧氏距离；典型 SEEG depth 3.5 或 5 mm）。real 分析与 model 观测用**同一间距**。
- **尺度不匹配的处理（两基底，各自原生尺度）**：
  - **rate field = 主观测基底**：原生 cm 尺（`sef_hfo_lif._DEFAULT_L=12`，Step-0 跑 L=12–16 mm），是 PDE、便宜；3.5 mm 间距在 12–16 mm 片上自然成立（数触点/杆×多杆→≥7）。Increment-3a 升为观测主路。
  - **SNN**：2mm 介观 = sub-electrode（不可用 3.5mm 电极读）；按 user 指示**先放大到数十 mm（几 cm）**作同尺度 mechanism+observation，build 更重，**后面再调**。SNN @2mm 仍作密集-oracle 机制检查（`anisotropy_front`，已 44.9°）。

**13.2 figure discipline（每张"声称 SEEG 读出"的图必须可诊断）**
- 不需重做所有历史图；但**任何声称"SEEG 读出方向/模板"的图**必须叠加：**模型平面 + 踢点 + 传播方向 + shaft/contact 位置 + contact spacing + sheet size**。否则读者无法区分"读出失败"是 **模型问题 / 估计子问题 / 电极摆位问题**（本轮 SNN smoke 的混淆正源于此）。

**13.3 per-contact read-out 面板（让链条可见）**
- 每个触点一条**时间轨迹**，旁标 contact 位置 + 所属 shaft；再展示由这些轨迹导出的 **lag / rank / endpoint axis**。使"**模型事件 → 电极观测 → 方向估计**"整条链在图上可见，不是黑箱出一个角度。

**13.4 LFP 两档（命名纪律）**
- **快版（当前）= 虚拟触点 firing envelope**（平滑放电密度，`snn_event_envelope`）——**必须显式标注"不是 LFP"**；只验方向顺序。
- **正式版 = current-based / field-based 读出**（突触电流 forward model，推广 `engine/lfp.py` |I_E|+|I_I| 至 montage 触点）——**只有这版才叫 LFP**，才适合与真实 SEEG **波形**幅度/形状对齐。两版别混名。

**13.5 不变量**：13.1–13.4 是观测**真实性 / 可诊断性**合同，不改 §3.5 估计子（endpoint-centroid，sparse-friendly）、§10 验收阈（25°/k_dir=3/τ_fail，永不调）、§-1 reframe claim（连接定无向轴）。
