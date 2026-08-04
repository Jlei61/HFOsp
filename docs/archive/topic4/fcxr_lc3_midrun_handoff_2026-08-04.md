# FCXR-LC3 运行中接管说明

写于 2026-08-04，提交 `93816262`，分支 `codex/topic4-fcxr-lc3`。
工作目录 `/home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2`。

## 一、现在有什么在跑

全部 `setsid nohup` 脱离会话，父进程为 1，不依赖任何交互会话。

| PID | 角色 | 说明 |
|---|---|---|
| 2358230 | E4 侦察 autopilot | 当前阶段 |
| 2358239 | E5 空间响应 autopilot | 等 2358230 结束 |
| 2358256 | E6/E7 标定与生命周期 autopilot | 等 2358239 |
| 2358269 | 收尾 autopilot | 等 2358256 |

接管第一步（**不要用 `pgrep -f`，会匹配到自己**）：

```bash
cd /home/honglab/leijiaxin/HFOsp/.worktrees/topic4-fcxr-lc2
ps -eo pid,ppid,sid,etimes,pcpu,rss,stat,cmd | grep fcxr_lc3 | grep -v grep
tail -n 40 results/topic4_sef_hfo/fcxr_lc3_dx_spatial_instability/dynamic_reconnaissance/nohup_autopilot.log
free -g && git status --short --branch
```

**不要因为接管而重启或杀掉它们。** 同机可能有其它 worktree 的仿真在跑，**绝不能碰**。

## 二、已经完成并验收的

- **E0 精确分叉**：中途存盘再续跑与一路跑到底逐位相同，三次全过。
- **E1 空间磨损场回放**：三条（两种强度 × 两个种子）标量逐点对齐。磨损沿两核之间**整条走廊**富集，核心/轴外 = 2.6 倍，跨种子稳（跨种子只可比区域/分布，逐细胞比较被审计显式禁用）。
- **四个准备态**：低态（稀疏放小尖波、10 秒 9 次可回落）、高态（`FINITE_HIGH_FIXED`、贴不应期天花板细胞 0%、正反馈变量斜率约 0）、H6 低态**注入哨兵**（不是自然平衡态，禁止写成"H6 低分支"）、H6 高态。
- **102 行冻结几何地图**（`geometry_map.json`，`GEOMETRY_DONE.json` 哈希已核对）：
  - 低起点自发点火 **0/42**
  - 高起点存活 **18/42**，边界是一条**完美竖线**：X≥0.80 存活、X≤0.65 掉回，**六档磨损没挪动一格**
  - 唯一例外 `Dmax_aX0p65_high` = `ELEVATED_EVENT_TRAIN`
  - 强直饱和 / 正反馈衰减未决 / 数值不安全 **各 0**，不应期天花板占比全场最大 **0.0000**
  - **关键陷阱**：`INTERICTAL_WORKPOINT` 只有上限没有下限，X≤0.80 的四列其实是**熄灭**（0.04–0.06 Hz），不是"稀疏小尖波"。所以"到处没点火"只在 X=1.00/0.90 两列有信息量。
  - 磨损的真实作用在**发放率**上：X 全开时 2.81→5.82 Hz 单调升；X=0.90 时健康组织熄灭（峰值 1.9 Hz）而磨损组织仍放出完整小尖波（峰值 53–61 Hz）。
- **慢流探针 12 落点**（`slow_vector_field/slow_vector_field.json`）：
  - **平均 X 漂移为正是假象**——85% 的轴外细胞在恢复，掩盖了核心内的消耗。
  - 区域分解后：X=0.80 一侧核心内 X **被消耗**，且**磨损越重消耗越深**（核心 B：健康 −0.013 → 最大磨损 −0.042，三倍单调）。这是全程唯一一处磨损在正确方向上的单调机制贡献。
  - 高态期间磨损**均匀累积**，轴外略高于核心 → 会**抹平**原有 2.6 倍走廊对比。

## 三、当前允许与禁止的表述

允许：
- 有限高发放态存在且稳健，**不需要磨损**（完全健康组织照样撑得住）。
- 高态能否维持**只由 X 决定**，磨损不移动这条界。
- 冻结几何层面**未观察到**低→高入口。

禁止：
- 不得称"双稳已确立"——存活的 18 格里 12 格只有 1.5 秒证据，只有 X=0.80 那 6 格是 5 秒证据。
- 不得给概率或 50% 等值线（每格单微状态单噪声，`probability_contours_authorized=false` 硬编码）。
- 不得说"够得着"——冻结几何 ≠ 真实轨迹可达，那是 E4 在答。
- **"终止与恢复互斥"只在均匀 X 坐标下成立**；真实 X 场非均匀（核心消耗、周围恢复），是否也互斥要等 E4 采到真实 X 场。

## 四、E4 判读要点（正在跑）

三条不打外力的轨迹（噪声 401/405/406），最少 32 秒、最多 45 秒；20 秒记录起始搜索状态。
**第一条就回答核心问题**（自己会不会烧起来），405/406 是噪声重复。

不可写成成功的形态：runaway、不应期强直平台、16 Hz 全局共同模态、kick 触发的高态、短暂 trough、只有终止没有回归的小尖波、冻结态几何冒充完整生命周期。

产物：`dynamic_reconnaissance/recon_noise{401,405,406}.json` + `aggregate.json`（需 `status=COMPLETE` 才解锁 E5）。
**行级断点续跑已存在**（`_run_once`：输出+完成标记齐全、状态 COMPLETE、行 ID 与源码锁提交号一致、哈希对得上才复用）。

## 五、本轮修掉的工程问题（都已提交 + 补测）

1. `cmd_manifest` 打印语句关键字撞名 → 102 行清单写盘后崩溃。提成 `geometry_manifest_summary()`，回归测试钉死原写法。
2. **四处** `KeyError: 'rng'`：`build_substrate` 不建噪声发生器，而地图 worker / 空间响应 / X 标定 / 慢流各自都要步进。统一到 `install_registered_noise_rng()`。加了**结构守卫测试**，用 glob 自动枚举 `scripts/*topic4_fcxr_lc3*.py`（**不要改回手写清单——慢流就是这样漏掉的**）。白名单两处，均有证据：`spatial.cmd_lock`（只算参考电流）、`geometry.cmd_field_audit`（只取区域掩膜和场统计）、`run_topic4_fcxr_lc3.py::_replay_family`（经 LC1 运行器步进，那边第 151 行自己按显式种子装发生器）。
3. `cmd_prepare` 无断点续跑 → 重跑会覆盖约 1.7 小时的准备态。加 `prepared_state_is_reusable()`。
4. worker 上限写死 2 → 改为按**实测**单行峰值定容并**每轮跟随内存重算**（`MAX_MAP_WORKERS=8`、`EXTENDED_ROW_RSS_SCALE=2.0`）。内存模型四次实测全对（1.5 s→6.79 / 5.0 s→8.66 / 5.06 s→8.69 / 10 s→11.74 GiB）。**加 worker 不改结果**：每行续跑时从存档整个恢复随机数状态（`topic4_fcxr_lc3.py:263`），已有测试用两个不同构造种子跑出逐位相同的输出。
5. **墙钟守卫定值系统性过小**：E4 被 18000 秒守卫杀掉（退出码 143，零完成行）。按实测单价（长轨迹全程存脉冲约 320–450 秒/仿真秒）重定：recon 18000→108000、spatial 28800→108000、X 标定 43200→86400、生命周期 57600→180000。几何守卫 64800 够用未动。

## 六、未完成 / 待办

1. **慢流缺注册判读标签**：计划 §4 要求输出 `DX_GEOMETRIC_PATH_PRESENT / _ABSENT / DX_DYNAMIC_VECTOR_MISALIGNED / DX_MAP_UNRESOLVED` 四选一，全库 grep **无此常量**。数据已冻结在 12 行里，可事后推导，**判读时必须用区域分解而非平均值**。等流水线静默再做（避免重锁扰动）。
2. **计划偏差记录**：计划正文写 H6 哨兵"交错穿插"，实际清单是 84 主行在前、18 哨兵在后**顺序执行**。对结果无影响（行独立、确定性、已逐行复现），但要写进归档偏差清单。
3. **进程池常驻内存**：`ProcessPoolExecutor` 按池大小拉起进程后不回收，常驻 = 池大小 × 衬底（8 × 5.5 GiB ≈ 44 GiB），而定容公式算的是活跃 worker 数。本轮内存充裕未构成风险，需写进归档。
4. **溯源分段**：地图 102 行产于锁 `824e56e0`，慢流产于 `4778cd73`。因慢流运行器属几何锁的 11 个源之一，修它必然移动锁的提交号；地图已完成故**未重跑**（重跑要 4.5 小时且零科学收益）。
5. 图 / `figures/README.md`（中文、目视后写）/ archive doc / 全量 pytest 收尾。
6. 用户 2026-08-04 指示：注册程序跑完后**按核心科学目标补跑实验**。设计需等 E4 结果落地——当前链条断在"低→高有没有入口"，补什么完全取决于这个答案。

## 七、纪律

- 每到一个 gate 先回答四问：这个结果回答哪个科学问题？是真分支/真生命周期，还是初值、阈值、分类器或短窗造成的假象？当前允许声称什么？哪些阶段因此解锁、哪些仍不得执行？
- 不在 40k 结果上临时调参；不重新设计模型；预注册的停机条件触发就停，保留失败产物，**禁止调阈值救活**。
- 出现真实工程 bug：确认相关仿真已退出 → 保留失败产物 → 声明旧锁作废 → 修复+补测+重新生成锁 → 只重跑受污染阶段。
- 失败产物一律改名 `*.superseded.json` 保留，不删。
- 全部测试：
  ```bash
  /home/honglab/leijiaxin/anaconda3/bin/python -m pytest -q \
    tests/test_topic4_fcxr_lc3.py tests/test_topic4_fcxr_lc3_geometry.py \
    tests/test_topic4_fcxr_lc3_slowflow.py tests/test_topic4_fcxr_lc3_recon.py \
    tests/test_topic4_fcxr_lc3_spatial.py tests/test_topic4_fcxr_lc3_xcal.py \
    tests/test_topic4_fcxr_lc3_finalize.py
  ```
  当前 63 个通过。
