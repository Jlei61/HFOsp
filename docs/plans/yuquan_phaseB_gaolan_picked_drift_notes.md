# gaolan picked-channel drift 根因笔记 (Phase B 调研)

> 时间: 2026-04-17
> 范围: gaolan / Phase B L1 调研
> 来源: 直接对比 `results/hfo_detection/gaolan/_refineGpu.npz` vs `/mnt/yuquan_data/yuquan_24h_edf/gaolan/_refineGpu.npz`

## 关键事实


| 指标                  | 老 detector | 新 detector              | 倍数    |
| ------------------- | ---------- | ----------------------- | ----- |
| 通道数                 | 120        | 130 (含未配对的 raw bipolar) | —     |
| events_count 总数     | 25,170     | 75,990                  | 3.02x |
| events_count mean   | 209.8      | 584.5                   | 2.79x |
| events_count std    | 338.3      | 1,039.8                 | 3.07x |
| events_count max    | 1,322      | 4,924                   | 3.72x |
| pick_k              | 1.9        | 1.9                     | 同     |
| 阈值 = mean + 1.9·std | 852.5      | 2,560.2                 | —     |
| picked 通道数          | 12         | 9                       | —     |


## 通道间放大倍数严重不均


| 通道族                | new/leg ratio |
| ------------------ | ------------- |
| `B'13, B'14, A'10` | 5.9–6.0x      |
| `C1, C2, C3`       | 3.2–3.5x      |
| `B2, B3, C4, C5`   | 2.2–2.4x      |
| `D/D'` 系列          | 1.1–1.4x      |


D / D' 系列在老 detector 里事件数排名 6–12（被 picked），在新 detector 里因为别的通道涨得更猛，相对位次跌出 picked。

## 老 picked vs 新 picked 对比

老 12 个：`C3 C2 C1 B2 B3 D'4 D'2 C5 D'3 D1 D2 C4`
新 9 个：`B'13 B'14 C1 C2 C3 A'10 A9 B2 B3`

- C 系列（C1–C3）和 B 系列（B2 B3）**两边都进**
- C4 / C5 老有，新阈值抬高后**跌出**
- D / D' 系列**全部跌出**新 picked
- 新增 `B'13 B'14 A'10 A9`（绝对 events_count 最高）

## 这意味着什么

1. 这**不是** refine 算法 bug。两边 refine 都是同一规则 `pick_idx = where(counts > mean + pick_k * std)`。
2. 这**不是** alias bug。zero alias collisions。
3. 这是 **detector 灵敏度结构在通道间非均匀放大**，被 mean+k·std 阈值法二次放大成 picked set 的剧变。

mean+k·std 阈值是相对于全脑通道分布的，对 std 敏感。新 detector 的 std 涨了 3 倍，门槛也跟着抬高。如果某个通道的 events_count 没跟上"全员涨 3 倍"的速度（比如 D 系列只涨 1.3 倍），它就会被相对剔除。

## Plan 怎么处理

`docs/plans/yuquan_lagpat_backfill_validation.plan.md` 第 §149-178 行明确：

> 本轮必须复刻旧 contract，不允许借机"修正"它。
> 如果要改 contract，那是另一个 PR，必须全 cohort 重算，不准夹带私货。

所以**不能**为了让 Phase B 通过去：

- 改 pick_k
- 改阈值规则成 fixed / percentile / top-k
- 强行 alias 到老 picked 集

可以做的是：**把这件事如实写进 Phase B drift 报告**，说"当 pick_k=1.9 这个阈值在新 detector 上选出的 picked 集与老 picked 集 Jaccard < 0.5 的 subject，端到端漂移不可控"，然后让 Topic 1/2 决策方决定：

- 是否接受 backfill subjects 的 picked 集与老 cohort 不直接同构
- 是否在 Topic 1/2 报告里把 41-subject 严格只用作 extended-cohort sensitivity，不替换 30-subject 主结论

## 还要看的事

- dongyiming / wangyiyang 的 picked 漂移幅度（first block 看 Jaccard 已经分别 0.86 / 0.88，远好于 gaolan）
- 8 个无 legacy lagPat 的 backfill subject，新 detector 的 events_count 总数和 std 是否也是 3x 量级 → 决定 backfill 后 picked 集稳定性
- gaolan 的 detector 重跑能否复现：是 detector 算法变了，还是 detector 参数（rel_thresh / abs_thresh / band 等）跟老不一致

