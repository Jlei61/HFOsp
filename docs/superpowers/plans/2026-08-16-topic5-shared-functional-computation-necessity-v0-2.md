# Topic 5.2 跨网络共同功能响应 necessity v0.2 — 执行记录

## Phase 0：P0 修复合同 — 完成

- [x] 定位 v0.1 泄漏：完整留出事件 future-field coordinate 进入删除中心与 support gate。
- [x] 方向、事件、剂量、对照、终点和 patient-first 统计保持不变。
- [x] 删除中心固定为训练阶段 phase curve；旧 `event_u` / `conditional_center` 读取后立即丢弃。
- [x] 增加 revision、target-free centre、support 和浮点重放自动检查。

## Phase 1：训练侧算子与留一方向 — 完成

- [x] 630/630 checkpoint 训练侧组织小片→未来触点响应表完成。
- [x] 168/168 “用另外三种网络定义、测试第四种网络”的方向冻结完成。
- [x] 新旧第一共同方向几乎相同：absolute cosine 中位 0.999993，最小 0.995747。
- [x] 模型与 decoder 参数 hash 630/630 不变。

## Phase 2：留出事件单方向删除 — 完成

- [x] 504/504 真实顺序网络单元完成，0 failure。
- [x] 699,925 个 state-family-dose 分支、1,602,882 个延迟决策通过共同 support。
- [x] 最大 reference replay error 5.45e-6，小于冻结容忍度 1e-5。
- [x] 共同方向、正交、高方差、打乱后半段网络方向和不删除基线全部同状态比较。

## Phase 3：累计前 1/2/3 维删除 — 完成

- [x] 504/504 单元完成，0 cell failure。
- [x] 3,310,720 个延迟决策通过 support。
- [x] rank 1/2/3、三档剂量、共同/打乱/高方差三类分支完成。
- [x] 修复分片并发汇总竞争：分片只写单元，最终单进程统一聚合。

## Phase 4：统计与审计 — 完成

- [x] reference → seed → fit → patient 聚合，患者分母 28。
- [x] 三个 primary 检验 Holm 校正；四种待测网络分别报告。
- [x] 单方向 audit 20/20 PASS；累计删除 audit 11/11 PASS。
- [x] 27 项相关回归测试通过。
- [x] 即时和早/中/晚结果仅作 sensitivity，不覆盖 primary。

## Phase 5：图与报告 — 完成

- [x] Primary 裁定 `NECESSITY_UNSUPPORTED`，Figure 6 主图不改。
- [x] 生成 v0.2 两面板补充图，PNG/PDF/SVG 同状态目视检查通过。
- [x] v0.1 spec/plan/report 标为 superseded；新增 v0.2 spec、执行记录和收口报告。
- [x] 主 closeout、Topic 5 archive index 与 Figure README 同步。
