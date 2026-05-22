# Module Refactor Summaries (Phase 1 preprocessing + Phase 2 hfo_detector + Module 5 visualization) — Archived 2026-05-22

> **归档说明**：合并 `docs/DEVELOP_PLAN.md` §4 旧 L454–539（Phase 1 preprocessing 重构 2026-01-30）、L553–624（Phase 2 hfo_detector 重构 2026-01-31）、L1855–1879（Module 5 visualization 重构 2026-01-16）三段 postmortem。
>
> 这些是已完成的 refactor 总结，内容主要是 "Before/After" code snippets + 性能测试结果。代码已 merge 进 `src/`，本归档只为历史溯源保留设计决策。
>
> **不要从本归档抄 API 签名** — 实际 API 以 `src/preprocessing.py` / `src/hfo_detector.py` / `src/visualization.py` 当前代码为准。

---

## Phase 1 重构总结（preprocessing.py, 2026-01-30）

**核心改进：消除重复的条件判断**

```python
# Before: 散落各处的 GPU 判断
if self.use_gpu and HAS_GPU:
    return self._apply_filters_gpu(...)  # 70+ lines
else:
    return self._apply_filters_cpu(...)  # 30+ lines

# After: 初始化时决定后端，运行时零判断
self.filter_backend = GpuFilterBackend() if use_gpu else CpuFilterBackend()

def _apply_filters(self, data, sfreq):
    if self.notch_freqs:
        data = self.filter_backend.apply_notch(data, sfreq, self.notch_freqs)
    if self.bandpass:
        data = self.filter_backend.apply_bandpass(data, sfreq, *self.bandpass)
    return data
```

**删除的废代码**（零破坏性）：
- ❌ `PreBipolarDetector` (15行) - 空壳类，总是返回 False
- ❌ `validate_against_gpu_results()` (12行) - 运行时抛异常的废函数
- ❌ `exclude_last_n` 参数 - 从所有类和函数签名中移除

**新增的架构组件**：

```python
# 抽象接口
class FilterBackend:
    def apply_notch(data, sfreq, freqs): ...
    def apply_bandpass(data, sfreq, low, high): ...

# CPU 实现（scipy filtfilt）
class CpuFilterBackend(FilterBackend): ...

# GPU 实现（CuPy + chunked processing）
class GpuFilterBackend(FilterBackend):
    def __init__(self, chunk_sec=20.0): ...
    # 自动处理 GPU OOM - 20s chunks + reflect padding
```

**扩展新后端的方法**：

```python
# 例如：添加 Apple Metal 支持
class MetalFilterBackend(FilterBackend):
    def apply_notch(self, data, sfreq, freqs):
        # 使用 PyTorch MPS 或 Metal Performance Shaders
        ...

    def apply_bandpass(self, data, sfreq, low, high):
        ...

# 使用：
preprocessor = SEEGPreprocessor(
    use_gpu=False,  # 不使用 CUDA
    # 手动替换 backend（高级用法）
)
preprocessor.filter_backend = MetalFilterBackend()
```

**收益**：
- ✅ 代码行数: 1223 → 1199 (-24行，质量提升显著)
- ✅ GPU 条件分支: 4处 → 0处
- ✅ 可扩展性: 添加新 backend (Metal/ROCm/OpenCL) 只需实现 2 个方法
- ✅ 可测试性: 每个 backend 可独立单元测试
- ✅ 零破坏性: 所有外部调用接口保持向后兼容

### 关键技术决策（preprocessing 历史记录）

- **不再猜**：不再根据"某些 contact 缺失"去推断 EDF 是否已做 bipolar；那是通道选择策略，不是重参考证据。
- **重参考策略（显式）**:
  - `'bipolar'`: 同 shaft 相邻触点差分；**命名为明确的 `A1-A2`**，避免与单极通道混淆
  - `'car'`: 每个 shaft 内部做 CAR
  - `'none'`: 保持 EDF 原始参考（单极/参考电极体系），不做任何推断
- **GPU 通道差异来源（已确认，chengshuai/FC10477Q）**：
  - `*_gpu.npz` 的 `chns_names` 是 EDF 清洗后通道的子集；
  - 差异主要来自**通道选择/剔除策略**（例如每个 shaft 缺少末端若干 contact），并非 bipolar 推断依据；
  - 若要复现 GPU 通道集合，使用 `include_channels=gpu['chns_names']`（显式）。

---

## Phase 2 重构总结（hfo_detector.py, 2026-01-31）

**核心改进：封装 + 并行化（但注意适用场景）**

```python
# Before: 分散的函数调用
env = bqk.return_hil_enve_norm(data, fs, freqband)  # 内部循环K次滤波
events = bqk.find_high_enveTimes(env, ...)

# After: 清晰的类封装 + 可选并行
detector = BQKDetector(sfreq=fs, freqband=(80,250), n_jobs=1)  # 默认串行
env = detector.compute_envelope(data)  # 预计算滤波器系数
events = detector.detect_events(data)  # 端到端检测
```

**删除的代码**（零破坏性，-348行）：
- ❌ `mad_hysteresis` 算法及所有相关函数（~200行）
- ❌ GPU相关代码（`cupy_hilbert`, `_HAS_CUPY`）
- ❌ Ictal检测相关（`_detect_ictal_mask`, `_mad`, `_moving_average`, `_find_runs`）

**性能测试结果**（2026-01-31, cuda_env）：

| 场景 | n_jobs=1 | n_jobs=-1 | 结论 |
|------|----------|-----------|------|
| 小数据 (8ch×10s) | 0.078s | 0.902s | ❌ 并行慢12x (进程开销) |
| 大数据 (16ch×30s) | 0.457s | 1.100s | ❌ 并行慢2.4x |
| 数值一致性 | - | diff=1.81e-10 | ✅ 完美 |

**关键发现（Amdahl's Law陷阱）**：
```
并行开销 = joblib进程创建 (~500ms) + 数据序列化 (~200ms)
计算时间 = K × (滤波+Hilbert) ≈ 0.078s (K=9, 8ch×10s)

当 计算时间 < 并行开销 → 串行更快
```

**并行化适用场景**：
- ✅ **长时程无chunk**：`chunk_sec=None` + 单文件>2分钟 + K>20
- ❌ **默认chunked处理**：30s chunk → 计算 <200ms → n_jobs=1 更快

**推荐配置**：

```python
# 默认配置（推荐）：n_jobs=1
config = HFODetectionConfig(
    band='ripple',
    chunk_sec=30.0,  # 分块处理
    n_jobs=1,        # ← 串行避免进程开销
)

# 全文件处理（特殊场景）：
config = HFODetectionConfig(
    chunk_sec=None,  # ← 整个文件一次性处理
    n_jobs=-1,       # ← 可能有收益（需测试）
)
```

**收益**：
- ✅ 代码: 632行 → 284行 (-55%)
- ✅ 算法: 单一BQK路径
- ✅ 封装: `BQKDetector` 类，滤波器系数预计算
- ✅ 数值: 与原 `bqk_utils.py` 误差 <1e-9
- ⚠️ 并行: **仅长时程场景有效，默认场景反而变慢**

**关键技术决策**:
- **算法纯化**: 只保留 BQK，删除所有非 BQK 代码
- **类封装**: `BQKDetector` 预计算滤波器系数（不再每 chunk 重复）
- **Chunked 处理**: 30s chunk + 1s overlap，避免内存爆炸
- **并行化策略**: 默认 `n_jobs=1`（实测更快），避免 joblib 开销
- **双阈值策略**: `rel_thresh × local_median` ∧ `abs_thresh × global_median`

---

## Module 5 visualization 重构总结（visualization.py, 2026-01-16）

✅ **完成的重构**：
1. ✅ 将 TF 质心计算从 `visualization.py` 迁移到 `group_event_analysis.py`
2. ✅ 添加 `compute_tf_centroids()` 计算 2D TF 质心（时间+频率）
3. ✅ `*_groupAnalysis.npz` 现在包含 `tf_centroid_time` 和 `tf_centroid_freq`
4. ✅ `plot_paper_fig2_*` 重构为读取预计算质心，不再内部计算 STFT
5. ✅ 新增便利函数：
   - `plot_lag_heatmaps_from_group_analysis()` - 一键读取+绘制热图
   - `plot_lag_statistics()` - 统计分析三联图
   - `plot_tf_centroid_statistics()` - TF质心分布统计
6. ✅ `chengshuai_hfo_analysis.ipynb` 重构为纯可视化终端（不做计算）

**架构验证**：
- ✅ visualization.py 不再有任何 STFT/质心计算逻辑
- ✅ 所有复杂计算在 `group_event_analysis.py`
- ✅ notebook 只调用 visualization 函数，不写内联绘图代码
- ✅ 中间结果可复用，避免重复计算

**关键技术决策**:
- **波形颜色**: `tableau_20_no_red` 避免与 HFO 红色标记冲突
- **事件标注**: 默认 `style='tick'` 细线，不遮挡波形
- **TF质心**: 使用 wavelet TFR + 动态基线 log 校正，再在该 TF 图上计算 2D 质心
