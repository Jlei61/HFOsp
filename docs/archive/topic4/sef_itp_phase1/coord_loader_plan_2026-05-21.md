# SEEG Coord Loader PR — Plan-of-Record v3.1（landed 2026-05-21）

> **状态**：**已实现 + 49 unit tests GREEN + 27-subject real-data smoke pass**
> **版本历史**：v1 (early plan) → v2 (strict schema 2026-05-21 早) → v3 (MNI152 auto-discovery 2026-05-21 中) → **v3.1 (canonical ID + loud-fail MRI miss 2026-05-21 晚)**
> **范围**：`src/seeg_coord_loader.py` —— SEEG 通道 3D 坐标 reader，Yuquan → fs_native_ras_mm；Epilepsiae 自动发现 MRI → mni152_1mm
> **上游 framework**：`docs/topic4_sef_itp_framework.md` v1.0.2 + plan_2026-05-21.md
> **数据合同源**：`docs/topic0_methodology_audits.md` §3.2.4（v3.1）—— 单一权威 schema
> **不属于**：H1/H2 实跑、coord 主分析数字、Phase 0a 重跑修复、Epilepsiae normalization_certainty 升级（找 warp field 文档归 future PR）

---

## 0. 一句话承诺（v3.1 落地版）

实现了 `src/seeg_coord_loader.py`：Yuquan 输出 `fs_native_ras_mm`（subject-native T1 RAS mm），Epilepsiae 自动发现 MNI152 grid MRI + 应用 affine 输出 `mni152_1mm`（mm，cohort-comparable）。10 条 invariant 锁住科学污染口（NULL 显式 missing / bipolar 两端必备 / 顺序锚点 / 空间合同 / canonical ID 防 cross-patient 串号 / MRI 默认 loud fail / affine byte-精确匹配 MNI152）。49 unit tests + 27-subject real-data smoke 全 GREEN。

## 0.1 v3.1 critical fixes（2026-05-21 晚，用户 audit 触发）

v3（早）首版有两个静默科学污染口，已在 v3.1 修复：

1. **短 ID cross-patient pollution**：旧版 fuzzy glob `*<id>*.sql` 让 `subject_id='115'` 静默匹配 pat_115002 SQL（与 pat_11502 MRI 来自不同病人）。
   - 修：加 `_canonicalize_epilepsiae_subject_id()` —— 规则 `<id>02 if not ending in '02'`（例：`'115' → '11502'`, `'1150' → '115002'`）
   - SQL glob 改 **精确** `pat_<canonical>_*.sql`；0 或 >1 → raise
   - MRI 搜 **精确** `pat_<canonical>/`；与 SQL canonical 必须一致

2. **MRI miss 静默降级 voxel**：旧版 MRI 没找到时默认返回 voxel sensitivity 模式，subject id 拼错 / mount 漏挂的错误会悄悄通过。
   - 修：默认 MRI miss → `FileNotFoundError` raise
   - 需要 voxel 模式必须显式 `allow_voxel_fallback=True`

3. **回归测试**：5 个新 case 锁死 cross-patient 不会再发生（`115 ≠ 115002`, `1150 ≠ 11502`, `620 ≠ 86202`, ambiguous SQL match raises, MRI miss default raises）

---

## 1. 为什么是 v2 严格合同

用户审阅 2026-05-21 指出 v1 合同（schema 主要给 `{channel_name: coord}` dict）有三个静默污染口：

1. **dict → array 转换错位**：Phase 1 代码吃 ordered numpy array。如果 loader 输出 dict，下游必须重排，**任何 sort 错位 = 端点距离静默错误**
2. **坐标空间不标识**：Yuquan 是 mm，Epilepsiae 是 voxel —— 不显式标会让消费侧默默把 voxel 当 mm 算距离
3. **bipolar 一端缺失允许只用一端**：会让缺一端的 bipolar 通道得到错误坐标

v2 合同把三条堵死。

---

## 2. 数据源（已确认）

### 2.1 Yuquan

- **位置**：`/mnt/yuquan_data/yuquan_images/nii格式及点电极坐标/caseAndMRI/yuquan_24h_mriCT/patients_elecs_reGen/<subject>/`
- **主文件**：`chnXyzDict.npy` — Python pickle dict，key = shaft 前缀（A / B / ...），value = (n_contacts, 3) numpy 数组
- **备用**：`<shaft>.dat` — ASCII 文本，每行 `x y z`
- **坐标空间**：`fs_native_ras_mm`（subject-native FreeSurfer / T1 RAS，单位 mm，**非 MNI**）
- **判定证据**：见 `topic0_methodology_audits.md` §3.2.1（CT→T1 rigid only / useRealRAS 1 header / 3.501 mm pitch / inv(mri_affine) 反推 / 含负值）
- **覆盖**：20 个 subject 中 11 个有目录；stable_k=2 cohort 8/16 全量映射

### 2.2 Epilepsiae

- **位置**：`/mnt/epilepsia_data/all_data_sqls/pat_*.sql`
- **表**：`electrode (id, array, name, moniker, focus_rel, invasive, supplier, coord_x, coord_y, coord_z, commentary)`
- **坐标空间**：`mri_native_voxel_ijk`（subject-native MRI voxel index，**非 mm，非 MNI**）
- **判定证据**：见 `topic0_methodology_audits.md` §3.2.2（整数 .0 / 范围 12-200 voxel-like / legacy readme 明写 ijk / legacy 用 aff_matrix 转 native）
- **覆盖**：20 个 subject 全有 SQL；stable_k=2 cohort 14/18 全量映射；NULL 通道带 `Mikrokontakt / nicht abgrenzbar` commentary

### 2.3 MRI affine（Epilepsiae voxel → mm 转换需要）

- **未确认源**：SQL `files` 表有 `mri_*.img/.hdr`（Analyze 格式 header），理论上含 affine
- **本 PR 范围**：loader 接受 `mri_affine: Optional[np.ndarray] = None` 参数；如果调用方提供则做 voxel→mm 转换，否则原样输出 voxel
- **MRI affine 实际推导**：归 future PR（先不阻塞主分析能跑 Yuquan）

---

## 3. 模块布局

```
src/seeg_coord_loader.py
├── # === 主入口 ===
├── load_subject_coords(dataset, subject_id, channel_names_requested,
│                       mri_affine=None) -> CoordResult
│
├── # === 数据源 readers ===
├── _read_yuquan_chnXyzDict(subject_id) -> Dict[shaft_prefix, np.ndarray]
├── _read_epilepsiae_electrode_sql(subject_id) -> List[ElectrodeRow]
│
├── # === 通道名解析 + 匹配 ===
├── _parse_channel_to_query_keys(channel_name) -> ChannelQuery
│     # bipolar 'HLA1-HLA2' -> {"left": ("HLA", 1), "right": ("HLA", 2), "is_bipolar": True}
│     # monopolar 'GC4'    -> {"left": ("GC", 4),  "right": None,        "is_bipolar": False}
│
├── # === bipolar midpoint 严格组合 ===
├── _resolve_bipolar(left_coord, right_coord) -> Optional[np.ndarray]
│     # 两端都 found → mean；任一端 None → None（整对 missing）
│
├── # === affine 转换 ===
├── _apply_affine(voxel_coord, mri_affine) -> ras_coord
│     # ras_h = aff @ [x, y, z, 1]; return ras_h[:3]
│
└── # === schema validation ===
    └── _validate_coord_result(result) -> None
        # 抛 ValueError 如果违反 v2 invariant

tests/test_seeg_coord_loader.py  (TDD)
├── # === schema invariants ===
├── test_output_array_aligned_to_requested_order
├── test_mapped_mask_marks_missing_channels_false
├── test_coord_space_yuquan_is_fs_native_ras_mm
├── test_coord_space_epilepsiae_default_is_voxel_ijk
├── test_coord_space_epilepsiae_with_affine_becomes_ras_mm
├── test_provenance_records_absolute_paths
│
├── # === channel parsing ===
├── test_parse_monopolar_channel
├── test_parse_bipolar_channel
├── test_parse_primed_shaft
│
├── # === bipolar midpoint contract ===
├── test_bipolar_midpoint_both_endpoints_required
├── test_bipolar_partial_endpoint_marks_pair_missing
│
├── # === NULL handling ===
├── test_sql_null_coord_marks_missing_with_reason
├── test_commentary_mikrokontakt_recorded
│
├── # === ordering invariant ===
├── test_channel_order_preserved_under_dict_shuffle
├── test_array_row_index_matches_requested_index
│
└── # === error paths ===
    ├── test_unknown_dataset_raises
    ├── test_subject_not_found_raises
    └── test_voxel_passed_to_mm_consumer_raises  # cross-cutting w/ Phase 1
```

---

## 4. 输出 schema（v2 lock）

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal
import numpy as np


@dataclass
class CoordResult:
    """SEEG 3D coord loader output (v2 strict schema, 2026-05-21).

    Aligned with topic0_methodology_audits.md §3.2.4 v2 strict invariants.
    """
    schema_version: str = "coord_loader_v2"
    dataset: Literal["yuquan", "epilepsiae"] = ...
    subject_id: str = ...

    # === ORDER ANCHOR ===
    # channel_names_requested + coords_array_in_requested_order +
    # mapped_mask_in_requested_order must all be aligned by index.
    # Any reordering breaks v2 invariant #5.
    channel_names_requested: List[str] = ...
    coords_array_in_requested_order: np.ndarray = ...   # shape (n_requested, 3)
                                                        # unmapped rows = NaN
    mapped_mask_in_requested_order: np.ndarray = ...    # shape (n_requested,) bool

    # === SPACE / UNITS (v2 invariant #6) ===
    coord_space: Literal[
        "fs_native_ras_mm",          # Yuquan default
        "mri_native_voxel_ijk",      # Epilepsiae default
        "ras_mm_via_affine",         # Epilepsiae with mri_affine applied
    ] = ...
    coord_units: Literal["mm", "voxel"] = ...

    # === PROVENANCE ===
    provenance: Dict[str, Any] = ...   # source_path, affine_path?, loader_version

    # === MISSING ===
    missing: List[MissingEntry] = ...

    # === BIPOLAR RESOLUTION (only when bipolar channels present) ===
    bipolar_resolution: Optional[Dict[str, BipolarRes]] = None


@dataclass
class MissingEntry:
    channel: str
    reason: Literal[
        "sql_null",
        "name_not_found",
        "bipolar_partial_endpoint",
        "commentary_mikrokontakt",
        "commentary_nicht_abgrenzbar",
        "commentary_other",
    ]
    index_in_requested: int
    commentary: Optional[str] = None


@dataclass
class BipolarRes:
    left_endpoint: str
    right_endpoint: str
    left_coord: Optional[Tuple[float, float, float]]
    right_coord: Optional[Tuple[float, float, float]]
    midpoint_strategy: Literal["mean_both_required"] = "mean_both_required"
```

---

## 5. 严格 invariant 测试用例

每条 invariant 必须有至少一个测试覆盖。**TDD：测试先写**。

### Invariant 1: NULL 显式 missing

```python
def test_sql_null_coord_marks_missing_with_reason():
    """Epilepsiae SQL 中 NULL coord 必须显式列入 missing[]，coords_array_in_requested_order
    对应行为 NaN，mapped_mask 对应位置为 False。"""
    # synthetic SQL with one channel coord = NULL + commentary 'nicht abgrenzbar!'
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="synthetic_with_null",
        channel_names_requested=["GC4-GC5", "HLA5-HLA6"],  # HLA5 has NULL coord
    )
    # HLA5-HLA6 should be missing because HLA5 has NULL
    null_entries = [m for m in result.missing if m.channel == "HLA5-HLA6"]
    assert len(null_entries) == 1
    assert null_entries[0].reason == "bipolar_partial_endpoint"
    # Array row for missing channel is NaN
    idx = result.channel_names_requested.index("HLA5-HLA6")
    assert np.all(np.isnan(result.coords_array_in_requested_order[idx]))
    assert result.mapped_mask_in_requested_order[idx] == False
```

### Invariant 2: Bipolar midpoint 两端必须都 found

```python
def test_bipolar_partial_endpoint_marks_pair_missing():
    """Bipolar 'HLA1-HLA2'：如果 HLA1 找到、HLA2 NULL，整对 missing 不允许只用 HLA1。"""
    result = load_subject_coords(
        dataset="epilepsiae",
        subject_id="synthetic_partial",
        channel_names_requested=["HLA1-HLA2"],
    )
    # both endpoints required
    bp = result.bipolar_resolution["HLA1-HLA2"]
    assert bp.midpoint_strategy == "mean_both_required"
    assert bp.left_coord is not None
    assert bp.right_coord is None  # HLA2 NULL
    # → entire pair missing
    assert result.mapped_mask_in_requested_order[0] == False
    assert len([m for m in result.missing if m.channel == "HLA1-HLA2"]) == 1
```

### Invariant 5: 顺序锚点

```python
def test_channel_order_preserved_under_dict_shuffle():
    """如果用户请求 ['B1-B2', 'A1-A2']（B 在前），输出数组也必须 [B coord, A coord]，
    不允许 loader 内部按 shaft 字母排序后再返回。"""
    requested = ["B1-B2", "A1-A2"]  # B first
    result = load_subject_coords("yuquan", "chengshuai", requested)
    assert result.channel_names_requested == requested
    # row 0 is B1-B2 midpoint, row 1 is A1-A2 midpoint
    # synthetic data should have B at x≈10, A at x≈0
    assert result.coords_array_in_requested_order[0, 0] != result.coords_array_in_requested_order[1, 0]


def test_array_row_index_matches_requested_index():
    """For every channel name in channel_names_requested, the array row
    at the same index MUST be that channel's coord (or NaN if missing)."""
    requested = [f"A{i+1}-A{i+2}" for i in range(5)]
    result = load_subject_coords("yuquan", "chengshuai", requested)
    for i, name in enumerate(result.channel_names_requested):
        assert name == requested[i]  # order preserved
        if result.mapped_mask_in_requested_order[i]:
            assert not np.any(np.isnan(result.coords_array_in_requested_order[i]))
        else:
            assert np.all(np.isnan(result.coords_array_in_requested_order[i]))
```

### Invariant 6: 空间合同

```python
def test_coord_space_yuquan_is_fs_native_ras_mm():
    result = load_subject_coords("yuquan", "chengshuai", ["A1-A2"])
    assert result.coord_space == "fs_native_ras_mm"
    assert result.coord_units == "mm"


def test_coord_space_epilepsiae_default_is_voxel_ijk():
    result = load_subject_coords("epilepsiae", "108402", ["GC4-GC5"])
    assert result.coord_space == "mri_native_voxel_ijk"
    assert result.coord_units == "voxel"


def test_coord_space_epilepsiae_with_affine_becomes_ras_mm():
    fake_affine = np.eye(4) * 0.5  # synthetic 0.5 mm/voxel
    fake_affine[3, 3] = 1.0
    result = load_subject_coords(
        "epilepsiae", "108402", ["GC4-GC5"], mri_affine=fake_affine,
    )
    assert result.coord_space == "ras_mm_via_affine"
    assert result.coord_units == "mm"
    assert result.provenance["affine_path"] is not None or "affine" in result.provenance.get("loader_version", "")
```

### Phase 1 cross-cutting 合同

```python
def test_voxel_coord_rejected_by_phase1_mm_consumers():
    """compute_h1_compactness / compute_h2_spatial_reversal 接受 coords 时必须
    断言 coord_units=='mm'。voxel 必须 raise。"""
    # Wrapped in coord-aware version (future Phase 1 update):
    voxel_coord_result = load_subject_coords("epilepsiae", "108402", ["GC4-GC5"])
    # When phase1 wraps these:
    with pytest.raises(ValueError, match="coord_units must be 'mm'"):
        run_phase1_with_coords(voxel_coord_result, ...)  # to be implemented
```

---

## 6. 实施步骤（TDD 顺序）

```
Step 1: tests/test_seeg_coord_loader.py — 写 12-15 个测试 (synthetic data)
        全 RED
Step 2: src/seeg_coord_loader.py — 实现 _parse_channel_to_query_keys
        (Step 1 中 channel parsing 测试 GREEN)
Step 3: 实现 _resolve_bipolar + bipolar midpoint 测试 GREEN
Step 4: 实现 _read_yuquan_chnXyzDict + Yuquan schema 测试 GREEN
Step 5: 实现 _read_epilepsiae_electrode_sql + Epilepsiae schema 测试 GREEN
Step 6: 实现 _apply_affine + affine 转换测试 GREEN
Step 7: 实现 load_subject_coords 整合 + 顺序锚点测试 GREEN
Step 8: _validate_coord_result 内部 invariant check 测试 GREEN
Step 9: Phase 1 cross-cutting: 在 src/sef_itp_phase1.py 的 compute_h1_compactness /
        compute_h1_descriptive / compute_h2_spatial_reversal 加 coord_units 断言
        (新增 phase1 test asserting voxel rejected)
Step 10: 文档化 results/coord_loader/per_subject/<sid>.json 输出路径 (optional cache)
```

---

## 7. Out of scope（本 PR 不做）

- ❌ MRI affine 从 Analyze header / nii 文件**推导** —— future PR；本 PR 接受 affine 作为外部输入
- ❌ Yuquan reGen 配准空间的二次验证（用户 2026-05-21 已确认 fs_native_ras_mm）
- ❌ Epilepsiae voxel size 一致性核查（本 PR 不假设；调用方决定怎么处理）
- ❌ Coord 主分析数字（本 PR 只读坐标，不跑 H1 / H2）
- ❌ Cortical surface mesh / SC distance（远 future）
- ❌ Cross-subject MNI normalization（永远不做：subject-level 分析）
- ❌ git commit / 真实 cohort 落地数据（autonomous 块禁止）

---

## 8. 红线

1. **禁止把 voxel 当 mm 用** —— 任何这么做的代码必须 raise
2. **禁止顺序错位** —— `coords_array_in_requested_order` 必须严格按 `channel_names_requested` index 对齐
3. **禁止 bipolar 只用一端** —— 任一端缺整对 missing
4. **禁止用默认值（0 / NaN-但-mask-True / centroid）填 NULL** —— missing 必须显式
5. **禁止跨 subject 配准** —— subject-level
6. **禁止假设 voxel size 一致** —— 必须 explicit affine
7. **禁止保留 v1 dict-only schema** —— 本 PR 直接给 v2 ordered-array

---

## 9. 自检清单

- [x] 数据源 §2 已确认两套（Yuquan + Epilepsiae SQL）
- [x] Schema v2 锁定（§4）
- [x] Invariants 1-7 每条有测试用例 §5
- [x] TDD 实施顺序 §6
- [x] Phase 1 cross-cutting 合同（coord_units 断言）
- [x] Out of scope §7
- [x] 红线 §8
- [x] CLAUDE.md §8 大白话风格（codename 仅作括号补注）

---

## 10. 一句话承诺

写 `src/seeg_coord_loader.py`，7 条严格 invariant + 12+ TDD 测试。Yuquan 给 mm，Epilepsiae 给 voxel + 显式标识；任何顺序错位 / 单位错位 / NULL 默填都让代码 raise，不让科学污染静默发生。完成后 SEF-ITP Phase 1 H1 三层 + H2 spatial reversal 可在 Yuquan stable_k=2 ~8 subject 上启动；Epilepsiae 待 MRI affine 可得后入主分析（在此之前是 sensitivity）。
