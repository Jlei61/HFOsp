#!/usr/bin/env python3
"""Export an Epilepsiae subject's MNI-grid T1 and SEEG coordinates.

The CSV is produced through the repository's canonical coordinate loader.
For Epilepsiae, raw SQL coordinates are MRI voxel IJK; the loader applies the
subject MRI affine and returns MNI152 1-mm world coordinates in millimetres.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np


def _find_repo_root(script_path: Path) -> Path:
    for candidate in script_path.resolve().parents:
        if (candidate / "src" / "seeg_coord_loader.py").is_file():
            return candidate
    raise RuntimeError(
        "This export script must run inside an HFOsp checkout containing "
        "src/seeg_coord_loader.py"
    )


ROOT = _find_repo_root(Path(__file__))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.seeg_coord_loader import (  # noqa: E402
    MNI152_1MM_AFFINE,
    MNI152_1MM_SHAPE,
    enumerate_subject_all_channels,
    load_subject_coords,
)


def _natural_channel_key(name: str) -> tuple[str, int, str]:
    match = re.fullmatch(r"([A-Za-z]+\'?)\s*(\d+)", name.strip())
    if match is None:
        return (name.upper(), -1, name)
    return (match.group(1).upper(), int(match.group(2)), name)


def _split_channel(name: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"([A-Za-z]+\'?)\s*(\d+)", name.strip())
    if match is None:
        return (name, None)
    return (match.group(1), int(match.group(2)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _convert_analyze_to_nifti(source_img_path: Path, output_path: Path) -> None:
    source = nib.load(str(source_img_path))
    data = np.asanyarray(source.dataobj)
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D T1 (or singleton 4D), got shape {data.shape}")
    if tuple(int(v) for v in data.shape) != MNI152_1MM_SHAPE:
        raise ValueError(
            f"MRI shape {data.shape} does not match MNI152 1-mm grid "
            f"{MNI152_1MM_SHAPE}"
        )
    affine = np.asarray(source.affine, dtype=float)
    if not np.allclose(affine, MNI152_1MM_AFFINE, atol=1e-3):
        raise ValueError("MRI affine does not match the canonical MNI152 1-mm affine")

    converted = nib.Nifti1Image(data, affine)
    # Code 2 means aligned anatomy. The grid is MNI152, while the exact historic
    # warp implementation remains unverified because no warp field was shipped.
    converted.set_qform(affine, code=2)
    converted.set_sform(affine, code=2)
    nib.save(converted, str(output_path))


def _write_csv(output_path: Path, result: object, *, subject_short_id: str) -> None:
    fieldnames = [
        "subject_short_id",
        "subject_canonical_id",
        "electrode_name",
        "shaft",
        "contact_number",
        "mni_x_mm",
        "mni_y_mm",
        "mni_z_mm",
        "coord_space",
        "coord_units",
        "world_coordinate_convention",
        "normalization_certainty",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for name, coord in zip(
            result.channel_names_requested,
            result.coords_array_in_requested_order,
            strict=True,
        ):
            shaft, contact_number = _split_channel(name)
            writer.writerow(
                {
                    "subject_short_id": subject_short_id,
                    "subject_canonical_id": result.subject_id,
                    "electrode_name": name,
                    "shaft": shaft,
                    "contact_number": contact_number,
                    "mni_x_mm": f"{float(coord[0]):.6f}",
                    "mni_y_mm": f"{float(coord[1]):.6f}",
                    "mni_z_mm": f"{float(coord[2]):.6f}",
                    "coord_space": result.coord_space,
                    "coord_units": result.coord_units,
                    "world_coordinate_convention": "RAS+",
                    "normalization_certainty": result.normalization_certainty,
                }
            )


def _write_readme(
    output_path: Path,
    *,
    subject_short_id: str,
    subject_canonical_id: str,
    contact_count: int,
) -> None:
    output_path.write_text(
        f"""# Epilepsiae {subject_short_id} MNI spatial bundle

本目录包含 Epilepsiae subject {subject_short_id}（canonical database ID: {subject_canonical_id}）的 T1 和全部颅内电极坐标。

## 文件

- `epilepsiae_1146_T1_mni152_1mm.nii.gz`：数据库分发的 skull-stripped subject T1，从 Analyze `.img/.hdr` 无空间变换地转换为压缩 NIfTI；shape 为 182×218×182，1 mm isotropic，保留原 affine。
- `epilepsiae_1146_electrodes_mni152_1mm.csv`：全部 {contact_count} 个 invasive contacts 的 MNI152 1-mm world coordinates。
- `manifest.json`：来源路径、affine、坐标合同和文件 SHA-256。
- `export_epilepsiae_mni_bundle.py`：生成本 bundle 的脚本副本。

## 坐标合同

SQL 中的原始电极坐标是配套 MRI 的 voxel IJK，不是直接的 RAS mm。CSV 由仓库 canonical `src.seeg_coord_loader` 生成：应用 MRI affine 后输出 `coord_space=mni152_1mm`、单位 mm、world convention 为 RAS+。

不要再手工翻转 x，也不要额外施加 CT→T1 变换。本地分发中没有找到术后 CT、CT→T1 变换或原始 MNI warp field。当前正规化证据标签为 `grid_confirmed_warp_type_unverified`：MNI152 grid 已确认，但历史 warp 的具体实现不可重建。
""",
        encoding="utf-8",
    )


def export_bundle(subject_id: str, output_dir: Path, expected_contacts: int | None) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_names = sorted(
        enumerate_subject_all_channels("epilepsiae", subject_id),
        key=_natural_channel_key,
    )
    result = load_subject_coords("epilepsiae", subject_id, channel_names)

    if result.coord_space != "mni152_1mm" or result.coord_units != "mm":
        raise ValueError(
            f"Unexpected coordinate contract: {result.coord_space}/{result.coord_units}"
        )
    if not bool(np.all(result.mapped_mask_in_requested_order)):
        missing = [entry.channel for entry in result.missing]
        raise ValueError(f"Not all invasive contacts mapped: {missing}")
    if expected_contacts is not None and len(channel_names) != expected_contacts:
        raise ValueError(
            f"Expected {expected_contacts} contacts, found {len(channel_names)}"
        )

    subject_short_id = (
        result.subject_id[:-2] if result.subject_id.endswith("02") else str(subject_id)
    )
    source_img_path = Path(result.provenance["affine_path"])
    t1_path = output_dir / f"epilepsiae_{subject_short_id}_T1_mni152_1mm.nii.gz"
    csv_path = output_dir / f"epilepsiae_{subject_short_id}_electrodes_mni152_1mm.csv"
    readme_path = output_dir / "README.md"
    script_copy_path = output_dir / Path(__file__).name
    manifest_path = output_dir / "manifest.json"

    _convert_analyze_to_nifti(source_img_path, t1_path)
    _write_csv(csv_path, result, subject_short_id=subject_short_id)
    _write_readme(
        readme_path,
        subject_short_id=subject_short_id,
        subject_canonical_id=result.subject_id,
        contact_count=len(channel_names),
    )
    shutil.copy2(Path(__file__).resolve(), script_copy_path)

    files = [t1_path, csv_path, readme_path, script_copy_path]
    manifest = {
        "schema_version": "epilepsiae_mni_bundle_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "subject_short_id": subject_short_id,
        "subject_canonical_id": result.subject_id,
        "contact_count": len(channel_names),
        "coord_space": result.coord_space,
        "coord_units": result.coord_units,
        "world_coordinate_convention": "RAS+",
        "source_coord_type": result.source_coord_type,
        "normalization_certainty": result.normalization_certainty,
        "source_sql": result.provenance["source_path"],
        "source_mri_img": str(source_img_path.resolve()),
        "source_mri_hdr": str(source_img_path.with_suffix(".hdr").resolve()),
        "mni152_1mm_shape": list(MNI152_1MM_SHAPE),
        "mni152_1mm_affine": MNI152_1MM_AFFINE.tolist(),
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in files
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    zip_path = output_dir.parent / f"{output_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in [*files, manifest_path]:
            archive.write(path, arcname=f"{output_dir.name}/{path.name}")
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="1146")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-contacts", type=int)
    args = parser.parse_args()

    output_dir = args.output_dir or (
        ROOT / "exports" / f"epilepsiae_{args.subject}_mni_bundle"
    )
    expected_contacts = args.expected_contacts
    if expected_contacts is None and args.subject == "1146":
        expected_contacts = 114
    zip_path = export_bundle(args.subject, output_dir.resolve(), expected_contacts)
    print(f"Bundle directory: {output_dir.resolve()}")
    print(f"ZIP archive: {zip_path.resolve()}")


if __name__ == "__main__":
    main()
