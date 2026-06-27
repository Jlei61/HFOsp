from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.import_yuquan_electrode_csv import (
    ImportErrorWithContext,
    build_chn_xyz_dict,
    import_electrode_csv,
    read_electrode_csv,
)
from src.seeg_coord_loader import load_subject_coords


def _write_electrode_csv(path: Path, rows: list[dict[str, object]]) -> None:
    header = [
        "Label",
        "ElectrodeName",
        "ContactIndex0",
        "ContactNumber",
        "Status",
        "PositionStatus",
        "LocationType",
        "tkrRAS_X",
        "tkrRAS_Y",
        "tkrRAS_Z",
        "MNI305_X",
        "MNI305_Y",
        "MNI305_Z",
        "MNI152_X",
        "MNI152_Y",
        "MNI152_Z",
    ]
    lines = [",".join(header)]
    for row in rows:
        lines.append(",".join(str(row.get(col, "")) for col in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows() -> list[dict[str, object]]:
    out = []
    for shaft, base in (("A", 0.0), ("B", 10.0)):
        for i in range(1, 4):
            out.append(
                {
                    "Label": f"{shaft}{i}",
                    "ElectrodeName": shaft,
                    "ContactIndex0": i - 1,
                    "ContactNumber": i,
                    "Status": "detected" if i == 1 else "interpolated",
                    "PositionStatus": "detected" if i == 1 else "interpolated",
                    "LocationType": "sEEG",
                    "tkrRAS_X": base + i,
                    "tkrRAS_Y": base + i * 2,
                    "tkrRAS_Z": base + i * 3,
                    "MNI305_X": base + i + 100,
                    "MNI305_Y": base + i * 2 + 100,
                    "MNI305_Z": base + i * 3 + 100,
                    "MNI152_X": base + i + 200,
                    "MNI152_Y": base + i * 2 + 200,
                    "MNI152_Z": base + i * 3 + 200,
                }
            )
    return out


def test_build_chn_xyz_dict_groups_contiguous_contacts(tmp_path):
    csv_path = tmp_path / "electrodes.csv"
    _write_electrode_csv(csv_path, _rows())

    rows = read_electrode_csv(csv_path)
    chn = build_chn_xyz_dict(rows)

    assert sorted(chn) == ["A", "B"]
    assert chn["A"].shape == (3, 3)
    np.testing.assert_allclose(chn["A"][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(chn["B"][2], [13.0, 16.0, 19.0])


def test_import_writes_loader_compatible_chnxyzdict(tmp_path):
    csv_path = tmp_path / "electrodes.csv"
    _write_electrode_csv(csv_path, _rows())

    output_root = tmp_path / "patients_elecs_reGen"
    metadata = import_electrode_csv(
        subject="synth",
        csv_path=csv_path,
        output_root=output_root,
        subject_edf_dir=tmp_path / "missing_edf_dir",
    )

    assert (output_root / "synth" / "chnXyzDict.npy").exists()
    assert (output_root / "synth" / "chnXyzDict.import_metadata.json").exists()
    assert metadata["n_contacts"] == 6

    res = load_subject_coords(
        "yuquan",
        "synth",
        ["A1", "A1-A2", "B3"],
        yuquan_root=output_root,
    )
    assert res.coord_space == "fs_native_ras_mm"
    assert int(res.mapped_mask_in_requested_order.sum()) == 3
    np.testing.assert_allclose(res.coords_array_in_requested_order[1], [1.5, 3.0, 4.5])

    sidecar = json.loads(
        (output_root / "synth" / "chnXyzDict.import_metadata.json").read_text()
    )
    assert sidecar["pipeline_coord_space"] == "fs_native_ras_mm"
    assert sidecar["written"]["chnXyzDict"].endswith("chnXyzDict.npy")
    assert sidecar["post_write_loader_check"]["coord_units"] == "mm"


def test_duplicate_label_is_rejected(tmp_path):
    csv_path = tmp_path / "electrodes.csv"
    rows = _rows()
    rows[1]["Label"] = rows[0]["Label"]
    with pytest.raises(ImportErrorWithContext, match="duplicate Label"):
        _write_electrode_csv(csv_path, rows)
        read_electrode_csv(csv_path)


def test_noncontiguous_contacts_are_rejected(tmp_path):
    csv_path = tmp_path / "electrodes.csv"
    rows = [row for row in _rows() if row["Label"] != "A2"]
    _write_electrode_csv(csv_path, rows)

    with pytest.raises(ImportErrorWithContext, match="contiguous"):
        build_chn_xyz_dict(read_electrode_csv(csv_path))
