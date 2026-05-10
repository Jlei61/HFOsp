"""Tests for scripts/build_yuquan_block_inventory.py.

Uses monkeypatched probes — no real EDFs touched.
"""
from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.build_yuquan_block_inventory import (
    BlockProbeResult,
    probe_one_edf,
    write_block_inventory_csv,
)


def test_probe_one_edf_returns_expected_fields(tmp_path: Path):
    fake_edf = tmp_path / "FAKE001.edf"
    fake_edf.write_bytes(b"")  # presence only; mocked probe never reads

    with patch("scripts.build_yuquan_block_inventory.read_edf_start_time", return_value=1605835000.0), \
         patch("scripts.build_yuquan_block_inventory._probe_edf_metadata", return_value={
             "duration_sec": 7200.0,
             "sample_rate": 2000.0,
             "n_channels_total": 147,
         }):
        result = probe_one_edf("gaolan", fake_edf)

    assert result.subject == "gaolan"
    assert result.recording_id == "FAKE001"
    assert result.block_id == "FAKE001"
    assert result.block_stem == "FAKE001"
    assert result.block_start_epoch == 1605835000.0
    assert result.block_end_epoch == 1605835000.0 + 7200.0
    assert result.duration_sec == 7200.0
    assert result.sample_rate == 2000.0
    assert result.n_channels_total == 147
    assert result.edf_path == str(fake_edf)
    assert result.data_path == str(fake_edf)
    assert result.head_path == ""


def test_write_block_inventory_csv_round_trip(tmp_path: Path):
    rows = [
        BlockProbeResult(
            subject="gaolan",
            recording_id="FA0013KQ",
            block_id="FA0013KQ",
            block_stem="FA0013KQ",
            block_start_epoch=1605829619.0,
            block_end_epoch=1605836819.0,
            duration_sec=7200.0,
            sample_rate=2000.0,
            n_channels_total=130,
            head_path="",
            data_path="/mnt/yuquan_data/yuquan_24h_edf/gaolan/FA0013KQ.edf",
            edf_path="/mnt/yuquan_data/yuquan_24h_edf/gaolan/FA0013KQ.edf",
        ),
    ]
    out_csv = tmp_path / "yuquan_block_inventory.csv"
    write_block_inventory_csv(rows, out_csv)

    with open(out_csv) as f:
        read_rows = list(csv.DictReader(f))

    assert len(read_rows) == 1
    assert read_rows[0]["subject"] == "gaolan"
    assert float(read_rows[0]["block_start_epoch"]) == 1605829619.0
    assert read_rows[0]["block_id"] == "FA0013KQ"
