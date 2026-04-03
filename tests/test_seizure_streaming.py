"""Unit tests for seizure detectors and interval matching."""
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch
from src.preprocessing import (
    _flag_to_runs,
    _merge_close_runs,
    _robust_z,
    _stream_edf_channel_ll,
    _stream_epilepsiae_channel_ll,
    match_seizure_intervals,
    detect_seizure_by_spatial_extent,
    detect_seizure_by_spatial_extent_epilepsiae,
    detect_seizure_onsets_from_data,
)


class TestFlagToRuns:
    def test_empty(self):
        assert _flag_to_runs(np.array([], dtype=bool)) == []

    def test_single_run(self):
        flag = np.array([False, True, True, True, False])
        assert _flag_to_runs(flag) == [(1, 4)]

    def test_multiple_runs(self):
        flag = np.array([True, True, False, True, False, False, True])
        runs = _flag_to_runs(flag)
        assert runs == [(0, 2), (3, 4), (6, 7)]

    def test_trailing_run(self):
        flag = np.array([False, False, True, True])
        assert _flag_to_runs(flag) == [(2, 4)]


class TestMergeCloseRuns:
    def test_no_runs(self):
        assert _merge_close_runs([], np.array([]), 1.0) == []

    def test_far_apart(self):
        t = np.arange(100, dtype=np.float64)
        runs = [(0, 5), (50, 55)]
        result = _merge_close_runs(runs, t, 10.0)
        assert result == [(0, 5), (50, 55)]

    def test_close_merge(self):
        t = np.arange(100, dtype=np.float64)
        runs = [(0, 5), (8, 15)]
        result = _merge_close_runs(runs, t, 5.0)
        assert result == [(0, 15)]


class TestMatchSeizureIntervals:
    def test_empty_both(self):
        m = match_seizure_intervals([], [])
        assert m["recall"] == 1.0
        assert m["tp"] == []

    def test_all_fn(self):
        m = match_seizure_intervals([(0, 10), (20, 30)], [])
        assert len(m["fn"]) == 2
        assert m["recall"] == 0.0

    def test_all_fp(self):
        m = match_seizure_intervals([], [(5, 15)])
        assert len(m["fp"]) == 1
        assert m["precision"] == 0.0

    def test_perfect_match(self):
        m = match_seizure_intervals([(10, 20)], [(10, 20)])
        assert len(m["tp"]) == 1
        assert m["recall"] == 1.0
        assert m["precision"] == 1.0
        assert m["tp"][0]["onset_err"] == 0.0
        assert m["tp"][0]["offset_err"] == 0.0

    def test_overlap_match(self):
        m = match_seizure_intervals([(10, 20)], [(15, 25)])
        assert len(m["tp"]) == 1
        assert m["tp"][0]["onset_err"] == pytest.approx(5.0)
        assert m["tp"][0]["offset_err"] == pytest.approx(5.0)

    def test_no_overlap(self):
        m = match_seizure_intervals([(10, 20)], [(30, 40)])
        assert len(m["fn"]) == 1
        assert len(m["fp"]) == 1

    def test_multi_event(self):
        manual = [(10, 20), (50, 60), (100, 110)]
        detected = [(12, 22), (55, 65)]
        m = match_seizure_intervals(manual, detected)
        assert len(m["tp"]) == 2
        assert len(m["fn"]) == 1
        assert len(m["fp"]) == 0


class TestDetectSeizureFromSynthetic:
    """Integration test with synthetic data fed to detect_seizure_onsets_from_data."""

    def _make_seizure_data(self, sfreq=500, dur_sec=120, sz_on=50, sz_off=70, n_ch=10):
        n = int(dur_sec * sfreq)
        rng = np.random.RandomState(42)
        data = rng.randn(n_ch, n) * 0.5
        on_samp = int(sz_on * sfreq)
        off_samp = int(sz_off * sfreq)
        data[:, on_samp:off_samp] += rng.randn(n_ch, off_samp - on_samp) * 10.0
        return data, sfreq

    def test_detects_large_seizure(self):
        data, sfreq = self._make_seizure_data()
        res = detect_seizure_onsets_from_data(data, sfreq, ll_k=4.0, rms_k=4.0)
        assert len(res["onsets_sec"]) >= 1
        assert res["onsets_sec"][0] < 55.0
        assert res["offsets_sec"][-1] > 65.0

    def test_no_false_alarm_on_noise(self):
        rng = np.random.RandomState(99)
        data = rng.randn(10, 60000) * 0.5
        res = detect_seizure_onsets_from_data(data, 500.0, ll_k=6.0, rms_k=6.0)
        assert len(res["onsets_sec"]) == 0


class TestStreamEdfChannelLL:
    def test_streams_bipolar_ll_from_records(self, tmp_path):
        edf_path = tmp_path / "toy.edf"
        header = bytes(256)
        # 3 monopolar channels (A1, A2, A3), 4 samples each per record
        # rec0: A1=[1,2,4,7] A2=[0,0,0,0] A3=[1,1,1,1]
        rec0 = np.array([1, 2, 4, 7, 0, 0, 0, 0, 1, 1, 1, 1], dtype="<i2").tobytes()
        # rec1: A1=[7,7,7,7] A2=[1,3,6,10] A3=[2,2,2,2]
        rec1 = np.array([7, 7, 7, 7, 1, 3, 6, 10, 2, 2, 2, 2], dtype="<i2").tobytes()
        edf_path.write_bytes(header + rec0 + rec1)

        fake_header = {
            "header_n_bytes": 256,
            "n_records": 2,
            "record_duration": 1.0,
            "record_total_bytes": 24,
            "seeg_idx": [0, 1, 2],
            "seeg_labels": ["A1", "A2", "A3"],
            "n_seeg": 3,
            "spr": 4,
            "sample_offsets": np.array([0, 4, 8], dtype=np.int64),
            "gains": np.array([1.0, 1.0, 1.0], dtype=np.float64),
            "offsets": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "sfreq": 4.0,
        }

        with patch("src.preprocessing._parse_edf_header_for_streaming", return_value=fake_header):
            ll, record_duration, sfreq, n_channels, labels = _stream_edf_channel_ll(edf_path)

        assert record_duration == pytest.approx(1.0)
        assert sfreq == pytest.approx(4.0)
        assert n_channels == 2
        assert labels == ["A1-A2", "A2-A3"]
        assert ll.shape == (2, 2)
        # rec0: bipolar A1-A2 = [1,2,4,7]-[0,0,0,0] = [1,2,4,7] → LL=|1|+|2|+|3|=6
        #        bipolar A2-A3 = [0,0,0,0]-[1,1,1,1] = [-1,-1,-1,-1] → LL=0
        # rec1: bipolar A1-A2 = [7,7,7,7]-[1,3,6,10] = [6,4,1,-3] → LL=|−2|+|−3|+|−4|=9
        #        bipolar A2-A3 = [1,3,6,10]-[2,2,2,2] = [-1,1,4,8] → LL=|2|+|3|+|4|=9
        np.testing.assert_allclose(ll, np.array([[6.0, 9.0], [0.0, 9.0]]))


class TestSpatialExtentDetector:
    def test_detects_widespread_recruitment(self, tmp_path):
        ch_ll = np.zeros((6, 10), dtype=np.float64)
        ch_ll[:, 3:7] = 10.0
        edf_path = tmp_path / "fake.edf"
        edf_path.write_bytes(b"")

        with patch(
            "src.preprocessing._stream_edf_channel_ll",
            return_value=(ch_ll, 1.0, 1000.0, 6, [f"A{i+1}" for i in range(6)]),
        ):
            res = detect_seizure_by_spatial_extent(
                edf_path,
                per_channel_k=3.0,
                min_active_frac=0.5,
                min_duration_sec=2.0,
                merge_gap_sec=0.0,
            )

        np.testing.assert_allclose(res["onsets_sec"], np.array([3.0]))
        np.testing.assert_allclose(res["offsets_sec"], np.array([7.0]))
        np.testing.assert_allclose(res["participation"][3:7], np.ones(4))

    def test_rejects_single_channel_artifact(self, tmp_path):
        ch_ll = np.zeros((6, 10), dtype=np.float64)
        ch_ll[0, 2:6] = 10.0
        edf_path = tmp_path / "fake.edf"
        edf_path.write_bytes(b"")

        with patch(
            "src.preprocessing._stream_edf_channel_ll",
            return_value=(ch_ll, 1.0, 1000.0, 6, [f"A{i+1}" for i in range(6)]),
        ):
            res = detect_seizure_by_spatial_extent(
                edf_path,
                per_channel_k=3.0,
                min_active_frac=0.5,
                min_duration_sec=2.0,
                merge_gap_sec=0.0,
            )

        assert res["onsets_sec"].size == 0
        assert res["participation"].max() == pytest.approx(1.0 / 6.0)

    def test_onset_tracks_progressive_recruitment_threshold_crossing(self, tmp_path):
        ch_ll = np.zeros((4, 12), dtype=np.float64)
        ch_ll[0, 4:7] = 10.0
        ch_ll[1, 5:7] = 10.0
        ch_ll[2, 6:7] = 10.0
        ch_ll[3, 7:8] = 10.0
        edf_path = tmp_path / "fake.edf"
        edf_path.write_bytes(b"")

        with patch(
            "src.preprocessing._stream_edf_channel_ll",
            return_value=(ch_ll, 1.0, 1000.0, 4, [f"A{i+1}" for i in range(4)]),
        ):
            res = detect_seizure_by_spatial_extent(
                edf_path,
                per_channel_k=3.0,
                min_active_frac=0.75,
                min_duration_sec=1.0,
                merge_gap_sec=0.0,
            )

        np.testing.assert_allclose(res["onsets_sec"], np.array([6.0]))
        np.testing.assert_allclose(res["offsets_sec"], np.array([7.0]))
        np.testing.assert_allclose(
            res["participation"],
            np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 0.25, 0.0, 0.0, 0.0, 0.0]),
        )


class TestStreamEpilepsiaeChannelLL:
    def test_streams_intracranial_bipolar_ll_from_raw_block(self, tmp_path):
        head_path = tmp_path / "107300102_0000.head"
        data_path = tmp_path / "107300102_0000.data"
        head_path.write_text(
            "\n".join(
                [
                    "start_ts=2009-07-06 18:54:09.000",
                    "num_samples=8",
                    "sample_freq=4",
                    "conversion_factor=1",
                    "num_channels=4",
                    "duration_in_sec=2",
                    "sample_bytes=2",
                    "elec_names=[A1, A2, A3, ECG]",
                ]
            )
        )
        # Interleaved by sample: [A1, A2, A3, ECG] × 8 samples.
        samples = np.array(
            [
                [1, 0, 1, 9],
                [2, 0, 1, 9],
                [4, 0, 1, 9],
                [7, 0, 1, 9],
                [7, 1, 2, 9],
                [7, 3, 2, 9],
                [7, 6, 2, 9],
                [7, 10, 2, 9],
            ],
            dtype="<i2",
        )
        data_path.write_bytes(samples.reshape(-1).tobytes())

        ll, record_duration, sfreq, n_channels, labels = _stream_epilepsiae_channel_ll(
            data_path,
            head_path,
        )

        assert record_duration == pytest.approx(1.0)
        assert sfreq == pytest.approx(4.0)
        assert n_channels == 2
        assert labels == ["A1-A2", "A2-A3"]
        assert ll.shape == (2, 2)
        np.testing.assert_allclose(ll, np.array([[6.0, 9.0], [0.0, 9.0]]))

    def test_detector_reuses_same_core_for_epilepsiae_blocks(self, tmp_path):
        ch_ll = np.zeros((6, 10), dtype=np.float64)
        ch_ll[:, 3:7] = 10.0
        data_path = tmp_path / "fake.data"
        head_path = tmp_path / "fake.head"
        data_path.write_bytes(b"")
        head_path.write_text("")

        with patch(
            "src.preprocessing._stream_epilepsiae_channel_ll",
            return_value=(ch_ll, 1.0, 1000.0, 6, [f"A{i+1}" for i in range(6)]),
        ):
            res = detect_seizure_by_spatial_extent_epilepsiae(
                data_path,
                head_path,
                per_channel_k=3.0,
                min_active_frac=0.5,
                min_duration_sec=2.0,
                merge_gap_sec=0.0,
            )

        np.testing.assert_allclose(res["onsets_sec"], np.array([3.0]))
        np.testing.assert_allclose(res["offsets_sec"], np.array([7.0]))

    def test_sustained_active_mask_filters_short_runs(self):
        """_sustained_active_mask should keep only runs ≥ min_consec."""
        from src.preprocessing import _sustained_active_mask
        raw = np.array([
            [False, True, True, False, True, True, True, True, False, True],
            [False, False, True, True, True, True, True, True, True, False],
        ])
        result = _sustained_active_mask(raw, min_consec=3)
        expected = np.array([
            [False, False, False, False, True, True, True, True, False, False],
            [False, False, True, True, True, True, True, True, True, False],
        ])
        np.testing.assert_array_equal(result, expected)

    def test_sustained_active_mask_passthrough_when_1(self):
        from src.preprocessing import _sustained_active_mask
        raw = np.array([[True, False, True]])
        np.testing.assert_array_equal(_sustained_active_mask(raw, 1), raw)

    def test_detect_with_consec_reduces_detections(self, tmp_path):
        """min_channel_consec_sec should eliminate brief spikes."""
        n_ch, n_rec = 8, 100
        ch_ll = np.ones((n_ch, n_rec), dtype=np.float64) * 0.5
        ch_ll[:, 10:14] = 50.0
        ch_ll[:, 50:65] = 50.0

        data_path = tmp_path / "test.data"
        head_path = tmp_path / "test.head"
        data_path.write_bytes(b"\x00")
        head_path.write_text("")

        from unittest.mock import patch
        with patch(
            "src.preprocessing._stream_edf_channel_ll",
            return_value=(ch_ll, 1.0, 1000.0, n_ch, [f"A{i+1}" for i in range(n_ch)]),
        ):
            res_no = detect_seizure_by_spatial_extent(
                data_path, per_channel_k=3.0, min_active_frac=0.5,
                min_duration_sec=3.0, merge_gap_sec=1.0, min_channel_consec_sec=0.0,
            )
            res_yes = detect_seizure_by_spatial_extent(
                data_path, per_channel_k=3.0, min_active_frac=0.5,
                min_duration_sec=3.0, merge_gap_sec=1.0, min_channel_consec_sec=5.0,
            )
        assert len(res_no["onsets_sec"]) == 2
        assert len(res_yes["onsets_sec"]) == 1

    def test_rejects_non_integer_sample_frequency(self, tmp_path):
        head_path = tmp_path / "bad.head"
        data_path = tmp_path / "bad.data"
        head_path.write_text(
            "\n".join(
                [
                    "start_ts=2009-07-06 18:54:09.000",
                    "num_samples=5",
                    "sample_freq=2.5",
                    "conversion_factor=1",
                    "num_channels=2",
                    "duration_in_sec=2",
                    "sample_bytes=2",
                    "elec_names=[A1, A2]",
                ]
            )
        )
        data_path.write_bytes(np.zeros((5, 2), dtype="<i2").reshape(-1).tobytes())

        with pytest.raises(ValueError, match="Non-integer sample_freq"):
            _stream_epilepsiae_channel_ll(data_path, head_path)
