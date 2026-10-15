"""Controlled synthetic detector test, separate from every paper result."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from data_processing.src.hfo_detector import HFODetector, HFODetectionConfig
from data_processing.src.group_event_analysis import build_windows_from_detections


def main():
    fs = 800.0
    time = np.arange(6400) / fs
    signal = np.random.default_rng(42).normal(0, .02, (4, len(time)))
    for center in (2.0, 5.0):
        signal += 3 * np.exp(-.5 * ((time-center)/.02)**2) * np.sin(2*np.pi*120*time)
    channels = ["A1", "A2", "B1", "B2"]
    permissive = HFODetector(HFODetectionConfig(chunk_sec=None, side_thresh=None,
                                               n_jobs=1, max_last_ms=None)).detect(signal, sfreq=fs, ch_names=channels)
    assert all(len(x) == 2 for x in permissive.events_by_channel)
    # Filtering broadens these artificial events beyond the 200-ms duration gate.
    assert all(np.all(np.diff(x, axis=1) > .2) for x in permissive.events_by_channel)
    strict = HFODetector(HFODetectionConfig(chunk_sec=None, side_thresh=None,
                                           n_jobs=1, max_last_ms=200)).detect(signal, sfreq=fs, ch_names=channels)
    assert all(len(x) == 0 for x in strict.events_by_channel)
    windows = build_windows_from_detections(dict(zip(channels, permissive.events_by_channel)),
                                            window_sec=.5, t_max_sec=8.0)
    assert len(windows) == 2
    print("PASS: copied detector, duration gate and group-window construction; synthetic integration check only")


if __name__ == "__main__":
    main()
