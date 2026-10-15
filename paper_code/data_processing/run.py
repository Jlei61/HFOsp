"""Explicit raw-EDF preprocessing and HFO detection entry point."""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from data_processing.src.preprocessing import SEEGPreprocessor
from data_processing.src.hfo_detector import HFODetectionConfig, HFODetector
from data_processing.src.group_event_analysis import build_windows_from_detections
from paper_paths import CODE_ROOT, INPUT_ROOT


def process_edf(recording, configuration, output):
    config = json.loads(Path(configuration).read_text())
    required = {"reference", "target_sfreq", "notch_freqs", "exclude_channels"}
    if not required.issubset(config["preprocessing"]):
        raise ValueError(f"Specify all preprocessing choices: {sorted(required)}")
    output = Path(output).resolve()
    if output in (CODE_ROOT, INPUT_ROOT) or CODE_ROOT in output.parents or INPUT_ROOT in output.parents:
        raise ValueError("Write processing outputs outside the code and frozen input directories")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new, empty output directory")
    output.mkdir(parents=True, exist_ok=True)
    # Avoid creating FIF caches alongside the input recording.
    pre = SEEGPreprocessor(**config["preprocessing"], use_fif_cache=False).run(recording)
    detected = HFODetector(HFODetectionConfig(**config["detector"])).detect(pre)
    detections = dict(zip(detected.ch_names, detected.events_by_channel))
    windows = build_windows_from_detections(detections, **config["group_events"])
    np.savez_compressed(output / "channel_events.npz", sfreq=detected.sfreq,
                        ch_names=np.asarray(detected.ch_names),
                        **{f"events_{i:04d}": x for i, x in enumerate(detected.events_by_channel)})
    (output / "group_windows.json").write_text(json.dumps([asdict(w) for w in windows], indent=2))
    (output / "used_configuration.json").write_text(json.dumps(config, indent=2))
    return {"channels": len(detections), "group_events": len(windows), "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edf", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(process_edf(args.edf, args.config, args.output), indent=2))
