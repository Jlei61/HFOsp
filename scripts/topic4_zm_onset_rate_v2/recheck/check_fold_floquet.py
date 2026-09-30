"""Run the existing full-DDE Floquet audit without overwriting earlier results."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import floquet_zm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--nev", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    floquet_zm.PERIODIC_OUT = floquet_zm.DEST / "recheck_20260917"
    floquet_zm.compute(args.path, args.dt, args.nev, args.device, filtered=True)
