"""Create the immutable code snapshot from which the v0.4 cohort is run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "train_topic5_we_unit.py",
    "launch_topic5_rnn_motif_v0_4.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    out_root = args.out_root.resolve()
    snapshot = out_root / "run_snapshot"
    if snapshot.exists():
        raise FileExistsError(f"immutable snapshot already exists: {snapshot}")
    (snapshot / "scripts").mkdir(parents=True)
    shutil.copytree(ROOT / "src", snapshot / "src")
    for name in SCRIPT_NAMES:
        shutil.copy2(ROOT / "scripts" / name, snapshot / "scripts" / name)
    files = sorted(path for path in snapshot.rglob("*") if path.is_file())
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--short"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    payload = {
        "git_head": head,
        "git_status_at_snapshot": status,
        "input_manifest_sha256": sha256(out_root / "INPUT_MANIFEST.json"),
        "files": {str(path.relative_to(snapshot)): sha256(path) for path in files},
    }
    (snapshot / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(payload, indent=2))
    for path in snapshot.rglob("*"):
        if path.is_file():
            path.chmod(0o444)
        elif path.is_dir():
            path.chmod(0o555)
    snapshot.chmod(0o555)
    print(f"frozen {len(files)} files at {snapshot}; git={head[:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
