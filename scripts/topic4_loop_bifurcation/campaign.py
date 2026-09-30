"""Private scope for the authorized 2026-09-27 Figure 5 analysis."""
from pathlib import Path
import sys
import json
import hashlib

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
ROOT = Path('/data/hfosp/topic4_sef_hfo/fig5_loop_bifurcation_20260927')
PREVIOUS = Path('/data/hfosp/topic4_sef_hfo/fig5_autonomous_loop_zk_20260924')
NATIVE = ROOT / 'native_slices'
PYTHON = '/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            h.update(block)
    return h.hexdigest()
