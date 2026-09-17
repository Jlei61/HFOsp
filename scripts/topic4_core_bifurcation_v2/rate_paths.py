"""Resolve saved historical result paths inside the current checkout.

The archived JSON/CSV retain their original source paths and hashes. Published
figure consumers resolve them here instead of reading an unrelated live checkout.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HISTORICAL_ROOT = Path('/home/honglab/leijiaxin/HFOsp')


def saved_path(value):
    path = Path(value)
    if not path.is_absolute():
        return ROOT / path
    if path.is_relative_to(HISTORICAL_ROOT):
        return ROOT / path.relative_to(HISTORICAL_ROOT)
    if path.is_relative_to(ROOT):
        return path
    raise ValueError(f'Unrecognized saved result root: {path}')
