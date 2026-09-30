"""Identical float64 arrays with reclaimable backing for large transient caches.

The on-disk array is a temporary numerical cache, not a scientific artifact.
It remains mapped until the owner releases it and consumes no persistent path.
"""
import os
import tempfile
from pathlib import Path
import numpy as np


def available_bytes():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable is unavailable')


def allocate_host_array(shape, storage='auto'):
    assert storage in ['auto', 'memory', 'disk']
    nbytes = int(np.prod(shape, dtype=np.int64))*8
    available = available_bytes()
    chosen = 'disk' if storage == 'disk' or (storage == 'auto' and nbytes > .7*available) else 'memory'
    if chosen == 'memory':
        array = np.empty(shape, dtype=np.float64)
    else:
        folder = Path(os.environ.get('TMPDIR', '/data/hfosp/tmp_zm_onset_20260918'))
        folder.mkdir(parents=True, exist_ok=True)
        # mmap duplicates the temporary file descriptor. Closing this handle
        # leaves the mapping alive; its last release frees the backing file.
        with tempfile.TemporaryFile(prefix='zm_host_gains_', dir=folder) as handle:
            array = np.memmap(handle, mode='w+', dtype=np.float64, shape=shape)
    assert array.flags.c_contiguous and array.dtype == np.float64
    return array, dict(storage=chosen, nbytes=nbytes, available_at_allocation=available,
                       criterion='disk if cache bytes exceed 70 percent of currently available RAM',
                       arithmetic_change=False)


def sanity():
    expected = np.random.default_rng(919).normal(size=(17, 8, 23))
    arrays = [allocate_host_array(expected.shape, kind)[0] for kind in ['memory', 'disk']]
    for array in arrays:
        for lo in range(0, len(array), 5):
            array[lo:lo+5] = expected[lo:lo+5]
        if isinstance(array, np.memmap):
            array.flush()
        assert np.array_equal(array, expected)
        assert np.array_equal(array[3:11].copy(), expected[3:11])
    return dict(status='PASS', array_and_sliced_block_bitwise=True,
                temporary_descriptor_lifetime='Validated after allocator file handle closes')


if __name__ == '__main__':
    print(sanity())
