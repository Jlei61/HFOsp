"""Scheduling-only memory estimates; never changes the physical execution unit."""
import json
from pathlib import Path


def reserve_gib(root, stage, unit):
    """Reserve 9 GiB for construction; 6 GiB after verified 1 s integration.

    This is a dispatch estimate, not a memory limit. Controllers retain their
    18 GiB process-tree guard and the host's 30/40 GiB hard/dispatch margins.
    Missing, malformed or mismatched progress conservatively means construction.
    """
    cid, topology, noise = unit
    path = (Path(root) / stage / 'units' / cid / f'{topology}_{noise}' /
            'workers/trajectory.progress.json')
    try:
        progress = json.loads(path.read_text())
        job = progress['job']
        identity = (job['candidate'], job['stage'], int(job['topology_seed']),
                    int(job['dynamics_seed']))
        if (identity == (cid, stage, int(topology), int(noise)) and
                progress['status'] == 'SIMULATING' and
                float(progress['simulated_ms']) >= 1000.):
            return 6.
    except (OSError, ValueError, TypeError, KeyError):
        pass
    return 9.
