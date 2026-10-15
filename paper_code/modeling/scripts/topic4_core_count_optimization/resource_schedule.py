"""Reserve startup peaks separately from bounded, streaming simulation memory."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import json
import math
WORKER_KINDS = ('train', 'canary', 'confirmation')

def validated_limits(defaults, override, cpu_count):
    allowed = {'max_workers', 'simulation_worker_gib', 'max_building_workers', 'available_reserve_gib', 'emergency_available_gib', 'cpu_ceiling_percent'}
    if not set(override) <= allowed:
        raise ValueError('Unsupported resource override')
    limits = dict(defaults, **override)
    for name in ('max_workers', 'max_building_workers'):
        value = limits.get(name, limits['max_workers'])
        if type(value) is not int or not 1 <= value <= cpu_count:
            raise ValueError(f'{name} must be within the available CPU affinity')
        limits[name] = value
    value = limits.get('simulation_worker_gib', limits['projected_worker_gib'])
    if type(value) not in (int, float) or not math.isfinite(value) or (not 4.0 <= value <= limits['projected_worker_gib']):
        raise ValueError('Simulation reservation must be between 4 GiB and the startup peak')
    limits['simulation_worker_gib'] = value
    for name, low, high in [('available_reserve_gib', 20.0, math.inf), ('emergency_available_gib', 8.0, math.inf), ('cpu_ceiling_percent', 1.0, 95.0)]:
        if name not in limits:
            continue
        value = limits[name]
        if type(value) not in (int, float) or not math.isfinite(value) or (not low <= value <= high):
            raise ValueError(f'{name} is outside the supported safety range')
    if 'emergency_available_gib' in limits and limits['emergency_available_gib'] >= limits.get('available_reserve_gib', 0):
        raise ValueError('Emergency threshold must stay below dispatch memory reserve')
    return limits

def worker_reservation(root, active, rss_gib, limits, now):
    """Only progress from this exact process/task can release startup headroom."""
    kind, cores, restart, value = active['task']
    candidate, noise = value.rsplit(':', 1) if kind == 'confirmation' else (value, None)
    run = root / 'per_subject/epilepsiae_1146' / f'core{cores}' / f'restart{restart:02d}'
    unit = root / 'canary' / f'core{cores}' if kind == 'canary' else run / ('confirmation' if kind == 'confirmation' else 'units') / (f'{candidate}_{noise}' if noise else candidate)
    steady = False
    try:
        path = unit / 'progress.json'
        progress = json.loads(path.read_text())
        job = progress['job']
        updated = progress['updated_unix']
        steady = progress['status'] == 'SIMULATING' and progress['simulated_ms'] > 0 and (active['created'] <= updated <= now + 5) and (path.stat().st_mtime >= active['created']) and (job['phase'] == kind) and (job['core_count'] == cores) and (job['restart'] == restart) and (job['candidate'] == candidate) and (noise is None or job['dynamics_seed'] == int(noise))
    except (OSError, ValueError, KeyError, TypeError):
        pass
    steady = steady and rss_gib <= limits['simulation_worker_gib']
    projected = limits['simulation_worker_gib'] if steady else limits['projected_worker_gib']
    return dict(stage='SIMULATING' if steady else 'BUILDING_OR_UNVERIFIED', projected_gib=max(projected, rss_gib), growth_gib=max(0.0, projected - rss_gib))

def reservations(root, active, rss, limits, now):
    details = {}
    for ident, a in active.items():
        current = rss.get(ident, 0.0)
        details[ident] = worker_reservation(root, a, current, limits, now) if a['task'][0] in WORKER_KINDS else dict(stage='HELPER', projected_gib=max(4.0, current), growth_gib=max(0.0, 4.0 - current))
    return (details, sum((d['growth_gib'] for d in details.values())), sum((d['stage'] == 'BUILDING_OR_UNVERIFIED' for d in details.values())))
