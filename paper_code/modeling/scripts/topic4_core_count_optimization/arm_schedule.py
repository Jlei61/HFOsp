"""An added arm must pass its own native canary before any training dispatch."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import json

def validated_core_counts(root, planned):
    path = root / 'canary_validation.json'
    if not path.exists():
        return set()
    record = json.loads(path.read_text())
    if record.get('status') != 'PASS':
        return set()
    return {r['core_count'] for r in record.get('checks', [])} & set(planned)

def canary_queue(root, planned, validated):
    workers, helpers = ([], [])
    for k in planned:
        if k in validated:
            continue
        if (root / 'canary' / f'core{k}' / 'result.json').exists():
            helpers.append(('check', k, 0, ''))
        else:
            workers.append(('canary', k, 0, 'canary'))
    return (workers, helpers)
