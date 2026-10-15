"""Bounded early feedback scheduling; no changes to simulation or objective."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import json

def read(path):
    return json.loads(path.read_text())

def dispatch_rank(candidate):
    """Start one feedback configuration per ready stream, then drain initial runs."""
    if candidate.startswith('bo') and candidate.endswith('_00'):
        return 0
    return 1 if candidate.startswith('initial_') else 2

def training_queue(directory, plan, rows):
    """Return pending configurations and at most one next proposal batch.

    The first feedback batch may overlap retained initial runs. Later batches
    wait for every preceding evaluation. Small feasibility top-ups count toward
    the same total budget and do not masquerade as feedback optimization.
    """
    done = {r['candidate'] for r in rows}
    assert len(done) == len(rows)
    waiting = []
    reserved = []
    threshold = plan.get('feedback_min_scorable', 8)
    for batch in range(plan['epochs'] + 1):
        path = directory / 'proposals' / f'batch_{batch:02d}.json'
        if not path.exists():
            assert batch > 0, 'Initial proposal is missing'
            return (waiting, batch)
        ids = read(path)['ids']
        assert ids and (not set(ids) & set(reserved))
        reserved.extend(ids)
        assert len(reserved) <= plan['epochs'], 'Training budget exceeded'
        missing = [cid for cid in ids if cid not in done]
        waiting.extend(missing)
        if missing:
            early = batch == 0 and plan.get('early_first_feedback', False) and (sum((r['J'] is not None for r in rows)) >= threshold)
            if not early:
                return (waiting, None)
        elif waiting:
            return (waiting, None)
        if len(reserved) == plan['epochs']:
            return (waiting, None)
    raise RuntimeError('No bounded proposal schedule found')

def proposal_state(directory, plan, batch, rows):
    """Freeze the observed losses and reserved/pending coordinates separately."""
    reserved = []
    for previous in range(batch):
        reserved.extend(read(directory / 'proposals' / f'batch_{previous:02d}.json')['ids'])
    assert len(set(reserved)) == len(reserved)
    done = {r['candidate'] for r in rows}
    assert len(done) == len(rows) and done <= set(reserved)
    candidates = [read(directory / 'candidates' / f'{cid}.json') for cid in reserved]
    assert [c['epoch'] for c in candidates] == list(range(1, len(reserved) + 1))
    by_id = {c['id']: c for c in candidates}
    assert all((r['epoch'] == by_id[r['candidate']]['epoch'] for r in rows))
    pending = [c for c in candidates if c['id'] not in done]
    n_good = sum((r['J'] is not None for r in rows))
    threshold = plan.get('feedback_min_scorable', 8)
    if pending:
        assert batch == 1 and plan.get('early_first_feedback', False) and (n_good >= threshold)
    remaining = plan['epochs'] - len(reserved)
    assert remaining > 0
    size = min(plan['batch_size'], remaining)
    if n_good < threshold:
        size = min(size, threshold - n_good)
    return dict(rows=rows, reserved_candidates=candidates, pending_candidates=pending, reserved_count=len(reserved), batch_size=size, n_scorable=n_good)
