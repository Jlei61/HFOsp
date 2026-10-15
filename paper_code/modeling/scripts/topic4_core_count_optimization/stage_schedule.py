"""A bounded feedback stage can supersede the original full-search schedule."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import json

def stage_settings(plan, policy):
    stage = policy.get('stage')
    if stage is None:
        return None
    cores = stage['core_counts']
    rounds = stage['feedback_rounds']
    if not isinstance(cores, list) or not cores or len(set(cores)) != len(cores) or any((type(k) is not int or k not in plan['core_counts'] for k in cores)):
        raise ValueError('Stage core counts must be a nonempty subset of the plan')
    if type(rounds) is not int or rounds < 1:
        raise ValueError('Stage feedback rounds must be a positive integer')
    cap = plan['initial_conditions'] + rounds * plan['batch_size']
    if cap > plan['epochs']:
        raise ValueError('Stage cannot exceed the original evaluation budget')
    if stage.get('confirmations') != 'deferred':
        raise ValueError('This bounded training stage defers endpoint confirmations')
    return dict(stage, epochs=cap)

def stage_progress(directory, rows, stage):
    done = {r['candidate'] for r in rows}
    epochs = {r['epoch'] for r in rows}
    count = sum((e in epochs for e in range(1, stage['epochs'] + 1)))
    rounds = 0
    invalid = []
    for batch in range(1, stage['feedback_rounds'] + 1):
        path = directory / 'proposals' / f'batch_{batch:02d}.json'
        if not path.exists():
            continue
        proposal = json.loads(path.read_text())
        if proposal['method'] != 'Matern52_qLogEI_plus_global_uncertainty':
            invalid.append(batch)
        elif set(proposal['ids']) <= done:
            rounds += 1
    complete = count == stage['epochs'] and rounds == stage['feedback_rounds']
    return dict(stage_evaluated=count, stage_planned=stage['epochs'], feedback_rounds_completed=rounds, feedback_rounds_planned=stage['feedback_rounds'], stage_complete=complete, stage_review_required=bool(invalid) or (count == stage['epochs'] and (not complete)), non_feedback_batches=invalid)

def stage_dispatch_rank(candidate):
    """Use already-ready feedback before the few retained initialization tasks."""
    return 0 if candidate.startswith('bo') else 1
