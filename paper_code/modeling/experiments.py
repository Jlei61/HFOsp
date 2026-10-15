"""Compare core counts with equal search budgets on a newly built network."""
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from modeling.computational import optimize_cores
from workflows.common import write_json

_CONTEXT = None


def _search(job):
    substrate, contacts, observer, objective, output, budget = _CONTEXT
    k, restart = job
    result = optimize_cores(substrate, contacts, observer, objective,
                            k=k, budget=budget, restart=restart)
    write_json(output / f'core{k}_restart{restart}.json', result['rows'])
    return result


def search_core_counts(substrate, contacts, observer, objective, output, *,
                       core_counts=(1, 2, 3, 4), budget=2, restarts=1):
    """Each candidate runs a new SNN; workers share only the new starting graph."""
    global _CONTEXT
    output.mkdir(parents=True, exist_ok=True)
    _CONTEXT = substrate, contacts, observer, objective, output, budget
    jobs = [(k, restart) for k in core_counts for restart in range(restarts)]
    results = {}
    try:
        with ProcessPoolExecutor(max_workers=substrate['config']['max_workers'],
                mp_context=multiprocessing.get_context('fork')) as pool:
            futures = {pool.submit(_search, job): job for job in jobs}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
    finally:
        _CONTEXT = None
    return [results[job] for job in jobs]
