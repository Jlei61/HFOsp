"""Paper Video2 examples, selected from newly simulated events in each mode."""
import numpy as np
from scipy.stats import rankdata
from modeling.panel_analysis import classify_model


def mode_medoid(times, names):
    # Equal total weight for each shaft; missingness and normalized ranks both
    # enter the distance. Select the observed event nearest the mode's mean.
    mask = np.isfinite(times)
    ranks = np.zeros_like(times)
    for k, row in enumerate(times):
        n = int(mask[k].sum())
        ranks[k, mask[k]] = (rankdata(row[mask[k]], method='average') - 1) / (n - 1)
    weights = np.array([1 / np.sqrt(2 * sum(q.startswith(n[:3]) for q in names)) for n in names])
    features = np.concatenate([mask * weights, ranks * weights], axis=1) / np.sqrt(2)
    distance = np.sum((features - features.mean(0)) ** 2, axis=1)
    return int(np.argmin(distance))


def select_propagation_events(runs, templates, names, burnin_ms):
    comparison = sorted([run for run in runs if run['score']['J'] is not None],
                        key=lambda run: -run['score']['J'])
    selected = []
    for run in comparison:
        times = np.asarray(run['observed']['centroid_ms'], float)
        labels = classify_model(times, templates)
        duration = run['envelope'].shape[1] * run['envelope_dt_ms']
        ids = [i for i in run['observed']['primary_event_indices']
               if run['observed']['events'][i]['window_ms'][0] >= burnin_ms
               and run['observed']['events'][i]['window_ms'][1] <= duration]
        groups = [[i for i in ids if labels[i] == k] for k in range(2)]
        selected.append([group[mode_medoid(times[group], names)] for group in groups]
                        if all(groups) else None)
    ready = len(comparison) == 2 and all(ids is not None for ids in selected)
    return comparison, dict(events=selected, status='READY' if ready else 'NOT_ESTIMABLE',
                            rule='Paper shaft-balanced rank/participation mode medoid')
