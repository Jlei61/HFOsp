"""Prepare equal-budget random-start searches and preserve source artifacts."""
from paper_paths import CODE_ROOT, INPUT_ROOT, OUTPUT_ROOT
import copy
import pickle
import shutil
from scipy.stats import qmc
from modeling.scripts.topic4_core_count_optimization.common import *
native_imports()
from modeling.scripts import run_topic4_shape_output_response as physical

def freeze():
    inherited = read(PARENT / 'plan.json')['source_snapshot']
    for path, digest in inherited.items():
        if sha(path) != digest:
            raise RuntimeError('Historical implementation changed: ' + path)
    paths = [Path(p) for p in inherited] + list(HERE.glob('*.py'))
    paths += [WT / 'src/topic4_three_observable_objective.py', WT / 'src/topic4_three_observable_label_free.py', physical.base.PARENT]
    paths += [OUT / p for p in ('plan.json', 'objective.pkl', 'observer.json', 'anchor.json')]
    write(OUT / 'source_snapshot.json', dict(files={str(p): sha(p) for p in paths}, created_unix=time.time()))

def main():
    if (OUT / 'plan.json').exists():
        raise RuntimeError('Already prepared; use the controller to resume')
    for name in ('logs', 'tmp', 'global_graph_cache', 'canary_candidates'):
        (OUT / name).mkdir(parents=True, exist_ok=True)
    anchor = read(ANCHOR)
    write(OUT / 'anchor.json', anchor)
    parent = read(PARENT / 'plan.json')
    objective = Path(parent['objective']['artifact'])
    if sha(objective) != parent['objective']['sha256']:
        raise RuntimeError('Historical objective changed')
    shutil.copy2(objective, OUT / 'objective.pkl')
    observer = physical.rt.load_observation_contract(read(physical.base.PARENT))
    write(OUT / 'observer.json', observer)
    with (OUT / 'objective.pkl').open('rb') as f:
        obj = pickle.load(f)
    if any((hasattr(obj, name) for name in ('km', 'proportions', 'mode_means'))):
        raise RuntimeError('Expected historical label-free loss')
    plan = dict(version='e1146_core_count_equal_budget_v1', subject='epilepsiae_1146', created_unix=time.time(), authorization='2026-09-28 user requested 1/2/3-core comparison; 2026-09-30 authorized early feedback and random-position 4-core extension.', core_counts=[1, 2, 3, 4], restarts=4, initial_conditions=15, adaptive_batches=81, batch_size=8, epochs=96, early_first_feedback=True, feedback_min_scorable=8, adaptive_batches_is_upper_bound=True, train_simulations=1536, confirmation_simulations=32, formal_simulations_total=1568, canaries=4, duration_ms=60000.0, canary_ms=1000.0, burnin_ms=1500.0, topology_seed=2511, training_noise=847401, confirmation_noises=[928401, 928402], radius_mm=anchor['radii_mm'][0], outward_bounds=[0.9, 1.6], angle_bounds_deg=[-60.0, 60.0], random_seed=2026092801, initialization='Independent scrambled Sobol positions for every arm/restart; no previously optimized warm starts.', position_bounds='Full 20 mm sheet with radius+0.05 mm edge margin; centers separated by at least 2*radius+0.05 mm.', fixed_physics='Historical dense_axis3_12: radius/depth, EE_same_core=0.85, perpendicular_scale=1.5, independent core OU, mean=0.95; only K, centers, outward EE and angle vary.', objective=dict(source=str(objective), sha256=sha(objective), version=obj.version, labels_used=False, groups=GROUPS, weights=[1 / 3] * 3, scales=obj.scales, minimum_events=16, negative_values_clipped=False), optimizer='Matern-5/2 ARD GP; six local qLogEI and two global uncertainty proposals per delayed batch. Fewer than eight scorable rows uses explicitly labelled feasibility space filling.', nomination='Lowest finite training J per core count and optimizer restart; freeze before independent noise reruns. Confirmation cannot feed back into optimization.', epoch_definition='One newly evaluated 60 s configuration, in frozen proposal order; unscorable attempts count toward budget and never receive fabricated losses.', resources=dict(max_workers=48, max_helpers=1, projected_worker_gib=6.5, simulation_worker_gib=4.0, max_building_workers=4, available_reserve_gib=64.0, emergency_available_gib=28.0, tree_limit_gib=18.0, disk_reserve_gib=150.0, cpu_ceiling_percent=85.0, dispatch_hours=336.0, attempts=2), limitations=['Per-core radius and per-cell input are fixed: total excitable area/input support increases with K; this is a model-configuration comparison, not an isolated count effect at equal dose.', 'One subject and topology; optimizer restarts measure optimization variability, not independent patient samples.', 'Best-so-far training curves are nonincreasing by definition. Historical patient data are development data; confirmation tests only fresh simulation noise.', 'Two-core superiority is a hypothesis. There is no K penalty, arm-specific normalization or outcome-dependent budget.', 'The old B plot spans historical objective changes; its selected old points are excluded from this equal-budget experiment.'])
    write(OUT / 'plan.json', plan)
    for restart in range(plan['restarts']):
        for k in plan['core_counts']:
            d = run_dir(k, restart)
            for name in ('candidates', 'scores', 'units', 'proposals', 'confirmation'):
                (d / name).mkdir(parents=True, exist_ok=True)
            lo, hi = bounds(k)
            seed = plan['random_seed'] + 100 * restart + k
            pool = qmc.Sobol(2 * k + 2, scramble=True, seed=seed).random_base2(12)
            vectors = [lo + z * (hi - lo) for z in pool if valid(k, lo + z * (hi - lo))][:plan['initial_conditions']]
            assert len(vectors) == plan['initial_conditions']
            ids = []
            for i, x in enumerate(vectors):
                cid = f'initial_{i:02d}'
                ids.append(cid)
                write(d / 'candidates' / f'{cid}.json', candidate(k, restart, x, cid, i + 1, 0, 'random_space_filling'))
            write(d / 'proposals/batch_00.json', dict(ids=ids, seed=seed, method='random_space_filling', created_unix=time.time()))
            if restart == 0:
                c = candidate(k, restart, vectors[0], 'canary', 0, -1, 'physical_canary')
                if k == 2:
                    c['centers_mm'] = copy.deepcopy(anchor['centers_mm'])
                    c['parameters'] = copy.deepcopy(anchor['parameters'])
                    c['vector'] = np.r_[np.asarray(c['centers_mm']).ravel(), c['parameters']['EE_core_to_out_scale'], c['parameters']['EE_angle_offset_deg']].tolist()
                write(OUT / 'canary_candidates' / f'core{k}.json', c)
    freeze()
    write(OUT / 'status.json', dict(status='PREPARED_AWAITING_CANARY', formal_complete=0, planned_formal=plan['formal_simulations_total'], updated_unix=time.time()))
    print(json.dumps(dict(output=str(OUT), formal_simulations=plan['formal_simulations_total'], objective_sha256=sha(OUT / 'objective.pkl'))))
if __name__ == '__main__':
    main()
