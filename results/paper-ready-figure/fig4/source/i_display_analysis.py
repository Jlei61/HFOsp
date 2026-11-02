"""Reproduce old D exactly, then quantify conditional time-block uncertainty.

Run once with --extract to freeze the minimal inputs from the canonical package
and original trajectories. Later runs use only these local inputs. No SNN runs.
All outputs belong to this candidate; no formal figure pointers are changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/honglab/leijiaxin/HFOsp')
FORMAL = ROOT / 'results/paper-ready-figure/fig4'
SEARCH = Path('/data/hfosp/topic4_sef_hfo/label_free_dense_parameter_search_20260915')
METRICS = ('mean_rank_error', 'within_shaft_order_error', 'participation_error')
SEED, N_BOOT = 2026102404, 4000


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def extract():
    """Freeze only selected events, target summaries and provenance."""
    paths = [HERE / n for n in ('i_original_angle_response.csv', 'i_events.npz',
                               'i_target.json', 'i_provenance.json')]
    if any(p.exists() for p in paths):
        raise FileExistsError('Extraction never overwrites existing snapshots.')
    pointer, registry = (read(FORMAL / n) for n in ('current_version.json', 'figure4_panel_registry.json'))
    assert pointer['panel_d'] == registry['panel_d']
    assert pointer['panel_d']['version'] == 'angle_three_error_response_v1'
    table = FORMAL / 'source/D_connection_response.csv'
    rows = sorted((r for r in csv.DictReader(table.open()) if r['axis'] == '5'),
                  key=lambda r: float(r['parameter_value']))
    assert len(rows) == 13
    target_path = SEARCH / 'analysis/diagnostic_observables.json'
    target = read(target_path)['patient']['ALL']
    names = list(target['contacts'])
    records, arrays, baseline = [], {'contact_names': np.array(names)}, None
    plan = read(SEARCH / 'plan.json')
    assert plan['analysis']['burnin_ms'] == 1500
    assert read(SEARCH / 'completion.json')['additional_seeds'] == 0
    for row in rows:
        cid, source = row['candidate'], Path(row['source'])
        raw = read(source)
        assert sha(source) == row['source_sha256']
        assert raw['actual_duration_ms'] == 60000
        assert raw['physical_status'] == 'COMPLETE_NO_RUNAWAY'
        assert raw['job']['topology_seed'] == 2511 and raw['job']['dynamics_seed'] == 847401
        npz_path = source.with_suffix('.npz')
        assert sha(npz_path) == raw['arrays_sha256']
        design_path, applied_path = source.parent.parent / 'design.json', source.parent.parent / 'applied_physics.json'
        assert sha(design_path) == raw['design_sha256']
        assert sha(applied_path) == raw['applied_physics_sha256']
        design, applied = read(design_path), read(applied_path)
        candidate = design['candidate']
        assert candidate == applied['candidate']
        assert candidate['parameters']['EE_angle_offset_deg'] == float(row['parameter_value'])
        fixed = {k: candidate[k] for k in ('centers_mm', 'radii_mm', 'core_mean_rate_scale',
                                          'core_ou_correlation', 'shape', 'outgoing')}
        fixed['parameters'] = {k: v for k, v in candidate['parameters'].items() if k != 'EE_angle_offset_deg'}
        if baseline is None:
            baseline = fixed
        assert fixed == baseline
        with np.load(npz_path, allow_pickle=False) as z:
            assert z['contact_names'].tolist() == names
            ix = np.array([int(i) for i in z['primary_event_indices']
                           if raw['events'][int(i)]['window_ms'][0] >= 1500
                           and raw['events'][int(i)]['window_ms'][1] <= 60000])
            assert len(ix) == int(row['N'])
            arrays[cid + '_centroid_ms'] = z['centroid_ms'][ix]
            arrays[cid + '_event_time_ms'] = z['event_time_ms'][ix]
            arrays[cid + '_original_event_indices'] = ix
            assert np.all(np.diff(arrays[cid + '_event_time_ms']) > 0)
        score_path = SEARCH / 'scores' / (cid + '.json')
        score = read(score_path)
        assert score['N'] == len(ix) and score['source_sha256'] == row['source_sha256']
        np.testing.assert_allclose(score['J'], np.mean(score['components']), rtol=0, atol=1e-12)
        records.append(dict(candidate=cid, source=str(source), source_sha256=sha(source),
            npz_sha256=sha(npz_path), design_sha256=sha(design_path), applied_sha256=sha(applied_path),
            topology_seed=2511, dynamics_seed=847401, duration_ms=60000,
            observed_unit_seed_directories=sorted(p.name for p in source.parent.parent.parent.iterdir() if p.is_dir()),
            J=score['J'], J_components=score['components'], J_joint=score.get('J_joint'),
            score_source=str(score_path), score_sha256=sha(score_path),
            actual_kernel=applied['graph']['kernel'], N=len(ix)))
    csv_write(paths[0], rows)
    np.savez_compressed(paths[1], **arrays)
    write(paths[2], target)
    write(paths[3], dict(formal_version=pointer['version'], formal_panel_d=pointer['panel_d'],
        source_csv=str(table), source_csv_sha256=sha(table), target_source=str(target_path),
        target_source_sha256=sha(target_path), patient_fit_events=target['N'], fixed_parameters=baseline,
        source_code=[str(ROOT / '.worktrees/topic4-continuous-core-state-r1/scripts/report_topic4_label_free_dense.py'),
                     str(ROOT / '.worktrees/topic4-continuous-core-state-r1/scripts/report_topic4_three_observable_raw.py'),
                     str(ROOT / '.worktrees/topic4-continuous-core-state-r1/src/topic4_three_observable_objective.py'),
                     str(FORMAL / 'source/fig4_parameter_response_assets.py')],
        records=records, repeat_variance_status='NOT_ESTIMABLE: complete 13-angle sweep has one seed pair per angle',
        event_selection='primary_event_indices AND window_start >= 1500 ms AND window_end <= 60000 ms',
        no_new_simulations=True))


def features(t, names, target):
    """Sufficient statistics; missing contacts/pairs never count as zero ranks."""
    m = np.isfinite(t)
    r = np.zeros_like(t)
    for i in range(len(t)):
        assert m[i].sum() >= 2
        r[i, m[i]] = (rankdata(t[i, m[i]], method='average') - 1) / (m[i].sum() - 1)
    pair_names = list(target['pairs'])
    pairs = np.array([(names.index(k.split('→')[0]), names.index(k.split('→')[1])) for k in pair_names])
    delta = t[:, pairs[:, 1]] - t[:, pairs[:, 0]]
    pair_valid = np.isfinite(delta)
    pair_order = (delta > 0) + .5 * (delta == 0)
    return np.column_stack([np.ones(len(t)), m, r, pair_valid, pair_order])


def score_sums(sums, names, target):
    sums = np.atleast_2d(sums)
    n, c, p = sums[:, :1], len(names), len(target['pairs'])
    count, ranks = sums[:, 1:1+c], sums[:, 1+c:1+2*c]
    support = sums[:, 1+2*c:1+2*c+p]
    ordered = sums[:, 1+2*c+p:]
    # Every original patient contact/pair is supported. A bootstrap lacking one
    # is excluded explicitly, never made to look better by omitting that term.
    valid = (count > 0).all(1) & (support > 0).all(1) & (n[:, 0] > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        mr, pr, op = ranks/count, count/n, ordered/support
    target_r = np.array(target['mean_rank'])
    target_p = np.array([target['contacts'][name]['participation'] for name in names])
    target_o = np.array([v['order_probability'] for v in target['pairs'].values()])
    rank_error, part_error, order_error = [], [], []
    for shaft in ('SCL', 'ICL'):
        ix = [i for i, name in enumerate(names) if name.startswith(shaft)]
        pi = [i for i, v in enumerate(target['pairs'].values()) if v['shaft'] == shaft]
        rank_error.append(np.mean(abs(mr[:, ix] - target_r[ix]), axis=1))
        part_error.append(np.mean(abs(pr[:, ix] - target_p[ix]), axis=1))
        order_error.append(np.mean(abs(op[:, pi] - target_o[pi]), axis=1))
    result = np.column_stack([np.mean(rank_error, axis=0), np.mean(order_error, axis=0), np.mean(part_error, axis=0)])
    result[~valid] = np.nan
    return result


def analyze():
    rows = list(csv.DictReader((HERE / 'i_original_angle_response.csv').open()))
    target, provenance = read(HERE / 'i_target.json'), read(HERE / 'i_provenance.json')
    with np.load(HERE / 'i_events.npz', allow_pickle=False) as z:
        names = z['contact_names'].tolist()
        samples = [(features(z[r['candidate'] + '_centroid_ms'], names, target),
                    z[r['candidate'] + '_event_time_ms']) for r in rows]
    original = np.array([[float(r[k]) for k in METRICS] for r in rows])
    reproduced = np.array([score_sums(f.sum(0), names, target)[0] for f, _ in samples])
    np.testing.assert_allclose(reproduced, original, rtol=0, atol=1e-12)
    angles = np.array([float(r['parameter_value']) for r in rows])
    aggregate = original.mean(1)
    order = np.argsort(aggregate)
    best, runnerup = int(order[0]), int(order[1])
    boot, sensitivity, checks = {}, [], {}
    # Shared physical time-block draws preserve time-local cofluctuation across
    # angle trajectories; equal event indices across trajectories are NOT paired.
    for blocks in (13, 26, 6):
        edges = np.linspace(1500, 60000, blocks + 1)
        rng = np.random.default_rng(SEED + blocks)
        weights = rng.multinomial(blocks, np.full(blocks, 1/blocks), size=N_BOOT)
        draws = []
        for f, times in samples:
            bins = np.searchsorted(edges, times, side='right') - 1
            assert np.all((bins >= 0) & (bins < blocks))
            totals = np.zeros((blocks, f.shape[1]))
            np.add.at(totals, bins, f)
            draws.append(score_sums(weights @ totals, names, target))
        draws = np.stack(draws, axis=1)
        valid = np.isfinite(draws).all(axis=(1, 2))
        checks[str(blocks)] = dict(valid_draws=int(valid.sum()), rejected_draws=int((~valid).sum()))
        draws = draws[valid]
        assert len(draws) >= .99 * N_BOOT
        boot['components_' + str(blocks)] = draws
        totals = draws.mean(2)
        variances = totals.var(0, ddof=1)
        gap = totals[:, best] - totals[:, runnerup]
        sensitivity.append(dict(blocks=blocks, block_duration_ms=58500/blocks,
            valid_draws=len(draws), empirical_min_angle_deg=float(angles[best]),
            min_variance_angle_deg=float(angles[np.argmin(variances)]),
            min_mean_variance=variances[best], min_mean_sd=float(np.sqrt(variances[best])),
            runner_up_sd=float(np.sqrt(variances[runnerup])),
            best_minus_runner_up_q025=float(np.quantile(gap, .025)),
            best_minus_runner_up_q975=float(np.quantile(gap, .975)),
            best_lower_fraction=float(np.mean(gap < 0))))
    primary = boot['components_13']
    total_boot = primary.mean(2)
    variance = total_boot.var(0, ddof=1)
    component_var = primary.var(0, ddof=1)
    # Retain covariance: variance of the mean is NOT variance across components.
    covariances = np.array([np.cov(primary[:, i, :], rowvar=False, ddof=1) for i in range(13)])
    np.testing.assert_allclose(variance, covariances.sum(axis=(1, 2))/9, rtol=1e-12, atol=1e-16)
    pareto = [not any(aggregate[j] <= aggregate[i] and variance[j] <= variance[i]
                     and (aggregate[j] < aggregate[i] or variance[j] < variance[i])
                     for j in range(13)) for i in range(13)]
    output = []
    for i, (row, record) in enumerate(zip(rows, provenance['records'])):
        assert row['candidate'] == record['candidate']
        out = dict(row, display_loss=float(aggregate[i]),
            display_loss_bootstrap_variance=float(variance[i]), display_loss_bootstrap_sd=float(np.sqrt(variance[i])),
            display_loss_bootstrap_q025=float(np.quantile(total_boot[:, i], .025)),
            display_loss_bootstrap_q975=float(np.quantile(total_boot[:, i], .975)),
            display_loss_bootstrap_mean=float(total_boot[:, i].mean()),
            empirical_display_minimum=i == best, empirical_variance_minimum=i == int(np.argmin(variance)),
            mean_variance_pareto=pareto[i], optimizer_J=record['J'], optimizer_J_joint='',
            independent_run_count=1, bootstrap_blocks=13, bootstrap_draws=len(primary))
        for j, key in enumerate(METRICS):
            out[key + '_bootstrap_variance'] = float(component_var[i, j])
        output.append(out)
    csv_write(HERE / 'i_display_angle_response.csv', output)
    csv_write(HERE / 'i_display_block_sensitivity.csv', sensitivity)
    np.savez_compressed(HERE / 'i_display_bootstrap_draws.npz', angles_deg=angles, **boot)
    j_values = np.array([r['J'] for r in provenance['records']])
    metadata = dict(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW', parent_panel='formal D', new_panel='I',
        render_api='render_i(fig, rect), rect=(left,bottom,width,height) in normalized figure coordinates; returns Axes',
        requested_data_axis_mm=[65, 53], fontsize_pt=dict(label=9, tick=8, legend=7),
        scientific_question='Which sampled EE angle balances three patient-summary errors, and how stable is the ranking to within-run sampling?',
        parameter='EE_angle_offset_deg', angle_is_offset_not_absolute_orientation=True,
        subject='epilepsiae_1146', n_conditions=13, simulation_seconds=60, patient_fit_events=target['N'],
        aggregate=dict(name='Mean display error', symbol='L_display',
            formula='(mean_rank_error + within_shaft_order_error + participation_error) / 3',
            weights=[1/3]*3, units='dimensionless', range=[0, 1],
            justification='All three errors are absolute discrepancies of quantities on [0,1]; equal component weights are an explicit post hoc display convention, not a fitted or historically optimized loss.',
            shaft_weighting='Equal SCL/ICL weight within each component; contacts or within-shaft pairs equally weighted within a shaft.',
            mean_rank='Absolute difference of conditional mean normalized within-event ranks, averaged across contacts within each shaft, then across shafts.',
            order='Absolute difference of P(t_j>t_i)+0.5 P(t_j=t_i), conditional on joint participation; averaged within each shaft, then across shafts.',
            participation='Absolute difference of event participation probability, averaged within each shaft, then across shafts.',
            no_minmax_rescaling=True, not_optimizer_J=True, not_optimizer_J_joint=True),
        uncertainty=dict(kind='Paired nonoverlapping physical-time-block bootstrap, model side only',
            resampling_unit='All qualified model events within one 4.5 s block of the same trajectory',
            inferential_unit='One simulated network/noise realization at each angle; NOT 13 independent networks',
            fixed_reference='19770 patient FIT events summarized once, no patient resampling',
            start_ms=1500, end_ms=60000, n_blocks=13, block_ms=4500, draws=N_BOOT,
            rng='numpy.default_rng(2026102404 + block_count)', ddof=1,
            displayed_band='Original full-sample L_display plus/minus one bootstrap SD; not a confidence interval',
            variance_definition='sample variance over recomputed aggregate scores; all component covariance retained',
            pairing='Same drawn physical-time-block indices across all angles; no event-index pairing',
            assumptions='Post-burnin blocks treated as exchangeable; block-length sensitivity recorded. Longer dependencies and nonstationarity are not identified from one 60 s run.',
            limitations='Does not estimate independent seed/topology variance, patient sampling uncertainty, or independent validation error.',
            full_support_checks=checks),
        optimum=dict(empirical_display_minimum=dict(angle_deg=float(angles[best]), value=float(aggregate[best]), candidate=rows[best]['candidate']),
            runner_up=dict(angle_deg=float(angles[runnerup]), value=float(aggregate[runnerup]), candidate=rows[runnerup]['candidate']),
            absolute_gap=float(aggregate[runnerup]-aggregate[best]),
            individual_component_minima={k:float(angles[np.argmin(original[:, j])]) for j,k in enumerate(METRICS)},
            bootstrap_variance_minimum=dict(angle_deg=float(angles[np.argmin(variance)]), value=float(variance.min())),
            mean_variance_pareto_angles=angles[np.array(pareto)].tolist(),
            simultaneous_minimum_of_all_components_and_variance=False,
            optimizer_J_minimum=dict(angle_deg=float(angles[np.argmin(j_values)]), value=float(j_values.min())),
            J_joint_available=False,
            uniqueness='Empirical argmin on the sampled grid, not a unique population optimum; inspect paired gap and block sensitivity.',
            no_variance_penalty_selected='Variance has squared-error units. No unrequested tradeoff coefficient is imposed to force one robust optimum.'),
        optimizer_contract=dict(J='Equal mean of calibrated signed D_off terms: rank_pattern, local_and_interrod_timing, participation; full distribution features, not these three display MAEs.',
            J_joint='Joint event-cooccurrence objective from a later stage; unavailable for this frozen scan.',
            panel_C_integration='If C retains optimization best-training loss, label C as Training loss J and I as Display error. A common C/I definition requires recomputing one entire series, not relabeling.'),
        validation=dict(original_39_values_recomputed=True, original_values_preserved_in_csv_and_artists=True,
            maximum_absolute_reconstruction_error=float(abs(reproduced-original).max()),
            fixed_parameters_audited=True, complete_repeat_variance_available=False, no_new_simulations=True),
        block_sensitivity=sensitivity,
        provenance='i_provenance.json', source_snapshot_sha256={n:sha(HERE/n) for n in ['i_events.npz','i_target.json','i_original_angle_response.csv']})
    write(HERE / 'i_display_metadata.json', metadata)
    print(json.dumps(metadata['optimum'], indent=2))
    print(json.dumps(sensitivity, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extract', action='store_true')
    args = parser.parse_args()
    if args.extract:
        extract()
    analyze()
