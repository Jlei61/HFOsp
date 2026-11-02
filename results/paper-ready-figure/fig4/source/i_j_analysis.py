"""Actual training J and block-bootstrap SD. --extract freezes original maps."""
from __future__ import annotations
import argparse
import csv
import json
import pickle
from pathlib import Path
import shutil
import sys
import warnings
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/honglab/leijiaxin/HFOsp')
WT = ROOT / '.worktrees/topic4-continuous-core-state-r1'
SEARCH = Path('/data/hfosp/topic4_sef_hfo/label_free_dense_parameter_search_20260915')
GROUPS = ('rank_pattern', 'local_and_interrod_timing', 'participation')
MAPS = ('rank', 'mask', 'scl_pairs', 'icl_pairs', 'rod_lag')
SEED, N_BOOT = 2026102404, 4000


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def csv_write(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mapped_features(t, pars, spec):
    """Exact frozen label-free feature formula; checked against original scorer."""
    t = np.asarray(t, dtype=np.float64)
    m = np.isfinite(t)
    assert not np.isinf(t).any() and np.all(m.sum(1) >= 2)
    r = np.zeros_like(t)
    for i, row in enumerate(t):
        r[i, m[i]] = (rankdata(row[m[i]], method='average') - 1)/(m[i].sum()-1)
    m = m.astype(float)
    shafts = [np.array(spec[s]) for s in ('scl', 'icl')]
    raw = dict(rank=np.column_stack([np.column_stack([m[:, ix], r[:, ix]]) / np.sqrt(len(ix)*2)
                                     for ix in shafts])/np.sqrt(2),
               mask=np.column_stack([m[:, ix]/np.sqrt(len(ix)) for ix in shafts])/np.sqrt(2))
    for name, pair in zip(('scl_pairs','icl_pairs'), spec['pairs']):
        pair = np.array(pair)
        delta = t[:, pair[:,1]] - t[:, pair[:,0]]
        valid = np.isfinite(delta)
        delta = np.where(valid, delta, 0.)
        raw[name] = np.column_stack([valid, np.sign(delta), np.arcsinh(delta/spec['time_scales'][name])])/np.sqrt(3*len(pair))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        lag = np.nanmedian(t[:, shafts[0]], axis=1) - np.nanmedian(t[:, shafts[1]], axis=1)
    valid = np.isfinite(lag)
    lag = np.where(valid, lag, 0.)
    raw['rod_lag'] = np.column_stack([valid, np.sign(lag), np.arcsinh(lag/spec['time_scales']['rod_lag'])])/np.sqrt(3)
    mapped = {}
    for key, x in raw.items():
        y = np.cos(x @ pars[key+'_weights'] + pars[key+'_phases']) * np.sqrt(2/(3*spec['n_fourier']))
        mapped[key] = (np.column_stack([x/spec['linear_scales'][key],y])/np.sqrt(2)
                       if key in ('rank','mask') else y)
    return dict(rank_pattern=mapped['rank'], participation=mapped['mask'],
                local_and_interrod_timing=np.column_stack([mapped[k] for k in ('scl_pairs','icl_pairs','rod_lag')])/np.sqrt(3))


def d_off(phi, target):
    """Signed U-statistic relative to the FIXED patient embedding."""
    n = len(phi); s = phi.sum(0); q = np.sum(phi*phi)
    assert n >= 16
    return float((s@s-q)/(n*(n-1)) - 2*(s@target)/n + target@target)


def bootstrap_group(phi, target, scale, bins, weights, blocks):
    """Subtract event-occurrence self-pairs, then divide by resampled n(n-1)."""
    counts = np.bincount(bins, minlength=blocks)
    block_sum = np.zeros((blocks, phi.shape[1])); block_q = np.zeros(blocks)
    np.add.at(block_sum, bins, phi)
    np.add.at(block_q, bins, np.sum(phi*phi, axis=1))
    n = weights @ counts
    assert np.all(n >= 16)
    gram = block_sum @ block_sum.T
    sum_norm = np.einsum('bi,ij,bj->b', weights, gram, weights)
    q = weights @ block_q  # LINEAR multiplicities, not squared weights.
    target_dot = weights @ (block_sum @ target)
    return ((sum_norm-q)/(n*(n-1)) - 2*target_dot/n + target@target)/scale


def extract():
    dest = HERE/'i_j_frozen_parameters.npz'
    if dest.exists():
        raise FileExistsError('Never overwrite frozen training parameters.')
    contract = read(SEARCH/'analysis/objective_frozen.json')
    original = Path(contract['artifact'])
    assert sha(original) == contract['sha256']
    sys.path.insert(0, str(WT))
    obj = pickle.loads(original.read_bytes())
    assert type(obj).__name__ == 'LabelFreeThreeObservableObjective'
    spec = dict(names=obj.names, scl=obj.scl.tolist(), icl=obj.icl.tolist(),
        pairs=[p.tolist() for p in obj.pairs], time_scales=obj.time_scales,
        n_fourier=obj.n_fourier, map_seed=obj.seed, scales=obj.scales,
        linear_scales={k:obj.maps[k]['linear_scale'] for k in MAPS},
        map_bandwidths={k:obj.maps[k]['bandwidth'] for k in MAPS},
        original_pickle=str(original), original_pickle_sha256=sha(original),
        original_class=type(obj).__module__+'.'+type(obj).__name__, target_and_scales_refitted=False)
    arrays = {g+'_target':obj.targets[g] for g in GROUPS}
    for k in MAPS:
        arrays[k+'_weights'], arrays[k+'_phases'] = obj.maps[k]['weights'], obj.maps[k]['phases']
    rows = list(csv.DictReader((HERE/'i_original_angle_response.csv').open())); checks = []
    with np.load(HERE/'i_events.npz') as z:
        assert z['contact_names'].tolist() == obj.names
        for row in rows:
            cid = row['candidate']; t = z[cid+'_centroid_ms']
            features = mapped_features(t, arrays, spec)
            original_features = obj.features(t, device='cpu')
            original_score = obj.score(t, device='cpu')
            historical = read(SEARCH/'scores'/f'{cid}.json')
            assert historical['objective_sha256'] == contract['sha256']
            for g in GROUPS:
                np.testing.assert_allclose(features[g], original_features[g], rtol=0, atol=1e-14)
            components = np.array([d_off(features[g], obj.targets[g])/obj.scales[g] for g in GROUPS])
            np.testing.assert_allclose(components, historical['components'], rtol=0, atol=1e-12)
            np.testing.assert_allclose(components.mean(), original_score['J'], rtol=0, atol=1e-12)
            # Independent event-pair kernel and unequal-block multiplicity checks.
            for g in GROUPS:
                x, target = features[g], obj.targets[g]; kernel = x@x.T
                explicit = (kernel.sum()-np.trace(kernel))/(len(x)*(len(x)-1)) - 2*(x@target).mean()+target@target
                np.testing.assert_allclose(explicit, d_off(x,target), rtol=0, atol=1e-12)
            bins = np.searchsorted(np.linspace(1500,60000,14),z[cid+'_event_time_ms'],side='right')-1
            test_weights = np.array([[2,0,1,3,0,1,1,0,2,1,0,1,1]])
            indices = np.concatenate([np.tile(np.flatnonzero(bins==b),int(test_weights[0,b])) for b in range(13)])
            explicit_score = obj.score(t[indices],device='cpu')
            test_scores = [bootstrap_group(features[g],obj.targets[g],obj.scales[g],bins,test_weights,13)[0] for g in GROUPS]
            np.testing.assert_allclose(test_scores,[explicit_score['groups'][g]['scaled'] for g in GROUPS],rtol=0,atol=1e-12)
            checks.append(dict(candidate=cid,J=historical['J'],recomputed_J=float(components.mean()),
                absolute_error=float(abs(components.mean()-historical['J'])),feature_map_agreement=True,
                explicit_offdiagonal_kernel_agreement=True,duplicated_blocks_materialized_scorer_agreement=True))
    np.savez_compressed(dest, **arrays); write(HERE/'i_j_frozen_spec.json',spec)
    folder=HERE/'i_j_original_source'; folder.mkdir(exist_ok=True); sources={}
    for name in ('topic4_three_observable_objective.py','topic4_three_observable_label_free.py'):
        f=WT/'src'/name; shutil.copyfile(f,folder/name); sources[str(f)]=sha(f)
    write(HERE/'i_j_scorer_validation.json',dict(checks=checks,scales=obj.scales,source_code_sha256=sources,
        original_pickle_sha256=sha(original),all_13_historical_J_and_39_scaled_components_reproduced=True,
        maximum_J_absolute_error=max(c['absolute_error'] for c in checks)))


def analyze():
    rows=list(csv.DictReader((HERE/'i_original_angle_response.csv').open()))
    spec=read(HERE/'i_j_frozen_spec.json'); provenance=read(HERE/'i_provenance.json')
    with np.load(HERE/'i_j_frozen_parameters.npz') as z:
        pars={k:z[k] for k in z.files}
    samples=[]
    with np.load(HERE/'i_events.npz') as z:
        for row in rows:
            cid=row['candidate']
            samples.append((mapped_features(z[cid+'_centroid_ms'],pars,spec),z[cid+'_event_time_ms']))
    angles=np.array([float(r['parameter_value']) for r in rows])
    actual_J=np.array([r['J'] for r in provenance['records']])
    original_components=np.array([r['J_components'] for r in provenance['records']])
    point_components=np.array([[d_off(f[g],pars[g+'_target'])/spec['scales'][g] for g in GROUPS] for f,_ in samples])
    np.testing.assert_allclose(point_components,original_components,rtol=0,atol=1e-12)
    np.testing.assert_allclose(point_components.mean(1),actual_J,rtol=0,atol=1e-12)
    rank=np.argsort(actual_J); best,second=int(rank[0]),int(rank[1]); boot,sensitivity={},[]
    for blocks in (13,26,6):
        weights=np.random.default_rng(SEED+blocks).multinomial(blocks,np.full(blocks,1/blocks),size=N_BOOT)
        edges=np.linspace(1500,60000,blocks+1); draws=[]
        for f,times in samples:
            bins=np.searchsorted(edges,times,side='right')-1
            draws.append(np.column_stack([bootstrap_group(f[g],pars[g+'_target'],spec['scales'][g],bins,weights,blocks) for g in GROUPS]))
        draws=np.stack(draws,axis=1); boot['scaled_components_'+str(blocks)]=draws
        totals=draws.mean(2); variance=totals.var(0,ddof=1); gap=totals[:,best]-totals[:,second]
        sensitivity.append(dict(blocks=blocks,block_ms=58500/blocks,J_argmin_deg=float(angles[best]),
            minimum_J_sd=float(np.sqrt(variance[best])),runner_up_J_sd=float(np.sqrt(variance[second])),
            variance_argmin_deg=float(angles[np.argmin(variance)]),
            best_minus_runner_up_q025=float(np.quantile(gap,.025)),best_minus_runner_up_q975=float(np.quantile(gap,.975)),
            best_lower_fraction=float(np.mean(gap<0))))
    primary=boot['scaled_components_13'];totals=primary.mean(2);variance=totals.var(0,ddof=1)
    covariance=np.array([np.cov(primary[:,i,:],rowvar=False,ddof=1) for i in range(13)])
    np.testing.assert_allclose(variance,covariance.sum(axis=(1,2))/9,rtol=1e-12,atol=1e-14)
    pareto=[not any(actual_J[j]<=actual_J[i] and variance[j]<=variance[i] and
              (actual_J[j]<actual_J[i] or variance[j]<variance[i]) for j in range(13)) for i in range(13)]
    out=[]
    for i,row in enumerate(rows):
        r=dict(row,optimizer_J=float(actual_J[i]),J_bootstrap_variance=float(variance[i]),
            J_bootstrap_sd=float(np.sqrt(variance[i])),J_bootstrap_mean=float(totals[:,i].mean()),
            J_bootstrap_q025=float(np.quantile(totals[:,i],.025)),J_bootstrap_q975=float(np.quantile(totals[:,i],.975)),
            empirical_J_minimum=i==best,J_variance_pareto=pareto[i],bootstrap_draws=N_BOOT,
            bootstrap_blocks=13,independent_runs=1)
        for j,g in enumerate(GROUPS):
            r['J_component_'+g]=float(original_components[i,j])
            r['J_component_'+g+'_bootstrap_variance']=float(primary[:,i,j].var(ddof=1))
        out.append(r)
    csv_write(HERE/'i_angle_response.csv',out); csv_write(HERE/'i_block_sensitivity.csv',sensitivity)
    np.savez_compressed(HERE/'i_bootstrap_draws.npz',angles_deg=angles,**boot)
    meta=dict(status='CANDIDATE_PENDING_AUTHOR_VISUAL_REVIEW',panel='I',primary_metric='Actual frozen training loss J',
        render_api='render_i(fig, rect); normalized data-axis rect; returns primary Axes, right Axes at ax._i_loss_axis',
        intended_data_axis_mm=[53,51],compatible_width_mm=[53,55,57],font_pt=dict(label=9,tick=8,legend=7),parent_rect_mm=[253,15,53,51],parent_canvas_mm=[320,245],
        parameter='EE_angle_offset_deg',angle_is_offset=True,subject='epilepsiae_1146',
        left_axis=dict(label='Error',metrics=['mean_rank_error','within_shaft_order_error','participation_error'],
            meaning='Original raw descriptive MAEs; preserved exactly; NOT literal additive summands of J.'),
        right_axis=dict(label='Training loss J',metric='optimizer_J',
            curve='13 original stored J values independently reproduced from raw events and frozen scorer',
            shade='Original J plus/minus one conditional time-block bootstrap SD; not a confidence interval'),
        aggregate=dict(formula='J=(1/3) sum_g D_off,g/s_g',groups=GROUPS,
            D_off_formula='(S_g dot S_g-Q_g)/(n*(n-1)) - 2*(S_g dot mu_patient,g)/n + mu_patient,g dot mu_patient,g',
            S_g='Sum of event feature vectors phi_g(x_i)',Q_g='Sum of squared norms of individual event feature vectors',
            equivalent='A_g-B_g; A_g=||S_g/n-mu_patient,g||^2; B_g=(Q_g/n-||S_g/n||^2)/(n-1)',
            normalization='Ordered distinct event-occurrence pairs n*(n-1); n changes with resampled block event counts',
            scales=spec['scales'],scales_fixed_during_bootstrap=True,patient_targets_fixed=True,
            feature_maps_fixed=True,negative_values_clipped=False,minimum_events=16,
            no_new_display_objective=True,J_joint='Not evaluated for this historical scan'),
        uncertainty=dict(statistical_unit='A 4.5 s physical-time block of events within one 60 s simulated trajectory',
            independent_runs_per_angle=1,seed_pair=[2511,847401],patient_fit_events=19770,
            burnin_ms=1500,end_ms=60000,blocks=13,draws=N_BOOT,ddof=1,
            same_block_weights_across_angles=True,rng='numpy.default_rng(2026102404+block_count)',
            duplicate_handling='Repeated blocks create repeated event occurrences; subtract Q with LINEAR multiplicities. Distinct copies remain in off-diagonal pairs, matching the original scorer on a materialized bootstrap sample.',
            covariance='Variance of jointly recomputed J, including covariance of its three calibrated components',
            limitations='Conditional finite-event uncertainty; no independent seed/network variance or patient uncertainty. Block exchangeability and finite temporal dependence are assumptions; block-size sensitivity retained.',
            all_draws_have_minimum_events=True),
        optimum=dict(J_argmin_deg=float(angles[best]),J_min=float(actual_J[best]),candidate=rows[best]['candidate'],
            J_min_bootstrap_sd=float(np.sqrt(variance[best])),J_min_bootstrap_variance=float(variance[best]),
            runner_up_angle_deg=float(angles[second]),runner_up_J=float(actual_J[second]),
            empirical_gap=float(actual_J[second]-actual_J[best]),variance_argmin_deg=float(angles[np.argmin(variance)]),
            minimum_variance=float(variance.min()),J_variance_pareto_angles=angles[np.array(pareto)].tolist(),
            individual_scaled_component_argmin_deg={g:float(angles[np.argmin(original_components[:,j])]) for j,g in enumerate(GROUPS)},
            star='Computed argmin of original J, not a forced reference marker',
            claim='Empirical minimum within 13 tested angles; neither unique population optimum nor joint minimum of all components and variance.'),
        panel_C_F_integration='Same algebraic definition J=(1/3)sum D_off,g/s_g. Frozen maps/scales may differ across experiments: do not assert identical calibration without checking parent sources. C repeats optimizations; I resamples within one run.',
        diagnostic=dict(renderer='i_display_panel.py',data='i_display_angle_response.csv',metadata='i_display_metadata.json',
            empirical_display_argmin_deg=16.66666666666667,display_mean_is_not_training_loss=True),
        validation=dict(all_39_raw_errors_preserved=True,all_13_J_and_39_actual_scaled_components_recomputed=True,
            original_scorer_and_explicit_event_pair_kernel_agree=True,materialized_unequal_block_bootstrap_agrees=True,
            source='i_j_scorer_validation.json'),block_sensitivity=sensitivity,
        frozen_training_parameters=dict(spec='i_j_frozen_spec.json',arrays='i_j_frozen_parameters.npz',
            original_pickle_sha256=spec['original_pickle_sha256']))
    write(HERE/'i_metadata.json',meta)
    print(json.dumps(meta['optimum'],indent=2)); print(json.dumps(sensitivity,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--extract',action='store_true')
    args=parser.parse_args()
    if args.extract:
        extract()
    analyze()
