"""Fit templates, statistics and spatial fields from newly computed event tables."""
import re
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import wilcoxon
from analysis.src.lagpat_rank_audit import build_masked_kmeans_features
from analysis.src.interictal_propagation import build_cluster_templates, compute_legacy_mi
from analysis.src.interictal_spatial_information_gain import compute_crossfit_all_event_spatial_information_gain
from analysis.src.topic5_interictal_direction_rose import fit_event_directions_3d
from analysis.src.topic5_template_axis_field import z_earliness, make_normalized_plane, build_interictal_template_field_record
from analysis.src.topic5_gradient_grid_field import make_adaptive_grid, build_grid_field


def cluster_events(events, k=2, seed=0):
    ranks, mask = events['ranks'], events['participating']
    valid = mask.sum(axis=0) >= 3
    if valid.sum() < 2*k:
        raise ValueError('Too few events with >=3 participating contacts')
    features = build_masked_kmeans_features(ranks[:,valid], mask[:,valid], impute='event_median')
    if isinstance(features, tuple):
        features = features[0]
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(features)
    order = np.argsort(-np.bincount(km.labels_, minlength=k)); remap=np.argsort(order)
    labels = np.full(ranks.shape[1], -1); labels[valid] = remap[km.labels_]
    templates = build_cluster_templates(ranks[:,valid], mask[:,valid], labels[valid], n_clusters=k)
    return dict(labels=labels, templates=templates, valid=valid, k=k,
                support=np.array([mask[:,labels==j].mean(axis=1) for j in range(k)]))


def patient_statistics(events, clustered, permutations=200, seed=0):
    r, b = events['ranks'], events['participating']
    valid = b.sum(0)>=3
    overall = compute_legacy_mi(r[:,valid], b[:,valid], n_permutations=permutations, seed=seed)
    within = []
    for k in range(clustered['k']):
        ix = clustered['labels']==k
        within.append(compute_legacy_mi(r[:,ix], b[:,ix], n_permutations=permutations, seed=seed+k+1))
    return dict(overall=overall, within=within, scope='one patient; events are not cohort replicates')


def cohort_statistics(patient_results):
    """One row per independent patient; no event-level pseudo-replication."""
    if len(set(patient_results)) != len(patient_results):
        raise ValueError('Patient identifiers must be unique')
    data=np.array([v['overall']['mi_mean'] for v in patient_results.values()])
    null=np.array([v['overall']['permuted_mean_median'] for v in patient_results.values()])
    within=np.array([np.mean([x['mi_mean'] for x in v['within']]) for v in patient_results.values()])
    ok=np.isfinite(data)&np.isfinite(null)&np.isfinite(within)
    result=dict(n_patients=int(ok.sum()), observed=data, null=null, within=within,
                unit='patient', p_vs_null=None, p_within_vs_overall=None)
    if ok.sum()>=2:
        def paired(a,b):
            return 1.0 if np.allclose(a,b) else float(wilcoxon(a,b,alternative='greater').pvalue)
        result.update(p_vs_null=paired(data[ok],null[ok]),p_within_vs_overall=paired(within[ok],data[ok]))
    return result


def align_coordinates(events, coordinates):
    return np.asarray([coordinates.get(str(n), [np.nan]*3) for n in events['channel_names']],float)


def direction_and_crossfit(events, xyz, *, n_null=1000, n_train_null=50):
    masked=np.where(events['participating'],events['ranks'],np.nan)
    directions=fit_event_directions_3d(masked,xyz)
    result=compute_crossfit_all_event_spatial_information_gain(events['ranks'],events['participating'],
        directions['directions'],events['block_ids'],xyz,n_null=n_null,n_train_spatial_null=n_train_null)
    return directions,result


def template_fields(events, clustered, xyz, *, n_boot=200, subject_id="computed_example", subject="Y1", dataset="raw_example"):
    """Actual current field constructor; preserve its geometry/axis qualification."""
    names=events['channel_names']; sh=[re.sub(r'\d.*$', '', str(n)) for n in names]
    return build_interictal_template_field_record(subject_id=subject_id,dataset=dataset,
        subject=subject,stable_k=clustered['k'],names=names,coords=xyz,
        rank_ta=clustered['templates'][0],rank_tb=clustered['templates'][1],shafts=sh,
        support_ta=clustered['support'][0],support_tb=clustered['support'][1],
        support_source='new raw detection and clustering',n_axis_boot=n_boot,n_pair_boot=n_boot,seed=0)


def illustrative_grid(events, clustered, xyz, k, plane=None):
    """Compute the same Gaussian field even if the short excerpt fails axis QC.

    Caller must label a failed-QC illustration; it is never promoted to a
    qualified interictal template field. Both templates can use the TA plane.
    """
    rank=clustered['templates'][k]; support=clustered['support'][k]
    use=np.isfinite(xyz).all(1)&np.isfinite(rank)&(support>0)
    direction=fit_event_directions_3d(np.where(use,rank,np.nan)[:,None],xyz)['directions'][0]
    if not np.isfinite(direction).all():
        raise ValueError('Template direction not estimable')
    if plane is None:
        plane=make_normalized_plane(xyz[use],direction)
    # Project both fields into the SAME plane origin/axis/scale, using current support.
    rel=xyz[use]-plane['origin']
    points=np.column_stack([rel@plane['u'],rel@plane['w']])/plane['scale_mm']
    values=z_earliness(rank[use]); weights=support[use]
    grid=make_adaptive_grid(points,plane['sigma'],float(weights.sum()),model='computed')
    return dict(plane=plane,points=points,values=values,support=weights,grid=grid,
                field=build_grid_field(grid['X'],grid['Y'],points,weights,values,plane['sigma']),direction=direction)


def shared_field_similarity(record,n_perm=1000,seed=0):
    """Fig2D actual observable: contact-evaluated shared-plane field correlation.

    This is different from the cosine between the two rank-gradient axes.
    Full-contact shuffling moves support+earliness together on a fixed plane.
    """
    from analysis.shared_field_null import _pearson_r,_null_correlations,_permutation_indices
    fields=record.get('interictal_field',{});models=fields.get('field_models',{})
    if not record.get('axis_pair',{}).get('geometry_2d_supported') or 'shared_a' not in models:
        return dict(status='NOT_ESTIMABLE',reason='No supported shared-plane fields')
    a,b=models['shared_a'],models['shared_b'];observed=_pearson_r(a['template_field'],b['template_field'])
    perm,unique,exact=_permutation_indices(list(fields['shafts']),'channel',n_perm,np.random.default_rng(seed))
    null=_null_correlations(np.asarray(a['template_field']),np.asarray(fields['earliness_b']),
        np.asarray(fields['support_b']),np.asarray(b['points']),b['sigma'],perm)
    p=float(np.mean(null<=observed+1e-12)) if exact else float((1+np.sum(null<=observed))/(len(null)+1))
    return dict(status='OK',observed_shared_field_r=observed,channel_null=null,p_negative=p,n_unique=unique,exact=exact,unit='one patient')


def spatial_templates(events,xyz,directions):
    """Final descriptive fields use an all-event Timing+Space fit, after CV.

    This full fit does not enter the held-out scores. Missing spatial views keep
    their temporal information, exactly as in the current paper contract.
    """
    from analysis.src.interictal_spatial_information_gain import fit_full_all_event_spatial_template_model
    fit=fit_full_all_event_spatial_template_model(events['ranks'],events['participating'],directions['directions'],xyz)
    return dict(labels=fit['labels'],templates=fit['templates'],support=fit['supports'],k=2,
                valid=np.ones(events['ranks'].shape[1],bool),fit=fit)
