"""Fig4 G/H/I readouts computed from new model events and patient observations."""
import numpy as np
from scipy.stats import rankdata
from analysis.from_events import cluster_events
from sklearn.cluster import KMeans
from modeling.src.topic4_interictal_repaired_evaluation import rank_features
from analysis.src.interictal_propagation import assign_events_to_templates
from modeling.src.topic4_d6_natural_kmeans import (contact_split_folds,patient_profiles,
    crossfit_patient_readout,assign_to_profiles,normalize_event_ranks)
from modeling.src.topic4_nlc_null_calibration import contact_permutation_matrix_draws,crossfit_matrix
from analysis.src.lagpat_rank_audit import mask_phantom_ranks


def event_ranks(centroid_ms):
    times=np.asarray(centroid_ms,float);mask=np.isfinite(times);r=np.full_like(times,np.nan)
    for i,row in enumerate(times):r[i,mask[i]]=rankdata(row[mask[i]],method='average')-1
    return r


def centroid_events(centroid_ms, names):
    """Patient or model event times -> the same masked rank representation."""
    ranks = event_ranks(centroid_ms)
    return dict(ranks=np.where(np.isfinite(ranks), ranks, -1).T,
                participating=np.isfinite(ranks).T, channel_names=np.asarray(names))


def fit_patient_templates(times,names,coords_mm,reference=None):
    """Timing+Space patient profiles and the paper's rank-only model classifier.

    Discover templates from the supplied interictal observations; train the
    rank-only classifier on FIT. Model trajectories never change either fit
    or their semantic TA/TB mapping.
    """
    from analysis.from_events import spatial_templates
    from analysis.src.topic5_interictal_direction_rose import fit_event_directions_3d
    events=centroid_events(times,names) if reference is None else reference
    if reference is not None:
        np.testing.assert_array_equal(reference['channel_names'],names)
    order=np.arange(len(names)) if reference is None else reference['discovery_contact_order']
    events=dict(ranks=events['ranks'][order],participating=events['participating'][order])
    # The paper first removes phantom-rank gaps, then fits the signed gradient.
    directions=fit_event_directions_3d(mask_phantom_ranks(events['ranks'],events['participating']),coords_mm[order])
    result=spatial_templates(events,coords_mm[order],directions)
    inverse=np.argsort(order)
    result['templates']=result['templates'][:,inverse]
    result['support']=result['support'][:,inverse]
    km=KMeans(2,n_init=10,random_state=2026090619).fit(rank_features(times))
    if reference is None:
        valid=result['labels']>=0
        swap=np.mean(km.labels_[valid]==result['labels'][valid])<.5
    else:
        result['discovery_labels']=result['labels'].copy()
        result['labels']=result['labels'][reference['development_indices']]
        predicted=km.predict(rank_features(reference['development_centroid_ms']))
        swap=np.mean(predicted==result['labels'])<.5
    result.update(mode_classifier=km,mode_remap=np.array([1,0] if swap else [0,1]),
                  classifier_rule='Patient FIT rank KMeans; semantic alignment to '+
                      ('full interictal' if reference is not None else 'FIT')+' Timing+Space discovery')
    return result


def classify_model(times,patient_model):
    times=np.asarray(times,float);readable=np.isfinite(times).sum(1)>=2
    labels=np.full(len(times),-1,int)
    labels[readable]=patient_model['mode_remap'][patient_model['mode_classifier'].predict(rank_features(times[readable]))]
    return labels


def model_patient_comparison(model_centroids,patient_centroids,names,contact_contract,patient_model,*,n_null=1000,patient_ranks=None):
    model=event_ranks(model_centroids)
    # G/H retain the measured patient rank values (including gaps), then
    # normalize each event's range. Missing contacts remain NaN throughout.
    patient=event_ranks(patient_centroids) if patient_ranks is None else np.asarray(patient_ranks,float)
    np.testing.assert_array_equal(np.isfinite(patient),np.isfinite(patient_centroids))
    labels=patient_model['labels'];keep=labels>=0
    profiles=patient_profiles(patient[keep],labels[keep])
    # Contact folds are fixed by the supplied interictal geometry contract.
    rows=contact_contract['contacts'];assert set(r['contact_name'] for r in rows)==set(names)
    contract=dict(contacts=[dict(r,contact_index=names.index(r['contact_name'])) for r in rows])
    folds=contact_split_folds(contract)
    crossfit=crossfit_patient_readout(model,patient[keep],labels[keep],folds)
    assigned=classify_model(model_centroids,patient_model)
    normalized=normalize_event_ranks(model)
    mean=[]
    for k in range(2):
        x=normalized[assigned==k];count=np.isfinite(x).sum(0)
        mean.append(np.divide(np.nansum(x,0),count,out=np.full(len(names),np.nan),where=count>0))
    np.testing.assert_allclose(crossfit_matrix(normalized,profiles,folds),crossfit['matrix'],rtol=1e-12,atol=1e-12,equal_nan=True)
    shaft_by_name={r['contact_name']:r['shaft_id'] for r in rows}
    null=contact_permutation_matrix_draws(model,patient[keep],labels[keep],folds,draws=n_null,seed=20260918,
        shaft_ids=np.asarray([shaft_by_name[n] for n in names]))
    matrix=np.asarray(crossfit['matrix']);p=[]
    for k in range(2):
        good=np.isfinite(null[:,k,k]);p.append(float((1+np.sum(null[good,k,k]>=matrix[k,k]))/(1+good.sum())) if good.any() and np.isfinite(matrix[k,k]) else None)
    return dict(patient_profiles=profiles,model_profiles=np.asarray(mean),matrix=matrix,
        model_labels=assigned,patient_labels=labels,folds=crossfit,n_null=n_null,
        within_shaft_null=null,diagonal_p=p,unit='one newly simulated network; not a patient cohort')
