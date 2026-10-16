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


def event_ranks(centroid_ms):
    times=np.asarray(centroid_ms,float);mask=np.isfinite(times);r=np.full_like(times,np.nan)
    for i,row in enumerate(times):r[i,mask[i]]=rankdata(row[mask[i]],method='average')-1
    return r


def centroid_events(centroid_ms, names):
    """Patient or model event times -> the same masked rank representation."""
    ranks = event_ranks(centroid_ms)
    return dict(ranks=np.where(np.isfinite(ranks), ranks, -1).T,
                participating=np.isfinite(ranks).T, channel_names=np.asarray(names))


def fit_patient_templates(times,names,coords_mm):
    """Timing+Space patient profiles and the paper's rank-only model classifier.

    Both are fitted only to patient FIT observations. Model trajectories never
    change the patient templates or their semantic TA/TB mapping.
    """
    from analysis.from_events import spatial_templates
    from analysis.src.topic5_interictal_direction_rose import fit_event_directions_3d
    events=centroid_events(times,names)
    directions=fit_event_directions_3d(np.where(events['participating'],events['ranks'],np.nan),coords_mm)
    result=spatial_templates(events,coords_mm,directions)
    km=KMeans(2,n_init=10,random_state=2026090619).fit(rank_features(times))
    valid=result['labels']>=0
    swap=np.mean(km.labels_[valid]==result['labels'][valid])<.5
    result.update(mode_classifier=km,mode_remap=np.array([1,0] if swap else [0,1]),
                  classifier_rule='Patient FIT rank KMeans; semantic alignment to Timing+Space by FIT agreement')
    return result


def classify_model(times,patient_model):
    times=np.asarray(times,float);readable=np.isfinite(times).sum(1)>=2
    labels=np.full(len(times),-1,int)
    labels[readable]=patient_model['mode_remap'][patient_model['mode_classifier'].predict(rank_features(times[readable]))]
    return labels


def select_propagation_events(runs, templates, names, burnin_ms):
    """First chronological eligible event of each patient template at each workpoint."""
    comparison = sorted([run for run in runs if run['score']['J'] is not None],
                        key=lambda run: -run['score']['J'])
    selected = []
    for run in comparison:
        labels = classify_model(run['observed']['centroid_ms'],templates)
        eligible = run['observed']['primary_event_indices']
        found = [[i for i in eligible if labels[i] == k and
                  run['observed']['events'][i]['window_ms'][0] >= burnin_ms]
                 for k in range(2)]
        selected.append([group[0] for group in found] if all(found) else None)
    ready = len(comparison) == 2 and all(ids is not None for ids in selected)
    return comparison, dict(events=selected,
        status='READY' if ready else 'NOT_ESTIMABLE',
        rule='first eligible chronological event per freshly fitted patient template')


def model_patient_comparison(model_centroids,patient_centroids,names,contact_contract,patient_model,*,n_null=1000):
    model=event_ranks(model_centroids);patient=event_ranks(patient_centroids)
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
