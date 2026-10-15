"""Fig4 G/H/I readouts computed from new model events and patient observations."""
import numpy as np
from scipy.stats import rankdata
from analysis.from_events import cluster_events
from modeling.src.topic4_d6_natural_kmeans import (contact_split_folds,patient_profiles,
    crossfit_patient_readout,assign_to_profiles,normalize_event_ranks)
from modeling.src.topic4_nlc_null_calibration import contact_permutation_matrix_draws,crossfit_matrix


def event_ranks(centroid_ms):
    times=np.asarray(centroid_ms,float);mask=np.isfinite(times);r=np.full_like(times,np.nan)
    for i,row in enumerate(times):r[i,mask[i]]=rankdata(row[mask[i]],method='dense')-1
    return r


def model_patient_comparison(model_centroids,patient_centroids,names,contact_contract,*,n_null=1000):
    model=event_ranks(model_centroids);patient=event_ranks(patient_centroids)
    e=dict(ranks=np.where(np.isfinite(patient),patient,-1).T,participating=np.isfinite(patient).T,channel_names=np.asarray(names))
    cluster=cluster_events(e);labels=cluster['labels'];keep=labels>=0
    profiles=patient_profiles(patient[keep],labels[keep])
    # Contact folds are fixed by the supplied interictal geometry contract.
    rows=contact_contract['contacts'];assert set(r['contact_name'] for r in rows)==set(names)
    contract=dict(contacts=[dict(r,contact_index=names.index(r['contact_name'])) for r in rows])
    folds=contact_split_folds(contract)
    crossfit=crossfit_patient_readout(model,patient[keep],labels[keep],folds)
    assigned,_=assign_to_profiles(model,profiles,np.arange(len(names)))
    normalized=normalize_event_ranks(model)
    mean=[]
    for k in range(2):
        x=normalized[assigned==k];count=np.isfinite(x).sum(0)
        mean.append(np.divide(np.nansum(x,0),count,out=np.full(len(names),np.nan),where=count>0))
    np.testing.assert_allclose(crossfit_matrix(normalized,profiles,folds),crossfit['matrix'],rtol=1e-12,atol=1e-12,equal_nan=True)
    shaft_by_name={r['contact_name']:r['shaft_id'] for r in rows}
    null=contact_permutation_matrix_draws(model,patient[keep],labels[keep],folds,draws=n_null,seed=20261015,
        shaft_ids=np.asarray([shaft_by_name[n] for n in names]))
    matrix=np.asarray(crossfit['matrix']);p=[]
    for k in range(2):
        good=np.isfinite(null[:,k,k]);p.append(float((1+np.sum(null[good,k,k]>=matrix[k,k]))/(1+good.sum())) if good.any() and np.isfinite(matrix[k,k]) else None)
    return dict(patient_profiles=profiles,model_profiles=np.asarray(mean),matrix=matrix,
        model_labels=assigned,patient_labels=labels,folds=crossfit,n_null=n_null,
        within_shaft_null=null,diagonal_p=p,unit='one newly simulated network; not a patient cohort')
