"""Raw signals -> band energy -> fixed spatial fields and signed alignment.

Input can be newly simulated synaptic-current proxy or real referenced SEEG;
keep their units, baselines and interpretation separate in the notebook.
"""
import numpy as np
from scipy.signal import spectrogram
from scipy.stats import rankdata
from analysis.src.topic5_template_axis_field import make_field_scorer,score_field_batch,z_earliness,make_normalized_plane
from analysis.src.topic5_interictal_direction_rose import fit_event_directions_3d
from analysis.src.topic5_gradient_grid_field import build_grid_field


def band_energy(samples,sfreq,*,band=(30,80),window_sec=.1,step_sec=.01):
    x=np.asarray(samples,float)
    if x.ndim!=2 or not np.isfinite(x).all() or band[1]>=sfreq/2:
        raise ValueError('Finite channel x sample input; band must be below Nyquist')
    n=round(window_sec*sfreq);overlap=n-round(step_sec*sfreq)
    f,t,p=spectrogram(x,fs=sfreq,window='hann',nperseg=n,noverlap=overlap,
                       detrend='constant',scaling='density',mode='psd',axis=-1)
    use=(f>=band[0])&(f<=band[1])
    if use.sum()<2:raise ValueError('Spectral window cannot resolve this band')
    power=np.sum(p[:,use],axis=1)*(f[1]-f[0])
    return t,power


def fit_onset_geometry(observations, coords_mm):
    """Rebuild Fig5's timing-only TA/TB reference from interictal observations."""
    from sklearn.cluster import KMeans
    from analysis.src.lagpat_rank_audit import build_masked_kmeans_features, mask_phantom_ranks
    from analysis.src.interictal_propagation import _legacy_hist_mean_rank
    from analysis.src.propagation_skeleton_geometry import assign_events_to_templates
    from analysis.src.topic5_template_axis_field import _fit_propagation_axis, shared_bisector
    order=observations['discovery_contact_order']
    ranks=observations['ranks'][order];mask=observations['participating'][order]
    features=build_masked_kmeans_features(ranks,mask,impute='event_median')
    labels=KMeans(2,n_init=10,random_state=0).fit_predict(features)
    modes=np.argsort(-np.bincount(labels))  # TA is the larger timing-only cluster.
    templates=np.array([np.argsort(np.argsort(_legacy_hist_mean_rank(
        ranks[:,labels==k],mask[:,labels==k]))) for k in modes])
    # The paper's support is measured after nearest-template Spearman assignment.
    assigned=assign_events_to_templates(mask_phantom_ranks(ranks,mask,normalize=True),*templates)
    support=np.array([mask[:,assigned==k].mean(1) for k in range(2)])
    xyz=np.asarray(coords_mm)[order]
    axes=[_fit_propagation_axis(xyz,z_earliness(r)) for r in templates]
    direction=shared_bisector(*axes)['u']
    plane=make_normalized_plane(xyz,direction)
    if plane['status']!='ok':raise ValueError('Shared interictal plane is not estimable')
    # Keep the original discovery order for the fit, then return input contact order.
    inverse=np.argsort(order);plane['points']=plane['points'][inverse]
    templates=templates[:,inverse];support=support[:,inverse]
    scorer=make_field_scorer(z_earliness(templates[0]),plane['points'],support[0],plane['sigma'])
    return dict(plane=plane,scorer=scorer,contact_mask=np.ones(len(order),bool),
                rank_a=templates[0],rank_b=templates[1],support_a=support[0],support_b=support[1])


def energy_field_trajectory(samples,sfreq,xy,template_rank,support,*,baseline_sec,band=(30,80),window_sec=.1,step_sec=.01,geometry=None):
    t,power=band_energy(samples,sfreq,band=band,window_sec=window_sec,step_sec=step_sec)
    logp=np.log(np.maximum(power,np.finfo(float).tiny))
    # Complete windows must be inside the declared baseline interval.
    baseline=(t-window_sec/2>=baseline_sec[0])&(t+window_sec/2<=baseline_sec[1])
    if baseline.sum()<20:raise ValueError('Insufficient complete baseline windows')
    location=np.median(logp[:,baseline],axis=1);scale=1.4826*np.median(abs(logp[:,baseline]-location[:,None]),axis=1)
    valid=scale>1e-12;z=np.full_like(logp,np.nan);z[valid]=(logp[valid]-location[valid,None])/scale[valid,None]
    # Mirror symmetry must be around a frozen interictal axis, not y=0 at
    # the lower edge of the 20-mm simulation sheet. Fit geometry from ranks
    # only; ictal/current amplitudes never determine this plane.
    xy=np.asarray(xy,float)
    xyz=xy if xy.shape[1]==3 else np.column_stack([xy,np.zeros(len(xy))])
    rank=np.asarray(template_rank,float);support=np.asarray(support,float)
    use=np.isfinite(xyz).all(1)&np.isfinite(rank)&(support>0)
    if geometry is None:
        direction=fit_event_directions_3d(np.where(use,rank,np.nan)[:,None],xyz)['directions'][0]
        plane=make_normalized_plane(xyz[use],direction)
        if plane['status']!='ok':raise ValueError('Interictal template plane is not estimable')
        scorer=make_field_scorer(z_earliness(rank[use]),plane['points'],support[use],plane['sigma'])
    else:
        plane,scorer,use=geometry['plane'],geometry['scorer'],geometry['contact_mask']
    sigma=plane['sigma']
    scores=score_field_batch(scorer,z[use].T)
    return dict(time_sec=t,power=power,log_power=logp,robust_z=z,baseline_sec=baseline_sec,
                scores=scores,sigma=sigma,scorer=scorer,plane=plane,contact_mask=use)


def native_entry(time_ms,rate_E,*,dt_ms,threshold_hz=200,hold_ms=200):
    """First full 200-ms run of >=200-Hz nonoverlapping 10-ms all-E bins."""
    width=round(10/dt_ms);n=len(rate_E)//width
    b=np.asarray(rate_E[:n*width]).reshape(n,width).mean(1);count=0
    for i,value in enumerate(b):
        count=count+1 if value>=threshold_hz else 0
        if count>=round(hold_ms/10):
            return dict(onset_sec=(i+1-count)*.01,confirmation_sec=(i+1)*.01,criterion='high-rate entry; not proof of seizure oscillation')
    return dict(onset_sec=None,confirmation_sec=None,criterion='high-rate entry not observed')


def patient_energy_analysis(raw_seizure,interictal_record,*,band=(30,80)):
    """Fig3/Fig5 clinical extension from samples, with frozen interictal geometry.

    raw_seizure keys: samples_V (already referenced), sfreq, names,
    start_rel_tspectral_sec. No anatomy/template is learned from ictal samples.
    Returns full time scores; selection/onset windows must be specified upstream.
    """
    from analysis.src.topic5_tspectral_field_concordance import distal_baseline_robust_z,exact_name_align_matrix
    from analysis.src.topic5_template_axis_field import scorers_from_interictal_record,score_scorer_bundle_batch
    t,power=band_energy(raw_seizure['samples_V'],raw_seizure['sfreq'],band=band,window_sec=1.,step_sec=.1)
    t=t+raw_seizure['start_rel_tspectral_sec']
    norm=distal_baseline_robust_z(np.log(np.maximum(power,np.finfo(float).tiny)),t,(-120.,-90.))
    aligned=exact_name_align_matrix(interictal_record,raw_seizure['names'],norm['delta'])
    return dict(time_rel_tspectral_sec=t,normalization=norm,alignment=aligned,
                scores=score_scorer_bundle_batch(scorers_from_interictal_record(interictal_record),aligned['values'].T))
