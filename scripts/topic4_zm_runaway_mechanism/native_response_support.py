"""Training-input support audit for frozen local response, no fitting."""
from common import OUT,np,read,write
from native_input_fixed_readout import direct_features
from scipy.spatial import cKDTree
from lif_mc import PARAMS
import time

DEST=OUT/'native_input_response_support'


def main():
    DEST.mkdir(exist_ok=True);assert not (DEST/'result.json').exists()
    z=np.load(OUT/'native_input_bridge/selected_input_history.npz');t=z['time_ms'];r=z['reconstructed'];m={k:z['moments'][:,j] for j,k in enumerate(z['moment_names'])}
    tm=np.where(z['population']==0,20.,10.);ta=PARAMS['tau_r_AMPA']+PARAMS['tau_d_AMPA'];tg=PARAMS['tau_r_GABA']+PARAMS['tau_d_GABA']
    physical=np.stack([r[:,0]-m['z']*r[:,1]-m['mcurrent'],2*ta/tm*r[:,6],2*tg/tm*m['z']**2*r[:,7]],axis=2)
    f=direct_features(physical,z['theta']);rows=[];references={};rng=np.random.default_rng(920124)
    for p,pop in enumerate('EI'):
        folder=OUT/'conditioned_refractory_rate';train=np.load(folder/f'training_arrays/{pop}.npz')['flux_features'].astype(float)
        normalization=np.load(folder/f'conditioning/{pop}.npz');center=normalization['center'];A=normalization['transform']
        white=(train-center)@A.T;lower=train.min(0);upper=train.max(0);wl=white.min(0);wu=white.max(0)
        norm=np.linalg.norm(white,axis=1);q=np.quantile(norm,[.5,.9,.99,.999,1.])
        indices=rng.choice(len(train),12288,replace=False);tree=cKDTree(white[indices[:8192]])
        distance=tree.query(white[indices[8192:]],workers=1)[0];dq=np.quantile(distance,[.5,.9,.99,1.])
        references[pop]=dict(training_rows=len(train),train_white_norm_quantiles=q.tolist(),
            heldout_training_neighbor_distance_quantiles=dq.tolist(),reference_count=8192,training_probe_count=4096)
        for j in np.flatnonzero(z['population']==p):
            features=f[:,j];w=(features-center)@A.T;outside=(features<lower)|(features>upper)
            # Ten-ms subsampling is declared only for neighbor-distance cost;
            # full0.1ms feature-box/norm checks retained.
            select=np.arange(0,len(t),100);d=tree.query(w[select],workers=1)[0]
            for lo,hi in [(9000,9420),(9420,9868.5),(9868.5,10370)]:
                keep=(t>=lo)&(t<hi);ks=(t[select]>=lo)&(t[select]<hi);normn=np.linalg.norm(w[keep],axis=1)
                rows.append(dict(group=int(z['groups'][j]),pop=pop,window_ms=[lo,hi],
                    fraction_outside_any_training_component=float(outside[keep].any(1).mean()),
                    fraction_outside_instantaneous_input_components=float(outside[keep,:3].any(1).mean()),
                    fraction_beyond_training_white_norm_max=float((normn>q[-1]).mean()),
                    fraction_beyond_training_white_norm_p999=float((normn>q[-2]).mean()),
                    white_norm_p50_p99_max=np.quantile(normn,[.5,.99,1.]).tolist(),
                    neighbor_distance_p50_p90_max=np.quantile(d[ks],[.5,.9,1.]).tolist(),
                    fraction_neighbor_distance_beyond_training_probe_p99=float((d[ks]>dq[-2]).mean()),
                    maximum_component_excess=float(np.maximum(np.maximum(lower-features[keep],features[keep]-upper),0).max())))
        print('SUPPORT',pop,references[pop],flush=True)
    np.savez_compressed(DEST/'native_features.npz',features=f,time_ms=t,physical=physical,groups=z['groups'],theta=z['theta'],population=z['population'])
    write(DEST/'result.json',dict(status='SUPPORT_DIAGNOSTIC_COMPLETE',rows=rows,training_reference=references,
        definitions='Input component bounds, conditioned norm, and nearest-neighbor distances to one fixed8192training-row subset;4096othertraining rows provide a scale reference. Within-bounds does not prove interpolation/convex-hull membership or response validity.',
        data='Nativeobserved groupcounts-derived inputhistories;1s preparation excluded; no native firing or LIF targets used to change response.',
        scope='Read-only fixedresponse input-domain diagnosis. Onehistory, no independentreplicates, no modelrepair or promotion.',seed=920124))
    print('SUPPORT COMPLETE',rows,flush=True)


if __name__=='__main__':main()
