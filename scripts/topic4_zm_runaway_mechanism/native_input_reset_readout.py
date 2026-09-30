"""Fixed post-fit diagnostic on the two existing native-input mean contrasts.

No fitting or autonomous network: determine whether the failed broad local
candidate nevertheless repaired the specific preentry response discrepancy.
"""
from reset_memory_rate import *
from reset_memory_numerics import direct
from native_input_fixed_readout import direct_features


def main():
    target=DEST/'native_input_diagnostic';target.mkdir(exist_ok=True);assert not (target/'result.json').exists()
    write(target/'contract.json',dict(status='FIXED_POST_FIT_DIAGNOSTIC',question='Did own-reset memory remove the measured preentry local response discrepancy?',
        design='Two existing supplied inputmean contrasts times six fixed groups; locked weights, no native firing in refractory/reset history.1s preparation and all previous50ms windows retained.',
        gate='Broad local validation alreadyFAIL; this diagnostic cannot override it or authorize network/bifurcation promotion. No new fitting or LIF data.',
        numerical='Nativeclock0.1ms; existing general0.05ms waveform gate includes5failures. This additional descriptive assay does not establish step convergence.'))
    src=OUT/'native_input_bridge';z=np.load(src/'selected_input_history.npz');data=np.load(OUT/'native_input_response_support/native_features.npz')
    physical=data['physical'];t=z['time_ms'];G=len(z['groups']);m={k:z['moments'][:,j] for j,k in enumerate(z['moment_names'])}
    measured=physical.copy();measured[:,:,0]=m['net'];nets,bases,locked=load_models();rates={};pred={}
    ref=np.load(src/'fixed_readout.npz');starts=ref['bin_start_ms']
    for label,inputs in [('projected_mean',physical),('measured_mean',measured)]:
        f=direct_features(inputs,z['theta']);r=np.empty((len(t),G))
        for j in range(G):
            pop='EI'[int(z['population'][j])];b=bases[pop].evaluate(inputs[:,j],z['theta'][j])
            r[:,j],_,_=direct(nets[pop],f[:,j],b,.1)
        rates[label]=r;counts=[]
        for lo in starts:
            keep=(t>=lo)&(t<lo+50);assert keep.sum()==500
            counts.append(r[keep].sum(0)*.1/1000*z['group_size'])
        pred[label]=np.array(counts)
    oldactual=np.load(src/'local_lif/measured_mean_contrast/fixed_rate.npz')['counts'];rows=[]
    for label in rates:
        old=ref['counts_projected_private'] if label=='projected_mean' else oldactual
        mc=np.load(src/('local_lif/dt0.1.npz' if label=='projected_mean' else 'local_lif/measured_mean_contrast/dt0.1.npz'))
        for j,g in enumerate(z['groups']):
            nr=int(mc['replicates'][j]);expected=mc['counts'][j,:nr].mean(0)*z['group_size'][j]
            for lo,hi in [(9000,9420),(9420,9868.5),(9868.5,10370)]:
                keep=(starts>=lo)&(starts+50<=hi);den=max(np.linalg.norm(expected[keep]),1)
                rows.append(dict(mean=label,group=int(g),window_ms=[lo,hi],native_count=int(ref['native_counts'][keep,j].sum()),LIF_count=float(expected[keep].sum()),parent_rate_count=float(old[keep,j].sum()),reset_rate_count=float(pred[label][keep,j].sum()),
                    parent_rate_vs_LIF_L2=float(np.linalg.norm(old[keep,j]-expected[keep])/den),reset_rate_vs_LIF_L2=float(np.linalg.norm(pred[label][keep,j]-expected[keep])/den)))
    np.savez_compressed(target/'predictions.npz',time_ms=t,bin_start_ms=starts,**rates,**{k+'_counts':v for k,v in pred.items()})
    write(target/'result.json',dict(status='DESCRIPTIVE_DIAGNOSTIC_COMPLETE',rows=rows,weights=locked['files'],model_promoted=False,
        scope='Supplied native-derived input histories, own predicted flux histories. No closed network, no statistical population inference, no bifurcation certification.'))
    log('NATIVE RESET READOUT',[(r['mean'],r['group'],round(r['parent_rate_vs_LIF_L2'],4),round(r['reset_rate_vs_LIF_L2'],4)) for r in rows if r['window_ms'][0]==9000])


if __name__=='__main__':main()
