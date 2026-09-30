"""One paired diagnostic: replace projected mean with measured native mean.

Raw private variance forcing, Z, threshold, random stream and time grid stay
identical to the completed local reference. Does not fit or launch a network.
"""
from native_input_local_lif import simulate,condition,DEST,LOCAL,np,read,write
from native_input_fixed_readout import direct_features,flux,load_models,physical_from_features
from datetime import datetime
import torch,argparse


def main(device):
    target=LOCAL/'measured_mean_contrast';target.mkdir(exist_ok=True)
    assert not (target/'contract.json').exists()
    c=read(LOCAL/'contract.json')
    write(target/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Do observed groupmean currenterrors explain native shortevent tails missed even by GaussianLIF?',
        only_change='Use actualpre-step native netinputmean instead of projected AMPA minus Z GABA minusM. Identicalrawprivatevarianceforcing,postfilterZ,meangroupthreshold,noise seed andtwo timesteps.',
        groups=c['groups'],replicates=c['min_replicates_per_case'],seed=c['seed'],dt_ms=c['dt_ms'],
        budget='Sixgroups attwo steps; fixedrate response onthese samemeans; nofit ornetworkrun.',
        comparisons='Samepredeclared50mswindows andthreeexistingtimeintervals; pairedCRN counts plus fixedreadout. Report errors continuously, no posthoc passgate.',
        scope='Teacher-forced diagnostic only; exact inputmean does not reconstruct actualcurrentdistribution or spikeinputcorrelation.'))
    z=np.load(DEST/'selected_input_history.npz');raw=np.load(DEST/'selected_raw_variance_forcing.npz');r=z['reconstructed'];t=z['time_ms']
    m={k:z['moments'][:,j] for j,k in enumerate(z['moment_names'])};G=len(z['groups']);sizes=z['group_size'].astype(int)
    old=np.load(LOCAL/'dt0.1.npz');nrep=old['replicates'][:G];R=int(nrep.max())
    theta=np.broadcast_to(z['theta'][:,None],(G,R)).copy()
    wave=np.stack([m['net'],raw['raw_variances'][:,2],raw['raw_variances'][:,3],m['z']],axis=1).transpose(2,1,0).copy()
    for dt in c['dt_ms']:
        pars=np.array([condition(0.,z['theta'][j],1.,1.,'E' if z['population'][j]==0 else 'I',dt=dt) for j in range(G)])
        counts=simulate(pars,wave,theta,nrep,dt,1000.,27,50.,c['seed'],device)
        np.savez_compressed(target/f'dt{dt:g}.npz',counts=counts,replicates=nrep,group_sizes=sizes,groups=z['groups'],dt_ms=dt)
    from lif_mc import PARAMS
    tm=np.where(z['population']==0,20.,10.);ref=np.where(z['population']==0,2.,1.)
    ta=PARAMS['tau_r_AMPA']+PARAMS['tau_d_AMPA'];tg=PARAMS['tau_r_GABA']+PARAMS['tau_d_GABA']
    physical=np.stack([m['net'],2*ta/tm*r[:,6],2*tg/tm*m['z']**2*r[:,7]],axis=2)
    f=direct_features(physical,z['theta']);nets,bases,_=load_models();ell=np.empty((len(t),G))
    for pop,key in [(0,'E'),(1,'I')]:
        mask=z['population']==pop;features=f[:,mask].reshape(-1,39);ths=np.broadcast_to(z['theta'][mask],(len(t),mask.sum())).ravel()
        base=bases[key].evaluate(physical_from_features(features,ths),ths)
        with torch.no_grad():ell[:,mask]=nets[key].logits(torch.tensor(features),torch.tensor(base)).numpy().reshape(len(t),mask.sum())
    rates=flux(ell,ref);fixed=np.load(DEST/'fixed_readout.npz');starts=fixed['bin_start_ms'];pred=[]
    for lo in starts:pred.append(rates[(t>=lo)&(t<lo+50)].sum(0)*.1/1000*sizes)
    pred=np.array(pred);np.savez_compressed(target/'fixed_rate.npz',time_ms=t,rate_hz=rates,bin_start_ms=starts,counts=pred)
    new=np.load(target/'dt0.1.npz');fine=np.load(target/'dt0.05.npz');rows=[]
    for j,g in enumerate(z['groups']):
        nr=int(nrep[j]);N=sizes[j];a=old['counts'][j,:nr].mean(0)*N;b=new['counts'][j,:nr].mean(0)*N
        ff=fine['counts'][j,:nr].mean(0)*N
        numerical=float(np.linalg.norm(b-ff)/max(np.linalg.norm(ff),1))
        for lo,hi in [(9000,9420),(9420,9868.5),(9868.5,10370)]:
            keep=(starts>=lo)&(starts+50<=hi);native=fixed['native_counts'][keep,j];norm=max(np.linalg.norm(native),1)
            rows.append(dict(group=int(g),window_ms=[lo,hi],native_count=int(native.sum()),
                MC_projected_mean_count=float(a[keep].sum()),MC_measured_mean_count=float(b[keep].sum()),rate_measured_mean_count=float(pred[keep,j].sum()),
                projected_mean_MC_vs_native_L2=float(np.linalg.norm(a[keep]-native)/norm),
                measured_mean_MC_vs_native_L2=float(np.linalg.norm(b[keep]-native)/norm),
                measured_mean_rate_vs_MC_L2=float(np.linalg.norm(pred[keep,j]-b[keep])/max(np.linalg.norm(b[keep]),1)),
                fullwindow_dt_L2=numerical))
    write(target/'result.json',dict(status='PAIRED_MEAN_DIAGNOSTIC_COMPLETE',rows=rows,model_promoted=False,
        boundary='SameGaussianprivateinputassumption andsameconditioned nativeZ/M; nativeprediction range not automatically calibrated. Stage errors have commonnative normalization; nofit oracceptancepromotion.'))
    print(rows,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)
