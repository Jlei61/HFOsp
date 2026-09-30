"""Fixed rate readout under actual early native g40 inputs; no fitting.

Counts upstream are teacher forcing, while local refractory history uses the
rate model's own expected flux. This never counts as autonomous validation.
"""
from common import OUT, model, np, read, write, log
from scipy import sparse
from scipy.linalg import expm
from numba import njit
from conditioned_refractory_rate import load_models
from nonlinear_rate_response import physical_from_features
from reconstruct_native_inputs import filter_inputs
import torch, argparse, os, gc

DEST=OUT/'native_early_surround_inputs'


def arrivals(s,counts,selected,device):
    import cupy as cp
    cp.cuda.Device(device).use();T=len(counts);G=len(selected)
    matrices=[sparse.load_npz(s.folder/f'{name}.npz')[selected].tocsr() for name in
        ['mean_ampa','mean_gaba','variance_ampa','variance_gaba']]
    matrices += [sparse.load_npz(OUT/f'physical_delay_variance_split/physical_private_{kind}.npz')[selected].tocsr() for kind in ['ampa','gaba']]
    kernel=cp.RawKernel(r'''
extern "C" __global__ void apply(const int* ptr,const int* col,const double* val,
 const unsigned short* counts,const double* sizes,double* out,int P,int G,int channel){
 int g=blockIdx.x,k=blockIdx.y,lane=threadIdx.x;double value=0.;
 for(int j=ptr[g]+lane;j<ptr[g+1];j+=blockDim.x){
  int source=col[j]%P,lag=col[j]/P+1,slot=k-lag;
  if(slot>=0)value+=val[j]*counts[(long long)slot*P+source]/(.1*sizes[source]);
 }
 __shared__ double buffer[128];buffer[lane]=value;__syncthreads();
 for(int n=64;n>0;n/=2){if(lane<n)buffer[lane]+=buffer[lane+n];__syncthreads();}
 if(lane==0)out[((long long)k*6+channel)*G+g]=buffer[0];
}''','apply',options=('--fmad=false',))
    observed=cp.empty((T,6,G));sp=cp.asarray(counts);sizes=cp.asarray(s.sizes)
    for ch,m in enumerate(matrices):
        args=(cp.asarray(m.indptr.astype('i4')),cp.asarray(m.indices.astype('i4')),cp.asarray(m.data))
        kernel((G,T),(128,),(*args,sp,sizes,observed,np.int32(s.P),np.int32(G),np.int32(ch)))
        cp.cuda.get_current_stream().synchronize()
    result=observed.get();checks=[]
    for k in [0,1,50,4999,T-1]:
        # Separate CPU construction of all delay slots, including exact zero history.
        slots=k-np.arange(1,len(s.delays)+1);history=np.zeros((len(slots),s.P))
        valid=slots>=0;history[valid]=counts[slots[valid]]/s.sizes/.1
        oracle=np.array([m@history.ravel() for m in matrices])
        error=float(np.max(abs(result[k]-oracle)));assert error<1e-8,(k,error)
        checks.append(dict(index=k,max_abs_error=error))
    write(DEST/'input_operator_check.json',dict(status='PASS',checks=checks,
        object='Originalphysical g40mean/fullvariance and repairedprivatevariance; actualnativecounts dividedbyoriginalgroupN/0.1ms. Truezero pre0history.',
        native_future_spikes_used_as_diagnostic_inputs=True,autonomous_validation=False))
    del observed,sp,sizes;gc.collect();cp.get_default_memory_pool().free_all_blocks()
    return result


@njit
def features(physical,theta):
    T,P,_=physical.shape;out=np.empty((T,P,39));h=np.zeros((P,3,4,3));taus=np.array([1.,4.,16.,64.])
    for k in range(T):
        for g in range(P):
            sc=theta[g]-11.;mu,ve,vi=physical[k,g]
            u=np.array([np.arcsinh((mu-11)/sc)/3,np.log1p(ve/sc**2)/2,np.log1p(vi/sc**2)/2])
            out[k,g,:3]=u
            for ch in range(3):
                for j in range(4):
                    b=.1/taus[j];e=np.exp(-b);h1,h2,h3=h[g,ch,j]
                    h[g,ch,j,0]=e*h1+(1-e)*u[ch]
                    h[g,ch,j,1]=e*(h2+b*h1)+(1-e*(1+b))*u[ch]
                    h[g,ch,j,2]=e*(h3+b*h2+.5*b*b*h1)+(1-e*(1+b+.5*b*b))*u[ch]
                    for q in range(3):out[k,g,3+12*ch+3*j+q]=h[g,ch,j,q]-u[ch]
    return out


@njit
def expected_flux(ell,ref):
    T,G=ell.shape;rate=np.zeros((T,G));worst=0.
    for k in range(T):
        for g in range(G):
            occupied=0.
            for lag in range(1,round(ref[g]/.1)):
                if k-lag>=0:occupied+=rate[k-lag,g]*.1
            v=ell[k,g];p=1/(1+np.exp(-v)) if v>=0 else np.exp(v)/(1+np.exp(v))
            rate[k,g]=(1-occupied)*p/.1
            worst=max(worst,occupied+.1*rate[k,g])
    return rate*1000,worst


def main(device):
    assert read(DEST/'replay_audit.json')['status']=='PASS'
    assert not (DEST/'input_response_result.json').exists(),'No silent reanalysis replacement'
    write(DEST/'analysis_jobs.json',dict(status='RUNNING',pid=os.getpid(),phase='load'))
    s=model(40);geo=np.load(DEST/'membership.npz');selected=geo['selected_groups'];G=len(selected)
    chunks=sorted((DEST/'inputs').glob('*.npz'));cs=[];ms=[];ds=[];ts=[]
    for path in chunks:
        with np.load(path) as z:
            cs.append(z['spikes']);ms.append(z['moments']);ds.append(z['external_rate_per_ms']);ts.append(z['time_ms'])
    counts=np.concatenate(cs);moments=np.concatenate(ms);drive=np.concatenate(ds);t=np.concatenate(ts)
    del cs,ms,ds,ts
    assert len(t)==30000 and np.allclose(t,np.arange(30000)*.1,atol=1e-9,rtol=0)
    names=read(DEST/'contract.json');from_names=['ampa','ampa2','gaba','gaba2','zgaba','zgaba2','mcurrent','mcurrent2','net','net2','voltage','voltage2','z','z2','ampa_gaba','ampa_zgaba','net_voltage']
    m={key:moments[:,i] for i,key in enumerate(from_names)}
    arr=arrivals(s,counts,selected,device);As=[];Bs=[]
    for tr,td in zip(s.rise,s.decay):
        Q=np.array([[-2/tr,0,0],[1/td,-1/tr-1/td,0],[0,2/td,-2/td]])
        A=expm(Q*.1);B=np.linalg.solve(Q,(A-np.eye(3))@np.array([1/tr**2,0,0]));As.append(A);Bs.append(B)
    filtered=filter_inputs(arr,drive,s.tm[selected],s.jext[selected],s.rise,s.decay,s.area,
        np.zeros((10,G)),np.array(As),np.array(Bs))
    assert np.isfinite(filtered).all() and filtered[:,4:].min()>-1e-10
    raw=np.stack([s.tm[selected]*s.area[0]**2*(arr[:,4]+s.jext[selected]**2*drive),
                  s.tm[selected]*s.area[1]**2*arr[:,5]],axis=1)
    np.savez_compressed(DEST/'projected_inputs.npz',time_ms=t,selected_groups=selected,filtered_moments=filtered,
        raw_private_variance_forcing=raw,names=['ampa_mean_cont','gaba_mean_cont','ampa_mean_native_step','gaba_mean_native_step',
        'ampa_var_full','gaba_var_full','ampa_var_private','gaba_var_private'])
    del arr
    scale=2*(s.rise+s.decay)[:,None]/s.tm[selected]
    varA=m['ampa2']-m['ampa']**2;varG=m['zgaba2']-m['zgaba']**2
    assert varA.min()>-1e-7 and varG.min()>-1e-7
    actual_marginals=np.stack([m['net'],scale[0]*np.maximum(varA,0),scale[1]*np.maximum(varG,0)],axis=2)
    measured=np.stack([m['net'],scale[0]*filtered[:,6],scale[1]*m['z']**2*filtered[:,7]],axis=2)
    projected=measured.copy();projected[:,:,0]=filtered[:,0]-m['z']*filtered[:,1]-m['mcurrent']
    nets,bases,_=load_models();outputs={};numeric=[]
    for label,physical in [('native_mean_private_variance',measured),('projected_mean_private_variance',projected),('native_mean_native_marginals',actual_marginals)]:
        write(DEST/'analysis_jobs.json',dict(status='RUNNING',pid=os.getpid(),phase=label))
        f=features(physical,s.theta[selected]);ell=np.empty((len(t),G));max_roundtrip=0.
        for pop,key in [(0,'E'),(1,'I')]:
            mask=s.geo['population'][selected]==pop;ff=f[:,mask].reshape(-1,39)
            ths=np.broadcast_to(s.theta[selected][mask],(len(t),mask.sum())).ravel();xx=physical[:,mask].reshape(-1,3)
            values=[]
            for start in range(0,len(ff),32768):
                sl=slice(start,start+32768);reconstructed=physical_from_features(ff[sl],ths[sl])
                max_roundtrip=max(max_roundtrip,float(np.max(abs(reconstructed-xx[sl]))))
                base=bases[key].evaluate(reconstructed,ths[sl])
                with torch.no_grad():values.append(nets[key].logits(torch.tensor(ff[sl]),torch.tensor(base)).numpy())
            ell[:,mask]=np.concatenate(values).reshape(len(t),mask.sum())
        rates,occupancy=expected_flux(ell,s.ref[selected]);assert occupancy<=1+1e-12 and rates.min()>=-1e-9 and np.isfinite(rates).all()
        assert max_roundtrip<1e-7
        outputs[label]=rates
        numeric.append(dict(label=label,physical_feature_roundtrip_max_error=max_roundtrip,maximum_refractory_occupancy=occupancy))
        log('EARLY NATIVE FIXED READOUT',label)
        del f,ell
    starts=np.arange(500,3000,50);groupN=s.sizes[selected]
    actual=counts[:,selected];native_bins=[];bins={k:[] for k in outputs}
    for lo in starts:
        keep=(t>=lo)&(t<lo+50);assert keep.sum()==500
        native_bins.append(actual[keep].sum(0))
        for label,rates in outputs.items():bins[label].append(rates[keep].sum(0)*.1/1000*groupN)
    native_bins=np.array(native_bins);bins={k:np.array(v) for k,v in bins.items()}
    with np.load(OUT/'physical_delay_count_rate/recorded_drive_binomial_seed1/trajectory.npz') as z:
        free=z['group_rate_hz'][:3000,selected].astype(float)
        bins['free_corrected_rate']=np.array([free[lo:lo+50].sum(0)/1000*groupN for lo in starts])
    def describe(pred,truth):
        return dict(total_count=float(pred.sum()),native_total_count=int(truth.sum()),
            count_ratio_to_native=float(pred.sum()/truth.sum()) if truth.sum() else None,
            fixed50ms_count_L2=float(np.linalg.norm(pred-truth)/max(np.linalg.norm(truth),1.)),
            mean_absolute_count_error_per50ms=float(abs(pred-truth).mean()))
    rows=[]
    for j,group in enumerate(selected):
        row=dict(names['selected_groups'][j]);row['comparisons']={label:describe(v[:,j],native_bins[:,j]) for label,v in bins.items()}
        primary=(t>=500)&(t<3000)
        row['input_mean_errors_mV_RMS']=dict(continuous_net=float(np.sqrt(np.mean((projected[primary,j,0]-m['net'][primary,j])**2))),
            native_step_net=float(np.sqrt(np.mean((filtered[primary,2,j]-m['z'][primary,j]*filtered[primary,3,j]-m['mcurrent'][primary,j]-m['net'][primary,j])**2))))
        rows.append(row)
    sets={'surround_E':np.arange(G)<9,'core_E':(np.arange(G)>=9)&(np.arange(G)<11),'I':np.arange(G)>=11}
    summary=[]
    for role,mask in sets.items():
        truth=native_bins[:,mask].sum(1)
        summary.append(dict(role=role,groups=int(mask.sum()),neurons=int(groupN[mask].sum()),
            comparisons={label:describe(v[:,mask].sum(1),truth) for label,v in bins.items()}))
    np.savez_compressed(DEST/'fixed_readout.npz',time_ms=t,selected_groups=selected,**outputs,bin_start_ms=starts,
        native_counts=native_bins,free_rate_1ms_hz=free,**{'counts_'+k:v for k,v in bins.items()})
    write(DEST/'input_response_result.json',dict(status='FIXED_RESPONSE_DIAGNOSTIC_COMPLETE',rows=rows,regional_summary=summary,
        numerical_checks=numeric,window_ms=[500,3000],statistical_unit='One native trajectory, geometric16targetgroups; bins/groups are dependent descriptive comparisons, not independentnetworkreplicates.',
        forcing='Originalnativegroupcountsandexternalinput drivefixedphysicaloperators. NativeZ/Mprescribed; readoutusesitsownexpected refractoryhistory, no nativefutureoutputfedthere.',
        initial='Native startsat originalinitialstate; all projected filters/covariances/history andownrefractoryhistory zeroat0s. Primaryafter500ms.',
        limitations='No formalpassgate orfitting. Measuredmarginalvariancecontains spatialheterogeneity andcorrelations notguaranteedrepresentedby independentGaussianinput. Responsefailurecannotalone distinguish fixedsurrogatefromGaussianinputclosure; conditionalLIFreference wouldseparate them.',
        model_promoted=False,bifurcation_type='NOT_ESTABLISHED'))
    write(DEST/'analysis_jobs.json',dict(status='COMPLETE',pid=os.getpid()))
    log('EARLY INPUT RESPONSE COMPLETE',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',type=int,default=1);main(p.parse_args().device)
