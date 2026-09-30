"""Apply locked rate readout to supplied native input moments, without fitting.

Actual marginal current spread includes heterogeneity/covariance absent from
the readout assumptions. Neither this nor the projected-input arm is an
autonomous simulation or a Gaussian-LIF reference assay.
"""
from common import OUT,np,read,write
from conditioned_refractory_rate import load_models
from nonlinear_rate_response import physical_from_features
from numba import njit
import torch

DEST=OUT/'native_input_bridge'


@njit
def direct_features(physical,theta):
    T,P,_=physical.shape;out=np.empty((T,P,39));h=np.zeros((P,3,4,3));taus=np.array([1.,4.,16.,64.])
    for k in range(T):
        for g in range(P):
            sc=theta[g]-11.;mu,ve,vi=physical[k,g]
            u=np.array([np.arcsinh((mu-11)/sc)/3,np.log1p(ve/sc**2)/2,np.log1p(vi/sc**2)/2])
            if k==0:
                for ch in range(3):h[g,ch,:,:]=u[ch]
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
def flux(ell,ref):
    T,P=ell.shape;rate=np.zeros((T,P))
    for k in range(T):
        for g in range(P):
            occupied=0.
            for lag in range(1,round(ref[g]/.1)):
                if k-lag>=0:occupied+=rate[k-lag,g]*.1
            p=1/(1+np.exp(-ell[k,g]));rate[k,g]=(1-occupied)*p/.1
    return rate*1000


def main():
    assert read(DEST/'input_comparison.json')['status']=='INPUT_BRIDGE_DIAGNOSTIC_COMPLETE'
    z=np.load(DEST/'selected_input_history.npz');t=z['time_ms'];a=z['moments'];r=z['reconstructed'];theta=z['theta'];pop=z['population']
    m={key:a[:,j] for j,key in enumerate(z['moment_names'])};P=len(theta)
    tm=np.where(pop==0,20.,10.);ref=np.where(pop==0,2.,1.)
    from lif_mc import PARAMS
    times=np.array([PARAMS['tau_r_AMPA']+PARAMS['tau_d_AMPA'],PARAMS['tau_r_GABA']+PARAMS['tau_d_GABA']])
    measured=np.stack([m['net'],2*times[0]/tm*np.maximum(m['ampa2']-m['ampa']**2,0),
        2*times[1]/tm*np.maximum(m['zgaba2']-m['zgaba']**2,0)],axis=2)
    projected=np.stack([r[:,0]-m['z']*r[:,1]-m['mcurrent'],2*times[0]/tm*r[:,6],
        2*times[1]/tm*m['z']**2*r[:,7]],axis=2)
    net,bases,_=load_models();predictions={}
    for label,physical in [('measured_marginals',measured),('projected_private',projected)]:
        f=direct_features(physical,theta);ell=np.empty((len(t),P))
        for p,key in [(0,'E'),(1,'I')]:
            mask=pop==p;features=f[:,mask].reshape(-1,39);ths=np.broadcast_to(theta[mask],(len(t),mask.sum())).ravel()
            reconstructed=physical_from_features(features,ths)
            assert np.max(abs(reconstructed-physical[:,mask].reshape(-1,3)))<1e-7
            base=bases[key].evaluate(reconstructed,ths)
            with torch.no_grad():values=net[key].logits(torch.tensor(features),torch.tensor(base)).numpy()
            ell[:,mask]=values.reshape(len(t),mask.sum())
        out=flux(ell,ref);assert np.isfinite(out).all() and out.min()>-1e-8
        predictions[label]=out
    # Fixed nonoverlapping50ms count windows, last incomplete window excluded.
    starts=np.arange(9000,10370-50+1e-6,50);native=[];pred={k:[] for k in predictions}
    for lo in starts:
        keep=(t>=lo)&(t<lo+50);assert keep.sum()==500
        native.append(z['spikes'][keep].sum(0))
        for key,v in predictions.items():pred[key].append(v[keep].sum(0)*.1/1000*z['group_size'])
    native=np.array(native);pred={k:np.array(v) for k,v in pred.items()};rows=[]
    for j,g in enumerate(z['groups']):
        row=dict(group=int(g),N=int(z['group_size'][j]),theta_mv=float(theta[j]),native_count=int(native[:,j].sum()))
        for key,v in pred.items():
            row[key]=dict(expected_count=float(v[:,j].sum()),absolute_window_count_error=float(abs(v[:,j]-native[:,j]).mean()),
                count_waveform_L2_descriptive=float(np.linalg.norm(v[:,j]-native[:,j])/max(np.linalg.norm(native[:,j]),1)))
        rows.append(row)
    np.savez_compressed(DEST/'fixed_readout.npz',time_ms=t,**predictions,bin_start_ms=starts,native_counts=native,
        **{'counts_'+k:v for k,v in pred.items()})
    write(DEST/'fixed_readout.json',dict(status='LOCKED_READOUT_DIAGNOSTIC_COMPLETE',rows=rows,
        preprocessing='Already-filtered current variances converted to effective static variance units; no second synaptic filtering. Input history initialized at8s instantaneous values;1s preparation excluded.',
        data_used='Only supplied input means,variances,Z/M; native future firing is not fed to refractory readout history. Projected inputs upstream were calculated from native group spikes.',
        boundaries='Counts are descriptive one-history comparison. No finiteNnoise acceptance, no localGaussianLIF reference, no autonomous model promotion or parameter fitting.',
        weights=str(OUT/'conditioned_refractory_rate/fit/locked_weights.json')))
    print(rows,flush=True)


if __name__=='__main__':main()
