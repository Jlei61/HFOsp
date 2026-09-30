"""One physically bounded mean-memory diagnostic, retaining DC/linear response.

The stored coordinate saturates only above threshold and stays asymptotically
linear under inhibition. It is a response coordinate, not the mean voltage.
No coefficient is fitted to waveform or native network targets.
"""
from common import OUT, ROOT, np, read, write, log, ResponseParams
from transfer_spline import TransferSpline
from response_voltage_units import VoltageScaledResponseTable
from datetime import datetime
import argparse

DEST = OUT / 'threshold_bounded_mean_memory'


def encode(mu, theta):
    gap=theta-11.;x=(mu-theta)/gap;root=np.hypot(x,2.)
    y=np.where(x>=0,2./(root+x),(root-x)/2.)
    return theta-gap*y


def decode(value, theta):
    gap=theta-11.;y=(theta-value)/gap
    assert np.min(y)>0, 'Bounded coordinate hit threshold; no clipping permitted'
    return theta+gap*(1./y-y)


def register():
    p=OUT/'threshold_bounded_mean_memory_contract.json';assert not p.exists()
    write(p,dict(created_local=datetime.now().astimezone().isoformat(),
        question='Can preventing accumulation of arbitrarily high supra-threshold mean-input memory improve strong-input recovery, while retaining negative-current history and the original DC/local response?',
        rationale='Exact membrane-balance replay shows large accumulated refractory-clamp and reset corrections. Earlier rate-coordinate memory also compressed the negative-input tail; this coordinate remains asymptotically linear there.',
        candidate_count=1,datasets=['in_domain_waveform','factorial_waveform'],
        equations='gap=theta-reset; x=(mu-theta)/gap; y=(sqrt(x*x+4)-x)/2; psi=theta-gap*y. tau_s h_dot=psi(mu)-h; decoded_mu_s=theta+gap*(1/yh-yh), yh=(theta-h)/gap. Substitute decoded_mu_s for old mu_s in the unchanged history-conditioned response.',
        parameters='Native theta and reset define upper asymptote and scale; no new fitted constants. Existing poles, variance filters, voltage-unit correction and static LIF transfer unchanged.',
        identity='Exact DC and first-order response unchanged by inverse-coordinate cancellation; therefore existing linear-response failures are NOT repaired by this diagnostic.',
        checks=['inverse/DC', 'small-perturbation all-channel identity at one E and I point', 'reproduce prior history-prediction arrays', 'same15percent waveform/10percent mean gates on24already-observed conditions'],
        acceptance=dict(normalized_waveform_RMSE_max=.15,relative_cycle_mean_error_max=.1),
        stop='One fixed coordinate only. Do not tune cap/width to diagnostics or launch a network if local gates fail. Even waveform pass requires an independently accepted linear response before model promotion.',
        scope='One nonlinear response-coordinate diagnostic, not a voltage-conservation closure, new blind test, accepted rate field or onset result. No particle network.'))


def main():
    c=read(OUT/'threshold_bounded_mean_memory_contract.json')
    response=ResponseParams(OUT/'frozen_data/response_closure/closure.json')
    transfer={p:TransferSpline(OUT/f'frozen_data/transfer_table/table_{p}.npz') for p in 'EI'}
    params=read(ROOT/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g20/prepared.json')['params']
    syn=[(params['tau_r_'+k]+params['tau_d_'+k])/2 for k in ['AMPA','GABA']]
    DEST.mkdir(exist_ok=True);rows=[];identities=[];reproductions=[]
    for dataset in c['datasets']:
        source=OUT/dataset;assert read(source/'independent_audit.json')['status']=='COUNT_LEVEL_AUDIT_PASS'
        info=read(source/'preparation.json');result=read(source/'result.json')
        data=np.load(source/'prepared.npz');obs=np.load(source/'response.npz')
        w=data['wave'].shape[-1];period=float(data['T_ms']);lam=2j*np.pi*np.arange(w//2+1)/period
        def filt(x,tau):return np.fft.irfft(np.fft.rfft(x)/(1+lam*tau),n=w)
        phase=((np.arange(result['steps'])+1)*result['dt_ms']/period)%1
        b=obs['measured_hz'].shape[1];bins=np.minimum((phase*b).astype(int),b-1);exposure=np.bincount(bins,minlength=b)
        pos=phase*w;lo=np.floor(pos).astype(int)%w;hi=(lo+1)%w;frac=pos-np.floor(pos)
        def aggregate(x):return np.bincount(bins,weights=(1-frac)*x[lo]+frac*x[hi],minlength=b)/exposure
        curves=[]
        for j,g in enumerate(info['rows']):
            pop,theta=g['pop'],g['theta_mv'];th=np.full(w,theta)
            pole=response.poles[pop];tab=VoltageScaledResponseTable(response.tables[pop])
            def predict(wave,transformed):
                mu,raw_e,raw_i=wave;ve,vi=filt(raw_e,syn[0]),filt(raw_i,syn[1])
                mus=decode(filt(encode(mu,theta),pole['tau_s']),theta) if transformed else filt(mu,pole['tau_s'])
                vef,vif=filt(ve,pole['tau_cE']),filt(vi,pole['tau_cI'])
                ves,vis=filt(ve,pole['tau_vE']),filt(vi,pole['tau_vI'])
                (alpha,ae,ai,ee,ei),_=tab.evaluate(mus,ves,vis,th)
                effective=alpha*mu+(1-alpha)*mus+ee*(ve-vef)+ei*(vi-vif)
                e=np.maximum(ae*ve+(1-ae)*ves,0);i=np.maximum(ai*vi+(1-ai)*vis,0)
                return transfer[pop].evaluate(effective,e,i,th)['rate']*1000
            baseline=aggregate(predict(data['wave'][j],False))
            error=float(np.max(abs(baseline-obs['predicted_hz'][j,1])))
            assert error<1e-9;reproductions.append(error)
            raw=predict(data['wave'][j],True);pred=aggregate(raw);curves.append(pred)
            target=obs['measured_hz'][j]
            waveform=float(np.linalg.norm(pred-target)/max(np.linalg.norm(target),np.sqrt(b)))
            mean=float(np.average(pred,weights=obs['occupancy_ms']))
            bias=abs(mean-result['rows'][j]['MC_mean_hz'])/max(result['rows'][j]['MC_mean_hz'],1)
            physical=bool(np.min(raw)>=-1e-10 and np.max(raw)<=1000/params['tau_ref_'+pop]+1e-10)
            rows.append(dict(dataset=dataset,**g,waveform_L2=waveform,relative_mean_error=bias,
                predicted_mean_hz=mean,physical_bounds_pass=physical,
                passed=bool(physical and waveform<=.15 and bias<=.1)))
            if dataset==c['datasets'][0] and j in [0,9]:
                x=np.array([-500.,-100.,11.,14.,18.,60.,300.,1000.])
                inverse_error=float(np.max(abs(decode(encode(x,theta),theta)-x)));assert inverse_error<1e-8
                t=np.arange(w)*period/w;gap=theta-11
                base=np.array([np.full(w,11+.6*gap),np.full(w,(gap*.7)**2),np.full(w,(gap*1.4)**2)])
                direction=np.array([gap*np.sin(2*np.pi*t/period),gap**2*.1*np.cos(2*np.pi*t/period),gap**2*.2*np.sin(2*np.pi*t/period+.7)])
                dc=float(np.max(abs(predict(base,True)-predict(base,False))));assert dc<1e-8
                errs=[]
                for eps in [1e-4,5e-5]:
                    a=(predict(base+eps*direction,True)-predict(base-eps*direction,True))/(2*eps)
                    b0=(predict(base+eps*direction,False)-predict(base-eps*direction,False))/(2*eps)
                    errs.append(float(np.linalg.norm(a-b0)/max(np.linalg.norm(b0),1.)))
                assert errs[-1]<1e-6,errs
                identities.append(dict(pop=pop,inverse_error=inverse_error,DC_difference_hz=dc,first_order_relative_errors=errs))
        np.savez_compressed(DEST/f'{dataset}.npz',predicted_hz=curves,measured_hz=obs['measured_hz'],
                            phase_centres=obs['phase_centres'],T_ms=period)
    q=dict(status='BOUNDED_MEAN_MEMORY_DIAGNOSTIC_COMPLETE',rows=rows,
        identities=identities,baseline_prediction_max_difference=max(reproductions),
        all_waveform_cases_pass=all(r['passed'] for r in rows),
        linear_response_status='UNCHANGED_PREVIOUS_FAILURES',model_promoted=False,scope=c['scope'])
    write(DEST/'result.json',q)
    log('BOUNDED MEAN MEMORY',q['all_waveform_cases_pass'])
    for row in rows:log(row['dataset'],row['source_label'],row.get('condition',row.get('amplitude')),row['waveform_L2'],row['relative_mean_error'],row['passed'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--register',action='store_true');a=p.parse_args()
    register() if a.register else main()
