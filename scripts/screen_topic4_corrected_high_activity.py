#!/usr/bin/env python3
"""Bounded autonomous high-state assay of the corrected, delayed spatial rate model."""
from validate_topic4_fixed_rate_base import ROOT, OUT as BASE, read, write
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import numpy as np
from scipy import sparse
from scipy.signal import find_peaks, periodogram
from src.topic4_patient_zm_meanfield import load_patient_coarse_model, transfer_rates

OUT=ROOT/'results/topic4_sef_hfo/corrected_rate_high_activity_screen_v1'


def diagnostics(fields, model):
    # All cells are retained; classification cannot rely on the sheet average alone.
    e=fields[:,0];global_e=np.average(e,axis=1,weights=model.count_e)
    local_amplitude=np.ptp(e[len(e)//2:],axis=0)
    cell=int(np.argmax(local_amplitude))
    signals={'global_E':global_e,'largest_local_E':e[:,cell]}
    result={'largest_local_cell':cell,'local_amplitude_p50_p90_max_hz':np.quantile(local_amplitude,[.5,.9,1]).tolist(),'signals':{}}
    for name,signal in signals.items():
        pieces=np.array_split(signal[len(signal)//2:],3)
        amplitudes=[float(np.ptp(x)) for x in pieces]
        tail=signal[len(signal)//2:];amp=float(np.ptp(tail));mean=float(tail.mean())
        peaks,_=find_peaks(tail,prominence=max(.05,.15*amp),distance=2)
        intervals=np.diff(peaks);cv=float(intervals.std()/intervals.mean()) if len(intervals)>1 else None
        f,p=periodogram(tail,fs=1000,detrend='linear');idx=int(np.argmax(p[1:])+1)
        ratio=amplitudes[-1]/max(amplitudes[0],1e-12)
        periodic=(len(peaks)>=8 and cv is not None and cv<.25 and .7<ratio<1.3)
        label='PERSISTENT_OSCILLATION_CANDIDATE' if periodic and amp>=1 else ('SMALL_PERIODIC_RIPPLE' if periodic else 'DECAYING_RINGING' if ratio<.5 and amplitudes[0]>=1 else 'TONIC_OR_LOW_STATIONARY' if amp<.25 else 'UNRESOLVED_SHORT_OR_NONPERIODIC')
        result['signals'][name]={'mean_hz':mean,'min_hz':float(tail.min()),'max_hz':float(tail.max()),
            'peak_to_peak_hz':amp,'relative_peak_to_peak':amp/max(mean,1e-12),'last_three_amplitudes_hz':amplitudes,
            'last_first_amplitude_ratio':ratio,'peaks':len(peaks),'period_cv':cv,
            'peak_interval_frequency_hz':float(1000/intervals.mean()) if len(intervals) else None,
            'dominant_frequency_hz':float(f[idx]) if amp>.01 else None,
            'fraction_below_1hz':float(np.mean(tail<1.)),
            'fraction_below_5hz':float(np.mean(tail<5.)),
            'label':label}
    result['candidate']=any(v['label']=='PERSISTENT_OSCILLATION_CANDIDATE' for v in result['signals'].values())
    return result


def run(q,tau,grid=10,duration=3000.,dt=.1,initial='high',preserve_native_area=False):
    started=time.time();folder=BASE/f'coarse_{grid}';model=load_patient_coarse_model(folder/'model.npz');cfg=read(folder/'prepared.json')
    factor=int(round(cfg['dt_ms']/dt));assert factor>=1 and abs(factor*dt-cfg['dt_ms'])<1e-10
    n=model.n_cells;delay=cfg['max_delay_steps']*factor
    ops={key:sparse.load_npz(folder/f'delay_{key}.npz') for key in ('ee','ei','ie','ii')}
    if factor!=1:
        for key,op in ops.items():
            coo=op.tocoo();block=coo.col//n;col=((block+1)*factor-1)*n+coo.col%n
            ops[key]=sparse.csr_matrix((coo.data,(coo.row,col)),shape=(n,n*delay))
    te,ti=model.tau_mem_e_ms,model.tau_mem_i_ms;ra,rg=cfg['tau_r_ampa_ms'],cfg['tau_r_gaba_ms'];ta=model.tau_ampa_ms
    nu=model.nu_ext_per_ms
    re=np.full(n,.1 if initial=='high' else 0.);ri=re.copy()
    he=np.tile(re,(delay,1));hi=np.tile(ri,(delay,1))
    rise=np.array([ra,rg,ra,rg,ra,ra])[:,None];decay=np.array([ta,tau,ta,tau,ta,ta])[:,None]
    membrane=np.array([te,te,ti,ti,te,ti])[:,None];ar=np.exp(-dt/rise);ad=np.exp(-dt/decay)
    jump_dt=dt*np.ones_like(rise)
    if preserve_native_area:
        # Refine integration at the native calibrated impulse area, not a new DC operating point.
        native_dt=cfg['dt_ms'];jump_dt=native_dt*(1-ar)/(1-np.exp(-native_dt/rise))
    ext_e=np.full(n,model.j_ext_e_mv*nu);ext_i=np.full(n,model.j_ext_i_mv*nu)
    drive=np.stack([ops['ee']@he.ravel(),q*(ops['ei']@hi.ravel()),ops['ie']@he.ravel(),q*(ops['ii']@hi.ravel()),ext_e,ext_i])
    gating=jump_dt*membrane/rise*drive/(1-ar) if initial=='high' else np.zeros((6,n));current=gating.copy()
    nsteps=int(round(duration/dt));stride=int(round(1/dt));fields=np.empty((nsteps//stride,2,n),np.float32)
    tau_re=read(BASE/'colored_response_diagnostic.json')['best_tau_rate_ms'];tau_ri=read(BASE/'colored_response_diagnostic_I.json')['best_tau_rate_ms']
    for step in range(nsteps):
        drive=np.stack([ops['ee']@he.ravel(),q*(ops['ei']@hi.ravel()),ops['ie']@he.ravel(),q*(ops['ii']@hi.ravel()),ext_e,ext_i])
        gating=gating*ar+jump_dt*membrane/rise*drive;current=gating+(current-gating)*ad
        exc_e=te*(model.v_ee@re+model.j_ext_e_mv**2*nu);inh_e=te*q*q*(model.v_ei@ri)
        exc_i=ti*(model.v_ie@re+model.j_ext_i_mv**2*nu);inh_i=ti*q*q*(model.v_ii@ri)
        ve=exc_e+inh_e;vi=exc_i+inh_i
        shift_e=2.065/2*np.sqrt(np.maximum((exc_e*(ra+ta)+inh_e*(rg+tau))/te,0))
        shift_i=2.065/2*np.sqrt(np.maximum((exc_i*(ra+ta)+inh_i*(rg+tau))/ti,0))
        pe,pi=transfer_rates(model,current[0]-current[1]+current[4]-shift_e,np.sqrt(np.maximum(ve,1e-12)),
            current[2]-current[3]+current[5]-shift_i,np.sqrt(np.maximum(vi,1e-12)))
        ne=re+dt/tau_re*(pe-re);ni=ri+dt/tau_ri*(pi-ri)
        he[1:]=he[:-1].copy();hi[1:]=hi[:-1].copy();he[0]=re;hi[0]=ri;re,ri=ne,ni
        if (step+1)%stride==0:fields[step//stride]=np.array([re,ri])*1000
    assert np.isfinite(fields).all()
    name=f'q{q:g}_gaba{tau:g}_grid{grid}_dt{dt:g}_{initial}_{duration:g}ms'
    if preserve_native_area:name+='_native_area'
    dest=OUT/'runs';dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/f'{name}.npz',fields_hz=fields,frame_ms=1.,rate_e=re,rate_i=ri,
        gating=gating,current=current,history_e=he,history_i=hi)
    result={'name':name,'q':q,'tau_gaba_ms':tau,'grid':grid,'dt_ms':dt,'duration_ms':duration,'initial':initial,
        'input':'constant nominal external Poisson rate in diffusion closure; both OU modulations off','nu_ext_per_ms':nu,
        'q_definition':'global multiplier of I-to-E and I-to-I synaptic jumps; mean scales q, variance q^2; Z/M dynamics off',
        'preserve_native_impulse_area':preserve_native_area,
        'seconds':time.time()-started,'diagnostics':diagnostics(fields,model)}
    write(dest/f'{name}.json',result);return result


def batch():
    tau0=read(BASE/'reconstruction.json')['params']['tau_d_GABA']
    jobs=[(q,tau) for tau in (9.,tau0,42.) for q in (.25,.5,.75,1.,1.25)]
    protocol={'status':'FROZEN_BEFORE_RESULTS','substrate':str(ROOT/'config/topic4_rate_model_dynamics_validation_v1.json'),
        'authorization':'User explicitly released high-activity common-mechanism test without waiting for patient interictal operating-point selection.',
        'scope':'Capability of corrected closure on one reference substrate; not evidence of geometry-independent universality or patient seizure recovery.',
        'jobs':[{'q':q,'tau_gaba_ms':t} for q,t in jobs],'grid':10,'duration_ms':3000,'dt_ms':.1,'initial':'high rates/history plus stationary synaptic expectation at those rates',
        'max_workers':4,'noise':'OU off and constant nominal input so external fluctuations cannot masquerade as an autonomous oscillation',
        'screen_labels':'Exploratory diagnostic only: >=8 peaks, period CV<0.25, last/first amplitude ratio 0.7..1.3 in late half; >=1 Hz absolute amplitude for candidate; all amplitudes/frequencies reported without fixed frequency band.',
        'confirmation':'Extend candidates; compare half step preserving physical delay, low initial condition, finer grid, and matching SNN. No Hopf label from spectral peak alone.'}
    write(OUT/'protocol.json',protocol);completed=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,q,t) for q,t in jobs]
        for f in as_completed(futures):
            row=f.result();completed.append(row);write(OUT/'status.json',{'status':'RUNNING','completed':len(completed),'total':len(jobs),'results':completed});print(row['name'],row['diagnostics']['candidate'],flush=True)
    write(OUT/'status.json',{'status':'SCREEN_COMPLETE','completed':len(completed),'total':len(jobs),'results':completed})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch',action='store_true');p.add_argument('--q',type=float,default=1);p.add_argument('--tau',type=float,default=20.611550480127335)
    p.add_argument('--grid',type=int,default=10);p.add_argument('--duration',type=float,default=3000);p.add_argument('--dt',type=float,default=.1);p.add_argument('--initial',choices=['high','low'],default='high');p.add_argument('--preserve-native-area',action='store_true');a=p.parse_args()
    if a.batch:batch()
    else:print(run(a.q,a.tau,a.grid,a.duration,a.dt,a.initial,a.preserve_native_area),flush=True)
