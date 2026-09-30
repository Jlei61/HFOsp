"""Matched finite-amplitude protocol for the fixed refractory-rate candidate.

Keeps each original assay's step, unmodulated burn and oscillator clock.
No fitting, model promotion or new spatial simulation is performed.
"""
from refractory_rate_response import *
from validate_refractory_rate_response import load_models
from nonlinear_rate_response import physical_from_features
from pathlib import Path
from datetime import datetime
import argparse,hashlib,os

LDIR=DEST/'matched_linear_protocol'

@njit
def driven_features(values,dt,theta,A,b,c):
    cov=np.zeros((2,3));h=np.zeros((3,4,3));out=np.empty((len(values),39))
    scale=np.array([3.,2.,2.]);taus=np.array([1.,4.,16.,64.]);sc=theta-11.
    for k in range(len(values)):
        x=values[k]
        for ch in range(2):cov[ch]=A[ch]@cov[ch]+b[ch]*x[ch+1]
        u=np.array([np.arcsinh((x[0]-11)/sc),np.log1p(c[0]*cov[0,2]/sc**2),np.log1p(c[1]*cov[1,2]/sc**2)])/scale
        for ch in range(3):
            for j in range(4):
                v=dt/taus[j];e=np.exp(-v);h1,h2,h3=h[ch,j]
                h[ch,j,0]=e*h1+(1-e)*u[ch]
                h[ch,j,1]=e*(h2+v*h1)+(1-e*(1+v))*u[ch]
                h[ch,j,2]=e*(h3+v*h2+.5*v*v*h1)+(1-e*(1+v+.5*v*v))*u[ch]
        out[k,:3]=u
        for ch in range(3):
            for j in range(4):
                for l in range(3):out[k,3+12*ch+3*j+l]=h[ch,j,l]-u[ch]
    return out

def implementation():
    rng=np.random.default_rng(920067);wave=rng.uniform(.1,1.,(3,128));wave[0]=11+20*wave[0];wave[1:]*=50
    burn=500;steps=500;dt=.1;T=100.;time=(np.arange(-burn,steps)+1)*dt
    pos=((time/T)%1)*128;lo=np.floor(pos).astype(int)%128;hi=(lo+1)%128;a=pos-np.floor(pos)
    values=((1-a)*wave[:,lo]+a*wave[:,hi]).T.copy();errors=[]
    for pop in 'EI':
        A,b,c,_,_=covariance_matrices(pop,dt)
        lhs=driven_features(values,dt,18.,A,b,c);rhs=features(wave,T,dt,burn,steps,pop,include_burn=True)
        error=float(np.max(abs(lhs-rhs)));assert error<1e-12;errors.append(error)
    return max(errors)

def register():
    LDIR.mkdir(exist_ok=True);assert not (LDIR/'contract.json').exists()
    check=implementation();locked=read(DEST/'fit/locked_weights.json')
    write(LDIR/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),status='FIXED_MODEL_PROTOCOL_DIAGNOSTIC',
        conditions=282,source_contract=str(OUT/'conditional_density_linear_response_contract.json'),weights=locked['files'],
        numerical='Each original dt(0.05or0.1ms),amplitude,burn,duration and oscillator phase preserved. Unmodulated burn begins from reset/zero history; model uses own predicted refractory history.',
        reason='Close the finite-amplitude versus infinitesimal-response ambiguity before attributing a gain failure to the rate model. Correct preliminary analytic covariance/history factors to each case own dt; preserve previous analytic file as historical.',
        driven_features_max_difference=check,training_changes=False,
        boundary='Local waveform rejection remains. These measurements cannot alone promote the spatial model or classify onset. No additional fitting or simulation extension.'))

def run():
    contract=read(LDIR/'contract.json');nets,bases,locked=load_models();assert locked['files']==contract['weights']
    cases=read(OUT/'conditional_density_linear_response_contract.json')['cases'];dest=LDIR/'cases';dest.mkdir(exist_ok=True)
    progress=dict(status='RUNNING',expected=282,completed=[],pid=os.getpid());assert not (LDIR/'jobs.json').exists();write(LDIR/'jobs.json',progress)
    for case in cases:
        q=case['workpoint'];pop=q['pop'];dt=case['dt_ms'];burn=round(case['burn_ms']/dt);steps=round(case['duration_ms']/dt);f=case['frequency_hz'];amp=case['absolute_amplitude'];ch=case['channel']
        time=(np.arange(burn+steps)+1)*dt;phase=2*np.pi*f*time/1000;osc=np.sin(phase) if f else np.ones(len(time));osc[:burn]=0
        physical=np.array([q['mu'],q['ve'],q['vi']]);A,b,c,_,_=covariance_matrices(pop,dt);rates=[]
        for sign in [1,-1]:
            values=np.tile(physical,(burn+steps,1));values[:,ch]+=sign*amp*osc;assert values[:,1:].min()>=0
            fs=driven_features(values,dt,q['theta'],A,b,c);ell=[]
            with torch.no_grad():
                for start in range(0,len(fs),4096):
                    ff=fs[start:start+4096];bl=bases[pop].evaluate(physical_from_features(ff,q['theta']),q['theta'])
                    ell.append(nets[pop].logits(torch.tensor(ff),torch.tensor(bl)).numpy())
            r,m=implicit_flux(np.concatenate(ell),dt,PARAMS['tau_ref_'+pop]);assert np.isfinite(r).all();rates.append(r[burn:])
        diff=rates[0]-rates[1]
        gain=np.mean(diff*(np.sin(phase[burn:])+1j*np.cos(phase[burn:])))/amp if f else complex(np.mean(diff)/(2*amp))
        fs=np.zeros((1,39));fs[:,:3]=normalized_input(physical[None],q['theta'])/SCALE
        bl,bg=bases[pop].evaluate(physical[None],q['theta'],True);ig=normalized_jacobian(physical[None],q['theta']);bank,cov,K=transfer_factors(pop,[f],dt=dt)
        analytic=nets[pop].linear_response(torch.tensor(fs),torch.tensor(bl),torch.tensor(bg),torch.tensor(ig),torch.tensor([ch]),torch.tensor(bank),torch.tensor(cov),torch.tensor(K)).detach().numpy()[0]
        path=dest/f'case{case["id"]:03d}.npz';assert not path.exists()
        np.savez_compressed(path,rate_hz=np.array(rates),gain=np.array([gain.real,gain.imag]),analytic_gain=np.array([analytic.real,analytic.imag]),
            dt_ms=dt,burn_ms=case['burn_ms'],duration_ms=case['duration_ms'],frequency_hz=f,amplitude=amp,case_id=case['id'])
        progress['completed'].append(case['id']);write(LDIR/'jobs.json',progress)
        if len(progress['completed'])%12==0:log('REFRACTORY MATCHED LINEAR',len(progress['completed']),282)
    progress['status']='COMPLETE';write(LDIR/'jobs.json',progress)
    write(LDIR/'predictions_locked.json',dict(status='LOCKED_BEFORE_MATCHED_SCORE',weights=locked['files'],case_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.glob('*.npz')},cases=282))

def score():
    locked=read(LDIR/'predictions_locked.json');rows=[];maximum=0.
    for case in read(OUT/'conditional_density_linear_response_contract.json')['cases']:
        path=LDIR/f'cases/case{case["id"]:03d}.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==locked['case_hashes'][path.name]
        z=np.load(path);r=z['rate_hz'];diff=r[0]-r[1];dt=float(z['dt_ms']);f=float(z['frequency_hz']);amp=float(z['amplitude']);T=float(z['duration_ms'])
        phase=2*np.pi*f/1000*((np.arange(len(diff))+1)*dt+float(z['burn_ms']))
        gain=complex(np.dot(diff,np.sin(phase)),np.dot(diff,np.cos(phase)))*dt/T/amp if f else complex(diff.sum()*dt/T/(2*amp))
        maximum=max(maximum,abs(gain-complex(*z['gain'])));ref=case['reference'];den=abs(ref['dc_measured']);measured=complex(*ref['measured']);analytic=complex(*z['analytic_gain'])
        error=abs(gain-measured)/den if den else None;ae=abs(analytic-measured)/den if den else None
        signok=np.sign(gain.real)==np.sign(measured.real) if f<=25 and ref['counted'] else None
        rows.append(dict(id=case['id'],pop=case['workpoint']['pop'],channel=case['channel'],frequency_hz=f,dt_ms=dt,counted=ref['counted'],
            gain=[gain.real,gain.imag],analytic_gain=[analytic.real,analytic.imag],error=error,analytic_error=ae,
            passed=bool(error<=ref['tol']) if ref['counted'] else None,low_frequency_sign_ok=bool(signok) if signok is not None else None,
            finite_vs_infinitesimal_difference=abs(gain-analytic)/den if den else None))
    assert maximum<1e-9,maximum
    ac=[r for r in rows if r['counted'] and r['frequency_hz']];dc=[r for r in rows if r['counted'] and not r['frequency_hz']]
    result=dict(status='MATCHED_LOCAL_RESPONSE_DIAGNOSTIC_COMPLETE',conditions=len(rows),AC_counted=len(ac),AC_failed=sum(not r['passed'] for r in ac),
        DC_counted=len(dc),DC_failed=sum(not r['passed'] for r in dc),low_frequency_sign_failed=sum(r['low_frequency_sign_ok'] is False for r in rows),
        analytic_correct_dt_AC_failed=sum(r['analytic_error']>next(c['reference']['tol'] for c in read(OUT/'conditional_density_linear_response_contract.json')['cases'] if c['id']==r['id']) for r in ac),
        independent_demodulation_max_difference=maximum,rows=rows,model_promoted=False,training_modified=False)
    write(LDIR/'result.json',result);log('MATCHED LINEAR RESULT',{k:v for k,v in result.items() if k!='rows'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run','score']);a=p.parse_args();{'register':register,'run':run,'score':score}[a.command]()
