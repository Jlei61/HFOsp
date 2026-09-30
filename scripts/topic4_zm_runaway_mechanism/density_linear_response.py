"""Same-protocol DC and frequency-response tests of the density candidate.

No target values enter the predictor. Protocol amplitudes, clocks and time
steps are copied from the original assays, including their unmodulated burn.
"""
from common import OUT,BASE,ROOT,np,read,write,log
from conditional_current_density import simulate,voltage_grid
from lif_mc import condition
from datetime import datetime
from pathlib import Path
import argparse,hashlib,subprocess,sys,time,os
from concurrent.futures import ThreadPoolExecutor,as_completed

DEST=OUT/'conditional_density_linear_response'
CHANNELS=['mean','variance_E','variance_I']


def register():
    path=OUT/'conditional_density_linear_response_contract.json';assert not path.exists()
    audit=read(OUT/'conditional_current_density/independent_audit.json')
    full=[r for r in audit['rows'] if r['grid']==256]
    assert len(full)==24 and all(r['waveform_pass'] and r['numerical_pass'] for r in full)
    ref=read(BASE/'dynamic_assay/validation_result.json')['rows'];cases=[];parity=[]
    parent=ROOT/'results/topic4_sef_hfo/fig5_zm_rate_synchronized_20260917'
    for lab in ['local_response','local_response_additional_mode_groups']:
        source=read(parent/lab/'result.json')['rows'];protocol=read(parent/lab/'contract.json')
        dt=protocol.get('dt_ms',.1)
        for kind in ['mean_raw','variance_raw']:
            raw=np.load(parent/lab/(kind+'.npz'))
            for position,index in enumerate(raw['indices']):
                row=source[int(index)];p=condition(row['mu_mv'],row['threshold_mv'],row['variance_E'],row['variance_I'],row['population'],dt=dt)
                old=raw['parameters'][position]
                delta=float(np.max(abs(p[6:20]-old[6:20])));parity.append(delta)
                assert np.allclose(p[6:20],old[6:20],rtol=1e-12,atol=1e-12), (lab,index,delta)
        for row in source:
            f=row['frequency_hz'];ch=CHANNELS.index(row['channel'])
            q=dict(pop=row['population'],mu=row['mu_mv'],theta=row['threshold_mv'],ve=row['variance_E'],vi=row['variance_I'])
            dc=next(r for r in source if r['state']==row['state'] and r['group']==row['group'] and r['channel']==row['channel'] and r['frequency_hz']==0.)
            if f:
                candidates=[(i,r) for i,r in enumerate(ref) if r['kind']=='v2_assayed' and r['state']==row['state'] and r['group']==row['group'] and r['channel']==row['channel'] and r['frequency_hz']==f]
                assert len(candidates)==1;reference_index,reference=candidates[0]
            else:
                reference_index=None
                value=complex(*row['measured']);snr=abs(value)/max(row['complex_sem'],1e-12)
                reference=dict(measured=[value.real,value.imag],sem=row['complex_sem'],dc_measured=value.real,counted=snr>=10,tol=.15,dc_snr=snr)
            cases.append(dict(kind='v2_assayed',workpoint=q,channel=ch,frequency_hz=f,
                absolute_amplitude=row['amplitude'],dt_ms=dt,burn_ms=1000.,duration_ms=protocol['duration_ms'],
                reference_index=reference_index,reference=reference,source=f'{lab}/result.json',state=row['state'],group=row['group']))
    rng=np.random.default_rng(11)
    for pop in 'EI':
        for _ in range(12):
            theta=18. if rng.uniform()<.5 or pop=='I' else rng.uniform(14.2,17.5)
            scale=theta-11;x=rng.uniform(-1,4);se=np.exp(rng.uniform(np.log(.3),np.log(3)));si=np.exp(rng.uniform(np.log(.1),np.log(4)))
            q=dict(pop=pop,theta=theta,mu=11+scale*x,ve=(scale*se)**2,vi=(scale*si)**2)
            for ch in range(3):
                for f in [0.,15.,60.]:
                    rows=[(i,r) for i,r in enumerate(ref) if r['kind']=='random' and r['pop']==pop and r['x']==x and r['channel']==CHANNELS[ch] and (f==0 or r['frequency_hz']==f)]
                    assert len(rows)==(2 if f==0 else 1)
                    reference_index,reference=rows[0]
                    if not f:
                        reference_index=None;reference=dict(measured=[reference['dc_measured'],0.],dc_measured=reference['dc_measured'],dc_snr=reference['dc_snr'],counted=reference['counted'],tol=.15,sem=None)
                    cases.append(dict(kind='random',workpoint=q,channel=ch,frequency_hz=f,
                        absolute_amplitude=.15 if ch==0 else .05*q['ve' if ch==1 else 'vi'],
                        dt_ms=.1,burn_ms=300.,duration_ms=4000.,reference_index=reference_index,reference=reference,x=x))
    for i,case in enumerate(cases):case['id']=i
    assert len(cases)==282 and sum(c['frequency_hz']>0 and c['reference']['counted'] for c in cases)==146
    source=Path(__file__).with_name('conditional_current_density.py')
    c=dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the new deterministic response reproduce the original eligible DC and complex gains, including gain sign, before it can supply a spatial stability analysis?',
        response_source=str(source),response_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        grid_nodes=256,wave_samples=4096,workers=12,cases=cases,
        source_synaptic_membrane_parameter_parity_max=max(parity),
        acceptance=dict(AC_counted=146,AC_failed_max=14,AC_original_per_row_tolerance=True,
            DC_relative_tolerance=.15,low_frequency_real_sign_preserved=True,
            mass_error_max=1e-8,current_moment_error_max=1e-7),
        budget='282pairedconditions=96DC+186AC at32originalworkpoints; no MonteCarlo retraining or fitted parameters. One256node pass. A numerical refinement must be separately specified before use; no network or bifurcation launch in this batch.',
        scope='Already observed validation targets, independent of this candidate calculation but not newly blind. Physical constants fixed. Population-density benchmark is not yet a compact accepted spatial rate field.',
        burn_protocol='Unmodulated original burn; oscillator clock starts at beginning of burn and continues during recording, exactly matching reference. Native0.05ms additional modes kept distinct from0.1ms assays.')
    write(path,c);DEST.mkdir(exist_ok=True);log('DENSITY LINEAR REGISTERED',len(cases),max(parity))


def one(case_id):
    c=read(OUT/'conditional_density_linear_response_contract.json')
    assert hashlib.sha256(Path(c['response_source']).read_bytes()).hexdigest()==c['response_sha256']
    case=c['cases'][case_id];q=case['workpoint'];ch=case['channel'];f=case['frequency_hz']
    dt=case['dt_ms'];burn=round(case['burn_ms']/dt);steps=round(case['duration_ms']/dt)
    period=1000/f if f else case['duration_ms'];W=c['wave_samples']
    waveform=np.repeat(np.array([q['mu'],q['ve'],q['vi']])[:,None],W,axis=1)
    oscillation=np.sin(2*np.pi*np.arange(W)/W) if f else np.ones(W)
    p=condition(0,q['theta'],1.,1.,q['pop'],dt=dt)
    grid=voltage_grid(q['theta'],11.,c['grid_nodes'])
    rates=[];evidence=[];means=[];started=time.monotonic()
    for sign in [1.,-1.]:
        drive=waveform.copy();drive[ch]+=sign*case['absolute_amplitude']*oscillation
        assert np.min(drive[1:])>0
        answer=simulate(p,drive,grid,dt,period,burn,steps,128,q['mu'],q['ve'],q['vi'],case['burn_ms'])
        rates.append(answer[3]);evidence.append(answer[6]);means.append(float(answer[3].mean()))
    rates=np.array(rates);evidence=np.array(evidence)
    phase=2*np.pi*f/1000*((np.arange(steps)+1)*dt+case['burn_ms'])
    diff=rates[0]-rates[1]
    if f:
        gain=np.mean(diff*(np.sin(phase)+1j*np.cos(phase)))/case['absolute_amplitude']
    else:
        gain=complex(np.mean(diff)/(2*case['absolute_amplitude']))
    ref=case['reference'];value=complex(*ref['measured'])
    error=float(abs(gain-value)/max(abs(ref['dc_measured']),1e-12))
    num_ok=bool(evidence[:,0].max()<1e-8 and evidence[:,1].max()<1e-7 and evidence[:,2].max()<1e-10 and evidence[:,3].max()<1e-8)
    result=dict(status='DENSITY_LINEAR_CASE_COMPLETE',case_id=case_id,kind=case['kind'],workpoint=q,channel=CHANNELS[ch],
        frequency_hz=f,dt_ms=dt,absolute_amplitude=case['absolute_amplitude'],
        predicted=[gain.real,gain.imag],measured=ref['measured'],dc_measured=ref['dc_measured'],
        normalized_error=error,counted=ref['counted'],tolerance=ref['tol'],passed=bool(error<=ref['tol']) if ref['counted'] else None,
        sign_ok=bool(np.sign(gain.real)==np.sign(value.real)) if f and f<=10 and ref['counted'] else None,
        numerical_pass=num_ok,numerical=evidence.tolist(),mean_rates_hz=means,elapsed_seconds=time.monotonic()-started,
        scope=c['scope'],model_promoted=False)
    prefix=DEST/f'case{case_id:03d}';assert not prefix.with_suffix('.json').exists()
    np.savez_compressed(prefix.with_suffix('.npz'),rate_hz=rates,dt_ms=dt,frequency_hz=f,burn_ms=case['burn_ms'],amplitude=case['absolute_amplitude'])
    write(prefix.with_suffix('.json'),result);log('DENSITY LINEAR',case_id,f,error,result['passed'],num_ok)


def dispatch():
    c=read(OUT/'conditional_density_linear_response_contract.json')
    def worker(case):
        path=DEST/f'case{case["id"]:03d}.json'
        assert not path.exists(), 'Dispatch once; audit terminal state before any retry'
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--case',str(case['id'])]
        with (DEST/f'case{case["id"]:03d}.log').open('w') as handle:
            process=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT)
            code=process.wait()
        return dict(id=case['id'],pid=process.pid,exit_code=code,result=path.name)
    status=dict(status='RUNNING',pid=os.getpid(),conditions=len(c['cases']),workers=c['workers'],finished=[])
    write(DEST/'jobs.json',status)
    with ThreadPoolExecutor(max_workers=c['workers']) as pool:
        for future in as_completed([pool.submit(worker,case) for case in c['cases']]):
            status['finished'].append(future.result());write(DEST/'jobs.json',status)
            log('LINEAR CASES',len(status['finished']),len(c['cases']))
    status['status']='COMPLETE' if all(r['exit_code']==0 for r in status['finished']) else 'EXECUTION_FAILURE'
    write(DEST/'jobs.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--register',action='store_true')
    parser.add_argument('--dispatch',action='store_true');parser.add_argument('--case',type=int)
    a=parser.parse_args()
    if a.register:register()
    elif a.dispatch:dispatch()
    else:one(a.case)
