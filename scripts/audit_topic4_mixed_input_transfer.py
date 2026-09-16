#!/usr/bin/env python3
"""Stationary native-LIF checks of the existing mixed AMPA/GABA transfer closure."""
from topic4_spatial_boundary_common import OUT, OLD, REFERENCE, read, write
from topic4_e_only_z_rate import EOnlySystem
from src.topic4_patient_zm_meanfield import lif_rate_gauss_legendre
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import time
import os


def cases():
    s=EOnlySystem();m=s.m;n=s.n
    native=np.load(OUT/'native/z9400_history9400.npz');src=np.load(OLD/'external_input.npz')
    ag=np.zeros((100,400));ag[src['cell_e'],native['cell_e']]=1
    field=(native['field_e_count_1ms']@ag.T)/src['count_e']*1000
    sm=field[-1000:].reshape(200,5,100).mean(1)
    cells=list(dict.fromkeys([int(np.argmax(sm.mean(0))),int(np.argmax(sm.std(0)))]))
    if len(cells)==1:cells.append(int(np.argsort(sm.std(0))[-2]))
    rate=np.load(OLD/'rate/native_replay_expected.npz');r=rate['fields_hz']/1000;z=rate['z'];expected=src['expected_rate_per_ms']
    jobs=[]
    for population in ('E','I'):
        rows=[]
        for t in (8000,8400,8800,9400):
            re,ri=r[t-50:t].mean(0);nu_e,nu_i=expected[(t-50)*10:t*10].mean(0)
            for cell in cells:
                if population=='E':
                    ae=(m.w_ee@re)[cell]+m.j_ext_e_mv*nu_e[cell]
                    be=(m.v_ee@re)[cell]+m.j_ext_e_mv**2*nu_e[cell]
                    ai=(m.w_ei@ri)[cell];bi=(m.v_ei@ri)[cell];q=float(z[t-1,cell])
                else:
                    ae=(m.w_ie@re)[cell]+m.j_ext_i_mv*nu_i[cell]
                    be=(m.v_ie@re)[cell]+m.j_ext_i_mv**2*nu_i[cell]
                    ai=(m.w_ii@ri)[cell];bi=(m.v_ii@ri)[cell];q=1.
                rows.append({'name':f't{t}_cell{cell}','time_ms':t,'cell':cell,'aE':float(ae),'bE':float(be),
                    'aI':float(ai),'bI':float(bi),'z':q})
        j=m.j_ext_e_mv if population=='E' else m.j_ext_i_mv
        for extra in (0.,.25,1.,4.):
            nu=m.nu_ext_per_ms+extra
            rows.append({'name':f'external_only_extra{extra:g}','cell':cells[0],'aE':j*nu,'bE':j*j*nu,'aI':0.,'bI':0.,'z':1.})
        jobs.append({'population':population,'cases':rows,'neurons_per_case':2048,'duration_ms':4000,'burn_ms':2000})
    return jobs


def current_var_factor(dt,rise,decay,tm):
    a=np.exp(-dt/rise);b=np.exp(-dt/decay)
    return dt*(tm/rise*(1-b)/(a-b))**2*(a*a/(1-a*a)+b*b/(1-b*b)-2*a*b/(1-a*b))


def run(job):
    start=time.time();s=EOnlySystem();m=s.m;pop=job['population'];rows=job['cases'];n=len(rows);nn=job['neurons_per_case'];dt=s.dt
    tm=m.tau_mem_e_ms if pop=='E' else m.tau_mem_i_ms
    tref=m.tau_ref_e_ms if pop=='E' else m.tau_ref_i_ms
    rng=np.random.default_rng(9108701 if pop=='E' else 9108702)
    ae=np.array([r['aE'] for r in rows]);be=np.array([r['bE'] for r in rows]);ai=np.array([r['aI'] for r in rows]);bi=np.array([r['bI'] for r in rows]);q=np.array([r['z'] for r in rows])
    je=be/ae;le=ae/je;ji=np.divide(bi,ai,out=np.zeros(n),where=ai>0);li=np.divide(ai,ji,out=np.zeros(n),where=ji>0)
    # Equivalent independent Poisson inputs match both first and second physical edge-weight sums.
    assert np.allclose(le*je,ae) and np.allclose(le*je*je,be)
    assert np.allclose(li*ji,ai) and np.allclose(li*ji*ji,bi)
    thresholds=np.empty((n,nn));pred=[];pred_discrete=[]
    for k,row in enumerate(rows):
        if pop=='E':nodes=m.threshold_nodes_e[row['cell']];w=m.threshold_weights_e[row['cell']]
        else:nodes=np.array([m.v_threshold_i_mv]);w=np.array([1.])
        quantiles=(np.arange(nn)+.5)/nn;thresholds[k]=nodes[np.minimum(np.searchsorted(np.cumsum(w),quantiles),len(nodes)-1)]
        mu=tm*(s.ga*ae[k]-s.gg*q[k]*ai[k]);ex=tm*be[k];inh=tm*q[k]*q[k]*bi[k]
        shift=2.065/2*np.sqrt((ex*(s.ra+s.ta)+inh*(s.rg+s.tau))/tm)
        phi=lif_rate_gauss_legendre(mu-shift,np.sqrt(ex+inh),tau_mem_ms=tm,tau_ref_ms=tref,
            v_threshold_mv=nodes,v_reset_mv=m.v_reset_mv)
        pred.append(float(np.average(phi,weights=w)*1000))
        un,cnt=np.unique(thresholds[k],return_counts=True)
        phi2=lif_rate_gauss_legendre(mu-shift,np.sqrt(ex+inh),tau_mem_ms=tm,tau_ref_ms=tref,v_threshold_mv=un,v_reset_mv=m.v_reset_mv)
        pred_discrete.append(float(np.average(phi2,weights=cnt)*1000))
    v=np.full((n,nn),m.v_reset_mv);ref=np.zeros((n,nn),np.int32)
    se=np.broadcast_to(tm*s.ga*ae[:,None],v.shape).copy();si=np.broadcast_to(tm*s.gg*ai[:,None],v.shape).copy()
    ce=se.copy();ci=si.copy();steps=round(job['duration_ms']/dt);rates=np.empty((steps,n),np.float32)
    moments=np.zeros((4,n));moment_samples=0
    for step in range(steps):
        se*=np.exp(-dt/s.ra);si*=np.exp(-dt/s.rg)
        se+=rng.poisson(le[:,None]*dt,size=v.shape)*(tm/s.ra*je[:,None])
        si+=rng.poisson(li[:,None]*dt,size=v.shape)*(tm/s.rg*ji[:,None])
        ce=se+(ce-se)*np.exp(-dt/s.ta);ci=si+(ci-si)*np.exp(-dt/s.tau)
        ref=np.maximum(ref-1,0);free=ref==0;cur=ce-q[:,None]*ci
        v=np.where(free,cur+(v-cur)*np.exp(-dt/tm),m.v_reset_mv)
        sp=free&(v>=thresholds);v[sp]=m.v_reset_mv;ref[sp]=round(tref/dt)
        rates[step]=sp.mean(1)/dt*1000
        if step*dt>=job['burn_ms'] and step%100==0:
            moments+=np.array([ce.mean(1),ci.mean(1),ce.var(1),ci.var(1)]);moment_samples+=1
        if step%10000==0:write(OUT/'mixed_transfer_progress'/f'{pop}.json',{'status':'RUNNING','time_ms':step*dt,'elapsed_s':time.time()-start})
    measured=rates[round(job['burn_ms']/dt):].mean(0,dtype=np.float64)
    blocks=rates[round(job['burn_ms']/dt):].reshape(8,-1,n).mean(1,dtype=np.float64)
    moments/=moment_samples
    for k,row in enumerate(rows):
        row.update({'native_LIF_rate_hz':float(measured[k]),'existing_Phi_hz':pred[k],
            'discretized_threshold_Phi_hz':pred_discrete[k], 'block_rates_hz':blocks[:,k].tolist(),
            'measured_mean_AMPA':float(moments[0,k]),'expected_mean_AMPA':float(tm*s.ga*ae[k]),
            'measured_mean_GABA':float(moments[1,k]),'expected_mean_GABA':float(tm*s.gg*ai[k]),
            'measured_variance_AMPA':float(moments[2,k]),'expected_variance_AMPA':float(current_var_factor(dt,s.ra,s.ta,tm)*be[k]),
            'measured_variance_GABA':float(moments[3,k]),'expected_variance_GABA':float(current_var_factor(dt,s.rg,s.tau,tm)*bi[k])})
    folder=OUT/'mixed_transfer';folder.mkdir(exist_ok=True)
    np.savez_compressed(folder/f'{pop}.npz',rate_hz=rates,thresholds=thresholds,dt_ms=dt)
    result={'status':'COMPLETE','population':pop,'seconds':time.time()-start,'cases':rows,
        'scope':'Independent moment-matched Poisson input to original LIF/filter/reset equations, not the recurrent native network. This isolates the stationary transfer approximation; it does not validate dynamic susceptibility.'}
    write(folder/f'{pop}.json',result);return result


def main():
    jobs=cases();write(OUT/'mixed_transfer_protocol.json',{'status':'DEFINED_BEFORE_ASSAYS','jobs':jobs,
        'reason':'Previous rate-response calibration used excitation-only pulses. Verify the mixed AMPA/GABA transfer at actual boundary contexts before attributing mismatch solely to Z averaging.',
        'selection':'Two cells: highest native mean and native 5-ms SD under frozen Z9.4; contexts use preceding 50-ms mean rate/input from prescribed-native-Z rate trajectory at 8/8.4/8.8/9.4 s. Four excitation-only sanity cases per population.',
        'intervention':'No fitting. Original threshold/reset/membrane/synapse/Z-target parameters; compound Poisson inputs match local first and second connection-weight sums; this replaces recurrence only to isolate the transfer function.',
        'workers':2,'resource_gate':'Start after 20x20 resolution jobs finish. These are the final two bounded reduced/closure sensitivity assays.'})
    while True:
        p=OUT/'resolution_status.json';status=read(p) if p.exists() else {}
        if status.get('status')=='COMPLETE':break
        if status.get('status')=='FAILED':raise RuntimeError(status)
        os.kill(read(OUT/'resolution_process.json')['pid'],0);time.sleep(10)
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(run,j) for j in jobs]):
            rows.append(f.result());write(OUT/'mixed_transfer_status.json',{'status':'RUNNING','completed':len(rows),'rows':rows})
    write(OUT/'mixed_transfer_status.json',{'status':'COMPLETE','rows':rows})


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'mixed_transfer_status.json',{'status':'FAILED','error':repr(exc)});raise
