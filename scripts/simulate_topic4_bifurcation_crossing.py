#!/usr/bin/env python3
"""Carry rates, synaptic states and all delay history across a parameter change."""
from analyze_topic4_corrected_bifurcation import *
from screen_topic4_corrected_high_activity import diagnostics
from concurrent.futures import ProcessPoolExecutor, as_completed
import time


def run(qstart,qend,duration=8000.,pulse_mv=0.,name=None):
    started=time.time();s=System();m=s.m;n=s.n;dt=s.dt;D=s.cfg['max_delay_steps']
    a=np.load(OUT/'fixed_points.npz');key=min([k for k in a.files if k.startswith('low_')],key=lambda k:abs(float(k[4:])-qstart))
    r,err,ok=s.solve(qstart,a[key]);assert ok
    re,ri=r[:n].copy(),r[n:].copy();he=np.tile(re,(D,1));hi=np.tile(ri,(D,1))
    ra,rg,ta,tau=s.ra,s.rg,s.ta,s.tau;te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms;nu=m.nu_ext_per_ms
    rise=np.array([ra,rg,ra,rg,ra,ra])[:,None];decay=np.array([ta,tau,ta,tau,ta,ta])[:,None]
    membrane=np.array([te,te,ti,ti,te,ti])[:,None];ar=np.exp(-dt/rise);ad=np.exp(-dt/decay)
    ext_e=np.full(n,m.j_ext_e_mv*nu);ext_i=np.full(n,m.j_ext_i_mv*nu);ops=s.ops
    def drive(q):return np.array([ops['ee']@he.ravel(),q*(ops['ei']@hi.ravel()),ops['ie']@he.ravel(),q*(ops['ii']@hi.ravel()),ext_e,ext_i])
    g=dt*membrane/rise*drive(qstart)/(1-ar);c=g.copy()
    # Same small rate displacement on both sides; no repeated stimuli or resets.
    re*=1.01
    geometry=np.load(BASE/'coarse_10/geometry.npz');cell=geometry['cell_e'];pos=geometry['positions_e']
    coords=np.array([np.bincount(cell,weights=pos[:,k],minlength=n)/m.count_e for k in range(2)]).T
    center=np.array([3.0639104904,18.7198569555]);mask=np.linalg.norm(coords-center,axis=1)<2.5
    nsteps=round(duration/dt);stride=round(1/dt);fields=np.empty((nsteps//stride,2,n),np.float32);qtrace=np.empty(len(fields),np.float32)
    for step in range(nsteps):
        t=step*dt;q=qstart if t<2000 else qend
        g=g*ar+dt*membrane/rise*drive(q);c=g+(c-g)*ad
        ex=np.r_[te*(m.v_ee@re+m.j_ext_e_mv**2*nu),ti*(m.v_ie@re+m.j_ext_i_mv**2*nu)]
        inh=np.r_[te*q*q*(m.v_ei@ri),ti*q*q*(m.v_ii@ri)]
        mu=np.r_[c[0]-c[1]+c[4],c[2]-c[3]+c[5]]
        if 2000<=t<2020:mu[:n]+=pulse_mv*mask
        p=s.phi(mu,ex,inh);r=np.r_[re,ri];nr=r+dt/s.tr*(p-r)
        he[1:]=he[:-1].copy();hi[1:]=hi[:-1].copy();he[0]=re;hi[0]=ri;re,ri=nr[:n],nr[n:]
        if (step+1)%stride==0:fields[step//stride]=nr.reshape(2,n)*1000;qtrace[step//stride]=q
    assert np.isfinite(fields).all();name=name or f'q{qstart:g}_to_{qend:g}_pulse{pulse_mv:g}'
    folder=OUT/'crossings';folder.mkdir(exist_ok=True);np.savez_compressed(folder/f'{name}.npz',fields_hz=fields,q=qtrace,frame_ms=1.,gating=g,current=c,history_e=he,history_i=hi)
    row={'name':name,'qstart':qstart,'qend':qend,'duration_ms':duration,'switch_ms':2000,'pulse_mv':pulse_mv,'pulse_ms':20,'initial':'Exact low equilibrium with stationary filters and history; E rates multiplied 1.01 once at t=0.',
        'pulse_mask':'Cells within 2.5 mm of frozen core A; transient added mean E voltage, not a change to topology.',
        'no_resets_at_switch':True,'seconds':time.time()-started,'diagnostics':diagnostics(fields,m)}
    write(folder/f'{name}.json',row);print(name,row['diagnostics']['signals']['global_E'],flush=True);return row


def batch():
    jobs=[(.76,.76,8000,0),(.76,.72,8000,0),(.76,.76,8000,2),(.76,.72,8000,2),(.75,.75,8000,0),(.75,.75,8000,5)]
    write(OUT/'crossing_protocol.json',{'jobs':jobs,'columns':['qstart','qend','duration_ms','local_pulse_mv'],'switch_ms':2000,'max_workers':3,
        'question':'Does loss of low-state stability cause sustained recruitment with full state carried; does a finite local pulse switch states while the equilibrium is still locally stable?',
        'scope':'Corrected rate capability on existing reference geometry; these low states are not certified patient interictal events.'})
    rows=[]
    with ProcessPoolExecutor(max_workers=3) as pool:
        fut=[pool.submit(run,*j) for j in jobs]
        for f in as_completed(fut):
            rows.append(f.result());write(OUT/'crossing_status.json',{'status':'RUNNING','completed':len(rows),'total':len(jobs),'results':rows})
    write(OUT/'crossing_status.json',{'status':'COMPLETE','completed':len(rows),'total':len(jobs),'results':rows})


def supplemental(group):
    import screen_topic4_corrected_high_activity as screen
    screen.OUT=OUT
    if group=='recruitment':
        protocol=read(OUT/'recruitment_scan_protocol.json');jobs=[(q,t,protocol['duration_ms']) for q,t in protocol['jobs']];status='recruitment_scan_status.json'
    else:
        protocol=read(OUT/'branch_followup_protocol.json');jobs=[(q,20.611550480127335,d) for q,d in protocol['jobs_q_duration_ms']];status='branch_followup_status.json'
    rows=[]
    with ProcessPoolExecutor(max_workers=protocol['max_workers']) as pool:
        for f in as_completed([pool.submit(screen.run,q,t,10,d) for q,t,d in jobs]):
            rows.append(f.result());write(OUT/status,{'status':'RUNNING','completed':len(rows),'total':len(jobs),'rows':rows})
    write(OUT/status,{'status':'COMPLETE','completed':len(rows),'total':len(jobs),'rows':rows})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['crossing','recruitment','branch'],default='crossing');args=p.parse_args()
    if args.group=='crossing':batch()
    else:supplemental(args.group)
