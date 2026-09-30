"""Observe an unchanged native 0-3 s replay at the current g40 resolution.

Only geometric target selection and recording are new. This supplies inputs
for a fixed-readout diagnostic; it is not a new autonomous rate experiment.
"""
from native_same_history_feedback import native, ROOT
import replay
from datetime import datetime
import argparse, os, time

np=native.np
OUT=ROOT/'results/topic4_sef_hfo/fig5_zm_runaway_mechanism_20260918'
DEST=OUT/'native_early_surround_inputs'
GEO=ROOT/'results/topic4_sef_hfo/interictal_brunel_spatial_bifurcation_20260917/operators/g40/geometry.npz'
NAMES=['ampa','ampa2','gaba','gaba2','zgaba','zgaba2','mcurrent','mcurrent2',
       'net','net2','voltage','voltage2','z','z2','ampa_gaba','ampa_zgaba','net_voltage']


def select(geo):
    pos=geo['positions']; pop=geo['population']; size=geo['group_size']; region=geo['group_region']
    A,B=geo['centers_mm']; axis=(B-A)/np.linalg.norm(B-A); normal=np.array([-axis[1],axis[0]])
    selected=[]; rows=[]
    def add(target,mask,role):
        eligible=np.flatnonzero(mask & (size>=5) & ~np.isin(np.arange(len(size)),selected))
        assert len(eligible)
        g=int(eligible[np.argmin(np.linalg.norm(pos[eligible]-target,axis=1))])
        selected.append(g); rows.append(dict(group=g,role=role,target_mm=target.tolist(),position_mm=pos[g].tolist(),
            population=int(pop[g]),region=int(region[g]),N=int(size[g]),theta_mv=float(geo['threshold_mv'][g])))
    for u in [.2,.5,.8]:
        for offset in [-3.,0.,3.]:
            add(A+u*(B-A)+offset*normal,(pop==0)&(region==2),f'surround_u{u:g}_offset{offset:g}')
    for k,center in enumerate([A,B]):add(center,(pop==0)&(region==k),f'core_{"AB"[k]}')
    for u in [0.,.2,.5,.8,1.]:add(A+u*(B-A),pop==1,f'I_u{u:g}')
    assert len(set(selected))==16
    return rows


def register():
    native.check_reference_sources(); assert native.read(native.REPLAY_RUN/'replay_qa.json')['status']=='PASS'
    DEST.mkdir(exist_ok=True); assert not (DEST/'contract.json').exists()
    geo=dict(np.load(GEO)); rows=select(geo)
    native.write(DEST/'contract.json',dict(created_local=datetime.now().astimezone().isoformat(),
        question='Does the fixed current rate response recover early native surround activity when driven by native rather than self-generated group inputs?',
        reason='Repaired whole rate field has0complete earlyevents vs11native in0.5-3s and much weaker surroundZdepletion; the previous inputbridge coveredonly8-10.37s atg20.',
        interval_ms=[0,3000],primary_window_ms=[500,3000],record_step_ms=.1,grid=40,groups=3479,
        geometry_sha256=native.sha(GEO),selected_groups=rows,
        selection='Geometry only:9surround E groups near three core-axis coordinates and three transverse offsets, nearest core E groups,5Iaxis groups; N>=5. No event or response result selection.',
        recording='All3479group actualcounts per0.1ms; currentmoments,pre-stepZ/M/V and externalrate for16selectedtargets. Originalnativechunk/fields/checkpoint observers unchanged.',
        native_model='Same original40000cells, graph, individualthresholds, seed9108401, drive andZM. Startfromoriginalinitialization; ZandMdynamic.',
        audit='Six500ms chunks andexisting10mspercellZM/current/input records bitwiseequal tooriginal. Compare selectedgroup counts tophysicalpopulation/spatialtotals; independent moment reaggregation at originalsparse current snapshots.',
        downstream='Currentconditioned39 frozenresponse only. Compare measuredgroupmeans and projectedphysicalmeans/privatevariance. Nativefuturecounts are diagnostic inputs, never accepted as autonomousrate validation.50msfixed countbins,1mseventprofiles,nonnegativeZ/Mandratebounds. No fitting or onset label.',
        budget='One3s observationalreplay; no newseed,parameterchange,ratefit,whole-networklaunch orcontinuation. If identityfails, stop downstream and report exactreason.'))
    print('EARLY INPUT REGISTERED',rows,flush=True)


def run(device):
    c=native.read(DEST/'contract.json'); assert native.sha(GEO)==c['geometry_sha256']
    assert not (DEST/'progress.json').exists(),'No silent restart or second replay'
    ref=native.check_reference_sources(); geo=dict(np.load(GEO)); selected=np.array([r['group'] for r in c['selected_groups']])
    allgroups=geo['cell_group']; sizes=geo['group_size']; P=len(sizes); G=len(selected)
    assert np.array_equal(np.bincount(allgroups,minlength=P),sizes)
    table=np.full(P,-1,int); table[selected]=np.arange(G)
    cells=np.flatnonzero(table[allgroups]>=0); ids=table[allgroups[cells]]
    assert np.array_equal(np.bincount(ids,minlength=G),sizes[selected])
    def avg(v):return np.bincount(ids,weights=np.asarray(v)[cells],minlength=G)/sizes[selected]
    # Reuse the exact M0 recorder with a shorter stop and separate destination.
    base=DEST/'replay'; base.mkdir(); replay.base_dir=lambda canary:base
    replay.REPLAY_END_MS=3000; replay.Z_TIMES_MS=[3000]
    original_job=replay.reference_job
    def job(seed):
        value=dict(original_job(seed)); value['device']=device; value['horizon_s']=3.
        return value
    replay.reference_job=job
    folder=base/'runs'/job(native.MAIN_SEED)['name']
    native.write(DEST/'progress.json',dict(status='RUNNING',pid=os.getpid(),time_ms=0,source=str(native.REPLAY_RUN)))
    original=native.core.old.simulate_kick
    globals()['compute_nu_theta']=original.__globals__['compute_nu_theta']
    def observer(params,net,*args,**kwargs):
        assert kwargs.get('resume_state') is None
        assert np.array_equal(net['pos'],geo['original_positions'])
        slow=kwargs['slow']; previous_current=kwargs.get('current_observer')
        previous_input=kwargs['input_observer'];previous_spike=kwargs['spike_observer'];previous_sink=kwargs['checkpoint_sink']
        threshold=np.broadcast_to(kwargs['V_th_per_neuron'],(len(allgroups),))
        np.savez_compressed(DEST/'membership.npz',**geo,selected_groups=selected,selected_cells=cells,
            selected_cell_group_index=ids,actual_threshold_mv=threshold)
        moments=np.empty((5000,len(NAMES),G)); counts=np.empty((5000,P),dtype=np.uint16)
        drive=np.empty((5000,G));tm_cur=np.empty(5000);tm_sp=np.empty(5000);tm_in=np.empty(5000)
        n=ns=ni=0; start=0
        def current(tm,ie,ii,voltage):
            nonlocal n
            if previous_current is not None:previous_current(tm,ie,ii,voltage)
            z=slow.z;m=slow.cfg.eta_m*slow.m;zi=z*ii;neti=ie-zi-m
            values=[ie,ie*ie,ii,ii*ii,zi,zi*zi,m,m*m,neti,neti*neti,voltage,voltage*voltage,z,z*z,ie*ii,ie*zi,neti*voltage]
            for j,v in enumerate(values):moments[n,j]=avg(v)
            tm_cur[n]=tm;n+=1
        def inputs(tm,nu,xi):
            nonlocal ni
            previous_input(tm,nu,xi);drive[ni]=avg(nu);tm_in[ni]=tm;ni+=1
        def spikes(tm,spk):
            nonlocal ns
            previous_spike(tm,spk);counts[ns]=np.bincount(allgroups,weights=spk,minlength=P).astype(np.uint16)
            tm_sp[ns]=tm;ns+=1
        def sink(k,engine):
            nonlocal n,ns,ni,start
            try:previous_sink(k,engine)
            finally:
                if k%5000==0:
                    assert n==ns==ni==k-start
                    assert np.array_equal(tm_cur[:n],tm_sp[:ns]) and np.array_equal(tm_cur[:n],tm_in[:ni])
                    out=DEST/'inputs';out.mkdir(exist_ok=True)
                    np.savez_compressed(out/f'{start:010d}_{k:010d}.npz',time_ms=tm_cur[:n],moments=moments[:n],
                        moment_names=NAMES,spikes=counts[:n],external_rate_per_ms=drive[:n],start_step=start,end_step=k)
                    native.write(DEST/'progress.json',dict(status='RUNNING',pid=os.getpid(),time_ms=k*.1,closed_chunks=k//5000))
                    print('EARLY INPUT CHUNK',start,k,flush=True);n=ns=ni=0;start=k
        kwargs.update(current_observer=current,input_observer=inputs,spike_observer=spikes,checkpoint_sink=sink)
        return original(params,net,*args,**kwargs)
    native.core.old.simulate_kick=observer
    result=replay.worker(native.MAIN_SEED);assert result['status']=='COMPLETE'
    audit()


def audit():
    c=native.read(DEST/'contract.json');geo=dict(np.load(DEST/'membership.npz'));P=len(geo['group_size'])
    selected=geo['selected_groups']; groups=geo['cell_group'];sizes=geo['group_size'];E=geo['population']==0
    folder=DEST/'replay/runs'/native.reference_job(native.MAIN_SEED)['name']
    assert native.read(folder/'result.json')['status']=='COMPLETE'
    rows=[];samples=0
    for f in sorted((DEST/'inputs').glob('*.npz')):
        source=native.REPLAY_RUN/'chunks'/f.name
        with np.load(folder/'chunks'/f.name) as x,np.load(source) as y:
            mismatches=[key for key in y.files if key not in x or x[key].dtype!=y[key].dtype or not np.array_equal(x[key],y[key])]
            assert not mismatches,(f.name,mismatches)
            pop=x['population_0p1ms'];field=x['field_0p1ms'];common=len(y.files)
        with np.load(folder/'fields'/f.name) as x,np.load(native.REPLAY_RUN/'fields'/f.name) as y:
            mismatches=[key for key in y.files if key not in x or not np.array_equal(x[key],y[key])]
            assert not mismatches,(f.name,mismatches)
        with np.load(f) as z:
            sp=z['spikes'];mom=z['moments'];times=z['time_ms'];start_step=int(z['start_step'])
            assert np.array_equal(sp[:,E].sum(1),pop[:,0])
            assert np.array_equal(sp[:,~E].sum(1),pop[:,1])
            # The current g40 cell IDs reduce to the physical original20x20 field.
            cells=np.floor(geo['positions']/1.).astype(int); coarse=cells[:,1]*20+cells[:,0]
            check=np.zeros_like(field)
            for g in np.flatnonzero(E):check[:,coarse[g]]+=sp[:,g]
            assert np.array_equal(check,field)
            assert np.isfinite(mom).all() and np.allclose(np.diff(times),.1,rtol=0,atol=1e-10)
            samples+=len(times);assert np.max(sp-sizes[None,:])<=0
            # Check the newly selected E moments against existing raw native fields.
            with np.load(native.REPLAY_RUN/'fields'/f.name) as raw:
                zm={key:raw[key] for key in ['z','m']}
                for j,step in enumerate(raw['zm_step']):
                    where=int(step-start_step); assert abs(times[where]-step*.1)<1e-9
                    for q,g in enumerate(selected):
                        if not E[g]:continue
                        mask=groups[:32000]==g
                        for name,key in [('z','z'),('mcurrent','m')]:
                            value=zm[key][j,mask].mean()*(.0005 if key=='m' else 1)
                            assert abs(value-mom[where,NAMES.index(name),q])<1e-12
        rows.append(dict(chunk=f.name,original_keys_bitwise=common,raw_fields_bitwise=True,group_counts_restrict_exactly=True))
    assert len(rows)==6 and samples==30000
    native.write(DEST/'replay_audit.json',dict(status='PASS',rows=rows,samples=samples,selected_groups=selected.tolist(),
        full_terminal_engine_comparison='No independently stored3s engine. Original sixchunk records andpercell10msfields matchbitwise; new3s fullengine saved forrecord identity.',
        scope='Unchanged observational replay only; no rate-response or autonomous validation inferred.'))
    native.write(DEST/'progress.json',dict(status='COMPLETE_REPLAY_AUDIT_PASS',pid=os.getpid(),time_ms=3000))
    print('EARLY INPUT REPLAY AUDIT PASS',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['register','run','audit']);p.add_argument('--device',type=int,default=1);a=p.parse_args()
    {'register':register,'run':lambda:run(a.device),'audit':audit}[a.command]()
