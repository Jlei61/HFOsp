"""Observe one full trajectory of the existing g40 mean kinetic candidate.

Same integration equations and operators as run_topic4_spatial_kinetic_candidate.
Adds fresh t=0 initialization and passive Fig.5 raster/resource observers only.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from run_topic4_spatial_kinetic_candidate import (
    ROOT, SOURCE, OUT as MODEL, NE, NI, DT, write, init_drive,
    SpatialOUDrive, SpatialOUConfig, EventDelay,
)

OUT = ROOT/'results/topic4_sef_hfo/fig5_spatial_kinetic_full_trajectory_20260916'


def run(a):
    folder = OUT/('canary' if a.start_ms else 'run')
    folder.mkdir(parents=True, exist_ok=False)
    started = time.time()
    write(folder/'status.json', dict(status='RUNNING', pid=os.getpid(), config=vars(a)))
    grid = MODEL/'coarse_40'
    prep = json.loads((grid/'prepared.json').read_text())
    geo = dict(np.load(grid/'geometry.npz'))
    md = dict(np.load(grid/'model.npz'))
    observed = dict(np.load(SOURCE/'replay/geometry.npz'))
    obsgeo = dict(np.load(SOURCE/'approx/coarse_20/geometry.npz'))
    protocol = json.loads((ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/protocol.json').read_text())
    assert prep['graph_identity'] == protocol['identity']
    p = prep['params']; n = 1600; depth = prep['max_delay_steps']
    cell = np.r_[geo['cell_e'], geo['cell_i']+n]
    counts = np.r_[geo['count_e'], geo['count_i']].astype(float)
    safe = np.maximum(counts, 1.)
    rng = np.random.default_rng(a.seed)
    if a.start_ms:
        cp = dict(np.load(SOURCE/f'replay/runs/eta0.0005_s{a.seed}/checkpoints/t{a.start_ms}ms.npz'))
        meta = json.loads(str(cp['__meta__']))
        v, ref, sa, ia, sg, ig = [cp[k].copy() for k in ('V','ref','s_E','I_E','s_I','I_I')]
        z, m = cp['slow__z'].copy(), cp['slow__m'].copy()
        pending_a, pending_g = cp['ring_sE'], cp['ring_sI']
        rng.bit_generator.state = meta['rng_state']
        drive = init_drive(geo, prep, cp, meta, a.seed)
        xi = meta['xi']
    else:
        # The native engine consumes these two recorder draws before its first input.
        rng.choice(NE, size=80, replace=False)
        rng.choice(NI, size=20, replace=False)
        v = np.full(NE+NI, p['V_reset'], dtype=float)
        ref = np.zeros(NE+NI, np.int32)
        sa, ia, sg, ig = [np.zeros(NE+NI) for _ in range(4)]
        z, m = np.ones(NE+NI), np.zeros(NE+NI)
        spec = dict(prep['spatial_ou']); spec.pop('role'); spec.pop('seed_offset')
        spec['seed'] = a.seed+500000
        drive = SpatialOUDrive(geo['positions_e'], 20., DT, SpatialOUConfig(**spec))
        xi = 0.; pending_a = pending_g = np.empty((0, NE+NI))
        meta = dict(step=0)
    theta = np.r_[geo['vtheta_e'], geo['vtheta_i']]
    assert (theta[:NE]<18).sum()==781 and not (theta[:NE]>18).any()
    tau = np.r_[np.full(NE,p['tau_m_E']),np.full(NI,p['tau_m_I'])]
    dv = np.exp(-DT/tau)
    scale_a, scale_g = tau/p['tau_r_AMPA'], tau/p['tau_r_GABA']
    ext_incr = scale_a*np.r_[np.full(NE,p['J_ext_E']),np.full(NI,p['J_ext_I'])]
    rsa, ria = np.exp(-DT/p['tau_r_AMPA']),np.exp(-DT/p['tau_d_AMPA'])
    rsg, rig = np.exp(-DT/p['tau_r_GABA']),np.exp(-DT/p['tau_d_GABA'])
    ref_steps = np.r_[np.full(NE,round(p['tau_ref_E']/DT)),np.full(NI,round(p['tau_ref_I']/DT))]
    oua = np.exp(-DT/p['tau_n'])
    oub = p['sigma_n']*1e-3*np.sqrt(p['tau_n']/2.)*np.sqrt(1-oua*oua)
    weights = {k:load_npz(grid/f'delay_{k}.npz') for k in ('ee','ei','ie','ii')}
    delay = EventDelay(weights, None, n, depth)
    del weights
    steps = round(a.duration_ms/DT)
    assert steps%100==0
    samples = observed['sample_ids']
    raster = np.zeros((steps,len(samples)),bool)
    fields = np.zeros((steps//10,400),np.uint16)
    finefields = np.zeros((steps//10,n),np.uint16)
    populations = np.zeros((steps//10,2),np.uint16)
    regions = np.zeros((steps//10,3),np.uint16)
    region_ix = [np.flatnonzero(geo['g175']==j) for j in range(3)]
    zstats = np.zeros((steps//50+1,8)); mstats = np.zeros((steps//50+1,4))
    def record_slow(j):
        ze, me = z[:NE], m[:NE]
        zstats[j] = [ze.mean(),ze.std(),*np.quantile(ze,[.1,.5,.9]),*[ze[ix].mean() for ix in region_ix]]
        mstats[j] = [me.mean(),*[me[ix].mean() for ix in region_ix]]
    record_slow(0)
    block = np.zeros(400,np.int32); fineblock = np.zeros(n,np.int32)
    popblock = np.zeros(2,np.int32); regblock = np.zeros(3,np.int32)
    ext_rate = np.empty(steps,np.float32)
    qa = dict(model='g40_mean', Z='dynamic', M='dynamic', unchanged_biological_parameters=True,
              graph_identity_match=True, no_poststart_native_activity_or_resources=True,
              starts_from_native_checkpoint=bool(a.start_ms), fixed_sample_ids=samples.tolist(),
              external_checkpoint_checks={})
    last_log = time.time(); chunk_start=0
    for t in range(steps):
        means,_ = delay.take(t)
        mean_a,mean_g = means
        sa *= rsa; sg *= rsg
        sa += mean_a[cell]*scale_a; sg += mean_g[cell]*scale_g
        if t<pending_a.shape[0]:
            slot=(meta['step']+t)%pending_a.shape[0]
            sa += pending_a[slot]; sg += pending_g[slot]
        xi=oua*xi+oub*rng.standard_normal()
        nu_now=max(float(md['nu_ext_per_ms'])+xi,0.)
        nu=np.full(NE+NI,nu_now)
        nu[:NE]=np.maximum(nu_now+drive.step(a.start_ms+t*DT),0.)
        sa += rng.poisson(nu*DT,size=NE+NI)*ext_incr
        ext_rate[t]=nu_now
        ia=sa+(ia-sa)*ria; ig=sg+(ig-sg)*rig
        current=ia-z*ig-.0005*m
        ref-=1; np.maximum(ref,0,out=ref)
        free=ref==0
        v=np.where(free,current+(v-current)*dv,p['V_reset'])
        spk=free & (v>=theta)
        v[spk]=p['V_reset']; ref[spk]=ref_steps[spk]
        z[:NE] += DT/5000.*((ig[:NE]<protocol['I_th']).astype(float)-z[:NE])
        np.clip(z[:NE],0.,1.,out=z[:NE])
        m[:NE] -= DT/1000.*m[:NE]; m[:NE] += spk[:NE]
        s=np.bincount(cell[spk],minlength=2*n).astype(np.int32)
        delay.push(t,s[:n]/safe[:n],s[n:]/safe[n:])
        raster[t]=spk[samples]
        fineblock+=s[:n]
        block+=np.bincount(obsgeo['cell_e'][spk[:NE]],minlength=400).astype(np.int32)
        regblock+=np.bincount(geo['g175'][spk[:NE]],minlength=3).astype(np.int32)
        popblock += [int(spk[:NE].sum()),int(spk[NE:].sum())]
        if (t+1)%10==0:
            j=t//10; fields[j]=block; finefields[j]=fineblock; populations[j]=popblock; regions[j]=regblock
            block.fill(0);fineblock.fill(0);popblock.fill(0);regblock.fill(0)
        if (t+1)%50==0:record_slow((t+1)//50)
        elapsed_ms=(t+1)*DT; absolute_ms=a.start_ms+elapsed_ms
        if (t+1)%5000==0 or t+1==steps:
            end=(t+1)//10
            chunk=folder/'chunks';chunk.mkdir(exist_ok=True)
            np.savez_compressed(chunk/f'{chunk_start:06d}_{end:06d}.npz',field_1ms=fields[chunk_start:end],
                fine_field_1ms=finefields[chunk_start:end],spikes_1ms=populations[chunk_start:end],
                regions_1ms=regions[chunk_start:end],raster=raster[chunk_start*10:end*10],
                slow_time_ms=a.start_ms+np.arange(chunk_start//5,end//5+1)*5,
                Z=zstats[chunk_start//5:end//5+1],M=mstats[chunk_start//5:end//5+1],start_ms=a.start_ms+chunk_start)
            chunk_start=end
        if (t+1)%100==0:
            path=SOURCE/f'replay/runs/eta0.0005_s{a.seed}/checkpoints/t{int(absolute_ms)}ms.npz'
            if path.exists():
                with np.load(path) as native:
                    nm=json.loads(str(native['__meta__']))
                    check=dict(rng=rng.bit_generator.state==nm['rng_state'],xi=xi==nm['xi'],
                        spatial_rng=drive._rng.bit_generator.state==nm['external_drive__rng_state'],
                        spatial_state=np.array_equal(drive._state,native['external_drive__field_state']),
                        spatial_cache=np.array_equal(drive._cached,native['external_drive__cached']))
                assert all(check.values()),check
                qa['external_checkpoint_checks'][str(int(absolute_ms))]={k:bool(v) for k,v in check.items()}
        if time.time()-last_log>25:
            end=(t+1)//10; lo=max(0,end-100)
            rate=float(populations[lo:end,0].sum()/NE/max((end-lo)*.001,.001))
            print('time_s',round(absolute_ms/1000,3),'E_Hz',round(rate,2),'Z',round(z[:NE].mean(),4),'wall_s',round(time.time()-started),flush=True)
            write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),completed_ms=elapsed_ms,config=vars(a),wall_s=time.time()-started))
            last_log=time.time()
    assert np.array_equal(fields.sum(1),populations[:,0])
    assert np.array_equal(finefields.sum(1),populations[:,0])
    assert np.array_equal(regions.sum(1),populations[:,0])
    assert all(np.isfinite(x).all() for x in (v,ia,ig,z,m))
    assert np.all((z>=0)&(z<=1)) and np.all(m>=0)
    assert np.all(z[NE:]==1) and np.all(m[NE:]==0)
    qa.update(count_conservation=True,finite_physical_state=True)
    if a.start_ms==8000:
        old=MODEL/f'runs/g40_mean_s{a.seed}_8000_2500ms/fields.npz'
        with np.load(old) as prior:
            qa['existing_candidate_observation_parity']=bool(np.array_equal(fields,prior['field_e20_1ms'][:len(fields)]) and
                np.array_equal(regions,prior['regions_1ms'][:len(regions)]) and np.array_equal(ext_rate,prior['global_external_rate'][:steps]))
        assert qa['existing_candidate_observation_parity']
    np.savez_compressed(folder/'trajectory.npz',field_1ms=fields,fine_field_1ms=finefields,
        spikes_1ms=populations,regions_1ms=regions,raster=raster,slow_time_ms=a.start_ms+np.arange(len(zstats))*5,
        Z=zstats,M=mstats,start_ms=a.start_ms,global_external_rate=ext_rate,
        sample_ids=samples,cell_e_counts=obsgeo['count_e'],fine_cell_e_counts=geo['count_e'],
        region_counts=np.bincount(geo['g175'],minlength=3),centers_mm=geo['centers_mm'],core_radius_mm=1.5)
    np.savez_compressed(folder/'end_state.npz',V=v,ref=ref,s_E=sa,I_E=ia,s_I=sg,I_I=ig,Z=z,M=m)
    write(folder/'qa.json',qa)
    write(folder/'status.json',dict(status='COMPLETE',config=vars(a),wall_s=time.time()-started,
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print('COMPLETE',folder,'wall_s',round(time.time()-started),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--start-ms',type=int,default=0)
    parser.add_argument('--duration-ms',type=int,default=12500)
    parser.add_argument('--seed',type=int,default=9108401)
    run(parser.parse_args())
