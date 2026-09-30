"""Spatial kinetic population candidate; qualify against native SNN before bifurcation.

Empirical joint voltage/current/refractory/Z/M distributions are represented by
particles. Recurrent communication uses the realized graph's spatial-delay
operators, not a stationary firing-rate transfer function. This first closure
test retains native particle counts to isolate the communication approximation.
It is a candidate, not an established equivalent or a dimension-reduced ODE.
"""
import os
for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.sparse import load_npz

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.topic4_spatial_ou_drive import SpatialOUDrive, SpatialOUConfig
from topic4_kinetic_delay import EventDelay

SOURCE = ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1'
OUT = ROOT/'results/topic4_sef_hfo/fig5_spatial_kinetic_equivalence_20260916'
NE, NI, DT = 32000, 8000, .1


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False,
                   default=lambda x: x.item() if isinstance(x,np.generic) else str(x))+'\n')


def init_drive(geo, prep, cp, meta, seed):
    spec = dict(prep['spatial_ou']); spec.pop('role'); spec.pop('seed_offset')
    spec['seed'] = seed+500000
    drive = SpatialOUDrive(geo['positions_e'], 20., DT, SpatialOUConfig(**spec))
    drive._rng.bit_generator.state = meta['external_drive__rng_state']
    drive._state = cp['external_drive__field_state'].copy()
    drive._cached = cp['external_drive__cached'].copy()
    drive._next_step = meta['external_drive__next_step']
    drive._last_step = meta['external_drive__last_step']
    return drive


def sample_jump(mean, var, rng):
    """Positive effective shot noise with specified conditional first two moments.

    Moment matching is a closure assumption, not the original quenched graph.
    An independent RNG must be used so native external innovations stay paired.
    """
    amplitude = np.divide(var, mean, out=np.zeros_like(mean), where=mean>1e-15)
    lam = np.divide(mean, amplitude, out=np.zeros_like(mean), where=amplitude>1e-15)
    return amplitude*rng.poisson(np.maximum(lam, 0.))


def run(a):
    tag = f'g{a.grid}_{a.closure}_s{a.seed}_{a.start_ms}_{a.duration_ms}ms'
    if a.z_field_ms is not None:
        tag += f'_z{a.z_field_ms}_h{a.history_ms}'
    folder = OUT/'runs'/tag
    folder.mkdir(parents=True, exist_ok=False)
    start = time.time()
    write(folder/'status.json', dict(status='RUNNING', pid=os.getpid(), config=vars(a)))
    grid = SOURCE/f'approx/coarse_{a.grid}' if a.grid in (10,20) else OUT/f'coarse_{a.grid}'
    prep = json.loads((grid/'prepared.json').read_text())
    geo = dict(np.load(grid/'geometry.npz')); md = dict(np.load(grid/'model.npz'))
    protocol = json.loads((ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/protocol.json').read_text())
    assert prep['graph_identity'] == protocol['identity']
    p = prep['params']; n = a.grid**2; depth = prep['max_delay_steps']
    history_ms = a.start_ms if a.history_ms is None else a.history_ms
    checkpoint = SOURCE/f'replay/runs/eta0.0005_s{a.seed}/checkpoints/t{history_ms}ms.npz'
    cp = dict(np.load(checkpoint)); meta = json.loads(str(cp['__meta__']))
    if history_ms!=a.start_ms:
        anchor=dict(np.load(checkpoint.parent/f't{a.start_ms}ms.npz'))
        anchor_meta=json.loads(str(anchor['__meta__']))
        for key in ('ring_sE','ring_sI'):
            original=cp[key];shift=(anchor_meta['step']-meta['step'])%len(original)
            cp[key]=np.roll(original,shift,axis=0)
            assert np.array_equal(original[meta['step']%len(original)],cp[key][anchor_meta['step']%len(original)])
        for key in ('external_drive__field_state','external_drive__cached'):cp[key]=anchor[key]
        meta=anchor_meta
    if a.z_field_ms is not None:
        with np.load(checkpoint.parent/f't{a.z_field_ms}ms.npz') as zsource:cp['slow__z']=zsource['slow__z'].copy()
    assert meta['absolute_time_ms'] == a.start_ms
    cell = np.r_[geo['cell_e'], geo['cell_i']+n]
    counts = np.r_[geo['count_e'], geo['count_i']].astype(float)
    safe_counts = np.maximum(counts,1.)
    assert np.all(counts[cell]>0)
    obsgeo = dict(np.load(SOURCE/'approx/coarse_20/geometry.npz'))
    W = {k: load_npz(grid/f'delay_{k}.npz') for k in ('ee','ei','ie','ii')}
    Q = {k: load_npz(grid/f'vdelay_{k}.npz') for k in W} if a.closure=='shot' else None
    event_delay = EventDelay(W, Q, n, depth) if a.delay_engine=='event' else None
    histories = [np.zeros((depth, n)), np.zeros((depth, n))]
    # Native per-neuron synaptic state and pending pre-start input are retained.
    v, ref, sa, ia, sg, ig = [cp[k].copy() for k in ('V','ref','s_E','I_E','s_I','I_I')]
    z, m = cp['slow__z'].copy(), cp['slow__m'].copy()
    pending_a, pending_g = cp['ring_sE'], cp['ring_sI']
    theta = np.r_[geo['vtheta_e'], geo['vtheta_i']]
    tau = np.r_[np.full(NE,p['tau_m_E']),np.full(NI,p['tau_m_I'])]
    dv = np.exp(-DT/tau)
    scale_a = tau/p['tau_r_AMPA']; scale_g = tau/p['tau_r_GABA']
    ext_incr = scale_a*np.r_[np.full(NE,p['J_ext_E']),np.full(NI,p['J_ext_I'])]
    rsa, ria = np.exp(-DT/p['tau_r_AMPA']),np.exp(-DT/p['tau_d_AMPA'])
    rsg, rig = np.exp(-DT/p['tau_r_GABA']),np.exp(-DT/p['tau_d_GABA'])
    ref_steps = np.r_[np.full(NE,round(p['tau_ref_E']/DT)),np.full(NI,round(p['tau_ref_I']/DT))]
    rng = np.random.default_rng(); rng.bit_generator.state = meta['rng_state']
    recurrent_rng = np.random.default_rng(a.seed+700000)
    drive = init_drive(geo, prep, cp, meta, a.seed)
    xi = meta['xi']; oua = np.exp(-DT/p['tau_n'])
    oub = p['sigma_n']*1e-3*np.sqrt(p['tau_n']/2.)*np.sqrt(1-oua*oua)
    steps = int(round(a.duration_ms/DT)); assert steps%10 == 0
    field = np.zeros((steps//10, 2*n), np.int32)
    obsfield = np.zeros((steps//10,400),np.int32)
    obsblock = np.zeros(400,np.int32)
    regions = np.zeros((steps//10,3),np.int32)
    region_counts = np.bincount(geo['g175'],minlength=3)
    region_block = np.zeros(3,np.int32)
    slow = np.zeros((steps//100, 4*n), np.float32)
    global_rates = np.empty(steps, np.float32)
    block = np.zeros(2*n, np.int32)
    qa = dict(graph_identity_match=True, original_parameters=True, Z='dynamic' if a.z_field_ms is None else 'frozen field', M='dynamic',
              native_particle_count=NE+NI, no_native_poststart_activity_forcing=True)
    last_print = time.time()
    for t in range(steps):
        if event_delay is not None:
            means,variances = event_delay.take(t)
            mean_a,mean_g = means
        else:
            he, hi = histories[0].ravel(), histories[1].ravel()
            mean_a = np.r_[W['ee']@he, W['ie']@he]
            mean_g = np.r_[W['ei']@hi, W['ii']@hi]
        if a.closure == 'shot':
            if event_delay is not None:
                var_a,var_g = variances
            else:
                var_a = np.r_[Q['ee']@he, Q['ie']@he]
                var_g = np.r_[Q['ei']@hi, Q['ii']@hi]
            jump_a = sample_jump(mean_a[cell], var_a[cell], recurrent_rng)*scale_a
            jump_g = sample_jump(mean_g[cell], var_g[cell], recurrent_rng)*scale_g
        else:
            jump_a = mean_a[cell]*scale_a
            jump_g = mean_g[cell]*scale_g
        sa *= rsa; sg *= rsg
        sa += jump_a; sg += jump_g
        if t < pending_a.shape[0]:
            slot = (meta['step']+t)%pending_a.shape[0]
            sa += pending_a[slot]; sg += pending_g[slot]
        xi = oua*xi+oub*rng.standard_normal()
        nu_now = max(float(md['nu_ext_per_ms'])+xi, 0.)
        nu = np.full(NE+NI, nu_now)
        nu[:NE] = np.maximum(nu_now+drive.step(a.start_ms+t*DT), 0.)
        sa += rng.poisson(nu*DT, size=NE+NI)*ext_incr
        global_rates[t] = nu_now
        ia = sa+(ia-sa)*ria; ig = sg+(ig-sg)*rig
        current = ia-z*ig-.0005*m
        ref -= 1; np.maximum(ref, 0, out=ref)
        free = ref==0
        v = np.where(free, current+(v-current)*dv, p['V_reset'])
        spk = free & (v>=theta)
        v[spk] = p['V_reset']; ref[spk] = ref_steps[spk]
        if a.z_field_ms is None:
            z[:NE] += DT/5000.*((ig[:NE]<protocol['I_th']).astype(float)-z[:NE])
            np.clip(z[:NE], 0., 1., out=z[:NE])
        m[:NE] -= DT/1000.*m[:NE]; m[:NE] += spk[:NE]
        s = np.bincount(cell[spk], minlength=2*n).astype(np.int32)
        if event_delay is not None:
            event_delay.push(t,s[:n]/safe_counts[:n],s[n:]/safe_counts[n:])
        else:
            histories[0][1:] = histories[0][:-1]
            histories[1][1:] = histories[1][:-1]
            histories[0][0] = s[:n]/safe_counts[:n]
            histories[1][0] = s[n:]/safe_counts[n:]
        block += s
        obsblock += np.bincount(obsgeo['cell_e'][spk[:NE]],minlength=400).astype(np.int32)
        region_block += np.bincount(geo['g175'][spk[:NE]],minlength=3).astype(np.int32)
        if (t+1)%10==0:
            field[t//10] = block; block.fill(0)
            obsfield[t//10] = obsblock; obsblock.fill(0)
            regions[t//10] = region_block; region_block.fill(0)
        if (t+1)%100==0:
            slow[t//100,:2*n] = np.bincount(cell,weights=z,minlength=2*n)/safe_counts
            slow[t//100,2*n:] = np.bincount(cell,weights=m,minlength=2*n)/safe_counts
        if time.time()-last_print>25:
            lo=max(0,(t+1)//10-100); hi=(t+1)//10
            print(tag,'time_ms',a.start_ms+(t+1)*DT,'E_hz',field[lo:hi,:n].sum()/NE/((hi-lo)*.001), 'elapsed',round(time.time()-start),flush=True)
            write(folder/'status.json',dict(status='RUNNING',pid=os.getpid(),config=vars(a),completed_ms=(t+1)*DT,wall_s=time.time()-start))
            last_print=time.time()
    end_path = checkpoint.parent/f't{a.start_ms+a.duration_ms}ms.npz'
    if end_path.exists():
        end = dict(np.load(end_path)); em = json.loads(str(end['__meta__']))
        qa['external_rng_matches_native'] = rng.bit_generator.state == em['rng_state']
        qa['global_ou_matches_native'] = xi == em['xi']
        qa['spatial_ou_matches_native'] = bool(np.array_equal(drive._state,end['external_drive__field_state']) and np.array_equal(drive._cached,end['external_drive__cached']))
        qa['spatial_rng_matches_native'] = drive._rng.bit_generator.state == em['external_drive__rng_state']
        assert all(qa[k] for k in ('external_rng_matches_native','global_ou_matches_native','spatial_ou_matches_native','spatial_rng_matches_native')), qa
    assert np.isfinite(v).all() and np.isfinite(ia).all() and np.isfinite(ig).all()
    assert (z>=0).all() and (z<=1).all() and (m>=0).all()
    qa['physical_Z_M_and_finite_state'] = True
    qa['I_Z_1_M_0'] = bool(np.all(z[NE:]==1) and np.all(m[NE:]==0))
    if a.start_ms==8000 and a.duration_ms<=4500:
        with np.load(SOURCE/f'approx/input/replay_s{a.seed}.npz') as original_input:
            qa['entire_global_external_rate_matches_native'] = bool(np.array_equal(global_rates,original_input['global_rate_per_ms'][:steps]))
        assert qa['entire_global_external_rate_matches_native']
    if a.start_ms==10370 and a.seed==9108401:
        with np.load(SOURCE/'approx/input/W1.npz') as original_input:
            qa['entire_global_external_rate_matches_native'] = bool(np.array_equal(global_rates,original_input['global_rate_per_ms'][:steps]))
        assert qa['entire_global_external_rate_matches_native']
    if a.z_field_ms is not None:
        qa['frozen_Z_still_applied_and_bitwise_constant']=bool(np.array_equal(z,cp['slow__z']))
        qa['M_changed']=bool(np.any(m[:NE]!=cp['slow__m'][:NE]))
        assert qa['frozen_Z_still_applied_and_bitwise_constant'] and qa['M_changed']
    assert np.array_equal(obsfield.sum(1),field[:,:n].sum(1))
    np.savez_compressed(folder/'fields.npz',counts=counts,spikes_1ms=field,slow_10ms=slow,
                        region_counts=region_counts,regions_1ms=regions,
                        field_e20_1ms=obsfield,count_e20=obsgeo['count_e'],
                        start_ms=a.start_ms,dt_ms=DT,global_external_rate=global_rates)
    np.savez_compressed(folder/'end_state.npz',V=v,ref=ref,s_E=sa,I_E=ia,s_I=sg,I_I=ig,Z=z,M=m)
    write(folder/'qa.json',qa)
    write(folder/'status.json',dict(status='COMPLETE',config=vars(a),wall_s=time.time()-start,
           final_Z_E=float(z[:NE].mean()),final_M_E=float(m[:NE].mean()),scope='kinetic closure pilot; equivalence not established'))
    print(tag,'COMPLETE',round(time.time()-start,1),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--grid',type=int,choices=(10,20,40),default=20)
    ap.add_argument('--closure',choices=('mean','shot'),default='shot')
    ap.add_argument('--seed',type=int,choices=(9108401,9108402),default=9108401)
    ap.add_argument('--start-ms',type=int,default=8000)
    ap.add_argument('--duration-ms',type=int,default=1000)
    ap.add_argument('--delay-engine',choices=('event','matrix'),default='event')
    ap.add_argument('--z-field-ms',type=int,default=None)
    ap.add_argument('--history-ms',type=int,default=None)
    run(ap.parse_args())
