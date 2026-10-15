#!/usr/bin/env python3
"""Bounded, isolated synaptic-duration interventions with a fixed observer."""
from pathlib import Path
import argparse, copy, hashlib, importlib.util, json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'src/snn_engine'))
import numpy as np
from scipy.sparse import csr_matrix
from scripts import run_topic4_multievent_distribution_v2_1 as execution
from scripts import run_topic4_xy_research as base
OUT=ROOT/'results/topic4_sef_hfo/local_width_mechanism_pilot'
SOURCE=ROOT/'results/topic4_sef_hfo/contact_native_integrated_pilot'
ARMS={'baseline':3.5,'recurrent_7ms':7.,'recurrent_14ms':14.,'external_7ms':7.}
PAIRS=[(6101,842901),(6102,842901)]

def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)

def load_engine():
    spec=importlib.util.spec_from_file_location('local_width_engine',OUT/'engine_snapshot.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def make_engine():
    source=ROOT/'src/snn_engine/kick_probe.py'
    code=source.read_text()
    substitutions=[
      ('    t0 = time.time()\n', '    _lw_base = np.zeros(N)\n    _lw_alt = np.zeros(N)\n    _lw_decay = np.exp(-dt / LOCAL_WIDTH_TAU_MS)\n    t0 = time.time()\n'),
      ('        I_E = s_E + (I_E - s_E) * decay_IE\n',
       '        if LOCAL_WIDTH_ARM == "baseline":\n            I_E = s_E + (I_E - s_E) * decay_IE\n        else:\n            _lw_base = s_E + (_lw_base - s_E) * decay_IE\n            I_E = _lw_base\n'),
      ('        if pathway_trace_on and t % pathway_trace_stride == 0:\n',
       '        _lw_actual_rec = I_E_rec if track_rec else None\n'
       '        if LOCAL_WIDTH_ARM != "baseline":\n'
       '            if not track_rec:\n                raise RuntimeError("split kinetics requires recurrent tracking")\n'
       '            _lw_is_rec = LOCAL_WIDTH_ARM.startswith("recurrent")\n'
       '            _lw_source = s_E_rec if _lw_is_rec else s_E - s_E_rec\n'
       '            _lw_original = I_E_rec if _lw_is_rec else _lw_base - I_E_rec\n'
       '            _lw_alt = _lw_source + (_lw_alt - _lw_source) * _lw_decay\n'
       '            I_E = _lw_base + (_lw_alt - _lw_original)\n'
       '            _lw_actual_rec = _lw_alt if _lw_is_rec else I_E_rec\n'
       '        if LOCAL_WIDTH_HOOK is not None:\n'
       '            LOCAL_WIDTH_HOOK.currents(t, tm, I_E, I_I, _lw_actual_rec, nu_vec, ext)\n'
       '        if pathway_trace_on and t % pathway_trace_stride == 0:\n'),
      ('            pathway_recurrent_E_to_E_mean[index] = np.mean(I_E_rec[:NE])',
       '            pathway_recurrent_E_to_E_mean[index] = np.mean(_lw_actual_rec[:NE])'),
      ('            pathway_recurrent_E_to_I_mean[index] = np.mean(I_E_rec[NE:])',
       '            pathway_recurrent_E_to_I_mean[index] = np.mean(_lw_actual_rec[NE:])'),
      ('        # ----- record -----\n',
       '        if LOCAL_WIDTH_HOOK is not None:\n            LOCAL_WIDTH_HOOK.spikes(t, spk)\n        # ----- record -----\n')]
    for old,new in substitutions:
        if code.count(old)!=1:raise RuntimeError(f'engine patch location changed: {old}')
        code=code.replace(old,new)
    code+='\n# Isolated pilot controls; the production engine remains untouched.\nLOCAL_WIDTH_ARM="baseline"\nLOCAL_WIDTH_TAU_MS=3.5\nLOCAL_WIDTH_HOOK=None\n'
    (OUT/'engine_snapshot.py').write_text(code)
    return dict(original_engine=str(source),original_sha256=base.sha(source),isolated_engine_sha256=base.sha(OUT/'engine_snapshot.py'))

class Recorder:
    """Diagnostic Gaussian samplers; primary full-contact observer is unchanged."""
    def __init__(self,sub):
        from src.topic4_node_dualmode import neuron_contact_sampling_weights
        self.ne=sub.net['NE'];self.dt=sub.params.dt;self.stride=round(1/self.dt)
        self.names=sub.contact_names;self.xy=sub.contact_xy
        weights=neuron_contact_sampling_weights(sub.positions_e,sub.contact_xy,kernel_width_mm=.25)
        # Retain effectively all mass, sparse only for diagnostic runtime.
        trimmed=np.where(weights>=1e-10,weights,0.)
        self.omitted_mass=float(np.max(1-trimmed.sum(axis=1)))
        self.w=csr_matrix(trimmed/trimmed.sum(axis=1,keepdims=True))
        wi=neuron_contact_sampling_weights(sub.net['pos'][self.ne:],sub.contact_xy,kernel_width_mm=.25)
        wi=np.where(wi>=1e-10,wi,0.);self.wi=csr_matrix(wi/wi.sum(axis=1,keepdims=True))
        self.current=[];self.firing=[];self.times=[]
        self.accE=np.zeros(self.ne);self.accI=np.zeros(sub.net['NI'])
        self.innov=hashlib.sha256();self.sampled_steps=0
    def currents(self,t,tm,E,I,rec,rate,ext):
        if t<1000:self.innov.update(np.asarray(ext,dtype=np.int32).tobytes());self.sampled_steps+=1
        if t%self.stride:return
        self.times.append(tm)
        self.current.append(np.stack([self.w@E[:self.ne],self.w@I[:self.ne],self.w@rec[:self.ne],self.w@(E[:self.ne]-rec[:self.ne]),self.w@rate[:self.ne]]))
    def spikes(self,t,spk):
        self.accE+=spk[:self.ne];self.accI+=spk[self.ne:]
        if (t+1)%self.stride==0:
            self.firing.append(np.stack([self.w@self.accE,self.wi@self.accI])*1000/(self.stride*self.dt))
            self.accE.fill(0);self.accI.fill(0)
    def save(self,path,arm,tau,p):
        np.savez_compressed(path,time_ms=np.array(self.times,dtype=np.float32),
           currents=np.array(self.current,dtype=np.float32),spike_rate_hz=np.array(self.firing,dtype=np.float32),
           contact_xy_mm=self.xy,contact_names=self.names)
        return dict(arm=arm,changed_decay_ms=tau,base_AMPA_decay_ms=p.tau_d_AMPA,
            trace_path=str(path),trace_sha256=base.sha(path),diagnostic_gaussian_omitted_mass=self.omitted_mass,
            current_columns=['total_E','GABA_to_E','recurrent_E','external_E','external_rate_per_ms'],
            firing_columns=['E_rate_Hz','I_rate_Hz'],time_convention='Currents sampled at 1ms bin start; firing bins integrate the following 1ms.',
            poisson_prefix_sha256=self.innov.hexdigest(),poisson_prefix_steps=self.sampled_steps,
            actual_base_params=vars(p))

def worker():
    # All original worker arguments are forwarded verbatim.
    from scripts import run_topic4_multidimensional_worker as w
    cfg=base.read(Path(sys.argv[sys.argv.index('--config')+1]));arm=sys.argv[sys.argv.index('--candidate-id')+1]
    output=Path(sys.argv[sys.argv.index('--out-json')+1])
    engine=load_engine();engine.LOCAL_WIDTH_ARM=arm;engine.LOCAL_WIDTH_TAU_MS=ARMS[arm]
    holder={};build=w.build_substrate
    def build_recorded(*a,**kw):
        sub=build(*a,**kw);holder['sub']=sub;return sub
    def run(p,net,**kw):
        hook=Recorder(holder['sub']);engine.LOCAL_WIDTH_HOOK=hook
        result=engine.simulate_kick(p,net,dump_pathway_trace=True,**kw)
        audit=hook.save(output.with_name(output.stem+'_local_currents.npz'),arm,ARMS[arm],p)
        audit.update(engine_sha256=base.sha(OUT/'engine_snapshot.py'),unchanged_noise_law=True,
                     unchanged_poisson_base_rate_calculation=True,readout_smoothing_sigma_ms=5.,duration_ms=p.T)
        write(output.with_name(output.stem+'_kinetics.json'),audit)
        return result
    w.build_substrate=build_recorded;w.simulate_kick=run;w.main()


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'design.json').exists():return base.read(OUT/'design.json')
    meta=make_engine()
    history=base.read(SOURCE/'confirmation_scores.json')['candidates']
    anchor=next(r['candidate'] for r in history if r['candidate_id']=='tshape_anchor3_B_minus')
    candidates=[]
    for arm,tau in ARMS.items():
        c=copy.deepcopy(anchor);c.update(candidate_id=arm,kinetic_intervention=dict(arm=arm,tau_ms=tau),parent_candidate_id=anchor['candidate_id']);candidates.append(c)
    folder=OUT/'execution';folder.mkdir(exist_ok=True)
    cp=folder/'execution_config.json';mp=folder/'candidate_manifest.json';sp=folder/'runtime_snapshot.json'
    cfg=base.read(SOURCE/'execution/confirmation/execution_config.json')
    cfg.update(output_root=str(folder),candidate_manifest=str(mp))
    cfg['search']['fit_network_seeds']=[6101,6102];cfg['search']['dynamics_seeds']=[842901]
    write(cp,cfg);write(mp,dict(config_sha256=base.sha(cp),candidates=candidates,frozen_before_simulation=True))
    hashes=base.read(SOURCE/'execution/confirmation/runtime_snapshot.json')['source_hashes']
    hashes={p:base.sha(ROOT/p) for p in hashes}
    from scripts import run_topic4_multidimensional_worker as loaded_worker
    hashes.update(loaded_worker._runtime_provenance(None)['runtime_module_sha256'])
    for p in [Path(__file__),OUT/'engine_snapshot.py']:
        hashes[str(p.relative_to(ROOT))]=base.sha(p)
    write(sp,dict(source_hashes=hashes,input_hashes={str(cp):base.sha(cp),str(mp):base.sha(mp)},identity_kind='isolated_duration_intervention'))
    d=dict(version='local_width_mechanism_pilot_v1',created_unix=time.time(),anchor=anchor,arms=ARMS,pairs=PAIRS,
       new_physical_runs=8,duration_ms=24000,maximum_workers=4,source=meta,
       question='Does longer internal excitation extend self-terminating local activity, compared with longer external-current filtering?',
       intervention='Separate recurrent versus external AMPA decay; baseline params including nu_theta computation remain unchanged. Recurrent arm changes E->E and E->I filters together.',
       invariants='Same graph, weights, thresholds, delays, OU law, Poisson baseline and fixed full-contact observer; Z/M and stimulation off.',
       primary='Per-network event distributions of local width, recruitment span, participation and centroid span; no TA/TB training.',
       caveat='The current kernel preserves isolated-spike integral but changes peak and duration. Not proof of a particular receptor mechanism; 14ms is a model capacity probe.',
       stop='After eight units and automatic analysis; no adaptive expansion, freeze or Fig5.',snapshot=str(sp))
    write(OUT/'design.json',d);return d


def main():
    if '--worker' in sys.argv:sys.argv.remove('--worker');worker();return
    ap=argparse.ArgumentParser();ap.add_argument('--prepare-only',action='store_true');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    if not 1<=args.workers<=4:raise ValueError('workers must be 1..4')
    d=prepare()
    if args.prepare_only:print(OUT);return
    import fcntl
    with open(OUT/'controller.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from scripts import run_topic4_observable_loss_physical_pilot as jobs
        folder=OUT/'execution';cp=folder/'execution_config.json';mp=folder/'candidate_manifest.json';sp=folder/'runtime_snapshot.json'
        # Reuse process-tree memory accounting and bounded dispatch, with this isolated wrapper.
        shim=OUT/'run_topic4_multidimensional_worker.py'
        expected='import runpy,sys\nsys.argv.insert(1,"--worker")\nrunpy.run_path('+repr(str(Path(__file__).resolve()))+',run_name="__main__")\n'
        if shim.exists() and shim.read_text()!=expected:raise RuntimeError('worker entry changed')
        shim.write_text(expected)
        execution.WORKER=shim;jobs.OUT=OUT
        joblist=[(arm,t,n) for arm in ARMS for t,n in PAIRS]
        jobs.run_jobs('KINETICS',joblist,(folder,cp,mp,sp),args.workers)
        subprocess.run([execution.PYTHON,str(ROOT/'scripts/analyze_topic4_local_width_mechanism_pilot.py')],cwd=ROOT,env=execution.ENV,check=True)
        write(OUT/'status.json',dict(status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',complete=8,total=8,updated_unix=time.time(),review=str(OUT/'scientific_review.md')))

if __name__=='__main__':main()
