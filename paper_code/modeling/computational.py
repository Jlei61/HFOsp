"""Build a native spatial LIF SNN, simulate, observe, fit and run Z/M controls.

No stored graph, trajectory, score or movie is used by these entry points.
Patient centroid tables and contact geometry are observations, not model output.
"""
from dataclasses import asdict
from types import SimpleNamespace
import copy
import json
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from modeling.src.snn_engine.params import Params, compute_nu_theta
from modeling.src.snn_engine.connectivity import place_neurons
from modeling.src.snn_engine.connectivity_rot import build_connectivity_rot
from modeling.cpu_acceleration import accelerated_engine
simulate_kick = accelerated_engine()
from modeling.src.snn_engine.mz_historical import MZSlowVars, MZSlowVarsConfig
from modeling.src.snn_engine.lfp import LFPRecorder
from modeling.src.topic4_streaming_spike_readout import simulate_streaming
from modeling.src.topic4_core_field import sample_core_quantiles, core_thresholds
from modeling.src.topic4_core_connectivity_v2 import core_index_for, scale_pathway_blocks
from modeling.scripts.topic4_core_count_optimization.physics import CoreCountOU
from modeling.src.topic4_observation_repaired import observe, calibrate_reference
from modeling.src.topic4_three_observable_objective import ThreeObservableObjective
from modeling.src.topic4_three_observable_label_free import LabelFreeThreeObservableObjective
from modeling.cohort_objective import summaries,descriptive


def load_configuration(path):
    return json.loads(Path(path).read_text())


def build_substrate(config, *, angle_offset=0.0):
    """New positions + no-autapse delayed adjacency from the declared topology seed."""
    p=Params(**config['parameters']);rng=np.random.default_rng(p.seed)
    pos,labels,ne,ni=place_neurons(p,rng)
    net=build_connectivity_rot(p,pos,labels,ne,ni,rng,
        theta_EE=np.deg2rad(config['theta_deg']+angle_offset), AR=config['aspect_ratio'],verbose=False)
    net['rng']=rng
    return dict(params=p,net=net,config=copy.deepcopy(config),angle_offset=float(angle_offset))


def core_field(substrate, centers, outward=1.25):
    """Lowering-only quantiles, circular membership, EE same-core/outward weights."""
    p,base=substrate['params'],substrate['net'];cfg=substrate['config'];ne=base['NE'];ni=base['NI']
    centers=np.asarray(centers,float);radius=cfg['radius_mm'];k=len(centers)
    if centers.shape!=(k,2) or k not in (1,2,3,4) or not np.isfinite(centers).all():
        raise ValueError('Expected 1..4 finite core centers')
    if np.any(centers-radius<0) or np.any(centers+radius>p.L):
        raise ValueError('Core outside sheet')
    if any(np.linalg.norm(centers[i]-centers[j])<=2*radius for i in range(k) for j in range(i)):
        raise ValueError('Core disks must be disjoint')
    ec=core_index_for(base['pos'][:ne],centers,[radius]*k)
    ic=core_index_for(base['pos'][ne:],centers,[radius]*k)
    if any(np.sum(ec==i)==0 for i in range(k)):
        raise ValueError('Empty core at this spatial sampling density')
    raw=core_thresholds(sample_core_quantiles(ne,cfg['quantile_seed']),cfg['core_mean'],cfg['core_std'])
    v=np.full(ne+ni,p.V_th);v[:ne]-=np.where(ec>=0,np.minimum(np.maximum(p.V_th-raw,0),p.V_th-11),0)
    prm=dict(EE_same_core_scale=.85,EE_core_to_out_scale=float(outward),EE_out_to_out_scale=1.,
             EI_same_core_scale=1.,IE_same_core_scale=1.,II_same_core_scale=1.)
    ampa,gaba,audit=scale_pathway_blocks(base['ampa_by_delay'],base['gaba_by_delay'],ne,ec,ic,prm)
    net={key:value for key,value in base.items() if key not in ('ampa_flat','gaba_flat')}
    net.update(ampa_by_delay=ampa,gaba_by_delay=gaba)
    return dict(net=net,vtheta=v,ec=ec,ic=ic,centers=centers,outward=outward,audit=audit)


class TracedHistoricalSlow(MZSlowVars):
    """Preserve historical arithmetic; sample traces at 2 ms without changing state."""
    def __init__(self,*args,dt,**kwargs):
        super().__init__(*args,**kwargs);self.dt=dt;self.samples=[]
    def _record_traces(self,spk):
        if self._step_i % round(2/self.dt)==0:
            self.samples.append([(self._step_i+1)*self.dt,self.z[:self.NE].mean(),self.m[:self.NE].mean()])
    def trace_arrays(self):
        a=np.asarray(self.samples)
        return dict(time_ms=a[:,0],z_mean=a[:,1],m_mean=a[:,2])


class CurrentReadout:
    """The Fig5 LFP proxy: weighted abs(I_E)+abs(I_I), sampled at 2000 Hz.

    Uses native E-cell currents and the original finite-radius radial kernel;
    this is not extracellular voltage and not the Fig4 spike-density envelope.
    """
    def __init__(self,p,net,contacts):
        self.rec=LFPRecorder(p,net['pos'],net['labels'],sites=contacts)
        self.stride=round(.5/p.dt);self.times=[];self.currents=[]
    def observe(self,t,tm,xi,nu_now,ext,delta_rate,V,I_E,I_I,spk):
        if t%self.stride==0:
            self.times.append(tm);self.currents.append(self.rec.sample(I_E,I_I))


def simulate(substrate, centers, contacts, *, outward=1.25, seed=847401,
             duration_ms=None, zm=None, record_current=False):
    p=copy.copy(substrate['params']); cfg=substrate['config'];p.T=duration_ms or p.T
    field=core_field(substrate,centers,outward);net=field['net'];ne,ni=net['NE'],net['NI']
    net['rng']=np.random.default_rng(seed)
    drive=CoreCountOU(field['ec'],p.nu_ext_ratio*compute_nu_theta(p)[0],.95,0,p.dt,p.tau_n,p.sigma_n,seed)
    slow=None if zm is None else TracedHistoricalSlow(ne+ni,p.V_th,MZSlowVarsConfig(**zm),NE=ne,core_mask_E=field['ec']>=0,dt=p.dt)
    observer=CurrentReadout(p,net,contacts) if record_current else None
    result=simulate_streaming(simulate_kick,p,net,KICK_BOOST=0.0,t_kick=1e9,
        positions=net['pos'][:ne],montage=SimpleNamespace(contacts=np.asarray(contacts)),
        V_th_per_neuron=field['vtheta'],slow=slow,global_ou_loading=np.zeros(ne+ni),
        deterministic_external_mask=np.r_[field['ec']<0,np.ones(ni,bool)],external_e_rate_drive=drive,
        step_observer=observer,early_stop_runaway=True,es_thresh_hz=120 if zm is None else 200,es_dur_ms=100 if zm is None else 200,
        post_runaway_record_ms=(cfg.get("zm_post_entry_ms",1500) if zm is not None else 500),verbose=False)
    spikes=result.pop('E_spk_bool');env,dt,_=spikes.envelope();movie=spikes.native()
    out=dict(envelope=env,envelope_dt_ms=dt,movie=movie['activity_counts'],frame_ms=movie['frame_ms'],
        time_ms=result['times'],rate_E=result['rate_E'],rate_I=result['rate_I'],centers=np.asarray(centers),
        outward=outward,vtheta=field['vtheta'],positions=net['pos'],ec=field['ec'],contact_xy=np.asarray(contacts),
        physical_status=('RUNAWAY' if zm is None else 'HIGH_RATE_SAFEGUARD_STOP') if result['runaway_early_stop_ms'] is not None else 'COMPLETE',
        parameters=asdict(p),seed=seed,wall_seconds=result['wall_s'],angle_offset=substrate['angle_offset'])
    if slow is not None:out['slow']=slow.trace_arrays()
    if observer is not None:out.update(current_time_ms=np.asarray(observer.times),contact_current=np.asarray(observer.currents).T)
    return out


def fit_objective(fit, cal, names, *, n_fourier=1024, draws=200, seed=2026091401, cal_draw_indices=None):
    """Fit Fourier maps and target moments from patient FIT; scale on separate CAL.

    The exact original feature map is specific to four SCL + eleven ICL contacts.
    The label-free wrapper never consumes a K-means label or a class proportion.
    """
    feature_source=ThreeObservableObjective(names,None,[.5,.5],seed=seed,n_fourier=n_fourier)
    feature_source.fit_maps(fit)
    objective=LabelFreeThreeObservableObjective(feature_source);objective.fit_targets(fit)
    rng=np.random.default_rng(seed+1)
    indices = [rng.choice(len(cal),16,replace=False) for _ in range(draws)] if cal_draw_indices is None else cal_draw_indices
    objective.calibrate(cal,[dict(cal_indices=np.asarray(ix,int)) for ix in indices])
    objective.reference_summary=summaries(fit,names)
    return objective


def score_simulation(run,observer,objective,burnin_ms=1500):
    obs=observe(run['envelope'],run['envelope_dt_ms'],observer)
    ids=np.asarray([i for i in obs['primary_event_indices'] if obs['events'][i]['window_ms'][0]>=burnin_ms],int)
    times=np.asarray(obs['centroid_ms']).reshape(-1,len(observer['contact_names']))
    score=objective.score(times[ids])
    if hasattr(objective,'reference_summary'):
        score['descriptive']=descriptive(times[ids],objective.reference_summary,objective.names)
    if run['physical_status']!='COMPLETE':score.update(status='RUNAWAY',J=None)
    run.update(observed=obs,primary_centroids_ms=times[ids],score=score)
    return score


def optimize_cores(substrate,contacts,observer,objective,*,k=2,budget=8,restart=0,seed=20261015):
    """Bounded, sequential Sobol + local refinement; every loss runs a new SNN.

    Equal proposal budget across K. This transparent notebook optimizer is not
    the historical delayed Matern-GP/qLogEI batch scheduler (copied separately).
    Missing events/runaway remain unscorable, with no artificial penalty value.
    """
    cfg=substrate['config'];r=cfg['radius_mm'];lo=r+.05;hi=substrate['params'].L-r-.05
    sobol=qmc.Sobol(2*k+2,scramble=True,seed=seed+100*k+restart)
    rng=np.random.default_rng(seed+restart);rows=[];best=None;best_run=None;worst_run=None;worst_j=-np.inf
    for epoch in range(budget):
        for attempt in range(1000):
            if best is not None and epoch>=budget//2 and attempt<50:
                vec=np.asarray(best['vector'])+rng.normal(0,[1.]*(2*k)+[.08,8.])
                centers=np.clip(vec[:2*k].reshape(k,2),lo,hi);outward=float(np.clip(vec[-2],.9,1.6));angle=float(np.clip(vec[-1],-60,60))
            else:
                z=sobol.random(1)[0];centers=(lo+(hi-lo)*z[:2*k]).reshape(k,2);outward=.9+.7*z[-2];angle=-60+120*z[-1]
            if all(np.linalg.norm(centers[i]-centers[j])>2*r+.05 for i in range(k) for j in range(i)):break
        else:raise RuntimeError('No nonoverlapping candidate')
        # Rotation is part of the graph, not a post-hoc rotation of a picture.
        angle = angle if cfg.get('optimize_angle',True) else 0.0
        candidate_sub=build_substrate(cfg,angle_offset=angle) if angle else substrate
        run=simulate(candidate_sub,centers,contacts,outward=outward,seed=cfg['training_seed'])
        score=score_simulation(run,observer,objective,burnin_ms=cfg['burnin_ms'])
        row=dict(epoch=epoch,k=k,restart=restart,vector=[*centers.ravel(),outward,angle],centroid_ms=run['primary_centroids_ms'].tolist(),**score)
        rows.append(row)
        print(f"K={k} restart={restart} epoch={epoch}: {score['status']} N={score['N']} J={score['J']}",flush=True)
        if score['J'] is not None and (best is None or score['J']<best['J']):best=row;best_run=run
        if score['J'] is not None and score['J']>worst_j:worst_j=score['J'];worst_run=run
    return dict(rows=rows,best=best,best_run=best_run,worst_run=worst_run)


def angle_response(config,centers,contacts,observer,objective,*,angles=(-60,0,60),outward=1.25):
    rows=[]
    for angle in angles:
        sub=build_substrate(config,angle_offset=float(angle));run=simulate(sub,centers,contacts,outward=outward,seed=config['training_seed'])
        score=score_simulation(run,observer,objective,config['burnin_ms'])
        rows.append(dict(angle=float(angle),centroid_ms=run['primary_centroids_ms'].tolist(),**score))
    return rows


def save_run(path,run):
    arrays={k:v for k,v in run.items() if isinstance(v,np.ndarray)}
    arrays.update({f'slow_{k}':v for k,v in run.get('slow',{}).items()})
    np.savez_compressed(path,**arrays)
