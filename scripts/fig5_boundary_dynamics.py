"""Nonlinear fixed-Z delayed dynamics on both sides of spectral boundaries."""
import sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fig5_z_gaba_phase_map as p
import numpy as np
from scipy import linalg
from concurrent.futures import ProcessPoolExecutor,as_completed
from src.topic4_patient_zm_meanfield import transfer_rates
from src.topic4_dual_core_spatial_z_delay import simulate_delayed_ou_trajectory

DATA=p.b.base.DATA/'boundary_nonlinear_dynamics'
FIG=p.b.base.FIG/'boundary_nonlinear_dynamics'
CASES=[(p.b.base.rec.IDS[0],4,[34.3,40.3]),(p.b.base.rec.IDS[1],0,[30.,34.5,42.])]


def lift_mode(f,x,tau_ms,freq):
    m,z,z2,eta,tm,ops=f.at(tau_ms/f.base.tau_gaba_ms);n=m.n_cells
    s=2j*np.pi*freq/1000;c=p.DelayCharacteristic(f,x)
    vals,vec=linalg.eig(c.matrix(s,tau_ms/f.base.tau_gaba_ms));v=vec[:,np.argmin(abs(vals))]
    v*=np.exp(-1j*np.angle(np.average(v[:n],weights=m.count_e)))
    mu=np.exp(s*ops.dt_ms);a=np.expm1(s*ops.dt_ms)/ops.dt_ms;re,ri=v[:n],v[n:]
    he=np.concatenate([re*mu**(-k) for k in range(1,ops.max_delay_steps+1)])
    hi=np.concatenate([ri*mu**(-k) for k in range(1,ops.max_delay_steps+1)])
    syn=[m.tau_mem_e_ms*(ops.w_ee_history@he)/(1+a*m.tau_ampa_ms),
         m.tau_mem_e_ms*(ops.w_ei_history@hi)/(1+a*m.tau_gaba_ms),
         m.tau_mem_i_ms*(ops.w_ie_history@he)/(1+a*m.tau_ampa_ms),
         m.tau_mem_i_ms*(ops.w_ii_history@hi)/(1+a*m.tau_gaba_ms)]
    full=np.concatenate([re,ri,*syn,he,hi,re/(a+1/tm)])
    residual=np.linalg.norm(p.corrected_delay_matrix(f,x,tau_ms/f.base.tau_gaba_ms)@full-mu*full)/np.linalg.norm(full)
    out=full.real;out*=.0001/np.sqrt(np.average(out[:n]**2,weights=m.count_e))
    return out,float(residual)


def prepare(cid,index,taus):
    out=DATA/cid;out.mkdir(parents=True,exist_ok=True)
    seed=p.read(p.DATA/cid/'seeds.json')['rows'][index]
    x=np.load(p.DATA/cid/'seeds.npz')['rates'][index];f=p.b.get_family(cid,'tau_gaba',True);f.z_anchor=seed['lambda']
    m,z,z2,eta,tm,ops=f.at(1.);n=m.n_cells;re,ri=x[:n],x[n:]
    syn=np.stack([m.tau_mem_e_ms*m.w_ee@re,m.tau_mem_e_ms*m.w_ei@ri,
                  m.tau_mem_i_ms*m.w_ie@re,m.tau_mem_i_ms*m.w_ii@ri])
    he=np.tile(re,(ops.max_delay_steps,1));hi=np.tile(ri,(ops.max_delay_steps,1));adapt=tm*re
    equilibrium=np.concatenate([re,ri,syn.ravel(),he.ravel(),hi.ravel(),adapt])
    row=p.read(p.DATA/cid/f'row_{index:03d}.json');cross=row['crossings'][0]
    specs=[(cross['tau_gaba_ms'],cross['frequency_hz'])]
    if cid==p.b.base.rec.IDS[1]:
        sec=next(r for r in p.read(p.DATA/cid/'secondary_boundary.json')['results'] if r['index']==index)['native']
        specs.append((sec['tau_gaba_ms'],sec['frequency_hz']))
    modes=[lift_mode(f,x,*spec) for spec in specs];kick=sum(v for v,r in modes)
    # Same complete perturbed state for every tau; retain positivity and rate caps.
    upper=np.concatenate([np.full(n,1/m.tau_ref_e_ms),np.full(n,1/m.tau_ref_i_ms),np.full(4*n,np.inf),
        np.full(ops.max_delay_steps*n,1/m.tau_ref_e_ms),np.full(ops.max_delay_steps*n,1/m.tau_ref_i_ms),np.full(n,np.inf)])
    # Near-silent cells can have equilibrium rates ~1e-30 and numerical mode
    # components ~1e-17. A global positivity rescale would erase the intended
    # perturbation everywhere. Project only the infeasible components instead.
    state=np.clip(equilibrium+kick,0.,upper)
    projection_error=state-equilibrium-kick
    rate_projection_relative=float(np.linalg.norm(projection_error[:2*n])/np.linalg.norm(kick[:2*n]))
    if rate_projection_relative>1e-3:raise RuntimeError('physical projection changed the intended rate perturbation')
    np.savez_compressed(out/'initial_state.npz',equilibrium=equilibrium,state=state,rates=x,z=z,z2=z2)
    protocol=dict(candidate_id=cid,index=index,z_loss_lambda=seed['lambda'],mean_z=float(np.average(z,weights=m.count_e)),
        equilibrium_e_hz=float(np.average(re,weights=m.count_e)*1000),tau_gaba_ms=taus,duration_ms=12000.,dt_ms=ops.dt_ms,
        tau_m_ms=tm,eta_m=eta,noise='No OU or external innovations; deterministic nonlinear reduction.',
        initial_condition='Identical complete perturbed state across tau values; small real oscillatory eigenmode perturbation(s) of critical equilibria, including rate, synapse, M and delay histories.',
        critical_modes=specs,mode_lift_residuals=[r for v,r in modes],physical_projection_rate_relative_error=rate_projection_relative,
        initial_e_rms_hz=float(1000*np.sqrt(np.average((state[:n]-re)**2,weights=m.count_e))),
        scope='Fixed spatial Z; dynamic M and synapses. This is the nonlinear system underlying the phase map, not a new 40000-neuron SNN run.')
    p.write(out/'protocol.json',protocol);return protocol


def integrate(cid,tau_ms,duration_ms=12000.,record_stride=10):
    out=DATA/cid;cfg=p.read(out/'protocol.json');a=np.load(out/'initial_state.npz')
    f=p.b.get_family(cid,'tau_gaba',True);f.z_anchor=cfg['z_loss_lambda']
    m,z,z2,eta,tm,ops=f.at(tau_ms/f.base.tau_gaba_ms);n=m.n_cells;delay=ops.max_delay_steps;dt=ops.dt_ms
    state=a['state'];x=a['rates'];re=state[:n].copy();ri=state[n:2*n].copy();syn=state[2*n:6*n].reshape(4,n).copy()
    he=state[6*n:6*n+delay*n].reshape(delay,n).copy();hi=state[6*n+delay*n:6*n+2*delay*n].reshape(delay,n).copy();adapt=state[-n:].copy()
    te,ti=m.tau_mem_e_ms,m.tau_mem_i_ms;ta,tg=m.tau_ampa_ms,m.tau_gaba_ms;times=np.array([ta,tg,ta,tg])[:,None]
    steps=int(round(duration_ms/dt));traces=[];cells=[];clipped=0;started=time.monotonic()
    def record(t):
        traces.append([t,1000*np.average(re,weights=m.count_e),1000*np.average(ri,weights=m.count_i),
            np.average(z*syn[1],weights=m.count_e),np.average(eta*adapt,weights=m.count_e),
            1000*np.sqrt(np.average((re-x[:n])**2,weights=m.count_e))])
        cells.append(1000*re.copy())
    record(0.)
    for step in range(steps):
        drive=np.stack([te*(ops.w_ee_history@he.ravel())/ta,te*(ops.w_ei_history@hi.ravel())/tg,
                        ti*(ops.w_ie_history@he.ravel())/ta,ti*(ops.w_ii_history@hi.ravel())/tg])
        next_syn=syn+dt*(drive-syn/times)
        mu_e=syn[0]-z*syn[1]-eta*adapt+te*m.j_ext_e_mv*m.nu_ext_per_ms
        mu_i=syn[2]-syn[3]+ti*m.j_ext_i_mv*m.nu_ext_per_ms
        var_e=te*(m.v_ee@re+z2*(m.v_ei@ri)+m.j_ext_e_mv**2*m.nu_ext_per_ms)
        var_i=ti*(m.v_ie@re+m.v_ii@ri+m.j_ext_i_mv**2*m.nu_ext_per_ms)
        pe,pi=transfer_rates(m,mu_e,np.sqrt(np.maximum(var_e,1e-12)),mu_i,np.sqrt(np.maximum(var_i,1e-12)))
        ne=re+dt*(-re+pe)/te;ni=ri+dt*(-ri+pi)/ti
        clipped+=int(np.any(ne<0) or np.any(ne>1/m.tau_ref_e_ms) or np.any(ni<0) or np.any(ni>1/m.tau_ref_i_ms))
        ne=np.clip(ne,0,1/m.tau_ref_e_ms);ni=np.clip(ni,0,1/m.tau_ref_i_ms)
        next_adapt=np.maximum(adapt+dt*(-adapt/tm+re),0.)
        he[1:]=he[:-1].copy();he[0]=re;hi[1:]=hi[:-1].copy();hi[0]=ri
        re,ri,syn,adapt=ne,ni,next_syn,next_adapt
        if (step+1)%record_stride==0:record((step+1)*dt)
    return dict(trace=np.array(traces),cells_e_hz=np.array(cells),final_rates=np.r_[re,ri],final_synapses=syn,
        final_m=adapt,final_he=he,final_hi=hi,clipped_steps=clipped,seconds=time.monotonic()-started)


def verify():
    records=[]
    for cid,index,taus in CASES:
        a=np.load(DATA/cid/'initial_state.npz');cfg=p.read(DATA/cid/'protocol.json');f=p.b.get_family(cid,'tau_gaba',True);f.z_anchor=cfg['z_loss_lambda']
        m,z,z2,eta,tm,ops=f.at(taus[0]/f.base.tau_gaba_ms);n=m.n_cells;d=ops.max_delay_steps;state=a['state']
        old=simulate_delayed_ou_trajectory(m,ops,state[:2*n],z_field=z,z_second_moment=z2,eta_m=eta,tau_m_slow_ms=tm,
            initial_synapses=state[2*n:6*n].reshape(4,n),initial_m=state[-n:],
            initial_history_e=state[6*n:6*n+d*n].reshape(d,n),initial_history_i=state[6*n+d*n:6*n+2*d*n].reshape(d,n),
            ou_rate_e=np.zeros((20,n)),tail_steps=20)
        new=integrate(cid,taus[0],duration_ms=2.,record_stride=1)
        err=max(np.max(abs(new['final_rates']-old['final_rates'])),np.max(abs(new['final_synapses']-old['final_synapses'])),np.max(abs(new['final_m']-old['final_adaptation_state'])))
        if err>1e-10:raise RuntimeError('nonlinear integration differs from accepted delay engine')
        records.append(dict(candidate_id=cid,steps=20,max_state_difference=float(err)))
    p.write(DATA/'nonlinear_engine_check.json',records)


def run_one(cid,tau):
    dest=DATA/cid/f'tau_{tau:.1f}'
    if Path(str(dest)+'.json').exists():return
    r=integrate(cid,tau);meta={k:r.pop(k) for k in ['clipped_steps','seconds']}
    # Decimal-containing names require appending, not Path.with_suffix.
    np.savez_compressed(str(dest)+'.npz',**r)
    trace=r['trace'];tail=trace[:,0]>=10000
    meta.update(tau_gaba_ms=tau,finite=bool(np.all(np.isfinite(trace))),
        tail_e_mean_hz=float(np.mean(trace[tail,1])),tail_e_peak_to_peak_hz=float(np.ptp(trace[tail,1])),
        initial_e_rms_hz=float(trace[0,5]),final_e_rms_hz=float(trace[-1,5]),
        minimum_e_hz=float(np.min(trace[:,1])),maximum_e_hz=float(np.max(trace[:,1])))
    p.write(Path(str(dest)+'.json'),meta);print(cid,tau,meta,flush=True)


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    FIG.mkdir(parents=True,exist_ok=True)
    for bi,(cid,index,taus) in enumerate(CASES):
        cfg=p.read(DATA/cid/'protocol.json');a=np.load(DATA/cid/'initial_state.npz');f=p.b.get_family(cid,'tau_gaba',True);f.z_anchor=cfg['z_loss_lambda']
        m,z,z2,eta,tm,ops=f.at(1.);n=m.n_cells;eq=a['equilibrium'];ieq=np.average(z*eq[3*n:4*n],weights=m.count_e);meq=np.average(eta*eq[-n:],weights=m.count_e)
        fig,axes=plt.subplots(4,len(taus),figsize=(4.2*len(taus),10),sharex='col',layout='constrained')
        ranges=[[] for _ in range(4)]
        for col,tau in enumerate(taus):
            r=np.load(DATA/cid/f'tau_{tau:.1f}.npz');v=r['trace'];t=v[:,0]/1000
            axes[0,col].plot(t,v[:,1],color='#383838',lw=.65);axes[0,col].axhline(cfg['equilibrium_e_hz'],color='.6',ls='--',lw=.7)
            axes[0,col].set_title(f'GABA decay = {tau:g} ms');axes[0,col].set_ylabel('Mean E rate (Hz)')
            axes[1,col].plot(t,v[:,3]-ieq,color='#267d8e',lw=.7);axes[1,col].set_ylabel('Effective inhibitory input\nΔ(z Iᵢ) (mV)')
            axes[2,col].plot(t,v[:,4]-meq,color='#b05d21',lw=.7);axes[2,col].set_ylabel('M adaptation input\nΔ(ηₘ M) (mV)')
            axes[3,col].semilogy(t,np.maximum(v[:,5],1e-12),color='#803781',lw=1);axes[3,col].set_ylabel('Spatial E-rate deviation\nRMS (Hz)');axes[3,col].set_xlabel('Time after perturbation (s)')
            for j,values in enumerate([v[:,1],v[:,3]-ieq,v[:,4]-meq,np.maximum(v[:,5],1e-12)]):ranges[j].extend([float(np.min(values)),float(np.max(values))])
            for ax in axes[:,col]:ax.set_xlim(0,12);ax.ticklabel_format(axis='x',useOffset=False)
        for j in range(4):
            lo,hi=min(ranges[j]),max(ranges[j])
            if j==3:bounds=(max(lo/2,1e-12),hi*2)
            else:
                pad=max((hi-lo)*.06,1e-8);bounds=(lo-pad,hi+pad)
            for ax in axes[j]:
                ax.set_ylim(*bounds)
                if j<3:ax.ticklabel_format(axis='y',useOffset=False)
        fig.suptitle(f'Base {bi+1}: nonlinear dynamics across oscillatory boundaries\nFixed Z: λ = {cfg["z_loss_lambda"]:.4f}, mean Z = {cfg["mean_z"]:.4f} · identical small initial perturbation · no noise',fontsize=13)
        stem=FIG/f'fig5-boundary-dynamics-base{bi+1}';fig.savefig(str(stem)+'.png',dpi=180);fig.savefig(str(stem)+'.pdf');plt.close(fig)
    (FIG/'README.md').write_text('\n'.join(f'### fig5-boundary-dynamics-base{i+1}.png\n固定同一空间 Z 场和高活动平衡态，在 GABA 时间常数边界两侧从完全相同的微小初态扰动开始，直接积分非线性降阶延迟系统。四行依次展示群体 E 放电率、有效抑制电流变化、M 适应电流变化和空间放电率偏离平衡态的 RMS；未施加 OU 噪声，Z 不随时间演化。\n**关注点**：这是相图对应的高活动态局部动力学检验，并非原始 40000 神经元 SNN 的间期到 runaway 重演；有限时长振荡不自动证明稳定极限环。\n' for i in range(2)))


if __name__=='__main__':
    DATA.mkdir(parents=True,exist_ok=True)
    for case in CASES:prepare(*case)
    verify()
    jobs=[(cid,tau) for cid,index,taus in CASES for tau in taus]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(run_one,*j) for j in jobs])):
            f.result();p.write(DATA/'status.json',dict(stage='NONLINEAR_BOUNDARY_SIMULATIONS',completed=i+1,total=len(jobs)))
    render();p.write(DATA/'status.json',dict(stage='COMPLETE_PENDING_VISUAL_REVIEW',completed=len(jobs),total=len(jobs)))
