"""Two-parameter high-equilibrium spectral map, without attractor attribution.

Continue the independently checked baseline oscillatory mode in spatial-Z
loss and GABA kinetics. Negative growth of one mode is not full stability.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fig5_near_runaway_branches as b
from src.topic4_fig5_delay_characteristic import DelayCharacteristic
from src.topic4_fig5_local_bifurcation import corrected_delay_matrix
from src.topic4_xy_fig5_followup import read, write, sha
import numpy as np
from scipy.sparse.linalg import eigs, ArpackNoConvergence
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse, time

DATA = b.base.DATA / 'z_gaba_phase_map'
FIG = b.base.FIG / 'z_gaba_phase_map'
TAUS = np.unique(np.r_[np.linspace(12, 60, 17), 18., 24.])


def advance(c, start, target, step=.04):
    """Adaptive small-step continuation; never replace a failed mode silently."""
    p = start['parameter']; g = start['growth_per_ms']; w = start['frequency_hz']
    history = []
    while abs(target-p) > 1e-10:
        h = np.sign(target-p)*min(step, abs(target-p))
        for trial in range(7):
            r = c.follow_mode(p+h, w, initial_growth_per_ms=g)
            if (r['converged'] and abs(r['growth_per_ms']-g) < .002
                    and abs(r['frequency_hz']-w) < 1.5 and r['frequency_hz'] > 1):
                break
            h /= 2
        else:
            return None, history
        history.append(r); p = r['parameter']; g = r['growth_per_ms']; w = r['frequency_hz']
    return dict(parameter=p, growth_per_ms=g, frequency_hz=w, converged=True), history


def prepare(cid):
    out = DATA/cid; out.mkdir(parents=True, exist_ok=True)
    if (out/'seeds.json').exists(): return read(out/'seeds.json')
    f = b.get_family(cid, 'tau_gaba', True)
    anchor = read(b.DATA/cid/'anchor.json'); p0 = anchor['lambda_anchor']
    fold = anchor['fold']['parameter']
    upper = min(2., read(b.base.DATA/cid/'prepared.json')['z_loss_upper'])
    targets = np.unique(np.r_[fold+.003, fold+.01, p0, np.linspace(p0, upper, 11)])
    initial = np.load(b.DATA/cid/'anchor_roots.npz')['rates'][-1]
    initial_mode = read(b.DATA/cid/'stability/tau_gaba_anchor_1.0.json')['modes'][0]
    rows = []; rates = []; failures = []
    for direction in [-1, 1]:
        x = initial.copy(); p = p0; mode = dict(initial_mode)
        for target in sorted([q for q in targets if (q-p0)*direction >= -1e-10], reverse=direction<0):
            while abs(target-p) > 1e-10:
                step = np.sign(target-p)*min(.008, abs(target-p))
                for trial in range(8):
                    f.z_anchor = p+step; xx, ok, err = f.solve(1., x)
                    if ok and np.max(abs(xx-x)) < .035:
                        c = DelayCharacteristic(f, xx)
                        mm = c.follow_mode(1., mode['frequency_hz'], initial_growth_per_ms=mode['growth_per_ms'])
                        if (mm['converged'] and abs(mm['growth_per_ms']-mode['growth_per_ms']) < .002
                                and abs(mm['frequency_hz']-mode['frequency_hz']) < 1.5):
                            break
                    step /= 2
                else:
                    failures.append(dict(target=float(target), stopped_lambda=float(p), reason='ROOT_OR_MODE_CONTINUATION_FAILED'))
                    break
                x=xx; p+=step; mode=mm
            if abs(p-target)>1e-9: break
            if any(abs(r['lambda']-p)<1e-9 for r in rows): continue
            f.z_anchor=p; m,z,*_=f.at(1.)
            rows.append(dict(index=len(rows), **{'lambda':float(p)},
                mean_z=float(np.average(z,weights=m.count_e)),
                equilibrium_hz=float(np.average(x[:m.n_cells],weights=m.count_e)*1000),
                mode=mode, equilibrium_residual=float(np.max(abs(f.f(x,1.))))))
            rates.append(x.copy())
            print('seed',cid,p,rows[-1]['equilibrium_hz'],flush=True)
    np.savez_compressed(out/'seeds.npz',rates=rates)
    result=dict(candidate_id=cid,rows=rows,failures=failures,fold=anchor['fold'],
        tau_baseline_ms=f.base.tau_gaba_ms, requested_lambda=targets.tolist())
    write(out/'seeds.json',result);return result


def scan_row(cid, index):
    out=DATA/cid; dest=out/f'row_{index:03d}.json'
    if dest.exists(): return read(dest)
    seed=read(out/'seeds.json')['rows'][index]
    x=np.load(out/'seeds.npz')['rates'][index]
    f=b.get_family(cid,'tau_gaba',True); f.z_anchor=seed['lambda']
    c=DelayCharacteristic(f,x); native=f.base.tau_gaba_ms
    start=dict(seed['mode'],parameter=1.)
    points=[dict(start,tau_gaba_ms=native)]; failures=[]; history=[]
    for direction in [-1,1]:
        last=start.copy()
        for tau in sorted([t for t in TAUS if (t-native)*direction>1e-9],reverse=direction<0):
            r,hist=advance(c,last,float(tau/native));history.extend(hist)
            if r is None:
                failures.append(dict(tau_target=float(tau),reason='MODE_TRACKING_FAILED'));break
            r['tau_gaba_ms']=float(tau);points.append(r);last=r
    points.sort(key=lambda r:r['tau_gaba_ms']);crossings=[]
    for a,q in zip(points[:-1],points[1:]):
        if a['growth_per_ms']*q['growth_per_ms']>=0:continue
        r=c.crossing((a['frequency_hz']+q['frequency_hz'])/2,(a['parameter']+q['parameter'])/2)
        valid=r['converged'] and a['parameter']<=r['scale']<=q['parameter']
        if valid:
            check,h=advance(c,a,r['scale'],.01)
            valid=check is not None and abs(check['growth_per_ms'])<1e-6 and abs(check['frequency_hz']-r['frequency_hz'])<.02
        r['tracked_mode_verified']=bool(valid);r['tau_gaba_ms']=r['scale']*native
        if valid:
            left=c.follow_mode(r['scale']-.005,r['frequency_hz'])
            right=c.follow_mode(r['scale']+.005,r['frequency_hz'])
            r['growth_slope_per_ms_per_scale']=(right['growth_per_ms']-left['growth_per_ms'])/.01
            r['transverse']=bool(left['converged'] and right['converged'] and left['growth_per_ms']*right['growth_per_ms']<0)
        crossings.append(r)
    result=dict(candidate_id=cid,seed=seed,points=points,crossings=crossings,failures=failures,
        continuation_steps=len(history),max_characteristic_residual=max([r.get('residual',0.) for r in history],default=0.))
    write(dest,result); print('row',cid,index,'crossings',[r['tau_gaba_ms'] for r in crossings],flush=True)
    return result


def spectrum(cid, index, tau):
    out=DATA/cid;dest=out/f'spectrum_{index:03d}_{tau:.4f}.json'
    if dest.exists():return read(dest)
    seed=read(out/'seeds.json')['rows'][index]
    f=b.get_family(cid,'tau_gaba',True);f.z_anchor=seed['lambda']
    x=np.load(out/'seeds.npz')['rates'][index];p=tau/f.base.tau_gaba_ms
    mat=corrected_delay_matrix(f,x,p);t=time.monotonic()
    try:
        vals,vec=eigs(mat,k=6,ncv=48,which='LM',tol=2e-8,maxiter=2500,
            v0=np.random.default_rng(1831).normal(size=mat.shape[0]));ok=True
    except ArpackNoConvergence as e:vals,vec=e.eigenvalues,e.eigenvectors;ok=False
    modes=[]
    for v,q in zip(vals,vec.T):
        modes.append(dict(growth_per_ms=float(np.log(abs(v))/.1),
            frequency_hz=float(abs(np.angle(v))*1000/(2*np.pi*.1)),
            residual=float(np.linalg.norm(mat@q-v*q)/np.linalg.norm(q))))
    modes.sort(key=lambda r:r['growth_per_ms'],reverse=True)
    result=dict(candidate_id=cid,index=index,**{'lambda':seed['lambda']},tau_gaba_ms=tau,
        converged=ok,modes=modes,seconds=time.monotonic()-t)
    write(dest,result);print('spectrum',cid,index,tau,ok,modes[:1],flush=True);return result


def observed_comparison(cid):
    d=read(b.base.rec.DATA/cid/'trajectory.json')
    a=np.load(b.base.rec.DATA/cid/'trajectory.npz');z=np.load(b.base.DATA/cid/'z_fields.npz')
    early=z['early'];delta=z['pre']-early;reg=a['region_E'];counts=np.bincount(reg)
    re=np.bincount(reg,weights=early)/counts;rd=np.bincount(reg,weights=delta)/counts
    rows=[]
    for label,t in [('early',1000.),('pre',d['states_ms']['pre_onset']),
                    ('first_regional',d['trajectory']['earliest_regional_recruitment_ms']),
                    ('population_onset',d['trajectory']['onset_ms'])]:
        i=np.argmin(abs(a['slow_time_ms']-t));zr=a['region_z'][i]
        mean=np.average(zr,weights=counts);lam=(mean-early.mean())/delta.mean()
        err=np.sqrt(np.average((zr-re-lam*rd)**2,weights=counts))
        loss=np.sqrt(np.average((zr-re)**2,weights=counts))
        rows.append(dict(label=label,time_ms=float(a['slow_time_ms'][i]),mean_z=float(mean),
            mean_equivalent_lambda=float(lam),three_region_shape_rmse=float(err),
            relative_shape_rmse=float(err/max(loss,1e-12))))
    result=dict(rows=rows,note='Mean-equivalent coordinate is not the full neuronal Z field. Three-region error only diagnoses coarse shape mismatch.')
    write(DATA/cid/'observed_z_comparison.json',result);return result


def continuous_crossing(cid,index):
    out=DATA/cid;dest=out/f'continuous_{index:03d}.json'
    if dest.exists():return read(dest)
    row=read(out/f'row_{index:03d}.json')
    f=b.get_family(cid,'tau_gaba',True);f.z_anchor=row['seed']['lambda']
    x=np.load(out/'seeds.npz')['rates'][index];c=DelayCharacteristic(f,x);results=[]
    for native in row['crossings']:
        if not native['tracked_mode_verified'] or not native.get('transverse'):continue
        previous=native;chain=[]
        for dt in [.09,.08,.07,.06,.05,.04,.03,.025,.02,.01,0.]:
            r=c.crossing(previous['frequency_hz'],previous['scale'],dt)
            if not r['converged'] or abs(r['frequency_hz']-previous['frequency_hz'])>1 or abs(r['scale']-previous['scale'])>.2:break
            r['tau_gaba_ms']=r['scale']*f.base.tau_gaba_ms;chain.append(r);previous=r
        results.append(dict(native=native,chain=chain,continuous_verified=bool(chain and chain[-1]['dt_ms']==0.)))
    result=dict(index=index,results=results);write(dest,result);return result


def refine():
    """Refine the observed nonmonotonic crossing-onset interval in base 1."""
    cid=b.base.rec.IDS[0];out=DATA/cid;s=read(out/'seeds.json')
    xs=list(np.load(out/'seeds.npz')['rates']);new=[];f=b.get_family(cid,'tau_gaba',True)
    targets=np.linspace(s['rows'][3]['lambda'],s['rows'][4]['lambda'],9)[1:-1]
    for target in targets:
        existing=next((r for r in s['rows'] if abs(r['lambda']-target)<1e-9),None)
        if existing:new.append(existing['index']);continue
        nearest=min(s['rows'],key=lambda r:abs(r['lambda']-target))
        x=xs[nearest['index']].copy();dose=nearest['lambda'];mode=nearest['mode']
        while abs(target-dose)>1e-10:
            dose+=np.sign(target-dose)*min(.004,abs(target-dose));f.z_anchor=dose
            x,ok,err=f.solve(1.,x)
            if not ok:raise RuntimeError('refinement root lost')
            c=DelayCharacteristic(f,x)
            r=c.follow_mode(1.,mode['frequency_hz'],initial_growth_per_ms=mode['growth_per_ms'])
            if not r['converged'] or abs(r['frequency_hz']-mode['frequency_hz'])>1:raise RuntimeError('refinement mode lost')
            mode=r
        m,z,*_=f.at(1.);idx=len(xs);xs.append(x.copy());new.append(idx)
        s['rows'].append(dict(index=idx,**{'lambda':float(target)},mean_z=float(np.average(z,weights=m.count_e)),
            equilibrium_hz=float(np.average(x[:m.n_cells],weights=m.count_e)*1000),mode=mode,equilibrium_residual=float(err)))
    s['adaptive_refinement']='Seven extra Z rows where the oscillatory zero-crossing first appears; checks nonmonotonic GABA response.'
    np.savez_compressed(out/'seeds.npz',rates=xs);write(out/'seeds.json',s)
    with ProcessPoolExecutor(max_workers=4) as pool:
        for ft in as_completed([pool.submit(scan_row,cid,i) for i in new]):ft.result()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for ft in as_completed([pool.submit(continuous_crossing,cid,i) for i in new]):ft.result()


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(16,8.3),width_ratios=[1.4,1.1,1.],layout='constrained')
    summaries=[]
    for ci,cid in enumerate(b.base.rec.IDS):
        s=read(DATA/cid/'seeds.json'); rows=[read(p) for p in sorted((DATA/cid).glob('row_*.json'))]
        rows.sort(key=lambda r:r['seed']['lambda'])
        lam=np.array([r['seed']['lambda'] for r in rows]);taus=TAUS
        growth=np.full((len(taus),len(lam)),np.nan)
        for j,r in enumerate(rows):
            for q in r['points']:
                ix=np.argmin(abs(taus-q['tau_gaba_ms']))
                if abs(taus[ix]-q['tau_gaba_ms'])<1e-8:growth[ix,j]=q['growth_per_ms']*1000
        ax=axes[ci,0]
        mesh=ax.pcolormesh(lam,taus,growth,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-15,vcenter=0,vmax=30),shading='nearest',rasterized=True)
        fold=s['fold']['parameter'];ax.axvspan(0,fold,color='.93',zorder=-2)
        ax.axvline(fold,color='#a43578',lw=1.4,ls='--')
        crossings=[(r['seed']['lambda'],c) for r in rows for c in r['crossings'] if c['tracked_mode_verified'] and c.get('transverse')]
        for sign in [-1,1]:
            part=[(p,c) for p,c in crossings if c['growth_slope_per_ms_per_scale']*sign>0]
            if part:ax.plot([p for p,c in part],[c['tau_gaba_ms'] for p,c in part],'-o',color='#76266c',ms=3,lw=1.5)
        continuous=[]
        for r in rows:
            path=DATA/cid/f'continuous_{r["seed"]["index"]:03d}.json'
            if path.exists():
                for q in read(path)['results']:
                    if q['continuous_verified']:continuous.append((r['seed']['lambda'],dict(q['chain'][-1],growth_slope_per_ms_per_scale=q['native']['growth_slope_per_ms_per_scale'])))
        for sign in [-1,1]:
            part=[(p,c) for p,c in continuous if c['growth_slope_per_ms_per_scale']*sign>0]
            if part:ax.plot([p for p,c in part],[c['tau_gaba_ms'] for p,c in part],'--',color='#27485f',lw=1.3)
        secondary_path=DATA/cid/'secondary_boundary.json'
        secondary=sorted(read(secondary_path)['results'],key=lambda r:r['lambda']) if secondary_path.exists() else []
        if secondary:
            ax.plot([r['lambda'] for r in secondary],[r['native']['tau_gaba_ms'] for r in secondary],'-o',color='#c06b13',ms=3,lw=1.5)
            cont=[r for r in secondary if r['continuous_verified']]
            if cont:ax.plot([r['lambda'] for r in cont],[r['dt_chain'][-1]['tau_gaba_ms'] for r in cont],'--',color='#c06b13',lw=1.3)
        tertiary_path=DATA/cid/'tertiary_boundary.json'
        tertiary=sorted(read(tertiary_path)['results'],key=lambda r:r['lambda']) if tertiary_path.exists() else []
        if tertiary:
            ax.plot([r['lambda'] for r in tertiary],[r['native']['tau_gaba_ms'] for r in tertiary],'-o',color='#c06b13',ms=3,lw=1.5)
            cont=[r for r in tertiary if r['continuous_verified']]
            if cont:ax.plot([r['lambda'] for r in cont],[r['dt_chain'][-1]['tau_gaba_ms'] for r in cont],'--',color='#c06b13',lw=1.3)
            if read(tertiary_path).get('status')=='CONTINUATION_INCOMPLETE':
                end=tertiary[0];ax.scatter(end['lambda'],end['native']['tau_gaba_ms'],marker='x',s=50,color='black',zorder=9)
        ipath=DATA/cid/'two_mode_intersection.json'
        if ipath.exists():
            it=read(ipath);ax.scatter(it['lambda'],it['tau_gaba_ms'],marker='D',s=28,color='black',zorder=9)
        ax.axhline(s['tau_baseline_ms'],color='black',ls=':',lw=1)
        ax.text(.98,.025,f'Baseline GABA: {s["tau_baseline_ms"]:.0f} ms\nNegative color ≠ full stability',transform=ax.transAxes,va='bottom',ha='right',fontsize=9,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
        for p in (DATA/cid).glob('spectrum_*.json'):
            q=read(p)
            if q['converged'] and q['modes']:
                color='#067481' if q['modes'][0]['growth_per_ms']<0 else '#c3551c'
                ax.scatter(q['lambda'],q['tau_gaba_ms'],s=48,marker='s',color=color,edgecolor='white',lw=.7,zorder=8)
        ax.set(xlim=(fold-.025,2.03),ylim=(11.5,61),xlabel='Spatial Z-loss coordinate λ',ylabel='GABA decay time (ms)',title=f'Base {ci+1} · high-branch spectral map')
        fig.colorbar(mesh,ax=ax,label='Growth (s⁻¹)',shrink=.8)
        ax=axes[ci,1]
        loc=read(b.base.DATA/cid/'z_loss.json');arrays=np.load(loc['arrays']['path']);m=b.base.family(cid,'z_loss').base
        for branch in loc['branches']:
            sl=slice(branch['start'],branch['stop'])
            rates=np.average(arrays['rates'][sl,:m.n_cells],axis=1,weights=m.count_e)*1000
            ax.plot(arrays['parameter'][sl],rates,color='.7',lw=.8)
        ax.plot(lam,[r['seed']['equilibrium_hz'] for r in rows],color='#76266c',lw=2,label='Mapped high branch')
        ax.axvline(fold,color='#a43578',ls='--',lw=1)
        obs=observed_comparison(cid);onset=next(q for q in obs['rows'] if q['label']=='population_onset')
        ax.axvline(onset['mean_equivalent_lambda'],color='#b77716',ls=':',lw=1.5)
        ax.axvspan(0,1,color='#cae1df',alpha=.35)
        ax.text(.03,.96,f'SNN onset: λ(mean) ≈ {onset["mean_equivalent_lambda"]:.3f}\nDotted line: mean-only projection',transform=ax.transAxes,va='top',fontsize=9)
        ax.text(.5,100,'Early → pre-onset\nλ = 0 → 1',ha='center',fontsize=9)
        ax.set(xlim=(0,2.03),ylim=(-12,510),xlabel='Spatial Z-loss coordinate λ',ylabel='Equilibrium E rate (Hz)',title='Branch context and observed onset')
        ax=axes[ci,2]
        if crossings:
            for sign in [-1,1]:
                part=[(p,c) for p,c in crossings if c['growth_slope_per_ms_per_scale']*sign>0]
                if part:ax.plot([p for p,c in part],[c['frequency_hz'] for p,c in part],'-o',color='#76266c',ms=4)
                part=[(p,c) for p,c in continuous if c['growth_slope_per_ms_per_scale']*sign>0]
                if part:ax.plot([p for p,c in part],[c['frequency_hz'] for p,c in part],'--',color='#27485f',lw=1.3)
            if secondary:
                ax.plot([r['lambda'] for r in secondary],[r['native']['frequency_hz'] for r in secondary],'-o',color='#c06b13',ms=3,lw=1.5)
                cont=[r for r in secondary if r['continuous_verified']]
                if cont:ax.plot([r['lambda'] for r in cont],[r['dt_chain'][-1]['frequency_hz'] for r in cont],'--',color='#c06b13',lw=1.3)
            if tertiary:
                ax.plot([r['lambda'] for r in tertiary],[r['native']['frequency_hz'] for r in tertiary],'-o',color='#c06b13',ms=3,lw=1.5)
                cont=[r for r in tertiary if r['continuous_verified']]
                if cont:ax.plot([r['lambda'] for r in cont],[r['dt_chain'][-1]['frequency_hz'] for r in cont],'--',color='#c06b13',lw=1.3)
        else:
            ax.text(.5,.55,'No zero crossing found\non this tracked branch\nwithin 12–60 ms',transform=ax.transAxes,ha='center')
        ax.set(xlabel='Spatial Z-loss coordinate λ',ylabel='Frequency at crossing (Hz)',title='Hopf-candidate boundary frequency')
        ax.spines[['top','right']].set_visible(False)
        summaries.append(dict(candidate_id=cid,rows=len(rows),crossings=[dict(**{'lambda':p},**c) for p,c in crossings],fold=s['fold'],failures=s['failures']+[q for r in rows for q in r['failures']]))
    fig.suptitle('Spatial inhibition loss × inhibitory synaptic kinetics\nHigh-equilibrium spectral map · color: reference mode only · fixed Z',fontsize=15)
    fig.legend(handles=[Line2D([],[],color='#76266c',marker='o',label='Reference-mode crossing: dt = 0.1 ms'),Line2D([],[],color='#c06b13',marker='o',label='Additional oscillatory boundary'),Line2D([],[],color='#27485f',ls='--',label='Continuous-time limit'),Line2D([],[],color='#a43578',ls='--',label='Equilibrium fold'),Line2D([],[],marker='s',ls='',color='#067481',label='Full-matrix sampled modes decay'),Line2D([],[],marker='s',ls='',color='#c3551c',label='Full matrix: growing mode detected')],loc='outside lower center',ncol=3,frameon=False)
    FIG.mkdir(parents=True,exist_ok=True);stem=FIG/'fig5-z-gaba-spectral-phase-map'
    fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
    write(DATA/'summary.json',dict(bases=summaries,interpretation='Color is ONE continuously tracked oscillatory mode, not the spectral abscissa. Negative color alone is not full stability. Fold bounds this continued high branch, not all possible high roots. Observed onset is projected by mean Z only.',author_visual_acceptance=False))
    (FIG/'README.md').write_text('### fig5-z-gaba-spectral-phase-map.png\n两套基底的空间 Z 损失与 GABA 衰减时间二维谱图；紫线为参考振荡模态过零线，橙线为额外振荡边界，黑色菱形标记两个不同振荡零模曲线的交点。颜色只表示参考模态的增长率，方块表示完整矩阵的抽样特征对；中列对照平衡分支与实际 onset 的平均 Z 投影，右列给出边界频率，黑色叉号表示延续失败的最后保存点。\n**关注点**：蓝色与青色方块均不认证全系统稳定；黑叉末端不是生物学边界，未识别周期轨道及 Hopf 非线性类型，也不能将高态 Hopf 候选直接称为 runaway 起始分岔。\n')


def run():
    DATA.mkdir(parents=True,exist_ok=True)
    write(DATA/'protocol.json',dict(tau_gaba_ms=TAUS.tolist(),lambda_domain=[0,2],
        z_path='Per-neuron early + lambda*(pre-early), with preserved cellwise second moments.',
        branch='Upper root continued from independently checked anchor; other roots not exhaustively mapped.',
        scope='Two-parameter spectral continuation, not full attractor phase diagram.',
        code={str(Path(__file__)):sha(Path(__file__))}))
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(prepare,b.base.rec.IDS))
    jobs=[(cid,r['index']) for cid in b.base.rec.IDS for r in read(DATA/cid/'seeds.json')['rows']]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(scan_row,*j) for j in jobs])):
            f.result();write(DATA/'status.json',dict(stage='SPECTRAL_ROWS',completed=i+1,total=len(jobs)))
    render()
    write(DATA/'status.json',dict(stage='ROWS_COMPLETE_SPECTRAL_VALIDATION_PENDING',completed=len(jobs),total=len(jobs)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','render','spectrum','refine']);p.add_argument('--cid');p.add_argument('--index',type=int);p.add_argument('--tau',type=float);a=p.parse_args()
    if a.mode=='run':run()
    elif a.mode=='render':render()
    elif a.mode=='refine':refine();render()
    else:spectrum(a.cid,a.index,a.tau)
