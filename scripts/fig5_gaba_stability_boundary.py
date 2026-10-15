"""Locate the oscillatory stability crossing and check numerical dt dependence."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fig5_near_runaway_branches as base
from src.topic4_fig5_delay_characteristic import DelayCharacteristic
from src.topic4_xy_fig5_followup import read,write,sha
import numpy as np
from scipy import linalg
from concurrent.futures import ProcessPoolExecutor


def analyze(cid):
    out=base.DATA/cid
    if (out/'gaba_oscillatory_boundary.json').exists():
        saved=read(out/'gaba_oscillatory_boundary.json')
        if saved.get('discovery') and saved.get('tracking'):return saved
    f=base.get_family(cid,'tau_gaba',True)
    x=np.load(out/'anchor_roots.npz')['rates'][-1]
    c=DelayCharacteristic(f,x)
    previous=read(out/'stability/tau_gaba_anchor_1.0.json')['modes'][0]
    value=c.small_eigenvalue(previous['growth_per_ms']+2j*np.pi*previous['frequency_hz']/1000,1.)
    if abs(value)>1e-6:raise RuntimeError('condensed characteristic disagrees with full delay map')
    discovery=[];p=1.;growth=previous['growth_per_ms'];freq=previous['frequency_hz']
    prev=None
    while p<1.5-1e-10 and growth>0:
        step=min(.001 if prev is None else .005,1.5-p)
        g0,f0=growth,freq
        if prev is not None:
            g0+=step*(growth-prev[1])/(p-prev[0]);f0+=step*(freq-prev[2])/(p-prev[0])
        r=c.follow_mode(p+step,f0,.1,initial_growth_per_ms=g0)
        if not r['converged'] or abs(r['growth_per_ms']-growth)>.002 or abs(r['frequency_hz']-freq)>1:
            raise RuntimeError('native leading-mode discovery lost continuation')
        prev=(p,growth,freq);p+=step;growth=r['growth_per_ms'];freq=r['frequency_hz'];discovery.append(r)
    if growth>=0:
        curve=[dict(parameter=1.,growth_per_ms=previous['growth_per_ms'],frequency_hz=previous['frequency_hz'],converged=True)]+discovery
        for r in curve:r['tau_gaba_ms']=r['parameter']*f.base.tau_gaba_ms
        result=dict(candidate_id=cid,crossings=[],mode_curve=curve,discovery=discovery,
            tracking=discovery,search_status='NO_CROSSING_FROM_BASELINE_TO_1.5',
            high_equilibrium_hz=float(np.average(x[:f.base.n_cells],weights=f.base.count_e)*1000),
            full_delay_map_crosscheck_residual=abs(value),anchor=read(out/'anchor.json'),
            interpretation='Tracked growing mode persists through 1.5x GABA decay. A subdominant mode crossing must not be substituted for stabilization.',
            code_sources={str(p):sha(p) for p in [Path(__file__),base.ROOT/'src/topic4_fig5_delay_characteristic.py']})
        write(out/'gaba_oscillatory_boundary.json',result);return result
    crossings=[];scale=p
    # Small dt steps prevent jumping between nearby oscillatory mode families.
    for dt in [.1,.09,.08,.07,.06,.05,.04,.03,.025,.02,.01,0.]:
        r=c.crossing(freq,scale,dt)
        if not r['converged']:raise RuntimeError('crossing root failed')
        r['tau_gaba_ms']=r['scale']*f.base.tau_gaba_ms
        delta=.005
        r['below']=c.follow_mode(r['scale']-delta,r['frequency_hz'],dt)
        r['above']=c.follow_mode(r['scale']+delta,r['frequency_hz'],dt)
        if not all(r[q]['converged'] for q in ['below','above']):raise RuntimeError('mode tracking failed')
        r['growth_slope_per_ms_per_scale']=(r['above']['growth_per_ms']-r['below']['growth_per_ms'])/(2*delta)
        r['crosses']=r['below']['growth_per_ms']*r['above']['growth_per_ms']<0
        matrix=c.matrix(2j*np.pi*r['frequency_hz']/1000,r['scale'],dt)
        singular=linalg.svdvals(matrix)
        r['two_smallest_characteristic_singular_values']=singular[-2:].tolist()
        if dt in [.1,.05,.025,0.]:crossings.append(r)
        freq=r['frequency_hz'];scale=r['scale']
    native=crossings[0]
    scales=np.unique(np.r_[np.linspace(max(.5,native['scale']-.4),native['scale']+.3,17),1.,native['scale']])
    curve=[];tracking=[]
    # Start from the independently computed leading full-map mode at baseline.
    # Reinitializing every solve at growth=0 can jump to a different damped mode.
    for direction in (-1,1):
        targets=sorted([float(s) for s in scales if (s-1)*direction>=-1e-12],reverse=direction<0)
        p=1.;growth=previous['growth_per_ms'];frequency=previous['frequency_hz'];last=None
        for target in targets:
            while abs(target-p)>1e-10:
                step=np.sign(target-p)*min(.005,abs(target-p))
                gp,fp=growth,frequency
                if last is not None:
                    gp+=step*(growth-last[1])/(p-last[0]);fp+=step*(frequency-last[2])/(p-last[0])
                r=c.follow_mode(p+step,fp,.1,initial_growth_per_ms=gp)
                if not r['converged'] or abs(r['growth_per_ms']-growth)>.002 or abs(r['frequency_hz']-frequency)>1:
                    raise RuntimeError('oscillatory mode continuation lost identity')
                last=(p,growth,frequency);p+=step;growth=r['growth_per_ms'];frequency=r['frequency_hz'];tracking.append(r)
            r=dict(parameter=target,growth_per_ms=growth,frequency_hz=frequency,converged=True,
                   tau_gaba_ms=target*f.base.tau_gaba_ms)
            if not any(abs(q['parameter']-target)<1e-10 for q in curve):curve.append(r)
    curve.sort(key=lambda r:r['parameter'])
    at_cross=min(curve,key=lambda r:abs(r['parameter']-native['scale']))
    if abs(at_cross['growth_per_ms'])>1e-6 or abs(at_cross['frequency_hz']-native['frequency_hz'])>.01:
        raise RuntimeError('crossing does not belong to the baseline tracked mode')
    result=dict(candidate_id=cid,crossings=crossings,mode_curve=curve,tracking=tracking,discovery=discovery,
        high_equilibrium_hz=float(np.average(x[:f.base.n_cells],weights=f.base.count_e)*1000),
        anchor=read(out/'anchor.json'),full_delay_map_crosscheck_residual=abs(value),
        interpretation='Oscillatory stability boundary of a high-rate equilibrium. This is not an interictal-to-runaway onset boundary. Generic Hopf type and periodic-orbit branches not established.',
        method='Exact elimination of native-map delay histories and synaptic filters; 200-dimensional complex characteristic matrix. Smaller dt holds physical delays and equilibrium fixed.',
        code_sources={str(p):sha(p) for p in [Path(__file__),base.ROOT/'src/topic4_fig5_delay_characteristic.py']})
    write(out/'gaba_oscillatory_boundary.json',result)
    return result


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),layout='constrained')
    for row,cid in enumerate(base.base.rec.IDS):
        r=read(base.DATA/cid/'gaba_oscillatory_boundary.json');c=r['crossings'][0] if r['crossings'] else None
        curve=[q for q in r['mode_curve'] if q['converged']]
        tau=np.array([q['tau_gaba_ms'] for q in curve]);growth=np.array([q['growth_per_ms']*1000 for q in curve]);freq=np.array([q['frequency_hz'] for q in curve])
        ax=axes[row,0];ax.axhline(0,color='.4',lw=.8)
        if c:ax.axvline(c['tau_gaba_ms'],color='#803781',ls='--',lw=1)
        ax.plot(tau,growth,'-',color='#803781',lw=1.8)
        for side in ['below','above']:
            check=base.DATA/cid/f'gaba_crossing_full_spectrum_{side}.json'
            if check.exists():
                q=read(check)
                if q['converged']:
                    ax.scatter(q['parameter']*base.get_family(cid,'tau_gaba').base.tau_gaba_ms,
                        q['modes'][0]['growth_per_ms']*1000,s=40,facecolor='white',edgecolor='black',zorder=7)
        ax.fill_between(tau,0,growth,where=growth>0,color='#e9bba1',alpha=.5)
        ax.fill_between(tau,0,growth,where=growth<0,color='#97cece',alpha=.5)
        ax.set(xlabel='GABA decay time (ms)',ylabel='Oscillatory-mode growth (s⁻¹)',
            title=f'Base {row+1} · high equilibrium ≈ {r["high_equilibrium_hz"]:.0f} Hz')
        annotation=f'Candidate crossing: {c["tau_gaba_ms"]:.2f} ms\nFrequency: {c["frequency_hz"]:.2f} Hz' if c else 'Growing mode persists\nthrough 1.5× GABA decay'
        ax.text(.98,.95,annotation,transform=ax.transAxes,ha='right',va='top',fontsize=10,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.85))
        ax=axes[row,1]
        for sign in [1,-1]:
            ax.plot(growth,sign*freq,'-',color='.7',lw=1)
            im=ax.scatter(growth,sign*freq,c=tau,cmap='viridis',s=28)
            if c:ax.scatter(0,sign*c['frequency_hz'],marker='D',s=38,color='#803781')
        ax.axvline(0,color='.4',lw=.8)
        ax.set(xlabel='Real growth exponent (s⁻¹)',ylabel='Imaginary exponent / 2π (Hz)',title='Conjugate oscillatory modes')
        fig.colorbar(im,ax=ax,label='GABA decay time (ms)',shrink=.75)
    fig.suptitle('High-state oscillatory stability in the reduced model\nFixed spatial Z and equilibrium activity · projected delays · dt = 0.1 ms',fontsize=15)
    base.FIG.mkdir(parents=True,exist_ok=True);stem=base.FIG/'fig5-gaba-oscillatory-stability-boundary'
    fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
    with (base.FIG/'README.md').open('a') as f:
        f.write('\n### fig5-gaba-oscillatory-stability-boundary.png\n固定两套基底各自折点附近的空间 Z，沿 GABA 衰减时间追踪同一高率平衡根的振荡特征模态。左列显示增长率过零，右列显示共轭模态跨越稳定性边界；0.1、0.05、0.025 ms 及连续时间极限均单独核查。\n**关注点**：这是高活动平衡态的振荡稳定性转变，不是已建立的间期事件到 runaway 的起始分岔；周期轨道及 Hopf 非线性类型尚未确定。\n')


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:
        for result in pool.map(analyze,base.base.rec.IDS):
            print(result['candidate_id'],result['crossings'],flush=True)
    render()
