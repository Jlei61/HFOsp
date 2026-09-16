"""Run local parameter-branch analysis on both fixed two-core GIF bases."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import fields
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import src
ZROOT=ROOT.parent/'topic4-dual-core-z-bifurcation'
src.__path__.append(str(ZROOT/'src'))
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigs,ArpackNoConvergence
import recover_fig5_two_gif_bases as rec
from src.topic4_xy_fig5_followup import read,write,sha
from src.topic4_patient_zm_meanfield import (load_patient_coarse_model,
    build_patient_coarse_model,save_patient_coarse_model,spatial_cell_index)
from src.topic4_dual_core_spatial_z_delay import build_coarse_delay_operators,CoarseDelayOperators
from src.topic4_fig5_local_bifurcation import Family,arc_continue,polish_fold,corrected_delay_matrix

DATA=ROOT/'results/topic4_sef_hfo/fig5_local_parameter_bifurcation'
FIG=ROOT/'results/paper-ready-figure/fig5_local_parameter_bifurcation/figures'
BOUNDS={'z_loss':(0.,2.),'ee':(.75,1.25),'e_to_i':(.6,1.4),'i_to_e':(.6,1.4),
        'eta_m':(0.,4.),'tau_m':(.5,2.),'tau_m_matched':(.5,2.),'tau_gaba':(.5,1.5)}
LABELS={'z_loss':'Retained Z loss, λ','ee':'E → E strength / baseline',
        'e_to_i':'E → I strength / baseline','i_to_e':'I → E strength / baseline',
        'eta_m':'Adaptation strength / baseline','tau_m':'Adaptation time constant / baseline',
        'tau_m_matched':'Adaptation time / baseline\n(fixed ηM τM)',
        'tau_gaba':'GABA decay time / baseline'}


def prepare(cid):
    out=DATA/cid;out.mkdir(parents=True,exist_ok=True)
    if (out/'prepared.json').exists():return read(out/'prepared.json')
    s,cfg,regions,fp,job,protocol=rec.build(cid)
    model=build_patient_coarse_model(s,n_grid=10,threshold_groups=8)
    source=ROOT/'results/topic4_sef_hfo/fig5_full_two_bases'/cid/'coarse_model.npz'
    previous=load_patient_coarse_model(source)
    for f in fields(model):
        if not np.array_equal(getattr(model,f.name),getattr(previous,f.name)):
            raise RuntimeError('substrate reduction changed: '+f.name)
    model_record=save_patient_coarse_model(out/'model.npz',model)
    ops=build_coarse_delay_operators(s,model)
    for key in ('ee','ei','ie','ii'):
        sparse.save_npz(out/('delay_'+key+'.npz'),getattr(ops,'w_'+key+'_history'))
    early=rec.old.load_checkpoint(rec.DATA/cid/'checkpoint_reference.npz')['slow']['z'][:s.n_e]
    pre=rec.old.load_checkpoint(rec.DATA/cid/'checkpoint_pre_onset.npz')['slow']['z'][:s.n_e]
    delta=pre-early
    cell=spatial_cell_index(s.positions_e,n_grid=model.n_grid,sheet_l_mm=model.sheet_l_mm)
    def avg(a):return np.bincount(cell,weights=a,minlength=model.n_cells)/model.count_e
    zm=np.stack([avg(early),avg(delta),avg(early**2),avg(early*delta),avg(delta**2)])
    np.savez_compressed(out/'z_fields.npz',moments=zm,early=early,pre=pre,cell=cell)
    physical_upper=min(np.min(-early[delta<0]/delta[delta<0]) if np.any(delta<0) else np.inf,
                       np.min((1-early[delta>0])/delta[delta>0]) if np.any(delta>0) else np.inf)
    record=dict(candidate_id=cid,model=model_record,eta_m=job['config']['eta_m'],
        tau_m_ms=500.,tau_z_ms=job['config']['tau_z_ms'] if 'tau_z_ms' in job['config'] else 5000.,
        dt_ms=ops.dt_ms,max_delay_steps=ops.max_delay_steps,
        z_path_physical_upper=float(physical_upper),z_loss_upper=min(2.,.999*float(physical_upper)),
        z_mean_early=float(early.mean()),z_mean_pre=float(pre.mean()),
        substrate_fingerprint=fp,source_coarse_model=str(source),
        checkpoint_sources={str(q):sha(q) for q in [rec.DATA/cid/'checkpoint_reference.npz',rec.DATA/cid/'checkpoint_pre_onset.npz']})
    write(out/'prepared.json',record);return record


def family(cid,parameter,with_ops=False):
    out=DATA/cid;p=read(out/'prepared.json')
    model=load_patient_coarse_model(out/'model.npz')
    with np.load(out/'z_fields.npz') as a:zm=a['moments']
    ops=None
    if with_ops:
        ops=CoarseDelayOperators(dt_ms=p['dt_ms'],max_delay_steps=p['max_delay_steps'],
            **{'w_'+key+'_history':sparse.load_npz(out/('delay_'+key+'.npz')) for key in ('ee','ei','ie','ii')})
    return Family(model,zm,p['eta_m'],parameter,ops)


def scan(cid,parameter):
    out=DATA/cid;dest=out/(parameter+'.json')
    if dest.exists():return read(dest)
    start=time.monotonic();fam=family(cid,parameter);bounds=BOUNDS[parameter]
    if parameter=='z_loss':bounds=(0.,read(out/'prepared.json')['z_loss_upper'])
    rows=[];branches=[];failures=[];folds=[];catalog=[];attempts=[]
    grid=np.unique(np.r_[np.linspace(*bounds,9),1.])
    grid=grid[(grid>=bounds[0])&(grid<=bounds[1])]
    rng=np.random.default_rng(941)
    for p in grid:
        guesses=[np.full(2*fam.base.n_cells,g) for g in (.005,.05,.2,.45)]
        if abs(p-1)<1e-8:
            guesses += [np.clip(.1+rng.normal(0,.06,2*fam.base.n_cells),.001,.45),
                        np.clip(.3+rng.normal(0,.08,2*fam.base.n_cells),.001,.49)]
        for j,guess in enumerate(guesses):
            x,ok,err=fam.solve(float(p),guess)
            attempts.append(dict(parameter=float(p),guess=j,converged=ok,residual=err))
            if ok and not any(abs(p-q)<1e-10 and np.max(abs(x-y))<1e-6 for y,q in catalog):
                catalog.append((x,float(p)))
    if not catalog:raise RuntimeError('NO_ROOTS '+cid+' '+parameter)
    means=np.array([np.average(x[:fam.base.n_cells],weights=fam.base.count_e) for x,p in catalog])
    seeds_to_trace=[]
    for side,idx in [('low',int(np.argmin(means))),('high',int(np.argmax(means)))]:
        x,p=catalog[idx]
        for direction in (-1,1):
            p2=p+direction*.002
            if not bounds[0]<=p2<=bounds[1]:continue
            y,ok,err=fam.solve(p2,x)
            if ok:seeds_to_trace.append((side+('_down' if direction<0 else '_up'),[(x,p),(y,p2)]))
            else:failures.append(dict(side=side,reason='SECOND_ROOT_FAILED',parameter=p2,residual=err))
    for side,seeds in seeds_to_trace:
        points,reason=arc_continue(fam,*seeds,bounds,max_steps=650,step=.008)
        offset=len(rows)
        rows.extend(points)
        branches.append(dict(side=side,start=offset,stop=len(rows),reason=reason))
        for i in range(1,len(points)):
            if points[i-1][2]*points[i][2]>=0:continue
            x,p,_=points[i]
            try:record,rfold=polish_fold(fam,x,p)
            except (ValueError,RuntimeError) as e:
                record={'confirmed':False,'message':str(e)};rfold=None
            record.update(branch=side,near_index=offset+i)
            if rfold is not None:
                record['rates_key']='fold_'+str(len(folds));record['_rates']=rfold
            folds.append(record)
    if not rows:raise RuntimeError('no roots for '+cid+' '+parameter)
    arrays=dict(rates=np.stack([r[0] for r in rows]),parameter=np.array([r[1] for r in rows]),
                tangent=np.array([r[2] for r in rows]),catalog_rates=np.stack([x for x,p in catalog]),
                catalog_parameters=np.array([p for x,p in catalog]))
    for f in folds:
        if '_rates' in f:arrays[f['rates_key']]=f.pop('_rates')
    np.savez_compressed(out/(parameter+'.npz'),**arrays)
    rates=arrays['rates'];parameters=arrays['parameter']
    record=dict(candidate_id=cid,parameter=parameter,bounds=bounds,branches=branches,folds=folds,
        root_count=len(rows),root_max_residual=max(float(np.max(np.abs(fam.f(x,p)))) for x,p in zip(rates,parameters)),
        failures=failures,root_seed_attempts=attempts,catalog_root_count=len(catalog),duration_s=time.monotonic()-start,
        arrays={'path':str(out/(parameter+'.npz')),'sha256':sha(out/(parameter+'.npz'))},
        claim='Equilibrium branches only; stability pending delay-aware analysis.')
    write(dest,record);return record


def stability_point(cid,parameter,index):
    out=DATA/cid/'stability';out.mkdir(exist_ok=True)
    dest=out/f'{parameter}_{index}.json'
    if dest.exists():return read(dest)
    fam=family(cid,parameter,True)
    with np.load(DATA/cid/(parameter+'.npz')) as a:
        if str(index).startswith('fold_'):
            x=a[str(index)];summary=read(DATA/cid/(parameter+'.json'))
            p=next(f['parameter'] for f in summary['folds'] if f.get('rates_key')==index)
        else:x=a['rates'][int(index)];p=float(a['parameter'][int(index)])
    mat=corrected_delay_matrix(fam,x,p)
    # Deterministic starting vector; largest magnitude, not just near-zero shift.
    start=time.monotonic()
    try:
        values,vectors=eigs(mat,k=8,which='LM',tol=2e-8,maxiter=12000,
                           v0=np.random.default_rng(381).normal(size=mat.shape[0]))
        converged=True
    except ArpackNoConvergence as exc:
        values,vectors=exc.eigenvalues,exc.eigenvectors;converged=False
    modes=[]
    for value,vec in zip(values,vectors.T):
        residual=float(np.linalg.norm(mat@vec-value*vec)/np.linalg.norm(vec))
        modes.append(dict(real=float(value.real),imag=float(value.imag),
            growth_per_ms=float(np.log(abs(value))/fam.operators.dt_ms),
            frequency_hz=float(abs(np.angle(value))*1000/(2*np.pi*fam.operators.dt_ms)),
            residual=residual))
    modes.sort(key=lambda a:a['growth_per_ms'],reverse=True)
    record=dict(parameter=p,index=index,converged=converged,modes=modes,
                duration_s=time.monotonic()-start,matrix_shape=mat.shape,
                population_hz=float(np.average(x[:fam.base.n_cells],weights=fam.base.count_e)*1000),
                claim='Native-dt tangent stability of the deterministic reduction; not an SNN spectrum.')
    write(dest,record);return record


def run():
    DATA.mkdir(parents=True,exist_ok=True)
    protocol=dict(parameters=BOUNDS,substrates=rec.IDS,
        question='Which core controls change equilibrium branches or delayed stability near runaway?',
        fixed='Each original graph, core XY, threshold field and drive; actual pre-onset spatial Z at non-Z scans.',
        z_scan='Clamp early + lambda*(pre-early); preserve first and second within-cell moments.',
        reduction='10x10 cells, eight E threshold groups; diffusion LIF transfer, realized delays, dynamic M, no added OU.',
        continuation='Nine-point grid plus working point; four uniform rate guesses and two random spatial guesses at working point; distinct-root catalog. Trace minimum/maximum-rate discovered roots both directions. This does not exhaust all spatial branches.',
        fold='F=0 and Jv=0; isolated zero, transversality and quadratic nondegeneracy. Not alone proof of stable-state loss.',
        stability='Native 0.1 ms delay tangent at self-consistent adapted input. Historical eta=0 derivative bug corrected locally.',
        tau_z='Absent in frozen-Z subsystem; controls motion toward boundaries, not a parameter of this frozen family.',
        tau_m_control='Also vary tau_M with eta_M*tau_M fixed to separate equilibrium adaptation load from kinetics.',
        tau_gaba='Change decay only; reduced stationary transfer holds integrated weights and diffusion moments fixed. Finite-rise/colored-current approximation requires SNN validation.',
        sources={'Brunel2000':'https://doi.org/10.1023/A:1008925309027','Dhooge2003':'https://doi.org/10.1145/779359.779362'},
        author_accepted=False)
    write(DATA/'protocol.json',protocol)
    for cid in rec.IDS:prepare(cid)
    tasks=[(cid,key) for key in BOUNDS for cid in rec.IDS]
    write(DATA/'status.json',dict(stage='EQUILIBRIUM_CONTINUATION',completed=0,total=len(tasks)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(scan,*t) for t in tasks]
        for i,f in enumerate(as_completed(futures)):
            r=f.result();print(r['candidate_id'],r['parameter'],r['root_count'],flush=True)
            write(DATA/'status.json',dict(stage='EQUILIBRIUM_CONTINUATION',completed=i+1,total=len(tasks)))
    render()
    jobs=[]
    for cid,key in tasks:
        r=read(DATA/cid/(key+'.json'))
        with np.load(DATA/cid/(key+'.npz')) as a:
            pp=a['parameter']
            for branch in r['branches']:
                inds=np.arange(branch['start'],branch['stop'])
                # Endpoints, working point, near folds, and coarse branch samples.
                selected=set(inds[np.linspace(0,len(inds)-1,5).astype(int)].tolist())
                selected.add(int(inds[np.argmin(abs(pp[inds]-1))]))
                for fold in r['folds']:
                    if fold['branch']==branch['side']:
                        k=fold['near_index'];selected.update([max(branch['start'],k-3),min(branch['stop']-1,k+3)])
                jobs.extend((cid,key,i) for i in sorted(selected))
            jobs.extend((cid,key,f['rates_key']) for f in r['folds'] if f.get('confirmed'))
    write(DATA/'status.json',dict(stage='DELAY_STABILITY',completed=0,total=len(jobs)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(stability_point,*j) for j in jobs]
        for i,f in enumerate(as_completed(futures)):
            f.result();write(DATA/'status.json',dict(stage='DELAY_STABILITY',completed=i+1,total=len(jobs)))
    render()
    write(DATA/'status.json',dict(stage='COMPLETE_REDUCED_BRANCH_SCREEN_PENDING_REVIEW',completed=len(jobs),total=len(jobs),
        limitation='Periodic-orbit continuation, spatial resolution convergence and finite-SNN validation not established.'))


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False})
    summaries=[]
    for cid in rec.IDS:
        fig,axes=plt.subplots(2,4,figsize=(15,7),layout='constrained')
        for ax,key in zip(axes.flat,BOUNDS):
            path=DATA/cid/(key+'.json')
            if not path.exists():ax.set_visible(False);continue
            r=read(path);summaries.append(r)
            fam=family(cid,key)
            with np.load(r['arrays']['path']) as a:params=a['parameter'];rates=a['rates']
            mean=np.average(rates[:,:fam.base.n_cells],weights=fam.base.count_e,axis=1)*1000
            for branch in r['branches']:
                ix=slice(branch['start'],branch['stop'])
                ax.plot(params[ix],mean[ix],color='#777777',lw=1.2,alpha=.8)
            for fold in r['folds']:
                if fold.get('confirmed'):
                    ax.scatter(fold['parameter'],fold['population_hz'],marker='D',s=38,color='#bc3580',zorder=6)
            for q in (DATA/cid/'stability').glob(key+'_*.json') if (DATA/cid/'stability').exists() else []:
                s=read(q)
                if not s['converged'] or not s['modes']:continue
                g=s['modes'][0]['growth_per_ms'];color='#007f86' if g<0 else '#d06528'
                ax.scatter(s['parameter'],s['population_hz'],s=18,color=color,zorder=5)
            ax.axvline(1,color='#888888',ls=':',lw=.8)
            ax.set(xlabel=LABELS[key],ylim=(0,510),xlim=BOUNDS[key])
            ax.set_ylabel('Equilibrium E rate (Hz)')
        which='Base 1: threshold gain 0.7, GABA 18 ms' if cid==rec.IDS[0] else 'Base 2: threshold gain 1.0, GABA 24 ms'
        fig.suptitle(which+'\nReduced fixed-Z branches near runaway',fontsize=15)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([],[],color='#777777',label='Equilibrium branch'),
            Line2D([],[],marker='D',ls='',color='#bc3580',label='Verified equilibrium fold'),
            Line2D([],[],marker='o',ls='',color='#007f86',label='Delay-stable sampled root'),
            Line2D([],[],marker='o',ls='',color='#d06528',label='Delay-unstable sampled root')],
            loc='outside lower center',ncol=4,frameon=False)
        FIG.mkdir(parents=True,exist_ok=True);stem=FIG/('fig5-local-branches-'+cid)
        fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
    write(DATA/'analysis.json',dict(branch_summaries=summaries,author_accepted=False,
        interpretation='Equilibrium folds require delay stability before attributing loss of an attracting state.'))
    (FIG/'README.md').write_text('\n'.join('### fig5-local-branches-'+cid+'.png\n固定各自双核位置、连接底物与阈值场，对八条控制参数路径追踪降阶模型平衡分支。Z 使用真实早期与转变前逐神经元场的均值及二阶矩；其余参数扫描固定转变前 Z。菱形为通过零模和非退化检验的平衡折点，蓝绿/橙点分别为离散延迟系统中采样根稳定/不稳定。\n**关注点**：灰线本身不声明稳定性，未采样处不能推断；这是降阶模型候选分析，不等于完整 SNN runaway 已由分岔解释。\n' for cid in rec.IDS))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','prepare','scan','stability','render'])
    parser.add_argument('candidate',nargs='?');parser.add_argument('parameter',nargs='?');parser.add_argument('index',nargs='?')
    args=parser.parse_args()
    if args.mode=='run':run()
    elif args.mode=='prepare':print(prepare(args.candidate))
    elif args.mode=='scan':print(scan(args.candidate,args.parameter))
    elif args.mode=='stability':print(stability_point(args.candidate,args.parameter,args.index))
    else:render()
