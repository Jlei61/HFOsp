"""Parameter cuts anchored just beyond each newly measured high-state fold.

The anchor is selected by the deterministic reduction, not claimed to be
the SNN transition point. Original pre-onset cuts remain separate.
"""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT)]
import fig5_local_bifurcation as base
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import eigs,ArpackNoConvergence
from concurrent.futures import ProcessPoolExecutor,as_completed
from src.topic4_fig5_local_bifurcation import Family,arc_continue,polish_fold,corrected_delay_matrix
from src.topic4_xy_fig5_followup import read,write,sha
import argparse,time

DATA=base.DATA/'near_high_state_fold'
FIG=base.FIG/'near_high_state_fold'
BOUNDS={'ee':(.9,1.1),'e_to_i':(.85,1.15),'i_to_e':(.85,1.15),
        'eta_m':(0.,2.),'tau_m':(.5,2.),'tau_m_matched':(.5,2.),'tau_gaba':(.5,1.5)}


def get_family(cid,key,ops=False):
    f=base.family(cid,key,ops)
    f.z_anchor=read(DATA/cid/'anchor.json')['lambda_anchor']
    return f


def prepare(cid):
    out=DATA/cid;out.mkdir(parents=True,exist_ok=True)
    if (out/'anchor.json').exists():return
    loc=read(base.DATA/cid/'z_loss.json')
    folds=[f for f in loc['folds'] if f.get('confirmed') and f['population_hz']>250]
    if not folds:raise RuntimeError('No verified high-state equilibrium fold for '+cid)
    fold=min(folds,key=lambda r:r['parameter']);anchor=fold['parameter']+.03
    f=base.family(cid,'z_loss')
    with np.load(loc['arrays']['path']) as a:
        nearest=np.argsort(abs(a['parameter']-anchor))[:32]
        guesses=[np.full(2*f.base.n_cells,.00003)]+[a['rates'][i] for i in nearest]
    roots=[]
    for guess in guesses:
        x,ok,err=f.solve(anchor,guess)
        if ok and not any(np.max(abs(x-y))<1e-6 for y in roots):roots.append(x)
    roots.sort(key=lambda x:np.average(x[:f.base.n_cells],weights=f.base.count_e))
    np.savez_compressed(out/'anchor_roots.npz',rates=roots)
    m,z,z2,eta,tau,_=f.at(anchor)
    write(out/'anchor.json',dict(lambda_anchor=anchor,fold=fold,n_roots=len(roots),
        root_rates_hz=[float(np.average(x[:m.n_cells],weights=m.count_e)*1000) for x in roots],
        mean_Z=float(np.average(z,weights=m.count_e)),
        selection='First verified high-state fold plus 0.03 lambda; local reduced-model anchor, not the empirical SNN onset.',
        source=str(base.DATA/cid/'z_loss.json'),source_sha256=sha(base.DATA/cid/'z_loss.json')))


def scan(cid,key):
    dest=DATA/cid/(key+'.json')
    if dest.exists():return read(dest)
    f=get_family(cid,key);n=f.base.n_cells
    with np.load(DATA/cid/'anchor_roots.npz') as a:seeds=a['rates']
    # Lowest then highest, then other discovered roots. Avoid duplicate branches.
    seeds=seeds[[0,len(seeds)-1]+list(range(1,len(seeds)-1))]
    rows=[];branches=[];folds=[];failures=[]
    for si,seed in enumerate(seeds):
        if rows and min(np.linalg.norm(x-seed)/np.sqrt(len(x)) for x,p,t in rows if abs(p-1)<.01)<1e-4:
            continue
        for direction in (-1,1):
            ok=False
            for delta in (.00002,.000002,.0000002):
                p2=1+direction*delta;x,solved,err=f.solve(p2,seed)
                ok=solved and np.linalg.norm(x-seed)/np.sqrt(len(x))<.005
                if ok:break
            if not ok:
                failures.append(dict(seed=si,direction=direction,reason='SECOND_ROOT_FAILED'));continue
            branch,reason=arc_continue(f,(seed,1.),(x,p2),BOUNDS[key],max_steps=650,step=.006)
            start=len(rows);rows.extend(branch);bid=f'seed{si}_'+('down' if direction<0 else 'up')
            branches.append(dict(id=bid,start=start,stop=len(rows),reason=reason))
            for j in range(1,len(branch)):
                if branch[j-1][2]*branch[j][2]>=0:continue
                x,p,_=branch[j]
                try:r,xf=polish_fold(f,x,p)
                except (ValueError,RuntimeError) as e:r={'confirmed':False,'message':str(e)};xf=None
                r.update(branch=bid,near_index=start+j)
                if xf is not None:r['_rates']=xf;r['rates_key']='fold_'+str(len(folds))
                folds.append(r)
    arrays=dict(rates=np.stack([x for x,p,t in rows]),parameter=np.array([p for x,p,t in rows]),
                tangent=np.array([t for x,p,t in rows]))
    for r in folds:
        if '_rates' in r:arrays[r['rates_key']]=r.pop('_rates')
    np.savez_compressed(dest.with_suffix('.npz'),**arrays)
    rec=dict(parameter=key,candidate_id=cid,bounds=BOUNDS[key],branches=branches,folds=folds,
        failures=failures,root_count=len(rows),max_residual=max(float(np.max(abs(f.f(x,p)))) for x,p,t in rows),
        arrays={'path':str(dest.with_suffix('.npz')),'sha256':sha(dest.with_suffix('.npz'))})
    write(dest,rec);return rec


def stability(cid,key,kind,value):
    out=DATA/cid/'stability';out.mkdir(exist_ok=True)
    dest=out/f'{key}_{kind}_{value}.json'
    if dest.exists():return read(dest)
    f=get_family(cid,key,True);n=f.base.n_cells
    if kind=='root':
        with np.load(DATA/cid/(key+'.npz')) as a:x=a['rates'][int(value)];p=float(a['parameter'][int(value)])
    else:
        with np.load(DATA/cid/'anchor_roots.npz') as a:x=a['rates'][-1]
        p=float(value);x,ok,err=f.solve(p,x)
        if not ok:raise RuntimeError('kinetic control root lost')
    mat=corrected_delay_matrix(f,x,p);start=time.monotonic()
    try:
        vals,vec=eigs(mat,k=2,which='LM',tol=2e-8,maxiter=5000,ncv=24,
            v0=np.random.default_rng(382).normal(size=mat.shape[0]));ok=True
    except ArpackNoConvergence as e:vals,vec=e.eigenvalues,e.eigenvectors;ok=False
    modes=[]
    for v,q in zip(vals,vec.T):
        modes.append(dict(growth_per_ms=float(np.log(abs(v))/.1),frequency_hz=float(abs(np.angle(v))*1000/(2*np.pi*.1)),
            residual=float(np.linalg.norm(mat@q-v*q)/np.linalg.norm(q))))
    modes.sort(key=lambda r:r['growth_per_ms'],reverse=True)
    record=dict(parameter=p,modes=modes,converged=ok,duration_s=time.monotonic()-start,
        population_hz=float(np.average(x[:n],weights=f.base.count_e)*1000),kind=kind,value=value)
    write(dest,record);return record


def jobs():
    result=[]
    for cid in base.rec.IDS:
        for key in BOUNDS:
            r=read(DATA/cid/(key+'.json'))
            with np.load(r['arrays']['path']) as a:
                pp=a['parameter'];rr=a['rates'];mean=np.average(rr[:,:100],weights=get_family(cid,key).base.count_e,axis=1)*1000
            chosen=set()
            # Three distinct baseline branches; endpoint points on recruited branches.
            if key=='ee':
                # All parameter families coincide at multiplier=1; solve once.
                for target in read(DATA/cid/'anchor.json')['root_rates_hz']:
                    ix=np.argmin(abs(pp-1)*5000+abs(mean-target));chosen.add(int(ix))
            folds=[f for f in r['folds'] if f.get('confirmed') and f['population_hz']>250]
            if folds:
                fold=min(folds,key=lambda f:abs(f['parameter']-1))
                b=next(b for b in r['branches'] if b['id']==fold['branch']);k=fold['near_index']
                chosen.update([max(b['start'],k-3),min(b['stop']-1,k+3)])
            for p in [BOUNDS[key][0],BOUNDS[key][1]]:
                inds=np.flatnonzero(mean>250)
                if len(inds):
                    closest=inds[np.abs(pp[inds]-p)<np.min(abs(pp[inds]-p))+.002]
                    chosen.add(int(closest[np.argmax(mean[closest])]))
            result.extend((cid,key,'root',i) for i in sorted(chosen))
    return result


def render():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,
        'axes.spines.top':False,'axes.spines.right':False})
    for ci,cid in enumerate(base.rec.IDS):
        fig,axes=plt.subplots(2,4,figsize=(15,7.5),layout='constrained')
        anchor=read(DATA/cid/'anchor.json')
        for ax,key in zip(axes.flat,['z_loss']+list(BOUNDS)):
            folder=base.DATA if key=='z_loss' else DATA
            if not (folder/cid/(key+'.json')).exists():ax.set_visible(False);continue
            r=read(folder/cid/(key+'.json'))
            with np.load(r['arrays']['path']) as a:pp=a['parameter'];rr=a['rates']
            m=base.family(cid,key).base;mean=np.average(rr[:,:m.n_cells],weights=m.count_e,axis=1)*1000
            for b in r['branches']:
                sl=slice(b['start'],b['stop']);ax.plot(pp[sl],mean[sl],lw=1,color='.55')
            for fold in r['folds']:
                if fold.get('confirmed'):ax.scatter(fold['parameter'],fold['population_hz'],marker='D',s=34,color='#b7357c',zorder=5)
            if key=='z_loss':ax.axvline(anchor['lambda_anchor'],color='#00818a',ls='--',lw=1)
            else:
                points=list((DATA/cid/'stability').glob(key+'_root_*.json')) if (DATA/cid/'stability').exists() else []
                if key!='ee' and (DATA/cid/'stability').exists():
                    points += [q for q in (DATA/cid/'stability').glob('ee_root_*.json') if abs(read(q)['parameter']-1)<1e-9]
                for q in points:
                    s=read(q)
                    if not s['converged'] or not s['modes']:continue
                    g=s['modes'][0]['growth_per_ms'];color='#00818a' if g<-1e-6 else '#d36625'
                    ax.scatter(s['parameter'],s['population_hz'],s=23,color=color,zorder=6)
                retry=DATA/cid/'silent_root_spectrum_retry.json'
                if retry.exists():
                    sr=read(retry)
                    if sr['converged'] and sr['modes'][0]['growth_per_ms']<0:
                        ax.scatter(1,anchor['root_rates_hz'][0],s=23,color='#00818a',zorder=6)
            ax.axvline(1,color='.6',ls=':',lw=.6)
            ax.set(xlabel=base.LABELS[key],ylabel='Equilibrium E rate (Hz)',ylim=(-10,510),
                   xlim=base.BOUNDS[key] if key=='z_loss' else BOUNDS[key])
        fig.suptitle(f'Base {ci+1}: core parameters near the high-state fold\nFixed spatial Z for parameter cuts: mean Z = {anchor["mean_Z"]:.3f}',fontsize=15)
        fig.legend(handles=[Line2D([],[],color='.55',label='Equilibrium branch'),
            Line2D([],[],ls='',marker='D',color='#b7357c',label='Verified equilibrium fold'),
            Line2D([],[],ls='',marker='o',color='#00818a',label='Delay-stable sampled root'),
            Line2D([],[],ls='',marker='o',color='#d36625',label='Delay-unstable sampled root')],
            loc='outside lower center',ncol=4,frameon=False)
        FIG.mkdir(parents=True,exist_ok=True);stem=FIG/('fig5-near-runaway-'+cid)
        fig.savefig(stem.with_suffix('.png'),dpi=180);fig.savefig(stem.with_suffix('.pdf'));plt.close(fig)
    (FIG/'README.md').write_text('\n'.join('### fig5-near-runaway-'+cid+'.png\n第一格在真实 Z 空间场方向上定位降阶模型高态折点，其余七格将 Z 固定在该折点后 λ+0.03，单独改变连接、适应或抑制动力学参数。灰线仅为求得的平衡分支，菱形为验证的平衡折点，彩色点表示实际检查过的延迟稳定性。\n**关注点**：近乎静默的低平衡支不是间期事件支；工作点由降阶模型选定，不等于 SNN onset，未经采样的稳定性及周期轨道分岔仍未确定。\n' for cid in base.rec.IDS))


def run():
    DATA.mkdir(parents=True,exist_ok=True)
    write(DATA/'protocol.json',dict(parameters=BOUNDS,
        anchor='Per substrate first verified high-state fold +0.03 retained-loss units.',
        inference='Local deterministic reduced-model branch and stability screen; no finite-SNN attribution yet.',
        root_seeds='Three or more distinct roots at anchor located from both fold branches plus silent root; bidirectional pseudo-arclength.',
        code_sources={str(p):sha(p) for p in [Path(__file__),Path(base.__file__),ROOT/'src/topic4_fig5_local_bifurcation.py']},
        references=['https://doi.org/10.1023/A:1008925309027','https://doi.org/10.1145/779359.779362']))
    for cid in base.rec.IDS:prepare(cid)
    tasks=[(cid,key) for key in BOUNDS for cid in base.rec.IDS]
    write(DATA/'status.json',dict(stage='NEAR_FOLD_CONTINUATION',completed=0,total=len(tasks)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(scan,*t) for t in tasks])):
            r=f.result();print(r['candidate_id'],r['parameter'],r['root_count'],flush=True)
            write(DATA/'status.json',dict(stage='NEAR_FOLD_CONTINUATION',completed=i+1,total=len(tasks)))
    render();todo=jobs()
    write(DATA/'status.json',dict(stage='NEAR_FOLD_DELAY_STABILITY',completed=0,total=len(todo)))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,f in enumerate(as_completed([pool.submit(stability,*j) for j in todo])):
            f.result();write(DATA/'status.json',dict(stage='NEAR_FOLD_DELAY_STABILITY',completed=i+1,total=len(todo)))
            if (i+1)%12==0:render()
    render()
    write(DATA/'status.json',dict(stage='REDUCED_LOCAL_SCREEN_COMPLETE_PENDING_REVIEW',completed=len(todo),total=len(todo)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['run','render']);a=p.parse_args()
    if a.mode=='run':run()
    else:render()
