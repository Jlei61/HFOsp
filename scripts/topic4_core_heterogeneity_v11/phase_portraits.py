"""Actual state-space trajectories projected onto Core A E/I rates.

This is not a two-dimensional autonomous closure: hidden populations, filters
and physical delay histories are retained during every integration. Therefore
no planar nullclines, separatrices or state-independent vector field are drawn.
"""
from common import *
from integrate import simulate,constant_state,orbit_state,describe
from spectral import leading
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.spatial import cKDTree
import argparse,hashlib

DEST=OUT/'phase_portraits';DEST.mkdir(exist_ok=True)
FIG=DEST/'figures';FIG.mkdir(exist_ok=True)
BLUE='#2465b0';RED='#cf4a45';GRAY='#959aa1'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
                     'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})


def exact_path(h,g,direction):
    p=OUT/'validated_attractors'/f'h{h:.5f}_{direction}_g{g:.5f}.json'
    d=read(p)
    assert d['stable_periodic_attractor']
    return ROOT/d['source']


def worker(kind):
    if kind.startswith('rest'):
        h,g=1.,1.1;s=System(h);r,err,ok=s.solve(g,np.array([.0002,.0002,0,0,0,0]));assert ok
        initial=r.copy();initial[[0,3]]=(np.array([.04,.40]) if kind=='rest_left' else np.array([.48,.22]))/1000
        state=constant_state(s,initial,.05);duration=1000
        meta=dict(h=h,g=g,initial_history='constant six rates and stationary synaptic filters',initial_rates_hz=(initial*1000).tolist(),equilibrium_hz=(r*1000).tolist())
    elif kind=='burst_entry':
        h,g=1.,1.14;s=System(h);initial=np.array([.0002,.0002,0,0,0,0])
        state=constant_state(s,initial,.05);duration=3000
        meta=dict(h=h,g=g,initial_history='constant low-rate six-population history',initial_rates_hz=(initial*1000).tolist())
    else:
        h,g=0.,1.38;s=System(h);direction='up' if kind.startswith('mixed') else 'down'
        path=exact_path(h,g,direction);z=np.load(path);r=z['r'];T=float(z['T'])
        # The same phase and a complete, consistent history are used. Scale
        # all rates and filters/history together; do not perturb just a 2-D dot.
        r=np.roll(r,-int(np.argmax(r[:,0])),axis=0)
        state=orbit_state(s,r,T,.05);factor=(.96 if kind=='mixed' else .99999) if kind.startswith('mixed') else 1.025
        state=(state[0]*factor,state[1]*factor,state[2]);duration=1500
        meta=dict(h=h,g=g,initial_history='uniform amplitude perturbation of a complete periodic history',factor=factor,initial_rates_hz=(state[0][:6]*1000).tolist(),orbit_source=str(path.relative_to(ROOT)))
    rate,state=simulate(s,g,duration_ms=duration,dt=.05,state=state,save_dt=.25)
    meta.update(dt_ms=.05,sample_dt_ms=.25,duration_ms=duration,description=describe(rate,.25))
    if kind.startswith('rest'):
        meta['final_equilibrium_error_hz']=float(abs(rate[-1]*1000-np.array(meta['equilibrium_hz'])).max())
        assert meta['final_equilibrium_error_hz']<.002
    elif kind!='burst_entry':
        meta['final_rate_distance_to_exact_orbit_hz']=float(cKDTree(z['r']*1000).query(rate[-1]*1000)[0])
    np.savez_compressed(DEST/f'{kind}.npz',r=rate,sample_dt=.25,h=h,g=g,initial_rate=np.array(meta['initial_rates_hz'])/1000)
    write(DEST/f'{kind}.json',meta)
    return kind,meta


def equilibrium(h,g,low=True):
    s=System(h)
    guesses=[np.array([.0002,.0002,0,0,0,0])] if low else [np.array([a,a,a*.001,a,a,a*.001]) for a in [.05,.1,.25,.35]]
    for initial in guesses:
        r,err,ok=s.solve(g,initial)
        if ok:break
    assert ok
    roots=leading(s,r,g,40);roots64=leading(s,r,g,64)
    assert min(abs(roots[0]-roots64))<1e-4 and abs(roots[0].real-roots64[0].real)<1e-4
    row=dict(h=h,g=g,r_hz=(r*1000).tolist(),residual=err,leading_real_per_s=float(roots[0].real),stable=bool(roots[0].real<0))
    write(DEST/f'equilibrium_h{h:.5f}_g{g:.5f}_{"low" if low else "selected_unstable"}.json',row)
    return row


def arrows(ax,xy,color,number=3,closed=False):
    """Arrowheads follow recorded time order, spaced in displayed arclength."""
    if closed:xy=np.r_[xy,xy[:1]]
    scale=np.array([np.diff(ax.get_xlim())[0],np.diff(ax.get_ylim())[0]])
    steps=np.linalg.norm(np.diff(xy,axis=0)/scale,axis=1)
    arc=np.r_[0,np.cumsum(steps)]
    if arc[-1]<.015:return
    for frac in np.linspace(.18,.84,number):
        target=frac*arc[-1];span=min(.035,arc[-1]/14)
        i=int(np.searchsorted(arc,max(0,target-span)));j=int(np.searchsorted(arc,min(arc[-1],target+span)))
        if j<=i:continue
        ax.annotate('',xy=xy[j],xytext=xy[i],arrowprops=dict(arrowstyle='-|>',color=color,lw=1.05,mutation_scale=13),zorder=8)


def trajectory(ax,r,color=GRAY,lw=1.1,closed=False,initial=False,number=3,alpha=1.):
    xy=r[:,[0,3]]*1000
    drawn=np.r_[xy,xy[:1]] if closed else xy
    ax.plot(drawn[:,0],drawn[:,1],color=color,lw=lw,alpha=alpha,zorder=4 if closed else 2)
    arrows(ax,xy,color,number,closed)
    if initial:ax.plot(*xy[0],'o',color='black',ms=4,zorder=9)


def eq_marker(ax,row):
    ax.plot(row['r_hz'][0],row['r_hz'][3],marker='*',mfc='black' if row['stable'] else 'white',mec='black',mew=1.1,ms=12,zorder=12,clip_on=False)


def setup(ax,title,h,g,rest=False):
    ax.set_title(title,loc='left',fontweight='bold',pad=30)
    ax.text(0,1.025,rf'$h={h:g},\quad J_{{\mathrm{{EE,core}}}}={g:g}$',transform=ax.transAxes,fontsize=11)
    ax.set_xlabel(r'$r_E^A$ (Hz / cell)');ax.set_ylabel(r'$r_I^A$ (Hz / cell)')
    ax.set_box_aspect(1)
    if rest:ax.set(xlim=(-.01,.55),ylim=(-.018,.44));ax.set_xticks([0,.2,.4]);ax.set_yticks([0,.2,.4])
    else:ax.set(xlim=(-10,405),ylim=(-18,650));ax.set_xticks([0,200,400]);ax.set_yticks([0,200,400,600])


def save(fig,name):
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)


def main_portrait():
    fig,axes=plt.subplots(1,3,figsize=(12.5,5.1));fig.subplots_adjust(wspace=.30,bottom=.30,top=.80)
    setup(axes[0],'a   Rest',1,1.1,rest=True)
    for key,col in [('rest_left',GRAY),('rest_right','#60666d')]:
        d=np.load(DEST/f'{key}.npz');trajectory(axes[0],np.r_[d['initial_rate'][None,:],d['r']],col,lw=1.2,initial=True,number=2)
    eq_marker(axes[0],equilibrium(1,1.1))
    setup(axes[1],'b   Regular burst',1,1.14)
    entry=np.load(DEST/'burst_entry.npz');trans=np.r_[entry['initial_rate'][None,:],entry['r']]
    trajectory(axes[1],trans[:6000],GRAY,initial=True,number=2)
    z=np.load(exact_path(1,1.14,'up'));trajectory(axes[1],z['r'],BLUE,lw=2.,closed=True,number=4)
    eq_marker(axes[1],equilibrium(1,1.14,low=False))
    setup(axes[2],'c   Two stable cycles',0,1.38)
    for key,col,direction in [('mixed_local',BLUE,'up'),('high',RED,'down')]:
        d=np.load(DEST/f'{key}.npz');trajectory(axes[2],np.r_[d['initial_rate'][None,:],d['r'][:2400]],GRAY,lw=.85,initial=True,number=1,alpha=.60)
        z=np.load(exact_path(0,1.38,direction));trajectory(axes[2],z['r'],col,lw=1.8,closed=True,number=3)
    handles=[Line2D([],[],color=GRAY,lw=1,label='Transient'),Line2D([],[],color=BLUE,lw=2,label='Burst cycle'),Line2D([],[],color=RED,lw=2,label='High-background cycle'),
             Line2D([],[],color='black',marker='o',ls='',ms=4,label='Initial state'),Line2D([],[],color='black',marker='*',ls='',ms=10,label='Stable equilibrium'),Line2D([],[],color='black',mfc='white',marker='*',ls='',ms=10,label='Unstable equilibrium')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.045),ncol=3,frameon=False,fontsize=10)
    fig.text(.07,.018,'Fixed parameters in each panel; arrows show time direction. Full delay-system trajectories projected onto Core A E/I rates.',fontsize=9.5)
    save(fig,'core_A_phase_portraits')


def heterogeneity_portrait():
    fig,axes=plt.subplots(1,3,figsize=(12.5,5.1));fig.subplots_adjust(wspace=.30,bottom=.30,top=.80)
    seed=V2/'periodic/g1.14000000_N2048.npz';seed_orbit=np.load(seed)['r']
    for ax,h,label in zip(axes,[1.,.975,.5],['a   Sustained burst','b   Return to rest','c   Return to rest']):
        setup(ax,label,h,1.14)
        if h==1:
            z=np.load(exact_path(1,1.14,'up'));trajectory(ax,z['r'],BLUE,lw=1.9,closed=True,number=4)
            ax.plot(seed_orbit[0,0]*1000,seed_orbit[0,3]*1000,'o',color='black',ms=4,zorder=10)
        else:
            path=OUT/'orbit_seed_scan'/f'h{h:.5f}_g1.14000.npz';z=np.load(path)
            r=np.r_[seed_orbit[:1],z['r']]
            trajectory(ax,r,BLUE,lw=1.7,initial=False,number=3)
            ax.plot(seed_orbit[0,0]*1000,seed_orbit[0,3]*1000,'o',color='black',ms=4,zorder=10)
            row=equilibrium(h,1.14);eq_marker(ax,row)
            assert abs(r[-1]*1000-np.array(row['r_hz'])).max()<.002
            inset=ax.inset_axes([.53,.65,.42,.30]);inset.set(xlim=(0,.65 if h>.9 else .3),ylim=(-.001,.02))
            inset.plot(r[:,0]*1000,r[:,3]*1000,color=BLUE,lw=1.2);eq_marker(inset,row)
            inset.set_xticks([0,.3,.6] if h>.9 else [0,.1,.2]);inset.set_yticks([0,.01,.02]);inset.tick_params(labelsize=8)
            inset.set_title('Near rest',fontsize=9,pad=3);inset.set_facecolor('#fafafa')
    fig.text(.07,.09,r'EE coupling and the mean threshold are fixed; only $h=\sigma_A/\sigma_{A,0}$ changes.',fontsize=11)
    fig.text(.07,.035,'Same initial periodic history in all three panels. Points mark its initial state; stars mark the reached equilibrium.',fontsize=10)
    save(fig,'heterogeneity_state_space')


def document():
    source_paths=[exact_path(1,1.14,'up'),exact_path(0,1.38,'up'),exact_path(0,1.38,'down'),V2/'periodic/g1.14000000_N2048.npz']
    source_paths += [OUT/'orbit_seed_scan'/f'h{h:.5f}_g1.14000.npz' for h in [.975,.5]]
    source_paths += [DEST/f'{key}.npz' for key in ['rest_left','rest_right','burst_entry','mixed_local','high']]
    source_paths += sorted(DEST.glob('equilibrium_*.json'))
    metadata=dict(projection=['Core A E rate','Core A I rate'],units='Hz/cell',
        equation='Frozen six-population rate model with all filters and physical delays',
        nullclines=False,autonomous_2D_vector_field=False,separatrix=False,
        cycles='Exact periodic orbits with existing transverse Floquet validation',
        equilibria='Full six-population equilibrium solve and N40/N64 DDE spectrum',
        stability_scope='Only marked solutions; no claim of exhaustive attractors or planar basins',
        initial_histories='Constant histories for rest examples; full periodic histories for cycle perturbations and the h comparison',
        h_comparison='EE and threshold mean fixed; identical initial periodic history; basin result for this history only',
        sources=[dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in source_paths],
        human_acceptance='PENDING')
    write(DEST/'metadata.json',metadata)
    (DEST/'analysis_note.md').write_text('''# 相轨迹图与参数平面图的区别

这次按用户参考图改为**状态空间相轨迹图**：每幅图固定参数，横轴 Core A 的 E rate，纵轴 Core A 的 I rate；曲线按真实时间连接，箭头沿时间方向。稳定平衡态显示为星号，稳定周期轨道显示为闭合曲线；同参数下的两个稳定周期轨道在同一平面叠加。

上一版横轴 EE、纵轴异质性，属于**参数平面状态图/双参数分岔图**：它回答参数改变后出现哪一种状态，不是概率或细胞数量的分布图。两种图回答不同问题，不能用其中一张替代另一张。

## 本次两张图

- `figures/core_A_phase_portraits.pdf`：静息、规则 burst、两个周期吸引子共存。第一格放大低率范围，三格的坐标范围并不全部相同；最后一格的蓝、红周期轨道参数完全相同。淡色细线为实际扰动或参数切换后的暂态；蓝/红粗线来自此前经过 Floquet 稳定性验证的精确周期轨道。
- `figures/heterogeneity_state_space.pdf`：固定 EE 和平均阈值，三个面板采用同一段初始周期历史，只改变 A 的阈值标准差比例。reference 时维持周期 burst，所展示的较低 h 条件下同一初始历史回到静息态；这是该初值下的结果，不是证明该参数下不存在其他吸引子。主轴一致，局部 inset 放大收敛终点。

## 与参考图不能直接照搬的部分

我们的六率模型还包含突触滤波和延迟历史，所以这只是完整系统在 A E/I 平面上的投影。投影曲线可以相交；相同的 A E/I rate 配合不同 B、surround 或历史，后续方向可以不同。因而不能直接画一个处处唯一的二维箭头流场，也不能把冻结其他状态后算出的条件零增长线称为完整系统的 nullclines 或吸引域边界。

图中平衡点的稳定性来自完整延迟系统；空心星号只标出一个实际求得的不稳定平衡点，不声称枚举全部平衡点。闭合轨道的稳定性复用此前精确周期解的横向 Floquet 检验。没有把人为椭圆、插值连线或二维闭环外观当作新的分岔证据。

源码为 `scripts/topic4_core_heterogeneity_v11/phase_portraits.py`。`prepare` 生成少量完整历史扰动轨迹，`plot` 复用已有验证轨道和本次轨迹生成 PNG/PDF/SVG；未改动上一版扫描与分岔结果。
''')
    (FIG/'README.md').write_text('''# Core A 状态空间相轨迹图

### core_A_phase_portraits.pdf

三格分别展示静息平衡点、规则 burst 周期轨道和同参数下两个稳定周期轨道的 E/I 投影。箭头来自真实时间顺序，第一格的低率范围单独放大；同名 PNG/SVG 为同一版本。**关注点**：点、闭合轨道和不同初始历史如何对应不同动力学状态。

### heterogeneity_state_space.pdf

固定 EE 强度与平均阈值，采用相同初始周期历史，对比不同 Core A 阈值异质性。主坐标一致，两个局部窗显示最终收敛的低率平衡点。**关注点**：异质性改变后的实际轨迹，而非把参数平面的色块当作相轨迹。

图为完整延迟系统的二维投影，没有另造二维 nullclines；待用户人工目视验收。
''')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','plot']);p.add_argument('--one');a=p.parse_args()
    if a.action=='prepare':
        if a.one:print(worker(a.one),flush=True)
        else:
            with ProcessPoolExecutor(max_workers=4) as pool:
                for key,row in pool.map(worker,['rest_left','rest_right','mixed_local','high','burst_entry']):print(key,row,flush=True)
    else:main_portrait();heterogeneity_portrait();document()
