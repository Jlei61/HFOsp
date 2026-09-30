"""All-rate composite: every example is an equilibrium or a solved periodic BVP.
No finite-startup burst is assigned to an asymptotic periodic branch.
"""
from plot_rate_periodic_completion import *
from scipy.interpolate import CubicSpline
from src.snn_contact_display import CONTACT_ORDER,contact_indices,SHAFT_COLORS
from audit_rate_filter_states import filter_state_minima

CASES=[('a',None,.934,'Resting equilibrium'),
       ('b','smallA_J0.942000000_N64',.942,'Small oscillation'),
       ('c','refined_J0.942000000_N1536',.942,'Alternating lead order'),
       ('d','branch095_J0.950000000_N512',.95,'B-leading burst'),
       ('e','branch_J1.300000000_N256',1.3,'A-leading burst')]


def loadcase(s,name,J):
    if name is None:
        r,ok,_=s.solve(J);assert ok
        return np.tile(r,(256,1)),1000.
    path=PERIODIC_OUT/f'orbits/{name}.npz'
    manifest=PERIODIC_OUT/'composite_case_resolution.json'
    if manifest.exists():
        check=read(manifest);assert check['status']=='COMPLETE','Finish exact-case resolution checks before drawing'
        matches=[q for q in check['rows'] if Path(q['original_orbit'])==path]
        assert len(matches)==1 and matches[0]['resolution']['status']=='RESOLUTION_CHECKED'
        path=Path(matches[0]['orbit'])
    z=np.load(path);assert abs(float(z['J'])-J)<1e-9
    physical=filter_state_minima(s,z['r'],float(z['T']))
    assert physical['positive'],f'Physical rate-filter resolution required before showing this case: {path}: {physical}'
    return z['r'],float(z['T'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='spatial_rate_periodic_composite')
    a=p.parse_args()
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    s=RateField();fs=families();geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    xy=geo['contact_xy'];order=contact_indices(geo['contact_names'].tolist())
    cell=s.geo['group_cell'];sz=s.geo['group_size'];ct=np.bincount(cell[s.E],weights=sz[s.E],minlength=400)
    fig=plt.figure(figsize=(22,13.2));grid=fig.add_gridspec(5,4,width_ratios=[1.3,1,1.55,1.15],left=.055,right=.985,bottom=.085,top=.91,wspace=.35,hspace=.65)
    ax=fig.add_subplot(grid[:3,0]);plot_branch(ax,0,fs);ax.set_title('A  Spatial rate bifurcation',loc='left',weight='bold')
    ax.set_ylabel('Core A rate (Hz / cell)')
    inset=fig.add_subplot(grid[3:,0]);plot_branch(inset,0,fs,(.935,.976),True);inset.set_ylim(.65,1.7);inset.set_title('Hopf-born branches: period mean')
    ax.legend(handles=[Line2D([0],[0],color='black',lw=1.8,label='Cycle mean'),
        Line2D([0],[0],color='black',lw=.6,alpha=.45,label='Cycle min / max'),
        Line2D([0],[0],marker='o',ls='',color='black',label='H: Hopf')]+[Line2D([0],[0],marker=m,ls='',mfc='white',mec='black',label=l) for m,l in [('o','LP: equilibrium fold'),('s','LPC: checks pending'),('v','PD: checks pending'),('D','TR: torus')]]+
        [Line2D([0],[0],marker=m,ls='',mfc='black',mec='black',label=l) for m,l in [('s','LPC: orbit + mode checked'),('v','PD: orbit + mode checked')]],loc='lower right',frameon=True,facecolor='white',edgecolor='none',framealpha=1,fontsize=8)
    pointmeta=[]
    for i,(letter,name,J,title) in enumerate(CASES):
        r,T=loadcase(s,name,J);N=len(r);reg=np.array([s.regional_rates(v) for v in r]);t=np.arange(N)*T/N
        mean=reg.mean(0);ax.scatter(J,mean[0],s=28,c='white',edgecolor='black',zorder=10)
        off={'a':(-16,-17),'b':(-8,12),'c':(-24,4),'d':(12,12),'e':(0,12)}[letter]
        ax.annotate(letter,(J,mean[0]),xytext=off,textcoords='offset points',fontsize=12,weight='bold')
        pointmeta.append(dict(case=letter,J_EE_core=J,orbit=name,title=title,T_ms=T if name else None,mean_rates_hz=mean))
        w=fig.add_subplot(grid[i,1])
        # Center phase on the minimum of summed core activity, so both burst peaks are visible.
        shift=int(np.argmin(reg[:,:2].sum(1)));rr=np.roll(r,-shift,axis=0);rg=np.roll(reg,-shift,axis=0)
        for k in [0,1]:w.plot(t,rg[:,k],color=COL[k],lw=1.1)
        w.plot(t,rg[:,2],color='#555555',lw=.75)
        w.set_xlim(0,T);w.set_ylim(bottom=0);w.set_ylabel('Hz / cell');w.set_xlabel('Time (ms)');style(w)
        w.set_title(f'{letter}  {title}\n'+rf'$J_{{\mathrm{{EE,core}}}}={J:g}$'+(f'  |  T={T:.1f} ms' if name else ''),loc='left',fontsize=10)
        sub=grid[i,2].subgridspec(1,4,wspace=.20)
        if name is None or letter=='b':ids=(np.arange(4)*N//4).tolist()
        elif letter=='c':
            # Both distinct lead orders within one full period; two frames per burst.
            ids=[]
            for k in [0,1]:
                peaks=find_peaks(np.tile(rg[:,k],3),height=20,distance=N//4)[0]
                peaks=peaks[(peaks>=N)&(peaks<2*N)]-N
                if len(peaks)!=2:raise RuntimeError('Expected two bursts per core in the exact double cycle')
                ids.extend(peaks.tolist())
            ids=sorted(ids)
        else:
            pk=int(np.argmax(rg[:,1 if letter=='d' else 0]));ids=[int((pk+offset/T*N)%N) for offset in [-20,0,40,80]]
        for j,ix in enumerate(ids):
            fld=np.bincount(cell[s.E],weights=sz[s.E]*rr[ix,s.E]*1000,minlength=400)/np.maximum(ct,1)
            f=fig.add_subplot(sub[j]);imf=f.imshow(fld.reshape(20,20),origin='lower',extent=(0,20,0,20),cmap='inferno',norm=PowerNorm(.55,0,500))
            for center in s.geo['centers_mm']:f.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
            f.scatter(xy[:,0],xy[:,1],s=7,facecolors='none',edgecolors='cyan',linewidths=.55)
            f.set(xticks=[0,20],yticks=[0,20],title=f'{t[ix]:.0f} ms');f.tick_params(labelsize=8)
            if j: f.tick_params(labelleft=False)
            if i==4:f.set_xlabel('x (mm)')
            if j==0:f.set_ylabel('y (mm)')
        c=fig.add_subplot(grid[i,3]);contact=rr@s.geo['contact_rate_weights']*1000
        im=c.imshow(contact[:,order].T,origin='upper',aspect='auto',extent=(0,T,14.5,-.5),cmap='magma',norm=PowerNorm(.5,0,200))
        c.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)');c.tick_params(axis='y',labelsize=7,length=2);c.axhline(3.5,color='white',lw=.6)
        for tick,n in zip(c.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
        if i==0:c.set_title('D  SEEG-site rate readout',loc='left',weight='bold')
    fig.text(.326,.956,'B  Core / surround activity',weight='bold',fontsize=13)
    fig.text(.527,.956,'C  Same-orbit spatial propagation',weight='bold',fontsize=13)
    handles=[Line2D([0],[0],color=COL[k],label=f'Core {"AB"[k]}') for k in [0,1]]+[Line2D([0],[0],color='#555555',label='Surround')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.32,.008),ncol=1,frameon=False,fontsize=9)
    fig.colorbar(imf,cax=fig.add_axes([.55,.037,.18,.011]),orientation='horizontal',label='E rate (Hz / cell)')
    fig.colorbar(im,cax=fig.add_axes([.82,.037,.13,.011]),orientation='horizontal',label='Contact-weighted rate (Hz / cell)')
    legend=[Line2D([0],[0],color=FAMILY[n],label=l) for n,l in [('A','H1 cycle'),('B','H2 cycle'),('double','Alternating'),('single','A-leading'),('Bleading','B-leading')]]
    if 'PDchild' in fs:legend.append(Line2D([0],[0],color=FAMILY['PDchild'],label='PD1 child'))
    if 'PDupperchild' in fs:legend.append(Line2D([0],[0],color=FAMILY['PDupperchild'],label='PD2 child'))
    if 'PDreturnchild' in fs:legend.append(Line2D([0],[0],color=FAMILY['PDreturnchild'],label='PD3 child'))
    fig.legend(handles=legend,loc='lower left',bbox_to_anchor=(.047,.008),ncol=3,
        frameon=False,fontsize=8.5,columnspacing=.8,handletextpad=.5,handlelength=1.7)
    save(fig,a.output)
    metadata='composite_cases.json' if a.output=='spatial_rate_periodic_composite' else a.output+'_cases.json'
    write(PERIODIC_OUT/metadata,dict(cases=pointmeta,source='All panels: same autonomous spatial rate DDE. Equilibrium / periodic BVP; no SNN trajectories or startup transients.',
      periodic_branch_coverage={family:dict(points=len(rows),J_min=min(q['J_EE_core'] for q in rows),J_max=max(q['J_EE_core'] for q in rows)) for family,rows in fs.items()},
      branch_lines='Colored lines give converged periodic mean (thick) and min/max (thin); stability coverage is separate.',
      LPC_markers='Filled squares require both independently checked modes and a positive constituent-filter profile at the same root. Open squares retain located folds with pending checks. Neither symbol establishes adjacent orbit stability.',
      PD_markers='Filled triangles require independently checked PD modes and positive constituent filters in the plotted parent. Open triangles retain locations requiring follow-up; a question mark additionally denotes the existing PD resolution review. The unresolved secondary-PD candidate is not plotted.',
      readout='Contact-weighted firing-rate envelope, not electrical SEEG voltage.',field_scale='Shared across all cases; low-rate fields appear dark.',
      case_resolution_source=str(PERIODIC_OUT/'composite_case_resolution.json')))
    update_readme({a.output:'左右均来自同一空间 rate DDE，a–e 为平衡解或周期解；代表轨道除连续方程残差外，也检查两个构成放电率的滤波状态非负，c 行在同一周期展示两种核领先顺序。实心方块或三角要求对应母轨道通过物理状态检查且临界模态独立通过验证，空心标记保留仍需补测的位置；新增欠分辨的二次 PD 候选不纳入已确认标记。**关注点**：彩色线表示已计算分支几何，不能据其线型判断整段稳定性；右列是 SEEG 位置的加权放电率，整张图仍为待人工审阅的部分完成版本。'})

if __name__=='__main__':main()
