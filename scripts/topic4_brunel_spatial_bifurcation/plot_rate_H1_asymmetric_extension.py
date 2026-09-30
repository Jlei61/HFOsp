"""Same-rate spatial readout of the checked, strongly asymmetric H1 segment."""
from plot_rate_branch_completion import *


def main():
    source=DATA/'H1_to_PD3_display_extension.json'
    progress=read(source)
    # A fixed delivered example, not a moving last-point selection.
    rows=[q for q in progress['rows'] if 85<=q['index']<=98]
    assert [q['index'] for q in rows]==list(range(85,99))
    assert all(q['status']=='PASS' and q['branch_match_pass'] and
               q['check']['filter_state_check']['positive'] and
               q['check']['maximum_group_defect_Hz']<.001 for q in rows)
    metadata=[read(Path(q['orbit']).with_suffix('.json')) for q in rows]
    s=RateField();z=np.load(rows[-1]['orbit']);r=z['r'];T=float(z['T']);N=len(r)
    regional=np.array([s.regional_rates(x) for x in r])
    shift=int(np.argmin(regional[:,:2].sum(1)))
    r=np.roll(r,-shift,axis=0);regional=np.roll(regional,-shift,axis=0)
    t=np.arange(N)*T/N;contact=r@s.geo['contact_rate_weights']*1000
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    order=contact_indices(geo['contact_names'].tolist())
    mass=s.geo['group_size'];cell=s.geo['group_cell'];e=s.E
    count=np.bincount(cell[e],weights=mass[e],minlength=400)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(15.2,7.6))
    grid=fig.add_gridspec(2,3,width_ratios=[1,1.45,1.25],left=.07,right=.98,
                        bottom=.13,top=.88,hspace=.57,wspace=.38)
    J=np.array([q['J_EE_core'] for q in metadata])
    mean=np.array([q['mean_rates_hz'] for q in metadata])
    ax=fig.add_subplot(grid[0,0])
    for k,c in enumerate(COL):
        ax.plot(J,mean[:,k],':',color=c,lw=1.8,label=f'Core {"AB"[k]}')
        ax.plot(J[-1],mean[-1,k],'o',mfc='white',mec=c,ms=5)
    ax.set(xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Period mean (Hz / E cell)',
           title='A   Checked H1 continuation')
    ax.legend(frameon=False,fontsize=8,loc='upper left');style(ax)
    ax=fig.add_subplot(grid[0,1])
    ax.plot(J,[q['T_ms'] for q in metadata],':',color='#168469',lw=1.8)
    ax.plot(J[-1],T,'o',mfc='white',mec='#168469',ms=5)
    ax.set(xlabel=r'$J_{\mathrm{EE,core}}$',ylabel='Full-network period (ms)',
           title='B   Period along the same segment');style(ax)
    ax=fig.add_subplot(grid[0,2])
    for k,c in enumerate([*COL,'#555555']):
        ax.plot(t,regional[:,k],color=c,lw=1.2,label=['Core A','Core B','Surround'][k])
    ax.set(xlim=(0,T),xlabel='Time (ms)',ylabel='Rate (Hz / E cell)',
           title='C   Actual example cycle')
    ax.legend(frameon=False,fontsize=8);style(ax)
    fieldgrid=grid[1,:2].subgridspec(1,4,wspace=.18)
    peak=int(np.argmax(regional[:,0]))
    ids=[(peak+int(round(dt/T*N)))%N for dt in [-20,0,20,40]]
    for i,idx in enumerate(ids):
        field=np.bincount(cell[e],weights=mass[e]*r[idx,e]*1000,
                          minlength=400)/np.maximum(count,1)
        ax=fig.add_subplot(fieldgrid[i])
        im=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),
            cmap='inferno',norm=LogNorm(.03,500))
        for center in s.geo['centers_mm']:
            ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
        ax.scatter(geo['contact_xy'][:,0],geo['contact_xy'][:,1],s=11,
                   facecolors='none',edgecolors='cyan',linewidths=.7)
        ax.set(xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)',title=f'{t[idx]:.1f} ms')
        if i==0:ax.set_ylabel('y (mm)')
        else:ax.tick_params(labelleft=False)
    ax=fig.add_subplot(grid[1,2])
    ci=ax.imshow(contact[:,order].T,origin='upper',aspect='auto',
        extent=(0,T,14.5,-.5),cmap='magma',norm=LogNorm(.03,200))
    ax.axhline(3.5,color='white',lw=.7)
    ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)',
           title='E   Same-cycle contact-rate readout')
    ax.tick_params(axis='y',length=2,labelsize=7)
    for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
    fig.text(.07,.452,'D   Same-cycle spatial field',weight='bold')
    fig.colorbar(im,cax=fig.add_axes([.16,.06,.40,.013]),orientation='horizontal',
                 label='E rate (Hz / cell; log scale)')
    fig.colorbar(ci,cax=fig.add_axes([.77,.06,.18,.013]),orientation='horizontal',
                 label='Contact-weighted rate (Hz / cell)')
    fig.suptitle('H1: asymmetric core oscillation | Stability unclassified\n'+
        rf'Example $J_{{\mathrm{{EE,core}}}}={float(z["J"]):.7f}$'+f' | T={T:.2f} ms',
        fontsize=13,y=.985)
    name='H1_asymmetric_core_extension';save_new(fig,name)
    write(DATA/(name+'.json'),dict(source=str(source),indices=list(range(85,99)),
        example_index=98,example_orbit=rows[-1]['orbit'],J_EE_core=float(z['J']),T_ms=T,
        regional_mean_Hz=regional.mean(0),regional_minimum_Hz=regional.min(0),
        regional_maximum_Hz=regional.max(0),phase_shift_samples=shift,
        snapshot_times_ms=t[ids],selection='Checked H1 segment with Core A amplitude much larger than Core B; fixed endpoint 98 illustrates this state without adding a critical marker to the main figure.',
        scope='Asymmetric periodic rate activity in the existing continuation. Stability unclassified, not a new bifurcation or an established connection to the other burst families. The spatial field and contact rates are from this same orbit; no electrical SEEG voltage or event-rank claim.'))
    path=OUTPUT/'figures/README.md';body=path.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    body+='\n\n### '+name+'.png\n展示 H1 返回路径已核验的第85–98点及第98点的完整周期波形、二维场和固定触点率读出，Core A 的起伏明显大于 Core B。示例固定于本次交付的第98点，空心圆只对应示例，不代表临界点。**关注点**：这是同一分支上的不对称周期活动，稳定性未分类；不能由此证明新的分岔、连接到其他 burst 分支或恢复患者传播统计。\n'
    path.write_text(body)


if __name__=='__main__':main()
