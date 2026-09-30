"""Checked b/c spectra, waveforms and spatial/contact activity at identical J."""
from plot_rate_branch_completion import *
from audit_rate_survey_filter_states import fingerprint


def main():
    source=DATA/'sameJ_small_burst_stability_readout.json';audit=read(source)
    assert audit['status']=='SAME_J_TWO_NUMERICALLY_STABLE_PERIODIC_STATES'
    s=RateField();cases={q['letter']:q for q in load_cases(s)}
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    names=geo['contact_names'].tolist();assert names==audit['contact_names']
    order=contact_indices(names);cell=s.geo['group_cell'];size=s.geo['group_size']
    count=np.bincount(cell[s.E],weights=size[s.E],minlength=400)
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(19.5,7.4))
    grid=fig.add_gridspec(2,4,width_ratios=[.95,1.25,2.25,1.4],
        left=.045,right=.99,top=.83,bottom=.19,hspace=.65,wspace=.38)
    fields=plt.get_cmap('inferno').copy();contacts=plt.get_cmap('magma').copy()
    for cmap in [fields,contacts]:cmap.set_bad('black');cmap.set_under('black')
    displayed=[]
    for i,evidence in enumerate(audit['rows']):
        q=cases[evidence['case']];r=q['r'];rg=q['regional'];N=len(r);T=q['T'];t=q['time']
        assert fingerprint(q['orbit'])==evidence['profile_fingerprint']
        assert q['J']==audit['J_EE_core']
        mu=values(evidence['classification']);color='#159b83' if i==0 else '#dc6646'
        ax=fig.add_subplot(grid[i,0]);theta=np.linspace(0,2*np.pi,400)
        ax.plot(np.cos(theta),np.sin(theta),'--',color='#222222',lw=.85)
        ax.axhline(0,color='#aaaaaa',lw=.5);ax.axvline(0,color='#aaaaaa',lw=.5)
        ax.scatter(mu.real,mu.imag,s=38,color=color,edgecolors='white',linewidths=.4,zorder=3)
        ax.set(xlim=(-1.12,1.12),ylim=(-1.12,1.12),aspect='equal',
            xticks=[-1,0,1],yticks=[-1,0,1],xlabel=r'Re $\mu$',ylabel=r'Im $\mu$',
            title='Largest returned modulus\n'+rf'$|\mu|={max(abs(mu)):.3f}$')
        style(ax)
        ax=fig.add_subplot(grid[i,1])
        for k,c in enumerate([*COL,'#555555']):ax.plot(t,rg[:,k],color=c,lw=1.35 if k<2 else .8)
        ax.set(xlim=(0,T),ylim=(0,max(.9,rg[:,:2].max()*1.1)),
            xlabel='Time (ms)',ylabel='Rate (Hz / E cell)',
            title=f'{q["letter"]}  {q["title"]}\nT = {T:.1f} ms')
        style(ax)
        if i==0:ids=(np.arange(4)*N//4).tolist()
        else:
            ids=[]
            for k in [0,1]:
                peaks=find_peaks(np.tile(rg[:,k],3),height=20,distance=N//4)[0]
                peaks=peaks[(peaks>=N)&(peaks<2*N)]-N
                assert len(peaks)==2
                ids.extend(peaks.tolist())
            ids=sorted(ids)
        sub=grid[i,2].subgridspec(1,4,wspace=.16)
        for j,ix in enumerate(ids):
            field=np.bincount(cell[s.E],weights=size[s.E]*r[ix,s.E]*1000,minlength=400)/np.maximum(count,1)
            ax=fig.add_subplot(sub[j]);imf=ax.imshow(field.reshape(20,20),origin='lower',extent=(0,20,0,20),
                cmap=fields,norm=LogNorm(.03,500))
            for center in s.geo['centers_mm']:ax.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
            ax.scatter(geo['contact_xy'][:,0],geo['contact_xy'][:,1],s=8,facecolors='none',edgecolors='cyan',linewidths=.55)
            ax.set(xticks=[0,20],yticks=[0,20],title=f'{t[ix]:.0f} ms');ax.tick_params(labelsize=8)
            if j:ax.tick_params(labelleft=False)
            else:ax.set_ylabel('y (mm)',fontsize=9)
            if i==1:ax.set_xlabel('x (mm)',fontsize=9)
        ax=fig.add_subplot(grid[i,3]);readout=r@s.geo['contact_rate_weights']*1000
        imc=ax.imshow(readout[:,order].T,origin='upper',aspect='auto',extent=(0,T,14.5,-.5),
            cmap=contacts,norm=LogNorm(.03,200))
        ax.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)',
            title='No SCL detections' if i==0 else 'SCL in one of two events / period')
        ax.tick_params(axis='y',labelsize=7,length=2);ax.axhline(3.5,color='white',lw=.6)
        for tick,name in zip(ax.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[name[:3]])
        displayed.append(dict(case=q['letter'],orbit=q['orbit'],snapshots_ms=t[ids]))
    fig.suptitle(r'Same spatial rate model, $J_{\mathrm{EE,core}}=0.942$: two locally stable periodic states',
        fontsize=15,y=.985)
    for x,title in [(.045,'A  Checked Floquet multipliers'),(.224,'B  Core / surround activity'),
                    (.46,'C  Same-orbit spatial activity'),(.812,'D  Contact-weighted rate')]:
        fig.text(x,.91,title,weight='bold',fontsize=11)
    fig.legend(handles=[Line2D([0],[0],color=c,label=n) for c,n in
        zip([*COL,'#555555'],['Core A','Core B','Surround'])],
        loc='lower left',bbox_to_anchor=(.22,.025),ncol=3,frameon=False,fontsize=9)
    fig.colorbar(imf,cax=fig.add_axes([.48,.095,.26,.014]),orientation='horizontal',label='E rate (Hz / cell; log scale)')
    fig.colorbar(imc,cax=fig.add_axes([.82,.095,.16,.014]),orientation='horizontal',label='Contact rate (Hz / cell; log scale)')
    name='sameJ_stable_small_and_burst';save_new(fig,name)
    write(DATA/(name+'.json'),dict(source=str(source),displayed_cases=displayed,
        model=audit['model'],observer=audit['observer_source'],
        phase_mode_removed=True,field_scales_shared_across_rows=True,
        spectral_display='Returned checked nontrivial multipliers, not every stable eigenvalue. The polynomial filter suppresses a stable spectral cluster; stability uses the separate outside-unit-disk coverage check.',
        scope='Numerical local stability and different readout on two physical cycles at the same J. No basin boundary, spontaneous switching or connecting bifurcation is established.'))
    path=OUTPUT/'figures/README.md';body=path.read_text()
    body=re.sub(r'^### '+name+r'\.png\s*\n.*?(?=^### |\Z)','',body,flags=re.M|re.S).rstrip()
    body+='\n\n### '+name+'.png\n在同一 J=0.942 下，展示主图 b、c 两条经过物理状态及双步长谱检查的周期解：非平凡 Floquet 乘子、两核与 surround 波形、400 格二维场和固定触点率读出。两者都获得数值局部稳定性支持；交替 burst 每周期的两个合格事件中有一个包含 SCL，小振荡没有 SCL 检出，也没有合格群事件。左列显示返回并核验的乘子，最大值仅针对返回谱；稳定性另以单位圆外谱覆盖检查判断。**关注点**：这是小振荡与 burst 的模型内数值共存，不能表述为 resting 与 burst 共存，也尚未确定吸引域边界、状态切换机制或原生 SNN 共存。\n'
    path.write_text(body)


if __name__=='__main__':
    main()
