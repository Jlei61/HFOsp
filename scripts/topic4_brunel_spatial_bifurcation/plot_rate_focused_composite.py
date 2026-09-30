"""Focused, all-rate figures with explicit sampled stability evidence.

Keep primary cycle families and period means in the composite. Put the
Hopf neighbourhood, peak/trough envelopes and extended families on separate
axes. No interpolation of sampled Floquet verdicts into stable intervals.
"""
from plot_rate_periodic_composite import *
from matplotlib.ticker import FixedLocator, FuncFormatter
from audit_rate_survey_filter_states import fingerprint
from datetime import datetime, timezone
import hashlib

DEST=ROOT/'results/topic4_sef_hfo/interictal_rate_bifurcation_focus_20260919'
FIG=DEST/'figures'
NAMES={'A':'H1 small cycle','B':'H2 small cycle','double':'Alternating bursts',
       'Bleading':'B-leading bursts','single':'A-leading bursts'}
LINESTYLE=(0,(1.3,2.1))
CONTINUATION_LABEL='Stability unclassified along line'
SHOW_PD=False
EXTRA_RETURN_FOLDS=[]
H2_RETURN_ROWS=None
EXACT_STABILITY_SAMPLES=[]


def export(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    for ext in ['png','pdf','svg']:
        fig.savefig(FIG/f'{name}.{ext}',dpi=210,bbox_inches='tight')
    plt.close(fig)


def hopf_departures(fs):
    roots={q['label']:q for q in critical()};out={}
    for name,label,limit in [('A','LPC_A1',289),('B','LPC_B1',166)]:
        root=roots[label];rows=fs[name][:limit]
        index=min(range(len(rows)),key=lambda i:abs(rows[i]['J_EE_core']-root['J_EE_core'])+
                  abs(rows[i]['T_ms']-root['T_ms'])/1000)
        out[name]=rows[:index+1]
    return out


def primary(fs):
    # Keep the local Hopf departures, and the original traced burst families.
    # Later H1/H2 returns and the upper A-leading continuation stay in the
    # companion figures rather than disappearing from the source inventory.
    return dict(**hopf_departures(fs),
        double=fs['double'],Bleading=fs['Bleading'],
        single=[q for q in fs['single'] if not Path(q['path']).stem.startswith('arcSingleUpperConnect_')])


def physical_stability_samples():
    folder=PERIODIC_OUT/'stability_coverage'
    review=read(folder/'constituent_filter_audit.json');out={}
    for row in review['rows']:
        source=Path(row['result_source'])
        if not source.exists():continue
        q=read(source)
        actual=Path(q.get('analyzed_orbit',row['orbit']))
        # A live worker may have replaced a result since the last audit.
        # Never apply the old profile check to its new orbit.
        if not actual.exists() or row['profile_fingerprint']!=fingerprint(actual):continue
        if not row['filter_state_check']['positive']:continue
        if q['status'] not in ['NUMERICALLY_STABLE','UNSTABLE']:continue
        orbit=q.get('orbit')
        if orbit:out[str(Path(orbit).resolve())]=dict(status=q['status'],source=str(source),orbit=str(actual))
    return out


def cycle_line(ax,rows,k,color,stat='mean_rates_hz',width=1.6,alpha=1.):
    bounds=[0,*continuation_breaks(rows),len(rows)]
    for start,stop in zip(bounds[:-1],bounds[1:]):
        rr=rows[start:stop]
        ax.plot([q['J_EE_core'] for q in rr],[q[stat][k] for q in rr],
                color=color,lw=width,ls=LINESTYLE,alpha=alpha,zorder=2)


def equilibrium(ax,k,max_J=1.018):
    z=np.load(RATE_OUT/'equilibrium_branch.npz');j=z['J'];h=read(RATE_OUT/'hopfs.json')['rows'][0]
    # Stop at the first equilibrium fold; secondary stationary branches are
    # part of the complete inventory, not this low-activity onset display.
    turns=np.flatnonzero(np.diff(j)[:-1]*np.diff(j)[1:]<0)+1
    end=int(turns[0])+1;j=j[:end];y=z['regional'][:end,k]
    cut=np.flatnonzero(j>=h['J_EE_core'])[0]
    ax.plot(np.r_[j[:cut],h['J_EE_core']],np.r_[y[:cut],h['rates_hz'][k]],color='#292929',lw=1.7,zorder=3)
    # Draw the post-Hopf low equilibrium as unclassified geometry. Exact
    # positive-root sites are shown independently, never filled between them.
    ax.plot(np.r_[h['J_EE_core'],j[cut:]],np.r_[h['rates_hz'][k],y[cut:]],
            color='#777777',lw=1.1,ls=LINESTYLE,zorder=1)


def draw_samples(ax,fs,k,evidence):
    for name,rows in fs.items():
        for q in rows:
            ev=evidence.get(str(Path(q['path']).resolve()))
            if ev is None:continue
            # Classify only the exact fine profile, not a coarse plotted
            # sample when the refinement changed its displayed mean.
            meta=read(Path(ev['orbit']).with_suffix('.json'))
            if ev['status']=='NUMERICALLY_STABLE':
                ax.plot(meta['J_EE_core'],meta['mean_rates_hz'][k],'o',ms=4.0,
                        color=FAMILY[name],mec='white',mew=.45,zorder=6)
            else:
                ax.plot(meta['J_EE_core'],meta['mean_rates_hz'][k],'x',ms=4.2,
                        color=FAMILY[name],mew=1.,zorder=5)
    # Exact witnesses need not coincide with a continuation mesh point.
    # Plot them independently; never insert or connect them by J sorting.
    displayed={str(Path(q['path']).resolve()) for rows in fs.values() for q in rows}
    for sample in EXACT_STABILITY_SAMPLES:
        if sample['family'] not in fs or str(Path(sample['orbit']).resolve()) in displayed:
            continue
        stable=sample['status']=='NUMERICALLY_STABLE'
        ax.plot(sample['J_EE_core'],sample['mean_rates_hz'][k],
                'o' if stable else 'x',ms=4.0 if stable else 4.2,
                color=FAMILY[sample['family']],
                mec='white' if stable else FAMILY[sample['family']],
                mew=.45 if stable else 1.,zorder=6 if stable else 5)


def cycle_critical(ax,k,names,annotate=True,annotation_offsets=None):
    physical={str(Path(q['orbit']).resolve()):q for q in
              read(PERIODIC_OUT/'rate_filter_state_positivity_audit.json')['rows']}
    for q in critical():
        if q['label'] not in names:continue
        meta=read(Path(q['orbit']).with_suffix('.json'));label=CRITICAL_LABELS[q['label']]
        source=PERIODIC_OUT/(q['label']+'_validation.json')
        v=read(source) if source.exists() else {}
        root=v.get('mesh_checks',[{}])[-1]
        matching=(root.get('N')==q['N'] and
            Path(root.get('orbit','')).resolve()==Path(q['orbit']).resolve() and
            abs(v.get('J_EE_core',float('inf'))-q['J_EE_core'])<1e-10 and
            abs(v.get('T_ms',float('inf'))-q['T_ms'])<1e-7)
        cached=physical.get(str(Path(q['orbit']).resolve()),{})
        positive=cached.get('positive',False) and cached.get('N')==q['N']
        continuous=v.get('continuous_defect',{})
        if matching and Path(continuous.get('orbit','')).resolve()==Path(q['orbit']).resolve():
            positive=continuous.get('filter_state_check',{}).get('positive',positive)
        passed=v.get('status')=='VALIDATED_CYCLE_FOLD' and matching and positive
        marker='s'
        if q['label'].startswith('PD'):
            parent=v.get('filter_state_followup',{})
            parent_profile=parent.get('continuous_check',parent.get('continuous_orbit_check',{}))
            parent_positive=(Path(parent_profile.get('orbit','')).resolve()==Path(q['orbit']).resolve() and
                             parent_profile.get('filter_state_check',{}).get('positive',False))
            direct=v.get('continuous_orbit_check',{})
            direct_positive=(Path(direct.get('orbit','')).resolve()==Path(q['orbit']).resolve() and
                             direct.get('filter_state_check',{}).get('positive',False))
            passed=(v.get('full_acceptance',False) and
                Path(v.get('accepted_parent_orbit','')).resolve()==Path(q['orbit']).resolve() and
                (parent_positive or
                 direct_positive or
                 (cached.get('positive',False) and cached.get('N')==q.get('accepted_parent_mesh_N',q['N']))))
            marker='v'
            if passed and v.get('criticality') in ['SUBCRITICAL_PD','SUPERCRITICAL_PD']:
                label+='\n'+v['criticality'].split('_')[0].lower()
        ax.plot(q['J_EE_core'],meta['mean_rates_hz'][k],marker,ms=5,
                mfc='#222222' if passed else 'white',mec='#222222',mew=.9,zorder=7)
        if annotate:
            offsets={'LPC_double_low':(-34,-38),'LPC_double_high':(-15,18),
                     'LPC_Bleading_low':(15,19),'LPC_burst_low':(12,-20),'LPC_burst_high':(-48,-22),
                     'LPC_A1':(-30,12),'LPC_B1':(-30,-22),
                     'LPC_B2':(-48,-28),
                     'LPC_A_stage4_turn1':(12,22),'LPC_A_stage4_turn2':(12,-23),
                     'LPC_A_stage4_turn3':(15,-12)}
            offsets.update(PD_double_low=(-43,-25),PD_double_upper=(-55,30),PD_H2_after_LPC13=(20,32))
            if k==0:offsets.update(LPC_B2=(42,-6),PD_H2_after_LPC13=(42,26))
            else:offsets.update(PD_H2_after_LPC13=(42,10),LPC_A_low_extension=(10,-18))
            if annotation_offsets:offsets.update(annotation_offsets)
            offset=offsets.get(q['label'],(10,12))
            ax.annotate(label,(q['J_EE_core'],meta['mean_rates_hz'][k]),xytext=offset,
                textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.6,color='#333333'))


def mean_axis(ax,k,selected,evidence,case_data=None):
    equilibrium(ax,k)
    for name,rr in selected.items():cycle_line(ax,rr,k,FAMILY[name])
    draw_samples(ax,selected,k,evidence)
    # The upper fold is separated enough for a clear label at this scale;
    # the tightly spaced low-J folds are labelled in the onset companion.
    cycle_critical(ax,k,['LPC_burst_high'])
    if SHOW_PD:cycle_critical(ax,k,['PD_double_low','PD_double_upper']+
                             (['PD_H2_after_LPC13'] if H2_RETURN_ROWS is not None else []))
    includes_return=min(q['J_EE_core'] for q in selected['A'])<.8
    if includes_return:cycle_critical(ax,k,['LPC_A1','LPC_A_low_extension',*EXTRA_RETURN_FOLDS])
    ax.set(xlim=(.695 if includes_return else .912,1.662),ylim=(.52,160),yscale='log',
        xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=rf'$\langle r_{{{ "AB"[k] }}}\rangle$ (Hz / E cell)')
    ticks=([.7] if includes_return else [])+[.95,1.1,1.3,1.5,1.65]
    ax.set_xticks(ticks);ax.set_xticklabels([f'{value:.2f}' for value in ticks])
    ax.yaxis.set_major_locator(FixedLocator([.5,1,2,5,10,20,50,100]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
    ax.tick_params(axis='both',which='minor',length=0);style(ax)
    if case_data:
        offsets={'a':(14,-9),'b':(30,13),'c':(16,-6),'d':(15,16),'e':(10,12)}
        if k==1:offsets.update(b=(30,5),c=(16,18),d=(16,-18))
        for q in case_data:
            J=q['J'];y=q['regional'].mean(0)[k]
            ax.plot(J,y,'o',mfc='white',mec='#222222',ms=4.5,zorder=8)
            ax.annotate(q['letter'],(J,y),xytext=offsets[q['letter']],textcoords='offset points',
                weight='bold',fontsize=11,arrowprops=dict(arrowstyle='-',lw=.65,color='#333333'),
                bbox=dict(facecolor='white',edgecolor='none',pad=.35),zorder=10)


def legends(fig):
    family=[Line2D([0],[0],color=FAMILY[n],lw=1.8,ls=LINESTYLE,label=NAMES[n]) for n in NAMES]
    fig.legend(handles=family,loc='lower left',bbox_to_anchor=(.036,.022),ncol=2,frameon=False,
               fontsize=8.2,columnspacing=.8,handlelength=2)
    status=[Line2D([0],[0],color='#222222',lw=1.6,label='Stable equilibrium'),
            Line2D([0],[0],color='#555555',ls=LINESTYLE,lw=1.4,label=CONTINUATION_LABEL),
            Line2D([0],[0],marker='o',ls='',color='#222222',ms=4,label='Stable cycle: checked sample'),
            Line2D([0],[0],marker='x',ls='',color='#222222',ms=5,label='Unstable cycle: checked sample'),
            Line2D([0],[0],marker='s',ls='',color='#222222',ms=5,label='Cycle fold: checked'),
            Line2D([0],[0],marker='s',mfc='white',mec='#222222',ls='',ms=5,label='Fold: checks pending')]
    if SHOW_PD:
        status.extend([Line2D([0],[0],marker='v',ls='',color='#222222',ms=5,label='PD: parent and mode checked'),
            Line2D([0],[0],marker='v',ls='',mfc='white',mec='#222222',ms=5,label='PD: checks pending')])
    return status


def load_cases(s):
    rows=[]
    manifest=read(PERIODIC_OUT/'composite_case_resolution.json')
    for letter,name,J,title in CASES:
        r,T=loadcase(s,name,J);regional=np.array([s.regional_rates(x) for x in r]);N=len(r)
        shift=int(np.argmin(regional[:,:2].sum(1)))
        actual=next((q['orbit'] for q in manifest['rows'] if q['case']==letter),None)
        rows.append(dict(letter=letter,name=name,J=J,title=title,r=np.roll(r,-shift,axis=0),
            regional=np.roll(regional,-shift,axis=0),T=T,time=np.arange(N)*T/N,orbit=actual))
    return rows


def composite(s,selected,evidence,cases,log_readout=False):
    from matplotlib.colors import LogNorm
    geo=np.load(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1/replay/geometry.npz')
    xy=geo['contact_xy'];order=contact_indices(geo['contact_names'].tolist())
    cell=s.geo['group_cell'];sz=s.geo['group_size'];ct=np.bincount(cell[s.E],weights=sz[s.E],minlength=400)
    field_cmap=plt.get_cmap('inferno').copy();contact_cmap=plt.get_cmap('magma').copy()
    for cmap in [field_cmap,contact_cmap]:cmap.set_bad('black');cmap.set_under('black')
    fig=plt.figure(figsize=(22,12.6))
    grid=fig.add_gridspec(5,4,width_ratios=[1.3,1,1.6,1.2],left=.048,right=.985,
                       bottom=.16,top=.915,wspace=.39,hspace=.78)
    left=grid[:,0].subgridspec(2,1,hspace=.40)
    for k in [0,1]:
        ax=fig.add_subplot(left[k]);mean_axis(ax,k,selected,evidence,cases)
        ax.set_title(f'A{k+1}  Core {"AB"[k]}: period mean',loc='left',weight='bold',fontsize=12)
    for i,q in enumerate(cases):
        r=q['r'];rg=q['regional'];T=q['T'];t=q['time'];N=len(r);letter=q['letter']
        ax=fig.add_subplot(grid[i,1])
        for k,c in enumerate([*COL,'#555555']):ax.plot(t,rg[:,k],color=c,lw=1.25 if k<2 else .8)
        ax.set(xlim=(0,T),ylim=(0,max(rg[:,:2].max()*1.10,.9)),ylabel='Hz / E cell',xlabel='Time (ms)')
        ax.set_title(f'{letter}  {q["title"]}\n'+rf'$J_{{\mathrm{{EE,core}}}}={q["J"]:g}$'+
            (f'  |  T={T:.1f} ms' if q['name'] else ''),loc='left',fontsize=10);style(ax)
        if q['name'] is None or letter=='b':ids=(np.arange(4)*N//4).tolist()
        elif letter=='c':
            ids=[]
            for k in [0,1]:
                peaks=find_peaks(np.tile(rg[:,k],3),height=20,distance=N//4)[0]
                peaks=peaks[(peaks>=N)&(peaks<2*N)]-N
                assert len(peaks)==2,'Retain both lead orders of the exact full-network period'
                ids.extend(peaks.tolist())
            ids=sorted(ids)
        else:
            peak=int(np.argmax(rg[:,1 if letter=='d' else 0]));ids=[int((peak+dt/T*N)%N) for dt in [-20,0,40,80]]
        sub=grid[i,2].subgridspec(1,4,wspace=.16)
        for j,index in enumerate(ids):
            fld=np.bincount(cell[s.E],weights=sz[s.E]*r[index,s.E]*1000,minlength=400)/np.maximum(ct,1)
            f=fig.add_subplot(sub[j]);imf=f.imshow(fld.reshape(20,20),origin='lower',extent=(0,20,0,20),
                cmap=field_cmap,norm=LogNorm(.03,500) if log_readout else PowerNorm(.55,0,500))
            for center in s.geo['centers_mm']:f.add_patch(plt.Circle(center,1.5,fill=False,color='white',lw=.8))
            f.scatter(xy[:,0],xy[:,1],s=7,facecolors='none',edgecolors='cyan',linewidths=.55)
            f.set(xticks=[0,20],yticks=[0,20],title=f'{t[index]:.0f} ms');f.tick_params(labelsize=8)
            if j:f.tick_params(labelleft=False)
            else:f.set_ylabel('y (mm)')
            if i==4:f.set_xlabel('x (mm)')
        c=fig.add_subplot(grid[i,3]);contact=r@s.geo['contact_rate_weights']*1000
        im=c.imshow(contact[:,order].T,origin='upper',aspect='auto',extent=(0,T,14.5,-.5),
            cmap=contact_cmap,norm=LogNorm(.03,200) if log_readout else PowerNorm(.5,0,200))
        c.set(yticks=np.arange(15),yticklabels=CONTACT_ORDER,xlabel='Time (ms)')
        c.tick_params(axis='y',labelsize=7,length=2);c.axhline(3.5,color='white',lw=.6)
        for tick,n in zip(c.get_yticklabels(),CONTACT_ORDER):tick.set_color(SHAFT_COLORS[n[:3]])
        q['snapshot_times_ms']=[float(t[index]) for index in ids]
    for x,label in [(.318,'B  Core / surround activity'),(.531,'C  Same-orbit spatial propagation'),(.814,'D  SEEG-site rate readout')]:
        fig.text(x,.958,label,weight='bold',fontsize=12)
    status=legends(fig)
    fig.legend(handles=status,loc='lower left',bbox_to_anchor=(.302,.008),ncol=2,
               frameon=False,fontsize=8.2,columnspacing=1.,handlelength=2.)
    fig.legend(handles=[Line2D([0],[0],color=c,label=n) for c,n in zip([*COL,'#555555'],
        ['Core A','Core B','Surround'])],loc='lower left',bbox_to_anchor=(.312,.087),ncol=3,frameon=False,fontsize=9)
    suffix='; log scale' if log_readout else ''
    fig.colorbar(imf,cax=fig.add_axes([.56,.104,.18,.011]),orientation='horizontal',label=f'E rate (Hz / cell{suffix})')
    fig.colorbar(im,cax=fig.add_axes([.826,.104,.13,.011]),orientation='horizontal',label=f'Contact-weighted rate (Hz / cell{suffix})')
    export(fig,'spatial_rate_focused_composite')


def standalone(selected,evidence,cases):
    fig,axs=plt.subplots(1,2,figsize=(12,5.6))
    fig.subplots_adjust(left=.075,right=.98,top=.90,bottom=.23,wspace=.26)
    for k,ax in enumerate(axs):
        mean_axis(ax,k,selected,evidence,cases)
        ax.set_title(f'Core {"AB"[k]}',loc='left',weight='bold')
    status=legends(fig)
    fig.legend(handles=status,loc='lower left',bbox_to_anchor=(.48,.018),ncol=2,
        frameon=False,fontsize=8,columnspacing=.9)
    export(fig,'primary_bifurcation_core_A_B')


def local_onset(fs,evidence):
    # Stop each Hopf family at its first broad fold. Tiny resonant turns near
    # H2 remain in the lower magnification, but later extended returns do not.
    local=hopf_departures(fs)
    if H2_RETURN_ROWS is not None:local['B']=H2_RETURN_ROWS
    local.update({name:primary(fs)[name] for name in ['double','Bleading','single']})
    fig,axs=plt.subplots(2,2,figsize=(12,8.6))
    fig.subplots_adjust(left=.08,right=.975,bottom=.17,top=.935,hspace=.42,wspace=.26)
    hopfs=read(RATE_OUT/'hopfs.json')['rows']
    for k in [0,1]:
        for row in [0,1]:
            ax=axs[row,k];equilibrium(ax,k)
            draw=local if row==0 else {name:local[name] for name in ['A','B']}
            for name,rr in draw.items():cycle_line(ax,rr,k,FAMILY[name])
            draw_samples(ax,draw,k,evidence)
            for i,h in enumerate(hopfs):
                ax.plot(h['J_EE_core'],h['rates_hz'][k],'o',color='#222222',ms=4.5,zorder=8)
                offset=(-16,16) if i==0 else (7,18 if row==0 else 14)
                if i==1 and row==0 and k==1 and H2_RETURN_ROWS is not None:offset=(-10,12)
                ax.annotate(f'H{i+1}',(h['J_EE_core'],h['rates_hz'][k]),xytext=offset,
                    textcoords='offset points',fontsize=9,
                    arrowprops=dict(arrowstyle='-',lw=.5,color='#333333'))
            ax.set(xlim=(.932,.976),xlabel=r'$J_{\mathrm{EE,core}}$',
                ylabel=rf'$\langle r_{{{ "AB"[k] }}}\rangle$ (Hz / E cell)')
            if row==0:
                ax.set(yscale='log',ylim=(.55,35),title=f'{"AB"[k]}  Core {"AB"[k]}: small cycles and bursts')
                ax.yaxis.set_major_locator(FixedLocator([1,2,5,10,20]));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
                cycle_critical(ax,k,['LPC_double_low','LPC_double_high','LPC_Bleading_low','LPC_burst_low'])
                if SHOW_PD:cycle_critical(ax,k,['PD_double_low','PD_double_upper']+
                                         (['PD_H2_after_LPC13'] if H2_RETURN_ROWS is not None else []))
            else:
                ax.set(ylim=(.68,1.65),title=f'{"CD"[k]}  Core {"AB"[k]}: Hopf neighbourhood')
                cycle_critical(ax,k,['LPC_A1','LPC_B1'])
                if H2_RETURN_ROWS is not None:cycle_critical(ax,k,['LPC_B2'])
                if SHOW_PD and H2_RETURN_ROWS is not None:cycle_critical(ax,k,['PD_H2_after_LPC13'])
            style(ax)
    fig.legend(handles=[Line2D([0],[0],color=FAMILY[n],ls=LINESTYLE,lw=1.6,label=NAMES[n]) for n in NAMES],
        loc='lower center',bbox_to_anchor=(.5,.065),ncol=5,frameon=False,fontsize=9)
    fig.legend(handles=[
        Line2D([0],[0],color='#777777',ls=LINESTYLE,label=CONTINUATION_LABEL),
        Line2D([0],[0],marker='o',ls='',color='#222222',ms=4,label='Stable cycle sample'),
        Line2D([0],[0],marker='x',ls='',color='#222222',ms=5,label='Unstable cycle sample'),
        Line2D([0],[0],marker='s',ls='',color='#222222',ms=5,label='Locally checked fold'),
        Line2D([0],[0],marker='s',ls='',mfc='white',mec='#222222',ms=5,label='Fold: checks pending')]+([
        Line2D([0],[0],marker='v',ls='',color='#222222',ms=5,label='PD: parent and mode checked'),
        Line2D([0],[0],marker='v',ls='',mfc='white',mec='#222222',ms=5,label='PD: checks pending')] if SHOW_PD else []),
        loc='lower center',bbox_to_anchor=(.5,-.002),ncol=3,frameon=False,fontsize=8.5)
    export(fig,'onset_bifurcation_detail')


def envelopes(selected):
    fig,axs=plt.subplots(2,3,figsize=(13.6,7.5))
    fig.subplots_adjust(left=.065,right=.985,bottom=.13,top=.91,wspace=.28,hspace=.4)
    for j,name in enumerate(['double','Bleading','single']):
        for k in [0,1]:
            ax=axs[k,j];rows=selected[name]
            for stat,width,alpha in [('min_rates_hz',.8,.6),('max_rates_hz',.8,.6),('mean_rates_hz',1.9,1.)]:
                cycle_line(ax,rows,k,FAMILY[name],stat,width,alpha)
            ax.set(xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=f'Core {"AB"[k]} rate (Hz / E cell)',
                yscale='log',ylim=(.006,550),title=NAMES[name] if k==0 else '')
            ax.yaxis.set_major_locator(FixedLocator([.01,.1,1,10,100]));ax.yaxis.set_major_formatter(FuncFormatter(lambda v,pos:f'{v:g}'))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f' if name!='single' else '%.2f'));style(ax)
    fig.legend(handles=[Line2D([0],[0],color='black',ls=LINESTYLE,lw=1.9,label='Period mean'),
        Line2D([0],[0],color='black',ls=LINESTYLE,lw=.8,alpha=.6,label='Minimum / maximum'),
        Line2D([0],[0],color='#777777',ls=LINESTYLE,label=CONTINUATION_LABEL)],
        loc='lower center',ncol=3,frameon=False)
    export(fig,'burst_envelopes_by_family')


def extended(fs):
    fig,axs=plt.subplots(2,2,figsize=(12,8))
    fig.subplots_adjust(left=.08,right=.98,bottom=.10,top=.94,hspace=.42,wspace=.28)
    for row,name in enumerate(['A','B']):
        for k in [0,1]:
            ax=axs[row,k];cycle_line(ax,fs[name],k,FAMILY[name])
            ax.set(xlabel=r'$J_{\mathrm{EE,core}}$',ylabel=rf'$\langle r_{{{ "AB"[k] }}}\rangle$ (Hz / E cell)',
                   title=f'H{row+1} extended family | Core {"AB"[k]}',yscale='log')
            if name=='A':cycle_critical(ax,k,[f'LPC_A_stage4_turn{i}' for i in [1,2,3]])
            style(ax)
    fig.legend(handles=[Line2D([0],[0],color='#777777',ls=LINESTYLE,label='Period mean; stability unclassified along line'),
        Line2D([0],[0],marker='s',color='#222222',ls='',label='Locally checked cycle fold')],
        loc='lower center',ncol=2,frameon=False,fontsize=9)
    export(fig,'extended_hopf_families_separate')


def main():
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
    fs=families();selected=primary(fs);evidence=physical_stability_samples();s=RateField();cases=load_cases(s)
    print('DRAW PRIMARY', {name:len(rr) for name,rr in selected.items()},flush=True)
    composite(s,selected,evidence,cases);standalone(selected,evidence,cases)
    local_onset(fs,evidence);envelopes(selected);extended(fs)
    meta=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        source_model='Frozen autonomous 400-cell / 935-population spatial rate DDE; Z=1 and dynamic M unchanged',
        producer=str(Path(__file__).resolve()),source_periodic_directory=str(PERIODIC_OUT),
        primary_family_names=NAMES,primary_orbits={name:[q['path'] for q in rr] for name,rr in selected.items()},
        total_source_families=len(fs),total_source_geometry_points=sum(len(rr) for rr in fs.values()),
        omitted_from_primary='Later H1/H2 returns, upper A-leading extension, higher stationary-branch Hopf families and PD children; retained in source inventory, not evidence of their absence.',
        line_semantics='Dotted: traced periodic geometry with interval stability unclassified. Solid black: inherited stable low equilibrium below H1. Filled circles / crosses: exact physically checked stable / unstable cycle samples; do not interpolate verdicts between samples. White lettered circles identify the displayed waveforms, not stability.',
        critical_semantics='Filled squares require physical parent and independent critical mode. Open squares retain fold coordinates with pending checks. Neither implies stability of neighbouring cycles.',
        cases=[dict(case=q['letter'],J_EE_core=q['J'],title=q['title'],orbit=q['orbit'],
                    T_ms=q['T'] if q['name'] else None,mean_A_B_surround_Hz=q['regional'].mean(0),
                    snapshot_times_ms=q['snapshot_times_ms']) for q in cases],
        exact_sample_stability=evidence,readout='Same periodic rate orbit; contact-weighted firing rate, not electrical SEEG voltage',
        validation='Display revision only; no additional root, stability interval or native-SNN equivalence inferred',
        figures={f.stem:{ext:hashlib.sha256(f.with_suffix('.'+ext).read_bytes()).hexdigest() for ext in ['png','pdf','svg']}
                 for f in FIG.glob('*.png')},human_visual_acceptance='PENDING')
    write(DEST/'figure_metadata.json',meta)
    entries={
      'spatial_rate_focused_composite':'左侧按 Core A、B 分别展示五类主要解的周期均值，右侧沿用同一 rate 方程下的 a–e 波形、二维场和固定十五触点读出；c 行仍包含同一完整周期中的两种领先顺序。点线表示整段稳定性未判定，稳定或不稳定只标在通过物理波形检查的采样位置。**关注点**：右侧周期波形已检查，但不因展示而宣称是已验证的稳定吸引子；接触读出不是电压。',
      'primary_bifurcation_core_A_B':'单独放大主图的 Core A、B 两个均值分岔面板，采用相同参数与纵轴范围。后续复杂返回支及高阶 Hopf 分支移出此展示，全部原始计算保留。**关注点**：a–e 与合图完全对应；曲线交叉不表示分支连接。',
      'onset_bifurcation_detail':'上排放大低 J 区间的小振荡与 burst 分支，下排仅看两条 Hopf 出发分支的早期均值变化。标记保留真实求解坐标，空心折点表示局部检查仍待完成。**关注点**：不能将 H1、H2 与大幅 burst 的周期折叠合并成一个起始点。',
      'burst_envelopes_by_family':'三列分别展示交替领先、B 领先和 A 领先 burst 的均值及峰谷，Core A、B 分开成两行。将峰谷线从主图移至此处，保留真实折返和数值范围。**关注点**：粗细区分统计量，点线不代表稳定周期。',
      'extended_hopf_families_separate':'将 H1、H2 的后续延续分别放在独立行，每列对应一个 core 的周期均值。标出已复核的 LPC61–63，避免将它们与主要 burst 分支重叠展示。**关注点**：LPC61–63 的邻域稳定性与临界模态是不同证据；延续曲线不是时间轨迹。'}
    (FIG/'README.md').write_text('\n\n'.join(f'### {name}.png\n{text}' for name,text in entries.items())+'\n')
    print('SAVED',DEST,flush=True)


if __name__=='__main__':main()
