"""No trajectory overlays: physical-D branches plus corresponding spatial fields."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from matplotlib.colors import PowerNorm,LogNorm
from matplotlib.ticker import FixedLocator,FuncFormatter
from scipy.signal import resample
from topic4_fig5_D_physical_model import Equilibrium,OUT,ROOT

FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
SOURCE=ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1'
META=ROOT/'results/topic4_sef_hfo/fig5_preentry_event_audit_20260914/clean_panels_v2/review_square_complete_20260915/fig5_metadata.json'
CENTERS=np.array([[4.19921431597,9.12890135365],[16.47920304044,3.965511533]])
COLORS=dict(a='#D88721',b='#9561AD');CYAN='#28D0CD'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,'axes.linewidth':1.,'pdf.fonttype':42,'svg.fonttype':'none'})

def read(p):return json.loads(p.read_text())
def save(fig,name):
    assert all(ax.get_title()=='' for ax in fig.axes)
    for ext in ['png','pdf','svg']:fig.savefig(FIG/f'{name}.{ext}',dpi=200,bbox_inches='tight',facecolor='white')
    plt.close(fig)

def map_panel(ax,field,norm,label,color='k',showy=True):
    displayed=np.array(field).reshape(20,20)
    if isinstance(norm,LogNorm):
        # Zero and round-off-negative rates belong below the color scale, not
        # to masked white pixels. Keep the underlying saved field unchanged.
        assert displayed.min()>-1e-7
        displayed=np.maximum(displayed,norm.vmin)
    im=ax.imshow(displayed,origin='lower',extent=[0,20,0,20],cmap='magma',norm=norm,interpolation='nearest')
    for j,xy in enumerate(CENTERS):
        ax.add_patch(Circle(xy,1.5,ec=CYAN,fc='none',lw=1.5))
        ax.text(xy[0],xy[1]+2.1,'AB'[j],color='#11666A',fontsize=10,ha='center',weight='bold',bbox=dict(fc='white',ec='none',pad=.25,alpha=.9))
    ax.set(xlabel='x (mm)',xticks=[0,10,20],yticks=[0,10,20],xlim=(0,20),ylim=(0,20))
    if showy:ax.set_ylabel('y (mm)')
    else:ax.set_yticklabels([])
    ax.text(.5,1.04,label,ha='center',va='bottom',transform=ax.transAxes,color=color,fontsize=12)
    return im

def native_reference():
    meta=read(META);fig,axes=plt.subplots(1,4,figsize=(11.4,3.5));fig.subplots_adjust(left=.06,right=.89,bottom=.20,top=.76,wspace=.26)
    names=['Rest','Interictal','Pre-ictal','Onset'];colors=['#657080','#258FB4','#E38520','#D62546']
    qa=[]
    for i,(ax,row,snap) in enumerate(zip(axes,meta['native_maps'],meta['snapshots'])):
        im=map_panel(ax,row['rate_Hz'],PowerNorm(.6,0,500),f"{i+1}  {names[i]}\n{snap['time_s']:.3f} s",colors[i],i==0)
        qa.append(dict(number=i+1,time_window_s=row['time_window_s'],source=str(META),rate_Hz=row['rate_Hz']))
    cax=fig.add_axes([.915,.22,.015,.53]);fig.colorbar(im,cax=cax,ticks=[0,250,500],label='E rate (Hz)')
    save(fig,'fig_native_spatial_reference')
    (OUT/'native_spatial_reference_data.json').write_text(json.dumps(qa,indent=2)+'\n')

def cycle_rows(q=1.25):
    rows=[]
    for core in 'ab':
        for row in read(OUT/f'q{q:g}_cycles_{core}_N32/summary.json'):
            amp=row['control_amplitude_hz'];cert=OUT/f'q{q:g}_floquet_{core}_amp{amp:g}_degree3.json'
            if not cert.exists():continue
            a=read(cert);mods=np.array(a['moduli']);neutral=int(np.argmin(abs(mods-1)))
            unstable=bool(np.any(mods>1+1e-5))
            if unstable:status='unstable'
            elif abs(mods[neutral]-1)<1e-5 and a['phase_return_error']<1e-5 and np.max(np.delete(mods,neutral))<1-1e-5:status='stable'
            else:continue
            path=Path(a.get('source',row['filename']));z=np.load(path);m=Equilibrium(q).m;r=resample(z['r'],1024,axis=0)
            macro=(r[:,:3200]*m.w_u).reshape(1024,400,8).sum(2)*1000;glob=macro@m.count_e/m.count_e.sum()
            row=dict(row,filename=str(path),D=float(z['s']),period_ms=float(z['T']),global_mean_hz=float(glob.mean()),global_min_hz=float(glob.min()),global_max_hz=float(glob.max()),stability=status,floquet_source=str(cert))
            rows.append(row)
    return rows

def diagram(focus=False):
    q=1.25;eq=Equilibrium(q);m=eq.m;certs=read(OUT/'q1.25_branch_stability.json');cy=cycle_rows(q);Hs={c:read(OUT/f'q1.25_H{c}.json') for c in 'ab'}
    seed=np.load(OUT/'q1.25_Ha.npz');rest,er,ok=eq.solve(seed['r_hz'],0.);assert ok
    folds=read(OUT/'new_stationary_folds.json') if (OUT/'new_stationary_folds.json').exists() else []
    fig=plt.figure(figsize=(14.5,7.6));ax=fig.add_axes([.075,.22,.53,.72]);inset=ax.inset_axes([.48,.18,.49,.48]);inset.set_zorder(20);maps=[]
    for i in range(4):maps.append(fig.add_axes([.665+(i%2)*.17,.60-(i//2)*.40,.155,.28]))
    draw=[]
    for name in ['low','middle','middle_extension','high']:
        p=OUT/f'q1.25_{name}.npz'
        if not p.exists():continue
        a=np.load(p);D=a['s'];rate=np.average(a['r_hz'][:,:400],weights=m.count_e,axis=1)
        known={r['index']:r for r in certs if r['branch']==name};ids=sorted(known);pieces=[]
        for ia,ib in zip(ids[:-1],ids[1:]):
            sa,sb=known[ia]['stability'],known[ib]['stability']
            if sa!=sb:
                between=[f for f in folds if f['branch']==name and ia<=f['turn_index']<ib and f['type']=='SADDLE_NODE_OF_EQUILIBRIA']
                if len(between)==1:
                    f=between[0];kk=f['turn_index']
                    for ids,xx,yy,status,prepend in [(np.arange(ia,kk+1),f['D'],f['mean_e_hz'],sa,False),(np.arange(kk+1,ib+1),f['D'],f['mean_e_hz'],sb,True)]:
                        x=np.r_[xx,D[ids]] if prepend else np.r_[D[ids],xx];y=np.r_[yy,rate[ids]] if prepend else np.r_[rate[ids],yy]
                        pieces.append((x,y,status))
                    draw.append(dict(branch=name,indices=[ia,ib],stability='split at certified saddle-node',fold=f['name']))
                continue
            sel=np.arange(ia,ib+1);sel=sel[(D[sel]>=0)&(D[sel]<=1)]
            if len(sel)<2:continue
            pieces.append((D[sel],rate[sel],sa))
            draw.append(dict(branch=name,indices=[ia,ib],stability=sa))
        # Join contiguous intervals before rendering; resetting a dash pattern
        # at every short continuation segment would make it look solid.
        joined=[]
        for x,y,status in pieces:
            if joined and joined[-1][2]==status and abs(joined[-1][0][-1]-x[0])<1e-10 and abs(joined[-1][1][-1]-y[0])<1e-8:
                px,py,_=joined[-1];joined[-1]=(np.r_[px,x[1:]],np.r_[py,y[1:]],status)
            else:joined.append((x,y,status))
        for x,y,status in joined:
            for dest in [ax,inset]:dest.plot(x,y,c='#202020',ls='-' if status=='stable' else (0,(5,3)),lw=1.25,zorder=2)
    # The low branch is explicitly split at the located critical point; do not
    # omit the transition merely because sampled stability straddles it.
    low=np.load(OUT/'q1.25_low.npz');dd=np.r_[0.,low['s'],Hs['a']['D'],Hs['b']['D']];yy=np.r_[np.average(rest[:400],weights=m.count_e),np.average(low['r_hz'][:,:400],weights=m.count_e,axis=1),Hs['a']['mean_e_hz'],Hs['b']['mean_e_hz']];ix=np.argsort(dd);dd,yy=dd[ix],yy[ix]
    for dest in [ax,inset]:
        for stable in [True,False]:
            use=(dd>=0)&((dd<=Hs['a']['D']) if stable else (dd>=Hs['a']['D']))
            dest.plot(dd[use],yy[use],c='#202020',ls='-' if stable else '--',lw=1.35)
        for core in 'ab':
            rr=sorted([r for r in cy if r['core']==core],key=lambda x:x['control_amplitude_hz']);col=COLORS[core]
            for status in ['stable','unstable']:
                rrs=[r for r in rr if r['stability']==status]
                if not rrs:continue
                xx=[Hs[core]['D']]+[r['D'] for r in rrs];ym=[Hs[core]['mean_e_hz']]+[r['global_mean_hz'] for r in rrs]
                dest.plot(xx,ym,c=col,ls='-' if status=='stable' else '--',lw=1.7,zorder=5)
                for key in ['global_min_hz','global_max_hz']:
                    dest.plot([r['D'] for r in rrs],[r[key] for r in rrs],ls='none',marker='s',ms=4.2,mec=col,mfc=col if status=='stable' else 'white',mew=1.,zorder=6)
        for core in 'ab':
            h=Hs[core];dest.plot(h['D'],h['mean_e_hz'],marker='D',ms=6,mfc='white',mec='k',zorder=8)
    if not focus:
        for core,offset in [('a',(-35,15)),('b',(8,14))]:
            h=Hs[core];inset.annotate(r'$NS_'+core.upper()+'$',(h['D'],h['mean_e_hz']),xytext=offset,textcoords='offset points',fontsize=11)
    if (OUT/'new_stationary_folds.json').exists():
        folds=[r for r in read(OUT/'new_stationary_folds.json') if r['type']=='SADDLE_NODE_OF_EQUILIBRIA']
        ax.plot([r['D'] for r in folds],[r['mean_e_hz'] for r in folds],ls='none',marker='*',ms=6,mec='k',mfc='white',zorder=8)
        if focus:
            inset.plot([r['D'] for r in folds],[r['mean_e_hz'] for r in folds],ls='none',marker='*',ms=7,mec='k',mfc='white',zorder=8)
    ax.set(xlim=(0,1),ylim=(.045,510),xlabel=r'$D=1-\langle Z_E\rangle$',ylabel='Global E rate (Hz / neuron)');ax.set_yscale('log');ax.yaxis.set_major_locator(FixedLocator([.05,.1,1,10,100,500]));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
    inset.set(xlim=(.057,.083),ylim=(.052,.12),xticks=[.06,.07,.08],yticks=[.06,.08,.10]);inset.tick_params(labelsize=10);inset.set_xlabel(r'$D$',fontsize=11,labelpad=0)
    if focus:
        ax.set_xlim(0,.56)
        inset.set(xlim=(.503,.526),ylim=(410,421),xticks=[.505,.515,.525],yticks=[412,416,420])
        for name,label,offset in [('q1.25_middle_extension_SN1',r'$SN_L$',(-45,15)),('q1.25_high_SN1',r'$SN_H$',(-38,-21))]:
            fold=next(f for f in folds if f['name']==name)
            dest=ax if name.endswith('extension_SN1') else inset
            dest.annotate(label,(fold['D'],fold['mean_e_hz']),xytext=offset,textcoords='offset points',fontsize=12)
    ax.spines[['top','right']].set_visible(False);ax.text(0,1.025,'A',transform=ax.transAxes,fontsize=18,weight='bold');ax.text(.98,.025,r'$q_{I\to E}=1.25$',transform=ax.transAxes,ha='right',fontsize=13)
    # Spatial fields are drawn at exactly the selected branch solutions.
    fields=[rest[:400]];pointinfo=[dict(number=1,D=0.,kind='stable equilibrium',mean_e_hz=float(np.average(rest[:400],weights=m.count_e)),rate_hz=rest[:400].tolist())]
    for core in ('a' if focus else 'ab'):
        row=max([r for r in cy if r['core']==core],key=lambda r:r['control_amplitude_hz']);a=np.load(row['filename']);r=resample(a['r'],1024,axis=0);macro=(r[:,:3200]*m.w_u).reshape(1024,400,8).sum(2)*1000;glob=macro@m.count_e/m.count_e.sum();j=int(np.argmax(glob));fields.append(macro[j]);pointinfo.append(dict(number=len(fields),D=row['D'],kind=f'core {core.upper()} cycle at global maximum',cycle_stability=row['stability'],mean_e_hz=row['global_mean_hz'],phase_fraction=j/1024,rate_hz=macro[j].tolist(),source=row['filename']))
    if focus:
        selected=next(f for f in folds if f['name']=='q1.25_middle_extension_SN1')
        state=np.load(OUT/(selected['name']+'.npz'));fields.append(state['r_hz'][:400]);pointinfo.append(dict(number=3,D=selected['D'],kind='saddle-node equilibrium, not an oscillation snapshot',mean_e_hz=selected['mean_e_hz'],rate_hz=fields[-1].tolist(),source=str(OUT/(selected['name']+'.npz'))))
    high=np.load(OUT/'q1.25_high.npz')
    if focus:
        # Earliest sampled stable point on the high branch, as D increases.
        choice=min((c for c in certs if c['branch']=='high' and c['stability']=='stable'),key=lambda c:c['D']);high_index=choice['index']
    else:high_index=0
    fields.append(high['r_hz'][high_index,:400]);pointinfo.append(dict(number=4,D=float(high['s'][high_index]),kind='stable high equilibrium, not a global oscillation' if focus else 'high equilibrium',mean_e_hz=float(np.average(fields[-1],weights=m.count_e)),rate_hz=fields[-1].tolist(),source=str(OUT/'q1.25_high.npz'),source_index=high_index))
    for i,(dest,f,row) in enumerate(zip(maps,fields,pointinfo)):
        identity=(['Equilibrium','A-cycle maximum',r'$SN_L$ equilibrium','High equilibrium'] if focus else ['Equilibrium','A-cycle maximum','B-cycle maximum','Equilibrium'])[i]
        color=(['k',COLORS['a'],'k','k'] if focus else ['k',COLORS['a'],COLORS['b'],'k'])[i]
        im=map_panel(dest,f,LogNorm(.01,500),f"{i+1}  {identity}\nD = {row['D']:.4f}",color=color,showy=i%2==0)
        if focus and i<2:dest.set_xlabel('')
        if i==0:dest.text(-.30,1.04,'B',transform=dest.transAxes,fontsize=18,weight='bold')
        y=float(np.average(f,weights=m.count_e));row['displayed_phase_global_E_hz']=y
        target=(inset if i==3 else ax) if focus else (inset if i in (1,2) else ax)
        offset=([(8,-10),(4,10),(9,-10),(8,3)] if focus else [(8,-10),(-15,9),(7,8),(-17,-17)])[i]
        target.annotate(str(i+1),(row['D'],y),xytext=offset,textcoords='offset points',fontsize=12,weight='bold',zorder=25)
        if i in (0,3):target.plot(row['D'],y,'o',ms=4,mfc='white',mec='k',clip_on=False,zorder=10)
        if focus and i==3:
            ax.plot(row['D'],y,'o',ms=4,mfc='white',mec='k',zorder=10)
            ax.annotate('4',(row['D'],y),xytext=(9,4),textcoords='offset points',fontsize=12,weight='bold')
    cbax=fig.add_axes([.665,.075,.325,.019]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal',ticks=[.01,1,100,500],label='E rate (Hz)');cb.set_ticklabels(['0.01','1','100','500'])
    handles=[Line2D([],[],color='k',lw=1.5,label='Equilibrium: stable'),Line2D([],[],color='k',ls='--',lw=1.5,label='Equilibrium: unstable'),
      Line2D([],[],color=COLORS['a'],lw=1.7,label='Cycle mean: stable'),Line2D([],[],color=COLORS['b'],ls='--',lw=1.7,label='Cycle mean: unstable'),
      Line2D([],[],color=COLORS['a'],marker='s',ls='none',mfc=COLORS['a'],label='Cycle max / min: stable'),Line2D([],[],color=COLORS['b'],marker='s',ls='none',mfc='white',label='Cycle max / min: unstable'),
      Line2D([],[],color='k',marker='D',ls='none',mfc='white',label='Neimark–Sacker (Hopf)'),Line2D([],[],color='k',marker='*',ls='none',mfc='white',ms=9,label='Saddle-node')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.073,.002),ncol=2,frameon=False,fontsize=10.5,columnspacing=1.5)
    save(fig,'fig_D_fold_focus_spatial' if focus else 'fig_physical_D_bifurcation_spatial')
    target_data=FIG.parent/'focus_figure_data.json' if focus else OUT/'figure_data.json'
    target_data.write_text(json.dumps(dict(q_ie=q,cycles=cy,segments=draw,spatial_points=pointinfo,native_reference_separate=True,no_native_time_trajectory=True,spatial_color_norm='log, .01–500 Hz',human_visual_acceptance='PENDING'),indent=2)+'\n')

def working_point_map():
    rows=read(OUT/'working_point_screen.json');fig,ax=plt.subplots(figsize=(5.5,4))
    for core in 'ab':
        rr=sorted([r for r in rows if r['core']==core and r['physical']],key=lambda r:r['q_ie']);ax.plot([r['q_ie'] for r in rr],[r['D'] for r in rr],'-o',c=COLORS[core],label=f'Core {core.upper()}')
    ax.set(xlabel=r'I$\to$E weight multiplier',ylabel=r'Critical depletion $D$',ylim=(0,.4));ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False);save(fig,'fig_working_point_shift')

def critical_modes():
    if not (OUT/'critical_spatial_modes.json').exists():return
    rows=read(OUT/'critical_spatial_modes.json');fig,axes=plt.subplots(1,4,figsize=(11.4,3.5));fig.subplots_adjust(left=.06,right=.89,bottom=.20,top=.76,wspace=.26)
    for i,(ax,row) in enumerate(zip(axes,rows)):
        im=map_panel(ax,np.array(row['cell_mode_energy_fraction'])*100,PowerNorm(.5,0,100),row['label'],showy=i==0)
    cax=fig.add_axes([.915,.22,.015,.53]);fig.colorbar(im,cax=cax,ticks=[0,25,50,100],label='Mode energy / cell (%)');save(fig,'fig_old_critical_spatial_modes')

def original_network():
    # Keep the user-requested original-time layout as a distinct q=1 figure.
    # Native points are observations, not declarations of equilibrium/cycle
    # membership. The old stable/unstable line ambiguity is removed.
    from plot_topic4_fig5_D_fast_slow import branch_data
    branches,certified,rest,events,cycles=branch_data()
    m=Equilibrium(1.).m;meta=read(META);projection=np.load(ROOT/'results/topic4_sef_hfo/fig5_D_fast_slow_20260916/native_D_projection.npz')
    fig=plt.figure(figsize=(14.5,7.6));ax=fig.add_axes([.075,.22,.53,.72]);draw=[]
    for b in branches:
        D=b['D'];rate=b['rate'];valid=np.array([(b['name'],k) in certified and D[k]>=0 for k in range(len(D))]);edges=valid[:-1]&valid[1:]
        start=None
        for k in range(len(edges)+1):
            hit=k<len(edges) and edges[k]
            if hit and start is None:start=k
            if not hit and start is not None:
                ax.plot(D[start:k+1],rate[start:k+1],color='k',ls=(0,(5,3)),lw=1.2);draw.append(dict(branch=b['name'],indices=[start,k],status='unstable',source='prior exact-map instability certificate'));start=None
    for name,label in [('old_TP','SN1'),('old_TP2','SN2')]:
        a=read(OUT/f'{name}.json');ax.plot(a['D'],a['mean_e_hz'],'*',mfc='white',mec='k',ms=10,zorder=5);ax.annotate(label,(a['D'],a['mean_e_hz']),xytext=(-32,10) if label=='SN1' else (-35,12),textcoords='offset points',fontsize=12)
    ax.set(xlim=(0,.33),ylim=(.02,510),xlabel=r'$D=1-\langle Z_E\rangle$',ylabel='Global E rate (Hz / neuron)');ax.set_yscale('log');ax.yaxis.set_major_locator(FixedLocator([.03,.1,1,10,100,500]));ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:g}'));ax.spines[['top','right']].set_visible(False);ax.text(0,1.025,'A',transform=ax.transAxes,fontsize=18,weight='bold');ax.text(.98,.025,r'$q_{I\to E}=1.00$',transform=ax.transAxes,ha='right',fontsize=13)
    colors=['#657080','#258FB4','#E38520','#D62546'];names=['Rest','Interictal','Pre-ictal','Onset'];rows=[]
    for i,(snap,native) in enumerate(zip(meta['snapshots'],meta['native_maps'])):
        D=float(np.interp(snap['time_s'],projection['field_t'],projection['D']));rate=float(np.average(native['rate_Hz'],weights=m.count_e))
        ax.plot(D,rate,'o',color=colors[i],ms=14,zorder=7);ax.text(D,rate,str(i+1),color='white',ha='center',va='center',fontsize=10,weight='bold',zorder=8)
        dest=fig.add_axes([.665+(i%2)*.17,.60-(i//2)*.40,.155,.28]);im=map_panel(dest,native['rate_Hz'],PowerNorm(.6,0,500),f"{i+1}  {names[i]}\n{snap['time_s']:.3f} s",color=colors[i],showy=i%2==0)
        if i==0:dest.text(-.3,1.04,'B',transform=dest.transAxes,fontsize=18,weight='bold')
        rows.append(dict(number=i+1,time_s=snap['time_s'],time_window_s=native['time_window_s'],D=D,global_E_hz=rate,rate_hz=native['rate_Hz'],classification='native snapshot, not a bifurcation or reduced equilibrium'))
    cax=fig.add_axes([.665,.075,.325,.019]);fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[0,250,500],label='E rate (Hz)')
    fig.legend(handles=[Line2D([],[],color='k',ls='--',label='Reduced equilibrium: unstable'),Line2D([],[],color='k',marker='*',ms=10,mfc='white',ls='none',label='Saddle-node'),Line2D([],[],color='#657080',marker='o',ls='none',label='Native SNN snapshot (1–4)')],loc='lower left',bbox_to_anchor=(.075,.02),frameon=False,ncol=1,fontsize=11)
    save(fig,'fig_original_network_bifurcation_spatial');(OUT/'original_network_figure_data.json').write_text(json.dumps(dict(q_ie=1.,native_points=rows,reduced_segments=draw,no_native_trajectory=True,unclassified_segments_omitted=True,scope='The paired reduction previously failed native state correspondence. Snapshot observations are not placed onto or fitted to a reduced branch.'),indent=2)+'\n')

if __name__=='__main__':
    native_reference();working_point_map();critical_modes();original_network()
    if (OUT/'q1.25_branch_stability.json').exists():diagram()
