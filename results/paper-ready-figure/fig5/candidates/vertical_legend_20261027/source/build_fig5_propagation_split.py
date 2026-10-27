#!/usr/bin/env python3
"""New Fig5: one frozen spatial-SNN workpoint and the pending core-count contrast."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
import numpy as np
from scipy.ndimage import uniform_filter1d
try:
    import pymupdf as fitz
except ImportError:
    sys.path.insert(0,'/tmp/hfosp_fig5_pdf_tools')
    import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/paper-ready-figure/fig5/candidates/propagation_split_20261024'
EXP=Path('/data/hfosp/topic4_sef_hfo/cohort_core_count_20261024')
RED,BLUE,PURPLE='#C63D3A','#287FA1','#75628F'
MM=72/25.4

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def safe(x):
    if isinstance(x,dict):return {k:safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple,np.ndarray)):return [safe(v) for v in x]
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,(np.integer,np.bool_)):return x.item()
    if isinstance(x,Path):return str(x)
    return x
def write(p,x):Path(p).write_text(json.dumps(safe(x),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def export(fig,path):
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer();outside=[]
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text():continue
        b=t.get_window_extent(renderer)
        if b.x0<-.5 or b.y0<-.5 or b.x1>fig.bbox.x1+.5 or b.y1>fig.bbox.y1+.5:outside.append(t.get_text())
    assert not outside,outside
    for ext in ('png','pdf','svg'):fig.savefig(path.with_suffix('.'+ext),dpi=300,facecolor='white')
    plt.close(fig)

def core_outlines(ax,physics,*,shift=-10.,color=PURPLE,lw=.9):
    c=physics['candidate']
    for center,r in zip(c['centers_mm'],c['radii_mm']):
        ax.add_patch(Circle(np.asarray(center)+shift,r,fill=False,ec=color,lw=lw,ls=(0,(3,2)),zorder=5))

def schematic(src,figdir,physics,arrays):
    fig=plt.figure(figsize=(76/25.4,78/25.4));ax=fig.add_axes([.16,.14,.79,.77])
    with np.load(src/'overview_neuron_samples.npz') as z:
        for key,marker,color in [('E','^','#e77777'),('I','o','#72aadb')]:
            ax.scatter(*z[key].T,s=2.2,marker=marker,c=color,lw=0,alpha=.55,zorder=1)
    with np.load(src/'rigid_contact_geometry.npz') as z:xy=z['rigid_xy_mm'];names=z['names'].astype(str)
    for shaft in ['SCL','ICL']:
        pts=xy[np.char.startswith(names,shaft)]
        ax.plot(*pts[[0,-1]].T,c='#777777',lw=.8,zorder=2)
        ax.scatter(*pts.T,facecolors='white',edgecolors='#666666',s=8,lw=.5,zorder=3)
    core_outlines(ax,physics)
    centers=np.asarray(physics['candidate']['centers_mm'])-10
    for i,center in enumerate(centers):
        ax.add_patch(Circle(center,physics['candidate']['radii_mm'][i],fc=PURPLE,alpha=.07,lw=0,zorder=1))
        ax.text(center[0],center[1]+2.2,f'Core {"AB"[i]}',ha='center',va='bottom',color='#594871',fontsize=7.5)
    kernel=physics['graph']['kernel'];theta=kernel['theta_deg'];u=np.array([np.cos(np.deg2rad(theta)),np.sin(np.deg2rad(theta))])
    center=np.array([0.,1.1]);major=9.0;minor=major*kernel['l_perp']/kernel['l_par']
    ax.add_patch(Ellipse(center,major,minor,angle=theta,fc=RED,ec='none',alpha=.075,zorder=1))
    p,q=center-4.7*u,center+4.7*u
    ax.plot(*np.vstack([p,q]).T,color=RED,lw=1.15,ls=(0,(4,2.5)),zorder=4)
    ax.text(0,5.2,'E→E axis',ha='center',color=RED,fontsize=8)
    ax.set(xlim=(-10,10),ylim=(-10,10),xticks=[-10,0,10],yticks=[-10,0,10],xlabel='x (mm)',ylabel='y (mm)',aspect='equal')
    ax.tick_params(length=2.5,pad=2)
    for spine in ax.spines.values():spine.set_visible(True)
    ax.legend([Line2D([],[],ls='',marker='^',ms=3,color='#e77777'),Line2D([],[],ls='',marker='o',ms=3,color='#72aadb')],
        ['E','I'],loc='upper right',fontsize=6.5,frameon=True,edgecolor='none',framealpha=.95,borderpad=.25,handletextpad=.3)
    export(fig,figdir/'fig5-panela')
    return dict(workpoint='xy_left_20',core_centers_mm=physics['candidate']['centers_mm'],core_radii_mm=physics['candidate']['radii_mm'],
        coordinate_translation_mm=[-10,-10],actual_EE_angle_deg=theta,ellipse_axis_ratio=kernel['l_par']/kernel['l_perp'],
        ellipse_scope='Schematic magnification of the actual EE kernel aspect ratio and orientation, not its spatial length scale.',
        core_boundaries='Exact physical circular core membership boundaries; dashed outline.',
        working_point_scope='Author selected the same accepted demonstration as panels B/C/E on 2026-10-24; not the later global J_joint minimum.')

def sequence(src,figdir,physics,arrays):
    case=next(c for c in read(src/'propagation_manifest.json')['cases'] if c['candidate']=='xy_left_20')
    run=read(src/'trajectory.json');assert sha(src/'trajectory.npz')==run['arrays_sha256']
    counts=arrays['sheet_activity_counts'];dt=float(arrays['sheet_activity_frame_ms']);assert dt==2 and counts.shape[1:]==(20,20)
    # Five consecutive native bins = a centered 10-ms mean in the original count units.
    smooth=uniform_filter1d(counts.astype(float),size=5,axis=0,mode='nearest')
    frames=[]
    for ev in case['events']:
        index=ev['event'];lo,hi=run['events'][index]['window_ms'];assert [lo,hi]==ev['window_ms']
        first,last=round(lo/dt),round(hi/dt);activity=smooth[first:last].sum((1,2));peak=float(activity.max())
        active=np.flatnonzero(activity>=.01*peak)
        assert len(active)>5
        # The first >=1% bin exposes initiation; extend to the first <=1% bin after the burst.
        start=first+int(active[0]);end=first+int(active[-1])+1
        step=int(np.ceil((end-start)/5));indices=start+np.arange(6)*step
        assert indices[-1]<last and np.all(np.diff(indices)==step)
        images=smooth[indices]
        grid=np.linspace(-9.5,9.5,20);xx,yy=np.meshgrid(grid,grid)
        totals=images.sum((1,2));centroids=np.column_stack([(images*xx).sum((1,2)),(images*yy).sum((1,2))])
        centroids=np.divide(centroids,totals[:,None],out=np.full_like(centroids,np.nan),where=totals[:,None]>0)
        theta=np.deg2rad(physics['graph']['kernel']['theta_deg'])
        projection=centroids@np.array([np.cos(theta),np.sin(theta)])
        frames.append(dict(mode='M'+ev['mode'],event=index,window_ms=[lo,hi],
            first_frame_ms=start*dt,absolute_times_ms=indices*dt,relative_times_ms=np.arange(6)*step*dt,
            frame_indices=indices,step_ms=step*dt,selection_threshold_total_spikes=.01*peak,
            total_spikes_at_frames=images.sum((1,2)),peak_smoothed_total_spikes=peak,
            activity_centroids_mm=centroids,EE_axis_projection_mm=projection,
            final_to_peak_fraction=float(images[-1].sum()/peak),images=images))
    fig=plt.figure(figsize=(171/25.4,83/25.4))
    cmap=LinearSegmentedColormap.from_list('model_activity',['#fbfcfd','#d4e4eb','#83b3c9','#397b9c','#15364e'])
    norm=PowerNorm(gamma=.5,vmin=0,vmax=75)
    with np.load(src/'rigid_contact_geometry.npz') as z:xy=z['rigid_xy_mm']
    # Shared absolute spatial and intensity scales across all twelve frames.
    for row,item in enumerate(frames):
        y=.56 if row==0 else .15
        for col,(values,t) in enumerate(zip(item['images'],item['relative_times_ms'])):
            ax=fig.add_axes([.07+col*.142,y,.132,.272])
            im=ax.imshow(values,origin='lower',extent=[-10,10,-10,10],cmap=cmap,norm=norm,interpolation='nearest')
            core_outlines(ax,physics,color='#775b83',lw=.55)
            ax.scatter(*xy.T,s=1.8,fc='white',ec='#6d767a',lw=.2,zorder=6)
            ax.set(xlim=(-10,10),ylim=(-10,10),xticks=[],yticks=[])
            ax.set_title(f'{t:g} ms',fontsize=7.5,pad=3)
            for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#ccd1d5');spine.set_linewidth(.45)
            if row==1 and col==0:
                ax.plot([-8,-3],[-8,-8],color='#333333',lw=1.1);ax.text(-5.5,-6.8,'5 mm',ha='center',fontsize=5.5)
        fig.text(.047,y+.136,item['mode'],ha='right',va='center',fontsize=8.5,color=RED if row==0 else BLUE,weight='bold')
    bar=fig.colorbar(im,cax=fig.add_axes([.945,.18,.012,.59]),ticks=[0,25,50,75])
    bar.ax.tick_params(labelsize=6.5,length=2,pad=2);bar.outline.set_visible(False)
    fig.text(.505,.965,'Time from event initiation',ha='center',va='top',fontsize=8)
    fig.text(.51,.035,'E spikes / (2 ms · mm²)',ha='center',fontsize=7.5)
    export(fig,figdir/'fig5-paneld')
    np.savez_compressed(src/'selected_propagation_frames.npz',**{f'{item["mode"]}_frames':item.pop('images') for item in frames})
    return dict(events=frames,native_bin_ms=2,temporal_display_average_ms=10,spatial_bin_mm=1,
        spatial_interpolation='none',spatial_smoothing='none',color_range=[0,75],powernorm_gamma=.5,
        per_frame_normalization=False,selection='Same two historical model-mode medoids; six equal steps from 1% whole-sheet activity onset through return below 1%; no event reselection.',
        time_origin='Each row uses its own event initiation; rows are distinct events, not simultaneous recordings.')

def compose(figdir,cohort_complete):
    doc=fitz.open();page=doc.new_page(width=300*MM,height=181*MM)
    # x/y/w/h in millimetres, with y measured down from the top.
    boxes={'A':[3,7,76,78],'B':[86,8,130,80.1],'C':[237,7,49,80.4],
           'D':[1,96,171,83],'E':[177,106,58,50.5],'F':[240,106,58,52.2]}
    checks={}
    for letter,box in boxes.items():
        x,y,w,h=box;target=fitz.Rect(x*MM,y*MM,(x+w)*MM,(y+h)*MM)
        source=fitz.open(figdir/f'fig5-panel{letter.lower()}.pdf')
        assert page.rect.contains(target),f'Panel {letter} extends beyond the canvas'
        page.show_pdf_page(target,source,0,keep_proportion=True)
        label_y=8 if letter in 'ABC' else 100
        label_x={'A':2,'B':84,'C':234,'D':2,'E':176,'F':239}[letter]
        page.insert_text((label_x*MM,label_y*MM),letter,fontsize=12,fontname='hebo')
        checks[letter]=dict(source_sha256=sha(figdir/f'fig5-panel{letter.lower()}.pdf'),target_mm=box,native_size_pt=list(source[0].rect)[2:])
    if not cohort_complete:
        page.insert_text((247*MM,166*MM),'1-core comparison pending',fontsize=7,color=(.45,.45,.45))
    dest=figdir/'fig5-complete-layout.pdf';doc.save(dest,garbage=4,deflate=True)
    page.get_pixmap(matrix=fitz.Matrix(2.4,2.4),alpha=False).save(figdir/'fig5-complete-layout.png')
    page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(figdir/'fig5-complete-layout-preview.png')
    (figdir/'fig5-complete-layout.svg').write_text(page.get_svg_image())
    doc.close();return checks

def cohort_panel(figdir,src):
    report_path=EXP/'template_recovery_summary.json'
    if not report_path.exists():return False
    report=read(report_path)
    if report['status']!='TRAINING_COMPLETE_REVIEW_PENDING':return False
    fig,ax=plt.subplots(figsize=(70/25.4,64/25.4));fig.subplots_adjust(left=.22,right=.97,bottom=.2,top=.88)
    rows=report['subject_arms'];rng=np.random.default_rng(20261024);jitter=rng.uniform(-.08,.08,25)
    for mode in (0,1):
        patients=[r['subject'] for r in rows if r['core_count']==1]
        for j,sid in enumerate(patients):
            pair=[next(r for r in rows if r['subject']==sid and r['core_count']==k)['matched_rhos'][mode] for k in (1,2)]
            if all(v is not None and np.isfinite(v) for v in pair):
                ax.plot(np.array([mode*3,mode*3+1])+jitter[j],pair,c='.82',lw=.35,zorder=0)
        for k in (1,2):
            values=np.array([r['matched_rhos'][mode] for r in rows if r['core_count']==k],float)
            color=[RED,BLUE][mode];x=mode*3+(k-1)
            ax.scatter(x+jitter,values,s=6,facecolors='white' if k==1 else color,edgecolors=color,lw=.45)
            good=values[np.isfinite(values)]
            if len(good):
                q1,med,q3=np.quantile(good,[.25,.5,.75]);ax.plot([x+.16,x+.16],[q1,q3],c=color,lw=1);ax.plot([x+.07,x+.25],[med,med],c=color,lw=1.3)
    ax.axhline(0,c='.8',ls='--',lw=.6);ax.set(ylim=(-1,1),yticks=[-1,0,1],xticks=[0,1,3,4],xticklabels=['1','2','1','2'],ylabel='Template similarity (ρ)',xlabel='Number of cores')
    ax.text(.5,1.04,'TA–MTA',transform=ax.get_xaxis_transform(),ha='center',color=RED,fontsize=8)
    ax.text(3.5,1.04,'TB–MTB',transform=ax.get_xaxis_transform(),ha='center',color=BLUE,fontsize=8)
    export(fig,figdir/'fig5-panelf');shutil.copy2(report_path,src/'completed_core_count_summary.json');return True

def main():
    global OUT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=OUT);a=p.parse_args();OUT=a.output
    src=OUT/'source';figdir=OUT/'figures';assert src.exists();figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8.5,'axes.linewidth':.65,
        'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    physics=read(src/'applied_physics.json')
    with np.load(src/'trajectory.npz') as z:arrays={k:z[k] for k in ('sheet_activity_counts','sheet_activity_frame_ms','contact_xy_mm','contact_names')}
    A=schematic(src,figdir,physics,arrays);D=sequence(src,figdir,physics,arrays)
    cohort_complete=cohort_panel(figdir,src)
    checks=compose(figdir,cohort_complete)
    preserved={}
    for new,old in {'b':'f','c':'g','e':'h','f':'i'}.items():
        if new=='f' and cohort_complete:continue
        for ext in ('png','pdf','svg'):
            before=src/f'fig4-panel{old}.{ext}';after=figdir/f'fig5-panel{new}.{ext}'
            assert sha(before)==sha(after)
            preserved[f'{new}.{ext}']=sha(after)
    record=dict(figure='Fig5',asset_id='spatial_snn_propagation_and_core_count',version='propagation_split_20261024',
        status='PENDING_AUTHOR_VISUAL_REVIEW',human_visual_acceptance='PENDING',
        producer=str(Path(__file__).relative_to(ROOT)),workpoint='xy_left_20',panel_a=A,panel_d=D,
        panel_map={'A':'Same-workpoint core geometry and EE axis','B':'Accepted old Fig4F','C':'Accepted old Fig4G',
                   'D':'Same-workpoint MTA/MTB native field sequence','E':'Accepted old Fig4H',
                   'F':'Matched 25-patient 1/2-core comparison' if cohort_complete else 'Accepted old Fig4I; 1-core comparison pending'},
        source_panels_byte_identical=preserved,layout=checks,cohort_experiment=str(EXP),cohort_comparison_complete=cohort_complete,
        sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in src.iterdir() if p.is_file()},
        outputs_sha256={str(p.relative_to(ROOT)):sha(p) for p in figdir.iterdir() if p.suffix in ('.png','.pdf','.svg')})
    write(OUT/'figure5_manifest.json',record)
    write(OUT/'validation.json',dict(status='PASS',source_panel_equality='PASS',six_equal_spaced_frames_per_event=True,
        same_workpoint_all_example_panels=True,new_simulations_for_panels_A_to_E=0,human_visual_acceptance='PENDING'))
    print(json.dumps(dict(output=str(figdir),cohort_complete=cohort_complete,frames=[{'mode':x['mode'],'step_ms':x['step_ms'],'absolute_times_ms':x['absolute_times_ms'].tolist()} for x in D['events']]),indent=2))

if __name__=='__main__':main()
