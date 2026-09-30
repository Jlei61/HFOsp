"""Actual spatial frames prepared for transition-branch annotations.

These are finite-trajectory observations, not equilibrium/cycle certificates.
The two D=.24 frames are consecutive burst peaks, with their identity retained.
The D=.25 frame is near the first sustained interval, not the late plateau.
"""
from analyze_qualification import *
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import argparse


def load_case(name):
    folder=OUT/'recurrence_searches'/name
    assert json.load(open(folder/'status.json'))['status']=='COMPLETE'
    cfg=json.load(open(folder/'config.json'))
    with np.load(folder/'trajectory.npz') as z:
        rates=z['rate_1ms'];field=z['field_1ms'];count=z['count_e']
    return folder,cfg,rates,field,count


def main(args):
    frames=[];rows=[]
    for name,kind in [('D2375_burst_fp64_confirmation','single'),
                      ('D240_from_burst_fp64','successive'),(args.entry_case,'entry')]:
        source,cfg,rates,field,count=load_case(name)
        smooth=uniform_filter1d(rates,50,axis=0,mode='nearest')
        peaks,_=find_peaks(smooth[:,0],distance=100,prominence=10)
        peaks=peaks[(peaks>=25)&(peaks<len(rates)-25)]
        if kind=='single':selected=peaks[-1:]
        elif kind=='successive':selected=peaks[-2:]
        else:
            r10=rates[:len(rates)//10*10].reshape(-1,10,4).mean(1)
            sustained=[(s,t) for s,t in stretches(r10[:,0]>=5.) if t-s>=50]
            assert sustained
            start,end=sustained[0];candidates=peaks[(peaks>=start*10+25)&(peaks<min(end*10-25,start*10+300))]
            selected=candidates[:1] if len(candidates) else np.array([start*10+50])
        for peak in selected:
            # Rebin the same cells to the accepted 1-mm display grid; keep
            # counts as weights so spatial averaging preserves global rate.
            native=field[peak-25:peak+25].mean(0)
            bins=count.reshape(20,2,20,2).sum((1,3)).ravel()
            image=(native*count).reshape(20,2,20,2).sum((1,3)).ravel()/np.maximum(bins,1)
            global_rate=float(rates[peak-25:peak+25,0].mean())
            assert abs(image@bins/bins.sum()-global_rate)<1e-10
            weights=bins/bins.sum()
            row=dict(source=str(source),D=cfg['D'],source_initial_ms=cfg['initial_ms'],
                peak_index_1ms=int(peak),center_ms=cfg['initial_ms']+float(peak),
                spatial_window_ms=[cfg['initial_ms']+float(peak)-25,cfg['initial_ms']+float(peak)+25],
                selection=kind,mean_E_hz=global_rate,
                regional_mean_hz=rates[peak-25:peak+25].mean(0),
                E_fraction_in_display_bins_above_50Hz=float(weights@(image>=50)),
                spatial_mean_rate_exactly_matches_temporal_readout=True)
            if kind=='entry':
                row['entry_selection_rule']='First interval with global 10-ms E rate >=5 Hz for at least 500 ms; choose its first 50-ms peak within 300 ms'
                row['selection_interval_ms']=[cfg['initial_ms']+start*10,cfg['initial_ms']+end*10]
                row['entry_scope']='Finite sustained interval; this frame does not establish permanent entry or an invariant attractor'
            frames.append(image.reshape(20,20));rows.append(row)
    # Descriptive difference at the same two consecutive D=.24 observations.
    x,y=frames[1].ravel(),frames[2].ravel();dx=x-weights@x;dy=y-weights@y
    contrast=dict(weighted_RMSE_hz=float(np.sqrt(weights@((x-y)**2))),
        weighted_spatial_correlation=float(weights@(dx*dy)/np.sqrt((weights@(dx*dx))*(weights@(dy*dy)))),
        interpretation='Changes in recruitment, not evidence that either core causes the bifurcation')
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    geo=np.load(OUT/'operators/selected_g40_theta0.25/geometry.npz')
    regional_weights=np.array([np.sum(geo['group_size'][(geo['population']==0)&(geo['group_region']==k)])/32000. for k in range(3)])
    difference=np.array(rows[2]['regional_mean_hz'][1:])-np.array(rows[1]['regional_mean_hz'][1:])
    contrast['regional_weights']=regional_weights
    contrast['regional_contribution_to_global_mean_change_hz']=difference*regional_weights
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.linewidth':1.,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,4,figsize=(10.6,3.0),layout='constrained',sharex=True,sharey=True)
    for i,(ax,frame,row) in enumerate(zip(axes,frames,rows)):
        im=ax.imshow(frame,origin='lower',extent=(0,20,0,20),cmap='inferno',vmin=0,vmax=500,interpolation='nearest')
        for j,(x,y) in enumerate(geo['centers_mm']):
            ax.add_patch(Circle((x,y),1.45,fill=False,color='#20c7d2',lw=1.2))
            ax.text(x,y+1.8,'AB'[j],ha='center',va='bottom',color='#20c7d2',fontsize=10)
        ax.text(0,1.05,f'{i+1}   $D={row["D"]:g}$',transform=ax.transAxes,ha='left',va='bottom')
        ax.set(xlim=(0,20),ylim=(0,20),xticks=[0,10,20],yticks=[0,10,20],xlabel='x (mm)')
        ax.tick_params(direction='out',length=3)
    axes[0].set_ylabel('y (mm)')
    bar=fig.colorbar(im,ax=axes,shrink=.82,pad=.025,ticks=[0,250,500]);bar.set_label('E rate (Hz)')
    stem=args.stem
    for ext in ('png','pdf','svg'):fig.savefig(folder/f'{stem}.{ext}',dpi=220,bbox_inches='tight')
    plt.close(fig)
    metadata_stem=stem.removeprefix('fig_')
    np.savez_compressed(OUT/f'{metadata_stem}.npz',frames=np.asarray(frames))
    report=dict(status='SPATIAL_OBSERVATIONS_COMPLETE',rows=rows,successive_D240_contrast=contrast,
        observable_order=['global_E','coreA_E','coreB_E','other_E'],display_bin_mm=1.,spatial_window_ms=50,
        model='Autonomous conditional-density model; fixed spatial Z, dynamic M',
        scope='Actual spatial snapshots prepared for branch annotations. No certified branch, stability or critical type inferred.',
        human_visual_acceptance='PENDING')
    (OUT/f'{metadata_stem}.json').write_text(json.dumps(safe(report),indent=2)+'\n')
    readme=folder/'README.md';old=readme.read_text() if readme.exists() else ''
    entry=f'\n### {stem}.png / .pdf / .svg\n\n固定空间 Z、保留动态 M 的密度模型真实空间帧：D=0.2375 的最后一个完整峰附近、D=0.24 的相邻两个峰，以及 D={rows[-1]["D"]:g} 首次持续活动段附近。每帧均用同一次实际运行中连续 50 ms 的放电率，按细胞数加权到 1 mm 展示网格，共用 0–500 Hz 色标；确切时窗见 {metadata_stem}.json。该图为分支标注准备空间观测，尚不赋予周期稳定性或分岔类型，待用户人工检查。**关注点**：相邻爆发的招募差异和持续活动开始时的空间分布；不能据此判哪个核造成分岔。\n'
    if f'### {stem}.' not in old:readme.write_text(old+entry)
    print(json.dumps(safe(report),indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--entry-case',default='D250_from_burst_fp64')
    ap.add_argument('--stem',default='fig_transition_spatial_anchors');main(ap.parse_args())
