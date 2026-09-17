"""Numbered, observable-matched native waveform/raster atlas."""
from common import *
import numpy as np,csv,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.ndimage import gaussian_filter1d
sys.path.append(str(ROOT/'scripts/topic4_burst_regime'))
import metrics_v2
OLD=ROOT/'results/topic4_sef_hfo/burst_regime_map_20260914'
VALUES=[.5,.7,.85,1.,1.10,1.14,1.17623,1.17632,1.24,1.25,1.34,1.355,1.37,1.375,1.38,1.385,1.395,1.45,1.12181,1.12183]
LABELS=['Original resting example','Original irregular example','Original intermediate example','Original regular example','Below reduced fold','Above reduced fold','Below reduced recruitment interval','Above reduced recruitment interval','Below reduced waveform-change region','Above reduced waveform-change region','Below reduced PD1','Above reduced PD1 / below reduced LP1','Above reduced LP1','Below reduced PD3','Above reduced PD3','Below reduced PD2','Above reduced PD2','Right continuation','Below newly found reduced cycle fold','Above newly found reduced cycle fold']
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
COL=['#286aa4','#9553a3','#28865c','#df8243','#c05f88','#80804a']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':15,'axes.labelsize':13,'legend.fontsize':10,'axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42})

def fetch(n,g):
    folder=(OLD/'per_run'/f'ee{g:g}_d1_n1_t2511_s848101') if n<=4 else (OUT/'native/per_run'/f'J{g:g}_s848101')
    if not (folder/'result.json').exists():return None
    result=read(folder/'result.json');path=folder/'trajectory.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==result['arrays_sha256']
    z=np.load(path);names=z['group_names'].tolist();counts=z['spike_counts_2ms'];t=(np.arange(len(counts))+.5)*.002
    region=np.load(V2/'projected_graph.npz')['region'];sizes=np.bincount(region,minlength=6)
    exact='exact_spike_time_ms' in z
    if exact:
        six=z['six_group_counts_2ms'];rt=z['exact_spike_time_ms']/1000;ri=z['exact_spike_cell']
    else:
        six=None;rt=z['raster_time_ms']/1000-.001;ri=z['raster_cell']
    rate=counts/z['group_sizes'][None,:]/.002
    full=(counts[:,names.index('allE')]+counts[:,names.index('allI')])/40000/.002
    a=rate[:,names.index('coreAE')];b=rate[:,names.index('coreBE')]
    keep=(t>=2)&(t<12)
    # Recompute both cores on the same 2-12s observation window; original names remain historical labels.
    short=OUT/'native_metrics'/f'{n:02d}';short.mkdir(parents=True,exist_ok=True)
    met={}
    import metrics
    for gn in ('coreAE','coreBE','allE'):
        j=names.index(gn);size=int(z['group_sizes'][j]);m=metrics.summarize(z['active_counts_10ms'][:1200,j]/size,counts[:6000,j],size,burnin_s=2.,runaway=result['runaway_early_stop_ms'] is not None)
        met[gn]=metrics_v2.amend(m)
    (short/'metrics.json').write_text(json.dumps(met,indent=2)+'\n')
    means=[float(a[keep].mean()),float(b[keep].mean())] if keep.any() else [None,None]
    row=dict(number=n,g=g,context=LABELS[n-1],path=str(path),sha256=result['arrays_sha256'],means=means,network_mean=float(full[keep].mean()) if keep.any() else None,
        window_s=[2.,min(12.,len(t)*.002)],requested_window_s=[2,12],actual_duration_ms=len(t)*2,exact_spike_raster=exact,
        labels={k:v['label'] for k,v in met.items()},cv={k:v.get('cv') for k,v in met.items()},
        cv2={k:v.get('cv2') for k,v in met.items()},n_bursts={k:v['n_bursts'] for k,v in met.items()},
        iei_median_s={k:v.get('iei_median_s') for k,v in met.items()})
    return row,dict(z=z,t=t,full=full,a=a,b=b,rate=rate,names=names,six=six,sizes=sizes,rt=rt,ri=ri,region=region,sample=z['raster_sample_ids'],exact=exact)

def figure(row,d,zoom):
    # Every condition uses identical windows; no visually selected best event.
    start,end=(4.,4.3) if zoom else (4.,7.)
    if d['t'][-1]<end:start=max(0.,d['t'][-1]-(.3 if zoom else 3.));end=d['t'][-1]
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),gridspec_kw={'height_ratios':[1,2.1]})
    fig.subplots_adjust(left=.065,right=.985,bottom=.1,top=.86,wspace=.28,hspace=.18)
    fig.suptitle(f"{row['number']:02d}    Native SNN    $J_{{\\mathrm{{EE,core}}}}={row['g']:g}$",fontsize=19,y=.975)
    fig.text(.5,.918,row['context'],ha='center',fontsize=14)
    titles=['Whole network (E + I)','Core A','Core B']
    for col in range(3):
        ax,ras=axes[:,col];ax.set_title(titles[col]);ax.set_xlim(start,end);ras.set_xlim(start,end)
        mask=(d['t']>=start)&(d['t']<=end)
        if col==0:
            traces=[(d['full'],'All cells','#222222')]
            for gn,name,color in [('allE','E',COL[0]),('allI','I',COL[3])]:traces.append((d['rate'][:,d['names'].index(gn)],name,color))
            groups=list(range(6));pergroup=20
        else:
            k=col-1;traces=[(d['a'] if k==0 else d['b'],'E',COL[k])]
            if d['six'] is not None:traces.append((d['six'][:,k+3]/d['sizes'][k+3]/.002,'I',COL[k+3]))
            groups=[k,k+3] if d['exact'] else [k];pergroup=30
        for rate,name,color in traces:
            yy=gaussian_filter1d(rate,1.5)
            ax.plot(d['t'][mask],yy[mask],color=color,lw=1.6,label=name)
        ax.set_ylim(bottom=0);ax.set_ylabel('Rate (Hz / cell)');ax.legend(loc='upper right',frameon=False,ncol=len(traces));ax.tick_params(labelbottom=False)
        yt=[];yl=[];offset=0
        for group in groups:
            allids=np.flatnonzero(d['region']==group);canonical=allids[np.linspace(0,len(allids)-1,min(100,len(allids)),dtype=int)]
            ids=np.intersect1d(d['sample'],canonical);ids=ids[np.linspace(0,len(ids)-1,min(pergroup,len(ids)),dtype=int)]
            if not len(ids):continue
            m=np.isin(d['ri'],ids)&(d['rt']>=start)&(d['rt']<=end)
            rr=np.searchsorted(ids,d['ri'][m])+offset
            ras.scatter(d['rt'][m],rr,s=(34 if col else 18),marker='|',linewidths=1.3,color=COL[group],rasterized=True)
            yt.append(offset+(len(ids)-1)/2);yl.append(['A E','B E','Sur E','A I','B I','Sur I'][group])
            # Empty separators distinguish populations and keep the sparse
            # legacy I labels readable. They do not add recorded neurons.
            offset+=max(len(ids),10)+3
            ras.axhline(offset-2,color='#dddddd',lw=.65)
        ras.set_ylim(-1,offset);ras.set_yticks(yt,yl);ras.set_xlabel('Time (s)');ras.set_ylabel('Fixed sampled neurons')
    return fig,[start,end]

def main():
    rows=[];manifest=[];captions=[]
    with PdfPages(FIG/'native_network_AB_atlas.pdf') as book:
        for n,g in enumerate(VALUES,1):
            obj=fetch(n,g)
            if obj is None:continue
            row,d=obj;rows.append(row)
            for zoom in (False,True):
                name=f'native_{n:02d}_'+('zoom' if zoom else 'waveform_raster')
                fig,window=figure(row,d,zoom)
                for ext in ('png','pdf'):fig.savefig(FIG/f'{name}.{ext}',dpi=190)
                book.savefig(fig);plt.close(fig)
                manifest.append(dict(name=name,number=n,g=g,window_s=window,source=row['path'],exact_raster=row['exact_spike_raster']))
                typ='0.1ms积分步真实spike' if row['exact_spike_raster'] else '旧样本的2ms内至少一次spike记录'
                captions.append(f"### {name}.png / .pdf\n编号{n}，J={g:g}。三列为全网络、A、B，展示全群体放电率和固定细胞样本的 raster；率以2ms计数并用σ=3ms平滑，时间窗{window}秒。Raster为{typ}，全网络图的E/I人口分母为32000/8000；原编号1–4核心raster只显示E细胞。\n**关注点**：{row['context']}；这是原生SNN有限时窗例子，临界点名称仅指对应的确定性模型参数附近。\n")
    write('native_coordinates.json',rows);write('native_figure_manifest.json',manifest)
    (FIG/'NATIVE_README.md').write_text('\n'.join(captions))
    with (OUT/'native_conditions.csv').open('w') as f:
        cols=['number','g','context','mean_A_hz','mean_B_hz','network_mean_hz','A_label','B_label','A_CV','B_CV','A_CV2','B_CV2','A_n_bursts','B_n_bursts','source']
        w=csv.DictWriter(f,fieldnames=cols);w.writeheader()
        for r in rows:w.writerow(dict(zip(cols,[r['number'],r['g'],r['context'],*r['means'],r['network_mean'],r['labels']['coreAE'],r['labels']['coreBE'],r['cv']['coreAE'],r['cv']['coreBE'],r['cv2']['coreAE'],r['cv2']['coreBE'],r['n_bursts']['coreAE'],r['n_bursts']['coreBE'],r['path']])))
    print('NATIVE_FIGURES',len(rows),len(manifest),flush=True)
if __name__=='__main__':main()
