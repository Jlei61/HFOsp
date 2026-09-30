"""Validation figures, deliberately not an unvalidated bifurcation diagram."""
from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Rectangle,Patch
from scipy.ndimage import gaussian_filter1d
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
NAMES=read(V10/'native/a/observation_contract.json')['contact_names']
DISPLAY=['SCL9','SCL8','SCL7','SCL6']+[f'ICL{k}' for k in range(11,0,-1)]
ORDER=[NAMES.index(n) for n in DISPLAY]
COLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
SIZES=np.bincount(np.load(V10/'native/a/trajectory.npz')['region'],minlength=6)
ENTRIES=[]

def save(fig,name,meaning,focus):
    fig.savefig(FIG/f'{name}.png',dpi=170,bbox_inches='tight',pad_inches=.12)
    fig.savefig(FIG/f'{name}.svg',bbox_inches='tight',pad_inches=.12);plt.close(fig)
    ENTRIES.append(f'### {name}.png\n{meaning} SVG 为同图可编辑版本。**关注点**：{focus}\n')

def contacts(ax,labels=True):
    ax.set(yticks=range(15),yticklabels=DISPLAY if labels else [],ylim=(14.5,-.5))
    for i,t in enumerate(ax.get_yticklabels()):t.set_color('#35a6b7' if i<4 else '#d48024')
    ax.axhline(3.5,color='white',lw=.8);ax.tick_params(axis='y',labelsize=8)

def rankplot(ax,summary,color,label):
    x=np.array(summary['mean_rank'],float)[ORDER]
    for ix in [np.arange(4),np.arange(4,15)]:ax.plot(x[ix],ix,'o-',ms=4,color=color,label=label if ix[0]==0 else None)
    ax.set(xlim=(-.05,1.05),xlabel='Mean normalized rank')
    contacts(ax)

def observations_figure():
    z=np.load(OUT/'native/848101/trajectory.npz');native=read(OUT/'summaries/native_848101.json')
    fig,axs=plt.subplots(4,2,figsize=(12,10.8),gridspec_kw={'width_ratios':[3.6,1.1]})
    vmax=float(z['contact_envelope'][2000:3000].max()/.002)
    for i,g in enumerate([None,10,20,40]):
        key='contact_envelope' if g is None else f'contact_envelope_{g}'
        env=z[key][2000:3000][:,ORDER].T/.002
        im=axs[i,0].imshow(env,extent=[4,6,14.5,-.5],aspect='auto',cmap='magma',vmin=0,vmax=vmax)
        contacts(axs[i,0]);axs[i,0].set_title('Native contact readout' if g is None else f'Observed SNN spikes, averaged within {20/g:g} mm bins',loc='left',pad=8)
        q=native if g is None else read(OUT/f'summaries/projection{g}_848101.json')
        rankplot(axs[i,1],native['summary'],'black','Native')
        if g:rankplot(axs[i,1],q['summary'],COLORS[i-1],f'{20/g:g} mm')
        axs[i,1].set_title(f'Valid events: {q["N"]}')
        if i==0:axs[i,1].legend(frameon=False,fontsize=8)
    axs[-1,0].set_xlabel('Time (s)')
    fig.suptitle('Does spatial averaging preserve the measured propagation?',fontsize=14)
    fig.tight_layout(rect=(0,0,.92,.96));cb=fig.add_axes([.95,.16,.015,.65]);fig.colorbar(im,cax=cb,label='Contact-weighted E rate (Hz)')
    save(fig,'01_observation_projection',
        '同一原生SNN的逐细胞触点读出，与2/1/0.5 mm分区后的真实spike活动读出比较；均使用同一触点核、时间平滑和冻结事件观察器。右侧显示所有有效事件的平均rank。',
        '这里尚未使用任何rate方程，差异只来自空间平均与随后的事件读出，不能归因于动力学闭合。')

def dynamics_figure():
    paths=[OUT/'native/848101/trajectory.npz']+[OUT/'rate'/f'grid{g}_seed848101.npz' for g in (10,20)]
    labels=['Native SNN','Spatial rate: 2 mm','Spatial rate: 1 mm']
    fig,axs=plt.subplots(3,2,figsize=(14,9.2),gridspec_kw={'width_ratios':[1.25,1.65]})
    vmax=max(float(np.load(p)['contact_envelope'][2000:3000].max()/.002) for p in paths)
    for row,(path,label) in enumerate(zip(paths,labels)):
        z=np.load(path);r=gaussian_filter1d(z['six_counts']/SIZES/.002,2.5,axis=0);t=(np.arange(len(r))+.5)*.002
        ix=(t>=4)&(t<=6);ax=axs[row,0]
        for j,c,n in [(0,COLORS[0],'Core A E'),(1,COLORS[1],'Core B E')]:ax.plot(t[ix],r[ix,j],color=c,lw=1.1,label=n)
        net=np.average(r[:,:3],weights=SIZES[:3],axis=1);ax.plot(t[ix],net[ix],color='black',lw=1.,label='All E')
        ax.set(xlim=(4,6),ylim=(0,500),ylabel='Rate (Hz / cell)',title=label)
        if row==0:ax.legend(frameon=False,fontsize=8,ncol=3)
        im=axs[row,1].imshow(z['contact_envelope'][ix][:,ORDER].T/.002,extent=[4,6,14.5,-.5],aspect='auto',cmap='magma',vmin=0,vmax=vmax)
        contacts(axs[row,1])
        if row==2:ax.set_xlabel('Time (s)');axs[row,1].set_xlabel('Time (s)')
    fig.suptitle(r'Matched substrate and OU input: $J_{\mathrm{EE,core}}=1.355$',fontsize=15)
    fig.tight_layout(rect=(0,0,.92,.95));cb=fig.add_axes([.95,.16,.015,.65]);fig.colorbar(im,cax=cb,label='Contact-weighted E rate (Hz)')
    save(fig,'02_dynamics_comparison',
        '固定拓扑2511及同一848101核内OU输入，比较原生SNN和2/1 mm空间rate候选的A/B、全E波形及触点包络。三行同为4–6秒窗口并共用绝对率范围。',
        '是否同时保留核心爆发及可检测的空间传播；同一OU不要求有限Poisson网络逐spike或逐事件相位完全相同。')

def metrics_figure():
    report=read(OUT/'comparison.json');ref=read(OUT/'summaries/native_848101.json')
    candidates=[(f'Rate {20/g:g} mm (N={read(OUT/f"summaries/rate{g}_848101.json")["N"]})',g) for g in (10,20)]
    fig=plt.figure(figsize=(14,5.8));gs=fig.add_gridspec(1,3,width_ratios=[1.5,1,1],wspace=.45)
    ax=fig.add_subplot(gs[0,0]);xs=np.arange(3);base=np.asarray([p['errors'] for p in report['native_pairs']],float)
    for j in range(3):
        ax.plot([j,j],[base[:,j].min(),base[:,j].max()],color='black',lw=7,alpha=.25)
        ax.scatter(np.full(3,j),base[:,j],color='black',s=20,label='SNN seed pairs' if j==0 else None)
    for i,(name,g) in enumerate(candidates):
        q=read(OUT/f'summaries/rate{g}_848101.json');v=np.array(q['errors_to_native'],float)
        ax.plot(xs+(.08 if i else -.08),v[0],'o-',color=COLORS[i],ms=6,label=name)
        for j in range(3):
            valid=v[:,j][np.isfinite(v[:,j])]
            if len(valid):ax.plot([j+(.08 if i else -.08)]*2,[valid.min(),valid.max()],color=COLORS[i],lw=1)
        if q['N']==0:ax.text(.04,.54,f'{20/g:g} mm: no valid events',transform=ax.transAxes,color=COLORS[i],fontsize=10)
    ax.set(xticks=xs,xticklabels=['Mean rank','Within-shaft\norder','Participation'],ylabel='Direct error relative to SNN',ylim=(0,1.02))
    ax.legend(frameon=False,fontsize=9,loc='upper left');ax.set_title('Three propagation summaries')
    ax=fig.add_subplot(gs[0,1]);rankplot(ax,ref['summary'],'black','SNN')
    for i,(name,g) in enumerate(candidates):rankplot(ax,read(OUT/f'summaries/rate{g}_848101.json')['summary'],COLORS[i],name)
    ax.set_title('Contact rank');ax.legend(frameon=False,fontsize=8)
    ax=fig.add_subplot(gs[0,2])
    for color,label,q in [('black','SNN',ref)]+[(COLORS[i],name,read(OUT/f'summaries/rate{g}_848101.json')) for i,(name,g) in enumerate(candidates)]:
        v=[q['summary']['contacts'][n]['participation'] for n in DISPLAY]
        for ix in [np.arange(4),np.arange(4,15)]:ax.plot(np.array(v,float)[ix],ix,'o-',color=color,ms=4)
    contacts(ax);ax.set(xlim=(-.05,1.05),xlabel='Participation probability',title='Contact participation')
    fig.suptitle('Model-to-model validation; unchanged event and contact definitions',fontsize=14,y=.98)
    fig.subplots_adjust(left=.07,right=.98,bottom=.13,top=.84);save(fig,'03_propagation_metrics',
        '三项误差直接以原生SNN为参照；黑色点和范围来自固定拓扑的三个噪声种子配对，彩色点为rate对同输入SNN的误差，竖线为对三条SNN的误差范围。另列逐触点rank及参与概率，不用平均分掩盖局部缺失。',
        '原生范围只是三种子的描述性参照，不是统计等价置信区间；缺失指标不填零。')

def spatial_figure():
    cfg=read(OUT/'model_config.json');xy=np.load(OUT/'grid20/model.npz')['contact_xy']
    sources=[('Native SNN',OUT/'native/848101/trajectory.npz','native_848101',20),
             ('Rate 2 mm',OUT/'rate/grid10_seed848101.npz','rate10_848101',10),
             ('Rate 1 mm',OUT/'rate/grid20_seed848101.npz','rate20_848101',20)]
    selections=[];maxima=[]
    for label,path,name,grid in sources:
        q=read(OUT/f'summaries/{name}.json');ob=read(OUT/f'observations/{name}.json')
        eligible=q['valid_event_ids']
        event=eligible[0] if eligible else None
        zero=float(np.nanmin(np.array(ob['centroid_ms'][event],float))) if event is not None else 4000.
        z=np.load(path);selections.append((label,z,name,grid,event,zero));maxima.append(float(np.max(z['field_counts'][1000:])/(20/grid)**2))
    vmax=max(maxima);fig,axs=plt.subplots(3,4,figsize=(12.6,9.4),sharex=True,sharey=True)
    for row,(label,z,name,grid,event,zero) in enumerate(selections):
        for col,offset in enumerate([-20,0,40,80]):
            frame=int(round((zero+offset)/2-.5));ax=axs[row,col]
            im=ax.imshow(z['field_counts'][frame]/(20/grid)**2,origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=vmax,interpolation='nearest')
            for c,r in zip(cfg['core_centers'],cfg['core_radii']):ax.add_patch(Circle(c,r,fill=False,ec='white',lw=1.))
            ax.scatter(xy[:,0],xy[:,1],s=12,facecolors='none',edgecolors='cyan',lw=.6)
            ax.set(xticks=[0,10,20],yticks=[0,10,20],title=f'{offset:+d} ms')
            if col==0:ax.set_ylabel(label+('\nNo valid event; t = 4 s' if event is None else '\nEvent-aligned')+'\ny (mm)')
            if row==2:ax.set_xlabel('x (mm)')
    fig.subplots_adjust(left=.09,right=.9,top=.9,bottom=.08,hspace=.33,wspace=.15)
    cb=fig.add_axes([.93,.2,.014,.57]);fig.colorbar(im,cax=cb,label='E spikes / 2 ms / mm² (rate: expected count)')
    fig.suptitle('Spatial activity: event-aligned; no-event model shown at 4 s',fontsize=14)
    save(fig,'04_spatial_propagation',
        '各模型独立选择2–12秒统计窗内第一个有效事件，并以最早触点质心对齐，显示−20/0/+40/+80 ms空间活动；没有有效事件时显示公共波形窗口起点4秒附近的场，并明确标出。原生为真实spike计数，rate为期望计数，统一换算每平方毫米并共享绝对色标。',
        '这是按预定规则展示的诊断事件，不宣称三行是同一个微观事件；结论必须结合所有事件的传播统计。')
    write(OUT/'snapshot_selection.json',[dict(model=x[0],event=x[4],zero_ms=x[5]) for x in selections])

def mixed_figure():
    z=np.load(OUT/'mixed_response.npz');r=gaussian_filter1d(z['microscopic_rate_hz'],50,axis=0)
    t=np.arange(len(r))*.0001;p=gaussian_filter1d(z['closure_rate_hz'],50,axis=0)
    fig,axes=plt.subplots(2,2,figsize=(13,6.6),gridspec_kw={'width_ratios':[1,1.5]})
    for row,name in enumerate(['Core A E','Core A I']):
        ix=(t>=.5)&(t<=3);ax=axes[row,0]
        for k,col,lab in [(0,COLORS[0],'Excitatory'),(1,COLORS[1],'Inhibitory')]:
            ax.plot(t[ix],z['currents'][ix,row,k],color=col,lw=1.1,label=lab)
        ax.set(xlabel='Time (s)',ylabel='Voltage-equivalent synaptic input (mV)',title=name+' mean input',xlim=(.5,3))
        if row==0:ax.legend(frameon=False,fontsize=8)
        ax=axes[row,1]
        ax.plot(t[ix],r[ix,row],color='black',lw=1.1,label='Microscopic LIF ensemble')
        ax.plot(t[ix],p[ix,0,row],color=COLORS[0],lw=1.3,label='Rate: E 5 / I 2.5 ms')
        ax.plot(t[ix],p[ix,1,row],color=COLORS[1],lw=1.3,ls='--',label='Rate: E 20 / I 10 ms')
        ax.set(xlabel='Time (s)',ylabel=name+' rate (Hz)',title='Prescribed native input moments',xlim=(.5,3))
    axes[0,1].legend(frameon=False,fontsize=8)
    fig.suptitle('Mixed E/I input replay: auxiliary response diagnostic, no fitting',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.95));save(fig,'05_mixed_input_response',
        '将原生SNN前3秒的群体发放率按原始延迟投影成时变输入一、二阶矩，驱动具有实际阈值与突触/膜/不应期参数的孤立LIF集合。左列为集合平均E/I输入，右列比较微观响应及两种未拟合的rate响应时间常数。',
        '这是规定输入下的局部响应诊断，丢弃了输入相关性与闭环反馈；均值接近不等于完整空间网络动力学对应。')

def adaptive_figure():
    z=np.load(OUT/'adaptive_readout.npz');q=read(OUT/'adaptive_readout.json');cfg=read(OUT/'model_config.json')
    fig,axes=plt.subplots(1,2,figsize=(12.8,6.2),gridspec_kw={'width_ratios':[1.15,1]})
    ax=axes[0];colors={2.:'#f6f6f6',1.:'#cce5f3',.5:'#f7d5a9'}
    for x,y,w in z['tiles']:ax.add_patch(Rectangle((x,y),w,w,facecolor=colors[w],edgecolor='#888888',lw=.35))
    for n,c,r in zip(['A','B'],cfg['core_centers'],cfg['core_radii']):
        ax.add_patch(Circle(c,r,fill=False,ec='black',lw=1.3));ax.text(c[0],c[1],n,ha='center',va='center',fontsize=12)
    xy=z['contact_xy'];ax.scatter(xy[:,0],xy[:,1],s=30,fc='#35a6b7',ec='black',lw=.5,zorder=4)
    ax.set(xlim=(0,20),ylim=(0,20),xlabel='x (mm)',ylabel='y (mm)',aspect='equal',title='382 spatial tiles; 813 E/I and region populations')
    ax.legend(handles=[Patch(fc=colors[w],ec='#888888',label=f'{w:g} mm') for w in [2.,1.,.5]],ncol=3,frameon=False,loc='upper right')
    ax=axes[1];native=np.array([v['errors'] for v in read(OUT/'comparison.json')['native_pairs']]);values=np.array([v['errors'] for v in q['rows']])
    for i in range(3):
        for offset,v,color,label in [(-.06,native[:,i],'black','SNN seed pairs'),(.06,values[:,i],COLORS[0],'Adaptive projection')]:
            ax.plot([i+offset]*2,[v.min(),v.max()],color=color,lw=2)
            ax.scatter(np.full(3,i+offset),v,s=45,color=color,label=label if i==0 else None)
    ax.set(xticks=range(3),xticklabels=['Mean rank','Within-shaft\norder','Participation'],ylabel='Direct error relative to native readout',ylim=(0,.43),title='Observed-spike projection only')
    ax.legend(frameon=False,fontsize=10)
    fig.suptitle('Refine the SEEG contact footprint; preserve explicit 2D geometry',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94));save(fig,'06_adaptive_spatial_readout',
        '仅按空间几何建立2 mm背景、1 mm核心邻域、0.5 mm触点邻域的分区；触点邻域定义为原高斯核的3倍sigma范围。右列用三条真实SNN活动检验该分区的读出损失，保留28/27/29个有效事件，三项误差与全局0.5 mm分区相同。',
        '这支持把局部细化作为下一版空间离散候选，尚不证明该网格能自主复现完整二维传播动力学。')

def main():
    observations_figure();dynamics_figure();metrics_figure();spatial_figure();mixed_figure();adaptive_figure()
    (FIG/'README.md').write_text('\n'.join(ENTRIES))
    write(OUT/'figure_manifest.json',dict(figures=[p.name for p in FIG.glob('*.png')],human_visual_acceptance='PENDING'))

if __name__=='__main__':main()
