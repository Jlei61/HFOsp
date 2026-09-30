"""Matched-time spatial readout for the equal-global-Z intervention."""
from native_path import *
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    q=read(OUT/'equal_mean_Z_control.json');assert q['status']=='COMPLETE'
    s=model();cases={row['condition']:row for row in q['rows']}
    labels=['affine_same_global_D','actual7770'];zs=[np.load(cases[k]['source']) for k in labels]
    counts=zs[0]['cell_counts'];assert np.array_equal(counts,zs[1]['cell_counts'])
    g=zs[0]['field_E_hz']@(counts/counts.sum())
    smooth=np.convolve(g,np.ones(50)/50,mode='same')
    peaks=find_peaks(smooth,height=20,distance=150)[0]
    peak=int(next(x for x in peaks if 6000<=x<=len(g)-150));centers=[peak,peak+100]
    E=s.E;cells=s.geo['group_cell'];size=s.sizes
    delta=zs[1]['Z_source']-zs[0]['Z_source']
    field=np.bincount(cells[E],weights=(size*delta)[E],minlength=400)/counts
    lim=float(np.max(abs(field)))
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig=plt.figure(figsize=(10.8,6.2));grid=fig.add_gridspec(2,3,width_ratios=[1.25,1,1],
        left=.075,right=.87,bottom=.12,top=.93,wspace=.47,hspace=.33)
    left=fig.add_subplot(grid[:,0]);di=left.imshow(field.reshape(20,20),origin='lower',extent=[0,20,0,20],
        cmap='RdBu_r',vmin=-lim,vmax=lim)
    left.set_xlabel('x (mm)');left.set_ylabel('y (mm)')
    left.text(-.20,1.06,'A',transform=left.transAxes,fontweight='bold',fontsize=15)
    bar=fig.colorbar(di,ax=left,orientation='horizontal',fraction=.07,pad=.15)
    bar.set_label(r'$Z_{\mathrm{recorded}}-Z_{\mathrm{interpolated}}$')
    axes=[left];windows=[]
    for row,z in enumerate(zs):
        for col,t in enumerate(centers):
            ax=fig.add_subplot(grid[row,col+1]);axes.append(ax)
            lo,hi=t-25,t+25;rate=z['field_E_hz'][lo:hi].mean(0)
            im=ax.imshow(rate.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
            ax.text(-.19,1.05,chr(ord('B')+row*2+col),transform=ax.transAxes,fontweight='bold',fontsize=15)
            if row==0:ax.text(.5,1.05,['Matched peak','+100 ms'][col],transform=ax.transAxes,ha='center')
            if col==0:ax.set_ylabel(['Interpolated Z','Recorded Z'][row]+'\ny (mm)')
            if row==1:ax.set_xlabel('x (mm)')
            windows.append(dict(condition=labels[row],start_sample_index=lo,stop_sample_index_exclusive=hi,
                start_ms=float(z['time_ms'][lo]),last_sample_ms=float(z['time_ms'][hi-1]),
                global_rate_hz=float(rate@(counts/counts.sum()))))
    for ax in axes:
        ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
        for center,label in zip(s.geo['centers_mm'],'AB'):
            ax.add_patch(Circle(center,1.5,fill=False,color='#18c1cf',lw=1.1))
            ax.text(center[0],center[1]+2.,label,color='#18c1cf',ha='center',fontsize=9)
    cax=fig.add_axes([.905,.23,.017,.55]);fig.colorbar(im,cax=cax,label='E rate (Hz)',ticks=[0,250,500])
    folder=OUT/'figures';name='fig_equal_mean_Z_spatial_control'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=180)
    plt.close(fig)
    write(folder/f'{name}.json',dict(source=str(OUT/'equal_mean_Z_control.json'),D=q['D'],global_Z=q['global_Z'],
        windows=windows,selection='First peak of affine-condition 50-ms global rate after continuation6s; exact same windows in both conditions',
        difference_field='Recorded rate7770ms Z minus affine endpoint-slice Z at the identical global mean',
        scope='Matched deterministic finite8s history; not a bifurcation type or global-saturation claim',human_visual_acceptance='PENDING'))
    readme=folder/'README.md';heading=f'### {name}.png / .pdf / .svg'
    text=readme.read_text();description=('两组全局平均Z完全相同、完整快状态与M及延迟历史相同，仅Z空间分布不同；左侧显示逐格Z差。'
        '右侧上排为端点直线插值场，下排为rate自身7.770s记录场，两排使用续接后完全相同的50ms时间窗；列为上排所选事件峰值及其后100ms，色标均为0–500Hz。'
        '**关注点**：相同平均耗减下，一组恢复低活动，另一组仍有局部持续，说明本次有限时间状态不能仅由平均Z决定；记录场时刻与续接显示时刻是两个不同的时间标签。\n')
    if heading not in text:readme.write_text(text+'\n'+heading+'\n'+description)
    log('EQUAL Z FIGURE',folder/f'{name}.png')


if __name__=='__main__':main()
