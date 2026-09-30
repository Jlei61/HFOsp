"""Actual fine-time Z fields: regular events, irregular events, local persistence."""
from native_path import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    s=model();cases=read(OUT/'fine_Z_path_controls.json')
    rows={q['actual_Z_time_ms']:q for q in cases['rows']}
    assert cases['status']=='COMPLETE'
    plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none'})
    fig,axs=plt.subplots(2,3,figsize=(10.0,6.9),sharex=True,sharey=True,
                         gridspec_kw=dict(left=.09,right=.87,bottom=.12,top=.87,wspace=.19,hspace=.19))
    records=[]
    for j,t in enumerate([7750,7760,7770]):
        q=rows[t];z=np.load(q['source']);f=z['field_E_hz'];counts=z['cell_counts']
        g=f@(counts/counts.sum());smooth=np.convolve(g,np.ones(50)/50,mode='same')
        if t==7750:
            peak=len(g)-4000+int(np.argmax(smooth[-4000:-300]))
            selection='Largest 50-ms global E rate in the last4s, excluding final300ms'
        elif t==7760:
            ev=next(e for e in q['events'] if e['duration_ms']>=200)
            lo,hi=int(ev['start_ms']),int(ev['end_ms'])
            peak=lo+int(np.argmax(g[lo:hi]));selection='Peak of first complete event lasting at least200ms'
        else:
            peak=100+int(np.argmax(smooth[100:1000]))
            selection='Largest 50-ms global E rate in the first100--1000ms after Z transplant'
        windows=[]
        for i,center in enumerate([peak,peak+100]):
            ax=axs[i,j];win=[center-25,center+25];assert 0<=win[0]<win[1]<=len(g)
            rate=f[win[0]:win[1]].mean(0)
            im=ax.imshow(rate.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
            ax.set_xticks([0,10,20]);ax.set_yticks([0,10,20])
            if j==0:ax.set_ylabel(('Peak' if i==0 else '+100 ms')+'\ny (mm)')
            if i==1:ax.set_xlabel('x (mm)')
            for c,label in zip(s.geo['centers_mm'],'AB'):
                ax.add_patch(Circle(c,1.5,fill=False,ec='#20c4cf',lw=1.1))
                ax.text(c[0],c[1]+2.,label,color='#20c4cf',ha='center',fontsize=9)
            if i==0:
                ax.text(-.15,1.18,chr(65+j),transform=ax.transAxes,fontweight='bold',fontsize=15)
                ax.text(.5,1.08,f'$Z({t/1000:.3f}\\,s)$\n$D={q["D"]:.6f}$',
                        transform=ax.transAxes,ha='center',fontsize=11)
            windows.append(dict(window_ms=win,global_mean_hz=float(rate@(counts/counts.sum()))))
        records.append(dict(actual_Z_time_ms=t,D=q['D'],category=q['category'],
                            source=q['source'],selection=selection,windows=windows))
    cax=fig.add_axes([.90,.2,.018,.57]);fig.colorbar(im,cax=cax,ticks=[0,250,500],label='E rate (Hz)')
    dest=OUT/'figures';name='fig_actual_fine_Z_state_transition'
    for ext in ['png','pdf','svg']:fig.savefig(dest/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(dest/f'{name}.json',dict(status='COMPLETE',rows=records,shared_initial=cases['shared_initial'],
        Z='Actual recorded spatial fields, held',M='dynamic',
        scope='Separate actual-field control; not points on the affine conditional bifurcation branch',
        time_labels='Source time of the transplanted Z field; image windows are continuation times in metadata',
        human_visual_acceptance='PENDING'))
    f=dest/'README.md';text=f.read_text();heading=f'### {name}.png / .pdf / .svg'
    if heading not in text:
        f.write_text(text+'\n'+heading+'\n'
            '三列分别冻结实际rate轨迹7.750、7.760、7.770秒记录的完整空间Z场，从相同快状态、M和延迟历史续接；M始终动态。'
            '分别出现规则自限、含长事件但仍可自限、不规则局部持续；每列展示一次活动峰值及100ms后实际50ms窗口，使用相同0–500Hz色标。'
            '**关注点**：列上时间是Z场来源时刻，不是续接图像的时钟；这是实际细路径的对照，不能与主图直线Z参数切面的点直接拼接。\n')
    log('FINE Z FIGURE',dest/f'{name}.png')


if __name__=='__main__':main()
