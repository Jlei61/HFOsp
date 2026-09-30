"""Spatial windows of the final complete event in the held-Z 72 s experiment."""
from native_path import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def main():
    q=read(OUT/'native_Z_long_readout.json');assert q['status']=='COMPLETE'
    event=q['canonical_whole']['events'][-1]
    fields=np.concatenate([np.load(path)['field_E_hz'] for path in q['source_files']])
    start,end=event['start_ms'],event['end_ms']
    centers=[int(start-50),int(start+150),int((start+end)/2),int(end+60)]
    s=model();plt.rcParams.update({'font.size':12,'pdf.fonttype':42,'svg.fonttype':'none',
                                   'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,4,figsize=(11.5,3.4),layout='constrained',sharey=True)
    windows=[]
    for i,(a,t) in enumerate(zip(axs,centers)):
        lo,hi=t-25,t+25;rate=fields[lo:hi].mean(0)
        im=a.imshow(rate.reshape(20,20),origin='lower',extent=[0,20,0,20],cmap='magma',vmin=0,vmax=500)
        a.text(0,1.06,f'{"ABCD"[i]}   {t/1000:.3f} s',transform=a.transAxes)
        a.set_xticks([0,10,20]);a.set_yticks([0,10,20]);a.set_xlabel('x (mm)')
        if i==0:a.set_ylabel('y (mm)')
        for c,label in zip(s.geo['centers_mm'],'AB'):
            a.add_patch(Circle(c,1.5,fill=False,ec='#20c4cf',lw=1.1))
            a.text(c[0],c[1]+2,label,color='#20c4cf',ha='center',fontsize=9)
        windows.append(dict(panel='ABCD'[i],start_sample=lo,end_sample_exclusive=hi))
    fig.colorbar(im,ax=axs.tolist(),fraction=.025,pad=.025,label='E rate (Hz)',ticks=[0,250,500])
    name='fig_native_Z78_late_self_termination';folder=OUT/'figures'
    for ext in ['png','pdf','svg']:fig.savefig(folder/f'{name}.{ext}',dpi=190)
    plt.close(fig)
    write(folder/f'{name}.json',dict(source=str(OUT/'native_Z_long_readout.json'),event=event,
        windows=windows,selection='Last complete event; before onset50ms, onset+150ms, event midpoint, end+60ms; all50ms windows',
        time_origin='Start of the continuous held-native-Z rate continuation; not original Fig.5 time',
        D=q['D'],global_Z=q['global_Z'],Z='held',M='dynamic',human_visual_acceptance='PENDING'))
    readme=folder/'README.md';text=readme.read_text();heading=f'### {name}.png / .pdf / .svg'
    description=('原生空间Z路径上全局Z固定为.7804，M动态；图示72秒连续rate续接中最后一个完整事件，其持续3.43秒。'
        '四列依次取事件前50ms、起点后150ms、事件中点和结束后60ms，显示时间以这次续接初态为零，每幅均为50ms平均空间率。'
        '**关注点**：在临近69秒仍能回到低活动；长事件不能仅凭短窗口认定为永久runaway，此图不证明渐近吸引态或分岔类型。\n')
    if heading not in text:readme.write_text(text+'\n'+heading+'\n'+description)
    print(folder/f'{name}.png')


if __name__=='__main__':main()
