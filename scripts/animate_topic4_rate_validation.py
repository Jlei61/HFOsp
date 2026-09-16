#!/usr/bin/env python3
"""Show the complete finite diagnostic window, without picking successful events."""
from validate_topic4_fixed_rate_base import OUT, read, write
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from PIL import Image


def main(name='sham_7101'):
    snn=np.load(OUT/'snn'/f'{name}.npz');dt=float(snn['dt_ms'])
    models=[('SNN',snn['field_e_hz_20'],snn['rate_e_hz'],'black')]
    for closure,label,color in [('cascade','Original rate','#228833'),('cascade_colored','Corrected rate (development)','#cc3311')]:
        a=np.load(OUT/'rate'/f'{name}_grid20_{closure}.npz')
        models.append((label,a['field_rates_hz'][:,0],a['rates_hz'][:,0],color))
    # Same 8-ms presentation bins; rate uses four sampled values, not spike counts.
    fields=[f[:len(f)//4*4].reshape(-1,4,400).mean(1).reshape(-1,20,20) for _,f,_,_ in models]
    vmax=max(float(f.max()) for f in fields)
    fig=plt.figure(figsize=(10.5,6.2),layout='constrained');gs=fig.add_gridspec(2,3,height_ratios=[1.5,1])
    maps=[]
    for j,(label,_,_,_) in enumerate(models):
        ax=fig.add_subplot(gs[0,j]);im=ax.imshow(fields[j][0],origin='lower',extent=[0,20,0,20],vmin=0,vmax=vmax,cmap='magma')
        ax.set(title=label,xlabel='x (mm)',ylabel='y (mm)');maps.append(im)
    fig.colorbar(maps[-1],ax=[im.axes for im in maps],label='Local E rate (Hz)',shrink=.75)
    ax=fig.add_subplot(gs[1,:]);t=np.arange(len(models[0][2]))*dt
    for label,_,rate,color in models:ax.plot(t,uniform_filter1d(rate.astype(float),100),color=color,lw=1.2,label=label)
    cursor=ax.axvline(0,color='#777777',ls='--');ax.set(xlabel='Time (ms)',ylabel='Mean E rate (Hz)',xlim=(0,t[-1]))
    ax.legend(fontsize=8,ncol=3,loc='upper right');ax.set_ylim(-1,ax.get_ylim()[1]*1.2)
    title=fig.suptitle('Same frozen network and external input | t = 0 ms\nComplete 2.4-s diagnostic; Z/M off; 8-ms display bins',fontsize=12)
    frames=[];dest=OUT/'figures';dest.mkdir(exist_ok=True)
    for k in range(len(fields[0])):
        for j,im in enumerate(maps):im.set_data(fields[j][k])
        cursor.set_xdata([k*8,k*8]);title.set_text(f'Same frozen network and external input | t = {k*8} ms\nComplete 2.4-s diagnostic; Z/M off; 8-ms display bins')
        fig.canvas.draw();frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
        frames.append(frame.convert('P',palette=Image.Palette.ADAPTIVE,colors=128))
        if k in (12,62,125,187,250):frame.save(dest/f'{name}_comparison_frame_{k:03d}.png')
    stem=dest/f'{name}_full_window_comparison';frames[0].save(str(stem)+'.gif',save_all=True,append_images=frames[1:],duration=40,loop=0)
    plt.close(fig)
    write(stem.with_suffix('.json'),{'source':name,'simulation_duration_ms':len(fields[0])*8,'frames':len(frames),
        'display_interval_ms':8,'playback_ms_per_frame':40,'color_scale_hz':[0,vmax],
        'selection':'entire simulation, no event filtering or alignment','spatial_readout':'native sheet, no electrode smoothing',
        'sampling':'SNN 8-ms spike-count rates; rate average of four 2-ms samples',
        'status':'DEVELOPMENT_DIAGNOSTIC_NOT_PATIENT_OR_DYNAMICS_ACCEPTANCE','author_visual_acceptance':False})
    p=dest/'README.md';old=p.read_text() if p.exists() else '';header=f'### {stem.name}.gif'
    if header not in old:p.write_text(old+f'\n{header}\n完整2.4秒无刺激窗口，对比同一固定双核和实际外源输入下SNN、原rate及输入相关性修正rate；不挑选事件，不做电极空间平滑。统一颜色范围，8毫秒展示一帧；SNN为计数率，rate为四个2毫秒采样的平均，播放慢放5倍。\n**关注点**：多次活动的产生、回落与二维传播是否一致；修正模型仍是开发诊断，不能因群体均值相近就验收。\n')
    print(stem,flush=True)


if __name__=='__main__':main()
