"""Unselected six-second native movie with all-contact readout and core rates.

No event-window concatenation: one fixed interval immediately after analysis
burn-in. Native 2-ms samples are displayed every 4 ms, with an explicit clock.
"""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from PIL import Image,ImageDraw,ImageFont
from scripts import analyze_topic4_propagation_recovery_night as review
an=review.an;rt=review.rt
FFMPEG=Path('/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/ffmpeg')


def render(c,seed,r,a,lo,hi,out):
    dt=float(a['sheet_activity_frame_ms']);assert dt==2.
    frames=np.arange(round(lo/dt),round(hi/dt),2);movie=a['sheet_activity_counts'];vmax=max(float(movie[frames].max()),1.)
    fig=plt.figure(figsize=(11,5),dpi=100);grid=fig.add_gridspec(1,2,left=.055,right=.985,bottom=.12,top=.83,width_ratios=[1,1.75],wspace=.25)
    left=fig.add_subplot(grid[0,0]);right=grid[0,1].subgridspec(2,1,height_ratios=[2,1],hspace=.3)
    readout=fig.add_subplot(right[0,0]);rates=fig.add_subplot(right[1,0]);names=list(a['contact_names']);order=[names.index(x) for x in an.DISPLAY]
    im=left.imshow(np.zeros((20,20)),origin='lower',extent=[0,20,0,20],cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
    left.set(xlabel='x (mm)',ylabel='y (mm)',title='全部原生 2 ms 活动',xticks=[0,10,20],yticks=[0,10,20])
    fig.colorbar(im,ax=left,location='bottom',fraction=.035,pad=.18,label='活动 E 神经元数 / 1 mm 网格')
    edt=float(a['contact_envelope_dt_ms']);env=a['contact_envelope'][round(lo/edt):round(hi/edt),order].T
    evmax=max(float(env.max()),1e-20)
    readout.imshow(env,aspect='auto',extent=[lo/1000,hi/1000,14.5,-.5],cmap='magma',vmin=0,vmax=evmax,interpolation='nearest')
    readout.set(yticks=range(15),yticklabels=an.DISPLAY,xticks=[],title='全部触点：连续发放密度包络');readout.tick_params(axis='y',labelsize=6)
    tt=a['trace_time_ms'];sel=(tt>=lo)&(tt<hi)
    for group,color,label in [('coreAE','#bd3934','左核 E'),('coreBE','#2679b0','右核 E'),('surroundE','#777777','核外 E')]:
        rate=gaussian_filter1d(a['trace_'+group+'_spikes'].astype(float)*1000/len(a['group_'+group]),5.)
        rates.plot(tt[sel]/1000,rate[sel],color=color,lw=.7,label=label)
    rates.set(xlim=(lo/1000,hi/1000),xlabel='实际时间 (s)',ylabel='每神经元率 (Hz)');rates.legend(loc='upper right',fontsize=7,ncol=3)
    rates.tick_params(labelsize=7)
    fig.suptitle(review.display(c)+f'｜网络 {c["topology"]}，噪声 {seed}',fontsize=11,y=.98)
    fig.canvas.draw();W,H=fig.canvas.get_width_height();base=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy())
    # Pixel locations are derived from the real axes; no changed spatial mapping.
    def point(ax,x,y):
        px,py=ax.transData.transform((x,y));return float(px),float(H-py)
    x0,ybottom=point(left,0,0);x1,ytop=point(left,20,20)
    box=(round(x0),round(ytop),round(x1),round(ybottom));size=(box[2]-box[0],box[3]-box[1])
    contact_pixels=[point(left,*xy) for xy in a['contact_xy_mm']]
    circles=[]
    for xy,rad in zip(c['centers_mm'],c['radii_mm']):
        p0=point(left,xy[0]-rad,xy[1]+rad);p1=point(left,xy[0]+rad,xy[1]-rad);circles.append((*p0,*p1))
    boxes=[]
    for ax in [readout,rates]:
        p0=point(ax,lo/1000,ax.get_ylim()[0]);p1=point(ax,hi/1000,ax.get_ylim()[1]);boxes.append((min(p0[0],p1[0]),min(p0[1],p1[1]),max(p0[0],p1[0]),max(p0[1],p1[1])))
    plt.close(fig)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    cmap=plt.get_cmap('inferno');lut=(cmap(np.linspace(0,1,256))[:,:3]*255).astype(np.uint8)
    stem=f'{c["id"]}_{seed}_continuous_native';mp4=out/f'{stem}.mp4';gif=out/f'{stem}.gif';poster=out/f'{stem}.png'
    maximum_frame=int(frames[np.argmax(movie[frames].sum((1,2)))])
    with (out/f'{stem}.ffmpeg.log').open('w') as log:
      proc=subprocess.Popen([str(FFMPEG),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','20','-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(mp4)],stdin=subprocess.PIPE,stderr=log)
      try:
        for index in frames:
            t=(index+.5)*dt;canvas=base.copy();pixels=lut[np.clip(movie[index][::-1]/vmax*255,0,255).astype(np.uint8)]
            canvas.paste(Image.fromarray(pixels).resize(size,Image.Resampling.NEAREST),box[:2]);draw=ImageDraw.Draw(canvas)
            for circle in circles:draw.ellipse(circle,outline='white',width=1)
            for shaft,col in [('ICL','#dc8722'),('SCL','#42a9b5')]:
                ix=[i for i,n in enumerate(names) if n.startswith(shaft)];draw.line([contact_pixels[i] for i in ix],fill=col,width=1)
                for i in ix:
                    x,y=contact_pixels[i];draw.ellipse((x-2,y-2,x+2,y+2),outline=col,width=1)
            for bi,b in enumerate(boxes):
                x=b[0]+(t-lo)/(hi-lo)*(b[2]-b[0]);draw.line((x,b[1],x,b[3]),fill='white' if bi==0 else 'black',width=1)
            draw.text((20,36),f'Native time {t/1000:.3f} s | 2 ms activity sampled every 4 ms | playback 20 fps',fill='black',font=font)
            if index==maximum_frame:canvas.save(poster)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close();returncode=proc.wait()
        if returncode:raise RuntimeError(f'ffmpeg encode failed: {returncode}')
      except BaseException:
        proc.kill();proc.wait();raise
      subprocess.run([str(FFMPEG),'-hide_banner','-loglevel','error','-y','-i',str(mp4),'-filter_complex','fps=20,scale=880:-1:flags=neighbor,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse=dither=none','-threads','2','-loop','0',str(gif)],stderr=log,check=True)
    count=0;duration=0
    with Image.open(gif) as img:
      for index in range(img.n_frames):img.seek(index);img.load();count+=1;duration+=img.info.get('duration',0)
    assert duration==len(frames)*50
    return dict(candidate=c['id'],seed=seed,window_ms=[lo,hi],selection='first fixed 6s after analysis burn-in, independent of events/labels/outcomes',
        native_bin_ms=dt,native_sample_step_ms=4,playback_fps=20,source_frames=len(frames),gif_frames=count,gif_duration_ms=duration,
        native_vmax=vmax,native_statistic='number of E neurons active at least once in each 2ms x 1mm bin',
        contact_scale='one common full-window maximum across all contacts',rates='per-neuron Hz, display-only Gaussian5ms',
        poster_rule='largest total native activity among sampled frames in fixed interval',poster_native_time_ms=(maximum_frame+.5)*dt,
        mp4=str(mp4),mp4_sha256=rt.sha(mp4),gif=str(gif),gif_sha256=rt.sha(gif),poster=str(poster),arrays_sha256=r['arrays_sha256'])


def main(phase,candidate):
    old,plan,spec,cases=review.stage_cases(phase);cases=[c for c in cases if c['id']==candidate or c['base_id']==candidate]
    if not cases:raise ValueError('candidate missing from phase')
    out=review.night.OUT/('continuous_native_'+phase);F=out/'figures';F.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':8,'pdf.fonttype':42})
    records=[]
    for c in cases:
      seeds=sorted({int(s) for cid,t,s in spec['units'] if cid==c['base_id'] and int(t)==c['topology']}) or old['seeds']
      for seed in seeds:
        path=an.run.result_path(c['output_stage'],c['base_id'],c['topology'],seed);unit=an.load_unit(path,old['analysis']['burnin_ms'])
        if unit is None:continue
        r,a,_=unit;lo=old['analysis']['burnin_ms'];hi=min(lo+6000,r['actual_duration_ms'])
        records.append(render(c,seed,r,a,lo,hi,F))
    rt.write(out/f'{candidate}_manifest.json',dict(status='COMPLETE_READ_ONLY',records=records,producer=__file__,producer_sha256=rt.sha(__file__),no_event_filter=True))
    descriptions=[]
    for file in sorted(F.glob('*.gif')):
        descriptions.append(f'### {file.name}\n\n固定显示启动排除期之后连续6秒，与事件标签和分数无关，不拼接事件窗口、不压缩事件间隔。左为全部原生2ms活动每4ms采样，右为完整电极包络及两核/核外每细胞发放率；实际时钟与20fps播放速度分开，MP4同源、PNG为此窗全场活动最多的一帧。**关注点**：未经事件筛选的核内外活动、连续节律和潜在的多个局部burst；单凭随机输入只在核内不能断言所有活动均由核因果驱动。')
    (F/'README.md').write_text('\n\n'.join(descriptions)+'\n');print(json.dumps(dict(output=str(out),clips=len(records)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',default='wave1');parser.add_argument('--candidate',required=True);args=parser.parse_args();main(args.phase,args.candidate)
