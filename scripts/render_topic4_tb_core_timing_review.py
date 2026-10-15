"""Show fixed, subgroup-mean TB examples with all-native activity and Fig2C."""
from pathlib import Path
import sys,json,io
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scripts import analyze_topic4_shape_output_response as an
from scripts.render_topic4_shape_output_gifs import field_tile,save_background
BASE=Path('/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913')
SOURCE=BASE/'core_phase_tb_fold/selected_tb_core_trace_manifest.json'
OUT=BASE/'tb_core_timing_native_review'

def main():
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9})
    m=an.rt.read(SOURCE);path=Path(m['source']['trajectory']);r,a,_=an.an.load_unit(path,1500.)
    ap=an.rt.read(path.parents[1]/'applied_physics.json');c=ap['candidate'];c['_applied_threshold']=ap['threshold']
    patient=an.figreview.patient_payloads();canonical,meta,pat,names,porder=patient;e=pat['TB']
    order=an.figreview.display.contact_indices(a['contact_names']);tt=a['trace_time_ms'];dt=float(a['contact_envelope_dt_ms'])
    vmax=max(float(np.quantile(a['sheet_activity_counts'][750:],.999)),1)
    rate_max=0
    for event in m['events']:
        lo,hi=event['window_ms'];sel=(tt>=lo)&(tt<hi)
        for group in ['coreAE','coreBE']:
            rate=a['trace_'+group+'_spikes'][sel]*1000/(len(a['group_'+group])*float(np.median(np.diff(tt))))
            rate_max=max(rate_max,float(rate.max()))
    xlo=min(-130.,e['tile_lo_ms'],*[v['window_ms'][0]-v['zero_absolute_ms'] for v in m['events']])
    xhi=max(160.,e['tile_hi_ms'],*[v['window_ms'][1]-v['zero_absolute_ms'] for v in m['events']])
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',16)
    records=[]
    for event in m['events']:
        i=event['event'];lo,hi=event['window_ms'];zero=event['zero_absolute_ms'];mu=a['centroid_ms'][i];part=np.isfinite(mu[order])
        fig,axes=an.plt.subplots(2,2,figsize=(13.5,11.4));fig.subplots_adjust(left=.085,right=.97,bottom=.17,top=.85,wspace=.29,hspace=.38)
        an.figreview.display.patient_readout(axes[0,0],e,names,canonical,(xlo,xhi),an.figreview.an.MODE_COLOR['TB'])
        axes[0,0].set(title='患者 TB：固定Fig2C真实STFT',xlabel='相对患者最早参与质心 (ms)');axes[0,0].set_box_aspect(None)
        axes[0,0].set_ylim(14.5,-.5)
        field=axes[0,1];im=field.imshow(np.zeros((20,20)),origin='lower',extent=[0,20,0,20],cmap='inferno',vmin=0,vmax=vmax,interpolation='nearest')
        field.set(xlabel='x (mm)',ylabel='y (mm)',title='模型全部E活动：原生2ms帧')
        cb=fig.colorbar(im,ax=field,fraction=.035,pad=.035);cb.set_label('活跃E细胞数 / 2ms / 1mm²',fontsize=8)
        sel=(tt>=lo)&(tt<hi);ax=axes[1,0]
        for group,label,col in [('coreAE','左核','#bd7939'),('coreBE','右核','#327fb0')]:
            ax.plot(tt[sel]-zero,a['trace_'+group+'_spikes'][sel]*1000/(len(a['group_'+group])*float(np.median(np.diff(tt)))),c=col,lw=1,label=label)
        ax.set(xlim=(xlo,xhi),ylim=(0,rate_max*1.05),xlabel='相对模型最早参与质心 (ms)',ylabel='核内E群体发放率 (Hz/细胞)',title='原始1ms记录；两例共用幅度轴');ax.legend(frameon=False)
        env=a['contact_envelope'][round(lo/dt):round(hi/dt),order].T.copy();env/=np.maximum(env.max(1,keepdims=True),1e-20);env[~part]=np.nan
        cmap=an.plt.get_cmap('magma').copy();cmap.set_bad('#777');ax=axes[1,1]
        ax.imshow(env,aspect='auto',extent=[lo-zero,hi-zero,14.5,-.5],cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        an.figreview.display.centroid_lines(ax,mu[order]-zero,color='#397fb0');an.figreview.display.contact_axis(ax)
        ax.set(xlim=(xlo,xhi),ylim=(14.5,-.5),xlabel='相对模型最早参与质心 (ms)',title='模型 TB：发放密度包络');ax.set_facecolor('#b6b6b6')
        for ax in [axes[0,0],axes[1,0],axes[1,1]]:ax.axvline(0,c='#999',ls='--',lw=.6)
        fig.suptitle('同一条记录中的两种TB过程｜'+event['group']+f'｜事件{i}',fontsize=15,y=.985)
        fig.text(.085,.905,'左移圆核＋向外EE增强25%；基础网络2511／噪声847101。选例为各时序亚组自身均值附近。',fontsize=10)
        fig.text(.085,.055,'上左患者仅作为固定形态参照；没有与这次模型事件逐事件配对，也没有时间拉伸。SCL、ICL始终分杆排列，未参与保留灰色。\n移动光标只属于模型。上右白线为两个core，青色触点为同一SEEG布局；完整E活动未按lineage、核心或传播好坏筛选。\n累计10%时间只是既有250ms窗口内的后验分层，不代表起燃位置或核间因果关系。需看原生场区分单前沿与双前沿叠加。',fontsize=9)
        fig.canvas.draw();field_box=field.get_position().bounds;cursor_boxes=[axes[1,j].get_position().bounds for j in [0,1]]
        background=save_background(fig);bw,bh=background.size
        x,y,w,h=field_box;fx,fy=round(x*bw),round((1-y-h)*bh);fw,fh=round(w*bw),round(h*bh)
        frames=[];times=np.arange(lo,hi,2.)
        for t in times:
            canvas=background.copy();tile=field_tile(a['sheet_activity_counts'][round(t/2)],c,a,ap,vmax,size=max(fw,fh));canvas.paste(tile.resize((fw,fh),Image.Resampling.NEAREST),(fx,fy));draw=ImageDraw.Draw(canvas)
            draw.text((round(.085*bw),round(.111*bh)),f'模型实际时间 {t/1000:.3f} s ｜ 相对模型最早质心 {t-zero:+.1f} ms',font=font,fill='black')
            for x,y,w,h in cursor_boxes:
                px=(x+(t-zero-xlo)/(xhi-xlo)*w)*bw;draw.line((px,(1-y-h)*bh,px,(1-y)*bh),fill='#00bfc6',width=2)
            frames.append(canvas)
        file=OUT/'figures'/f'tb_event_{i}_field_core_readout.gif'
        frames[0].save(file,save_all=True,append_images=frames[1:],duration=60,loop=0)
        native_peak=int(np.argmax(a['sheet_activity_counts'][round(lo/2):round(hi/2)].sum((1,2))))
        frames[native_peak].save(file.with_name(file.stem+'_native_peak.png'))
        # Equally spaced snapshots show the whole frozen window, including quiet
        # periods; they are not selected for agreement with the patient.
        snap_ids=np.linspace(0,len(frames)-1,6,dtype=int);sheet=Image.new('RGB',(3*400,2*440),'white');draw=ImageDraw.Draw(sheet)
        for k,j in enumerate(snap_ids):
            t=times[j];tile=field_tile(a['sheet_activity_counts'][round(t/2)],c,a,ap,vmax,size=400);ox,oy=(k%3)*400,(k//3)*440;sheet.paste(tile,(ox,oy+40));draw.text((ox+5,oy+5),f'{t-zero:+.1f} ms',font=font,fill='black')
        sheet.save(file.with_name(file.stem+'_six_times.png'))
        zoom_offsets=np.arange(-25.,86.,10.)
        zoom=Image.new('RGB',(1200,990),'white');draw=ImageDraw.Draw(zoom);zoom_times=[]
        for k,offset in enumerate(zoom_offsets):
            t=2*round((zero+offset)/2);assert lo<=t<hi;zoom_times.append(float(t))
            tile=field_tile(a['sheet_activity_counts'][round(t/2)],c,a,ap,vmax,size=300);ox,oy=(k%4)*300,(k//4)*330
            zoom.paste(tile,(ox,oy+30));draw.text((ox+5,oy+3),f'{t-zero:+.1f} ms',font=font,fill='black')
        zoom.save(file.with_name(file.stem+'_10ms_native_sequence.png'))
        with Image.open(file) as im:
            n=im.n_frames
            for j in range(n):im.seek(j);im.load()
        assert n==len(times)
        records.append(dict(**event,file=str(file),nframes=n,frame_ms=2,playback_ms_per_frame=60,native_color_limits=[0,vmax],patient_event=e['event'],native_peak_preview_frame=native_peak,six_snapshot_frames=snap_ids.tolist(),zoom_native_times_ms=zoom_times,zoom_selection='same fixed -25..85ms relative contact-centroid range every 10ms for both events; nearest native 2ms frame'))
        del frames
    an.rt.write(OUT/'manifest.json',dict(source_manifest=str(SOURCE),source_manifest_sha256=an.rt.sha(SOURCE),source=m['source'],events=records,selection='existing subgroup-own-mean examples, not patient-nearest; does not replace chronological all-event GIFs',display_order=an.figreview.display.CONTACT_ORDER,model_cursor_only=True,relative_time_limits_ms=[xlo,xhi],field_layer='all E, 2 ms, no lineage restriction',human_review='PENDING'))
    (OUT/'figures/README.md').write_text(''.join(f'### {p.name}\n两例取自同一条60秒记录的TB标签内部，按两核累计10%活动先后分组，各选自身特征均值附近事件。患者固定为Fig2C真实TB频谱；模型原生场、核内发放率和接触包络同步，全部触点保持固定分杆顺序。\n**关注点**：两核前沿是否叠加；亚组与接触顺序的关联不等于核间因果驱动或患者双模式恢复。\n\n' for p in sorted((OUT/'figures').iterdir())))
    print('Two diagnostic native 2ms GIFs and six-time snapshots generated and decoded.')

if __name__=='__main__':main()
