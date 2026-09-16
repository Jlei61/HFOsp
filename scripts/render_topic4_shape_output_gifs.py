"""Native field and fixed-shaft readout animations from complete, immutable runs."""
from pathlib import Path
import io,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.paper_figures import plot_topic4_recovery_review as figreview


def core_pixels(c,physics,scale,offset):
    th=physics['threshold'];theta=np.deg2rad(th['ellipse_A_angle_deg']);u=np.array([np.cos(theta),np.sin(theta)]);v=np.array([-u[1],u[0]])
    angles=np.linspace(0,2*np.pi,181);a,b=th['ellipse_A_semiaxes_mm']
    curves=[np.asarray(c['centers_mm'][0])+a*np.cos(angles)[:,None]*u+b*np.sin(angles)[:,None]*v,
            np.asarray(c['centers_mm'][1])+c['radii_mm'][1]*np.stack([np.cos(angles),np.sin(angles)],1)]
    return [[(float(offset[0]+scale*x),float(offset[1]+scale*(20-y))) for x,y in curve] for curve in curves]


def field_tile(frame,c,a,physics,vmax,size=400):
    cmap=plt.get_cmap('inferno');rgb=(255*cmap(np.clip(frame[::-1]/max(vmax,1),0,1))[...,:3]).astype(np.uint8)
    im=Image.fromarray(rgb).resize((size,size),Image.Resampling.NEAREST);draw=ImageDraw.Draw(im)
    for line in core_pixels(c,physics,size/20,(0,0)):draw.line(line,fill='white',width=2)
    for x,y in a['contact_xy_mm']:
        x,y=x*size/20,(20-y)*size/20;draw.ellipse((x-3,y-3,x+3,y+3),outline='cyan',width=1)
    return im


def save_background(fig):
    buf=io.BytesIO();fig.savefig(buf,format='png',dpi=100);buf.seek(0);im=Image.open(buf).convert('RGB').copy();plt.close(fig);return im


def rotation_clip(c,seed,r,a,physics,out,rotation):
    tracks=[t for t in rotation['tracks'] if t['half_turn_candidate']]
    if not tracks:return dict(status='NO_HALF_TURN_CANDIDATE')
    tr=max(tracks,key=lambda t:t['best_fixed_turns']);lo=max(1500.,tr['start_ms']-150);hi=min(r['actual_duration_ms'],tr['end_ms']+150)
    center=tr['center_mm'];radius=tr['best_fixed_ring']['radius_mm'];vmax=max(float(np.quantile(a['sheet_activity_counts'][round(lo/2):round(hi/2)],.999)),1)
    frames=[]
    for t in np.arange(lo,hi,2):
        canvas=Image.new('RGB',(650,660),'white');tile=field_tile(a['sheet_activity_counts'][int(t/2)],c,a,physics,vmax,size=550);canvas.paste(tile,(50,85));draw=ImageDraw.Draw(canvas)
        draw.text((10,10),f'{c["id"]} | noise {seed} | {t/1000:.3f} s',fill='black');draw.text((10,32),f'Rotation SCREEN candidate; {tr["best_fixed_turns"]:.2f} turn; track {tr["duration_ms"]:.0f} ms',fill='black')
        draw.text((10,54),'Native 2 ms frames; cyan fixed ring; not a confirmed spiral rotor',fill='black')
        x,y=50+center[0]*27.5,85+(20-center[1])*27.5;rad=radius*27.5;draw.ellipse((x-rad,y-rad,x+rad,y+rad),outline='cyan',width=1)
        if tr['start_ms']<=t<=tr['end_ms']:
            point=min(tr['points'],key=lambda p:abs(1500+2*p[0]+1-t));px,py=50+point[1]*27.5,85+(20-point[2])*27.5;draw.line((px-6,py,px+6,py),fill='cyan',width=2);draw.line((px,py-6,px,py+6),fill='cyan',width=2)
        frames.append(canvas)
    name=f'{c["id"]}_{seed}_rotation_candidate.gif';frames[0].save(out/name,save_all=True,append_images=frames[1:],duration=45,loop=0)
    with Image.open(out/name) as im:
        for j in range(im.n_frames):im.seek(j);im.load()
    return dict(status='COMPLETE',file=name,selection='largest native fixed-ring turn among predeclared half-turn candidates; illustration, not prevalence estimate',track=tr,window_ms=[lo,hi],frame_ms=2)


def render(c,seed,r,a,ids,physics,out,patient):
    canonical,meta,pat,names,porder=patient;order=figreview.display.contact_indices(a['contact_names']);chosen=[]
    for m in [1,0]:chosen.extend(ids[a['event_mode'][ids]==m][:3].tolist())
    chosen.sort(key=lambda i:r['events'][i]['window_ms'][0]);manifest=[]
    if not chosen:return dict(status='NO_ELIGIBLE_EVENTS',seed=seed)
    vmax=max(float(np.quantile(a['sheet_activity_counts'][750:],.999)),1)
    images=[]
    for i in chosen:
        lab='TA' if a['event_mode'][i]==1 else 'TB';lo,hi=r['events'][i]['window_ms'];times=a['centroid_ms'][i];part=np.isfinite(times);zero=float(np.nanmin(times));dt=float(a['contact_envelope_dt_ms'])
        mass=a['contact_envelope'][round(lo/dt):round(hi/dt),order].T;mass=mass/np.maximum(mass.max(1,keepdims=True),1e-20)
        e=pat[lab];xlo=min(e['tile_lo_ms'],lo-zero);xhi=max(e['tile_hi_ms'],hi-zero)
        fig,axes=plt.subplots(1,2,figsize=(10,5.4));fig.subplots_adjust(left=.09,right=.99,bottom=.13,top=.84,wspace=.26)
        ax=axes[0];cmap=plt.get_cmap('magma').copy();cmap.set_bad('#777777');mass[~part[order]]=np.nan
        ax.imshow(mass,aspect='auto',extent=[lo-zero,hi-zero,14.5,-.5],cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
        figreview.display.centroid_lines(ax,times[order]-zero,color=figreview.an.MODE_COLOR[lab]);figreview.display.contact_axis(ax);ax.set(xlim=(xlo,xhi),xlabel='相对最早参与质心 (ms)',title='模型 '+lab+'：发放密度包络');ax.tick_params(labelsize=8);ax.set_facecolor('#b6b6b6')
        figreview.display.patient_readout(axes[1],e,names,canonical,(xlo,xhi),figreview.an.MODE_COLOR[lab]);axes[1].set(title='患者 '+lab+'：Fig2C真实STFT',xlabel='相对最早参与质心 (ms)');axes[1].set_box_aspect(None);axes[1].tick_params(labelsize=8)
        for ax in axes:ax.set_ylim(14.5,-.5)
        fig.canvas.draw();boxes=[ax.get_position().bounds for ax in axes];background=save_background(fig)
        for t in np.arange(lo,hi,4.):
            canvas=Image.new('RGB',(1420,580),'white');canvas.paste(background,(420,30));canvas.paste(field_tile(a['sheet_activity_counts'][round(t/2)],c,a,physics,vmax),(10,120));draw=ImageDraw.Draw(canvas)
            draw.text((12,15),f'{c["id"]} | topology {c["topology"]} | noise {seed}',fill='black')
            draw.text((12,42),f'{lab} event {i} | t={t:.0f} ms | event offset={t-lo:.0f} ms',fill='black')
            draw.text((12,72),'All native E activity; native 2 ms bins, display step 4 ms',fill='black')
            draw.text((12,544),f'Color: 0 .. {vmax:.0f} active cells / 2 ms / 1 mm2 (fixed within replay)',fill='black')
            for x,y,w,h in boxes:
                px=420+(x+(t-zero-xlo)/(xhi-xlo)*w)*background.width
                top=30+(1-y-h)*background.height;bottom=30+(1-y)*background.height
                draw.line((px,top,px,bottom),fill='#00ffff',width=2)
            images.append(canvas)
        manifest.append(dict(event=int(i),mode=lab,window_ms=[lo,hi],patient_event=e['event'],selection='first three primary events per mode, chronological; no patient-distance selection'))
    stem=f'{c["id"]}_{seed}_multievent_field_readout'
    images[0].save(out/(stem+'.gif'),save_all=True,append_images=images[1:],duration=55,loop=0);images[0].save(out/(stem+'_preview.png'))
    with Image.open(out/(stem+'.gif')) as im:
        nframes=im.n_frames
        for j in range(nframes):im.seek(j);im.load()
    # Fixed continuous time, independent of any event label or score.
    lo,hi=1500.,min(7500.,r['actual_duration_ms']);env=a['contact_envelope'][round(lo/dt):round(hi/dt),order].T
    fig,ax=plt.subplots(figsize=(10,5.4));fig.subplots_adjust(left=.1,right=.99,bottom=.13,top=.84)
    ax.imshow(env/max(np.quantile(env,.99),1e-20),aspect='auto',extent=[lo/1000,hi/1000,14.5,-.5],vmin=0,vmax=1,cmap='magma');figreview.display.contact_axis(ax);ax.set(xlabel='实际时间 (s)',title='固定1.5–7.5秒，全部接触点连续读出')
    box=ax.get_position().bounds;bg=save_background(fig);continuous=[]
    for t in np.arange(lo,hi,40):
        canvas=Image.new('RGB',(1420,580),'white');canvas.paste(bg,(420,30));canvas.paste(field_tile(a['sheet_activity_counts'][round(t/2)],c,a,physics,vmax),(10,120));draw=ImageDraw.Draw(canvas)
        draw.text((12,15),f'{c["id"]} | topology {c["topology"]} | noise {seed} | {t/1000:.2f} s',fill='black');draw.text((12,72),'Continuous diagnostic: native 2 ms activity sampled every 40 ms',fill='black')
        x,y,w,h=box;px=420+(x+(t-lo)/(hi-lo)*w)*bg.width;draw.line((px,30+(1-y-h)*bg.height,px,30+(1-y)*bg.height),fill='cyan',width=2);continuous.append(canvas)
    continuous[0].save(out/(f'{c["id"]}_{seed}_continuous_field_readout.gif'),save_all=True,append_images=continuous[1:],duration=40,loop=0)
    return dict(status='COMPLETE',multievent_file=stem+'.gif',nframes=nframes,events=manifest,field_color_limits=[0,vmax],field_layer='all native activity, no lineage filter',continuous_interval_ms=[lo,hi],
        display_order=list(figreview.display.CONTACT_ORDER),display_ylim=[14.5,-.5],continuous_sampling_ms=40,multievent_sampling_ms=4,native_bin_ms=2,
        patient_signal='real STFT',model_signal='spike-density envelope',interpretation='Patient/model cursors share relative milliseconds, not a claim of matched individual events.')
