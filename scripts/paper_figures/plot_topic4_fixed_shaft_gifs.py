"""Re-render the three reviewed TB windows with the fixed patient/model Y axis."""
from pathlib import Path
import importlib.util,json,hashlib,sys
MAIN=Path('/home/honglab/leijiaxin/HFOsp')
sys.path.insert(0,str(MAIN))
from src import snn_contact_display as display
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageSequence
NIGHT=Path('/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911')
OUT=NIGHT/'tb_fixed_shaft_review';F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('original_zigzag_review',NIGHT/'tb_zigzag_review/analyze_and_render.py')
source=importlib.util.module_from_spec(spec);spec.loader.exec_module(source)
plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9})


def render(seed,event,unit,selection):
    r,a,ids=unit;assert event in ids
    order=display.contact_indices(a['contact_names']);xy=a['contact_xy_mm']
    times=a['centroid_ms'][event];zero=float(np.nanmin(times));part=np.isfinite(times)
    lo,hi=r['events'][event]['window_ms'];start,end=int(lo/2),int(hi/2)
    env=a['contact_envelope'][start:end];scale=np.maximum(env.max(0),1e-20)
    normalized=(env/scale)[:,order].T;normalized[~part[order]]=np.nan
    raw=a['sheet_activity_counts'];vmax=max(1,int(raw[start:end].max()))
    fig,axes=plt.subplots(1,3,figsize=(13.6,4.8),dpi=120,
        gridspec_kw={'width_ratios':[1,1,1.55]},layout='constrained')
    im=axes[0].imshow(raw[start],origin='lower',extent=[0,20,0,20],interpolation='nearest',cmap='inferno',vmin=0,vmax=vmax)
    source.geometry(axes[0],a,True);axes[0].set_title('全部原生 E 活动')
    fig.colorbar(im,ax=axes[0],location='bottom',pad=.12,shrink=.85,label='活动细胞 / 2 ms / 1 mm 网格')
    source.geometry(axes[1],a)
    dots=axes[1].scatter(*xy.T,c=np.zeros(15),s=62,cmap='magma',vmin=0,vmax=1,
        edgecolors=[display.SHAFT_COLORS[str(n)[:3]] for n in a['contact_names']],linewidths=1)
    axes[1].set_title('接触读出：真实二维位置')
    fig.colorbar(dots,ax=axes[1],location='bottom',pad=.12,shrink=.85,label='各触点自身峰值归一化包络')
    cmap=plt.get_cmap('magma').copy();cmap.set_bad('#777777')
    axes[2].imshow(normalized,extent=[lo-zero,hi-zero,14.5,-.5],aspect='auto',cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
    display.contact_axis(axes[2]);display.centroid_lines(axes[2],times[order]-zero)
    axes[2].set(xlim=(-115,180),xlabel='相对最早参与质心 (ms)',title='固定 Y 轴：SCL9–6，然后 ICL11–1')
    for j in np.flatnonzero(~part[order]):
        axes[2].text(30,j,'未参与 / 无有效质心',color='white',ha='center',va='center',fontsize=7)
    cursor=axes[2].axvline(lo-zero,color='white',lw=1)
    title=fig.suptitle('');frames=[]
    for index in range(start,end):
        t=2*index+1;im.set_data(raw[index]);values=a['contact_envelope'][index]/scale;values[~part]=np.nan;dots.set_array(values)
        cursor.set_xdata([t-zero,t-zero])
        title.set_text(f'TB 事件 {event}｜噪声 {seed}｜t={t/1000:.3f} s（相对质心 {t-zero:+.1f} ms）\n{selection}；2 ms 原生帧，20 fps 慢放；两杆之间不连接质心线')
        fig.canvas.draw();frame=Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy());frames.append(frame)
        if index==int((zero+50)//2):frame.save(F/f'{seed}_event{event}.png')
    path=F/f'{seed}_event{event}.gif';frames[0].save(path,save_all=True,append_images=frames[1:],duration=50,loop=0);plt.close(fig)
    count=0
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):frame.load();count+=1
    assert count==125
    return dict(seed=seed,event=event,selection=selection,window_ms=[lo,hi],zero_absolute_ms=zero,
        display_contact_order=list(display.CONTACT_ORDER),display_ylim=list(display.YLIM),display_xlim_ms=[-115,180],frames=count,
        original_arrays_sha256=r['arrays_sha256'],gif=str(path),gif_sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':
    records=[];data={s:source.load(s) for s in [847101,847102]}
    for seed,event,why in [(847101,72,'用户截图中的既有代表例'),(847101,123,'同噪声的非完整锯齿反例'),(847102,150,'另一噪声的既有代表例')]:
        records.append(render(seed,event,data[seed],why));print('rendered',seed,event,flush=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(status='COMPLETE_PENDING_USER_VISUAL_REVIEW',
        display_contract='E1146_fixed_shaft_rows_v1',producer=str(Path(__file__)),records=records,
        old_display_archive=str(NIGHT/'tb_zigzag_review'),new_simulations=0),ensure_ascii=False,indent=2))
    notes=[]
    for rec in records:
        for ext in ['gif','png']:
            name=f'{rec["seed"]}_event{rec["event"]}.{ext}'
            notes.append(f'### {name}\n\n{rec["selection"]}，与旧版完全相同的事件及250 ms原始窗口。左侧为未筛选原生E活动，中间为真实触点布局，右侧按SCL9–6、ICL11–1固定15行；橙杆ICL、青杆SCL，未参与行保留，杆间不连线。'+('动画保留125个2 ms帧，以20 fps慢放。' if ext=='gif' else '静帧取相对最早参与质心约+50 ms。')+'**关注点**：与患者、不同事件和后续版本使用相同Y轴；固定行序不改变任何传播结果。')
    (F/'README.md').write_text('\n\n'.join(notes)+'\n')
