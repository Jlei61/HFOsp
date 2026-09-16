"""Review the first completed remaining geometry unit, without score selection."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import analyze_topic4_shape_output_response as an
from scripts.render_topic4_shape_output_gifs import render
from src.topic4_pdf_font_guard import install
install();an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3})
P=Path('/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912')
CID='up3__ellipse4__EI_same_core_scale_0.75__y_plus10'
OUT=P.parent/'overnight_exploration_20260913/ellipse_yplus1_first_noise'


def main():
    F=OUT/'figures';F.mkdir(parents=True,exist_ok=True);rt=an.rt
    p=P/'response/units'/CID/'2511_847101/workers/trajectory.json'
    c=rt.read(P/'candidates'/f'{CID}.json');c.update(topology=2511,display_name='左椭圆4:1；两核内E→I×0.75；左核再上移1 mm')
    r,a,ids=an.an.load_unit(p,1500.);assert r['actual_duration_ms']==60000 and r['physical_status']=='COMPLETE_NO_RUNAWAY'
    physics=rt.read(p.parents[1]/'applied_physics.json');c['_applied_threshold']=physics['threshold']
    fr=an.figreview;patient=fr.patient_payloads();units={847101:(r,a,ids)}
    entries=[fr.spectral_comparison(c,units,[847101,847102],F,patient,'primary')]
    m=fr.four_panel(c,847101,r,a,ids,F,physics,'primary')
    fields=[fr.native_timing(r,a,v['event'])[0] for v in fr.representatives(a,ids).values()]
    finite=np.concatenate([x[np.isfinite(x)] for x in fields]) if fields else np.array([])
    m['native_shared_color_limits_ms']=[float(finite.min()),max(float(finite.max()),float(finite.min())+1)] if len(finite) else None
    entries.extend([m,render(c,847101,r,a,ids,physics,F,patient)])
    rt.write(OUT/'manifest.json',dict(status='COMPLETE_FIRST_NOISE_MEDIA_PENDING_REVIEW',source=str(p),arrays_sha256=r['arrays_sha256'],producer=str(Path(__file__)),producer_sha256=rt.sha(Path(__file__)),
        available_noises=[847101],pending_noises=[847102],selection='first completed remaining geometry unit; own-mode representatives; first3 primary events per mode; no patient-nearest selection',entries=entries,updated_unix=time.time()))
    (F/'README.md').write_text('# 左椭圆再上移1mm：首条完整噪声\n\n'+''.join(f'### {p.name}\n患者为固定Fig2C真实STFT，模型为全部E发放密度及原生场。15个接触点按杆固定，第二噪声明确待完成；GIF展示每类前三个合格事件和固定连续片段。\n**关注点**：上部SCL是否过度参与，以及TB固定接触顺序是否恢复；一条噪声不支持重复性结论。\n\n' for p in sorted(F.iterdir()) if p.suffix in ['.png','.gif']))
    print(json.dumps(dict(output=str(OUT),primary=len(ids))))


if __name__=='__main__':main()
