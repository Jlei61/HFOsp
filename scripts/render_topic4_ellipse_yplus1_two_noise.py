"""Complete paired-noise review of the predeclared upward geometry probe."""
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts import analyze_topic4_shape_output_response as an
from scripts.render_topic4_shape_output_gifs import render
from src.topic4_pdf_font_guard import install
P=Path('/data/hfosp/topic4_sef_hfo/core_recruitment_tradeoff_followup_20260912')
CID='up3__ellipse4__EI_same_core_scale_0.75__y_plus10'
OUT=P.parent/'overnight_exploration_20260913/ellipse_yplus1_two_noise'

def main():
    rt=an.rt;paths={s:P/'response/units'/CID/f'2511_{s}/workers/trajectory.json' for s in [847101,847102]}
    assert all(p.exists() for p in paths.values()),'Both full physical runs required'
    install();an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3})
    F=OUT/'figures';F.mkdir(parents=True,exist_ok=True)
    c=rt.read(P/'candidates'/f'{CID}.json');c.update(topology=2511,display_name='左椭圆4:1；两核内E→I×0.75；左核再上移1 mm')
    units={s:an.an.load_unit(p,1500.) for s,p in paths.items()}
    assert all(r['actual_duration_ms']==60000 and r['physical_status']=='COMPLETE_NO_RUNAWAY' for r,a,ids in units.values())
    physics=rt.read(paths[847101].parents[1]/'applied_physics.json');c['_applied_threshold']=physics['threshold']
    fr=an.figreview;patient=fr.patient_payloads();entries=[fr.spectral_comparison(c,units,[847101,847102],F,patient,'primary')]
    for s,(r,a,ids) in units.items():
        m=fr.four_panel(c,s,r,a,ids,F,physics,'primary')
        fields=[fr.native_timing(r,a,v['event'])[0] for v in fr.representatives(a,ids).values()]
        v=np.concatenate([x[np.isfinite(x)] for x in fields]) if fields else np.array([])
        m['native_shared_color_limits_ms']=[float(v.min()),max(float(v.max()),float(v.min())+1)] if len(v) else None;entries.append(m)
    # The first-noise movies are already preserved and checked. Render the
    # second noise under exactly the same time-based example selection rule.
    r,a,ids=units[847102];entries.append(render(c,847102,r,a,ids,physics,F,patient))
    first=P.parent/'overnight_exploration_20260913/ellipse_yplus1_first_noise'
    rt.write(OUT/'manifest.json',dict(status='COMPLETE_TWO_NOISE_MEDIA_PENDING_REVIEW',sources=[dict(noise=s,path=str(paths[s]),arrays_sha256=units[s][0]['arrays_sha256']) for s in paths],first_noise_movie_manifest=str(first/'manifest.json'),first_noise_movie_manifest_sha256=rt.sha(first/'manifest.json'),entries=entries,producer=str(Path(__file__)),producer_sha256=rt.sha(Path(__file__)),updated_unix=time.time()))
    (F/'README.md').write_text('# 左椭圆再上移1mm：两条完整噪声\n\n首条噪声动画保留在[首条审阅](../../ellipse_yplus1_first_noise/figures/README.md)，此处新增第二噪声，并将两条完整结果同时与患者比较。两次噪声来自同一基础网络，不能当作两张独立网络。\n\n'+''.join(f'### {p.name}\n患者为固定Fig2C真实STFT，模型为全体E发放密度包络与原生场；固定15行分杆显示。静态选例取各运行自身模式均值附近，动画取该噪声各模式前三个事件及固定连续片段。\n**关注点**：上部SCL参与、TA杆间时差与TB回返是否在两噪声中保留；分数改善不能替代完整传播验收。\n\n' for p in sorted(F.iterdir()) if p.suffix in ['.png','.gif']))
    print(json.dumps(dict(output=str(OUT),primary_by_noise={s:len(x[2]) for s,x in units.items()})))

if __name__=='__main__':main()
