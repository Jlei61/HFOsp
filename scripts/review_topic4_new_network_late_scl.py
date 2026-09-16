"""Illustrate the observed late-SCL tail, separately from normal model examples."""
from pathlib import Path
import sys,json,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from PIL import Image,ImageDraw
from scripts import analyze_topic4_shape_output_response as an
from scripts.render_topic4_shape_output_gifs import render
from src.topic4_pdf_font_guard import install


def main():
    base=Path('/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913')
    out=base.parent/'overnight_exploration_20260913/late_scl_TA_native_2711'
    out.mkdir(exist_ok=True)
    source=base/'response/units/bridge_circle_out125/2711_847401/workers/trajectory.json'
    key=hashlib.sha256(str(source).encode()).hexdigest()[:20]
    analysis=base/'analysis/units'/key/'result.json';summary=an.rt.read(analysis)
    r,a,ids=an.an.load_unit(source,1500.);assert r['arrays_sha256']==summary['source_sha256']
    ta=[e for e in summary['events'] if e['primary'] and e['mode']=='TA']
    tail=[e for e in ta if e['centroid_SCL_minus_ICL_ms'] is not None and e['centroid_SCL_minus_ICL_ms']>0]
    median=np.median([e['centroid_SCL_minus_ICL_ms'] for e in tail])
    e=min(tail,key=lambda q:(abs(q['centroid_SCL_minus_ICL_ms']-median),q['event']))
    c=an.rt.read(base/'candidates/bridge_circle_out125.json');c['topology']=2711
    physics=an.rt.read(source.parents[1]/'applied_physics.json')
    install();an.plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':9,'pdf.fonttype':3})
    record=render(c,847401,r,a,np.array([e['event']],int),physics,out,an.figreview.patient_payloads())
    record['events'][0]['selection']='Diagnostic only: primary TA with SCL-ICL rod median difference >0; nearest own subgroup median, event index breaks ties. Not patient-distance selected.'
    gif=out/record['multievent_file']
    with Image.open(gif) as im:
        chosen={round(i*(im.n_frames-1)/8) for i in range(9)};tiles=[]
        for k in range(im.n_frames):
            im.seek(k);im.load()
            if k in chosen:
                tile=im.convert('RGB');tile.thumbnail((710,290));bg=Image.new('RGB',(710,315),'white');bg.paste(tile,(0,25));ImageDraw.Draw(bg).text((8,4),f'Frame {k}',fill='black');tiles.append(bg)
        n=im.n_frames
    montage=Image.new('RGB',(2130,945),'white')
    for i,t in enumerate(tiles):montage.paste(t,((i%3)*710,(i//3)*315))
    montage.save(out/'diagnostic_nine_frames.png')
    an.rt.write(out/'manifest.json',dict(created_unix=time.time(),source=str(source),analysis=str(analysis),arrays_sha256=r['arrays_sha256'],
        TA_n=len(ta),late_SCL_TA_n=len(tail),all_late_TA_events=tail,selected_event=e,selection=record['events'][0]['selection'],
        gif=record,decoded_frames=n,physical_runs_added=0,objective_changes=0,producer=str(Path(__file__))))
    (out/'README.md').write_text(''.join(f'### {p.name}\n本图是新网络2711、原位置、噪声847401中TA晚SCL尾部的解释性诊断，不能代替正常选例。8/78个TA满足SCL杆间中位差大于零，取最接近这个亚组自身中位数的一个事件；没有根据患者相似度选择。\n**关注点**：原生前沿与较晚SCL读出的关系；核心窗口t10不是因果起点，全部事件仍保留于分布图。\n\n' for p in out.iterdir() if p.suffix in ['.png','.gif']))
    print(dict(output=str(out),event=e['event'],TA=len(ta),late_TA=len(tail)),flush=True)


if __name__=='__main__':main()
