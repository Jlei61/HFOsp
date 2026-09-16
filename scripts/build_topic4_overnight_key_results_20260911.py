"""A short, versioned reading package for the completed overnight round.

Consumes a human-readable scientific selection record; it never nominates a
candidate, dispatches a run or replaces the full collaborator report.
"""
from pathlib import Path
import datetime
import json
import os
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_topic4_model_collaborator_report_v7 import Report,BLUE,read,sha,Image
from build_topic4_model_collaborator_report_v8 import fixed_shaft_figure

NIGHT=Path('/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911')
OUT=Path('/home/honglab/leijiaxin/HFOsp/results/topic4_sef_hfo/overnight_key_results_20260911')


def build():
    selection=NIGHT/'key_review_selection.json';spec=read(selection)
    assert spec['status']=='COMPLETE_PENDING_USER_REVIEW'
    assert read(NIGHT/'final_B_analysis_complete.json')['status']=='COMPLETE_PENDING_SCIENTIFIC_REVIEW'
    now=datetime.datetime.now().astimezone();folder=OUT/'builds'/now.strftime('%Y%m%dT%H%M%S%f');folder.mkdir(parents=True)
    r=Report(folder,now.strftime('%Y-%m-%d %H:%M %z'),version=8)
    r.source(selection,'scientific_selection');r.source(Path(__file__),'short_report_producer')
    r.source(Path(__file__).with_name('build_topic4_model_collaborator_report_v7.py'),'shared_renderer')
    for i,page in enumerate(spec['pages']):
        r.start(page['title'],page.get('landscape',False))
        if 'figure' in page:r.figure(fixed_shaft_figure(Path(page['figure'])),'figure_'+str(i+1),page.get('maxheight',360 if page.get('landscape') else 590))
        if 'table' in page:r.table(page['table'],widths=page.get('widths'),size=page.get('table_size',10))
        for text in page.get('paragraphs',[]):r.para(text,page.get('font_size',10.6))
        for text in page.get('boundaries',[]):r.para(text,page.get('font_size',10.2),BLUE)
    assert r.y>35
    r.log[-1]['content_bottom_pt']=r.y
    assert all(not figure['vector'] for p in r.log for figure in p['figures'])
    r.c.setTitle('夜间SNN实验：传播恢复与参数响应的关键证据');r.c.save()
    pdf=folder/'overnight_key_results.pdf';shutil.copy2(folder/'layout.pdf',pdf)
    (folder/'source_manifest.json').write_text(json.dumps(dict(sources=r.sources,pages=r.log,text=r.text),ensure_ascii=False,indent=2))
    F=folder/'figures';F.mkdir()
    result=subprocess.run(['pdftoppm','-scale-to','1100','-png',str(pdf),str(F/'page')],capture_output=True,text=True,check=True)
    pngs=sorted(F.glob('page-*.png'));assert len(pngs)==len(spec['pages'])
    for p in pngs:
        with Image.open(p) as im:im.load()
    (F/'README.md').write_text('\n\n'.join(f'### {p.name}\n\n本次精简审阅包第{i+1}页，同源PDF及各原始图的来源保存在source_manifest。**关注点**：模型传播的支持与残差并列；图件可读不代表用户已经接受模型。' for i,p in enumerate(pngs))+'\n')
    delivery=dict(status='COMPLETE_PENDING_USER_REVIEW',generated_at=now.isoformat(),pages=len(pngs),snapshot=str(pdf),sha256=sha(pdf),
        report=str(OUT/pdf.name),renderer_warnings=result.stderr,model_scientific_acceptance=spec.get('model_scientific_acceptance',False),user_visual_acceptance=False)
    (folder/'delivery_checks.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2))
    temp=OUT/'report.tmp.pdf';shutil.copy2(pdf,temp);os.replace(temp,OUT/pdf.name)
    temp=OUT/'current.tmp.json';temp.write_text(json.dumps(delivery,ensure_ascii=False,indent=2));os.replace(temp,OUT/'current.json')
    print(json.dumps(delivery,ensure_ascii=False),flush=True)


if __name__=='__main__':build()
