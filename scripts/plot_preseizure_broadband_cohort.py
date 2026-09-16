#!/usr/bin/env python3
"""Describe broadband eligibility separately from missing data and labelability."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic5_preseizure_template_association_broadband/cohort_20260909'


def main():
    manifest=pd.read_csv(OUT/'cohort_manifest.csv',dtype={'subject':str})
    rows=[]
    for _,r in manifest[manifest.selected].iterrows():
        folder=OUT/'per_subject'/f'epilepsiae_{r.subject}'
        for p in sorted((folder/'per_seizure').glob('*.json')):
            q=json.loads(p.read_text());reasons=q.get('qualification_exclusions',[])
            if q['status']=='UNAVAILABLE':category='Data unavailable'
            elif any(x!='does_not_meet_matched_window_broadband_5of6' for x in reasons):category='Window / baseline ineligible'
            elif not q['passes_spectral_5of6']:category='Broadband criterion not met'
            elif q['qualified_source_label'] in ('TA','TB'):category=q['qualified_source_label']+' source'
            else:category='Broadband, source unclear'
            rows.append(dict(subject=r.subject,sz=q['sz'],primary_geometry=r.primary_geometry,category=category,
                spectral_5of6=q.get('passes_spectral_5of6'),exclusions=';'.join(reasons)))
    df=pd.DataFrame(rows);df.to_csv(OUT/'qualification_flow.csv',index=False)
    subjects=sorted(df.subject.unique(),key=int)
    categories=['TA source','TB source','Broadband, source unclear','Broadband criterion not met','Window / baseline ineligible','Data unavailable']
    colors=['#b2182b','#2166ac','#9970ab','#e6ab02','#bdbdbd','#525252']
    fig,ax=plt.subplots(figsize=(12,6.2),layout='constrained');left=np.zeros(len(subjects))
    for category,color in zip(categories,colors):
        counts=np.array([sum((df.subject==s)&(df.category==category)) for s in subjects])
        ax.barh(np.arange(len(subjects)),counts,left=left,color=color,label=category)
        for i,(n,l) in enumerate(zip(counts,left)):
            if n>=3:ax.text(l+n/2,i,str(n),ha='center',va='center',fontsize=9,color='white' if category in ['TA source','TB source','Data unavailable'] else 'black')
        left+=counts
    ax.set(yticks=np.arange(len(subjects)),yticklabels=['E'+s+(' (1D)' if s=='139' else '') for s in subjects],xlabel='Number of seizures',title='Broadband eligibility before signed source labelling')
    ax.invert_yaxis();ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='lower right',frameon=False,fontsize=9)
    fp=OUT/'figures';fp.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(fp/f'cohort_broadband_qualification.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    (fp/'README.md').write_text('### cohort_broadband_qualification.png\n按患者展示宽频筛选、数据不可用、时间窗不合格和最终明确source标签的数量，PDF为同图矢量版。类别互斥：数据不可用优先，其次时间窗/基线不合格，再判断频谱资格；因此灰色不能解释为缺乏宽频增强。E139仅作一维几何敏感性病例，TA/TB为患者内多数/少数模板。\n**关注点**：通过宽频判据仍不保证能区分source；E1084大量短发作不支持固定0–10秒发作内窗口，不能据此称其没有宽频增强。图待人工检查。\n',encoding='utf-8')
    stats_path=OUT/'all_patient_association_tests.csv'
    stats=pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    details=['# 本批次读出与限制','','资格流程见`qualification_flow.csv`及`figures/cohort_broadband_qualification.png`。分频增强、时间窗完整性和source可分性分别保留，不能把未获得标签统一解释为缺乏broadband。','']
    for s in subjects:
        d=df[df.subject==s]
        details.append(f'- E{s}：已处理{len(d)}次；'+ '，'.join(f'{c} {sum(d.category==c)}' for c in categories)+'。')
    if len(stats):
        primary=stats[stats.primary_geometry]
        details+=['',f'二维主队列共运行{len(primary)}项患者内候选检验；跨患者×特征BH q<0.05有{sum(primary.q_across_primary_patients_and_features<.05)}项。这是探索性筛选，非预测验证。',
            'E1084多数标注发作短于10秒，当前固定窗口不适用；这些排除不支持“缺乏宽频增强”的生物学判断。缩短窗口会同时改变能量标签和宽频判据，需作为单独敏感性分析。',
            '冻结模板事件映射遵循all-event producer原始列索引，保留空间视图缺失但timing标签存在的事件，不重新使用旧三触点方向资格过滤。']
    (OUT/'RESULT_NOTES.md').write_text('\n'.join(details)+'\n',encoding='utf-8')
    print(df.groupby(['subject','category']).size().to_string())


if __name__=='__main__':main()
