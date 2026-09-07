#!/usr/bin/env python3
"""Finish rendering and descriptive analysis when the two fixed-base jobs finish."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/topic4_sef_hfo/fig5_two_gif_bases'
IDS=['support_rank__vth_low','old_joint__tau_d_GABA_ms_high']


def read(p): return json.loads(p.read_text())


def main():
    last=None
    while True:
        statuses={cid:read(DATA/cid/'status.json') for cid in IDS}
        failures={cid:read(DATA/cid/'failure.json') for cid in IDS if (DATA/cid/'failure.json').exists()}
        signature=[(cid,[(DATA/cid/f).stat().st_mtime if (DATA/cid/f).exists() else None
                        for f in ['trajectory.json','probe.json']]) for cid in IDS]
        if signature!=last and any((DATA/cid/'trajectory.json').exists() for cid in IDS):
            subprocess.run([sys.executable,str(ROOT/'scripts/paper_figures/plot_fig5_two_gif_bases.py')],cwd=ROOT,check=True)
            last=signature
        if all(statuses[cid]['stage'].startswith('COMPLETE') or cid in failures for cid in IDS):break
        time.sleep(30)
    results=[]
    for cid in IDS:
        row=dict(candidate_id=cid,status=statuses[cid],failure=failures.get(cid))
        if (DATA/cid/'trajectory.json').exists():row['trajectory']=read(DATA/cid/'trajectory.json')['trajectory']
        if (DATA/cid/'probe.json').exists():
            probe=read(DATA/cid/'probe.json');states=list(probe['states_ms'])
            first=[r['extra_spikes_200ms'] for r in probe['rows'] if r['state']==states[0]]
            later=[r['extra_spikes_200ms'] for r in probe['rows'] if r['state']==states[1]]
            row['probe_comparison']=dict(states_ms=probe['states_ms'],first=first,later=later,
                sites_with_increase=sum(b>a for a,b in zip(first,later)),n_sites=len(first),
                qualified_pretransition=probe['qualified_pretransition'])
        results.append(row)
    report=dict(status='COMPLETE_PENDING_VISUAL_REVIEW' if not failures else 'INCOMPLETE_WITH_FAILURES',
        results=results,author_accepted=False,
        boundary='Two fixed complete substrates, one graph/noise seed each. No substrate optimization, new patient fit validation, or bifurcation scan. All negative and absent-transition outcomes retained.')
    (DATA/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# 两组 GIF 基底的 Fig. 5 扰动比较','',
        '两组分别完整保留原 GIF 的双核位置、VTH 映射、连接和 GABA 时间常数；它们同时有多项差异，因此不是 GABA 的单因素对照。共同施加原有 Z/M 参考参数，未搜索底物。',
        '关闭 Z/M 的 100 ms 二维活动前缀与旧 GIF 数据逐格一致；多维参数审计完全匹配。每组为 topology/dynamics seed 2511 的探索实例。','']
    for row in results:
        lines.append('## '+row['candidate_id'])
        if row.get('failure'):lines.append('运行失败：'+str(row['failure']))
        if 'trajectory' in row:lines.append('轨迹：'+json.dumps(row['trajectory'],ensure_ascii=False))
        if 'probe_comparison' in row:
            p=row['probe_comparison'];lines.append(f"200 ms 下游响应在 {p['sites_with_increase']}/{p['n_sites']} 个配对刺激位置增加。原始有符号响应和完整时间曲线保留；位置不是独立模型重复。")
            if not p['qualified_pretransition']:lines.append('没有合格的转变前轨迹。此处仅比较早期与后期参考状态，不能解释为发作前易感性。')
        lines.append('')
    lines+=['图的 D 部分分别回答刺激位置效应、固定中心刺激后的响应位置和累积响应时间过程。地图不平滑，不混合不同刺激来源；颜色保留负值，两个基底共用色标。',
            'PNG/PDF 已由脚本输出（以 render_status.json 为准）；自动完成不代表 Agent 目视检查或作者验收。旧 Figure 5 的 E/F 分叉与参数扫描结果没有搬到这些新基底。']
    (DATA/'analysis.md').write_text('\n\n'.join(lines)+'\n')


if __name__=='__main__':main()
