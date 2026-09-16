#!/usr/bin/env python3
"""Bounded figure/report delivery for the declared fixed-model mechanism batches."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import argparse,csv,json,time,sys,subprocess
from pathlib import Path
import analyze_topic4_fixed_zm_termination as analysis

ROOT=analysis.OUT
FOLDERS=[ROOT,ROOT/'matched_spatial_round2',ROOT/'hyperpolar_spatial_round3',ROOT/'source_sahp_round4']
if (ROOT/'sahp_bracket_round5/protocol.json').exists():FOLDERS.append(ROOT/'sahp_bracket_round5')
for sub in ['positive_candidate_confirmation','low_fraction_round6','autonomous_recurrence_continuation']:
    if (ROOT/sub/'protocol.json').exists():FOLDERS.append(ROOT/sub)
PRODUCER=Path(analysis.__file__).resolve()

def update(folder,render=False):
    analysis.OUT=folder;analysis.run.OUT=folder
    comparison=analysis.collect();rows=comparison['rows']
    columns=['name','complete','observed_s','entry_s','return_start_s','return_confirm_s','second_entry_s','late_E_Hz','late_core_A_Hz','late_core_B_Hz','late_quiet_E','Z_end','native_M_end_mV','finite_events_before_entry','category']
    table=[]
    for r in rows:
        table.append(dict(name=r['name'],complete=r['complete'],observed_s=r['observed_s'],
            entry_s=r['entries'][0]['onset_s'] if r['entries'] else None,
            return_start_s=r['recoveries'][0]['start_s'] if r['recoveries'] else None,
            return_confirm_s=r['recoveries'][0]['confirmation_s'] if r['recoveries'] else None,
            second_entry_s=r['entries'][1]['onset_s'] if len(r['entries'])>1 else None,
            late_E_Hz=r['late_mean_Hz'][0],late_core_A_Hz=r['late_mean_Hz'][1],late_core_B_Hz=r['late_mean_Hz'][2],
            late_quiet_E=r['late_quiet_fraction'][0],Z_end=r['mean_Z_end'],native_M_end_mV=r['M_current_end'],
            finite_events_before_entry=r['finite_events_before_first_entry'],category=r['category_label']))
    with (folder/'measured_outcomes.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader();writer.writerows(table)
    if render:
        for r in rows:
            subprocess.run([sys.executable,str(PRODUCER),'plot','--root',str(folder),'--name',r['name']],check=True)
        analysis.write(folder/'delivery_complete.json',dict(time=time.time(),n_figures=len(rows),entry_return_gate_pass_count=sum(bool(r['recoveries']) for r in rows),full_Fig5_scientific_acceptance='NOT_ESTABLISHED',human_review='PENDING',full_batch=comparison['complete']))
    return comparison

def main(watch=False):
    deadline=json.load(open(ROOT/'protocol.json'))['deadline_epoch'];rendered=set()
    for folder in FOLDERS:
        if (folder/'delivery_complete.json').exists():rendered.add(folder)
    while True:
        status=[];complete=True
        for folder in FOLDERS:
            comp=update(folder)
            if comp['complete'] and folder not in rendered:
                update(folder,render=True);rendered.add(folder)
            complete=complete and comp['complete']
            status.append(dict(folder=str(folder),expected=comp['expected'],finished=sum(r['complete'] for r in comp['rows']),
                measured=[dict(name=r['name'],t=r['observed_s'],category=r['category_label'],entries=r['entries'],recoveries=r['recoveries']) for r in comp['rows']],
                figures_delivered=folder in rendered))
        analysis.write(ROOT/'delivery_status.json',dict(time=time.time(),complete=complete,batches=status))
        lines=['# 固定双核 Z/M：自主终止机制 Fig.5 候选\n',
            '本目录所有图来自当前40k手放双核 SNN。原文LAS复现不列作当前模型的成功结果。状态由实际轨迹确定；每张图仍待人工验收。\n',
            'A为连续raster及core E放大，B为原Z/M（慢钾分支另列真实gK），C为同一时序的原生2D场，D为实测状态投影，E为本批离散条件，F在基线足够时使用实际电极电流与固定Fig3C患者。D不是nullcline或分岔证明。\n']
        for item in status:
            folder=Path(item['folder']);lines.append(f"\n## {folder.name} ({item['finished']}/{item['expected']})\n")
            for row in item['measured']:
                path=folder/'figures'/f"fig5_{row['name']}.png"
                lines.append(f"- {row['name']}: {row['category']}; 已观察{row['t']:g}s"+(f" — [Fig5]({path})" if path.exists() else '')+'\n')
        (ROOT/'figure_index.md').write_text('\n'.join(lines))
        if not watch or complete:break
        if time.time()>=deadline:
            for folder in FOLDERS:
                if folder not in rendered:update(folder,render=True)
            break
        time.sleep(30)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch',action='store_true');args=parser.parse_args();main(args.watch)
