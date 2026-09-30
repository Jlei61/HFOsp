#!/usr/bin/env python3
"""Paired native finite-window graph responses; no bifurcation interpolation."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import contextlib
import io
import json
import numpy as np
import analyze_topic4_loop_zk_conditional as analysis
import run_topic4_loop_axis_conditional as run


def main():
    rows=[];arrays={}
    contract=run.native.base.read(run.CONTRACT)
    names=run.native.base.read(run.OUT/'rotated/queue.json')['names']
    for condition,root in [('current',run.PRIMARY),('rotated',run.OUT/'rotated'),('isotropic',run.OUT/'isotropic')]:
        previous=analysis.OUT;analysis.OUT=root
        try:
            for name in names:
                if not (root/'runs'/name/'result.json').exists():continue
                row,inputs=analysis.analyze(name)
                row['axis_condition']=condition;row['source_root']=str(root)
                rows.append(row);arrays[(condition,name)]=inputs
            if condition!='current':
                with contextlib.redirect_stdout(io.StringIO()):analysis.main()
        finally:analysis.OUT=previous
    comparisons=[]
    for name in names:
        present=[c for c in ['current','rotated','isotropic'] if (c,name) in arrays]
        if len(present)<2:continue
        first=arrays[(present[0],name)]
        for condition in present[1:]:
            assert np.array_equal(first,arrays[(condition,name)]),(name,present[0],condition)
        chosen=[r for r in rows if r['name']==name]
        by_graph={r['axis_condition']:r for r in chosen}
        readouts=[]
        for condition in present[1:]:
            difference=analysis.compare_readouts(by_graph[present[0]],by_graph[condition])
            difference.update(reference_graph=present[0],comparison_graph=condition)
            readouts.append(difference)
        comparisons.append(dict(name=name,graphs=present,paired_future_input_records_exact=True,
            finite_window_states={r['axis_condition']:r['finite_window_state'] for r in chosen},
            all_completed_states_equal=len({r['finite_window_state'] for r in chosen})==1,
            paired_native_readout_differences=readouts,
            interpretation='Equal coarse labels do not establish identical native states; regional, spatial and contact readout differences remain descriptive.'))
    complete=sum(r['axis_condition']!='current' for r in rows)
    run.native.write(run.OUT/'comparison.json',dict(status='COMPLETE_FINITE_WINDOW' if complete==16 else 'RUNNING_PARTIAL',
        new_completed=complete,new_total=16,rows=rows,paired_comparisons=comparisons,
        contract=contract,formal_bifurcation_status='NOT_ESTABLISHED',
        limits='Four coordinates per graph and two carried histories. A state difference bounds conditional responses; no difference does not prove equal boundaries. Outdegree differs; rotated anisotropy attenuates.'))
    lines=['# 匹配空间结构下的条件响应','','原图复用8条，其余两图各8条；共同Z/K空间场、两个完整历史及未来输入。末10秒状态是30秒有限窗读出，不是平衡点或周期分支认证。同类标签不等于同一空间状态；JSON同时保留两核/核外率、400格点平均率和15触点电流差，传播形态另看原生事件图。','','|Z|K/gL|历史|结构|末10秒状态|平均全E率Hz|短事件数|dZ/s|dK/s|','|---|---|---|---|---|---|---|---|---|']
    for row in rows:
        j=row['job'];dz,dk=row['counterfactual_drift_mean_allE_A_B_other'][0]
        lines.append(f"|{j['target_Z']}|{j['target_K']}|{row['name'].rsplit('_',1)[-1]}|{row['axis_condition']}|{row['finite_window_state']}|{row['tail_mean_Hz'][0]:.2f}|{row['tail_brief_events']}|{dz:.5f}|{dk:.5f}|")
    (run.OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(new_complete=complete,total=16,paired_graph_checks=len(comparisons))),flush=True)


if __name__=='__main__':main()
