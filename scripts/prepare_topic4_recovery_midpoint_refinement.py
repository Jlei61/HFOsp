"""Eight prospective local refinements after the complete sixteen-run long panel.

The remaining eight units are reserved for a separate review. No automatic
confirmation of a model that still has systematic propagation residuals.
"""
import copy
import numpy as np
from scripts import run_topic4_propagation_recovery_night as n
from scripts import run_topic4_core_tonic_recovery as tonic

def prepare():
    dest=n.OUT/'final_A_selection.json'
    if dest.exists():return n.rt.read(dest)
    lp=n.rt.read(n.OUT/'long_selection.json')
    assert all(n.old.complete(n.old.result_path(lp['stage'],c['id'],2511,s)) for c in lp['candidates'] for s in lp['seeds'])
    lookup={c['id']:c for c in lp['candidates']}
    centers=(np.asarray(lookup['refine_near_EE075']['centers_mm'])+np.asarray(lookup['refine_mid_EE075']['centers_mm']))/2
    stage='recovery_final_refinement_20260911';cases=[]
    for recipe,parent,depth,mean in [
        ('EE075','refine_mid_EE075',1.,1.),('EE075_A115','refine_mid_EE075',1.15,1.),
        ('EE085_mean095','refine_mid_EE085_mean095',1.,.95),('EE085_A115_mean095','refine_mid_EE085_mean095',1.15,.95)]:
        c=copy.deepcopy(lookup[parent]);c.update(id='refine_midpoint_'+recipe,stage=stage,layout='midpoint_near_and_up3',centers_mm=centers.tolist(),
            parent_id=parent if depth==1 else 'refine_midpoint_'+('EE075' if recipe.startswith('EE075') else 'EE085_mean095'),
            changed_parameter='core_center' if depth==1 else 'depth_A_scale',changed_value=None if depth==1 else depth,
            changed_parameters={} if depth==1 else {'depth_A_scale':depth},core_mean_rate_scale=mean,
            geometry_change=dict(from_centers_mm=lookup[parent]['centers_mm'],to_centers_mm=centers.tolist()) if depth==1 else None,
            display_name='两已测位置的中点；'+('核内EE×0.75' if recipe.startswith('EE075') else 'EE×0.85、输入均值×0.95')+('、左核降幅×1.15' if depth!=1 else ''))
        c['parameters']['depth_A_scale']=depth;c['parameters']['depth_B_scale']=1.
        assert np.linalg.norm(centers[0]-centers[1])>sum(c['radii_mm'])
        assert all(min(x[0],x[1],20-x[0],20-x[1])>=r for x,r in zip(centers,c['radii_mm']))
        p=n.old.OUT/'candidates'/f'{c["id"]}.json'
        if p.exists() and n.rt.read(p)!=c:raise RuntimeError('candidate collision')
        n.rt.write(p,c);cases.append(c)
    reason=('完整第二批16条仍未恢复患者两类传播：近SCL的增强左核条件TA占主导但ICL缺段；中间位置EE0.75补回TA参与，却有SCL系统性偏晚。'
        '中间位置EE0.85和左核降幅1.15仍主要为TB；均值0.95增加合格事件量而TA稀少。因此末16条先分8条在两个已测位置的几何中点做局部精调，'
        '保留EE0.75与EE0.85加均值0.95两种有事件支持的参数组合，各比较左核降幅1和1.15。两套组合的差异不能归因于单一EE因素。')
    record=dict(status='FROZEN_BEFORE_DISPATCH',stage=stage,phase='final_A',candidates=cases,topology_seed=2511,seeds=[847101,847102],duration_ms=60000.,
        worker_script=str(tonic.SCRIPT),reason=reason,geometry_rule='Arithmetic midpoint of the two already tested A locations; B unchanged. No event-specific source, route or stimulus.',
        interpretation='Four fixed prospective conditions, each replayed with two noises. Vth contrasts within the same midpoint geometry are paired; movement changes the input-cell support.',
        remaining_eight='After this complete eight-unit batch, review full propagation, distributions and actual support. Use the final <=8 units for new-graph replay if propagation improves materially, otherwise another bounded targeted refinement. Freeze before dispatch.',
        budget='Final A 8 plus Final B <=8 replaces the former final16 confirmation allocation. Total remains <=48 new / <=168 formal; same 8-worker and 08:12/10:12 limits.',
        unchanged='Original observer, patient target, classifier, primary rule and loss; nonpositive core-E threshold offsets; external randomness only in E-core; GABA18ms; no kick, spatial OU, external slow I, Z/M.')
    n.rt.write(dest,record)
    plan=n.rt.read(n.OUT/'plan.json');n.rt.write(n.OUT/'plan_before_final_subdivision.json',plan)
    plan['confirmation']['amendment']=dict(kind='two_bounded_eight_unit_stages_prioritize_propagation_recovery',first_selection=str(dest),first_units=8,last_units_max=8,
        no_automatic_new_graph_replay=True,total_new_max=48)
    n.rt.write(n.OUT/'plan.json',plan)
    with (n.OUT/'execution_amendment.md').open('a') as f:
        f.write('\n\n## 03:55：末16条先分成两个最多8条的阶段\n\n'+reason+'\n\n第一阶段固定4条件×2原噪声×60秒，拓扑2511。第二阶段最多8条，等待首8条完整结果；传播有实质改善再做新图重演，否则继续有边界的针对性组合。不是16条已拟合候选确认，也不是DE；先前未执行的2×2新图确认生成器仅为草案。全夜48条新增/168条正式、最多8 worker及时间上限不变。\n')
    return record

if __name__=='__main__':print(prepare()['stage'])
