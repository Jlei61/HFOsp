"""Bounded final panel: replay two observed effects and test their combination.

Preparation is allowed only after all 16 long-panel outputs are complete.
The added corner is a prospective interaction experiment, not a candidate already
selected for good training fit. Four conditions x two graphs x two inputs = 16.
"""
import argparse,copy
from scripts import run_topic4_propagation_recovery_night as n
from scripts import run_topic4_core_tonic_recovery as tonic

def prepare(layout,reason):
    assert layout in ['near','mid']
    dest=n.OUT/'confirmation_selection.json'
    if dest.exists():
        record=n.rt.read(dest)
        if record['layout']!=layout:raise RuntimeError('Cannot change a frozen confirmation selection')
        return record
    long=n.rt.read(n.OUT/'long_selection.json')
    assert all(n.old.complete(n.old.result_path(long['stage'],c['id'],2511,s)) for c in long['candidates'] for s in long['seeds']), 'Long panel incomplete'
    old=n.rt.read(n.old.OUT/'plan.json');tonic_plan=n.rt.read(tonic.PLAN)
    for p,h in old['source_snapshot'].items():
        if n.rt.sha(p)!=h:raise RuntimeError('Frozen physics changed: '+p)
    seeds=n.rt.read(n.old.OUT/'confirmation/seed_freeze.json')
    topologies=seeds['topology_seeds'];dynamics=seeds['dynamics_seeds'];assert len(topologies)==len(dynamics)==2
    stage='recovery_factorial_confirmation_20260911'
    prefix='refine_'+layout+'_'
    cases=[n.rt.read(n.old.OUT/'candidates'/f'{prefix}{suffix}.json') for suffix in ['EE085','EE085_A115','EE085_mean095']]
    joint=copy.deepcopy(cases[1]);joint.update(id=prefix+'EE085_A115_mean095',stage=stage,core_mean_rate_scale=.95,
        parent_id=cases[1]['id'],display_name=('靠近上部 SCL' if layout=='near' else '端点原位上移 3 mm')+'；EE×0.85＋左核降幅×1.15＋输入均值×0.95',
        changed_parameter='core_mean_rate_scale',changed_value=.95,changed_parameters={'core_mean_rate_scale':.95},
        experimental_role='Prospective joint corner; no earlier training trajectory or fit-based nomination')
    candidate_path=n.old.OUT/'candidates'/f'{joint["id"]}.json'
    if candidate_path.exists() and n.rt.read(candidate_path)!=joint:raise RuntimeError('Candidate identity collision')
    n.rt.write(candidate_path,joint);cases.append(joint)
    for c in cases:
        assert c['parameters']['EE_same_core_scale']==.85
        assert c['parameters']['depth_B_scale']==1.
        assert c['centers_mm']==cases[0]['centers_mm'] and c['radii_mm']==cases[0]['radii_mm']
        assert {k:v for k,v in c['parameters'].items() if k!='depth_A_scale'}=={k:v for k,v in cases[0]['parameters'].items() if k!='depth_A_scale'}
    assert {(c['parameters']['depth_A_scale'],c['core_mean_rate_scale']) for c in cases}=={(a,m) for a in [1.,1.15] for m in [1.,.95]}
    record=dict(status='FROZEN_BEFORE_DISPATCH',stage=stage,layout=layout,candidates=cases,topology_seeds=topologies,seeds=dynamics,
        duration_ms=60000.,worker_script=str(tonic.SCRIPT),source_plan=str(tonic.PLAN),
        design='2 left-core threshold-depth values x 2 core afferent-mean values x 2 new graphs x 2 new dynamics inputs; same EE0.85 and chosen geometry.',
        selection_reason=reason,statistical_scope='Three previously observed corners receive new graph/input replay; the fourth is a prospective interaction test, not confirmation of an already fitted optimum.',
        comparison='Vth changes preserve actual input on a fixed geometry; mean changes preserve input law parameters except the mean, not identical realizations. All four corners share topology and dynamics identifiers.',
        budget='Replaces the original final 16 replay units, no addition beyond 48 new / 168 total; existing 08:12 dispatch and 10:12 hard stop retained.',
        continued_limits='No raised core-E threshold, no random input outside E-core, GABA18ms, no directed stimulation, no new route/label loss or relaxed primary qualification.',
        source_snapshot=dict(old['source_snapshot'],**{str(tonic.SCRIPT):n.rt.sha(tonic.SCRIPT),str(tonic.PLAN):n.rt.sha(tonic.PLAN)}))
    n.rt.write(dest,record)
    plan=n.rt.read(n.OUT/'plan.json');plan['confirmation']['amendment']=dict(selection=str(dest),kind='prospective_2x2_interaction_and_new_graph_replay',runs=16)
    n.rt.write(n.OUT/'plan.json',plan)
    with (n.OUT/'execution_amendment.md').open('a') as f:
        f.write('\n\n## 最后16条：两个已观察作用的组合及新图重演\n\n'+reason+'\n\n固定所选几何及核内EE0.85，左核降阈值倍数1/1.15 × 核内输入均值1/0.95构成2×2。三个既有角点在两个新拓扑、每图两条新噪声中重演；第四个组合是预期性新实验，不能称为已拟合最优点的确认。16条替代原最后16条确认，正式预算和时间上限不变；保持原患者目标、事件资格、观察器与物理限制。\n')
    return record

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--layout',choices=['near','mid'],required=True);parser.add_argument('--reason',required=True)
    a=parser.parse_args();print(prepare(a.layout,a.reason)['stage'])
