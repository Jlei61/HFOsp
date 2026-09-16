"""Freeze the final eight units after complete midpoint propagation review.

Two previously observed backgrounds, two OU correlations, two noise replays.
No physiological labels, event timing or routes enter the input process.
"""
import copy
import time
from scripts import run_topic4_propagation_recovery_night as n
from scripts import run_topic4_core_ou_correlation_recovery as worker


def prepare():
    dest=n.OUT/'final_B_selection.json'
    if dest.exists():return n.rt.read(dest)
    audit=n.rt.read(n.OUT/'core_ou_correlation_canary_audit.json')
    assert audit['status']=='PASS_ENGINEERING_ONLY'
    assert all(audit['rho1_exact_array_parity'].values())
    a=n.rt.read(n.OUT/'final_A_units.json')
    assert all(n.old.complete(n.old.result_path(a['stage'],*u)) for u in a['units'])
    assert (n.OUT/'final_A_preselection_review/manifest.json').exists()
    for path,h in n.rt.read(worker.PLAN)['source_snapshot'].items():
        assert n.rt.sha(path)==h,path
    lp=n.rt.read(n.OUT/'long_selection.json');lookup={c['id']:c for c in lp['candidates']}
    stage='recovery_core_OU_correlation_20260911';cases=[]
    for recipe,parent,title in [
        ('mid_EE075','refine_mid_EE075','端点上移3mm；核内EE×0.75'),
        ('mid_EE085_mean095','refine_mid_EE085_mean095','端点上移3mm；EE×0.85、输入均值×0.95')]:
        for rho in [.5,0.]:
            c=copy.deepcopy(lookup[parent])
            c.update(id=f'coreOU_{recipe}_rho{round(rho*100):03d}',stage=stage,
                parent_id=parent,changed_parameter='core_ou_correlation',changed_value=rho,
                changed_parameters={'core_ou_correlation':rho},core_ou_correlation=rho,
                display_name=title+f'；两核慢输入相关系数ρ={rho:g}')
            path=n.old.OUT/'candidates'/f'{c["id"]}.json'
            if path.exists() and n.rt.read(path)!=c:raise RuntimeError('candidate identity collision')
            n.rt.write(path,c);cases.append(c)
    reason=('中点4条件×2噪声全部完整：1035个全检测、651个原primary。EE0.75两条件的TA仍有约100ms的SCL偏晚；'
        'EE0.85加均值0.95和左核降幅1.15使TA杆间中位转负，但常见示例失去左端ICL及上部SCL，不能将换了参与集合后的杆间中位解释为完整恢复。'
        '因此没有提名中点工作点。末8条回到同一上移3mm布局的两个完整既有条件：EE0.75有两噪声中的少见完整TA，'
        'EE0.85加均值0.95曾出现更相容的TB个例；这些是能力诊断而非恢复验收。检验共享核输入是否限制不同传播响应的出现机会。')
    record=dict(status='FROZEN_BEFORE_DISPATCH',frozen_unix=time.time(),phase='final_B',stage=stage,
        candidates=cases,topology_seed=2511,seeds=[847101,847102],duration_ms=60000.,worker_script=str(worker.SCRIPT),reason=reason,
        review_sources=[str(n.OUT/x) for x in ['final_A_complete_rapid_summary.json','final_A_preselection_review/manifest.json','phase3_review.md',
            'phase2_review.md','capacity_examples_long/refine_mid_EE075/selection.json','capacity_examples_long/refine_mid_EE085_mean095/selection.json']],
        hypothesis='Shared core-wide OU may restrict access to distinct responses; this is not an established explanation of the propagation residual.',
        contrast='Within each background only rho changes from inherited1 to0.5 or0. Marginal OU mean/variance/tau are identical in law before clipping; finite empirical values are logged. Independent cell Poisson remains only within cores.',
        fixed='Same geometry, graph, delays, static thresholds and other physical parameters within each parent contrast. Original primary observer, labels, patient target, positive loss scales and no-route-input rule unchanged.',
        random_identity='Seeds paired by identity, not a claim of identical realized per-step Poisson/global innovations across rho. Independent core OU has a separate reproducible namespace.',
        interpretation=['More TA/TB labels without better conditional contact paths: access/composition response only.',
            'Better conditional paths and broader support: candidate improvement requiring later new-graph/noise confirmation.',
            'Little change or worse propagation: no support for this correlation intervention as the remedy; do not automatically repeat noise search.'],
        no_new_gate='Actual event support remains explicit; no requirement that every replay or bin contain both modes.',
        budget='Final B8 completes48 new /168 formal units. No new-graph confirmation in this allocation because no full propagation substrate was accepted. Additional engineering3x500ms is separately disclosed.',
        engineering_audit=str(n.OUT/'core_ou_correlation_canary_audit.json'),
        inherited_metadata_note='The canary plan retains historical score_status SPECIFIED_NOT_IMPLEMENTED_OR_CALIBRATED from its ancestor; current offline scorer and positive calibration are already implemented and frozen. This stale field is not an execution gate or a new scoring contract.')
    n.rt.write(dest,record)
    plan=n.rt.read(n.OUT/'plan.json');plan['confirmation']['amendment']['last_selection']=str(dest)
    plan['confirmation']['amendment']['last_units']=8;plan['confirmation']['amendment']['new_graph_confirmation_executed']=False
    n.rt.write(n.OUT/'plan.json',plan)
    with (n.OUT/'execution_amendment.md').open('a') as f:
        f.write('\n\n## 05:04：完整中点审阅后，冻结最后8条核输入相关性实验\n\n'+reason+'\n\n每个背景只改变两核慢输入OU相关系数ρ=0.5/0，ρ=1参照已完成；2背景×2相关系数×2原噪声×60秒。逐核OU边际规律不增加方差，核外无随机输入，无路由或标签刺激。静态图和阈值保持父条件身份，相同seed不冒称相同实际Poisson输入。3条500ms工程检查已通过，ρ=1与原执行器62项数组完全相同。完整观察器审计与全分布图继续并行产出；没有等待其排版完成才派发，也未放宽原primary规则。全夜48新增/168正式上限不变；本轮没有新图确认。\n')
    return record


if __name__=='__main__':print(prepare()['stage'])
