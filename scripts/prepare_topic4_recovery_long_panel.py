"""Freeze the second overnight panel from the completed first-wave evidence."""
import copy
import numpy as np
from scripts import run_topic4_propagation_recovery_night as n
from scripts import run_topic4_core_tonic_recovery as t

def prepare():
    dest=n.OUT/'long_selection.json'
    if dest.exists():return n.rt.read(dest)
    plan=n.rt.read(n.OUT/'plan.json');w=plan['wave1']
    assert all(n.old.complete(n.old.result_path(w['stage'],c['id'],2511,s)) for c in w['candidates'] for s in w['seeds'])
    assert n.rt.read(n.OUT/'tonic_canary_audit.json')['status']=='PASS'
    parent=next(c for c in w['candidates'] if c['id']=='recovery_upper_wide_recurrence')
    old=n.rt.read(n.old.OUT/'plan.json');endpoint=next(c for c in old['candidates'] if c['id']=='endpoint__baseline')
    mid=copy.deepcopy(endpoint['centers_mm']);mid[0][1]+=3.
    stage='recovery_long_20260911';cases=[]
    recipes=[('EE075',.75,1.,1.,'核内 EE ×0.75'),('EE085',.85,1.,1.,'核内 EE ×0.85'),
        ('EE085_A115',.85,1.15,1.,'EE ×0.85＋左核降幅 ×1.15'),('EE085_mean095',.85,1.,.95,'EE ×0.85＋核内输入均值 ×0.95')]
    for layout,centers,title in [('near',parent['centers_mm'],'靠近上部 SCL'),('mid',mid,'端点原位上移 3 mm')]:
      for name,ee,depth,mean,meaning in recipes:
        c=copy.deepcopy(parent);c.update(id=f'refine_{layout}_{name}',stage=stage,layout='near_upper' if layout=='near' else 'intermediate_up3',
            centers_mm=copy.deepcopy(centers),display_name=f'{title}；{meaning}',core_mean_rate_scale=mean,
            changed_parameter='declared_refinement',changed_value=None)
        c['parameters'].update(EE_same_core_scale=ee,depth_A_scale=depth,depth_B_scale=1.)
        c['parent_id']=('recovery_upper_wide_recurrence' if layout=='near' else 'refine_near_EE075') if name=='EE075' else f'refine_{layout}_'+('EE075' if name=='EE085' else 'EE085')
        c['changed_parameters']=({'EE_same_core_scale':ee} if name=='EE085' else {'depth_A_scale':depth} if name=='EE085_A115' else {'core_mean_rate_scale':mean} if name=='EE085_mean095' else {})
        c['geometry_change']=dict(from_centers_mm=parent['centers_mm'],to_centers_mm=centers) if layout=='mid' and name=='EE075' else None
        c['radii_mm']=[c['parameters'][f'radius_{k}_mm'] for k in ['A','B']]
        assert np.linalg.norm(np.asarray(centers[0])-centers[1])>sum(c['radii_mm'])
        assert all(min(x[0],x[1],20-x[0],20-x[1])>=r for x,r in zip(centers,c['radii_mm']))
        assert all(old['axes'][k]['adaptive_bounds'][0]<=v<=old['axes'][k]['adaptive_bounds'][1] for k,v in c['parameters'].items())
        path=n.old.OUT/'candidates'/f'{c["id"]}.json'
        if path.exists() and n.rt.read(path)!=c:raise RuntimeError('candidate collision')
        n.rt.write(path,c);cases.append(c)
    selection=dict(status='FROZEN_BEFORE_DISPATCH',stage=stage,candidates=cases,topology_seed=2511,seeds=[847101,847102],duration_ms=60000.,
        worker_script=str(t.SCRIPT),source_plan=str(t.PLAN),
        reason='Completed 16-run wave: 1125 detections, 120 primary; only weaker intraccore EE condition has both replays scorable (21/22 primary), but TA scarce. Interpolate recurrence, adjust left-core excitability, compare a nearer-prior geometry and one core-input mean perturbation.',
        allocation='16 runs replaces original 16 long-support-only replays; no increase in formal 48-new/168-total or 8-10h limits',
        controls='Each EE/depth/input effect has a same-geometry 60s direct comparator; geometry contrast changes only left-core location and its associated membership/input support.',
        display_correction='Fig2C main displays participants only. Retain full-contact QC, add participant-conditioned comparison. This does not change the frozen scorer or event selection.',
        no_change='patient target, classifier, primary selection, global GABA18ms, core nonpositive threshold shifts, outside deterministic mean and no external slow I/ZM/kick')
    n.rt.write(dest,selection);n.rt.write(n.OUT/'plan_before_long_amendment.json',plan)
    plan['long_support']['amendment']=dict(kind='bounded_residual_refinement_and_long_support',selection=str(dest),run_count=16,
        input_assay='Only the mean095 arms alter afferent means; all others use the exact scale1 engine path validated against62 arrays.')
    n.rt.write(n.OUT/'plan.json',plan)
    with (n.OUT/'execution_amendment.md').open('a') as f:
        f.write('''\n\n## 第二阶段修订：2026-09-11 01时，首批16条完成后\n\n首批1125次检测、120次原primary；减弱核内EE至0.75的条件有21/22个primary，两核完整平均率约5Hz，但仍以TB为主。强弱阈值偏置分别产生几乎单一TA或TB，尚非完整双模式。下一批将原16条单纯长重演改为8条件×2噪声×60秒：靠近上部SCL与端点原位上移3mm两布局，各比较核内EE0.75、EE0.85、EE0.85加左核降幅1.15、EE0.85加核内输入均值0.95。每个参数效应有同几何60秒对照；总48条新增/168条正式上限及时间限制不变。\n\n输入均值是独立版本的小对照，经已有external_e_rate_drive接口仅向E-core加入常数速率偏移。条件Poisson方差随均值自然改变，OU参数、核外确定均值、GABA及原观察器不变。3条500ms实施检查通过，scale1与原引擎62/62数组完全相同；这不是传播阳性证据。正式source与参数在tonic_input_plan.json、long_selection.json中冻结。\n\n对照图的层次也予以明确：此前全触点黑底图包含患者未参与触点的亮信号，是全窗口QC；Fig2C主图实际只使用参与且质心可用的触点。补充参与定义匹配的显示，同时保留全触点QC和完整模型原生场；不更改训练mask/资格，不把Fig2C方向筛选示例当自然模式全分布。\n''')
    return selection

if __name__=='__main__':print(prepare()['stage'])
