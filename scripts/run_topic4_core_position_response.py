"""Structured endpoint-prior position interventions on the core-input substrate.

No simulator changes. Reuses verified input-pilot baselines and preserves all
previous artifacts. Explicitly separate E-core position and external I state.
"""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src/snn_engine')]
import numpy as np
from scripts import run_topic4_core_driven_input_pilot as core
rt=core.rt
PARENT=core.OUT
OUT=Path('/data/hfosp/topic4_sef_hfo/core_position_scl_response_20260910')
SCRIPT=Path(__file__).resolve()
GEOMETRY_AUDIT=Path('/data/hfosp/topic4_sef_hfo/geometry_threshold_refinement_20260909/anchor_audit.json')


def configure():
    core.OUT=OUT;core.SCRIPT=SCRIPT


def prepare():
    if (OUT/'plan.json').exists():return
    OUT.mkdir(parents=True,exist_ok=True)
    parent=rt.read(PARENT/'plan.json');base=copy.deepcopy(next(c for c in parent['candidates'] if c['id']=='core_ou_only_Istate_off'))
    audit=rt.read(GEOMETRY_AUDIT);weighted=[x['weighted_center'] for x in audit['rows']]
    with np.load(PARENT/'formal/units/core_ou_only/847101/workers/trajectory.npz') as a:
        contacts=a['contact_xy_mm'];names=a['contact_names'].astype(str)
    upper=contacts[[list(names).index(n) for n in ['SCL9','SCL8']]].mean(0)
    original=np.asarray(base['centers_mm'][0]);vector=upper-original;distance=np.linalg.norm(vector)
    # A geometry-guided probe: the circular edge lies 0.5 mm from the upper
    # pair's geometric center, along the ray from the endpoint prior center.
    near=upper-vector/distance*(base['radii_mm'][0]+.5)
    cases=[]
    def add(cid,label,centers,state=False,reference=None,dy=None,group='position'):
        c=copy.deepcopy(base);c.update(id=cid,label=label,centers_mm=copy.deepcopy(centers),retain_I_state=state,
            preserve_exact_baseline=reference is not None,stage=group,reference_source=reference,y_shift_mm=dy)
        cases.append(c)
    centers=base['centers_mm']
    add('base_off','端点几何原位',centers,reference='core_ou_only_Istate_off',dy=0.)
    p=copy.deepcopy(centers);p[0]=weighted[0];add('weighted_A_off','仅左核参与加权中心',p,group='weighted_prior')
    add('weighted_AB_off','两核参与加权中心',weighted,group='weighted_prior')
    for dy in [1.5,3.,4.5]:
        p=copy.deepcopy(centers);p[0][1]+=dy
        add('A_up_'+str(dy).replace('.','p')+'_off',f'左核上移 {dy:g} mm',p,dy=dy,group='vertical_series')
    p=copy.deepcopy(centers);p[0]=near.tolist();add('A_near_upper_SCL_off','左核边缘靠近上部 SCL',p,group='geometric_probe')
    add('base_on','原位＋旧慢 I 状态',centers,state=True,reference='core_ou_only',dy=0.,group='state_bridge')
    p=copy.deepcopy(centers);p[0][1]+=3.
    add('A_up_3p0_on','上移 3 mm＋旧慢 I 状态',p,state=True,dy=3.,group='state_bridge')
    plan=copy.deepcopy(parent)
    plan.update(version='core_position_response_v1',output_root=str(OUT),candidates=cases,default_candidate='base_off',max_parallel=14,
        formal_runs=18,new_formal_runs=14,reused_formal_runs=4,canary_duration_ms=500.,
        source_snapshot={str(p):rt.sha(p) for p in [*core.SOURCE_FILES,SCRIPT]},
        position_contract='E-core geometry and its shared-OU loading move together. Radius, per-neuron latent threshold draws, gain, graph/weights/delays, tonic background and observation contract fixed. No threshold dose compensation.',
        primary_state_contract='External slow I state OFF for all primary geometry contrasts. Two ON conditions are a separately labeled paired bridge with historical I indices/loading held fixed.',
        statistical_unit='One topology 2511, two paired seed runs per condition; events nested in runs. Common seeds are not claimed to imply identical Poisson innovations after changing rates.',
        prior_source=dict(path=str(GEOMETRY_AUDIT),sha256=rt.sha(GEOMETRY_AUDIT),role='development geometry prior; weighted centers use frozen patient FIT participation, not independent validation'),
        upper_scl_probe=dict(contacts=['SCL9','SCL8'],pair_center_mm=upper.tolist(),core_center_mm=near.tolist(),edge_to_pair_center_mm=.5,interpretation='geometry probe, no event-order or named route stimulus'),
        stop='Fixed 14 new 20-second trajectories plus four existing baseline trajectories; all-condition analysis and report update. No adaptive extra wave or model acceptance by score.')
    rt.write(OUT/'plan.json',plan)
    (OUT/'candidates').mkdir();(OUT/'logs').mkdir()
    for c in cases:
        rt.write(OUT/'candidates'/f'{c["id"]}.json',c)
        if c['reference_source']:
            for kind,seeds in [('formal',plan['seeds']),('canary',plan['seeds'][:1])]:
                for seed in seeds:
                    source=PARENT/kind/'units'/c['reference_source']/str(seed)
                    result=source/'workers/trajectory.json';rr=rt.read(result)
                    assert rr['status']=='COMPLETE' and rt.sha(result.with_suffix('.npz'))==rr['arrays_sha256']
                    dest=OUT/kind/'units'/c['id']/str(seed);dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(source,target_is_directory=True)
    contact_rows=[]
    for c in cases:
        for i,(name,xy) in enumerate(zip(names,contacts)):
            dd=np.linalg.norm(xy-np.asarray(c['centers_mm'][0]));contact_rows.append(dict(candidate=c['id'],label=c['label'],contact=name,
                center_distance_mm=float(dd),edge_gap_mm=float(max(0,dd-c['radii_mm'][0])),inside_left_core=bool(dd<=c['radii_mm'][0])))
    rt.write(OUT/'geometry_to_contact_distances.json',contact_rows)
    lines=['# 端点先验下的 core 位置—传播响应','',
        '本轮明确测试位置变化，而非重开全部参数搜索。原位、左核/双核参与加权中心、左核上移1.5/3/4.5 mm，以及让左核边缘距SCL9/SCL8几何中心0.5 mm的斜向位置，全部事前固定；最大位移是扩大范围的几何探针，不称为已优化的小幅微调。',
        '', '主比较关闭外加慢I状态，使用上一轮已经完成的同模型对照为基线，避免移动E易激区时把旧位置的慢I投影混入主要解释。保留原位和上移3 mm的慢I开启配对桥；开启组的I投影固定在历史位置，不冒称整个E/I局部状态都移动。',
        '', '半径约1.753 mm、右核原位（双加权例外）、图/突触权重/时延、降阈值抽样规律和强度、GABA18ms、全场固定均值Poisson、核内共享OU、空间OU关闭均固定。移动的是E阈值支持与它的共享OU支持，静态节点重新入/出core；实际成员数及阈值总量按结果记录，不补偿。',
        '', '拓扑2511，噪声847101/847102，各20秒；14条新正式轨迹，复用4条已完成原位轨迹。7条新500ms位置canary只核对参数/输出；不要求先看见某类或拟合好才继续。最多14 worker，8 GiB进程余量预算、60 GiB机器余量、18 GiB实际进程树保护。',
        '', '主要看逐触点参与概率（SCL9/SCL8单列，同时保留全部SCL/ICL）、不分模式和TA/TB条件分布、成对顺序错误、具体毫秒时序、完整黑底包络及原生多事件活动。原始检测、窗口排除、实际每类N和原生连续轨迹全部保留；SCL变亮不能抵消ICL丢失或TB错误。',
        '', '先交位置×观测配对响应：上移是否增加SCL、是否丢ICL、是否改善两杆相对顺序，作用是否在两条噪声同向。事件不足或runaway单独呈现，不硬补一个loss。已用几何不作为独立验证；分类标签不能代替传播恢复。',
        '', '本批位置效果来自指定输入/阈值模型；固定seed配对不宣称更改输入率后Poisson创新逐时刻相同。无参数自适应选择、无额外路线刺激、无自动下一轮。分析与PDF更新在正式轨迹结束后自动执行。']
    (OUT/'execution_plan.md').write_text('\n'.join(lines)+'\n')


def worker(cid,seed,canary):
    configure();core.worker(cid,seed,canary)
    result=core.result_path(cid,seed,canary);unit=result.parent.parent
    plan=rt.read(OUT/'plan.json');c=next(c for c in plan['candidates'] if c['id']==cid)
    reference=PARENT/'formal/units'/('core_ou_only' if c['retain_I_state'] else 'core_ou_only_Istate_off')/str(seed)/'workers/trajectory.npz'
    with np.load(result.with_suffix('.npz')) as a,np.load(reference) as b:
        common=(a['h']>0)&(b['h']>0);outside=(a['h']==0)&(b['h']==0)
        checks=dict(no_raised_thresholds=bool(np.all(a['vtheta']<=18)),same_static_positions=bool(np.array_equal(a['positions_E'],b['positions_E'])),
            common_core_thresholds_unchanged=bool(np.array_equal(a['vtheta'][:32000][common],b['vtheta'][:32000][common])),
            common_outside_thresholds_unchanged=bool(np.array_equal(a['vtheta'][:32000][outside],b['vtheta'][:32000][outside])),
            historical_I_projection_fixed=bool(np.array_equal(a['I_target_indices'],b['I_target_indices']) and np.array_equal(a['I_loading'],b['I_loading'])))
        checks['n_core_E']=int((a['h']>0).sum());checks['actual_total_lowering_mV']=float(np.sum(18.-a['vtheta'][:32000]))
    if not all(v for k,v in checks.items() if isinstance(v,bool)):raise RuntimeError(f'position application mismatch: {checks}')
    rr=rt.read(result);br=rt.read(reference.with_suffix('.json'))
    checks['fixed_connectivity']=all(v==br['static_array_identity'][k] for k,v in rr['static_array_identity'].items() if k.startswith(('positions_','ampa_','gaba_')))
    if not checks['fixed_connectivity']:raise RuntimeError('connectivity changed in position experiment')
    rt.write(unit/'position_application_audit.json',checks)


def controller():
    prepare();configure();core.queue(True)
    plan=rt.read(OUT/'plan.json')
    for c in plan['candidates']:
        if c['reference_source']:continue
        path=core.result_path(c['id'],plan['seeds'][0],True)
        assert rt.read(path)['actual_duration_ms']==500.
        assert (path.parent.parent/'position_application_audit.json').exists()
    rt.write(OUT/'canary_audit.json',dict(status='PASS',new_canaries=7,reused=2,meaning='position application and fixed-parameter checks; not patient propagation success'))
    core.queue(False)
    rt.write(OUT/'status.json',dict(status='ANALYZING',new_formal_runs=14,reused_formal_runs=4,updated_unix=time.time()))
    subprocess.run([rt.PYTHON,str(ROOT/'scripts/analyze_topic4_core_position_response.py')],cwd=ROOT,env=rt.ENV,check=True)
    rt.write(OUT/'status.json',dict(status='COMPLETE_PENDING_SCIENTIFIC_REVIEW',formal_runs=18,new_formal_runs=14,reused_formal_runs=4,updated_unix=time.time()))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','worker','controller']);ap.add_argument('--candidate');ap.add_argument('--seed',type=int);ap.add_argument('--canary',action='store_true');a=ap.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='worker':worker(a.candidate,a.seed,a.canary)
    else:
        try:controller()
        except Exception as exc:rt.write(OUT/'status.json',dict(status='FAILED',error=repr(exc),updated_unix=time.time()));raise
