"""Bounded first native burst-regime map, with resumable result-based accounting."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse, csv, hashlib, json, subprocess, sys, time
import psutil
import metrics

HERE=Path(__file__).resolve().parent
MAIN=HERE.parents[1]
OUT=MAIN/'results/topic4_sef_hfo/burst_regime_map_20260914'
PYTHON=Path('/home/honglab/leijiaxin/anaconda3/envs/cuda_env/bin/python')

def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(path)

def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def name(j):return f'ee{j["ee"]:g}_d{j["depth"]:g}_n{int(j["noise"])}_t{j["topology"]}_s{j["seed"]}'

def prepare():
    ee=[.5,.7,.85,1.,1.2];depth=[0.,.4,.7,1.,1.3];seeds=[848101,848102]
    jobs=[]
    for d in depth:
        for e in ee:
            for s in seeds:jobs.append(dict(ee=e,depth=d,seed=s,topology=2511,noise=True,family='grid',duration_ms=20000.))
    for d in [0.,1.]:
        for e in [.5,.85,1.2]:jobs.append(dict(ee=e,depth=d,seed=seeds[0],topology=2511,noise=False,family='noise_off',duration_ms=20000.))
    for e,d in [(.5,0.),(.85,1.),(1.2,1.3)]:
        for s in seeds:jobs.append(dict(ee=e,depth=d,seed=s,topology=2711,noise=True,family='new_network',duration_ms=20000.))
    assert len(jobs)==62
    # Complete anchors and a coarse diagonal early, while retaining all predeclared cells.
    priorities=[(.85,1.),(.5,0.),(1.2,1.3),(.5,1.),(1.2,0.)]
    jobs.sort(key=lambda j:(0 if j['family']=='grid' and (j['ee'],j['depth']) in priorities else 1,
        priorities.index((j['ee'],j['depth'])) if (j['ee'],j['depth']) in priorities else 99,j['family'],j['seed'],j['depth'],j['ee']))
    sources={str(p):sha(p) for p in [HERE/'runtime.py',HERE/'metrics.py']}
    reference=Path('/data/hfosp/topic4_sef_hfo/core_multiseed_response_curves_20260913/plan.json')
    inherited=read(reference)['source_snapshot']
    sources.update({p:sha(p) for p in inherited})
    plan=dict(schema='topic4.native_burst_regime.v1',created_unix=time.time(),
        authorization='2026-09-14 user requests model simulation of irregular burst and normal-region phase map',
        question='Does the existing full spatial E/I model have a reproducible irregular self-limited burst region adjacent to low-activity background?',
        ee_values=ee,depth_values=depth,seeds=seeds,jobs=jobs,budget=dict(grid=50,noise_off=6,new_network=6,total=62),
        primary_topology=2511,duration_ms=20000.,burnin_ms=2000.,max_workers=16,
        parameters='same-core E->E absolute multiplier and both-core lowering-depth multiplier; all other parameters and geometry frozen',
        input='core-only independent OU (rho=0) plus Poisson; outside E and all I deterministic expected input; core mean scale=.95; no Z/M/spatial OU/kick',
        normal_definition='Low native activity without detected population bursts; operational model reference, not validated healthy human tissue',
        source_reference=str(reference),source_snapshot=sources,
        burst=dict(observable='Fraction of actual native neurons firing at least once per nonoverlapping 10 ms frame',
            onset_fraction=.10,offset_fraction=.03,join_gap_ms=20.,sensitivity_onset=[.075,.125],
            observation_groups=['coreAE','coreBE','coreUnionE','allE','surroundE'],
            sparse_minimum_bursts=8,sustained_max_episode_s=1.,sustained_duty=.75,
            regular='CV<=.30 AND CV2<=.40 AND half-median ratio<=1.5 AND native ACF peak>=.20',
            irregular='CV>=.40 AND CV2>=.50, with recurrent separate bursts and no sustained-activity criterion',
            residual='variable_bursts; neither an oscillator claim nor forced into irregular',
            synchrony='Peak fraction of native cells active in 2 ms per detected episode',
            CV='sample SD of complete consecutive onset intervals / mean; no cross-run pooling',
            CV2='mean(2*abs(diff(IEI))/(IEI_next+IEI_previous))'),
        scope='finite-time operational regime map; no mathematical phase boundary, healthy-tissue validation, patient fit or autonomous-attractor identification',
        stop='Finish these 62 units and aggregate all outcomes, including sparse/ambiguous/sustained. No automatic additional search or Fig5 entry.')
    if (OUT/'plan.json').exists():
        old=read(OUT/'plan.json')
        for key in ['jobs','source_snapshot','burst']:
            if old[key]!=plan[key]:raise RuntimeError('frozen plan changed: '+key)
        return old
    write(OUT/'plan.json',plan)
    text='''# 模型原生 burst 动力学区域图：第一版执行合同（2026-09-14）

科学问题：在当前完整空间 E/I SNN 中，不规则且可自限的群体 burst 是否占据一个可重演的参数区域，并与低活动背景相邻？irregular 是待检验结果，不预设它必然出现。

横轴为同核 E→E 权重倍率 0.5/0.7/0.85/1/1.2，纵轴为两核降阈值幅度 0/0.4/0.7/1/1.3。当前基线是 (0.85,1)。固定32,000 E + 8,000 I、当前几何、其余连接、限核随机输入与GABA18ms；逐E细胞ΔVth≤0。depth=0保留几何和输入支持，仅撤去病理降阈值，所以“背景区”是模型操作定义，不称健康人组织。

25格×2新噪声=50条，每条20秒，剔除前2秒。另预先固定6条关闭OU和Poisson随机性的对照，保持整流后平稳期平均输入；3个锚点×2噪声在另一张已冻结网络2711上复测，共62条。20秒不足以检验分钟级漂移，稀疏事件不因难估计而删除，也不提高输入凑事件。

原生10ms内至少放电一次的细胞比例定义burst：起始阈值10%、延续阈值3%，低活动间隙≤20ms合并；起始阈值7.5%/12.5%作敏感性。两个core、合并core及全E分别计算。CV/CV2由完整相邻起点间隔给出；n<8标稀疏、不中断运行；单段≥1秒或burst占时≥75%标持续活动。规则区需CV≤0.30、CV2≤0.40、前后半段中位间隔比≤1.5、原生活动ACF峰≥0.20；不规则区需CV≥0.40且CV2≥0.50；中间状态保留为变化性burst。它们是预先约定的有限时间表型，不是数学分岔或自主振荡判定。

同步性另外统计每次burst峰值2ms内活跃细胞比例。全部原生活动进入统计，不使用患者分类器、loss或250ms孤立形态筛选。每次运行是实验单位，事件是运行内样本；两噪声不一致、检测敏感性及新网络不复现均明确显示。

复用冻结构图与膜/突触/输入方程，只有独立进程内的有序scatter实现加速和原生记录器改变。正式运行前验证原执行器/加速器完整状态逐位一致，并与既有基线轨迹相同种子的1秒前缀核对。最多16个进程，保留至少40GiB主机内存；每进程预留9GiB，峰值超过18GiB记工程失败。

交付：区域图、连续CV/CV2/率/同步读数、每类实际存在状态的原生raster和时间序列、全部逐运行表及解释。不存在的状态不画伪造示例；全部62条结束后停止，不自动增加波次、冻结模型或进入Fig5。图由Agent自查后交用户目视验收。
'''
    (OUT/'execution_plan.md').write_text(text)
    return plan

def summarize(job):
    folder=OUT/'per_run'/name(job);result=read(folder/'result.json')
    assert result['status']=='COMPLETE' and sha(folder/'trajectory.npz')==result['arrays_sha256']
    if not (folder/'metrics.json').exists():write(folder/'metrics.json',metrics.run_metrics(folder/'trajectory.npz',result))
    return read(folder/'metrics.json')

def controller():
    import fcntl
    plan=prepare()
    validation=read(OUT/'validation.json')
    assert validation['status']=='PASS' and validation['runtime_sha256']==sha(HERE/'runtime.py')
    for path,h in plan['source_snapshot'].items():
        if sha(path)!=h:raise RuntimeError('source changed: '+path)
    with (OUT/'controller.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pending=[];complete=[]
        for job in plan['jobs']:
            if (OUT/'per_run'/name(job)/'result.json').exists():summarize(job);complete.append(job)
            else:pending.append(job)
        active={};failures=[]
        while pending or active:
            for key,(proc,job,log) in list(active.items()):
                if proc.poll() is not None:
                    log.close();del active[key]
                    if proc.returncode:failures.append(dict(name=key,returncode=proc.returncode,log=str(OUT/'logs'/f'{key}.log')))
                    else:summarize(job);complete.append(job)
            rss={}
            for key,(proc,job,log) in active.items():
                try:
                    tree=[psutil.Process(proc.pid)]+psutil.Process(proc.pid).children(recursive=True)
                    rss[key]=sum(p.memory_info().rss for p in tree)/2**30
                    if rss[key]>18.:
                        for p in reversed(tree):p.terminate()
                        failures.append(dict(name=key,error='process_tree_memory_guard'))
                except psutil.NoSuchProcess:pass
            available=psutil.virtual_memory().available/2**30
            growth=sum(max(0,9-rss.get(k,0)) for k in active)
            if pending and not failures and len(active)<plan['max_workers'] and available>40+growth+9:
                job=pending.pop(0);key=name(job);log=(OUT/'logs'/f'{key}.log').open('a')
                cmd=[str(PYTHON),'-u',str(HERE/'runtime.py'),'--ee',str(job['ee']),'--depth',str(job['depth']),
                    '--seed',str(job['seed']),'--topology',str(job['topology']),'--duration',str(job['duration_ms'])]
                if not job['noise']:cmd.append('--no-noise')
                proc=subprocess.Popen(cmd,cwd=MAIN,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                active[key]=(proc,job,log)
            if failures and not active:break
            snapshot=dict(status='RUNNING' if not failures else 'FAILURE_DRAINING',complete=len(complete),total=len(plan['jobs']),queued=len(pending),
                active=[dict(name=k,pid=p.pid,rss_gib=rss.get(k),progress=read(OUT/'per_run'/k/'progress.json') if (OUT/'per_run'/k/'progress.json').exists() else {}) for k,(p,j,f) in active.items()],
                failures=failures,updated_unix=time.time())
            write(OUT/'status.json',snapshot)
            time.sleep(2)
        write(OUT/'status.json',dict(status='COMPLETE' if not failures else 'FAILED',complete=len(complete),total=len(plan['jobs']),failures=failures,updated_unix=time.time()))
        if failures:raise RuntimeError(str(failures))
        subprocess.run([str(PYTHON),str(HERE/'plot.py')],cwd=MAIN,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args()
    if a.action=='prepare':print(json.dumps(prepare()['budget']))
    else:controller()
