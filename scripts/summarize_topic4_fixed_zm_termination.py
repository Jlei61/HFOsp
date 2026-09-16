#!/usr/bin/env python3
"""Scientific acceptance report for the fixed-native-model termination screen."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import csv,datetime,json
from pathlib import Path
import numpy as np
import analyze_topic4_fixed_zm_termination as a

ROOT=a.OUT

def main():
    folders=['','matched_spatial_round2','hyperpolar_spatial_round3','source_sahp_round4','sahp_bracket_round5','positive_candidate_confirmation','low_fraction_round6','autonomous_recurrence_continuation']
    all_rows=[]
    for sub in folders:
        root=ROOT/sub
        if not (root/'protocol.json').exists():continue
        a.OUT=root;a.run.OUT=root
        for j in a.run.prepare()['initial_jobs']:
            r=a.analyze(j['name'])
            if r:all_rows.append(dict(batch=sub or ROOT.name,source=str(root/'runs'/j['name']),same_trajectory_extension=sub=='autonomous_recurrence_continuation',**r))
    index={r['name']:r for r in all_rows}
    columns=['batch','name','observed_s','complete','same_trajectory_extension','gamma','K_gain','seed','first_entry_s','return_confirmation_s','second_entry_s','max_E_10ms_Hz','late_mean_E_Hz','mean_Z_end','finite_events_before_first_entry','category']
    csvrows=[]
    for r in all_rows:
        j=r['job'];csvrows.append(dict(batch=r['batch'],name=r['name'],observed_s=r['observed_s'],complete=r['complete'],same_trajectory_extension=r['same_trajectory_extension'],gamma=j['gamma'],K_gain=j.get('sahp_gain',0),seed=j['seed'],first_entry_s=r['entries'][0]['onset_s'] if r['entries'] else None,return_confirmation_s=r['recoveries'][0]['confirmation_s'] if r['recoveries'] else None,second_entry_s=r['entries'][1]['onset_s'] if len(r['entries'])>1 else None,max_E_10ms_Hz=r['max_E_10ms_Hz'],late_mean_E_Hz=r['late_mean_Hz'][0],mean_Z_end=r['mean_Z_end'],finite_events_before_first_entry=r['finite_events_before_first_entry'],category=r['category_label']))
    with (ROOT/'all_native_conditions.csv').open('w') as h:
        w=csv.DictWriter(h,fieldnames=columns);w.writeheader();w.writerows(csvrows)
    hit=index['sahp1.5_g0.5_s9108401'];base=index['native_s9108401']
    duration={}
    for r in [base,hit]:
        onset=r['entries'][0]['onset_s'];v=[e['duration_s'] for e in r['finite_events'] if e['end_s']<onset]
        duration[r['name']]=dict(n=len(v),min_s=min(v),median_s=float(np.median(v)),max_s=max(v))
    energy=json.load(open(ROOT/'sahp_bracket_round5/analysis/sahp1.5_g0.5_s9108401_early_energy.json'))
    assert energy['model_sampling_Hz']==10000.
    causal=json.load(open(ROOT/'state_matched_sahp_ablation/causal_comparison.json'))
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    n_unique=sum(not r['same_trajectory_extension'] for r in all_rows)
    doc=[f'# 固定 Z/M 双核模型：自主终止结果与验收\n\n更新时间：{stamp}。\n',
    '## 科学判断\n\n在原有40k手放双核SNN上保留Z/M，增加空间全局反馈及独立的慢钾适应后，已经找到固定参数下自行进入高活动、随后退出的轨迹；状态匹配撤除慢钾的对照支持它对终止有因果贡献。但目前尚未同时保留原工作点的短间期事件、跨噪声稳健入态及患者早期能量空间模式，因此**不冻结为通过验收的Fig5模型**。\n',
    '原文LAS复现只作为机制来源，不混入当前SNN结果。该批次沿用原拓扑、两核位置及阈值场，原生eta_M=0.0005、tau_M=1s、tau_Z=5s、阈值95.198513mV等效量。所有主要轨迹保持参数常数，原生Z/M都开；没有手动reset、Z夹持、定时刺激或额外10s全局滤波。\n',
    '## 实际加入的机制\n\n全局反馈是全E放电驱动的单层15ms滤波，按gamma分配局部和空间均匀抑制；全部仍受逐细胞Z调节。新增k=gK/gL按5s衰减，每次E放电增加0.01×gain，钾反转电位为原生坐标中的−30mV。它是独立新增的sAHP电导，不能称为原有M本身已经足够；也不能称为全局抑制单独实现了终止。\n',
    'E细胞的膜方程新增部分是`gG(EG−V)+k(EK−V)`；保留`IE−Z(1−gamma)II−eta_M M−V`。Z的函数形式保留，但资源输入改为明确声明的`J=(1−gamma)II+gamma C_R R_G`，C_R来自原基线配对均值匹配。导电型全球项与电流型原网络只在参考V=18mV匹配；J不是逐时刻的氯离子通量。该映射是当前实现的模型假设，不能等同于完整复现原文氯动力学。\n',
    '机制依据：[Liou et al., 2020, eLife](https://elifesciences.org/articles/50927)。原文包含距离无关抑制和放电适应；这里将这些机制接入现有显式E/I网络，未替换成原文网络。\n',
    '## 观测与数值判据\n\n单位是同一拓扑下的一条噪声轨迹。全E10ms率≥200Hz连续200ms标记进入；随后连续两个1s窗，全E及两核分别满足平均率<50Hz、至少20%的10ms窗<5Hz，才确认退出。记录全体神经元计数，raster只是固定样本。通过此判据不自动等于持续发作吸引子、Hopf分岔或恢复患者传播分布。\n',
    '| 条件 | 观察时长(s) | 首次进入(s) | 退出确认(s) | 当前状态 |\n|---|---:|---:|---:|---|']
    selected=[('native_s9108401','原工作点'),('hyperpolar_g0.5_phi0_s9108401','gamma=1/2，全局反馈，K=0'),('sahp1_g0.5_s9108401','gamma=1/2，K×1'),('sahp1.25_g0.5_s9108401','gamma=1/2，K×1.25'),('sahp1.5_g0.5_s9108401','gamma=1/2，K×1.5'),('sahp1.75_g0.5_s9108401','gamma=1/2，K×1.75'),('sahp2_g0.5_s9108401','gamma=1/2，K×2'),('sahp4_g0.5_s9108401','gamma=1/2，K×4'),('sahp0.25_g0.166667_s9108401','gamma=1/6，K×0.25'),('sahp0.5_g0.166667_s9108401','gamma=1/6，K×0.5')]
    for name,label in selected:
        if name not in index:continue
        r=index[name];en=f"{r['entries'][0]['onset_s']:.2f}" if r['entries'] else '未达到';ex=f"{r['recoveries'][0]['confirmation_s']:.2f}" if r['recoveries'] else '未确认'
        status='完成' if r['complete'] else '仍在记录或时间截尾'
        doc.append(f"| {label} | {r['observed_s']:g} | {en} | {ex} | {status} |")
    doc.append(f'\n共记录{n_unique}条不同条件/噪声的冷启动轨迹；另有同轨迹延长、观察器重放及状态匹配干预，不重复算独立样本。完整表见[all_native_conditions.csv]({ROOT}/all_native_conditions.csv)。未达到阈值只表示该观察窗内未达到，不能叫恢复，也不能断言永久不发作。\n')
    doc.append('## 自主退出与因果检验\n\nK×1.5在6.33s达到高态判据，7.12–9.12s通过退出窗；K×1.25在5.22s进入、9.34s确认退出。K×1.75在10.85s进入，持续传播后于20.79s确认退出。三条主要轨迹均没有参数干预。二维场显示移动的高率带及波后停止，而非已验证的全局持续振荡。波前接触边界，因此不单凭动画把所有终止归于慢适应。\n')
    off=causal['rows'][1]['mean_E_Hz_10to12']
    doc.append(f'从K×1.25的同一6s完整状态分叉，保留慢钾的母轨迹在10–12s全E平均率为0Hz；只撤除新增钾通道后为{off:.1f}Hz，两核也持续高率，Z继续下降。未改参数的6–8s续跑，已保存的raster、放电计数、突触电流读出、Z/M汇总、区域与空间场及输入历史逐位一致。这个对照说明慢适应影响终止，但它是有干预的诊断，不是新的自主成功轨迹。\n')
    doc.append(f'![状态匹配对照]({ROOT}/state_matched_sahp_ablation/figures/carried_state_sahp_removal.png)\n')
    noise=index.get('sahp1.5_g0.5_s9108402')
    if noise:
        doc.append(f"## 尚未通过的验收内容\n\n同参数换噪声9108402，已观察{noise['observed_s']:g}s，最高全E10ms率{noise['max_E_10ms_Hz']:.1f}Hz，进入次数{len(noise['entries'])}、退出次数{len(noise['recoveries'])}。因此不能把开发种子的单次通过称为跨噪声稳健恢复。\n")
    d0,d1=duration[base['name']],duration[hit['name']]
    doc.append(f"原工作点首次入态前有{d0['n']}个有限事件，时长{d0['min_s']:.2f}–{d0['max_s']:.2f}s、中位{d0['median_s']:.2f}s；K×1.5仅有{d1['n']}个已结束的前期事件，时长{d1['min_s']:.2f}–{d1['max_s']:.2f}s。因此图中写Long event，未硬标为间期HFO。该比较用于指出明显动力学改变，不是用不同观察时长估计患者事件率差。\n")
    doc.append(f"原生10kHz电流重放与原轨迹逐位一致。按固定Fig3C的1–150Hz谱与robust-z流程，模型{energy['model_positive_contacts']}/15触点增强，但与锁定的E10/SZ3空间相关rho={energy['contact_rho']:.3f}；没有根据新结果重挑患者、翻转颜色或重排触点。模型只有5个基线谱帧，患者基线更长，幅值也不能直接等同。500Hz与10kHz的该例相关相同，最大robust-z差约0.0028；本例空间不符不能归咎于采样。\n")
    timing_path=ROOT/'sahp_bracket_round5/early_window_sensitivity.json'
    if timing_path.exists():
        timing=json.load(open(timing_path));early=timing['rows'][0]
        doc.append(f"按持续事件起始规则重取招募早期{early['window_s'][0]:.2f}–{early['window_s'][1]:.2f}s，15/15触点增强，但与同一患者相关仍仅rho={early['rho']:.3f}。这说明窗口语义确实不同，空间匹配仍未建立。该窗口不是按患者相关最大值挑选；原高率门槛后的结果保留。另交付[招募起始版本Fig5]({ROOT}/sahp_bracket_round5/figures/fig5_sahp1.5_g0.5_s9108401_early_recruitment.png)，C中的Recruitment与F使用同一事件起点。\n")
    rec=index.get('sahp1.5_g0.5_s9108401_to50s')
    if rec:doc.append(f"同一K×1.5轨迹已延长到{rec['observed_s']:g}s，当前共记录{len(rec['entries'])}次进入、{len(rec['recoveries'])}次退出。续跑完整携带Z/M/gK、膜、突触、延迟和噪声历史；观察延长不是新种子。是否再次发作以此完整记录判断，不从Z回升自行推断。\n")
    doc.extend(['## 图与下一步界限\n',f"![Fig5候选]({ROOT}/sahp_bracket_round5/figures/fig5_sahp1.5_g0.5_s9108401.png)\n",
    '候选沿用Fig5：A连续raster和两核E放大；B原Z/M及单独注明的新增gK；C同时间点真实二维场；D实测三维轨迹；E离散条件结果；F固定患者比较。D不是nullcline、向量场或分岔证明，E也不是连续参数相图。全图索引在[figure_index.md](figure_index.md)，未通过用户人工验收。\n',
    '下一版首先应保住原来的短事件，再检验从已招募状态自主退出。当前小占比/弱适应检验专门针对这一缺口；不能只按“有return”选工作点。若仍只能得到抑制入态、移动长波或单种子成功，应审阅全局资源输入J与实际导电电流之间的映射、初始化暂态及局部适应强度，而不是继续只扫更多随机位置或把数值阈值改得更容易通过。\n'])
    (ROOT/'scientific_review.md').write_text('\n'.join(doc))
    a.write(ROOT/'scientific_acceptance.json',dict(updated_utc=stamp,n_unique_cold_start_runs=n_unique,completed=sum(r['complete'] for r in all_rows if not r['same_trajectory_extension']),gate_pass_names=[r['name'] for r in all_rows if r['recoveries'] and not r['same_trajectory_extension']],full_Fig5_acceptance='NOT_ESTABLISHED',human_review='PENDING',native_short_event_durations=duration,clinical_comparison=energy,causal_ablation=causal))
    print(ROOT/'scientific_review.md')

if __name__=='__main__':main()
