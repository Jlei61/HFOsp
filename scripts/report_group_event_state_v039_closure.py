#!/usr/bin/env python3
"""Chinese closeout prose from the complete, auditable design-pilot summary."""
import argparse,csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def number(value):
    if value is None:return '不可估'
    if abs(value)<1e-10:return '0.0000'
    return f'{value:.4f}' if abs(value)>=1e-4 or value==0 else f'{value:.2e}'


def effect(value):
    if not value['n']:return '不可估'
    vs=value['values'];return f"{number(value['median'])} [{number(min(vs))}, {number(max(vs))}]"


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'


def effect_list(values):
    values=[v for v in values if v is not None and np.isfinite(v)]
    return effect(dict(n=len(values),median=float(np.median(values)) if values else None,values=values))



def support_label(rows):
    """Held-out denominator behind a transfer cell.

    The main table already prints非重叠留出目标窗 next to every effect; the
    transfer tables carry the only surviving positive lead and printed no
    denominator at all.  E1096's expression probe has 24 held-out anchors while
    E253's has 147, so the case with the least support is the one with the
    positive -- a reader has to be able to see that in the same row.
    """
    counts = [
        row[key] for row in (rows or []) for key in ('n_selection_anchors', 'n_anchors')
        if isinstance(row, dict) and isinstance(row.get(key), (int, float))
    ]
    if not counts:
        return '不可估'
    low, high = min(counts), max(counts)
    return str(int(low)) if low == high else f'{int(low)}–{int(high)}'


# Share strict pairing/provenance checks with the machine-readable review audit.
import sys
sys.path.insert(0, str(ROOT))
from src.topic5_group_event_state.v039.instrument_audit import known_truth_margin, convergence_audit


def report(folder):
    s=json.loads((folder/'summary_main.json').read_text());root=Path(s['root']);subjects=list(s['single_view_family_selection']['family'])
    def grouped(section,**filters):
        matches=[r for r in s['grouped'][section] if all(r.get(k)==v for k,v in filters.items())]
        if len(matches)!=1:raise ValueError(('Ambiguous summary row',section,filters,len(matches)))
        return matches[0]
    def primary(subject):return s['single_view_family_selection']['family'][subject]
    def mse(section,subject,endpoint,view='joint',arm='state'):
        row=grouped(section,subject=subject,family=primary(subject),history_hours=8.,view=view,endpoint=endpoint,arm=arm)
        return effect(row['effects']['gain_over_constant_floored'])
    def anchors(section,subject,endpoint,view='joint',arm='state'):
        row=grouped(section,subject=subject,family=primary(subject),history_hours=8.,view=view,endpoint=endpoint,arm=arm)
        return support_label(row.get('physical_support'))
    h1=[];nonlinear=[];gradient=[]
    for subject in subjects:
        family=primary(subject);row=grouped('main',subject=subject,family=family,history_hours=8.,lead_hours=2,support_subset='all_anchors')
        long=grouped('paired',subject=subject,contrast=family+':H8_over_H0.5',lead_hours=2,support_subset='all_anchors')
        recent=grouped('paired',subject=subject,contrast=family+':H8_over_H0.5',lead_hours=2,support_subset='recent_input_available')
        support=next(r for r in s['data_support'][subject] if r['phase']=='SELECTION' and r['lead_hours']==2)
        h1.append([subject.replace('epilepsiae_','E'),family,support['n_nonoverlapping_target_windows'],effect(row['effects']['gain_over_constant_floored']),effect(long['effects']['gain']),effect(recent['effects']['gain'])])
        n_l=grouped('paired',subject=subject,contrast='N_over_L',lead_hours=2,support_subset='all_anchors');n_f=grouped('paired',subject=subject,contrast='N_over_F',lead_hours=2,support_subset='all_anchors')
        usage=[r['curvature_to_full_drift'] for r in s['nonlinear_usage']['rows'] if r['subject']==subject and r['history_hours']==8 and r['view']=='joint']
        nonlinear.append([subject.replace('epilepsiae_','E'),effect(n_l['effects']['gain']),effect(n_f['effects']['gain']),f'{number(min(usage))}–{number(max(usage))}'])
        gs=[r for r in s['gradient_audit'] if r['subject']==subject and r['family']==family and r['history_hours']==8 and r['view']=='joint']
        values=[g for r in gs for g in r['actual_loss_gradient_l1_6_to_8h']]
        gradient.append([subject.replace('epilepsiae_','E'),family,', '.join(str(r['selected_step']) for r in sorted(gs,key=lambda r:r['seed'])),
            f"{sum(v>1e-12 for v in values)}/{len(values)}",f"{number(min(values))}–{number(max(values))}"])
    contact=[]
    for subject in subjects:
        for endpoint,label in [('exact_next_subset','同prefix下一集合'),('next_subset_all','全部下一集合'),('stop','STOP')]:
            contact.append([subject.replace('epilepsiae_','E'),label,anchors('contact',subject,endpoint),
                mse('contact',subject,endpoint),mse('contact',subject,endpoint,arm='functional')])
    expr=[]
    for endpoint,label in [('band_centroid','频带时间质心'),('band_log_energy','频带能量'),('band_log_peak','频带峰值'),('cross_band_lag','跨频带延迟'),
            ('waveform_statistics','波形统计'),('propagation_span_s','传播跨度'),('next_group_delay_s','下一组延迟'),('contact_coupling','触点coupling')]:
        expr.append([label]+[f"{mse('expression',subject,endpoint)}（n={anchors('expression',subject,endpoint)}）"
            for subject in subjects])
    cross=[]
    for subject in subjects:
        for view,endpoint,label in [('count','cross_future_recruitment_vector_2h','计数 → 粗招募完整向量'),('recruitment','cross_future_count_log1p_2h','粗招募 → log1p计数')]:
            own=[r['metrics']['2']['gain_over_refitted_constant_floored'] for r in s['single_view_training'] if r['subject']==subject and r['view']==view]
            cross.append([subject.replace('epilepsiae_','E'),label,anchors('expression',subject,endpoint,view),effect_list(own),mse('expression',subject,endpoint,view),mse('expression',subject,endpoint,view,'functional')])
    seizure=[]
    for r in s['seizure_transfer']['rows']:
        seizure.append([r['subject'].replace('epilepsiae_','E'),'/'.join(str(r['onsets_by_phase'][p]) for p in ['FIT','INNER','SELECTION']),
            '/'.join(str(r['risk']['observed_seizures_by_phase'][p]) for p in ['FIT','INNER','SELECTION']),'NOT_ESTIMABLE'])
    sensitivities=[]
    for subject in subjects:
        family=primary(subject);rows=[r for r in s['sensitivity'] if r['subject']==subject]
        eo=[r for r in rows if r['family']==family and r['event_only']]
        ro=[r for r in rows if r['family']==family and r['forecast']=='rollout']
        wide=[r for r in rows if r['width']==32]
        sensitivities.append([subject.replace('epilepsiae_','E'),family,
            effect_list([r['metrics']['2']['gain_over_refitted_constant_floored'] for r in eo]),
            effect_list([r['metrics']['2']['gain_over_refitted_constant_floored'] for r in ro]),
            effect_list([r['metrics']['6']['gain_over_refitted_constant_floored'] for r in ro]),
            effect_list([r['paired_comparisons']['2']['gain_over_direct_same_family'] for r in wide]),
            effect_list([r['paired_comparisons']['2']['gain_over_N16'] for r in wide])])
    repairs=[f"{r['subject'].replace('epilepsiae_','E')}：删除{r['removed_windows']}个受事件core污染的背景窗" for r in s['input_repair']['rows']]
    optimization=s['optimization'];capped=sum(v['event']['optimization_limited'] for v in optimization.values());initial=sum(v['event']['initial_selected'] for v in optimization.values())
    replay=sum(r['replay_pass'] for r in s['gradient_audit']);fd=sum(r['finite_difference_pass'] for r in s['gradient_audit'])
    chosen_primary=[r for r in s['main_rows'] if r['family']==primary(r['subject']) and r['history_hours']==8 and r['lead_hours']==2 and r['support_subset']=='all_anchors']
    primary_positive=sum(r['gain_over_constant_floored'] is not None and r['gain_over_constant_floored']>1e-8 for r in chosen_primary)
    primary_statement=(f'主要H8历史、提前2小时的比较中，按INNER预先选定的三个家族，在{len(chosen_primary)}条优化重复中均未超过背景与重拟合常数封底。相对其他模型名次更好，不能代替正的事件净贡献。'
        if primary_positive==0 else f'主要H8历史、提前2小时比较的{len(chosen_primary)}条优化重复中，有{primary_positive}条事件净收益为正；这是三个已见患者内的优化重复计数，不是独立患者支持率。')
    synthetic_n={r['family']:r['held_out']['total'] for r in s['instruments'] if r['experiment']=='joint' and r['case']=='nonlinear_transition'}
    nl_margin=known_truth_margin(s['instruments'],'nonlinear_transition','N','L')
    cutoff_text=('' if nl_margin.get('shared_selected_step') is None else
        f"该场景四个家族**都停在同一预算截断（第{nl_margin['shared_selected_step']}步）**、"
        f"同一学习率、{nl_margin['n_seeds']}个训练seed，因此这个margin是同一截断点上的读数，不是收敛后的margin；"
        "若N与L收敛速度不同，收敛后的差可能更大也可能更小。")
    instrument_statement=(f"已知真值的非线性转移场景中，N的留出联合损失为{number(synthetic_n['N'])}，L为{number(synthetic_n['L'])}，"
        f"完整F为{number(synthetic_n['F'])}；N相对L/F的收益分别为{number(synthetic_n['L']-synthetic_n['N'])}/{number(synthetic_n['F']-synthetic_n['N'])}。"
        f"这是旧预算下的历史读数；这里只跑一个训练seed，且预算边缘仍有改善，"
        f"不能据此宣称所有患者条件下仪器已经充分灵敏。{cutoff_text}"
        "真值块自陈 real_patient_power_calibrated=false：没有任何检验显示它能在真实患者的样本量和效应量下检出非线性。")
    conv=convergence_audit(root)
    if conv is not None:
        # This sidecar binds the report to validated cards and per-episode scores.
        (folder/'instrument_budget_audit.json').write_text(json.dumps(conv,ensure_ascii=False,indent=2)+'\n')
        nl=conv['N_over_L'];nf=conv['N_over_F'];lf=conv['L_over_F']
        stops=('全部由INNER耐心规则停止' if conv['all_patience_stopped'] else
               f"其中{conv['budget_limited_runs']}次仍由预算停止")
        instrument_statement+=(
            f"\n\n**提高至2400步预算后的配对复核（{len(conv['seeds'])}个优化seed，{stops}）不再支持旧N-over-L优势。**"
            f"N相对L为{number(nl['median'])}（{nl['positive']}/{nl['n']}个优化seed为正），"
            f"旧480步读数为{number(synthetic_n['L']-synthetic_n['N'])}；"
            f"L的选中步为{min(conv['selected_steps']['L'])}–{max(conv['selected_steps']['L'])}。"
            f"N相对完整F为{number(nf['median'])}（{nf['positive']}/{nf['n']}），L相对F为{number(lf['median'])}（{lf['positive']}/{lf['n']}）。"
            "三次是同一合成数据seed上的优化重复，不是三个独立真值数据集。"
            "耐心停止只说明当前学习率和验证规则下进入平台；尚未验证降低学习率、容量或其他训练配方后的稳定性，不能写成已充分收敛。"
            "当前场景分不开非线性与线性转移：N-over-L没有正对照支持，人体阴性不能据此排除非线性。"
            "L/N超过这套七核F，只构成该合成场景下学习历史整合的正向线索；不能自动认证真实患者功效或生理转移被识别。"
            "逐卡来源、停止原因和评分hash见同目录instrument_budget_audit.json。")
    precision_max=max(r['maximum_absolute_difference'] for r in s['saved_score_precision']['by_subject'].values())
    precision_text=f'{precision_max:.3g}'
    plain=f'''# v0.3.8审阅收口与v0.3.9设计pilot结论

**v0.3.8的审阅修复包已完成；原始“统一、非线性病理状态及事件反馈”科学闭环仍未建立。v0.3.9已把缺失的可学习转移比较和同一冻结状态的未训练任务迁移实际跑完，结果应按下面各层分别解读。**

本轮是E1096、E1125、E253三个已经看过的设计病例，不是新患者独立确认。已完成{ s['registered_human_fits'] }个有限人体拟合、{s['selected_main_states']}个按INNER选择的主状态和{s['selected_single_view_states']}个单视图状态；没有打开development/sealed，不按H1阳性筛掉迁移状态，也没有以发作标签反选上游。

{primary_statement}这一点不证明事件中没有状态信息，尤其不能把分母小或优化受限的结果变成生物学阴性。

## 一、旧报告收口与这次实际修复

旧包已恢复对照封底、逐卡源码版本、错时/窗口诊断和最内层基线原点诊断；撤掉没有定义的完成分数。最内层在第0步不自动等于后续所有强历史读出都是截距，需分层看实际拟合。用户复核撤回的“候选标志自相矛盾”也不再保留为问题，宽joint与附加严格证据链分别解释。

沿原始生产代码追溯还发现旧触点选择使用全记录计数，三例FIT-only触点集合确实改变，因此重建466个块的事件测量和9个新触点contact decoder。此项限制写入v0.3.8平行附录；旧结果保留为条件于回顾性测量字典的线索，不能补写成严格前瞻证据。

跨200秒处理段的事件恢复clock/count/参与，细mark保留缺失；背景随后单独修复。{'；'.join(repairs)}。E253一个零事件短块补提取6个背景窗。新旧所有事件输入、历史、anchor、目标与划分逐项一致，背景及其FIT归一化重新计算，所有依赖人体拟合都按原配方重跑。旧背景结果被隔离，不混入本报告。

## 二、事件状态和长历史究竟带来什么

F是完整固定多尺度历史；L是可学习线性转移；N额外允许状态依赖的非线性转移。表中家族由joint H8的INNER选择，学习率也只由INNER选择。主要目标从查询时刻向前推2小时后，评分随后30分钟；H8和H0.5指真实事件历史长度。

收益为正才表示改善。数值为三个优化seed的**中位数 [最小, 最大]**，区间不是置信区间；物理窗口单列。绝对值小于1e-10的数仅在正文显示为0，机器文件保留原值。

{table(['病例','INNER家族','非重叠留出目标窗','超过背景/重拟合常数封底','H8超过H0.5','近期有发布事件的配对子集'],h1)}
主表效应按全部合格查询anchor平均；非重叠窗列单独说明物理支持，逐窗文件另给非重叠目标的重算结果。三个主家族在非重叠目标重算后仍均无正净收益。近期状态需要等待原始块结束才收到token，所以长短历史还包含信息可用性差异。右列专门检查近期确有已发布输入的子集；不能把全部H8收益直接命名为八小时生理记忆。逐物理窗口结果见[配对分数](paired_window_scores.csv)，不由seed补分母。

非线性必须另外超过重新训练的L和完整F，不能由N模型名称、输入矩阵更新或非零梯度代替：

{table(['病例','N超过L','N超过F','转移曲率相对总drift的范围'],nonlinear)}
曲率量是相对于FIT均值处仿射切线的偏离，只描述冻结模型在已观测状态上的非线性使用。它可以非零，而留出净收益仍不成立；也不能跨患者当成同一生理量比较。

固定预算的敏感性比较单列，不回头改变主家族：

{table(['病例','主家族','event-only净收益/lead2h','自主rollout净收益/lead2h','自主rollout净收益/lead6h','L32超过L16/lead2h','L32超过N16/lead2h'],sensitivities)}
event-only保留时钟、覆盖及发布延迟元信息，移除测得背景波形；其收益相对自己的时钟父预测和重拟合常数。rollout仅从当前状态自主推进，不能读取未来实际事件或背景。所有lead的目标宽度仍为30分钟，lead6h不是从现在累积六小时事件。容量比较是重新拟合的L32，不用无作用参数凑数。

E253的event-only两小时结果保留了局部正值，但三个seed并非一致，加入强背景后的主事件净贡献仍未成立。自主rollout也未形成三seed一致的正净收益。E1096六小时lead没有合格留出目标，表内不可估不是零收益；不把较短提前量的结果外推到六小时。
## 三、训练与长程梯度

{table(['病例','INNER家族','三个seed选中事件训练步','6–8h真实事件梯度非零窗口/审计窗口','实际损失梯度L1范围'],gradient)}
全部72个冻结状态的独立事件重放通过{replay}/72，有限差分检查通过{fd}/72。重放从真实事件逐条重建输入，再回到实际held-out计数/招募损失；不是画预设指数响应。N在实际状态上的曲率及局部Jacobian变化另见机器汇总，非零非线性分支也可能近似线性。

梯度L1依赖输入归一化和事件数，不能用作患者间的生理作用强度比较。F的长历史路径仍来自固定核，不能把它的梯度称为学到的时间常数；新合同最长H8，也不声称本轮验证了16–32小时记忆。

上述逐事件与缓存重算在相同float64精度下比较；另将其与原保存的float32评分核对，登记重放窗口中的最大差为{precision_text}。这不是全数据误差上界，也不声称逐bit一致；极小数值差异不升级为独立科学支持。

不能把所有卡统称“训练充分”：270卡中{capped}卡事件阶段触及预算且未触发耐心停止，{initial}卡由INNER选回初始事件残差。前者标优化受限，后者表示这次拟合选择了背景回退；两者都不能作为生物学阴性。实际更新次数、LR、batch、初始化、每层形状和归一化均有记录。

## 四、同一冻结状态能否预测没有参与上游训练的事件表达

下表沿用各病例INNER所选家族的H8状态，比较相同冻结checkpoint的原始latent与上游任务功能读出。背景、初始化、完整F历史和重拟合常数使用一致的读出结构、选模规则与预算；输入维度不同仍会带来参数数量差异，并非总参数严格相等。表内收益先按父基线/常数封底。

{table(['病例','端点','留出锚点数','原始latent净收益','上游功能读出净收益'],contact)}
contact主端点精确条件于前两组触点身份和第三组大小，分叉资格只在FIT确定；STOP单列。它是离线测量后的结构条件预测，尚不是实时原始波形的prefix识别器。

细表达原始latent的全部登记端点如下，收益单位为FIT标准化后的均方误差差值：

{table(['未训练端点']+[s.replace('epilepsiae_','E') for s in subjects],expr)}
E253只评分FIT实际可测的四个频带和六个频带对，最高频带明确缺失。各频带时间质心不是中心频率；延迟和coupling在映射触点上，频带和波形统计在完整新FIT传感集合上。

局部线索仍应保留：E1096的N状态在未训练的频带时间质心上，三个优化seed均有正收益，并均超过完整F和初始化表征；但同一主状态的上游联合事件净收益未成立，contact同prefix集合也没有一致收益。因此这是一项患者内跨任务线索，不能补成完整证据链。

**读这张表时必须同时看分母。** 三例的留出锚点数相差数倍（E1096最少、E253最多，见表内n），E1096 的频带时间质心线索只有24个锚点。锚点可能共享历史和原始记录块，不能把这些数字当作独立样本数；也不能凭三个病例断言分母与效应存在系统关系。下一轮须同时导出原始块、会话和互不重叠窗口的配对分数。

F的原始latent是确定性的固定历史，同一病例同一历史长度在不同seed下完全相同。因此其确定性ridge探针出现三条相同分数，只是一份表征的重复来源，不是三次独立支持；机器汇总另报每个臂实际有多少种不同特征数组。

联合多头模型可能把几个独立原因装进一个向量，因此另做两个单视图的冻结对向探针：

这里的count-view/recruitment-view限制的是上游训练目标，输入仍可包含全部已发布的过去事件特征；它们不是“只准读过去计数”或“只准读过去招募”的输入消融。

{table(['病例','未训练方向','留出锚点数','上游自身任务净收益','跨任务latent净收益','跨任务功能读出净收益'],cross)}
上游自身任务分别用计数NB或招募BCE评分；跨任务探针用FIT标准化均方误差，不能把这两种单位直接相减。自身任务与新任务必须沿同一状态卡分别核验，不能只挑对向任务的一条正值。

E1096“招募 → 计数”的最大latent收益来自选中第0步的状态，和初始化表征完全相同；另一条正值的上游招募任务仍无净增益。E253的count-view功能读出在两个seed上改善未来粗招募，但其中一条没有超过初始化表征，且第三个seed为负。这些差异均保留在完整对照表中，尚不是稳定的共享状态结论。
不能只凭latent跨任务收益就宣称一个共同病理因子。六类已知真值仪器同时保留固定历史充分、线性状态/非线性观测、非线性转移、背景决定结果、独立原因及无事件反馈反例；所有结果均在[机器汇总](summary_main.json)和图中列出。

功能读出的维度更低，也会丢掉状态信息；它没有迁移不能单独证伪多维共同状态。这里将它与完整历史、初始化表征、原始latent及独立原因反例共同解读，不把它换成新的统一否决门槛。

{instrument_statement}

## 五、发作关系和H3

全部上游冻结后才打开登记的发作分母：

{table(['病例','原始发作 FIT/INNER/留出','有过去支持的下一发作 FIT/INNER/留出','风险及早期空间/路径'],seizure)}
风险网格重新生成：每5分钟查询，仅要求此前8小时至少90%测量支持、且此前历史不与已开始的发作重叠，不按未来是否纯间期筛选。每个查询只对应之后6小时内的第一场发作，不把同一查询后的一串发作都算成独立可预测事件。

E1096的原始3/1/4次通过最低数量门槛，但带过去支持的下一发作只剩2/0/1次；其INNER发作前无法满足登记的历史/重置条件。三例都未达到风险可估条件，空间任务也未同时满足FIT/INNER/留出发作门槛，未拟合风险或空间读出。支持数仍是进一步数值和随访核验前的上界，不能称为发作关系阴性。H1按未来纯间期筛选的anchor没有被误用成风险查询集。

H3仍未被人体数据识别：IED到来后模型状态更新，是观察器吸收证据；损失追溯到旧事件，是模型敏感性。这两件事都不等于IED改变了生理状态。下一步需让“共同原因/无反馈”“计数反馈”“内容反馈”三种生成模型都允许观察更新，再比较独立未来观测是否需要反馈项。

## 六、可据此写出的结论边界

现在可以按同一患者、同一seed、同一冻结checkpoint评估事件历史、线性/非线性转移、未训练事件表达和可估发作结果；原先被跨患者拼接的证据不再算一条链。上面的配对收益说明每个设计病例支持到哪里，并不自动建立统一机制。

本轮只有三个已见设计病例，且部分窗口很少、部分优化受限、三例发作迁移不可估。已补的真实窗口合成诊断直接把真因子交给读出器，是乐观的条件灵敏度；没有完成整个状态学习和发作流程在患者支持下的功效证明。因此仍不能称队列结论、科学阴性或临床预测器。后续按[独立确认合同]({ROOT/'docs/archive/topic5/group_event_state_v0_3_9_independent_confirmation_contract_2026-09-05.md'})推进，保持有限预算和预先登记。
'''
    tech=['# v0.3.9技术收口与可复现索引','',f'正式人体根目录：`{root}`。所有人体数据和依赖拟合使用背景修正版；旧模拟仪器及新触点decoder按冻结hash复用。',
        '', '## P0/P1审阅发现及处理', '',
        table(['级别','实际问题','影响与处理'],[
            ['P0','旧触点选择读取全记录目录计数','不能当成仅过去信息的测量算子；三例FIT集合实际改变，重建466块及新触点decoder，旧测量前瞻资格撤回。'],
            ['P1','对照损害被计入状态收益','旧汇总恢复父基线封底和对照损害诊断；新H1、contact和表达探针均在同一评分支持聚合后封底。'],
            ['P1','逐卡版本/窗口质量被旧汇总丢弃','恢复审计；新人体数据统一修正版，源码按角色分层，混版本旧seed不借新结果取得资格。'],
            ['P1','最内层原点被误读成整体强基线合格','逐层记录选中步、真实更新和预算，基础臂与后续历史臂分别判断；不把q0自动扩写为所有后续读出仅截距。'],
            ['P1','跨200秒段事件细mark缺失导致时钟被删','保留检测可确定的clock/count/participation/coupling，未知细mark为缺失；记录覆盖不因细mark缺失删去。'],
            ['P1','背景避让清单没有同步恢复的事件core','剔除60/215/540个重叠背景窗，零事件短块补6窗；背景标准化和依赖拟合按原有限配方重跑。'],
            ['P1','一个不可测频带让完整响应向量全部不可估','按不同FIT anchor的响应支持选列，保存分量索引；E253保留4频带和6频带对，不以缺失当阴性。'],
            ['P1','潜在的风险查询人群错误','H1未来纯间期筛选不能沿用到风险集；新H2b先核验原始发作分母，合格时才生成仅基于过去支持的查询。']]),
        '', '## 输入、模型与优化', '',
        '输入按完整原始块结束发布，H按事件实际时刻裁切，波形/细mark可用性另有显式位。计数是未来30分钟事件数；粗招募是每个物理shaft内参与比例的事件均值，不是“窗口内是否至少出现一次”的二值标签。零事件时该招募目标未定义，不能填零当阴性。',
        '', '新FIT事件传感集合为E1096六触点、E1125九触点、E253九触点。E1125的HR11没有可用MRI坐标，因此contact decoder与映射空间目标为八触点；其频带/波形探针仍在完整九触点集合，范围不混写。',
        '', '事件特征冻结FIT median/MAD，clip=8，计数按FIT事件/小时缩放；背景独立FIT中心/尺度及可用性，再构造七个固定历史核。E1096/E1125输入135/195维、背景389维；E253事件195维、背景341维（少一个不可支持频带）。实际维度以数据卡为准。',
        '', 'F宽度=7×事件输入维度，无可学习转移参数；L/N主状态16维，L32仅敏感性。L的A为反对称矩阵加负对角；N增加rank8的U tanh(Vs+b)。K的完整16×16参数化有对称冗余，不能把全部256项都称独立动力自由度。事件写入B形状16×输入维度，N的U/V为16×8和8×16。',
        '', '背景读出：(背景维度+lead1)→32 Tanh→(计数1+粗招募维度)；事件残差：(状态维度+lead1)→32 Tanh→同输出。另有一个NB dispersion标量。没有64/128隐藏层，也没有BatchNorm或LayerNorm；batch=128与层宽32是不同概念。',
        '', '人体AdamW，weight_decay=1e-4，全局梯度裁剪2；背景/常数LR=.003，上限各800步；事件LR=.001/.003按INNER选择，上限800步。每25步验证，耐心背景/常数8次、事件12次，初始权重可被选中。批抽样保留原CUDA随机流，逐批核验FIT索引，记录实际batch最小/最大及序列hash。',
        '', 'L/N以RK4最大5分钟步长积分，长间隔插零输入；full-history BPTT使用32步activation checkpoint而不detach。F解析指数衰减。gradient审计另以float64从个体真实事件重放，检查中心有限差分和半步积分；梯度不是生理冲激响应。',
        '', '## 全部有限拟合的训练状态', '']
    opt=[]
    with (folder/'model_parameter_inventory.csv').open() as stream:inventory=list(csv.DictReader(stream))
    counts={}
    for row in inventory:
        if float(row['history_hours'])==8 and row['view']=='joint' and int(row['seed'])==20260905:
            key=(row['subject'],row['family'],row['module']);counts[key]=counts.get(key,0)+int(row['parameters'])
    dimensions=[]
    for subject in subjects:
        for family in 'FLN':
            dimensions.append([subject.replace('epilepsiae_','E'),family]+[counts.get((subject,family,m),0) for m in ['observer','residual','background','constant']])
    tech += [table(['病例','家族','转移/写入参数','事件读出参数','背景读出参数','常数对照参数'],dimensions),
        '这里统计各阶段实际存在的参数，冻结后的requires_grad=False不意味着它从未训练。F另有固定核及完整写入buffer，不记可学习参数；F的完整历史使读出输入更宽，所以各家族并非总参数严格相等，另用有效L32容量比较解释N的优势或劣势。','']
    for batch_folder,stages in optimization.items():
        for stage,v in stages.items():opt.append([batch_folder,stage,v['cards'],v['initial_selected'],v['optimization_limited'],v['optimizer_updates']])
    tech.append(table(['实验','阶段','卡数','选初始步','预算受限','实际更新总数'],opt))
    tech += ['','## 模拟仪器及优化边界','',
        '独立episode仪器：每类3072条历史，2048 FIT/512 INNER/512留出，一个训练seed，两个预设LR，480步上限。C/F/L/N联合48卡，单视图72卡，L32容量12卡，自主rollout36卡，修正完整招募向量后的冻结交叉探针36卡。上限触及仍在改善时标优化受限，不能反过来为了N获胜扩搜索。','']
    instr=[]
    for row in s['instruments']:
        if row['experiment']=='joint':instr.append([row['case'],row['family'],row['selected_lr'],row['selected_step'],row['optimization_limited'],number(row['held_out']['total'])])
    tech.append(table(['真值场景','家族','INNER所选LR','选中步','预算受限','留出联合损失'],instr))
    tech += ['','## 下游与统计','',
        'contact adapter：8单元GELU，冻结decoder后先做静态logit校准400步，再独立背景400步，冻结父预测；state/initialized/F/functional/constant各600步上限。同维度输入共享初始化与抽样seed，恒等特征复用同一结果。每批均匀抽anchor，再抽该anchor一个事件。精确枚举剩余触点中大小为k的无序集合，评分第三组条件概率；不是单触点softmax平均。','',
        '表达探针：FIT按anchor加权中心/尺度，ridge=.01/.1/1/10，INNER选取，零残差回退始终可选。仅用FIT可测分量选列，保存索引，不用INNER/留出响应选择列。均方误差先在同anchor事件平均，再跨anchor平均。功能读出为上游实际训练任务在2h lead的事件残差输出。','',
        '所有对照收益在同一评分集聚合后由父预测封底；错时收益重新取真实matched donor子集，不直接相减不同分母的两个总体均值。H1错时只在连续5分钟评分段内半圈移位；contact错时保持prefix和组大小、至少2小时错开、排除中间发作，缺失配对记不可估。','',
        '主要效果表是三个seed的中位数和范围，没有把seed当独立患者，没有对重叠anchor做制造精度的bootstrap。非重叠30分钟目标仍可能共用八小时历史及生理状态；三例是设计病例。','',
        '## 可追溯文件','',
        '- [机器汇总](summary_main.json)：全部主状态、敏感性、端点、梯度、曲率、发作分母、模拟与版本分层。',
        '- [每层参数清单](model_parameter_inventory.csv)：真实shape与参数数量。',
        '- [主端点评分](main_endpoint_scores.csv)与[配对比较](paired_history_transition_comparisons.csv)：全部0/2/6小时lead，全部/非重叠/近期输入子集。',
        '- [contact评分](frozen_contact_scores.csv)与[细表达评分](frozen_expression_scores.csv)：完整端点与对照，不只列正值。',
        '- [逐窗分数](paired_window_scores.csv)、[文件完整性](artifact_integrity.json)、[图说明](figures/README.md)。',
        '',f"文件哈希核对{s['integrity']['checks']}项，结果{'PASS' if s['integrity']['passed'] else 'FAIL'}；版本按生产角色分层，不把不同源码seed混算一个中位数。",'',
        '旧仪器第一次启动的导入失败、旧背景人体队列的一张CUDA索引异常、背景修复前队列和被替代的探针都保留在旧根目录，未作为生物学阴性或正式完整卡。CUDA索引异常两GPU独立重放未复现，原因仍未定位；新批索引审计不被写成根因证明。','',
        '修复背景后另有两次初次运行失败：E1096 F 自主预测（LR=0.003，seed=20260906）出现 CUDA illegal instruction；E1125 F count-view（LR=0.001，同seed）出现索引断言。两次都在GPU 1，尚不能定位为软件或硬件根因。每张只登记一次相同源码、输入、种子和预算的重跑，开启 CUDA_LAUNCH_BLOCKING=1。原队列分别保留125/126、35/36成功的初次状态，最终闭环另列恢复记录，不称零失败，日志和进度均归档。这些运行错误不进入生物学阴性分母。','',
        '本轮没有人为清除旧输出、重写旧checkpoint、打开development/sealed、提交或发布代码。源码快照和队列状态是本地可复现资产；后续确认需独立数据及全流程功效校准。']
    for name,text in [('group_event_state_v039_closeout_plain.md',plain),('group_event_state_v039_closeout_technical.md','\n'.join(tech)+'\n')]:
        p=folder/name
        if p.exists():raise FileExistsError(p)
        p.write_text(text)
    print(json.dumps(dict(status='COMPLETE',reports=2)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reports',type=Path,required=True);report(p.parse_args().reports)
