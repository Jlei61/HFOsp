"""Current overnight evidence, without relabelling older input/state models as current.

Static scientific report only. All builds retained; no experiment dispatch.
"""
from pathlib import Path
import csv,datetime,json,os,shutil,subprocess,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_topic4_model_collaborator_report_v7 import Report,BLUE,GRAY,sha,read,rows,Image
ROOT=Path('/home/honglab/leijiaxin/HFOsp')
SEARCH=Path('/data/hfosp/topic4_sef_hfo/core_connectivity_search_20260910')
NIGHT=Path('/data/hfosp/topic4_sef_hfo/core_propagation_recovery_20260911')
OUT=ROOT/'results/topic4_sef_hfo/model_collaborator_report_v8_2026-09-11'
PHASE_TITLES={'wave1':'首批组合','long':'60秒精调','final_A':'中点位置精调','final_B':'末批针对性实验','confirmation':'新图与新噪声验证'}

def fixed_shaft_figure(path):
    """Current patient/model comparisons must use fixed rows, not archived order."""
    path=Path(path)
    if path.name.endswith('_patient_spectra_model_envelopes.png') and not path.parent.parent.name.endswith('_shaft_fixed'):
        folder=path.parent.parent
        path=folder.with_name(folder.name+'_shaft_fixed')/'figures'/path.name
        if not path.exists():raise FileNotFoundError(f'Fixed contact-row figure required: {path}')
    return path

def build():
    now=datetime.datetime.now().astimezone();directory=OUT/'builds'/now.strftime('%Y%m%dT%H%M%S%f');directory.mkdir(parents=True)
    r=Report(directory,now.strftime('%Y-%m-%d %H:%M %z'),version=8)
    p=read(SEARCH/'plan.json');review=read(SEARCH/'analysis/response_summary.json');state=read(NIGHT/'status.json');np=read(NIGHT/'plan.json')
    for path,key in [(Path(__file__),'report_producer'),(Path(__file__).with_name('build_topic4_model_collaborator_report_v7.py'),'shared_report_renderer'),(SEARCH/'plan.json','screen_plan'),(NIGHT/'plan.json','night_plan'),(NIGHT/'status.json','night_status'),(NIGHT/'execution_amendment.md','execution_amendment'),(ROOT/'docs/topic4_patient_geometry_prior_snn.md','shared_contract')]:r.source(path,key)
    r.start('当前判断：参数作用已出现，患者双模式尚未恢复')
    r.para('核心问题不变：在患者传播端点几何先验附近，用局部易激双核和 E/I 连接产生患者相容的两种传播，再建立参数改变与观测改变的对应关系。两个分类标签、较低损失或两杆都被招募，均不能单独作为成功。',12)
    r.table([['证据层','本次实际状态'],['已完成的系统筛查','60条件×2噪声×20秒，共120条；全部完成，无工程失败、无物理runaway。3个布局、14类可调量，包含19个单参数探针及各布局基线。'],['原120条的训练集合','排除启动期及不完整边界后，有7735个检测事件，其中485个符合原primary规则；7250个因250 ms窗口重叠排除。只有1条运行达到N≥16，没有一个条件的两条噪声均可评分。'],['实际传播','扩大范围、增强局部递归或离核输出能提高SCL参与，但存在ICL缺失、过度招募或两类标签都由左核较早活动的问题；目前不接受完整恢复。'],['夜间继续',f"状态 {state['status']}；当前阶段完成 {state.get('complete',0)}/{state.get('total',0)}。新增上限48条，原120加新增不超过168；08:12停止新派发，最迟10:12停止物理执行。"]],widths=[105,406],size=10.4)
    r.para('本版正文只把严格限核随机输入版本称为当前。旧全场Poisson、混合阈值、外加慢I状态、C/Z分支均保留原身份；旧报告v7和v6保持可追溯，不套用新模型解释。',10.3,BLUE)
    r.start('当前模型：先把输入、core和连接说清楚')
    r.table([['对象','本轮实际执行定义'],['双核几何','右核固定；左核采用端点原位、上移4.5mm、靠近上部SCL三种布局。几何来源是患者先验，不能再作为独立验证。'],['局部易激性','E阈值 Vth=18−a×max(18−Vraw,0)，下界11mV；核内每个E相对背景都不升阈值。核外和I阈值18mV；实际降阈值细胞、未变细胞及半径分别记录。'],['随机输入','截至中点批为共享OU＋独立Poisson仅在两核E；末批另改变两核OU相关性。核外E与全部I接收确定期望到达，经原AMPA滤波。无全场空间OU、无定向kick、无标签刺激；外加慢I及Z/M关闭。'],['连接参数','分开调整核内EE、核→外EE、外→外EE、同核EI/IE/II权重；实际入度、EE横向/纵向范围和方向另列。权重不强制总量归一化。'],['几何与图身份','单纯权重/阈值干预保持图和时延；半径改变核成员及输入支持；入度/范围/方向会改变实际邻接，仅共享拓扑随机身份。'],['时间与随机性','GABA衰减固定18ms。配对噪声847101/847102；相同seed不保证参数改变后Poisson逐时刻创新相同。事件是运行内样本，不是独立网络。']],widths=[100,411],size=10.1)
    r.para('评分采用已冻结的去自配对分布统计量及联合参与结构；它不是传播机制证明。原合格窗口规则保持不变，未把全部检测诊断重新包装成正式患者匹配分数。',10.5,BLUE)
    for fname,title,body in [
        ('parameter_recruitment_tradeoff','参数 → 招募：已有具体作用，也存在取舍','固定上移布局、同一拓扑，四列一次只改变一个参数。扩大左核增加SCL却减少ICL，并使TA标签占比接近1；提高核内EE或减弱核内EI也偏向TA。横向范围扩大可以保留两杆，但模式占比对噪声仍敏感。'),
        ('core_timing_mode_separation','两种标签是否保留不同的两核先后关系？','原位的TA/TB更接近相反的核活动先后，但SCL缺失。提高SCL招募的两个条件中，TA与TB标签往往都由左核先积累活动。这里的10%质量时间不等于因果起源，必须结合完整原生场。')]:
        r.start(title,True);r.figure(NIGHT/f'initial_review/figures/{fname}.png',fname,330);r.para(body,10.5)
        r.para('模型为全部检测的开发诊断；患者参考来自原合格FIT表，资格不同，不能作正式分布验收。圆/方或实/虚线分别保留两条噪声，未合并为一个均值掩盖不一致。',10,BLUE)
    r.start('先修正比较层：Fig2C主图、全触点QC不是同一对象')
    r.para('Fig2C主图使用本次事件实际参与且质心可用的触点，左侧是冻结的真实STFT幅度。此前黑底对照是同一患者源事件的全触点HFO包络，连未参与触点也保留；这是重要QC，但不能把其中最亮位置当成本事件传播起点。',11)
    r.para('具体例子：患者TB937的SCL8/9不参与该群体事件，虽然它们的原始包络在窗口早期很亮。参与触点中，SCL6/7质心约在最早参与触点之后50/55ms，ICL9约76ms，而ICL11约51ms。此前关于SCL早晚的目视判断必须按这个事件层重新检查，不能用显示层混淆解释模型成败。',11)
    r.para('本版同时提供真实STFT—模型包络、参与定义匹配的包络图、全触点QC和完整模型原生场。患者STFT幅度、患者HFO包络与模型发放密度是不同信号；各图不使用时间拉伸。Fig2C患者例经过方向可读性筛选；模型例取自身模式均值附近，两者都不能替代全分布检验。',10.5,BLUE)
    black=SEARCH/'rapid_audit_20260911/all_condition_time_review/figures'
    for cid,title in [('endpoint__baseline','端点原位'),('up4p5__EE_core_to_out_scale_1.25','左核上移＋离核输出增强'),('near_upper__EE_kernel_perp_scale_1.5','靠近上部SCL＋横向范围扩大')]:
        r.start(title+'：全触点窗口QC',True);r.figure(black/f'{cid}_event_scale.png','black_'+cid,335)
        r.para('旧图标题“患者Fig2C”指源事件包络，不是原Fig2C频谱，也不是其participant-only主图。固定TA/TB在左；模型两条噪声在右。模型例最接近自身模式均值，缺少该类时保留空格；全部检测诊断不替换原训练资格。',10.1)
        r.para('纵轴保留真实触点顺序，横轴250ms不拉伸；患者HFO包络与模型发放密度不是同一种物理信号，更不是频谱等价。观察参与缺口、两杆先后与并行活动，不只看颜色形状。',10.1,BLUE)
    r.start('夜间第一批：针对残差做组合，不冒充已有DE优化')
    r.para('原计划DE没有可评分的双噪声父代，因此本次明确改为残差指导的固定组合，派发前冻结实际参数。第一批8条件×2噪声×20秒，既有直接出发点可配对；不把组合效应归因于单一参数。',11.2)
    data=[['直接出发点','新增改变']]
    for c in np['wave1']['candidates']:
        parent='上移＋离核EE×1.25' if c['parent_id'].startswith('up4p5') else '近SCL＋横向范围×1.5'
        mapping={'EE_kernel_perp_scale':'EE横向倍数','radius_A_mm':'左核半径mm','depth_A_scale':'左核降幅倍数','depth_B_scale':'右核降幅倍数','EI_same_core_scale':'核内E→I倍数','IE_same_core_scale':'核内I→E倍数','EE_same_core_scale':'核内EE倍数','EE_core_to_out_scale':'离核EE倍数'}
        data.append([parent,'；'.join(f'{mapping.get(k,k)}={v:g}' for k,v in c['changed_parameters'].items())])
    r.table(data,widths=[160,351],size=9.8)
    r.para('随后依据真实时序决定剩余采样和新图/新噪声确认；当前分配及任何有依据的修订写入执行记录。延长运行增加证据支持，不假定它能改变错误的空间路径。',10.5,BLUE)
    long_plan=NIGHT/'long_selection.json'
    if long_plan.exists():
        r.source(long_plan,'long_selection');r.source(NIGHT/'tonic_canary_audit.json','tonic_canary')
        r.start('第二阶段：减弱递归后，能否同时保留两类传播？')
        r.para('第一批16条全部完成：1125个检测、120个原primary。横向范围扩大并减弱同核EE至0.75，使两核完整平均率降至约5Hz；两噪声有21/22个primary，是首个均能评分的条件。但TA仅1/2个，正式Loff约16.34/13.19仍不理想，不能把“可评分”写成“拟合好”。',11)
        r.table([['布局','每布局4个60秒条件'],['靠近上部SCL','同核EE0.75；EE0.85；EE0.85＋左核降幅1.15；EE0.85＋核内输入均值0.95'],['端点原位上移3mm','完全相同的四个参数条件，右核不动；每条件两条配对噪声']],widths=[120,391],size=10.5)
        r.para('共16条，替代原16条单纯长重演，不增加48条新增上限。输入均值通过独立执行器只改变E-core的常数到达率，OU规律和核外确定均值不变；Poisson方差随均值自然改变。scale1与原执行器62个数组一致，3条500ms应用检查通过。这些是实施验证，不是传播恢复证据。',10.5,BLUE)
    final_a=NIGHT/'final_A_selection.json'
    if final_a.exists():
        r.source(final_a,'final_A_frozen_selection');r.start('长记录之后：优先继续寻找传播基底')
        r.para('第二批16条已全部完成，近SCL与端点上移3mm两个位置都没有完整恢复患者两种传播。在全检测诊断中，增强EE对TA比例的影响随位置改变而反向；原primary在部分条件只有4–10例，不能把全检测的交互直接推广到正式合格集合。中间位置EE0.75虽补齐较多参与触点，SCL相对ICL仍系统性偏晚。',11)
        r.table([['左核新位置','先运行的4个条件，每条件两条原噪声、60秒'],['两个已测位置的几何中点：(4.9557, 12.8515) mm','EE0.75；EE0.75＋左核降幅1.15；EE0.85＋输入均值0.95；EE0.85＋输入均值0.95＋左核降幅1.15']],widths=[190,321],size=10.3)
        r.para('最后16条先分配8条局部精调，另8条等待完整审阅后决定。出现实质传播改善再用于新图重演，否则继续有边界的针对性组合；不自动将错误传播做成“最佳候选确认”。这是预先固定的局部参数提案，不是DE。EE0.75与EE0.85加均值0.95是两套设置，不能将它们的差异归因于一个参数。',10.5,BLUE)
    if (NIGHT/'final_A_analysis_complete.json').exists():
        r.source(NIGHT/'phase3_review.md','midpoint_scientific_review')
        r.start('中点位置完整结果：较低分数仍不等于完整传播')
        r.para('8条60秒运行完整完成，无工程失败或物理runaway；启动排除后1035个全检测、651个原合格孤立窗，其中TA45、TB606。EE0.75背景仍以SCL晚约100ms的TA为主，位置插值没有消除时序残差。',11)
        r.para('EE0.85、输入均值0.95、左核降幅1.15的联合训练分数6.563/7.287，低于此前上移3mm、EE0.75的7.508/7.309。改善主要来自特征均值A项；但其TA典型例缺失ICL左端与SCL8/9，上部SCL平均参与仅0.444/0.375。',11)
        r.para('杆间时差按每次实际参与触点取中位数。这个候选的TA杆间中位虽然转为负值，同时却改变了参与集合，不能据此单独宣称时序恢复。逐触点和完整图形比较仍是决定性证据，本轮不提名该条件为已恢复基底。',10.5,BLUE)
    final_b=NIGHT/'final_B_selection.json'
    if final_b.exists():
        r.source(final_b,'final_B_frozen_selection');r.source(NIGHT/'core_ou_correlation_canary_audit.json','core_OU_correlation_canary')
        r.start('末批8条：共同输入是否限制了不同传播的出现机会？')
        r.para('回到同一上移3mm几何的两个已完成背景：EE0.75，以及EE0.85加核内输入均值0.95。前者两噪声均出现过较相容完整TA，后者出现过较相容TB个例；这些是事后能力诊断，不是恢复验收或最佳分数提名。',11)
        r.table([['改变','固定内容'],['两核慢OU输入相关系数 1 → 0.5 / 0','每核边际均值、方差和150ms时间常数保持相同规律；核外无随机输入。图、时延、阈值和几何相对各自父条件相同。'],['2背景×2相关系数×2噪声×60秒','ρ=1直接使用完整父条件。共有8条新运行，不另做新图确认，不增加48条新增上限。']],widths=[190,321],size=10.4)
        r.para('若只增加某类标签而不改善其条件路径，只能接受出现机会/组成的改变；若条件路径和分布也改善，才形成待独立确认的候选。相关性改变后的实际Poisson创新不保证和父条件逐步相同。3条500ms工程检查已通过，ρ=1与原执行器62/62数组一致，不能把检查通过视为科研阳性。',10.4,BLUE)
        image=NIGHT/'core_ou_effect_final_B/figures/input_to_native_core_response.png'
        if image.exists():
            r.source(image.parent.parent/'manifest.json','core_OU_native_response')
            r.start('输入相关性 → 完整原生核心活动',True);r.figure(image,'core_OU_native_response',365)
            r.para('左、中图核对实际输入关联和逐核边际散布，右图使用全部原生核心发放，无电极读出或事件筛选。相关系数为运行内描述，有限记录不要求精确等于理论值；时间互相关不是因果路径或双稳态证明。',10.5,BLUE)
    summary=NIGHT/'scientific_review_summary.json'
    if summary.exists():
        r.source(summary,'final_scientific_judgment')
        item=read(summary)
        r.start('本夜最终审阅：能接受什么，仍缺什么')
        for para in item['paragraphs']:r.para(para,10.8)
        r.para('参数效应均以实际运行及其直接对照为单位；本轮未进行新拓扑确认。图件已由Agent检查，用户人工验收仍待完成。',10.2,BLUE)
    for relative,title,body in [
        ('core_ou_propagation_response/figures/core_input_correlation_to_propagation.png','输入相关性：事件出现与条件传播分开看','两个背景的每个点是一条60秒原合格集合。输入共同程度降低，在EE0.85/均值0.95背景增加TA支持，但TA的SCL仍偏晚、TB的左端顺序仍偏向患者分布的另一侧。中间误差棒及患者灰带是事件5–95%范围，不是置信区间。'),
        ('parameter_response_summary/figures/primary_parameter_response_summary.png','单参数响应：正式合格事件与完整核心活动','三列分别改变核内EE、左核阈值降幅、核内输入均值，颜色标识参数，圆/方标识两条噪声。EE列同时保留近SCL与上移3mm两种布局；其他两列固定近SCL布局。没有TA时，其条件参与率留空；原生核心率使用完整轨迹，不经过事件筛选。'),
        ('parameter_response_summary/figures/all_detected_parameter_response_summary.png','单参数响应：全检测开发诊断','此页保留所有检测，不受相邻250ms群体窗口重叠排除；患者FIT资格不同，因此灰色患者参考只作描述。两种资格下的响应需要同时读，不能仅选择趋势最清楚的一页作为正式恢复结论。'),
        ('core_to_contact_timing/figures/working_points_native_to_contact_timing.png','两核先后已能不同，接触招募仍可不相容','横轴是右核相对左核的窗口内10%累计发放时间，纵轴是参与SCL相对参与ICL的质心中位差。每点一个合格事件，颜色组织TA/TB，形状区分噪声；单杆事件不能计算纵轴，其数量另行保留。核心时间和接触时间不是同一种观测，也都不是因果起源定位。'),
        ('early_contact_geometry/figures/current_847101_early_contact_probability.png','最早三个参与触点：空间身份也必须匹配','每事件取最早三个参与触点质心，着色为进入该集合的频率，各触点概率和为3；保持真实电极位置，不插值。模型可以改善早期触点的时间跨度，同时仍选中错误的触点。此页不是神经元源定位或新的训练约束。'),
        ('core_prior_extent_audit/figures/endpoint_cloud_and_core_extent.png','几何先验保留了中心，却未保留相同程度的空间范围','已采用的左侧三个端点摘要分散约11mm，右侧三个端点集中在约3mm范围；相同1.75mm半径的圆对两者是很不同的近似。电极无需位于core内，这也不证明患者有拉长的神经元灶；它提示应区分core范围与向外连接范围，不能只移动同一个小圆。')]:
        image=NIGHT/relative
        if image.exists():
            r.start(title,True);r.figure(image,'new_diagnostic_'+image.stem,355);r.para(body,10.2)
    main_review=NIGHT/'main_review_wave1'
    if (main_review/'manifest.json').exists():
        r.source(main_review/'manifest.json','main_layout_and_patient_spectra')
        for cid,title in [('recovery_up_out_transverse','离核输出增强与横向范围组合'),('recovery_upper_wide_recurrence','减弱核内EE：TB顺序与罕见TA分别看')]:
            r.start(title+'：同一网络',True);r.figure(main_review/f'figures/{cid}@2511_847101_same_network.png','main_'+cid,330)
            r.para('四列为实际阈值降低场、两类各自均值附近的单事件原生时间场、完整连续电极读出。原生场为1mm网格内2ms活动神经元数的累计10%时间，未插值；不是因果起源。完整原生动画和全触点QC另存。',10.3,BLUE)
            r.start(title+'：实际患者STFT与模型包络',True);r.figure(fixed_shaft_figure(main_review/f'figures/{cid}@2511_patient_spectra_model_envelopes.png'),'spectra_'+cid,330)
            r.para('患者左列保留Fig2C冻结STFT、cell edges及质心；右两列为模型发放密度包络，未冒称HFO频谱。所有面板固定SCL9–6、ICL11–1这15行，未参与触点保留原位，杆间不连质心线。仅平移时间零点，完整全触点信号另见QC。',10.2)
            if cid.endswith('recurrence'):
                r.para('减弱EE后的TB中SCL不再统一早于ICL，但ICL末端顺序仍不等同患者示例。TA仅1/2例，其中噪声847101事件9的SCL9在窗口两端有两次局部burst；计算质心处实际包络为零，不能把红色质心折线解释为持续传播。',10.1,BLUE)
    burst=NIGHT/'local_burst_audit_wave1/audit.json'
    if burst.exists():
        r.source(burst,'local_burst_observer_audit');r.source(burst.with_name('summary.csv'),'local_burst_summary')
        r.start('窗口与局部事件：仍有一条具体的观测缺口')
        r.para('原primary排除相互重叠的群体观察窗，并不保证每个触点在一个窗里只有一次局部活动。冻结模型观察器对窗口内全部正包络质量取质心；患者Fig2C质心则来自主频谱增强连通区。两者可以在多个局部burst时给出不同含义。',11)
        r.para('已对所有首批及直接对照的保存包络重放冻结观察器，事件与参与mask不变，质心差异仅为保存float32所致的极小数值差。额外报告每事件多burst触点比例、质心处包络高度和不直接贡献群体合格段的触点；这是只读诊断，不是本轮新增训练惩罚或患者校准的资格门。',11)
        r.para('这项发现不意味着所有事件都在“作弊”，也不允许单凭修读出宣布物理传播恢复。它限定了当前标签和损失能支持的结论：必须联合看真实接触活动、原生场、参与分布和模式支持量。',10.5,BLUE)
    curated=NIGHT/'curated_review_selection.json'
    if curated.exists():
        r.source(curated,'curated_scientific_examples')
        for item in read(curated)['items']:
            folder=NIGHT/('main_review_'+item['phase']+('_primary' if item['population']=='primary' else ''))/'figures'
            stem=item['candidate']+'@'+str(item['topology'])
            image=fixed_shaft_figure(folder/(stem+'_patient_spectra_model_envelopes.png'))
            if not image.exists():continue
            r.start(item['title']+'：患者真实STFT对照',True)
            r.figure(image,'curated_spectra_'+stem,345)
            r.para(item['interpretation'],10.4)
            r.para('患者两例沿用Fig2C；模型各例最接近该运行自身模式均值，未按患者相似度挑选。模型信号是发放密度包络，不冒称HFO频谱；共同实际毫秒轴不拉伸。标签不等于患者相容路径。',10.2,BLUE)
            image=folder/(stem+'_'+str(item['seed'])+'_same_network.png')
            if image.exists():
                r.start(item['title']+'：同一网络与完整记录',True)
                r.figure(image,'curated_native_'+stem,345)
                r.para('左侧是实际仅降阈值场及真实电极位置，中间为对应两事件的原生累计活动时间场，右侧保留完整连续电极读出。累计时间不是因果起燃位置；完整原生动画与局部帧检查另存。',10.4,BLUE)
    for phase in ['wave1','long','final_A','final_B','confirmation']:
        A=NIGHT/f'analysis_{phase}'
        if not (A/'review_state.json').exists():continue
        s=read(A/'review_state.json');r.source(A/'review_state.json','review_'+phase)
        if (A/'run_observations.csv').exists():r.source(A/'run_observations.csv','observations_'+phase)
        fig=A/'figures/participation_and_core_timing.png'
        if fig.exists():
            phase_title=PHASE_TITLES[phase]
            r.start(f'{phase_title}：招募与两核时间关系',True);r.figure(fig,'night_'+phase,345)
            r.para('各条件、各拓扑和噪声分别保留；图中事件数是实际观测量。局部宽度、成对顺序、杆间质心时差、两核质量时间和正式评分同时落盘，不能用其中一项替代完整传播。',10.1,BLUE)
    paired=NIGHT/'paired_input_streams_wave1.json'
    if paired.exists():
        r.source(paired,'verified_stepwise_input_pairs');r.source(NIGHT/'length_prefix_replay_audit.json','long_short_prefix_replay')
        r.start('哪些参数比较真的保留了同一随机输入？')
        r.para('首批16组候选与直接出发点的比较，有14组每个0.1ms步的外部输入完全相同；每秒摘要包含OU值与所有40000个神经元的外部到达。只有扩大左核的两组改变了输入支持，因此不能称为相同逐步噪声。均值0.95条件也不能据相同seed宣称输入相同。',11)
        r.para('这里初始膜电位统一为V_reset；动力学seed改变输入随机流，并未同时抽样一套随机初始电压。EE0.75的两条60秒记录与对应20秒记录各有44项静态/动态前缀核查完全一致，包络仅排除短记录末端16ms卷积边界。因此延长记录增加支持量，不会重写已有的前20秒传播。',11)
        r.para('这些核查加强的是配对参数解释与复现范围，不是患者传播的阳性证据；也不推广到未核查的历史执行器。',10.5,BLUE)
    for phase in ['long','final_A','final_B','confirmation']:
        A=NIGHT/f'analysis_{phase}'
        if not (A/'run_observations.csv').exists():continue
        rr=rows(A/'run_observations.csv')
        spec=read(NIGHT/f'{phase}_units.json');allowed={(cid,str(t),str(s)) for cid,t,s in spec['units']}
        actual=[x for x in rr if (x['base_id'],x['topology_seed'],x['seed']) in allowed and x['layer']=='primary' and x['mode']=='ALL']
        if not actual:continue
        phase_label=PHASE_TITLES[phase]
        r.start(f'{phase_label}：实际事件支持，逐运行列出')
        table=[['条件 / 图 / 噪声','原primary TA / TB','全检测 TA / TB']]
        for x in actual:
            key=(x['candidate'],x['seed'])
            def nn(layer,mode):
                q=next((z for z in rr if (z['candidate'],z['seed'])==key and z['layer']==layer and z['mode']==mode),None)
                return q['n'] if q else '缺失'
            label=x['display_name']
            table.append([f"{label} / {x['topology_seed']} / {x['seed']}",nn('primary','TA')+' / '+nn('primary','TB'),nn('all_detected','TA')+' / '+nn('all_detected','TB')])
        r.table(table,widths=[285,113,113],size=8.5)
        r.para('仅列有完整输出且已分析的运行；尚未完成不写成“未产生模式”。原primary和全部检测资格不同，后者只作开发诊断。更多事件或更均衡的标签，均不直接代表患者条件传播分布恢复。',10,BLUE)
        pp=NIGHT/f'parameter_response_{phase}'
        if (pp/'manifest.json').exists():
            r.source(pp/'manifest.json','paired_response_'+phase);r.source(pp/'paired_changes.csv','paired_response_values_'+phase)
            for mode in ['ALL','TA','TB']:
                image=pp/f'figures/{mode}_all_detected_paired_changes.png'
                if not image.exists():continue
                mode_label='不区分模式' if mode=='ALL' else mode
                r.start(f'{phase_label}：{mode_label} 参数改变 → 观测差值',True);r.figure(image,'paired_'+phase+'_'+mode,355)
                r.para('每行只改变写明的一个因素，后者减前者；颜色区分EE、阈值、均值、位置及两核慢输入相关性，符号区分实际拓扑/噪声重演。参与概率上升本身不等于接近患者，需同时看参与误差、顺序误差和实际事件支持。这里为全部检测开发层；原primary图与所有数值另存。',10.2,BLUE)
            image=pp/'figures/native_group_rate_paired_changes.png'
            if image.exists():
                r.start(f'{phase_label}：局部参数能否重排整张网络的活动？',True)
                r.figure(image,'native_rates_'+phase,375)
                r.para('这张图直接使用全部群体发放，不经电极读出或事件筛选。只降低左核阈值时，右核也可能有明显响应；同一输入和固定图的核查可以排除误把参数应用到右核，但不能仅据群体平均率确定作用经过哪条回路。',10.5,BLUE)
    native=NIGHT/'core_timing_long/figures/refine_near_EE085_A115_2511_847102.png'
    if native.exists():
        r.start('长记录暴露的限制：标签变化伴随核心时间关系变化',True)
        r.figure(native,'core_timing_A115',365)
        r.para('靠近SCL、EE0.85、左核阈值降幅1.15的这一噪声重演，两核约每213ms出现一个群体峰。前50秒左右以TA标签为主，后段更多TB，同时核心周期坐标差发生变化。这是实际群体时序关联，不是已证明的双稳态或核心因果驱动。',10.4)
        r.para('完整核内发放先定义群体峰，再叠加事件标签；没有按接触事件筛选核心轨迹。周期坐标依赖峰识别，漏峰会改变它；参数敏感性和逐运行结果保留。不能把这类标签改变直接写成患者两条路径恢复。',10.2,BLUE)
    residual=NIGHT/'initial_review/TB_terminal_order_residual.json'
    if residual.exists():
        r.source(residual,'TB_terminal_order_residual')
        r.start('为什么仍不接受TB恢复：完整分布中的反证')
        r.para('患者FIT中ICL11和ICL9共同参与的4879个TB事件，约69%为ICL9较晚；时差ICL9−ICL11中位约+6.0ms，5–95%约−14.3到+32.0ms。第一批减弱核内EE的两条噪声各20个primary TB中，均为ICL9较早，中位约−14.2/−14.3ms。模型主要落在患者分布的一侧，问题不只是一张示例不像。',11)
        r.para('这40个模型TB事件的两个触点都只有一次局部检测，质心处包络有实际活动，并直接贡献群体合格段。因此，这一具体残差不能由“两个burst取了中间质心”解释。患者存在方差，不要求每次都是同一个方向；必须恢复相容的条件分布。',11)
        r.para('逐触点均值、中位数、样本方差、5–95%范围及全部触点对顺序矩阵均已保存，primary与开发层分开。未定义跨维度权重的“总体方差恢复率”。',10.5,BLUE)
    block=NIGHT/'patient_block_context/figures/patient_block_context.png'
    if block.exists():
        r.source(block.parent.parent/'manifest.json','patient_block_context')
        r.start('患者本身有块间变化，但不能解释全部残差',True)
        r.figure(block,'patient_block_context',360)
        r.para('患者28个原FIT数据块分别保留，不把全部事件当作一批独立重复。TA比例确有变化；但TA杆间时差在绝大多数块内仍为SCL较早，TB中心接近零。图中模型是上移3mm、EE0.75的两条60秒合格集合，其SCL系统性偏晚。',10.5)
        r.para('横轴是实际事件数，竖线是事件5–95%范围，不是均值置信区间；未匹配患者块和模型的记录时长，也未把经验块范围设成新验收阈值。该图补充来源块背景，不替代逐触点联合传播检验。',10.2,BLUE)
    capability=NIGHT/'capacity_examples_long/refine_mid_EE075'
    if (capability/'selection.json').exists():
        r.source(capability/'selection.json','posthoc_capability_selection')
        r.start('少见相容个例：能出现，与常见分布恢复是两回事',True)
        r.figure(capability/'figures/patient_compatible_extremes.png','posthoc_capacity',345)
        r.para('在同一上移3mm、EE0.75条件内，每条噪声每类选择患者同类邻域距离最小的原合格事件。TA两例均15触点参与，SCL相对ICL约早24.5/26.4ms，且各参与触点直接贡献群体合格段，没有多burst触点。它们证明曾出现较相容时序，不能替换常见TA过晚和TB末端顺序偏差。',10.3)
        r.para('本页明确事后择优，只作已观察能力诊断；常规主图仍使用自身模式均值附近事件，完整逐帧原生GIF与时间分箱另存，不把最佳个例当作分布恢复。',10.2,BLUE)
    window=NIGHT/'core_window_alias_long/figures/core_window_847101.png'
    if window.exists():
        r.start('完整核心时序：不要把两个活动峰直接当成两次起源',True)
        r.figure(window,'core_window_alias',350)
        r.para('同一运行内，常见TA例先出现ICL传播，SCL到窗口末段才被招募，并伴随左核再次活动；事后较相容TA例则两杆在较接近时间被招募。再次核心活动可能包含返回传播，不能只凭两峰判定独立起始或窗口拼接。原生逐帧场与完整包络共同限定机制解释。',10.3,BLUE)
    final=NIGHT/'scientific_review.md'
    if final.exists():
        r.source(final,'night_scientific_review');r.start('本轮科学审阅与下一步边界')
        for para in [x.strip() for x in final.read_text().split('\n\n') if x.strip() and not x.lstrip().startswith(('#','|','-'))][:9]:
            r.para(para,10.1)
            if r.y<140:break
    r.start('完整参数响应附录的读法')
    r.para('后续每页一个参数，列为三种布局；黑色不分TA/TB，红TA，蓝TB；实线/虚线为两条噪声，点线为患者FIT参考。原primary事件数很少的点只能描述，空值不能填成零。全部检测版本、逐触点表、完整60条件黑底页和原生多事件GIF另存于同一分析目录。',11.2)
    r.para('本报告正文及附录保留阴性证据。未通过患者传播的人工验收前，不冻结工作点，不进入Fig5，不把图件排版完成称为科研目标已达成。历史C/Z图及方程随后原样保留，并继续标注其独立模型身份。',11.2,BLUE)
    for fam,label in [('EE_same_core_scale','同核E→E'),('EE_core_to_out_scale','核→外E→E'),('EE_out_to_out_scale','外→外E→E'),('EI_same_core_scale','核内E→I'),('IE_same_core_scale','核内I→E'),('II_same_core_scale','核内I→I'),('depth_A_scale','左核降阈值幅度'),('depth_B_scale','右核降阈值幅度'),('radius_A_mm','左核半径'),('radius_B_mm','右核半径'),('EE_core_to_out_degree_scale','核→外实际入度'),('EE_kernel_perp_scale','EE横向范围'),('EE_kernel_parallel_scale','EE纵向范围'),('EE_angle_offset_deg','EE连接轴方向')]:
        r.start(label+'：原合格事件的配对响应');r.figure(SEARCH/f'analysis/figures/response_{fam}.png','response_'+fam,625);r.para('保持原primary规则；每个点旁标实际N。低支持量不视为缺乏某模式的机制证明。',9.7,BLUE)
    pdf,pages=r.finish();preview=directory/'figures';preview.mkdir()
    result=subprocess.run(['pdftoppm','-scale-to','850','-png',str(pdf),str(preview/'page')],capture_output=True,text=True,check=True)
    images=sorted(preview.glob('page-*.png'));assert len(images)==pages
    for file in images:
        with Image.open(file) as im:im.load()
    (preview/'README.md').write_text('\n\n'.join(f'### {file.name}\n\n本构建PDF第{i+1}页的同源预览，所有图源见source_manifest。**关注点**：图文与物理版本一致性；解码通过不等于科学验收。' for i,file in enumerate(images)))
    info=dict(generated_at=now.isoformat(),report=str(OUT/pdf.name),snapshot=str(pdf),pages=pages,sha256=sha(pdf),screen_runs=review['runs'],night_state=state,user_scientific_acceptance=False,decoded_pages=len(images),renderer_warnings=result.stderr)
    (directory/'delivery_checks.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
    temp=OUT/'report.tmp.pdf';shutil.copy2(pdf,temp);os.replace(temp,OUT/pdf.name)
    temp=OUT/'current.tmp.json';temp.write_text(json.dumps(info,ensure_ascii=False,indent=2));os.replace(temp,OUT/'current.json')
    print(json.dumps(info,ensure_ascii=False))

if __name__=='__main__':build()
