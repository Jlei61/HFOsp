"""Chinese scientific review rendered from the repaired evidence summary."""
from __future__ import annotations

import json
from pathlib import Path


def render(summary, technical=False):
    closure = summary['scientific_closure']
    short = lambda value: str(value).replace('epilepsiae_', 'E')
    names = lambda key: '、'.join(short(x) for x in closure.get(key, [])) or '无'
    stage = '修复已验收' if summary.get('status') == 'REVIEW_REPAIR_COMPLETE' else '阶段性复核，尚未最终验收'
    root = Path(summary['output_paths']['summary_json']).parents[1]
    lineage = json.loads(Path(summary['state_lineage']['path']).read_text())
    diagnostics = summary.get('repair_diagnostics', {})
    h1_limited = [row for row in diagnostics.get('h1_stage_qualification', []) if row.get('training_budget_exhausted')]
    h2a_unqualified = [row for row in diagnostics.get('h2a_stage_qualification', []) if not row['qualified']]
    lines = ['# v0.3.8 科学与工程审阅修复' + ('：技术报告' if technical else '：白话报告'), '',
             f'**{stage}。原结果保留，本报告以修复目录的机器汇总为依据。**', '',
             '## 1. 一句话判断', '',
             '**已有患者内预测线索和真实长历史梯度；尚未建立一个由间期事件学习、具有非线性演化、'
             '能统一解释事件形态与发作的病理状态。六小时双流模型的动态收益尤其不能直接归给事件分支。**', '',
             '原报告把不同 seed 的通过票拼成完整证据，并把不同模型、初始化状态和已学习状态混在一起。'
             '修复后必须逐患者、逐 seed、逐 checkpoint、逐预测端点判断。', '',
             '## 2. 完成程度与原始假设', '',
             '**审阅修复工作包完成；原始核心科学闭环尚未建立。** '
             '局部预测线索、尚未有效检验、优化受限和不可估分别报告，不再使用没有统一含义的总分。', '',
             '|原始问题|目前可说的话|主要边界|', '|---|---|---|',
             '|能否从间期事件学习动态状态|E1096短尺度有同seed完整预测对照线索；E1125六小时双流模型有完整模型层面的收益|E1096严格特征复现未过；E1125事件分支常数反而更好；完整事件状态资格尚未建立|',
             '|是否学到非线性演化|尚未建立|时间常数固定，主要学习线性输入矩阵；归一化读出的非线性不能替代该检验|',
             f"|是否表达多种病理信息|H1 可检查负荷、community、coupling、repertoire 和合并 mark；完整 H2a 多端点候选：{names('multi_pathology_h2a_subjects')}|预测多个已训练目标不等于跨任务或跨患者泛化；须另查上游是否真的学习|",
             f"|是否有发作前特异性|通过修复后全部资格的重复发作候选：{names('repeatable_seizure_transfer_subjects')}|少量发作、聚集与分区问题仍限制解释；不能据此宣称生物学阴性|",
             '|每次事件是否改变生理状态|H3 未建立|模型中每次事件触发更新是结构假设；梯度与单分支历史删除不是生理干预|', '',
             '## 3. P0 / P1关键问题及修复', '',
             '**P0：同 seed 交集已修复。** E922 原 event/grid 的三个对照各自可以有3/5为正，'
             '但同一 seed 三项同时通过只有1/5。因此撤回原“可重复发作候选”的表述。', '',
             '**P1：端点身份也固定。** 多端点候选必须由同一对非同义端点在同一批至少三个seed上通过。'
             '每个seed各自任取两项、但在seed间更换组合，只保留为诊断，不能写成固定多端点复现。', '',
             '**P0：后缀评分已修复。** 给定前两组触点后，只评分之后的预测，保留前缀末端是否 STOP 的判断。'
             '正确、常数、错时和 B_mark 使用相同后缀事件；改善已经给定的前缀不能再制造分叉增益。', '',
             '**P0：训练和来源分开核验。** 165个模型单元中，选中参数分类为：'
             + '；'.join(f'{key}={value}' for key, value in sorted(lineage['state_classes'].items())) + '。'
             '这些是模型单元数，不是独立患者数。E253 event/grid 上游保留初始化、E1077主要学习背景等情况会单列，'
             '不能用下游 adapter 的学习替代上游状态学习。', '',
             '**P1：补齐明确不足的承重训练。** E1096六个随机对照在相同学习率和初始化下增加上限后均按原耐心停止；'
             'E253的状态头和静态对照也按原日志补预算。登记为优化受限的承重对照会阻止对应科学候选放行，'
             '不会把“预算用完”自动当作没有效应。', '',
             '全队列收尾另外登记了中尺度22个H2a单元，以及H1剩余15条random/current-background触顶路径。'
             '只补相应承重拟合，不根据SELECTION调学习率。更新后的背景基线是独立对照，原科学模型保持冻结；'
             '若原对照分数不能重放则标来源受限，再次触顶则标优化受限。两者都不能当科学阴性。', '',
             '**P1：预处理来源已加锁。** 新缓存绑定实际使用的事件、波形和词典，保存标准化、残差化及 PCA 变换。'
             '旧缓存及旧结果保留。报告分别记录端点分数是否复现与特征是否达到严格数值容差；'
             '保存一个新源码快照不能补回原先未保存的预处理历史。', '',
             '## 4. 科学证据与归因边界', '',
             '**E1096的短尺度预测线索保留。** event的0.5小时和grid的0.5/2小时均有同样三个seed通过原H1完整对照。'
             '这些seed的输入矩阵确有更新，新预处理的端点分数也在原容差内；但严格特征重放未通过，因此不写成来源链全部合格的结论。', '']
    for row in summary.get('same_model_h1_credit_multitarget', {}).get('rows', []):
        if row['same_model_functional_state_candidate']:
            lines += [f"**{short(row['subject'])}，dual，{row['horizon_seconds']/3600:g}小时未来目标："
                      f"{row['joint_evidence']['n_joint_positive']}/5 seeds 同时具有完整双流H1收益、已更新事件输入矩阵、相同权重的长程credit和固定的两类非计数端点。** "
                      'mixture与embedding合并算一个repertoire组。这里连接的是聚合统计预测，不能把它写成已经通过contact decoder或发作迁移。', '']
    lines += ['**但E1125五个seed的事件分支固定为FIT均值后，六小时预测反而更好。** '
              '事件分支对正确时刻的偏好为5/5，但胜过自身常数为0/5；背景分支胜过自身常数5/5、胜过自身错时4/5。'
              '这说明完整双流模型的动态收益不能直接归给事件分支；也不能据此认定背景已学到生理机制。'
              '五个seed中，学到的背景分支同样未胜过匹配随机背景组件。完整分支对照已覆盖55个dual checkpoint。', '',
              '梯度检验追溯到真实旧事件，且检查了选中输入矩阵是否更新。'
              '历史删除还说明“有梯度”与“删除一定使预测变差”是两回事；正负变化均保留。'
              '这些分析支持计算路径的可达性和敏感性，不发现新的生理时间常数，也不证明 IED 的因果作用。', '',
              'H1 的多端点在训练时已经作为目标。留出期表现可支持患者内预测价值；'
              '它不能单独证明未训练病理表型的迁移。合并 rich mark 的增益，也不能拆成频带、延迟、waveform 三个独立阳性。', '',
              '本轮完整重放的实际结果：', '',
              '|输入边界|已完成模型|端点分数在原容差内|同时满足严格特征与分数容差|', '|---|---:|---:|---:|']
    for name, row in summary.get('repair_diagnostics', {}).get('preprocessing_replay', {}).items():
        lines.append(f"|{name}|{row['completed_models']}|{row['rebuilt_endpoint_score_pass']}|{row['strict_feature_and_endpoint_pass']}|")
    lines += ['', '特征检查比端点评分检查更严格；两者分别记录。未达到严格特征容差，不自动等于预测效应消失；'
              '端点分数通过也不等于找回了历史运行中的每一个中间数值。', '',
              '## 5. 发作检验为什么仍不能下强结论', '',
              'H2b始终使用原冻结间期特征，发作梯度不能更新其上游。风险和距发作时间来自同一个survival读出，'
              '不能当作两个独立发现。新增同宽度历史投影用于检查压缩与容量替代解释；它不是初始化observer对照。', '',
              'E922仍只有三次留出onset；按六小时聚集规则，它们属于同一个跨分区簇。'
              '大量重叠anchor或person-period行不能增加独立发作分母。该聚集分析是审阅后的稳健性检查，'
              '不是独立预注册确认，也不把时间聚集直接认定为同一个生理事件。', '',
              '风险读出保存了最终权重、归一化和各正则化配方，检查返回权重处的梯度。'
              '未来发作/未来空间场正对照明确带有信息泄漏，只用于仪器诊断；它证明强信号可以被读出，'
              '不保证对微弱生理效应有足够统计功效。缺少有效训练触点或发作分母的端点为不可估。', '',
              f"本次可估的选中风险读出 {summary.get('repair_diagnostics',{}).get('estimated_hazard_readouts',0)} 个，"
              f"其中通过最终梯度闸门 {summary.get('repair_diagnostics',{}).get('stationary_hazard_readouts',0)} 个。"
              '这仍是读出数量，不是独立发作数量。', '',
              '旧工作树没有正确挂到bb150能量场缓存；新的复核显式绑定已有主缓存和触点名称。'
              '因此旧能量场缺失是未测量，不是阴性。', '',
              '## 6. 工程配置、可复现性与训练充分性', '',
              f"有限补修之后，H1仍有 **{len(h1_limited)}条路径触及预算上限**；H2a承重stage仍有 **{len(h2a_unqualified)}条未通过训练资格**。"
              '它们保留为优化受限，不能据此宣称训练已经充分，也不能据此作生理阴性判断。'
              '精确患者、seed、路径和停止步数列在机器汇总 repair_diagnostics 中。', '',
              '|组件|实际结构/训练|解释|', '|---|---|---|',
              '|慢事件observer|burden 7→14（98参数），grammar 24→21（504参数），共602可训练参数；状态42维|固定7个指数时间常数600–57600秒；不是64层或128维RNN|',
              '|背景observer|18→14（252参数），加7个可靠性量，状态21维|背景轨迹与事件轨迹分别审计|',
              '|H1优化|AdamW，β=(0.9,0.999)，eps=1e-8，clip=2；每步完整FIT目标|event/grid状态LR=1e-3、head=3e-3；dual状态/背景LR=3e-4、head=1e-3|',
              '|H1预算与初始化|q 3600、B_mark 1800、state 1800、原random 900；每25步验证，耐心16次|输入矩阵normal std=0.02；状态读出起点std=0.01；有父臂回退，不能强迫选非零步|',
              '|H1其他学习率与正则|q LR=3e-3，B_mark=2e-3，random head=1e-3|event/grid一般weight decay=1e-3，dual=1e-4，B_mark=1e-4；event/grid状态预热100步，dual为0|',
              '|H2a事件内部decoder|64×64递归矩阵受结构mask约束；E253 dual adapter 160→8，再调制64节点/8触点/STOP，共2376参数|64是事件内部组织节点，不能与小时状态混为同一隐藏层|',
              '|E253逐层示例|事件内recurrent存储4096权重；STOP为4→16→1，共97参数；decoder全部参数4331，H2a时全部冻结|state adapter为160→8（1280参数），GELU后接64/64/8/1输出（512/512/64/8参数）；几何矩阵与mask另列为buffer|',
              '|H2a优化|AdamW；static LR=1e-3、state LR=8e-4；batch=512事件；clip=1；state预热5 epoch|static/state耐心分别12/40 epoch；精确逐单元预算在卡片和队列manifest|',
              '|H2b风险读出|float64 LBFGS，LR=0.5，strong_wolfe；6个预设危险率时间段|原160步，必要时再480步；第二阶段停止精度1e-12；最终max梯度阈值1e-5|',
              '|归一化|FIT/pre-INNER估计的尺度、词典和PCA；H1主要用median/MAD并clip到±12|observer没有BatchNorm/LayerNorm；新缓存绑定实际FIT输入与变换|', '',
              '上述维度是本队列代表性配置；H1逐患者、逐层精确形状、参数数和选中变化保存在state_lineage.json，'
              '全部120个可估H2a adapter和冻结decoder的逐层清单在model_inventory.json。'
              '“训练充分”在本报告中只指登记的路径、耐心、预算及数值条件合格，并非证明全局最优。'
              'v0.3.7原先已有优化搜索和梯度审计，不能说完全没有；问题是这些检查原来没有贯穿所有承重对照和最终结论。', '',
              '## 7. 最短后续路线', '',
              '本轮已修复能定位的实现和汇总问题，完成登记的有限预算与审计；仍触顶的拟合保留为优化受限。'
              '原运行未保存的精确预处理历史无法事后补造，有来源限制的旧结果保留为诊断。'
              '之后如要检验非线性演化，应在不使用发作标签选择模型的条件下，预注册容量匹配的线性/非线性转移比较。'
              '如要证明统一状态，应由同一患者、同一冻结权重贯穿H1、长程credit、接触点表达和有足够独立发作的H2b；'
              '确认必须使用尚未参与当前判断的数据。H3需要能区分观察器记忆更新与生理反馈的识别设计。', '',
              '本轮所有再评估仍属于已经检查过的SELECTION；development/sealed未用于训练、选择或结论。'
              '优化受限、初始化回退、来源差异、不可估和未建立分别记录，不合并成科学阴性。', '',
              '## 8. 审阅后补充的三项口径修正（2026-09-05）', '',
              '**对照封底。** 常数臂与匹配随机臂嵌套在强基线之上，可能比强基线本身还差；此时"胜过自身对照"计入的是对照的损害而非状态的能力。'
              'E1096的匹配随机组件比强基线差0.34（0.5小时档5个seed中3个、2小时档5/5），因此原先0.51–0.55的"胜过随机"'
              '在封底到强基线后为0.078–0.122，与其胜过强基线的干净嵌套值相同；E1077与E1125的对照未变差，数值不变。'
              '判定不受影响（封底后仍为正），改变的只是效应量。机器汇总新增 dynamic_over_constant_floored、'
              'gain_over_random_floored 与 control_worse_than_strong_baseline。', '',
              '**代码版本审计已恢复。** v0.3.7曾加入的逐卡源码哈希审计在v0.3.8汇总器中缺失。恢复后显示：370张卡中30张无版本记录（均为NOT_ESTIMABLE存根），'
              'h1_train.py存在2个版本、h2a.py存在3个版本。H1的版本分裂沿队列边界（长/短两层本就不合并），'
              'h2a长队列event家族的分裂发生在同一被试的seed之间，其5-seed中位数跨了两个版本，须按来源受限读。', '',
              '**最内层基线选择初始回退已单列。** E1077与E1146各15个模型单元的q层选中第0步，'
              '说明这一层没有贡献选中的增量；不能据此断言后面的历史或背景层都退回截距。'
              '例如E1077的dual背景层仍选中了非零步。机器汇总同时列出后续各层选中步数。'
              'qualified只表示登记的优化条件通过，不表示这一层学出了有效信号；初始回退也不自动等于优化失败。', '',
              '**错时与物理窗口诊断已恢复。** 按队列逐卡列出实际donor位移、错时臂是否比常数更差，以及每个预测尺度的独立物理窗。'
              '这些量不能由seed重复数或重叠anchor数量替代。版本混用按队列、患者、family和seed分组报告；'
              'hash不同本身不能判断变化是否影响数值，旧队列因此仍只作development诊断。', '',
              '**下一版的判据调整。** 随机投影若未丢失信息，后续读出可能吸收学习投影，因此不再无条件要求胜过每一种随机组件。'
              '改为固定历史F、可学习线性转移L、非线性转移N的受控比较，并冻结同一状态检验未参与训练的任务。'
              '同seed的宽joint_evidence与另加学习、重放条件的composite交集是不同对象；此前对二者“自相矛盾”的怀疑已撤回。', '']
    if technical:
        lines += ['## 逐队列执行记录', '', '|队列|状态|完成|不可估|失败|待运行/运行|', '|---|---|---:|---:|---:|---:|']
        for name, queue in summary['repair_queues'].items():
            lines.append(f"|{name}|{queue.get('status')}|{queue.get('complete',0)}|{queue.get('not_estimable',0)}|{queue.get('failed',0)}|{queue.get('pending',0)}/{queue.get('running',0)}|")
        lines += ['', '## H1完整对照候选与上游身份', '', '|队列/family|患者|未来小时|同seed完整通过|物理窗|状态身份|', '|---|---|---:|---:|---:|---|']
        for cohort, families in summary['h1'].items():
            for family, data in families.items():
                for row in data['rows']:
                    if not row['directional_dynamic_state_candidate']: continue
                    classes = ', '.join(sorted(set(row['selected_state_classes'].values())))
                    lines.append(f"|{cohort}/{family}|{short(row['subject'])}|{row['horizon_seconds']/3600:g}|{row['joint_evidence']['n_joint_positive']}/5|{row['independent_selection_windows']}|{classes}|")
        lines += ['', '## H2a各患者同seed多端点交集', '', '|队列/family|患者|完整通过|其中上游event矩阵更新|', '|---|---|---:|---:|']
        for cohort, families in summary['h2a'].items():
            for family, data in families.items():
                for row in data['rows']:
                    lines.append(f"|{cohort}/{family}|{short(row['subject'])}|{row['multi_endpoint_joint_evidence']['n_joint_positive']}/5|{row['learned_upstream_multi_endpoint_joint_evidence']['n_joint_positive']}/5|")
        lines += ['', '所有方向性seed门槛仅衡量优化重复；它们不是独立样本显著性检验或置信区间。', '']
        lines += ['## 有限预算后仍触顶的H1路径', '', '|患者|family|seed|路径|运行步|选中步|', '|---|---|---:|---|---:|---:|']
        for row in h1_limited:
            lines.append(f"|{short(row['subject'])}|{row['family']}|{row['seed']}|{row['stage']}|{row['steps_run']}|{row['selected_step']}|")
        lines += ['']
    validation_path = Path(summary['output_paths']['summary_json']).parent / 'validation/regression_validation.json'
    if not validation_path.exists(): validation_path = root / 'validation/regression_validation.json'
    if validation_path.exists():
        validation = json.loads(validation_path.read_text())
        lines += [f"**回归验证：{validation['passed_tests']}项通过，退出码{validation['returncode']}。** "
                  f"[完整命令、日志与代码hash]({validation_path})。测试证明相应实现与保护条件有效，不证明生理假设成立。", '']
    lines += ['## 可追溯文件', '',
              f"- [机器汇总]({summary['output_paths']['summary_json']})：全部方向、资格、原卡/修复卡hash及队列。",
              f"- [状态权重与来源审计]({summary['state_lineage']['path']})：全部层、配置、实际权重变化与下游绑定。",
              f"- [逐层参数清单]({root / 'model_inventory.json'})：H2a所有adapter、decoder参数与固定buffer分开计数。",
              f"- [冻结执行合同]({root / 'execution_contract.md'})：每轮补修范围和预算。",
              f"- 新预处理重放：`{root / 'checkpoint_replay_verified'}`；保留条件下的原输入重放：`{root / 'checkpoint_replay'}`。", '']
    return '\n'.join(lines)
