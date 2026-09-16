#!/usr/bin/env python3
"""Summarize the bounded exploration; never dispatch or truncate a simulation."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import snapshot_topic4_fig5_exploration as inventory

ROOT, BASE, WINDOW = inventory.ROOT, inventory.BASE, inventory.WINDOW
read = inventory.read


def main(final=False):
    now = datetime.now().astimezone()
    window = read(WINDOW/'window.json')
    deadline = datetime.fromisoformat(window['deadline'])
    if final:
        assert now >= deadline, 'The authorized exploration window has not ended.'
    inventory.main()
    snapshot = read(WINDOW/'comprehensive_inventory/latest.json')
    inventory_path=WINDOW/'comprehensive_inventory/latest.json'
    if final:
        inventory_path=WINDOW/'comprehensive_inventory/inventory_at_deadline.json'
        inventory_path.write_bytes((WINDOW/'comprehensive_inventory/latest.json').read_bytes())
        (WINDOW/'comprehensive_inventory/inventory_at_deadline.md').write_bytes(
            (WINDOW/'comprehensive_inventory/latest.md').read_bytes())
    protocol = read(BASE/'m_parameter_modes_fig5_20260913/protocol.json')
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
                    for p,h in protocol['source_hashes'].items())
    assert unchanged, 'Physical source drift requires review before delivery.'
    m = read(BASE/'m_parameter_modes_fig5_20260913/analysis_summary.json')
    energy = read(WINDOW/'early_energy_high_resolution_replay/energy_comparison.json')
    reset = read(WINDOW/'reset_matched_90s/analysis.json')
    best = WINDOW/'early_energy_high_resolution_replay/candidate'
    bestmeta = read(best/'fig5_metadata.json')
    assert bestmeta['metrics']['full_1_to_5_observed'] and energy['contact_result_unchanged']
    original_m_complete = snapshot['batches']['m_parameter_modes_fig5_20260913']['result_files']
    z_complete = snapshot['batches']['m_on_z_kinetics_20260912']['result_files']
    seed2_folder=WINDOW/'early_z_refill_branches/runs/early_z_refill_s9108402'
    seed2_result=read(seed2_folder/'result.json')
    seed2_progress=read(seed2_folder/'progress.json')
    seed2_tracker=seed2_result.get('tracker',seed2_progress)
    seed2_path=WINDOW/'early_energy_high_resolution_replay_seed9108402/candidate/fig5_metadata.json'
    if not seed2_path.exists():
        seed2_path=WINDOW/'early_z_refill_branches/candidates/early_z_refill_s9108402/fig5_metadata.json'
    if not seed2_path.exists():
        for item in read(WINDOW/'ongoing_fig5_prefixes/index.json').values():
            if item['source_run']==str(seed2_folder):
                seed2_path=Path(item['figure']).parent.parent/'fig5_metadata.json'
                break
    seed2_meta=read(seed2_path)
    late=read(WINDOW/'late_Z_refill_fig5/latest.json')
    requirements = [
        dict(item='Fixed manual substrate, native Z/M and continuous state', status='VERIFIED',
             evidence='Nine physics sources unchanged; real full-state numerical QA and replay observation parity.'),
        dict(item='One complete finite→high→Z refill→return→second high trajectory',status='OBSERVED_EXTERNAL_RETURN',
             evidence=str(best/'fig5_metadata.json')),
        dict(item='Autonomous termination from M at fixed parameters',status='NOT_ESTABLISHED',
             evidence='Completed recurrence example used external Z refill; external M-gain step is a separate intervention.'),
        dict(item='Native sustained low-frequency burst in the high state',status='NOT_ESTABLISHED',
             evidence=str(WINDOW/'recurrent_high_native_rhythm/analysis.json')),
        dict(item='Patient-compatible early energy increase',status='NOT_REPRODUCED_IN_COMPLETE_EXAMPLE',
             evidence=str(WINDOW/'early_energy_high_resolution_replay/energy_comparison.json')),
        dict(item='Full M grid followups and corresponding F surface',status='COMPLETE' if original_m_complete==40 else 'RUNNING',
             completed=original_m_complete,total=40,first_endpoints=m.get('established_first_endpoints'),
             complete_paired_F_cells=m.get('complete_F_cells'),total_F_cells=20),
        dict(item='Fixed-M Z kinetics grid',status='COMPLETE' if z_complete==147 else 'RUNNING',
             completed=z_complete,total=147,distinct_M_setting='.02 / 2s'),
        dict(item='Reset causal diagnosis',status='PARTIAL_LONG_FOLLOWUP_RUNNING',
             evidence='Z-only1000s and full-fast90s finished; M-clear1000s is tracked independently.'),
        dict(item='Figure layout and observation QA',status='AGENT_REVIEWED',
             evidence='Full and zoom figures viewed; raw0.1ms E2 observer verified without changing the trajectory.'),
        dict(item='Human visual and scientific acceptance',status='PENDING',model_frozen=False),
    ]
    payload=dict(time=now.isoformat(),status='WINDOW_COMPLETE_PENDING_USER_REVIEW' if final else 'DRAFT_WINDOW_ACTIVE',
        window_start=window['start'],deadline=window['deadline'],bounded_exploration_complete=bool(final),
        M_and_Z_parameter_grids_complete=original_m_complete==40 and z_complete==147,
        model_scientific_acceptance=False,requirements=requirements,physics_unchanged=unchanged,
        inventory=str(inventory_path),
        preferred_complete_figure=str(best/'figures/fig5.png'),
        existing_authorized_batches_continue=True,new_exploration_dispatch_closed=bool(final),
        no_simulations_dispatched_or_stopped_by_report=True)
    tag='final' if final else 'draft'
    (WINDOW/f'requirements_audit_{tag}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines=[f'# Fig5 自主探索审阅 · {now:%Y-%m-%d %H:%M}', '',
        ('已结束01:40–09:40的限定探索，以下是实际交付及仍按原合同继续的计算。' if final else
         '这是进行中的审阅稿，09:40前仍会随已授权结果更新，不能当作8小时探索已经完成。'),'',
        '## 目前能回答什么','',
        '弱而快M（ηM=.005、τM=1秒）已经产生完整五状态：10.59秒首次进入，12.8–13.8秒只补Z，15.2秒确认恢复，26.89秒再次进入，完整观察至37.5秒。M全程开启且未清零。这排除了“补Z后模型结构上必然不能再runaway”，但返回是外部干预造成的。', '',
        (f'第二个预定噪声种子也已再次进入：首次13.74秒，补Z15.95–16.95秒，恢复确认18.35秒，第二次28.47秒。其尾部随访'+
         (f'已完成至{seed2_result["end_s"]:.1f}秒。' if seed2_result else '仍在运行，尚未标为完整。'))
        if len(seed2_tracker.get('entries',[]))>=2 else '第二个预定噪声种子的再进入随访尚在进行。','',
        '原ηM=.02、τM=2秒工作点的Z-only随访已到1000秒；释放后923.5秒未再次进入。另一个共享状态的完整90秒配对中，额外清M和额外清所有细胞快状态仍未再次进入，后期平均Z、M及E率很接近。既有M的直接衰减残留不能独自解释长期现象；清除后的活动及噪声实现仍可影响状态。90秒阴性不等于永久稳定。', '',
        '外部将同一高态的M增益调至2可压制高放电；调至.2仍保持高率。前者趋于近沉默，既不是固定参数下的自主终止，也不是恢复有限事件的证据。', '',
        '为什么仅打开M不一定足够：代码中每个E细胞每次发放给M加1，之后按τM衰减，反馈电流是ηM×M。在近似恒定放电率下，时间平均M约为τM(s)×r(Hz)，因此ηM=.005、τM=1秒、约500Hz时适应电流只有约2.5mV等效值。这个量级估计解释了“放得很快却没有被弱M压下”的可能性；它不是稳定性或分岔证明。', '',
        '这不只是量级猜测：原弱M长随访的第一个种子在59–60秒，所有E细胞的原始GABA均在耗竭门槛以上，平均Z约4.45×10⁻⁵，E平均率500Hz，M反馈2.500mV；兴奋输入约1579mV，而乘Z后的抑制仅0.079mV。第二种子49–50秒及τM=2秒两种子29–30秒也持续越过门槛。此时现有方程令Z继续趋向零，M已接近发放与衰减的平衡，并非还会无限积累。这不支持“只要继续等待M累积就会自然返回”的解释；仍不能据此排除噪声作用、其他参数或其他状态中的返回。', '',
        '原ηM=.02、τM=2秒的75.5秒实测高态，M约457，当前反馈约9.15mV，而平均净驱动约514mV等效值。保持同一状态代数地改ηM到.2或2，净驱动分别约432和−392；随后真正SNN的5秒续跑也分别维持高率和被压制。两层证据共同支持反馈相对回返兴奋不足这一解释，但尚未证明固定参数能兼顾进入与自主返回。', '',
        '## 两个尚未补上的科学缺口','',
        '高态目前更像快速持续放电：完整例子的第二高态采样E神经元约2.1ms一次发放，目标1–150Hz的群体burst包络很弱。虚拟电极30–80Hz滤波后的形状不能证明原网络具有所需振荡。', '',
        f'早期能量经真实0.1ms重放核查，只有{energy["after"]["n_model_contacts_above_baseline"]}/15模型电极和{energy["after"]["native_cells_with_increased_power"]}/400原生网格增强。与固定Fig3C的空间ρ={energy["after"]["model_patient_rho"]:.3f}不能替代能量增强。1ms采样的混叠确实抬高部分主招募带低频功率，但校正后仍未复现目标。', '',
        (f'第二种子的模型电极增强{seed2_meta["E2"]["n_model_contacts_above_baseline"]}/15，ρ={seed2_meta["E2"]["model_patient_rho"]:.3f}，也未复现临床早期增强。其原生空间图记录分辨率为{seed2_meta["E2"].get("native_spatial_bin_ms",1)}ms；两种子使用同一Fig3C，不按相关值挑选参照。'
         if seed2_meta.get('E2',{}).get('status')=='MEASURED' else '第二种子能量观测仍待完成核查。'),'',
        '## 完整新版图候选','',
        '| 实际模式 | 参数与观察 | 图 |','|---|---|---|',
        f'| 补Z后返回并再次高率 | ηM=.005 / τM=1s，37.5s完整；E2用0.1ms原生计数 | [完整Fig5]({best}/figures/fig5.png) · [转变放大]({best}/figures/fig5_transition_zoom.png) |',
        f'| 补Z返回，长程未再进入 | ηM=.02 / τM=2s，1000s | [完整Fig5]({WINDOW}/completed_Z_only_1000s_fig5/figures/fig5.png) |',
        f'| 高率未自行恢复 | ηM=.02 / τM=2s，原有90s数据 | [完整Fig5]({WINDOW}/existing_mode_figures/weak_fast/figures/fig5.png) |',
        f'| 观察窗内未进入高率 | ηM=.02 / τM=20s，原有90s数据 | [完整Fig5]({WINDOW}/existing_mode_figures/weak_20s/figures/fig5.png) |', '',
        (f'[第二噪声种子的当前Fig5]({seed2_path.parent}/figures/fig5.png)，实际图中覆盖{seed2_meta.get("source_duration_s")}秒；仿真完整随访是否结束以上述结果文件为准。' if seed2_meta else ''),'',
        '同一条轨迹的旧图、完整图、能量修正版和数值重放只算一条证据；原有90秒数据不会计入本轮40个完整随访样本。所有候选仍待用户目视及科学验收。', '',
        (f'原弱M网格中较晚补Z的同种子对照已保存至{late["source_duration_s"]:.1f}秒：先保留首次高态后60秒原生演化，再于{late["restore_s"]:.2f}–{late["release_s"]:.2f}秒只补Z，已观察恢复。'+
         ('该保存前缀也已包含第二次高态。' if len(late['entries'])>=2 else '第二次高态尚未在该保存前缀中观察到。')+
         f'这是原M网格的进行中前缀，完整随访仍未结束，不能记为最终阴性或新增独立种子。[对应整图]({late["figure"]})。'
         if late else '较晚补Z的原M网格对照继续保留其原定随访；未保存完成的阶段不据进度估计补画。'), '',
        '## 扫描与剩余计算','',
        f'M网格完整随访 {original_m_complete}/40；首次终点已确定 {m.get("established_first_endpoints",0)}/40；两种子齐备的F格 {m.get("complete_F_cells",0)}/20。',
        f'固定M=.02/τM=2秒的Z扫描 {z_complete}/147 完成。它和弱M示例是不同工作点，不能把这张Z图直接当作弱M的同条件F。',
        '首次进入时间从保存的全E发放计数核验，以≥200Hz持续200ms的确认时间计；未进入且未跑满180秒保留待定，不能提前填删失。完整返回/再进入随访继续独立进行。', '',
        f'[运行和checkpoint清单]({WINDOW}/comprehensive_inventory/latest.md) · [M参数图]({BASE}/m_parameter_modes_fig5_20260913/figures/M_entry_time_and_fraction.png) · [Z参数图]({BASE}/m_on_z_kinetics_20260912/figures/entry_time_and_fraction.png)', '',
        '## 继续与停止标准','',
        '继续既定40条M轨迹（20个参数组合×2种子）、147条Z轨迹（49个组合×3种子）及已开始的有限对照，完成实际首次进入、原生恢复与再进入分类。高率计算明显更慢，GPU替换仅改变已验证逐位一致的加和执行，不缩短观察窗。',
        '不增加90秒电压/突触拆分，因为先决条件“全快状态清除可再次进入”没有出现。09:40以后不增加探索条件；已有队列保留原参数、种子和完整终点。',
        '下一里程碑先收齐原M网格并逐条区分：未进入、进入后自行返回、人工补Z后返回、再次进入、以及未恢复高率。首次进入时间F与这些完整轨迹分类分别交付；不能用首次终点齐备替代60秒原生高态观察和返回后的随访。',
        '对既定网格中实际出现的自主返回候选，再沿用目前0.1ms原生计数审计局部和群体burst包络，并检查返回后是否恢复有限事件。只有这两端成立，才值得用同底物的简化模型复现、延拓固定点/周期轨道并命名分岔。若没有候选通过，当前M扫描应如实收束，提出反馈机制的明确改变后进入下一版；不通过重新滤波、挑窗口或额外几何变量来替代这一缺口。',
        '在固定参数下同时看到有限事件、自主高态结束以及再次事件之前，不把这条线当作已完成的自主发作周期模型。若全部预定M参数组合仍无法兼顾两端，应先修正高态反馈与原生节律机制，再讨论患者能量匹配和分岔命名。三维轨迹仅是观察投影，不能直接充当闭合向量场、nullcline或Hopf证据。', '',
        f'[90秒配对诊断]({WINDOW}/reset_matched_90s/scientific_review.md) · [原生节律审计]({WINDOW}/recurrent_high_native_rhythm/analysis.json) · [能量观测审计]({WINDOW}/early_energy_high_resolution_replay/scientific_review.md) · [弱M高态实测平衡]({WINDOW}/weak_M_high_state_balance_snapshot.json) · [M电流量级]({WINDOW}/M_current_headroom/scientific_review.md) · [同状态增益续跑]({WINDOW}/high_state_M_gain_probe/scientific_review_complete.md)']
    (WINDOW/f'scientific_review_{tag}.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=payload['status'],review=str(WINDOW/f'scientific_review_{tag}.md'))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    main(final=args.final)
