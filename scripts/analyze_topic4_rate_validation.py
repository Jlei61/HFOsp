#!/usr/bin/env python3
"""Summarize the actual V1 evidence and keep later gates closed on mismatch."""
from validate_topic4_fixed_rate_base import OUT, ROOT, read, write
from pathlib import Path
import numpy as np
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def summaries(rate,dt):
    smooth=uniform_filter1d(np.asarray(rate,float),int(round(10/dt)),mode='nearest')
    post=smooth[int(round(500/dt)):]
    sensitivity=[]
    for threshold in (1.,5.,10.):
        on=post>threshold;starts=np.flatnonzero(np.diff(np.r_[False,on].astype(int))==1)
        ends=np.flatnonzero(np.diff(np.r_[on,False].astype(int))==-1)+1
        durations=(ends-starts)*dt
        complete=(starts>0)&(ends<len(on))
        sensitivity.append({'threshold_hz':threshold,'segments_including_censored':len(starts),
            'upcrossings':len(starts)-int(on[0]),
            'episodes_at_least_10ms':int(np.sum(durations>=10)),
            'complete_segments_at_least_10ms':int(np.sum(complete&(durations>=10))),
            'median_complete_segment_duration_ms':float(np.median(durations[complete])) if complete.any() else None,
            'fraction_above':float(np.mean(on)),
            'left_censored':bool(on[0]),
            'right_censored':bool(on[-1])})
    return {'mean_after_500ms_hz':float(np.mean(rate[int(500/dt):])),
        'peak_10ms_after_500ms_hz':float(post.max()),
        'last_300ms_mean_hz':float(np.mean(rate[-int(300/dt):])),
        'population_excursion_sensitivity':sensitivity}


def main():
    rows=[]
    for path in sorted((OUT/'snn').glob('*.json')):
        meta=read(path);name=path.stem
        if name.startswith('prefix'):continue
        snn=np.load(path.with_suffix('.npz'));dt=float(snn['dt_ms'])
        row={'name':name,'snn':summaries(snn['rate_e_hz'],dt),'pulse':meta['args'],'rate':{},'colored_development':{}}
        onset=meta['args']['onset'];lo=int((onset-50)/dt);hi=int(onset/dt)
        row['pre_pulse_snn_50ms_mean_hz']=float(snn['rate_e_hz'][lo:hi].mean())
        for g in (10,20):
            rp=OUT/'rate'/f'{name}_grid{g}_cascade.npz'
            if rp.exists():
                model=np.load(rp);row['rate'][str(g)]=summaries(model['rates_hz'][:,0],dt)
            cp=OUT/'rate'/f'{name}_grid{g}_cascade_colored.json'
            if cp.exists():
                colored=np.load(cp.with_suffix('.npz'))
                row['colored_development'][str(g)]=summaries(colored['rates_hz'][:,0],dt)
        rows.append(row)
    sham=next((r for r in rows if r['name']=='sham_7101'),None)
    failure=bool(sham and '20' in sham['rate'] and
        sham['snn']['peak_10ms_after_500ms_hz']>10 and
        sham['rate']['20']['peak_10ms_after_500ms_hz']<1)
    status=read(OUT/'v1_batch_status.json') if (OUT/'v1_batch_status.json').exists() else {}
    report={'status':'V1_ORIGINAL_CLOSURE_FAILED_REPAIR_UNVALIDATED' if failure else 'V1_UNRESOLVED',
        'batch_status':status,'runs':rows,
        'failure_rule':'Necessary phenotype mismatch: SNN has >10 Hz population excursions after burn-in whereas 1-mm rate stays below 1 Hz. This is a broad diagnostic separation, not a calibrated patient-equivalence threshold.',
        'excursion_labels':'Population rate excursions only, not patient primary event labels or lineage event counts.',
        'onset_state_limit':'Pulse at 1000 ms may occur during recovery from spontaneous activity; pre-pulse state reported. Do not call the single-seed dose response a resting ignition threshold.',
        'V2_released':False,'V3_released':False,'V4_released':False,'Hopf_released':False,
        'repair_attempts':['literal rise/decay cascade and native-step DC gain','both external OU components and actual clipped afferent rate replay','1-mm vs 2-mm spatial cells','exact empirical threshold averaging','finite-size Poisson population diagnostic'],
        'next_repair':'Validate the corrected response under mixed excitatory/inhibitory currents, then state-matched finite and paired pulses. Keep physical SNN geometry/weights fixed; no V2/Hopf on mean-rate agreement.'}
    if (OUT/'isolated_response_calibration.json').exists():report['isolated_response']=read(OUT/'isolated_response_calibration.json')
    report['colored_calibration']={pop:read(OUT/f'colored_response_diagnostic{suffix}.json')
        for pop,suffix in [('E',''),('I','_I')] if (OUT/f'colored_response_diagnostic{suffix}.json').exists()}
    report['colored_network_status']='Development correction restores excursions in short prefix; timing, spatial propagation, longer persistence and finite pulse response not accepted.'
    report['closure_versus_physics']='Calibrated rate response times are reduction parameters, not alterations to SNN membrane or synaptic parameters.'
    if (OUT/'response_correction_ablation.json').exists():report['response_correction_ablation']=read(OUT/'response_correction_ablation.json')
    if (OUT/'reference_variation_check.json').exists():
        report['reference_variation_check']=read(OUT/'reference_variation_check.json')
        report['corrected_rate_adjudication']='Positive evidence for repeated self-ending activity; quantitative fidelity not established. Single-window differences alone do not establish failure relative to SNN variability.'
    write(OUT/'analysis.json',report)
    lines=['# 固定双核底物：降阶动力学验证执行报告','',f"状态：`{report['status']}`。物理批次：{len(status.get('completed',[]))}/8 对照条件完成SNN和两种空间分辨率rate计算。",'',
        '## 已验证','',
        '6101网络的8项节点/连接身份重建一致；600 ms与2.4 s无刺激重演的原生二维spike计数均与冻结历史轨迹完全一致，活动率在历史float32保存精度下一致。Z/M关闭，两个旧GIF基底未混入。', '',
        '## 原生动力学与近似','',
        '|条件|SNN整段500ms后均值|rate 2mm|rate 1mm|','|---|---:|---:|---:|']
    for row in rows:
        vals=[f"{row['rate'][str(g)]['mean_after_500ms_hz']:.3f}" if str(g) in row['rate'] else '待完成' for g in (10,20)]
        lines.append(f"|{row['name']}|{row['snn']['mean_after_500ms_hz']:.3f}|{vals[0]}|{vals[1]}|")
    lines += ['', '### 输入相关性修正的开发结果','',
        '采用Fourcaud–Brunel短相关时间展开的阈值/重置平移启发式，单群体弱、中脉冲校准得到E响应5 ms、I响应2.5 ms。E的两剂量归一化均方误差之和从3.355降为0.289；I修正后为0.168。膜时间常数和所有SNN物理参数保持不变。强脉冲此前已查看，因此新修正的强脉冲结果是开发诊断，不再称盲留出。', '',
        '20×20网络短前缀恢复了反复升高、回落，整窗平均14.82 Hz，SNN为15.23 Hz；均值接近不能替代逐事件检验，同一时间的空间快照还受事件相位偏移影响，不能据此单独断言传播模式错误。GABA混合输入的方差加权相关时间为启发式，且GABA/膜时间常数比不满足小参数假设，因此不能将文献公式直接视为本网络已验证闭合。', '',
        '|无重置长探针|修正rate 1mm：500ms后均值Hz|10ms峰值Hz|','|---|---:|---:|']
    for row in rows:
        if '20' in row['colored_development']:
            rr=row['colored_development']['20'];lines.append(f"|{row['name']}|{rr['mean_after_500ms_hz']:.3f}|{rr['peak_10ms_after_500ms_hz']:.3f}|")
    if sham and '20' in sham['colored_development']:
        ss=sham['snn']['population_excursion_sensitivity'][0]
        cc=sham['colored_development']['20']['population_excursion_sensitivity'][0]
        lines += ['', f"无刺激窗口的1 Hz诊断切分：SNN/修正rate高于阈值的时间比例为{ss['fraction_above']:.3f}/{cc['fraction_above']:.3f}，完整片段时长中位数为{ss['median_complete_segment_duration_ms']:.1f}/{cc['median_complete_segment_duration_ms']:.1f} ms。这是单窗口的描述性差异，不能单独作为修正近似失败的依据；需对照SNN自身变异。只统计完整片段，左右窗边界截断片段另记。1/5/10 Hz均为开发敏感性读出，不是患者事件定义或接受界限。", '']
    if 'reference_variation_check' in report:
        lines += ['', '### 对上一轮判断的校正：先对照SNN自身变异','',
            '将固定6101的7101/7102两段24秒历史轨迹切为同长度1.9秒窗口，以共同2ms观测步长重算。SNN窗口内完整片段时长中位数在1/5/10 Hz阈值下的范围分别为76–149、61–110、51–88 ms；修正rate单窗为142、108、86 ms，偏长但仍在这些描述性范围内。因此上一轮不能凭142对90 ms就断言事件终止/恢复不匹配。', '',
            '这些窗口来自两个随机种子，不是多个独立网络或独立种子；范围比较不是正式等价检验。当前应接受修正模型已有反复产生并自行回落的定性正证据，定量保真、刺激阈值和恢复曲线尚未建立；空间评价应按各自事件阶段对齐，而非要求噪声轨迹逐时刻重合。', '']
    lines += ['', '### 下一项有界验证与判据','',
        '已完成600ms前缀的两项拆分诊断：原闭合、仅加快rate响应、仅修正输入相关性、两者合并的整窗平均依次为0.566、6.489、3.525、14.816 Hz。两个部分都影响结果；短前缀均值与幅度不足以证明自限事件保真，更不能将此解释为患者网络机制。详见response_correction_ablation.json。', '',
        '下一校准必须包含原SNN所经历的混合E/I输入，因为现有单群体实验仅有外源AMPA。检验安静、接近触发、强输入及撤除后的响应；若单群体仍失配，优先改静态传递和历史依赖，暂不靠细化网络参数补偿。', '',
        '单群体合格后，从原SNN完整状态选取明确安静与恢复阶段，配对有限刺激及无刺激分支，保存膜电位、不应期、突触和延迟历史。不同剂量使用共同的预生成全局/空间OU轨迹，并核对刺激前一致；否则当前共享RNG的全局OU会在刺激后分叉。成对脉冲用来测第二次响应随间隔恢复，不从单脉冲均值推断恢复常数。', '',
        '空间传播在各自事件起点对齐后评价参与范围、core间时差和波前顺序；背景输入相同不要求每个随机spike相同。误差容限从指定SNN重复样本及观测误差定义，不能在查看rate结果后调阈值让其通过。达到V1再开无重置连续事件V2；持续振荡与积累仍分别受后续门控。', '']
    lines += ['', '以上均值包含事件，不能称为安静背景放电率。群体1/5/10 Hz阈值敏感性仅作动力学诊断，既不是患者分类器，也不等于原谱系事件计数。完整二维场、每步E/I群体率、每步实际外源率和格内方差保存在snn/，rate对照保存在rate/。', '',
        '## 关键更正','',
        '实际引擎有两种OU：局部空间OU与sigma_n=3.3、tau_n=150 ms的全局OU。后者与Poisson采样共用RNG；施加刺激后，不能只凭相同整数seed声称其逐步共同随机数仍相同。本轮完整记录全局OU及E/I实际输入后给rate重放。', '',
        '固定jump/rise改变GABA decay不改变归一化电流的无限时窗脉冲面积。18/20.61155/42 ms的单位jump面积均为1.05083319448，峰值与时延不同；原先关于面积变化的文字已纠正，冻结SNN源文件没有改动。', '',
        '## 闭合修复与继续条件','',
        '已尝试补齐双指数突触、保留实际输入、提高空间分辨率、完整经验阈值积分及有限群体计数噪声诊断；短前缀均未恢复反复高幅自限事件。有限群体诊断不是严格推导的mesoscopic模型，尤其未正确闭合全部相关性和不应期历史，不能据其阴性排除有限尺寸机制。', '',
        '原白噪声闭合的单群体校准：弱、中剂量选出的单一tau_rate=20 ms仍在首次强剂量测试上失配。强脉冲SNN峰值182.49 Hz、峰时6.6 ms，而该rate闭合为95.89 Hz、21.4 ms；简单把时间常数调快也不能同时匹配弱输入。输入相关性修正已执行，结果见上节；不在患者loss上重搜物理参数来补偿此误差。', '',
        '本次1000 ms刺激可能落在自发事件恢复期，因此暂不报告安静态点火阈值、刺激导致的独立传播时差或双脉冲恢复通过。若rate不能先保留无刺激动态，即使一次强刺激后能返回也不放行V2。V2连续多事件、V3持续振荡和V4积累跨越均未通过，未运行Hopf扫描。', '',
        '机制参考：[Schwalger et al., 2017](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005507)讨论有限群体波动与spike历史的联合闭合；本轮Poisson诊断不是该论文方法的完整实现。[Fourcaud & Brunel, 2002](https://www-sop.inria.fr/members/Olivier.Faugeras/MVA/Articles_11/fourcaud-brunel-02.pdf)给出突触滤波对群体响应的影响及小参数近似；本轮混合输入处理仍需实测校准。', '']
    (OUT/'ANALYSIS.md').write_text('\n'.join(lines))
    manifest_path=ROOT/'config/topic4_rate_model_dynamics_validation_v1.json'
    manifest=read(manifest_path);manifest['status']=report['status']
    manifest['validation_state']['new_simulations_run']=len(list((OUT/'snn').glob('*.json')))
    manifest['validation_state']['new_simulations_count_definition']='completed full-network SNN runs, including prefix; isolated calibration assays excluded'
    manifest['validation_state']['analysis_path']=str(OUT/'analysis.json')
    write(manifest_path,manifest)
    render_response()
    print(report['status'],len(rows),flush=True)


def render_response():
    a=np.load(OUT/'isolated_response.npz');dt=float(a['dt_ms']);s=uniform_filter1d(a['snn_rate_hz'],50)
    rates=uniform_filter1d(a['rate_closures_hz'],50,axis=0);taus=a['tau_rate_ms']
    c=np.load(OUT/'colored_response_diagnostic.npz')
    ci=int(np.argmin(abs(c['tau_rate_ms']-5.)))
    corrected=uniform_filter1d(c['rate_closures_hz'][:,ci],50)
    fig,axes=plt.subplots(1,3,figsize=(13,3.9),layout='constrained')
    for axis,onset,dose in zip(axes,[500,1000,1500],[.25,1.,4.]):
        ix=slice(int((onset-20)/dt),int((onset+100)/dt));t=np.arange(len(s))[ix]*dt-onset
        axis.axvspan(0,18,color='#dddddd');axis.plot(t,s[ix],color='black',lw=2,label='LIF population')
        for j in (0,2,4):axis.plot(t,rates[ix,j],label=f'Rate τ={taus[j]:g} ms',lw=1.3)
        axis.plot(t,corrected[ix],label='Colored correction, τ=5 ms',color='#cc3311',ls='--',lw=2)
        axis.set(title=f'Added input {dose:g} spikes/ms',xlabel='Time from pulse (ms)',ylabel='E rate (Hz)')
        axis.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8);fig.suptitle('Single-population response calibration: original LIF parameters, no recurrence')
    dest=OUT/'figures';dest.mkdir(exist_ok=True);stem=dest/'isolated_response_calibration'
    fig.savefig(str(stem)+'.png',dpi=170);fig.savefig(str(stem)+'.pdf');plt.close(fig)
    path=dest/'README.md';old=path.read_text() if path.exists() else ''
    header='### isolated_response_calibration.png'
    entry=header+'\n原LIF参数和全部经验阈值下，对三种18毫秒外源率脉冲比较单群体spiking响应、白噪声rate响应时标及输入相关性修正。移除递归仅为隔离群体响应闭合，不是网络机制删减后通过验证；强脉冲对新修正而言已是开发数据。\n**关注点**：原固定rate时标不能同时对齐弱输入与强输入峰时；修正虽改善，仍需混合E/I输入和网络验证。\n'
    if header in old:
        start=old.index(header);end=old.find('\n### ',start+len(header));end=len(old) if end<0 else end
        old=old[:start]+entry+old[end:]
    else:old+='\n'+entry
    path.write_text(old)


if __name__=='__main__':main()
