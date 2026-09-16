"""Close the authorized review window without terminating bounded simulations."""
from pathlib import Path
import argparse,datetime,hashlib,json,re,time
import psutil

W=Path('/data/hfosp/topic4_sef_hfo/overnight_exploration_20260913')
MAIN=Path('/home/honglab/leijiaxin/HFOsp')


def read(p):return json.loads(p.read_text())
def write(p,j):p.write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n')


def main(packet):
    window=read(W/'window.json');now=time.time()
    if now<window['review_due_unix']:raise RuntimeError('Eight-hour window has not elapsed')
    out=Path(packet);manifest=read(out/'manifest.json');qa=read(out/'artifact_qa.json')
    assert manifest['window_elapsed_review'] and qa['status']=='PASS_AGENT_VISUAL_REVIEW_HUMAN_PENDING'
    states={k:read(W.parent/name/'status.json') for k,name in [
        ('followup','core_recruitment_tradeoff_followup_20260912'),
        ('multiseed','core_multiseed_response_curves_20260913'),
        ('global_axis','global_axis_residual_probe_20260913')]}
    live=[]
    for name,s in states.items():
        for worker in s.get('active',[]):
            try:
                p=psutil.Process(worker['pid']);assert p.status()!=psutil.STATUS_ZOMBIE
                live.append(dict(queue=name,unit=worker['unit'],pid=p.pid,create_time=p.create_time(),cmdline=p.cmdline()))
            except psutil.NoSuchProcess:pass
    lines=['|队列|完整运行|正在运行|尚未派发|','|---|---:|---:|---:|',
        f'|续跑参数响应|{manifest["followup_cumulative"]["response"]}/60|0|0|']
    for name,title,budget in [('followup','续跑确认',16),('multiseed','多网络响应',108),('global_axis','全局EE方向',6)]:
        s=states[name];n=s.get('complete',s.get('formal_complete',0));q=s.get('queued',len(s.get('pending',[])))
        lines.append(f'|{title}|{n}/{budget}|{sum(r["queue"]==name for r in live)}|{q}|')
    status='\n'.join(lines)
    stamp=datetime.datetime.now().astimezone().isoformat()
    text=(W/'scientific_window_review_draft.md').read_text().replace('# 8小时探索：科学审阅草稿（窗口结束后补入实际状态）','# 8小时探索：科学审阅')
    intro=f'\n\n窗口：2026-09-13 01:32:26–09:32:26（北京时间）；交付快照：{stamp}。**8小时工作窗口已完成，全部物理批次尚未结束。**\n\n'+status+'\n\n续跑60条中48条为该批新运行、12条为历史直接对照复用；原140条已完成，不能把复用再算一次。上表是各队列累计完成量，不是都在本8小时内新增。\n\n'
    text=text.replace('\n\n',intro,1)
    paired=W/'position_new_network_replay/scientific_note.md'
    if paired.exists():
        text+='\n## 本窗口最后完成的新网络位置配对\n\n'+paired.read_text().split('\n',1)[1]+'\n'
    text+='\n## 图件与来源\n\n'+f'- [统一图集PDF]({out}/overnight_review.pdf)与[逐图说明]({out}/figures/README.md)。\n'
    for title,path in [('完整事件时差分布','bridge_event_distributions/scientific_note.md'),('范围取舍','radius_tradeoff_review/scientific_note.md'),('新噪声配对','position_new_noise_replay/scientific_note.md'),('晚SCL解释性原生GIF','late_scl_TA_native_2711/README.md'),('核先后与TB原生过程','tb_core_timing_native_review/figures/README.md')]:
        text+=f'- [{title}]({W/path})。\n'
    text+='- [新噪声正常选例及每类前三例GIF]('+str(W.parent/'core_multiseed_response_curves_20260913/analysis/first_noise_review/README.md')+')。\n'
    (out/'interpretation_addendum.md').write_text(text)
    write(W/'window_closeout.json',dict(status='EIGHT_HOUR_REVIEW_COMPLETE_BOUNDED_QUEUES_CONTINUE',
        reviewed_unix=now,authorized_deadline_unix=window['review_due_unix'],elapsed_seconds=now-window['started_unix'],
        packet=str(out),states=states,live_workers=live,model_acceptance='NOT_ACCEPTED_AS_FULL_PATIENT_DUAL_PROPAGATION',
        continuation='Only the already authorized followup16 and multiseed108 budgets continue; G dispatch deadline unchanged; no new search, model freeze or Fig5.'))
    doc=MAIN/'docs/snn_model_report_current.md';old=doc.read_text()
    archive=MAIN/'docs/archive/topic4/snn_overnight_progress_2026-09-13.md'
    if archive.exists():raise RuntimeError('Refuse to overwrite the historical report archive')
    def absolute_link(m):
        target=m.group(1)
        if target.startswith(('/', '#')) or '://' in target:return m.group(0)
        path,sep,anchor=target.partition('#')
        return ']('+str((doc.parent/path).resolve())+(sep+anchor if sep else '')+')'
    archive.write_text('# 9月13日窗口结束前的SNN协作者报告\n\n原位置：'+str(doc)+'；下文保留当时的过程与历史结论，最新状态见当前报告。\n\n'+re.sub(r'\]\(([^)]+)\)',absolute_link,old))
    digest=hashlib.sha256(old.encode()).hexdigest()
    assert doc.read_text()==old,'Concurrent shared-report edit; preserve and review before replacing'
    current=f'''# SNN 当前协作者报告

**2026-09-13：01:32–09:32的8小时自主探索及审阅完成；既定物理批次仍在运行。** 当前接受部分参数—观测响应，尚不接受患者TA/TB完整传播恢复，未冻结模型或进入Fig5。

先看[本次科学判断]({out}/interpretation_addendum.md)、[统一图集PDF]({out}/overnight_review.pdf)和[新噪声多事件GIF]({W.parent}/core_multiseed_response_curves_20260913/analysis/first_noise_review/README.md)。图件已由Agent核查，仍待用户人工目视验图。

## 完成量与继续执行

{status}

以上为{stamp}快照；续跑响应含48条该批新运行及12条历史复用，原140条已完成，不能重复计算。实时状态与存活执行单元见[窗口交付记录]({W}/window_closeout.json)。已授权的确认与108条多网络响应继续至各自预算结束；方向探针09:32停止新增派发，已派发轨迹完整收尾。完成后停在审阅点。

## 当前科学判断

- 左核左移0.75mm在开发网络的两条开发噪声及一条新噪声下，改善TA上部SCL参与与杆间时差；仍须按每张网络自己的直接对照判断重复性。
- 扩大core、拉长／上移core或改变EE方向会在参与、时序和模式数量间产生取舍。扩大且匹配降阈值总量时，左核总发放增加，TA事件数量反而减少；不能只以覆盖电极更多为优。
- TA的时差中心改善后，完整事件散布仍不相符；TB多数事件仍在约33–35ms的跨杆延迟附近聚集，固定ICL局部顺序仍偏离患者。患者参考跨杆中位差约1.19ms；完整时差分布、患者STFT与原生GIF共同判断，不以两个标签或低分替代传播恢复。
- 核间先后与电极路径并非一一对应。新网络原位置的晚SCL尾部个例显示后段沿上缘传播；这是诊断个例，不证明全部事件同机制或边界反射。约半圈旋转候选也尚不能接受为稳定螺旋。

## 模型、观测与解释口径

见[共享设计口径]({MAIN}/docs/topic4_patient_geometry_prior_snn.md)。两核E阈值只降或保持背景，随机OU/Poisson输入限核；核外保留确定期望输入与递归传播。Z/M、空间OU、慢I、定向刺激关闭，GABA18ms及loss固定。冻结训练系数为全体联合特征0.25、模式条件0.25、参与组合0.50，各自先除以正尺度。TA/TB模式信息参与拟合，神经元输入不读取标签或指定细路线。

患者FIT：TA13,165、TB6,605。运行是实验单位，事件是运行内样本；患者分布、逐网络重复性与机制解释分开。上部SCL参与为SCL9/8概率平均；杆间差为每事件两杆参与触点质心的中位数之差，再汇总事件。部分旧图注误写均值，已更正，数值不变；见[勘误]({W}/rod_definition_caption_correction/correction.json)。患者STFT与模型发放包络不等价，固定分杆15行、缺失保留、实际毫秒轴不拉伸。

## 图与历史入口

- [全事件时差分布及均值／中位数／方差／范围]({W}/bridge_event_distributions/scientific_note.md)。
- [全部14种既有局部参数的配对作用]({W}/completed_parameter_effects/scientific_note.md)。
- [范围三点曲线与实际阈值剂量]({W}/radius_tradeoff_review/scientific_note.md)。
- [旧32条几何确认的网络依赖]({W}/geometry_confirmation_effects/scientific_note.md)。
- [窗口内过程及之前各版本报告]({archive})；[逐轨迹来源]({out}/completed_run_summary.csv)。
'''
    if paired.exists():
        current=current.replace('仍须按每张网络自己的直接对照判断重复性。','新网络2711上，左移使TA时差和参与分布误差变差，整体收益没有跨网络保留。')
        current=current.replace('## 模型、观测与解释口径','最新新网络配对见[逐项结果]('+str(paired)+')：参与增加和顺序误差降低部分同向，但患者相似度并非普遍改善；不能冻结该位置。\n\n## 模型、观测与解释口径')
    doc.write_text(current)
    window.update(status='EIGHT_HOUR_REVIEW_COMPLETE_BOUNDED_QUEUES_CONTINUE',reviewed_unix=now,
        latest_review=str(out.relative_to(W)/'interpretation_addendum.md'),latest_review_pdf=str(out.relative_to(W)/'overnight_review.pdf'),
        final_status='window_closeout.json',shared_report=str(doc),previous_shared_report_sha256=digest)
    write(W/'window.json',window)
    (W/'README.md').write_text('# 8小时自主探索：交付入口\n\n'+f'[科学报告]({out}/interpretation_addendum.md)、[图集]({out}/overnight_review.pdf)、[完成与运行状态](window_closeout.json)。\n\n'+status+'\n\n8小时窗口已结束，原授权有界队列继续；不能称为全部实验完成或双模式已恢复。\n')
    print(json.dumps(dict(packet=str(out),shared_report=str(doc),live_workers=len(live)),ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);main(p.parse_args().packet)
