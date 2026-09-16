"""Finalize only after actual bounded samplers end; preserve all draft/history files."""
import sys,time,json,subprocess,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.patient_state_v1.common import ROOT,RUN,write_json
start=1788968996.;now=time.time();elapsed=(now-start)/3600
assert 8<=elapsed<=10,elapsed
paths={'ou':'particle_posterior_v1_2/ou','ou_history':'particle_posterior_v1_2/ou_history','two_ou_history':'two_scale_particle_posterior_v1_15_16chains'}
for proc in Path('/proc').glob('[0-9]*/cmdline'):
    try:c=proc.read_bytes().replace(b'\0',b' ')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert not any(x in c for x in [b'patient_state_v1/particle_mcmc.py --',b'patient_state_v1/two_scale_posterior.py --']),('sampler_live',str(proc))
meta={n:json.loads((RUN/p/'checkpoint.json').read_text()) for n,p in paths.items()}
assert all(m['status'] in ['COMPLETE','CHECKPOINTED_TIME_LIMIT'] for m in meta.values()),meta
subprocess.run([sys.executable,str(ROOT/'scripts/patient_state_v1/review_posteriors.py'),'--plots'],check=True)
diag=json.loads((RUN/'posterior_diagnostics/status.json').read_text());diag['status']='FINAL_TERMINAL_DIAGNOSTICS';write_json(RUN/'posterior_diagnostics/status.json',diag);by={r['model']:r for r in diag['results']}
for n,m in meta.items():assert by[n]['iteration']==m['iteration'] and by[n]['sampler_status']==m['status']
subprocess.run([sys.executable,str(ROOT/'scripts/patient_state_v1/compare_posterior_methods.py')],check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/patient_state_v1/delivery_audit.py')],check=True)
assert json.loads((RUN/'joint_grid_refit_v1_26/closeout.json').read_text())['status']
lines=[]
for n in paths:
    r=by[n];lines.append(f"| {n} | {r['iteration']} | {r['sampler_status']} | {r['status']} | {max(s['r_hat'] for s in r['summary']):.5f} |")
table='| 采样器 | 最终迭代 | 实际终点 | 数值验收 | 最大Rhat |\n|---|---:|---|---|---:|\n'+'\n'.join(lines)
endtext=f'本次实际探索{elapsed:.2f}小时（起点2026-09-09 23:49:56，北京时间；完成核对时刻{time.strftime("%Y-%m-%d %H:%M:%S")}），在用户8–10小时范围内。所有本任务采样器已实际退出；完成、时间预算退出和数值验收分开报告。直接联合网格、新生成、边界/初态校准和最终源数据复核均已完成。图已做Agent检查，尚未经过用户人工图面验收。'
s=(RUN/'scientific_review_draft.md').read_text().replace('夜间第一版科学审阅（执行中草稿）','夜间第一版科学审阅')
s=re.sub(r'这是一份执行中的汇总稿。[^\n]+',endtext,s,count=1)
s=s.replace('普通OU与双时间OU长链的终点仍待更新。','长链实际终点与数值资格见末尾终点表。')
s=s.replace('![当前患者状态回顾](figures/patient_state_recap.png)','![完整患者状态回顾与平滑](figures/patient_state_filter_and_smoother.png)')
s=s.replace('图中状态参数来自全记录后验中位数，曲线仅消费各查询时刻以前已完成的标签；因此是','图中参数来自全记录后验中位数；过滤曲线仅消费查询前已完成的标签，最下行平滑则使用未来标签。过滤部分属于')
insert='概率校准使用相同16,157个前推事件。常数、近期比例记忆、OU和OU-history的Brier分数分别为0.23086、0.22077、0.22039和0.22041；OU与简单记忆接近，不能凭这项描述宣布复杂模型额外优势。固定0.1宽度的概率箱及5,000次按折分层的6小时块重采样已保存；仅14个测试块，区间是开发记录上的描述性不确定性。见[概率校准](final_observation_audit_v1_40/calibration_summary.csv)。\n\n'
s=s.replace('纳入参数不确定性后，两种基础模型各生成512条',insert+'纳入参数不确定性后，两种基础模型各生成512条')
s+='\n\n## 7. 实际计算终点与接受状态\n\n'+table+'\n\n两组基础OU参数的主要区间来自已经通过其自身预定门槛的独立粒子重要性抽样；未通过的MCMC不能被合并称为数值通过。双时间后验的数值限制不等于已经证明该生理机制不可识别。详情见[终点诊断](posterior_diagnostics/status.json)。\n\n'+endtext+'\n'
(RUN/'scientific_review.md').write_text(s)
r=(RUN/'read_first.md').read_text().replace('主要数据实验已完成，整夜目标仍待长链终点与最终收口；本页是候选结果入口。',f'本轮实际探索{elapsed:.2f}小时，计算与自审已收口；本页是结果入口。').replace('[科学审阅草稿](scientific_review_draft.md)','[完整科学审阅](scientific_review.md)');(RUN/'read_first.md').write_text(r)
a=(RUN/'completion_audit_draft.md').read_text();a=a.replace('原始目标逐项验收：运行中草稿','原始目标逐项验收：最终候选交付');a=re.sub(r'当前不是完成声明。[^\n]+',endtext,a,count=1)
a=a.replace('主要OU参数已完成；长链最终状态尚待更新，复杂模型不预先接受','主要OU参数已完成；长链最终状态按下表分别验收，复杂模型不因计算结束而接受')
a=a.replace('持续完成中；初态与直接网格检验已关闭，新增读出解释无新参数','已完成；初态、直接网格、条件组成与校准检查均关闭')
a=a.replace('三组长链与联合网格仍活跃','三组长链与联合网格均已到实际终点').replace('尚未到时限；不能用预定时长替代实际执行',f'实际{elapsed:.2f}小时，全部本任务采样器已退出')
a=a.replace('草稿已交付；最终报告、最终源码快照和终点审阅尚未完成','最终科学报告、源码快照及终点诊断已生成；代表图待用户人工验收')
a=a.replace('`scientific_review_draft.md`已汇总','`scientific_review.md`已汇总')
a=re.sub(r'最终完成前必须重新核对：[^\n]+','最终已核对实际采样终点、全部生成关闭结果、输入重读一致性与图面文件完整性。阴性检验已经完成，机制验收仍以科学报告中的明确边界为准。',a)
a+='\n\n'+table+'\n\n新增交付：`final_observation_audit_v1_40`保存四个固定模型的概率校准及当前坐标完整过滤/平滑时间线；全部12发作的小图保存在`figures/joint_preseizure_state_all12.png`，两TB例另有完整网格数值核对。\n'
(RUN/'completion_audit.md').write_text(a)
write_json(RUN/'completion_audit.json',dict(status='COMPLETE_SCIENTIFIC_PROTOTYPE_WITH_LIMITATIONS',started_unix=start,completed_unix=time.time(),elapsed_hours=(time.time()-start)/3600,user_budget_hours=[8,10],all_task_samplers_exited=True,samplers=meta,numerical_diagnostics=[dict(model=r['model'],status=r['status'],iteration=r['iteration'],max_rhat=max(s['r_hat'] for s in r['summary'])) for r in diag['results']],source_audit='input_delivery_audit.json',figure_audit='figure_file_audit.json',scientific_report='scientific_review.md',snn_core_r1_zm_unchanged=True,human_visual_acceptance='PENDING',accepted='Statistical continuous mode-state prototype and explicit tested limitations; not DDM threshold, biological reset, unknownK or SNN physical parameter identification'))
print(json.dumps(dict(status='COMPLETE',elapsed_hours=(time.time()-start)/3600,report=str(RUN/'scientific_review.md'))))
