#!/usr/bin/env python3
"""Close the bounded numerical package, leaving scientific/visual review to the agent/user."""
from plot_topic4_z_bifurcation_audit import OUT,ROOT,REF,FIG,native_controls,diagrams,smooth
from validate_topic4_fixed_rate_base import read,write
import numpy as np
import time
import subprocess
import sys


def main():
    while read(OUT/'native_batch_status.json')['status']=='RUNNING':time.sleep(10)
    batch=read(OUT/'native_batch_status.json');assert batch['status']=='COMPLETE' and batch['completed']==10
    checks={}
    for filename in ['external_input_qa.json','rate_equation_qa.json','rate_tangent_qa.json','conditional_phase_qa.json','critical_plane_plot_qa.json']:
        assert read(OUT/filename)['status']=='PASS',filename;checks[filename]='PASS'
    assert read(OUT/'native_resume_qa.json')['exact_resume_verified']
    assert read(OUT/'replay_status.json')['all_observed_steps_identical']
    assert all(read(OUT/'phase_replay_status.json')['identity'].values())
    rows=[]
    for row in batch['rows']:
        a=np.load(OUT/'native'/f"{row['name']}.npz");assert row['count_conservation']
        z0=a['initial_z_e'];zf=a['final_z_e'];assert np.isfinite(zf).all() and zf.min()>=0 and zf.max()<=1
        if row['mode']=='frozen':assert np.array_equal(z0,zf)
        rows.append({k:row[k] for k in ['name','initial_Z_mean','final_Z_mean','late_E_mean_hz','late_E_quiet_fraction','late_E_cv']})
    rb=read(OUT/'rate_batch_status.json');assert rb['status']=='COMPLETE' and len(rb['rows'])==8
    for row in rb['rows']:
        a=np.load(OUT/'rate'/f"{row['name']}.npz");v=a['fields_hz'];assert np.isfinite(v).all() and v.min()>=0 and v[:,0].max()<=500.0001 and v[:,1].max()<=1000.0001
    native=np.load(REF/'trajectory.npz');noref=np.load(OUT/'native/autonomous_no_refill_t10680.npz')
    paired=[]
    for name,e in [('external_refill',native['rate_e_hz'][106800:126800]),('continue_Z_ODE',noref['rate_e_hz'])]:
        late=smooth(e[10000:],50)
        paired.append({'condition':name,'absolute_window_s':[11.68,12.68],'E_mean_hz':float(late.mean()),'quiet_fraction_5ms':float(np.mean(late<1))})
    native_controls();diagrams()
    manifest=[]
    for png in sorted(FIG.glob('*.png')):
        pdf=png.with_suffix('.pdf');assert pdf.exists()
        info=subprocess.run(['pdfinfo',str(pdf)],check=True,capture_output=True,text=True).stdout
        pages=int(next(line.split(':')[1].strip() for line in info.splitlines() if line.startswith('Pages:')));assert pages==1
        manifest.append({'png':str(png),'pdf':str(pdf),'pdf_pages':pages})
    report={'status':'READY_FOR_AGENT_VISUAL_REVIEW','native_runs':rows,'paired_counterfactual':paired,'checks':checks,'figures':manifest,
            'scope':'Completed bounded candidate package; no native SNN bifurcation type is assigned and no formal figure/model is frozen.',
            'acceptance':{'native_autonomous_entry_external_return':'SUPPORTED_AT_REFERENCE_WORKING_POINT',
                          'manual_restore_causal_counterfactual':'SUPPORTED_ON_PAIRED_TRAJECTORY',
                          'reduced_fast_response':'QUALITATIVE_CAPABILITY_SUPPORTED',
                          'reduced_autonomous_Z_quantitative_match':'NOT_PASSED',
                          'low_activity_complex_crossing':'NUMERICALLY_SUPPORTED_IN_DELAYED_RATE',
                          'native_sustained_recruitment_equals_that_Hopf':'NOT_ESTABLISHED',
                          'all_global_attractors_or_closed_EI_nullclines':'NOT_ESTABLISHED',
                          'human_visual_acceptance':'PENDING'}}
    write(OUT/'closeout.json',report)
    md=FIG/'README.md';text=md.read_text()
    extra={
      'native_frozen_z_raster_and_recruitment.png / .pdf':'从 5 个真实检查点连续携带全部快状态并冻结各自逐神经元 Z，展示 2 s 延续、最后 0.5 s 的真实 spike raster 和最后 1 s 空间平均 E rate。蓝绿色圆圈只标示固定双核位置；它们不是新的刺激或修改后的 core。**关注点**：安静间隔的消失早于数百 Hz 高率，空间招募随状态改变；2 s 结果不能当作渐近稳定性证明。',
      'native_counterfactual_and_sensitivity.png / .pdf':'上部从相同的 10.68 s 完整状态和相同未来随机输入出发，对比补回 Z 与保留原 Z 自主演化；下部在同一 9.8 s Z 场上比较小电位扰动和两组新未来噪声。各轨迹仅改变标明的条件，膜/突触/延迟状态连续携带。**关注点**：返回是否由 Z 干预造成，以及持续活动能否在有限扰动与有限新噪声样本下保留；这不是完整吸引域或噪声稳健性扫描。',
      'reduced_z_gaba_stability_diagram.png / .pdf':'展示 E-only Z 的已追踪平衡支、有限时间动态范围、低率复模态越界曲线及高率端部分特征根。第二个 Z 场族由真实检查点插值，曲线只表示已追踪模态的局部越界，不是全部吸引子的全局相图；短 GABA 衰减端的零频退化未定性。**关注点**：低率失稳、局部振荡、空间招募与高率平台不能合并成一个未经验证的 Hopf 叙事。'}
    for name,description in extra.items():
        if f'### {name}' not in text:text+=f'\n\n### {name}\n\n{description}\n'
    md.write_text(text)
    print('Numerical package ready; agent must inspect new figures and finish scientific closeout.',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        write(OUT/'closeout.json',{'status':'FAILED','error':repr(exc)});raise
