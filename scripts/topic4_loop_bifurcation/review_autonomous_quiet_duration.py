#!/usr/bin/env python3
"""Explain the observed long quiet interval from the native K and Z laws."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import shutil
import numpy as np
from campaign import ROOT,read,write,sha
import analyze_constant_tau_return as a

OUT=ROOT/'autonomous_quiet_duration_review'


def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'result.json').exists()
    start, end=16.868,49.31
    m=a.part(a.BASE,'mechanism_chunks','time_ms',['global_E_rate_Hz','global_raw_conductance_ratio'],16.8,49.4)
    k=a.part(a.BASE,'intrinsic_adaptation_chunks','time_ms',['sahp_mean_conductance_ratio'],16.8,49.4)
    z=a.part(a.BASE,'chunks','slow_time_ms',['Z'],16.8,49.4)
    assert np.array_equal(k['time_ms'],m['time_ms'])
    t=m['time_ms']/1000;R=m['global_E_rate_Hz'];G=m['global_raw_conductance_ratio'];K=k['sahp_mean_conductance_ratio']
    lo,hi=np.searchsorted(t,[start,end]);assert abs(t[lo]-start)<1e-10 and abs(t[hi]-end)<1e-10
    # Backward bound from each following 1ms sample: every omitted spike adds
    # nonnegative R, so an intervening value cannot exceed R(right)*exp(1/15).
    upper=float(R[lo+1:hi+1].max()*np.exp(.001/.015))
    assert upper<5
    prediction=K[lo]*np.exp(-(t[lo:hi+1]-start)/5)
    relative=float(np.max(abs(K[lo:hi+1]/prediction-1)))
    assert relative<1e-9
    effective=float(5*np.log(K[lo]/K[hi]));assert abs(effective-(end-start))<1e-8
    block=95.19851312666987/(18+17.662847938268442)
    predicted_G_clear=start+.5*np.log(G[lo]/block)
    measured_G_clear=float(t[lo+np.flatnonzero(G[lo:hi+1]<block)[0]])
    assert abs(predicted_G_clear-measured_G_clear)<=.0011
    times=z['slow_time_ms']/1000;Z=z['Z'][:,[5,6]]
    # Choose the first stored post-clear resource sample before checking the
    # original discrete fastest-recovery law throughout the entire quiet tail.
    iz=np.searchsorted(times,measured_G_clear,side='right');jz=np.searchsorted(times,end,side='right')
    steps=np.rint((times[iz:jz]-times[iz])*10000).astype(int)
    bound=1-(1-Z[iz])[None,:]*(1-.1/5000)**steps[:,None]
    errors=np.max(abs(bound-Z[iz:jz]),axis=0)
    assert np.max(Z[iz:jz]-bound)<1e-9
    global_start=float(times[iz]);global_errors=errors.copy()
    budget=a.part(a.BASE,'z_budget_chunks','time_ms',['values'],17.5,end,right=True)
    bt=budget['time_ms']/1000;loss=budget['values'][:,1:3,3]
    nonzero=np.flatnonzero((loss>1e-12).any(1))
    full_start=float(bt[nonzero[-1]])
    assert ((bt>full_start).any() and (loss[bt>full_start]<1e-12).all())
    iz=np.searchsorted(times,full_start);assert abs(times[iz]-full_start)<1e-10
    steps=np.rint((times[iz:jz]-times[iz])*10000).astype(int)
    bound=1-(1-Z[iz])[None,:]*(1-.1/5000)**steps[:,None]
    errors=np.max(abs(bound-Z[iz:jz]),axis=0);assert max(errors)<1e-9
    reference=np.array(read(ROOT/'natural_exit_mediator_probes/mediator_analysis/result.json')['rows'][0]['core_reference'])
    minimum=np.log((1-reference)/(1-Z[iz]))/np.log(1-.1/5000)/10000
    result=dict(status='COMPLETE_NATIVE_QUIET_TAIL_EQUATION_CHECK',interval_s=[start,end],duration_s=end-start,
        continuous_R_upper_bound_Hz=upper,K_start=float(K[lo]),K_at_first_return=float(K[hi]),
        K_exact_exponential_max_relative_error=relative,observed_K_log_ratio_times_tau_s=effective,
        G_blocking_bound=block,G_start=float(G[lo]),G_unblocking_predicted_s=predicted_G_clear,G_unblocking_observed_s=measured_G_clear,
        first_post_global_clear_Z_sample_s=global_start,maximum_recovery_bound_gap_at_that_start=global_errors.tolist(),
        complete_core_eligibility_interval_start_s=full_start,Z_prediction_start_s=float(times[iz]),core_Z_prediction_start=Z[iz].tolist(),core_reference=reference.tolist(),
        fastest_discrete_core_reference_elapsed_s=minimum.tolist(),core_Z_max_discrete_prediction_error=errors.tolist(),
        core_Z_last_sample_before_return=Z[jz-1].tolist(),
        interpretation='Measured32.442s low-activity interval follows the original5s K decay exactly and has no timed hold. The K value at the first return is measured retrospectively, not a prespecified universal release threshold or an independent onset-time prediction. G crossing is necessary but not immediately sufficient for every core cell: small residual CoreA consumption remains in the first post-clear budget. After the last nonzero-consumption block, both core means follow the maximum-recovery discrete law through the remaining quiet segment. Other exits, backgrounds and intervening events need separate assessment.',
        statistical_unit='One existing autonomous native episode; no new simulation, seed or loop count.',
        formal_bifurcation_allowed=False,producer_sha256=sha(__file__))
    write(OUT/'result.json',result);shutil.copy2(__file__,OUT/'producer.py')
    (OUT/'review.md').write_text(f'''# 长低活动窗由原式衰减产生\n\n这一原生片段从{start:.3f}秒进入低率，到{end:.3f}秒首次返回短事件，持续{end-start:.3f}秒。1毫秒样本之间的因果率严格上界仍小于5Hz；因此整段一直处在原式的5秒K衰减分段，不存在未记录的高率累积或快速衰减切换。\n\n原始K从{K[lo]:.4f}降到{K[hi]:.5f}，全段逐点满足K(t)=K0 exp(−t/5秒)，最大相对误差{relative:.2g}。5 ln(K0/K返回)恰为观测持续时间。此处K返回是事后观测值，不是模型内的释放门，也不是无需噪声信息即可预测返回时刻的普适阈值。\n\nG从{G[lo]:.4f}消退到允许资源恢复的必要上限2.6694，解析预计{predicted_G_clear:.6f}秒，记录为{measured_G_clear:.3f}秒。G越过上限后，CoreA短暂仍有局部负荷；从{full_start:.2f}秒起，预算确认两核消耗项均为零，之后核心Z在余下完整低活动尾部遵循原离散恢复方程的最快曲线；这不是重置Z，也不是人为规定恢复目标。两核达到原间期参考后还继续恢复到接近1，才出现首次返回。\n\n该解释把“终止活动”“解除恢复阻断”“积累资源余量”“活动重新出现”按实际时间顺序连接起来。它只说明这一个已有自主片段，不增加闭环数，不认证正式分岔。\n''')
    print(result,flush=True)


if __name__=='__main__':main()
