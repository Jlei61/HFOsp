"""Close the bounded two-parameter spectral analysis, preserving unknowns."""
import sys, csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import fig5_z_gaba_phase_map as p
import numpy as np
from scipy import linalg
from scipy.optimize import brentq


def intersection():
    cid=p.b.base.rec.IDS[1];out=p.DATA/cid
    if (out/'two_mode_intersection.json').exists():return p.read(out/'two_mode_intersection.json')
    f=p.b.get_family(cid,'tau_gaba',True);seeds=p.read(out/'seeds.json')['rows'];xs=np.load(out/'seeds.npz')['rates'];history=[]
    def at(lam):
        seed=min(seeds,key=lambda r:abs(r['lambda']-lam));f.z_anchor=lam
        x,ok,err=f.solve(1.,xs[seed['index']])
        if not ok:raise RuntimeError('intersection equilibrium failed')
        c=p.DelayCharacteristic(f,x);a=c.crossing(20.,33/24);b=c.crossing(12.5,33/24)
        if not(a['converged'] and b['converged']) or abs(a['frequency_hz']-b['frequency_hz'])<2:raise RuntimeError('intersection mode identity failed')
        history.append(dict(**{'lambda':lam},first=a,second=b,equilibrium_residual=err));return a['scale']-b['scale']
    lam=brentq(at,1.7108914968962134,1.7308914968962135,xtol=1e-10);at(lam)
    r=history[-1];r.update(tau_gaba_ms=r['first']['scale']*24,evaluations=history[:-1],
        scope='Two distinct oscillatory zero-mode curves intersect; generic double-Hopf unfolding and periodic branches are not established.')
    p.write(out/'two_mode_intersection.json',r);return r


def slow_mode_audit():
    cid=p.b.base.rec.IDS[0];out=p.DATA/cid
    if (out/'slow_real_mode_check.json').exists():return p.read(out/'slow_real_mode_check.json')
    f=p.b.get_family(cid,'tau_gaba',True);f.z_anchor=2.;x=np.load(out/'seeds.npz')['rates'][12]
    c=p.DelayCharacteristic(f,x);par=60/18;r=c.follow_mode(par,0.,initial_growth_per_ms=-.002001)
    s=r['growth_per_ms'];m,z,z2,eta,tau,ops=f.at(par);n=m.n_cells
    vals,vec=linalg.eig(c.matrix(s,par));v=vec[:,np.argmin(abs(vals))];re,ri=v[:n],v[n:]
    mu=np.exp(s*.1);a=np.expm1(s*.1)/.1
    he=np.concatenate([re*mu**(-k) for k in range(1,ops.max_delay_steps+1)])
    hi=np.concatenate([ri*mu**(-k) for k in range(1,ops.max_delay_steps+1)])
    syn=[m.tau_mem_e_ms*(ops.w_ee_history@he)/(1+a*m.tau_ampa_ms),
         m.tau_mem_e_ms*(ops.w_ei_history@hi)/(1+a*m.tau_gaba_ms),
         m.tau_mem_i_ms*(ops.w_ie_history@he)/(1+a*m.tau_ampa_ms),
         m.tau_mem_i_ms*(ops.w_ii_history@hi)/(1+a*m.tau_gaba_ms)]
    full=np.concatenate([re,ri,*syn,he,hi,re/(a+1/tau)]);mat=p.corrected_delay_matrix(f,x,par)
    r['full_matrix_lift_relative_residual']=float(np.linalg.norm(mat@full-mu*full)/np.linalg.norm(full))
    r['finding']='A slower real mode was missed by the six converged LM eigenpairs. Do not equate eigensolver convergence with a complete leading spectrum.'
    p.write(out/'slow_real_mode_check.json',r);return r


def main():
    it=intersection();slow=slow_mode_audit();rows=[];boundaries=[];spectra=[];unknown=[]
    for cid in p.b.base.rec.IDS:
        p.observed_comparison(cid)
        for q in sorted((p.DATA/cid).glob('row_*.json')):
            r=p.read(q);rows.append(r)
            for c in r['crossings']:
                boundaries.append(dict(candidate_id=cid,mode='reference',**{'lambda':r['seed']['lambda']},
                    tau_gaba_ms=c['tau_gaba_ms'],frequency_hz=c['frequency_hz'],residual=c['residual']))
        for name in ['secondary_boundary','tertiary_boundary']:
            q=p.DATA/cid/(name+'.json')
            if not q.exists():continue
            r=p.read(q)
            if r.get('status')=='CONTINUATION_INCOMPLETE':unknown.append(dict(candidate_id=cid,branch=name,**r['failure']))
            for c in r['results']:
                boundaries.append(dict(candidate_id=cid,mode=name,**{'lambda':c['lambda']},
                    tau_gaba_ms=c['native']['tau_gaba_ms'],frequency_hz=c['native']['frequency_hz'],residual=c['native']['residual']))
        spectra.extend(p.read(q) for q in (p.DATA/cid).glob('spectrum_*.json'))
    with (p.DATA/'boundaries.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(boundaries[0]));writer.writeheader();writer.writerows(boundaries)
    stats=dict(z_rows=len(rows),reference_mode_grid_points=sum(len(r['points']) for r in rows),
        reference_zero_crossings=sum(len(r['crossings']) for r in rows),all_boundary_points=len(boundaries),
        full_matrix_samples=len(spectra),full_matrix_converged_samples=sum(q['converged'] for q in spectra),
        grid_tracking_failures=[q for r in rows for q in r['failures']],additional_branch_unknowns=unknown,
        max_boundary_characteristic_residual=max(q['residual'] for q in boundaries),
        stability_certification='NOT_COMPLETE: converged sparse modes may omit slower modes; positive eigenvalues certify instability, negative samples do not certify global stability.',
        two_mode_intersection=it,slow_real_mode_audit=slow)
    p.write(p.DATA/'analysis.json',stats)
    report=f'''# Z 累积损失 × GABA 动力学：二维分岔候选图

已完成第一轮有边界的二维谱分析：{stats['z_rows']} 个 Z 水平、{stats['reference_mode_grid_points']} 个参考模态网格点、{stats['reference_zero_crossings']} 个参考模态过零点；扩展追踪后共保存 {stats['all_boundary_points']} 个振荡边界点。6 个新的完整延迟矩阵抽查均返回收敛特征对，但不能据此宣称完整的最右谱已求全。

## 回答原始问题

可以把二维分岔候选图作为 Fig. 5E 的机制分析基础。当前图说明 Z 的累积损失与抑制动力学共同改变高活动平衡态的振荡稳定性；尚未解释间期活动为何进入 runaway。自然轨迹是否穿过吸引子失稳边界，与高活动平衡态自身存在什么 Hopf 候选，是不同的问题。

横轴 λ 定义每个 E 神经元的 Z = Zearly + λ (Zpre − Zearly)，保留空间格内一、二阶矩；λ=0/1 是原始轨迹早期/转变前检查点。背景网格的 GABA 范围为 12–60 ms。Z>1 的 λ 是检查点场的物理可行外推，不是声称患者或 SNN 自然沿该直线演化。其余图、阈值、EE/EI 强度、延迟及 M 参数沿用两套各自的基底。M 自洽且动态，Z 冻结；因此不能从这张图推断 τZ 的慢快分岔。

## 主要发现

1. 基底 1 原锚点 λ=1.3952：GABA 延长至 60 ms 后，仍有 +3.238 s⁻¹、12.53 Hz 的增长模态。更大的 Z 损失使振荡过零线出现。在 λ=1.470806，同一模态于 46.029 和 58.669 ms 两次过零，说明不能只画一条单调阈值线。λ=1.516165 的下边界为 37.322 ms；完整矩阵在 ±0.5 ms 两侧分别检出 +0.1145 和 −0.1096 s⁻¹ 的对应模态。
2. 基底 2 原锚点 λ=1.730891：参考模态在 32.331 ms 过零，另一个模态在 37.948 ms 过零；在 60 ms，完整矩阵检出 +11.298 s⁻¹、8.70 Hz 的增长模态。因此不能将参考模态变蓝的整个区域称为稳定。
3. 基底 2 的两条候选边界在 λ={it['lambda']:.9f}、GABA={it['tau_gaba_ms']:.6f} ms 相交，两个频率为 {it['first']['frequency_hz']:.3f} / {it['second']['frequency_hz']:.3f} Hz。这里只建立了两个不同振荡零模同时存在；尚未确定通用 double-Hopf 展开、周期轨道或其稳定性。
4. 基底 2 在 λ=2、GABA=60 ms 又检出 +1.078 s⁻¹、9.44 Hz 的增长模态。从该模态向回追踪，过零位于 55.920 ms。该第三条候选曲线向较小 λ 延伸至最后保存点 1.757802 后，下一目标 1.730891 的延续失败，保留未确定，末端不是生物学边界。

## 实际 onset 与相图是否接上

| 基底 | 原始 SNN 群体 runaway onset 的平均 Z 等效 λ | 已追踪高态折点 λ |
|---|---:|---:|
| 1 | 1.343613 | 1.365207 |
| 2 | 1.532581 | 1.700891 |

两者实际 onset 的平均投影均在该高态折点之前。三大空间区上的 Z 形状偏离扫描直线，分别约为早期到该时刻 Z 改变量的 2.48% / 3.87%；这只是粗粒度诊断，并非全神经元场一致性证明。当前不能声称自然的 IED→runaway 是穿过这些高态 Hopf 边界造成的，也不能仅凭不重合就否定 SNN 中存在其他分岔：扩散 LIF 降阶、固定 Z、空间形状与原始随机非平衡轨迹均可能影响对应。

## 稳定性审计与图的读法

颜色只编码一条连续追踪参考模态的增长率，紫线是该模态过零线，橙线是额外检出的振荡边界；虚线同时给出连续时间极限，避免把 native dt=0.1 ms 的精确数值当成连续时间阈值。新增边界均另存 dt 序列。部分延续线在追踪中超出 60 ms（最高约 91 ms），仅是边界延续扩展；背景网格仍止于 60 ms。

方块表示完整延迟矩阵的抽样特征对；橙色方块确证有增长模态，青色只表示已返回模态衰减。我们明确发现 eigs(which=LM) 即使收敛也会漏掉更慢模态：基底 1、λ=2、60 ms 的六个返回模态中最慢为 −12.541 s⁻¹，但另有 −2.001424 s⁻¹ 的实模态，提升回完整矩阵的相对残差为 {slow['full_matrix_lift_relative_residual']:.2e}。因此本轮不将任何仅靠负特征对抽样得到的区域标为“全系统稳定”。这也限制上一轮 sparse leading-spectrum 的全局稳定性措辞；正增长模态及经过核对的振荡过零本身不受此影响。

7 个既有数值测试通过，覆盖平衡残差导数、M 修正、路径增益、延迟矩阵一步线性化以及压缩特征方程与小系统完整谱的一致性。本图仍是二维谱候选图，不是完整吸引子相图；未完成周期轨道延续、第一 Lyapunov 系数、全右半平面根计数或原始固定 Z SNN 对照。是否适合正式 Fig. 5E，需要作者在这些范围内目视审阅。

## 复现入口

- `scripts/fig5_z_gaba_phase_map.py run`：两套基底的初始 26 条谱扫描，4 个并行 worker。
- `scripts/fig5_z_gaba_phase_map.py refine`：基底 1 非单调过零出现区间增加 7 条 Z 扫描。
- 该模块 `continuous_crossing(cid,index)`：每条参考过零线沿 dt 从 0.1 ms 到 0 延续。
- `scripts/fig5_z_gaba_phase_map.py spectrum --cid ... --index ... --tau ...`：独立完整矩阵抽查；不要将收敛标志当成最右谱完整性证明。
- `scripts/fig5_z_gaba_secondary_boundary.py`：从基底 2 原锚点 60 ms 的完整矩阵模态追踪第二边界；加 `--anchor-index 12 --output-name tertiary_boundary` 追踪第三边界。
- `scripts/analyze_fig5_z_gaba_phase_map.py`：生成审计、表和图；输出目录中的各 JSON 保留实际参数与数值残差。

方法层次可对照原始研究：[Visser et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3478171/)，其延迟神经网络分析也分别处理平衡稳定性、Hopf 类型与周期轨道，不能互相替代。
'''
    (p.DATA/'ANALYSIS.md').write_text(report)
    p.render()
    p.write(p.DATA/'status.json',dict(stage='BOUNDED_TWO_PARAMETER_SPECTRAL_ANALYSIS_COMPLETE_WITH_UNRESOLVED_BRANCH',
        z_rows=len(rows),grid_points=stats['reference_mode_grid_points'],additional_branch_unknowns=unknown,
        full_attractor_phase_diagram_complete=False,author_visual_acceptance=False))
    p.write(p.DATA/'delivery_manifest.json',dict(code={str(q):p.sha(q) for q in [Path(__file__),Path(p.__file__),Path(__file__).with_name('fig5_z_gaba_secondary_boundary.py')]},
        analysis=str(p.DATA/'ANALYSIS.md'),figure=str(p.FIG/'fig5-z-gaba-spectral-phase-map.png'),
        source_data=str(p.b.base.DATA),tests='7 numerical tests passed; all new boundary results retain residuals and dt continuation.',author_visual_acceptance=False))


if __name__=='__main__':main()
