"""Close the bounded kinetic-model validation batch only after all runs finish."""
import json
from pathlib import Path
import hashlib

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/topic4_sef_hfo/fig5_spatial_kinetic_equivalence_20260916'


def main():
    statuses={p.parent.name:json.loads(p.read_text()) for p in (OUT/'runs').glob('*/status.json')}
    assert len(statuses)==10 and all(d['status']=='COMPLETE' for d in statuses.values()),'batch still running'
    comparison=json.loads((OUT/'comparison.json').read_text())
    frozen=json.loads((OUT/'frozen_comparison.json').read_text())
    assert len(comparison['runs'])==8 and len(frozen)==2,'analysis does not yet cover all ten runs'
    qa={name:json.loads((OUT/'runs'/name/'qa.json').read_text()) for name in statuses}
    assert all(v['graph_identity_match'] and v['original_parameters'] and v['physical_Z_M_and_finite_state'] for v in qa.values())
    rows=[];events=[]
    def fmt(value, digits=3):return '不可估' if value is None else f'{value:.{digits}f}'
    for name,r in comparison['runs'].items():
        cfg=r['config']
        if cfg['duration_ms']<2000:continue
        state=r.get('resource_at_high_onset');p=r['event_propagation']
        rows.append(f"| {cfg['grid']}×{cfg['grid']} {cfg['closure']} | {cfg['seed']} | {state['candidate_onset_s']:.2f} | {state['candidate_D']:.4f} | {state['native_D']:.4f} | {state['Z_field_weighted_rmse']:.4f} | {state['Z_field_spearman']:.3f} |" if state else f"| {cfg['grid']}×{cfg['grid']} {cfg['closure']} | {cfg['seed']} | 未确认 | — | — | — | — |")
        events.append(f"| {cfg['grid']}×{cfg['grid']} {cfg['closure']} | {cfg['seed']} | {p['native_events']} / {p['candidate_events']} | {fmt(p['median_recruited_E_iou'])} | {fmt(p['median_arrival_order_spearman'])} | {fmt(p['median_offset_removed_mae_ms'],2)} |")
    fr=[]
    for r in frozen.values():
        n,k=r['native'],r['candidate'];sp=r['tail_spatial_field']
        fr.append(f"| {r['z_field_ms']/1000:.2f} | {n['finite_events']} / {k['finite_events']} | {n['quiet_fraction']:.3f} / {k['quiet_fraction']:.3f} | {n['mean_global_E_hz']:.2f} / {k['mean_global_E_hz']:.2f} | {n['tail500ms_mean_hz']:.2f} / {k['tail500ms_mean_hz']:.2f} | {sp['weighted_correlation']:.3f} |")
    text='''# 二维空间动理学候选模型：本轮科学审阅

原问题是：先找到能恢复同一原生SNN发作附近事件结构、两核/核外传播和资源控制关系的近似模型，再解释分岔。本轮模型族中已见到旧率模型缺失的自限事件、主要二维传播和固定Z两端动力学的恢复；这些证据分别属于下文明确列出的版本。完整自主轨迹及空间粒度稳健性尚未达到可称为“已经等效”的程度。本轮接受这条模型路线的可行性，不接受以它给原SNN命名分岔或宣告一个标量D临界值。

**本轮优先保留0.5mm mean闭合作为后续候选**：它保留脉冲历史、具有正确的固定图细化极限，两条自主续接均恢复自限事件到持续高率的过程；进入D为0.2581/0.2567，原生0.2563/0.2565，但时间仍晚0.31/0.43s。20×20 shot的冻结Z对照是该版本的证据，**不能直接算作40×40 mean已经通过冻结Z验收**。③附近的局部交替持续/全局不规则活动转换尚未由中间场验证，当前高率读出也不是一个同步程度指标。

## 实际换了什么

使用完整20×20mm二维空间，初筛20×20格、再细化为40×40格，最终优先保留后者的mean版本，每格区分E/I。用膜电位、双指数突触电流、不应期、Z/M的联合分布代替静态Phi；原实际图的四条E/I通路逐0.1ms延迟投影，群体输出的成串脉冲参与空间回返。逐细胞阈值和原外部OU/Poisson输入保留，未改变q_IE或其他生物参数。空间比较统一按原SNN的20×20观测格读出，不能把观测格误作0.5mm版本的计算格。

这轮先保留40000个分布粒子以隔离连接通信近似的影响。因此它是空间动理学/连接粗化原型，**尚不是已经降到800个状态、可以直接延续的低维ODE**。模型定义、各假设及方程在[model_definition.md](model_definition.md)，数值实现为`scripts/run_topic4_spatial_kinetic_candidate.py`。

自主续接中Z和M都动态更新，未来原生放电场与Z/M均不输入模型。冻结对照只固定Z，仍乘在GABA电流上，M继续更新。精确外部创新重放与额外回返抽样使用分开的RNG。

## 测了什么与统计单位

共10条：1mm两种闭合8–9s短程；1mm shot两条原参考输入8–12.5s；0.5mm shot两条输入8–10.5s；1mm shot两个冻结Z场各2s；根据空间细化极限诊断再补0.5mm mean两条8–10.5s。原生数据均来自既有逐位复现/移植实验，未改原SNN。两条参考噪声都是已看过的开发资料；格点、事件与成对比较不冒充独立网络样本。

补mean的理由是：源格与目的格各只有一个神经元时，已知源脉冲通过原固定边的输入应为确定J，但shot规则仍产生J·Poisson(1)。mean具有退回原固定图SNN的细化极限，shot额外抽样则没有；这不等于已经证明当前0.5mm误差就是这一项造成的，也不保证有限网格mean准确。追加范围与上限8→10的变更已写入执行方案，未用新生物参数调拟合。

全局率为E神经元加权平均、10ms读出。自限事件沿用低于5Hz持续≥20ms分隔且峰值≥20Hz的定义；进入高率是连续200ms达到全局200Hz。**此操作性进入点不等于分岔点。** 核率沿用原记录的1.75mm观测半径，图中圆是原1.5mm阈值核；二者未混作同一掩模。事件传播用5ms尾随局部率、持续5ms超过50Hz，已经在窗口开始活动的格点左删失；对相同早期核活动标签的全部事件对比较，没有取最相似的事件。

## 两端固定Z的对照

相同8s快速状态/M，统一10.37s的外部输入时钟；比较原生与候选各自共同的前2s。表中成对量均为“原生 / 候选”。

| Z来源(s) | 自限事件数 | 静息占比 | 全窗平均率(Hz) | 尾500ms平均率(Hz) | 尾场加权相关 |
|---|---|---|---|---|---|
'''+ '\n'.join(fr)+'''

低耗减场保留自限事件；高耗减场恢复持续活动及其斜向空间支持。高场的强空间相关是持续态场的一致性，不是已证明瞬时传播/所有状态都一致；低场的时间平均会受事件相位影响。此次2s结果也不能替代原10–20s分类、中间Z场、另一M历史或更多噪声的验证。

## 自主进入：时钟与资源坐标分开判断

原生两条轨迹的操作性进入点均为9.87s。新1mm模型晚约0.83–0.84s，说明完整自主推进有偏差；但进入时D与Z空间场明显比按同一绝对时刻比较更接近。下表中候选资源状态取进入10ms率窗之前最后一帧，未使用未来帧。

| 网格 | 输入 | 候选进入(s) | 候选D | 原生D | Z场加权RMSE | Z场Spearman |
|---|---|---|---|---|---|---|
'''+ '\n'.join(rows)+'''

D=1−〈Z_E〉只是整张场的摘要。即便这些进入D接近，也不能把不同空间场、M/快速历史和噪声压成普适D阈值；不能凭时钟偏差就断言快系统边界错了，也不能凭D接近就断言慢反馈和整个模型已等效。0.5mm两条轨迹用于查粒度敏感性，不把更细网格或单条更接近者自动选为已验证版本。

## 事件与二维传播

窗口统一8–9.42s。表中重叠/秩相关/误差是条件化的全部跨事件对的中位数，非独立样本检验。

| 网格 | 输入 | 原生/候选事件数 | 招募集合IoU | 到达秩相关 | 去整体偏移后误差(ms) |
|---|---|---|---|---|---|
'''+ '\n'.join(events)+'''

原生两条输入之间的相同读出也有很大事件差异，见`comparison.json`中的`native_noise_event_baseline`。两条输入与少量事件不足以建立事先噪声容限或总体等效检验。首个事件的二维帧显示空间推进可恢复，但它只是示例，不独自承担验收；按进入点对齐的图与绝对时钟图都保留，避免只展示有利对齐。

## 原因判断与下一步

保留膜电位/不应历史和时间组织的回返输入后，在未调生物参数的情况下恢复了自限burst，这支持旧平稳率闭合遗漏了关键快动力学。现在的残差可能涉及格内固定连接异质性、共享输入相关和Z对抑制电流阈值占据的积累；这些是有针对性的候选解释，当前实验未把它们逐一归因。

还做了一个反证诊断：保持原生逐细胞抑制电流轨迹不变，比较“逐细胞过阈值”与“先取格内平均电流再过阈值”。用共同10ms抽样推进Z，在原生9.87s处，1mm空间平均只使Z再降低约0.00061–0.00084，0.5mm只降低0.00018–0.00021；方向是略加快耗减，不能解释候选自主推进偏晚。10ms采样自身最大重构误差约0.00065–0.00077。结果在`z_threshold_averaging.json`，这只排除单纯算术平均误差作为主要解释；并未隔离自主网络反馈和输入相关误差。因此不应直接改Z时间常数来补时间差。

下一版仍应沿有完整二维场和快历史的动力学路线，优先检查：

1. 对选中的0.5mm mean先补8s/9.87s两端，再用原生中间Z场9.30/9.42s及两种M/快速历史，比较相同条件下的局部持续、两核交替与全局招募；目标是当前③附近，不能只凭别的版本的两端验收。
2. 将资源推进与快响应分开：同一冻结Z场的快过程若相符，而自主Z不同，检查抑制电流越阈占比及其时间相关；若冻结Z下也不符，检查连接粗化和格内异质性。保留同一物理参数，避免把SNN调成适合近似的网络。
3. 在已匹配的工作点冻结闭合，再用新的原生噪声建立容限，验证事件变异、二维招募和小扰动响应；增加网格/减少粒子都需单独检验，不能仅因图更平滑而接受。
4. 上述通过后，沿完整物理Z场的指定路径做条件分岔，明确M动态/冻结条件；可延续对象应是闭合群体演化方程或经过Markov/记忆检验的粗粒化演化算子。粒子轨迹的峰谷、一次高率进入和D≈0.26均不提供saddle-node/Hopf的类型证据。

本轮不重用旧率模型的临界点标签，不进入新的延续计算。接受模型路线的可行性和两端有限窗恢复；完整科学等效与分岔解释仍未通过。

## 交付与核查

数据与逐项比较在`runs/*`中的NPZ/QA及逐条件比较、`comparison.json`、`frozen_comparison.json`。10条都有原参数和物理Z/M核查；输入重放、实际延迟卷积、非负跳变矩、已知二维波及左删失测试见`numerical_qa.json`。最初两条短试验在NPZ保存后发生NumPy布尔量JSON序列化错误，元数据已修复并用纯输入重放重新核验，没有重跑或替换其动力学数据；不能将该历史说成0工程错误。

主要图均有PNG/PDF/SVG与[中文说明](figures/README.md)。优先候选的时序、同钟空间图及关键冻结Z图已完成Agent目视自查，具体范围见`visual_qa.json`；用户人工图形验收仍待定。此目录独立于原生结果和其他线程的图，不替换正式图。

方法背景：[Schwalger, Deger & Gerstner (2017)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005507)讨论保留不应与有限群体涨落的群体动力学；本项目的空间图/双指数电流闭合是候选实现，不能继承该文的验证结论。
'''
    (OUT/'scientific_report.md').write_text(text)
    code={str(ROOT/'scripts'/name):hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in (
        'run_topic4_spatial_kinetic_candidate.py','topic4_kinetic_delay.py','prepare_topic4_kinetic_grid.py',
        'analyze_topic4_spatial_kinetic_candidate.py','write_topic4_kinetic_review.py','diagnose_topic4_kinetic_z_averaging.py')}
    (OUT/'provenance.json').write_text(json.dumps(dict(source=str(ROOT/'results/topic4_sef_hfo/fig5_current_network_z_state_v1'),
           upstream_audit=str(ROOT/'results/topic4_sef_hfo/fig5_native_reduction_correspondence_20260916'),
           script_sha256=code,particle_count=40000,original_graph='topology6101, original q_IE=1',
           coupled_space_mm=[20,20],grid_resolutions_mm=[1,.5],Z_M='both dynamic except explicitly frozen Z controls'),indent=2)+'\n')
    (OUT/'status.json').write_text(json.dumps(dict(status='EXECUTION_COMPLETE',runs=statuses,numerical_qa='PASS',
          native_correspondence='PARTIAL_RECOVERY_NOT_FULL_EQUIVALENCE',scientific_acceptance='NOT_ESTABLISHED',
          preferred_next_candidate='grid40_mean',particle_reduction='NOT_YET_TESTED',
          frozen_Z_scope='two endpoints, one history, one input, first2s',bifurcation='NOT_STARTED_DEPENDENCY',
          human_visual_acceptance='PENDING'),indent=2)+'\n')
    print('Ten runs reviewed; partial recovery, equivalence NOT_ESTABLISHED; no bifurcation launched.')


if __name__=='__main__':main()
