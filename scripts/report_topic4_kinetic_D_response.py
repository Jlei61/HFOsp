"""Scientific disposition of fixed-D response, with explicit classical gap."""
import json
import hashlib
from pathlib import Path
from topic4_kinetic_D_response import OUT, ROOT

LABELS={'self_limited_events':'自限事件','persistent_intermediate_activity':'中等强度持续',
    'broad_high_activity':'广泛高活动','low_activity':'低活动','unresolved':'未定'}


def main():
    summary=json.loads((OUT/'response_summary.json').read_text());rows=summary['runs']
    states=[json.loads(p.read_text()) for p in (OUT/'runs').glob('*/status.json')]
    complete=len(rows)==24 and len(states)==24 and all(s['status']=='COMPLETE' for s in states)
    comparisons=json.loads((OUT/'native_anchor_comparison.json').read_text())
    pair=lambda d:[next(r for r in rows if r['D']==d and r['history_ms']==h) for h in (8000,10370)]
    lines=['# 当前简化模型的固定 D 响应：科学报告','',
        '**已计算的对象是g40 mean空间动理学模型自身的有限时间参数响应。经典稳定/不稳定平衡分支、周期分支与分岔类型仍未建立。**', '',
        '这张图回答：沿一条指定的物理空间Z场路径增加耗减，候选何时从会自行终止的活动进入局部/广泛持续活动，以及这种响应是否依赖完整初始历史。它不把原图的一条随时间漂移轨迹直接当作分岔曲线。', '',
        '## 实际模型与实验','',
        '仍为原拓扑6101、q_IE=1、0.5mm计算格、40000分布粒子、0.1ms步长和原延迟；没有改工作点、tau或传递函数。原完整Fig.5布局图中Z和M双动态；本轮为定义参数，固定E粒子的Z_i(D)，M始终动态。', '',
        '路径为Z_i=(Z_i^ref)^alpha，Z^ref来自原Fig.5的9.42s逐细胞场；alpha由D=1−mean_E Z反解。D=0/1分别对应E的Z=1/0，所有空间资源都在0–1。D≈0.228844761复现该原图③的场；其他D是此明确场族的截面，不是原时间轨迹的重放。I的Z=1/M=0。', '',
        f'当前包含{len(rows)}个已保存4s响应（目标12个D×2历史）；其中{sum(r["duration_s"]>=8 for r in rows)}条已延长至8s。全部来自同一未来输入W1/seed9108401，从统一10.37s输入时钟出发。初始历史分别保留原8.00/10.37s完整快状态、M和延迟待到达输入，之后活动完全自主。输入实现而不是时间bin才是随机重复单位，本轮没有跨输入显著性或等效性结论。', '',
        '## 关键定位','',
        '当前模型在D=0到0.20都保留自限事件，不能把这一侧画成已经证明的低率平衡点。短窗D≈0.225–0.234覆盖最早的自限消失区域：0.225两历史仍会停下，原③场D≈0.228845出现历史依赖，0.234134两历史在共同2–4s窗口均持续。', '',
        '**8s延长改变了上侧判断：D≈0.234134的低历史在6–8s窗口重新出现120ms低活动，不能视作已确定的持续侧临界点。** 两种历史均经延长支持的外侧参照是D=.225仍自限、D=.25持续；中间D≈.229–.234具有历史和观察窗依赖。不能把短窗的类别跳变提前标成SN，也不能把这个有限时间区间当作已定位的数学分岔。', '',
        '再增加D时，空间招募继续扩大：D=.25的主窗全局均值约197Hz、平均空间占据约56%，D=.30约297Hz、占据约77%。这些结果把自限消失与随后空间招募的增长分开了；某个放电率/占据阈值不能自动成为新的分岔。', '',
        '本轮为离散参数响应，点间没有延续连线；未采样参数既不代表无解，也不构成平衡分支断裂。高耗减一侧以宽范围点为参照，计算密度集中在用户关注的早期转变。', '',
        '## 同窗口结果','',
        '每格斜线前/后分别是低历史/高历史；均使用固定后的2–4s同一未来输入窗口。放电率是每个E神经元的Hz，不是爆发重复频率；空间占据是1mm观测格率≥50Hz的E细胞数加权比例，再对时间平均。', '',
        '| D | 全局均值 Hz | 低活动比例 | 空间占据比例 | 状态 |',
        '|---|---:|---:|---:|---|']
    for d in sorted(set(r['D'] for r in rows)):
        if sum(r['D']==d for r in rows)!=2:continue
        pp=[r['first_terminal'] for r in pair(d)]
        fmt=lambda k,scale=1:f'{pp[0][k]*scale:.1f} / {pp[1][k]*scale:.1f}'
        lines.append(f'| {d:.6f} | {fmt("mean_E_hz")} | {fmt("quiet_fraction",100)}% | {fmt("mean_occupied_E_fraction",100)}% | '+
            ' / '.join(LABELS[p['category']] for p in pp)+' |')
    lines+=['','状态是预先写定的有限窗读出，不是吸引子认证。“中等强度持续”涵盖局部持续到较大范围招募；“广泛高活动”额外要求至少90%时间全局率≥200Hz，且至少90%时间有80%以上的E加权观测格≥50Hz。它仍不是同步性指标。','',
        '## 延长检查','',
        '| D | 历史 | 2–4s 均值/静息比例 | 6–8s 均值/静息比例 | 后窗状态 |',
        '|---|---|---:|---:|---|']
    for r in rows:
        if r['duration_s']<8:continue
        a,b=r['first_terminal'],r['terminal']
        lines.append(f'| {r["D"]:.6f} | {r["history_ms"]/1000:.2f}s | {a["mean_E_hz"]:.1f}Hz / {a["quiet_fraction"]:.1%} | {b["mean_E_hz"]:.1f}Hz / {b["quiet_fraction"]:.1%} | {LABELS[b["category"]]} |')
    lines+=['','每一行的两个窗口不是独立重复。M仍动态；后窗差别只能称有限时间历史依赖，不能由两条轨迹直接认证双稳态、separatrix或saddle-node。','',
        '## 与原SNN在③同场的对应','',
        '复用原生冻结9.42s场、相同快历史的已完成对照，W1与本轮未来输入配对，W2提供原生噪声背景。此比较未用于驱动候选的未来活动。','',
        '| 历史 | 窗口 | 原生输入 | 原生均值 / 静息比例 | 候选W1均值 / 静息比例 |',
        '|---|---|---|---:|---:|']
    for p in comparisons['pairs']:
        a,b=p['native'],p['candidate']
        lines.append(f'| {p["history_ms"]/1000:.2f}s | {p["window_s"][0]}–{p["window_s"][1]}s | {p["native_input"]} | {a["mean_E_hz"]:.1f}Hz / {a["quiet_fraction"]:.1%} | {b["mean_E_hz"]:.1f}Hz / {b["quiet_fraction"]:.1%} |')
    lines+=['','高历史后窗的率与空间占据接近原生；低历史候选仍能停下，而两个原生输入在该后窗均未出现全局低活动。这一处临界响应尚未确认等效，不能把候选图的区间直接写成原SNN的分岔位置；两条原生输入也不足以构成统计容差。','',
        '## 图义、QA与验收','',
        '主图无总标题、无灰色小字注释、无蓝色原生时间轨迹，显示共同2–4s时间均值及10–90%时间分位范围。它们没有被标为平衡、稳定周期、周期极值或不稳定分支；因此没有采用会暗示Floquet稳定性的实心/空心周期方块，也没有未经判型的星号。空间图均来自真实模拟50ms窗口，统一色标0–500Hz；第四图使用最早已采样且持续、平均空间占据≥50%的低历史条件，避免拿饱和末态代替开始扩展。','',
        '新推进器与原g40 mean canary的空间计数、两核计数、外驱和8项粒子状态逐位相同；存储恢复的状态、延迟队列与输入也逐位一致。用于后续批次的扁平延迟索引优化保留累加顺序，密集/稀疏及跨环测试与原实现逐位相同，另通过完整canary。各条件Z恒定且仍作用于GABA电流，M动态、状态有限且物理，空间/区域计数守恒，主外驱与已存W1逐位相同。','',
        '图已经生成，人工目视验收仍待用户；Agent目检单独记录在visual_qa.json。当前结果只接受为模型自身的探索性响应图。传统分岔图未完成，实际缺少经验证的确定性群体演化对象/粗状态闭合及稳定性计算；见analysis_object.md。不得用旧Phi模型的Hopf/SN/TP标签填补。','',
        '后续最有信息的工作应围绕D≈0.23的两种活动：先检验候选与原生的固定Z响应、噪声与长时存活，再在能恢复这些响应的群体分布方程中延续不稳定分支并求空间临界模态。只增加相似时间轨迹或拟合一条S形曲线都不能确定分岔类型。','',
        '## 文件','',
        '- 主图：figures/fig_kinetic_fixed_D_response.png / .pdf / .svg。',
        '- 全轨迹与延长核查：figures/fig_all_fixed_D_traces.*，figures/fig_duration_check.*。',
        '- 数据：response_summary.json/csv，spatial_panels.json，native_anchor_comparison.json。',
        '- 完整可恢复状态与计数：runs/；计算范围与读出：execution_plan.md。','']
    (OUT/'scientific_report.md').write_text('\n'.join(lines))
    files=['scripts/topic4_kinetic_D_response.py','scripts/topic4_kinetic_delay.py','scripts/topic4_kinetic_delay_flat.py',
        'scripts/run_topic4_spatial_kinetic_candidate.py','scripts/analyze_topic4_kinetic_D_response.py',
        'scripts/report_topic4_kinetic_D_response.py']
    (OUT/'producer_identity.json').write_text(json.dumps({f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},indent=2)+'\n')
    verdict=dict(status='RESPONSE_DIAGRAM_COMPLETE' if complete else 'RESPONSE_IN_PROGRESS',
        conditions_with_4s=len(rows),conditions_with_8s=sum(r['duration_s']>=8 for r in rows),
        model='g40_mean',input_repetitions=1,Z='frozen physical spatial field',M='dynamic',
        classical_equilibrium_periodic_branches='NOT_COMPUTED',bifurcation_type='NOT_ESTABLISHED',
        native_critical_response_equivalence='NOT_ESTABLISHED',human_visual_acceptance='PENDING')
    (OUT/'status.json').write_text(json.dumps(verdict,indent=2)+'\n')
    print(json.dumps(verdict,indent=2))


if __name__=='__main__':main()
