"""Verify the scientific distinctions made in v6 and write the result report."""
from common import *
import numpy as np
from PIL import Image
import csv

def mu(source,dt=None):
    source=Path(source).resolve();found=[]
    for base in [OUT,V5]:
        folder=base/'poincare'/source.parent.name/source.stem
        if folder.exists():
            for p in folder.glob('rk4_orthogonal_dt*.json'):
                a=read(p)
                if dt is None or abs(a['dt_ms']-dt)<1e-5:found.append(a)
    if not found:raise FileNotFoundError(('Floquet missing',str(source),dt))
    return min(found,key=lambda a:a['dt_ms'])

def main():
    old=read(V5/'numerical_validation.json');assert old['status']=='PASS'
    seq=read(OUT/'displayed_curve_sequences.json');count=0
    for a in seq:
        for r in a:
            lo,m,hi=map(np.asarray,[r['lo'],r['mean'],r['hi']])
            assert np.all(lo<=m+1e-7) and np.all(m<=hi+1e-7);count+=1
    pairs=read(OUT/'same_J_recruitment.json');assert pairs[0]['g']==pairs[1]['g']
    coexist=[]
    for a in pairs:
        meta=read(Path(a['path']).with_suffix('.json'));fl=mu(a['path'])
        assert meta['offgrid_defect_hz']<1e-6 and fl['max_transverse']<.99
        coexist.append(dict(**a,floquet=fl,offgrid_defect_hz=meta['offgrid_defect_hz']))
    peak=read(OUT/'peak_exchange.json');peakmu=mu(peak['source'])
    assert abs(peak['primary_hz']-peak['secondary_hz'])<1e-6 and abs(peak['arc_dJ'])>1e-7 and peakmu['max_transverse']<.99
    cross=read(OUT/'projection_intersection.json');smu=mu(cross['stable']['source']);umu=mu(cross['unstable']['source'])
    assert abs(cross['observable_difference_hz'])<1e-6
    assert abs(cross['stable']['T_ms']-cross['unstable']['T_ms'])>1
    assert smu['max_transverse']<.99 and umu['max_transverse']>100
    steps=[]
    for name in ['recruitment_up','recruitment_down']:
        a=read(OUT/'transitions'/f'{name}_dt0.025.json');b=read(OUT/'transitions'/f'{name}_dt0.0125.json')
        dm=float(abs(np.array(a['mean_hz'])-b['mean_hz']).max());dt=abs(a['recurrence']['period_ms']-b['recurrence']['period_ms'])
        assert dm<.01 and dt<.01 and b['recurrence']['recurrence_error']<1e-5
        steps.append(dict(name=name,max_tail_mean_change_hz=dm,period_change_ms=dt,refined_period_ms=b['recurrence']['period_ms']))
    folds=[]
    for name,label in [('surround_first_fold','LP0a'),('surround_second_fold','LP0b'),('surround_recruited_fold','LP0c')]:
        a=read(V5/'folds'/f'{name}_N2048.json');root=V5 if label=='LP0c' else OUT;b=read(root/'folds'/f'{name}_N4096.json')
        dg=abs(a['g']-b['g']);assert dg<1e-11 and b['fixed_parameter_null_residual']<1e-8
        folds.append(dict(label=label,grid_J_difference=dg,**b))
    pdlo=read(OUT/'flips/surround_micro_flip_N2048.json');pd=read(OUT/'flips/surround_micro_flip_N4096.json');pdflo=mu(pd['source'])
    assert abs(pdlo['g']-pd['g'])<1e-11 and pd['anti_null_residual']<1e-7
    assert abs(pdflo['multipliers'][0][0]+1)<.005
    neighborhood=read(OUT/'arcs/micro_neighborhood/progress.json');near=[]
    for a in neighborhood:
        if a['fraction'] in [.25,.75,1.25]:
            f=mu(a['source']);near.append(dict(**a,floquet=f));assert (f['max_transverse']<1)==(a['fraction']<1)
    children=[]
    for p in sorted((OUT/'periodic/surround_period2').glob('*.json')):
        a=read(p);f=mu(a['source']);assert a['half_period_difference']>1e-5
        children.append(dict(**a,floquet=f))
    assert len(children)>=3
    children.sort(key=lambda a:a['amplitude'])
    assert children[0]['g']>pd['g'] and children[0]['floquet']['max_transverse']<1
    assert children[1]['floquet']['max_transverse']<1 and children[2]['floquet']['max_transverse']>1
    mode=[]
    for name,source in [(a['label'],a['source']) for a in folds]+[('PD0',pd['source'])]+[(a['name'],None) for a in read(V5/'critical_mode_components.json')]:
        if source:
            z=np.load(source);N=len(z['r']);v=z['mode'] if name=='PD0' else z['right_null'][:-1].reshape(N,6)
            l=z['left_mode'] if name=='PD0' else z['left_null'][:-1].reshape(N,6)
            a=dict(name=name,right_rate_fraction=(np.sum(v*v,0)/np.sum(v*v)).tolist(),left_rate_fraction=(np.sum(l*l,0)/np.sum(l*l)).tolist())
        else:a=next(x for x in read(V5/'critical_mode_components.json') if x['name']==name)
        mode.append(a)
    write('critical_mode_components.json',mode)
    grids=read(OUT/'critical_grid_and_modes.json');dense=[]
    for folder,n in [('approach_first_fold',21),('between_low_folds',21),('between_fold_and_flip',25)]:
        a=read(OUT/'arcs'/folder/'progress.json');assert len(a)==n and max(x['residual'] for x in a)<2e-11;dense.extend(a)
    validation=dict(status='PASS',observable_order_checked=count,dense_micro_orbit_count=len(dense),critical_orbit_and_mode_convergence=grids,same_J_coexistence=coexist,peak_exchange=dict(**peak,floquet=peakmu),
        crossing=dict(**cross,stable_floquet=smu,unstable_floquet=umu),time_step_checks=steps,fold_grid_checks=folds,
        PD0=dict(**pd,grid_J_difference=abs(pdlo['g']-pd['g']),floquet=pdflo,classification='supercritical toward increasing J'),micro_neighborhood=near,PD0_children=children,
        prior_right_branch_validation=str(V5/'numerical_validation.json'),scope='targeted local and projection checks; not exhaustive global attractor census')
    write('numerical_validation.json',validation)
    with (OUT/'critical_points.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['label','JEE_core','T_ms','A_mean_hz','B_mean_hz','Surround_mean_hz'])
        for a in folds+[dict(label='PD0',**pd)]:writer.writerow([a['label'],a['g'],a['T_ms'],*a['mean_hz'][:3]])
        for label,path in [('LP1','folds/burst_end_fold_N4096.json'),('PD1','flips/mixed_lower_flip_N2048.json'),('PD2','flips/mixed_flip_N2048.json'),('PD3','flips/tonic_lower_flip_N2048.json')]:
            a=read(V5/path);writer.writerow([label,a['g'],a['T_ms'],*a['mean_hz'][:3]])
    table='\n'.join(f'| {a["label"]} | {a["g"]:.13f} | {a["T_ms"]:.6f} | {a["mean_hz"][0]:.6f} | {a["mean_hz"][1]:.6f} |' for a in folds+[dict(label='PD0',**pd)])
    right=read(OUT/'same_J_right.json');cm='\n'.join(f'| {a["name"]} | '+ ' | '.join(f'{x*100:.3f}%' for x in a['right_rate_fraction'])+' |' for a in mode)
    childtext='\n'.join(f'| {a["amplitude"]:g} | {a["g"]:.13f} | {a["T_ms"]:.6f} | {a["half_period_difference"]:.6g} | {a["floquet"]["max_transverse"]:.6g} |' for a in children)
    report=f'''# Core A/B 分岔投影、尖点和周期共存核查 v6 — 2026-09-15

## 回答的问题与结论

本次区分三个现象：1.176附近真正的周期轨道折返/失稳；同一周期内两个峰交换最大值身份；右侧不同周期轨道、不同统计量在二维图上的交叠。主图中的每条线都来自已保存的周期边值解，不能把曲线尖角、坐标交点或稳定支的可见端点直接命名为新分岔。

分析对象沿用v2–v5的确定性六群体延迟率闭合模型，群体为[A E,B E,周边E,A I,B I,周边I]。统计量是每群体每神经元率的完整周期均值及波形极值；稳定性单位是联合网络的一条完整周期轨道。它不是原生随机SNN逐细胞Jacobian，也不研究临床发作转变。原生SNN四状态波形/raster和编号仍保留，但未重新仿真或重新分类。

B图的编号1–4表示与A图相同的SNN运行条件，四状态名称沿用A核原分类；没有把B的独立事件CV分类视作已完成。

## 1.176附近的针状突起

粗横轴把宽度约5.2×10^-5的共存区压成了竖线。低周边周期族与招募周边周期族必须分别绘制，不能按“每个J只取一个值”串成唯一轨道。下面的折点都由带相位约束的周期边值Jacobian零模定位；PD0由反周期线性化零模定位。

| 标记 | J_EE,core | 完整周期 / ms | A E均值 / Hz | B E均值 / Hz |
|---|---:|---:|---:|---:|
{table}

LP0a→LP0b之间的参数折返仅约{folds[0]['g']-folds[1]['g']:.3g}。LP0b之后还有一段极短的稳定T周期支，随后在PD0发生实Floquet乘子穿过−1。LP0b与PD0的参数距离约{pd['g']-folds[1]['g']:.3g}；主图不可能分辨这一尺度，新增的微区图使用局部坐标显示。先前把这一整段都笼统称为“不稳定连接”的解释不充分，本版补入实际稳定性检查。

PD0的N=2048/4096参数差为{abs(pdlo['g']-pd['g']):.3g}；N=4096反周期零模残差{pd['anti_null_residual']:.3g}，独立返回映射临界乘子{pdflo['multipliers'][0][0]:.9f}。极小参数差是这一冻结数值模型内部的结果，不代表生物参数或模型闭合误差也达到这种精度。

PD0分出的2T解如下；非零半周期差排除了把同一T波形复制两次的伪解。子支超/亚临界结论以参数方向和横向乘子联合判断，不仅看分支向哪边画。

| 振幅坐标 | J | 2T周期 / ms | 非零半周期差 | 最大横向乘子模 |
|---|---:|---:|---:|---:|
{childtext}

最靠近PD0的两个2T样本位于更大J一侧且稳定，故PD0为向右增大J时的超临界倍周期。继续增大2T振幅后，实主乘子已低于−1；后续再次失稳的精确临界点和4T支未在本次穷尽。不能把整条2T子支都涂为稳定，也不能据此直接断言存在混沌或解释原生SNN的不规则burst。

在同一个J={coexist[0]['g']:.10f}，低周边与招募周边轨道都稳定，最大横向乘子模分别为{coexist[0]['floquet']['max_transverse']:.6f}和{coexist[1]['floquet']['max_transverse']:.6f}。周边E均值从{coexist[0]['mean'][2]:.3f}到{coexist[1]['mean'][2]:.3f}Hz，A均值从{coexist[0]['mean'][0]:.3f}到{coexist[1]['mean'][0]:.3f}Hz，而B从{coexist[0]['mean'][1]:.3f}降至{coexist[1]['mean'][1]:.3f}Hz。A在招募支每个完整周期中多一次放电，B没有同步复制这一变化。

从LP0a完整延迟历史向上跳至J=1.17632会到达招募周边周期态；从LP0c向下跳至1.17623返回低周边周期态。dt=0.025和0.0125ms得到相同去向，最大尾段均值差小于0.01Hz，周期差小于0.01ms。这里的尾段3秒均值不等于完整周期均值，报告的精确轨道均值均来自周期边值解。这两个有限参数阶跃不等同于精确准静态迟滞阈值。

两族的全部远端不稳定连接、PD0子支更远处的分岔尚未穷尽；图中不跨未验证的全局断口补线。已确认的折点、局部失稳、共存和指定历史下的切换应与这个未完成的全局问题分开。

## 绿色最大值为何有尖角

在招募周边支，A每周期有较早和较晚两个局部峰。J={peak['g']:.10f}时两峰同高{peak['primary_hz']:.6f}Hz，分别在周边峰前约82.61ms和后约10.18ms。最大值定义为二者较大者，所以最高峰身份交换时，其参数导数可以出现折角。

该点周期轨道仍光滑，伪弧长dJ分量非零，最大横向乘子模为{peakmu['max_transverse']:.6f}，远离1；这里没有发现新的轨道分岔。橙色均值的大范围针状形状则来自窄区内分支的快速弯曲及共存，不能把均值和峰值尖角混作同一机制。

## 右侧交叠与从零到高谷值

原主图同时放置均值和峰谷，容易把上升的谷值线认成最大值线。新版将最大值和最小值分色，并单独给出A/B×均值/最大值/最小值六联图。

一个已数值求根验证的交点在J={cross['g']:.10f}：不稳定轨道A均值={cross['unstable']['mean_hz'][0]:.6f}Hz，稳定轨道A谷值={cross['stable']['refined_A_min_hz']:.6f}Hz，差{abs(cross['observable_difference_hz']):.3g}Hz。两者周期为{cross['unstable']['T_ms']:.6f}与{cross['stable']['T_ms']:.6f}ms，A波形显著不同；横向主乘子模约{umu['max_transverse']:.3g}与{smu['max_transverse']:.6f}。因此这是不同轨道的不同读出量相等，不是轨道连接或新临界点。强不稳定轨道的精细次主谱不作解释。

同一轨道始终满足r_min≤⟨r⟩≤r_max；这个不等式不约束两条不同轨道之间的均值/谷值大小。

在J=1.38已经有两种稳定状态共存：

| 状态 | A均值 | A谷值 | A峰值 | B均值 | B谷值 | B峰值 |
|---|---:|---:|---:|---:|---:|---:|
| A burst / B高活动 | {right[0]['mean'][0]:.2f} | {right[0]['lo'][0]:.2f} | {right[0]['hi'][0]:.2f} | {right[0]['mean'][1]:.2f} | {right[0]['lo'][1]:.2f} | {right[0]['hi'][1]:.2f} |
| A高活动 / B高活动 | {right[1]['mean'][0]:.2f} | {right[1]['lo'][0]:.2f} | {right[1]['hi'][0]:.2f} | {right[1]['mean'][1]:.2f} | {right[1]['lo'][1]:.2f} | {right[1]['hi'][1]:.2f} |

单位均为Hz/神经元。A两支峰值相似，谷值却由接近0变为约207Hz；B两支都处于高背景。PD3处T周期母支的A谷值已经约131Hz，因为这里改变稳定性的是一条有限振幅周期轨道，不是从零振幅产生的新Hopf周期。只画稳定极值时，两族之间当然留下空白；补画不稳定轨道后可看到中间读出值。实际随历史走到哪条吸引子，还取决于共存和失稳后的去向。V5已验证从PD2向右阶跃到1.39进入两核高背景振荡，不能把该吸引子切换又误称为PD3处“谷值从零出生”。

## A/B是同一分岔的不同投影

同一个J同时缩放A、B核内EE权重，均值矩乘J、二阶矩乘J²。A/B全部直接跨核E/I连接均为0，两核通过周边群体间接作用。两核不是精确镜像：E细胞数720/742，平均阈值约17.26395/17.28482mV，基线投影EE权重也不同。因此不要求r_A(t)=r_B(t)。审计数值见model_AB_audit.json；这些差别说明对称性并不存在，并未单独归因于某一个参数差。

对每一条联合网络轨道，A/B投影共享同一套特征根/Floquet谱和临界J；投影到B时临界参数不变，均值、峰谷及模态可见程度会变。同J下不同共存轨道仍各有自己的谱。若把A或B孤立出来重算，则改变了模型，不能叫当前同一网络的另一个读出。

以下是临界右率模态在六群体的平方范数比例，未按细胞数加权；不是因果贡献百分比。LP还含周期变化分量，此处仅展示率分量。

| 临界点 | A E | B E | 周边E | A I | B I | 周边I |
|---|---:|---:|---:|---:|---:|---:|
{cm}

LP0a由周边E率分量主导，与周边招募位置一致；LP1由B E主导，PD2由A E主导，解释为何后续高背景变化先明显表现在B、再表现在A。PD1/PD3的右率模主要在对应核I分量。以上是定位联合临界模式及其与波形变化的一致性，未经耦合消融不能把它提升为独立因果链证明。

## 数值方法、复现与交付

固定点沿用v2的特征方程。周期轨道使用自由周期傅里叶边值问题、显式物理延迟及相位条件；转弯处使用伪弧长图而非固定J插值。均值为整周期积分，峰谷由傅里叶加密/极值优化求得。折点用周期Jacobian的左右零向量和非退化检查；PD用v(t+T)=−v(t)的反周期线性化零模；独立返回映射移除自治相位方向后计算Floquet乘子。

周期折点需要额外的非平凡+1乘子，不能把自治相位乘子当作分岔。方法判据参照[DDE-BIFTOOL官方周期折点示例](https://ddebiftool.sourceforge.net/demos/neuron/html/demo1_POfold.html)与[官方手册](https://arxiv.org/abs/1406.7144)；这里实际运行的是本地Python实现。

本次曾有固定J=1.176275的高周边Newton求解失败；改用已有弧长点的精确J并加倍网格后得到低残差解，同时求出同J另一稳定周期轨道。失败记录保留，未把不收敛位置当作分岔。微区PD求根采用两个近临界反周期算子的联合行列式测试，避免最近特征值换序使单模态测试漏根。

模型仍保留368个物理延迟bin、六群体阈值分布和双指数突触核，E/I率响应时间常数5/2.5ms属于闭合假设，尚未完成原生SNN动态响应标定。参数数值精度不能消除这一模型层限制。

代码：scripts/topic4_core_observable_bifurcation_v6/。主图输入：displayed_curve_sequences.json；核查：numerical_validation.json；临界表：critical_points.csv；逐图说明：figures/README.md。失败和中间尝试与被接受的结果分开保存。运行figures.py重画已完成的数值结果，运行validate_report.py重做本报告涉及的验证。

候选图已生成并执行本地自查，仍待用户目视检查。旧版输出保持原位。
'''
    (OUT/'scientific_report.md').write_text(report)
    print('VALIDATION_PASS',count,'observable points',flush=True)

if __name__=='__main__':main()
