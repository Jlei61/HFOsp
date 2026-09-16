#!/usr/bin/env python3
"""Render the current model results as a Chinese, figure-led scientific PDF.

Uses existing scientific figures, preserving vector PDF content. No model reruns.
Runtime: cuda_env Python; reportlab and pypdf from /tmp/hfosp_report_pdf_runtime.
"""
import sys
sys.path.insert(0,'/tmp/hfosp_report_pdf_runtime')
from pathlib import Path
import json,hashlib,shutil,re,html
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph,Table,TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader,PdfWriter,Transformation
import copy

ROOT=Path('/home/honglab/leijiaxin/HFOsp')
OUT=ROOT/'results/topic4_sef_hfo/model_core_results_report_2026-09-08'
MULTI=ROOT/'.worktrees/topic4-substrate-autapse-fix/results/topic4_sef_hfo/multievent_distribution_search_v2_1'
PILOT=MULTI.parent/'contact_timing_shape_pilot'
RASTER=ROOT/'results/topic4_sef_hfo/snn_raster_inhibition_transition_v1'
BIF=ROOT/'results/topic4_sef_hfo/corrected_rate_bifurcation_v1'
BLUE=colors.HexColor('#24536c');GRAY=colors.HexColor('#536976');PALE=colors.HexColor('#eaf2f6')
pdfmetrics.registerFont(TTFont('CN','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('LatinBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))

def markup(text):
    return re.sub(r'([\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]+)',r'<font name="CN">\1</font>',html.escape(text))

sources={
 'parameters':MULTI/'visual_review_delivery/figures/confirmation_parameters.pdf',
 'metrics':MULTI/'visual_review_delivery/figures/confirmation_metrics.pdf',
 'pilot':PILOT/'figures/physical_parameters_and_observables.pdf',
 'envelopes':MULTI/'fig2c_contact_envelope_comparison/figures/contact_envelopes_all_contacts.pdf',
 'native_summary':MULTI/'native_activity_shortcut_audit/figures/native_activity_summary.pdf',
 'native_examples':MULTI/'native_activity_shortcut_audit/figures/full_activity_partition_examples.pdf',
 'raster_base':RASTER/'figures/raster_base.pdf',
 'raster_jump':RASTER/'figures/raster_jump.pdf',
 'bifurcation':BIF/'figures/bifurcation_and_recruitment.pdf',
 'nullclines':BIF/'figures/conditional_nullclines_and_phase_portraits.pdf',
 'bistability':BIF/'figures/bistability_and_spatial_propagation.pdf',
}
manifest=[]
for key,path in sources.items():
    dest=OUT/'assets'/f'{key}.pdf';shutil.copy2(path,dest)
    manifest.append({'id':key,'source':str(path),'snapshot':str(dest.relative_to(OUT)), 'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})

PDF=OUT/'model_core_results_report.pdf'
c=canvas.Canvas(str(OUT/'text_layout.pdf'),pagesize=A4,pageCompression=1)
c.setTitle('双核空间 SNN：数据驱动拟合、事件组织与分岔分析')
c.setAuthor('HFOsp research project')
page=0;W,H=A4;y=0;page_log=[];textlog=[]

def para(text,size=10.2,color=None,space=8,bold=False,x=42,width=None):
    global y
    width=width or W-84
    style=ParagraphStyle('p',fontName='LatinBold' if bold else 'Latin',fontSize=size,leading=size*1.55,textColor=color or colors.HexColor('#23333c'),wordWrap='CJK')
    p=Paragraph(markup(text),style);_,height=p.wrap(width,1000);p.drawOn(c,x,y-height);y-=height+space
    textlog.append({'page':page,'text':text})

def start(title,wide=False,subtitle=None):
    global page,W,H,y
    if page:
        assert y>34,(page,y)
        page_log[-1]['content_bottom_pt']=y;c.showPage()
    page+=1;W,H=landscape(A4) if wide else A4;c.setPageSize((W,H));y=H-26
    para('双核空间 SNN 模型结果报告',size=8,color=GRAY,space=0)
    c.setStrokeColor(colors.HexColor('#b8cbd6'));c.line(42,H-43,W-42,H-43)
    c.setFont('Latin',8);c.setFillColor(GRAY);c.drawRightString(W-42,22,str(page))
    c.drawString(42,22,'HFOsp  |  2026-09-08  |  Core results')
    y=H-57;para(title,size=17,color=BLUE,space=7,bold=True)
    if subtitle:para(subtitle,size=8.8,color=GRAY,space=7)
    page_log.append({'page':page,'title':title,'size_pt':[W,H],'figures':[]})

def figure(key,maxheight,crop=(0,0,1,1)):
    global y
    p=PdfReader(str(OUT/'assets'/f'{key}.pdf')).pages[0];fw=float(p.mediabox.width);fh=float(p.mediabox.height)
    x0,y0,x1,y1=crop;cw=fw*(x1-x0);ch=fh*(y1-y0);scale=min((W-84)/cw,maxheight/ch)
    ww,hh=cw*scale,ch*scale;xx=(W-ww)/2;yy=y-hh
    y=yy-9
    page_log[-1]['figures'].append({'id':key,'crop_bottom_origin':crop,'bbox_pt':[xx,yy,ww,hh]})

def table(rows,widths=None,size=9):
    global y
    widths=widths or [(W-84)/len(rows[0])]*len(rows[0]);style=ParagraphStyle('cell',fontName='Latin',fontSize=size,leading=size*1.45,wordWrap='CJK')
    data=[[Paragraph(markup(str(v)),style) for v in row] for row in rows]
    t=Table(data,colWidths=widths,hAlign='LEFT');t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#aec4d0')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d9e3e9')),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));_,hh=t.wrap(W-84,1000);t.drawOn(c,42,y-hh);y-=hh+12

start('双核空间 SNN：数据驱动拟合、\n事件组织与分岔分析'.replace('\n',''))
para('核心结果汇编｜按现有实验与原始图件整理｜2026-09-08',9,color=GRAY,space=14)
para('摘要',13,color=BLUE,bold=True)
para('数据驱动搜索已改善患者逐事件特征分布，但尚未恢复完整的局部包络与原生场传播组织。最新空间 SNN raster 显示：原参数下已有反复、自限的放电事件；降低全局 GABA 幅度后出现更强招募。简化空间 rate 模型进一步定位到 Hopf 型失稳、固定点鞍结和低率／爆发态共存。')
para('1. 模型与结果的对应关系',13,color=BLUE,bold=True)
table([['分支','模型与目的','本报告的核心结果'],['Data-driven 双核 SNN','32,000 E + 8,000 I；空间连接、LIF 与随机输入；优化事件分布','7 个确认条件及最新时间形状 pilot'],['空间 SNN raster','同一参考图，逐神经元记录；人工改变 GABA gain','原参数的自限事件；减弱抑制后的强招募及外部恢复'],['Corrected spatial rate','10×10 格，E/I 各 100 个率变量；保留突触与实际延迟','参数分岔、跨界后的持续爆发、条件 nullcline']],widths=[104,196,W-384],size=9)
para('共同骨架',12,color=BLUE,bold=True)
para('LIF 膜时间常数 E/I＝20/10 ms，不应期＝2/1 ms，reset＝11 mV；AMPA rise/decay＝0.7/3.5 ms。双核通过空间阈值调制构成，EE 连接保留各向异性。当前参考候选的 GABA decay＝20.61 ms；比较候选的其他参数见下一页。',9.5)
para('阅读范围',12,color=BLUE,bold=True)
para('本报告仅汇总已有结果，不启动新搜索。Data-driven 与新 raster/rate 实验的噪声设置、时长不同；后两者沿用偏上的参考双核，不意味着最终患者间期工作点已冻结。所附 well-mixed 报告仅作为版式参考，其无空间网络结果未混入本报告。',9.5)
para('所有结果均为开发阶段证据。不同损失版本不直接比较数值；患者 HFO 包络与模型放电密度也不共享幅度单位。',9,color=GRAY)

start('2. Data-driven：实际参数组合与双核位置',True,'7 个确认条件；每条件 2 张新拓扑 × 2 段动力学噪声；每条轨道 24 s。')
figure('parameters',330)
para('图 1｜每个条件的双核几何和实际参数。联合优化同时改变位置、阈值偏移、EE、E→I、I→E、GABA decay、EE 轴角与轴比；连核线和名义 EE 轴不能直接解释为实测传播方向。',9.4)
para('固定网络连续产生多次事件，再比较事件分布；生成器不接收逐事件 TA/TB 指令。模式分类器参与评分，因而模式条件图并非完全独立于训练。',9.4)
para('来源：多事件分布搜索 v2.1／visual_review_delivery。所有候选均保留为比较条件。',8,color=GRAY)

start('3. Data-driven：总分改善不等于每个观测都改善',True,'每个点是一条完整运行；四点来自拓扑与噪声的组合，不是四名患者。')
figure('metrics',372)
para('图 2｜联合损失、参与误差、顺序误差、时差、方向、两类覆盖、支持率和事件数。联合候选确认期 L_off＝0.979，old-joint 基线＝1.079；相对基线仅 3/4 配对运行更低，不能据此声称稳定显著优势。',9.1)
para('后续分量审计显示：该两候选间改善主要来自模式项（0.208→0.124），不分标签整体项仅由 0.871→0.855。总分最低不等于全部传播细节最相似。',9.1)

start('4. 最新时间形状 pilot：参数与损失的关系',True,'三个固定双核位置；真实 SNN 联合扰动五项参数；下图为同一原图上半部。')
figure('pilot',330,crop=(0,.5,1,1))
para('图 3a｜依次为阈值偏移、E→E、E→I、I→E、GABA decay 五列。三行分别为原损失、时间／形状项和新总损失。新目标补充每个接触点的 t50 相对位置、上升／下降质量时间及形状不对称。',9.1)
para('训练最低新总损失为 2.2613（tshape_anchor1_B_plus）。但两名提名候选在留后患者块＋新噪声重演中，新时间距离均为 0/2 改善：相对各自起点平均增加 0.266、0.123。训练收益未保留。',9.1)
para('来源：contact_timing_shape_pilot。46 个患者训练波形、18 个按块留后诊断波形；均属已开发使用的数据。',8,color=GRAY)

start('5. 最新时间形状 pilot：局部持续、时差与重叠',True,'下图为图 3 同一原图下半部；每个点是一条真实模拟，其他参数并未逐项固定。')
figure('pilot',345,crop=(0,0,1,.5))
para('图 3b｜局部包络宽度、接触点质心跨度及接触点时间重叠。局部宽度仍主要约 15–20 ms，重叠大多接近零；改变参数使损失变化，并没有同步补齐患者较宽、较重叠的局部时间结构。',9.1)
para('蓝／橙／紫分别为三个固定位置；圆／三角／方块为起点、A 批、B 批。联合扰动散点只能表示参数组合与结果的关联，不支持把单列趋势解释为某一参数的独立因果作用。',9.1)

start('6. 间期模式对比：患者的局部活动更宽、更重叠')
figure('envelopes',330)
para('图 4｜左列为 Fig. 2C 的患者 TA/TB 原始代表事件，中、右列为联合候选与参考位置的模型事件。固定通道顺序、250 ms 真实时间窗；每通道按自己的完整窗口峰值归一化，不拉伸时间。星号为未参与但仍保留的通道。',9.5)
figure('native_summary',154)
para('图 5｜28 条确认轨道、1,082 个模型事件的原生场诊断。患者 TRAIN 波形的接触点时间重叠中位数 TA/TB＝0.686/0.677；联合候选和参考位置均为 0。这里的重叠是 [t10,t90] 区间交并比，不是相关系数。',9.5)
para('热图描述包络形状，不证明模型复现了患者的振荡载波；也不能比较两者绝对能量。来源：Fig. 2C 包络对比与 native_activity_shortcut_audit。',8.5,color=GRAY)

start('7. Core 内部活动：有错相与多家族，尚非两条患者路线')
figure('native_examples',448)
para('图 6｜每行是一个原有模型事件，展示全部原生场、最大活动家族、其余活动，以及两个 core 和核外的活动轨迹。比较联合候选与参考位置；选取各自首个相应标签事件，没有按外观挑选。',9.5)
table([['事件层诊断','联合候选','参考位置'],['两核活动峰绝对间隔：中位数','34 ms','26 ms'],['TA 中 core1 先到峰','45.1%','72.0%'],['TB 中 core1 先到峰','48.4%','35.3%']],size=9)
para('联合候选尚无“TA 由核1先发、TB 由核2先发”的稳定对应。活动家族依赖分割规则，不能直接称为独立振子或因果起源。来源：原生场冻结输出审阅。',9.2)

start('8. 新 SNN raster：原参数已有反复、自限事件',True,'真实逐神经元 spike；固定参考底物，gain＝1；OU 关闭，逐神经元 Poisson 输入保留。')
figure('raster_base',376)
para('图 7｜没有参数变化或外部刺激，1–6.5 s 仍记录到 15 个描述性事件，峰值 42.1–69.1 Hz；全 E 平均率 6.23 Hz，74.5% 的 5 ms 时间 bin 低于 1 Hz。活动有间隔，并非持续高率平台。',9.2)
para('Raster 固定抽样 60 core-A E、60 core-B E、120 核外 E、60 I，核内样本被富集；下方群体率来自全部 40,000 个神经元。单 seed 诊断，不等同于患者间期分布验收。',9.2)

start('9. 新 SNN raster：降低抑制幅度后，招募显著增强',True,'同图同 seed；GABA jump gain：1→0.5→1；膜电位、突触电流与延迟历史连续携带。')
figure('raster_jump',369)
para('图 8｜低 gain 保持期（2–3.5 s）全 E 均值 64.97 Hz、峰值 281.29 Hz；恢复后（5.5–6.5 s）均值 3.48 Hz、峰值 48.89 Hz。仍存在低率间隔，不能只凭峰值命名持续 runaway。',9.2)
para('gain 同时缩放 I→E 和 I→I 的新突触事件。恢复由实验者指定，并非活动依赖 Z 自行恢复；本图未开启 Z/M 动态，也未证明空间 SNN 的分岔点等于 rate 模型的临界值。',9.2)

start('10. 简化空间模型：定位参数分岔与状态切换',True,'Corrected rate 保留参考双核、突触动力学与 358 个实际延迟 bin；不属于无空间 well-mixed 模型。')
figure('bifurcation',378)
para('图 9｜GABA decay＝20.61 ms 时，q≈0.73805 出现约 3.99 Hz 的复共轭模态过界；完整耦合切向系统含 72,600 个变量。固定突触面积的步长细化保留该边界。另定位到 q≈0.70886、0.40622 的低／高率固定点鞍结。',9.1)
para('相同局部刺激在 q＝0.76 后自限，在 q＝0.72 后进入持续爆发。最大同时招募随 q 进一步降低而扩大；尚未证明“全局化”本身是独立分岔。',9.1)

start('11. Nullcline 与相轨道：交点存在，稳定性仍可改变')
figure('nullclines',448)
para('图 10｜上排在各自平衡点固定 E/I 空间形状，仅缩放群体幅度，并将突触和历史设为对应稳态输入，得到条件零导数曲线。两个 q 下都有交点，但完整延迟模型分别稳定与振荡失稳。',10)
para('下排为完整动态模型的真实 E–I 投影轨迹，保留演化中的突触电流与历史。上排二维面不是不变流形，不能把两条 nullcline 当成全部空间系统。',10)
para('严格术语：离散映射是复共轭乘子越过单位圆；对应固定面积连续延迟极限呈 Hopf 型过界。尚未计算第一 Lyapunov 系数，不赋予超／亚临界标签。',9.6)

start('12. 核心判断：爆发能力、传播组织与患者拟合分别检验')
figure('bistability',373)
para('图 11｜q＝0.75 保持不变时，无刺激回到低率态，一次有限局部刺激可转入持续爆发。下排为各一个晚期爆发的首次越过 20 Hz 时间：q＝0.72 的空间招募并集约 77.8%，q＝0.50 为 100%；这是事件内并集，不是同一时刻的招募比例。',9.6)
table([['目前支持','尚未建立'],['多事件特征拟合有改善；模型具备自限和强爆发两类能力','患者局部波形、TA/TB 原生传播与模式切换机制已恢复'],['Rate 模型出现 Hopf 型失稳、鞍结与状态共存','空间 SNN 的同一分岔证明；独立的全局化分岔'],['参数改变和有限扰动均可切换状态','活动积累驱动 Z/M 自主跨界的完整证据链']],widths=[(W-84)*.5,(W-84)*.5],size=9)
para('来源索引：图1–2＝多事件 v2.1；图3＝时间形状 pilot；图4–6＝患者包络与原生场审阅；图7–8＝新空间 SNN raster；图9–11＝corrected rate 分岔分析。逐图原始路径、快照与校验值另存 figure_sources.json。',8.4,color=GRAY)
assert y>34,(page,y);page_log[-1]['content_bottom_pt']=y;c.save()
writer=PdfWriter();base=PdfReader(str(OUT/'text_layout.pdf'))
for record,target in zip(page_log,base.pages):
    for f in record['figures']:
        source=copy.deepcopy(PdfReader(str(OUT/'assets'/f"{f['id']}.pdf")).pages[0]);fw=float(source.mediabox.width);fh=float(source.mediabox.height)
        x0,y0,x1,y1=f['crop_bottom_origin'];xx,yy,ww,hh=f['bbox_pt'];scale=ww/(fw*(x1-x0))
        source.cropbox.lower_left=(x0*fw,y0*fh);source.cropbox.upper_right=(x1*fw,y1*fh)
        target.merge_transformed_page(source,Transformation().scale(scale).translate(xx-x0*fw*scale,yy-y0*fh*scale))
    writer.add_page(target)
writer.add_metadata({'/Title':'双核空间 SNN：数据驱动拟合、事件组织与分岔分析','/Author':'HFOsp research project'})
with PDF.open('wb') as stream:writer.write(stream)
(OUT/'figure_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(OUT/'layout_manifest.json').write_text(json.dumps(page_log,ensure_ascii=False,indent=2))
(OUT/'report_text.json').write_text(json.dumps(textlog,ensure_ascii=False,indent=2))
print(str(PDF));print('pages',page,'lowest content',min(p['content_bottom_pt'] for p in page_log))
