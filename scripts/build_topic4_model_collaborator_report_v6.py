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
OUT=ROOT/'results/topic4_sef_hfo/model_collaborator_report_v6_2026-09-08'
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
 'model_overview':ROOT/'results/paper-ready-figure/fig4/figures/fig4-panela.pdf',
 'ranges':MULTI/'observable_distribution_recovery/figures/candidate_1_distribution_ranges.pdf',
 'contacts':MULTI/'all_event_parameter_distributions/figures/c1_contacts_all_and_modes.pdf',
 'rhythm':MULTI.parent/'patient_model_rhythm_comparison/figures/patient_model_event_rhythm.pdf',
 'interventions':MULTI.parent/'patient_model_rhythm_comparison/figures/parameter_rhythm_response.pdf',
 'core_rhythm':MULTI.parent/'core_burst_rhythm_audit/figures/core_activity_rhythm.pdf',
 'parameters':MULTI/'visual_review_delivery/figures/confirmation_parameters.pdf',


 'envelopes':MULTI/'fig2c_contact_envelope_comparison/figures/contact_envelopes_all_contacts.pdf',
 'native_summary':MULTI/'native_activity_shortcut_audit/figures/native_activity_summary.pdf',
 'native_examples':MULTI/'native_activity_shortcut_audit/figures/full_activity_partition_examples.pdf',
 'inhibition_cycle':RASTER/'figures/snn_raster_inhibition_cycle.pdf',
 'raster_c_ou':RASTER/'figures/raster_z_current_e_ou.pdf',
 'bifurcation':BIF/'figures/bifurcation_and_recruitment.pdf',
 'nullclines':BIF/'figures/conditional_nullclines_and_phase_portraits.pdf',
 'bistability':BIF/'figures/bistability_and_spatial_propagation.pdf',
}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'assets').mkdir(exist_ok=True)
manifest=[]
for key,path in sources.items():
    dest=OUT/'assets'/f'{key}.pdf';shutil.copy2(path,dest)
    if key == 'rhythm':
        # Source PDF embeds a CJK font that poppler cannot render; use its unchanged PNG.
        from reportlab.lib.utils import ImageReader
        png=path.with_suffix('.png');snapshot=OUT/'assets'/'rhythm.png';shutil.copy2(png,snapshot)
        bounds=PdfReader(str(path)).pages[0].mediabox;fw,fh=float(bounds.width),float(bounds.height)
        rc=canvas.Canvas(str(dest),pagesize=(fw,fh));rc.drawImage(ImageReader(str(snapshot)),0,0,width=fw,height=fh);rc.save()
        manifest.append({'id':key,'source':str(png),'snapshot':str(snapshot.relative_to(OUT)),'sha256':hashlib.sha256(snapshot.read_bytes()).hexdigest(),'embedded_pdf':'assets/rhythm.pdf','reason':'Original PDF CJK font embedding renders missing glyphs; unchanged source PNG used.'})
        continue
    manifest.append({'id':key,'source':str(path),'snapshot':str(dest.relative_to(OUT)), 'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})

PDF=OUT/'model_collaborator_report_v6.pdf'
c=canvas.Canvas(str(OUT/'text_layout.pdf'),pagesize=A4,pageCompression=1)
c.setTitle('双核空间 SNN：数据驱动拟合、事件组织与分岔分析')
c.setAuthor('HFOsp research project')
page=0;W,H=A4;y=0;page_log=[];textlog=[]

def para(text,size=10.2,color=None,space=8,bold=False,x=42,width=None):
    global y
    if W < H and size >= 9: size *= 1.07
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
    para('双核空间 SNN｜合作者讨论稿 · v6',size=8,color=GRAY,space=0)
    c.setStrokeColor(colors.HexColor('#b8cbd6'));c.line(42,H-43,W-42,H-43)
    c.setFont('Latin',8);c.setFillColor(GRAY);c.drawRightString(W-42,22,str(page))
    c.drawString(42,22,'HFOsp  |  2026-09-08  |  Collaborator discussion')
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

# Vector equations use the same math rendering as the scientific plotting stack.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype':42,'font.family':'DejaVu Sans','mathtext.fontset':'dejavusans'})
def eq(key, lines, height=None):
    fig=plt.figure(figsize=(8.8,2))
    heights=[]
    for line in lines:
        artist=fig.text(.02,.5,line,fontsize=16,va='center')
        fig.canvas.draw()
        bbox=artist.get_window_extent(fig.canvas.get_renderer())
        if bbox.width > .95*fig.bbox.width:
            artist.set_fontsize(16*.95*fig.bbox.width/bbox.width)
        fig.canvas.draw()
        heights.append(artist.get_window_extent(fig.canvas.get_renderer()).height)
    gap=12
    total=sum(heights)+gap*(len(lines)+1)
    fig.set_size_inches(8.8,total/fig.dpi)
    top=total-gap
    for artist,h in zip(fig.texts,heights):
        artist.set_position((.02,(top-h/2)/total))
        top-=h+gap
    fig.canvas.draw()
    for artist in fig.texts:
        bounds=artist.get_window_extent(fig.canvas.get_renderer())
        assert bounds.x1 < fig.bbox.width and bounds.y0 >= 0 and bounds.y1 <= fig.bbox.height, (key, bounds)
    path=OUT/'assets'/f'{key}.pdf';fig.savefig(path,facecolor='white');plt.close(fig)
    manifest.append({'id':key,'source':'equations verified against code; authored in this report','snapshot':str(path.relative_to(OUT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    figure(key,height or 30*len(lines)+8)
def explanation(why,how,result,next_,size=9.8):
    for label,txt in [('检验问题',why),('制作与读法',how),('结果与解释',result),('已完成的衔接',next_)]:para(label+'｜'+txt,size=size,space=7)

start('从患者间期传播到持续高活动：模型解释了什么？')
para('合作者讨论稿 · 第 6 版 · 2026-09-08',10,color=GRAY,space=18)
para('核心问题',14,color=BLUE,bold=True)
para('同一套固定网络能否在连续随机输入下产生多次、自行结束的事件，使不同事件分别接近患者两类传播模式（TA/TB）的事件分布；在同一套连接结构中，抑制反馈的变化又能否使自限事件转为持续爆发？前一个问题约束患者特异性，后一个问题检验动力学能力。二者需要分别成立，才能讨论间期到发作样状态的联系。',11)
para('目前的判断',14,color=BLUE,bold=True)
para('数据驱动候选改善了部分逐事件统计，仍有通道参与、时差和局部波形偏差。核内活动呈较快、较规则的重复爆发，电极采样和事件筛选会使这种规则性看起来减弱。因此，“能分到 TA/TB 两类”还不等于恢复了患者的传播机制。',11)
para('在独立机制实验中，降低脉冲网络（SNN）的 GABA 输入幅度确实增强放电与招募；仅减弱兴奋性神经元接收的抑制电流时，在原有 OU 背景下也能进入持续高率态，并在恢复抑制后退出；修正后的空间放电率模型则存在 Hopf 型失稳、自限／持续爆发转换和状态共存。尚未证明患者间期工作点、SNN 与放电率模型共享同一临界点，也尚未接通活动积累—Z/M 演化—自主跨界这条链。',11)
table([['阅读顺序','本节要回答什么'],['模型方法、差异与优化（第 2–9 页）','双核在哪里进入方程？哪些参数可调？训练的样本究竟是什么？'],['患者对照与节律（第 10–15 页）','分布哪些部分相似、哪些不相似？为什么进一步审计波形和核内活动？'],['SNN 干预与 Z（第 16–18 页）','四条件对照与 C＋OU 高态进出，及活动依赖 Z 的关键检验。'],['空间放电率模型与分岔（第 19–23 页）','近似保留什么？什么数学对象失稳？状态如何切换？'],['讨论与来源（第 24–25 页）','哪些结论可用，关键缺口在哪里，图由哪些已有实验产生？']],widths=[160,W-244],size=9.3)
para('模型说明现以作者提供的 Methods 为主线，保留已确定的方程与符号，另列文章和实际实验之间的差异；模型结构图和前版结果均保留。已有图保留数据及坐标；单参数图的错误 GABA 面积图注依据代码与脉冲核验更正。本报告不新增网络搜索，不替换正式 Fig. 5。',9,color=GRAY)

start('模型总览｜从局部 E/I 回路到患者电极记录')
figure('model_overview',280)
para('模型结构示意｜沿用 Figure 4A 的局部回路与患者特异空间网络图。红色三角为兴奋性 E 神经元，蓝色圆点为抑制性 I 神经元；右侧黑边白圆及连线表示患者电极接触点与电极排列。',10.2)
para('局部回路怎样产生和限制活动｜E→E 递归兴奋放大局部放电，E→I 招募抑制细胞，I→E 反馈约束兴奋性活动。红色椭圆表示 EE 连接的方向偏好，蓝色范围表示局部抑制作用示意；它们不是两个核区的轮廓。实际网络也包含图中未展开的 I→I 通路，后面的 C 对照正说明其作用不能忽略。',10.2)
para('局部活动怎样被电极观察｜右侧以 −10 至 10 mm 展示二维网络和电极几何。触点周围的淡绿色区域表示高斯空间采样范围（σ＝0.25 mm，示意 95% 权重半径约 0.61 mm），电极附近神经元的活动经加权汇总，形成各接触点的放电密度包络。虚线连接的是代表性局部回路与空间邻域，不表示这里已经识别出病理起源。',10.2)
para('Z 与 M 怎样对应方程｜蓝色 z 表示有效抑制资源／效能，紫色 m 表示由兴奋性神经元动作电位积累的适应变量。在现有实现中，Z 调节兴奋性神经元接收的 GABA 电流，M 提供 E 细胞的适应负反馈。数据驱动拟合实验未开启自主 Z/M；四种抑制干预中的 C 及 C＋OU 对照使用人工 z(t)，自主 Z/M 的因果检验见后面的衔接说明。',10.2)
para('双核在哪里｜该结构图沿用已接受的 Figure 4A 底图，没有叠加当前候选的双核阈值区域。双核由 E 细胞阈值场定义；本报告的实际中心、范围规则和连接参数见后面的“双核位置与参数”。下一页先给出单细胞及突触的实际更新方程。',10.2)

start('神经元与突触动力学｜单神经元动力学')
para('模型采用文章 Methods 的定义：网络包含兴奋性神经元 E 和抑制性神经元 I。每个神经元积累突触输入；膜电位达到阈值后发放一个动作电位，然后回到重置电位。这种模型称为漏电积分发放模型（LIF）。',10.5)
eq('methods_lif',[
 r'$\tau_m^I\frac{dV_i^I}{dt}=-V_i^I+I_i^{I,E}-I_i^{I,I},\qquad i\in I$',
 r'$\tau_m^E\frac{dV_i^E}{dt}=-V_i^E+I_i^{E,E}-z_i(t)I_i^{E,I}-\eta_m m_i(t),\qquad i\in E$'])
para('V 表示膜电位，−V 表示没有输入时膜电位的被动衰减。τm 表示膜电位变化的时间尺度。电流符号 I 的第一个上标表示接收输入的神经元类型，第二个上标表示提供输入的神经元类型。例如带上标 E,I 的电流项表示抑制性神经元传给兴奋性神经元的电流，即 I→E 输入。图中的箭头按“发出→接收”书写，公式的上标则按“接收、发出”排列。',10.5)
para('兴奋性输入通过 AMPA 突触产生，抑制性输入通过 A 型 GABA 受体介导的突触产生。两类电流均用非负幅值表示，膜电位方程中的加号和减号决定兴奋或抑制作用。由于电流已换算成膜电位方程中的有效输入，数值不能直接视为实验记录中的 pA 电流。',10.5)
eq('methods_reset',[
 r'$V_i^X(t)\geq V_\theta^X\ \Rightarrow\ V_i^X\leftarrow V_r^X,\qquad X\in\{E,I\}$'])
para('神经元发放后，在绝对不应期 τref 内保持重置电位，不再发放。E/I 的膜时间常数分别为 20/10 ms，不应期分别为 2/1 ms，重置电位均为 11 mV。具体实验的 E 阈值在空间上变化，详见双核参数页。',10.5)
para('兴奋性神经元还可受两个慢变量调节：m 随自身动作电位积累，产生适应性电流；z 控制接收到的抑制电流有多少真正发挥作用。ηm 决定适应性电流强度。抑制性神经元不使用这两个慢变量。',10.5)
para('与本报告实验的差异｜文章写出包含 z 和 m 的完整模型。数据驱动搜索关闭两个慢变量，相当于固定 z＝1、ηm＝0；人工抑制实验仅按指定时间改变增益。后面的放电率近似模型也未开启自主 z/m。这些实验检验完整模型的不同部分，不能写成已经运行了全部机制。',10.2,color=BLUE)

start('神经元与突触动力学｜突触电流怎样形成')
para('文章用两个连续变化的变量表示一次突触输入：s 接收动作电位到达时的瞬时增量，I 表示随后作用于膜电位的电流。对接收类型 X 和发出类型 Y，方程为：',10.5)
eq('methods_synapses',[
 r'$\frac{ds_i^{X,Y}}{dt}=-\frac{s_i^{X,Y}}{\tau_r^Y}+\sum_{j\in\mathcal{Y}}J_{ij}^{X,Y}\sum_k\delta(t-t_j^k-d_{ij})$',
 r'$\tau_d^Y\frac{dI_i^{X,Y}}{dt}=-I_i^{X,Y}+s_i^{X,Y}$',
 r'$s_i^{X,Y}((t_j^k+d_{ij})^+)=s_i^{X,Y}((t_j^k+d_{ij})^-)+J_{ij}^{X,Y}$'])
para('t 表示发放时刻，下标 j 指神经元，上标 k 指第几次发放；d 表示从 j 传到 i 的延迟。δ 表示到达瞬间发生一次输入，J 表示这次输入使 s 增加多少；上标 +/− 表示到达之后／之前。没有新输入时，s 按 τr 衰减，I 按 τd 跟随 s，因此电流先上升再下降，形成双指数时间过程。',10.5)
table([['突触类型','上升时间 τr','衰减时间 τd'],['AMPA：E→E 和 E→I','0.7 ms','3.5 ms'],['GABA：I→E 和 I→I','1 ms','当前参考为 20.612 ms；历史基线为 18 ms']],widths=[200,110,W-394],size=9.8)
para('与代码参数的对应｜文章中的 J 已经是 s 的实际瞬时增量。代码的基础参数 w 与 J 不是同一个数：生成连接时，程序先把 w 乘以接收神经元的膜时间常数，再除以上升时间常数。未经其他参数调整的基础连接满足：',10.5)
eq('methods_weight_mapping',[
 r'$J_{ij,\mathrm{code}}^{X,Y}=\frac{\tau_m^X}{\tau_r^Y}\,w_{XY,\mathrm{code}}A_{ij}^{X,Y}$'])
para('因此，文章写 J＝wA 时，必须说明文章的 w 指实际输入增量，还是程序中尚未换算的 w。若表格沿用程序的 w，公式需要保留上式的时间常数比；若表格给实际增量，应相应换算表格。当前候选还在基础连接上调整各类通路及单条 EE 连接的强度，见双核参数页。',10.2,color=BLUE)
para('数值计算与外部输入｜程序以 0.1 ms 步长，依次更新 s、电流 I 和膜电位。随机背景输入也进入 AMPA 通路：各神经元按指定瞬时率独立接收 Poisson 脉冲；部分实验还让输入率随时间和位置相关波动（OU 过程）。本次提供的 Methods 片段未包含外部输入定义，不能把省略理解为实验没有背景输入。',10.2)

start('神经元与突触动力学｜适应与抑制效能变化')
para('脉冲发放适应｜每当兴奋性神经元发放一个动作电位，适应变量 m 增加 1；没有新发放时，m 按时间常数 τadp 衰减。积累的 m 通过负的适应性电流减少兴奋性神经元的净输入。',10.5)
eq('methods_adaptation',[
 r'$\frac{dm_i}{dt}=-\frac{m_i}{\tau_{\mathrm{adp}}}+\sum_k\delta(t-t_i^k),\qquad m_i((t_i^k)^+)=m_i((t_i^k)^-)+1$',
 r'$I_i^{\mathrm{adp}}=\eta_m m_i$'])
para('ηm 控制每次发放增加多少适应性电流。持续发放可使负反馈逐渐增强，但是否足以终止网络高活动，仍取决于适应强度和网络输入，不能只凭方程中存在 m 就认定模型会自行恢复。',10.2)
para('活动依赖性抑制耗竭｜每个兴奋性神经元有一个 0–1 之间的抑制效能 z。z＝1 时接收到的抑制完整生效，z 降低时有效抑制减弱。文章方程为：',10.5)
eq('methods_depletion',[
 r'$I_{i,\mathrm{eff}}^{E,I}=z_i I_i^{E,I},\qquad 0\leq z_i\leq1$',
 r'$\tau_z\frac{dz_i}{dt}=z_{\infty,i}-z_i,\qquad z_{\infty,i}=H(I_{\mathrm{th}}^{E,I}-I_i^{E,I})$'])
para('当原始抑制电流低于阈值 Ith 时，z 趋向 1；当电流达到或高于阈值时，z 趋向 0。τz 同时控制耗竭和恢复的速度。为与文章的分段说明及代码一致，这里的阶跃函数 H 在输入恰为 0 时取 0。判断依据是乘 z 之前的抑制电流，不是已经减弱的有效电流。',10.5)
para('生理解释的范围｜z 将持续 GABA 能活动引起的抑制效能下降合并表示为一个慢过程。模型没有计算氯离子浓度、GABA 反转电位或转运体状态，因此可以讨论抑制失效的整体作用，但不能仅由 z 的下降确认某一种具体离子机制。',10.2)
para('与本报告结果的差异｜图 8 在原有 OU 背景下只把人工给定的 z(t) 乘到 E 神经元的 GABA 电流上，没有求解上面的 z 方程，也没有开启 m。人工恢复抑制后退出高态已经观察到；按活动自行恢复尚未验证。如果高态中的抑制电流一直超过 Ith，现有 z 方程会继续耗竭，需要检验适应等负反馈能否先降低活动，再允许 z 回升。',10.2,color=BLUE)

start('网络连接结构｜文章定义与实际连接生成')
para('文章把神经元放在二维空间，以距离决定哪些神经元更容易连接。CXY 表示接收类型 X 从发出类型 Y 接收多少条连接。仅 E→E 有方向偏好，其余三类连接只依赖距离。',10.5)
eq('methods_indegree',[
 r'$C_{EE}=800,\quad C_{IE}=800,\quad C_{EI}=200,\quad C_{II}=200$',
 r'$l_{EE}=0.380\ \mathrm{mm},\qquad l_{EI}=l_{IE}=l_{II}=0.250\ \mathrm{mm}$'])
para('文章的空间核可以拆成下列等价写法。ξ 是两细胞之间的位移，ϱY 是发出群体的面密度，l 控制连接范围，ρ 控制方向偏好；R 是按椭圆尺度计算的距离。',10.2)
eq('methods_spatial_kernel',[
 r'$R_{XY}(\xi)=\frac{\sqrt{\xi_1^2-2\rho_{XY}\xi_1\xi_2+\xi_2^2}}{l_{XY}\sqrt{1-\rho_{XY}^2}}$',
 r'$p_{XY}(\xi)=\frac{C_{XY}}{\varrho_Y}\frac{e^{-R_{XY}(\xi)}}{2\pi l_{XY}^2\sqrt{1-\rho_{XY}^2}}$',
 r'$\rho_{EE}\ne0,\quad \rho_{EI}=\rho_{IE}=\rho_{II}=0$'])
para('传导延迟仍按文章写法：固定延迟加上距离除以传导速度。本报告使用 τ0＝0.1 ms、vaxon＝0.3 mm/ms，再将结果取到 0.1 ms 的模拟时间格。',10.2)
eq('methods_delay',[r'$d_{ij}=\tau_0+\|\xi_{ij}\|/v_{\mathrm{axon}}$'])
para('需要统一的差异｜文章先写“固定入度”，随后又写每条连接独立抽样（Bernoulli 抽样）。独立抽样只能控制平均连接数，不能保证每个细胞恰好接收固定数量。当前代码按 exp(−R) 给每个来源细胞赋相对权重，再不重复地选择 C 个来源，并排除自连接。因此，本报告实验采用的是固定数量的加权抽样，不采用文章末尾的独立 Bernoulli 抽样。',10.2,color=BLUE)
para('方向参数的扩展｜文章只写 ρ，没有独立的旋转角。当前 EE 连接使用长轴角度 θ 和长短轴比，并在优化时重新分配已有边的强度。相同的方向偏好不意味着空间范围完全相同；参数映射与其他差异集中列于下一页。',10.2)

start('与文章 Methods 的差异｜哪些一致，哪些需要补充')
para('以下按当前已完成实验核对。正文沿用文章的膜电位、突触、m 和 z 方程；实际运行时关闭哪些项、改变哪些参数，必须另外说明。本次整理未改动文章原稿或模型代码。',10.4)
table([['项目','文章 Methods','本报告实验／需补充的说明'],['膜电位与突触正负号','先接收类型、后发出类型；兴奋加、抑制减','一致。电流数值是进入膜方程的有效驱动，需与实验 pA 电流区分。'],['z 和 m','完整模型包含两种活动依赖变量','数据驱动及当前放电率模型未开启；图 8 人工指定 z(t)，其余干预还可能改变 I 神经元输入。'],['发放阈值','同一类型写作统一 VθX','E 神经元使用双核空间阈值与固定个体差异；I 保持 18 mV。双核并未在提供的 Methods 片段中定义。'],['连接抽样','“固定入度”与独立 Bernoulli 同时出现','代码按距离权重不重复选定 C 个来源；排除自连接。两种抽样方式需在文章中统一。'],['突触权重','J＝wA，J 为一次到达的实际增量','程序基础 w 先乘 τm/τr；应统一正文、参数表和代码的符号对应。'],['EE 空间方向','由 ρ 控制方向偏好','当前用可旋转椭圆和轴比；长短轴尺度为 l√轴比、l/√轴比，并对已有边重新分配权重。'],['连接强度','每类连接使用固定 w','当前候选调 E→E、E→I、I→E 的全局倍率；EE 各边还受方向重加权。I→I 倍率固定为 1。'],['背景输入','本次粘贴片段未给出','模拟保留随机外源脉冲；部分实验叠加相关输入率波动。需在完整 Methods 中补足或回引。'],['数值求解','连续时间方程','原网络以 0.1 ms 更新；放电率近似另外引入空间分组和响应时间，见后面专节。']],widths=[78,165,W-327],size=9.4)
para('方向参数的具体区别｜在文章的椭圆核中，ρ＞0 时长轴沿 45°，长短尺度分别为 l√(1＋ρ)、l√(1−ρ)。程序使用可旋转的长短轴比。以 ρ＝0.6、轴比＝2 为例，二者长短比相同；但若都令 l＝0.380 mm，文章的尺度为 0.481/0.240 mm，程序为 0.537/0.269 mm。两套参数不能只换名称而不换尺度。',10,color=BLUE)
para('本报告的约定｜涉及完整生理模型时使用文章符号；涉及已有结果时，明确标注实际开启的机制和参数。关于氯离子等具体机制，保留文章的概括性解释，不把尚未直接计算的量写成实验结果。',10)

start('双核位置与参数｜双核阈值与全局 EE 方向偏好')
para('双核不是两组被指定轮流放电的振子。对每个 E 细胞，先计算其距两个中心的较小距离，再选取最近的 1,499 个细胞作为核区；选中细胞的 h＝1，其余为 0。中心位置改变时重新选取细胞，总数保持不变，因此网络边界附近的核区可能不是完整圆形。',10)
eq('eq_core',[
 r'$d_i=18-v_i^{\rm core},\quad v_i^{\rm core}\sim {\cal N}(17.5,1^2),\quad v_i^{\rm core}>11$',
 r'$V_{\theta,i}=18-\gamma h_i d_i\quad(i\in E),\qquad V_{\theta,i}=18\quad(i\in I)$'])
para('每个 E 细胞使用预先固定的随机数生成阈值，搜索时不重新抽样。γ 调整阈值相对 18 mV 的偏移程度：原本低于 18 mV 的会更低，原本高于 18 mV 的会更高。因此增大 γ 不等于所有核内细胞统一降阈值。本轮未额外调整阈值的离散程度；C1 的 γ＝0.910。',10)
para('EE 各向异性作用于整张网络的现存 E→E 边，核内、核外均有方向偏好；它不只存在于两核之间。连接随椭圆距离衰减；全局轴是无向轴，没有预先指定沿轴哪一端先放电。搜索只改变已有连接的相对强度，保留连接对象和传导延迟。',10)
eq('eq_axis',[
 r'$R_{ij}(\theta,\eta)=\sqrt{(d_{\parallel}/(\ell\sqrt{\eta}))^2+(d_{\perp}/(\ell/\sqrt{\eta}))^2}$',
 r'$b_{ij}=\exp\{{\rm clip}[-R_{ij}(\theta,\eta)+R_{ij}(\theta_0,\eta_0),-20,20]\}$',
 r'$\widetilde W_{ij}^{EE}=W_{ij}^{0}b_{ij}\,\frac{\sum_k W_{ik}^{0}}{\sum_k W_{ik}^{0}b_{ik}},\qquad W_{ij}^{EE}=g_{EE}\widetilde W_{ij}^{EE}$'])
para('i 为接收端，j 为发放端；ℓ 是固定的连接长度尺度，η 是轴比，θ 是轴角。R 表示椭圆距离，b 是相对原连接的调整倍率，W 表示连接强度，clip 表示把指数限制在 −20 至 20。先调整方向偏好并保持每个 E 细胞的总 EE 输入，再用 gEE 整体缩放。本轮不额外改变局部区域的连接强度；E→I、I→E 各有一个全局倍率，I→I 倍率固定为 1。',9.8)
table([['C1 及后续机制实验的参考参数','实际值'],['双核中心（mm）','(3.064,18.720)；(12.824,15.091)'],['全局 EE / E→I / I→E','1.159 / 1.205 / 1.110'],['GABA 衰减时间；EE 设定轴角 / 轴比','20.612 ms；−29.682° / 1.936']],widths=[220,W-304],size=9)
para('这里沿用偏上的已有参考网络，以如实对应已完成结果；它尚未确认为最终患者间期工作点。设定的 EE 轴、实际连接的加权方向、两核连线、事件传播方向是四个不同量。',9.2,color=GRAY)

start('评价定义｜优化多次事件的分布，不要求一次事件包含两类')
para('每组固定参数生成连续 24 s 轨道。模拟电极按归一化高斯权重汇总附近神经元活动；至少 8/15 个接触点达到固定检测条件形成群体候选。每通道需持续超过阈值至少 4 ms，并按固定扩展／合并规则组合。250 ms 形态窗相互重叠的事件均不纳入主要分析样本。每个合格事件可以接近 TA、TB 或患者支持范围以外；两类不分别重置网络或指定刺激。',10.4)
table([['观测层','定义与科学问题','是否进入 v2.1 目标'],['参与','逐接触点是否参与；是否招募了正确的点？','是，明确区分未参与接触点'],['顺序','参与点内的相对顺序 rank，0 最早、1 最晚；谁先谁后？','是；通道对逆序表为诊断'],['相对时差','接触点活动时间质心减最早质心，单位 ms','是；不同于局部波形宽度'],['空间概括量','参与、偏早、偏晚位置中心及电极参与比例','是；未指定 TB 的弯折路线'],['独立输出审阅','直接记录的二维活动起源、连续路径、局部包络、事件间隔及核内节律','不在原目标内；用于检验遗漏']],widths=[78,272,W-434],size=9.1)
para('每个事件形成 53 维观测，经预先固定的缩放和核特征变换（用于比较多项观测的联合分布）得到 512 维特征 X。下面的 Doff 扣除每个模型事件与自身比较带来的贡献；t 是患者训练数据变换后的平均特征，VN 是模型特征到其均值的平均平方距离。负值允许存在，不截为零；它也不是逐事件方差越小越好的惩罚。',9.8)
eq('eq_loss',[
 r'$D_{\rm off}=\|\bar X-t\|^2-\frac{V_N}{N-1},\qquad V_N=\frac{1}{N}\sum_e\|X_e-\bar X\|^2$',
 r'$L_{\rm run}=\frac{1}{2}\frac{D_{\rm global}}{a_{\rm global}}+\frac{1}{2}\frac{D_{\rm mode}}{a_{\rm mode}},\qquad L=\langle L_{\rm run}\rangle_{\rm runs}$'])
para('a 为预先固定的正尺度。模式部分使用事先由患者训练数据确定的分类规则及患者自然比例（TA 66.59%、TB 33.41%），兼顾类别出现和类内事件特征；不强制各半。任何一次正式评分运行的合格事件不足 16 个时，都不能只平均其余运行的得分。模型事件有时间依赖，去自配对校正不自动使其成为独立样本。N≥16 是为了估计短窗事件分布而设置的最低样本数，不应直接拿来淘汰自然低事件率工作点。',9.8)
para('优化先用 Sobol 方法分散选择初始参数，再用差分进化反复组合、比较和保留参数（DE/rand/1/bin）。参数可连续取值；每次模拟使用同一组参数生成多次事件。11 维包括两核 XY（4）、阈值倍率（1）、三条连接倍率（3）、GABA 衰减时间（1）、EE 轴角和轴比（2）。独立随机重复阶段用两张新连接网络、每张两段随机输入，四条完整运行等权，不能把其中数百个事件当数百次独立实验。',9.8)
para('已完成的比较：C1 确认 L＝0.979，旧联合工作点 C2 为 1.079；四个配对单位中 3 个改善。分量变化主要来自模式项 0.208→0.124，整体项仅 0.871→0.855。所以下面先看原始观测，而不把总分下降作为模型已经恢复患者机制的结论。',9.8)

start('图 1｜分布中心接近时，散布是否也相容？',True)
figure('ranges',333)
explanation('分别看 TA/TB，检查参与规模、时差尺度与二维位移，避免用相似的平均方向掩盖过宽分布。',
'患者 TA/TB 为 19,563/10,486 个合格事件；四条 C1 运行分开统计。点＝中位数，粉线＝均值，粗线＝25–75%，细线＝5–95% 事件范围；均非置信区间。位移由早／晚各三分之一参与接触点的中心相减。',
'TA 横向位移均值接近患者，但纵向标准差 为 5.06–6.02 mm，患者为 3.24 mm；TA 质心跨度均值 75–87 ms，患者 55 ms。TB 也有跨度偏长、纵向散布偏宽。覆盖患者范围不等于排除了不合理模式。',
'已继续分解到逐通道的参与、rank 和时差（图 2），以定位“总参与数相似却参与了不同的点”等误差。此空间位移是接触点摘要，不能定位真实起燃或证明完整 TB 路径。',size=9)

start('图 2｜哪些接触点的参与与先后关系仍不匹配？')
figure('contacts',442)
explanation('逐点检验相似性；先看按自然比例生成的全部事件，再用 TA/TB 条件行定位偏差。',
'黑线为患者，四种颜色为两张连接网络、每张两段随机输入。参与率用全部相应事件；rank 和质心时差仅在该点参与时计算，线为中位数、带／误差棒为 中间 50% 的事件范围。三列分别回答参与、次序、相对时间。',
'全部事件中 ICL11 参与率患者 84.4%，C1 仅 57.1–65.2%。TB 的 ICL1 应偏早，患者 rank 中位数 0.167，模型为 0.364–0.618；这支持 TB 的右侧早期组织仍未恢复。条件标签参与了目标，不能将两条模式行称为全新独立检验。',
'已检查完整时间包络与直接记录的二维活动（图 5–6），检验接触点质心顺序是否掩盖窄爆发和两核错相。命名的 TA/TB 路线仍留作验证，没有作为逐事件强制目标。',size=9.5)

start('图 3｜模型反复爆发的时间规律像患者吗？')
figure('rhythm',445)
explanation('逐事件内部顺序拟合不约束事件之间的间隔。这里直接比较患者事件、模型接触事件、模型核区爆发，检查事件检测与电极采样方法是否改变表面规则性。',
'双方用 23.5 s 窗、10 ms 起点计数、4 s Welch 段；功率谱的 0.5–10 Hz 面积归一。累积分布仅用窗内完整间隔；患者变异系数（间隔标准差／均值）来自 ≥8 事件的 1,814 个活跃窗。四行是时间形状分支的四个候选，不是图 1 的四个随机重复。',
'患者间隔中位数 0.594 s，模型接触事件逐运行 0.286–0.354 s，核区为 0.250–0.293 s；患者活跃窗变异系数中位数 0.613，核区为 0.119–0.343。接触事件的变异系数 虽可接近患者，核内活动仍更快、更规则。',
'已补做筛选审计：528 个完整接触检测事件中，97 个因形态窗重叠排除，主图红线为保留的 431 个。筛选前的间隔为 0.284–0.324 s，仍偏快。另完成历史参数配对（图 4）。患者为归档 46,683 事件，区别于图 1–2 的 30,049 事件。',size=9.3)

start('图 4｜哪些参数真正改变了核内重复爆发？',True)
# The old last footer line contains a false dose claim. Crop footer only, retain all axes/data.
figure('interventions',319,crop=(0,.069,1,1))
explanation('用实际单参数干预区分反馈作用，而非从多参数联合搜索的散点相关性推断因果。',
'三个历史双核位置×四个配对随机数种子；每次仅改变一项，12 s 轨道去前 0.5 s，两核指标先在运行内平均。五列的低／中／高取值见图顶；失控持续高活动的运行不作为节律状态绘入，状态表保留。',
'GABA 18→24 ms：典型间隔 248→285 ms，12/12 配对单元同向；EE×0.85：间隔 332 ms、变异系数（CV） 0.512、峰活跃比例 27.1%（基线 66.8%）。增加抑制路径强度也改变规则性，故不存在已识别的单一“节拍旋钮”。',
'已完成峰门槛与接触筛选审计：GABA 延长使完整接触间隔 273→303 ms，但筛选后反为 489→321 ms，说明不能只看筛选序列。脉冲核验显示固定单次输入增量时，完整电流积分面积不随衰减时间变化；旧末行剂量图注已撤去。',size=9)

start('图 5｜为什么仅拟合质心与 rank 仍然不够？')
figure('envelopes',347)
explanation('接触点质心给出一个时间坐标，却可能遗漏一次局部活动有多宽、上升下降是否合理、不同点是否重叠。',
'左列用 Fig. 2C 患者 TA/TB 代表事件；其余列为 C1 和参考位置的完整接触包络。固定通道顺序和真实 250 ms 时间窗，每通道按自己的峰值归一；未参与通道仍显示。两侧幅度单位不同，不能比较绝对能量。',
'患者局部活动较宽且重叠，模型多为短而错开的爆发。64 个诊断波形（TA/TB 各 32）局部宽度中位数约 88/95 ms，C1 约 18 ms。质心跨度偏长和局部波形偏窄可以同时发生，并不矛盾。',
'已做离线可辨识性检验：对称展宽不改变原 L＝0.979，压缩质心时差则能大幅降分。因此原目标确实看不见一部分形状误差。随后已执行时间波形初步试验，结果见下表。',size=9.8)
table([['已做的补充','结果及其含义'],['加入接触 t50、上升／下降质量时间及不对称','46 个训练波形；18 个按块留后诊断波形，均属开发材料'],['固定三种布局，物理联合扰动五个参数','训练最佳新 L＝2.261；两个提名在另一段随机输入×留后块各 0/2 改善'],['解释','新目标暴露了遗漏，但小批参数变化的训练收益未保留；不能直接据此冻结新工作点']],widths=[195,W-279],size=9)

start('图 6｜电极记录之下，两个核区实际怎样活动？')
figure('core_rhythm',385)
explanation('检查模型是否主要由窄而同步的核内群体爆发组成，以及低热点惩罚是否已改变这种状态。该诊断不使用主要分析中的接触事件筛选，也不用 TA/TB 标签。',
'读取 16 条已有 24 s 轨道；只取完全位于各核内部的 2 ms 直接记录的活跃细胞数，覆盖每个核区 216–355 个 E 细胞。图示固定 5–8 s 片段，谱和自相关使用完整有效记录。峰检测平滑仅用于定位，峰活跃比例读取原始计数。',
'另一段随机输入下四种候选的峰内活跃比例约 75–88%，主要重复频率 3.25–3.75 Hz。低热点惩罚候选反而更规则；两个核的错相可降低零延迟相关，却不能证明它们独立。此处频率是爆发重复率，不是 HFO 载波。',
'已补充患者同窗时间对照及配对参数分析（图 3–4），并新增逐神经元放电图（图 7–8）。另一次直接记录的二维活动审阅发现 C1 的 TA/TB 中核1先到峰分别约 45%/48%，尚无稳定的“两类对应两起源”证据。',size=10)
para('解释上的关键替代项：接触点的两类先后顺序，可能部分来自强爆发的核间错相及事件检测与电极采样方法筛选。现有结果尚未证明这一解释足够，也尚未排除真实传播；因此必须继续保留直接记录的二维活动与逐事件波形，而不能只给类别均值图。',10,color=BLUE)

start('图 7｜抑制作用于哪些细胞，决定了高活动的形式')
figure('inhibition_cycle',545)
explanation('比较抑制作用范围：同时减弱 E/I 所受抑制，与仅减弱 E 所受抑制，是否产生相同的网络状态？',
'固定同一双核与参数，6.5 s 内人工指定增益 1→0.5→1，全程保留膜电位、突触和延迟历史。A：全部 GABA 输入增量；B：E/I 两群的 GABA 电流；C：仅 E 的 GABA 电流；D：A 加回原 OU 背景。前三行保留相同 Poisson 输入。群体率使用全部 32,000 E 和 8,000 I，放电图为固定分层抽样。',
'A/B/D 主要表现为增强的反复爆发，C 出现持续高率平台，恢复抑制后均可回到间歇活动。C 保留了 I→I 抑制，说明抑制的作用范围会改变高活动的形式。',
'为排除 C 的进出仅在关闭 OU 时存在，已完成同一 C 参数下开启原有 OU 的对照，见图 8。该对照直接检查随机背景是否破坏高态进出。',size=9.2)

start('图 8｜C＋OU 开启：原参数下进入高态，再回到间歇活动',True)
figure('raster_c_ou',295)
explanation('保留原有随机背景后，仅调节 E 所受抑制，是否仍能进入并退出高态？这直接检验是否需要关闭 OU 或更换参数才能出现切换。',
'同一连接网络和阈值，输入随机种子 9108401，GABA 衰减 20.612 ms；恢复原有全局和空间 OU，保留逐细胞 Poisson 输入。E 所受 GABA 电流乘人工增益：0–1 s 为 1，1–2 s 降至 0.5，2–3.5 s 保持，3.5–4.5 s 恢复到 1。I 所受抑制不缩放，全程不重置状态。',
'E 平均率在初始窗（0.5–1 s）为 13.12 Hz，低抑制窗（2–3.5 s）为 454.87 Hz，恢复后（5.5–6.5 s）为 13.94 Hz；I 对应为 33.36、575.47、35.38 Hz。恢复后的 E 平均率接近初始窗，且 56% 的 5 ms 时间段低于 1 Hz，说明重新出现有间隔的活动。',
'这条单种子对照支持在同一参考网络、原有 OU 背景下可逆进入高率态；低抑制阶段的 E 率 5–95% 范围为 453.78–456.11 Hz，仍主要是高率平台。人工增益不是自主 Z，本结果也未定位分岔类型。下一页只保留从人工对照转向活动依赖 Z 的关键检验。',size=9)

start('沿 C＋OU 推进｜由活动驱动 Z，而非人工指定增益')
para('图 8 确定了后续的参照条件：在同一双核、相同连接和阈值、原有 OU 背景下，人工减弱并恢复 E 所受抑制，可以进入并退出高率态。后续保持这些条件，检验网络活动能否通过 Z 自行推动状态变化。',10.5)
para('方程沿用第 5 页，不重复列写。每个 E 神经元的 Z 由自身接收的原始 GABA 电流决定：低于阈值时恢复，高于或等于阈值时耗竭；M 随动作电位积累并减少净兴奋。Z 与人工增益的作用位置相同，但时间轨迹的来源不同。',10.5)
table([['需要回答的问题','检验与判读'],['Z 能否由间歇活动积累而下降？','固定图 8 的网络和 OU 输入规则，开启逐神经元 Z。记录每个细胞接收的电流、超过阈值的时间比例、Z 和放电率；不预设进入时刻。平均 Z 下降本身不等于已经发生状态切换。'],['什么促使网络从高态回来？','原规则在 GABA 电流持续超过阈值时继续耗竭，不会因进入高态而自动恢复。需要检查已有 M 能否先减弱活动，再允许 Z 回升；与只开启 Z、只开启 M 比较。'],['回来后是否保留原有活动？','同时检查核内、核外和全网络放电，区分高率平台与大幅振荡。确认恢复后仍有可自行结束的事件；患者 TA/TB 传播分布另作独立检验。'],['实际跨越了哪条动力学边界？','在仅调节 E 所受抑制的条件下分析平衡态和周期态。全局 q 同时改变 I→E 与 I→I，其既有分岔线不能直接用于解释 C。']],widths=[155,W-239],size=10)
para('图 7–8 是人工增益实验，均未开启自主 Z/M。它们证明该网络在原有随机背景下具备高态进出的能力；活动积累能否自主实现这一过程，需要另外用完整轨道检验。',10.3,color=BLUE)
para('后续以 C＋OU 为参照条件。自主 Z 的中途下降或短暂放电变化不作为完成依据；先确认活动积累与进入高态的因果关系，再解释自然退出，同时检查原有间歇活动是否保留。',10,color=GRAY)

start('简化模型 1｜每个空间格的方程是什么？')
para('这里的放电率模型保留双核与空间连接。将原网络的 40,000 个细胞按位置分为 10×10 格，每格各有 E/I 放电率；从真实连接按来源格、接收格和延迟汇总平均输入矩阵 W。E 阈值以每格 8 个代表值及相应权重来近似。它不同于无空间、仅一个 E/I 群体的 均匀混合模型。',10.5)
eq('eq_rate_drive',[
 r'$D_{a\leftarrow b,i}^{n}=q_{ab}\sum_{j,d}W_{a\leftarrow b,ij}^{(d)}r_{b,j}^{n-d},\quad q_{aI}=q,\quad q_{aE}=1$',
 r'$g_{ab,i}^{n+1}=e^{-\Delta t/\tau_{r,b}}g_{ab,i}^{n}+\Delta t\frac{\tau_{m,a}}{\tau_{r,b}}D_{a\leftarrow b,i}^{n}$',
 r'$c_{ab,i}^{n+1}=e^{-\Delta t/\tau_{d,b}}c_{ab,i}^{n}+(1-e^{-\Delta t/\tau_{d,b}})g_{ab,i}^{n+1}$',
 r'$\mu_{E,i}=c_{EE,i}-c_{EI,i}+c_{{\rm ext},E,i},\quad \mu_{I,i}=c_{IE,i}-c_{II,i}+c_{{\rm ext},I,i}$',
 r'$r_{a,i}^{n+1}=r_{a,i}^{n}+\frac{\Delta t}{\tau_{{\rm rate},a}}[\Phi_{a,i}(\mu_{a,i}-\Delta_{a,i},\sigma_{a,i})-r_{a,i}^{n}]$'])
para('本页 ab 下标统一为“接收 a←来源 b”；例如 EI 表示 I→E。g/c 分别是突触上升状态与电流状态，外源也经过 AMPA 两级滤波。内部放电率单位为每毫秒发放次数，图转为 Hz。τrate 为经响应诊断选定的 E 5 ms、I 2.5 ms，区别于膜时间常数 20/10 ms。',10)
para('延迟不是一个平均值：358 个延迟时间段、最大 35.8 ms 全部携带历史。当前方差近似仍用当前格率，均值驱动保留延迟；这项近似的准确性必须通过 SNN 对照检验，不能由方程来自同一网络自动保证。',10)
table([['保留','作出的近似'],['双核阈值、真实连接权重与延迟分布','每格同类细胞以一个平均放电率描述'],['AMPA/GABA 上升与衰减状态','单次动作电位输入用输入均值／方差替代'],['神经元阈值异质性与不应期','经验阈值积分；输入与平均放电率的对应关系，加有限响应时间'],['空间场随时间演化','当前实验关闭 OU、Z/M；外源 Poisson 用确定性的输入均值和方差的近似']],widths=[225,W-309],size=9.1)
para('这种放电率模型可用来检验“这套反馈是否具有持续爆发的能力”，但在单事件响应、噪声和传播分布与 SNN 充分对应前，不能作为已验证的能够保留原网络动力学的简化。',9.8,color=BLUE)

start('简化模型 2｜非线性、GABA 时间常数与稳定性')
para('Φ 将输入的均值和波动换算成平均放电率，具体采用电流型 LIF 的 Siegert 公式。积分上下限为 重置电位、发放阈值 相对均值的标准化距离；E 对经验阈值节点加权，I 使用统一阈值。',10)
eq('eq_transfer',[
 r'$\Phi(\mu,\sigma;V_\theta)=\left[\tau_{\rm ref}+\tau_m\sqrt{\pi}\int_{(V_{\rm reset}-\mu)/\sigma}^{(V_\theta-\mu)/\sigma}e^{u^2}(1+{\rm erf}\,u)\,du\right]^{-1}$',
 r'$\sigma_{a}^{2}=v_{a}^{E}+v_{a}^{I},\quad v_a^E=\tau_{m,a}(Q_{aE}r_E+J_{{\rm ext},a}^{2}\nu_{\rm ext}),\quad v_a^I=\tau_{m,a}q^2Q_{aI}r_I$',
 r'$\Delta_a=\frac{2.065}{2}\sqrt{\frac{v_a^E(\tau_{r,A}+\tau_{d,A})+v_a^I(\tau_{r,G}+\tau_{d,G})}{\tau_{m,a}}}$'],height=113)
para('Q 由真实边权平方汇总，区别于均值矩阵 W；所有式子逐空间格应用。Δ 是已实现的有限突触相关时间修正，在代码中从均值扣除。它是一种输入均值和方差的近似，并不是独立拟合出的生理资源。GABA 衰减时间除进入 c 的时间滤波外，也进入 Δ，因此边界移动不能全部归因于一个平均抑制量。',10)
para('q 和 τGABA 的区别',12,color=BLUE,bold=True)
para('q 缩放新 GABA 突触输入增量，使递归平均输入乘 q、方差乘 q²；τGABA 改变滤波的时间响应。当前归一化滤波对一个单位 s 增量 的完整电流面积为下式，与衰减时间无关。该结论是单次输入本身的性质；相互反馈的网络因放电数变化，累计抑制当然仍可变化。',10)
eq('eq_area',[r'$\Delta t\sum_{n\geq0}I^n=\frac{\Delta t}{1-e^{-\Delta t/\tau_{r,G}}}\quad(s^0=1,\ I^{-1}=0)$'],height=37)
para('本版核验时间步长 0.1 ms、上升时间 1 ms、衰减时间 12/18/24/42 ms：积分均为 1.050833，误差低于 10⁻¹²。若改变时间步长，s 更新对应的面积会改变，所以数值步长检验需保持原模拟的突触响应面积；不应把改时间步带来的工作点变化叫作分岔消失。',9.8)
para('究竟分析谁的特征值？',12,color=BLUE,bold=True)
para('稳定性分析针对放电率、突触状态及全部延迟历史构成的一步更新。描述小扰动随时间变化的完整线性系统有 72,600 个状态，另有 400 个可消去的外源衰减状态；不是 2×2 E–I Jacobian，也不是原始脉冲网络的 Jacobian。一对复共轭特征值的模从小于 1 变为大于 1，意味着扰动逐步放大，再由保持面积的连续延迟极限确认 Hopf 型振荡失稳。',10)
para('旧近似仅分析平均放电率的导数矩阵（Jacobian），可以求平衡点及其出现或消失的位置（鞍结分岔），却不能代表含突触和延迟的振荡稳定性。后续所有分岔图均使用修正后的方程和完整小扰动稳定性检验；τGABA＝42 ms 仍找到振荡失稳边界，不能因某条均值轨道平坦就说“42 ms 不振荡”。',9.8)

start('图 9｜参数变化是否真的导致失稳和放电状态切换？',True)
figure('bifurcation',310)
explanation('区分固定点分支、稳定性变化与实际持续爆发，避免把平均率的一次跳变直接命名为 Hopf。',
'在固定参考网络上逐步改变参数、追踪平衡点，计算完整状态每一步更新时小扰动的放大或缩小程度；q–GABA 平面追踪局部边界。另从低率状态比较相同局部 20 ms、2 mV 刺激，改变 q 后连续携带所有状态。',
'τGABA＝20.612 ms 时 q≈0.73805 出现约 3.99 Hz 的 Hopf 型振荡失稳；q＝0.76 的有限事件回落，降至 0.72 后维持约 2.80 Hz 大爆发，无重复刺激也持续。低／高率平衡点发生鞍结分岔的位置另在 q≈0.70886/0.40622；它们不是同一个切换点。',
'已做固定面积步长细化和完整根计数；q＝0.740 无不稳定根，0.736 有两个。随后检查条件零导数曲线、真实轨道与固定参数下的共存（图 10–11）。未收敛的延续点保留为空，不填作无分岔。',size=9)

start('图 10｜零导数曲线 能说明什么，不能说明什么？')
figure('nullclines',415)
explanation('解释为什么平衡点仍存在时，系统也可以从回落变成持续振荡；同时避免把高维系统的二维投影误当完整动力学。',
'上排固定各 q 平衡点的 E/I 空间形状，只改变两个群体幅度；突触与延迟历史设为对应稳态，计算人数加权平均率导数的零线。下排是完整动态模型真正走出的 E–I 投影轨道。',
'q＝0.76 与 0.72 的条件零线都有交点，但完整延迟系统前者稳定、后者振荡失稳。真实轨道不会始终停留在这个二维条件面上，因此轨道的前进方向不一定等于图中的二维箭头。稳定性结论来自完整状态的稳定性分析，不来自两线交点外观。',
'已进一步固定 q＝0.75，以不同初态和一次较强刺激检验低率与爆发是否共存（图 11）。尚未计算第一 Lyapunov 系数或延续小振幅周期支，因此本报告不正式命名超临界／亚临界 Hopf。',size=10)
para('这张图的用途是帮助读者理解“交点存在”和“交点稳定”是两个问题。它不能替代完整状态空间里的不同长期状态之间的初态分界，也不能直接展示 Z/M 随积累演化的轨迹。',10,color=BLUE)

start('图 11｜持续化是否必须跨分岔？是否等于全局化？')
figure('bistability',380)
explanation('检验另一种切换解释：控制参数不变时，有限扰动是否足以改变最终进入的长期状态；并把爆发持续性与空间招募范围分开。',
'q＝0.75 固定，对比小扰动与一次 5 mV、20 ms 核附近刺激；延长高初态轨道检查持续性。空间图用晚期单次爆发各格首次超过 20 Hz 的时间，按格内 E 细胞数统计招募。',
'小扰动回到约 0.073 Hz，强脉冲后维持约 2.58 Hz 爆发，支持低率与爆发态共存。q＝0.72 时代表爆发的招募并集约 77.8%，q＝0.50 为 100%；同一时刻招募和事件内依次招募不能混用。',
'已增加空间招募参数点：q 从 0.75 降至 0.50，最大同时招募约 43.7%→99.0%。目前呈逐步扩大，未识别独立的“全局化分岔”。周期支连接、不同长期状态之间的初态分界和 Z/M 活动驱动跨界仍待检验。',size=10)
para('因此，观察到跳变后有持续爆发，至少要区分两条机制：参数改变使原先的长期状态失去稳定性，或有限扰动在同时存在的长期状态之间切换。当前模型已经显示两种可能，不能只保留第一种解释。',10,color=BLUE)

start('合作者讨论｜当前证据可以支撑到哪里？')
table([['科学问题','已有证据','尚不能写成'],['同一网络可否产生不同事件？','固定网络连续噪声产生多事件；C1 有 TA/TB 分类事件','两个标签已证明两个起源核区 或两条完整传播路径'],['数据驱动是否有效？','部分参与、rank、空间分布相近；独立重复的平均损失降低','患者所有方差恢复、独立泛化成功或全局最优'],['为什么审计节律？','核内活动更快、更规则；模拟电极检测事件变异系数可掩盖底层规则性','患者具有同样核内同步或同一生理振子'],['抑制能否改变放电状态？','SNN 全局减抑制增强爆发，仅减弱 E 所受抑制时，原 OU 背景下也可进出高率态；放电率模型有持续振荡','该 SNN 已证明与放电率模型同一临界点或已复现临床发作'],['分岔是否存在？','修正后的放电率模型：Hopf 型失稳、鞍结分岔、状态共存','已确定 Hopf 超／亚临界，或发现单独全局化分岔'],['积累能否解释跨越？','C＋OU 已支持外部恢复下可退出；Z 的自主回升及负反馈仍待检验','已经完成放电积累驱动 Z/M 自主跨界']],widths=[110,190,W-384],size=9.4)
para('对当前工作的建议',13,color=BLUE,bold=True)
para('保留 C1 为可复现实验参考，接受其逐事件特征改善以及修正放电率模型能产生相关活动的结果。患者间期参数组合暂不冻结：参与位置、TB 早期顺序、局部波形与间隔规律仍有系统差距，不能靠继续降低同一个总分自动消除。',10.5)
para('下一项会改变判断的工作',13,color=BLUE,bold=True)
para('患者匹配线应先在固定事件检测与电极采样方法下同时保留逐事件空间组织、局部波形与事件间隔诊断，确认候选改善不由检测筛选产生，再决定是否把序列统计纳入新目标。机制线应验证相同条件的脉冲网络与放电率模型响应与持续态对应，并延续周期分支或确定不同初态会进入哪一种共存状态。C 的自主 Z/M 试验可以在固定参考网络上先行，作为模型动力学能力的检验，无需等待最终患者间期参数组合确认；同时保留脉冲网络与放电率模型对应与患者分布两条独立检验。再根据完整轨道判断是跨越失稳边界，还是在同时存在的长期状态之间切换。',10.5)
para('这些是供讨论的后续方案，本次报告整理没有执行新搜索或 Z/M 仿真。已有局部时间波形初步试验、直接记录的二维活动／节律审计、逐神经元放电干预和修正放电率模型分岔都已在前页标为完成；避免把“已经发现缺口”和“已经修好缺口”混写。',10,color=GRAY)

start('来源与版本｜每张图由什么实验产生？')
para('图编号为本报告内部编号。原图均来自已有计算输出，按题意重新选择与排列，未调整点位、坐标或结果。公式及其余图保留矢量；图 3 的源 PDF 中文嵌字渲染失败，改用同次绘图的原始高分辨率 PNG。图 4 仅撤去错误的末行积分剂量图注，具体更正与单脉冲复算另存。第 16–18 页保留一次四条件合图，以 C＋OU 为主要进出对照，再说明活动依赖 Z 的关键检验；原有重复展开页已合并。',10)
table([['图','来源包 / producer','统计或实验单位'],['模型总览','Figure 4A / build_fig4_panel_a_combined.py','局部 E/I 与患者电极几何示意；未表示当前双核候选'],['1–2','multievent_distribution_search_v2_1 / observable_distribution_recovery；all_event_parameter_distributions','现代 E10 30,049 合格事件；C1 四条 24 s 运行分开'],['3–4','patient_model_rhythm_comparison / compare_topic4_patient_core_rhythm.py','归档 E10 46,683 事件；8 次使用另一段随机输入的确认运行；另有历史配对参数实验'],['5','fig2c_contact_envelope_comparison / contact_envelopes_all_contacts','Fig. 2C 代表波形与模型完整模拟电极包络；64 波形审计另计'],['6','core_burst_rhythm_audit / audit_topic4_core_burst_rhythm.py','四候选×两张连接网络、每张两段随机输入，共 16 轨道'],['7–8（含 C＋OU）','snn_raster_inhibition_transition_v1 / run_topic4_snn_raster_transition.py；run_topic4_e_only_ou_followup.py','同一连接网络和随机数种子的 6.5 s 四种作用对照；新增 C 开启原 OU 背景'],['9–11','corrected_rate_bifurcation_v1 / analyze_topic4_corrected_bifurcation.py；simulate_topic4_bifurcation_crossing.py','固定参考网络的确定性放电率模型、含延迟的稳定性分析及非线性轨道']],widths=[40,278,W-402],size=9)
para('模型 Methods：保留作者本次提供的原文（author_model_methods.tex）。连接抽样与权重换算另核对 snn_engine/connectivity.py、connectivity_rot.py；z/m 对应 snn_engine/mz_slow_vars.py。双核及背景输入为本报告实验的补充，不能据此推定文章其他章节没有相关定义。',9.2)
para('公式的代码核对',12,color=BLUE,bold=True)
para('SNN：topic4_raster_protocol_engine.py / membrane_step 与 simulate_kick；阈值：topic4_core_field_rev9.py / reconstruct_node_from_h；双核与参数：topic4_zm_ictal_transition.py、topic4_multidimensional_parameters.py；EE 椭圆重加权：topic4_rev20_dual_core_mechanism.py。',9.6)
para('Rate：screen_topic4_corrected_high_activity.py、topic4_patient_zm_meanfield.py；稳定性：analyze_topic4_corrected_bifurcation.py、check_topic4_delayed_rate_spectrum.py。reference 配置为 topic4_rate_model_dynamics_validation_v1.json；该早期配置的状态字段不代替后续 corrected 结果状态。',9.6)
para('交付包含 PDF、逐图原始路径与快照、正文源、公式矢量资产、渲染预览及版面检查记录。旧版独立保留。C＋OU 的协议和逐窗统计见 e_only_ou_followup_protocol.json、e_only_ou_comparison.json。Agent 完成版面与来源自查；作者／合作者目视审阅仍待进行。',9.6)
assert y>34,(page,y);page_log[-1]['content_bottom_pt']=y;c.save()
writer=PdfWriter();base=PdfReader(str(OUT/'text_layout.pdf'))
for record,target in zip(page_log,base.pages):
    for f in record['figures']:
        source=copy.deepcopy(PdfReader(str(OUT/'assets'/f"{f['id']}.pdf")).pages[0]);fw=float(source.mediabox.width);fh=float(source.mediabox.height)
        x0,y0,x1,y1=f['crop_bottom_origin'];xx,yy,ww,hh=f['bbox_pt'];scale=ww/(fw*(x1-x0))
        source.cropbox.lower_left=(x0*fw,y0*fh);source.cropbox.upper_right=(x1*fw,y1*fh)
        target.merge_transformed_page(source,Transformation().scale(scale).translate(xx-x0*fw*scale,yy-y0*fh*scale))
    writer.add_page(target)
    writer.add_outline_item(record['title'],len(writer.pages)-1)
writer.add_metadata({'/Title':'双核空间 SNN：方程、患者对照与动力学证据','/Author':'HFOsp research project'})
with PDF.open('wb') as stream:writer.write(stream)
used={f['id'] for p in page_log for f in p['figures']}
manifest=[m for m in manifest if m['id'] in used]
(OUT/'figure_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
(OUT/'layout_manifest.json').write_text(json.dumps(page_log,ensure_ascii=False,indent=2))
(OUT/'report_text.json').write_text(json.dumps(textlog,ensure_ascii=False,indent=2))
print(str(PDF));print('pages',page,'lowest content',min(p['content_bottom_pt'] for p in page_log))
