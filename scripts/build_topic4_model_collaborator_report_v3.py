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
OUT=ROOT/'results/topic4_sef_hfo/model_collaborator_report_v3_2026-09-08'
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
 'raster_c':RASTER/'figures/raster_z_current_e.pdf',
 'raster_base':RASTER/'figures/raster_base.pdf',
 'raster_jump':RASTER/'figures/raster_jump.pdf',
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

PDF=OUT/'model_collaborator_report_v3.pdf'
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
    para('双核空间 SNN｜合作者讨论稿 · v3',size=8,color=GRAY,space=0)
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
    fig=plt.figure(figsize=(8.8,.48*len(lines)+.12))
    for k,line in enumerate(lines):fig.text(.02,1-(k+.65)/(len(lines)+.2),line,fontsize=16,va='center')
    path=OUT/'assets'/f'{key}.pdf';fig.savefig(path,facecolor='white');plt.close(fig)
    manifest.append({'id':key,'source':'equations verified against code; authored in this report','snapshot':str(path.relative_to(OUT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    figure(key,height or 30*len(lines)+8)
def explanation(why,how,result,next_,size=9.8):
    for label,txt in [('检验问题',why),('制作与读法',how),('结果与解释',result),('已完成的衔接',next_)]:para(label+'｜'+txt,size=size,space=7)

start('从患者间期传播到持续爆发：模型究竟解释了什么？')
para('合作者讨论稿 · 第 3 版 · 2026-09-08',10,color=GRAY,space=18)
para('核心问题',14,color=BLUE,bold=True)
para('同一张固定网络能否在连续随机输入下产生多次、自行结束的事件，使不同事件分别接近患者 TA/TB 的传播分布；在这个网络骨架上，抑制反馈的变化又能否使自限事件转为持续爆发？前一个问题约束患者特异性，后一个问题检验动力学能力。二者需要分别成立，才能讨论间期到发作样状态的联系。',11)
para('目前的判断',14,color=BLUE,bold=True)
para('数据驱动候选改善了部分逐事件统计，仍有通道参与、时差和局部波形偏差。核内活动呈较快、较规则的重复爆发，接触点读出会使这种规则性看起来减弱。因此，“能分到 TA/TB 两类”还不等于恢复了患者的传播机制。',11)
para('在独立机制实验中，降低 SNN 的 GABA 输入幅度确实增强放电与招募；E-only 电流干预还出现持续高率态并可在恢复抑制后退出；修正后的空间 rate 模型则存在 Hopf 型失稳、自限／持续爆发转换和状态共存。尚未证明患者间期工作点、SNN 与 rate 模型共享同一临界点，也尚未接通活动积累—Z/M 演化—自主跨界这条链。',11)
table([['阅读顺序','本节要回答什么'],['模型定义与优化（第 2–4 页）','双核在哪里进入方程？哪些参数可调？训练的样本究竟是什么？'],['患者对照与节律（第 5–10 页）','分布哪些部分相似、哪些不相似？为什么进一步审计波形和核内活动？'],['SNN 干预与 Z（第 11–15 页）','四条件对照，尤其 C 的高态进入／退出，及自主 Z 方案。'],['空间 rate 与分岔（第 16–20 页）','近似保留什么？什么数学对象失稳？状态如何切换？'],['讨论与来源（第 21–22 页）','哪些结论可用，关键缺口在哪里，图由哪些已有实验产生？']],widths=[160,W-244],size=9.3)
para('本版将旧报告的损失面板和联合扰动散点页撤换为原始观测量对照。已有图保留数据及坐标；单参数图的错误 GABA 面积图注依据代码与脉冲核验更正。本报告不新增网络搜索，不替换正式 Fig. 5。',9,color=GRAY)

start('模型定义 1｜神经元怎样接收输入并产生 spike？')
para('空间 SNN 由 32,000 个兴奋性 E 神经元和 8,000 个抑制性 I 神经元组成，位于 20×20 mm 平面。E/I 递归连接具有距离与延迟，已移除 autapse 自连接；E 的局部阈值场定义两个 core。以下为当前关闭 Z/M、适应及其他可选反馈时的实际电流型 LIF 更新，时间步 Δt＝0.1 ms。',10.5)
eq('eq_snn',[
 r'$s_{a,i}^{n+1}=e^{-\Delta t/\tau_{r,a}}s_{a,i}^{n}+A_{a,i}^{n}$',
 r'$I_{a,i}^{n+1}=e^{-\Delta t/\tau_{d,a}}I_{a,i}^{n}+(1-e^{-\Delta t/\tau_{d,a}})s_{a,i}^{n+1}$',
 r'$V_i^{n+1}=e^{-\Delta t/\tau_{m,i}}V_i^n+(1-e^{-\Delta t/\tau_{m,i}})(I_{E,i}^{n+1}-I_{I,i}^{n+1})$',
 r'$V_i^{n+1}\geq V_{\theta,i}\ \Rightarrow\ {\rm spike},\quad V_i\leftarrow V_{\rm reset},\quad t_{\rm ref,i}\ {\rm applied}$'])
para('s 是接收突触事件的上升状态，I 是再次滤波后的有效驱动；a 表示 AMPA 或 GABA。A 汇总这一时间步按真实轴突延迟到达的递归突触 jump；AMPA 还加入独立 Poisson 外源事件。递归 jump 已含目标细胞 τm/τrise 与边权，不能在方程中再乘一次。I 采用与膜电位相容的有效驱动单位，不能直接当实验电流 pA。',10)
para('每步先衰减 s，再加到达输入，再更新 I 和膜电位；不应期内膜电位保持 reset。连续写法 τm·dV/dt＝−V＋IE−II 有助于理解，但本报告数值结果和稳定性核验均以实际更新顺序为准。',10)
table([['参数','E / AMPA','I / GABA','作用'],['膜时间常数 τm','20 ms','10 ms','膜电位对驱动的响应速度'],['不应期 τref','2 ms','1 ms','spike 后暂不再次发放'],['突触 rise / decay','0.7 / 3.5 ms','1 / 可调 ms','先上升、后衰减的反馈时间结构'],['阈值 / reset','核外 18 / 11 mV','18 / 11 mV','核内 E 阈值另见下一页']],widths=[108,95,95,W-382],size=9)
para('Data-driven 运行保留全局及空间 OU 调制，并在给定瞬时率下抽取逐神经元 Poisson 输入；没有逐事件指定 TA/TB 或点火时刻。后面的 raster 机制对照关闭 OU，但保留 Poisson；rate 则使用确定性矩闭合。三者的输入层不同。',9.7)

start('模型定义 2｜双核阈值与全局 EE 方向偏好')
para('双核不是两组被指定轮流放电的振子。对每个 E 细胞，先计算距两个中心的较小距离，选择最近的固定 1,499 个细胞；由此确定共有的距离截止值和二值 h。位置改变时重新选择细胞，核心预算保持不变，边界可能截断圆形。',10)
eq('eq_core',[
 r'$d_i=18-v_i^{\rm core},\quad v_i^{\rm core}\sim {\cal N}(17.5,1^2),\quad v_i^{\rm core}>11$',
 r'$V_{\theta,i}=18-\gamma h_i d_i\quad(i\in E),\qquad V_{\theta,i}=18\quad(i\in I)$'])
para('每个 E 细胞的抽样分位数预先固定。γ 是有符号阈值偏移的倍率：它同时放大核内的较低和较高阈值，不能解释为所有核内细胞统一降阈值。当前搜索的阈值离散度收缩参数固定为 1；C1 的 γ＝0.910。',10)
para('EE 各向异性作用于整张网络的现存 E→E 边，核内、核外均有方向偏好；它不只存在于两核之间。连接随椭圆距离衰减；全局轴是无向轴，没有预先指定沿轴哪一端先放电。搜索改变既有边的相对权重，保留拓扑和延迟。',10)
eq('eq_axis',[
 r'$R_{ij}(\theta,\eta)=\sqrt{(d_{\parallel}/(\ell\sqrt{\eta}))^2+(d_{\perp}/(\ell/\sqrt{\eta}))^2}$',
 r'$b_{ij}=\exp\{{\rm clip}[-R_{ij}(\theta,\eta)+R_{ij}(\theta_0,\eta_0),-20,20]\}$',
 r'$\widetilde W_{ij}^{EE}=W_{ij}^{0}b_{ij}\,\frac{\sum_k W_{ik}^{0}}{\sum_k W_{ik}^{0}b_{ik}},\qquad W_{ij}^{EE}=g_{EE}\widetilde W_{ij}^{EE}$'])
para('i 为接收端，j 为发放端；ℓ 为冻结连接长度尺度。归一化先保持每个目标 E 细胞的总 EE 输入，gEE 再整体缩放。E→I、I→E 也有各自全局倍率；本轮局部学习边场关闭，I→I 倍率固定 1。',9.8)
table([['本报告 C1 / rate / raster 参考','实际值'],['双核中心（mm）','(3.064,18.720)；(12.824,15.091)'],['全局 EE / E→I / I→E','1.159 / 1.205 / 1.110'],['GABA decay；EE 名义轴角 / 轴比','20.612 ms；−29.682° / 1.936']],widths=[220,W-304],size=9)
para('这里沿用偏上的已有参考底物，以如实对应已完成结果；它未被验收为最终患者间期工作点。名义 EE 轴、加权实际边轴、两核连线、事件传播方向是四个不同量。',9.2,color=GRAY)

start('评价定义｜优化多次事件的分布，不要求一次事件包含两类')
para('每组固定参数生成连续 24 s 轨道。接触读出用近邻神经元的归一化高斯空间权重；至少 8/15 个接触点达到固定检测条件形成群体候选。每通道需持续超阈值 ≥4 ms，并按固定扩展／合并规则组合。250 ms 形态窗相互重叠的事件从 primary 样本中双侧排除。每个合格事件可以接近 TA、TB 或患者支持范围以外；两类不分别重置网络或指定刺激。',10.4)
table([['观测层','定义与科学问题','是否进入 v2.1 目标'],['参与','逐接触点是否参与；是否招募了正确的点？','是，保留未参与掩码'],['顺序','参与点内归一化 rank；谁先谁后？','是；通道对逆序表为诊断'],['相对时差','接触点活动时间质心减最早质心，单位 ms','是；不同于局部波形宽度'],['空间摘要','参与、偏早、偏晚位置中心及电极参与比例','是；未指定 TB 的弯折路线'],['独立输出审阅','原生场起源、连续路径、局部包络、事件间隔及核内节律','不在原目标内；用于检验遗漏']],widths=[78,272,W-434],size=9.1)
para('每个事件形成 53 维观测，经冻结缩放和核映射得到 512 维特征 X。下面 Doff 去掉模型事件的自配对贡献；t 是患者 FIT 目标的均值特征。VN＝平均平方离差。负值允许存在，不截为零；它也不是逐事件方差越小越好的惩罚。',9.8)
eq('eq_loss',[
 r'$D_{\rm off}=\|\bar X-t\|^2-\frac{V_N}{N-1},\qquad V_N=\frac{1}{N}\sum_e\|X_e-\bar X\|^2$',
 r'$L_{\rm run}=\frac{1}{2}\frac{D_{\rm global}}{a_{\rm global}}+\frac{1}{2}\frac{D_{\rm mode}}{a_{\rm mode}},\qquad L=\langle L_{\rm run}\rangle_{\rm runs}$'])
para('a 为冻结的正尺度。模式项使用 FIT 患者分类器及患者自然比例（TA 66.59%、TB 33.41%），兼顾类别出现和类内表征；不强制各半。任何正式评分运行事件数不足 16 时不部分平均。模型事件有时间依赖，去自配对校正不自动使其成为独立样本。N≥16 是短窗形态分布的估计支持约定，不应直接拿来淘汰自然低事件率工作点。',9.8)
para('优化方法是连续参数的 Sobol 初始覆盖加分批差分进化 DE/rand/1/bin，不是梯度训练。11 维包括两核 XY（4）、阈值倍率（1）、三条连接倍率（3）、GABA decay（1）、EE 轴角和轴比（2）。确认期用两张新拓扑×两段噪声，四条完整运行等权，不能把其中数百个事件当数百次独立实验。',9.8)
para('已完成的比较：C1 确认 L＝0.979，旧联合工作点 C2 为 1.079；四个配对单位中 3 个改善。分量变化主要来自模式项 0.208→0.124，整体项仅 0.871→0.855。所以下面先看原始观测，而不把总分下降作为模型已经恢复患者机制的结论。',9.8)

start('图 1｜分布中心接近时，散布是否也相容？',True)
figure('ranges',333)
explanation('分别看 TA/TB，检查参与规模、时差尺度与二维位移，避免用相似的平均方向掩盖过宽分布。',
'患者 TA/TB 为 19,563/10,486 个合格事件；四条 C1 运行分开统计。点＝中位数，粉线＝均值，粗线＝25–75%，细线＝5–95% 事件范围；均非置信区间。位移由早／晚各三分之一参与接触点的中心相减。',
'TA 横向位移均值接近患者，但纵向 SD 为 5.06–6.02 mm，患者为 3.24 mm；TA 质心跨度均值 75–87 ms，患者 55 ms。TB 也有跨度偏长、纵向散布偏宽。覆盖患者范围不等于排除了不合理模式。',
'已继续分解到逐通道的参与、rank 和时差（图 2），以定位“总参与数相似却参与了不同的点”等误差。此空间位移是接触点摘要，不能定位真实起燃或证明完整 TB 路径。',size=9)

start('图 2｜哪些接触点的参与与先后关系仍不匹配？')
figure('contacts',442)
explanation('逐点检验相似性；先看按自然比例生成的全部事件，再用 TA/TB 条件行定位偏差。',
'黑线为患者，四种颜色为两拓扑×两噪声。参与率用全部相应事件；rank 和质心时差仅在该点参与时计算，线为中位数、带／误差棒为 IQR。三列分别回答参与、次序、相对时间。',
'全部事件中 ICL11 参与率患者 84.4%，C1 仅 57.1–65.2%。TB 的 ICL1 应偏早，患者 rank 中位数 0.167，模型为 0.364–0.618；这支持 TB 的右侧早期组织仍未恢复。条件标签参与了目标，不能将两条模式行称为全新独立检验。',
'已检查完整时间包络与原生二维活动（图 5–6），检验接触点质心顺序是否掩盖窄爆发和两核错相。命名的 TA/TB 路线仍留作验证，没有作为逐事件强制目标。',size=9.5)

start('图 3｜模型反复爆发的时间规律像患者吗？')
figure('rhythm',445)
explanation('逐事件内部顺序拟合不约束事件之间的间隔。这里直接比较患者事件、模型接触事件、模型 core 爆发，检查观察器是否改变表面规则性。',
'双方用 23.5 s 窗、10 ms 起点计数、4 s Welch 段；PSD 的 0.5–10 Hz 面积归一。CDF 仅用窗内完整间隔；患者 CV 来自 ≥8 事件的 1,814 个活跃窗。四行是时间形状分支的四个候选，不是图 1 的四个 seed。',
'患者间隔中位数 0.594 s，模型接触事件逐运行 0.286–0.354 s，core 为 0.250–0.293 s；患者活跃窗 CV 中位数 0.613，core 为 0.119–0.343。接触读出 CV 虽可接近患者，底层 core 仍更快、更规则。',
'已补做筛选审计：528 个完整接触检测事件中，97 个因形态窗重叠排除，主图红线为保留的 431 个。筛选前的间隔为 0.284–0.324 s，仍偏快。另完成历史参数配对（图 4）。患者为归档 46,683 事件，区别于图 1–2 的 30,049 事件。',size=9.3)

start('图 4｜哪些参数真正改变了核内重复爆发？',True)
# The old last footer line contains a false dose claim. Crop footer only, retain all axes/data.
figure('interventions',319,crop=(0,.069,1,1))
explanation('用实际单参数干预区分反馈作用，而非从多参数联合搜索的散点相关性推断因果。',
'三个历史双核位置×四个配对 seed；每次仅改变一项，12 s 轨道去前 0.5 s，两核指标先在运行内平均。五列的低／中／高取值见图顶；runaway 不作为可测节律绘入，状态表保留。',
'GABA 18→24 ms：典型间隔 248→285 ms，12/12 配对单元同向；EE×0.85：间隔 332 ms、CV 0.512、峰活跃比例 27.1%（基线 66.8%）。增加抑制路径强度也改变规则性，故不存在已识别的单一“节拍旋钮”。',
'已完成峰门槛与接触筛选审计：GABA 延长使完整接触间隔 273→303 ms，但筛选后反为 489→321 ms，说明不能只看筛选序列。脉冲核验显示固定 jump 的完整积分面积不随 decay 变化；旧末行剂量图注已撤去。',size=9)

start('图 5｜为什么仅拟合质心与 rank 仍然不够？')
figure('envelopes',347)
explanation('接触点质心给出一个时间坐标，却可能遗漏一次局部活动有多宽、上升下降是否合理、不同点是否重叠。',
'左列用 Fig. 2C 患者 TA/TB 代表事件；其余列为 C1 和参考位置的完整接触包络。固定通道顺序和真实 250 ms 时间窗，每通道按自己的峰值归一；未参与通道仍显示。两侧幅度单位不同，不能比较绝对能量。',
'患者局部活动较宽且重叠，模型多为短而错开的爆发。64 个诊断波形（TA/TB 各 32）局部宽度中位数约 88/95 ms，C1 约 18 ms。质心跨度偏长和局部波形偏窄可以同时发生，并不矛盾。',
'已做离线可辨识性检验：对称展宽不改变原 L＝0.979，压缩质心时差则能大幅降分。因此原目标确实看不见一部分形状误差。随后已执行时间形状 pilot，结果见下表。',size=9.8)
table([['已做的补充','结果及其含义'],['加入接触 t50、上升／下降质量时间及不对称','46 个训练波形；18 个按块留后诊断波形，均属开发材料'],['固定三种布局，物理联合扰动五个参数','训练最佳新 L＝2.261；两个提名在新噪声×留后块各 0/2 改善'],['解释','新目标暴露了遗漏，但小批参数变化的训练收益未保留；不能直接据此冻结新工作点']],widths=[195,W-279],size=9)

start('图 6｜接触读出之下，两个 core 实际怎样活动？')
figure('core_rhythm',385)
explanation('检查模型是否主要由窄而同步的核内群体爆发组成，以及低热点惩罚是否已改变这种状态。该诊断不筛选接触 primary 事件，也不用 TA/TB 标签。',
'读取 16 条已有 24 s 轨道；只取完全位于各核内部的 2 ms 原生活跃计数，覆盖每核 216–355 个 E 细胞。图示固定 5–8 s 片段，谱和自相关使用完整有效记录。峰检测平滑仅用于定位，峰活跃比例读取原始计数。',
'新噪声下四种候选的峰内活跃比例约 75–88%，主要重复频率 3.25–3.75 Hz。低热点惩罚候选反而更规则；两个核的错相可降低零延迟相关，却不能证明它们独立。此处频率是爆发重复率，不是 HFO 载波。',
'已补充患者同窗时间对照及配对参数分析（图 3–4），并新增真正逐神经元 raster（图 7–8）。另一次原生场审阅发现 C1 的 TA/TB 中核1先到峰分别约 45%/48%，尚无稳定的“两类对应两起源”证据。',size=10)
para('解释上的关键替代项：接触点的两类先后顺序，可能部分来自强爆发的核间错相及观察器筛选。现有结果尚未证明这一解释足够，也尚未排除真实传播；因此必须继续保留原生场与逐事件波形，而不能只给类别均值图。',10,color=BLUE)

start('图 7｜真实 SNN spike 中，自限事件是否存在？',True)
figure('raster_base',315)
explanation('在讨论持续化之前，直接看完整网络是否具有可自行结束、可重复出现的放电事件。',
'固定 C1 底物、topology 6101、单 seed 9108401，模拟 6.5 s；OU 关闭，独立 Poisson 保留，q＝1。Raster 用真实 0.1 ms spike，固定抽样 60 核A E、60 核B E、120 核外 E、60 I；群体率用全体细胞、5 ms bin。',
'1–6.5 s 有 15 个描述性事件，E 峰 42.1–69.1 Hz，全 E 平均 6.23 Hz；74.5% 的 bin 低于 1 Hz。均值包含爆发，并不是静息基线一直 6 Hz。核内富集抽样也不能用来估计全场参与率。',
'同一张图、同一 seed 已执行连续状态的 GABA 干预（图 8）。本图证明该协议下有自限事件，尚不是患者间期分布验收，也不能用它的 OU-off 事件间隔替代图 3 的 OU-on 比较。',size=9)

start('图 8｜降低抑制幅度会把网络推向什么状态？',True)
figure('raster_jump',312)
explanation('在保留实际 spike、突触电流和延迟历史的网络中，检验抑制反馈减弱能否提高招募，而不是只在 rate 近似中看到强活动。',
'与图 7 同底物同 seed。全局 GABA jump gain q：0–1 s 为 1，1–2 s 降至 0.5，2–3.5 s 保持，3.5–4.5 s 回升至 1。q 同时作用 I→E 和 I→I；不是 Z 的自主耗竭，也不是仅调 I→E。',
'低 q 保持期全 E 平均率 64.97 Hz、峰 281.29 Hz；恢复后的 5.5–6.5 s 均值 3.48 Hz、峰 48.89 Hz。低 q 仍有 37.7% 的 bin 低于 1 Hz，故应称更强反复爆发与招募，不能只凭峰值命名无间断 runaway。',
'已用同一参考图的 corrected rate 模型分别求固定点、完整延迟稳定性和非线性轨道（图 9–11）。人工恢复 q 说明外部可逆干预，不证明模型能自主结束发作；下面的 rate 临界值亦未被直接转移到 SNN。',size=9)

start('图 8（续）｜四种抑制干预：同一状态连续进入与退出')
figure('inhibition_cycle',590)
para('同一双核、同一初态，6.5 s 内指定增益 1→0.5→1，全程保留膜电位、突触状态和延迟历史。A：全部 GABA jump；B：E/I 两群 GABA 电流；C：仅 E 的 GABA 电流；D：A 加回原 OU 背景。前三行保留相同 Poisson 输入。',9.3)
para('本图新增的关键是 C：仅改变抑制的目标群体，便从 A/B 的强反复爆发转为持续高率态，恢复抑制后又回到间歇活动。C 的数值与振荡判读见下一页；不能把四行统一称为同一种高态。',9.3,color=BLUE)

start('图 8C｜确认高态可逆切换，区分持续发放与大幅振荡',True)
figure('raster_c',305)
explanation('检验与原 Z 相同的 E-only 作用范围，是否足以让完整空间 SNN 进入高活动态并在恢复抑制后退出。',
'仅 E 使用 IE−z(t)II；I 仍为 IE−II。低增益窗 2–3.5 s，恢复后窗 5.5–6.5 s；群体率用全体细胞。除既有 5 ms 图，本次从保存的 0.1 ms spike 率复算 1 ms 与区域率，没有新增网络仿真。',
'C 的低增益 E/I 均值为 454.87/575.61 Hz；恢复后为 4.56/11.96 Hz，80.5% 的 5 ms bin 中 E<1 Hz。高态 E 的 1 ms SD 仅 1.24 Hz，核 A/B 均值约 462/461 Hz。因此支持高率态进入—退出，未显示 A/B 那样的大幅群体振荡；微小快波纹不能等同持续 burst。',
'已核验干预前 1 s 的三行 spike 完全一致。B→C 只保留 I 自身的 GABA 抑制，改变了 I→I 反馈；这是作用范围敏感性的直接对照。后续以 C 的 E-only 电流形式接入活动依赖 Z，不直接沿用全局 q 的 rate Hopf 临界值。',size=9)

start('沿 C 推进｜从人工增益轨迹到活动依赖 Z')
para('C 已给出可继续的物理基础：外部恢复有效抑制时，这张网络可以从高态回来。下一步要让网络自身的状态决定何时降低与恢复抑制；目标是恢复“进入—维持—退出”的因果过程，而不是逐点拟合人工线性斜坡。',10.4)
para('现有代码中与 C 相对应的 Z/M 方程',12,color=BLUE,bold=True)
eq('eq_zm_scope',[
 r'$I_{{\rm net},i}=I_{E,i}-z_i I_{I,i}-\eta_m m_i\quad(i\in E),\qquad I_{{\rm net},i}=I_{E,i}-I_{I,i}\quad(i\in I)$',
 r'$\tau_z\dot z_i={\bf 1}[I_{I,i}<I_{\rm th}]-z_i$',
 r'$\dot m_i=-m_i/\tau_{\rm adp}+\sum_k\delta(t-t_i^k)$'],height=102)
para('Z 乘在已滤波的 GABA 电流上，仅 E 的 Z 演化；M 是 spike 驱动的适应负反馈，当前 C 两者都未自主开启。已有 Z 规则在原始 GABA 电流持续高于阈值时满足 dz/dt＝−z/τz，因此只会进一步耗竭。C 低增益窗的原始 E-target GABA 电流群体均值约 1,630，基线约 45.8（模型驱动单位）；这提示高态时的恢复是关键缺口，但均值不能代替逐细胞阈值占空比。',10)
table([['下一里程碑的顺序','具体检验与判断标准'],['1. 固定 C 的作用范围与参考底物','保留当前几何、连接、阈值和 Poisson；z_i 只乘 E-target GABA 电流。人工轨迹作为阳性参照，不作为闭环输入。'],['2. 先测 Z-only 能否自然进入高态','记录逐细胞电流阈值占空比及 dz/dt；让连续活动驱动 Z，不预设点火或回升时刻。接受依据是活动先改变 Z，随后状态跨越，而非时间脚本触发。'],['3. 明确自然退出缺什么','若 Z-only 锁在高态，检查是否缺乏负反馈。优先测试代码已有 M：能否先减弱活动和抑制电流、再让 Z 回升；同时与 Z-only、M-only 比较，防止 M 只是阻止进入。'],['4. 同时检查高态类型及基底保留','分别报告高率平台、大幅持续振荡及小幅快波纹；检查核内／核外和群体率。恢复后是否仍有自限事件，进入前是否仍覆盖患者 TA/TB，均需独立检验。']],widths=[145,W-229],size=9)
para('不能把本报告的全局 q 分岔线套到 C：前者同时改变 I→E 和 I→I，C 仅改变 E 接受的抑制。应在 E-only 控制下重新定位相关固定点／周期态及吸引域，再检查 Z/M 实际走过哪条边界。图中的下降与回升轨迹有限速率，单凭进出阈值不同也不能直接命名静态迟滞。',10)
para('本次完成的是既有轨道复算与报告补充；上述自主 Z/M 检验尚未执行。先保住用户原始的“间期活动累积导致进入高态”目标；自然退出作为新增机制目标单独验收，不能为实现退出而破坏原有间期传播。',9.5,color=GRAY)

start('简化模型 1｜每个空间格的方程是什么？')
para('这里的 rate 模型保留双核与空间连接。原 40,000 细胞图投影为 10×10 格，每格各有 E/I 放电率；从真实连接按来源格、接收格和延迟汇总平均输入算子 W。E 阈值以每格 8 个经验积分节点保留。它不同于无空间、仅一个 E/I 群体的 well-mixed 模型。',10.5)
eq('eq_rate_drive',[
 r'$D_{a\leftarrow b,i}^{n}=q_{ab}\sum_{j,d}W_{a\leftarrow b,ij}^{(d)}r_{b,j}^{n-d},\quad q_{aI}=q,\quad q_{aE}=1$',
 r'$g_{ab,i}^{n+1}=e^{-\Delta t/\tau_{r,b}}g_{ab,i}^{n}+\Delta t\frac{\tau_{m,a}}{\tau_{r,b}}D_{a\leftarrow b,i}^{n}$',
 r'$c_{ab,i}^{n+1}=e^{-\Delta t/\tau_{d,b}}c_{ab,i}^{n}+(1-e^{-\Delta t/\tau_{d,b}})g_{ab,i}^{n+1}$',
 r'$\mu_{E,i}=c_{EE,i}-c_{EI,i}+c_{{\rm ext},E,i},\quad \mu_{I,i}=c_{IE,i}-c_{II,i}+c_{{\rm ext},I,i}$',
 r'$r_{a,i}^{n+1}=r_{a,i}^{n}+\frac{\Delta t}{\tau_{{\rm rate},a}}[\Phi_{a,i}(\mu_{a,i}-\Delta_{a,i},\sigma_{a,i})-r_{a,i}^{n}]$'])
para('本页 ab 下标统一为“接收 a←来源 b”；例如 EI 表示 I→E。g/c 分别是突触 rise/current 状态，外源也经过 AMPA 两级滤波。内部率单位为 spike/ms，图转为 Hz。τrate 为经响应诊断选定的 E 5 ms、I 2.5 ms，区别于膜时间常数 20/10 ms。',10)
para('延迟不是一个平均值：358 个实际 delay bin、最大 35.8 ms 全部携带历史。当前方差闭合仍用当前格率，均值驱动保留延迟；这项近似的准确性必须通过 SNN 对照检验，不能由方程来自同一网络自动保证。',10)
table([['保留','作出的近似'],['双核阈值、真实连接权重与延迟分布','每格同类细胞以一个 rate 描述'],['AMPA/GABA 上升与衰减状态','离散 spike 统计用输入均值／方差替代'],['神经元阈值异质性与不应期','经验阈值积分；静态 transfer 加有限响应时间'],['空间场随时间演化','当前实验关闭 OU、Z/M；外源 Poisson 用确定性 diffusion closure']],widths=[225,W-309],size=9.1)
para('这种模型可用来检验“这套反馈是否具有持续爆发的能力”，但在单事件响应、噪声和传播分布与 SNN 充分对应前，不能作为已验证的动力学等价降阶。',9.8,color=BLUE)

start('简化模型 2｜非线性、GABA 时间常数与稳定性')
para('非线性 Φ 使用电流型 LIF 的 Siegert 转移函数。积分上下限为 reset、threshold 相对均值的标准化距离；E 对经验阈值节点加权，I 使用统一阈值。',10)
eq('eq_transfer',[
 r'$\Phi(\mu,\sigma;V_\theta)=\left[\tau_{\rm ref}+\tau_m\sqrt{\pi}\int_{(V_{\rm reset}-\mu)/\sigma}^{(V_\theta-\mu)/\sigma}e^{u^2}(1+{\rm erf}\,u)\,du\right]^{-1}$',
 r'$\sigma_{a}^{2}=v_{a}^{E}+v_{a}^{I},\quad v_a^E=\tau_{m,a}(Q_{aE}r_E+J_{{\rm ext},a}^{2}\nu_{\rm ext}),\quad v_a^I=\tau_{m,a}q^2Q_{aI}r_I$',
 r'$\Delta_a=\frac{2.065}{2}\sqrt{\frac{v_a^E(\tau_{r,A}+\tau_{d,A})+v_a^I(\tau_{r,G}+\tau_{d,G})}{\tau_{m,a}}}$'],height=113)
para('Q 由真实边权平方汇总，区别于均值算子 W；所有式子逐空间格应用。Δ 是已实现的有限突触相关时间修正，在代码中从均值扣除。它是一种近似闭合，并不是独立拟合出的生理资源。GABA decay 除进入 c 的时间滤波外，也进入 Δ，因此边界移动不能全部归因于一个平均抑制量。',10)
para('q 和 τGABA 的区别',12,color=BLUE,bold=True)
para('q 缩放新 GABA jump，使递归平均输入乘 q、方差乘 q²；τGABA 改变滤波的时间响应。当前归一化滤波对一个单位 s jump 的完整电流面积为下式，与 decay 无关。该结论是单输入开环性质；闭环网络因放电数变化，累计抑制当然仍可变化。',10)
eq('eq_area',[r'$\Delta t\sum_{n\geq0}I^n=\frac{\Delta t}{1-e^{-\Delta t/\tau_{r,G}}}\quad(s^0=1,\ I^{-1}=0)$'],height=37)
para('本版核验 Δt＝0.1 ms、τrise＝1 ms、decay＝12/18/24/42 ms：积分均为 1.050833，误差低于 10⁻¹²。若改 dt，rise 更新的面积会改变，所以数值步长检验需保持 native 面积；不应把改时间步带来的工作点变化叫作分岔消失。',9.8)
para('究竟分析谁的特征值？',12,color=BLUE,bold=True)
para('稳定性分析针对 rates、突触状态及全部 delay history 构成的实际离散映射。耦合切向系统有 72,600 个状态，另有 400 个可消去的外源衰减状态；不是 2×2 E–I Jacobian，也不是整张 spiking SNN 的 Jacobian。复共轭乘子穿越单位圆，再由保持面积的连续延迟极限确认 Hopf 型过界。',10)
para('旧近似的静态 Jacobian 可以求固定点与 fold，却不能代表含突触和延迟的振荡稳定性。后续所有分岔图均使用修正后的方程和完整切向验证；τGABA＝42 ms 仍找到振荡失稳边界，不能因某条均值轨道平坦就说“42 ms 不振荡”。',9.8)

start('图 9｜参数变化是否真的导致失稳和放电状态切换？',True)
figure('bifurcation',310)
explanation('区分固定点分支、稳定性变化与实际持续爆发，避免把平均率的一次跳变直接命名为 Hopf。',
'在固定参考图上延续固定点，按实际离散切向映射求复共轭谱；q–GABA 平面追踪局部边界。另从低率状态比较相同局部 20 ms、2 mV 刺激，改变 q 后连续携带所有状态。',
'τGABA＝20.612 ms 时 q≈0.73805 出现约 3.99 Hz 的 Hopf 型过界；q＝0.76 的有限事件回落，降至 0.72 后维持约 2.80 Hz 大爆发，无重复刺激也持续。低／高率 fold 另在 q≈0.70886/0.40622；它们不是同一个切换点。',
'已做固定面积步长细化和完整根计数；q＝0.740 无不稳定根，0.736 有两个。随后检查条件 nullcline、真实轨道与固定参数下的共存（图 10–11）。未收敛的延续点保留为空，不填作无分岔。',size=9)

start('图 10｜Nullcline 能说明什么，不能说明什么？')
figure('nullclines',415)
explanation('解释为什么平衡点仍存在时，系统也可以从回落变成持续振荡；同时避免把高维系统的二维投影误当完整动力学。',
'上排固定各 q 平衡点的 E/I 空间形状，只改变两个群体幅度；突触与延迟历史设为对应稳态，计算人数加权平均率导数的零线。下排是完整动态模型真正走出的 E–I 投影轨道。',
'q＝0.76 与 0.72 的条件零线都有交点，但完整延迟系统前者稳定、后者振荡失稳。二维条件面不是不变流形；实际轨道的切向方向不必等于该条件二维场。稳定性结论来自完整谱，不来自两线交点外观。',
'已进一步固定 q＝0.75，以不同初态和一次较强刺激检验低率与爆发是否共存（图 11）。尚未计算第一 Lyapunov 系数或延续小振幅周期支，因此本报告不正式命名超临界／亚临界 Hopf。',size=10)
para('这张图的用途是帮助读者理解“交点存在”和“交点稳定”是两个问题。它不能替代完整状态空间里的吸引域边界，也不能直接展示 Z/M 随积累演化的轨迹。',10,color=BLUE)

start('图 11｜持续化是否必须跨分岔？是否等于全局化？')
figure('bistability',380)
explanation('检验另一种切换解释：控制参数不变时，有限扰动是否足以跨越吸引域；并把爆发持续性与空间招募范围分开。',
'q＝0.75 固定，对比小扰动与一次 5 mV、20 ms 核附近刺激；延长高初态轨道检查持续性。空间图用晚期单次爆发各格首次超过 20 Hz 的时间，按格内 E 细胞数统计招募。',
'小扰动回到约 0.073 Hz，强脉冲后维持约 2.58 Hz 爆发，支持低率与爆发态共存。q＝0.72 时代表爆发的招募并集约 77.8%，q＝0.50 为 100%；同一时刻招募和事件内依次招募不能混用。',
'已增加空间招募参数点：q 从 0.75 降至 0.50，最大同时招募约 43.7%→99.0%。目前呈逐步扩大，未识别独立的“全局化分岔”。周期支连接、吸引域边界和 Z/M 活动驱动跨界仍待检验。',size=10)
para('因此，观察到跳变后有持续爆发，至少要区分两条机制：参数改变使原吸引子失稳，或有限扰动在共存吸引子之间切换。当前模型已经显示两种可能，不能只保留第一种解释。',10,color=BLUE)

start('合作者讨论｜当前证据可以支撑到哪里？')
table([['科学问题','已有证据','尚不能写成'],['同一网络可否产生不同事件？','固定网络连续噪声产生多事件；C1 有 TA/TB 分类事件','两个标签已证明两个起源 core 或两条完整传播路径'],['数据驱动是否有效？','部分参与、rank、空间分布相近；确认平均 loss 降低','患者所有方差恢复、独立泛化成功或全局最优'],['为什么审计节律？','core 更快更规则；读出事件 CV 可掩盖底层规则性','患者具有同样核内同步或同一生理振子'],['抑制能否改变放电状态？','SNN 全局减抑制增强爆发，E-only 可进出高率态；rate 有持续振荡','该 SNN 已证明与 rate 同一临界点或已复现临床发作'],['分岔是否存在？','corrected rate：Hopf 型失稳、fold、状态共存','已确定 Hopf 超／亚临界，或发现单独全局化分岔'],['积累能否解释跨越？','C 已支持外部恢复下可退出；Z 的自主回升及负反馈仍待检验','已经完成放电积累驱动 Z/M 自主跨界']],widths=[110,190,W-384],size=9.4)
para('对当前工作的建议',13,color=BLUE,bold=True)
para('保留 C1 为可复现实验参考，接受其逐事件特征改善及 corrected rate 的机制能力结果。患者间期底物暂不冻结：参与位置、TB 早期顺序、局部波形与间隔规律仍有系统差距，不能靠继续降低同一个总分自动消除。',10.5)
para('下一项会改变判断的工作',13,color=BLUE,bold=True)
para('患者匹配线应先在固定观察器下同时保留逐事件空间组织、局部波形与事件间隔诊断，确认候选改善不由检测筛选产生，再决定是否把序列统计纳入新目标。机制线应验证相同条件的 SNN／rate 响应与持续态对应，并延续周期分支或定位共存吸引域。C 的自主 Z/M 试验可以在固定参考底物上先行，作为机制能力检验，无需等待最终患者间期底物验收；同时保留 SNN／rate 对应与患者分布两条独立检验。再根据完整轨道判断是跨越失稳边界，还是在共存吸引子间切换。',10.5)
para('这些是供讨论的后续方案，本次报告整理没有执行新搜索或 Z/M 仿真。已有局部时间形状 pilot、原生场／节律审计、raster 干预和 corrected rate 分岔都已在前页标为完成；避免把“已经发现缺口”和“已经修好缺口”混写。',10,color=GRAY)

start('来源与版本｜每张图由什么实验产生？')
para('图编号为本报告内部编号。原图均来自已有计算输出，按题意重新选择与排列，未调整点位、坐标或结果。公式及其余图保留矢量；图 3 的源 PDF 中文嵌字渲染失败，改用同次绘图的原始高分辨率 PNG。图 4 仅撤去错误的末行积分剂量图注，具体更正与单脉冲复算另存。新增第 13–15 页收录完整四条件图、C 的细时间尺度判读和自主 Z/M 的下一里程碑。',10)
table([['图','来源包 / producer','统计或实验单位'],['1–2','multievent_distribution_search_v2_1 / observable_distribution_recovery；all_event_parameter_distributions','现代 E10 30,049 合格事件；C1 四条 24 s 运行分开'],['3–4','patient_model_rhythm_comparison / compare_topic4_patient_core_rhythm.py','归档 E10 46,683 事件；8 新噪声确认运行；另有历史配对参数实验'],['5','fig2c_contact_envelope_comparison / contact_envelopes_all_contacts','Fig. 2C 代表波形与模型完整接触读出；64 波形审计另计'],['6','core_burst_rhythm_audit / audit_topic4_core_burst_rhythm.py','四候选×两拓扑×两噪声，共 16 轨道'],['7–8（含 C）','snn_raster_inhibition_transition_v1 / run_topic4_snn_raster_transition.py','同图同 seed 的 6.5 s 四种作用／噪声对照；C 的 1 ms 复算新增'],['9–11','corrected_rate_bifurcation_v1 / analyze_topic4_corrected_bifurcation.py；simulate_topic4_bifurcation_crossing.py','固定参考图的确定性 rate、延迟切向谱及非线性轨道']],widths=[40,278,W-402],size=9)
para('公式的代码核对',12,color=BLUE,bold=True)
para('SNN：topic4_raster_protocol_engine.py / membrane_step 与 simulate_kick；阈值：topic4_core_field_rev9.py / reconstruct_node_from_h；双核与参数：topic4_zm_ictal_transition.py、topic4_multidimensional_parameters.py；EE 椭圆重加权：topic4_rev20_dual_core_mechanism.py。',9.6)
para('Rate：screen_topic4_corrected_high_activity.py、topic4_patient_zm_meanfield.py；稳定性：analyze_topic4_corrected_bifurcation.py、check_topic4_delayed_rate_spectrum.py。reference 配置为 topic4_rate_model_dynamics_validation_v1.json；该早期配置的状态字段不代替后续 corrected 结果状态。',9.6)
para('交付包含 PDF、逐图原始路径与快照、正文源、公式矢量资产、渲染预览及版面检查记录。旧版独立保留。新增 C 的原始输出和 1 ms 复算见 c_high_state_audit.json。Agent 完成版面与来源自查；作者／合作者目视审阅仍待进行。',9.6)
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
