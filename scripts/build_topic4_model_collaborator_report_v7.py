#!/usr/bin/env python3
"""Refresh the collaborator report from completed artifacts; never launch physics.

Each build is retained. The stable PDF and current.json are replaced only after
layout, source, and retained C-figure checks pass. --watch-pilot performs one
terminal-state refresh, then exits, including on pilot failure.
"""
import sys
sys.path.insert(0, '/tmp/hfosp_report_pdf_runtime')
from pathlib import Path
import argparse, csv, datetime, hashlib, html, io, json, os, re, shutil, subprocess, time
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.utils import ImageReader
from pypdf import PdfReader, PdfWriter, Transformation
from PIL import Image

ROOT=Path('/home/honglab/leijiaxin/HFOsp')
OUT=ROOT/'results/topic4_sef_hfo/model_collaborator_report_v7_2026-09-10'
V6=ROOT/'results/topic4_sef_hfo/model_collaborator_report_v6_2026-09-08'
HISTORY=ROOT/'results/topic4_sef_hfo/lowering_only_core_historical_review_20260910'
GEOM=Path('/data/hfosp/topic4_sef_hfo/geometry_threshold_refinement_20260909')
EXTENT=Path('/data/hfosp/topic4_sef_hfo/core_extent_long_propagation_20260909')
PILOT=Path('/data/hfosp/topic4_sef_hfo/core_driven_input_pilot_20260910')
POSITION=Path('/data/hfosp/topic4_sef_hfo/core_position_scl_response_20260910')
CONNECTIVITY_DESIGN=ROOT/'results/topic4_sef_hfo/core_connectivity_search_design_20260910'
SEARCH=Path('/data/hfosp/topic4_sef_hfo/core_connectivity_search_20260910')
BLUE=colors.HexColor('#24536c'); GRAY=colors.HexColor('#536976')
pdfmetrics.registerFont(TTFont('CN','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('LatinBold','/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'))
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def markup(t): return re.sub(r'([\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef]+)',r'<font name="CN">\1</font>',html.escape(str(t)))
def rows(p):
    with Path(p).open() as f:return list(csv.DictReader(f))

def applied_input_figure():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','font.size':10,'pdf.fonttype':42})
    plan=read(PILOT/'plan.json');geometries=plan['candidates'][0]
    fig,axes=plt.subplots(1,4,figsize=(15,4.7),layout='constrained')
    origins=[]
    for j,cid in enumerate(['legacy_mixed_full','core_ou_only']):
        path=PILOT/'canary/units'/cid/'847101/workers/trajectory.npz'
        with np.load(path) as z:xy=z['positions_E'];names=z['contact_names'].astype(str);contacts=z['contact_xy_mm'];delta=z['vtheta'][:len(xy)]-18
        ratepath=path.with_name('input_rates.npz')
        with np.load(ratepath) as z:loading=z['global_ou_loading'][:len(xy)]
        for col,values,title in [(j,delta,('历史' if j==0 else '当前')+'：E 阈值偏移'),(j+2,loading,('历史' if j==0 else '当前')+'：共享 OU 加载')]:
            ax=axes[col]
            im=ax.scatter(*xy.T,c=values,s=.55,cmap='RdBu_r' if col<2 else 'Blues',norm=TwoSlopeNorm(vmin=-4,vcenter=0,vmax=4) if col<2 else None,vmin=0 if col>=2 else None,vmax=1 if col>=2 else None,rasterized=True)
            for shaft,color in [('SCL','#00b7c4'),('ICL','#ef9b22')]:
                sel=np.array([k for k,n in enumerate(names) if n.startswith(shaft)])
                sel=sel[np.argsort([int(names[k][len(shaft):]) for k in sel])]
                ax.plot(*contacts[sel].T,'-o',color=color,ms=3.5,lw=1,markeredgecolor='black',markeredgewidth=.3)
            for lab,center,radius in zip(['A','B'],geometries['centers_mm'],geometries['radii_mm']):
                ink='white' if col==2 else 'black'
                ax.add_patch(Circle(center,radius,fill=False,color=ink,lw=.9));ax.text(center[0],center[1]+radius+.7,lab,ha='center',fontsize=10,color=ink)
            ax.set(xlim=(0,20),ylim=(0,20),aspect='equal',xlabel='x (mm)',title=title)
            if col==0:ax.set_ylabel('y (mm)')
            else:ax.set_yticklabels([])
            fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.06,pad=.10,label='ΔVth (mV)' if col<2 else '共享 OU 加载系数')
        origins.append(dict(source=str(path),sha256=sha(path),rate_source=str(ratepath),rate_sha256=sha(ratepath),n_lower=int((delta<0).sum()),n_raised=int((delta>0).sum())))
    fig.suptitle('实际执行数组：同一端点几何与网络；橙色 ICL，青色 SCL\n仅展示 E 细胞；当前 I 不加载这项共享 OU，旧慢 I 输入状态另行控制')
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    for suffix in ['png','pdf']:fig.savefig(folder/f'applied_core_threshold_and_input.{suffix}',dpi=170)
    plt.close(fig)
    (folder/'applied_core_threshold_and_input.json').write_text(json.dumps(origins,indent=2))
    (folder/'README.md').write_text('### applied_core_threshold_and_input.png\n\n来自六臂检查中历史与当前核心方案的实际执行数组，保持同一端点几何、SEEG 及拓扑。前两图色值为 E 阈值相对18 mV的偏移，后两图为共享OU加载，单位与色条分开；橙色ICL、青色SCL，黑圈是core。\n\n**关注点**：核内正阈值尾部归零、共享OU限核；图只证明参数应用，不证明传播恢复。PDF为同图矢量/栅格混合导出，JSON记录来源。\n\n### applied_core_threshold_and_input.pdf\n\n与同名PNG来自同一实际数组、同次绘图，便于报告缩放。没有根据仿真活动重画或选点。\n\n**关注点**：参数定义与活动起源证据分开。\n')
    return folder/'applied_core_threshold_and_input.pdf'


class Report:
    def __init__(self, directory, stamp, version=7):
        self.out=directory; self.stamp=stamp; self.version=version; self.log=[]; self.text=[]; self.sources=[]
        self.c=canvas.Canvas(str(directory/'layout.pdf'),pagesize=A4,pageCompression=1)
        self.page=0; self.y=0; self.W,self.H=A4
    def para(self,text,size=10.8,color=None,space=9):
        p=Paragraph(markup(text),ParagraphStyle('p',fontName='Latin',fontSize=size,leading=size*1.52,textColor=color or colors.HexColor('#23333c'),wordWrap='CJK'))
        _,h=p.wrap(self.W-84,2000);self.y-=h;p.drawOn(self.c,42,self.y);self.y-=space
        self.text.append(dict(page=self.page,text=text))
    def start(self,title,wide=False):
        if self.page:
            assert self.y>35,(self.page,title,self.y)
            self.log[-1]['content_bottom_pt']=self.y;self.c.showPage()
        self.page+=1;self.W,self.H=landscape(A4) if wide else A4;self.c.setPageSize((self.W,self.H))
        self.y=self.H-24;self.para(f'患者几何先验下的空间 SNN｜协作者报告 v{self.version}',8,GRAY,0)
        self.c.setStrokeColor(colors.HexColor('#b8cbd6'));self.c.line(42,self.H-43,self.W-42,self.H-43)
        self.c.setFont('Latin',7);self.c.setFillColor(GRAY)
        self.c.drawString(42,20,'HFOsp | '+self.stamp+f' | v{self.version}');self.c.drawRightString(self.W-42,20,str(self.page))
        self.y=self.H-56;self.para(title,17,BLUE,12)
        self.log.append(dict(page=self.page,title=title,figures=[]))
    def table(self,data,widths=None,size=9.7):
        style=ParagraphStyle('cell',fontName='Latin',fontSize=size,leading=size*1.43,wordWrap='CJK')
        t=Table([[Paragraph(markup(v),style) for v in row] for row in data],colWidths=widths or [(self.W-84)/len(data[0])]*len(data[0]))
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf2f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.3,colors.HexColor('#c5d5df')),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        _,h=t.wrap(self.W-84,2000);self.y-=h;t.drawOn(self.c,42,self.y);self.y-=12
        self.text.append(dict(page=self.page,table=data))
    def source(self,path,key):
        path=Path(path);target=self.out/'assets'/(key+path.suffix)
        target.parent.mkdir(exist_ok=True);shutil.copy2(path,target)
        self.sources.append(dict(id=key,source=str(path),snapshot=str(target.relative_to(self.out)),sha256=sha(target)))
        return target
    def figure(self,path,key,maxheight):
        path=Path(path)
        if path.suffix=='.pdf' and path.with_suffix('.png').exists():
            # Matplotlib's CJK font subsets can render incorrect glyphs after
            # PDF merging. Keep the source PDF and embed its same-run PNG.
            self.source(path,key+'_original_vector')
            self.sources[-1]['embedding_note']='Original PDF retained; matched source PNG embedded to avoid CJK font-subset rendering corruption.'
            path=path.with_suffix('.png')
        target=self.source(path,key)
        if target.suffix=='.pdf':
            pg=PdfReader(str(target)).pages[0];w,h=float(pg.mediabox.width),float(pg.mediabox.height)
        else:
            with Image.open(target) as im:im.load();w,h=im.size
        scale=min((self.W-84)/w,maxheight/h);w*=scale;h*=scale;x=(self.W-w)/2;self.y-=h
        if target.suffix!='.pdf':self.c.drawImage(ImageReader(str(target)),x,self.y,width=w,height=h)
        self.log[-1]['figures'].append(dict(id=key,path=str(target),bbox=[x,self.y,w,h],vector=target.suffix=='.pdf'))
        self.y-=10
    def finish(self):
        assert self.y>35,(self.page,self.y)
        self.log[-1]['content_bottom_pt']=self.y;self.c.save()
        writer=PdfWriter()
        for record,pg in zip(self.log,PdfReader(str(self.out/'layout.pdf')).pages):
            for f in record['figures']:
                if not f['vector']:continue
                src=PdfReader(f['path']).pages[0];x,y,w,h=f['bbox'];scale=w/float(src.mediabox.width)
                pg.merge_transformed_page(src,Transformation().scale(scale).translate(x,y))
            writer.add_page(pg)
        # Retain the actual old C figure, caption and equations, with every page
        # explicitly identified as an experiment/methods archive, not v7 defaults.
        old=self.source(V6/'model_collaborator_report_v6.pdf','v6_source')
        oldpages=PdfReader(str(old)).pages
        for index in list(range(1,6))+list(range(15,23)):
            pg=oldpages[index];w,h=float(pg.mediabox.width),float(pg.mediabox.height)
            stream=io.BytesIO();c=canvas.Canvas(stream,pagesize=(w,h));c.setFillColor(colors.white);c.rect(0,h-44,w,44,fill=1,stroke=0);c.rect(0,0,w,29,fill=1,stroke=0)
            text=f'v{self.version} 附录｜9月8日历史方法/状态实验；参数与后续进展以本版正文为准'
            p=Paragraph(markup(text),ParagraphStyle('banner',fontName='Latin',fontSize=8,leading=11,textColor=BLUE));p.wrap(w-84,40);p.drawOn(c,42,h-35)
            c.setFont('Latin',8);c.setFillColor(GRAY);c.drawString(42,18,f'v6 source page {index+1} | historical evidence retained');c.drawRightString(w-42,18,str(len(writer.pages)+1));c.save()
            pg.merge_page(PdfReader(stream).pages[0]);writer.add_page(pg)
            self.log.append(dict(page=len(writer.pages),title=f'历史 v6 第 {index+1} 页',v6_page=index+1))
        writer.add_metadata({'/Title':f'患者几何先验下的空间 SNN：core 驱动、参数作用与传播证据 v{self.version}','/Author':'HFOsp research project'})
        path=self.out/f'model_collaborator_report_v{self.version}.pdf'
        with path.open('wb') as f:writer.write(f)
        check=PdfReader(str(path));alltext='\n'.join(p.extract_text() or '' for p in check.pages)
        assert '454.87' in alltext and '13.94' in alltext,'C result/caption lost'
        for key,value in [('layout_manifest',self.log),('report_text',self.text),('source_manifest',self.sources)]:
            (self.out/(key+'.json')).write_text(json.dumps(value,ensure_ascii=False,indent=2))
        return path,len(check.pages)


def position_pages(r):
    if not (POSITION/'plan.json').exists():return None
    plan=read(POSITION/'plan.json');state=read(POSITION/'status.json')
    completed=sum((POSITION/'formal/units'/c['id']/str(s)/'workers/trajectory.json').exists() for c in plan['candidates'] if not c['reference_source'] for s in plan['seeds'])
    done=state['status']=='COMPLETE_PENDING_SCIENTIFIC_REVIEW' and (POSITION/'analysis/position_response_summary.json').exists()
    r.source(POSITION/'plan.json','position_plan');r.source(POSITION/'status.json','position_status')
    r.start('回到位置主线：端点初值怎样改变双杆招募？',True)
    r.figure(POSITION/'analysis/figures/position_design.png','position_design',330)
    r.para(f'本轮14条新20秒位置干预，复用4条原位轨迹；本版新运行完成 {completed}/14，状态 {state["status"]}。所有位置在结果前固定，半径、图/连接及输入规律固定；只移动E阈值支持与其共享OU加载。',10.2)
    r.para('主系列关闭外加慢I状态，以已有关闭对照为原位；原位/上移3mm另保留I状态开启的桥。加权中心略向下移，不等于靠近SCL；上移4.5mm使左核边缘距SCL9从5.43降至1.13mm，却距ICL11从2.22增至6.36mm。',10.2,BLUE)
    if done:
        s=read(POSITION/'analysis/position_response_summary.json');r.source(POSITION/'analysis/position_response_summary.json','position_observations')
        if (POSITION/'analysis/agent_scientific_review.md').exists():
            r.source(POSITION/'analysis/agent_scientific_review.md','position_agent_review')
            r.start('位置结果：招募增加了，双模式仍未恢复',True)
            r.figure(POSITION/'analysis/key_result/figures/position_key_response.png','position_key_result',260)
            r.para('18条对照轨迹（14新＋4复用）全部完成，共463个合格事件。上移4.5mm使两杆联合参与从0升至60.9%/62.5%，但SCL9仍为0；TA仅2/23、0/16。斜向靠近上部SCL后，SCL9为8.8%/12.5%，两条噪声均未观察到TA标签。',11)
            r.para('这建立了位置→观测的具体对应，但不能接受完整恢复。ICL平均参与没有整体下降，部分提高来自TA/TB组成改变；TB条件内又存在过度招募。下一版需区分左核范围和向外招募能力，不能继续只沿y扫点或把两个标签当作患者传播。',10.5,BLUE)
        r.start('位置 → 观测：增加SCL时，有没有失去ICL？',True)
        r.figure(POSITION/'analysis/figures/vertical_position_response.png','position_effects',335)
        r.para('横轴只包含固定X的垂直位移；两条线分别为两条配对噪声。先看不分TA/TB的上排，再看条件分布；SCL9/8平均参与、ICL平均参与和两杆联合参与分别读。rank相关不能掩盖未参与触点。',10.2)
        r.para('每个点由本运行全部合格事件计算，n不是独立网络数。灰/黑参考为完整患者FIT的观测，未用两个展示事件估计概率。加权/斜向位置和慢I开启桥单列，不接入同一纯Y响应曲线。',10.2,BLUE)
        r.start('位置 → 触点对的顺序概率',True)
        r.figure(POSITION/'analysis/figures/vertical_position_pair_order.png','position_pair_order',285)
        r.para('对每对触点，计算两者均参与时i的质心早于j的概率，相同时记半次；与患者同一概率比较。曲线为有共同支持的触点对等权平均绝对偏差，数字为实际可读对数，最多105。它是离线分布诊断，不是新增训练损失。',10.4)
        r.para('可读触点对随位置改变，故分数高低不能脱离支持范围比较；逐对实际事件数已保存。缺失触点对没有填0，不能以删掉难拟合触点降低此误差来接受传播。',10.4,BLUE)
        r.start('位置响应的量化表：同网两条噪声分别保留')
        data=[['位置','seed末位','N','SCL9/8参与','ICL参与','两杆联合']]
        for x in s['all_event_rows']:
            def f(k):return '缺值' if x[k] is None else f'{x[k]:.3f}'
            data.append([x['label'],str(x['seed'])[-2:],x['n'],f('SCL_upper_participation'),f('ICL_contact_participation'),f('both_rods')])
        r.table(data,widths=[196,52,35,76,76,76],size=8.4)
        r.para('SCL9/8参与是两触点概率的平均，ICL参与是11触点概率的平均；两杆联合表示至少各有一个触点。它们回答不同问题，不能把SCL一处变亮当作整个模式已恢复。',10.3,BLUE)
        r.start('位置 → 两杆时间关系：不能只有参与增加',True)
        r.figure(POSITION/'analysis/figures/vertical_position_lag.png','position_lag',300)
        r.para('仅两杆均参与事件可估计此时差：每杆参与触点的质心时间先取中位数，再求SCL−ICL。负值为SCL较早；点为逐运行中位数、竖线为事件5–95%范围，数字是实际支持数。没有两杆事件的条件不填0。',10.3)
        r.para('这是质心相对时间，不是两核或两处组织的起燃时差；参与集合改变也会改变杆级摘要，因此必须结合下一页逐触点与完整包络。',10.3,BLUE)
        r.start('全部位置：保留每个SCL与ICL触点',True)
        r.figure(POSITION/'analysis/figures/all_positions_contact_participation.png','position_contact_probabilities',340)
        r.para('所有固定位置与两条噪声按计划顺序呈现，不按拟合好坏删选。颜色是该触点参与概率；三列为不分模式、TA和TB，灰色为不可估计，患者FIT参考在各列首行。',10.3)
        r.para('先检查是补足两杆、仅换了一根杆，还是增加了患者不常参与的触点；这些结果需要与下面预先指定的上移3mm/靠近上部SCL时序比较。',10.3,BLUE)
        for cid,title in [('A_up_3p0_off','左核上移3mm'),('A_near_upper_SCL_off','左核边缘靠近上部SCL')]:
            r.start(title+'：患者与两条噪声的完整时序',True)
            r.figure(POSITION/f'analysis/all_condition_time_review/figures/{cid}_event_scale.png','position_black_'+cid,340)
            r.para('这两个位置在结果前指定为报告展示对象；每条运行、每个模式取最接近自身特征均值的事件，不按患者相似度选例。无事件的格子保留为空，全部其他位置黑底图和多事件GIF也已交付。',10.2)
            r.para('局部HFO包络与模型发放密度仍有物理差距；具体SCL/ICL顺序、并行活动和模式条件分布共同决定传播判断，不能由单幅图接受机制。',10.2,BLUE)
        if (POSITION/'analysis/continuous_native_activity.csv').exists():
            r.source(POSITION/'analysis/continuous_native_activity.csv','position_continuous_native')
            r.source(POSITION/'analysis/core_mass_timing_summary.csv','position_core_mass_timing')
            r.start('TA少了，是否因为左核不再活动？',True)
            r.figure(POSITION/'analysis/continuous_native/figures/continuous_native_rate_response.png','position_native_rate',280)
            r.para('不是整体静默：完整1.5–20秒左核平均率从原位15.57/16.07Hz，上移3mm后为15.46/14.23Hz，上移4.5mm后为15.23/15.71Hz。全部发放按实际神经元数及秒数归一化，不依赖事件资格或TA/TB分类。',10.4)
            r.para('原生多事件帧显示，一些活动团从右下沿ICL向左上运动，却没有充分招募SCL上部；也存在左核先亮而仍归为TB的事件。平均率、核先后活动与患者传播标签必须分开。该结果不能唯一归因于EE/EI或某一连接机制。',10.4,BLUE)
            r.start('保留两核完整活动，避免只看筛选后的事件',True)
            r.figure(POSITION/'analysis/continuous_native/figures/continuous_core_activity.png','position_full_core_activity',350)
            r.para('红线左核A、蓝线右核B；20ms固定分箱，灰区为启动期。四个固定位置、两条噪声逐一保留。曲线说明两核仍有活动，不证明其全部burst都符合患者两种传播，也不证明一核因果驱动另一核。',10.4)
    else:
        r.para('位置结果尚未齐全；本版先给出可核验的实际位置及比较对象。完成后自动补入位置×观测、两杆时差、逐触点概率及固定位置的黑底图，不以输入核查或文档更新代替响应结果。',10.2,BLUE)
    return dict(status=state['status'],new_complete=completed,new_total=14,reused=4,results_included=done)


def connectivity_status_sentence():
    if not (SEARCH/'status.json').exists():
        return '当前仅设计和确定性候选表已完成，正式运行0条；物理实现、应用核验和新评分仍待执行。详见 docs/archive/topic4/core_connectivity_search_design_2026-09-10.md 及同名结果包 execution_prompt.md。'
    st=read(SEARCH/'status.json');canary=read(SEARCH/'canary_audit.json')['status'] if (SEARCH/'canary_audit.json').exists() else '未完成'
    done=sum((SEARCH/'screen/units'/c['id']/f"2511_{s}"/'workers/trajectory.json').exists() for c in read(SEARCH/'plan.json')['candidates'] for s in read(SEARCH/'plan.json')['seeds'])
    return f'本版更新：新版物理 core_connectivity_v2 已实现（只降阈值双核、核外确定期望输入、六块局部权重、真实入度增删、EE 核重采样），6 条 500 ms 应用核验 {canary}；第一阶段 120 条 20 秒运行已完成 {done}/120，当前状态 {st["status"]}。结果页见下文；候选表沿用 search_design.json，派发前冻结。'


def connectivity_pages(r):
    if not (SEARCH/'plan.json').exists():return None
    plan=read(SEARCH/'plan.json');st=read(SEARCH/'status.json');A=SEARCH/'analysis'
    r.source(SEARCH/'plan.json','connectivity_plan');r.source(SEARCH/'status.json','connectivity_status')
    if (SEARCH/'canary_audit.json').exists():r.source(SEARCH/'canary_audit.json','connectivity_canary')
    layouts=[l['id'] for l in plan['layouts']];zh={'endpoint':'端点几何原位','up4p5':'原位上移 4.5 mm','near_upper':'靠近上部 SCL'}
    r.start('新版物理 core_connectivity_v2：执行器实际应用了什么？')
    r.table([['部分','实际定义（以 applied_physics.json 为准）'],['E 阈值','Vth = 18 − a_k·max(18 − Vraw_i, 0)，核内只降不升；潜在 Vraw_i 按神经元身份固定（quantile_seed 20260806）；11 mV 下界并记录截断；核重叠取较强降低。'],['随机输入','共享 OU＋独立 Poisson 只作用两核 E；核外 E 与全部 I 每步接收确定的期望到达 nu·dt，经同一 AMPA 滤波；空间 OU、慢 I 状态、Z/M、kick 关闭。'],['权重块','EE 同核 / EE 核→外 / EE 外→外 / E→I 同核 / I→E 同核 / I→I 同核，按真实前后突触类型与 A/B/O 成员缩放；边与时延不变；不改 Params.w_EE。'],['真实入度','只改 EE 核→外块：每个外部 E 靶入边数=基线×因子确定取整，固定拓扑随机键增删，新边按距离重算时延；零基线靶记录。'],['核范围/方向','重采样全部 E→E 入边（每靶保持 800），核尺度 (l_par·s, l_perp·s, θ+Δ)，时延按距离；E→I/I→E/I→I 保留；只共享拓扑随机身份，不是同一邻接。'],['组合顺序','核基线 → 核→外密度 → 权重块，每步从不可变缓存图计算。']],widths=[84,427],size=9.6)
    if (SEARCH/'canary_audit.json').exists():
        ca=read(SEARCH/'canary_audit.json');data=[['500 ms 应用检查','A/B 成员','降阈值总量 mV','截断','全部核验','峰值内存 GiB']]
        for u in ca['units']:data.append([u['candidate'],f"{u['members'][0]}/{u['members'][1]}",f"{u['total_lowering_mV']:.1f}",str(u['floor_clipped']),'通过' if all(u['checks'].values()) else '未通过',f"{u['peak_rss_gib']:.1f}"])
        r.table(data,widths=[200,70,90,45,60,60],size=8.8)
        r.para(f'应用核验状态 {ca["status"]}：核对阈值、输入局部性（核外 E 速率偏差恰为 0）、六块权重、真实加边/重采样与 500 ms 完成；不以传播好坏作为闸门。',10.2,BLUE)
    if not (A/'response_summary.json').exists():
        r.para(f'第一阶段运行状态：{st["status"]}。响应图、黑底与原生 GIF 在 120 条完成后由已冻结分析器生成并加入本报告；本版不预写结论。',11);return dict(status=st['status'],results_included=False)
    summ=read(A/'response_summary.json');counts=rows(A/'per_run_counts.csv');obs=rows(A/'run_mode_observations.csv');r.source(A/'response_summary.json','connectivity_summary')
    for name in ['per_run_counts.csv','run_mode_observations.csv','paired_parameter_effects.csv','scores.csv','applied_parameters.csv','mask_score_calibration.json']:
        if (A/name).exists():r.source(A/name,'connectivity_'+name.split('.')[0])
    def val(x):
        try:return float(x)
        except (TypeError,ValueError):return None
    v2=[c for c in counts if c['source']=='v2' and c['physical_status']!='MISSING']
    r.start('输入基底桥：核外改为确定输入后，同布局基线变了什么？',True)
    r.figure(A/'figures/input_bridge.png','connectivity_input_bridge',330)
    lines=[]
    for layout in layouts:
        for seed in plan['seeds']:
            old=next((o for o in obs if o['candidate']==f'{layout}__legacy_input' and o['seed']==str(seed) and o['mode']=='ALL'),None);new=next((o for o in obs if o['candidate']==f'{layout}__baseline' and o['seed']==str(seed) and o['mode']=='ALL'),None)
            if old and new:lines.append(f"{zh[layout]} 噪声{seed}: N {old['n']}→{new['n']}，SCL9/8 {val(old['SCL_upper_participation']) or 0:.2f}→{val(new['SCL_upper_participation']) or 0:.2f}，两杆 {val(old['both_rods']) or 0:.2f}→{val(new['both_rods']) or 0:.2f}，左核率 {val(old['coreAE_rate_hz']) or 0:.1f}→{val(new['coreAE_rate_hz']) or 0:.1f} Hz")
    r.para('；'.join(lines) if lines else '桥数据缺失。',9.8)
    r.para('这是输入基底本身的效应（圆点旧版全场 Poisson，方点新版核外确定输入），后续所有连接效应都相对新版基线解释，不能把它归因于任何连接参数。',10.2,BLUE)
    r.start('第一阶段：120 条配对运行的实际支持')
    runaways=summ.get('runaways',[]);missing=summ.get('missing',[])
    scorable_conditions=sum(all(int(x['analysis_n'])>=16 for x in v2 if x['candidate']==cid) for cid in {x['candidate'] for x in v2})
    r.para(f"完成 {summ['runs']}/120 条（缺失 {len(missing)}，物理 runaway {len(runaways)}），单条运行 ≥16 合格事件且可评分 {summ['scorable_runs']} 条；两条噪声均达到事件数要求的条件 {scorable_conditions} 个。联合参与掩膜项的正尺度 a_mask={summ['a_mask']:.4f}（患者 FIT 块匹配 16 事件的有偏距离中位数）。",11)
    data=[['布局','条件数','合格事件合计','TA 合计','TB 合计','N≥16 的运行']]
    for layout in layouts:
        c=[x for x in v2 if x['layout']==layout]
        data.append([zh[layout],str(len({x['candidate'] for x in c})),str(sum(int(x['analysis_n']) for x in c)),str(sum(int(x['n_TA']) for x in c)),str(sum(int(x['n_TB']) for x in c)),str(sum(int(x['analysis_n'])>=16 for x in c))])
    r.table(data,widths=[120,60,90,70,70,100],size=9.5)
    r.para('事件是运行内样本，不是独立网络；少数模式缺失先报告支持量。逐运行完整表与逐触点参与、成对顺序、时差 CSV 在分析目录。',10.2,BLUE)
    fam_zh={'EE_same_core_scale':'同核 E→E 权重','EE_core_to_out_scale':'核→外 E→E 权重','EE_out_to_out_scale':'外→外 E→E 权重','EI_same_core_scale':'同核 E→I 权重','IE_same_core_scale':'同核 I→E 权重','II_same_core_scale':'同核 I→I 权重','depth_A_scale':'A 核降阈值幅度','depth_B_scale':'B 核降阈值幅度','radius_A_mm':'A 核半径','radius_B_mm':'B 核半径','EE_core_to_out_degree_scale':'核→外 E→E 实际入度','EE_kernel_perp_scale':'E→E 横向核尺度','EE_kernel_parallel_scale':'E→E 纵向核尺度','EE_angle_offset_deg':'E→E 轴方向'}
    for fam,label in fam_zh.items():
        p=A/f'figures/response_{fam}.png'
        if not p.exists():continue
        r.start(f'参数 → 观测：{label}')
        r.figure(p,'connectivity_response_'+fam,640)
        r.para('三列为三个布局；黑/红/蓝为不分模式/TA/TB，实线/虚线为两条配对噪声，点线为患者 FIT 参考，横轴写真实参数值，n 为实际事件数。缺值表示该模式无事件，不填零。',9.4,BLUE)
    for layout in layouts:
        p=A/f'figures/contact_participation_{layout}.png'
        if p.exists():
            r.start(f'{zh[layout]}：全部条件逐触点参与',True);r.figure(p,'connectivity_contacts_'+layout,330)
            r.para('患者 FIT 参考在首行；灰格为无可估计事件。先看 SCL9/8 是否补足、ICL 是否过度招募，再看 TA/TB 各自列。',9.8,BLUE)
    for layout in layouts:
        p=A/f'figures/scores_and_core_activity_{layout}.png'
        if p.exists():
            r.start(f'{zh[layout]}：分数与两核完整活动',True);r.figure(p,'connectivity_scores_'+layout,330)
            r.para('冻结 L_off、联合参与掩膜项和 L_search 只用于排序辅助；N<16 的运行留空。底行为不依赖事件资格的两核/核外完整放电率。',9.8,BLUE)
    for layout in layouts:
        p=A/f'time_review_{layout}/all_condition_time_review/figures/{layout}__baseline_event_scale.png'
        if p.exists():
            r.start(f'{zh[layout]} 新版基线：患者与两条噪声的完整时序',True);r.figure(p,'connectivity_black_'+layout,340)
            r.para('患者 Fig2C TA/TB 固定在左；模型例最接近自身模式均值，不按患者相似度选例；无事件格保留为空。全部 60 条件黑底页在分析目录 time_review_<布局>/all_condition_time_review/all_conditions.pdf。',9.8,BLUE)
    for stage,name in [('adaptive','联合阶段（有边界 DE）'),('confirmation','确认阶段（新拓扑/新噪声）')]:
        nom=SEARCH/stage/'nomination.json'
        if nom.exists():
            n=read(nom);r.source(nom,'connectivity_'+stage+'_nomination');r.start(name+'：状态与提名')
            r.para(json.dumps({k:v for k,v in n.items() if k in ('status','reason','conditions','reasons','rule','note')},ensure_ascii=False)[:1800],9.5)
            for extra in [SEARCH/'analysis_confirmation/figures/confirmation_paired_effects.png']:
                if extra.exists():r.figure(extra,'connectivity_confirmation_effects',330)
    review=A/'agent_scientific_review.md'
    if review.exists():
        r.source(review,'connectivity_agent_review');r.start('第一阶段科学审阅（Agent 自查，待用户目视）')
        for para in [x.strip() for x in review.read_text().split('\n\n') if x.strip() and not x.strip().startswith('#') and not x.strip().startswith('|') and not x.strip().startswith('-')][:9]:
            r.para(para,9.6)
            if r.y<120:break
    return dict(status=st['status'],results_included=True,runs=summ['runs'],scorable_runs=summ['scorable_runs'])


def build():
    now=datetime.datetime.now().astimezone();stamp=now.strftime('%Y-%m-%d %H:%M %z')
    directory=OUT/'builds'/now.strftime('%Y%m%dT%H%M%S%f');directory.mkdir(parents=True)
    r=Report(directory,stamp);status=read(PILOT/'status.json');plan=read(PILOT/'plan.json')
    complete=sum((PILOT/'formal/units'/c['id']/str(s)/'workers/trajectory.json').exists() for c in plan['candidates'] for s in plan['seeds'])
    done=status['status']=='COMPLETE_PENDING_SCIENTIFIC_REVIEW' and (PILOT/'analysis/delivery_checks.json').exists()
    r.source(PILOT/'status.json','pilot_status');r.source(PILOT/'plan.json','pilot_plan')
    r.source(Path(__file__),'report_producer');r.source(PILOT/'canary_audit.json','pilot_canary')
    if (PILOT/'scheduling_amendment.json').exists():r.source(PILOT/'scheduling_amendment.json','pilot_dispatch_amendment')
    for key,p in [('shared_contract',ROOT/'docs/topic4_patient_geometry_prior_snn.md'),('historical_review',HISTORY/'scientific_review.md'),('geometry_review',GEOM/'analysis/scientific_review.md'),('six_noise_review',EXTENT/'analysis_90000_6noise/six_noise_scientific_review.md')]:r.source(p,key)

    r.start('当前问题：怎样的局部 E/I 设置，产生怎样的传播？')
    r.para('以患者间期传播轴和端点区域的几何中心提供 core 初值；让随机输入主要访问局部易激 core，再由网络连接与局部状态决定跨区域招募。目标是同时恢复患者 TA/TB 的多事件传播分布，并建立参数变化与观测变化的对应关系。',12)
    r.para('当前决定：core 驱动是操作性模型假设。先快速核对输入修订与明显副作用，不再为“把输入限核后，核会更主导”另开长期验证。是否形成患者相容的传播，仍必须看完整时间过程。',11.3,BLUE)
    position_claim='左核上移、加权中心和靠近上部SCL的位置干预；连接保持固定，实际状态见位置章节。'
    if (POSITION/'analysis/agent_scientific_review.md').exists():position_claim='位置对照18/18已完成。上移4.5mm增加两杆参与至约61–63%，但SCL9仍缺失，TA支持下降；左核仍活跃。位置响应已建立，完整双模式尚未恢复。'
    r.table([['证据状态','本版判断'],['历史参数响应','范围、阈值分布与 I 输入作用域确实改变参与和顺序；部分响应在六条噪声下保留，但存在 TA/TB 取舍。'],['患者双模式恢复','尚未接受。SCL 招募、TB 的两杆相对时序和原生分离活动仍有缺口；不能用低损失或两类标签代替。'],['当前输入修订',f'六臂 500 ms 实施检查通过。正式 20 秒运行 {complete}/12 完成；状态：{status["status"]}。'+('核心方案36例、关慢I对照73例均缺SCL，已接位置响应。' if done else '尚不能给出新版传播改善的结论。')],['当前参数作用',position_claim]],widths=[100,411])
    r.para('本报告替换 v6 的当前结论与患者拟合章节；通用方程、C＋OU 图及图注、放电率分岔证据作为明确标注的历史附录保留。旧报告不覆盖、不倒改。',10,GRAY)

    r.start('模型定义：几何、易激性、连接与输入分别是什么？')
    r.table([['部分','当前输入 pilot 的实际定义'],['几何先验','端点中心 A=(4.1992,9.1289)、B=(16.4792,3.9655) mm；两核半径约 1.753 mm；同一拓扑 2511。几何先验不是独立验证结果。'],['E 易激性','正阈值偏移截为零，原负偏移保留。1,075 个 E 降阈值；424 个原升阈值 E 回到背景。允许降幅异质性，不要求每个 core 细胞都严格降低。'],['随机访问','共享 OU 只加载几何 core 的 E；核外 E 与 I 不接收这项 OU 波动。空间局部 OU 默认关闭；可选时只保留核内原随机场值。'],['背景与传播','固定均值 Poisson 背景仍在全网络；核外活动由背景和递归网络产生。不会删掉核外 spike，也不按预期源点筛选事件。'],['本轮固定项','原静态图与所有 EE/EI/IE/II 连接保持不变，学习型 EE/EI 系数为零；GABA 衰减 18 ms；自主 Z/M 关闭。'],['外加慢状态','核心方案保留原慢 I 输入状态以保持配对；另有关闭它的对照。它不是内源双稳态，也不把 TA/TB 标签写入刺激。']],widths=[84,427],size=10)
    r.para('这是明确的新模型版本，不是只恢复 XY，也不是完整复刻最早手放物理。几何 core、实际降阈值细胞、I 状态作用域须分别记录，不能用同一个圆框混称。',10.4,BLUE)

    r.start('当前几何与输入：执行器实际应用了什么？',True)
    r.figure(applied_input_figure(),'applied_core_input',305)
    r.para('从已完成的 500 ms 检查读取实际 E 阈值和共享 OU 加载向量；固定同一拓扑与 SEEG。这里没有重新拟合位置：两核仍在患者端点几何初值，输入空间加载改变。',10.4)
    r.para('阈值偏移与 OU 加载使用不同色条和单位。蓝色负阈值表示更易激，OU 色值表示输入加载系数；A/B 是核的几何编号，不把核编号直接等同于 TA/TB。这不是活动热图或传播恢复结果。',10.4,BLUE)

    r.start('先看什么：患者与模型的可比观测')
    r.table([['问题','观察量与单位','需要怎样比较'],['是否招募到该通道？','逐触点参与概率；整杆 SCL 缺失率','全事件及 TA/TB 条件分布；保留具体触点，不把整杆出现等于完整招募。'],['先后顺序像不像？','参与触点的质心 rank；成对反序率','未参与触点不填伪 rank；均值模板相关只是一项摘要。'],['招募间隔是否接近？','相对质心/10%/50%/90% 质量时间，ms；跨接触跨度','局部宽度和跨区间隔分开，保留分布、散布及相关结构。'],['每处活动持续多久？','各参与通道 10–90% 包络质量宽度，ms','患者 HFO 包络与模型放电密度不是相同物理信号；不通过拉伸时间或强调 GABA 假装等价。'],['传播过程是否连续？','完整通道×时间包络；原生 2 ms 空间活动','保留核内外活动和 SEEG 几何；分离亮区不直接等于独立因果源。'],['两种模式是否都有支持？','实际 TA/TB 事件数、比例、条件分布','标签组织比较，不证明两条路径；少数类缺失先报告有限观测。']],widths=[112,180,219],size=9.5)
    r.para('统计单位：事件是同一运行内的样本；先逐运行估计，再按噪声/拓扑配对。多噪声不是多患者，匹配 seed 也不保证改变速率后的 Poisson 创新逐时刻一致。',10.4)
    r.para('均值/中位数与 5–95% 范围描述模型和患者各自分布；区间重叠不自动等于恢复率。若要总体汇总，必须先声明跨观测权重与“恢复”的定义。',10.4,BLUE)

    r.start('哪些是训练目标，哪些是传播检验？')
    r.para('v2.1 曾用多事件特征的分布差异选择候选，包含全局项和患者模式相关项；模型神经元输入不读 TA/TB 标签，不等于整个评价过程从未使用患者聚类。当前输入 pilot 是预先固定的六臂配对干预，没有自适应优化，也没有新增路线损失。',11.1)
    r.para('去自配对统计量 D_off = A − B，其中 A 为特征均值到患者目标的平方距离，B 为模型事件散布除以 N−1。其有限样本值可负；保留原 D16 作稳定性诊断，不截断负值。它不识别传播因果链，也不能自动修复连续事件相关性。',11)
    r.table([['此前捷径','现在如何读证据'],['旧 selected-lineage 读出可忽略同窗其他活动','历史分数保留原身份；当前连续 R1 输入线读全神经元放电，原生图与接触图同时核对。'],['匹配质心 rank，局部宽度/早期包络仍错误','黑底完整时序前置；以患者固定 Fig2C 两例定位形态，再看全部事件条件分布。'],['两标签都有，但来自不相容传播或启动过程','报告逐运行支持、连续轨迹和模式条件结构；不以两个标签作为机制验收。'],['把合格事件增多看成原生事件增多','同时报告原始检测、完整窗口资格和合格计数；窗口重叠排除可以改变方向。']],widths=[196,315],size=10)
    r.para('局部 E/I 干预先回答“实际改变了什么”，不能在尚未检查完整传播时仅凭联合分数决定下一工作点。沿用的开发图不再称为盲验证。',11,BLUE)

    r.start('历史复核：同样叫 core，实际阈值并不相同',True)
    r.figure(HISTORY/'figures/threshold_maps.pdf','historical_threshold',280)
    r.para('原不重叠手放双核用两个阈值数组逐点 minimum 合并，另一核的背景会截掉高阈值抽样；后来 data-driven 的带符号映射保留了升阈值尾部。回滚圆形几何没有恢复原物理定义。',10.5)
    r.para('图为 rev20 的实际参数对照；它与当前端点几何不同。历史“均匀降低”保留净阈值降低总量并消除微观异质性；本轮截正保负增加净降低量。两种干预不能合并解释。',10.5,BLUE)

    r.start('历史全降对照：TA 改善，并没有同时修复 TB',True)
    r.figure(HISTORY/'figures/paired_mode_observations.pdf','historical_effects',310)
    r.para('12 个网络、每条 20 秒，灰线为相同 legacy seed 配对，紫线为网络等权平均。TA 参与误差 0.322→0.296、顺序相关 0.585→0.665；TB 参与误差 0.293→0.369、顺序相关 0.699→0.603。两类 SCL 缺失均下降，但完整恢复仍不成立。',10.2)
    r.para('这是底部旧双核、全场 OU、历史学习型连接重分配和 lineage 观测下的结果；不能据此判定当前端点先验＋core 输入版也必然失败。',10.2,BLUE)

    r.start('黑底对照：旧全降阈值也仍有完整时序缺口',True)
    r.figure(HISTORY/'figures/patient_model_seed_2521.png','historical_black',345)
    r.para('按网络编号展示第一个确认网络，并在每类合格完整活动族中取最接近自身模式均值的例子；没有按患者相似度挑选。显示完整发放包络，而星号沿用历史 lineage 参与判定，两层并不等价。',10)
    r.para('检查点是 SCL/ICL 的实际先后与断裂，不能把 ICL 上一条斜带或平均 rank 相关直接接受为患者 TA/TB。完整 12 网络对照另存于历史复核包。',10,BLUE)

    r.start('范围实验：哪些参数响应在多条噪声中保留？',True)
    r.figure(EXTENT/'analysis_90000_6noise/figures/six_noise_effect_intervals.pdf','six_noise_effects',305)
    r.para('一张拓扑、7 条件×6 条噪声×90 秒，42 条完整运行、1,530 个合格事件。左核扩大至 4 mm：TA 参与误差 6/6 改善，TB 顺序相关 5/6 下降、1/6 持平。右核 4 mm：两类整杆 SCL 缺失 6/6 下降，TB 运行等权约 32.3%→14.8%。',10.2)
    r.para('范围控制了招募与两类取舍，但不等于两模式恢复；区间是固定患者 FIT 下噪声及 15 秒块的开发 bootstrap，没有跨拓扑推断。该批仍是混合阈值＋全场空间 OU；未完成的 180 秒扩展不在本次继续。',10.2,BLUE)

    r.start('中心与阈值：总调制量和空间范围需要拆开',True)
    r.figure(GEOM/'analysis/figures/paired_parameter_changes.pdf','geometry_effects',315)
    r.para('新增 14 条件×两条噪声×90 秒均完成，1,245 个合格事件。左核 4 mm 下保持原正/负阈值偏移总量，TB 参与误差 0.211/0.201→0.150/0.116，平均顺序相关 0.571/0.304→0.768/0.718；TA 顺序却下降。',10.2)
    r.para('这说明扩大范围和增加调制总量有不同作用；总量匹配在此是辨别解释的对照，不是所有后续干预的限制。当前 core 定义已改为非正偏移，不能直接复用这组混合阈值操作作为新默认。',10.2,BLUE)

    r.start('最关键的残差：完整双杆传播仍不够像患者',True)
    r.figure(GEOM/'analysis/figures/patient_vs_two_models_event_scale.png','current_historical_black',345)
    r.para('固定患者 Fig2C 两例在左，两个历史诊断候选在右；保持相同通道顺序和真实 250 ms 时间轴。TB 仍可表现为右端活动与后续 ICL 招募分段，另一例则缺 SCL。较好的 rank 摘要没有解决这一差距。',10)
    r.para('患者为 HFO 包络，模型为全部神经元发放密度；这里不是二者频谱等价的证明。约 16–20 ms 的模型局部宽度与患者约 85 ms 的差距需保留，当前不强迫用 GABA 抹平。',10,BLUE)

    r.start('短程输入检查：实际执行内容与完成状态')
    r.table([['对照','阈值','共享 OU / 空间 OU','慢 I 状态'],['历史原样','有升有降','全 E/I / 全 E','开'],['仅修阈值','截正保负','全 E/I / 全 E','开'],['空间 OU 限核','截正保负','全 E/I / core E','开'],['两类 OU 限核','截正保负','core E / core E','开'],['核心方案','截正保负','core E / 关闭','开'],['再关慢 I 状态','截正保负','core E / 关闭','关']],widths=[116,99,210,86],size=10)
    r.para('所有条件固定几何、连接和其余参数；同一拓扑配对两条噪声，每条 20 秒，共 12 次正式运行。前置 6 条 500 ms 短段已经完成，实际加载核验通过；旧引擎对照前缀完全一致。',11)
    r.para('执行器实测：核心方案的核外 E 输入率没有 OU 偏移，原降阈值保留、升阈值归零。它证明参数确实应用，不证明真实活动一定来自 core，也不证明患者双模式已经恢复。',11,BLUE)
    r.para(f'本次报告生成时：{complete}/12 正式运行完成，状态 {status["status"]}。报告在本轮终态后自动刷新一次，保留每次生成快照；不会据分数自动开新搜索。',11)

    if done:
        counts=rows(PILOT/'analysis/per_run_counts.csv');obs=rows(PILOT/'analysis/per_mode_observations.csv')
        r.source(PILOT/'analysis/per_run_counts.csv','pilot_counts');r.source(PILOT/'analysis/per_mode_observations.csv','pilot_observations')
        r.start('新版输入：实际事件支持与逐模式响应',True)
        r.figure(PILOT/'analysis/figures/parameter_observations.pdf','pilot_effects',315)
        r.para('每条线是一条配对噪声，所有条件保留；断线表示该观测不可估计，不能填零。下面两页固定展示事前指定核心方案，不按完成后的分数选“最好看”的输入。',10.3)
        r.para('这是单拓扑、两次 20 秒重演的开发响应；标签数量、核内早期占比或更安静的核外场，都不能单独接受为 TA/TB 恢复。完整 12 次时序页与逐事件原生诊断在本轮分析目录。',10.3,BLUE)
        r.start('新版输入：时间尺度随条件怎样变化？',True)
        r.figure(PILOT/'analysis/figures/parameter_timing_distributions.pdf','pilot_timing',315)
        r.para('点和连线是各运行事件中位数，竖线是该运行事件的 5–95% 范围，不是置信区间。局部宽度、跨触点早期招募跨度和质心跨度分开；无可估计事件不填零。',10.3)
        r.para('这里只检验模型对输入变化的响应，不把患者 HFO 与模型发放密度视为同一信号，也不通过匹配局部宽度替代两杆空间顺序。单参与触点的跨度定义为零，应结合事件的参与触点数理解。',10.3,BLUE)
        r.start('新版输入：原生空间活动随条件怎样变化？',True)
        r.figure(PILOT/'analysis/figures/parameter_native_distributions.pdf','pilot_native_effects',315)
        r.para('早期定义为该事件窗口累计活动质量的前 10%。核内占比与每神经元密度比分开，避免 core 大小和核外神经元数混入同一个数。峰值分离分量为 1 mm 格、2 ms 活动、每格至少 2 个神经元的八邻接统计。',10.3)
        r.para('这些图帮助定位原生活动，但不证明因果起源或独立 roots，也不是新训练惩罚。核占比增加是输入改变后的预期方向；决定性问题仍是患者相容的两模式传播。',10.3,BLUE)
        r.start('新版输入：每条轨迹的实际观测量')
        data=[['条件','噪声末位','TA / TB 数','原始检测 / 合格','物理状态']]
        for x in counts:
            m=[next(y['n'] for y in obs if y['candidate']==x['candidate'] and y['seed']==x['seed'] and y['mode']==mode) for mode in ['TA','TB']]
            data.append([x['label'],x['seed'][-2:],'/'.join(m),x['detected']+' / '+x['analysis_n'],x['physical_status']])
        r.table(data,widths=[182,58,71,105,95],size=8.8)
        r.para('合格 = 排除启动期并具有完整形态窗口后的事件。窗口资格改变会改变数量；不能只看合格计数判断网络总发放多少。少数模式未被观察到时，先说明实际 N，不以固定标签门判定机制缺失。',10.3)
        r.start('核心方案：患者—模型完整时序',True)
        r.figure(PILOT/'analysis/all_condition_time_review/figures/core_ou_only_event_scale.png','pilot_default_black',345)
        r.para('左为固定患者 Fig2C TA/TB，中、右为两条噪声；模型例最接近自身本运行模式特征均值。没有合格事件的格子保留为空，不用别的条件替换。须同时核查 SCL/ICL、毫秒尺度和全部事件分布。',10.2)
        r.para('这是自动生成的完整读出对照，待 Agent/用户科学目视；不能因为本页出现两类或活动集中就写成已恢复。',10.2,BLUE)
        native=PILOT/'analysis/figures/core_ou_only_native_stills.png'
        r.start('核心方案：核内外全部活动与连续读出',True)
        if native.exists():r.figure(native,'pilot_default_native',340)
        else:r.figure(PILOT/'analysis/figures/core_ou_only_847101_continuous.png','pilot_default_continuous',320)
        r.para('原生图取第一条噪声每类最早三个合格事件，按时间排列；圆为实际 core，空心触点为 SEEG。多事件 GIF 与全部连续读出随本轮分析交付。没有合格事件时展示完整连续读出。',10.2)
        r.para('早期累计质量中 core 占比只是活动定位；按神经元数校正内外密度后仍不能直接证明因果起源。峰值空间分离也不等于独立 causal roots。',10.2,BLUE)
    else:
        r.start('输入结果页：按完成状态更新，不预写阳性结论')
        r.para('当前 12 条正式轨迹尚未全部分析完成。新版患者—模型黑底图、原生多事件图和参数响应将由已冻结分析器生成后加入此报告；当前不能把上面历史传播图当作新版效果。',12)
        r.table([['结果分支','如何解释'],['核外活动减少，传播仍断裂','输入修订落实了操作性假设，但还需查核内循环/向外招募和局部抑制；不继续只扫噪声。'],['活动几乎消失或少数模式没观测到','报告原生完整读出和实际事件量；先辨别 core 访问不足、抑制过强或观测窗资格，不能称为传播恢复。'],['两类完整时序改善','保留新输入基底做小范围局部连接配对响应，再在新噪声/拓扑确认。'],['两类标签仍有但原生场不相容','保持未接受；标签和特征损失不承担机制恢复结论。']],widths=[177,334],size=10.5)
        r.para('刷新只读取完成产物，不改变计划、物理参数、选例规则和训练排名；失败也写入报告状态并停止等待，不自动重启或扩预算。',10.8,BLUE)

    position_info=position_pages(r)
    connectivity_design_included=(CONNECTIVITY_DESIGN/'search_design.json').exists()
    if connectivity_design_included:
        r.source(CONNECTIVITY_DESIGN/'historical_scl_audit.json','historical_scl_radius_audit')
        r.source(CONNECTIVITY_DESIGN/'search_design.json','next_connectivity_design')
        r.source(ROOT/'docs/archive/topic4/core_connectivity_search_design_2026-09-10.md','next_connectivity_contract')
        r.start('历史手放：上部 SCL 确有招募，但大核不是完整答案',True)
        r.figure(CONNECTIVITY_DESIGN/'figures/historical_scl_radius_evidence.png','historical_scl_radius',305)
        r.para('同一个旧 seed 3，约4秒运行：半径1.5/2.5/4/6 mm的SCL9参与分别为0/14、0/16、0/7、4/11。6 mm保存示例中SCL9附近17个E细胞有15个发放；它有局部原生活动支持。左两图是旧producer按参与数/可读性保存的示例，并非随机选例。',10.5)
        r.para('降阈值细胞同时从773增至10507，且6 mm核跨越边界。旧坐标注册、图采样和方向标签与当前不同，不将它们直接当作患者TA/TB配对；本图支持研究招募范围，不证明已恢复两模式。',10.5,BLUE)
        r.start('下一版输入：严格限核随机驱动，尚未实施')
        r.para('已完成的输入/位置版只是把OU限于E-core，核外仍有Poisson随机到达。按照用户“噪声只能驱动核”的要求，拟议新版让核外E和全部I使用确定的期望到达量，经原突触滤波；核内保留共享OU和Poisson。背景均值保留，核外仍可由网络递归放电。',12)
        r.para('这比OU限核更强，是新模型假设；同布局的新旧基线单独对照，不把输入效应归因于连接。两核阈值只允许降低；空间OU、外加慢I、Z/M和kick关闭，GABA固定18 ms。',11.5,BLUE)
        r.para('连接范围需要独立调纵向与横向尺度：当前增大轴比会压窄横向，不能直接代表更宽扇形。权重缩放保持邻接，真实密度/范围干预必须改变实际边并更新时延；两种“同seed”对照的静态含义分别记录。',11.5)
        r.start('下一版系统搜索：先建立参数响应，再做联合选择')
        r.table([['阶段','明确预算与目的'],['固定配对响应','3布局 ×（基线＋19个单参数探针）× 2噪声 = 120条20秒。位置以原位、上移4.5 mm、靠近上部SCL开始。'],['有边界联合搜索','最多16条件×2噪声=32条20秒；两个布局人口，各4后代后统一更新，再各4后代。最多6个活跃参数。'],['新图/新噪声确认','4条件×2新拓扑×2新噪声=16条60秒；保留新版原位基线。正式总上限168条。']],widths=[110,401],size=10.5)
        r.para('探针包括EE内部/向外、核外EE、同核E→I/I→E/I→I、两核各自降阈值强度与半径、实际连接数、纵横范围和方向。直接输入总量可变；每个参数的两类取舍都交付，不只展示联合赢家。',10.5)
        r.para('提议保留原去自配对分数，并增加完整15触点联合参与掩膜分布项：分开事件的一杆活动不能替代同一事件两杆参与。新项尚待实现/校准，第一批固定配对探针不依赖它选点。完整黑底时序和原生多事件传播仍决定科学审阅，不能由分数代替。',10.5,BLUE)
        r.para(connectivity_status_sentence(),10.5)
    connectivity_info=connectivity_pages(r)
    r.start('位置响应之后：放松哪些局部连接参数？')
    r.para('先冻结新版实际工作点的几何、非正阈值场、输入规律和状态；每次只改变一个明确通路，保留两核身份。下表是下一批设计，不是当前已运行结果。',11)
    r.table([['操作对象','想区分的作用','重点观测'],['E_core→E_core','局部递归放大/重复响应，是否把 core 变成过强爆发团','局部 burst 与重复活动、两类支持；不假定越强越好。'],['E_core→E_out','核内活动能否向外招募，而非只在核里发亮','SCL/ICL 参与、跨区时间、原生前沿；与核内放大分开。'],['E_core→I_core','局部兴奋如何招募抑制，限制或组织事件','自限/持续活动、两类参与与时序取舍。'],['I_core→E_core','反馈抑制对本核持续和核间先后关系的影响','局部密度、宽度、起始相对时间、核外招募。']],widths=[119,193,199],size=10.2)
    r.para('core E 用当前几何成员，core I 用相同圆形几何选择 I 节点，不能直接沿用可能不同的旧慢 I 状态 mask。两核先分别干预；核间直接边如存在则另记，不能把所有核内边合并成一个旋钮。',10.3)
    r.para('第一版优先在固定图上缩放所选边的实际权重，直接保留总输入变化。0.8×/1.0×/1.2×可作小范围候选起点，但须按实际基底与资源冻结成批后才执行；不在本次报告更新中自动派发。',10.3,BLUE)

    r.start('连接数量可变：总量不是必须守恒的约束')
    r.para('直接减弱某通路与把相同总量重新分配到另一方向，是不同问题。病理 core 允许具有区别于背景的有效连接；不要求通过补偿把每个靶细胞的总输入拉回原值。',11.5)
    r.table([['干预方式','固定什么 / 实际改变什么'],['缩放权重','固定边、时延和静态细胞状态；所选边权重乘因子，实际输入总量改变。记录每靶输入总量、E/I 分路和空间分布。'],['减少连接数量','另设固定删边随机流，在同一候选图上用嵌套子集删边；保持保留边权重不补偿。避免重建整图把异质性一并改变。'],['方向或范围重分配','写明只改固定边权重，还是重新连边。只有检验方向作用是否独立于剂量时，才另加总量匹配对照。'],['多个自由度联合','先用单通路响应定位可解释方向，再做少量组合；不能把一次联合改善归因于某一个旋钮。']],widths=[113,398],size=10.3)
    r.para('最先交付同网同噪声的配对曲线：TA/TB 参与、顺序、招募跨度、局部宽度及原生空间过程分别画；实际检测/合格数作为支持量。再用新噪声和新拓扑检验趋势是否稳定，不把事件数当网络数。',11)
    r.para('拟合与干预能建立模型内的有效参数—观测映射；不能唯一倒推出患者真实细胞或突触病理。临床定位与状态/发作机制继续独立验收。',10.8,BLUE)

    r.start('四组承重图与报告更新方式')
    r.table([['图','固定读法与审阅重点'],['1. 黑底患者—模型时序','患者 TA/TB 固定在左，模型配对条件/噪声在右；250 ms 真时间，不按 rank 重排。先看 TB 与 SCL 缺口。'],['2. 原生多事件 2D＋连续读出','真实 SEEG 和 core 叠加、核内外全保留；多例按预定规则挑选，保留没有合格事件的连续轨迹。'],['3. 参数×各项观测','同一拓扑和噪声配对；用明确参数名和值，分 TA/TB，标实际 N、均值/中位数与散布。'],['4. 多噪声/拓扑稳健性','先逐运行，再逐拓扑；展示效应方向与区间、不同随机性的取舍，不只汇报合并分数。']],widths=[139,372],size=10.4)
    r.para('报告稳定入口为 docs/snn_model_report_current.md；每次构建保留 PDF、来源快照、图面布局与报告文字。当前输入 pilot 完成或失败后自动刷新一次并停止，不自动安排下一物理批次。',11)
    r.para('接受标准：完整两杆时序与原生传播支持患者两模式，并有足够多事件分布支持；损失、两个标签或核主导单独改善均不足。当前历史结果暂不接受完整恢复，新版结果按其实际轨迹单独审阅。',11,BLUE)

    r.start('状态分支保留，但它不是当前患者输入基底')
    r.para('v6 的 C＋OU 图及解释完整保留于附录：同一网络在原 OU 下 E 平均率 13.12→454.87→13.94 Hz，分别为间隔事件、持续高态及外部恢复后的返回。它不等于自主终止，也不证明当前患者传播恢复。',11)
    r.para('9 月 9 日补充已直接运行原逐神经元 Z：历史手放几何、manual_hard 阈值＋当前 C 快速连接，在 10.68 秒达到持续高活动判据，11.18–12.18 秒外部补回 Z，最后一秒返回约 12.49 Hz。进入与外部返回已观察到；原 SNN 分岔类型和患者模式匹配仍未建立。',11)
    r.figure(ROOT/'results/topic4_sef_hfo/historical_manual_hard_native_z_v1/figures/autonomous_z_manual_restore.png','native_z_update',295)
    r.para('下一附录按原图保留通用 LIF/突触/Z 方程，以及 C、rate 模型的证据。附录中的“下一步”是 9 月 8 日历史状态，应以上述后续结果和各分支文档为准；不把新版 core 输入套到旧状态曲线上。',10,BLUE)

    r.start('来源与版本：每一层证据来自哪里？')
    r.table([['资料包','内容 / 状态'],['lowering_only_core_historical_review_20260910','实际阈值数组、rev20 12 网络配对、黑底对照；历史 lineage 层与全发放层分开。'],['core_extent_long_propagation_20260909','42×90 秒、6 噪声范围响应；当前未自动续 180 秒。'],['geometry_threshold_refinement_20260909','新增 28×90 秒中心/阈值/I 状态范围干预；21 条件完整时序 42 页。'],['core_driven_input_pilot_20260910',f'6×500 ms 检查已过；本版正式 {complete}/12；状态 {status["status"]}。'],['historical_manual_hard_native_z_v1','原生 Z 进入高态、外部恢复及有限延续，独立状态分支。'],['model_collaborator_report_v6_2026-09-08','附录保留原第 2–6、16–23 页：方程、C 图、rate 机制；原始 PDF 不改。']],widths=[228,283],size=9.5)
    r.para('代码：scripts/build_topic4_model_collaborator_report_v7.py。当前科学定义：docs/topic4_patient_geometry_prior_snn.md。每次 PDF 随附 source_manifest.json、layout_manifest.json、report_text.json 和 current.json，结果与计划分开。',10.3)
    r.para('生成校验不是科学接受；新增图须经 Agent 与用户目视。报告更新不新增仿真，不改正式 Fig4/5，不自动提交或推送仓库。',10.5,BLUE)
    pdf,pages=r.finish()
    preview=directory/'figures';preview.mkdir(exist_ok=True)
    check=subprocess.run(['pdftoppm','-scale-to','700','-png',str(pdf),str(preview/'page')],capture_output=True,text=True,check=True)
    images=sorted(preview.glob('page-*.png'));assert len(images)==pages
    for p in images:
        with Image.open(p) as im:im.load();assert min(im.size)>100
    (preview/'README.md').write_text('\n\n'.join(f'### {p.name}\n\n本构建 PDF 第 {k+1} 页的渲染预览，与该目录的报告快照一致；不是新增模型数据。图源、阈值/输入版本及实验状态见报告正文与 source_manifest。\n\n**关注点**：图文可读性与版本身份，不能以解码成功代替科学接受。' for k,p in enumerate(images))+'\n')
    info=dict(generated_at=now.isoformat(),report=str(OUT/pdf.name),snapshot=str(pdf),pages=pages,pilot_status=status['status'],formal_complete=complete,formal_total=12,pilot_results_included=done,position_experiment=position_info,connectivity_search_design_included=connectivity_design_included,connectivity_search=connectivity_info,sha256=sha(pdf),rendered_and_decoded_pages=len(images),user_scientific_acceptance=False)
    (directory/'delivery_checks.json').write_text(json.dumps(dict(**info,layout_minimum_bottom_pt=min(p['content_bottom_pt'] for p in r.log if 'content_bottom_pt' in p),retained_C_text_checked=True,renderer_warnings=check.stderr),ensure_ascii=False,indent=2))
    for dest,source in [(OUT/pdf.name,pdf)]:
        tmp=dest.with_suffix('.tmp.pdf');shutil.copy2(source,tmp);os.replace(tmp,dest)
    tmp=OUT/'current.tmp.json';tmp.write_text(json.dumps(info,ensure_ascii=False,indent=2));os.replace(tmp,OUT/'current.json')
    print(json.dumps(info,ensure_ascii=False));return info


def watch(target=PILOT):
    OUT.mkdir(parents=True,exist_ok=True)
    # flock keeps a restarted report watcher from publishing concurrent builds.
    import fcntl
    with (OUT/('watch_position.lock' if target==POSITION else 'watch.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        start=time.time();last=None
        while True:
            status=read(target/'status.json');name=status['status']
            if name in ('COMPLETE_PENDING_SCIENTIFIC_REVIEW','FAILED'):
                result=build();(OUT/'watch_status.json').write_text(json.dumps(dict(status='REFRESHED_AND_STOPPED',result=result),ensure_ascii=False,indent=2));return
            if time.time()-start>4*3600:
                (OUT/'watch_status.json').write_text(json.dumps(dict(status='TIMEOUT_NO_EXPERIMENT_CHANGE',pilot_status=status)));return
            if name!=last:
                (OUT/'watch_status.json').write_text(json.dumps(dict(status='WAITING_FOR_PILOT_TERMINAL',pilot_status=name),indent=2));last=name
            time.sleep(20)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--watch-pilot',action='store_true');parser.add_argument('--watch-position',action='store_true');args=parser.parse_args()
    if args.watch_pilot or args.watch_position:
        try:watch(POSITION if args.watch_position else PILOT)
        except Exception as exc:
            (OUT/'watch_status.json').write_text(json.dumps(dict(status='REPORT_REFRESH_FAILED',error=repr(exc)),indent=2));raise
    else:build()
