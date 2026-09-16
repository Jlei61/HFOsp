"""Readability/completeness checks and descriptions after actual rendering."""
from common import *
from PIL import Image
import re,subprocess,xml.etree.ElementTree as ET

def main():
    fig=OUT/'figures';rows=[]
    for p in sorted(fig.glob('*.png')):
        with Image.open(p) as im:im.load();assert min(im.size)>700;size=list(im.size)
        pdf=p.with_suffix('.pdf');assert pdf.exists()
        xml=subprocess.check_output(['pdftotext','-bbox',str(pdf),'-'],stderr=subprocess.DEVNULL)
        root=ET.fromstring(xml);words=[];bad=[]
        for page in root.iter():
            if not page.tag.endswith('page'):continue
            width=float(page.attrib['width']);height=float(page.attrib['height'])
            for word in page.iter():
                if not word.tag.endswith('word'):continue
                text=word.text or '';x=float(word.attrib['xMin']);y=float(word.attrib['yMin']);xx=float(word.attrib['xMax']);yy=float(word.attrib['yMax']);words.append(text)
                if x<-.5 or y<-.5 or xx>width+.5 or yy>height+.5:bad.append(text)
        assert not bad,(p.name,bad)
        assert words,p
        rows.append(dict(name=p.name,pixels=size,pdf_words=len(words),outside_page_words=bad))
    books={}
    for name,wanted in [('native_network_AB_atlas.pdf',40),('reduced_numbered_waveforms.pdf',24),('joint_critical_waveforms_modes.pdf',12)]:
        output=subprocess.check_output(['pdfinfo',str(fig/name)],text=True);pages=int(re.search(r'^Pages:\s+(\d+)',output,re.M).group(1));assert pages==wanted,(name,pages);books[name]=pages
    desc=read(OUT/'figure_descriptions.json');parts=['# v7候选图说明\n\n确定性边值轨道与原生SNN放电记录使用同参数编号对应；二者不是同一数据层。图已经过文件、页边界与来源检查，仍待用户目视检查。\n']
    parts.append((fig/'NATIVE_README.md').read_text())
    for name,caption in sorted(desc.items()):
        assert (fig/f'{name}.png').exists()
        parts.append(f'### {name}.png / .pdf\n{caption}\n**关注点**：检查模型层、参数编号、A/B读出与稳定性证据的对应。\n')
    parts.extend(['### native_network_AB_atlas.pdf\n20条件原生SNN图库，每条件总览和放大各一页，共40页。记录来源与固定时间窗和单图相同。\n**关注点**：跨条件看序列差异，逐核与全网络看同一时段的活动。\n','### reduced_numbered_waveforms.pdf\n24页确定性波形，a/b表示同J的不同吸引子。平衡解、周期解均保留模型层标记。\n**关注点**：与同编号原生记录作参数对应，不能把波形相似当作分岔机制已验证。\n','### joint_critical_waveforms_modes.pdf\n12个周期临界轨道的全网络/A/B读出及左右临界模。模态平方范数没有人口加权，也不是因果贡献。\n**关注点**：同一临界点在两核和周边的不同表现。\n'])
    (fig/'README.md').write_text('\n'.join(parts))
    names={p.stem for p in fig.glob('*.png')};documented=set(desc)|{r['name'] for r in read(OUT/'native_figure_manifest.json')};assert names==documented,(names-documented,documented-names)
    write('figure_validation.json',dict(status='PASS',png_count=len(rows),figures=rows,books=books,descriptions_complete=True,human_visual_acceptance='PENDING'))
    print('FIGURE_QA',len(rows),books,flush=True)

if __name__=='__main__':main()
