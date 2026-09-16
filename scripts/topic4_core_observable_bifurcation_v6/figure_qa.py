"""Check PNG decoding, PDF page count and text lying inside the page."""
from common import *
from PIL import Image
import subprocess,xml.etree.ElementTree as ET

def main():
    rows=[]
    for a in read(OUT/'figure_manifest.json'):
        name=a['name'];png=OUT/'figures'/f'{name}.png';pdf=png.with_suffix('.pdf')
        with Image.open(png) as im:im.verify()
        info=subprocess.check_output(['pdfinfo',str(pdf)],text=True);pages=int(next(x.split(':')[1] for x in info.splitlines() if x.startswith('Pages:')));assert pages==1
        xml=subprocess.check_output(['pdftotext','-bbox',str(pdf),'-'],text=True)
        root=ET.fromstring(xml);page=next(x for x in root.iter() if x.tag.endswith('page'));w=float(page.attrib['width']);h=float(page.attrib['height']);outside=[]
        for x in page.iter():
            if not x.tag.endswith('word'):continue
            b=x.attrib
            if float(b['xMin'])<-.5 or float(b['yMin'])<-.5 or float(b['xMax'])>w+.5 or float(b['yMax'])>h+.5:outside.append(x.text)
        assert not outside,(name,outside)
        rows.append(dict(name=name,png_readable=True,pdf_pages=pages,text_outside_page=outside,insets=a['insets']))
    write('figure_qa.json',dict(status='PASS',figures=rows,human_acceptance='pending'))
    print('FIGURE_QA_PASS',len(rows),flush=True)

if __name__=='__main__':main()
