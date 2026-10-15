#!/usr/bin/env python3
"""Render the written scientific review and selected existing diagnostic figures to PDF."""
from pathlib import Path
import textwrap,json,subprocess,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/topic4_sef_hfo/nightly_central_workpoint'
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
def main():
    report=OUT/'scientific_review.md'
    lines=[]
    for line in report.read_text().splitlines():
        lines.extend(textwrap.wrap(line,width=68,replace_whitespace=False,break_long_words=True) or [''])
    images=[OUT/'figures/paired_parameter_responses.png',OUT/'figures/paired_observable_ranges.png',OUT/'batches/A/figures/geometry_proposals.png',OUT/'batches/A/figures/central_A_anchor1_geom1_per_contact.png',OUT/'batches/B/figures/central_B_GABA_long_per_contact.png',OUT/'batches/B/figures/broad_B_geometry_per_contact.png']
    images+=sorted((OUT/'figures').glob('*_contact_comparison.png'))
    images+=sorted((OUT/'figures').glob('*review_frames.png'))
    with PdfPages(OUT/'scientific_review.pdf') as pdf:
        for start in range(0,len(lines),34):
            fig=plt.figure(figsize=(11.7,8.3));fig.text(.05,.95,'\n'.join(lines[start:start+34]),va='top',fontsize=10,fontproperties=font,linespacing=1.6);pdf.savefig(fig);plt.close(fig)
        for p in images:
            if not p.exists():raise FileNotFoundError(p)
            with Image.open(p) as im:
                fig,ax=plt.subplots(figsize=(16,10));ax.imshow(im);ax.axis('off');fig.suptitle(p.name,fontsize=10);fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    results=[]
    for p in [OUT/'scientific_review.pdf']+images:
        if p.suffix=='.pdf':
            result=subprocess.run(['pdfinfo',str(p)],capture_output=True,text=True,check=True).stdout
        else:
            with Image.open(p) as im:im.load();result=str(im.size)
        results.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),validation=result))
    (OUT/'report_validation.json').write_text(json.dumps(dict(files=results,human_visual_acceptance='PENDING'),indent=2))
if __name__=='__main__':main()
