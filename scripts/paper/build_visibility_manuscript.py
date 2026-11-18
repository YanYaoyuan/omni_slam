"""Build the review manuscript from frozen results, without mapping or inference."""
from pathlib import Path
import hashlib
import html
import json
import re
import shutil
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, Preformatted
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'build/benchmarks/visibility_independent_20261112'
REVIEW=ROOT/'docs/reports/visibility_mechanism_review_2026-11-12'
OUT=ROOT/'docs/paper/visibility_revision'
PDF=ROOT/'output/pdf/omni_slam_visibility_revision_draft.pdf'
INPUTS={}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def use(p):
    p=Path(p);INPUTS[str(p.relative_to(ROOT))]=sha(p);return p
def read(p):return json.loads(use(p).read_text())
def arr(p):
    with np.load(use(p)) as z:return {k:z[k] for k in z.files}
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def result_tables():
    m={s:read(BASE/'evaluation'/s/'metrics.json')['methods'] for s in ['052','054','057','061']}
    methods=[('cg','CG (shared input)'),('history2','Strong baseline'),('sor','Baseline + SOR'),('visibility1','**Ours**'),('visibility3','Ours: 3-frame control'),('visibility1_no_depth','Ours: no depth gate'),('maskclustering','MaskClustering (shared)')]
    rows=[]
    for name,label in methods:
        v=[r[name]['score']['observed_voxel_instance_pq']*100 for r in m.values()]
        rows.append([label,*[f'{x:.3f}' for x in v],f'{np.mean(v):.3f}'])
    main=mdtable(['Method','052','054','057','061','Mean'],rows)
    means={n:np.mean([r[n]['score']['observed_voxel_instance_pq']*100 for r in m.values()]) for n,_ in methods}
    contrast=[('Ours - strong baseline','history2'),('Ours - SOR','sor'),('Ours - no depth gate','visibility1_no_depth'),('Ours - 3-frame control','visibility3')]
    ab=[]
    for label,n in contrast:
        d=[(r['visibility1']['score']['observed_voxel_instance_pq']-r[n]['score']['observed_voxel_instance_pq'])*100 for r in m.values()]
        ab.append([label,f'{np.mean(d):+.3f}',f'{sum(x>1e-9 for x in d)}/{sum(x< -1e-9 for x in d)}'])
    for n,label in [('visibility1','Ours locked - baseline'),('sor','SOR locked - baseline')]:
        d=[(r[n]['locked_assignment_score']['observed_voxel_instance_pq']-r['history2']['score']['observed_voxel_instance_pq'])*100 for r in m.values()]
        ab.append([label,f'{np.mean(d):+.3f}',f'{sum(x>1e-9 for x in d)}/{sum(x< -1e-9 for x in d)}'])
    diag=read(REVIEW/'scene057_diagnosis.json')
    rows=[[label,*[diag['pair_stats'][n][k] for k in ['correct','excess','missing']],f"{diag['pair_stats'][n]['iou']:.6f}"] for n,label in [('history2','Strong baseline'),('visibility1','Ours'),('sor','SOR'),('visibility1_locked','Ours (locked)')]]
    objects=read(REVIEW/'per_object.json');objrows=[]
    for scene in m:
        deltas=[o['delta_ours_baseline'] for o in objects if o['scene']==scene]
        objrows.append([scene,len(deltas),sum(x>1e-10 for x in deltas),sum(x< -1e-10 for x in deltas),sum(abs(x)<=1e-10 for x in deltas)])
    return {'MAIN_TABLE':main,'ABLATION_TABLE':mdtable(['Paired contrast','Mean PQ-point difference','Wins/losses'],ab),'CASE_TABLE':mdtable(['Method','Correct','Excess','Missed','IoU'],rows),'OBJECT_TABLE':mdtable(['Scene','GT objects','Improved','Reduced','Unchanged'],objrows)},means

def blocks(md):
    lines=md.splitlines();i=0
    while i<len(lines):
        s=lines[i]
        if not s.strip():i+=1;continue
        if s=='<!-- page -->':yield 'page','';i+=1;continue
        if s.startswith('#'):
            level=len(s)-len(s.lstrip('#'));yield 'h'+str(level),s[level:].strip();i+=1;continue
        match=re.fullmatch(r'!\[(.*)\]\((.*)\)',s)
        if match:yield 'image',match.groups();i+=1;continue
        if s.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].startswith('|'):
                row=[x.strip() for x in lines[i].strip('|').split('|')]
                if not all(re.fullmatch(r'[-:]+',x) for x in row):rows.append(row)
                i+=1
            yield 'table',rows;continue
        if s.startswith('    '):
            code=[]
            while i<len(lines) and lines[i].startswith('    '):code.append(lines[i][4:]);i+=1
            yield 'code','\n'.join(code);continue
        para=[]
        while i<len(lines) and lines[i].strip():para.append(lines[i]);i+=1
        yield 'p',' '.join(para)

def inline(s):return re.sub(r'\*\*(.*?)\*\*',r'<b>\1</b>',html.escape(s))

def render(md):
    parsed=list(blocks(md));parts=[]
    for kind,value in parsed:
        if kind=='page':parts.append('<hr class="page">')
        elif kind.startswith('h'):parts.append(f'<{kind}>{inline(value)}</{kind}>')
        elif kind=='p':parts.append('<p>'+inline(value)+'</p>')
        elif kind=='code':parts.append('<pre>'+html.escape(value)+'</pre>')
        elif kind=='image':parts.append(f'<figure><img src="{value[1]}" alt="{html.escape(value[0])}"><figcaption>{inline(value[0])}</figcaption></figure>')
        elif kind=='table':parts.append('<table>'+''.join('<tr>'+''.join(f'<{"th" if j==0 else "td"}>{inline(x)}</{"th" if j==0 else "td"}>' for x in row)+'</tr>' for j,row in enumerate(value))+'</table>')
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>omni_slam visibility revision - manuscript</title><style>body{max-width:1000px;margin:42px auto;padding:0 22px;color:#203449;font:17px/1.75 Georgia,serif}h1,h2,h3{font-family:system-ui;line-height:1.3}h1{font-size:34px}h2{margin-top:34px}table{width:100%;border-collapse:collapse;font:14px/1.5 system-ui}th,td{padding:9px;border-bottom:1px solid #d6e0e6;text-align:left}th{background:#eaf0f6}figure{margin:24px 0}img{width:100%;height:auto}figcaption{font:13px/1.6 system-ui;color:#52677a}pre{background:#edf4f7;padding:18px;white-space:pre-wrap;font:13px/1.6 monospace}.page{margin:46px 0;border:0;border-top:1px solid #d5e1e8}p{overflow-wrap:anywhere}</style><body><nav style="font:15px system-ui;margin-bottom:24px"><a href="visual_gallery.html">高清实验图册 / Figure gallery</a></nav>'+''.join(parts)+'</body></html>'
    (OUT/'index.html').write_text(page)
    fonts=Path('/usr/share/fonts/truetype/dejavu')
    for name,file in [('Paper','DejaVuSerif.ttf'),('PaperBold','DejaVuSerif-Bold.ttf'),('UI','DejaVuSans.ttf'),('UIBold','DejaVuSans-Bold.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Paper',normal='Paper',bold='PaperBold',italic='Paper',boldItalic='PaperBold')
    pdfmetrics.registerFontFamily('UI',normal='UI',bold='UIBold',italic='UI',boldItalic='UIBold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='BodyPaper',fontName='Paper',fontSize=10.1,leading=14.3,spaceAfter=8,textColor=colors.HexColor('#203449')))
    styles.add(ParagraphStyle(name='CaptionPaper',fontName='UI',fontSize=8.1,leading=11,spaceAfter=10,textColor=colors.HexColor('#52677a')))
    for n,size,leading in [('h1',20,25),('h2',14,19),('h3',11,15)]:styles.add(ParagraphStyle(name='Paper'+n,fontName='UIBold',fontSize=size,leading=leading,spaceBefore=8,spaceAfter=10,keepWithNext=True,textColor=colors.HexColor('#183247')))
    styles.add(ParagraphStyle(name='CellPaper',fontName='UI',fontSize=7.4,leading=10))
    width=A4[0]-84;story=[]
    for kind,value in parsed:
        if kind=='page':story.append(PageBreak())
        elif kind.startswith('h'):story.append(Paragraph(inline(value),styles['Paper'+kind]))
        elif kind=='p':story.append(Paragraph(inline(value),styles['CaptionPaper'] if value.startswith(('Table ','Research manuscript')) else styles['BodyPaper']))
        elif kind=='code':
            wrapped='\n'.join('\n'.join(textwrap.wrap(x,96,subsequent_indent='    ',replace_whitespace=False)) if x else '' for x in value.splitlines())
            story.extend([Preformatted(wrapped,ParagraphStyle('Code',fontName='Courier',fontSize=7.5,leading=11,spaceBefore=6,spaceAfter=10)),Spacer(1,4)])
        elif kind=='image':
            path=OUT/value[1];im=PILImage.open(path);h=width*im.height/im.width
            story.extend([Image(str(path),width=width,height=h),Paragraph(inline(value[0]),styles['CaptionPaper'])])
        elif kind=='table':
            cells=[[Paragraph(inline(x),styles['CellPaper']) for x in row] for row in value]
            count=len(cells[0]);first=165 if count==3 else 150
            widths=[first]+[(width-first)/(count-1)]*(count-1)
            t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
            commands=[('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f6')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,-1),.4,colors.HexColor('#d6e0e6'))]
            for j,row in enumerate(value):
                if row[0]=='**Ours**':commands.append(('BACKGROUND',(0,j),(-1,j),colors.HexColor('#e4f4ef')))
            t.setStyle(TableStyle(commands));story.extend([t,Spacer(1,10)])
    def footer(c,doc):
        c.setStrokeColor(colors.HexColor('#d6e0e6'));c.line(42,36,A4[0]-42,36);c.setFont('UI',7)
        c.setFillColor(colors.HexColor('#647b91'));c.drawString(42,24,'omni_slam | Research review draft | 13 November 2026');c.drawRightString(A4[0]-42,24,str(doc.page))
    PDF.parent.mkdir(parents=True,exist_ok=True)
    SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=38,bottomMargin=48,
                      title='omni_slam: Visibility-Guided Geometry Revision for Object-Level Mapping',author='omni_slam project').build(story,onFirstPage=footer,onLaterPages=footer)

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    from build_visibility_supplement import build as build_supplement
    build_supplement()
    frozen=read(REVIEW/'freeze_manifest.json')
    for path,want in frozen['source_and_evaluation_inputs'].items():assert sha(use(ROOT/path))==want,path
    for name,want in frozen['artifacts'].items():assert sha(use(REVIEW/name))==want,name
    for name in ['method_architecture','mechanism_decomposition','057_threshold_case','representative_objects','all_object_changes']:
        for ext in ['png','svg']:shutil.copyfile(use(REVIEW/(name+'.'+ext)),OUT/'figures'/(name+'.'+ext))
    from build_visibility_visuals import build as build_visuals
    use(ROOT/'scripts/paper/build_visibility_visuals.py')
    build_visuals(use);tables,means=result_tables()
    md=use(OUT/'manuscript.md').read_text()+use(OUT/'supplement.md').read_text()
    for name in ['supplement_manifest.json','mechanism_audit.json','environment_observed.json']:
        use(OUT/name)
    use(ROOT/'scripts/paper/build_visibility_supplement.py')
    use(OUT/'references.bib');use(OUT/'editorial_note_zh.md');use(OUT/'REPRODUCE.md')
    use(ROOT/'scripts/paper/verify_visibility_package.py');use(ROOT/'scripts/paper/reproduce_visibility_paper.sh')
    for token,value in tables.items():md=md.replace('{{'+token+'}}',value)
    assert '{{' not in md
    (OUT/'manuscript_resolved.md').write_text(md);render(md)
    for path,want in INPUTS.items():assert sha(ROOT/path)==want,path
    outputs=[OUT/'index.html',OUT/'visual_gallery.html',OUT/'manuscript_resolved.md',PDF,OUT/'supplement.md',OUT/'supplement_manifest.json',OUT/'mechanism_audit.json',OUT/'environment_observed.json',*sorted((OUT/'figures').glob('*'))]
    dump(OUT/'build_manifest.json',dict(complete=True,inputs_unchanged=True,no_new_experiments=True,
        primary='visibility1',means_pq100=means,input_sha256=INPUTS,
        output_sha256={str(p.relative_to(ROOT)):sha(p) for p in outputs},script_sha256=sha(__file__),
        pdf_render_review_pending=True))
    print(json.dumps(dict(pdf=str(PDF),html=str(OUT/'index.html'),inputs=len(INPUTS),outputs=len(outputs)),indent=2))

if __name__=='__main__':main()
