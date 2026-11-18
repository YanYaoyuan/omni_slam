"""Render the frozen paper plan; no inference, data downloads or experiment runs."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'docs/paper/frozen_v1'
NAVY = '#182c42'
TEAL = '#007f77'
GRAY = '#62758a'
AMBER = '#a35d12'


def architecture():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(16, 6.8), dpi=180)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0,1600), ylim=(680,0)); ax.axis('off')
    def text(x,y,s,size=15,color=NAVY,weight='normal',**kw):
        ax.text(x,y,s,fontsize=size,color=color,weight=weight,va='top',**kw)
    def box(x,y,w,h,fc='#f3f6f9',ec='#cdd6df',dash=False):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=12',
            linewidth=1.2,edgecolor=ec,facecolor=fc,linestyle='--' if dash else '-'))
    def arrow(a,b,color=GRAY,dash=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=13,lw=1.5,
                                   color=color,linestyle='--' if dash else '-'))
    text(40,22,'omni_slam',28,weight='bold')
    text(310,30,'Evidence-preserving object mapping',20)
    text(40,70,'Frozen paper v1  /  association first, category readout second',13,GRAY)
    text(1120,32,'Borrowed / integration',12,GRAY)
    text(1120,57,'Research mechanisms to evaluate',12,TEAL)
    ax.plot([1088,1108],[41,41],color=GRAY,lw=5)
    ax.plot([1088,1108],[66,66],color=TEAL,lw=5)
    # Each column describes the actual implemented boundary.
    cols=[(40,240),(320,250),(610,270),(920,290),(1250,310)]
    for idx,(x,w) in enumerate(cols):
        box(x,140,w,280,fc='#edf7f5' if idx in (3,4) else '#f3f6f9',
            ec='#79b7ad' if idx in (3,4) else '#cdd6df')
    text(58,158,'01  POSED INPUT',12,GRAY,weight='bold')
    text(58,194,'Posed RGB + 3D',15,weight='bold')
    text(58,232,'Public benchmark',13,weight='bold')
    text(58,257,'RGB-D + provided poses',12)
    text(58,295,'Robot support',13,weight='bold')
    text(58,320,'RGB / LiDAR / IMU',12)
    text(58,345,'FAST-LIVO2 → keyframes',12)
    text(58,385,'Two fixed sensor profiles',11,GRAY)
    text(338,158,'02  FROZEN FRONTEND',12,GRAY,weight='bold')
    text(338,195,'SAM2.1 Hiera L',17,weight='bold')
    text(338,230,'instance proposals / masks',12)
    text(338,275,'SigLIP384',17,weight='bold')
    text(338,309,'learned visual descriptor',12)
    text(338,336,'1152-D + frozen text bank',12)
    text(338,385,'Cached once for all methods',11,GRAY)
    text(628,158,'03  STRONG REFERENCE',12,GRAY,weight='bold')
    text(628,195,'ConceptGraphs core',17,weight='bold')
    text(628,236,'geometry + feature association',12)
    text(628,266,'native fusion + source guard',12)
    text(628,310,'Keep ambiguous matches',13,weight='bold')
    text(628,340,'and pre-final fragments',12)
    text(628,385,'No GT in inference',11,GRAY)
    text(938,158,'04  HISTORY ASSOCIATION',12,TEAL,weight='bold')
    text(938,195,'Repeated witnesses',17,weight='bold')
    # Small graph shows two positive witnesses and one conflict rejection.
    for x,y,label in [(963,266,'u'),(1100,266,'v'),(1100,340,'w')]:
        ax.add_patch(Circle((x,y),15,fc='white',ec=TEAL,lw=1.5))
        ax.text(x,y,label,ha='center',va='center',fontsize=13,color=NAVY)
    ax.plot([979,1084],[266,266],c=TEAL,lw=2)
    text(992,241,'k₁, k₂',12,TEAL)
    ax.plot([976,1084],[277,331],c=AMBER,lw=1.5,ls='--')
    text(982,309,'conflict',11,AMBER)
    text(938,370,'≥2 frames + component veto',12,TEAL)
    text(938,396,'OFFLINE graph finalization',11,AMBER,weight='bold')
    text(1268,158,'05  CATEGORY EVIDENCE',12,TEAL,weight='bold')
    text(1268,195,'Consensus readout',17,weight='bold')
    text(1268,237,'one largest mask / frame',12)
    text(1268,267,'median cosine centrality',12)
    text(1268,297,'retain upper half, keep ties',12)
    text(1268,327,'√area weights → text scores',12)
    text(1268,375,'Geometry + retention stay fixed',11,TEAL,weight='bold')
    for i in range(3):arrow((cols[i][0]+cols[i][1]+3,215),(cols[i+1][0]-5,215))
    # Separate native final processing is not implied to obey graph constraints.
    box(920,444,640,53,fc='#fff5e8',ec='#dbb88c')
    text(937,456,'Graph union → native denoise / filter / guarded final merge',12,AMBER)
    text(937,477,'Final native merges do not inherit the graph veto guarantee.',10,AMBER)
    arrow((1065,420),(1065,441),AMBER)
    arrow((1400,444),(1400,422),AMBER)
    # Evidence bank, versioned output, implementation boundary.
    box(40,455,840,103,fc='#f1f4f8')
    text(58,470,'IMMUTABLE OBSERVATION EVIDENCE',12,GRAY,weight='bold')
    text(58,500,'observation ID  ·  keyframe  ·  mask area  ·  measured support  ·  feature  ·  pose',12)
    text(58,530,'Object membership and derived outputs are versioned; identifiers may change after reassociation.',10,GRAY)
    arrow((444,421),(444,452))
    arrow((880,511),(918,511),TEAL,True)
    box(920,515,640,70,fc='#eef2f8')
    text(938,528,'OBJECT MAP',12,weight='bold')
    text(938,553,'members / 3D support / category / location / evidence / revision',12)
    ax.plot([1540,1580,1580],[420,420,550],c=TEAL,lw=1.5)
    arrow((1580,550),(1562,550),TEAL)
    text(40,596,'VALIDATED',11,TEAL,weight='bold')
    text(40,619,'RGB-D offline evaluation; LIVO cached replay; ROS category consumer.',12)
    text(870,606,'Live SAM2 / SigLIP producer + robot end-to-end timing:',11,AMBER)
    text(870,630,'pending system integration; no real-time claim in this paper freeze.',11,AMBER)
    fig.savefig(DEST/'architecture.svg',facecolor='white')
    fig.savefig(DEST/'architecture.png',facecolor='white',dpi=180)
    plt.close(fig)


def one_page():
    pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    pdfmetrics.registerFont(TTFont('DejaVu','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    out=ROOT/'output/pdf/omni_slam_paper_plan_v1.pdf';out.parent.mkdir(parents=True,exist_ok=True)
    w,h=595.276,841.89
    c=canvas.Canvas(str(out),pagesize=(w,h))
    c.setTitle('omni_slam | Frozen paper plan v1')
    c.setAuthor('omni_slam project')
    c.setFillColor(HexColor(NAVY));c.rect(0,h-87,w,87,fill=1,stroke=0)
    def line(s,x,y,size=10,color=NAVY):
        c.setFillColor(HexColor(color));c.setFont('STSong-Light',size);c.drawString(x,y,s)
    line('omni_slam  /  论文方案冻结 v1',30,h-36,22,'#ffffff')
    line('对象级语义建图：重复观测关联 + 可追溯类别证据',30,h-61,12,'#d9e8ef')
    style=ParagraphStyle('body',fontName='STSong-Light',fontSize=10,leading=14,textColor=HexColor(NAVY))
    small=ParagraphStyle('small',parent=style,fontSize=8.5,leading=11)
    def para(s,y,st=style,width=535):
        p=Paragraph(s,st);_,ph=p.wrap(width,1000);p.drawOn(c,30,y-ph);return y-ph
    y=h-103
    y=para('<b>目标与问题</b>　以 ICRA（CCF-B）为目标，研究重复观察中被拆开的对象如何恢复关联，以及类别证据如何在固定对象集合上融合。LIVO 提供位姿；不把定位、分割模型或三维框显示列为新算法。',y)
    y-=8
    c.drawImage(str(DEST/'architecture.png'),25,y-225,width=545,height=225,mask='auto');y-=237
    y=para('<b>冻结方法</b>　SAM2.1 Hiera L + SigLIP384 学习前端 → CG 原生融合与来源保护 → 两帧历史图 → 原生最终处理 → 一帧一最大掩码 → 一致性 / 平方根面积类别读出。主版本：omni_slam-paper-v1。',y)
    y-=8
    y=para('<b>关联规则</b>　<font name="DejaVu" size="9">A(u,v) = 1[|Wuv| ≥ 2] × 1[F(Cu) ∩ F(Cv) = ∅]</font>。W 为不同查询帧；F 为整个分量的帧集合。按票数降序与节点 ID 排序。历史图属于离线整理，最终原生合并不保证同帧互斥。',y)
    y-=7
    y=para('<b>类别规则</b>　<font name="DejaVu" size="9">qi = median(j≠i) cos(fi,fj); bi = 1[qi ≥ median(q)]; wi ∝ bi√ai</font>；类别取文本与归一化加权特征的最大内积。并列全保留，少于三视角回退平方根面积；固定原均值前景集合，不以新类别删除对象。',y)
    y-=12
    rows=[['论文内容','已有证据（PQ × 100）','定位与缺口'],
          ['历史关联','实例：42.4435→42.8842 / Replica 8\n新序列：11.6630→12.6519 / SceneNN 2','主要研究机制\n待固定测试与去约束消融'],
          ['类别证据融合','新序列：6.7434→7.7412\n面积加权得到相同分数','次要机制\n尚无超过面积基线的证据'],
          ['LIVO / ROS2 / 证据版本','330 帧回放；331 次更新\n5 类故障注入通过','工程支撑\n不等于机器人全链路实时']]
    data=[[Paragraph(x.replace('\n','<br/>'),small) for x in row] for row in rows]
    tab=Table(data,colWidths=[94,244,197]);tab.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),HexColor('#e5eeef')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,-1),.4,HexColor('#d5dde4')),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    _,th=tab.wrap(535,1000);tab.drawOn(c,30,y-th);y-=th+12
    y=para('<b>只补一轮</b>　SceneNN 036/038/041/045/047/049；八行对照、五套几何，最多 30 次建图。面积基线、无筛选平方根面积、一帧支持及去同帧否决必须保留。测试后不调参、不换场景；失败则收缩主张并写作。',y,small)
    y-=6
    y=para('<b>边界与时间</b>　现有结果均为观测域诊断，非官方榜单；新序列 025/032 现已查看。ICRA 2027 投稿已关闭，按 ICRA 2028 常规投稿准备，截止日待官方确认。机器人演示留作后续系统验证。',y,small)
    assert y>43, f'One-page layout overflow: {y}'
    line('2026-11-11  |  公式、证据链接及协议：docs/paper/frozen_v1/  |  1 / 1',30,23,8,GRAY)
    c.showPage();c.save()


if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    architecture();one_page()
    print('Rendered architecture SVG/PNG and one-page PDF.')
