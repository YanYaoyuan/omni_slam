"""Vector paper architecture using bundled measured evidence; no inference or mapping."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Ellipse

P=argparse.ArgumentParser();P.add_argument('--paper-dir',type=Path);args=P.parse_args()
HERE=Path(__file__).resolve()
ROOT=args.paper_dir or (HERE.parent.parent if HERE.parent.name=='tools' else HERE.parents[2]/'docs/paper/visibility_revision/latex')
DATA=ROOT/'data/architecture'; OUT=ROOT/'figures/current'
INK='#182f42'; MUTED='#5e7484';TEAL='#008e7b';BLUE='#3e75bc';PURPLE='#8456ae';CORAL='#dc775e';PALE='#eef7f4'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none','mathtext.fontset':'dejavusans'})
fig=plt.figure(figsize=(12,4.9),facecolor='white');a=fig.add_axes([0,0,1,1]);a.set(xlim=(0,1),ylim=(0,1));a.axis('off')
def txt(x,y,s,size=11,color=INK,weight='normal',**kw):return a.text(x,y,s,fontsize=size,color=color,weight=weight,va=kw.pop('va','center'),**kw)
def rect(x,y,w,h,fc='white',ec='#d8e2e7',lw=.8,rounded=False):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.002,rounding_size=0.012',facecolor=fc,edgecolor=ec,linewidth=lw) if rounded else Rectangle((x,y),w,h,facecolor=fc,edgecolor=ec,linewidth=lw)
 a.add_patch(p);return p
def arrow(x1,y1,x2,y2,color=MUTED,lw=1.5,style='-'):
 a.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,linewidth=lw,color=color,linestyle=style))
def dot(x,y,c=TEAL,r=.0045,ec='white'):
 a.add_patch(Ellipse((x,y),r*2,r*2*12/4.9,fc=c,ec=ec,lw=.6))
def line(x1,y1,x2,y2,c='#b8cbd3',lw=1.2,style='-'):a.plot([x1,x2],[y1,y2],color=c,lw=lw,ls=style,solid_capstyle='round')
# Three visual columns, with typography and whitespace carrying the hierarchy.
for x,w,title,c in [(.012,.217,'1  Observed evidence',INK),(.265,.480,'2  Evidence authorization',TEAL),(.792,.196,'3  Revised map',INK)]:
 txt(x,.955,title,12.7,c,'bold');line(x,.915,x+w,.915,c,1.8)
txt(.738,.955,'OURS',10.3,TEAL,'bold',ha='right',bbox={'boxstyle':'round,pad=.3','facecolor':PALE,'edgecolor':'none'})
# Input evidence: real RGB and exact prior-map crop.
ra=fig.add_axes([.012,.738,.217,.155]);ra.imshow(Image.open(DATA/'rgb_903.png'));ra.axis('off')
txt(.012,.707,'Posed RGB + measured LiDAR',10.2)
txt(.012,.657,'SAM3   /   SigLIP   /   CG',10.4,MUTED)
line(.012,.621,.229,.621,'#d8e2e7',.8)
txt(.012,.588,'Prior object map',11,INK,'bold')
z=np.load(DATA/'measured_crop.npz');bounds=z['bounds'];span=bounds[1]-bounds[0]
def cloud(rectangle,xyz,labels):
 ax=fig.add_axes(rectangle,facecolor='#f5f8fa');ax.set_aspect('equal');ax.set_xlim(bounds[0,0],bounds[1,0]);ax.set_ylim(bounds[0,1],bounds[1,1]);ax.set_xticks([]);ax.set_yticks([])
 for sp in ax.spines.values():sp.set_visible(False)
 take=np.arange(0,len(xyz),4);c=np.full((len(xyz),4),matplotlib.colors.to_rgba(BLUE));c[labels==1]=matplotlib.colors.to_rgba(PURPLE);c[labels==2]=matplotlib.colors.to_rgba(TEAL)
 ax.scatter(xyz[take,0],xyz[take,1],s=.55,c=c[take],linewidth=0,rasterized=False)
 # Shared scale bar, chosen in an empty corner of both measured crops.
 ax.plot([-116,-111],[32,32],color=INK,lw=1.4);ax.text(-113.5,32.7,'5 m',ha='center',fontsize=9,color=INK)
 return ax
cloud([.012,.145,.217,.410],z['prior_xyz'],np.zeros(len(z['prior_xyz']),int))
txt(.012,.112,'0009  /  actual fixed XY crop',9.8,MUTED)
# Shared proposal layer explicitly precedes both authorization decisions.
rect(.265,.797,.480,.089,fc='#f0f3f6',ec='none',rounded=True)
txt(.280,.856,'Shared measured-surface proposals',11.3,INK,'bold')
txt(.280,.819,'Visible cells + native masks + fixed receiver IDs',10.2,MUTED)
arrow(.231,.839,.262,.839)
arrow(.504,.793,.504,.765,color=TEAL)
# A: masks jointly explain historical IDs; evidence persists after one-view removal.
rect(.265,.468,.480,.292,fc=PALE,ec='#c8e2d9',rounded=True)
txt(.280,.720,'A  Authorize historical identity edits',12,TEAL,'bold')
for i,x in enumerate([.287,.394,.501]):
 rect(x,.554,.088,.119,fc='white',ec='#cededb',rounded=True)
 rect(x+.009,.571,.070,.075,fc='#def0e7',ec='#9bcec0',rounded=True)
 for dx,dy in [(.025,.593),(.031,.614),(.021,.625)]:dot(x+dx,dy,BLUE,.003)
 for dx,dy in [(.059,.587),(.065,.612),(.056,.628)]:dot(x+dx,dy,PURPLE,.003)
 txt(x+.044,.662,f'view {i+1}',8.9,MUTED,ha='center')
# One removed view: no fictional score, explicitly a mechanism schematic.
rect(.501,.554,.088,.119,fc='#ffffffce',ec='none')
line(.511,.563,.579,.663,MUTED,1.0,'--');line(.511,.663,.579,.563,MUTED,1.0,'--')
arrow(.600,.611,.626,.611,color=TEAL)
txt(.680,.637,r'$\Delta_{-1}>\varepsilon$',13,TEAL,ha='center')
txt(.680,.582,'still positive',10.1,TEAL,ha='center')
txt(.280,.505,'Shared vs. separate masks  +  complete-link compatibility',10.1)
# Aliased IDs feed the local certificate; certificate is also required for old edits.
arrow(.505,.466,.505,.420,color=TEAL)
txt(.520,.443,'authorized ID seeds',9.7,MUTED)
# B: fixed local graph, distinct observation-frame groups and competing ID.
rect(.265,.100,.480,.316,fc='#f0f7fa',ec='#cbdde6',rounded=True)
txt(.280,.376,'B  Certify local identity and new writes',12,TEAL,'bold')
pts={0:(.302,.283),1:(.347,.313),2:(.348,.251),3:(.398,.279),4:(.443,.306),5:(.447,.248),6:(.535,.277),7:(.492,.197),8:(.400,.196),9:(.348,.183)}
for u,v in [(0,1),(0,2),(1,3),(2,3),(2,9),(3,4),(3,5),(3,8),(4,6),(5,6),(5,7),(7,8),(8,9)]:line(*pts[u],*pts[v])
# Teal seed circles are different frame groups for the same ID; coral is another ID.
for k,(x,y) in pts.items():dot(x,y,'#b8ccd5',.0035)
for k in [0,9]:dot(*pts[k],TEAL,.0055)
dot(*pts[6],CORAL,.0055)
rect(.383,.251,.030,.056,fc='#ffffffa0',ec=TEAL,lw=1.3)
dot(*pts[3],TEAL,.0055)
txt(.297,.321,r'$t_1$',9.5,TEAL);txt(.328,.157,r'$t_2$',9.5,TEAL)
txt(.524,.320,'rival ID',9.2,CORAL,ha='center')
txt(.561,.290,r'$d^{(2)}_{\rm winner}<d_{\rm rival}-\varepsilon$',11.7,TEAL)
txt(.561,.239,'Two distinct frame groups',10.0)
txt(.561,.190,'Direct mask witness',10.0)
txt(.281,.127,'Uncertified evidence: abstain',10.1,MUTED)
# Actual result. Blue is unchanged old identity, purple relabeled old, teal new measurement.
labels=np.zeros(len(z['revised_xyz']),int);labels[z['edited']]=1;labels[z['new']]=2
cloud([.792,.465,.196,.410],z['revised_xyz'],labels)
txt(.792,.420,'Same region and scale',10.1,MUTED)
for y,c,t in [(.365,BLUE,'Preserved old IDs'),(.318,PURPLE,'Authorized ID edits'),(.271,TEAL,'Witnessed additions')]:
 dot(.801,y,c,.004);txt(.813,y,t,10.0)
# Both edit authorization and the local certificate are required at commit.
arrow(.748,.607,.775,.607,color=TEAL);line(.776,.607,.776,.217,TEAL,1.5);arrow(.748,.217,.790,.217,color=TEAL)
txt(.792,.165,'Copy measured coordinates',10.0,INK,'bold')
# A compact bottom rail makes provenance and role of LIVO explicit.
line(.012,.070,.988,.070,'#d8e2e7',.8)
txt(.012,.030,'Poses: benchmark inputs / robot LIVO',9.8,MUTED)
txt(.988,.030,'Middle: mechanism schematic     |     Side panels: recorded data',9.8,MUTED,ha='right')
OUT.mkdir(exist_ok=True)
for ext in ['pdf','svg','png']:
 fig.savefig(OUT/f'architecture.{ext}',dpi=220,bbox_inches=None,pad_inches=0)
plt.close(fig)
manifest=json.loads((DATA/'provenance.json').read_text());manifest.update({'script_sha256':hashlib.sha256(HERE.read_bytes()).hexdigest(),'display_stride':4,'color_semantics':{'blue':'old points with unchanged ID','purple':'old points whose ID was changed','teal':'new measured source points'},'data_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*') if p.name!='provenance.json'}})
(OUT/'architecture_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Updated architecture.pdf / .svg / .png using recorded evidence and labeled mechanism schematics.')
