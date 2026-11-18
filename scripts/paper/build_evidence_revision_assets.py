"""Build paper evidence from frozen outputs; never run or modify experiments."""
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'build/benchmarks/revision_assessment_v1'
DEST = ROOT / 'docs/paper/evidence_authorized_revision/latex'
FIG = DEST / 'figures'
METHODS = ['strong_baseline', 'unconditional', 'history_only', 'full_authorization']
NAMES = ['Strong baseline', 'Unconditional', 'History-only', r'\textbf{Ours}']
INK, TEAL, BLUE, RED = '#193446', '#008b78', '#447bd4', '#db6158'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
PINS = {}


def pin(path):
    path = Path(path)
    PINS[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path):
    return json.loads(pin(path).read_text())


def load(path):
    return dict(np.load(pin(path)))


def save(fig, name):
    fig.savefig(FIG / (name + '.pdf'), bbox_inches='tight', pad_inches=.03)
    fig.savefig(FIG / (name + '.png'), dpi=160, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)


def tab(name, header, rows, columns):
    text = '\\begin{tabular}{' + columns + '}\n\\toprule\n'
    text += ' & '.join(header) + ' \\\\\n\\midrule\n'
    for row in rows:
        text += ' & '.join(str(x) for x in row) + ' \\\\\n'
    text += '\\bottomrule\n\\end{tabular}\n'
    (DEST / 'tables' / (name + '.tex')).write_text(text)


def main():
    for d in [FIG, DEST / 'tables', DEST / 'data']:
        d.mkdir(parents=True, exist_ok=True)
    s = read(SRC / 'summary.json')
    audit = read(SRC / 'delivery_audit.json')
    assert audit['complete'] and s['prospective_gate']
    p = read(SRC / 'protocol.json')
    objects = read(SRC / 'per_object.json')['results']
    maps = read(SRC / 'comparison_maps_complete.json')
    robot = read(SRC / 'robot_correspondence_audit.json')
    dev = read(ROOT / 'build/benchmarks/revision_authorization_v1/summary.json')
    factor = read(ROOT / 'build/benchmarks/revision_factorial_v1/summary.json')
    cover = read(SRC / 'cover/prepared.json')
    for f in ['summary.json', 'per_object.json', 'protocol.json', 'delivery_audit.json', 'robot_correspondence_audit.json']:
        shutil.copyfile(pin(SRC / f), DEST / 'data' / f)
    rows = []
    for method, name in zip(METHODS, NAMES):
        values = [s['results'][key][method]['primary']['metrics']['allAp'] * 100 for key in s['results']]
        rows.append([name] + [f'{v:.3f}' for v in values] + [f'{np.mean(values):.3f}'])
    tab('primary', ['Method', '0009', '0010', 'Mean'], rows, 'lrrr')
    rows = [[name] + [f"{s['mean_ap'][mode][method]*100:.3f}" for mode in ['primary','raw_targets','uniform_scores']]
            for method, name in zip(METHODS, NAMES)]
    tab('diagnostics', ['Method','Primary','Canonical','Uniform'], rows, 'lrrr')
    rows = []
    for method, name in zip(METHODS, NAMES):
        rows.append([name] + [f"{s['results'][key][method]['primary']['metrics']['classes'][cls]['ap']*100:.3f}"
                             for key in s['results'] for cls in ['building','car']])
    tab('classes', ['Method','09 Bldg.','09 Car','10 Bldg.','10 Car'], rows, 'lrrrr')
    rows = []
    for method, name in [('baseline','Strong baseline'),('unconditional','Unconditional'),
                         ('identity_only','Identity-only$^*$'),('write_only','Write-only'),
                         ('current_only','No history edits'),('ours',r'\textbf{Ours}')]:
        rows.append([name] + [f"{dev['results'][key][method]['primary']['metrics']['allAp']*100:.3f}"
                             for key in ['kitti360_0000','kitti360_0002']] + [f"{dev['mean_ap']['primary'][method]*100:.3f}"])
    for method, name in [('authorized_history_only','History-only'),('authorized_history_write','History + witness')]:
        rows.append([name] + [f"{factor['results'][key][method]['primary']['metrics']['allAp']*100:.3f}"
                             for key in ['kitti360_0000','kitti360_0002']] + [f"{factor['mean_ap']['primary'][method]*100:.3f}"])
    tab('development', ['Method','0000','0002','Mean'], rows, 'lrrr')
    tab('robot', ['Method','Points','IDs','Old edits'],
        [[name, f"{robot['methods'][m]['points']:,}",robot['methods'][m]['nonempty_object_ids'],
          f"{robot['methods'][m]['old_point_id_changes']:,}"] for m,name in zip(METHODS,NAMES)], 'lrrr')
    cases = [('kitti360_0009',2168),('kitti360_0010',2086),('kitti360_0009',2164)]
    tab('objects', ['Window / object','Strong','Uncond.','History','Ours'],
        [[key[-4:]+' / '+str(gid)] + [f"{next(x for x in objects[key]['raw_targets'] if x['gt_id']==gid)['methods'][m]['best_iou']:.5f}" for m in METHODS]
         for key,gid in cases], 'lrrrr')
    # A compact vector overview, with actual RGB and actual source-point map.
    fig = plt.figure(figsize=(10.6,4.35)); ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,title,body,accent=INK):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008,rounding_size=0.012',
                                    facecolor='#eff7f5' if accent==TEAL else '#f1f4f7',edgecolor='#cfdce2',linewidth=.8))
        ax.text(x+.013,y+h-.04,title,color=accent,weight='bold',fontsize=10,va='top')
        if body:ax.text(x+.013,y+h-.105,body,color=INK,fontsize=8.5,va='top',linespacing=1.5)
    box(.02,.54,.27,.40,'1  Shared observations','')
    scene=p['scenes'][0];job=read(Path(scene['job']));frame=next(f for f in job['frames'] if f['frame_id']==903)
    rgb=np.asarray(Image.open(pin(frame['rgb'])).convert('RGB'))
    ia=fig.add_axes([.035,.68,.24,.16]);ia.imshow(rgb);ia.axis('off')
    ax.text(.035,.635,'Posed RGB + measured LiDAR\nSAM3 / SigLIP / CG association',fontsize=8.5,color=INK,va='top')
    box(.35,.54,.28,.40,'2  Surface proposals',
        'Measured cells + native masks\nVisible coverage minus leakage\nAccepted regions + fixed receivers')
    box(.69,.54,.29,.40,'3  Revised object map','')
    geometry=load(SRC/'comparison_maps/kitti360_0009/full_authorization/geometry.npz')
    ma=fig.add_axes([.72,.60,.23,.25]);take=np.arange(0,len(geometry['xyz']),20)
    ma.scatter(geometry['xyz'][take,0],geometry['xyz'][take,1],s=.12,c=plt.get_cmap('turbo')((geometry['instance'][take]*37%251)/250))
    ma.set_aspect('equal');ma.axis('off');ax.text(.71,.565,'Actual 0009 output',fontsize=8,color=INK)
    box(.02,.08,.46,.34,'OURS  A  Authorize historical edits',
        'Shared vs separate mask explanations\nMargin survives deletion of any one view\nPairwise-compatible groups + local agreement',TEAL)
    box(.53,.08,.45,.34,'OURS  B  Certify new surface writes',
        'Direct mask witness on a measured cell\nCompeting IDs supported by two view groups\nAppend measured points; abstain when uncertified',TEAL)
    for xy,xy2 in [((.295,.75),(.34,.75)),((.49,.53),(.25,.43)),((.50,.53),(.75,.43)),
                   ((.49,.24),(.52,.24)),((.85,.43),(.85,.53))]:
        ax.add_patch(FancyArrowPatch(xy,xy2,arrowstyle='-|>',mutation_scale=12,color=TEAL,linewidth=1.3))
    ax.text(.5,.015,'Public benchmark: official poses     |     Robot branch: LIVO poses     |     No generated 3D coordinates',ha='center',fontsize=8,color=INK)
    save(fig,'architecture')
    # Actual prediction errors on fixed diagnostic crops, not attractive-only cases.
    fig, axes=plt.subplots(3,3,figsize=(8.9,7.8),layout='constrained')
    vizrows=[]
    for row,(key,gid) in enumerate(cases):
        ref=load(SRC/'evaluation'/key/'reference.npz'); xyz=ref['xyz'];truth=ref['raw_target_instance']==gid
        bounds=np.array([xyz[truth,:2].min(0)-2,xyz[truth,:2].max(0)+2]); center=bounds.mean(0)
        radius=max((bounds[1]-bounds[0]).max()/2,2);bounds=np.array([center-radius,center+radius])
        inside=np.all((xyz[:,:2]>=bounds[0])&(xyz[:,:2]<=bounds[1]),axis=1)
        item=next(x for x in objects[key]['raw_targets'] if x['gt_id']==gid)
        for col,(m,label) in enumerate([('strong_baseline','Strong baseline'),('unconditional','Unconditional'),('full_authorization','Ours')]):
            pred=load(SRC/'evaluation'/key/m/'reference_prediction.npz')['instance']
            oid=item['methods'][m]['object_id']; selected=pred==oid
            a=axes[row,col]
            for mask,color in [(truth&~selected,'#b6c0ca'),(selected&~truth,RED),(truth&selected,TEAL)]:
                ids=np.flatnonzero(mask&inside);ids=ids[::max(1,int(np.ceil(len(ids)/10000)))]
                a.scatter(xyz[ids,0],xyz[ids,1],s=.6,color=color,rasterized=True)
            a.set(xlim=bounds[:,0],ylim=bounds[:,1]);a.set_aspect('equal');a.tick_params(labelsize=7)
            a.set_title(f"{label}  |  IoU {item['methods'][m]['best_iou']:.5f}",fontsize=9,color=TEAL if m=='full_authorization' else INK)
            if col==0:a.set_ylabel(f'{key[-4:]} / building {gid}\ny (m)',fontsize=9)
            if row==2:a.set_xlabel('x (m)',fontsize=8)
            excluded=int(np.count_nonzero(selected&~inside))
            a.text(.02,.02,f'Pred. outside crop: {excluded:,}',transform=a.transAxes,fontsize=6.7,color=INK,
                   bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
            vizrows.append(dict(window=key,gt=gid,method=m,object_id=oid,crop_xy=bounds.tolist(),predicted_points_outside_crop=excluded))
    fig.suptitle('Matched surface (teal)   |   Excess surface (coral)   |   Missing surface (gray)',fontsize=10,color=INK)
    save(fig,'object_cases')
    # Keep original images and map panels intact, including unchanged/empty frames.
    for f in (ROOT/'output/revision_assessment_v1').glob('*.png'):
        shutil.copyfile(pin(f),FIG/f.name)
    for f in ['frame_000126.png','vehicles.png','all_maps.png']:
        shutil.copyfile(pin(ROOT/'output/revision_robot_v1'/f),FIG/('robot_'+f))
    manifest=dict(source_inputs=PINS,source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  frozen_experiments_unchanged=True,new_inference=False,new_maps=False,
                  diagnostic_case_selection='Two new raw AP50 matches plus the building where unconditional beats Ours; selected after scoring, not representative random samples.',
                  diagnostic_crops=vizrows,cover_acceptance=cover['results'])
    (DEST.parent/'asset_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Generated tables, vector architecture, measured object cases, and portable evidence data.')


if __name__ == '__main__':main()
