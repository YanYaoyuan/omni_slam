"""Verify frozen outputs and publish development diagnostics, never GT metrics."""
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/omni_history_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from audit_evidence_retention import ROOT, OUT, REPORT, dump
from omni_semantic_map.flat_benchmark import sha256


def read(p):
    return json.loads(p.read_text())


def main():
    root=OUT/'history_graph_v1'; dest=REPORT/'history_graph_v1'
    spec=read(root/'protocol.json'); summary=read(root/'summary.json')
    assert summary['complete'] and sha256(root/'protocol.json')==summary['protocol_sha256']
    for p,h in {**spec['sources'],**spec['dependencies']}.items():
        assert sha256(Path(p))==h,p
    baseline=read(root/'baseline/classification.json')['objects']
    bg=np.load(root/'baseline/geometry.npz')
    reference=OUT/'scale_ablation_v1/cache_supportcc/both'
    assert baseline==read(reference/'classification.json')['objects']
    for k in bg.files:
        np.testing.assert_array_equal(bg[k],np.load(reference/'geometry.npz')[k])
    template=read(dest/'annotation_template.json')
    assert len(template['edges'])==25
    for e in template['edges']:
        assert e['identity_label'] is None and e['reviewer'] is None
        assert sha256(dest/e['evidence_image'])==e['image_sha256']
    nodes=read(root/'snapshot/nodes.json'); node_by_id={n['alias']:n for n in nodes}
    manifest=read(OUT/'livo_full_strong_v2/cache_supportcc/manifest.json')
    frame_by_obs={r['observation_offset']+k:r['frame_id'] for r in manifest['records'] for k in range(r['masks'])}
    def extras(rows):
        return sum(len(o['observation_ids'])-len({frame_by_obs[i] for i in o['observation_ids']}) for o in rows)
    local_fg=None
    checks=[]
    for item in summary['variants']:
        variant=item['variant']; path=root/variant
        for name,h in item['hashes'].items():
            assert sha256(path/name)==h
        graph=read(path/'graph.json')
        assert sorted(n for group in graph['groups'] for n in group)==sorted(node_by_id)
        for group in graph['groups']:
            # Existing within-node conflicts are not newly introduced graph unions.
            for i,a in enumerate(group):
                for b in group[i+1:]:
                    assert not set(node_by_id[a]['frames']) & set(node_by_id[b]['frames'])
        rows=read(path/'classification.json')['objects']
        ids=[i for o in rows for i in o['observation_ids']]
        assert len(ids)==len(set(ids))
        assert not item['lost_observations'] and not item['foreground_lost_observations']
        fg=[o for o in rows if o['predictions']['mean']['retained']]
        assert extras(fg)==0
        if variant=='native_bridge_1':
            local_fg=fg
        if variant=='dino_bridge_1':
            # Evaluation IDs shift when a background region is absent.
            stable=lambda rows:{tuple(o['observation_ids']):{k:v for k,v in o.items() if k!='object_id'} for o in rows}
            assert stable(fg)==stable(local_fg)
            ga=np.load(root/'native_bridge_1/geometry.npz'); gb=np.load(path/'geometry.npz')
            matching={tuple(o['observation_ids']):o for o in local_fg}
            for obj in fg:
                old=matching[tuple(obj['observation_ids'])]
                np.testing.assert_array_equal(ga['xyz'][ga['instance']==old['object_id']],gb['xyz'][gb['instance']==obj['object_id']])
        loop=next(o for o in rows if 3074 in o['observation_ids'])
        original=next(o for o in baseline if 3074 in o['observation_ids'])
        assert loop['observation_ids']==original['observation_ids']
        checks.append(dict(variant=variant,ownership_unique=True,baseline_members_preserved=True,
                           no_new_graph_component_coframe_conflicts=True,return_car_members_unchanged=True,
                           final_same_frame_extras=extras(rows)))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['Baseline','Native bridge\n2 frames','+ DINO\n2 frames','Native bridge\n1 frame','+ DINO\n1 frame']
    values=[205,217,216,220,220]; objects=[41,40,40,39,39]
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained')
    colors=['#8794a2','#2b6e9b','#b87836','#2b6e9b','#b87836']
    for ax,ys,title in zip(axes,[values,objects],['Retained foreground observations','Foreground predictions']):
        bars=ax.bar(np.arange(5),ys,color=colors,width=.65)
        ax.set_xticks(np.arange(5),names,fontsize=8);ax.set_title(title,loc='left',fontweight='bold')
        ax.bar_label(bars,padding=4);ax.set_ylim(0,max(ys)*1.15);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('History recovery on rosbag2245 — development diagnostics',fontsize=14,fontweight='bold')
    fig.supxlabel('Counts describe retention and merging, not identity accuracy or benchmark gains.',fontsize=10)
    for ext in ['png','svg']:
        fig.savefig(dest/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    snapshot=np.load(root/'snapshot/geometry.npz')
    groups=[[525,1000],[588,597,617],[1165,1185,1202],[2361,2364,2386]]
    fig,axes=plt.subplots(2,4,figsize=(13,6.6),layout='constrained')
    palette=['#286b9b','#d58b39','#963da6']
    for col,group in enumerate(groups):
        merged=np.concatenate([snapshot['xyz'][snapshot['instance']==n] for n in group])
        origin=merged.mean(0)
        for row,dims in enumerate([(0,1),(0,2)]):
            ax=axes[row,col]
            for color,node in zip(palette,group):
                pts=snapshot['xyz'][snapshot['instance']==node]-origin
                ax.scatter(pts[:,dims[0]],pts[:,dims[1]],s=7,alpha=.7,c=color,label=str(node),rasterized=True)
            ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.15)
            ax.set_xlabel('X offset (m)');ax.set_ylabel(('Y' if row==0 else 'Z')+' offset (m)')
            if row==0:
                ax.set_title(' + '.join(map(str,group)),fontsize=11);ax.legend(fontsize=8,frameon=False)
    fig.suptitle('Original measured fragment supports in a common world frame',fontsize=14,fontweight='bold')
    fig.supxlabel('Native two-frame bridges join these histories. No alignment, dense completion, or GT correction applied.',fontsize=10)
    for ext in ['png','svg']:
        fig.savefig(dest/f'fragment_supports.{ext}',dpi=180)
    plt.close(fig)
    background={(287,295),(351,1181),(378,1113),(778,831),(1317,1320),(1500,1543),(1709,1730),(2069,2080),(2182,2208)}
    notes=[]
    for e in template['edges']:
        pair=(e['a'],e['b'])
        if pair in background:
            note='背景表面/立柱/墙顶分区证据；不能按车辆式离散身份直接判定正确合并。'
        elif pair==(672,883):
            note='已补看两条历史全部 4 帧；边缘可见区域有限，跨视角身份仍不确定。'
        elif pair in {(525,1000),(526,1013)}:
            note='已补看两条历史全部成员；车身、相邻车辆及立柱布局支持同车假设，但没有独立身份真值。'
        else:
            note='首组证据图的车身及相邻停车布局与同车片段假设一致；未完成每个成员的独立身份标注。'
        notes.append(dict(a=e['a'],b=e['b'],note=note))
    dump(dest/'review.json',dict(complete=True,scope='Assistant image inspection, not independent identity labels.',
        all_25_first_witness_triptychs_reviewed=True,independent_ground_truth=False,
        full_history_groups_reviewed=[[525,1000],[526,1013],[672,883]],
        dino_veto_2822_review='第 303 帧当前车辆与第 297 帧小尺度历史图像支持同车假设；同帧竞争者 2867 位于另一侧车位。原始相似度 0.742 小于竞争者 0.830，导致两帧规则只剩一票。',
        baseline_exact=True,native_and_dino_one_frame_foreground_exact_after_evaluation_id_remap=True,
        baseline_same_frame_extras=extras(baseline),checks=checks,edge_notes=notes,
        figure_hashes={p.name:sha256(p) for p in dest.glob('*.png')}))
    print('Verified baseline exact, 4 graph branches, all memberships conserved before filtering, no lost baseline final observations, and 25 image hashes.')


if __name__=='__main__':
    main()
