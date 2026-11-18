import copy,json
import numpy as np
from run_partition_revision import export_method


def context_fixture():
    source=np.column_stack([np.arange(100),np.zeros((100,2))]).astype(float)
    labels=np.repeat([1,2,9],20)
    records={i:dict(object_id=i,observation_ids=[],predictions=dict(mean=dict(name='car',semantic_id=26,train_id=13,score=s,retained=True))) for i,s in [(1,.7),(2,.9),(9,.8)]}
    a,b=np.arange(20),np.arange(20,40);whole=np.r_[a,b,np.arange(80,100)]
    view=dict(frame_id=10,visible_cells=whole,queries=[whole],scores=np.array([1.]),query_overlap=np.ones((1,1)),used_for_prompt=False)
    bundle=dict(source_candidates=dict(keep=[a,b],split=[a,b],merge=[whole]),valid=dict(keep=True,split=True,merge=True),
                supports=[np.arange(80,90),np.arange(90,100),np.arange(80,90),np.arange(90,100)],
                history_masks=[np.zeros(60,bool)]*4,prompt_compliance=[True]*4)
    case=dict(row=dict(groups=[0,1],semantic_id=26,policies=dict(max_visible=dict(frame_id=5))),
              policies=dict(ordinary=bundle,symmetric=bundle),acquisition=dict(view,frame_id=5,used_for_prompt=True),validation=[view])
    return dict(source=source,cell=np.arange(100),sets=[a,b],groups=[dict(node_ids=[],seed_frames=[1,2,3]),dict(node_ids=[],seed_frames=[1,2,3])],
                nodes=[],base=dict(xyz=source[:60],instance=labels),records=records,classification=dict(objects=list(records.values())),
                group_oid={0:1,1:2},protected=np.arange(80),cases=[case])


def test_export_preserves_geometry_and_unrelated_objects_and_current_only_scores(tmp_path):
    ctx=context_fixture()
    for method in ['ours','current_only']:
        folder=tmp_path/method;folder.mkdir();run=export_method(ctx,method,folder);g=np.load(folder/'geometry.npz')
        np.testing.assert_array_equal(g['xyz'][:60],ctx['base']['xyz'])
        np.testing.assert_array_equal(g['xyz'][60:],ctx['source'][g['added_source_ids']])
        np.testing.assert_array_equal(g['instance'][40:60],np.full(20,9))
        np.testing.assert_array_equal(g['added_source_ids'],np.arange(80,100))
        objects={o['object_id']:o for o in json.loads((folder/'classification.json').read_text())['objects']}
        if method=='ours':
            assert run['old_point_ids_changed']==20 and 2 not in objects
            assert objects[1]['predictions']['mean']['score']==.9
        else:
            np.testing.assert_array_equal(g['instance'][:60],ctx['base']['instance'])
            assert objects[1]['predictions']['mean']['score']==.7 and objects[2]==ctx['records'][2]


def test_failed_birth_rejects_entire_transaction(tmp_path):
    ctx=context_fixture();ctx['group_oid']={};ctx['base']['instance']=np.full(60,9)
    ctx['records']={9:ctx['records'][9]};ctx['classification']['objects']=list(ctx['records'].values())
    ctx['nodes']=[dict(seed_frame=1,score=.05),dict(seed_frame=2,score=.05)]
    ctx['groups']=[dict(node_ids=[0],seed_frames=[1]),dict(node_ids=[1],seed_frames=[2])]
    run=export_method(ctx,'ours',tmp_path);g=np.load(tmp_path/'geometry.npz')
    assert run['birth_rejections']==1 and run['applied_pairs']==0 and run['added_points']==0
    np.testing.assert_array_equal(g['xyz'],ctx['base']['xyz'])
    np.testing.assert_array_equal(g['instance'],ctx['base']['instance'])
