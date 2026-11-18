"""Trace actual label-bearing observations behind frozen source anchors.

Read-only mechanism diagnosis on already exposed data. No reference labels,
new inference, maps, scoring or hyperparameter selection. Source acquisition
frames are deliberately kept distinct from associated mask observation frames.
"""
import time
from pathlib import Path
import numpy as np
from paper_final_common import ROOT,read,dump,sha

OUT=ROOT/'build/benchmarks/observation_anchor_provenance_v1'
B=ROOT/'build/benchmarks'


def contexts():
    rows=[]
    for key in ['kitti360_0000','kitti360_0002','kitti360_0006','rosbag2245']:
        new=key=='kitti360_0006';robot=key=='rosbag2245';root=B/'batch_transport_assessment_v1'
        graph=(root/'geometry_reference'/key/'maps/geometry_graph_reference' if new else B/'robot_geometry_reference_v1/maps/geometry_graph_reference' if robot else B/'signed_query_association_v1/maps'/key/'geometry_graph')
        nodes=graph if new or robot else B/'signed_query_association_v1/nodes'/key
        field=(root/'fields'/key/'field.npz' if new else B/'batch_transport_robot_v1/fields'/key/'field.npz' if robot else B/'batch_stable_transport_v1/fields'/key/'full/field.npz')
        full=(root/'full_measured'/key if new else B/'batch_transport_robot_v1/full_measured'/key if robot else B/'full_lidar_surface_inputs_v1'/key)
        rows.append(dict(key=key,graph=graph,nodes=nodes,field=field,full=full,
            sem=(root/'semantics/aggregate'/key/'source.npz' if new else B/'batch_transport_robot_v1/semantics/aggregate'/key/'source.npz' if robot else B/'closed_vocabulary_semantics_v1/aggregate'/key/'source.npz'),
            native=(root/'maps'/key/'largest' if new else B/'fixed_identity_recovery_control_v1/maps'/key/'largest'),
            front=(root if new else B/'discovery_query_benchmark_v1'),
            support=(root if new else B/'deferred_surface_support_v1')))
    return rows


def trace(context):
    c=context;key=c['key'];started=time.monotonic();sem=np.load(c['sem']);owner=sem['owners'];full=np.load(c['full']/'geometry.npz');mapping=np.load(c['full']/'old_to_full.npy');field=np.load(c['field']);cell=field['source_to_cell'][mapping];acquisition=full['source_frame'][mapping]
    groups=read(c['graph']/'groups.json');assignment={int(k):int(v) for k,v in groups['old_id_assignment'].items()};core=read(c['native']/'classification.json')['objects'];byobs={obs:assignment[row['object_id']] for row in core for obs in row['observation_ids']}
    pieces=[];native_pieces=0;raw_pieces=0;files={}
    def record(ids,oid,frame):
        ids=np.asarray(ids,np.int64);ids=ids[owner[ids]==oid]
        assert (acquisition[ids]<=frame).all(), 'Evidence precedes measurement'
        if len(ids):pieces.append(np.column_stack([ids,np.full(len(ids),oid),np.full(len(ids),frame)]))
        return len(ids)
    front=read(c['front']/'masks_complete.json')['scenes'][key]['records']
    for row in front:
        f=c['support']/'supports'/key/'all'/f'{row["frame_id"]:06d}.npz';d=np.load(f);files[str(f)]=sha(f)
        for j in range(row['masks']):
            oid=byobs.get(row['observation_offset']+j)
            if oid is not None:native_pieces+=record(d['point_ids'][d['offsets'][j]:d['offsets'][j+1]],oid,row['frame_id'])
    node_data=np.load(c['nodes']/'nodes.npz');metadata=read(c['nodes']/'nodes.json')
    for group in groups['groups']:
        oid=group['object_id']
        for j in group['node_ids']:
            m=metadata[j]
            if m['hypothesis_index']>=0:
                raw_pieces+=record(node_data['source_ids'][node_data['source_offsets'][j]:node_data['source_offsets'][j+1]],oid,m['seed_frame'])
    incidence=np.unique(np.concatenate(pieces) if pieces else np.empty((0,3),np.int64),axis=0)
    seen=np.bincount(incidence[:,0],minlength=len(owner));fg=np.flatnonzero(owner>0);missing=fg[seen[fg]==0]
    tokens=np.unique(np.column_stack([cell[incidence[:,0]],incidence[:,1:]]),axis=0)
    old=np.unique(np.column_stack([cell[fg],owner[fg],acquisition[fg]]),axis=0)
    np.testing.assert_array_equal(old,field['seeds'][field['seeds'][:,1]>0])
    width=int(owner.max())+1;old_keys,acq_count=np.unique(old[:,0]*width+old[:,1],return_counts=True)
    obs_keys,first,obs_count=np.unique(tokens[:,0]*width+tokens[:,1],return_index=True,return_counts=True)
    def lookup(keys):
        ix=np.searchsorted(obs_keys,keys);ok=ix<len(obs_keys);ok[ok]&=obs_keys[ix[ok]]==keys[ok]
        counts=np.zeros(len(keys),np.int64);frames=np.full(len(keys),-1,np.int64);counts[ok]=obs_count[ix[ok]];frames[ok]=tokens[first[ix[ok]],2]
        return counts,frames
    counts,_=lookup(old_keys)
    denied=np.flatnonzero((field['cell_owner']>0)&(field['frame_owner']!=field['cell_owner']))
    g=field['cell_groups'];nearest=g[field['cell_nearest'][denied]]
    a=nearest[:,0,1]*width+nearest[:,0,0];b=nearest[:,1,1]*width+nearest[:,1,0]
    ca,fa=lookup(a);cb,fb=lookup(b)
    # Two already certified spatial anchors alone have >=2 observation frames.
    # A possible recovery witness, NOT a new prediction or proof of accuracy.
    witness=(ca>0)&(cb>0)&((ca>=2)|(cb>=2)|(fa!=fb))
    folder=OUT/key;folder.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(folder/'incidence.npz',source_owner_observation_frame=incidence,cell_owner_observation_frame=tokens,foreground_sources_without_trace=missing,cell_owner_keys=old_keys,acquisition_frame_counts=acq_count,observation_frame_counts=counts,spatial_only_cells=denied,observed_two_frame_witness=witness)
    per_object=[]
    for oid in np.unique(owner[fg]):
        take=owner[fg]==oid;groups_take=old_keys%width==oid;denied_take=field['cell_owner'][denied]==oid
        per_object.append(dict(object_id=int(oid),source_anchors=int(take.sum()),sources_without_trace=int((seen[fg[take]]==0).sum()),sources_with_repeated_label_observations=int((seen[fg[take]]>=2).sum()),cell_owners=int(groups_take.sum()),single_acquisition_multiple_observations=int((groups_take&(acq_count==1)&(counts>=2)).sum()),multiple_acquisition_single_observation=int((groups_take&(acq_count>=2)&(counts==1)).sum()),spatial_only_cells=int(denied_take.sum()),two_observation_witness_cells=int((denied_take&witness).sum())))
    stats=dict(source_anchors=len(fg),unique_source_frame_incidence=len(incidence),native_incidence_before_dedup=native_pieces,raw_incidence_before_dedup=raw_pieces,source_anchors_without_label_trace=len(missing),sources_with_repeated_label_observations=int((seen[fg]>=2).sum()),label_observations_at_later_than_acquisition=int((incidence[:,2]>acquisition[incidence[:,0]]).sum()),acquisition_seed_records=len(old),observation_seed_records=len(tokens),cell_owner_count=len(old_keys),single_acquisition_multiple_observations=int(((acq_count==1)&(counts>=2)).sum()),multiple_acquisition_single_observation=int(((acq_count>=2)&(counts==1)).sum()),spatial_only_cells=len(denied),two_observation_witness_cells=int(witness.sum()),per_object=per_object,seconds=time.monotonic()-started)
    for path in [c['sem'],c['field'],c['full']/'geometry.npz',c['full']/'old_to_full.npy',c['graph']/'groups.json',c['native']/'classification.json',c['nodes']/'nodes.json',c['nodes']/'nodes.npz',c['front']/'masks_complete.json']:files[str(path)]=sha(path)
    dump(folder/'summary.json',stats);dump(folder/'inputs.json',files)
    print('OBSERVATION PROVENANCE',key,{k:v for k,v in stats.items() if k!='per_object'},flush=True)
    return stats


def main():
    assert not OUT.exists();OUT.mkdir()
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),source_sha256=sha(__file__),scenes=[c['key'] for c in contexts()],scope='Already exposed three public windows and robot, frozen anchor labels; zero GT/reference reads, new maps, model forwards or score searches.',question='Does first acquisition-frame grouping represent actual label-bearing observation provenance, and can repeated associated observations witness some spatial-control-only cells?',provenance='Native retained object observation IDs and all measured-mask supports, plus selected raw hypothesis connected supports in the final graph groups; retain incidence only where frozen source owner equals that final group identity. Intrinsic additions require actual associated mask containment. Duplicated raw queries in same source/frame collapse.',limitation='Conditional on frozen global identities, associations and assigned sources; not an end-to-end counterfactual with a frame removed. Observation multiplicity is not statistical independence or semantic correctness. Two-nearest-spatial-anchor witness is a feasibility bound, not a new map decision.'))
    results={c['key']:trace(c) for c in contexts()}
    dump(OUT/'summary.json',dict(complete=True,results=results,new_maps=0,new_inference=0,reference_labels_read=False,source_sha256=sha(__file__)))


if __name__=='__main__':main()
