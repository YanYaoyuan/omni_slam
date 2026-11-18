"""Full-window partition mechanism and same-observation additive control.

No reference labels are read here. These are intermediate object partitions;
an end-to-end map/score requires the separately frozen common exporter.
"""
import argparse
from pathlib import Path
import time
import numpy as np
from paper_final_common import ROOT,read,dump,sha
from prepare_censored_object_partition import OUT,verify
from omni_semantic_map.censored_object_partition import make_histograms,merge_gain,region_score
from omni_semantic_map.censored_partition_native import library,partition_native
from omni_semantic_map.surface_formation import partition_surfaces

METHODS=['pairwise','region']


def freeze():
    p=verify();prepared=read(OUT/'prepared.json');audit=read(OUT/'space_audit.json')
    assert audit['necessary_preflight_passed'] and not (OUT/'partition_protocol.json').exists()
    for f,h in prepared['files'].items():assert sha(f)==h,f
    files=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/censored_partition_native.py',
           ROOT/'omni_semantic_map/omni_semantic_map/censored_partition_native.cpp',
           ROOT/'omni_semantic_map/test/test_censored_partition_native.py',
           ROOT/'omni_semantic_map/omni_semantic_map/surface_formation.py']
    dump(OUT/'partition_protocol.json',dict(created_epoch=time.time(),methods=METHODS,
        input_protocol_sha256=sha(OUT/'input_protocol.json'),prepared_sha256=sha(OUT/'prepared.json'),
        audit_sha256=sha(OUT/'space_audit.json'),sources={str(f):sha(f) for f in files},
        mechanism='Start from the SAME frozen measured atoms. Region mode greedily accepts the largest positive whole-region marginal likelihood gain, updating categorical histories after each merge. No old-ID must-link constraint. Novelty/accuracy not established by an implementation.',
        control='Pairwise mode computes the EXACT same likelihood gain at each INITIAL atom edge, then greedily merges by the sum of frozen boundary edge costs. Uniformly scale all costs by0.01 so the inherited solver1e-12 cutoff equals the region solver1e-10 in original units; greedy ranking and objective are otherwise unchanged in real arithmetic. This isolates whole-region recomputation, sharing masks, semantic sampling, native scores, atoms, graph, initial singleton partition and information budget. It is an adapted mechanism control, not a named external-method reproduction.',
        compute='C++ implementation algebraically cancels noncommon categorical terms, equivalent real-arithmetic objective. Native libm/reduction order may differ from Python at floating ties. Stable minimum-root tie ordering, positive>1e-10, independent Python objective/remaining-boundary replay tolerance1e-7. Six tests passed, including30random fractional-count graphs with every chosen merge and greedy condition independently replayed.',
        budget='Two exposed complete windows x two complete partitions, CPU only, no new image inference/training/parameters. No methods or scenes selected after scoring, no retries with different priors/graph/birth rules.',
        attribution='Partition objective values are not comparable to the frozen baseline AP and are not new map metrics. Intermediate proposed splits/new regions are not committed historical edits or correct objects. Full common map export, strong baseline, history-lock ablation, same-module robot and independent assessment still required.'))
    print('COMPLETE PARTITION COMPARISON FROZEN',flush=True)


def verify_execution():
    p=verify();e=read(OUT/'partition_protocol.json')
    assert sha(OUT/'input_protocol.json')==e['input_protocol_sha256']
    assert sha(OUT/'prepared.json')==e['prepared_sha256']
    assert sha(OUT/'space_audit.json')==e['audit_sha256']
    for f,h in e['sources'].items():assert sha(f)==h,f
    return p


def solve(key):
    p=verify_execution();c=next(c for c in p['contexts'] if c['key']==key)
    prepared=read(OUT/'prepared.json');path=OUT/'prepared'/key/'atoms.npz'
    assert sha(path)==prepared['files'][str(path)]
    folder=OUT/'partitions'/key;folder.mkdir(parents=True,exist_ok=False)
    tick=time.monotonic();d=dict(np.load(path));n=len(d['cell_ids']);k=d['categories']
    hist=make_histograms(n,d['observation_atom'],d['observation_view'],d['observation_token'],d['observation_mass'],k)
    primitive=np.empty(len(d['left']),float)
    for i,(a,b) in enumerate(zip(d['left'],d['right'])):
        primitive[i]=merge_gain(hist[a],hist[b],k)
        if (i+1)%50000==0:print('PRIMITIVE COMPATIBILITY',key,i+1,'/',len(primitive),flush=True)
    np.save(folder/'primitive_gain.npy',primitive)
    field=np.load(c['field']);full_cells=field['cells'];source_cell=field['source_to_cell']
    cell_sources=np.bincount(source_cell,minlength=len(full_cells));base=np.load(Path(c['ours'])/'geometry.npz')
    # Exact grid correspondence, not a nearest-point/GT correspondence.
    dtype=np.dtype([('x',np.int64),('y',np.int64),('z',np.int64)])
    keys=np.ascontiguousarray(full_cells,np.int64).view(dtype).ravel()
    old_keys=np.ascontiguousarray(np.floor(base['xyz']/.2).astype(np.int64)).view(dtype).ravel()
    found=np.searchsorted(keys,old_keys);valid=found<len(keys)
    valid[valid] &= keys[found[valid]]==old_keys[valid]
    lookup=np.full(len(full_cells),-1,np.int64);lookup[d['cell_ids']]=np.arange(n)
    old_atom=np.full(len(old_keys),-1,np.int64);old_atom[valid]=lookup[found[valid]]
    original_ids=base['instance'];results={};files={};outputs={}
    native=library(folder/'native')
    for method in METHODS:
        begin=time.monotonic();dest=folder/method;dest.mkdir()
        if method=='pairwise':
            labels,trace=partition_surfaces(n,d['left'],d['right'],primitive*.01,np.zeros(n,np.int64))
            trace['affinity_scale']=.01
        else:
            labels,trace=partition_native(n,k,d['left'],d['right'],d['observation_atom'],d['observation_view'],
                d['observation_token'],d['observation_mass'],native)
        final=make_histograms(n,labels[d['observation_atom']],d['observation_view'],
                             d['observation_token'],d['observation_mass'],k)
        whole_score=float(sum(region_score(h,k) for h in final if h))
        if method=='region':
            assert abs(whole_score-trace['final_score'])<=1e-7*max(1.,abs(whole_score))
            boundary=np.unique(np.minimum(labels[d['left']],labels[d['right']])*n+np.maximum(labels[d['left']],labels[d['right']]))
            boundary=boundary[boundary//n!=boundary%n]
            remaining=max((merge_gain(final[a],final[b],k) for a,b in zip(boundary//n,boundary%n)),default=0.)
            assert remaining<=1e-7
        else:remaining=None
        counts=np.bincount(labels,weights=cell_sources[d['cell_ids']],minlength=n).astype(np.int64)
        viewpairs=np.unique(labels[d['observation_atom']]*len(k)+d['observation_view']//2)
        observed_frames=np.bincount(viewpairs//len(k),minlength=n)
        birth=(counts>=16)&(observed_frames>=3)
        known=old_atom>=0;point_region=np.full(len(old_atom),-1,np.int64);point_region[known]=labels[old_atom[known]]
        # These are potential history operations, not exported IDs or accuracy.
        represented=known.copy();represented[known]&=birth[point_region[known]]
        old_to_regions={int(o):np.unique(point_region[represented&(original_ids==o)]).tolist() for o in np.unique(original_ids)}
        maxold=int(original_ids.max(initial=0))+1
        pairs=np.unique(point_region[represented]*maxold+original_ids[represented])
        touched=np.bincount(pairs//maxold,minlength=n)
        stats=dict(atoms=n,regions=int(len(np.unique(labels))),birth_eligible_regions=int(birth.sum()),
            birth_eligible_by_class={str(cls):int((birth&(d['train_class']==cls)).sum()) for cls in [2,13]},
            original_points_with_atom=int(known.sum()),original_points_in_birth_eligible_regions=int(represented.sum()),
            old_ids_touching_multiple_regions=sum(len(v)>1 for v in old_to_regions.values()),
            regions_touching_multiple_old_ids=int((touched>1).sum()),
            whole_region_score=whole_score,python_remaining_merge_gain=remaining,
            seconds=time.monotonic()-begin,committed_history_edits=0,new_map_scores=0)
        np.savez_compressed(dest/'partition.npz',root=labels,source_count=counts,observed_frames=observed_frames,
            birth_eligible=birth,old_point_proposed_region=point_region)
        dump(dest/'trace.json',trace);dump(dest/'old_identity_overlap.json',old_to_regions);dump(dest/'run.json',stats)
        outputs[method]=labels;results[method]=stats
        print('COMPLETE PARTITION',key,method,stats,flush=True)
    # Minimum atom index is the canonical label in both implementations.
    changed=int((outputs['pairwise']!=outputs['region']).sum())
    for f in folder.rglob('*'):
        if f.is_file():files[str(f)]=sha(f)
    dump(folder/'complete.json',dict(complete=True,results=results,files=files,
        canonical_atom_labels_changed=changed,seconds=time.monotonic()-tick,
        protocol_sha256=sha(OUT/'partition_protocol.json'),gt_read=False,new_maps=0))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','solve']);parser.add_argument('--key')
    a=parser.parse_args();freeze() if a.action=='freeze' else solve(a.key)
