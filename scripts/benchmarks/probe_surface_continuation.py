"""One finite cross-window test of geometry-derived transport affinities.

Frozen predicted anchors supply pseudo edge labels. No reference annotations,
new masks, new maps or AP are read or generated in this screening experiment.
"""
import argparse,time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from paper_final_common import ROOT,read,dump,sha
from audit_transport_unknown_seeds import known_semantics
from omni_semantic_map.surface_continuation import geometry_features,pure_anchor_labels,FEATURES

OUT=ROOT/'build/benchmarks/surface_continuation_probe_v1'
B=ROOT/'build/benchmarks'
KEYS=['kitti360_0000','kitti360_0002']


def freeze():
    assert not OUT.exists();inputs={}
    for key in KEYS:
        for f in [B/'batch_stable_transport_v1/fields'/key/'full/field.npz',B/'closed_vocabulary_semantics_v1/aggregate'/key/'source.npz',B/'full_lidar_surface_inputs_v1'/key/'old_to_full.npy']:
            inputs[str(f)]=sha(f)
    sources=[Path(__file__),ROOT/'omni_semantic_map/omni_semantic_map/surface_continuation.py',ROOT/'omni_semantic_map/test/test_surface_continuation.py',ROOT/'scripts/benchmarks/audit_transport_unknown_seeds.py']
    dump(OUT/'protocol.json',dict(created_epoch=time.time(),input_hashes=inputs,source_hashes={str(f):sha(f) for f in sources},scenes=KEYS,
        motivation='Observation provenance audit finds very few additional multiview witnesses. Replace unsupported hard repetition requirement with learned surface continuation, only if source evidence supports cross-window discrimination.',
        labels='Pure single-owner graph cells from frozen source anchors; valid known background is0, unknown/conflicted=-1. Edge supervised only when both endpoints known. Target same owner vs different owner. Predicted pseudo-labels are not GT or semantic accuracy.',
        features=list(FEATURES),geometry='Existing20cmcell graph and firstactualsourcecenters. Local covariance from all symmetric adjacent nodes plus self, centered coordinates for numerical stability. Eigenvalue/normal/step invariants; unsupported normals explicitly marked. No instanceID, classID, absoluteXYZ or reference input.',
        fit='Two leave-one-window-out folds, each distance-only1feature and full14feature MLP(16,16),4fits total. Balanced at most50000edges/class using seed0; same sampled edges across arms. Train80epochs Adam lr0.001,batch4096,CPU,seed0. No early stopping,checkpoint/seed/resampling search.',
        screening='Both held-out windows full-feature AUROC>=distance-only+.05 and foreground-same-owner false-negative-rate at.5 no worse than distance-only. All known edges scored and four pair types reported. Necessary pseudo-label information test only, not sufficient for maps/paper.',
        prospective_map_rule='If screening passes, freeze a separate bounded full-map experiment. Shared old geometry/anchors, use continuous edgecost length*(1-log(max(affinity,1e-6))), no hard edge deletion; same two-spatial-cell acceptance and originalmap/IDs/scores immutable. Full-map controls must include same-input Euclidean and fixed geometric costs, not just weak distance classifier.',
        limits='This is learned graph affinity, an established technique; not novelty by itself. No new reference access/maps/newpublic scenes/robot in screening. No claim that predicted anchor labels are correct or independent observations.',
        previous_branch='Acquisition-frame certificate results/gates remain frozen. No observation-frame certificate map experiment added.'))
    print('SURFACE CONTINUATION SCREEN FROZEN',flush=True)


def verify():
    p=read(OUT/'protocol.json')
    for f,h in {**p['input_hashes'],**p['source_hashes']}.items():assert sha(f)==h,f
    return p


def prepare():
    verify();assert not (OUT/'features_complete.json').exists();files={};stats={}
    for key in KEYS:
        field=np.load(B/'batch_stable_transport_v1/fields'/key/'full/field.npz');n=len(field['centers']);graph=csr_matrix((field['graph_weights'],field['graph_indices'],field['graph_indptr']),shape=(n,n));data=geometry_features(field['centers'],graph)
        sem=known_semantics(np.load(B/'closed_vocabulary_semantics_v1/aggregate'/key/'source.npz'));owner=sem['owners'];active=(owner>0)|((owner==0)&(sem['seen']>0)&~np.isin(sem['mean_response'],[2,13]));ids=np.flatnonzero(active);mapping=np.load(B/'full_lidar_surface_inputs_v1'/key/'old_to_full.npy')
        labels=pure_anchor_labels(n,field['source_to_cell'][mapping[ids]],owner[ids]);a,b=labels[data['left']],labels[data['right']];known=(a>=0)&(b>=0);y=np.where(known,(a==b).astype(np.int8),-1)
        kind=np.full(len(y),-1,np.int8);kind[known&(a==0)&(b==0)]=0;kind[known&(a>0)&(a==b)]=1;kind[known&(a>0)&(b>0)&(a!=b)]=2;kind[known&((a==0)^(b==0))]=3
        assert ((kind>=0)==known).all();folder=OUT/'features'/key;folder.mkdir(parents=True,exist_ok=False);f=folder/'edges.npz';np.savez_compressed(f,**data,node_anchor_owner=labels,target=y,pair_kind=kind);files[str(f)]=sha(f)
        stats[key]=dict(nodes=n,edges=len(y),known_edges=int(known.sum()),positive=int((y==1).sum()),negative=int((y==0).sum()),unknown_edges=int((y<0).sum()),pair_types={k:int((kind==i).sum()) for i,k in enumerate(['same_background','same_foreground','different_foreground','foreground_background'])})
        assert stats[key]['negative']>=256;print('EDGE FEATURES',key,stats[key],flush=True)
    dump(OUT/'features_complete.json',dict(complete=True,files=files,stats=stats))


def network(dimensions):
    import torch
    return torch.nn.Sequential(torch.nn.Linear(dimensions,16),torch.nn.ReLU(),torch.nn.Linear(16,16),torch.nn.ReLU(),torch.nn.Linear(16,1))


def train():
    import torch
    from sklearn.metrics import roc_auc_score,average_precision_score
    verify();seal=read(OUT/'features_complete.json');assert seal['complete'] and not (OUT/'summary.json').exists()
    for f,h in seal['files'].items():assert sha(f)==h,f
    torch.set_num_threads(1);files={};results={};passed=True
    for testkey in KEYS:
        trainkey=next(k for k in KEYS if k!=testkey);train=np.load(OUT/'features'/trainkey/'edges.npz');test=np.load(OUT/'features'/testkey/'edges.npz');target=train['target'];rng=np.random.RandomState(0);count=min(50000,int((target==0).sum()),int((target==1).sum()));selected=np.r_[rng.choice(np.flatnonzero(target==0),count,replace=False),rng.choice(np.flatnonzero(target==1),count,replace=False)];results[testkey]={}
        testids=np.flatnonzero(test['target']>=0);truth=test['target'][testids];kind=test['pair_kind'][testids];test_features=test['features']
        for name,dim in [('distance_only',1),('surface_geometry',len(FEATURES))]:
            start=time.monotonic();torch.manual_seed(0);model=network(dim);x=train['features'][selected,:dim].copy();mean=x.mean(0);std=np.maximum(x.std(0),1e-6);x=torch.from_numpy((x-mean)/std);y=torch.tensor(target[selected],dtype=torch.float32);optimizer=torch.optim.Adam(model.parameters(),lr=.001);curve=[]
            for epoch in range(80):
                model.train();order=torch.randperm(len(x));total=0.
                for lo in range(0,len(x),4096):
                    ix=order[lo:lo+4096];optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x[ix]).squeeze(1),y[ix]);loss.backward();optimizer.step();total+=float(loss.detach())*len(ix)
                curve.append(total/len(x))
            model.eval();outputs=[]
            with torch.inference_mode():
                for lo in range(0,len(testids),16384):
                    xx=(test_features[testids[lo:lo+16384],:dim]-mean)/std;outputs.append(model(torch.tensor(xx)).squeeze(1).sigmoid().numpy())
            prediction=np.concatenate(outputs);per_type={}
            for i,label in enumerate(['same_background','same_foreground','different_foreground','foreground_background']):
                use=kind==i;per_type[label]=dict(edges=int(use.sum()),mean_affinity=float(prediction[use].mean()) if use.any() else None,predicted_same_fraction=float((prediction[use]>=.5).mean()) if use.any() else None)
            same_fg=kind==1;stats=dict(training_window=trainkey,held_out_window=testkey,train_edges=len(selected),test_edges=len(testids),auroc=float(roc_auc_score(truth,prediction)),positive_average_precision=float(average_precision_score(truth,prediction)),foreground_false_negative_rate=float((prediction[same_fg]<.5).mean()),pair_types=per_type,curve=curve,seconds=time.monotonic()-start)
            folder=OUT/'models'/testkey/name;folder.mkdir(parents=True,exist_ok=False);torch.save(dict(state=model.state_dict(),dimensions=dim,mean=mean,std=std,training=[trainkey],feature_names=list(FEATURES[:dim]),selected_training_edge_ids=selected),folder/'model.pt');np.savez_compressed(folder/'heldout_predictions.npz',edge_ids=testids,affinity=prediction,target=truth,pair_kind=kind);dump(folder/'metrics.json',stats)
            for f in folder.iterdir():files[str(f)]=sha(f)
            results[testkey][name]=stats;print('EDGE SCREEN',testkey,name,{k:v for k,v in stats.items() if k not in ['curve','pair_types']},flush=True)
        a=results[testkey]['surface_geometry'];b=results[testkey]['distance_only'];passed &= a['auroc']>=b['auroc']+.05 and a['foreground_false_negative_rate']<=b['foreground_false_negative_rate']
    dump(OUT/'summary.json',dict(complete=True,results=results,screening_gate=bool(passed),files=files,new_maps=0,reference_labels_read=False,semantic_accuracy_claimed=False,training_fits=4,protocol_sha256=sha(OUT/'protocol.json')))
    print('SURFACE CONTINUATION SCREENING GATE',passed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','train']);globals()[parser.parse_args().action]()
