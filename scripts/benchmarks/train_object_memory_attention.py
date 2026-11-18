"""Scene-disjoint supervised pilot: frozen DINO, trained mean vs trained attention.

Six fixed fits (two arms x three seeds). Fixed native candidate histories;
no new maps, end-to-end backbone fitting, hyperparameter search or test selection.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import torch

from probe_object_memory_attention import (ROOT, HISTORY, DINO, read, dump, sha,
                                         verify_hashes, load_scene, signature)
from omni_semantic_map.object_memory_attention import select_memory, read_memory
from omni_semantic_map.learned_object_memory import ObjectMemoryRanker, positive_mass_loss

OUT = ROOT/'build/benchmarks/learned_object_memory_20261112'
REPORT = ROOT/'docs/reports/learned_object_memory_2026-11-12'
OLD = ROOT/'build/benchmarks/object_memory_attention_20261112'
SPLITS = dict(train=['replica_office_2','replica_office_3','replica_office_4',
                     'replica_room_0','replica_room_1','scenenn_021'],
              validation=['replica_room_2'],
              evaluation=['replica_office_0','replica_office_1','scenenn_014','scenenn_015'])
SEEDS=[17,29,43]
MODES=['mean','attention']


def freeze():
    assert not OUT.exists()
    keys=sum(SPLITS.values(),[]); assert len(keys)==len(set(keys))==11
    all_scenes=read(HISTORY/'protocol.json')['scenes']
    scenes=[next(s for s in all_scenes if s['key']==k) for k in keys]
    inputs={str(HISTORY/'protocol.json'):sha(HISTORY/'protocol.json'),
            str(DINO/'protocol.json'):sha(DINO/'protocol.json')}
    references={}
    for s in scenes:
        cache=Path(s['cache']); m=read(cache/'manifest.json');fm=DINO/'features'/s['key']/'manifest.json'
        d=read(fm); assert m['complete'] and m['no_association'] and d['complete']
        assert d['protocol_sha256']==sha(DINO/'protocol.json')
        assert sha(cache/'manifest.json')==s['hashes'][str(cache/'manifest.json')]
        inputs[str(cache/'manifest.json')]=sha(cache/'manifest.json');inputs[str(fm)]=sha(fm)
        for name,h in m['files_sha256'].items():inputs[str(cache/name)]=h
        for r in m['records']:inputs[str(cache/'frames'/r['file'])]=r['sha256']
        for r in d['records']:inputs[str(fm.parent/r['file'])]=r['sha256']
        run=HISTORY/'runs'/s['key']; status=read(run/'run.json');assert status['complete']
        inputs[str(run/'events.json')]=status['artifacts']['events.json']
        references[s['reference']]=s['hashes'][s['reference']]
    verify_hashes(inputs);verify_hashes(references)
    sources=[Path(__file__),ROOT/'scripts/benchmarks/probe_object_memory_attention.py',
             ROOT/'omni_semantic_map/omni_semantic_map/object_memory_attention.py',
             ROOT/'omni_semantic_map/omni_semantic_map/learned_object_memory.py',
             ROOT/'omni_semantic_map/test/test_learned_object_memory.py']
    p=dict(created_epoch=time.time(),splits=SPLITS,scenes=scenes,seeds=SEEDS,modes=MODES,
           source_sha256={str(x):sha(x) for x in sources},input_sha256=inputs,reference_sha256=references,
           optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001,epochs=50,batch_events=16,clip_norm=1.),
           model=dict(input_dim=384,width=64,heads=4,max_views=16),max_fits=6,max_new_maps=0,
           features='Frozen existing DINOv2-small support-pooled 384D descriptors. No backbone fitting.',
           readers='Both learned arms share embedding, V/output projection and pair MLP, same native score, raw max/mean cosine and memory size. Attention adds learned Q/K; mean freezes Q/K and uses uniform weights. Parameter difference reported, not capacity-matched.',
           memory='Actual prior CG candidate members. One largest-area mask per frame, lowest ID tie; if >16 views, take deterministic evenly spaced temporal indices including first/last. Identical for all readers; never GT grouping.',
           supervision='Original support-union identity proxy: nearest valid reference within 5cm, >=5 points, >=70% foreground dominance including background denominator. Queries and ALL candidates must be usable, with at least one same and one different identity. Multi-positive listwise negative log probability of correct candidates. Labels not provided to model.',
           training='Only six train scenes enter gradients. Shuffle each epoch using fixed seed; inverse within-scene count weights equalize scene contributions. Fifty epochs per fit. No augmentation or class-ID embedding.',
           checkpoint='Best validation scene mean listwise NLL, strict lower improvement; epoch zero native-equivalent checkpoint eligible. Tie retains earlier epoch. No evaluation labels or scores read in training. All six fixed fits reported, no best test-seed selection.',
           baselines='Native CG; fixed mean/nearest/attention over same capped DINO memory; supervised mean residual ranker. Native score is supplied to both learned arms, not the appearance-only fixed controls.',
           evaluation='Predict all four held-out-from-current-fitting scenes after all six checkpoints sealed; only then attach proxy GT. Report all events and proxy coverage, primary subset counts, per-scene accuracy, corrected/harmed, each seed and aggregate. One-step native-history reranking, NOT rollout/PQ.',
           known_scope='All eleven scenes have been examined in prior project experiments. Scene-disjoint fitting in this round only; not pristine unseen test or paper generalization evidence.',
           stop='Six fits only. No retuning after evaluation, no new maps or automatic further training. Coverage failure remains a limitation, not relaxed to obtain positive results.')
    dump(OUT/'protocol.json',p)
    for x in sources:
        target=OUT/'sources'/x.name;target.parent.mkdir(exist_ok=True);target.write_bytes(x.read_bytes())
    print('FROZEN',sha(OUT/'protocol.json'),len(inputs),'inputs',flush=True)


def protocol():
    p=read(OUT/'protocol.json');verify_hashes(p['source_sha256']);return p


def labels_for_scene(s, supports, events):
    ref=np.load(s['reference']);xyz=np.load(Path(s['cache'])/'geometry.npz')['xyz']
    valid=ref['valid'] if 'valid' in ref.files else np.ones(len(ref['xyz']),bool)
    dist,idx=cKDTree(ref['xyz']).query(xyz,workers=1)
    supported=(dist<=.05)&valid[idx];projected=ref['instance'][idx]
    memo={}
    def sig(members):
        k=tuple(sorted(set(members)))
        if k not in memo:memo[k]=signature(np.concatenate([supports[i] for i in k]),projected,supported)
        return memo[k]
    output=[]
    for event in events:
        q=sig([event['query']]);cs=[sig(c['members']) for c in event['candidates']]
        resolved=q['usable'] and all(c['usable'] for c in cs)
        positive=[bool(resolved and c['dominant']==q['dominant']) for c in cs]
        output.append(dict(query=event['query'],query_signature=q,candidate_signatures=cs,
                           resolved=resolved,positives=positive,
                           discriminative=bool(resolved and any(positive) and not all(positive))))
    return output


def prepare():
    p=protocol();verify_hashes(p['input_sha256']);assert not (OUT/'data').exists()
    manifests={}; inventory=[]
    for s in p['scenes']:
        key=s['key'];role=next(role for role,keys in p['splits'].items() if key in keys)
        f,frames,areas,supports=load_scene(s,supports=role!='evaluation')
        features=f['dino'].astype(np.float32);norm=np.linalg.norm(features,axis=1)
        valid=np.isfinite(features).all(1)&(norm>1e-12)
        features[valid]/=norm[valid,None]
        raw=read(HISTORY/'runs'/key/'events.json');events=[]
        for e in raw:
            memories=[]
            for c in e['candidates']:
                ids=select_memory(c['members'],e['query'],frames,areas)
                if len(ids)>p['model']['max_views']:
                    ids=[ids[i] for i in np.linspace(0,len(ids)-1,p['model']['max_views'],dtype=int)]
                memories.append(ids)
            supported=bool(valid[e['query']] and all(valid[ids].all() for ids in memories))
            events.append(dict(query=e['query'],frame=e['frame'],memories=memories,supported=supported,
                native=next(i for i,c in enumerate(e['candidates']) if c['is_native']),
                native_scores=[c['native_score'] for c in e['candidates']]))
        folder=OUT/'data'/key;folder.mkdir(parents=True)
        np.savez_compressed(folder/'features.npz',features=features,frames=frames,areas=areas)
        dump(folder/'events.json',events)
        row=dict(scene=key,role=role,events=len(events),supported=sum(e['supported'] for e in events))
        if role!='evaluation':
            assert sha(s['reference'])==p['reference_sha256'][s['reference']]
            labels=labels_for_scene(s,supports,raw)
            dump(folder/'labels.json',labels)
            row['training_eligible']=sum(l['discriminative'] and e['supported'] for l,e in zip(labels,events))
            row['native_wrong_eligible']=sum(l['discriminative'] and e['supported'] and not l['positives'][e['native']] for l,e in zip(labels,events))
        inventory.append(row)
        manifests.update({str(x):sha(x) for x in folder.iterdir()})
        print('PREPARED',row,flush=True)
    assert not any((OUT/'data'/k/'labels.json').exists() for k in SPLITS['evaluation'])
    assert sum(r['training_eligible'] for r in inventory if r['role']=='train')>=20
    assert sum(r['training_eligible'] for r in inventory if r['role']=='validation')>=5
    dump(OUT/'data_ready.json',dict(complete=True,files_sha256=manifests,inventory=inventory,
        evaluation_labels_absent=True,protocol_sha256=sha(OUT/'protocol.json')))


def load_events(keys, labeled):
    result=[]
    for key in keys:
        folder=OUT/'data'/key;f=np.load(folder/'features.npz')['features'];events=read(folder/'events.json')
        labels=read(folder/'labels.json') if labeled else [None]*len(events)
        for i,(e,l) in enumerate(zip(events,labels)):
            if labeled and not(e['supported'] and l['discriminative']):continue
            result.append(dict(scene=key,event_index=i,features=f,event=e,label=l))
    return result


def batch(samples):
    q=[];h=[];native=[];positive=[];offsets=[0]
    for sample in samples:
        f,e=sample['features'],sample['event']
        for ids,score in zip(e['memories'],e['native_scores']):
            q.append(f[e['query']]);h.append(f[ids]);native.append(score)
        offsets.append(len(q))
        if sample['label'] is not None:positive.extend(sample['label']['positives'])
    mem=np.zeros((len(q),16,384),np.float32);valid=np.zeros((len(q),16),bool)
    for i,v in enumerate(h):mem[i,:len(v)]=v;valid[i,:len(v)]=True
    return (torch.from_numpy(np.array(q)),torch.from_numpy(mem),torch.from_numpy(valid),
            torch.tensor(native,dtype=torch.float32),offsets,torch.tensor(positive,dtype=torch.bool))


def measure(model,samples):
    model.eval();losses=[];correct=[]
    with torch.no_grad():
        for i in range(0,len(samples),16):
            ss=samples[i:i+16];q,h,m,n,offsets,y=batch(ss);scores=model(q,h,m,n)
            losses.extend(positive_mass_loss(scores,offsets,y).tolist())
            for sample,a,b in zip(ss,offsets[:-1],offsets[1:]):
                values=scores[a:b].tolist();native=sample['event']['native']
                chosen=max(range(len(values)),key=lambda j:(values[j],j==native,-j))
                correct.append(bool(sample['label']['positives'][chosen]))
    return dict(nll=float(np.mean(losses)),correct=sum(correct),total=len(correct),accuracy=float(np.mean(correct)))


def train():
    p=protocol();ready=read(OUT/'data_ready.json');verify_hashes(ready['files_sha256'])
    assert not (OUT/'fits').exists()
    # Training loader has no access to evaluation samples, targets or scores.
    training=load_events(p['splits']['train'],True);validation=load_events(p['splits']['validation'],True)
    assert {s['scene'] for s in training}.isdisjoint(p['splits']['evaluation'])
    assert {s['scene'] for s in validation}.isdisjoint(p['splits']['evaluation'])
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    counts=Counter(s['scene'] for s in training)
    scene_weights={k:len(training)/(len(counts)*n) for k,n in counts.items()}
    fits=[]
    for seed in p['seeds']:
        for mode in p['modes']:
            folder=OUT/'fits'/f'{mode}_{seed}';folder.mkdir(parents=True)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            model=ObjectMemoryRanker(mode);initial={k:v.detach().clone() for k,v in model.state_dict().items()}
            opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad],lr=p['optimizer']['lr'],weight_decay=p['optimizer']['weight_decay'])
            train0=measure(model,training);val0=measure(model,validation)
            best=val0['nll'];best_epoch=0
            torch.save(model.state_dict(),folder/'best.pt')
            history=[dict(epoch=0,train=train0,validation=val0)]
            order_rng=np.random.default_rng(seed);began=time.monotonic();steps=0;gradmax={name:0. for name in ['query','key','value','embed','score']}
            for epoch in range(1,p['optimizer']['epochs']+1):
                model.train();order=order_rng.permutation(len(training))
                for i in range(0,len(order),p['optimizer']['batch_events']):
                    ss=[training[j] for j in order[i:i+p['optimizer']['batch_events']]]
                    q,h,m,n,offsets,y=batch(ss);opt.zero_grad(set_to_none=True)
                    scores=model(q,h,m,n);losses=positive_mass_loss(scores,offsets,y)
                    weights=torch.tensor([scene_weights[s['scene']] for s in ss])
                    loss=(losses*weights).mean();assert torch.isfinite(loss)
                    loss.backward()
                    for name in gradmax:
                        module=getattr(model,name)
                        gradmax[name]=max(gradmax[name],sum(float(v.grad.abs().sum()) for v in module.parameters() if v.grad is not None))
                    torch.nn.utils.clip_grad_norm_(model.parameters(),p['optimizer']['clip_norm']);opt.step();steps+=1
                tr=measure(model,training);val=measure(model,validation)
                history.append(dict(epoch=epoch,train=tr,validation=val))
                if val['nll']<best:
                    best=val['nll'];best_epoch=epoch;torch.save(model.state_dict(),folder/'best.pt')
                if epoch%10==0:
                    print('TRAIN',mode,seed,epoch,'train',round(tr['nll'],4),'val',round(val['nll'],4),'best',best_epoch,flush=True)
                dump(folder/'progress.json',dict(complete=False,epoch=epoch,best_epoch=best_epoch,steps=steps))
            torch.save(model.state_dict(),folder/'last.pt')
            final_delta={name:sum(float((v-initial[k]).square().sum()) for k,v in model.state_dict().items() if k.startswith(name+'.'))**.5 for name in gradmax}
            assert steps>0 and final_delta['score']>0 and gradmax['score']>0
            if mode=='attention':assert all(final_delta[k]>0 and gradmax[k]>0 for k in ['query','key','value'])
            selected=ObjectMemoryRanker(mode);selected.load_state_dict(torch.load(folder/'best.pt',weights_only=True))
            selected_val=measure(selected,validation);assert abs(selected_val['nll']-best)<1e-7
            result=dict(complete=True,mode=mode,seed=seed,device='cpu',threads=2,steps=steps,
                train_samples=len(training),validation_samples=len(validation),train_scene_counts=dict(counts),
                trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
                best_epoch=best_epoch,selection='validation NLL only; epoch zero eligible',initial_train=train0,initial_validation=val0,
                selected_train=measure(selected,training),selected_validation=selected_val,
                final_train=tr,final_validation=val,history=history,gradient_abs_max=gradmax,final_weight_l2_changes=final_delta,
                checkpoints_sha256={str(folder/n):sha(folder/n) for n in ['best.pt','last.pt']},
                wall_s=time.monotonic()-began,protocol_sha256=sha(OUT/'protocol.json'))
            dump(folder/'run.json',result);dump(folder/'progress.json',dict(complete=True,epoch=50,best_epoch=best_epoch,steps=steps))
            fits.append(dict(path=str(folder/'run.json'),sha256=sha(folder/'run.json'),mode=mode,seed=seed,best_epoch=best_epoch))
    assert len(fits)==p['max_fits']
    dump(OUT/'training_complete.json',dict(complete=True,fits=fits,evaluation_used_in_training=False,
        trained_scenes=p['splits']['train'],validation_scenes=p['splits']['validation'],created_epoch=time.time()))


def predict():
    p=protocol();done=read(OUT/'training_complete.json');assert done['complete']
    assert not (OUT/'predictions').exists()
    ready=read(OUT/'data_ready.json');verify_hashes(ready['files_sha256'])
    torch.set_num_threads(2);models={}
    for fit in done['fits']:
        assert sha(fit['path'])==fit['sha256'];r=read(fit['path']);verify_hashes(r['checkpoints_sha256'])
        name=f"{fit['mode']}_{fit['seed']}";model=ObjectMemoryRanker(fit['mode'])
        model.load_state_dict(torch.load(Path(fit['path']).parent/'best.pt',weights_only=True));model.eval();models[name]=model
    sealed={}
    for key in p['splits']['evaluation']:
        samples=load_events([key],False);rows=[]
        for sample in samples:
            e=sample['event'];native=e['native'];choices={'native':native};score_table={}
            if e['supported']:
                q,h,m,n,offsets,y=batch([sample])
                with torch.no_grad():
                    for name,model in models.items():
                        values=model(q,h,m,n).tolist();score_table[name]=values
                        choices[name]=max(range(len(values)),key=lambda j:(values[j],j==native,-j))
                reads=[read_memory(sample['features'][e['query']],sample['features'][ids],.1) for ids in e['memories']]
                for reader in ['mean','nearest','attention']:
                    name='fixed_'+reader
                    if any(r is None for r in reads):choices[name]=native;score_table[name]=None
                    else:
                        values=[r['scores'][reader] for r in reads];score_table[name]=values
                        choices[name]=max(range(len(values)),key=lambda j:(values[j],j==native,-j))
            else:
                choices.update({name:native for name in list(models)+['fixed_mean','fixed_nearest','fixed_attention']})
            rows.append(dict(query=e['query'],frame=e['frame'],supported=e['supported'],choices=choices,scores=score_table))
        target=OUT/'predictions'/f'{key}.json';dump(target,dict(complete=True,no_gt=True,rows=rows));sealed[str(target)]=sha(target)
        print('PREDICTED',key,len(rows),flush=True)
    dump(OUT/'predictions_complete.json',dict(complete=True,files_sha256=sealed,
        training_complete_sha256=sha(OUT/'training_complete.json'),epoch=time.time(),new_maps=0))


def evaluate():
    p=protocol();seal=read(OUT/'predictions_complete.json');verify_hashes(seal['files_sha256'])
    assert seal['training_complete_sha256']==sha(OUT/'training_complete.json')
    assert not (OUT/'evaluation').exists()
    scenes=[]
    for key in p['splits']['evaluation']:
        s=next(s for s in p['scenes'] if s['key']==key)
        assert sha(s['reference'])==p['reference_sha256'][s['reference']]
        _,_,_,supports=load_scene(s,supports=True);raw=read(HISTORY/'runs'/key/'events.json')
        labels=labels_for_scene(s,supports,raw)
        # Independently reconstructed labels must match the previous immutable probe.
        old=read(OLD/'evaluation'/key/'events.json')
        for label,prior in zip(labels,old):
            assert label['query_signature']==prior['query_signature']
            assert label['candidate_signatures']==prior['candidate_signatures']
        predictions=read(OUT/'predictions'/f'{key}.json')['rows'];assert len(predictions)==len(labels)
        methods={};primary=[i for i,(r,l) in enumerate(zip(predictions,labels)) if r['supported'] and l['discriminative']]
        assert len(primary)==sum(r['primary'] for r in old)
        for name in predictions[0]['choices']:
            good=[bool(labels[i]['positives'][predictions[i]['choices'][name]]) for i in primary]
            transitions={}
            for base in ['native','fixed_mean','fixed_nearest']:
                b=[bool(labels[i]['positives'][predictions[i]['choices'][base]]) for i in primary]
                transitions[base]=dict(gained=sum(x and not y for x,y in zip(good,b)),lost=sum(not x and y for x,y in zip(good,b)))
            methods[name]=dict(correct=sum(good),total=len(good),accuracy100=100*np.mean(good) if good else None,transitions=transitions)
        row=dict(key=key,dataset=s['dataset'],events=len(predictions),primary=len(primary),methods=methods,
                 unresolved=sum(not l['resolved'] for l in labels),
                 same_only=sum(l['resolved'] and all(l['positives']) for l in labels),
                 no_positive=sum(l['resolved'] and not any(l['positives']) for l in labels))
        dump(OUT/'evaluation'/key/'labels.json',labels);dump(OUT/'evaluation'/key/'metrics.json',row);scenes.append(row)
        print('EVALUATED',key,{k:v['correct'] for k,v in methods.items()},flush=True)
    totals={}
    for method in scenes[0]['methods']:
        correct=sum(s['methods'][method]['correct'] for s in scenes);n=sum(s['primary'] for s in scenes)
        totals[method]=dict(correct=correct,total=n,accuracy100=100*correct/n,
             transitions={base:{v:sum(s['methods'][method]['transitions'][base][v] for s in scenes) for v in ['gained','lost']} for base in ['native','fixed_mean','fixed_nearest']})
    paired=[]
    for seed in p['seeds']:
        good=bad=0
        for s in scenes:
            rows=read(OUT/'predictions'/f"{s['key']}.json")['rows'];labels=read(OUT/'evaluation'/s['key']/'labels.json')
            for row,l in zip(rows,labels):
                if not (row['supported'] and l['discriminative']):continue
                a=l['positives'][row['choices'][f'attention_{seed}']];b=l['positives'][row['choices'][f'mean_{seed}']]
                good+=int(a and not b);bad+=int(b and not a)
        paired.append(dict(seed=seed,attention_vs_trained_mean=dict(gained=good,lost=bad)))
    dump(OUT/'summary.json',dict(complete=True,scenes=scenes,totals=totals,paired=paired,
        total_events=sum(s['events'] for s in scenes),primary_events=sum(s['primary'] for s in scenes),
        fits=read(OUT/'training_complete.json')['fits'],trained=True,backbone_frozen=True,new_maps=0,
        checkpoint_selection='validation NLL only, all seeds reported',
        coverage='Same 32 proxy-resolvable discriminative events as prior probe; insufficient to establish generalization or map improvement.',
        decision='Complete the six fixed fits and retain all results. No deployment, parameter sweep or automatic rollout.'))


def audit():
    p=protocol();verify_hashes(p['input_sha256']);verify_hashes(p['reference_sha256'])
    ready=read(OUT/'data_ready.json');verify_hashes(ready['files_sha256'])
    done=read(OUT/'training_complete.json');seal=read(OUT/'predictions_complete.json')
    verify_hashes(seal['files_sha256']);assert len(done['fits'])==6
    assert set(sum(p['splits'].values(),[]))=={s['key'] for s in p['scenes']}
    assert len(sum(p['splits'].values(),[]))==11
    torch.set_num_threads(2)
    validation=load_events(p['splits']['validation'],True)
    for fit in done['fits']:
        assert sha(fit['path'])==fit['sha256'];r=read(fit['path']);verify_hashes(r['checkpoints_sha256'])
        assert r['best_epoch']==min(range(51),key=lambda i:r['history'][i]['validation']['nll'])
        assert r['steps']>0 and r['final_weight_l2_changes']['score']>0
        if fit['mode']=='attention':assert all(r['gradient_abs_max'][k]>0 and r['final_weight_l2_changes'][k]>0 for k in ['query','key','value'])
        model=ObjectMemoryRanker(fit['mode']);model.load_state_dict(torch.load(Path(fit['path']).parent/'best.pt',weights_only=True));model.eval()
        assert measure(model,validation)==r['selected_validation']
        for key in p['splits']['evaluation']:
            rows=read(OUT/'predictions'/f'{key}.json')['rows'];samples=load_events([key],False)
            for row,sample in zip(rows,samples):
                if not row['supported']:continue
                q,h,m,n,_,_=batch([sample])
                with torch.no_grad():scores=model(q,h,m,n).tolist()
                name=f"{fit['mode']}_{fit['seed']}"
                np.testing.assert_allclose(scores,row['scores'][name],rtol=0,atol=0)
                assert row['choices'][name]==max(range(len(scores)),key=lambda j:(scores[j],j==sample['event']['native'],-j))
    for key in p['splits']['evaluation']:
        assert not (OUT/'data'/key/'labels.json').exists()
    for s in p['scenes']:
        cache=OUT/'data'/s['key'];a=np.load(cache/'features.npz');frames=a['frames']
        for event in read(cache/'events.json'):
            for memory in event['memories']:
                assert len(memory)<=16 and len({int(frames[i]) for i in memory})==len(memory)
                assert all(frames[i]<event['frame'] for i in memory)
    summary=read(OUT/'summary.json')
    for scene in summary['scenes']:
        rows=read(OUT/'predictions'/f"{scene['key']}.json")['rows'];labels=read(OUT/'evaluation'/scene['key']/'labels.json')
        for name,metric in scene['methods'].items():
            good=[l['positives'][r['choices'][name]] for r,l in zip(rows,labels) if r['supported'] and l['discriminative']]
            assert sum(good)==metric['correct'] and len(good)==metric['total']
    dump(OUT/'audit.json',dict(complete=True,source_and_inputs_unchanged=True,
        scene_disjoint_gradients_validation_evaluation=True,eval_labels_absent_from_training_data=True,
        six_actual_training_runs=True,attention_qkv_gradients_and_weight_changes=True,
        validation_only_checkpoint_selection_reproduced=True,predictions_reproduced_exactly=True,
        causal_memories_checked=True,identity_proxies_match_previous_independent_audit=True,
        primary_counts_recomputed=True,summary_sha256=sha(OUT/'summary.json')))
    print('AUDIT PASS',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['freeze','prepare','train','predict','evaluate','audit'])
    args=parser.parse_args();globals()[args.action]()
