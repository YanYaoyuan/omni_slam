"""Post-training fixed-state diagnosis; no fitting, labels or new maps."""
import numpy as np
import torch
from paper_final_common import read,dump,sha
from run_persistent_source_memory import OUT,PREPARED,KEYS,verify,seal
from omni_semantic_map.persistent_source_memory import PersistentSourceMemory


def run():
    verify();seal('trained');torch.set_num_threads(1);results={}
    for key in KEYS:
        for mode in ['geometry','no_geometry']:
            cp=torch.load(OUT/'models'/key/mode/'model.pt',map_location='cpu',weights_only=True)
            model=PersistentSourceMemory(**cp['configuration']).eval();model.load_state_dict(cp['state'],strict=True)
            memory=None;rows=[]
            with torch.inference_mode():
                for path in sorted((PREPARED/key).glob('[0-9][0-9][0-9].npz')):
                    data=np.load(path)
                    out=model(torch.tensor(data['context'].astype('float32')),torch.tensor(data['frame_ids']),
                              torch.tensor(data['relations']),memory=memory,use_geometry=mode=='geometry')
                    memory=out['memory'];q=torch.tensor(data['query'].astype('float32'))
                    logits=model.decode_sources(q,memory)
                    without_existence=logits.argmax(1).numpy()
                    foreground_spread=float((logits[:,1:].max(1).values-logits[:,1:].min(1).values).mean())
                    logits[:,1:]+=torch.nn.functional.logsigmoid(out['objectness'])[None]
                    owner=logits.argmax(1).numpy()
                    unit=torch.nn.functional.normalize(memory,dim=1);cos=(unit@unit.T).numpy()
                    off=cos[~np.eye(len(cos),dtype=bool)]
                    singular=torch.linalg.svdvals(memory).numpy();energy=singular**2
                    rows.append(dict(frame_index=int(path.stem),slot_cosine_mean=float(off.mean()),slot_cosine_min=float(off.min()),
                        slot_pairs_cosine_above_0999=float((off>.999).mean()),
                        first_singular_energy_fraction=float(energy[0]/energy.sum()),
                        mean_foreground_logit_spread=foreground_spread,
                        mask_argmax_foreground_fraction=float((without_existence>0).mean()),
                        mask_argmax_foreground_slots=np.unique(without_existence[without_existence>0]).tolist(),
                        foreground_query_fraction=float((owner>0).mean()),
                        winning_foreground_slots=np.unique(owner[owner>0]).tolist(),
                        objectness_min=float(out['objectness'].sigmoid().min()),objectness_max=float(out['objectness'].sigmoid().max())))
            results[key+'/'+mode]=dict(training=cp['training'],relation_bias=model.relation_bias.detach().tolist(),frames=rows,
                final=rows[-1],model_sha256=sha(OUT/'models'/key/mode/'model.pt'))
            print('FIXED MEMORY DIAGNOSIS',key,mode,rows[-1],flush=True)
    dump(OUT/'state_diagnosis.json',dict(complete=True,results=results,new_training=0,new_maps=0,gt_read=False,
        trained_seal_sha256=sha(OUT/'trained.json'),source_sha256=sha(__file__)))


if __name__=='__main__':run()
