"""Post-hoc certificate that convex reweighting cannot fix selected errors."""
import json
import numpy as np
from audit_evidence_retention import CACHE, OUT, REPORT, dump
from omni_semantic_map.full_ovo_evaluation import BACKGROUND


def main():
    folder=REPORT/'stage1';results=[]
    for scene in json.loads((folder/'summary.json').read_text())['scenes']:
        data=json.loads((folder/(scene+'.json')).read_text());rows={r['object_id']:r for r in data['objects']}
        with np.load(OUT/'observations'/scene/'metadata.npz') as arrays:features=arrays['features']
        text=np.load(CACHE/scene/'text_features.npy');text/=np.linalg.norm(text,axis=1,keepdims=True)
        vocab=json.loads((CACHE/scene/'vocabulary.json').read_text());background=np.isin(vocab['replica_class_ids'],BACKGROUND)
        for loss in data['filter_losses']:
            if loss['cause']!='direct_background_filter':continue
            row=rows[loss['object_id']];scores=features[row['selected_observation_ids']]@text.T
            dominators=[int(i) for i in np.flatnonzero(background) if np.all(scores[:,i]>scores[:,~background].max(1))]
            results.append(dict(scene=scene,object_id=row['object_id'],common_background_dominators=[vocab['names'][i] for i in dominators],
                                cannot_be_recovered_by_positive_weighting=bool(dominators)))
    dump(folder/'convex_readout_limits.json',dict(objects=results,blocked_objects=sum(r['cannot_be_recovered_by_positive_weighting'] for r in results),
        scope='A single fixed background vector outranks every foreground vector in every selected input descriptor. Any nonnegative weighted average with nonzero total weight remains background.'))


if __name__=='__main__':main()
