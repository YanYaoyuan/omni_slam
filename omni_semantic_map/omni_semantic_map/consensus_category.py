"""Category-only update for persistent objects; never decides object deletion.

Development candidate: one largest mask per frame, median cosine centrality,
retain centrality >= its median, then sqrt(area)-weighted SigLIP readout.
No DINO inference, taxonomy expansion, or geometry changes are required.
"""
import numpy as np
from .category_evidence import readout
from .shared_observations import select_views


def classify(members,features,frames,areas,text,vocabulary):
    ids=np.array(select_views(members,frames,areas,top_k=0),dtype=np.int64)
    if not len(ids):raise ValueError('object has no observations')
    # Setting exclusive area to raw area reduces the frozen quality term to
    # sqrt(area). The experiment did not establish an exclusivity benefit.
    out=readout(np.asarray(features)[ids],np.asarray(areas)[ids],np.asarray(areas)[ids],
                text,method='quality_siglip')
    order=np.argsort(-out['scores'],kind='stable');best=int(order[0])
    return dict(name=vocabulary['names'][best],replica_class_id=int(vocabulary['replica_class_ids'][best]),
                score=float(out['scores'][best]),margin=float(out['scores'][best]-out['scores'][order[1]]),
                selected_observation_ids=ids.tolist(),weights=out['weights'].tolist(),
                fallback=out['fallback'],method='consensus_sqrt_area')
