"""Instance matching and class correctness on a fixed annotated voxel domain."""
import numpy as np
from .full_ovo_evaluation import voxel_metrics


def semantic_metrics(truth,prediction,gt_classes,pred_classes):
    truth,prediction=np.asarray(truth),np.asarray(prediction)
    gids=set(map(int,np.unique(truth[truth>0])))
    if not gids<=set(gt_classes):raise ValueError('GT class missing on evaluated domain')
    instance=voxel_metrics(truth,prediction)
    matches=instance['matches']
    correct=[m for m in matches if pred_classes.get(m['predicted_id'],-1)==gt_classes[m['gt_id']]]
    ng=instance['gt_visible_instances'];npred=instance['predicted_instances'];tp=len(correct)
    denominator=.5*(ng+npred);sum_iou=sum(m['iou'] for m in correct)
    gclass=np.array([gt_classes.get(int(i),0) for i in truth])
    pclass=np.array([pred_classes.get(int(i),-1) if i>0 else 0 for i in prediction])
    fg=truth>0
    return dict(instance_pq=instance['observed_voxel_instance_pq'],class_aware_pq=sum_iou/denominator if denominator else 0.,
        class_aware_tp=tp,class_aware_fp=npred-tp,class_aware_fn=ng-tp,
        geometry_matches=len(matches),correct_class_matches=tp,
        matched_class_accuracy=tp/len(matches) if matches else None,
        foreground_voxel_class_accuracy=float((gclass[fg]==pclass[fg]).mean()) if fg.any() else None,
        unknown_predicted_classes=sum(pred_classes.get(int(i),-1)<0 for i in np.unique(prediction[prediction>0])),
        matches=[dict(**m,gt_class=gt_classes[m['gt_id']],pred_class=pred_classes.get(m['predicted_id'],-1),
                      class_correct=m in correct) for m in matches])
