"""Confidence-weighted KITTI-360 instance evaluation with a pinned AP kernel.

The upstream AP/matching functions are compiled unchanged from a hash-verified
source file. A local NumPy proxy supplies removed scalar aliases. The upstream
annotation helper has a mismatched constructor; the explicit adapter below
implements the confidence-weighted counts expected by the evaluation caller.
This is an adapted local evaluator, not evidence of an official submission.
"""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np

EVALUATOR_SHA256 = 'a0cf25f1019bf4df9ab496aca8b16a9a35793cb25780788393c94ab8703b932e'


class LegacyNumpy:
    def __getattr__(self, name):
        if name == 'float': return float
        if name == 'bool': return bool
        return getattr(np, name)


class WeightedInstance:
    def __init__(self, ids, confidence, instance_id):
        self.instance_id = int(instance_id)
        self.labelId = self.instance_id//1000
        self.vert_count = float(np.sum(confidence[ids == instance_id]))
        self.med_dist = -1
        self.dist_conf = 0.

    def to_dict(self): return dict(vars(self))


def kernel(source):
    path = Path(source)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == EVALUATOR_SHA256, 'Evaluator source changed'
    names = {'getGTInstances', 'assignGt2Preds', 'evaluateMatches', 'computeAverages'}
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {f.name for f in functions} == names
    args = SimpleNamespace(validClassLabels=['building', 'car'], validClassIds=[2, 13],
        id2label={2:'building', 13:'car'}, overlaps=np.append(np.arange(.5, .95, .05), .25),
        minRegionSizes=np.array([100]), distanceThs=np.array([np.inf]), distanceConfs=np.array([-np.inf]))
    def error(message): raise ValueError(message)
    namespace = dict(np=LegacyNumpy(), deepcopy=deepcopy, Annotation3DInstance=WeightedInstance,
                     args=args, printError=error)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return args, namespace


def evaluate(ground_truth_train_instance, confidence, predicted_instance, predictions, source):
    """Evaluate a single common reference domain; predictions indexed by object ID.

    GT IDs use official trainId*1000+instanceWithinClass, with void=0.
    Prediction records contain semantic_id (11 or 26) and score; no GT is used
    for class assignment, confidence ranking or defining the predicted masks.
    """
    truth, weight, predicted = map(np.asarray, [ground_truth_train_instance, confidence, predicted_instance])
    if truth.ndim != 1 or truth.shape != weight.shape or truth.shape != predicted.shape:
        raise ValueError('Reference and prediction shapes differ')
    if not np.issubdtype(truth.dtype, np.integer) or not np.issubdtype(predicted.dtype, np.integer):
        raise ValueError('Integer instance arrays required')
    if not np.isfinite(weight).all() or np.any((weight < 0) | (weight > 1)):
        raise ValueError('Finite confidence in [0,1] required')
    args, ns = kernel(source)
    ns['readGTImage'] = lambda _: (truth, weight.astype(np.float64))
    mapping = {11:2, 26:13}
    info = {}
    for oid, row in sorted(predictions.items()):
        if row['semantic_id'] not in mapping: continue
        if not np.isfinite(row['score']): raise ValueError('Nonfinite prediction score')
        info[len(info)] = dict(labelId=mapping[row['semantic_id']], conf=float(row['score']), mask=predicted == oid)
    gt, pred = ns['assignGt2Preds']('window', info, 'in_memory_reference')
    matches = {'window':dict(groundTruth=gt, prediction=pred)}
    aps = ns['evaluateMatches'](matches, args)
    averages = ns['computeAverages'](aps, args)
    return dict(averages=averages, ap=aps, matches=matches, thresholds=args.overlaps.copy())
