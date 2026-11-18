"""ScanNet AP-kernel adapter and deterministic point-to-mesh transfer.

SceneNN use is class-agnostic, not an official ScanNet/SceneNN benchmark.
The upstream Python 2 evaluator is loaded without executing its CLI. Changes
are Python 3 syntax, NumPy scalar aliases, and an empty-score guard (all
predictions ignored with GT present -> AP zero rather than IndexError).
"""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
from scipy.spatial import cKDTree

UPSTREAM = Path('build/benchmarks/mesh_instance_20261110/upstream')


def transfer_labels(points, labels, vertices, *, mode='nearest_5cm'):
    """Assigned points only; tie order fixed by input; no GT labels consumed."""
    points, labels, vertices = np.asarray(points), np.asarray(labels), np.asarray(vertices)
    if points.shape != (len(labels), 3) or vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError('incompatible point, label or mesh shapes')
    if not np.isfinite(points).all() or not np.isfinite(vertices).all():
        raise ValueError('non-finite coordinates')
    if mode not in ('nearest_5cm', 'nearest_unbounded', 'knn5_unbounded'):
        raise ValueError('unknown transfer mode')
    keep = labels > 0; points, labels = points[keep], labels[keep]
    if not len(points):
        return np.zeros(len(vertices), np.int64), np.full(len(vertices), np.inf)
    tree = cKDTree(points)
    if mode in ('nearest_5cm', 'nearest_unbounded'):
        distance, index = tree.query(vertices, k=1, workers=1)
        result = labels[index].copy()
        if mode == 'nearest_5cm':
            result[distance > .05] = 0
        return result, distance
    k = min(5, len(points))
    distances, indices = tree.query(vertices, k=k, workers=1)
    if k == 1:
        return labels[indices].copy(), distances
    values = labels[indices]
    # At most five neighbors; ties use the lowest identity, like torch.mode.
    count = (values[:, :, None] == values[:, None, :]).sum(axis=2)
    candidates = np.where(count == count.max(axis=1, keepdims=True), values, np.iinfo(np.int64).max)
    return candidates.min(axis=1), distances[:, 0]


def _functions(path, names):
    # lib2to3 only adapts syntax, not algorithmic behavior. The fetched source
    # remains byte-for-byte archived under upstream/ for provenance.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        from lib2to3.refactor import RefactoringTool, get_fixers_from_package
        tool = RefactoringTool(get_fixers_from_package('lib2to3.fixes'))
        source = str(tool.refactor_string(path.read_text(), str(path)))
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    class Compatibility(ast.NodeTransformer):
        def visit_Attribute(self, node):
            if isinstance(node.value, ast.Name) and node.value.id == 'np' and node.attr in ('float', 'bool'):
                return ast.copy_location(ast.Name(id=node.attr, ctx=ast.Load()), node)
            return self.generic_visit(node)
        def visit_If(self, node):
            if ast.unparse(node.test) == 'has_gt and has_pred':
                node.test = ast.parse('has_gt and has_pred and len(y_score) > 0', mode='eval').body
            return self.generic_visit(node)
    return ast.fix_missing_locations(Compatibility().visit(tree))


class ScanNetAP:
    """Unmodified matching/PR integration, with class-agnostic input encoding."""
    def __init__(self, upstream=UPSTREAM, min_region_size=100):
        self.env = dict(np=np, deepcopy=deepcopy, json=json,
                        CLASS_LABELS=['object'], VALID_CLASS_IDS=np.array([1]),
                        ID_TO_LABEL={1:'object'}, LABEL_TO_ID={'object':1},
                        opt=SimpleNamespace(overlaps=np.append(np.arange(.5,.95,.05),.25),
                            min_region_sizes=np.array([min_region_size]),
                            distance_threshes=np.array([np.inf]), distance_confs=np.array([-np.inf])))
        util_tree = _functions(upstream/'util_3d.py', {'Instance','get_instances'})
        exec(compile(util_tree, str(upstream/'util_3d.py'), 'exec'), self.env)
        def fail(message, **kwargs):
            raise ValueError(message)
        self.env['util'] = SimpleNamespace(print_error=fail)
        tree = _functions(upstream/'evaluate_semantic_instance.py',
                          {'evaluate_matches','compute_averages','assign_instances_for_scan'})
        exec(compile(tree, str(upstream/'evaluate_semantic_instance.py'), 'exec'), self.env)

    def match(self, truth, masks, confidences, scene):
        """truth >0 is an instance; <=0 is void, following ScanNet AP rules.

        masks is a sequence of vertex Boolean arrays and may overlap. Geometry
        benchmark projection produces exclusive masks, but AP itself does not.
        """
        truth = np.asarray(truth)
        if truth.ndim != 1 or len(masks) != len(confidences):
            raise ValueError('incompatible GT/masks/confidences')
        gt = np.zeros(len(truth), np.int64)
        positive = np.unique(truth[truth > 0])
        if len(positive) >= 1000:
            raise ValueError('single-class ScanNet encoding allows at most 999 instances')
        for k, original in enumerate(positive, 1):
            gt[truth == original] = 1000 + k
        values = {scene+'/gt':gt}; info = {}
        for i, (mask, confidence) in enumerate(zip(masks, confidences)):
            mask = np.asarray(mask)
            if mask.shape != truth.shape or not np.isfinite(confidence):
                raise ValueError('invalid mask/confidence')
            key = f'{scene}/prediction_{i}'; values[key] = mask
            info[key] = dict(label_id=1, conf=float(confidence))
        self.env['util_3d'] = SimpleNamespace(get_instances=self.env['get_instances'],
             load_ids=lambda key: values[key], read_instance_prediction_file=lambda *args: info)
        gt2pred, pred2gt = self.env['assign_instances_for_scan']('', scene+'/gt', '')
        return dict(gt=gt2pred, pred=pred2gt)

    def score(self, matches):
        ap = self.env['evaluate_matches'](matches)
        # Preserve undefined no-GT values as null in JSON, not a false zero.
        values = ap[0,0]
        safe = lambda x: float(x) if np.isfinite(x) else None
        return dict(ap=safe(values[:-1].mean()), ap50=safe(values[0]), ap25=safe(values[-1]),
                    iou_thresholds=self.env['opt'].overlaps.tolist(),
                    ap_by_iou=[safe(x) for x in values])
