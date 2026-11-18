"""Geometry evidence relative to an object's observed, partial surface.

The reference contains predictions, not ground truth. Its empirical ranks
remove object-wide feature offsets but do not certify correctness. Excluding
the query's complete physical cell prevents its direct self-reference.
"""
import numpy as np


class ObjectReferenceRanks:
    def __init__(self, cell_size=.2):
        if not np.isfinite(cell_size) or cell_size <= 0:
            raise ValueError('Positive finite cell size required')
        self.cell_size = float(cell_size)
        self.references = {}
        self.dimensions = None

    def fit(self, xyz, owners, features):
        xyz = np.asarray(xyz, float); owners = np.asarray(owners, np.int64)
        features = np.asarray(features, float)
        if xyz.ndim != 2 or xyz.shape[1] != 3 or owners.shape != (len(xyz),) or features.ndim != 2 or len(features) != len(xyz):
            raise ValueError('Aligned point, owner, feature arrays required')
        if not np.isfinite(xyz).all() or not np.isfinite(features).all() or np.any(owners <= 0):
            raise ValueError('Finite evidence with positive predicted IDs required')
        references = {}
        for oid in np.unique(owners):
            take = owners == oid; p = xyz[take]; f = features[take]
            cells = np.floor(p / self.cell_size).astype(np.int64)
            unique, inverse = np.unique(cells, axis=0, return_inverse=True)
            if len(unique) < 2: raise ValueError('At least two reference cells per object required')
            distance = np.sum((p - (cells + .5) * self.cell_size) ** 2, axis=1)
            # Actual measured representative nearest the cell centre; geometry
            # breaks ties, so order and repeated identical inputs do not matter.
            order = np.lexsort((p[:, 2], p[:, 1], p[:, 0], distance, inverse))
            first = np.r_[True, np.diff(inverse[order]) != 0]
            selected = order[first]; values = f[selected].copy()
            references[int(oid)] = dict(cells=unique, values=values,
                sorted=np.sort(values, axis=0),
                lookup={tuple(c): i for i, c in enumerate(unique)})
        self.references = references; self.dimensions = features.shape[1]
        return self

    def transform(self, xyz, owners, features):
        xyz = np.asarray(xyz, float); owners = np.asarray(owners, np.int64)
        features = np.asarray(features, float)
        if xyz.shape != (len(owners), 3) or features.shape != (len(owners), self.dimensions):
            raise ValueError('Aligned fitted feature dimensions required')
        if not np.isfinite(xyz).all() or not np.isfinite(features).all():
            raise ValueError('Finite query evidence required')
        if any(int(o) not in self.references for o in np.unique(owners)):
            raise ValueError('Missing object reference')
        result = np.empty(features.shape, np.float32)
        cells = np.floor(xyz / self.cell_size).astype(np.int64)
        for oid in np.unique(owners):
            rows = np.flatnonzero(owners == oid); ref = self.references[int(oid)]
            exclusion = np.array([ref['lookup'].get(tuple(c), -1) for c in cells[rows]])
            known = exclusion >= 0; count = len(ref['values']) - known.astype(int)
            excluded = ref['values'][np.maximum(exclusion, 0)]
            for j in range(self.dimensions):
                value = features[rows, j]; ordered = ref['sorted'][:, j]
                lower = np.searchsorted(ordered, value, side='left').astype(float)
                upper = np.searchsorted(ordered, value, side='right').astype(float)
                lower -= known & (excluded[:, j] < value)
                upper -= known & (excluded[:, j] <= value)
                result[rows, j] = (lower + upper) / count - 1.
        if not np.isfinite(result).all() or np.any(np.abs(result) > 1 + 1e-6):
            raise AssertionError('Invalid empirical rank')
        return result
