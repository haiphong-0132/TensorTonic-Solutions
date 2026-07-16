import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    try:
        matrix = np.asarray(matrix, dtype=float)

        if matrix.ndim != 2:
            return None
        if axis not in (None, 0, 1):
            return None
        if norm_type not in ('l1', 'l2', 'max'):
            return None

        if norm_type == 'l2':
            norm = np.linalg.norm(matrix, axis=axis, keepdims=True) if axis is not None else np.linalg.norm(matrix)
        elif norm_type == 'l1':
            norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
        else:  # max
            norm = np.max(np.abs(matrix), axis=axis, keepdims=True)

        norm = np.where(norm == 0, 1, norm)
        return matrix / norm

    except:
        return None