import numpy as np

MIN_PROB: float = 0.0001
MAX_PROB: float = 1.0 - MIN_PROB

def relative_entropy(p: np.ndarray, q: np.ndarray) -> float:
    _p: np.ndarray = p.clip(MIN_PROB, MAX_PROB)
    _q: np.ndarray = q.clip(MIN_PROB, MAX_PROB)
    re: float = np.dot(_p, np.log2(_p/_q))
    return re

def jaccard_like(p: np.ndarray, q: np.ndarray) -> float:
    _p = p[1:]
    _q = q[1:]

    max_set: np.ndarray = np.fmax(_p, _q)
    min_set:np.ndarray  = np.fmin(_p, _q)
    dist: float = 1.0 - min_set.sum() / max_set.sum()
    return dist

def reward(p: np.ndarray, q: np.ndarray) -> float:
    _p: np.ndarray = p.clip(MIN_PROB, MAX_PROB)
    _q: np.ndarray = q.clip(MIN_PROB, MAX_PROB)
    re: float = np.dot(_p, _p - _q)
    return re
