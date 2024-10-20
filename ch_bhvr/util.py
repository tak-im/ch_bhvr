import numpy as np

MIN_PROB: float = 0.0001
MAX_PROB: float = 1.0 - MIN_PROB

def relative_entropy(p: np.ndarray, q: np.ndarray) -> float:
    _p: np.ndarray = p.clip(MIN_PROB, MAX_PROB)
    _q: np.ndarray = q.clip(MIN_PROB, MAX_PROB)
    tmp: np.ndarray = _p * np.log(_p/_q)
    re: float = tmp.sum()
    return re
