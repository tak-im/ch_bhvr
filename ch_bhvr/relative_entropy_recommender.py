from ch_bhvr.simple_user_sim import Intervention
from types import List, Dict, Tuple
import numpy as np

class RERec():

    _is_no_intervation_utility_sample: bool = False

    _len_intervention: int
    _len_behavior: int

    # alpha, beta, by intervention, behabior
    _params: Dict[Tuple[int, int], Tuple[int, int]] = {}

    # hashmap
    _count_intervention: Dict[int, int] = {}# intervention -> count
    _count_behavior_by_intervention: Dict[Tuple[int, int], int] = {} # (intervention, behavior) -> count

    def __init__(self, len_intervention, len_behavior, is_no_intervention_utility_sample=False):
        self._len_intervention = len_intervention
        self._len_behavior = len_behavior
        self._is_no_intervation_utility_sample = is_no_intervention_utility_sample

    def set_params(self, params:Dict[Tuple[int, int], Tuple[int,int]]=None, count_intervention=None, count_behavior=None):
        if params is not None:
            self._params = params
        if count_intervention is not None:
            self._count_intervention = count_intervention
        if count_behavior is not None:
            self._count_behavior_by_intervention = count_behavior

    def select_intervention(self) -> Intervention:
        base_util = self._sample_utility(0)
        base_util_np = np.ndarray(base_util)
        base_util_np = base_util_np / base_util_np.sum()

        scores: np.ndarray = np.zeros(self._len_intervention)
        for iv in range(self._count_intervention):
            re = self._sample_re(iv, base_util_np)
            scores[iv] = re

        selected: int = np.argmax(scores)
        return selected

    def _sample_re(self, intervention: int, base_utilities_np: np.ndarray) -> float:
        utilities: List[float] = self._sample_utility(intervention)
        utilities_np = np.ndarray(utilities)
        utilities_np = utilities_np / utilities_np.sum()

        re = np.dot(utilities_np, np.log2(utilities_np / base_utilities_np))

        return re


    def _sample_utility(self, intervention: int) -> List[float]: # by behavior
        utilities: List[float] = [0] + self._len_behavior
        count_intervention: int = self._count_intervention.get(intervention, 0)

        # intervention 0: no intervention
        if intervention == 0 and self._is_no_intervation_utility_sample==False:
            for bhvr in range(len(self._len_behavior)):
                count_bhvr_ib: int = self._count_behavior_by_intervention.get((intervention, bhvr), 0)
                alpha, beta = self._alpha_beta((intervention, bhvr))
                utilities[bhvr] = float(alpha + count_bhvr_ib) / float(alpha + beta + count_intervention)
        else:
            for bhvr in range(len(self._len_behavior)):
                count_bhvr_ib: int = self._count_behavior_by_intervention.get((intervention, bhvr), 0)
                alpha, beta = self._alpha_beta((intervention, bhvr))
                utilities[bhvr] = np.random.beta(alpha + count_bhvr_ib, beta + count_intervention - count_bhvr_ib)

        return utilities

    def _alpha_beta(self, ib: Tuple[int, int]) -> Tuple[int, int]:
        a_b: Tuple[int, int] = self._params.get(ib, (1, 1))
        return a_b

    def update(self, intervention: int, observations: List[int]) -> None:
        self._count_intervention[intervention] = self._count_intervention.get(intervention, 0) + 1
        for bhvr in observations:
            self._count_behavior_by_intervention[(intervention, bhvr)] = self._count_behavior_by_intervention.get((intervention, bhvr), 0)
