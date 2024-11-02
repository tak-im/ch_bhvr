from ch_bhvr.simple_user_sim import Intervention, UserContext, ObservedUserBehavior
from ch_bhvr import util
from typing import List, Dict, Tuple
import numpy as np

class RERec():

    _is_no_intervation_utility_sample: bool = False

    _len_intervention: int
    _len_behavior: int

    # alpha, beta, by context, intervention, behabior
    _params: Dict[Tuple[int, int, int], Tuple[int, int]] = {}

    # hashmap
    _count_ci: Dict[Tuple[int, int], int] = {}# (context, intervention) -> count
    _count_cib: Dict[Tuple[int, int, int], int] = {} # (context, intervention, behavior) -> count

    def __init__(self, len_intervention, len_behavior, is_no_intervention_utility_sample=False):
        self._len_intervention = len_intervention
        self._len_behavior = len_behavior
        self._is_no_intervation_utility_sample = is_no_intervention_utility_sample

    def set_params(self, params: Dict[Tuple[int, int], Tuple[int,int]]=None, 
                   count_ci: Dict[Tuple[int, int], int]=None, 
                   count_cib: Dict[Tuple[int, int, int], int]=None):
        if params is not None:
            self._params = params
        if count_ci is not None:
            self._count_ci = count_ci
        if count_cib is not None:
            self._count_cib = count_cib

    def select_intervention(self, context: UserContext) -> Intervention:
        base_util = self._sample_utility(context.context, 0)
        print(base_util)
        base_util_np: np.array = np.array(base_util, dtype=float)
        base_util_np = base_util_np / base_util_np.sum()

        scores: np.ndarray = np.zeros(self._len_intervention)
        for iv in range(self._len_intervention):
            re = self._sample_re(context.context, iv, base_util_np)
            scores[iv] = re

        selected: int = np.argmax(scores)
        intervention: Intervention = Intervention()
        intervention.intervention = selected
        return intervention

    def _sample_re(self, context: int, intervention: int, base_utilities_np: np.ndarray) -> float:
        utilities: List[float] = self._sample_utility(context, intervention)
        utilities_np = np.array(utilities)
        utilities_np = utilities_np / utilities_np.sum()

        #re = np.dot(utilities_np, np.log2(utilities_np / base_utilities_np))
        re = util.relative_entropy(utilities_np, base_utilities_np)

        return re

    def _sample_utility(self, context: int, intervention: int) -> List[float]: # by behavior
        utilities: List[float] = [0] * self._len_behavior
        count_ci: int = self._count_ci.get((context, intervention), 0)

        # intervention 0: no intervention
        if intervention == 0 and self._is_no_intervation_utility_sample==False:
            for bhvr in range(self._len_behavior):
                count_cib: int = self._count_cib.get((context, intervention, bhvr), 0)
                alpha, beta = self._alpha_beta((context, intervention, bhvr))
                utilities[bhvr] = float(alpha + count_cib) / float(alpha + beta + count_ci)
        else:
            for bhvr in range(self._len_behavior):
                count_cib: int = self._count_cib.get((context, intervention, bhvr), 0)
                alpha, beta = self._alpha_beta((context, intervention, bhvr))
                #print("alpha: ", alpha, ", beta: ", beta, ", count_ci: ", count_ci, "count_cib: ", count_cib)
                utilities[bhvr] = np.random.beta(alpha + count_cib, beta + count_ci - count_cib)

        return utilities

    def _alpha_beta(self, cib: Tuple[int, int, int]) -> Tuple[int, int]:
        a_b: Tuple[int, int] = self._params.get(cib, (1, 1))
        return a_b

    def update(self, context: UserContext, intervention: Intervention, observed: ObservedUserBehavior) -> None:
        ci = (context.context, intervention.intervention)
        observedBhaviors: List[int] = observed.behaviors
        for bhvr in observedBhaviors:
            self._count_ci[ci] = self._count_ci.get(ci, 0) + 1
            cib = (context.context, intervention.intervention, bhvr)
            self._count_cib[cib] = self._count_cib.get(cib, 0) + 1
