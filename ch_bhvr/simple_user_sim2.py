from ch_bhvr.interface import IContext, IIntervention, IObservedBehavior, IRecords, IUserBehaviorSimulator
from typing import List, Tuple
import numpy as np
from ch_bhvr import util

BEHAVIOR_COST: float = 0.01

class UserContext(IContext):
    context: int

class ObservedUserBehavior(IObservedBehavior):
    behaviors: List[int]

class Intervention(IIntervention):
    intervention: int # 0:no intervention

class SimpleUserSimParams():

    # probrem size
    context_size: int
    behavior_size: int
    intervention_size: int
    behavior_observation_size: int

    # context
    prob_context: List[float]

    # user param (by context, behabior)
    true_utility: List[List[float]]
    base_utility: List[List[float]]
    utility_std: List[List[float]]
    utility_decay: List[List[List[float]]]

    # user behabior understanding (by context, behabior)
    understanding: List[List[float]]

    # intervention
    # (by context, intervention)
    prob_accept_interventions: List[List[float]]
    # (by intervention, behavior)
    effect_permanent_understanding: List[List[float]]
    effect_temporal_understanding: List[List[float]]


class Record():
    index: int
    context: int
    intervention: int
    tmp_true_utility: List[float]
    tmp_base_utility: List[float]
    tmp_understanding: List[float]
    prm_understanding: List[float]
    perceived_utility: List[float]
    recognition_re: float
    observed_behaviors: List[int]
    max_utility: float
    gained_utility: float
    utility_error: float

    

class Records(IRecords):
    params: SimpleUserSimParams
    records: List[Record]

class SimpleUserSimulator(IUserBehaviorSimulator):

    # initial param
    _init_params: SimpleUserSimParams = None

    # probrem size
    _context_size: int = 1
    _user_behavior_size: int = 5
    _intervention_size: int = 5
    _behavior_observation_size = 100

    # context
    _prob_context: List[float] = None

    # user param (by context, behabior)
    _true_utility : List[List[float]] = None # if understand 100%
    _base_utility : List[List[float]] = None # if understand 0%
    _utility_std: List[List[float]] = None
    _utility_decay: List[List[float]] = None

    # user behabior understanding (by context, behabior)
    _understanding: List[List[float]] = None  

    # intervention (by context, intervention)
    _prob_accept_interventions: List[List[float]] = None

    # intervention effect (by intervention, behavior)
    _effect_permanent_understanding: List[List[float]] = None
    _effect_temporal_understanding: List[List[float]] = None

    #temporal_state
    _current_context: UserContext = None
    _temporal_true_utility: np.ndarray = None
    _temporal_base_utility: np.ndarray = None
    _temporal_understanding: np.ndarray = None

    #rng
    _rng: np.random.Generator = None

    # records
    _index: int = 0
    _records: Records = None
    _current_record: Record = None

    # replay_records
    _replay_records: Records = None

    def __init__(self, params: SimpleUserSimParams):
        self._init_params = params
        self._records = Records()
        self._records.params = params
        self._records.records = []
        self._rng = np.random.default_rng()
        self.init_state()

    def set_replay_records(self, records: Records):
        self._replay_records = records
        self._init_params = records.params
        self.init_state()

    def init_state(self) -> None:
        params: SimpleUserSimParams = self._init_params

        self._context_size = params.context_size
        self._behavior_size = params.behavior_size
        self._intervention_size = params.intervention_size
        self._behavior_observation_size = params.behavior_observation_size

        self._prob_context = params.prob_context      

        self._true_utility  = params.true_utility 
        self._base_utility  = params.base_utility 
        self._utility_std = params.utility_std
        self._utility_decay = params.utility_decay

        self._understanding = params.understanding

        self._prob_accept_interventions = params.prob_accept_interventions
        self._effect_permanent_understanding = params.effect_permanent_understanding
        self._effect_temporal_understanding = params.effect_temporal_understanding

        self._index = 0

    def next_step(self) -> UserContext:
        context: UserContext = UserContext()

        if self._replay_records is None:
            # sample context
            context_index: int = np.random.choice(a=range(self._context_size), p=self._prob_context)
            context.context = context_index
            self._current_context = context

            # reset temp values
            self._temporal_understanding = np.array(self._understanding[context_index])

            # renew temporal state
            true_utility: List[float] = self._true_utility[context_index]
            base_utility: List[float] = self._base_utility[context_index]
            utility_std: List[float] = self._utility_std[context_index]
            self._temporal_true_utility = np.random.normal(true_utility , utility_std)
            self._temporal_true_utility  = self._temporal_true_utility.clip(util.MIN_PROB, util.MAX_PROB)
            self._temporal_base_utility = np.random.normal(base_utility , utility_std)
            self._temporal_base_utility  = self._temporal_base_utility.clip(util.MIN_PROB, util.MAX_PROB)

            #record
            self._append_record()
            self._current_record = Record()
            self._current_record.index = self._index
            self._current_record.context = context_index
            self._current_record.tmp_true_utility = self._temporal_true_utility.copy()
            self._current_record.tmp_base_utility = self._temporal_base_utility.copy()

        else:
            record: Record = self._replay_records[self._index]

            # sample context
            context_index = record.context
            context.context = context_index
            self._current_context = context

            # reset temp values
            self._temporal_understanding = np.array(self._understanding[context_index])

            # renew temporal state
            self._temporal_true_utility  = record.tmp_true_utility.copy()
            self._temporal_base_utility  = record.tmp_base_utility.corpy()

            #record
            self._append_record()
            self._current_record = Record()
            self._current_record.index = self._index
            self._current_record.context = context_index
            self._current_record.tmp_true_utility = self._temporal_true_utility.copy()
            self._current_record.tmp_base_utility = self._temporal_base_utility.copy()

        # index
        self._index += 1

        return context

    def _append_record(self):
        if self._current_record is not None:
            self._records.records.append(self._current_record)
            self._current_record = None

    def interaction(self, intervention: Intervention) -> ObservedUserBehavior:
        intervention_idx: int = intervention.intervention
        context_idx: int = self._current_context.context
        accept_rate: float = self._prob_accept_interventions[context_idx][intervention_idx]
        tmp_effect: List[float] = self._effect_temporal_understanding[intervention_idx]

        if self._rng.uniform(0.0, 1.0) < accept_rate:
            self._temporal_understanding += tmp_effect
            self._temporal_understanding = self._temporal_understanding.clip(util.MIN_PROB, util.MAX_PROB)
            # TODO: parmanent effect
        
        behavior: ObservedUserBehavior = self._generate_user_behavior()

        # record
        self._current_record.intervention = intervention_idx
        self._current_record.tmp_understanding = self._temporal_understanding.copy()
        self._current_record.prm_understanding = self._understanding.copy()
        self._current_record.observed_behaviors = behavior.behaviors.copy()

        return behavior


    def _generate_user_behavior(self) -> ObservedUserBehavior:
        behavior: ObservedUserBehavior = ObservedUserBehavior()
        observed_behaviors: List[int] = [0] * self._behavior_observation_size

        utility_decay: np.ndarray = np.array(self._utility_decay[self._current_context.context])
        true_utilities: np.ndarray = self._temporal_true_utility

        # user perception of behabior utilitys
        perceived_utilities: np.ndarray = self._temporal_true_utility * self._temporal_understanding + self._temporal_base_utility * (1.0 - self._temporal_understanding)
        #print("true utility: ", self._temporal_true_utility)
        #print("base utility: ", self._temporal_base_utility)
        #print("understanding: ", self._temporal_understanding)
        #print("perceived utility: ", perceived_utilities)


        # true
        true_history, max_utility = self._calc_util_2_behavior(true_utilities, utility_decay)
        """
        true_history: np.ndarray = np.zeros(self._behavior_size)
        max_utility: float = 0.0
        for trial in range(self._behavior_observation_size):
            decayed_true: np.ndarray = self._decayed_utility(true_utilities, utility_decay, true_history)
            true_behabior: int = np.argmax(decayed_true)
            true_history[true_behabior] += 1
            max_utility += decayed_true[true_behabior]
        """

        # observed
        behavior_history: np.ndarray = np.zeros(self._behavior_size)
        gained_utility: float = 0.0
        for trial in range(self._behavior_observation_size):
            decayed_true: np.ndarray = self._decayed_utility(true_utilities, utility_decay, behavior_history)
            decayed_perceived: np.ndarray = self._decayed_utility(perceived_utilities, utility_decay, behavior_history)

            # choice
            p_perceived: np.ndarray = decayed_perceived.clip(util.MIN_PROB, util.MAX_PROB)
            p_perceived = p_perceived / p_perceived.sum()
            observed_behavior: int = np.random.choice(a=self._behavior_size, p=p_perceived)
            behavior_history[observed_behavior] += 1
            observed_behaviors[trial] = observed_behavior
            gained_utility += decayed_true[observed_behavior]

        p = self._temporal_true_utility / self._temporal_true_utility.sum()
        q = perceived_utilities / perceived_utilities.sum()
        re: float = util.relative_entropy(p,q)

        # record
        self._current_record.perceived_utility = perceived_utilities
        self._current_record.recognition_re = re
        self._current_record.max_utility = max_utility
        self._current_record.gained_utility = gained_utility
        self._current_record.utility_error = max_utility - gained_utility

        behavior.behaviors = observed_behaviors

        return behavior

    def _calc_util_2_behavior(self, utility: np.ndarray, decay: np.ndarray) -> Tuple[np.ndarray, float]:
        # 確率的に選択しない場合の行動分布を計算
        history: np.ndarray = np.zeros(self._behavior_size)
        utility_max: float = 0.0
        for trial in range(self._behavior_observation_size):
            decayed_util: np.ndarray = self._decayed_utility(utility, decay, history)
            behabior: int = np.argmax(decayed_util)
            history[behabior] += 1
            utility_max += decayed_util[behabior]
        return history, utility_max


    def _decayed_utility(self, org_util: np.ndarray, decay_param: np.ndarray, history: np.ndarray) -> np.ndarray:
        decayed:  np.ndarray = np.zeros(self._behavior_size)
        decay_factor = np.exp(-decay_param * history)
        decayed = org_util * decay_factor
        #print("history: ", history)
        #print("org_util: ", org_util)
        #print("decayed: ", decayed)
        return decayed

    def get_current_record(self) -> Record:
        return self._current_record

    def get_records(self) -> Records:
        self._append_record()
        return self._records
    
    def print_internal_info(self):

        params: SimpleUserSimParams = self._init_params
        understanding_cb: np.ndarray = np.array(params.understanding)        
        understanding_cib: np.ndarray = np.repeat(understanding_cb[:, np.newaxis, :], params.intervention_size, axis=1)
        
        effect_ib: np.ndarray = np.array(params.effect_temporal_understanding)
        effect_cib: np.ndarray = np.tile(effect_ib, (params.context_size, 1, 1))

        intervened_understanding_cib: np.ndarray = understanding_cib + effect_cib
        intervened_understanding_cib.clip(util.MIN_PROB, util.MAX_PROB)

        print("=============== internal infomation =============")
        #print(effect_ib)
        #print(effect_cib)
        print("understanding cib")
        print(intervened_understanding_cib)

        true_utility_cb: np.ndarray = np.array(params.true_utility)
        true_utility_cib: np.ndarray = np.repeat(true_utility_cb[:,np.newaxis,:], params.intervention_size, axis=1)
        base_utility_cb: np.ndarray = np.array(params.base_utility)
        base_utility_cib: np.ndarray = np.repeat(base_utility_cb[:,np.newaxis,:], params.intervention_size, axis=1)
        utility_cib: np.ndarray = true_utility_cib * intervened_understanding_cib + base_utility_cib * (1.0 - intervened_understanding_cib)

        print("true utility cb")
        print(true_utility_cb)
        print("base utility cb")
        print(base_utility_cb)

        print("utility cib")
        print(utility_cib)

        for c in range(params.context_size):
            print("context: ", c)
            true_util: np.ndarray = true_utility_cb[c]
            decay: np.ndarray = np.array(self._utility_decay[c])
            true_behavior, max_util = self._calc_util_2_behavior(true_util, decay)
            print("true utility: ", true_util, ", behabior: ", true_behavior, ", total_utility", max_util)

            no_iv_util: np.ndarray = utility_cib[c, 0]
            no_iv_behavior, no_iv_total_util = self._calc_util_2_behavior(no_iv_util, decay)

            for i in range(params.intervention_size):
                print("intervention: ", i)
                utility: np.ndarray = utility_cib[c, i]
                behavior_i, total_util_i = self._calc_util_2_behavior(utility, decay)
                print("utility: ", utility, ", behabior: ", behavior_i, ", total_utility", total_util_i)

                accept_rate: float = params.prob_accept_interventions[c][i]

                p_true_util: np.ndarray = true_util / true_util.sum()
                p_iv_util: np.ndarray = utility / utility.sum()
                p_noiv_util: np.ndarray = no_iv_util / no_iv_util.sum()
                re: float = util.relative_entropy(p_true_util, p_iv_util)
                re_from_base: float = util.relative_entropy(p_iv_util, p_noiv_util)
                print("utility no iv -> iv: ", re_from_base, ", iv -> true: ", re)
                print("utility(exp) no iv -> iv: ", re_from_base * accept_rate, ", iv -> true", re * accept_rate)

                p_true_b: np.ndarray = true_behavior / true_behavior.sum()
                p_i_b: np.ndarray = behavior_i / behavior_i.sum()
                p_noiv_b: np.ndarray = no_iv_behavior / no_iv_behavior.sum()
                re: float = util.relative_entropy(p_true_b, p_i_b)
                re_from_base: float = util.relative_entropy(p_i_b, p_noiv_b)
                print("behavior no iv -> iv: ", re_from_base, ", iv -> true: ", re)
                print("behavior(exp) no iv -> iv: ", re_from_base * accept_rate, ", iv -> true", re * accept_rate)




        #util.relative_entropy(params.true_utility, )
