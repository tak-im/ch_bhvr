from ch_bhvr.interface import IContext, IIntervention, IObservedBehavior, IRecords, IUserBehaviorSimulator
from typing import List
import numpy as np
from ch_bhvr import util

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
    recognition_error: float
    observed_behaviors: List[int]
    

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

    # user behabior understanding (by behabior)
    _understanding: List[List[float]] = None  

    # intervention
    _prob_accept_interventions: List[List[float]] = None
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

    def __init__(self, params: SimpleUserSimParams):
        self._init_params = params
        self._records = Records()
        self._records.params = params
        self._records.records = []
        self._rng = np.random.default_rng()
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

        self._understanding = params.understanding

        self._prob_accept_interventions = params.prob_accept_interventions
        self._effect_permanent_understanding = params.effect_permanent_understanding
        self._effect_temporal_understanding = params.effect_temporal_understanding

        self._index = 0

    def next_step(self) -> UserContext:
        # index
        self._index += 1

        # sample context
        context_index: int = np.random.choice(a=range(self._context_size), p=self._prob_context)
        context: UserContext = UserContext()
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
        if self._current_record is not None:
            self._records.records.append(self._current_record)
        self._current_record = Record()
        self._current_record.index = self._index
        self._current_record.context = context_index
        self._current_record.tmp_true_utility = self._temporal_true_utility.copy()
        self._current_record.tmp_base_utility = self._temporal_base_utility.copy()

        return context

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

        # user perception of behabior utilitys
        perceived_utilities: np.ndarray = self._temporal_true_utility * self._temporal_understanding + self._temporal_base_utility * (1.0 - self._temporal_understanding)
        print("true utility: ", self._temporal_true_utility)
        print("base utility: ", self._temporal_base_utility)
        print("understanding: ", self._temporal_understanding)
        print("perceived utility: ", perceived_utilities)

        # user behavior
        perceived_utilities = perceived_utilities.clip(util.MIN_PROB, util.MAX_PROB)

        p = self._temporal_true_utility / self._temporal_true_utility.sum()
        q = perceived_utilities / perceived_utilities.sum()
        re: float = util.relative_entropy(p,q)

        # record
        self._current_record.perceived_utility = perceived_utilities
        self._current_record.recognition_error = re


        #print(perceived_utilities)
        prob_p_u: np.ndarray = perceived_utilities / perceived_utilities.sum()
        #print(prob_p_u)
        observed_behaviors: np.ndarray = np.random.choice(a=self._behavior_size, size=self._behavior_observation_size, p=prob_p_u)
        behavior.behaviors = observed_behaviors


        return behavior
    
    def get_records(self) -> Records:
        return self._records