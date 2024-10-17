from ch_bhvr.interface import IContext, IIntervention, IObservedBehavior, IRecords, IUserBehaviorSimulator
from typing import List
from scipy.stats import norm
import random
import numpy as np

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
    base_utility: List[List[float]]
    utility_std: List[List[float]]

    # user behabior understanding (by behabior)
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
    tmp_utility: List[float]
    tmp_understanding: List[float]
    prm_understanding: List[float]
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
    _base_utility: List[List[float]] = None
    _utility_std: List[List[float]] = None

    # user behabior understanding (by behabior)
    _understanding: List[List[float]] = None  

    # intervention
    _prob_accept_interventions: List[List[float]] = None
    _effect_permanent_understanding: List[List[float]] = None
    _effect_temporal_understanding: List[List[float]] = None

    #temporal_state
    _current_context: UserContext = None
    _temporal_utility: np.ndarray = None
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

        self._base_utility = params.base_utility
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
        base_utility: List[float] = self._base_utility[context_index]
        utility_std: List[float] = self._utility_std[context_index]
        self._temporal_utility = np.random.normal(base_utility, utility_std)
        self._base_utility = self._temporal_utility.clip(0.0, 1.0)

        #record
        if self._current_record is not None:
            self._records.records.append(self._current_record)
        self._current_record = Record()
        self._current_record.index = self._index
        self._current_record.context = context_index
        self._current_record.tmp_utility = self._temporal_utility.copy()

        return context

    def interaction(self, intervention: IIntervention) -> ObservedUserBehavior:
        intervention_idx: int = intervention
        context_idx: int = self._current_context.context
        accept_rate: float = self._prob_accept_interventions[context_idx][intervention_idx]
        tmp_effect: List[float] = self._effect_temporal_understanding[intervention_idx]

        if self._rng.uniform(0.0, 1.0) < accept_rate:
            self._temporal_understanding += tmp_effect
            self._temporal_understanding = self._temporal_understanding.clip(0.0, 0.99)
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
        #perceived_utilities: np.ndarray = self._temporal_utility * self._temporal_understanding
        #print(self._temporal_understanding)
        perceived_utilities: np.ndarray = np.random.normal(self._temporal_understanding, self._temporal_understanding * (self._temporal_understanding * (-1.0) + 1.0))

        # user behavior
        #print(perceived_utilities)
        perceived_utilities = perceived_utilities.clip(0.0, None)
        #print(perceived_utilities)
        prob_p_u: np.ndarray = perceived_utilities / perceived_utilities.sum()
        #print(prob_p_u)
        observed_behaviors: np.ndarray = np.random.choice(a=self._behavior_size, size=self._behavior_observation_size, p=prob_p_u)
        behavior.behaviors = observed_behaviors

        return behavior
    
    def get_records(self) -> Records:
        return self._records