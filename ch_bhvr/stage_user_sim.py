from ch_bhvr.interface import IContext, IIntervention, IObservedBehavior, IRecords, IUserBehaviorSimulator
from typing import List, Tuple
import numpy as np
from ch_bhvr import util

class UserContext(IContext):
    context: int

class ObservedUserBehavior(IObservedBehavior):
    behaviors: np.ndarray # 0: unrelated behavior
    hist: np.ndarray

class Intervention(IIntervention):
    intervention: int # 0:no intervention

class StageUserSimParams():
    # probrem size
    context_size: int
    behavior_size: int
    intervention_size: int
    behavior_observation_size: int

    # context
    prob_context: List[float]

    # user param 
    init_progress: float
    stage_size: int
    stage_distribution: List[List[float]] # (by stage, behabior)

    # intervention
    # (by stage, context, intervention)
    prob_accept_interventions: List[List[List[float]]]
    # (by stage, context, intervention, behavior)
    effect_to_temporal_distribution: List[List[List[List[float]]]]
    #effect_to_temporal_distribution_var: List[List[List[float]]]

    # behavior change
    auto_change_coeff: float
    change_coeff: float

class Record():
    index: int
    context: int
    auto_progress: float

    progress_before_intervention: float
    intervention: int
    progress_after_intervention: float

    observed_behaviors: np.ndarray
    behavior_hist: np.ndarray
   

class Records(IRecords):
    params: StageUserSimParams
    records: List[Record]

class StageUserSimulator(IUserBehaviorSimulator):

    # initial param
    _params: StageUserSimParams = None

    # user param 
    _progress: float
    
    #temporal_state
    _current_context: UserContext = None
    _current_behavior_distribution: np.ndarray # by behavior
    _temporal_behavior_distribution: np.ndarray # by behavior

    #rng
    _rng: np.random.Generator = None

    # records
    _index: int = 0
    _records: Records = None
    _current_record: Record = None

    # replay_records
    _replay_records: Records = None

    def __init__(self, params: StageUserSimParams):
        self._params = params
        self._records = Records()
        self._records.params = params
        self._records.records = []
        self._rng = np.random.default_rng()
        self.init_state()

    def set_replay_records(self, records: Records):
        self._replay_records = records
        self._params = records.params
        self.init_state()

    def init_state(self) -> None:
        params: StageUserSimParams = self._params
        self._progress = params.init_progress
        self._index = 0

    def next_step(self) -> UserContext:
        params: StageUserSimParams = self._params
        context: UserContext = UserContext()

        if self._replay_records is None:
            # sample context
            context_index: int = np.random.choice(a=range(self._context_size), p=params._prob_context)
            context.context = context_index
            self._current_context = context

            # progress(auto)
            auto_progress: float = self._auto_progress()

            # reset temporal vals
            self._current_behavior_distribution = self._calc_behavior_distribution()
            self._temporal_behavior_distribution = self._current_behavior_distribution.copy()

            #record
            self._append_record()
            self._current_record = Record()
            self._current_record.index = self._index
            self._current_record.context = context_index
            self._current_record.auto_progress = auto_progress
            self._current_record.progress_before_intervention = self._progress

        else:
            record: Record = self._replay_records[self._index]

            # sample context
            context_index = record.context
            context.context = context_index
            self._current_context = context

            # reset temp values
            self._temporal_understanding = np.array(self._understanding[context_index])

            # progress(auto)
            auto_progress = record.auto_progress
            self._progress += auto_progress

            # reset temporal vals
            self._current_behavior_distribution = self._calc_behavior_distribution()
            self._temporal_behavior_distribution = self._current_behavior_distribution.copy()

            #record
            self._append_record()
            self._current_record = Record()
            self._current_record.index = self._index
            self._current_record.context = context_index
            self._current_record.auto_progress = auto_progress
            self._current_record.progress_before_intervention = self._progress

        # index
        self._index += 1

        return context

    def _append_record(self):
        if self._current_record is not None:
            self._records.records.append(self._current_record)
            self._current_record = None

    """
    progressの自然変化、変化量を返す（再生用）
    """
    def _auto_progress(self) -> float:
        params: StageUserSimParams = self.params
        return self._rng.uniform(0, 1) * params.auto_change_coeff

    """
    progressから現在のstageと次のステージまでの進捗を計算
    返り値は(現在のステージ、次のステージ、現在のステージから次のステージに向けた進捗割合（0～1）)
    """
    def _progress_detail(self) -> Tuple[int, int, float]:
        params: StageUserSimParams = self._params
        a, b = divmod(self._progress, params.stage_size - 1)
        stage: int = int(a)
        next_stage: int = stage + 1
        local_progress: float = b * (params.stage_size - 1)
        return (stage, next_stage, local_progress)


    """
    progressから現在の行動確率を計算
    """
    def _calc_behavior_distribution(self) -> np.ndarray:
        params: StageUserSimParams = self._params
        stage, next_stage, local_progress = self._progress_detail()
        stage_d: np.ndarray = np.array(params.stage_distribution[stage])
        next_stage_d: np.ndarray = np.array(params.stage_distribution[next_stage])
        dist: np.ndarray = (1.0 - local_progress) * stage_d + local_progress * next_stage_d
        return dist


    """
    現在のコンテキストと介入に対して、介入の受け入れ確率をprogressから計算
    """
    def _calc_accept_rate(self, context: int, intervention: int) -> float:
        #ひとまず受け入れ確率は各ステージによるものとする
        params: StageUserSimParams = self._params
        stage, next_stage, local_progress = self._progress_detail()
        ar: float = params.prob_accept_interventions[stage][context][intervention]
        return ar

    """
    現在のコンテキストと介入に対して、介入の影響を考慮した行動分布
    返り値は、影響を考慮した行動分布、合計が1になるように行動0（行動しない）を調整
    """
    def _calc_temporal_distribution(self, context: int, intervention: int) -> np.ndarray:
        params: StageUserSimParams = self._params
        stage, next_stage, local_progress = self._progress_detail()
        effect: np.ndarray = np.array(params.effect_to_temporal_distribution[stage][context][intervention])
        tmp_dist: np.ndarray = self._current_behavior_distribution.copy()
        tmp_dist = tmp_dist + effect
        tmp_dist[0] = 1.0 - np.sum(tmp_dist[1:])
        return tmp_dist

    """
    実際のbehaviorからprogressを更新
    """
    def _progress(self, behavior: ObservedUserBehavior):
        #TODO
        pass

    def interaction(self, intervention: Intervention) -> ObservedUserBehavior:
        params: StageUserSimParams = self._params
        intervention_idx: int = intervention.intervention
        context_idx: int = self._current_context.context

        accept_rate: float = self._calc_accept_rate()
        tmp_effect: np.ndarray = self._calc_temporal_effect(context_idx, intervention_idx)

        if self._rng.uniform(0.0, 1.0) < accept_rate:
            self._temporal_behavior_distribution = self._calc_temporal_distribution(context_idx, intervention_idx)
        
        behavior: ObservedUserBehavior = self._generate_user_behavior()
        self._progress(behavior)

        # record
        self._current_record.intervention = intervention_idx
        self._current_record.observed_behaviors = behavior.behaviors.copy()
        self._current_record.behavior_hist = behavior.hist.copy()
        self._current_record.progress_after_intervention = self._progress

        return behavior


    def _generate_user_behavior(self) -> ObservedUserBehavior:
        params: StageUserSimParams = self._params
        behavior: ObservedUserBehavior = ObservedUserBehavior()

        observed_behaviors: np.ndarray = np.random.choice(a=params.behavior_size, size=params.behavior_observation_size, p=self._temporal_behavior_distribution)
        behavior.behaviors = observed_behaviors
        #print(observed_behaviors)

        behavior_histogram: np.ndarray = np.zeros(params._behavior_size)
        for bhvr in observed_behaviors:
            behavior_histogram[bhvr] += 1
        behavior.hist = behavior_histogram

        return behavior

    def get_current_record(self) -> Record:
        return self._current_record

    def get_records(self) -> Records:
        self._append_record()
        return self._records
    
    def print_internal_info(self):
        print("=============== internal infomation =============")
        print("trial: ")
        print("progress(before): ")
        print("stage: ")
        print("context: ")
        print("intervention: ")
        print("progress effect: ")
        print("progress(after): ")
        print("best intervention: ")
        print("progress effect(best choice): ")
        print("progress(after best choice): ")


