from ch_bhvr.interface import IContext, IIntervention, IObservedBehavior, IRecords, IUserBehaviorSimulator
from typing import List, Tuple
import numpy as np
from ch_bhvr import util
from dataclasses import dataclass

class UserContext(IContext):
    def __init__(self):
        self.context: int = 0

class ObservedUserBehavior(IObservedBehavior):
    def __init__(self):
        self.behaviors: np.ndarray = None # 0: unrelated behavior
        self.hist: np.ndarray = None

class Intervention(IIntervention):
    def __init__(self):
        intervention: int = 0 # 0:no intervention

class StageUserSimParams():
    def __init__(self):
        # probrem size
        self.context_size: int = 1
        self.behavior_size: int = 6
        self.intervention_size: int = 6
        self.behavior_observation_size: int = 100

        # context
        self.prob_context: List[float] = [1.0]

        # user param 
        self.init_progress: float = 0.1
        self.stage_size: int = 5
        self.stage_distribution: List[List[float]] = [] # (by stage, behabior) 

        # intervention
        # (by stage, context, intervention)
        self.prob_accept_interventions: List[List[List[float]]] = []
        # (by stage, context, intervention, behavior)
        self.effect_to_temporal_distribution: List[List[List[List[float]]]] = []
        #effect_to_temporal_distribution_var: List[List[List[float]]]

        # behavior change
        self.auto_change_coeff: float = 0.0
        self.change_coeff: float = 1.0
        # 勝手に行動が変わるのを防ぐ、(observed - base)/(next - base)がthreshold以下だとprogressは進まない
        self.change_threshold: float = 0.1
        # localprogressがこの値を超えると次のstageに進む
        self.next_stage_progress: float = 0.9

class Record():
    def __init__(self):       
        self.index: int = None
        self.context: int = None
        self.auto_progress: float = None

        self.progress_before_intervention: float = None
        self.intervention: int = None
        self.progress_after_intervention: float = None

        self.observed_behaviors: np.ndarray = None
        self.behavior_hist: np.ndarray = None
   

class Records(IRecords):
    params: StageUserSimParams
    records: List[Record]

class StageUserSimulator(IUserBehaviorSimulator):

    def __init__(self, params: StageUserSimParams):
        # instance var
        # initial param
        self._params: StageUserSimParams = params
        # user param 
        self._progress: float        
        #temporal_state
        self._current_context: UserContext = None
        self._current_behavior_distribution: np.ndarray # by behavior
        self._temporal_behavior_distribution: np.ndarray # by behavior
        #rng
        self._rng: np.random.Generator = np.random.default_rng()
        # records
        self._index: int = 0
        self._records: Records = Records()
        self._records.params = params
        self._records.records = []

        self._current_record: Record = None
        # replay_records
        self._replay_records: Records = None

        # initialize
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
            context_index: int = np.random.choice(a=range(params.context_size), p=params.prob_context)
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
        params: StageUserSimParams = self._params
        return self._rng.uniform(0, 1) * params.auto_change_coeff

    """
    progressから現在のstageと次のステージまでの進捗を計算
    返り値は(現在のステージ、次のステージ、現在のステージから次のステージに向けた進捗割合（0～1）)
    """
    def _progress_detail(self) -> Tuple[int, int, float]:
        params: StageUserSimParams = self._params
        #print(self._progress, ", ", (1.0 / (params.stage_size - 1)))
        a, b = divmod(self._progress, (1.0 / (params.stage_size - 1)))
        #print(a, ", ", b)
        stage: int = int(a)
        next_stage: int = stage + 1
        if next_stage >= params.stage_size:
            next_stage = params.stage_size - 1
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

        # effect は 次のstage行動分布に近づくように影響する（理解度）
        stage_dist: np.ndarray = np.array(params.stage_distribution[stage])
        next_dist: np.ndarray = np.array(params.stage_distribution[next_stage])
        effect = (next_dist - stage_dist) * effect

        tmp_dist = tmp_dist + effect
        tmp_dist[0] = 1.0 - np.sum(tmp_dist[1:])
        return tmp_dist

    """
    実際のbehaviorからprogressを更新
    """
    def _make_progress(self, behavior: ObservedUserBehavior):
        params: StageUserSimParams = self._params
        # 現在のstageで不足している行動
        stage, next_stage, local_progress = self._progress_detail()      
        #stage_d: np.ndarray = np.array(params.stage_distribution[stage])
        next_stage_d: np.ndarray = np.array(params.stage_distribution[next_stage])

        ## ステージ間で足らない行動
        #stage_diff: np.ndarray = next_stage_d - stage_d
        #stage_diff = stage_diff.clip(0, 1)
        #stage_diff[0] = 0.0

        print("===============================================================================")
        print("current_dist: ", self._current_behavior_distribution)
        print("next_dist: ", next_stage_d)

        # 現在の基本分布
        base_d: np.ndarray = self._current_behavior_distribution
        # 現在の分布が次のステージに不足している行動分布
        next_base_diff: np.ndarray = next_stage_d - base_d
        next_base_diff = next_base_diff.clip(0,1)
        next_base_diff[0] = 0.0

        # 経験した行動が足らない行動をどの程度満たしたか
        behavior_diff = behavior.hist.astype(np.float64)
        behavior_diff = behavior_diff / behavior_diff.sum()
        print("obesrved: ", behavior_diff)
        behavior_diff = behavior_diff - base_d
        behavior_diff = behavior_diff.clip(0, next_base_diff)
        behavior_diff[0] = 0.0

        print("observed-base: ", behavior_diff)
        print("next-base: ", next_base_diff)

        exp_progress: float = behavior_diff.sum() / next_base_diff.sum()
        print("progress: ", self._progress, ", stage: ", stage, ", local_progress: ", local_progress)
        print("exp_progress: ", exp_progress)
        #if exp_progress > local_progress:
        #    progress_diff: float = exp_progress - local_progress
        #    progress_diff = progress_diff / ( params.stage_size - 1 )
        #    self._progress = self._progress + progress_diff * params.change_coeff
        if exp_progress > params.change_threshold:
            progress_diff: float = (1.0 - local_progress) * exp_progress * params.change_coeff / (params.stage_size - 1)
            self._progress = self._progress + progress_diff * params.change_coeff
            stage_new, next_stage_new, local_progress_new = self._progress_detail()
            # local plogressが一定を超えたら次のstage
            if local_progress_new > params.next_stage_progress and next_stage_new < params.stage_size - 1:
                self._progress = next_stage_new / (params.stage_size - 1)


    def interaction(self, intervention: Intervention) -> ObservedUserBehavior:
        params: StageUserSimParams = self._params
        intervention_idx: int = intervention.intervention
        context_idx: int = self._current_context.context

        accept_rate: float = self._calc_accept_rate(context_idx, intervention_idx)

        if self._rng.uniform(0.0, 1.0) < accept_rate:
            self._temporal_behavior_distribution = self._calc_temporal_distribution(context_idx, intervention_idx)
        
        behavior: ObservedUserBehavior = self._generate_user_behavior()
        self._make_progress(behavior)

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

        behavior_histogram: np.ndarray = np.zeros(params.behavior_size)
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


