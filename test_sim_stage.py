from ch_bhvr.stage_user_sim import StageUserSimulator, StageUserSimParams, UserContext, ObservedUserBehavior, Intervention, Record, Records
#from ch_bhvr.simple_user_sim2 import SimpleUserSimulator, SimpleUserSimParams, UserContext, ObservedUserBehavior, Intervention, Record, Records
#from ch_bhvr.static_recommender import RERec
from ch_bhvr.progressive_recommender import ProgressiveRecommender
#from ch_bhvr.rec_test import RERec
from ch_bhvr.stage_record_viewer import RecordViewer
import numpy as np
from typing import List
import matplotlib.pyplot as plt

def main():
    # params
    params: StageUserSimParams = StageUserSimParams()
    params.context_size = 1
    params.behavior_size = 6
    params.intervention_size = 6
    params.behavior_observation_size = 500

    params.prob_context = [1.0]
    
    params.init_progress = 0.1
    params.stage_size = 5
    params.stage_distribution = [
        [0.95, 0.01, 0.01, 0.01, 0.01, 0.01], #stage0
        [0.84, 0.1, 0.02, 0.01, 0.01, 0.02], #stage1
        [0.76, 0.08, 0.1, 0.02, 0.01, 0.03], #stage2
        [0.71, 0.05, 0.08, 0.1, 0.02, 0.04], #stage3
        [0.67, 0.05, 0.05, 0.08, 0.1, 0.05], #stage4
    ]

    #params.prob_accept_interventions = [
    #    [[0.3, 0.3, 0.3, 0.3, 0.3, 0.3]],
    #    [[0.4, 0.4, 0.4, 0.4, 0.4, 0.4]],
    #    [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
    #    [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]],
    #    [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    #]
    params.prob_accept_interventions = [
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]],
        [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
    ]
    
    params.effect_to_temporal_distribution = [
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]],
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]],
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]],
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]],
        [[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        ]]
    ]

    params.auto_change_coeff = 0.0
    params.change_coeff = 0.2
    params.change_threshold = 0.5
    params.next_stage_progress = 0.9

    sim: StageUserSimulator = StageUserSimulator(params)

    """
    context: UserContext = sim.next_step()
    print("context: ", context.context)

    intervention: Intervention = Intervention()
    intervention.intervention = 0

    observed: ObservedUserBehavior = sim.interaction(intervention)

    print("intervention: ", intervention.intervention)
    print("observed: ", observed.behaviors)
    """

    rec: ProgressiveRecommender = ProgressiveRecommender(params.intervention_size, params.behavior_size, ucb_discount_rate=1.0, ucb_param=0.5)
    rec.init_pf(100, 0.02, np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1]))
    
    for t in range(100):
        print("----- prestep ", str(t), "-----")
        context: UserContext = sim.next_step()
        intervention: Intervention = Intervention()
        intervention.intervention = 0
        observed: ObservedUserBehavior = sim.interaction(intervention)
        rec.update(context, intervention, observed)

    for t in range(1000):
        print("===== step ", str(t), "=====")
        context: UserContext = sim.next_step()
        intervention: Intervention = rec.select_intervention(context)
        #intervention = Intervention()
        #intervention.intervention = 1
        observed: ObservedUserBehavior = sim.interaction(intervention)
        rec.update(context, intervention, observed)
        record: Record = sim.get_current_record()
        print("Intervention: ", intervention.intervention, ", progress: ", record.progress_before_intervention, " -> ", record.progress_after_intervention)

    sim.print_internal_info()
    records: Records = sim.get_records()

    viewer = RecordViewer(records)

    viewer.view_progress(50)
    viewer.view_intervention(50)
    plt.show()

    """
    trial_window_size = 100  
    trials: List[int] = []
    errors: List[float] = []
    ivs: List[List[float]] = [[] for _ in range(records.params.intervention_size)]
    tmp_size: float = 0.0
    tmp_err_sum: float = 0.0
    tmp_iv_sum: List[int] = [0] * records.params.intervention_size

    for record in records.records:
        tmp_size += 1
        tmp_err_sum += record.recognition_error
        tmp_iv_sum[record.intervention] += 1
        #print(record.index)
        #errors[record.index-1] = record.recognition_error
        if tmp_size >= trial_window_size:
            print(tmp_iv_sum)
            trials.append(record.index)
            errors.append(tmp_err_sum / tmp_size)
            for iv in range(records.params.intervention_size):
                #print(ivs[iv])
                #print(tmp_iv_sum[iv])
                (ivs[iv]).append(tmp_iv_sum[iv])
            tmp_size = 0.0
            tmp_err_sum = 0.0
            tmp_iv_sum = [0] * records.params.intervention_size

    #print(ivs)

    fig, ax = plt.subplots()
    ax.plot(trials, errors)
    #ax.hist(errors, bins=10)
    #plt.show()

    fig2, ax2 = plt.subplots()
    ivs_np: np.ndarray = np.array(ivs)
    bottom_np: np.ndarray = np.zeros(len(trials))
    for iv in range(records.params.intervention_size):
        ax2.bar(trials, ivs[iv], bottom=bottom_np.tolist(), width=20)
        bottom_np = bottom_np + ivs_np[iv]
    plt.show()
    """


if __name__ == "__main__":
    main()



