from ch_bhvr.simple_user_sim import SimpleUserSimulator, SimpleUserSimParams
from ch_bhvr.simple_user_sim import UserContext, ObservedUserBehavior, Intervention, Record, Records
from ch_bhvr.relative_entropy_recommender import RERec
import numpy as np
from typing import List
import matplotlib.pyplot as plt

def main():
    # params
    params: SimpleUserSimParams = SimpleUserSimParams()
    params.context_size = 1
    params.behavior_size = 5
    params.intervention_size = 6
    params.behavior_observation_size = 1000

    params.prob_context = [1.0]

    params.true_utility  = [[0.01, 0.2, 0.8, 0.1, 0.4]]
    params.base_utility  = [[0.1, 0.1, 0.1, 0.1, 0.1]]
    params.utility_std = [[0.01, 0.01, 0.01, 0.01, 0.01]]

    params.understanding = [[0.2, 0.2, 0.5, 0.1, 0.1]]

    params.prob_accept_interventions = [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]]
    params.effect_temporal_understanding = [[0.0, 0.0, 0.0, 0.0, 0.0],
                                             [0.5, 0.0, 0.0, 0.0, 0.0],
                                             [0.0, 0.5, 0.0, 0.0, 0.0],
                                             [0.0, 0.0, 0.5, 0.0, 0.0],
                                             [0.0, 0.0, 0.0, 0.5, 0.0],
                                             [0.0, 0.0, 0.0, 0.0, 0.5]]
    params.effect_permanent_understanding = np.zeros((6, 5)).tolist()

    sim: SimpleUserSimulator = SimpleUserSimulator(params)

    """
    context: UserContext = sim.next_step()
    print("context: ", context.context)

    intervention: Intervention = Intervention()
    intervention.intervention = 0

    observed: ObservedUserBehavior = sim.interaction(intervention)

    print("intervention: ", intervention.intervention)
    print("observed: ", observed.behaviors)
    """

    rec: RERec = RERec(params.intervention_size, params.behavior_size)

    for t in range(500):
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
        observed: ObservedUserBehavior = sim.interaction(intervention)
        rec.update(context, intervention, observed)
        record: Record = sim.get_current_record()
        print("Intervention: ", intervention.intervention, ", Error: ", record.recognition_error)

    records: Records = sim.get_records()

    size: int = len(records.records)
    print("size: ", str(size))
    trial_window_size = 50  
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



if __name__ == "__main__":
    main()



