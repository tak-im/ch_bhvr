from ch_bhvr.simple_user_sim import SimpleUserSimulator, SimpleUserSimParams
from ch_bhvr.simple_user_sim import UserContext, ObservedUserBehavior, Intervention, Record, Records
from ch_bhvr.relative_entropy_recommender import RERec
import numpy as np

def main():
    # params
    params: SimpleUserSimParams = SimpleUserSimParams()
    params.context_size = 1
    params.behavior_size = 5
    params.intervention_size = 6
    params.behavior_observation_size = 100

    params.prob_context = [1.0]

    params.true_utility  = [[0.01, 0.2, 0.8, 0.1, 0.4]]
    params.base_utility  = [[0.1, 0.1, 0.1, 0.1, 0.1]]
    params.utility_std = [[0.01, 0.02, 0.1, 0.01, 0.1]]

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

    for t in range(100):
        print("----- prestep ", str(t), "-----")
        context: UserContext = sim.next_step()
        intervention: Intervention = Intervention()
        intervention.intervention = 0
        observed: ObservedUserBehavior = sim.interaction(intervention)
        rec.update(context, intervention, observed)

    for t in range(100):
        print("===== step ", str(t), "=====")
        context: UserContext = sim.next_step()
        intervention: Intervention = rec.select_intervention(context)
        observed: ObservedUserBehavior = sim.interaction(intervention)
        rec.update(context, intervention, observed)
        record: Record = sim.get_current_record()
        print("Intervention: ", intervention.intervention, ", Error: ", record.recognition_error)


if __name__ == "__main__":
    main()



