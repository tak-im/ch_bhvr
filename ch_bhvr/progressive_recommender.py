from ch_bhvr.stage_user_sim import Intervention, UserContext, ObservedUserBehavior
from ch_bhvr import util
from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import multinomial
import math
from enum import Enum

def test():
    #p: Particle = Particle(5, 0.05, np.array([0.6, 0.1, 0.1, 0.1, 0.1]))
    #p.next_step()
    #print(p.behavior_distribution)

    pf: ParticleFilter = ParticleFilter(100, 5, 0.02, np.array([0.6,0.1,0.1,0.1,0.1]))
    print(pf.estimated_distribution)

    pf.update([58,8,11,10,15])
    print(pf.estimated_distribution)

class OperationMode(Enum):
    OBSERVE = 0
    INTERVENTION = 1


class Particle():
    def __init__(self, behavior_size: int, noise_std: float, init_dist: np.ndarray):
        self.behavior_size = behavior_size
        self.noise_std = noise_std
        self.behavior_distribution: np.ndarray = init_dist.copy()

    def set_distribution(self, distribution: np.ndarray):
        self.behavior_distribution = distribution.copy()

    def next_step(self):
        bhv_dist_sum: float = 2.0
        while bhv_dist_sum >= 1.0:
            noise: np.ndarray = np.random.normal(0.0, self.noise_std, self.behavior_size)
            self.behavior_distribution += noise
            self.behavior_distribution = self.behavior_distribution.clip(0.0, 1.0)
            bhv_dist_sum: float = np.sum(self.behavior_distribution[1:])
            self.behavior_distribution[0] = 1.0 - bhv_dist_sum

    def likelihood(self, obserbed_count: np.ndarray) -> float:
        count: int = np.sum(obserbed_count)
        likelihood: float = multinomial.pmf(obserbed_count, count, self.behavior_distribution)
        #print("############# likelihood")
        #print(obserbed_count)
        #print(count)
        #print(self.behavior_distribution)
        #print(likelihood)
        #if np.isnan(likelihood):
        #    exit()

        return likelihood

class ParticleFilter():

    def __init__(self, particle_num, behavior_size: int, noise_std: float, init_dist: np.ndarray):
        self.particle_num: int = particle_num
        self.behavior_size = behavior_size
        self.noise_std = noise_std
        self.init_dist = init_dist
        self.estimated_distribution: np.ndarray = init_dist.copy()
        self._particles: List[Particle] = []
        for idx in range(particle_num):
            ptcl: Particle = Particle(behavior_size, noise_std, init_dist)
            self._particles.append(ptcl)
        self._last_particles: List[Particle] = []
        for idx in range(particle_num):
            ptcl: Particle = Particle(behavior_size, noise_std, init_dist)
            self._last_particles.append(ptcl)


    def update(self, observed_count: np.ndarray):
        #print("============================")
        likelihoods: np.nparray = np.zeros(self.particle_num)
        #print(likelihoods)
        for idx in range(self.particle_num):
            ptcl: Particle = self._particles[idx]
            ptcl.next_step()
            likelihoods[idx] = ptcl.likelihood(observed_count)
        #print(likelihoods)

        weights: np.ndarray = likelihoods.copy()
        weights /= np.sum(weights)
        #print(weights)

        distribution: np.ndarray = np.zeros(self.behavior_size)
        for idx in range(self.particle_num):
            ptcl: Particle = self._particles[idx]
            distribution += ptcl.behavior_distribution * weights[idx]
        self.estimated_distribution = distribution
        #print(weights)
        #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #print("recommender estimate distribution: ", self.estimated_distribution)

        self._resampling(weights)

    def _resampling(self, weights: np.ndarray):
        # 入れ替え
        new_particles: List[Particle] = self._last_particles
        self._last_particles = self._particles
        self._particles = new_particles

        for idx in range(self.particle_num):
            selected = np.random.choice(self.particle_num, p=weights)
            self._particles[idx].set_distribution(self._last_particles[selected].behavior_distribution)


class ProgressiveRecommender():

    def __init__(self, len_intervention: int, len_behavior: int):
        self._len_intervention: int = len_intervention
        self._len_behavior: int = len_behavior

        # 動作モード
        self._mode: OperationMode = OperationMode.INTERVENTION

        # particles
        self._particle_num: int = 100
        self._noise_std: float = 0.01
        self._pf: ParticleFilter = None

        # context, intervention -> alpha by behabior
        self._params: Dict[Tuple[int, int, int], List[float]] = {}

        # hashmap
        self._count_c: Dict[int, int] = {} # context -> count
        self._count_ci: Dict[Tuple[int, int], int] = {} # (context, intervention) -> intervention count
        self._count_ci_b: Dict[Tuple[int, int], int] = {} # (context, intervention) -> all behavior count
        self._count_cib: Dict[Tuple[int, int, int], int] = {} # (context, intervention, behavior) -> count
        self._reward_ci: Dict[Tuple[int, int], float] = {} # (context, intervention) -> sum of reward

    def init_pf(self, particle_num: int, noise_std: float, init_dist: np.ndarray):
        self._particle_num = particle_num
        self._noise_std = noise_std
        self._pf = ParticleFilter(particle_num, self._len_behavior, noise_std, init_dist)

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
        #base_dist = self._sample_distribution(context.context, 0)
        #print(base_util)
        #base_util_np: np.array = np.array(base_dist, dtype=float)
        #base_util_np = base_util_np / base_util_np.sum()

        intervention: Intervention = Intervention()

        if self._mode == OperationMode.OBSERVE:
            intervention.intervention = 0
            self._mode = OperationMode.INTERVENTION
        else:
            base_distribution: np.ndarray = self._pf.estimated_distribution

            scores: np.ndarray = np.zeros(self._len_intervention)
            for iv in range(self._len_intervention):
                score: float = self._ucb_score(context.context, iv)
                scores[iv] = score

            selected: int = np.argmax(scores)
            intervention.intervention = selected
            self._mode = OperationMode.OBSERVE
        
        return intervention

    def _ucb_score(self, context: int, intervention: int) -> float:
        total_reward_ci: float = self._reward_ci.get((context, intervention), 0.0)
        count_ci: int = self._count_ci.get((context, intervention), 0)
        count_c: int = self._count_c.get(context, 0)

        score: float = 100.0
        if intervention == 0:
            score = 0.0
        elif count_ci > 0:
            score = total_reward_ci / count_ci
            score += math.sqrt(math.log(float(count_c)) / (2.0 * count_ci))
        return score

    def _sample_score(self, context: int, intervention: int, base_dist_np: np.ndarray) -> float:
        distribution: List[float] = self._sample_distribution(context, intervention)
        dist_np = np.array(distribution)
        dist_np = dist_np / dist_np.sum()

        #re = np.dot(utilities_np, np.log2(utilities_np / base_utilities_np))
        #re = util.relative_entropy(utilities_np, base_utilities_np)
        #print("base", base_utilities_np)
        #print("sample", utilities_np)
        score = util.jaccard_like(dist_np, base_dist_np)
        #print("dist", re)

        return score

    def _sample_distribution(self, context: int, intervention: int) -> List[float]: # by behavior
        distribution: List[float] = [0] * self._len_behavior
        count_ci_all_b: int = self._count_ci_b.get((context, intervention), 0)

        # intervention 0: no intervention
        if intervention == 0:
            for bhvr in range(self._len_behavior):
                count_cib: int = self._count_cib.get((context, intervention, bhvr), 0)
                distribution[bhvr] = float(count_cib) / float(count_ci_all_b)
        else:
            alphas: List[float] = self._alphas((context, intervention))
            for bhvr in range(self._len_behavior):
                count_cib: float = float(self._count_cib.get((context, intervention, bhvr), 0))
                alphas[bhvr] += count_cib
                #print("alphas: ", alphas)
            #distribution = np.random.beta(alpha + count_cib, beta + count_ci - count_cib)
            distribution = np.random.dirichlet(alpha=alphas).tolist()

        return distribution

    def _alphas(self, ci: Tuple[int, int]) -> List[float]:
        alphas: List[float] = self._params.get(ci, [1.0] * self._len_behavior)
        return alphas

    def update(self, context: UserContext, intervention: Intervention, observed: ObservedUserBehavior) -> None:

        if intervention.intervention == 0:
            self._pf.update(observed.hist)
            print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
            print("estimated distribution: ", self._pf.estimated_distribution)
        else:
            ci = (context.context, intervention.intervention)
            observedBhaviors: List[int] = observed.behaviors
            self._count_c[context.context] = self._count_c.get(context.context, 0) + 1
            self._count_ci[ci] = self._count_ci.get(ci, 0) + 1
            for bhvr in observedBhaviors:
                self._count_ci_b[ci] = self._count_ci.get(ci, 0) + 1
                cib = (context.context, intervention.intervention, bhvr)
                self._count_cib[cib] = self._count_cib.get(cib, 0) + 1

            base_distribution: np.ndarray = self._pf.estimated_distribution
            observed_dist: np.ndarray = observed.hist / observed.hist.sum()

            reward = util.jaccard_like(observed_dist, base_distribution)
            self._reward_ci[ci] = self._reward_ci.get(ci, 0) + reward



if __name__ == "__main__":
    test()
