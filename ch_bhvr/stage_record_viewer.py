from typing import List
import matplotlib.pyplot as plt
from ch_bhvr.stage_user_sim import Record, Records, StageUserSimParams
import numpy as np
import ch_bhvr.util as util

class RecordViewer():
    _records: Records

    def __init__(self, records: Records):
        self._records = records
        size: int = len(records.records)
        print("size: ", str(size))


    def view_progress(self, view_window_size: int):
        trials: List[int] = []
        progress: List[float] = []

        tmp_size: float = 0.0
        for record in self._records.records:
            tmp_size += 1.0
            if tmp_size >= view_window_size:
                trials.append(record.index)
                progress.append(record.progress_after_intervention)
                tmp_size = 0.0

        fig, ax = plt.subplots()
        ax.plot(trials, progress)
        ax.set_ylim(0.0, 1.0)
        #ax.plot(trials, changes)
        #plt.show()
        print(trials)
        print(progress)

    def view_intervention(self, view_window_size: int):
        trials: List[int] = []
        ivs: List[List[float]] = [[] for _ in range(self._records.params.intervention_size)]
        tmp_size: float = 0.0
        tmp_iv_sum: List[int] = [0] * self._records.params.intervention_size

        for record in self._records.records:
            tmp_size += 1.0
            tmp_iv_sum[record.intervention] += 1
            #print(record.index)
            #errors[record.index-1] = record.recognition_error
            if tmp_size >= view_window_size:
                #print(tmp_iv_sum)
                trials.append(record.index)
                for iv in range(self._records.params.intervention_size):
                    #print(ivs[iv])
                    #print(tmp_iv_sum[iv])
                    (ivs[iv]).append(tmp_iv_sum[iv])
                tmp_size = 0.0
                tmp_iv_sum = [0] * self._records.params.intervention_size

        fig, ax = plt.subplots()
        ivs_np: np.ndarray = np.array(ivs)
        bottom_np: np.ndarray = np.zeros(len(trials))
        for iv in range(self._records.params.intervention_size):
            ax.bar(trials, ivs[iv], bottom=bottom_np.tolist(), width=10)
            bottom_np = bottom_np + ivs_np[iv]
        #plt.show()
