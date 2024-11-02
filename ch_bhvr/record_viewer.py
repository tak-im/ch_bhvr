from typing import List
import matplotlib.pyplot as plt
from ch_bhvr.simple_user_sim import Record, Records, SimpleUserSimParams
import numpy as np

class RecordViewer():
    _records: Records

    def __init__(self, records: Records):
        self._records = records
        size: int = len(records.records)
        print("size: ", str(size))

    def print_best(self):
        params: SimpleUserSimParams = self._records.params
        

    def view_error(self, view_window_size: int):
        trials: List[int] = []
        errors: List[float] = []
        tmp_size: float = 0.0
        tmp_err_sum: float = 0.0

        for record in self._records.records:
            tmp_size += 1.0
            tmp_err_sum += record.recognition_error
            #print(record.index)
            #errors[record.index-1] = record.recognition_error
            if tmp_size >= view_window_size:
                trials.append(record.index)
                errors.append(tmp_err_sum / tmp_size)
                tmp_size = 0.0
                tmp_err_sum = 0.0

        fig, ax = plt.subplots()
        ax.plot(trials, errors)
        #plt.show()


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
                print(tmp_iv_sum)
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
            ax.bar(trials, ivs[iv], bottom=bottom_np.tolist(), width=20)
            bottom_np = bottom_np + ivs_np[iv]
        #plt.show()
