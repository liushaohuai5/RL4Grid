import pandas as pd
import numpy as np

class ForecastReader(object):
    def __init__(self, ppc, is_test=False):
        max_renewable_gen_p_filepath = f'data/{"test" if is_test else "train"}/{ppc["network"]}/max_renewable_gen_p.csv'
        def_max_renewable_gen_p = pd.read_csv(max_renewable_gen_p_filepath)
        self.max_renewable_gen_p_all = def_max_renewable_gen_p.values.tolist()
        load_p_filepath = f'data/{"test" if is_test else "train"}/{ppc["network"]}/load_p.csv'
        def_load_p = pd.read_csv(load_p_filepath)
        self.load_p_all = def_load_p.values.tolist()
        self.ppc = ppc

    def read_step_renewable_gen_p_max(self, t):
        cur_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t]
        if t == self.ppc['num_sample'] - 1:
            #TODO(@zenghsh3)
            next_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t]
        else:
            next_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t+1]
        return cur_step_renewable_gen_p_max, next_step_renewable_gen_p_max

    def read_step_load_p(self, t):
        if t == self.ppc['num_sample'] - 1:
            next_step_load_p = self.load_p_all[t]
        else:
            next_step_load_p = self.load_p_all[t+1]
            # print(f'cur_load_sum={sum(self.load_p_all[t])}, next={sum(self.load_p_all[t+1])}, diff={sum(self.load_p_all[t+1]) - sum(self.load_p_all[t])}')
        return next_step_load_p

    def read_Xstep_renewable_gen_p_max(self, t, x):
        if t + x > self.ppc['num_sample']:
            renewable_gen_p_max = self.max_renewable_gen_p_all[t+1:] + \
                                  [[0. for _ in range(len(self.ppc['renewable_ids']))] for _ in range(t+1+x - self.ppc['num_sample'])]
        else:
            renewable_gen_p_max = self.max_renewable_gen_p_all[t+1:t+1+x]

        renewable_gen_p_max = np.asarray(renewable_gen_p_max)
        if renewable_gen_p_max.shape[0] < x:
            for _ in range(x - renewable_gen_p_max.shape[0]):
                renewable_gen_p_max = np.concatenate((
                    renewable_gen_p_max,
                    np.zeros((1, renewable_gen_p_max.shape[1]))
                ), axis=0)
        return renewable_gen_p_max

    def read_Xstep_load_p(self, t, x):
        if t + x > self.ppc['num_sample']:
            load_p = self.load_p_all[t+1:] + \
                     [[0. for _ in range(self.ppc['num_load'])] for _ in range(t+1+x - self.ppc['num_sample'])]
        else:
            load_p = self.load_p_all[t+1:t+1+x]

        load_p = np.asarray(load_p)
        if load_p.shape[0] < x:
            for _ in range(x - load_p.shape[0]):
                load_p = np.concatenate((
                    load_p,
                    np.zeros((1, load_p.shape[1]))
                ), axis=0)
        return load_p
