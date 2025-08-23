import pandas as pd
import numpy as np
import os
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *

class ForecastReader(object):
    def __init__(self, ppc, bus_areas, renewable_masks, is_test=False):
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # max_renewable_gen_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/{ppc["network"]}/max_renewable_gen_p.csv'
        # def_max_renewable_gen_p = pd.read_csv(max_renewable_gen_p_filepath)
        # self.max_renewable_gen_p_all = def_max_renewable_gen_p.values.tolist()
        # load_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/{ppc["network"]}/load_p.csv'
        # def_load_p = pd.read_csv(load_p_filepath)
        # self.load_p_all = def_load_p.values.tolist()
        self.ppc = ppc
        max_solar_gen_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/solar.npy'
        max_wind_gen_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/wind.npy'
        load_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/load.npy'
        self.max_solar_gen_p_all = np.load(max_solar_gen_p_filepath)
        self.max_wind_gen_p_all = np.load(max_wind_gen_p_filepath)
        self.load_p_all = np.load(load_p_filepath)
        self.bus_areas = bus_areas
        self.renewable_masks = renewable_masks
        self.renewable_areas = self.bus_areas[self.ppc['gen_bus']][self.ppc['renewable_ids']]
        self.renewable_max = np.asarray(self.ppc['max_gen_p'])[self.ppc['renewable_ids']]

    def read_step_renewable_gen_p_max(self, t, last_random_noises):
        # try:
        #     cur_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t]
        # except:
        #     import ipdb
        #     ipdb.set_trace()
        # if t == self.ppc['num_sample'] - 1:
        #     next_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t]
        # else:
        #     next_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t+1]
        cur_step_solar_gen_p_max = self.renewable_max * self.max_solar_gen_p_all[t, self.renewable_areas]
        cur_step_wind_gen_p_max = self.renewable_max * self.max_wind_gen_p_all[t, self.renewable_areas]
        cur_step_renewable_gen_p_max = cur_step_solar_gen_p_max * (1 - self.renewable_masks) + cur_step_wind_gen_p_max * self.renewable_masks
        cur_step_renewable_gen_p_max *= last_random_noises
        next_step_solar_gen_p_max = self.renewable_max * self.max_solar_gen_p_all[t+1, self.renewable_areas]
        next_step_wind_gen_p_max = self.renewable_max * self.max_wind_gen_p_all[t+1, self.renewable_areas]
        next_step_renewable_gen_p_max = next_step_solar_gen_p_max * (1 - self.renewable_masks) + next_step_wind_gen_p_max * self.renewable_masks
        renewable_noises = np.random.uniform(0.95, 1.05, next_step_renewable_gen_p_max.shape)
        next_step_renewable_gen_p_max *= renewable_noises
        return cur_step_renewable_gen_p_max, next_step_renewable_gen_p_max, renewable_noises

    def read_step_load_p(self, t):
        # if t == self.ppc['num_sample'] - 1:
        #     next_step_load_p = self.load_p_all[t]
        # else:
        #     next_step_load_p = self.load_p_all[t+1]
            # print(f'cur_load_sum={sum(self.load_p_all[t])}, next={sum(self.load_p_all[t+1])}, diff={sum(self.load_p_all[t+1]) - sum(self.load_p_all[t])}')
        load_bus = self.ppc['load_bus']
        next_step_load_p = self.ppc['bus'][load_bus, PD] * self.load_p_all[t+1, self.bus_areas[load_bus]]
        load_noises = np.random.uniform(0.95, 1.05, next_step_load_p.shape)
        next_step_load_p *= load_noises
        return next_step_load_p, load_noises

    def read_Xstep_renewable_gen_p_max(self, t, x):
        # if t + x > self.ppc['num_sample']:
        #     renewable_gen_p_max = self.max_renewable_gen_p_all[t+1:] + \
        #                           [[0. for _ in range(len(self.ppc['renewable_ids']))] for _ in range(t+1+x - self.ppc['num_sample'])]
        # else:
        #     renewable_gen_p_max = self.max_renewable_gen_p_all[t+1:t+1+x]
        #
        # renewable_gen_p_max = np.asarray(renewable_gen_p_max)
        # if renewable_gen_p_max.shape[0] < x:
        #     for _ in range(x - renewable_gen_p_max.shape[0]):
        #         renewable_gen_p_max = np.concatenate((
        #             renewable_gen_p_max,
        #             np.zeros((1, renewable_gen_p_max.shape[1]))
        #         ), axis=0)

        solar_gen_p_max = self.renewable_max * self.max_solar_gen_p_all[t+1:t+1+x, self.renewable_areas]
        wind_gen_p_max = self.renewable_max * self.max_wind_gen_p_all[t+1:t+1+x, self.renewable_areas]
        renewable_gen_p_max = solar_gen_p_max * (1 - self.renewable_masks) + wind_gen_p_max * self.renewable_masks
        renewable_noises = np.random.uniform(0.95, 1.05, renewable_gen_p_max.shape)
        renewable_gen_p_max *= renewable_noises
        return renewable_gen_p_max

    def read_Xstep_load_p(self, t, x):
        # if t + x > self.ppc['num_sample']:
        #     load_p = self.load_p_all[t+1:] + \
        #              [[0. for _ in range(self.ppc['num_load'])] for _ in range(t+1+x - self.ppc['num_sample'])]
        # else:
        #     load_p = self.load_p_all[t+1:t+1+x]
        #
        # load_p = np.asarray(load_p)
        # if load_p.shape[0] < x:
        #     for _ in range(x - load_p.shape[0]):
        #         load_p = np.concatenate((
        #             load_p,
        #             np.zeros((1, load_p.shape[1]))
        #         ), axis=0)
        load_bus = self.ppc['load_bus']
        load_p = self.ppc['bus'][load_bus, PD] * self.load_p_all[t+1:t+1+x, self.bus_areas[load_bus]]
        load_noises = np.random.uniform(0.95, 1.05, load_p.shape)
        load_p *= load_noises
        return load_p
