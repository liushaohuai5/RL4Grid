import pandas as pd
import numpy as np
import os
from RL4Grid.rewards import renewable_consumption_reward
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *

class ForecastReader(object):
    def __init__(self, ppc, bus_areas=None, renewable_masks=None, is_test=False):
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.bus_areas = bus_areas
        if bus_areas is None:
            max_renewable_gen_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/{ppc["network"]}/max_renewable_gen_p.csv'
            def_max_renewable_gen_p = pd.read_csv(max_renewable_gen_p_filepath)
            self.max_renewable_gen_p_all = def_max_renewable_gen_p.values.tolist()
            load_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/{ppc["network"]}/load_p.csv'
            def_load_p = pd.read_csv(load_p_filepath)
            self.load_p_all = def_load_p.values.tolist()
        else:
            max_solar_gen_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/solar.npy'
            max_wind_gen_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/wind.npy'
            load_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/load.npy'
            self.max_solar_gen_p_all = np.load(max_solar_gen_p_filepath)
            self.max_wind_gen_p_all = np.load(max_wind_gen_p_filepath)
            self.load_p_all = np.load(load_p_filepath)
            self.bus_areas = bus_areas
            self.renewable_masks = renewable_masks
            self.renewable_areas = self.bus_areas[ppc['gen_bus']][ppc['renewable_ids']]
            self.renewable_max = np.asarray(ppc['max_gen_p'])[ppc['renewable_ids']]

        self.ppc = ppc

    def read_step_renewable_gen_p_max(self, t, last_random_noises):
        if self.bus_areas is None:
            cur_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t]
            next_step_renewable_gen_p_max = self.max_renewable_gen_p_all[t + 1]
            renewable_noises = np.ones_like(cur_step_renewable_gen_p_max)
        else:
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
        if self.bus_areas is None:
            next_step_load_p = self.load_p_all[t+1]
            load_noises = np.ones_like(next_step_load_p)
        else:
            load_bus = self.ppc['load_bus']
            next_step_load_p = self.ppc['bus'][load_bus, PD] * self.load_p_all[t+1, self.bus_areas[load_bus]]
            load_noises = np.random.uniform(0.95, 1.05, next_step_load_p.shape)
            next_step_load_p *= load_noises
        return next_step_load_p, load_noises

    def read_Xstep_renewable_gen_p_max(self, t, x):
        if self.bus_areas is None:
            renewable_gen_p_max = self.max_renewable_gen_p_all[t+1:t+1+x]
        else:
            solar_gen_p_max = self.renewable_max * self.max_solar_gen_p_all[t+1:t+1+x, self.renewable_areas]
            wind_gen_p_max = self.renewable_max * self.max_wind_gen_p_all[t+1:t+1+x, self.renewable_areas]
            renewable_gen_p_max = solar_gen_p_max * (1 - self.renewable_masks) + wind_gen_p_max * self.renewable_masks
            renewable_noises = np.random.uniform(0.95, 1.05, renewable_gen_p_max.shape)
            renewable_gen_p_max *= renewable_noises
        return renewable_gen_p_max

    def read_Xstep_load_p(self, t, x):
        if self.bus_areas is None:
            load_p = self.load_p_all[t+1:t+1+x]
        else:
            load_bus = self.ppc['load_bus']
            load_p = self.ppc['bus'][load_bus, PD] * self.load_p_all[t+1:t+1+x, self.bus_areas[load_bus]]
            load_noises = np.random.uniform(0.95, 1.05, load_p.shape)
            load_p *= load_noises
        return load_p
