import numpy as np
from gym import spaces
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *

class ActionSpace(object):
    def __init__(self, ppc):
        self.num_gen = ppc['num_gen']
        self.ramp_rate = ppc['ramp_rate']
        self.max_gen_p = ppc['max_gen_p']
        self.min_gen_p = ppc['min_gen_p']
        self.max_gen_v = ppc['max_gen_v']
        self.min_gen_v = ppc['min_gen_v']
        self.thermal_ids = ppc['thermal_ids']
        self.renewable_ids = ppc['renewable_ids']
        self.balanced_id = ppc['balanced_id']
        self.keep_decimal_digits = ppc['keep_decimal_digits']

    def get_p_range(self, gen_p, steps_to_recover_gen, steps_to_close_gen, steps_to_min_gen, nextstep_renewable_gen_p_max):
        # Initialization
        low = np.zeros([self.num_gen])
        high = np.zeros([self.num_gen])
        low[self.balanced_id] = -float('inf')
        high[self.balanced_id] = float('inf')

        self.update_thermal_p(low, high, gen_p, steps_to_recover_gen, steps_to_close_gen, steps_to_min_gen)
        self.update_renewable_p(low, high, gen_p, nextstep_renewable_gen_p_max)
        
        low = np.round(low, self.keep_decimal_digits)
        high = np.round(high, self.keep_decimal_digits)
        return low, high

    def update_thermal_p(self, low, high, gen_p, steps_to_recover_gen, steps_to_close_gen, steps_to_min_gen):
        # injection values are less than maximum limit and larger than minimum limit
        max_capa_adjust = [self.max_gen_p[i] - gen_p[i] for i in range(self.num_gen)]
        min_capa_adjust = [self.min_gen_p[i] - gen_p[i] for i in range(self.num_gen)]

        # adjust actions of thermal generators should less than ramp value
        max_ramp_adjust = [self.ramp_rate * ele for ele in self.max_gen_p]

        # default value of min&max adjust is 0
        # print(f'inner step={steps_to_min_gen}')
        for idx in self.thermal_ids:
            if gen_p[idx] == 0.0:
                low[idx] = 0.0

                high[idx] = self.min_gen_p[idx]
                if steps_to_recover_gen[idx] != 0 or steps_to_min_gen[idx] > 0:  # cannot turn on
                    high[idx] = 0.0 

            elif abs(gen_p[idx] - self.min_gen_p[idx]) < 1e-3:
                high[idx] = min(max_capa_adjust[idx], max_ramp_adjust[idx])

                if steps_to_close_gen[idx] == 0:  # can turn off
                    low[idx] = -self.min_gen_p[idx]
                else:  # cannot turn off
                    low[idx] = 0.0

            elif gen_p[idx] > self.min_gen_p[idx]:
                low[idx] = max(min_capa_adjust[idx], -max_ramp_adjust[idx])
                high[idx] = min(max_capa_adjust[idx], max_ramp_adjust[idx])
                if steps_to_close_gen[idx] == 0:  # can turn off
                    low[idx] = max(-gen_p[idx], -max_ramp_adjust[idx])
            else:
                gen_p[idx] = 0.0
                low[idx] = 0.0
                high[idx] = self.min_gen_p[idx]
                if steps_to_recover_gen[idx] != 0 or steps_to_min_gen[idx] > 0:  # cannot turn on
                    high[idx] = 0.0
            #     import ipdb
            #     ipdb.set_trace()
            #     assert False

    def update_renewable_p(self, low, high, gen_p, nextstep_renewable_gen_p_max):
        for i, idx in enumerate(self.renewable_ids):
            low[idx] = -gen_p[idx]
            high[idx] = nextstep_renewable_gen_p_max[i] - gen_p[idx]

    def get_v_range(self, gen_v):
        low = np.zeros([self.num_gen])
        high = np.zeros([self.num_gen])
        for i in range(self.num_gen):
            low[i] = self.min_gen_v[i] - gen_v[i]
            high[i] = self.max_gen_v[i] - gen_v[i]
        low = np.round(low, self.keep_decimal_digits)
        high = np.round(high, self.keep_decimal_digits)
        return low, high

    # def update(self, grid, steps_to_recover_gen, steps_to_close_gen, steps_to_min_gen, rounded_gen_p,
    #         nextstep_renewable_gen_p_max):
    #     gen_p = rounded_gen_p
    #     gen_v = grid.prod_v[0]
    #
    #     low_adjust_p, high_adjust_p = self.get_p_range(
    #         gen_p, steps_to_recover_gen, steps_to_close_gen, steps_to_min_gen, nextstep_renewable_gen_p_max
    #     )
    #     action_space_p = spaces.Box(low=low_adjust_p, high=high_adjust_p)
    #
    #     low_adjust_v, high_adjust_v = self.get_v_range(gen_v)
    #     action_space_v = spaces.Box(low=low_adjust_v, high=high_adjust_v)
    #
    #     action_space = {'adjust_gen_p': action_space_p, 'adjust_gen_v': action_space_v}
    #     return action_space

    def update(self, result, steps_to_recover_gen, steps_to_close_gen, steps_to_min_gen, rounded_gen_p,
            nextstep_renewable_gen_p_max):
        gen_p = rounded_gen_p
        gen_v = result['gen'][:, VG]

        low_adjust_p, high_adjust_p = self.get_p_range(
            gen_p, steps_to_recover_gen, steps_to_close_gen, steps_to_min_gen, nextstep_renewable_gen_p_max
        )
        action_space_p = spaces.Box(low=low_adjust_p, high=high_adjust_p)

        low_adjust_v, high_adjust_v = self.get_v_range(gen_v)
        action_space_v = spaces.Box(low=low_adjust_v, high=high_adjust_v)

        action_space = {'adjust_gen_p': action_space_p, 'adjust_gen_v': action_space_v}
        return action_space
