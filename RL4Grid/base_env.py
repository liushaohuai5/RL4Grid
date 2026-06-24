import sys
import time

from .observation import Observation
from .rewards import *
from .utilize.read_forecast_value import ForecastReader
from .utilize.line_cutting import Disconnect
from .utilize.action_space import ActionSpace
from .utilize.legal_action import *
import copy
import numpy as np
import pandas as pd
import warnings
import pypower
from pypower.api import *
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
from pypower.ppoption import ppoption
from pypower.opf import opf
from numpy import flatnonzero as find
from pypower.isload import isload
from pypower.totcost import totcost
import random
import os
warnings.filterwarnings('ignore')
from .model_jm.visualizer import Visualizer

def profile(func):
    from line_profiler import LineProfiler
    def wrapper(*args, **kwargs):
        lp = LineProfiler()
        lp_wrapper = lp(func)
        result = lp_wrapper(*args, **kwargs)
        lp.print_stats()

        return result
    return wrapper

def run_uopf(ppc):
    open_hot = np.zeros(ppc['num_gen'] + 1)
    close_hot = np.zeros(ppc['num_gen'] + 1)

    def _log_ppc_value_changes(before_ppc, after_ppc, tag):
        print(f'[DEBUG] ppc value changes around {tag}:')
        all_keys = sorted(set(before_ppc.keys()) | set(after_ppc.keys()))
        for key in all_keys:
            if key not in before_ppc:
                print(f'  - {key}: added')
                continue
            if key not in after_ppc:
                print(f'  - {key}: removed')
                continue

            before_val = before_ppc[key]
            after_val = after_ppc[key]

            if isinstance(before_val, np.ndarray) and isinstance(after_val, np.ndarray):
                if before_val.shape != after_val.shape:
                    print(f'  - {key}: shape {before_val.shape} -> {after_val.shape}')
                    continue

                equal_mask = (before_val == after_val)
                if np.issubdtype(before_val.dtype, np.floating) and np.issubdtype(after_val.dtype, np.floating):
                    equal_mask = equal_mask | (np.isnan(before_val) & np.isnan(after_val))
                changed_count = int(equal_mask.size - np.count_nonzero(equal_mask))
                if changed_count == 0:
                    print(f'  - {key}: unchanged')
                    continue

                if np.issubdtype(before_val.dtype, np.number) and np.issubdtype(after_val.dtype, np.number):
                    diff = np.abs(after_val - before_val)
                    max_abs_diff = float(np.nanmax(diff))
                    print(f'  - {key}: changed_elements={changed_count}/{before_val.size}, max_abs_diff={max_abs_diff}')
                else:
                    print(f'  - {key}: changed_elements={changed_count}/{before_val.size}')
                continue

            changed = before_val != after_val
            if changed:
                print(f'  - {key}: changed ({type(before_val).__name__} -> {type(after_val).__name__})')
            else:
                print(f'  - {key}: unchanged')

    ##-----  do combined unit commitment/optimal power flow  -----

    ## check for sum(Pmin) > total load, decommit as necessary
    on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc["gen"]))  ## gens in service
    onld = find((ppc["gen"][:, GEN_STATUS] > 0) & isload(ppc["gen"]))  ## dispatchable loads in service
    load_capacity = sum(ppc["bus"][:, PD]) - sum(ppc["gen"][onld, PMIN])  ## total load capacity
    Pmin = ppc["gen"][on, PMIN]
    while sum(Pmin) > load_capacity:
        thermal_on = list(set(on) & set(ppc['thermal_ids']))
        if len(thermal_on) == 0:
            break
        ## shut down most expensive unit
        Pmin_thermal_on = ppc["gen"][thermal_on, PMIN]
        avgPmincost = totcost(ppc["gencost"][thermal_on, :], Pmin_thermal_on) / Pmin_thermal_on
        # _, i_to_close = fairmax(avgPmincost)  ## pick one with max avg cost at Pmin
        avgPmincost = list(avgPmincost)
        i_to_close = avgPmincost.index(max(avgPmincost))  ## pick one with max avg cost at Pmin
        i = thermal_on[i_to_close]  ## convert to generator index

        ## set generation to zero
        ppc["gen"][i, [PG, QG, GEN_STATUS, PMIN, PMAX]] = 0

        ## update minimum gen capacity
        on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc["gen"]))  ## gens in service
        Pmin = ppc["gen"][on, PMIN]
        close_hot[i] = 1
        # print('Shutting down generator %d.\n' % i)

    Pmax = ppc["gen"][on, PMAX]
    off = find((ppc["gen"][:, GEN_STATUS] == 0))
    while sum(Pmax) < 1.2 * load_capacity:
        thermal_off = list(set(off) & set(ppc['thermal_ids']))
        if len(thermal_off) == 0:
            break
        ## restart cheapest unit
        # Pmin_thermal_off = ppc["gen"][thermal_off, PMIN]
        # avgPmincost = totcost(ppc['gen'][thermal_off, :], Pmin_thermal_off) / Pmin_thermal_off
        avgPmincost = ppc['gencost'][:, STARTUP][thermal_off]
        # _, i_to_restart = fairmax(-avgPmincost)
        avgPmincost = list(avgPmincost)
        i_to_restart = avgPmincost.index(min(avgPmincost))
        i = thermal_off[i_to_restart]

        # restart
        ppc['gen'][i, PG] = ppc['min_gen_p'][i]
        ppc['gen'][i, PMAX] = ppc['min_gen_p'][i]
        ppc['gen'][i, PMIN] = ppc['min_gen_p'][i]
        ppc['gen'][i, GEN_STATUS] = 1

        on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc['gen']))
        off = find((ppc["gen"][:, GEN_STATUS] == 0))
        Pmax = ppc["gen"][on, PMAX]
        open_hot[i] = 1
        # print('restarting generator %d.\n' % i)

    ppopt = ppoption(VERBOSE=0, OUT_ALL=0, CONTINGENCY_AWARE=0)
    # import ipdb
    # ipdb.set_trace()
    # x = time.time()
    ppc_before_rundcopf = copy.deepcopy(ppc)
    result = rundcopf(ppc, ppopt,
                      # fname='opf.log'
                      )
    _log_ppc_value_changes(ppc_before_rundcopf, ppc, 'rundcopf (first call)')
    if not result['success']:
        ppopt = ppoption(VERBOSE=1, OUT_ALL=1, CONTINGENCY_AWARE=0)
        ppc_before_rundcopf_retry = copy.deepcopy(ppc)
        result = rundcopf(ppc, ppopt,
                          fname=f'dcopf_{ppc["network"]}.log'
                          )
        _log_ppc_value_changes(ppc_before_rundcopf_retry, ppc, 'rundcopf (retry)')
        print(f'----------------------DCOPF failed in {ppc["network"]}--------------------')
        import ipdb
        ipdb.set_trace()
    # print(f'contingency-aware dcopf cost {time.time()-x}s')
    # import ipdb
    # ipdb.set_trace()
    # if result['success'] == False:
    #     import ipdb
    #     ipdb.set_trace()
        # result = rundcopf(ppc, ppopt)
        # if result['success'] == False:
        #     import ipdb
        #     ipdb.set_trace()
    # print(f'on gens={ppc["gen"][:, GEN_STATUS].sum()}')
    return result


class Environment:
    '''
    Environment Initialization
    '''
    def __init__(self, network_ppc, reward_type="EPRIReward", is_test=False, two_player=False, attack_all=False, deterministic=False):
        self.network_ppc = network_ppc
        for k, v in network_ppc.items():
            self.network = k
            self.ppc = copy.deepcopy(v)

        if self.network in ['Texas2000', 'Western10000']:
            self.visualizer = Visualizer(self.ppc)

        self.num_gen = self.ppc['gen'].shape[0]
        self.num_bus = self.ppc['bus'].shape[0]
        self.num_line = self.ppc['branch'].shape[0]
        self.ppopt = ppoption(PF_DC=False, VERBOSE=0, OUT_ALL=0)


        self.reward_type = reward_type
        self.done = True
        self.action_space_cls = ActionSpace(self.ppc)
        self.is_test = is_test
        self.two_player = two_player
        self.attack_all = attack_all
        self.load_bus = np.nonzero(self.ppc['bus'][:, PD])[0].tolist()
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.deterministic = deterministic

        self.last_load_noises = np.ones(self.ppc['num_bus'])[self.load_bus]
        self.last_renewable_noises = np.ones(len(self.ppc['renewable_ids']))
        if deterministic:
            self.load_p_filepath = root_path + f'/data/{"test" if is_test else "train"}/{self.network}/load_p.csv'
            self.load_q_filepath = root_path + f'/data/{"test" if is_test else "train"}/{self.network}/load_q.csv'
            self.load_p_profiles = pd.read_csv(self.load_p_filepath).values
            self.load_q_profiles = pd.read_csv(self.load_q_filepath).values
            self.forecast_reader = ForecastReader(self.ppc, is_test=is_test)
            self.num_sample = self.load_p_profiles.shape[0]
            self.bus_areas = None
        else:
            self.load_path = root_path + f'/data/{"test" if is_test else "train"}/load.npy'
            self.load_profiles = np.load(self.load_path)
            self.solar_path = root_path + f'/data/{"test" if is_test else "train"}/solar.npy'
            self.solar_profiles = np.load(self.solar_path)
            self.wind_path = root_path + f'/data/{"test" if is_test else "train"}/wind.npy'
            self.wind_profiles = np.load(self.wind_path)
            self.num_sample = self.load_profiles.shape[0]
            area_codes, areas = pd.factorize(self.ppc['bus'][:, BUS_AREA].tolist())
            # TODO: currently only 28 area profiles, largerst grid requires 38
            self.bus_areas = np.random.choice(list([i for i in range(self.load_profiles.shape[1])]), len(areas),
                                              replace=len(areas)>self.load_profiles.shape[1])
            self.bus_areas = np.asarray([self.bus_areas[i] for i in area_codes.tolist()])
            self.ori_ppc = copy.deepcopy(self.ppc)
            self.renewable_masks = np.asarray([int(random.random() > 0.5) for _ in range(len(self.ppc['renewable_ids']))])     # 0-solar, 1-wind
            self.forecast_reader = ForecastReader(self.ppc, self.bus_areas, self.renewable_masks, is_test=is_test)


    def reset_attr(self):
        # Reset attr in the base env
        self.done = False
        self.timestep = 0
        self.gen_start_flag = np.zeros(self.num_gen)
        self.steps_to_recover_gen = np.zeros(self.num_gen, dtype=int)
        self.steps_to_close_gen = np.zeros(self.num_gen, dtype=int)
        self.steps_to_min_gen = -np.ones(self.num_gen, dtype=int)

        for k, v in self.network_ppc.items():
            self.network = k
            self.ppc = copy.deepcopy(v)


    def readdata(self, scenario_idx):
        if self.deterministic:
            self.ppc['bus'][self.load_bus, PD] = self.load_p_profiles[scenario_idx]
            self.ppc['bus'][self.load_bus, QD] = self.load_q_profiles[scenario_idx]
        else:
            self.ppc['bus'][self.load_bus, PD] = self.ori_ppc['bus'][self.load_bus, PD] * self.load_profiles[scenario_idx, self.bus_areas[self.load_bus]] * self.last_load_noises
            self.ppc['bus'][self.load_bus, QD] = self.ori_ppc['bus'][self.load_bus, QD] * self.load_profiles[scenario_idx, self.bus_areas[self.load_bus]] * self.last_load_noises

    def rerun_opf(self, nextstep_renewable_gen_p_max):
        # GEN_DATA
        ppc = copy.deepcopy(self.ppc)
        balanced_id = ppc['balanced_id']
        renewable_ids = ppc['renewable_ids']
        max_gen_p = ppc['max_gen_p']
        min_gen_p = ppc['min_gen_p']

        ppc['gen'][:, GEN_STATUS] = np.ones_like(ppc['gen'][:, GEN_STATUS])  # status

        bal_gen_p_mid = (min_gen_p[balanced_id] + max_gen_p[balanced_id]) / 2
        ppc['gen'][:, PMAX] = np.array(max_gen_p)
        redundancy = (max_gen_p[balanced_id] - min_gen_p[balanced_id]) / 2 * 0.6
        ppc['gen'][balanced_id, PMAX] = bal_gen_p_mid + redundancy

        print(f'deterministic: {self.deterministic}')
        ppc['gen'][renewable_ids, PMAX] = np.array(nextstep_renewable_gen_p_max)[:len(renewable_ids)] * 0.2
        ppc['gen'][renewable_ids, GEN_STATUS] = 1
        ppc['gen'][:, PMIN] = np.array(min_gen_p)
        ppc['gen'][renewable_ids, PMIN] = 0.0
        ppc['gen'][balanced_id, PMIN] = bal_gen_p_mid - redundancy
        # ppc['bus'][:, VM] = 1.0
        # ppc['bus'][:, VA] = 0.0
        # ppc['gen'][:, VG] = 1.05
        # for idx in ppc['gen'][:, GEN_BUS].astype(int).tolist():
        #     i = ppc['bus'][:, BUS_I].astype(int).tolist().index(idx)
        #     ppc['bus'][i, VM] = 1.05

        # for i in ppc['thermal_ids']:
        #     if random.random() < 0.8:
        #         ppc['gen'][i, GEN_STATUS] = 1
        #     else:
        #         ppc['gen'][i, [GEN_STATUS, PMIN, PMAX]] = 0.0

        result = run_uopf(ppc)
        self.ppc['bus'] = result['bus']
        self.ppc['gen'] = result['gen']
        self.ppc['branch'] = result['branch']
        # print(f'lower than min={np.where(self.ppc["gen"][:, PG]<self.ppc["gen"][:, PMIN])}')
        # print(f'larger than max={np.where(self.ppc["gen"][:, PG]>self.ppc["gen"][:, PMAX])}')
        self.ppc['gen'][:, PG] = self.ppc['gen'][:, PG].clip(self.ppc['gen'][:, PMIN], self.ppc['gen'][:, PMAX])
        # return self.ppc

    # @profile
    def power_flow(self):
        # print(f'sample_idx={self.sample_idx}, p_sum={sum(self.ppc["gen"][:, PG])}, d_sum={sum(self.ppc["bus"][self.load_bus, PD])}')
        # self.ppc['gen'][self.ppc['renewable_ids'], PMAX] = self.ppc['gen'][self.ppc['renewable_ids'], PG]
        # self.ppc['gen'][self.ppc['renewable_ids'], PMIN] = self.ppc['gen'][self.ppc['renewable_ids'], PG]
        # self.ppc['gen'][self.ppc['thermal_ids'], PMAX] = self.ppc['gen'][self.ppc['thermal_ids'], PG]
        # self.ppc['gen'][self.ppc['thermal_ids'], PMIN] = self.ppc['gen'][self.ppc['thermal_ids'], PG]
        # diff_p = sum(self.ppc['bus'][:, PD]) - sum(self.ppc['gen'][:, PG])
        # import ipdb
        # ipdb.set_trace()
        # if abs(diff_p) > 100:
        #     self.ppc['gen'][self.ppc['balanced_id'], PG] += diff_p
        # self.ppc['gen'][self.ppc["balanced_id"], PMAX] = self.ppc['gen'][self.ppc["balanced_id"], PG]
        # self.ppc['gen'][self.ppc["balanced_id"], PMIN] = self.ppc['gen'][self.ppc["balanced_id"], PG]

        # import ipdb
        # ipdb.set_trace()
        # ppc = ext2int(self.ppc)
        # Ybus, _, _ = makeYbus(ppc['baseMVA'], ppc['bus'], ppc['branch'])
        # print(f'Ybus diag real min={np.diag(Ybus.toarray()).real.min()}, Ybus diag imag max={np.diag(Ybus.toarray()).imag.max()}')
        ppopt = ppoption(PF_DC=False, VERBOSE=0, OUT_ALL=0,
                         # PF_MAX_IT=50, PF_TOL=1e-6
                         )
        ppc = copy.deepcopy(self.ppc)
        # ppc['bus'][:, VM] = 1.0
        # ppc['bus'][:, VA] = 0.0
        # ppc['gen'][:, VM] = 1.05
        result, success = runpf(ppc, ppopt)
        if success:
            self.ppc['bus'] = result['bus']
            self.ppc['gen'] = result['gen']
            self.ppc['branch'] = result['branch']
        else:
            ppopt = ppoption(PF_DC=False, VERBOSE=1, OUT_ALL=1)
            ppc = copy.deepcopy(self.ppc)
            # ppc['bus'][:, VM] = 1.0
            # ppc['bus'][:, VA] = 0.0
            result, success = runpf(ppc, ppopt, fname=f'pf_{self.network}.log')
            print(f'----------------------ACPF failed in {self.network}--------------------')
            # import ipdb
            # ipdb.set_trace()
        balanced_id = self.ppc['balanced_id']
        if self.ppc['gen'][balanced_id, PG] > self.ppc['max_gen_p'][balanced_id] or \
                self.ppc['gen'][balanced_id, PG] < self.ppc['min_gen_p'][balanced_id]:
            # import ipdb
            # ipdb.set_trace()
            return False, f'balanced_gen_p out of limit {self.ppc["gen"][balanced_id, PG]:.3f}, ' \
                          f'max={self.ppc["max_gen_p"][balanced_id]:.3f},' \
                          f'min={self.ppc["min_gen_p"][balanced_id]:.3f}'
        # self.ppc['gen'][:, PG] = self.ppc['gen'][:, PG].clip(self.ppc['gen'][:, PMIN], self.ppc['gen'][:, PMAX])
        return success, 'power flow not converged' if not success else ' '

    def update_ppc_from_obs(self, obs):
        self.ppc['gen'][:, PG] = obs.gen_p
        self.ppc['gen'][:, QG] = obs.gen_q
        self.ppc['gen'][:, VG] = obs.gen_v
        # self.ppc['gen'][self.ppc['renewable_ids'], PMAX] = obs.nextstep_renewable_gen_p_max
        # self.ppc['gen'][self.ppc['renewable_ids'], PMIN] = 0.0
        # self.ppc['gen'][self.ppc['thermal_ids, PMAX] = obs.gen_p[self.ppc['thermal_ids] + obs.action_space['adjust_gen_p'].high[self.ppc['thermal_ids]
        # self.ppc['gen'][self.ppc['thermal_ids, PMAX] = obs.gen_p[self.ppc['thermal_ids] + obs.action_space['adjust_gen_p'].low[self.ppc['thermal_ids]
        self.ppc['bus'][self.load_bus, PD] = obs.nextstep_load_p
        self.ppc['bus'][self.load_bus, QD] = obs.load_q
        self.ppc['bus'][:, VM] = obs.bus_v
        self.ppc['bus'][:, VA] = obs.bus_ang
        self.ppc['branch'][:, BR_STATUS] = obs.line_status


    def reset(self, seed=None, start_sample_idx=None, ppc=None):
        # Reset states and attributes
        self.reset_attr()

        # Instead of using `np.random` use `self.np_random`.
        # It won't be affected when user using `np.random`.
        self.np_random = np.random.RandomState()
        if seed is not None:
            self.np_random.seed(seed=seed)
            np.random.seed(seed=seed)

        self.disconnect = Disconnect(self.np_random, self.ppc)

        if start_sample_idx is not None:
            self.sample_idx = start_sample_idx
        else:
            self.sample_idx = self.np_random.randint(0, self.num_sample - 300)
        assert self.num_sample > self.sample_idx >= 0

        # Read self.sample_idx timestep data
        self.readdata(self.sample_idx)
        # print(f'gen_p_2={self.ppc["gen"][:, PG].sum()}, load_p_2={self.ppc["bus"][:, PD].sum()}, sample_idx={self.sample_idx}')

        if ppc is not None:
            self.ppc['bus'][self.load_bus, PD] = ppc['bus'][self.load_bus, PD]
            self.ppc['bus'][self.load_bus, QD] = ppc['bus'][self.load_bus, QD]
            # self.ppc['bus'] = ppc['bus']
            # self.ppc['gen'] = ppc['gen']
            # self.ppc['branch'] = ppc['branch']

        # Update forecast value
        # print(f'reset, prev_renewable_noise_mean={self.last_renewable_noises.mean()}')
        curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max, self.last_renewable_noises = \
            self.forecast_reader.read_step_renewable_gen_p_max(self.sample_idx, self.last_renewable_noises)
        if ppc is not None:
            curstep_renewable_gen_p_max = ppc['curstep_renewable_gen_p_max']
            nextstep_renewable_gen_p_max = ppc['nextstep_renewable_gen_p_max']
        # print(f'reset, after_renewable_noise_mean={self.last_renewable_noises.mean()}')
        self.rerun_opf(nextstep_renewable_gen_p_max)
        # print(f'gen_p_2.5={self.ppc["gen"][:, PG].sum()}, load_p_2.5={self.ppc["bus"][:, PD].sum()}')
        rounded_gen_p = self._round_p(self.ppc['gen'][:, PG])
        self._update_gen_status(self.ppc['gen'][:, PG], is_reset=True)
        self._check_gen_status(self.ppc['gen'][:, PG], rounded_gen_p)

        self.last_injection_gen_p = copy.deepcopy(self.ppc['gen'][:, PG])
        rho, v_or, v_ex, a_or, a_ex = self._calc_rho_v2()

        # Update forecast value
        # curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max = \
        #     self.forecast_reader.read_step_renewable_gen_p_max(self.sample_idx)
        nextstep_load_p, self.last_load_noises = self.forecast_reader.read_step_load_p(self.sample_idx)
        if ppc is not None:
            nextstep_load_p = ppc['nextstep_load_p']
        action_space = self.action_space_cls.update(self.ppc, self.steps_to_recover_gen, self.steps_to_close_gen, self.steps_to_min_gen,
                                                     rounded_gen_p, nextstep_renewable_gen_p_max)

        future_renewable_gen_p_max = self.forecast_reader.read_Xstep_renewable_gen_p_max(self.sample_idx, self.ppc['renewable_forecast_horizon'])
        future_renewable_gen_p_max = np.array(future_renewable_gen_p_max).sum(-1).tolist()
        future_load_p = self.forecast_reader.read_Xstep_load_p(self.sample_idx, self.ppc['load_forecast_horizon'])
        future_load_p = np.array(future_load_p).sum(-1).tolist()
        self.obs = Observation(
            ppc=self.ppc, load_bus=self.load_bus, timestep=self.timestep, action_space=action_space,
            steps_to_reconnect_line=self.disconnect.steps_to_reconnect_line,
            count_soft_overflow_steps=self.disconnect.count_soft_overflow_steps, rho=rho,
            v_or=v_or, v_ex=v_ex, a_or=a_or, a_ex=a_ex,
            steps_to_recover_gen=self.steps_to_recover_gen,
            steps_to_close_gen=self.steps_to_close_gen,
            steps_to_min_gen=self.steps_to_min_gen,
            gen_start_flag=self.gen_start_flag,
            curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
            nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
            rounded_gen_p=rounded_gen_p,
            nextstep_load_p=nextstep_load_p,
            future_renewable_gen_p_max=future_renewable_gen_p_max,
            future_load_p=future_load_p
        )
        # print(f'PD={sum(self.ppc["bus"][:, PD])}, PG_after_reset={sum(self.ppc["gen"][:, PG])}, result_PG={sum(self.ppc["gen"][:, PG])}')
        return copy.deepcopy(self.obs)

    def load_and_step(self, snapshot, act):
        if self.done:
            raise Exception("The env is game over, please reset.")
        last_obs, sample_idx, gen_start_flag, steps_to_recover_gen, steps_to_close_gen, steps_to_min_gen = snapshot
        timestep = last_obs.timestep
        self.update_ppc_from_obs(last_obs)
        self.disconnect.load_from_obs(last_obs)
        self.gen_start_flag = gen_start_flag
        self.steps_to_recover_gen = steps_to_recover_gen
        self.steps_to_close_gen = steps_to_close_gen
        self.steps_to_min_gen = steps_to_min_gen

        self._check_action(act)
        act['adjust_gen_p'] = self._round_p(act['adjust_gen_p'])

        # Compute the injection value
        self.ppc['gen'][:, PG] = last_obs.gen_p + act['adjust_gen_p']
        # self.ppc['gen'][:, VG] = last_obs.gen_v + act['adjust_gen_v']

        # Judge the legality of the action
        legal_flag, fail_info = is_legal(act, last_obs, self.ppc)
        if not legal_flag:
            done = True
            new_obs, reward, done, info = self.return_res(fail_info, done)
            return (new_obs, sample_idx, self.gen_start_flag, self.steps_to_recover_gen, self.steps_to_close_gen, self.steps_to_min_gen), reward, done, info

        self.ppc['branch'][:, BR_STATUS] = 1
        disc_ids, steps_to_reconnect_line, count_soft_overflow_steps = self.disconnect.get_disc_line(
            last_obs, attack_all=self.attack_all, two_player=self.two_player, attack=None
        )
        self.ppc['branch'][disc_ids, BR_STATUS] = 0

        sample_idx += 1
        timestep += 1

        # Examine if exceeding historical scenarios limit
        if sample_idx >= self.num_sample - 288:
            self.done = True
            return self.return_res('Exceeding sample limit', self.done)

        # Read the power data of the next step from .csv file
        self.readdata(sample_idx)

        self._injection_auto_mapping()

        # Update generator running status
        self._update_gen_status(self.ppc['gen'][:, PG])

        # Power flow calculation
        sucess, info = self.power_flow()
        if not sucess:
            self.done = True
            return self.return_res(info, self.done)

        rounded_gen_p = self._round_p(self.ppc['gen'][:, PG])
        # print(np.asarray(rounded_gen_p)-np.asarray(injection_gen_p))

        self._check_gen_status(self.ppc['gen'][:, PG], rounded_gen_p)
        self.last_injection_gen_p = copy.deepcopy(self.ppc['gen'][:, PG])

        # Update forecast value
        curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max, self.last_renewable_noises = \
            self.forecast_reader.read_step_renewable_gen_p_max(sample_idx, self.last_renewable_noises)
        nextstep_load_p, self.last_load_noises = self.forecast_reader.read_step_load_p(sample_idx)

        future_renewable_gen_p_max = self.forecast_reader.read_Xstep_renewable_gen_p_max(self.sample_idx, self.ppc[
            'renewable_forecast_horizon'])
        future_renewable_gen_p_max = np.array(future_renewable_gen_p_max).sum(-1).tolist()
        future_load_p = self.forecast_reader.read_Xstep_load_p(self.sample_idx, self.ppc['load_forecast_horizon'])
        future_load_p = np.array(future_load_p).sum(-1).tolist()

        action_space = self.action_space_cls.update(self.ppc, steps_to_recover_gen, steps_to_close_gen, self.steps_to_min_gen,
                                                    rounded_gen_p, nextstep_renewable_gen_p_max)

        rho, v_or, v_ex, a_or, a_ex = self._calc_rho_v2()

        # pack obs
        self.obs = Observation(
            ppc=self.ppc, load_bus=self.load_bus, timestep=self.timestep, action_space=action_space,
            steps_to_reconnect_line=self.disconnect.steps_to_reconnect_line,
            count_soft_overflow_steps=self.disconnect.count_soft_overflow_steps, rho=rho,
            v_or=v_or, v_ex=v_ex, a_or=a_or, a_ex=a_ex,
            steps_to_recover_gen=self.steps_to_recover_gen,
            steps_to_close_gen=self.steps_to_close_gen,
            steps_to_min_gen=self.steps_to_min_gen,
            gen_start_flag=self.gen_start_flag,
            curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
            nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
            rounded_gen_p=rounded_gen_p,
            nextstep_load_p=nextstep_load_p,
            future_renewable_gen_p_max=future_renewable_gen_p_max,
            future_load_p=future_load_p
        )
        self.reward = self.get_reward(self.obs, last_obs)
        new_obs, reward, done, info = self.return_res()
        new_snapshot = (new_obs, sample_idx,
                        self.gen_start_flag, self.steps_to_recover_gen, self.steps_to_close_gen, self.steps_to_min_gen
                        )
        return new_snapshot, reward, done, info

    def step_only_attack(self, act):
        # print(f'sample_idx={self.sample_idx}')
        if self.done:
            raise Exception("The env is game over, please reset.")
        last_obs = copy.deepcopy(self.obs)

        self.ppc['branch'][:, BR_STATUS] = 1
        disc_ids = self.disconnect.get_disc_line(
            last_obs.rho, attack=act, attack_all=self.attack_all, two_player=self.two_player)
        self.ppc['branch'][disc_ids, BR_STATUS] = 0

        # Power flow calculation
        sucess, info = self.power_flow()
        if not sucess:
            self.done = True
            return self.return_res(info, self.done)

        rounded_gen_p = self._round_p(self.ppc['gen'][:, PG])

        self._check_gen_status(self.ppc['gen'][:, PG], rounded_gen_p)
        self.last_injection_gen_p = copy.deepcopy(self.ppc['gen'][:, PG])

        # Update forecast value
        curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max, self.last_renewable_noises = \
            self.forecast_reader.read_step_renewable_gen_p_max(self.sample_idxs, self.last_renewable_noises)
        nextstep_load_p, self.last_load_noises = self.forecast_reader.read_step_load_p(self.sample_idx)

        future_renewable_gen_p_max = self.forecast_reader.read_Xstep_renewable_gen_p_max(self.sample_idx, self.ppc[
            'renewable_forecast_horizon'])
        future_renewable_gen_p_max = np.array(future_renewable_gen_p_max).sum(-1).tolist()
        future_load_p = self.forecast_reader.read_Xstep_load_p(self.sample_idx, self.ppc['load_forecast_horizon'])
        future_load_p = np.array(future_load_p).sum(-1).tolist()

        action_space = self.action_space_cls.update(self.ppc, self.steps_to_recover_gen, self.steps_to_close_gen, self.steps_to_min_gen,
                                                     rounded_gen_p, nextstep_renewable_gen_p_max)

        rho, v_or, v_ex, a_or, a_ex = self._calc_rho_v2()

        # pack obs
        self.obs = Observation(
            ppc=self.ppc, load_bus=self.load_bus, timestep=self.timestep, action_space=action_space,
            steps_to_reconnect_line=self.disconnect.steps_to_reconnect_line,
            count_soft_overflow_steps=self.disconnect.count_soft_overflow_steps, rho=rho,
            v_or=v_or, v_ex=v_ex, a_or=a_or, a_ex=a_ex,
            steps_to_recover_gen=self.steps_to_recover_gen,
            steps_to_close_gen=self.steps_to_close_gen,
            steps_to_min_gen=self.steps_to_min_gen,
            gen_start_flag=self.gen_start_flag,
            curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
            nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
            rounded_gen_p=rounded_gen_p,
            nextstep_load_p=nextstep_load_p,
            future_renewable_gen_p_max=future_renewable_gen_p_max,
            future_load_p=future_load_p
        )
        self.reward = self.get_reward(self.obs, last_obs)
        return self.return_res()

    def step(self, act, ppc=None):
        print(f'in service line num={int(self.ppc["branch"][:, BR_STATUS].sum())}')
        # print(f'sample_idx={self.sample_idx}')
        if self.done:
            raise Exception("The env is game over, please reset.")
        last_obs = copy.deepcopy(self.obs)

        self._check_action(act)
        act['adjust_gen_p'] = self._round_p(act['adjust_gen_p'])

        # Compute the injection action
        self.ppc['gen'][:, PG] += act['adjust_gen_p']   # actual power adjustion
        # self.ppc['gen'][:, VG] += act['adjust_gen_v']   # generator voltage adjust
        # print(f'gen_p_3={self.ppc["gen"][:, PG].sum()}, load_p_3={self.ppc["bus"][:, PD].sum()}')

        # Check the legality of the action
        legal_flag, fail_info = is_legal(act, last_obs, self.ppc)
        if not legal_flag:
            self.done = True
            return self.return_res(fail_info, self.done)

        # Set random line attack & update line status
        self.ppc['branch'][:, BR_STATUS] = 1
        disc_ids = self.disconnect.get_disc_line(
            last_obs.rho, attack_all=self.attack_all, two_player=self.two_player, attack=None)
        # self.ppc['branch'][disc_ids, BR_STATUS] = 0     # N-1 enabled

        self.sample_idx += 1
        self.timestep += 1

        # Examine if exceeding historical scenarios limit
        if self.sample_idx >= self.num_sample - 288:
            self.done = True
            return self.return_res('Exceeding sample limit', self.done)

        # Read the power data of the next step
        self.readdata(self.sample_idx)
        if ppc is not None:
            self.ppc['bus'][self.load_bus, PD] = ppc['nextstep_load_p']

        # Map restart or close thermal dispatching
        self._injection_auto_mapping()

        # Update generator running status
        self._update_gen_status(self.ppc['gen'][:, PG])

        # Power flow calculation
        sucess, info = self.power_flow()
        if not sucess:
            self.done = True
            return self.return_res(info, self.done)

        rounded_gen_p = self._round_p(self.ppc['gen'][:, PG])
        # print(f'gen_p_4={self.ppc["gen"][:, PG].sum()}, load_p_4={self.ppc["bus"][:, PD].sum()}')
        self._check_gen_status(self.ppc['gen'][:, PG], rounded_gen_p)
        self.last_injection_gen_p = copy.deepcopy(self.ppc['gen'][:, PG])

        # Update forecast value
        # print(f'step, prev_renewable_noise_mean={self.last_renewable_noises.mean()}')
        curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max, self.last_renewable_noises = \
            self.forecast_reader.read_step_renewable_gen_p_max(self.sample_idx, self.last_renewable_noises)
        # print(f'step, after_renewable_noise_mean={self.last_renewable_noises.mean()}')
        nextstep_load_p, self.last_load_noises = self.forecast_reader.read_step_load_p(self.sample_idx)

        future_renewable_gen_p_max = self.forecast_reader.read_Xstep_renewable_gen_p_max(self.sample_idx, self.ppc['renewable_forecast_horizon'])
        future_renewable_gen_p_max = np.array(future_renewable_gen_p_max).sum(-1).tolist()
        future_load_p = self.forecast_reader.read_Xstep_load_p(self.sample_idx, self.ppc['load_forecast_horizon'])
        future_load_p = np.array(future_load_p).sum(-1).tolist()

        # Update next-step action space
        action_space = self.action_space_cls.update(self.ppc, self.steps_to_recover_gen, self.steps_to_close_gen, self.steps_to_min_gen,
                                                     rounded_gen_p, nextstep_renewable_gen_p_max)

        # Calculate line status
        rho, v_or, v_ex, a_or, a_ex = self._calc_rho_v2()

        # Pack obs
        self.obs = Observation(
            ppc=self.ppc, load_bus=self.load_bus, timestep=self.timestep, action_space=action_space,
            steps_to_reconnect_line=self.disconnect.steps_to_reconnect_line,
            count_soft_overflow_steps=self.disconnect.count_soft_overflow_steps, rho=rho,
            v_or=v_or, v_ex=v_ex, a_or=a_or, a_ex=a_ex,
            steps_to_recover_gen=self.steps_to_recover_gen,
            steps_to_close_gen=self.steps_to_close_gen,
            steps_to_min_gen=self.steps_to_min_gen,
            gen_start_flag=self.gen_start_flag,
            curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
            nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
            rounded_gen_p=rounded_gen_p,
            nextstep_load_p=nextstep_load_p,
            future_renewable_gen_p_max=future_renewable_gen_p_max,
            future_load_p=future_load_p
        )
        # if sum(self.obs.gen_p[self.ppc['renewable_ids']]) - sum(self.obs.curstep_renewable_gen_p_max) > 1.0:
        #     print('renewable exceeds')
        self.reward = self.get_reward(self.obs, last_obs)
        return self.return_res()

    def get_snapshot(self):
        return (self.obs, self.sample_idx)

    def _calc_rho_v2(self):
        limit = self.ppc['branch'][:, RATE_A]
        num_line = min(self.ppc['num_line'], len(limit))
        p_or = self.ppc['branch'][:, PF]
        p_ex = self.ppc['branch'][:, PT]
        q_or = self.ppc['branch'][:, QF]
        q_ex = self.ppc['branch'][:, QT]
        s_or = p_or + 1j * q_or
        s_ex = p_ex + 1j * q_ex
        f_bus = (self.ppc['branch'][:, F_BUS] - 1).astype(int).tolist()
        t_bus = (self.ppc['branch'][:, T_BUS] - 1).astype(int).tolist()
        _rho = [None] * num_line
        v_or, v_ex, a_or, a_ex = [], [], [], []
        for i in range(num_line):
            f_idx = np.where(self.ppc['bus'][:, BUS_I] == f_bus[i] + 1)[0][0]
            f_v_mag = self.ppc['bus'][f_idx, VM]
            f_v_ang = self.ppc['bus'][f_idx, VA]
            t_idx = np.where(self.ppc['bus'][:, BUS_I] == t_bus[i] + 1)[0][0]
            t_v_mag = self.ppc['bus'][t_idx, VM]
            t_v_ang = self.ppc['bus'][t_idx, VA]
            v_or_comp = f_v_mag * np.exp(1j * np.deg2rad(f_v_ang))
            v_ex_comp = t_v_mag * np.exp(1j * np.deg2rad(t_v_ang))
            v_or.append(np.abs(v_or_comp))
            v_ex.append(np.abs(v_ex_comp))
            a_or.append(np.abs(s_or[i] / v_or_comp))
            a_ex.append(np.abs(s_ex[i] / v_ex_comp))
            _rho[i] = max(a_or[i], a_ex[i]) / (limit[i] + 1e-3)
        return _rho, v_or, v_ex, a_or, a_ex

    def _injection_auto_mapping(self):
        """
        based on the last injection q, map the value of injection_gen_p
        from (0, min_gen_p) to 0/min_gen_p
        """
        for i in self.ppc['thermal_ids']:
            if self.ppc['gen'][i, PG] >= 0 and self.ppc['gen'][i, PG] < self.ppc['min_gen_p'][i]:
                if (self.last_injection_gen_p[i] - self.ppc['min_gen_p'][i]) < 1e-3:
                    self.ppc['gen'][i, PG] = 0.0  # close the generator
                elif self.last_injection_gen_p[i] - self.ppc['min_gen_p'][i] >= 1e-3:
                    self.ppc['gen'][i, PG] = self.ppc['min_gen_p'][i]  # mapped to the min_gen_p
                elif abs(self.last_injection_gen_p[i]) < 1e-3:
                    if i in self.ppc['fast_thermal_gen']:
                        self.ppc['gen'][i, PG] = self.ppc['min_gen_p'][i]  # open the generator
                    else:
                        self.ppc['gen'][i, PG] = 0.0
                else:
                    assert False  # should never in (0, min_gen_p)

    def _injection_auto_mapping_v2(self, inject_gen_p):
        """
        based on the last injection q, map the value of injection_gen_p
        from (0, min_gen_p) to 0/min_gen_p
        """
        injection_gen_p = inject_gen_p
        for i in self.ppc['thermal_ids']:
            if injection_gen_p[i] > 0 and injection_gen_p[i] < self.ppc['min_gen_p'][i]:
                if self.last_injection_gen_p[i] > self.ppc['min_gen_p'][i]:
                    # if injection_gen_p[i] < self.ppc['min_gen_p[]i]:
                    injection_gen_p[i] = self.ppc['min_gen_p'][i]
                elif abs(self.last_injection_gen_p[i] - self.ppc['min_gen_p'][i]) < 1e-3:
                    if injection_gen_p[i] < self.ppc['min_gen_p'][i]/2:
                        injection_gen_p[i] = 0.0
                    else:
                        injection_gen_p[i] = self.ppc['min_gen_p'][i]
                elif abs(self.last_injection_gen_p[i]) < 1e-3:
                    if injection_gen_p[i] > self.ppc['min_gen_p'][i]/2:
                        injection_gen_p[i] = self.ppc['min_gen_p'][i]
                    else:
                        injection_gen_p[i] = 0.0
        return injection_gen_p

    def _update_gen_status(self, injection_gen_p, is_reset=False):
        cnt = 0
        for i in self.ppc['thermal_ids']:
            if abs(injection_gen_p[i]) < 1e-3:
                if self.ppc['gen'][i, GEN_STATUS] == 1:  # the generator is open
                    assert self.steps_to_close_gen[i] == 0
                    self.ppc['gen'][i, GEN_STATUS] = 0  # close the generator
                    self.ppc['gen'][i, PG] = 0.0
                    self.ppc['gen'][i, PMAX] = 0.0
                    self.ppc['gen'][i, PMIN] = 0.0
                    if not is_reset:
                        self.steps_to_recover_gen[i] = self.ppc['max_steps_to_recover_gen'][i]
            elif abs(injection_gen_p[i] - self.ppc['min_gen_p'][i]) < 1e-3:
                if self.ppc['gen'][i, GEN_STATUS] == 0:  # the generator is shutdown
                    assert self.steps_to_recover_gen[i] == 0  # action isLegal function should have checked
                    if self.gen_start_flag[i] == 0:
                        if not is_reset:
                            cnt += 1
                            self.gen_start_flag[i] = 1
                            self.steps_to_min_gen[i] = self.ppc['thermal_start_response_steps']
                            injection_gen_p[i] = 0

            if self.steps_to_recover_gen[i] > 0:
                self.steps_to_recover_gen[i] -= 1  # update recover timesteps counter
            if self.steps_to_close_gen[i] > 0:
                self.steps_to_close_gen[i] -= 1  # update close timesteps counter

            if self.gen_start_flag[i] == 1:
                if self.steps_to_min_gen[i] > 0:
                    self.steps_to_min_gen[i] -= 1
                    injection_gen_p[i] = 0
                elif self.steps_to_min_gen[i] == 0:
                    self.gen_start_flag[i] = 0
                    self.ppc['gen'][i, GEN_STATUS] = 1
                    self.ppc['gen'][i, PG] = self.ppc['min_gen_p'][i]
                    self.ppc['gen'][i, PMAX] = self.ppc['min_gen_p'][i]
                    self.ppc['gen'][i, PMIN] = self.ppc['min_gen_p'][i]
                    self.steps_to_close_gen[i] = self.ppc['max_steps_to_close_gen'][i]
                    injection_gen_p[i] = self.ppc['min_gen_p'][i]
                    self.steps_to_min_gen[i] = -1

        if cnt > 1:
            print(f'multiple restarts, num={cnt}')


    def __update_gen_status(self, injection_gen_p, gen_status, steps_to_recover_gen, steps_to_close_gen):
        new_gen_status = copy.deepcopy(gen_status)
        new_steps_to_recover_gen = copy.deepcopy(steps_to_recover_gen)
        new_steps_to_close_gen = copy.deepcopy(steps_to_close_gen)
        for i in self.ppc['thermal_ids']:
            if abs(injection_gen_p[i]) < 1e-3:
                if gen_status[i] == 1:  # the generator is open
                    assert steps_to_close_gen[i] == 0
                    new_gen_status[i] = 0  # close the generator
                    new_steps_to_recover_gen[i] = self.ppc['max_steps_to_recover_gen'][i]
            elif abs(injection_gen_p[i] - self.ppc['min_gen_p'][i]) < 1e-3:
                if gen_status[i] == 0:  # the generator is shutdown
                    assert steps_to_recover_gen[i] == 0  # action isLegal function should have checked
                    new_gen_status[i] = 1  # open the generator
                    new_steps_to_close_gen[i] = self.ppc['max_steps_to_close_gen'][i]

            if steps_to_recover_gen[i] > 0:
                new_steps_to_recover_gen[i] -= 1  # update recover timesteps counter
            if steps_to_close_gen[i] > 0:
                new_steps_to_close_gen[i] -= 1  # update close timesteps counter
        return new_gen_status, new_steps_to_recover_gen, new_steps_to_close_gen

    def _check_gen_status(self, injection_gen_p, rounded_gen_p):
        # check gen_p value of thermal generators after calling power flow
        for i in self.ppc['thermal_ids']:
            # if self.ppc['gen'][i, GEN_STATUS] == 0 and self.gen_start_flag[i] == 0:
            if self.ppc['gen'][i, GEN_STATUS] == 0:
                assert abs(rounded_gen_p[i]) < 1e-3, (i, rounded_gen_p[i], self.ppc['min_gen_p'][i])
            else:
                if rounded_gen_p[i] < self.ppc['min_gen_p'][i] - 1e-3:
                    print('opened gen < min_gen_p')
                assert rounded_gen_p[i] >= self.ppc['min_gen_p'][i] - 1e-3, (i, rounded_gen_p[i], self.ppc['min_gen_p'][i])

            assert abs(injection_gen_p[i] - rounded_gen_p[i]) <= self.ppc['env_allow_precision'], (i, injection_gen_p[i], rounded_gen_p[i])

    def __check_gen_status(self, injection_gen_p, rounded_gen_p, gen_status):
        # check gen_p value of thermal generators after calling power flow
        for i in self.ppc['thermal_ids']:
            if gen_status[i] == 0:
                assert abs(rounded_gen_p[i]) < 1e-3
            else:
                # print(gen_status[i], rounded_gen_p[i], self.ppc['min_gen_p[]i])
                assert rounded_gen_p[i] >= self.ppc['min_gen_p'][i], (i, rounded_gen_p[i], self.ppc['min_gen_p'][i])

            assert abs(injection_gen_p[i] - rounded_gen_p[i]) <= self.ppc['env_allow_precision'], (i, injection_gen_p[i], rounded_gen_p[i])

        for i in self.ppc['renewable_ids']:
            assert abs(injection_gen_p[i] - rounded_gen_p[i]) <= self.ppc['env_allow_precision'], (i, injection_gen_p[i], rounded_gen_p[i])

    def _check_action(self, act):
        assert 'adjust_gen_p' in act
        assert 'adjust_gen_v' in act

        adjust_gen_p = act['adjust_gen_p']
        adjust_gen_v = act['adjust_gen_v']

        assert isinstance(adjust_gen_p, (list, tuple, np.ndarray))
        assert len(adjust_gen_p) == self.ppc['num_gen']

        assert isinstance(adjust_gen_v, (list, tuple, np.ndarray))
        assert len(adjust_gen_v) == self.ppc['num_gen']

    def _round_p(self, p):
        return [round(x, self.ppc['keep_decimal_digits']) for x in p]

    def get_reward(self, obs, last_obs):
        reward_dict = {
            "EPRIReward": EPRIReward,
            "line_over_flow_reward": line_over_flow_reward,
            "renewable_consumption_reward": renewable_consumption_reward,
            "running_cost_reward": running_cost_reward,
            "balanced_gen_reward": balanced_gen_reward,
            "gen_reactive_power_reward": gen_reactive_power_reward,
            "sub_voltage_reward": sub_voltage_reward,
        }
        reward_func = reward_dict[self.reward_type]
        return reward_func(obs, last_obs, self.ppc)

    def return_res(self, info=None, done=False):
        ret_obs = copy.deepcopy(self.obs)
        if done:
            if not info:
                return ret_obs, -10, True, {}
            else:
                return ret_obs, -10, True, {'fail_info': info}
        else:
            assert self.reward, "the reward are not calculated yet"
            return ret_obs, self.reward, False, {}
