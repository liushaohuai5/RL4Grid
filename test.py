import RL4Grid
from RL4Grid.utilize.form_action import form_action
import numpy as np
from pypower.api import *
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
import time
from pypower.opf_args import opf_args2
from pypower.ppoption import ppoption
from pypower.isload import isload
from pypower.totcost import totcost
from pypower.fairmax import fairmax
from numpy import flatnonzero as find
import copy



def run_uopf(ppc):
    open_hot = np.zeros(ppc['num_gen'] + 1)
    close_hot = np.zeros(ppc['num_gen'] + 1)
    t0 = time.time()  ## start timer

    ##-----  do combined unit commitment/optimal power flow  -----

    ## check for sum(Pmin) > total load, decommit as necessary
    on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc["gen"]))  ## gens in service
    onld = find((ppc["gen"][:, GEN_STATUS] > 0) & isload(ppc["gen"]))  ## disp loads in serv
    load_capacity = sum(ppc["bus"][:, PD]) - sum(ppc["gen"][onld, PMIN])  ## total load capacity
    Pmin = ppc["gen"][on, PMIN]
    Pmax = ppc["gen"][on, PMAX]
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
        print('test, Shutting down generator %d.\n' % i)

    off = find((ppc["gen"][:, GEN_STATUS] == 0))
    while sum(Pmax) < 1.2 * load_capacity:
        thermal_off = list(set(off) & set(ppc['thermal_ids']))
        if len(thermal_off) == 0:
            break
        ## restart cheapest unit
        # Pmin_thermal_off = ppc["gen"][thermal_off, PMIN]
        # avgPmincost = totcost(ppc['gen'][thermal_off, :], Pmin_thermal_off) / Pmin_thermal_off
        avgPmincost = np.array(ppc['gencost'][:, STARTUP])[thermal_off]
        # _, i_to_restart = fairmax(-avgPmincost)
        avgPmincost = list(avgPmincost)
        i_to_restart = avgPmincost.index(min(avgPmincost))
        i = thermal_off[i_to_restart]

        # restart
        ppc['gen'][i, [PG, PMIN, PMAX]] = ppc['min_gen_p'][i]
        ppc['gen'][i, GEN_STATUS] = 1

        on = find((ppc["gen"][:, GEN_STATUS] > 0) & ~isload(ppc['gen']))
        off = find((ppc["gen"][:, GEN_STATUS] == 0))
        Pmax = ppc["gen"][on, PMAX]
        open_hot[i] = 1
        print('test, restarting generator %d.\n' % i)

    ## run initial opf
    ppopt = ppoption(VERBOSE=0, OUT_ALL=0, CONTINGENCY_AWARE=0)
    results = rundcopf(ppc, ppopt,
                       # fname='opf.log'
                       )
    # if results['success'] == False:
    #     import ipdb
    #     ipdb.set_trace()

    ## compute elapsed time
    et = time.time() - t0

    return results, open_hot, close_hot


def rerun_opf(observation, ppc):
    bus_num = ppc['num_bus']
    gen_num = ppc['num_gen']
    load_num = ppc['num_load']
    renewable_num = len(ppc['renewable_ids'])

    gen_bus_lst = ppc['gen'][:, GEN_BUS].tolist()
    gen2busM = np.zeros((gen_num, bus_num))
    for i, bus in enumerate(gen_bus_lst):
        idx = ppc['bus'][:, BUS_I].tolist().index(bus)
        gen2busM[i, idx] = 1
    load_bus_lst = ppc['load_bus']
    ld2busM = np.zeros((load_num, bus_num))
    for i, bus in enumerate(load_bus_lst):
        ld2busM[i, int(bus)] = 1
    renewable2busM = np.zeros((renewable_num, bus_num))
    for i, gen in enumerate(ppc['renewable_ids']):
        renewable2busM[i] = gen2busM[gen]

    open_ids = np.where(observation.gen_status > 0)[0].tolist()
    close_ids = np.where(observation.gen_status == 0)[0].tolist()
    edge_ids = np.where(np.abs(observation.gen_p - np.asarray(ppc['min_gen_p'])) < 1e-3)[0].tolist()

    # BUS DATA
    load_p = np.asarray(observation.nextstep_load_p) * 1.01     # estimate transmission loss = 1%
    ppc['bus'][:, PD] = np.matmul(load_p, ld2busM)  # Pd
    load_q = np.asarray(observation.load_q)
    ppc['bus'][:, QD] = np.matmul(load_q, ld2busM)  # Qd
    Vm = np.asarray(observation.bus_v)
    ppc['bus'][:, VM] = Vm  # Vm
    # TODO: add bus angle information
    Va = np.asarray(observation.bus_ang)  # Va
    ppc['bus'][:, VA] = Va

    # GEN_DATA
    gen_p = np.asarray(observation.gen_p)
    ppc['gen'][:, PG] = gen_p  # Pg
    gen_q = np.asarray(observation.gen_q)
    ppc['gen'][:, QG] = gen_q  # Qg
    gen_v = np.asarray(observation.gen_v)
    ppc['gen'][:, VG] = gen_v  # Vg
    gen_status = np.asarray(observation.gen_status)
    ppc['gen'][:, GEN_STATUS] = gen_status  # status

    bal_gen_p_mid = (ppc['min_gen_p'][ppc['balanced_id']] + ppc['max_gen_p'][ppc['balanced_id']]) / 2
    redundancy = (ppc['max_gen_p'][ppc['balanced_id']] - ppc['min_gen_p'][ppc['balanced_id']]) / 2 * 0.8

    gen_p_upper = np.asarray(observation.gen_p) + observation.action_space['adjust_gen_p'].high
    gen_p_lower = np.maximum(np.asarray(observation.gen_p) + observation.action_space['adjust_gen_p'].low, ppc['min_gen_p'])
    # import ipdb
    # ipdb.set_trace()
    ppc['gen'][:, PMAX] = gen_p_upper
    # ppc['gen'][:, PMAX] = ppc['max_gen_p']
    ppc['gen'][ppc['balanced_id'], PMAX] = bal_gen_p_mid + redundancy
    ppc['gen'][close_ids, PMAX] = 0
    ratio = 0.98
    ppc['gen'][ppc['renewable_ids'], PMAX] = np.array(observation.nextstep_renewable_gen_p_max) * ratio
    ppc['gen'][:, PMIN] = gen_p_lower
    # ppc['gen'][:, PMIN] = ppc['min_gen_p']
    ppc['gen'][ppc['renewable_ids'], PMIN] = 0.0
    edge_n_thermal = list(set(edge_ids) & set(ppc['thermal_ids']))
    ppc['gen'][edge_n_thermal, PMIN] = ppc['gen'][edge_n_thermal, PMIN].clip(np.asarray(ppc['min_gen_p'])[edge_n_thermal], 1e6)
    ppc['gen'][ppc['balanced_id'], PMIN] = bal_gen_p_mid - redundancy
    ppc['gen'][close_ids, PMIN] = 0

    result, open_hot, close_hot = run_uopf(ppc)
    result['gen'][:, PG] = result['gen'][:, PG].clip(ppc['gen'][:, PMIN], ppc['gen'][:, PMAX])
    new_gen_p = copy.deepcopy(result['gen'][:, PG])
    recover_ids = np.where(open_hot[:gen_num] > 0)[0].tolist()
    close_ids = np.where(close_hot[:gen_num] > 0)[0].tolist()
    return new_gen_p, recover_ids, close_ids, result


# datacenter's influence on LMP, Tariff project
# cases = [
#     # bus_num, injected_load
#     [1636, 1000],   # case 1, houston connected 1GW
#     [18, 600],      # case 2, mid land connected 0.5GW
#     [1067, 600],    # case 3, austin connected 0.5GW
#     [570, 800],    # case 4, dallas connected 0.8GW
# ]
# for i in range(len(cases) + 1):
#     ppc = case2000()        # case 0, original, 2016 summer peak
#     # i = 2
#     if i > 0:
#         bus, load = cases[i-1]
#         ppc['bus'][bus, PD] += load
#     ppopt = ppoption(VERBOSE=1, OUT_ALL=0, CONTINGENCY_AWARE=0)
#     results = rundcopf(ppc, ppopt)
#     env.env.visualizer.plot(results, save_path=f'./compensations/{results["network"]}', step=i)
# import ipdb
# ipdb.set_trace()



ppc_dict = {
    # 'IEEE14': case14(),
    # 'IEEE39': case39(),
    # 'IEEE57': case57(),
    # 'SG126': case126(),
    # 'IEEE300': case300(),
    'Texas2000': case2000(),
}
for network, ppc in ppc_dict.items():
    env = RL4Grid.make_gridsim(network_ppc={network: ppc}, deterministic=True)
    ppc_lst = []
    target_dones = 0
    for i in range(0, 35132):
        obs = env.reset(start_sample_idx=i)
        action_high = obs.action_space['adjust_gen_p'].high
        action_low = obs.action_space['adjust_gen_p'].low
        ppc = copy.deepcopy(env.env.ppc)
        new_gen_p, _, _, result = rerun_opf(obs, ppc)
        if not result['success']:
            continue
        best_a = new_gen_p - np.asarray(obs.gen_p)
        best_a = best_a.clip(action_low, action_high)
        _, best_reward, target_done, info = env.step(best_a)
        # env.env.visualizer.plot(env.env.ppc, save_path=f'./figs/{env.env.ppc["network"]}', step=i)
        if target_done:
            mismatch_ids = np.where(np.abs(new_gen_p - env.env.ppc['gen'][:, PG])>1)[0].tolist()
            for idx in mismatch_ids:
                print(f'scenario={i}, gen_idx={idx}, gen_bus={env.env.ppc["gen"][idx, GEN_BUS]}, is_renewable={idx in env.env.ppc["renewable_ids"]}, is_thermal={idx in env.env.ppc["thermal_ids"]}, '
                      f'is_balanced={idx == env.env.ppc["balanced_id"]}, prev_p={new_gen_p[idx]}, now_gen_p={env.env.ppc["gen"][idx, PG]}')
            print(f'target done, {info}')
            target_dones += 1
        else:
            ppc_dict = {}
            ppc_dict['bus'] = env.env.ppc['bus']
            ppc_dict['branch'] = env.env.ppc['branch']
            ppc_dict['gen'] = env.env.ppc['gen']
            ppc_dict['target_gen_p'] = new_gen_p
            ppc_dict['index'] = i
            ppc_lst.append(ppc_dict)
            if i % 100 == 0:
                print("********************************************")
                print(f'{i}, network={network}, target_dones={target_dones}')
                print("--------------------------------------------")

    import pickle
    filehandler = open(f"ppc_lst_{network}.pkl", "wb")
    pickle.dump(ppc_lst, filehandler)
    # filehandler = open(f'ppc_lst_{network}.pkl', "rb")
    # data = pickle.load(filehandler)

import ipdb
ipdb.set_trace()