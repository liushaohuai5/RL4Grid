import numpy as np
from sklearn import preprocessing
import torch
from utilize.form_action import *
import copy
import torch.nn.functional as F
import math
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *

def formalize_obs(obs_lst):
    obs_lst = np.asarray(obs_lst)
    shape = obs_lst.shape
    obs_lst = obs_lst.reshape((shape[0], -1))
    return obs_lst

def state_append(state, state_dict, idx, input, input_name):
    state.append(np.reshape(np.array(input, dtype=np.float32), (-1,)))
    try:
        idx += len(input)
        state_dict.update({input_name: [idx - len(input), idx]})
    except:
        idx += 1
        state_dict.update({input_name: [idx - 1, idx]})

    return state, state_dict, idx

def calc_running_cost_rew(last_real_gen_p, ppc, gen_status=None, last_gen_status=None, is_real=False):
    r_c = 0.0
    for i in range(last_real_gen_p.shape[-1]):
        if i not in ppc['renewable_ids']:
            if len(last_real_gen_p.shape) == 2:
                r_c -= ppc['gencost'][:, -3][i] * (last_real_gen_p[:, i]) ** 2 + \
                       ppc['gencost'][:, -2][i] * \
                       last_real_gen_p[:, i] + ppc['gencost'][:, -1][i]
            elif len(last_real_gen_p.shape) == 3:
                r_c -= ppc['gencost'][:, -3][i] * (last_real_gen_p[:, :, i]) ** 2 + \
                       ppc['gencost'][:, -2][i] * \
                       last_real_gen_p[:, :, i] + ppc['gencost'][:, -1][i]
            else:
                r_c -= ppc['gencost'][:, -3][i] * (last_real_gen_p[i]) ** 2 + \
                       ppc['gencost'][:, -2][i] * \
                       last_real_gen_p[i] + ppc['gencost'][:, -1][i]
        if gen_status is not None:
            if gen_status[i] != last_gen_status[i] and i in ppc['thermal_ids']:
                r_c -= ppc['gencost'][:, 1][i]

    # return r_c / 1e5 if is_real else np.log(1 + r_c / 1e5)
    # return r_c / 1e5 if is_real else -(r_c/1e5)**2
    return r_c / 1e5 if is_real else r_c / 1e5


def combine_one_hot_action(state, action_one_hot, ready_thermal_mask, closable_thermal_mask, config, action_high, action_low, thermal_to_all, is_test=False):

    generator_num = config.generator_num
    one_hot_dim = config.one_hot_dim
    action = np.zeros(config.ppc['num_gen'])
    action[config.ppc['sorted_controlable_ids']] = action_one_hot[:generator_num]
    open_one_hot = np.matmul(action_one_hot[generator_num:generator_num+one_hot_dim], thermal_to_all)
    close_one_hot = np.matmul(action_one_hot[generator_num+one_hot_dim:], thermal_to_all)
    state = torch.from_numpy(state)
    action = torch.from_numpy(action)
    open_one_hot = torch.from_numpy(open_one_hot)
    close_one_hot = torch.from_numpy(close_one_hot)
    ready_thermal_mask = torch.from_numpy(ready_thermal_mask)[:-1]
    closable_thermal_mask = torch.from_numpy(closable_thermal_mask)[:-1]
    action_high = torch.from_numpy(action_high)
    action_low = torch.from_numpy(action_low)

    mu = (action + torch.ones_like(action)) / (2 * torch.ones_like(action)) * (action_high - action_low) + action_low
    # mu = action
    modified_mu = mu * (torch.ones_like(ready_thermal_mask) - ready_thermal_mask)
    modified_mu *= (torch.ones_like(closable_thermal_mask) - closable_thermal_mask)

    modified_mu += torch.clamp(mu*closable_thermal_mask, 0, 10000)

    if sum(open_one_hot) > 0:
        open_id = open_one_hot.argmax()
        if open_id in config.ppc['fast_thermal_gen']:
            modified_mu[open_id] = action_high[open_id]
            ready_thermal_mask[open_id] = 0

    if sum(close_one_hot) > 0:
        close_id = close_one_hot.argmax()
        if close_id in config.ppc['fast_thermal_gen']:
            modified_mu[close_id] = action_low[close_id]
            closable_thermal_mask[close_id] = 0

    # idxs = config.state_dict['steps_to_min_gen']
    # starting_gen_steps = state[idxs[0]:idxs[1]]
    # gen_starting_flag = (starting_gen_steps >= 0).float()
    # ready_thermal_mask[config.ppc['fast_thermal_gen']] = ready_thermal_mask[config.ppc['fast_thermal_gen']] * (1 - gen_starting_flag)
    # ready_ids = torch.where(ready_thermal_mask > 0)[0].tolist()
    # closable_ids = torch.where(closable_thermal_mask > 0)[0].tolist()
    # delta_load_p = state[0]
    # balance_up_redundency = state[-1]
    # balance_down_redundency = state[-2]
    # balanced_range = config.ppc['max_gen_p'][config.ppc['balanced_id']] - config.ppc['min_gen_p'][config.ppc['balanced_id']]
    # redundency_adjust = -(1 - np.sign(balance_up_redundency - balanced_range * 0.05)) / 2 * (
    #         balance_up_redundency - balanced_range * 0.05) + \
    #                     (1 - np.sign(balance_down_redundency - balanced_range * 0.05)) / 2 * (
    #                             balance_down_redundency - balanced_range * 0.05)
    # if config.parameters['only_power']:
    #     delta = delta_load_p - modified_mu.sum() + redundency_adjust
    # else:
    #     delta = delta_load_p - modified_mu[:generator_num].sum() + redundency_adjust
    # 
    # delta_futures = []
    # ahead = 0
    # idxs = config.state_dict['future_load_p']
    # future_load = state[idxs[0] + config.ppc['thermal_start_response_steps'] + ahead]
    # idxs = config.state_dict['future_renewable_gen_p_max']
    # future_renewable = state[idxs[0] + config.ppc['thermal_start_response_steps'] + ahead]
    # 
    # # TODO: add multi-step delta back
    # for i in range(config.ppc['thermal_start_response_steps'] + ahead - 1, config.ppc['thermal_start_response_steps'] + ahead):
    #     idxs = config.state_dict['future_load_p']
    #     current_load = state[idxs[0] + i]
    #     idxs = config.state_dict['future_renewable_gen_p_max']
    #     current_renewable = state[idxs[0] + i]
    #     delta_futures.append((future_load - current_load) -
    #                          (future_renewable - current_renewable))
    # 
    # delta_futures = torch.stack(delta_futures, dim=0)[0]
    # # delta_futures += redundency_adjust
    # cnt = 0
    # if delta_futures > balanced_range * 0.1:
    #     if is_test:
    #         print(f'delta_futures={delta_futures:.3f}')
    #     while delta_futures > balanced_range * 0.1 and cnt < 4:
    #         if delta_futures > 0:
    #             if sum(ready_thermal_mask) == 0:
    #                 break
    #             else:
    #                 del_id = np.argmin(np.abs(action_high[ready_ids] - delta_futures))
    #                 id = ready_ids[del_id]
    #                 assert ready_thermal_mask[id] == 1
    #                 modified_mu[id] = action_high[id]
    #                 ready_thermal_mask[id] = 0
    #                 del ready_ids[del_id]
    #                 delta_futures -= action_high[id]
    #         cnt += 1
    #         if cnt > 30:
    #             import ipdb
    #             ipdb.set_trace()
    #             print('error')
    # 
    # # cnt = 0
    # # if delta < -balanced_range * 0.1:
    # #     if is_test:
    # #         print(f'delta={delta:.3f}')
    # #     while delta < -balanced_range * 0.1 and cnt < 2:
    # #         if delta < 0:
    # #             if sum(closable_thermal_mask) == 0:
    # #                 break
    # #             else:
    # #                 del_id = np.argmin(np.abs(action_low[closable_ids] - delta))
    # #                 id = closable_ids[del_id]
    # #                 assert closable_thermal_mask[id] == 1
    # #                 modified_mu[id] = action_low[id]
    # #                 closable_thermal_mask[id] = 0
    # #                 del closable_ids[del_id]
    # #                 delta -= action_low[id]
    # #         cnt += 1
    # #         if cnt > 30:
    # #             import ipdb
    # #             ipdb.set_trace()
    # #             print('error')

    return modified_mu.cpu().numpy()

def modify_action_v3(action, ready_thermal_mask, closable_thermal_mask, config, action_high, action_low):

    # def atanh(x):
    #     return 0.5 * (torch.log(1 + x + 1e-6) - torch.log(1 - x + 1e-6))
    # print(action[-1])
    action = torch.from_numpy(action)
    # action = F.tanh(action)
    ready_thermal_mask = torch.from_numpy(ready_thermal_mask)
    closable_thermal_mask = torch.from_numpy(closable_thermal_mask)
    action_high = torch.from_numpy(action_high)
    action_low = torch.from_numpy(action_low)
    mu = (action + torch.ones_like(action)) / (2 * torch.ones_like(action)) * (action_high - action_low) + action_low
    # mu = action
    modified_mu = mu * (torch.ones_like(ready_thermal_mask) - ready_thermal_mask)
    modified_mu *= (torch.ones_like(closable_thermal_mask) - closable_thermal_mask)

    sum_ready_thermal_mask = ready_thermal_mask.sum()
    open_mu = torch.zeros_like(mu)

    if sum_ready_thermal_mask > 0:
        # open_id = ready_thermal_mask.nonzero()[F.softmax(action.gather(0, ready_thermal_mask.nonzero().squeeze())).argmax()][0]
        # open_id = ready_thermal_mask.nonzero()[action.gather(0, ready_thermal_mask.nonzero().squeeze()).argmax()][0]
        open_tmp = (F.sigmoid(action.gather(0, ready_thermal_mask.nonzero().squeeze())) > 0.65).sum()
        if open_tmp > 0:
            open_ids = ready_thermal_mask.nonzero()[torch.where(F.sigmoid(action.gather(0, ready_thermal_mask.nonzero().squeeze())) > 0.65)][0]
            # print(f'real_open_ids={open_ids}')
            open_mu[open_ids] = mu[open_ids]

    modified_mu += open_mu
    if config.enable_close_gen:
        sum_closable_thermal_mask = closable_thermal_mask.sum()
        close_mu = torch.zeros_like(mu)
        if sum_closable_thermal_mask > 0:
            # close_id = closable_thermal_mask.nonzero()[action.gather(0, closable_thermal_mask.nonzero().squeeze()).argmin()][0]
            close_tmp = (F.sigmoid(action.gather(0, closable_thermal_mask.nonzero().squeeze())) < 0.35).sum()
            if close_tmp > 0:
                close_ids = closable_thermal_mask.nonzero()[torch.where(F.sigmoid(action.gather(0, closable_thermal_mask.nonzero().squeeze())) < 0.35)][0]
                # print(f'real_close_ids={close_ids}')
                close_mu[close_ids] = mu[close_ids]
    else:
        close_mu = torch.clamp(mu * closable_thermal_mask, 0, 10000)  # prohibit thermal generator closing
    modified_mu += close_mu
    return modified_mu.cpu().numpy()

def print_log(obs, config):
    adjust_gen_p_max = sum(obs.action_space['adjust_gen_p'].high[config.ppc['sorted_controlable_ids']]) \
                       + config.ppc['max_gen_p'][config.ppc['balanced_id']] - obs.gen_p[config.ppc['balanced_id']]
    delta_load = sum(obs.nextstep_load_p) - sum(obs.load_p)
    print(f'delta_load={delta_load:.2f}, adjust_gen_p_max={adjust_gen_p_max}, balanced_gen={obs.gen_p[config.ppc["balanced_id"]]}')
    if delta_load > 0:
        if adjust_gen_p_max < delta_load:
            print('no solution')

    gen_ids = [i for i in range(config.ppc['num_gen'])]
    closed_gen_ids = np.where(obs.gen_status == 0)
    closed_gen_ids = closed_gen_ids[0].tolist()  # ??????id
    renewable_adjustable_ids = copy.copy(config.ppc['renewable_ids'])
    action_high = obs.action_space['adjust_gen_p'].high
    open_gen_ids = []  # ??????
    for i in config.ppc['thermal_ids']:
        if i not in closed_gen_ids:
            open_gen_ids.append(i)
    closable_gen_ids = []
    for i in open_gen_ids:
        if obs.steps_to_close_gen[i] == 0 and abs(obs.gen_p[i] - config.ppc['min_gen_p'][i])<1e-3:
            closable_gen_ids.append(i)
    adjustable_gen_ids = []
    for i in open_gen_ids:
        if i not in closable_gen_ids:
            adjustable_gen_ids.append(i)
    ready_thermal_ids = []  # ???????
    for i in closed_gen_ids:
        if obs.steps_to_recover_gen[i] == 0:
            ready_thermal_ids.append(i)
    print(f'closed_gen={closed_gen_ids}')
    print(f'step_restart={obs.steps_to_recover_gen[closed_gen_ids]}')
    # print(obs.action_space['adjust_gen_p'].high[closed_gen_ids])
    print(f'sum_renewable_action_high={sum(action_high[renewable_adjustable_ids])}')


def get_state_from_obs(obs, config, return_dict=False):
    parameters = config.parameters
    delta_load = sum(obs.nextstep_load_p) - sum(obs.load_p)
    # print(f'outer_load={sum(obs.load_p)}, outer_next_load={sum(obs.nextstep_load_p)}')
    state = []

    bus_v = obs.bus_v

    thermal_flag = np.zeros(len(config.ppc['thermal_ids']), dtype=np.float32)
    ready_thermal_mask = np.zeros(config.ppc['num_gen']+1, dtype=np.float32)
    closable_thermal_mask = np.zeros(config.ppc['num_gen']+1, dtype=np.float32)
    thermal_set = config.ppc['thermal_ids']
    for i, gen_idx in enumerate(thermal_set):
        ready_thermal_mask[-1] = 1
        if obs.gen_status[gen_idx] == 0 and obs.steps_to_recover_gen[gen_idx] == 0: #and obs.gen_start_flag[i] == 0:
            thermal_flag[i] = 1
            ready_thermal_mask[gen_idx] = 1
        closable_thermal_mask[-1] = 1
        if obs.gen_status[gen_idx] == 1 and abs(obs.gen_p[gen_idx] - config.ppc['min_gen_p'][gen_idx]) < 1e-3:
            if obs.steps_to_close_gen[gen_idx] == 0:
                thermal_flag[i] = -1
                closable_thermal_mask[gen_idx] = 1

    adjust_gen_p_max = sum(obs.action_space['adjust_gen_p'].high[config.ppc['sorted_controlable_ids']]) + \
                       config.ppc['max_gen_p'][config.ppc['balanced_id']] - obs.gen_p[config.ppc['balanced_id']]

    action_high, action_low = get_action_space(obs, config)
    action_high = action_high[config.ppc['sorted_controlable_ids']]
    action_low = action_low[config.ppc['sorted_controlable_ids']]

    renewable_consumption_rate = sum(np.array(obs.gen_p)[config.ppc['renewable_ids']]) / (sum(obs.curstep_renewable_gen_p_max) + 1e-2)
    idx = 0
    state_dict = {}

    state, state_dict, idx = state_append(state, state_dict, idx, delta_load, 'delta_load') # delta_load max = 467, min = -386
    state, state_dict, idx = state_append(state, state_dict, idx, obs.gen_p, 'gen_p') # config.ppc['min_gen_p'], config.ppc['max_gen_p']
    state, state_dict, idx = state_append(state, state_dict, idx, obs.load_p, 'load_p') # load_p max = 130, min = -150
    state, state_dict, idx = state_append(state, state_dict, idx, obs.nextstep_load_p, 'nextstep_load_p')
    # state, state_dict, idx = state_append(state, state_dict, idx, renewable_consumption_rate, 'renewable_consumption_rate')
    # TODO: add line measurements into state
    state, state_dict, idx = state_append(state, state_dict, idx, obs.a_or, 'a_or')
    state, state_dict, idx = state_append(state, state_dict, idx, obs.a_ex, 'a_ex')

    state, state_dict, idx = state_append(state, state_dict, idx, np.array(obs.steps_to_min_gen, dtype=np.float32)[config.ppc['fast_thermal_gen']], 'steps_to_min_gen')
    state, state_dict, idx = state_append(state, state_dict, idx, obs.future_renewable_gen_p_max, 'future_renewable_gen_p_max')
    state, state_dict, idx = state_append(state, state_dict, idx, obs.future_load_p, 'future_load_p')

    state, state_dict, idx = state_append(state, state_dict, idx, obs.rho, 'rho')
    state, state_dict, idx = state_append(state, state_dict, idx, bus_v, 'bus_v')
    state, state_dict, idx = state_append(state, state_dict, idx, obs.gen_q, 'gen_q')
    state, state_dict, idx = state_append(state, state_dict, idx, obs.load_q, 'load_q')
    state, state_dict, idx = state_append(state, state_dict, idx, action_high, 'action_high')
    state, state_dict, idx = state_append(state, state_dict, idx, action_low, 'action_low')
    state, state_dict, idx = state_append(state, state_dict, idx, obs.curstep_renewable_gen_p_max, 'curstep_renewable_gen_p_max')
    state, state_dict, idx = state_append(state, state_dict, idx, obs.nextstep_renewable_gen_p_max, 'nextstep_renewable_gen_p_max')

    bal_gen_p_mid = (config.ppc['min_gen_p'][config.ppc['balanced_id']]+config.ppc['max_gen_p'][config.ppc['balanced_id']]) / 2
    redundancy = (config.ppc['max_gen_p'][config.ppc['balanced_id']]-config.ppc['min_gen_p'][config.ppc['balanced_id']]) / 2 * 0.6
    state, state_dict, idx = state_append(state, state_dict, idx, obs.gen_p[config.ppc['balanced_id']] - (bal_gen_p_mid - redundancy), 'down_redundancy')
    state, state_dict, idx = state_append(state, state_dict, idx, (bal_gen_p_mid + redundancy) - obs.gen_p[config.ppc['balanced_id']], 'up_redundancy')

    state = np.concatenate(state)

    if return_dict:
        return state_dict
    else:
        if len(state)==1298:
            import ipdb
            ipdb.set_trace()
        return state, ready_thermal_mask, closable_thermal_mask

def get_action_space(obs, config):
    parameters = config.parameters
    if parameters['only_power']:
        if parameters['only_thermal']:
            action_high = obs.action_space['adjust_gen_p'].high[config.ppc['thermal_ids']]
            action_low = obs.action_space['adjust_gen_p'].low[config.ppc['thermal_ids']]
        else:
            action_high = obs.action_space['adjust_gen_p'].high
            action_high = np.where(np.isinf(action_high), np.full_like(action_high, 0),
                                   action_high)  # feed 0 to balanced generator threshold
            action_low = obs.action_space['adjust_gen_p'].low
            action_low = np.where(np.isinf(action_low), np.full_like(action_low, 0),
                                  action_low)  # feed 0 to balanced generator threshold
    else:
        action_high = np.asarray(
            [obs.action_space['adjust_gen_p'].high, obs.action_space['adjust_gen_v'].high]).flatten()
        action_high = np.where(np.isinf(action_high), np.full_like(action_high, 0),
                               action_high)  # feed 0 to balanced generator threshold
        action_low = np.asarray([obs.action_space['adjust_gen_p'].low, obs.action_space['adjust_gen_v'].low]).flatten()
        action_low = np.where(np.isinf(action_low), np.full_like(action_low, 0),
                              action_low)  # feed 0 to balanced generator threshold
    return action_high, action_low

def voltage_action(obs, config, type='reactive'):  # type = 'period' or '1'
    if type == 'period':
        err = np.maximum(np.asarray(config.ppc['min_gen_v']) - np.asarray(obs.gen_v), np.asarray(obs.gen_v) - np.asarray(config.ppc['max_gen_v']))  # restrict in legal range
        gen_num = len(err)
        action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
        adjust_gen_v = np.zeros(config.ppc['num_gen'])
        for i in range(gen_num):
            if err[i] <= 0:
                continue
            elif err[i] > 0:
                if obs.gen_v[i] < config.ppc['min_gen_v'][i]:
                    if err[i] < action_high[i]:
                        adjust_gen_v[i] = err[i]
                    else:
                        adjust_gen_v[i] = action_high[i]
                elif obs.gen_v[i] > config.ppc['max_gen_v'][i]:
                    if - err[i] > action_low[i]:
                        adjust_gen_v[i] = - err[i]
                    else:
                        adjust_gen_v[i] = action_low[i]
        return adjust_gen_v
    elif type == 'constant':
        err = np.asarray(obs.gen_v) - 1.02  # restrict at stable point 1.05
        gen_num = len(err)
        action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
        adjust_gen_v = np.zeros(config.ppc['num_gen'])
        for i in range(gen_num):
            if err[i] < 0:
                if abs(err[i]) > action_high[i]:
                    adjust_gen_v[i] = action_high[i]
                else:
                    adjust_gen_v[i] = abs(err[i])
            elif err[i] > 0:
                if err[i] > abs(action_low[i]):
                    adjust_gen_v[i] = action_low[i]
                else:
                    adjust_gen_v[i] = -err[i]
        return adjust_gen_v
    elif type == 'overflow':
        outline_ids = []
        for line_idx in range(config.ppc['num_line']):
            if not obs.line_status[line_idx]:
                outline_ids.append(line_idx)

        overflow_line_ids = []
        for line_idx in range(config.ppc['num_line']):
            if obs.rho[line_idx] > config.ppc['soft_overflow_bound'] - 0.1:
                overflow_line_ids.append(line_idx)

        if len(overflow_line_ids) > 0:
            adjust_gen_v = obs.action_space['adjust_gen_v'].high
        elif len(overflow_line_ids) == 0:
            err = np.asarray(obs.gen_v) - 1  # restrict at stable point 1.05
            gen_num = len(err)
            action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
            adjust_gen_v = np.zeros(config.ppc['num_gen'])
            for i in range(gen_num):
                if err[i] < 0:
                    if - err[i] > action_high[i]:
                        adjust_gen_v[i] = action_high[i]
                    else:
                        adjust_gen_v[i] = - err[i]
                elif err[i] > 0:
                    if - err[i] < action_low[i]:
                        adjust_gen_v[i] = action_low[i]
                    else:
                        adjust_gen_v[i] = - err[i]
        return adjust_gen_v
    elif type == 'reactive':
        adjust_num = 0
        err = False
        adjust_gen_v = np.zeros(config.ppc['num_gen'])
        adjustable_gen_ids = [i for i in range(config.ppc['num_gen'])]
        # first do voltage overstep adjust, then do reactive power overstep adjusts
        bus_v = obs.bus_v

        upgrade_gen_v_ids = []
        downgrade_gen_v_ids = []
        for gen_id in range(config.ppc['num_gen']):
            if obs.gen_q[gen_id] < config.ppc['min_gen_q'][gen_id]:
                err = True
                upgrade_gen_v_ids.append(gen_id)
                adjustable_gen_ids.remove(gen_id)
            elif obs.gen_q[gen_id] > config.ppc['max_gen_q'][gen_id]:
                err = True
                downgrade_gen_v_ids.append(gen_id)
                adjustable_gen_ids.remove(gen_id)

        if len(upgrade_gen_v_ids) > 0:
            adjust_gen_v += 0.01
            adjust_gen_v = np.clip(adjust_gen_v, obs.action_space['adjust_gen_v'].low,
                                   obs.action_space['adjust_gen_v'].high)
            return adjust_gen_v
        elif len(downgrade_gen_v_ids) > 0:
            # adjust_gen_v[downgrade_gen_v_ids] -= 0.01
            adjust_gen_v -= 0.01
            adjust_gen_v = np.clip(adjust_gen_v, obs.action_space['adjust_gen_v'].low,
                                   obs.action_space['adjust_gen_v'].high)
            return adjust_gen_v

        gen2busInfM = gen_to_bus_influence_matrix(obs, config.ppc)
        over_v_upbound_bus = []
        below_v_lowbound_bus = []
        for bus_idx in range(config.ppc['num_bus']):
            if bus_v[bus_idx] < config.ppc['min_bus_v'][bus_idx]:
                err = True
                below_v_lowbound_bus.append(bus_idx)
            elif bus_v[bus_idx] > config.ppc['max_bus_v'][bus_idx]:
                err = True
                over_v_upbound_bus.append(bus_idx)
        if adjust_num == 0:
            adjust_num += 1
            if len(over_v_upbound_bus) > 0:
                # adjust_gen_v -= 0.011
                err = np.asarray(obs.gen_v) - 1  # restrict at stable point 1.05
                gen_num = len(err)
                action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
                adjust_gen_v = np.zeros(config.ppc['num_gen'])
                for i in range(gen_num):
                    if err[i] < 0:
                        if - err[i] > action_high[i]:
                            adjust_gen_v[i] = action_high[i]
                        else:
                            adjust_gen_v[i] = - err[i]
                    elif err[i] > 0:
                        if - err[i] < action_low[i]:
                            adjust_gen_v[i] = action_low[i]
                        else:
                            adjust_gen_v[i] = - err[i]
            elif len(below_v_lowbound_bus) > 0:
                # adjust_gen_v += 0.011
                err = np.asarray(obs.gen_v) - 1.04  # restrict at stable point 1.05
                gen_num = len(err)
                action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
                adjust_gen_v = np.zeros(config.ppc['num_gen'])
                for i in range(gen_num):
                    if err[i] < 0:
                        if - err[i] > action_high[i]:
                            adjust_gen_v[i] = action_high[i]
                        else:
                            adjust_gen_v[i] = - err[i]
                    elif err[i] > 0:
                        if - err[i] < action_low[i]:
                            adjust_gen_v[i] = action_low[i]
                        else:
                            adjust_gen_v[i] = - err[i]
            else:
                err = np.asarray(obs.gen_v) - 1.02  # restrict at stable point 1.05
                gen_num = len(err)
                action_high, action_low = obs.action_space['adjust_gen_v'].high, obs.action_space['adjust_gen_v'].low
                adjust_gen_v = np.zeros(config.ppc['num_gen'])
                for i in range(gen_num):
                    if err[i] < 0:
                        if - err[i] > action_high[i]:
                            adjust_gen_v[i] = action_high[i]
                        else:
                            adjust_gen_v[i] = - err[i]
                    elif err[i] > 0:
                        if - err[i] < action_low[i]:
                            adjust_gen_v[i] = action_low[i]
                        else:
                            adjust_gen_v[i] = - err[i]
            adjust_gen_v = np.clip(adjust_gen_v, obs.action_space['adjust_gen_v'].low,
                                       obs.action_space['adjust_gen_v'].high)

        # if err:
        #     import ipdb
        #     ipdb.set_trace()
        #     print('false')
        return adjust_gen_v

    elif type == 'none':
        adjust_gen_v = np.zeros(config.ppc['num_gen'])
        return adjust_gen_v


def get_adjacent_matrix(obs, ppc, weighted=False):
    bus_num = ppc['num_bus']
    adjacent_matrix = np.zeros((bus_num, bus_num), dtype=np.float32)
    idx_or = 0
    for i in range(ppc['num_line']):
        fbus = ppc['branch'][i, F_BUS]
        tbus = ppc['branch'][i, T_BUS]
        if weighted:
            adjacent_matrix[fbus, tbus] = obs.rho[i]
            adjacent_matrix[tbus, fbus] = obs.rho[i]
        else:
            adjacent_matrix[fbus, tbus] = 1
            adjacent_matrix[tbus, fbus] = 1
        break
    return adjacent_matrix


def gen_to_bus_matrix(ppc):
    gen2busM = np.zeros((ppc['num_gen'], ppc['num_bus']), dtype=np.float32)
    for gen, bus in enumerate(ppc['gen'][:, GEN_BUS]):
        gen2busM[gen][bus] = 1
    return gen2busM

def load_to_bus_matrix(ppc):
    ld2busM = np.zeros((ppc['num_load'], ppc['num_bus']), dtype=np.float32)
    for ld, bus in enumerate(ppc['load_bus']):
        ld2busM[ld][bus] = 1
    return ld2busM

def bus_to_line_matrix(ppc):
    bus2lineM = np.zeros((ppc['num_bus'], ppc['num_line']))
    for i in ppc['num_line']:
        fbus = ppc['branch'][i, F_BUS]
        tbus = ppc['branch'][i, T_BUS]
        bus2lineM[fbus, i] = 1
        bus2lineM[tbus, i] = -1
    return bus2lineM

def gen_to_bus_influence_matrix(obs, ppc):
    adjacent_matrix = get_adjacent_matrix(obs, ppc)
    gen2busM = gen_to_bus_matrix(ppc)
    gen2busInfM = gen2busM + 0.3 * np.matmul(gen2busM, adjacent_matrix)
    return gen2busInfM

def gen_to_line_influence_matrix(obs, ppc):
    adjacent_matrix = get_adjacent_matrix(obs, ppc)
    gen2busM = gen_to_bus_matrix(ppc)
    bus2lineM = bus_to_line_matrix(ppc)
    SecondOrderA = np.matmul(adjacent_matrix, adjacent_matrix)
    ThirdOrderA = np.matmul(adjacent_matrix, np.matmul(adjacent_matrix, adjacent_matrix))
    temp = gen2busM \
           + 0.3*(np.matmul(gen2busM, adjacent_matrix)) \
           # + 0.1*(np.matmul(gen2busM, SecondOrderA)) \
           # + 0.03*(np.matmul(gen2busM, ThirdOrderA))
    gen2lineInfM = np.matmul(temp, bus2lineM)
    return gen2lineInfM