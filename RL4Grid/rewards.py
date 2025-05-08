import math


def line_over_flow_reward(obs, ppc):
    r = 1 - sum([min(i, 1) for i in obs.rho])/ppc['num_line']
    return r

def line_over_flow_reward_v2(obs, ppc):
    # r = 1 - sum([min(i, 1) for i in obs.rho])/ppc['num_line']
    r = 1 - sum([rho for rho in obs.rho])/ppc['num_line']
    return r

def line_disconnect_reward(obs, ppc):
    disc_num = ppc['num_line'] - sum(obs.line_status)
    r = -0.3 * (1.5) ** disc_num
    return r

def renewable_consumption_reward(obs, ppc):
    all_gen_p = 0.0
    all_gen_p_max = 0.0
    for i, j in enumerate(ppc['renewable_ids']):
        all_gen_p += obs.gen_p[j]
        all_gen_p_max += obs.curstep_renewable_gen_p_max[i]
    r = all_gen_p / all_gen_p_max
    return r

def thermal_backup_reward(obs, ppc):
    backup_gen_p = 0.0
    backup_gen_p_max = sum(ppc['min_gen_p'])
    for i, j in enumerate(ppc['thermal_ids']):
        backup_gen_p += max(abs(obs.action_space['adjust_gen_p'].high[j]), abs(obs.action_space['adjust_gen_p'].low[j]))
    r = backup_gen_p / backup_gen_p_max
    return r

def balanced_gen_reward(obs, ppc):
    r = 0.0
    idx = ppc['balanced_id']
    max_val = ppc['max_gen_p'][idx]
    min_val = ppc['min_gen_p'][idx]
    gen_p_val = obs.gen_p[idx]
    if gen_p_val > max_val:
        r += abs((gen_p_val - max_val) /
                 ((max_val - min_val)/2)
                 # max_val
                 )
    if gen_p_val < min_val:
        r += abs((gen_p_val - min_val) /
                 ((max_val - min_val)/2)
                 # min_val
                 )
    r = -10 * r   # Ensure the range of r is [-1,0]
    return r

def balanced_gen_reward_v2(obs, ppc):
    r = 0.0
    i = ppc['balanced_id']
    r += (inv_funnel_func(obs.gen_p[i], ppc['max_gen_p'][i], ppc['min_gen_p'][i]) - 0.75)
    return r

def running_cost_reward(obs, last_obs, ppc):
    r = 0.0
    for i in range(len(obs.gen_p)):
        if i not in ppc['renewable_ids']:
            r -= ppc['gencost'][:, -3][i] * (obs.gen_p[i]) ** 2 + \
                ppc['gencost'][:, -2][i] * \
                obs.gen_p[i] + ppc['gencost'][:, -1][i]
        if obs.gen_status[i] != last_obs.gen_status[i] and i in ppc['thermal_ids']:
            r -= ppc['gencost'][:, 1][i]
    temp = 100000.0
    reward = r / (ppc['num_gen'] * temp)
    reward = math.exp(reward) - 1
    return reward


def running_cost_reward_v2(obs, last_obs, ppc):
    r = 0.0
    for i in range(len(obs.gen_p)):
        if i not in ppc['renewable_ids']:
            r -= ppc['gencost'][:, -3][i] * (obs.gen_p[i]) ** 2 + \
                ppc['gencost'][:, -2][i] * \
                obs.gen_p[i] + ppc['gencost'][:, -1][i]
        if obs.gen_status[i] != last_obs.gen_status[i] and i in ppc['thermal_ids']:
            r -= ppc['gencost'][:, 1][i]
    temp = 100000.0
    reward = r / (ppc['num_gen'] * temp)
    reward = math.exp(reward) - 1
    return reward


def gen_reactive_power_reward(obs, ppc):
    r = 0.0
    for i in range(ppc['num_gen']):
        if obs.gen_q[i] > ppc['max_gen_q'][i]:
            r -= abs((obs.gen_q[i] - ppc['max_gen_q'][i]) /
                     ((ppc['max_gen_q'][i] - ppc['min_gen_q'][i])/2)
                     # ppc['max_gen_q'][i]
                     )
        if obs.gen_q[i] < ppc['min_gen_q'][i]:
            r -= abs((obs.gen_q[i] - ppc['min_gen_q'][i]) /
                     ((ppc['max_gen_q'][i] - ppc['min_gen_q'][i])/2)
                     # ppc['min_gen_q'][i]
                     )
    r = math.exp(r) - 1
    return r

# TODO: V2 is a dense version of reactive power reward(penalty)
def gen_reactive_power_reward_v2(obs, ppc):
    # r = 0.0
    # for i in range(ppc['num_gen):
    #     r += (inv_funnel_func(obs.gen_q[i], ppc['max_gen_q[]i], ppc['min_gen_q[]i]) - 0.75)
    # r /= ppc['num_gen
    # return r

    r = []
    for i in range(ppc['num_gen']):
        r.append(inv_funnel_func(obs.gen_q[i], ppc['max_gen_q'][i], ppc['min_gen_q'][i]) - 0.75)
    return min(r)


def inv_funnel_func(x, upperB, lowerB):
    mu = (upperB + lowerB) / 2
    sigma = (upperB - lowerB) / 3
    return math.exp((-(x-mu)**2)/(sigma**2))


def sub_voltage_reward(obs, ppc):
    r = 0.0
    for i in range(len(ppc['max_bus_v'])):
        if obs.bus_v[i] > ppc['max_bus_v'][i]:
            # print(f'bus{i}, v={obs.bus_v[i]}')
            r -= abs((obs.bus_v[i] - ppc['max_bus_v'][i]) /
                     (ppc['max_bus_v'][i] - ppc['min_bus_v'][i])
                     # ppc['max_bus_v[i]
                     )
        if obs.bus_v[i] < ppc['min_bus_v'][i] and obs.bus_v[i] > 0.0:
            # print(f'bus{i}, v={obs.bus_v[i]}')
            r -= abs((obs.bus_v[i] - ppc['min_bus_v'][i]) /
                     (ppc['max_bus_v'][i] - ppc['min_bus_v'][i])
                     # ppc['min_bus_v'][i]
                     )
    r /= len(ppc['max_bus_v'])
    r = math.exp(r) - 1
    return r


def EPRIReward(obs, last_obs, ppc):
    r = ppc['coeff_line_over_flow'] * line_over_flow_reward(obs, ppc) + \
        ppc['coeff_renewable_consumption'] * renewable_consumption_reward(obs, ppc) + \
        ppc['coeff_balanced_gen'] * balanced_gen_reward(obs, ppc) + \
        ppc['coeff_sub_voltage'] * sub_voltage_reward(obs, ppc) + \
        ppc['coeff_gen_reactive_power'] * gen_reactive_power_reward(obs, ppc) + \
        ppc['coeff_running_cost'] * running_cost_reward(obs, last_obs, ppc)
    return r


def self_reward(obs, last_obs, config):
    ppc = config.ppc
    # r = config.coeff_renewable_consumption * renewable_consumption_reward(obs, ppc)

    # config.coeff_line_disconnect * line_disconnect_reward(obs, ppc) #+ \
    #
    r = config.coeff_line_over_flow * line_over_flow_reward_v2(obs, ppc) + \
        config.coeff_renewable_consumption * renewable_consumption_reward(obs, ppc) + \
        config.coeff_balanced_gen * balanced_gen_reward(obs, ppc) + \
        config.coeff_running_cost * running_cost_reward_v2(obs, last_obs, ppc) #+ \
        # config.coeff_sub_voltage * sub_voltage_reward(obs, ppc)
    # config.coeff_gen_reactive_power * gen_reactive_power_reward(obs, ppc) + \


        # config.coeff_line_disconnect * line_disconnect_reward(obs, ppc) + \
        # config.coeff_balanced_gen * balanced_gen_reward(obs, ppc) + \
        # config.coeff_gen_reactive_power * gen_reactive_power_reward(obs, ppc)
        # config.coeff_thermal_backup * thermal_backup_reward(obs, ppc)
    #
    return r
