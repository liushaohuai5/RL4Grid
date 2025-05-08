# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for 39 bus New England system.
"""
import numpy as np
import os
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
from pypower.idx_cost import *
path = os.path.dirname(__file__) + '/'

def case2000():

    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = np.load(path+'TX2000_bus.npy')

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = np.load(path+'TX2000_gen.npy')
    ppc["gen"][39, GEN_BUS] = 1091


    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = np.load(path+'TX2000_branch.npy')

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = np.load(path+'TX2000_gencost.npy')

    ppc['bus'][:, BUS_AREA] = 1
    # ppc['branch'][:, [BR_R, BR_X, BR_B]] = np.abs(ppc['branch'][:, [BR_R, BR_X, BR_B]])
    # ppc['branch'][:, BR_X] /= 10
    # ppc['branch'][:, [BR_R, BR_X]] = ppc['branch'][:, [BR_R, BR_X]].clip(1e-3, 1000)
    # ppc['branch'][:, BR_B] /= 10
    # ppc['branch'][:, ANGMIN] = -360.0
    # ppc['branch'][:, ANGMAX] = 360.0
    # ppc['branch'][:, [RATE_A, RATE_B, RATE_C]] = ppc['branch'][:, [RATE_A, RATE_B, RATE_C]].clip(1e3, 1e8)

    ppc['network'] = 'Texas2000'
    ppc['num_gen'] = ppc['gen'].shape[0]
    ppc['num_bus'] = ppc['bus'].shape[0]
    ppc['num_line'] = ppc['branch'].shape[0]  # 185 ori
    ppc['load_bus'] = np.nonzero(ppc['bus'][:, PD])[0].tolist()
    ppc['num_load'] = len(ppc['load_bus'])
    ppc['gen_type'] = np.load(path+'TX2000_gen_type.npy')
    balanced_bus = ppc['bus'][np.where(ppc['bus'][:, BUS_TYPE] == 3)[0][0], BUS_I]
    ppc['balanced_id'] = np.where(ppc['gen'][:, GEN_BUS] == balanced_bus)[0][0]
    ppc['gen_type'][ppc['balanced_id']] = 2
    ppc['thermal_ids'] = np.where(ppc['gen_type'] == 1)[0].tolist()
    ppc['renewable_ids'] = np.where(ppc['gen_type'] == 5)[0].tolist()
    # ppc['gencost'][ppc['renewable_ids'], -2] = 0
    # ppc['gencost'][ppc['renewable_ids'], -1] = 0
    ppc['sorted_controlable_ids'] = sorted(ppc['renewable_ids'] + ppc['thermal_ids'])
    ppc['min_gen_p'] = ppc['gen'][:, PMIN].tolist()
    ppc['gen'][ppc['balanced_id'], PMAX] *= 5
    ppc['max_gen_p'] = ppc['gen'][:, PMAX].tolist()
    for i in range(ppc['num_gen']):
        if i in ppc['thermal_ids']:
            ppc['min_gen_p'][i] = np.around(0.04 * ppc['gen'][i, PMAX], decimals=2).tolist()
    for i, bus in enumerate(ppc['gen'][:, GEN_BUS].tolist()):
        bus_idx = ppc['bus'][:, BUS_I].tolist().index(bus)
        if int(ppc['bus'][bus_idx, BUS_TYPE]) not in [2, 3]:
            ppc['bus'][bus_idx, BUS_TYPE] = 2
        ppc['bus'][bus_idx, BUS_TYPE] = 3 if i == ppc['balanced_id'] else 2

    # ppc['gen'][:, QMIN] *= 10
    # ppc['gen'][:, QMAX] *= 10
    ppc['min_gen_q'] = ppc['gen'][:, QMIN]
    ppc['max_gen_q'] = ppc['gen'][:, QMAX]
    ppc['min_gen_v'] = [0.9 for _ in range(ppc['num_gen'])]
    ppc['max_gen_v'] = [1.1 for _ in range(ppc['num_gen'])]
    ppc['min_bus_v'] = [0.9 for _ in range(ppc['num_bus'])]
    ppc['max_bus_v'] = [1.1 for _ in range(ppc['num_bus'])]

    # overflow parameters
    ppc['soft_overflow_bound'] = 1
    ppc['max_steps_soft_overflow'] = 4
    ppc['hard_overflow_bound'] = 1.35

    # line disconnection parameters
    ppc['prob_disconnection'] = 0.01
    ppc['max_steps_to_reconnect_line'] = 16
    ppc['line_thermal_limit'] = ppc['branch'][:, RATE_A]
    ppc['white_list_random_disconnection'] = [6, 37, 52, 71, 153, 136, 66, 75, 85, 98, 123, 138, 141]

    # ref/balanced generator
    ppc['min_balanced_gen_bound'] = 0.9
    ppc['max_balanced_gen_bound'] = 1.1

    ppc['ramp_rate'] = 0.05
    ppc['max_steps_to_recover_gen'] = []
    ppc['max_steps_to_close_gen'] = []
    ppc['fast_thermal_gen'] = []
    thresholds = [200, 400]     # TODO: need adjust
    for i in range(ppc['num_gen']):
        if i in ppc['thermal_ids']:
            if ppc['max_gen_p'][i] <= thresholds[0]:
                ppc['max_steps_to_recover_gen'].append(10)
                ppc['max_steps_to_close_gen'].append(10)
                ppc['fast_thermal_gen'].append(i)
            elif ppc['max_gen_p'][i] <= thresholds[1]:
                ppc['max_steps_to_recover_gen'].append(20)
                ppc['max_steps_to_close_gen'].append(20)
                ppc['fast_thermal_gen'].append(i)
            else:
                ppc['max_steps_to_recover_gen'].append(40)
                ppc['max_steps_to_close_gen'].append(40)
        else:
            ppc['max_steps_to_recover_gen'].append(40)
            ppc['max_steps_to_close_gen'].append(40)
    ppc['thermal_start_response_steps'] = 5

    # renewable generators
    ppc['renewable_forecast_horizon'] = 20
    ppc['load_forecast_horizon'] = 50

    # eval reward coeffs
    ppc['coeff_line_over_flow'] = 1
    ppc['coeff_renewable_consumption'] = 2
    ppc['coeff_running_cost'] = 1
    ppc['coeff_balanced_gen'] = 4
    ppc['coeff_gen_reactive_power'] = 1
    ppc['coeff_sub_voltage'] = 1

    # others
    ppc['keep_decimal_digits'] = 2
    ppc['env_allow_precision'] = 0.1
    ppc['action_allow_precision'] = 1e-5
    ppc['num_sample'] = 35132

    return ppc
