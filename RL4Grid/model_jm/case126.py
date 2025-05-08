# Copyright (c) 1996-2015 PSERC. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

"""Power flow data for 39 bus New England system.
"""
import numpy as np
from pypower.idx_bus import *
from pypower.idx_gen import *
from pypower.idx_brch import *
import os
path = os.path.dirname(__file__) + '/'

def case126():

    ppc = {"version": '2'}

    ##-----  Power Flow Data  -----##
    ## system MVA base
    ppc["baseMVA"] = 100.0

    ## bus data
    # bus_i type Pd Qd Gs Bs area Vm Va baseKV zone Vmax Vmin
    ppc["bus"] = np.load(path+'SG126_bus.npy')

    ## generator data
    # bus, Pg, Qg, Qmax, Qmin, Vg, mBase, status, Pmax, Pmin, Pc1, Pc2,
    # Qc1min, Qc1max, Qc2min, Qc2max, ramp_agc, ramp_10, ramp_30, ramp_q, apf
    ppc["gen"] = np.load(path+'SG126_gen.npy')

    ## branch data
    # fbus, tbus, r, x, b, rateA, rateB, rateC, ratio, angle, status, angmin, angmax
    ppc["branch"] = np.load(path+'SG126_branch.npy')

    ##-----  OPF Data  -----##
    ## generator cost data
    # 1 startup shutdown n x1 y1 ... xn yn
    # 2 startup shutdown n c(n-1) ... c0
    ppc["gencost"] = np.load(path+'SG126_gencost.npy')

    ppc['network'] = 'SG126'
    ppc['num_gen'] = ppc['gen'].shape[0]
    ppc['num_bus'] = ppc['bus'].shape[0]
    ppc['num_line'] = ppc['branch'].shape[0]    # 185 ori
    ppc['load_bus'] = np.nonzero(ppc['bus'][:, PD])[0].tolist()
    ppc['num_load'] = len(ppc['load_bus'])
    ppc['gen_type'] = np.array([5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 1, 1, 2, 1, 5, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 5])
    ppc['thermal_ids'] = np.where(ppc['gen_type'] == 1)[0].tolist()
    ppc['renewable_ids'] = np.where(ppc['gen_type'] == 5)[0].tolist()
    balanced_bus = ppc['bus'][np.where(ppc['bus'][:, BUS_TYPE] == 3)[0][0], BUS_I]
    ppc['balanced_id'] = np.where(ppc['gen'][:, GEN_BUS]+1==balanced_bus)[0][0]
    # ppc['balanced_id'] = np.where(ppc['gen_type'] == 2)[0].tolist()[0]
    ppc['sorted_controlable_ids'] = sorted(ppc['renewable_ids'] + ppc['thermal_ids'])
    ppc['min_gen_p'] = ppc['gen'][:, PMIN].tolist()
    ppc['max_gen_p'] = ppc['gen'][:, PMAX].tolist()
    for i in range(ppc['num_gen']):
        if i in ppc['thermal_ids']:
            ppc['min_gen_p'][i] = np.around(0.5 * ppc['gen'][i, PMAX], decimals=2).tolist()
    ppc['min_gen_q'] = [-180.0 for _ in range(ppc['num_gen'])]
    ppc['max_gen_q'] = [100.0 for _ in range(ppc['num_gen'])]
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
    ppc['line_thermal_limit'] = [313.92, 472.14, 899.01, 547.54, 593.39, 510.21, 340.29, 332.76,
    421.88, 504.58, 598.21, 445.0, 154.59, 327.68, 426.04, 338.65, 327.49, 374.02,
    370.94, 1120.62, 560.22, 602.95, 534.57, 490.72, 370.68, 346.75, 474.19, 670.54,
    1595.64, 458.65, 546.74, 660.22, 235.09, 2126.18, 1081.66, 578.59, 558.97, 677.71,
    393.76, 329.76, 515.94, 407.59, 592.89, 565.51, 609.72, 707.28, 1825.1, 1884.33,
    2000.32, 5042.53, 2348.87, 982.76, 925.44, 900.04, 297.88, 590.58, 699.74, 816.95,
    717.04, 425.57, 683.69, 720.87, 655.34, 648.31, 839.28, 439.4, 404.84, 241.6,
    354.29, 540.04, 505.36, 413.21, 258.2, 984.56, 795.51, 810.9, 529.76, 820.43,
    539.92, 303.62, 278.99, 243.14, 356.43, 354.18, 353.73, 284.72, 210.79, 335.84,
    1019.7, 6365.77, 2085.32, 781.7, 651.42, 382.77, 465.75, 508.24, 4382.12, 534.4,
    430.05, 725.12, 538.49, 1477.84, 1012.15, 522.97, 1026.97, 473.95, 449.73, 413.86,
    466.4, 454.41, 715.01, 337.2, 465.05, 348.73, 380.07, 148.98, 299.2, 3357.97,
    1295.32, 1289.9, 647.34, 972.3, 784.19, 436.1, 443.27, 680.91, 734.17, 795.4,
    353.95, 561.84, 609.4, 1212.41, 321.33, 937.52, 842.59, 733.43, 639.9, 1218.54,
    760.96, 526.56, 1184.53, 679.77, 949.06, 1147.56, 413.43, 1558.73, 948.07, 963.0,
    1230.32, 1137.6, 371.19, 659.98, 480.15, 1871.8, 656.35, 296.3, 351.13, 535.05,
    683.05, 437.39, 444.37, 851.91, 424.47, 659.23, 745.1, 885.21, 421.04, 1651.13,
    1098.42, 442.49, 278.21, 273.69, 225.8, 365.48, 498.46, 665.39, 876.37, 532.05,
    660.22, 892.74, 778.44, 871.3, 1651.13, 445.0, 546.74]
    # ppc['white_list_random_disconnection'] = [43, 44, 113, 114, 115, 118, 66, 75, 85, 98, 123, 138, 141]
    ppc['white_list_random_disconnection'] = [6, 37, 52, 71, 153, 136, 66, 75, 85, 98, 123, 138, 141]

    # ref/balanced generator
    ppc['min_balanced_gen_bound'] = 0.9
    ppc['max_balanced_gen_bound'] = 1.1

    # thermal generator
    ppc['ramp_rate'] = 0.05
    ppc['max_steps_to_recover_gen'] = []
    ppc['max_steps_to_close_gen'] = []
    ppc['fast_thermal_gen'] = []
    thresholds = [80, 140]
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
    ppc['num_sample'] = 106820

    return ppc